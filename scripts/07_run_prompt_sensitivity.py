#!/usr/bin/env python
"""Step 7 - prompt-sensitivity check on a Task 2 few-shot strategy.

`scripts/05_run_task2_fewshot.py` compared few-shot *strategies* on the **test**
split (`outputs/task2/final_all/`): `curated` ("curated concise") won, `random`
("random concise") lost. This script takes one of them - `--strategy` (default:
the winner, `curated`; `--strategy random` for the loser) - and perturbs it
along three axes that should not change the meaning of the prompt, then scores
every perturbation on the test split, so the spread of the numbers tells you
how much of that strategy's score is real signal vs. prompt luck:

  1. instruction phrasing  - 4 full paraphrases of the winner's instruction
                             block (context + both label definitions + the
                             judgment rules + the JSON-answer instruction),
                             each preserving every definition and rule
                             (prompt_sensitivity.PHRASINGS)
  2. demo order            - the same 4 demos in each of pos,neg,pos,neg /
                             pos,pos,neg,neg / neg,neg,pos,pos / neg,pos,neg,pos
                             (the strategy's own file order is the baseline)
  3. output schema         - the 3 fields reordered; and the 3 fields renamed
                             (label->sentiment, confidence->certainty,
                             evidence->justification) everywhere they appear
                             (format line, demos, instruction prose), mapped
                             back before scoring so label quality stays comparable

One axis is changed at a time; the other two stay at baseline. The baseline
combination reproduces the strategy's prompt file byte for byte
(`prompt_sensitivity.assert_baseline_matches`) and is scored here too, as the
reference point every axis is measured against. `curated` and `random` share
the identical instruction block, so one set of paraphrases covers both.

Everything else - model, decoding, the strategy's 4 demo reviews and their
hand-authored answers, and the exact eval rows - is held fixed, so any movement
in accuracy / macro-F1 / schema-compliance / ECE is attributable to the
perturbation and nothing else.

Produces, under outputs/task2_sensitivity/<run_id>/  (run_id defaults to
"<strategy>_<timestamp>"):
    run_config.json          the exact config used
    env_record.json          Python/PyTorch/CUDA/Transformers/GPU/OS
    eval_sample.json          seed + shape of the shared eval subset
    variants.json             every variant's full prompt text
    <axis>__<variant>/        one folder per perturbation (+ baseline/), each:
        prompt.json                 this variant's system prompt + user template
        prompt.txt                  the system prompt as plain text, for eyeballing
        predictions_task2.csv / errors_task2.csv / invalid_outputs_task2.csv
        metrics_task2.json          accuracy, macro-F1, per-class, confusion, invalid rate
        quality_task2.json          evidence relevance, calibration, format compliance
        results_task2.md            human-readable summary
        run_meta.json               seed, eval_sample, latency, tokens, GPU memory
    sensitivity_summary.csv   one row per variant, with deltas vs. baseline
    sensitivity_report.md     per-axis spread (range / std) + a verdict

Usage:
    # build every variant's prompt and inspect it, no GPU needed
    python scripts/07_run_prompt_sensitivity.py --dry-run

    # full run on the test split (1000 stratified rows) - the winner, curated
    python scripts/07_run_prompt_sensitivity.py

    # the loser, random ("random concise")
    python scripts/07_run_prompt_sensitivity.py --strategy random

    # 200 test rows, or all of test.csv, or just one axis
    python scripts/07_run_prompt_sensitivity.py --strategy random --n-samples 200
    python scripts/07_run_prompt_sensitivity.py --n-samples 3000 --run-id curated_full
    python scripts/07_run_prompt_sensitivity.py --only-axis demo_order
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src import dataset, env_record, eval_task2, prompt_sensitivity
from src import metrics as metrics_mod
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("task2_sensitivity")

BANNER_WIDTH = 66
_METRIC_KEYS = ("accuracy_overall", "macro_f1", "invalid_output_rate", "schema_compliance_rate")


# --------------------------------------------------------------------------
# Scoring one variant
# --------------------------------------------------------------------------
def _score_variant(
    lm, strategy: str, variant: dict[str, str], eval_df: pd.DataFrame, cfg: dict[str, Any],
    out_dir, eval_sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one prompt variant over eval_df; save its artefacts; return
    (metrics, quality, run_meta)."""
    from src import gpu_monitor, model as model_mod

    label = variant["label"]
    batch_size = cfg["task2"]["batch_size"]
    rows = eval_df.to_dict("records")
    records: list[dict[str, Any]] = []
    batch_stats: list[dict[str, Any]] = []

    wall_start = time.perf_counter()
    with gpu_monitor.MemoryTracker() as mem:
        for start in tqdm(range(0, len(rows), batch_size), desc=label, unit="batch"):
            batch = rows[start:start + batch_size]
            messages = [
                prompt_sensitivity.build_messages(
                    strategy, variant["phrasing"], variant["demo_order"], variant["schema"], r["text"]
                )
                for r in batch
            ]
            completions, stats = model_mod.generate_batch(lm, messages, cfg, temperature=0.0)
            batch_stats.append(stats)
            per_sample = stats["batch_seconds"] / max(len(batch), 1)
            for row, raw in zip(batch, completions):
                r = prompt_sensitivity.parse_variant(raw, variant["schema"])
                records.append({
                    "uid": row["uid"], "text": row["text"],
                    "gold": row["label"], "predicted": r.label if r.ok else "unparsed",
                    "latency_seconds": round(per_sample, 4),
                    "raw_output": raw[:500], **r.as_dict(),
                })
    wall_seconds = time.perf_counter() - wall_start

    preds = pd.DataFrame(records)
    m = metrics_mod.compute(preds)
    q = eval_task2.evaluate(preds)

    latencies = sorted(preds["latency_seconds"].dropna().tolist())
    total_generated = sum(s["generated_tokens"] for s in batch_stats)

    strat_dir = out_dir / label
    run_meta = {
        "axis": variant["axis"], "variant": variant["variant"], "label": label,
        "phrasing": variant["phrasing"], "demo_order": variant["demo_order"],
        "schema": variant["schema"],
        "checkpoint": lm.checkpoint, "revision": lm.revision, "precision": lm.precision,
        "prompt_version": f"sensitivity-{strategy}-{label}",
        "strategy": strategy,
        "seed": cfg["seed"], "source": cfg["task2"]["source"],
        "eval_sample": eval_sample,
        "wall_seconds": round(wall_seconds, 2),
        "latency_batched": {
            "mean_seconds_per_sample": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "median_seconds_per_sample": round(statistics.median(latencies), 4) if latencies else None,
            "p95_seconds_per_sample": round(latencies[int(0.95 * (len(latencies) - 1))], 4) if latencies else None,
            "throughput_samples_per_second": round(len(preds) / wall_seconds, 3) if wall_seconds else None,
        },
        "tokens": {
            "generated_total": total_generated,
            "generated_per_second": round(total_generated / wall_seconds, 2) if wall_seconds else None,
        },
        "peak_vram_gb": mem.report.get("peak_reserved_gb"),
        "total_vram_gb": mem.report.get("total_vram_gb"),
    }

    metrics_mod.save_all(preds, m, run_meta, strat_dir, tag="task2")
    write_json(run_meta, strat_dir / "run_meta.json")
    write_json(prompt_sensitivity.variant_metadata(strategy, variant), strat_dir / "prompt.json")
    (strat_dir / "prompt.txt").write_text(
        prompt_sensitivity.build_system_prompt(
            strategy, variant["phrasing"], variant["demo_order"], variant["schema"]
        ),
        encoding="utf-8",
    )
    write_json(q, strat_dir / "quality_task2.json")
    with open(strat_dir / "results_task2.md", "a", encoding="utf-8") as f:
        f.write("\n" + eval_task2.to_markdown(q))

    return m, q, run_meta


# --------------------------------------------------------------------------
# Summary + report
# --------------------------------------------------------------------------
def _row_for(variant: dict[str, str], m: dict[str, Any], q: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis": variant["axis"],
        "variant": variant["variant"],
        "label": variant["label"],
        "phrasing": variant["phrasing"],
        "demo_order": variant["demo_order"],
        "schema": variant["schema"],
        "n": m["n"],
        "accuracy": m["accuracy"],
        "accuracy_overall": m["accuracy_overall"],
        "macro_f1": m["macro_f1"],
        "invalid_output_rate": m["invalid_output_rate"],
        "schema_compliance_rate": (q.get("format_compliance") or {}).get("schema_compliance_rate"),
        "mean_evidence_relevance": (q.get("evidence") or {}).get("mean_relevance"),
        "null_evidence_relevance": (q.get("evidence") or {}).get("null_relevance"),
        "evidence_relevance_lift": (q.get("evidence") or {}).get("relevance_lift"),
        "ece": (q.get("calibration") or {}).get("ece"),
    }


def _spread(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "max": None, "range": None, "std": None}
    return {
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "range": round(max(vals) - min(vals), 4),
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
    }


def _write_report(
    df: pd.DataFrame, strategy: str, out_dir, cfg: dict[str, Any], eval_sample: dict[str, Any]
) -> None:
    base = df[df["axis"] == "baseline"].iloc[0]
    role = "test winner" if strategy == prompt_sensitivity.WINNING_STRATEGY else "test loser"
    lines = [
        "# Task 2 prompt-sensitivity report",
        "",
        f"- Strategy under test: `{strategy}`  ({role})",
        f"- Checkpoint: `{cfg['model']['checkpoint']}`  |  Precision: `{cfg['model']['precision']}`  |  Seed: `{cfg['seed']}`",
        f"- Eval sample: `{eval_sample['source']}`, n `{eval_sample['n_evaluated']}`, "
        f"stratified `{eval_sample['stratified']}`, seed `{eval_sample['seed']}`",
        f"- Baseline (reproduces `{strategy}`'s prompt exactly): "
        f"acc_overall **{base['accuracy_overall']}**, macro-F1 **{base['macro_f1']}**, "
        f"invalid **{base['invalid_output_rate']:.2%}**, schema-compliance **{base['schema_compliance_rate']:.2%}**",
        "",
        "A meaning-preserving prompt change should move these numbers only a "
        "little. A large `range` or `std` on an axis means this strategy's "
        "score leans on wording that a reasonable person could have written "
        "differently.",
        "",
    ]

    axis_titles = {
        "phrasing": "Axis 1 - instruction phrasing (4 paraphrases)",
        "demo_order": "Axis 2 - demo order",
        "schema": "Axis 3 - output schema (field order / field names)",
    }
    for axis, title in axis_titles.items():
        sub = df[df["axis"].isin([axis, "baseline"])].copy()
        if len(sub) <= 1:
            continue
        sub = sub.sort_values("axis")  # baseline first
        lines += [
            f"## {title}",
            "",
            "| variant | acc_overall | macro_f1 | invalid | schema_compliance | mean_evidence_rel | ece |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            tag = "baseline" if r["axis"] == "baseline" else r["variant"]
            lines.append(
                f"| {tag} | {r['accuracy_overall']} | {r['macro_f1']} | "
                f"{r['invalid_output_rate']:.2%} | {r['schema_compliance_rate']:.2%} | "
                f"{r['mean_evidence_relevance']} | {r['ece']} |"
            )
        lines.append("")
        lines += ["| metric | min | max | range | std (population) |", "|---|---|---|---|---|"]
        for key in _METRIC_KEYS:
            s = _spread(sub[key].tolist())
            lines.append(f"| {key} | {s['min']} | {s['max']} | **{s['range']}** | {s['std']} |")
        lines.append("")

    # headline: worst-case swing on each axis, macro-F1 and acc_overall
    lines += ["## Verdict", "", "| axis | macro_f1 range | acc_overall range |", "|---|---|---|"]
    for axis in axis_titles:
        sub = df[df["axis"].isin([axis, "baseline"])]
        if len(sub) <= 1:
            continue
        f1s = _spread(sub["macro_f1"].tolist())
        accs = _spread(sub["accuracy_overall"].tolist())
        lines.append(f"| {axis} | {f1s['range']} | {accs['range']} |")
    lines += [
        "",
        "_Notes._",
        "",
        "- `schema` / `renamed`: scored after mapping "
        "`sentiment`/`certainty`/`justification` back to "
        "`label`/`confidence`/`evidence`, so its `schema_compliance_rate` "
        "means 'structurally a clean 3-field object', not 'used the baseline "
        "field names'.",
        "- `schema` / `reordered`: only the key order in the `Output format:` "
        "line and the demo answers changes; the instruction prose still lists "
        "the fields label-first.",
        "- `phrasing` / `p2`-`p4` are hand-written full paraphrases of `p1`; "
        "an offline test checks each keeps every example snippet and rule, but "
        "wording judgement is ultimately manual.",
        "",
    ]
    (out_dir / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prompt-sensitivity check on Task 2's winning few-shot prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--strategy", default=prompt_sensitivity.DEFAULT_STRATEGY,
                     choices=list(prompt_sensitivity.SUPPORTED_STRATEGIES),
                     help="which Task 2 strategy's prompt to perturb "
                          f"(default: {prompt_sensitivity.DEFAULT_STRATEGY}, the test winner)")
    ap.add_argument("--run-id", default=None,
                     help="output subfolder name (default: <strategy>_<timestamp>)")
    ap.add_argument("--n-samples", type=int, default=None,
                     help="eval rows from --source (default: config's task2.n_samples; "
                          "a value >= the split size uses all of it)")
    ap.add_argument("--source", default="test", choices=list(dataset.PARTITIONS),
                     help="split to score on (default: test)")
    ap.add_argument("--only-axis", default=None,
                     choices=["phrasing", "demo_order", "schema"],
                     help="restrict to one perturbation axis (baseline always included)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="build and save every variant's prompt, then stop (no model load)")
    ap.add_argument("--hf-token", default=None,
                     help="HuggingFace token; normally omitted so it is read from .env")
    return ap.parse_args()


def _dry_run(strategy: str, variants: list[dict[str, str]], out_dir) -> None:
    for v in variants:
        strat_dir = out_dir / v["label"]
        strat_dir.mkdir(parents=True, exist_ok=True)
        write_json(prompt_sensitivity.variant_metadata(strategy, v), strat_dir / "prompt.json")
        (strat_dir / "prompt.txt").write_text(
            prompt_sensitivity.build_system_prompt(
                strategy, v["phrasing"], v["demo_order"], v["schema"]
            ),
            encoding="utf-8",
        )
    print("\n" + "=" * BANNER_WIDTH)
    print(f"DRY RUN - {strategy}: {len(variants)} variant prompt(s) written, no model loaded")
    print("=" * BANNER_WIDTH)
    print(f"  {'label':<22} {'phrasing':<9} {'demo_order':<16} {'schema':<10}")
    for v in variants:
        print(f"  {v['label']:<22} {v['phrasing']:<9} {v['demo_order']:<16} {v['schema']:<10}")
    print("=" * BANNER_WIDTH)
    print(f"Prompts: {out_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg["task2"]["n_samples"] = args.n_samples
    cfg["task2"]["source"] = args.source
    if args.batch_size is not None:
        cfg["task2"]["batch_size"] = args.batch_size

    strategy = args.strategy
    prompt_sensitivity.assert_baseline_matches(strategy)
    variants = prompt_sensitivity.build_variants(strategy, only_axis=args.only_axis)

    run_id = args.run_id or f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = resolve(cfg["task2"].get("sensitivity_out_dir", "outputs/task2_sensitivity")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Strategy: %s  |  Run id: %s  ->  %s", strategy, run_id, out_dir)

    set_seed(cfg["seed"])
    cfg["_sensitivity_strategy"] = strategy
    write_json(cfg, out_dir / "run_config.json")
    write_json(
        {v["label"]: prompt_sensitivity.variant_metadata(strategy, v) for v in variants},
        out_dir / "variants.json",
    )

    if args.dry_run:
        _dry_run(strategy, variants, out_dir)
        return

    write_json(env_record.collect(), out_dir / "env_record.json")

    source = cfg["task2"]["source"]
    n_requested = cfg["task2"]["n_samples"]
    stratified = cfg["task2"]["stratified_sample"]
    pool = dataset.load_split(cfg, source)
    eval_df = dataset.sample_for_eval(pool, n=n_requested, seed=cfg["seed"], stratified=stratified)

    eval_sample = {
        "seed": cfg["seed"], "source": source, "n_requested": n_requested,
        "n_evaluated": len(eval_df), "stratified": stratified,
        "label_counts": eval_df["label"].value_counts().to_dict(),
    }
    write_json(eval_sample, out_dir / "eval_sample.json")
    role = "test winner" if strategy == prompt_sensitivity.WINNING_STRATEGY else "test loser"
    print("\n" + "=" * BANNER_WIDTH)
    print(f"PROMPT SENSITIVITY  (strategy: {strategy} - {role})")
    print("=" * BANNER_WIDTH)
    print(f"  Eval split           : {source}")
    print(f"  Rows (req -> eval)    : {n_requested} -> {len(eval_df)} (stratified={stratified})")
    print(f"  Label counts         : {eval_sample['label_counts']}")
    print(f"  Variants             : {len(variants)} "
          f"({', '.join(sorted({v['axis'] for v in variants}))})")
    print("=" * BANNER_WIDTH)

    from src import model as model_mod

    lm = model_mod.load(cfg, hf_token=args.hf_token)

    summary_rows: list[dict[str, Any]] = []
    for v in variants:
        logger.info("Scoring variant: %s (axis=%s)", v["label"], v["axis"])
        m, q, _ = _score_variant(lm, strategy, v, eval_df, cfg, out_dir, eval_sample)
        summary_rows.append({"strategy": strategy, **_row_for(v, m, q)})

    df = pd.DataFrame(summary_rows)
    base = df[df["axis"] == "baseline"].iloc[0]
    df["delta_macro_f1"] = (df["macro_f1"] - base["macro_f1"]).round(4)
    df["delta_accuracy_overall"] = (df["accuracy_overall"] - base["accuracy_overall"]).round(4)
    df.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    _write_report(df, strategy, out_dir, cfg, eval_sample)

    print("\n" + "=" * BANNER_WIDTH)
    print(f"SENSITIVITY SUMMARY  ({run_id}, {source}, n={len(eval_df)})")
    print("=" * BANNER_WIDTH)
    print(f"  {'label':<22} {'acc_ovr':>8} {'macroF1':>8} {'dF1':>7} {'invalid':>8} {'schema':>8}")
    for _, r in df.iterrows():
        print(f"  {r['label']:<22} {r['accuracy_overall']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{r['delta_macro_f1']:>+7.4f} {r['invalid_output_rate']:>7.2%} "
              f"{r['schema_compliance_rate']:>7.2%}")
    print("=" * BANNER_WIDTH)
    for axis in ("phrasing", "demo_order", "schema"):
        sub = df[df["axis"].isin([axis, "baseline"])]
        if len(sub) > 1:
            s = _spread(sub["macro_f1"].tolist())
            print(f"  {axis:<12} macro-F1 range = {s['range']}  (min {s['min']}, max {s['max']})")
    print("=" * BANNER_WIDTH)
    print(f"\nArtefacts: {out_dir}")
    print(f"Report:    {out_dir / 'sensitivity_report.md'}")


if __name__ == "__main__":
    main()
