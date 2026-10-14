#!/usr/bin/env python
"""Step 5 - test Task 2 few-shot prompt strategies (curated vs random vs ...).

Scores a stratified sample of one split under every prompts2.STRATEGIES entry
and reports, for each:

  - final-label quality: accuracy, macro-F1, per-class, confusion matrix,
    invalid-output rate (via `metrics.compute()`, shared with Task 1)
  - evidence relevance, confidence calibration, and format compliance
    (via `eval_task2.evaluate()`)

so the example-selection methods can be compared before picking a winner.

"random" and "basic_random" are special only in provenance, not mechanics:
their few-shot demos are baked into their .txt files like every other
strategy's, but that text was generated from a single precomputed, fixed
draw in data/processed/random_fewshot_samples.csv (see src/prompts2.py and
scripts/10_select_random_fewshot_samples.py) - 2 positive + 2 negative
reviews from one seed. Both strategies use the *same* 4 rows, so the
comparison isolates the system prompt's rules, not example choice. The
drawn UIDs are saved to that strategy's prompt.json (`fewshot_seed`,
`fewshot_example_uids`), so the draw can be audited later.

Default source is "val" (matching the intended use: test prompt strategies on
val, then switch --source test --n-samples 1000 for the final comparison).

Every strategy in one run is scored on the *exact same* evaluation rows -
`dataset.sample_for_eval(pool, n, seed, stratified)` is deterministic in
(pool, n, seed, stratified). The console banner and each strategy's
run_meta.json record an `eval_sample` block (seed, source, n, stratified
flag, label counts): two runs sharing those - against the same frozen split -
saw the identical val subset, so their numbers are directly comparable.

Produces, under outputs/task2/<run_id>/:
    run_config.json         the exact config used
    env_record.json         Python/PyTorch/CUDA/Transformers/PEFT/GPU/OS
    eval_sample.json        seed + shape of the shared eval subset
    <strategy>/             one folder per prompt strategy, each holding:
        prompt.json                system prompt, user template, example uids
        predictions_task2.csv / errors_task2.csv / invalid_outputs_task2.csv
        metrics_task2.json         accuracy, macro-F1, per-class, confusion, invalid rate
        quality_task2.json         evidence relevance, calibration, format compliance
        results_task2.md           human-readable summary
        run_meta.json              seed, eval_sample, latency, tokens, GPU memory

Usage:
    python scripts/05_run_task2_fewshot.py
    python scripts/05_run_task2_fewshot.py --n-samples 100
    python scripts/05_run_task2_fewshot.py --strategy curated
    python scripts/05_run_task2_fewshot.py --strategy representative
    python scripts/05_run_task2_fewshot.py --source test --n-samples 1000 --run-id final
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
from src import (
    dataset, env_record, eval_task2, gpu_monitor, json_parser,
    metrics as metrics_mod, model as model_mod, prompts2,
)
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("task2")

BANNER_WIDTH = 62


def eval_sample_note(
    eval_df: pd.DataFrame, seed: int, source: str, n_requested: int | None, stratified: bool
) -> dict[str, Any]:
    """Record of how the shared evaluation subset was drawn.

    `sample_for_eval` is a pure function of (pool, n, seed, stratified): two
    runs with the same seed, source, n and stratified flag - against the same
    frozen split - score the identical val rows, so their numbers are directly
    comparable. This block is echoed into every strategy's run_meta.json.
    """
    return {
        "seed": seed,
        "source": source,
        "n_requested": n_requested,
        "n_evaluated": len(eval_df),
        "stratified": stratified,
        "label_counts": eval_df["label"].value_counts().to_dict(),
    }


def _print_eval_sample_banner(es: dict[str, Any]) -> None:
    print("\n" + "=" * BANNER_WIDTH)
    print("EVAL SAMPLE  (shared by every strategy in this run)")
    print("=" * BANNER_WIDTH)
    print(f"  Source / split       : {es['source']}")
    print(f"  Seed                 : {es['seed']}")
    print(f"  Rows (req -> eval)    : {es['n_requested']} -> {es['n_evaluated']}  "
          f"(stratified={es['stratified']})")
    print(f"  Label counts         : {es['label_counts']}")
    print("=" * BANNER_WIDTH)


def _print_strategy_banner(
    run_id: str, strategy: str, m: dict[str, Any], q: dict[str, Any],
    run_meta: dict[str, Any],
) -> None:
    """One box per strategy, laid out like Task 1's console summary."""
    tokens = run_meta.get("tokens") or {}
    latency = run_meta.get("latency_batched") or {}
    ece = q["calibration"]["ece"]
    ev_rel = q["evidence"]["mean_relevance"]
    ev_null = q["evidence"]["null_relevance"]
    ev_lift = q["evidence"]["relevance_lift"]
    acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a (no valid outputs)"

    print("\n" + "=" * BANNER_WIDTH)
    print(f"TASK 2 RESULTS - {strategy.upper()}  ({run_id} / {strategy})")
    print("=" * BANNER_WIDTH)
    print(f"  n evaluated          : {m['n']}")
    print(f"  Accuracy (valid only): {acc}")
    print(f"  Accuracy (overall)   : {m['accuracy_overall']:.4f}")
    print(f"  Macro F1             : {m['macro_f1']:.4f}")
    print(f"  Invalid-output rate  : {m['invalid_output_rate']:.2%}")
    print(f"  Schema compliance    : {q['format_compliance']['schema_compliance_rate']:.2%}")
    print(f"  Evidence relevance   : {ev_rel if ev_rel is not None else 'n/a'}")
    print(f"  Evidence null base.  : {ev_null if ev_null is not None else 'n/a'}")
    print(f"  Evidence rel. lift   : {ev_lift if ev_lift is not None else 'n/a'}")
    print(f"  Calibration ECE      : {ece if ece is not None else 'n/a'}")
    print(f"  Tokens generated     : {tokens.get('generated_total')} "
          f"({tokens.get('generated_per_second')} tok/s)")
    print(f"  Peak GPU memory      : {run_meta['peak_vram_gb']} / {run_meta['total_vram_gb']} GB")
    print(f"  Latency (batched)    : {latency.get('mean_seconds_per_sample')} s/sample")
    print(f"  Wall time            : {run_meta['wall_seconds']} s")
    print("=" * BANNER_WIDTH)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Test Task 2 few-shot prompt strategies.")
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--run-id", default=None, help="output subfolder name (default: timestamp)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--source", default=None, choices=list(dataset.PARTITIONS))
    ap.add_argument("--strategy", default=None, choices=list(prompts2.STRATEGIES),
                     help="run only one strategy (default: all)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--hf-token", default=None,
                     help="HuggingFace token; normally omitted so it is read from .env")
    return ap.parse_args()


def _score_strategy(
    lm: model_mod.LoadedModel, strategy: str, eval_df: pd.DataFrame,
    cfg: dict[str, Any], out_dir, eval_sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one prompt strategy over eval_df and save its artefacts under out_dir/<strategy>/.

    Returns (metrics, quality, run_meta).
    """
    label = strategy
    batch_size = cfg["task2"]["batch_size"]
    rows = eval_df.to_dict("records")
    records = []
    batch_stats = []

    wall_start = time.perf_counter()
    with gpu_monitor.MemoryTracker() as mem:
        for start in tqdm(range(0, len(rows), batch_size), desc=label, unit="batch"):
            batch = rows[start:start + batch_size]
            messages = [prompts2.build_messages(strategy, r["text"]) for r in batch]
            completions, stats = model_mod.generate_batch(lm, messages, cfg, temperature=0.0)
            batch_stats.append(stats)
            per_sample = stats["batch_seconds"] / max(len(batch), 1)
            for row, raw in zip(batch, completions):
                r = json_parser.parse_fewshot(raw)
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
    fewshot_seed = prompts2.random_fewshot_seed() if strategy in prompts2.RANDOM_STRATEGIES else None
    run_meta = {
        "strategy": strategy, "checkpoint": lm.checkpoint, "revision": lm.revision,
        "precision": lm.precision, "prompt_version": f"fewshot-{label}",
        "fewshot_seed": fewshot_seed, "seed": cfg["seed"], "source": cfg["task2"]["source"],
        # How the shared eval subset was drawn (seed/source/n/stratified) -
        # identical across every strategy in a run, comparable across runs.
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
    paths = metrics_mod.save_all(preds, m, run_meta, strat_dir, tag="task2")
    write_json(run_meta, strat_dir / "run_meta.json")
    # For "random"/"basic_random" this also records fewshot_seed +
    # fewshot_example_uids - exactly which train.csv rows were drawn as
    # few-shot demos.
    write_json(prompts2.prompt_metadata(strategy), strat_dir / "prompt.json")

    # Quality dimensions beyond label accuracy: evidence relevance, confidence
    # calibration, format compliance. Written separately so metrics.py stays
    # shared/generic between Task 1 and Task 2.
    write_json(q, strat_dir / "quality_task2.json")
    with open(paths["results_md"], "a", encoding="utf-8") as f:
        f.write("\n" + eval_task2.to_markdown(q))

    return m, q, run_meta


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg["task2"]["n_samples"] = args.n_samples
    if args.source is not None:
        cfg["task2"]["source"] = args.source
    if args.batch_size is not None:
        cfg["task2"]["batch_size"] = args.batch_size

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve(cfg["task2"]["out_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("task2_run")
    log.info("Run id: %s  ->  %s", run_id, out_dir)

    set_seed(cfg["seed"])
    write_json(cfg, out_dir / "run_config.json")
    write_json(env_record.collect(), out_dir / "env_record.json")

    source = cfg["task2"]["source"]
    n_requested = cfg["task2"]["n_samples"]
    stratified = cfg["task2"]["stratified_sample"]
    pool = dataset.load_split(cfg, source)
    eval_df = dataset.sample_for_eval(
        pool, n=n_requested, seed=cfg["seed"], stratified=stratified,
    )

    # Seed / eval-sample note: record how the shared eval subset was drawn
    # (seed/source/n/stratified), so a later run can be checked for having
    # used the same val subset. Saved to eval_sample.json and echoed into
    # every strategy's run_meta.json.
    eval_sample = eval_sample_note(eval_df, cfg["seed"], source, n_requested, stratified)
    write_json(eval_sample, out_dir / "eval_sample.json")
    log.info("Eval sample | source=%s seed=%s n=%d/%d stratified=%s | labels: %s",
             source, cfg["seed"], eval_sample["n_evaluated"], len(pool),
             stratified, eval_sample["label_counts"])
    _print_eval_sample_banner(eval_sample)

    lm = model_mod.load(cfg, hf_token=args.hf_token)

    strategies = [args.strategy] if args.strategy else list(prompts2.STRATEGIES)
    results = {}
    for strategy in strategies:
        log.info("Scoring strategy: %s", strategy)
        results[strategy] = _score_strategy(lm, strategy, eval_df, cfg, out_dir, eval_sample)

    # --- console summary: one Task 1-style box per strategy ----------------
    for strategy, (m, q, run_meta) in results.items():
        _print_strategy_banner(run_id, strategy, m, q, run_meta)

    # --- leaderboard: only meaningful when >1 strategy ran -----------------
    if len(results) > 1:
        print("\n" + "=" * BANNER_WIDTH)
        print(f"TASK 2 FEW-SHOT COMPARISON  ({run_id}, {source}, n={len(eval_df)})")
        print("=" * BANNER_WIDTH)
        print(f"  {'strategy':<20} {'acc_overall':>11} {'macro_f1':>9} {'invalid':>8} {'schema':>8}")
        for strategy, (m, q, _) in sorted(
            results.items(), key=lambda kv: kv[1][0]["macro_f1"], reverse=True
        ):
            print(f"  {strategy:<20} {m['accuracy_overall']:>11.4f} {m['macro_f1']:>9.4f} "
                  f"{m['invalid_output_rate']:>7.2%} "
                  f"{q['format_compliance']['schema_compliance_rate']:>7.2%}")
        print("=" * BANNER_WIDTH)

    print(f"\nEval sample: seed={eval_sample['seed']}, {source}, "
          f"n={eval_sample['n_evaluated']}, stratified={eval_sample['stratified']}")
    print(f"Artefacts: {out_dir}")


if __name__ == "__main__":
    main()
