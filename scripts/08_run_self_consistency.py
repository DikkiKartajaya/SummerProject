#!/usr/bin/env python
"""Step 8 - Task 2 few-shot inference with optional self-consistency voting.

Runs Task 2's best and worst strategies on the test comparison - `curated`
("curated concise", the test-set winner) and `random` ("random concise", the
test-set loser) - and, when asked, replaces the single greedy prediction per
review with N stochastic predictions + majority vote
(Wang et al. 2022, self-consistency).

  --self-consistency 1   (default)  greedy, temperature 0, one reply per review.
                                    Byte-identical to scripts/05_run_task2_fewshot.py's
                                    inference - the greedy path is NOT changed.
  --self-consistency N   (N > 1)    N samples per review in ONE generate() call
                                    (num_return_sequences=N, temperature=0.7,
                                    top_p=0.95), seeded from cfg.seed for
                                    determinism. Majority label wins; ties go to
                                    the highest-confidence candidate. See
                                    src/self_consistency.py for the vote rules.

The output dataframe keeps every column scripts/05 writes (so
`eval_task2.evaluate()` is unaffected) and adds exactly two:
  - vote_margin    winning votes / valid votes  (1.0 for the greedy path)
  - n_valid_votes  how many candidates parsed   (1 for the greedy path)

`run_meta.json` records `self_consistency_n` and `temperature`.

CAVEAT: with N > 1 a row counts as valid if *any* candidate parses, a weaker
bar than greedy's "the one reply parses", so `invalid_output_rate` from a
self-consistency run is not apple-to-apple with the single-sample baseline (it
can only drop). Compare accuracy / macro-F1 across modes; read the invalid rate
only within self-consistency runs.

Artefacts land under outputs/task2_selfconsistency/<run_id>/<strategy>/ with the
same files scripts/05 produces, plus self_consistency_summary.csv at the run root.

Usage:
    # greedy baseline on both strategies, 200 test rows
    python scripts/08_run_self_consistency.py --n-samples 200

    # 3 samples + majority vote (the requested experiment)
    python scripts/08_run_self_consistency.py --n-samples 200 --self-consistency 3

    # one strategy only, full comparison size
    python scripts/08_run_self_consistency.py --strategy curated --self-consistency 5
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
    metrics as metrics_mod, model as model_mod, prompts2, self_consistency,
)
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("task2_sc")

BANNER_WIDTH = 64
DEFAULT_STRATEGIES = ["curated", "random"]  # test-set best + worst


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Task 2 few-shot inference with optional self-consistency voting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--run-id", default=None, help="output subfolder name (default: timestamp)")
    ap.add_argument("--n-samples", type=int, default=None,
                     help="eval rows from --source (default: config's task2.n_samples)")
    ap.add_argument("--source", default="test", choices=list(dataset.PARTITIONS),
                     help="split to score on (default: test)")
    ap.add_argument("--self-consistency", type=int, default=1, metavar="N",
                     help="N stochastic predictions + majority vote (default 1 = greedy, unchanged)")
    ap.add_argument("--strategy", default=None, choices=list(prompts2.STRATEGIES),
                     help=f"run only one strategy (default: {' + '.join(DEFAULT_STRATEGIES)})")
    ap.add_argument("--batch-size", type=int, default=None,
                     help="lower this if you OOM - each row generates N sequences when --self-consistency > 1")
    ap.add_argument("--hf-token", default=None,
                     help="HuggingFace token; normally omitted so it is read from .env")
    return ap.parse_args()


def _record_greedy(row: dict[str, Any], raw: str, per_sample: float) -> dict[str, Any]:
    """Greedy path: identical to scripts/05's record, plus the two vote columns
    as constants (1 vote, full margin)."""
    r = json_parser.parse_fewshot(raw)
    return {
        "uid": row["uid"], "text": row["text"], "gold": row["label"],
        "predicted": r.label if r.ok else "unparsed",
        "latency_seconds": round(per_sample, 4),
        "raw_output": raw[:500], **r.as_dict(),
        "vote_margin": 1.0 if r.ok else 0.0,
        "n_valid_votes": 1 if r.ok else 0,
    }


def _record_voted(row: dict[str, Any], candidates: list[str], per_sample: float) -> dict[str, Any]:
    """Self-consistency path: majority vote over N candidates."""
    agg = self_consistency.aggregate(candidates)
    predicted = agg.pop("_label")
    return {
        "uid": row["uid"], "text": row["text"], "gold": row["label"],
        "predicted": predicted if agg["parse_ok"] else "unparsed",
        "latency_seconds": round(per_sample, 4),
        "raw_output": " ||| ".join(c.strip() for c in candidates)[:800],
        **agg,
    }


def _score_strategy(
    lm: model_mod.LoadedModel, strategy: str, eval_df: pd.DataFrame,
    cfg: dict[str, Any], out_dir, eval_sample: dict[str, Any], sc_n: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one strategy over eval_df (greedy or voted), save artefacts, return
    (metrics, quality, run_meta)."""
    batch_size = cfg["task2"]["batch_size"]
    rows = eval_df.to_dict("records")
    records: list[dict[str, Any]] = []
    batch_stats: list[dict[str, Any]] = []

    if sc_n > 1:
        import transformers
        # Deterministic sampling for this strategy: seed once, then let the
        # fixed batch order consume the RNG reproducibly.
        transformers.set_seed(cfg["seed"])

    desc = f"{strategy} (N={sc_n})"
    wall_start = time.perf_counter()
    with gpu_monitor.MemoryTracker() as mem:
        for start in tqdm(range(0, len(rows), batch_size), desc=desc, unit="batch"):
            batch = rows[start:start + batch_size]
            messages = [prompts2.build_messages(strategy, r["text"]) for r in batch]

            if sc_n == 1:
                completions, stats = model_mod.generate_batch(lm, messages, cfg, temperature=0.0)
                per_sample = stats["batch_seconds"] / max(len(batch), 1)
                for row, raw in zip(batch, completions):
                    records.append(_record_greedy(row, raw, per_sample))
            else:
                grouped, stats = model_mod.generate_batch_samples(
                    lm, messages, cfg, n=sc_n,
                    temperature=self_consistency.SC_TEMPERATURE,
                    top_p=self_consistency.SC_TOP_P,
                )
                per_sample = stats["batch_seconds"] / max(len(batch), 1)
                for row, cands in zip(batch, grouped):
                    records.append(_record_voted(row, cands, per_sample))

            batch_stats.append(stats)
    wall_seconds = time.perf_counter() - wall_start

    preds = pd.DataFrame(records)
    m = metrics_mod.compute(preds)
    q = eval_task2.evaluate(preds)

    latencies = sorted(preds["latency_seconds"].dropna().tolist())
    total_generated = sum(s["generated_tokens"] for s in batch_stats)
    strat_dir = out_dir / strategy

    run_meta = {
        "strategy": strategy, "checkpoint": lm.checkpoint, "revision": lm.revision,
        "precision": lm.precision, "prompt_version": f"fewshot-{strategy}-sc{sc_n}",
        "seed": cfg["seed"], "source": cfg["task2"]["source"],
        # --- self-consistency block (spec) --------------------------------
        "self_consistency_n": sc_n,
        "temperature": 0.0 if sc_n == 1 else self_consistency.SC_TEMPERATURE,
        "top_p": 1.0 if sc_n == 1 else self_consistency.SC_TOP_P,
        "sample_seed": cfg["seed"] if sc_n > 1 else None,
        "vote_margin_mean": round(float(preds["vote_margin"].mean()), 4) if len(preds) else None,
        "n_valid_votes_mean": round(float(preds["n_valid_votes"].mean()), 4) if len(preds) else None,
        "invalid_rate_note": (
            "single-sample: comparable to the greedy baseline"
            if sc_n == 1 else
            "N>1: a row is valid if ANY candidate parses - not apple-to-apple "
            "with the greedy baseline's invalid_output_rate (can only drop)."
        ),
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
    write_json(prompts2.prompt_metadata(strategy), strat_dir / "prompt.json")
    write_json(q, strat_dir / "quality_task2.json")
    with open(paths["results_md"], "a", encoding="utf-8") as f:
        f.write("\n" + eval_task2.to_markdown(q))
        f.write(f"\n\n## Self-consistency\n\n"
                f"- N (predictions per review): **{sc_n}**"
                f"{'  _(greedy, unchanged)_' if sc_n == 1 else ''}\n")
        if sc_n > 1:
            f.write(f"- Sampling: temperature `{self_consistency.SC_TEMPERATURE}`, "
                    f"top_p `{self_consistency.SC_TOP_P}`, seed `{cfg['seed']}`\n"
                    f"- Mean vote margin: **{run_meta['vote_margin_mean']}**  |  "
                    f"mean valid votes: **{run_meta['n_valid_votes_mean']}** / {sc_n}\n"
                    f"- ⚠️ `invalid_output_rate` above is **not** comparable to the "
                    f"greedy baseline: a row is valid if any of the {sc_n} candidates "
                    f"parses, so it can only be lower. Compare accuracy / macro-F1.\n")

    return m, q, run_meta


def _print_strategy_banner(strategy: str, m: dict[str, Any], q: dict[str, Any], rm: dict[str, Any]) -> None:
    acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
    print("\n" + "=" * BANNER_WIDTH)
    print(f"TASK 2 SELF-CONSISTENCY - {strategy.upper()}  (N={rm['self_consistency_n']})")
    print("=" * BANNER_WIDTH)
    print(f"  n evaluated          : {m['n']}")
    print(f"  Accuracy (valid only): {acc}")
    print(f"  Accuracy (overall)   : {m['accuracy_overall']:.4f}")
    print(f"  Macro F1             : {m['macro_f1']:.4f}")
    print(f"  Invalid-output rate  : {m['invalid_output_rate']:.2%}"
          f"{'' if rm['self_consistency_n'] == 1 else '  (not vs-baseline comparable)'}")
    print(f"  Schema compliance    : {q['format_compliance']['schema_compliance_rate']:.2%}")
    if rm["self_consistency_n"] > 1:
        print(f"  Mean vote margin     : {rm['vote_margin_mean']}")
        print(f"  Mean valid votes     : {rm['n_valid_votes_mean']} / {rm['self_consistency_n']}")
    print(f"  Latency (per review) : {rm['latency_batched']['mean_seconds_per_sample']} s")
    print(f"  Peak GPU memory      : {rm['peak_vram_gb']} / {rm['total_vram_gb']} GB")
    print("=" * BANNER_WIDTH)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg["task2"]["n_samples"] = args.n_samples
    cfg["task2"]["source"] = args.source
    if args.batch_size is not None:
        cfg["task2"]["batch_size"] = args.batch_size

    sc_n = args.self_consistency
    if sc_n < 1:
        raise SystemExit("--self-consistency must be >= 1")
    strategies = [args.strategy] if args.strategy else list(DEFAULT_STRATEGIES)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve(cfg["task2"].get("selfconsistency_out_dir", "outputs/task2_selfconsistency")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run id: %s  ->  %s", run_id, out_dir)

    set_seed(cfg["seed"])
    write_json(cfg, out_dir / "run_config.json")
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

    print("\n" + "=" * BANNER_WIDTH)
    print(f"TASK 2 SELF-CONSISTENCY  (N={sc_n}, {'greedy' if sc_n == 1 else 'sampled'})")
    print("=" * BANNER_WIDTH)
    print(f"  Strategies           : {', '.join(strategies)}")
    print(f"  Eval split           : {source}")
    print(f"  Rows (req -> eval)    : {n_requested} -> {len(eval_df)} (stratified={stratified})")
    print(f"  Label counts         : {eval_sample['label_counts']}")
    if sc_n > 1:
        print(f"  Sampling             : temp {self_consistency.SC_TEMPERATURE}, "
              f"top_p {self_consistency.SC_TOP_P}, seed {cfg['seed']}")
        print("  NOTE                 : invalid_output_rate not comparable to the "
              "greedy baseline (valid if any candidate parses)")
    print("=" * BANNER_WIDTH)

    lm = model_mod.load(cfg, hf_token=args.hf_token)

    results = {}
    for strategy in strategies:
        logger.info("Scoring strategy: %s (N=%d)", strategy, sc_n)
        results[strategy] = _score_strategy(lm, strategy, eval_df, cfg, out_dir, eval_sample, sc_n)

    for strategy, (m, q, rm) in results.items():
        _print_strategy_banner(strategy, m, q, rm)

    summary_rows = []
    for strategy, (m, q, rm) in results.items():
        summary_rows.append({
            "strategy": strategy,
            "self_consistency_n": sc_n,
            "temperature": rm["temperature"],
            "n": m["n"],
            "accuracy": m["accuracy"],
            "accuracy_overall": m["accuracy_overall"],
            "macro_f1": m["macro_f1"],
            "invalid_output_rate": m["invalid_output_rate"],
            "schema_compliance_rate": q["format_compliance"]["schema_compliance_rate"],
            "mean_evidence_relevance": q["evidence"]["mean_relevance"],
            "ece": q["calibration"]["ece"],
            "vote_margin_mean": rm["vote_margin_mean"],
            "n_valid_votes_mean": rm["n_valid_votes_mean"],
            "sec_per_sample": rm["latency_batched"]["mean_seconds_per_sample"],
        })
    df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    df.to_csv(out_dir / "self_consistency_summary.csv", index=False)

    if len(results) > 1:
        print("\n" + "=" * BANNER_WIDTH)
        print(f"COMPARISON  (N={sc_n}, {source}, n={len(eval_df)})")
        print("=" * BANNER_WIDTH)
        print(f"  {'strategy':<12} {'acc_overall':>11} {'macro_f1':>9} {'invalid':>8} {'schema':>8}")
        for _, r in df.iterrows():
            print(f"  {r['strategy']:<12} {r['accuracy_overall']:>11.4f} {r['macro_f1']:>9.4f} "
                  f"{r['invalid_output_rate']:>7.2%} {r['schema_compliance_rate']:>7.2%}")
        print("=" * BANNER_WIDTH)

    print(f"\nArtefacts: {out_dir}")


if __name__ == "__main__":
    main()
