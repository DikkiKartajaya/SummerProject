#!/usr/bin/env python
"""Step 3 - run Task 1 (zero-shot baseline + prompting ablations) and write
every artefact.

One script drives all three Task 1 prompting methods - `--strategy` picks
which: `zero_shot` (plain `{"label": ...}`, `prompts.py` / `configs/task1.yaml`),
`carp` (Clue -> Reasoning -> Label, `prompts_carp.py` / `configs/task1_carp.yaml`),
or `basic` (a bare POSITIVE/NEGATIVE word, `prompts_basic.py` /
`configs/task1_basic.yaml`). Same model, same data, same retry-aware
`inference.run()` loop, same metrics for all three - only the prompt module
and its matching decoding budget (CARP needs far more `max_new_tokens` for
its reasoning chain) differ, which is why each strategy has its own config
file; `--config` overrides the strategy's default if needed.

Every strategy writes into the SAME `outputs/task1/` directory (see each
config's `task1.out_dir`), tagged with its `"prompt_method"` in
run_meta.json, so `scripts/04_summarize_runs.py` compares all three in one
table without any extra wiring.

Produces, under outputs/task1/<run_id>/:
    run_config.json      the exact config used (copy, not a reference)
    env_record.json      Python/PyTorch/CUDA/Transformers/PEFT/GPU/OS
    prompt.json          system prompt, user template, retry suffixes
    predictions_task1.csv / errors_task1.csv / invalid_outputs_task1.csv
    metrics_task1.json   accuracy, macro-F1, per-class, confusion matrix, invalid-output rate
    results_task1.md     human-readable summary table
    run_stats.json       latency, tokens, GPU memory
    confusion_matrix_task1.png

A crash-safe `predictions.jsonl` checkpoint exists only while the run is in
progress (so a killed/interrupted run can resume with `--run-id` + the same
partition instead of starting over) - it is deleted once the run finishes
successfully and its rows have been folded into predictions_task1.csv.

`predictions_task1.csv`'s `extra_keys` column will show `["clue", "reasoning"]`
for every successfully parsed CARP row, and every basic-prompt row's
`parse_strategy` will read `bare_label` - both expected, not schema
violations: the parser only requires the `label` key and reports whatever
else it finds, and the basic prompt never asks for JSON in the first place.

Usage:
    python scripts/03_run_task1.py
    python scripts/03_run_task1.py --n-samples 1000
    python scripts/03_run_task1.py --n-samples all --run-id full_pool
    python scripts/03_run_task1.py --precision nf4 --run-id ablation_nf4
    python scripts/03_run_task1.py --strategy carp --n-samples 200 --run-id carp_200
    python scripts/03_run_task1.py --strategy basic --n-samples 200 --run-id basic_200
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime

import _bootstrap  # noqa: F401
from src import (
    dataset, env_record, inference, metrics as metrics_mod,
    model as model_mod, prompts, prompts_basic, prompts_carp,
)
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("task1")

# Everything that differs between Task 1's three prompting methods: which
# module builds the chat messages, which config has the matching decoding
# budget (CARP's reasoning chain needs far more `max_new_tokens` than a bare
# label does), the run-id prefix used when `--run-id` is omitted (so runs
# stay distinguishable by timestamp alone), and the console summary title.
STRATEGIES = {
    "zero_shot": {
        "module": prompts,
        "config": "configs/task1.yaml",
        "run_id_prefix": "",
        "title": "TASK 1 RESULTS",
    },
    "carp": {
        "module": prompts_carp,
        "config": "configs/task1_carp.yaml",
        "run_id_prefix": "carp_",
        "title": "TASK 1 RESULTS - CARP",
    },
    "basic": {
        "module": prompts_basic,
        "config": "configs/task1_basic.yaml",
        "run_id_prefix": "basic_",
        "title": "TASK 1 RESULTS - BASIC PROMPT",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Task 1: zero-shot baseline + prompting ablations.")
    ap.add_argument("--strategy", default="zero_shot", choices=list(STRATEGIES),
                    help="which prompting method to run (default: zero_shot)")
    ap.add_argument("--config", default=None,
                    help="override the strategy's default config file")
    ap.add_argument("--run-id", default=None,
                    help="output subfolder name (default: <strategy prefix><timestamp>)")
    ap.add_argument("--n-samples", default=None,
                    help="number of examples, or 'all' (overrides the config)")
    ap.add_argument("--source", default=None, choices=list(dataset.PARTITIONS),
                    help="which partition to evaluate on (default: config -> all). "
                         "'holdout' is the independent HF `test` split")
    ap.add_argument("--precision", default=None, choices=["bf16", "fp16", "nf4", "int8"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start over")
    ap.add_argument("--latency-samples", type=int, default=30,
                    help="examples used for the batch-size-1 latency probe (0 = skip)")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token; normally omitted so it is read from .env")
    return ap.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags win over the YAML file, so ablations need no file edits."""
    if args.n_samples is not None:
        cfg["task1"]["n_samples"] = None if args.n_samples.lower() == "all" else int(args.n_samples)
    if args.source is not None:
        cfg["task1"]["source"] = args.source
    if args.precision is not None:
        cfg["model"]["precision"] = args.precision
    if args.batch_size is not None:
        cfg["task1"]["batch_size"] = args.batch_size
    return cfg


def main() -> None:
    args = parse_args()
    strat = STRATEGIES[args.strategy]
    prompt_mod = strat["module"]
    cfg = apply_overrides(load_config(args.config or strat["config"]), args)

    run_id = args.run_id or f"{strat['run_id_prefix']}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = resolve(cfg["task1"]["out_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger("task1_run")
    log.info("Run id: %s (strategy=%s)  ->  %s", run_id, args.strategy, out_dir)

    # --- reproducibility: seed + full environment snapshot ------------------
    set_seed(cfg["seed"])
    env = env_record.collect()
    write_json(env, out_dir / "env_record.json")
    print(env_record.summarize(env))

    write_json(cfg, out_dir / "run_config.json")
    write_json(prompt_mod.prompt_metadata(), out_dir / "prompt.json")

    # --- data ---------------------------------------------------------------
    source = cfg["task1"]["source"]
    pool = dataset.load_split(cfg, source)
    eval_df = dataset.sample_for_eval(
        pool,
        n=cfg["task1"]["n_samples"],
        seed=cfg["seed"],
        stratified=cfg["task1"]["stratified_sample"],
    )
    log.info("Evaluating %d / %d rows from '%s' | labels: %s",
             len(eval_df), len(pool), source, eval_df["label"].value_counts().to_dict())

    # --- model --------------------------------------------------------------
    lm = model_mod.load(cfg, hf_token=args.hf_token)

    # --- inference ----------------------------------------------------------
    checkpoint = out_dir / "predictions.jsonl"
    preds, run_stats = inference.run(
        lm, eval_df, cfg, checkpoint_path=checkpoint, resume=not args.no_resume,
        build_messages=prompt_mod.build_messages,
    )

    # --- single-request latency (separate from batched throughput) ----------
    latency_single = {}
    if args.latency_samples > 0:
        log.info("Measuring batch-size-1 latency on %d examples...", args.latency_samples)
        latency_single = inference.latency_probe(
            lm, eval_df.to_dict("records"), cfg, n=args.latency_samples,
            build_messages=prompt_mod.build_messages,
        )
    run_stats["latency_single"] = latency_single
    write_json(run_stats, out_dir / "run_stats.json")

    # --- metrics ------------------------------------------------------------
    m = metrics_mod.compute(preds)
    run_meta = {
        "run_id": run_id,
        "prompt_method": args.strategy,
        "checkpoint": lm.checkpoint,
        "revision": lm.revision,
        "precision": lm.precision,
        "hf_token_source": lm.token_source,  # source only, never the value
        "seed": cfg["seed"],
        "source": source,
        "decoding": model_mod.build_generation_kwargs(cfg),
        "wall_seconds": run_stats["wall_seconds"],
        "latency_batched": run_stats["latency_batched"],
        "latency_single": latency_single,
        "tokens": run_stats["tokens"],
        "peak_vram_gb": run_stats["gpu_memory"].get("peak_reserved_gb"),
        "total_vram_gb": run_stats["gpu_memory"].get("total_vram_gb"),
    }
    paths = metrics_mod.save_all(preds, m, run_meta, out_dir, tag="task1")
    write_json(run_meta, out_dir / "run_meta.json")

    # Convenience: keep a copy of the latest results at the task root.
    shutil.copy(paths["results_md"], resolve(cfg["task1"]["out_dir"]) / "LATEST_results.md")

    # The checkpoint's only job was crash-safety during this run; every row is
    # now in predictions_task1.csv, so drop it rather than leave a duplicate.
    checkpoint.unlink(missing_ok=True)

    # --- console summary ----------------------------------------------------
    print("\n" + "=" * 62)
    print(f"{strat['title']}  ({run_id})")
    print("=" * 62)
    print(f"  n evaluated          : {m['n']}")
    print(f"  Accuracy (valid only): {m['accuracy']:.4f}" if m['accuracy'] is not None
          else "  Accuracy (valid only): n/a (no valid outputs)")
    print(f"  Macro F1             : {m['macro_f1']:.4f}")
    print(f"  Invalid-output rate  : {m['invalid_output_rate']:.2%}")
    print(f"  Tokens generated     : {run_stats['tokens']['generated_total']} "
          f"({run_stats['tokens']['generated_per_second']} tok/s)")
    print(f"  Peak GPU memory      : {run_meta['peak_vram_gb']} / {run_meta['total_vram_gb']} GB")
    print(f"  Latency (batched)    : {run_stats['latency_batched']['mean_seconds_per_sample']} s/sample")
    if latency_single:
        print(f"  Latency (bs=1 median): {latency_single['median_seconds']} s")
    print(f"  Wall time            : {run_stats['wall_seconds']} s")
    print("=" * 62)
    print(f"Artefacts: {out_dir}")
    if args.strategy != "zero_shot":
        print("Compare against the other Task 1 prompting strategies with:")
        print("  python scripts/04_summarize_runs.py")


if __name__ == "__main__":
    main()
