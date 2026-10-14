#!/usr/bin/env python
"""Step 12 - Task 3: QLoRA fine-tune Llama-3.1-8B-Instruct for binary sentiment.

One `--seed` = one full cycle: load the base model in 4-bit NF4 (double
quant), attach a LoRA adapter, build the instruction-formatted training set
from `data/processed/train.csv`, train, pick the checkpoint with the best
*validation macro-F1*, and score that adapter on the full val + test splits.
`--seeds` runs several and reports the mean / std (task sheet requirement 5).

What the task sheet asks for, and where it lands:
  1. 4-bit NF4 + double quant, bf16 compute when supported  -> src/qlora.load_base_model
  2. prepare_model_for_kbit_training + gradient checkpointing + LoRA targets
                                                             -> src/qlora
  3. r / alpha / dropout / targets / trainable %  -> printed + lora_summary.json
  4. instruction format, train on train split only, select by val macro-F1
                                       -> src/instruction_format + MacroF1Trainer
  5. train/val loss, LR, time, peak GPU memory, >=2 seeds, diagrams
                     -> training_log.csv/json, run_meta.json, summary.*, *.png

Produces, under outputs/task3/<run_id>/:
    run_config.json / env_record.json
    seed_<seed>/
        lora_summary.json
        training_log.csv / training_log.json
        training_curves_seed<seed>.png
        metrics_{val,test}_task3.json / predictions_{val,test}_task3.csv
        confusion_matrix_{val,test}_task3.png
        adapter/                      the selected LoRA adapter
        run_meta.json / results_task3.md
    summary.json / summary.md
    curves_across_seeds.png / macro_f1_by_seed.png / resources_by_seed.png

Usage:
    python scripts/12_run_task3_qlora.py
    python scripts/12_run_task3_qlora.py --seeds 42 1337 2024
    python scripts/12_run_task3_qlora.py --max-train-samples all --epochs 2 --run-id full
    python scripts/12_run_task3_qlora.py --seeds 42 --max-train-samples 500   # quick smoke
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime

import pandas as pd

import _bootstrap  # noqa: F401
from src import dataset, env_record, plot_task3, train_task3
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("task3")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Task 3: QLoRA fine-tuning.")
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--run-id", default=None, help="output subfolder (default: qlora_<timestamp>)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="random seeds; overrides config `seeds` (default: [42, 1337])")
    ap.add_argument("--max-train-samples", default=None,
                    help="rows sampled (stratified) from train.csv, or 'all'")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--select-max-samples", type=int, default=None,
                    help="val rows used for macro-F1 checkpoint selection during training")
    ap.add_argument("--final-max-samples", default=None,
                    help="cap the final val/test scoring set, or 'all'")
    ap.add_argument("--hf-token", default=None, help="normally omitted; read from .env")
    return ap.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.max_train_samples is not None:
        v = args.max_train_samples
        cfg["train"]["max_train_samples"] = None if str(v).lower() == "all" else int(v)
    if args.epochs is not None:
        cfg["train"]["num_train_epochs"] = args.epochs
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.select_max_samples is not None:
        cfg["eval"]["select_max_samples"] = args.select_max_samples
    if args.final_max_samples is not None:
        v = args.final_max_samples
        cfg["eval"]["final_max_samples"] = None if str(v).lower() == "all" else int(v)
    return cfg


def _stratified(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Stratified subsample keeping the label balance; n=None -> the whole df."""
    return dataset.sample_for_eval(df, n=n, seed=seed, stratified=True)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    seeds = list(cfg.get("seeds") or [cfg.get("seed", 42)])

    run_id = args.run_id or f"qlora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = resolve(cfg["out_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run id: %s  ->  %s  |  seeds: %s", run_id, out_dir, seeds)

    set_seed(cfg.get("seed", 42))
    env = env_record.collect()
    write_json(env, out_dir / "env_record.json")
    print(env_record.summarize(env))
    write_json(cfg, out_dir / "run_config.json")

    # --- frozen split (shared with Tasks 1-2) ---------------------------
    train_full = dataset.load_split(cfg, cfg["train"]["source"])
    val_full = dataset.load_split(cfg, "val")
    test_full = dataset.load_split(cfg, "test")

    n_train = cfg["train"]["max_train_samples"]
    select_n = cfg["eval"].get("select_max_samples")
    logger.info("train.csv: %d rows (using %s) | val: %d | test: %d",
                len(train_full), "all" if n_train is None else n_train,
                len(val_full), len(test_full))

    # The train subsample is drawn once from the base seed so every seed
    # trains on the identical rows - the only thing that varies across seeds
    # is initialisation + data order (Trainer's `seed`/`data_seed`).
    train_df = _stratified(train_full, None if n_train is None else int(n_train), cfg.get("seed", 42))
    select_val_df = _stratified(val_full, None if not select_n else int(select_n), cfg.get("seed", 42))

    # --- one full cycle per seed --------------------------------------
    seed_results, log_dfs = [], {}
    for seed in seeds:
        logger.info("=" * 70)
        logger.info("SEED %s", seed)
        logger.info("=" * 70)
        res = train_task3.train_one_seed(
            cfg, seed, train_df, val_full, test_full, select_val_df,
            seed_dir=out_dir / f"seed_{seed}", hf_token=args.hf_token,
        )
        seed_results.append(res)
        log_dfs[seed] = pd.read_csv(res["log_csv"])
        write_json({"seeds_done": [r["seed"] for r in seed_results]}, out_dir / "_progress.json")

    # --- aggregate + diagrams ---------------------------------------
    logger.info("All %d seed(s) done; aggregating summary + cross-seed diagrams ...", len(seeds))
    summary = _summarize(cfg, run_id, seeds, seed_results)
    write_json(summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    figs = plot_task3.plot_cross_seed(seed_results, log_dfs, out_dir)
    logger.info("Summary + diagrams written to %s", out_dir)

    _print_console(summary, out_dir, figs)


def _agg(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(vals), 4),
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def _summarize(cfg: dict, run_id: str, seeds: list[int], results: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "checkpoint": cfg["model"]["checkpoint"],
        "seeds": seeds,
        "lora": {k: cfg["lora"][k] for k in ("r", "alpha", "dropout", "target_modules")},
        "train": {
            "max_train_samples": cfg["train"]["max_train_samples"],
            "num_train_epochs": cfg["train"]["num_train_epochs"],
            "effective_batch_size": cfg["train"]["per_device_train_batch_size"]
            * cfg["train"]["gradient_accumulation_steps"],
            "learning_rate": cfg["train"]["learning_rate"],
        },
        "per_seed": results,
        "aggregate": {
            "test_macro_f1": _agg([r["test"]["macro_f1"] for r in results]),
            "val_macro_f1": _agg([r["val"]["macro_f1"] for r in results]),
            "test_accuracy_overall": _agg([r["test"]["accuracy_overall"] for r in results]),
            "test_invalid_output_rate": _agg([r["test"]["invalid_output_rate"] for r in results]),
            "best_eval_macro_f1": _agg([r["best_eval_macro_f1"] for r in results]),
            "train_seconds": _agg([r["train_seconds"] for r in results]),
            "peak_reserved_gb": _agg([r["peak_reserved_gb"] for r in results]),
        },
    }


def _summary_md(s: dict) -> str:
    a = s["aggregate"]
    lines = [
        f"# Task 3 - QLoRA fine-tune: summary ({s['run_id']})",
        "",
        f"- Base model: `{s['checkpoint']}`  |  seeds: {s['seeds']}",
        f"- LoRA: r={s['lora']['r']}, alpha={s['lora']['alpha']}, dropout={s['lora']['dropout']}",
        f"- target modules: `{' '.join(s['lora']['target_modules'])}`",
        f"- Train: {s['train']['max_train_samples']} samples, {s['train']['num_train_epochs']} epoch(s), "
        f"effective batch {s['train']['effective_batch_size']}, lr {s['train']['learning_rate']}",
        "",
        "## Headline (test split, across seeds)",
        "",
        f"- **Macro-F1: {a['test_macro_f1']['mean']} ± {a['test_macro_f1']['std']}** "
        f"(min {a['test_macro_f1']['min']}, max {a['test_macro_f1']['max']}, n={a['test_macro_f1']['n']})",
        f"- Accuracy (overall): {a['test_accuracy_overall']['mean']} ± {a['test_accuracy_overall']['std']}",
        f"- Invalid-output rate: {a['test_invalid_output_rate']['mean']} ± {a['test_invalid_output_rate']['std']}",
        "",
        "## Per seed",
        "",
        "| Seed | Best step | Val F1 (select) | Val F1 (full) | Test F1 | Test acc (overall) | Train time (min) | Peak VRAM (GB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in s["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['best_step']} | {r['best_eval_macro_f1']} | "
            f"{r['val']['macro_f1']} | {r['test']['macro_f1']} | {r['test']['accuracy_overall']} | "
            f"{round(r['train_seconds'] / 60, 1)} | {r['peak_reserved_gb']} |"
        )
    lines += [
        "",
        "## Resources (across seeds)",
        "",
        f"- Training time: {a['train_seconds']['mean']} s ± {a['train_seconds']['std']} "
        f"({round((a['train_seconds']['mean'] or 0) / 60, 1)} min avg)",
        f"- Peak GPU memory: {a['peak_reserved_gb']['mean']} GB ± {a['peak_reserved_gb']['std']} (torch reserved)",
        "",
        "## Diagrams",
        "",
        "- `curves_across_seeds.png` - train/val loss + validation macro-F1 over steps",
        "- `macro_f1_by_seed.png` - val/test macro-F1 per seed with mean ± std",
        "- `resources_by_seed.png` - training time + peak GPU memory per seed",
        "- `seed_<seed>/training_curves_seed<seed>.png` - per-seed loss / F1 / LR",
        "",
    ]
    return "\n".join(lines)


def _print_console(summary: dict, out_dir, figs) -> None:
    a = summary["aggregate"]
    print("\n" + "=" * 66)
    print(f"TASK 3 - QLoRA RESULTS  ({summary['run_id']})")
    print("=" * 66)
    print(f"  Base model        : {summary['checkpoint']}")
    print(f"  Seeds             : {summary['seeds']}")
    print(f"  LoRA r/alpha/drop : {summary['lora']['r']} / {summary['lora']['alpha']} / {summary['lora']['dropout']}")
    for r in summary["per_seed"]:
        print(f"   - seed {r['seed']:>5}: test macro-F1 {r['test']['macro_f1']:.4f}  "
              f"| val {r['val']['macro_f1']:.4f}  | best step {r['best_step']}  "
              f"| {round(r['train_seconds'] / 60, 1)} min  | {r['peak_reserved_gb']} GB")
    print("-" * 66)
    print(f"  Test macro-F1     : {a['test_macro_f1']['mean']} ± {a['test_macro_f1']['std']}")
    print(f"  Test invalid rate : {a['test_invalid_output_rate']['mean']}")
    print(f"  Train time (avg)  : {round((a['train_seconds']['mean'] or 0) / 60, 1)} min")
    print(f"  Peak VRAM (avg)   : {a['peak_reserved_gb']['mean']} GB")
    print("=" * 66)
    print(f"Artefacts: {out_dir}")
    for f in figs:
        print(f"  figure: {f}")


if __name__ == "__main__":
    main()
