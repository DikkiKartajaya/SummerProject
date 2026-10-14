#!/usr/bin/env python
"""Step 13 (optional) - compare every Task 3 run under outputs/task3/ in one table.

Task 3 runs nest as outputs/task3/<run_id>/seed_<seed>/, so this walks both
levels and reports one row per (run_id, seed) plus the run-level aggregate
read straight from summary.json. Nothing is recomputed.

Usage:
    python scripts/13_summarize_task3_runs.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import read_json, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize all Task 3 runs.")
    ap.add_argument("--runs-dir", default="outputs/task3")
    ap.add_argument("--out", default="outputs/task3/runs_summary.csv")
    args = ap.parse_args()

    runs_dir = resolve(args.runs_dir)
    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summ = run / "summary.json"
        agg = read_json(summ).get("aggregate", {}) if summ.exists() else {}
        for seed_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            meta_f = seed_dir / "run_meta.json"
            if not meta_f.exists():
                continue
            meta = read_json(meta_f)
            fm = meta.get("final_metrics", {})
            lora = meta.get("lora", {})
            pk = meta.get("peak_gpu_memory", {})
            rows.append({
                "run_id": run.name,
                "seed": meta.get("seed"),
                "checkpoint": meta.get("checkpoint"),
                "compute_dtype": meta.get("compute_dtype"),
                "lora_r": lora.get("lora_r"),
                "lora_alpha": lora.get("lora_alpha"),
                "lora_dropout": lora.get("lora_dropout"),
                "trainable_pct": lora.get("trainable_percentage"),
                "train_examples": (meta.get("train_dataset") or {}).get("n_examples"),
                "best_step": meta.get("best_step"),
                "best_eval_macro_f1": meta.get("best_eval_macro_f1"),
                "val_macro_f1": (fm.get("val") or {}).get("macro_f1"),
                "test_macro_f1": (fm.get("test") or {}).get("macro_f1"),
                "test_accuracy_overall": (fm.get("test") or {}).get("accuracy_overall"),
                "test_invalid_rate": (fm.get("test") or {}).get("invalid_output_rate"),
                "train_minutes": round((meta.get("train_seconds") or 0) / 60, 1),
                "peak_reserved_gb": pk.get("torch_peak_reserved_gb"),
                "run_test_f1_mean": (agg.get("test_macro_f1") or {}).get("mean"),
                "run_test_f1_std": (agg.get("test_macro_f1") or {}).get("std"),
            })

    if not rows:
        raise SystemExit(f"No completed Task 3 runs found in {runs_dir}.")

    df = pd.DataFrame(rows).sort_values(["run_id", "seed"])
    out = resolve(args.out)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
