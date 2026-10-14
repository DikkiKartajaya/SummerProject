#!/usr/bin/env python
"""Step 4 (optional) - compare every run under outputs/task1/ in one table.

Useful once you have ablations (bf16 vs nf4, different sample sizes, or the
three Task 1 prompts - plain zero-shot (03), CARP (07), and the basic prompt
(08) - all of which write here on purpose; see configs/task1*.yaml and the
`prompt_method` column below). Reads each run's metrics + meta files; nothing
is recomputed.

Usage:
    python scripts/04_summarize_runs.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import read_json, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize all Task 1 runs.")
    ap.add_argument("--runs-dir", default="outputs/task1")
    ap.add_argument("--out", default="outputs/task1/runs_summary.csv")
    args = ap.parse_args()

    runs_dir = resolve(args.runs_dir)
    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        mfile, metafile = run / "metrics_task1.json", run / "run_meta.json"
        if not (mfile.exists() and metafile.exists()):
            continue  # incomplete run
        m, meta = read_json(mfile), read_json(metafile)
        rows.append({
            "run_id": run.name,
            "prompt_method": meta.get("prompt_method", "zero_shot"),  # older runs predate this field
            "checkpoint": meta.get("checkpoint"),
            "precision": meta.get("precision"),
            "source": meta.get("source"),
            "n": m["n"],
            "accuracy": m["accuracy"],
            "accuracy_overall": m["accuracy_overall"],
            "macro_f1": m["macro_f1"],
            "invalid_output_rate": m["invalid_output_rate"],
            "peak_vram_gb": meta.get("peak_vram_gb"),
            "tokens_per_sec": (meta.get("tokens") or {}).get("generated_per_second"),
            "sec_per_sample": (meta.get("latency_batched") or {}).get("mean_seconds_per_sample"),
            "bs1_median_sec": (meta.get("latency_single") or {}).get("median_seconds"),
            "wall_seconds": meta.get("wall_seconds"),
        })

    if not rows:
        raise SystemExit(f"No completed runs found in {runs_dir}.")

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out = resolve(args.out)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
