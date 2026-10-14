#!/usr/bin/env python
"""Step 6 (optional) - compare every run under outputs/task2/ in one table.

Task 2 runs nest one level deeper than Task 1 (outputs/task2/<run_id>/<strategy>/),
since each run scores one or both few-shot strategies. This walks both levels,
reads each strategy's metrics + meta + quality files, and reports one row per
(run_id, strategy) pair; nothing is recomputed.

Usage:
    python scripts/06_summarize_task2_runs.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import read_json, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize all Task 2 runs.")
    ap.add_argument("--runs-dir", default="outputs/task2")
    ap.add_argument("--out", default="outputs/task2/runs_summary.csv")
    args = ap.parse_args()

    runs_dir = resolve(args.runs_dir)
    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for strat_dir in sorted(p for p in run.iterdir() if p.is_dir()):
            mfile, metafile = strat_dir / "metrics_task2.json", strat_dir / "run_meta.json"
            if not (mfile.exists() and metafile.exists()):
                continue  # incomplete run
            m, meta = read_json(mfile), read_json(metafile)
            qfile = strat_dir / "quality_task2.json"
            q = read_json(qfile) if qfile.exists() else {}
            rows.append({
                "run_id": run.name,
                "strategy": meta.get("strategy", strat_dir.name),
                "label": strat_dir.name,
                "seed": meta.get("seed"),
                "fewshot_seed": meta.get("fewshot_seed"),
                "checkpoint": meta.get("checkpoint"),
                "precision": meta.get("precision"),
                "source": meta.get("source"),
                "n": m["n"],
                "accuracy": m["accuracy"],
                "accuracy_overall": m["accuracy_overall"],
                "macro_f1": m["macro_f1"],
                "invalid_output_rate": m["invalid_output_rate"],
                "schema_compliance_rate": (q.get("format_compliance") or {}).get("schema_compliance_rate"),
                "mean_evidence_relevance": (q.get("evidence") or {}).get("mean_relevance"),
                # null baseline (same evidence vs. a random other review) and the
                # lift over it - a mean_relevance that doesn't clear the null is
                # generic vocabulary overlap, not grounding.
                "null_evidence_relevance": (q.get("evidence") or {}).get("null_relevance"),
                "evidence_relevance_lift": (q.get("evidence") or {}).get("relevance_lift"),
                "ece": (q.get("calibration") or {}).get("ece"),
                "peak_vram_gb": meta.get("peak_vram_gb"),
                "tokens_per_sec": (meta.get("tokens") or {}).get("generated_per_second"),
                "sec_per_sample": (meta.get("latency_batched") or {}).get("mean_seconds_per_sample"),
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
