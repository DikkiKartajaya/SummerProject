#!/usr/bin/env python
"""Step 7c (optional) - compare every prompt-sensitivity run in one table.

The Task 2 counterpart of `scripts/06_summarize_task2_runs.py`, one level deeper:
sensitivity runs nest as
`outputs/task2_sensitivity/<run_id>/<axis>__<variant>/` (with a `baseline/`
per run). This walks both levels and reports one row per (run_id, variant),
reading each variant's `metrics_task2.json` + `run_meta.json` + `quality_task2.json`
- nothing is recomputed.

Every column `06_summarize_task2_runs.py` emits is kept (run_id, strategy,
label, seed, fewshot_seed, checkpoint, precision, source, n, accuracy,
accuracy_overall, macro_f1, invalid_output_rate, schema_compliance_rate,
mean_evidence_relevance, null_evidence_relevance, evidence_relevance_lift, ece,
peak_vram_gb, tokens_per_sec, sec_per_sample, wall_seconds), plus the columns
that identify a perturbation (axis, variant, phrasing, demo_order, schema) and
`delta_macro_f1` / `delta_accuracy_overall` vs. that run's own `baseline`
variant. `fewshot_seed` is always empty here - a sensitivity run's meta does
not record it - kept only for column parity with step 6.

Usage:
    python scripts/07c_summarize_prompt_sensitivity_runs.py
    python scripts/07c_summarize_prompt_sensitivity_runs.py --runs-dir outputs/task2_sensitivity
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import read_json, resolve

# Same variable set as scripts/06_summarize_task2_runs.py, in the same order.
_BASE_COLS = [
    "run_id", "strategy", "label", "seed", "fewshot_seed", "checkpoint",
    "precision", "source", "n", "accuracy", "accuracy_overall", "macro_f1",
    "invalid_output_rate", "schema_compliance_rate", "mean_evidence_relevance",
    "null_evidence_relevance", "evidence_relevance_lift", "ece", "peak_vram_gb",
    "tokens_per_sec", "sec_per_sample", "wall_seconds",
]
# Perturbation-identifying columns unique to the sensitivity runs.
_AXIS_COLS = ["axis", "variant", "phrasing", "demo_order", "schema"]
_DELTA_COLS = ["delta_macro_f1", "delta_accuracy_overall"]


def _row(run_id: str, strat_dir, m: dict, meta: dict, q: dict) -> dict:
    return {
        "run_id": run_id,
        "strategy": meta.get("strategy"),
        "label": strat_dir.name,
        "seed": meta.get("seed"),
        "fewshot_seed": meta.get("fewshot_seed"),  # not recorded for sensitivity runs
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
        # null baseline (same evidence vs. a random other review) and the lift
        # over it - a mean_relevance that doesn't clear the null is generic
        # vocabulary overlap, not grounding.
        "null_evidence_relevance": (q.get("evidence") or {}).get("null_relevance"),
        "evidence_relevance_lift": (q.get("evidence") or {}).get("relevance_lift"),
        "ece": (q.get("calibration") or {}).get("ece"),
        "peak_vram_gb": meta.get("peak_vram_gb"),
        "tokens_per_sec": (meta.get("tokens") or {}).get("generated_per_second"),
        "sec_per_sample": (meta.get("latency_batched") or {}).get("mean_seconds_per_sample"),
        "wall_seconds": meta.get("wall_seconds"),
        "axis": meta.get("axis"),
        "variant": meta.get("variant"),
        "phrasing": meta.get("phrasing"),
        "demo_order": meta.get("demo_order"),
        "schema": meta.get("schema"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize all Task 2 prompt-sensitivity runs.")
    ap.add_argument("--runs-dir", default="outputs/task2_sensitivity")
    ap.add_argument("--out", default="outputs/task2_sensitivity/runs_summary.csv")
    args = ap.parse_args()

    runs_dir = resolve(args.runs_dir)
    rows = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for strat_dir in sorted(p for p in run.iterdir() if p.is_dir()):
            mfile, metafile = strat_dir / "metrics_task2.json", strat_dir / "run_meta.json"
            if not (mfile.exists() and metafile.exists()):
                continue  # incomplete variant
            qfile = strat_dir / "quality_task2.json"
            q = read_json(qfile) if qfile.exists() else {}
            rows.append(_row(run.name, strat_dir, read_json(mfile), read_json(metafile), q))

    if not rows:
        raise SystemExit(f"No completed sensitivity runs found in {runs_dir}.")

    df = pd.DataFrame(rows)

    # delta vs. each run's own baseline variant (axis == "baseline")
    df["delta_macro_f1"] = pd.NA
    df["delta_accuracy_overall"] = pd.NA
    for run_id, grp in df.groupby("run_id"):
        base = grp[grp["axis"] == "baseline"]
        if base.empty:
            continue
        b_f1 = base["macro_f1"].iloc[0]
        b_acc = base["accuracy_overall"].iloc[0]
        mask = df["run_id"] == run_id
        df.loc[mask, "delta_macro_f1"] = (df.loc[mask, "macro_f1"] - b_f1).round(4)
        df.loc[mask, "delta_accuracy_overall"] = (df.loc[mask, "accuracy_overall"] - b_acc).round(4)

    df = df[_BASE_COLS[:3] + _AXIS_COLS + _BASE_COLS[3:] + _DELTA_COLS]
    # baseline first within each run, then by descending macro_f1
    df["_base_first"] = (df["axis"] != "baseline").astype(int)
    df = df.sort_values(["run_id", "_base_first", "macro_f1"], ascending=[True, True, False])
    df = df.drop(columns="_base_first")

    out = resolve(args.out)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
