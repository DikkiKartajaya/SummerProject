#!/usr/bin/env python
"""Step 2 - exploratory data analysis on the cleaned pool.

Reads data/processed/clean_pool.csv (written by 01_prepare_data.py) and writes
figures + eda_stats.json. Purely descriptive - never touches the split or the
model, and safe to re-run any time after step 1.

Usage:
    python scripts/02_run_eda.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src import dataset, eda
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EDA on the cleaned pool.")
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--out-dir", default="outputs/eda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = dataset.load_split(cfg, "all")
    eda.run(pool, out_dir=args.out_dir)  # logs its own one-line summary


if __name__ == "__main__":
    main()
