#!/usr/bin/env python
"""Step 1 - download, clean, and create the frozen split ONCE.

Idempotent by design: re-running reuses the existing split files on disk.
Use --force only if you deliberately want to discard the old split.
EDA is a separate step - run scripts/02_run_eda.py after this one.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --force        # DANGER: re-splits
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src import dataset
from src.utils import get_logger, load_config, resolve, set_seed, write_json

logger = get_logger("prepare_data", log_file="outputs/logs/prepare_data.log")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download, clean and split the dataset.")
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split (normally forbidden)")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token; normally omitted so it is read from .env "
                         "(same credential used for the gated model checkpoint)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    logger.info("Cleaning %s [%s]", cfg["data"]["hf_dataset"], cfg["data"]["pool_split"])
    df = dataset.load_clean(cfg, hf_token=args.hf_token)
    if len(df) < 200:
        raise SystemExit(f"Only {len(df)} usable rows; the task requires at least 200.")

    # Cleaned with the exact same rules as the pool, then de-overlapped so it
    # stays genuinely independent.
    holdout = dataset.load_holdout(cfg, pool=df, hf_token=args.hf_token)

    manifest = dataset.build_split(cfg, df, holdout=holdout, force=args.force)
    write_json(manifest, resolve(cfg["split"]["out_dir"]) / "split_manifest.json")

    h = manifest["holdout"]
    print("\n===== Data ready =====")
    print(f"Pool (HF train split) : {manifest['pool_size']} rows")
    print(f"  split sizes         : {manifest['sizes']}")
    print(f"Holdout (HF test split): {h['n_rows']} rows"
          + ("" if h["present"] else "  (not configured - optional)"))
    print(f"Location              : {resolve(cfg['split']['out_dir'])}")
    print("\nRoles:")
    print("  train / val / test -> Tasks 2-4 (fine-tuning, model selection, final score)")
    print("  pool ('all')       -> Task 1 samples from here; no training, so no leakage")
    print("  holdout            -> independent cross-check, never split or trained on")
    print("\nNext: python scripts/02_run_eda.py")


if __name__ == "__main__":
    main()
