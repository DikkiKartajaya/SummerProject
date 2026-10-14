#!/usr/bin/env python
"""Pre-select the "random"/"basic_random" few-shot strategies' demo reviews.

Draws --n-per-class positive + negative reviews (2+2 = 4 by default) from
data/processed/train.csv for a single seed - unfiltered, no length/quality
curation, which is the whole point of the "random"/"basic_random" (vs
"curated"/"basic") ablation - and writes them to
data/processed/random_fewshot_samples.csv.

This file is a *record* of the draw, not something read at prompt-build
time: prompts/task2_fewshot_random.txt and prompts/task2_fewshot_basic_random.txt
bake the actual example text (review + hand-authored JSON answer) in
directly, exactly like curated's/basic's demos, so build_messages() needs no
runtime templating for any strategy. The same role
selected_sample.csv/mixed_sentiment_samples.csv/contrastive_samples.csv
already play for the other strategies (see scripts/09_select_hard_samples.py):
documents which rows were selected and makes the draw reproducible/auditable,
while the prompt text itself is authored (or regenerated) from it as a
separate, deliberate step - re-running this script alone does not change
either .txt file.

Why a precomputed file instead of each strategy sampling train.csv
independently: "random" and "basic_random" must be built from the *exact
same* 4 demo reviews, so that pairing isolates the system prompt's
rules/definitions as the only variable, the same way "curated" vs "basic"
do. Drawing once here and hand-authoring both .txt files from that one
output guarantees that parity by construction.

One seed, one file, 4 rows: earlier versions of this script drew several
seeds and ran "random"/"basic_random" once per seed to average out a lucky
or unlucky draw. That added a seed dimension to every run directory and
report table for a strategy-selection ablation that only needs one fixed,
reproducible draw - so this version keeps exactly one.

Usage:
    python scripts/10_select_random_fewshot_samples.py
    python scripts/10_select_random_fewshot_samples.py --seed 123 --n-per-class 3
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import get_logger, resolve

logger = get_logger(__name__)

# Must match src/prompts2.py's expectation of a single-seed draw.
DEFAULT_SEED = 42
DEFAULT_N_PER_CLASS = 2


def select_samples(df: pd.DataFrame, seed: int, n_per_class: int) -> pd.DataFrame:
    """One row per sampled review, `n_per_class` per label. Unfiltered draw -
    no length/quality filtering - via `DataFrame.sample(random_state=seed)`,
    so the draw is reproducible from train.csv alone."""
    rows = []
    for label in ("positive", "negative"):
        subset = df[df["label"] == label].sample(n=n_per_class, random_state=seed)
        for _, r in subset.iterrows():
            rows.append({"seed": seed, "uid": r["uid"], "label": r["label"], "text": r["text"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/train.csv")
    ap.add_argument("--out", default="data/processed/random_fewshot_samples.csv")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="the one seed to draw this file's demos from")
    ap.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                     help="positive + negative reviews drawn (must match "
                          "curated's/basic's example count - see src/prompts2.py)")
    args = ap.parse_args()

    df = pd.read_csv(resolve(args.input))
    out = select_samples(df, args.seed, args.n_per_class)

    out_path = resolve(args.out)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows (seed=%d, 2 classes x %d/class) -> %s",
                len(out), args.seed, args.n_per_class, out_path)
    logger.info("  label counts: %s", out["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
