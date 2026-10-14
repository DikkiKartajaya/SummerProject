#!/usr/bin/env python
"""Select the most *representative* few-shot demo reviews from train.csv.

Where scripts/10_select_random_fewshot_samples.py draws an unfiltered random
pair per class, this script picks the pair per class that sits closest to the
*centre* of its class in embedding space - the "prototype" reviews - using
`sentence-transformers/all-MiniLM-L6-v2` sentence embeddings.

Method
------
1. Optionally keep only reviews within [--min-words, --max-words] so the demos
   are short enough to paste into a prompt (the same reason curated's demos
   are ~25 words). Representativeness is then measured *within* that pool.
2. Embed every candidate review with all-MiniLM-L6-v2 (mean-pooled,
   L2-normalised - identical to the sentence-transformers default). Uses the
   `sentence-transformers` package if installed, otherwise a plain
   `transformers` mean-pool fallback that produces the same vectors.
3. For each label, the class centroid is the (re-normalised) mean of its
   members' vectors. Score each candidate by:
     - centroid_similarity: cosine(candidate, own-class centroid)   [ranked on]
     - margin: cosine(candidate, own centroid) - cosine(candidate, other centroid)
   High centroid_similarity => textually typical of the class; positive margin
   => unambiguously that class rather than a fence-sitter.
4. Greedily take the top --n-per-class by centroid_similarity, skipping any
   candidate that is near-duplicate (cosine > --dedup-threshold) of one
   already picked for the same class, so the two demos are not paraphrases.

Output (data/processed/representative_example.csv), one row per demo:
    uid, label, n_words, centroid_similarity, margin, text

Like random_fewshot_samples.csv, this file is a *record* of the selection,
not read at prompt-build time: prompts/task2_fewshot_representative.txt and
prompts/task2_fewshot_basic_representative.txt bake the chosen review text
(plus a hand-authored JSON answer) in directly, exactly like curated's/
random's demos. Re-running this script does not rewrite those .txt files -
regenerating the baked-in example text from the new CSV is a separate,
deliberate step.

Usage:
    python scripts/11_select_representative_samples.py
    python scripts/11_select_representative_samples.py --min-words 20 --max-words 150
    python scripts/11_select_representative_samples.py --n-per-class 3 --no-length-filter
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import LABELS, get_logger, resolve, set_seed

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEED = 42
DEFAULT_N_PER_CLASS = 2
DEFAULT_MIN_WORDS = 20
DEFAULT_MAX_WORDS = 150
DEFAULT_DEDUP_THRESHOLD = 0.9


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def _embed_with_sentence_transformers(texts: list[str]) -> np.ndarray | None:
    """Canonical path: the `sentence-transformers` package, if it is installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Embedding with sentence-transformers (%s) on %s", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model.encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )


def _embed_with_transformers(texts: list[str]) -> np.ndarray:
    """Fallback: plain `transformers`. Mean-pool the last hidden state over the
    attention mask, then L2-normalise - the exact recipe all-MiniLM-L6-v2's
    model card gives, so the vectors match the sentence-transformers output."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Embedding with transformers fallback (%s) on %s", MODEL_NAME, device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    out = []
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        enc = tok(batch, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy())
        logger.info("  embedded %d / %d", min(start + 64, len(texts)), len(texts))
    return np.vstack(out)


def embed(texts: list[str]) -> np.ndarray:
    vecs = _embed_with_sentence_transformers(texts)
    if vecs is None:
        vecs = _embed_with_transformers(texts)
    # Guard against any tiny drift from fp math so cosine == dot product.
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_representative(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_per_class: int,
    dedup_threshold: float,
) -> pd.DataFrame:
    """Top `n_per_class` reviews per label by cosine similarity to the class
    centroid, de-duplicated so no two picks are near-paraphrases."""
    df = df.reset_index(drop=True)
    centroids = {}
    for label in LABELS:
        member_vecs = embeddings[df.index[df["label"] == label]]
        c = member_vecs.mean(axis=0)
        centroids[label] = c / np.linalg.norm(c).clip(min=1e-12)

    rows = []
    for label in LABELS:
        idx = np.asarray(df.index[df["label"] == label])
        vecs = embeddings[idx]
        sim_own = vecs @ centroids[label]
        other = LABELS[1] if label == LABELS[0] else LABELS[0]
        margin = sim_own - (vecs @ centroids[other])

        order = idx[np.argsort(-sim_own)]
        picked: list[int] = []
        for cand in order:
            if any(float(embeddings[cand] @ embeddings[p]) > dedup_threshold for p in picked):
                continue
            picked.append(int(cand))
            if len(picked) == n_per_class:
                break
        if len(picked) < n_per_class:
            logger.warning("Only %d de-duplicated picks for label=%s (wanted %d)",
                            len(picked), label, n_per_class)

        pos_of = {int(v): i for i, v in enumerate(idx)}
        for cand in picked:
            r = df.loc[cand]
            rows.append({
                "uid": r["uid"],
                "label": label,
                "n_words": len(str(r["text"]).split()),
                "centroid_similarity": round(float(sim_own[pos_of[cand]]), 4),
                "margin": round(float(margin[pos_of[cand]]), 4),
                "text": r["text"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/processed/train.csv")
    ap.add_argument("--out", default="data/processed/representative_example.csv")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                     help="positive + negative demos to select (2 => 2 pos + 2 neg)")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    ap.add_argument("--no-length-filter", action="store_true",
                     help="rank over all of train.csv, ignoring --min/--max-words")
    ap.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD,
                     help="skip a candidate this cosine-similar to an already-picked "
                          "same-class demo")
    args = ap.parse_args()

    set_seed(args.seed)
    df = pd.read_csv(resolve(args.input))
    logger.info("Loaded %d rows from %s | %s",
                len(df), args.input, df["label"].value_counts().to_dict())

    if not args.no_length_filter:
        n_words = df["text"].str.split().str.len()
        keep = n_words.between(args.min_words, args.max_words)
        df = df.loc[keep].copy()
        logger.info("%d rows within [%d, %d] words kept as the candidate pool | %s",
                    len(df), args.min_words, args.max_words,
                    df["label"].value_counts().to_dict())

    embeddings = embed(df["text"].astype(str).tolist())
    out = select_representative(df, embeddings, args.n_per_class, args.dedup_threshold)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(out), out_path)
    for _, r in out.iterrows():
        logger.info("  [%s] sim=%.4f margin=%+.4f uid=%s | %s",
                    r["label"], r["centroid_similarity"], r["margin"], r["uid"],
                    str(r["text"])[:90].replace("\n", " "))


if __name__ == "__main__":
    main()
