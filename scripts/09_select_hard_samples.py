#!/usr/bin/env python
"""Select Mixed Sentiment and Contrastive example reviews from train.csv.

Candidate pool (shared by both outputs): reviews whose word length falls
between the Q1 and Q2 (median) of the whole train.csv length distribution,
with mixedness >= 0.5 and sent_count >= 6.

  - mixedness: sentence-level positive/negative balance, using NLTK's
    Hu & Liu opinion lexicon. Each sentence is scored pos/neg/neutral by
    which lexicon has more hits; mixedness = 2*min(pos,neg)/(pos+neg).
    1.0 = perfectly balanced, 0.0 = one-sided.
  - sent_count: number of sentences (NLTK sent_tokenize).

Outputs (data/processed/):
  - mixed_sentiment_samples.csv: top 5 positive + top 5 negative candidates,
    ranked by mixedness (ties broken by sent_count).
  - contrastive_samples.csv: 5 positive/negative pairs whose reviews share
    the most content-word vocabulary (Jaccard on stopword-free word sets),
    picked greedily by descending Jaccard score with each review used once.

Usage:
    python scripts/09_select_hard_samples.py
"""

from __future__ import annotations

import argparse
import re
from itertools import product

import nltk
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils import LABELS, get_logger, resolve

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"[a-z']+")


def _ensure_nltk() -> tuple[set[str], set[str], set[str]]:
    """Download the small NLTK resources this script needs, if missing."""
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("opinion_lexicon", "corpora/opinion_lexicon"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    stopwords = set(nltk.corpus.stopwords.words("english"))
    from nltk.corpus import opinion_lexicon

    return stopwords, set(opinion_lexicon.positive()), set(opinion_lexicon.negative())


def _word_set(text: str, stopwords: set[str]) -> set[str]:
    """Content-word set for Jaccard: lowercase tokens, stopwords/short words dropped."""
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in stopwords and len(t) > 2}


def _sentence_sentiment_stats(
    text: str, pos_words: set[str], neg_words: set[str]
) -> tuple[int, int, int, float]:
    """Return (sent_count, n_pos_sent, n_neg_sent, mixedness) for one review."""
    sentences = nltk.sent_tokenize(text)
    n_pos = n_neg = 0
    for sent in sentences:
        tokens = TOKEN_RE.findall(sent.lower())
        pos_hits = sum(1 for t in tokens if t in pos_words)
        neg_hits = sum(1 for t in tokens if t in neg_words)
        if pos_hits > neg_hits:
            n_pos += 1
        elif neg_hits > pos_hits:
            n_neg += 1
    denom = n_pos + n_neg
    mixedness = 2 * min(n_pos, n_neg) / denom if denom > 0 else 0.0
    return len(sentences), n_pos, n_neg, mixedness


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Full candidate pool: length between Q1-Q2, mixedness >= 0.5, sent_count >= 6."""
    stopwords, pos_words, neg_words = _ensure_nltk()

    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    q1, q2 = df["n_words"].quantile([0.25, 0.5])
    logger.info("Word-length quartiles over train.csv: Q1=%.1f, Q2 (median)=%.1f", q1, q2)

    length_ok = df["n_words"].between(q1, q2)
    n_total = len(df)
    df = df.loc[length_ok].copy()
    logger.info("%d / %d rows within [Q1, Q2] word length", len(df), n_total)

    stats = df["text"].apply(lambda t: _sentence_sentiment_stats(t, pos_words, neg_words))
    df["sent_count"] = [s[0] for s in stats]
    df["n_pos_sent"] = [s[1] for s in stats]
    df["n_neg_sent"] = [s[2] for s in stats]
    df["mixedness"] = [s[3] for s in stats]

    candidates = df[(df["mixedness"] >= 0.5) & (df["sent_count"] >= 6)].copy()
    candidates["word_set"] = candidates["text"].apply(lambda t: _word_set(t, stopwords))
    logger.info("%d candidates pass mixedness>=0.5 and sent_count>=6", len(candidates))
    return candidates


def select_mixed_sentiment(candidates: pd.DataFrame, n_per_class: int = 5) -> pd.DataFrame:
    """Top-N per class by mixedness (ties broken by sent_count), from the candidate pool."""
    picked = []
    for label in LABELS:
        subset = candidates[candidates["label"] == label].sort_values(
            ["mixedness", "sent_count"], ascending=False
        )
        if len(subset) < n_per_class:
            logger.warning("Only %d mixed-sentiment candidates for label=%s (wanted %d)",
                            len(subset), label, n_per_class)
        picked.append(subset.head(n_per_class))
    out = pd.concat(picked, ignore_index=True)
    return out[["uid", "label", "n_words", "sent_count", "n_pos_sent", "n_neg_sent",
                "mixedness", "text"]]


def select_contrastive(candidates: pd.DataFrame, n_pairs: int = 5) -> pd.DataFrame:
    """Greedy Pos x Neg pairing by descending Jaccard overlap, each review used once."""
    pos_rows = candidates[candidates["label"] == "positive"]
    neg_rows = candidates[candidates["label"] == "negative"]

    scored = []
    for (pi, prow), (ni, nrow) in product(pos_rows.iterrows(), neg_rows.iterrows()):
        a, b = prow["word_set"], nrow["word_set"]
        union = a | b
        score = len(a & b) / len(union) if union else 0.0
        scored.append((score, prow["uid"], nrow["uid"]))
    scored.sort(key=lambda x: x[0], reverse=True)

    used_pos, used_neg, pairs = set(), set(), []
    for score, pos_uid, neg_uid in scored:
        if pos_uid in used_pos or neg_uid in used_neg:
            continue
        pairs.append((pos_uid, neg_uid, score))
        used_pos.add(pos_uid)
        used_neg.add(neg_uid)
        if len(pairs) == n_pairs:
            break

    if len(pairs) < n_pairs:
        logger.warning("Only found %d contrastive pairs (wanted %d)", len(pairs), n_pairs)

    rows = []
    by_uid = candidates.set_index("uid")
    for pair_id, (pos_uid, neg_uid, score) in enumerate(pairs, start=1):
        for uid in (pos_uid, neg_uid):
            r = by_uid.loc[uid]
            rows.append({
                "pair_id": pair_id,
                "jaccard_score": round(score, 4),
                "uid": uid,
                "label": r["label"],
                "n_words": r["n_words"],
                "text": r["text"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/train.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--n-pairs", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(resolve(args.input))
    candidates = build_candidates(df)

    mixed = select_mixed_sentiment(candidates, n_per_class=args.n_per_class)
    contrastive = select_contrastive(candidates, n_pairs=args.n_pairs)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = out_dir / "mixed_sentiment_samples.csv"
    contrastive_path = out_dir / "contrastive_samples.csv"
    mixed.to_csv(mixed_path, index=False)
    contrastive.to_csv(contrastive_path, index=False)

    logger.info("Wrote %d mixed-sentiment rows -> %s", len(mixed), mixed_path)
    logger.info("Wrote %d contrastive rows (%d pairs) -> %s",
                len(contrastive), len(contrastive) // 2, contrastive_path)


if __name__ == "__main__":
    main()
