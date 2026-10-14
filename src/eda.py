"""Exploratory data analysis on the cleaned pool.

Produces a JSON of statistics plus three PNG figures. Everything here is
descriptive only - it never touches the split or the model.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import nltk

matplotlib.use("Agg")  # headless: write PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import plot_style
from .utils import get_logger, resolve, write_json

logger = get_logger(__name__)
plot_style.apply()

# Words carrying no sentiment signal, removed before the top-token counts.
try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# HTML/markup remnants and generic filler that swamped every label equally
STOPWORDS |= {
    "br", "www", "http", "https",
    "get", "like", "one", "see", "still", "really", "would", "even",
    "movie", "film",  # near-universal in a movie-review corpus, so not discriminative
}

TOKEN_RE = re.compile(r"[a-z']+")


def _fig_path(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.debug("Saved figure: %s", path.name)
    return str(path)


def _plot_label_distribution(df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    colors = plot_style.label_palette(counts.index)
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                hue=counts.index, palette=colors, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom",
                fontsize=9, color=plot_style.INK_SECONDARY)
    ax.set(title="Label distribution (cleaned pool)", xlabel="", ylabel="count")
    ax.margins(y=0.15)
    sns.despine(fig)
    return _save(fig, _fig_path(out_dir, "eda_label_distribution.png"))


def _plot_length_distribution(df: pd.DataFrame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["n_chars"], bins=60, ax=axes[0], color=plot_style.CATEGORICAL[0])
    axes[0].set(title="Characters per review", xlabel="characters", ylabel="count")
    colors = plot_style.label_palette(df["label"].unique())
    sns.boxplot(data=df, x="label", y="n_words", ax=axes[1],
                hue="label", palette=colors, legend=False)
    axes[1].set(title="Words per review by label", xlabel="", ylabel="words")
    sns.despine(fig)
    return _save(fig, _fig_path(out_dir, "eda_length_distribution.png"))


def _plot_top_tokens(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> str:
    """Most frequent content words per label - a cheap sanity check that the
    labels actually carry signal (e.g. 'love/great' under positive)."""
    labels = sorted(df["label"].unique())
    colors = plot_style.label_palette(labels)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        tokens = Counter()
        for text in df.loc[df["label"] == label, "text"]:
            tokens.update(
                t for t in TOKEN_RE.findall(text.lower())
                if t not in STOPWORDS and len(t) > 2
            )
        common = tokens.most_common(top_n)
        sns.barplot(x=[c for _, c in common], y=[w for w, _ in common],
                    ax=ax, color=colors[label])
        ax.set(title=f"Top tokens: {label}", xlabel="count", ylabel="")
    sns.despine(fig)
    return _save(fig, _fig_path(out_dir, "eda_top_tokens.png"))


def run(df: pd.DataFrame, out_dir: str | Path = "outputs/eda") -> dict[str, Any]:
    """Compute EDA statistics and write figures. Returns the stats dict."""
    out_dir = resolve(out_dir)
    df = df.copy()
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()

    stats: dict[str, Any] = {
        "n_rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "label_ratios": df["label"].value_counts(normalize=True).round(4).to_dict(),
        # Imbalance ratio > ~3 would justify macro-F1 over plain accuracy.
        "imbalance_ratio": round(
            df["label"].value_counts().max() / df["label"].value_counts().min(), 3
        ),
        "text_length_chars": df["n_chars"].describe().round(2).to_dict(),
        "text_length_words": df["n_words"].describe().round(2).to_dict(),
        # 99th percentile guides the max_input_tokens setting in the config.
        "words_p95": float(df["n_words"].quantile(0.95)),
        "words_p99": float(df["n_words"].quantile(0.99)),
        "length_by_label": df.groupby("label")["n_words"].mean().round(2).to_dict(),
        "duplicate_texts_remaining": int(df["text"].duplicated().sum()),
    }

    stats["figures"] = [
        _plot_label_distribution(df, out_dir),
        _plot_length_distribution(df, out_dir),
        _plot_top_tokens(df, out_dir),
    ]

    write_json(stats, out_dir / "eda_stats.json")
    logger.info("EDA: %d rows, imbalance ratio %.2f, %d figures -> %s",
                stats["n_rows"], stats["imbalance_ratio"], len(stats["figures"]), out_dir)
    return stats
