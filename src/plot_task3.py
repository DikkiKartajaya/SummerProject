"""Diagrams for Task 3 QLoRA (task sheet requirement 5).

Per seed (`plot_seed_curves`):
  - training vs. validation loss over steps
  - validation macro-F1 over steps, with the selected checkpoint marked
  - learning-rate schedule

Across seeds (`plot_cross_seed`):
  - loss curves overlaid (one colour per seed)
  - validation macro-F1 overlaid
  - test / val macro-F1 per seed as bars with the mean +- std
  - training time and peak GPU memory per seed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_style
from .utils import get_logger, resolve

logger = get_logger(__name__)
plot_style.apply()

_C = plot_style.CATEGORICAL


def _series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows of `df` where both `step` and `col` are present, sorted by step."""
    if col not in df.columns:
        return pd.DataFrame(columns=["step", col])
    out = df[["step", col]].dropna()
    return out.sort_values("step")


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return str(path)


# --------------------------------------------------------------------------
def plot_seed_curves(log_df: pd.DataFrame, seed: int, out_dir: str | Path) -> str:
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    tr, ev = _series(log_df, "train_loss"), _series(log_df, "eval_loss")
    axes[0].plot(tr["step"], tr["train_loss"], color=_C[0], label="train loss", lw=1.6)
    if not ev.empty:
        axes[0].plot(ev["step"], ev["eval_loss"], color=_C[1], marker="o", ms=4,
                     label="val loss", lw=1.4)
    axes[0].set(title=f"Loss (seed {seed})", xlabel="step", ylabel="loss")
    axes[0].legend()

    f1 = _series(log_df, "eval_macro_f1")
    if not f1.empty:
        axes[1].plot(f1["step"], f1["eval_macro_f1"], color=_C[2], marker="o", ms=4, lw=1.6)
        best = f1.loc[f1["eval_macro_f1"].idxmax()]
        axes[1].axvline(best["step"], color=plot_style.BASELINE, ls="--", lw=1)
        axes[1].scatter([best["step"]], [best["eval_macro_f1"]], color=_C[7], zorder=5,
                        label=f"selected: step {int(best['step'])} (F1={best['eval_macro_f1']:.3f})")
        axes[1].legend()
    axes[1].set(title=f"Validation macro-F1 (seed {seed})", xlabel="step", ylabel="macro-F1")

    lr = _series(log_df, "learning_rate")
    axes[2].plot(lr["step"], lr["learning_rate"], color=_C[3], lw=1.6)
    axes[2].set(title=f"Learning rate (seed {seed})", xlabel="step", ylabel="lr")

    return _save(fig, out_dir / f"training_curves_seed{seed}.png")


# --------------------------------------------------------------------------
def plot_cross_seed(
    seed_results: list[dict[str, Any]],
    log_dfs: dict[int, pd.DataFrame],
    out_dir: str | Path,
) -> list[str]:
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # --- 1. loss + macro-F1 curves overlaid across seeds -----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (seed, df) in enumerate(sorted(log_dfs.items())):
        col = _C[i % len(_C)]
        tr, ev = _series(df, "train_loss"), _series(df, "eval_loss")
        axes[0].plot(tr["step"], tr["train_loss"], color=col, lw=1.4, label=f"seed {seed} (train)")
        if not ev.empty:
            axes[0].plot(ev["step"], ev["eval_loss"], color=col, lw=1.3, ls="--", marker="o",
                         ms=3, label=f"seed {seed} (val)")
        f1 = _series(df, "eval_macro_f1")
        if not f1.empty:
            axes[1].plot(f1["step"], f1["eval_macro_f1"], color=col, lw=1.6, marker="o", ms=3,
                         label=f"seed {seed}")
    axes[0].set(title="Loss across seeds", xlabel="step", ylabel="loss")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Validation macro-F1 across seeds", xlabel="step", ylabel="macro-F1")
    axes[1].legend(fontsize=8)
    paths.append(_save(fig, out_dir / "curves_across_seeds.png"))

    # --- 2. per-seed test/val macro-F1 with mean +- std ------------------
    seeds = [r["seed"] for r in seed_results]
    test_f1 = np.array([r["test"]["macro_f1"] for r in seed_results], dtype=float)
    val_f1 = np.array([r["val"]["macro_f1"] for r in seed_results], dtype=float)
    x = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(1.6 * len(seeds) + 3, 4.5))
    ax.bar(x - 0.2, val_f1, width=0.38, color=_C[0], label="val")
    ax.bar(x + 0.2, test_f1, width=0.38, color=_C[1], label="test")
    for xi, vv, tv in zip(x, val_f1, test_f1):
        ax.text(xi - 0.2, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, tv, f"{tv:.3f}", ha="center", va="bottom", fontsize=8)
    if len(test_f1) > 1:
        ax.axhline(test_f1.mean(), color=_C[1], ls="--", lw=1)
        ax.text(x[-1] + 0.45, test_f1.mean(),
                f"test mean {test_f1.mean():.3f} ± {test_f1.std():.3f}",
                va="center", fontsize=8, color=plot_style.INK_SECONDARY)
    ax.set(title="Macro-F1 by seed", xlabel="seed", ylabel="macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.margins(y=0.18)
    ax.legend()
    paths.append(_save(fig, out_dir / "macro_f1_by_seed.png"))

    # --- 3. training time + peak GPU memory per seed --------------------
    mins = [r["train_seconds"] / 60 for r in seed_results]
    mem = [r["peak_reserved_gb"] for r in seed_results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(x, mins, color=_C[4], width=0.5)
    for xi, mv in zip(x, mins):
        axes[0].text(xi, mv, f"{mv:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set(title="Training time by seed", xlabel="seed", ylabel="minutes")
    axes[0].set_xticks(x); axes[0].set_xticklabels(seeds); axes[0].margins(y=0.18)

    axes[1].bar(x, mem, color=_C[6], width=0.5)
    for xi, mv in zip(x, mem):
        axes[1].text(xi, mv, f"{mv:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].set(title="Peak GPU memory by seed", xlabel="seed", ylabel="GB (torch reserved)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(seeds); axes[1].margins(y=0.18)
    paths.append(_save(fig, out_dir / "resources_by_seed.png"))

    return paths
