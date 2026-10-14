"""Metrics and reporting for Task 1.

Reports: accuracy, macro-F1, per-class precision/recall/F1, a confusion
matrix, the invalid-output rate, and (via run_meta) latency/tokens/GPU memory.
Headline metric is macro-F1, not accuracy: the pool is imbalanced, and a
model that ignores "neutral" entirely can still look decent on accuracy.

Accuracy is computed over valid (parseable) rows only - it answers "of the
times the model committed to a label, how often was it right?" - and is kept
separate from invalid_output_rate, which answers "how often did it fail to
commit at all?" A model that refuses to answer half the time and gets the
rest perfect should not look 50% accurate. Rows the model never produced a
valid label for still count as wrong for macro-F1 and the per-class/confusion
breakdowns (they show up as their own "unparsed" row/column there), since
those are meant to reflect overall label-quality including parse failures.

A third figure, accuracy_overall, folds invalid_output_rate back into a
single end-to-end number: correct / all rows, so an unparsed row counts as
wrong instead of being dropped from the denominator. It answers "of every
review the model was asked to classify, how often did the full pipeline
(generate + parse) land on the right label?" and is what should be quoted
when comparing strategies that differ in how often they fail to parse -
`accuracy` alone can make a high-invalid-rate strategy look artificially
strong since its denominator is smaller. Equivalent to
accuracy * (1 - invalid_output_rate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import plot_style
from .utils import LABELS, UNPARSED, get_logger, resolve, write_json

logger = get_logger(__name__)
plot_style.apply()


def _confusion(preds: pd.DataFrame) -> tuple[list[str], list[str], list[list[int]]]:
    """Confusion matrix: gold is always one of LABELS, predicted may also be
    'unparsed'. Rectangular (3 gold rows x up to 4 predicted columns) rather
    than square, since gold never takes the 'unparsed' value."""
    pred_labels = LABELS + ([UNPARSED] if (~preds["predicted"].isin(LABELS)).any() else [])
    y_pred = preds["predicted"].fillna(UNPARSED).where(preds["predicted"].isin(LABELS), UNPARSED)
    cm = pd.crosstab(preds["gold"], y_pred).reindex(
        index=LABELS, columns=pred_labels, fill_value=0
    )
    return LABELS, pred_labels, cm.to_numpy().tolist()


def compute(preds: pd.DataFrame) -> dict[str, Any]:
    """Metrics dict for a predictions dataframe (needs gold + predicted)."""
    y_true = preds["gold"].tolist()
    y_pred = preds["predicted"].fillna(UNPARSED).tolist()

    per_class = classification_report(y_true, y_pred, labels=LABELS,
                                       output_dict=True, zero_division=0)
    gold_labels, pred_labels, matrix = _confusion(preds)
    n = len(preds)
    valid_mask = preds["predicted"].isin(LABELS)
    n_valid = int(valid_mask.sum())
    n_invalid = n - n_valid

    return {
        "n": n,
        "n_valid": n_valid,
        # Valid-only: excludes unparsed rows from both numerator and
        # denominator, so failing to produce a label doesn't get scored as
        # a wrong label. See module docstring.
        "accuracy": round(accuracy_score(
            preds.loc[valid_mask, "gold"], preds.loc[valid_mask, "predicted"]
        ), 4) if n_valid else None,
        # Overall/end-to-end accuracy: unparsed rows count as wrong instead of
        # being dropped, so it folds invalid_output_rate into a single number.
        # y_pred already maps unparsed rows to UNPARSED, which never matches a
        # gold label, so they score as incorrect automatically. See module
        # docstring.
        "accuracy_overall": round(accuracy_score(y_true, y_pred), 4) if n else None,
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=LABELS), 4),
        "per_class": {k: v for k, v in per_class.items() if k in LABELS},
        "confusion_matrix": {
            "gold_labels": gold_labels, "predicted_labels": pred_labels, "matrix": matrix,
        },
        "invalid_output_rate": round(n_invalid / n, 4) if n else None,
    }


def plot_confusion(preds: pd.DataFrame, out_dir: str | Path, tag: str = "task1") -> str:
    """Confusion matrix heatmap (counts + row-normalised)."""
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_labels, pred_labels, matrix = _confusion(preds)
    cm = pd.DataFrame(matrix, index=gold_labels, columns=pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    heatmap_kwargs = dict(
        cmap=plot_style.SEQUENTIAL_CMAP, linewidths=1, linecolor=plot_style.SURFACE,
        annot_kws={"fontsize": 9},
    )
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cbar=False, **heatmap_kwargs)
    axes[0].set(title="Confusion matrix (counts)", xlabel="predicted", ylabel="gold")

    # row-normalised = per-class recall on the diagonal
    cm_norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=axes[1], vmin=0, vmax=1, cbar=False,
                **heatmap_kwargs)
    axes[1].set(title="Row-normalised (recall)", xlabel="predicted", ylabel="")
    for ax in axes:
        ax.tick_params(axis="both", length=0)

    path = out_dir / f"confusion_matrix_{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.debug("Saved figure: %s", path.name)
    return str(path)


_TASK_TITLES = {
    "task1": "Task 1 - Zero-Shot Sentiment Analysis",
    "task2": "Task 2 - Few-Shot Sentiment Analysis",
}


def to_markdown(metrics: dict[str, Any], run_meta: dict[str, Any], tag: str = "task1") -> str:
    """Render a compact human-readable results summary."""
    if tag == "task1":
        # Only shown when set: plain Task 1 runs have one prompt, so it's
        # noise there, but ablations (e.g. CARP, see prompt_method) need it.
        method = run_meta.get("prompt_method")
        method_part = f"  |  Prompt method: `{method}`" if method else ""
        precision_line = f"- Precision: `{run_meta.get('precision')}`  |  Seed: `{run_meta.get('seed')}`{method_part}"
    else:
        precision_line = (
            f"- Precision: `{run_meta.get('precision')}`  |  Prompt: `{run_meta.get('prompt_version')}`"
            f"  |  Seed: `{run_meta.get('seed')}`"
        )
    lines = [
        f"# {_TASK_TITLES.get(tag, tag)}: Results",
        "",
        f"- Checkpoint: `{run_meta.get('checkpoint')}` (revision `{run_meta.get('revision')}`)",
        precision_line,
        f"- Evaluated on {metrics['n']} examples from `{run_meta.get('source')}`",
    ]
    # Seed / eval-sample note: how the scored rows were drawn. Two runs with
    # the same seed/source/n/stratified (against the same frozen split) saw
    # the identical eval subset. Only present for runs that record it (Task 2).
    es = run_meta.get("eval_sample")
    if es:
        lines.append(
            f"- Eval sample: seed `{es.get('seed')}`, source `{es.get('source')}`, "
            f"n `{es.get('n_evaluated')}`, stratified `{es.get('stratified')}`"
        )
    lines += [
        "",
        "## Classification metrics",
        "",
        f"- Accuracy (valid outputs only, n={metrics['n_valid']}): **{metrics['accuracy']}**",
        f"- Accuracy (overall, unparsed counted as wrong, n={metrics['n']}): **{metrics['accuracy_overall']}**",
        f"- Macro F1: **{metrics['macro_f1']}**",
        "",
        "## Per-class",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label in LABELS:
        c = metrics["per_class"].get(label, {})
        lines.append(
            f"| {label} | {c.get('precision', 0):.3f} | {c.get('recall', 0):.3f} "
            f"| {c.get('f1-score', 0):.3f} | {int(c.get('support', 0))} |"
        )

    cm = metrics["confusion_matrix"]
    lines += [
        "",
        "## Confusion matrix (rows = gold, columns = predicted)",
        "",
        "| gold \\ pred | " + " | ".join(cm["predicted_labels"]) + " |",
        "|---" * (len(cm["predicted_labels"]) + 1) + "|",
    ]
    for label, row in zip(cm["gold_labels"], cm["matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")

    tokens = run_meta.get("tokens") or {}
    lines += [
        "",
        "## Reliability",
        "",
        f"- Invalid-output rate: **{metrics['invalid_output_rate']:.2%}**",
        "",
        "## Performance",
        "",
        f"- Latency (batched, mean/sample): {run_meta.get('latency_batched', {}).get('mean_seconds_per_sample')} s",
        f"- Latency (single-request, median): {run_meta.get('latency_single', {}).get('median_seconds')} s",
        f"- Tokens generated: {tokens.get('generated_total')} ({tokens.get('generated_per_second')} tok/s)",
        f"- Peak GPU memory: {run_meta.get('peak_vram_gb')} GB of {run_meta.get('total_vram_gb')} GB",
        "",
    ]
    return "\n".join(lines)


def save_all(
    preds: pd.DataFrame, metrics: dict[str, Any], run_meta: dict[str, Any],
    out_dir: str | Path, tag: str = "task1",
) -> dict[str, str]:
    """Write metrics.json, results.md and the confusion matrix figure."""
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics_json": str(write_json(metrics, out_dir / f"metrics_{tag}.json")),
        "confusion_matrix": plot_confusion(preds, out_dir, tag),
    }
    md_path = out_dir / f"results_{tag}.md"
    md_path.write_text(to_markdown(metrics, run_meta, tag=tag), encoding="utf-8")
    paths["results_md"] = str(md_path)

    csv_path = out_dir / f"predictions_{tag}.csv"
    preds.to_csv(csv_path, index=False)
    paths["predictions_csv"] = str(csv_path)

    # error slice: quick qualitative inspection of what the model got wrong
    errors = preds[preds["gold"] != preds["predicted"]]
    err_path = out_dir / f"errors_{tag}.csv"
    errors.to_csv(err_path, index=False)
    paths["errors_csv"] = str(err_path)

    # invalid-output slice: rows that never produced a parseable label
    invalid = preds[~preds["predicted"].isin(LABELS)]
    invalid_path = out_dir / f"invalid_outputs_{tag}.csv"
    invalid.to_csv(invalid_path, index=False)
    paths["invalid_outputs_csv"] = str(invalid_path)

    logger.info("Saved metrics + figures to %s", out_dir)
    return paths
