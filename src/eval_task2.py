"""Task 2 quality metrics beyond label accuracy.

`metrics.py` (shared with Task 1) already covers final-label quality -
accuracy, macro-F1, per-class, confusion matrix, invalid-output rate. This
module covers the three dimensions that are specific to Task 2's richer
{label, confidence, evidence} schema:

- **evidence relevance** - is the one-sentence evidence actually grounded in
  the review, or a generic/fabricated sentence that happens to fit the label?
  Measured as content-word overlap between the evidence and the review text.
  Movie reviews share a lot of generic vocabulary ("film", "story", "acting",
  "character"), so even an ungrounded sentence scores a non-trivial overlap.
  A raw overlap number is therefore not interpretable on its own; we also
  report a **null baseline** - the same evidence scored against a *random
  other* review - and the **lift** over it. Lift near zero means the evidence
  is no more tied to *this* review than to any review.
- **confidence calibration** - when the model says "0.9 confident", is it
  right about 90% of the time? Measured with Expected Calibration Error (ECE)
  and Brier score against the model's own correctness. Rows without a numeric
  confidence are dropped from the ECE, and those rows correlate with invalid
  output - so a strategy with a high invalid rate is otherwise scored on an
  easier subset and its ECE is not comparable to a cleaner strategy's. We
  report `coverage` alongside every ECE, and accept an optional
  `common_index` so several strategies can be scored on an identical row set.
- **format compliance** - how often is the reply *exactly* the required
  3-key JSON schema (not just recoverable via the lenient extraction ladder)?
  The headline `schema_compliance_rate` counts unparseable rows as failures,
  so across strategies it largely re-measures the invalid-output rate. The
  `*_among_valid` rates restrict to rows that produced a usable label, which
  isolates format-following from parse success.

All are heuristic/deterministic (no extra model calls), matching the
project's "no training, no hidden judging model" baseline philosophy.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z']+")
_LABEL_ECHO_WORDS = ("positive", "negative", "label")

# 5 permutations, averaged, so the null baseline does not swing on one unlucky
# shuffle. Fixed seeds keep the whole metric deterministic.
_NULL_BASELINE_SEEDS = range(42, 47)

try:
    import nltk

    try:
        _STOPWORDS = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except ImportError:  # nltk is a hard dep elsewhere, but keep this module standalone-safe
    _STOPWORDS = set()


def _content_words(text: Any) -> set[str]:
    """Lowercase content words (stopwords and 2-letter words dropped)."""
    return {w for w in _WORD_RE.findall(str(text).lower()) if w not in _STOPWORDS and len(w) > 2}


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------
# 1. Evidence relevance
# --------------------------------------------------------------------------
def _overlap(ev_words: set[str], text_words: set[str]) -> float:
    """Fraction of the evidence's content words that also appear in the review."""
    return len(ev_words & text_words) / len(ev_words)


def _overlap_scores(ev_word_sets: list[set[str]], text_word_sets: list[set[str]]) -> list[float]:
    """Overlap for each aligned (evidence, review) pair. Both inputs must
    already be filtered to evidence-bearing rows and stay index-aligned."""
    return [_overlap(ev, txt) for ev, txt in zip(ev_word_sets, text_word_sets)]


def _derangement(n: int, seed: int) -> np.ndarray | None:
    """A permutation of ``range(n)`` with no fixed point (``perm[i] != i`` for
    every ``i``), derived deterministically from ``seed``.

    The null baseline is supposed to contain *zero* correct evidence-review
    pairings; a plain shuffle leaves some evidence matched with its own
    review, which inflates the "random" score. Forcing a derangement keeps the
    baseline honest. Returns ``None`` when ``n <= 1``, where no derangement
    exists (and the caller reports ``null_relevance = None``).
    """
    if n <= 1:
        return None
    perm = np.random.RandomState(seed).permutation(n)
    fixed = [i for i in range(n) if perm[i] == i]
    if len(fixed) == 1:
        # Value i sits at position i, so it cannot also sit at position i+1;
        # swapping i with its neighbour therefore always clears the fixed point.
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    elif len(fixed) > 1:
        # Cycle the fixed positions' values by one - each ends up holding a
        # different fixed index, so none stays put.
        rotated = fixed[1:] + fixed[:1]
        for pos, new_val in zip(fixed, rotated):
            perm[pos] = new_val
    return perm


def _empty_evidence(n: int, evidence_present_rate: float | None = None) -> dict[str, Any]:
    return {
        "n": n,
        "evidence_present_rate": evidence_present_rate,
        "mean_relevance": None,
        "median_relevance": None,
        "null_relevance": None,
        "relevance_lift": None,
        "mean_evidence_length_words": None,
        "label_echo_rate_among_present": None,
    }


def evaluate_evidence(preds: pd.DataFrame) -> dict[str, Any]:
    """Evidence-relevance summary for one strategy's predictions."""
    n_total = int(len(preds))
    if n_total == 0 or "evidence" not in preds.columns:
        return _empty_evidence(n_total)

    # Filter to evidence-bearing rows *first*, into two parallel lists. This
    # ordering matters for the null baseline: it permutes `txt_only`, so every
    # permuted pair is real-evidence vs. real-(other)-review. Filtering after
    # the shuffle instead (drop no-evidence rows post-zip) would permute over
    # all rows and only then discard some, shifting the pairing distribution.
    ev_only: list[set[str]] = []
    txt_only: list[set[str]] = []
    ev_texts: list[str] = []
    for e, t in zip(preds["evidence"].tolist(), preds["text"].tolist()):
        if not (isinstance(e, str) and e.strip()):
            continue
        ev_words = _content_words(e)
        if not ev_words:  # evidence was all stopwords / short words - nothing to score
            continue
        ev_only.append(ev_words)
        txt_only.append(_content_words(t))
        ev_texts.append(e)

    evidence_present_rate = round(len(ev_only) / n_total, 4)
    if not ev_only:
        return _empty_evidence(n_total, evidence_present_rate)

    real_scores = _overlap_scores(ev_only, txt_only)
    mean_relevance = _mean_or_none(real_scores)

    # Null baseline: the same evidence vs. a RANDOM OTHER review. Movie reviews
    # share generic vocabulary, so ungrounded evidence still overlaps
    # something; the real-minus-null gap is the actual grounding signal. Real
    # and null are computed from the identical filtered population. Averaged
    # over several fixed seeds so one unlucky shuffle can't dominate.
    m = len(txt_only)
    null_means: list[float] = []
    for seed in _NULL_BASELINE_SEEDS:
        perm = _derangement(m, seed)
        if perm is None:  # m <= 1: no derangement possible
            break
        shuffled = [txt_only[j] for j in perm]
        null_means.append(float(np.mean(_overlap_scores(ev_only, shuffled))))
    null_relevance = _mean_or_none(null_means)

    relevance_lift = (
        round(mean_relevance - null_relevance, 4)
        if mean_relevance is not None and null_relevance is not None
        else None
    )

    # Evidence length, so a high relevance score can be read as "more grounded"
    # vs. merely "longer sentence, more chances to overlap".
    ev_lengths = [len(_WORD_RE.findall(e.lower())) for e in ev_texts]

    # Same denominator as mean_relevance: evidence-bearing rows only.
    label_echo = [any(w in e.lower() for w in _LABEL_ECHO_WORDS) for e in ev_texts]

    return {
        "n": n_total,
        "evidence_present_rate": evidence_present_rate,
        "mean_relevance": round(mean_relevance, 4) if mean_relevance is not None else None,
        "median_relevance": round(float(np.median(real_scores)), 4) if real_scores else None,
        "null_relevance": round(null_relevance, 4) if null_relevance is not None else None,
        "relevance_lift": relevance_lift,
        "mean_evidence_length_words": round(float(np.mean(ev_lengths)), 2) if ev_lengths else None,
        # evidence that just restates the label word instead of citing the review
        "label_echo_rate_among_present": round(float(np.mean(label_echo)), 4) if label_echo else None,
    }


# --------------------------------------------------------------------------
# 2. Confidence calibration
# --------------------------------------------------------------------------
def evaluate_calibration(
    preds: pd.DataFrame, n_bins: int = 5, common_index: pd.Index | None = None
) -> dict[str, Any]:
    """Expected Calibration Error + Brier score of self-reported confidence
    against the model's own correctness (gold == predicted).

    `n_bins` defaults to 5: the model emits only a handful of discrete
    confidence values, so 10 bins leave most of them empty.

    `common_index`, when given, restricts scoring to that row set. Rows that
    fail to parse also tend to carry no numeric confidence, so a strategy with
    a high invalid rate would otherwise have its ECE measured on a smaller,
    easier subset - not comparable to a cleaner strategy's. Pass the same
    index to several strategies to score them on an identical population.
    `coverage` (scored rows / rows considered) is always reported so a low
    number is visible next to the ECE.
    """
    scored = preds
    if common_index is not None:
        scored = preds.loc[preds.index.intersection(common_index)]
    total = len(scored)

    df = scored.dropna(subset=["confidence"]).copy()
    n_excluded = total - len(df)
    if df.empty:
        return {
            "n": 0,
            "n_excluded_no_confidence": int(n_excluded),
            "coverage": 0.0 if total else None,
            "ece": None, "brier_score": None,
            "mean_confidence": None, "accuracy_on_scored_rows": None,
            "confidence_value_counts": {}, "bins": [],
        }

    df["confidence"] = df["confidence"].clip(0, 1)
    df["correct"] = (df["gold"] == df["predicted"]).astype(float)

    edges = [i / n_bins for i in range(n_bins + 1)]
    df["bin"] = pd.cut(df["confidence"], bins=edges, include_lowest=True)

    bins = []
    ece = 0.0
    n = len(df)
    for interval, group in df.groupby("bin", observed=True):
        if group.empty:
            continue
        bin_conf, bin_acc = float(group["confidence"].mean()), float(group["correct"].mean())
        ece += (len(group) / n) * abs(bin_acc - bin_conf)
        bins.append({
            "range": f"{interval.left:.1f}-{interval.right:.1f}",
            "n": int(len(group)),
            "mean_confidence": round(bin_conf, 4),
            "accuracy": round(bin_acc, 4),
        })

    brier = float(((df["confidence"] - df["correct"]) ** 2).mean())
    value_counts = df["confidence"].round(3).value_counts().head(10)
    return {
        "n": int(n),
        "n_excluded_no_confidence": int(n_excluded),
        "coverage": round(n / total, 4) if total else None,
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "mean_confidence": round(float(df["confidence"].mean()), 4),
        "accuracy_on_scored_rows": round(float(df["correct"].mean()), 4),
        "confidence_value_counts": {f"{k:.3f}": int(v) for k, v in value_counts.items()},
        "bins": bins,
    }


# --------------------------------------------------------------------------
# 3. Format compliance
# --------------------------------------------------------------------------
def evaluate_format_compliance(preds: pd.DataFrame) -> dict[str, Any]:
    """How often the reply was exactly the required 3-key JSON schema, vs
    needing the lenient extraction ladder or missing fields.

    `schema_compliance_rate` is over *all* rows, so it folds in the invalid
    rate; `strict_rate_among_valid` / `repaired_rate_among_valid` restrict to
    rows that produced a usable label, isolating format-following from parse
    success. `repaired_rate_among_valid` is the complement of the strict rate
    within valid rows (anything that needed a non-strict rung of the ladder),
    so it does not depend on the parser's internal strategy names.
    """
    n = len(preds)
    parse_strategy = preds["parse_strategy"]
    strategy_counts = parse_strategy.fillna("no_json_found").value_counts().to_dict()

    extra_keys_present = preds["extra_keys"].apply(lambda x: bool(x) if isinstance(x, list) else False)
    confidence_valid = preds["confidence_valid"].fillna(False).astype(bool)
    evidence_present = preds["evidence"].notna()

    schema_ok = (
        (parse_strategy == "strict")
        & ~extra_keys_present
        & confidence_valid
        & evidence_present
    )

    # "valid" = the parser found a usable JSON object (any rung of the ladder),
    # i.e. parse_strategy is set and is not the no-JSON sentinel.
    valid = parse_strategy.notna() & (parse_strategy != "no_json_found")
    n_valid = int(valid.sum())
    strict_among_valid = int((valid & (parse_strategy == "strict")).sum())
    # Complement of strict within valid rows: robust to whatever the parser
    # names its lenient rungs ("fenced", "substring", "repaired", ...).
    repaired_among_valid = int((valid & (parse_strategy != "strict")).sum())

    return {
        "n": int(n),
        "schema_compliance_rate": round(float(schema_ok.mean()), 4) if n else None,
        "n_valid": n_valid,
        "strict_rate_among_valid": round(strict_among_valid / n_valid, 4) if n_valid else None,
        "repaired_rate_among_valid": round(repaired_among_valid / n_valid, 4) if n_valid else None,
        "strategy_breakdown": strategy_counts,
        "confidence_present_rate": round(float(confidence_valid.mean()), 4) if n else None,
        "evidence_present_rate": round(float(evidence_present.mean()), 4) if n else None,
        "extra_keys_rate": round(float(extra_keys_present.mean()), 4) if n else None,
    }


# --------------------------------------------------------------------------
def evaluate(preds: pd.DataFrame, common_index: pd.Index | None = None) -> dict[str, Any]:
    """Bundle of the three quality dimensions beyond label accuracy.

    `common_index` is forwarded to `evaluate_calibration` so callers can score
    several strategies' ECE on an identical row set (see that function's note).
    """
    return {
        "evidence": evaluate_evidence(preds),
        "calibration": evaluate_calibration(preds, common_index=common_index),
        "format_compliance": evaluate_format_compliance(preds),
    }


def to_markdown(q: dict[str, Any]) -> str:
    """Render the quality bundle as a markdown block, appended to results_task2.md."""
    ev, cal, fmt = q["evidence"], q["calibration"], q["format_compliance"]

    lines = [
        "## Evidence relevance",
        "",
        f"- Evidence present: **{ev['evidence_present_rate']:.2%}**" if ev["evidence_present_rate"] is not None else "- Evidence present: n/a",
        (f"- Mean relevance (content-word overlap with the review): **{ev['mean_relevance']}**"
         if ev["mean_relevance"] is not None else "- Mean relevance: n/a (no evidence present)"),
    ]
    if ev["null_relevance"] is not None:
        lines += [
            f"- Null baseline (same evidence vs. a random other review, mean of "
            f"{len(_NULL_BASELINE_SEEDS)} permutations): **{ev['null_relevance']}**",
            f"- Relevance lift over null (grounding signal; ~0 = not tied to *this* review): "
            f"**{ev['relevance_lift']}**",
        ]
    if ev["mean_evidence_length_words"] is not None:
        lines.append(f"- Mean evidence length: **{ev['mean_evidence_length_words']}** words")
    if ev["label_echo_rate_among_present"] is not None:
        lines.append(
            f"- Label-echo rate among evidence-bearing rows (evidence just repeats the "
            f"label word): **{ev['label_echo_rate_among_present']:.2%}**"
        )
    lines += [
        "",
        "## Confidence calibration",
        "",
    ]
    if cal["ece"] is not None:
        coverage = f"{cal['coverage']:.2%}" if cal["coverage"] is not None else "n/a"
        lines += [
            f"- Expected Calibration Error (ECE): **{cal['ece']}** "
            f"(n={cal['n']}, coverage={coverage}; lower is better, 0 = perfectly calibrated)",
            f"- Brier score: **{cal['brier_score']}** (lower is better)",
            f"- Mean confidence vs. accuracy, over the {cal['n']} rows with a numeric confidence: "
            f"{cal['mean_confidence']} vs. {cal['accuracy_on_scored_rows']}",
        ]
        if cal["n_excluded_no_confidence"]:
            lines.append(f"- Excluded (no numeric confidence returned): {cal['n_excluded_no_confidence']}")
        if cal["confidence_value_counts"]:
            pretty = ", ".join(f"{k} (n={v})" for k, v in cal["confidence_value_counts"].items())
            lines.append(f"- Confidence values returned (top 10): {pretty}")
        lines += [
            "",
            "| Confidence bin | n | mean confidence | accuracy |",
            "|---|---|---|---|",
        ]
        for b in cal["bins"]:
            lines.append(f"| {b['range']} | {b['n']} | {b['mean_confidence']} | {b['accuracy']} |")
    else:
        lines.append("- No row returned a numeric confidence; calibration cannot be computed.")

    lines += [
        "",
        "## Format compliance",
        "",
        f"- Full schema compliance over all rows (strict JSON, exactly label+confidence+evidence, "
        f"valid types): **{fmt['schema_compliance_rate']:.2%}**",
    ]
    if fmt["strict_rate_among_valid"] is not None:
        lines += [
            f"- Strict JSON among the {fmt['n_valid']} parseable rows (excludes invalid output): "
            f"**{fmt['strict_rate_among_valid']:.2%}**",
            f"- Needed a non-strict rung of the ladder (fenced/substring/repair), among parseable "
            f"rows: **{fmt['repaired_rate_among_valid']:.2%}**",
        ]
    lines += [
        f"- Confidence present & numeric: **{fmt['confidence_present_rate']:.2%}**",
        f"- Evidence present: **{fmt['evidence_present_rate']:.2%}**",
        f"- Extra/unexpected keys present: **{fmt['extra_keys_rate']:.2%}**",
        "",
        "| Extraction strategy | count |",
        "|---|---|",
    ]
    for k, v in fmt["strategy_breakdown"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)
