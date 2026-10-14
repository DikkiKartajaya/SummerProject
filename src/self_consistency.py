"""Self-consistency voting for Task 2 few-shot inference.

The Task 2 pipeline (`scripts/05_run_task2_fewshot.py`) is greedy: one
deterministic reply per review. Self-consistency instead draws N stochastic
replies per review and takes the majority label
(Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning").

`scripts/08_run_self_consistency.py` is the runner; this module is just the
vote aggregation, kept separate so it is unit-testable offline. It does not
touch `json_parser`, `metrics.py`, or `eval_task2.py`: each candidate is parsed
by the existing `json_parser.parse_fewshot`, and `aggregate()` returns exactly
the fields `FewShotResult.as_dict()` produces (so `eval_task2.evaluate()` sees
a compatible row) plus two new columns, `vote_margin` and `n_valid_votes`.

Rules (spec):
  - a row is VALID if at least one candidate parses
  - label = majority vote over the valid candidates
  - tie -> the label of the single highest-confidence valid candidate
  - confidence = mean confidence of the candidates that voted for the winner
  - evidence  = the first winner-agreeing candidate's evidence
  - vote_margin  = winning votes / valid votes
  - n_valid_votes = number of candidates that parsed

CAVEAT (invalid_output_rate is no longer apple-to-apple): "valid if >=1
candidate parses" is a far weaker bar than the greedy pipeline's "the one
reply parses". A self-consistency run's `invalid_output_rate` can only be <=
the single-sample baseline's - drawing more samples never makes a row less
recoverable - so it is not directly comparable across the two modes. Compare
accuracy / macro-F1; read the invalid rate only within self-consistency runs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import json_parser

# Fixed sampling params for the N > 1 path (spec). vLLM would express this as
# SamplingParams(n=N, temperature=0.7, top_p=0.95, seed=42); this project runs
# on HF transformers, where the equivalent is a single generate() call with
# num_return_sequences=N and do_sample=True (see model.generate_batch_samples).
SC_TEMPERATURE = 0.7
SC_TOP_P = 0.95

_AS_DICT_KEYS = (
    "parse_ok", "parse_strategy", "confidence", "confidence_valid",
    "evidence", "extra_keys", "parse_error",
)


def _mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(nums) / len(nums), 4) if nums else None


def _unparsed_row() -> dict[str, Any]:
    """Row shape for 'no candidate parsed' - mirrors a failed FewShotResult."""
    return {
        "parse_ok": False,
        "parse_strategy": None,
        "confidence": None,
        "confidence_valid": False,
        "evidence": None,
        "extra_keys": [],
        "parse_error": "no_valid_candidate",
        "vote_margin": 0.0,
        "n_valid_votes": 0,
        "_label": None,
    }


def aggregate(candidates_raw: list[str]) -> dict[str, Any]:
    """Majority-vote a review's N raw candidate replies into one record.

    Returns the `FewShotResult.as_dict()` fields plus `vote_margin`,
    `n_valid_votes`, and `_label` (the voted label, or None). The caller pops
    `_label` into the `predicted` column and spreads the rest.
    """
    parsed = [json_parser.parse_fewshot(raw) for raw in candidates_raw]
    valid = [p for p in parsed if p.ok]
    n_valid = len(valid)
    if n_valid == 0:
        return _unparsed_row()

    counts = Counter(p.label for p in valid)
    top_count = counts.most_common(1)[0][1]
    tied = [label for label, c in counts.items() if c == top_count]

    if len(tied) == 1:
        winner = tied[0]
    else:
        # Tie-break: the label of the single highest-confidence valid candidate
        # (confidence None sorts lowest).
        best = max(valid, key=lambda p: p.confidence if p.confidence is not None else -1.0)
        winner = best.label

    agree = [p for p in valid if p.label == winner]
    rep = agree[0]  # representative parse: its strategy/extra_keys stand in for the row

    return {
        "parse_ok": True,
        "parse_strategy": rep.strategy,
        "confidence": _mean([p.confidence for p in agree]),
        "confidence_valid": any(p.confidence_valid for p in agree),
        "evidence": next((p.evidence for p in agree if p.evidence), rep.evidence),
        "extra_keys": rep.extra_keys,
        "parse_error": None,
        "vote_margin": round(len(agree) / n_valid, 4),
        "n_valid_votes": n_valid,
        "_label": winner,
    }
