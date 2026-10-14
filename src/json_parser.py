"""Parse and validate the model's constrained-JSON answer.

Tries a ladder of increasingly permissive strategies (strict JSON -> fenced
JSON -> substring JSON -> regex key/value -> bare label word) and records
which one succeeded, so the report can state how often the strict path
was enough. `bare_label` is the only lenient one, tracked separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .utils import LABELS

# --- Regexes, compiled once ------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_KV_RE = re.compile(r'["\']?label["\']?\s*[:=]\s*["\']?\s*(\w+)', re.IGNORECASE)
_WORD_RE = re.compile(r"\b(positive|negative)\b", re.IGNORECASE)

# A closing quote is followed (after whitespace) by one of these, or by
# nothing at all. Any other quote inside a string is an unescaped inner
# quote (e.g. `"evidence": "Calls it "terrible" and..."`) and gets escaped.
# A bare trailing comma is NOT enough on its own - prose routinely has a
# comma right after a quoted word (`"terrible", mentions ...`) - so a comma
# only counts when it's actually followed by the next `"key":` pair.
_CLOSING_QUOTE_RE = re.compile(r'\s*(?:\}|\]|:|$)')
_CLOSING_QUOTE_COMMA_RE = re.compile(r'\s*,\s*"[^"]*"\s*:')

# Common near-misses mapped onto the canonical label set.
_ALIASES = {
    "pos": "positive", "positve": "positive", "good": "positive",
    "neg": "negative", "negatve": "negative", "bad": "negative",
}


def _repair_unescaped_quotes(text: str) -> str:
    """Escape stray `"` characters inside JSON string values.

    Models frequently quote a word inside `evidence` (`"Calls it "terrible""`)
    without escaping it, which breaks `json.loads` even though the object is
    otherwise well-formed. Walk the text tracking string state; a `"` is only
    treated as a real closing quote when it's followed by `,`, `}`, `]`, `:`
    or end-of-string (allowing surrounding whitespace) - anything else is an
    inner quote and gets escaped.
    """
    out: list[str] = []
    in_string = False
    escape = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escape = False
            i += 1
            continue

        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            if _CLOSING_QUOTE_RE.match(text, i + 1) or _CLOSING_QUOTE_COMMA_RE.match(text, i + 1):
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _try_parse(text: str) -> dict | None:
    """`json.loads` that only ever returns a dict, never raises."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _try_parse_with_repair(text: str) -> tuple[dict | None, bool]:
    """Try `text` as-is, then with unescaped inner quotes repaired.

    Returns (obj, needed_repair) so callers can pick a strategy label.
    """
    obj = _try_parse(text)
    if obj is not None:
        return obj, False
    return _try_parse(_repair_unescaped_quotes(text)), True


@dataclass
class ParseResult:
    """Outcome of parsing one model reply."""
    ok: bool
    label: str | None = None
    strategy: str | None = None       # which ladder rung succeeded
    error: str | None = None
    raw: str = ""
    extra_keys: list[str] = field(default_factory=list)  # schema violations seen

    def as_dict(self) -> dict[str, Any]:
        return {
            "parse_ok": self.ok,
            "parse_strategy": self.strategy,
            "parse_error": self.error,
            "extra_keys": self.extra_keys,
        }


def normalize_label(value: Any) -> str | None:
    """Map a raw value onto one of the three canonical labels, or None."""
    if not isinstance(value, str):
        return None
    v = value.strip().strip("\"'.,!").lower()
    if v in LABELS:
        return v
    return _ALIASES.get(v)


def _from_obj(obj: Any) -> tuple[str | None, list[str]]:
    """Pull `label` out of a decoded JSON object; also report extra keys."""
    if not isinstance(obj, dict):
        return None, []
    extra = [k for k in obj if k != "label"]
    return normalize_label(obj.get("label")), extra


def parse(raw: str) -> ParseResult:
    """Run the strategy ladder over one raw model reply."""
    text = (raw or "").strip()
    if not text:
        return ParseResult(ok=False, error="empty_output", raw=raw)

    # 1. strict - the reply is exactly a JSON object (optionally with
    #    unescaped inner quotes repaired, e.g. a stray `"` inside a value)
    obj, repaired = _try_parse_with_repair(text)
    if obj is not None:
        label, extra = _from_obj(obj)
        if label:
            return ParseResult(True, label, "repaired" if repaired else "strict", raw=raw, extra_keys=extra)

    # 2. fenced - ```json { ... } ```
    fence = _FENCE_RE.search(text)
    if fence:
        obj, repaired = _try_parse_with_repair(fence.group(1).strip())
        if obj is not None:
            label, extra = _from_obj(obj)
            if label:
                return ParseResult(True, label, "repaired" if repaired else "fenced", raw=raw, extra_keys=extra)

    # 3. substring - first balanced-looking {...} anywhere in the reply
    for match in _OBJ_RE.finditer(text):
        obj, repaired = _try_parse_with_repair(match.group(0))
        if obj is not None:
            label, extra = _from_obj(obj)
            if label:
                return ParseResult(True, label, "repaired" if repaired else "substring", raw=raw, extra_keys=extra)

    # 4. regex_kv - malformed JSON but the key/value pair is readable
    kv = _KV_RE.search(text)
    if kv:
        label = normalize_label(kv.group(1))
        if label:
            return ParseResult(True, label, "regex_kv", raw=raw)

    # 5. bare_label - no JSON at all, but exactly one label word is present
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if len(words) == 1:
        return ParseResult(True, words.pop(), "bare_label", raw=raw)
    if len(words) > 1:
        return ParseResult(False, error="ambiguous_multiple_labels", raw=raw)

    return ParseResult(False, error="no_label_found", raw=raw)


def validate(result: ParseResult, strict_schema: bool = False) -> ParseResult:
    """Apply schema rules on top of a successful parse.

    With `strict_schema=True` the reply must be a clean JSON object with the
    single key `label` (strategy "strict" and no extra keys). Useful as an
    ablation to measure raw format-following ability.
    """
    if not result.ok:
        return result
    if strict_schema and (result.strategy != "strict" or result.extra_keys):
        return ParseResult(
            False, error=f"schema_violation({result.strategy})",
            raw=result.raw, extra_keys=result.extra_keys,
        )
    return result


def parse_and_validate(raw: str, strict_schema: bool = False) -> ParseResult:
    """Convenience wrapper: parse, then validate."""
    return validate(parse(raw), strict_schema=strict_schema)


# --------------------------------------------------------------------------
# Task 2: label + confidence + evidence
# --------------------------------------------------------------------------
@dataclass
class FewShotResult:
    """Outcome of parsing a Task 2 {label, confidence, evidence} reply."""
    ok: bool
    label: str | None = None
    confidence: float | None = None
    confidence_valid: bool = False    # was `confidence` present and numeric, pre-clamping
    evidence: str | None = None
    strategy: str | None = None       # which extraction stage found the JSON: strict|fenced|substring
    extra_keys: list[str] = field(default_factory=list)  # schema violations seen
    error: str | None = None
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "parse_ok": self.ok,
            "parse_strategy": self.strategy,
            "confidence": self.confidence,
            "confidence_valid": self.confidence_valid,
            "evidence": self.evidence,
            "extra_keys": self.extra_keys,
            "parse_error": self.error,
        }


_FEWSHOT_KEYS = {"label", "confidence", "evidence"}


def _extract_object(text: str) -> tuple[dict | None, str | None]:
    """First JSON object found in text: whole string, then fenced, then substring.

    Each candidate is also retried with unescaped inner quotes repaired (see
    `_repair_unescaped_quotes`) - the model routinely quotes a word inside
    `evidence` without escaping it, which strict `json.loads` rejects even
    though the object is otherwise well-formed.

    Returns (obj, strategy) so callers can tell a clean reply from one that
    needed lenient extraction - the basis of the format-compliance metric.
    """
    obj, repaired = _try_parse_with_repair(text)
    if obj is not None:
        return obj, "repaired" if repaired else "strict"

    fence = _FENCE_RE.search(text)
    if fence:
        obj, repaired = _try_parse_with_repair(fence.group(1).strip())
        if obj is not None:
            return obj, "repaired" if repaired else "fenced"

    for match in _OBJ_RE.finditer(text):
        obj, repaired = _try_parse_with_repair(match.group(0))
        if obj is not None:
            return obj, "repaired" if repaired else "substring"
    return None, None


def parse_fewshot(raw: str) -> FewShotResult:
    """Parse a Task 2 reply: {"label": ..., "confidence": ..., "evidence": ...}."""
    text = (raw or "").strip()
    if not text:
        return FewShotResult(False, error="empty_output", raw=raw)

    obj, strategy = _extract_object(text)
    if obj is None:
        return FewShotResult(False, error="no_json_found", raw=raw)

    extra_keys = [k for k in obj if k not in _FEWSHOT_KEYS]

    label = normalize_label(obj.get("label"))
    if label is None:
        return FewShotResult(False, error="invalid_label", raw=raw,
                              strategy=strategy, extra_keys=extra_keys)

    raw_confidence = obj.get("confidence")
    confidence_valid = isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool)
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = None
        confidence_valid = False

    evidence = obj.get("evidence")
    evidence = evidence.strip() if isinstance(evidence, str) and evidence.strip() else None

    return FewShotResult(
        True, label=label, confidence=confidence, confidence_valid=confidence_valid,
        evidence=evidence, strategy=strategy, extra_keys=extra_keys, raw=raw,
    )
