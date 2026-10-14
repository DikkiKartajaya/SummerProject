"""Task 2 prompt-sensitivity: perturb a Task 2 few-shot prompt and see how much
the numbers move.

Task 2 (`scripts/05_run_task2_fewshot.py`) compared few-shot strategies on the
**test** split (`outputs/task2/final_all/`): **`curated`** ("curated concise")
won, **`random`** ("random concise") lost. That comparison changed one big
thing at a time (which examples, how many rules). This module changes small,
meaning-preserving things instead, along three axes, to measure how brittle a
strategy is to wording it "shouldn't" care about:

  1. instruction phrasing  - 4 full paraphrases of the instruction block
                             (context + the two label definitions + the
                             judgment rules + the JSON-answer instruction),
                             each preserving every definition and rule
  2. demo order            - the same 4 demos in each of pos,neg,pos,neg /
                             pos,pos,neg,neg / neg,neg,pos,pos / neg,pos,neg,pos
                             (the strategy's own file order is the baseline)
  3. output schema         - the 3 fields reordered; and the 3 fields renamed
                             (label->sentiment, confidence->certainty,
                             evidence->justification), everywhere they appear

`curated` and `random` share the *identical* instruction block (they differ
only in their 4 demo reviews), so one set of hand-written paraphrases (`p2`-`p4`)
covers both; `assert_baseline_matches` checks that assumption per strategy.
Everything else - model, decoding, the strategy's 4 demo reviews and their
hand-authored answers, the eval rows - is held fixed.

Nothing is re-implemented: each strategy's instruction and demos are read back
out of its `prompts/task2_fewshot*.txt` at call time, and
`build_system_prompt(strategy, **baseline_for(strategy))` is asserted to
reproduce that file byte for byte.

Parsing note: the `renamed` schema variant makes the model emit
`{"sentiment": ..., "certainty": ..., "justification": ...}`. `parse_variant`
maps those keys back to `label`/`confidence`/`evidence` *before* the normal
`json_parser.parse_fewshot` validation, so label quality stays comparable. The
extraction strategy (`strict`/`repaired`/...) is still taken from the real
reply, so `schema_compliance_rate` for that variant means "structurally a
clean 3-field object", not "used the literal baseline field names".
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from . import json_parser, prompts2
from .utils import get_logger

logger = get_logger(__name__)

# The Task 2 strategies this harness can perturb. They must share one
# instruction block (only their demos differ) so the p2-p4 paraphrases apply to
# all of them - assert_baseline_matches enforces it.
SUPPORTED_STRATEGIES = ("curated", "random")

# Default target: the strategy that won the Task 2 test comparison.
WINNING_STRATEGY = "curated"
DEFAULT_STRATEGY = WINNING_STRATEGY


# --------------------------------------------------------------------------
# Baseline pieces, read back out of a strategy's prompt file
# --------------------------------------------------------------------------
def _load_baseline_parts(strategy: str) -> tuple[str, str, list[dict[str, Any]], str]:
    """(instruction, format_line, demos, user_template) from a strategy's .txt.

    File layout (see prompts/task2_fewshot.txt / task2_fewshot_random.txt):

        <instruction block: context + label definitions + rules + JSON instr>

        Output format:
        <format_line>

        Examples:

        Review: <text>
        JSON: <json answer>
        ...
    """
    system, user_template = prompts2._load_template(strategy)
    head, examples = system.split("\nExamples:\n", 1)

    instr_block, fmt_block = head.split("\nOutput format:\n", 1)
    instruction = instr_block.strip()
    format_line = fmt_block.strip()

    demos: list[dict[str, Any]] = []
    for block in examples.strip().split("\n\n"):
        block = block.strip()
        if not block.startswith("Review: ") or "\nJSON: " not in block:
            continue
        review, answer = block.split("\nJSON: ", 1)
        parsed = json.loads(answer.strip())
        demos.append({
            "text": review[len("Review: "):].strip(),
            "label": parsed["label"],
            "confidence": parsed["confidence"],
            "evidence": parsed["evidence"],
        })
    if len(demos) < 2:
        raise RuntimeError(
            f"Expected >=2 demos in {strategy}'s prompt, parsed {len(demos)}."
        )
    return instruction, format_line, demos, user_template.strip()


def _detect_demo_order(demos: list[dict[str, Any]]) -> str:
    """Name the demo layout a strategy's file already uses, so it becomes the
    baseline and the other three orders are the perturbations."""
    labels = tuple(d["label"] for d in demos)
    known = {
        ("positive", "negative", "positive", "negative"): "alternating",
        ("negative", "positive", "negative", "positive"): "neg_alternating",
        ("positive", "positive", "negative", "negative"): "pos_first",
        ("negative", "negative", "positive", "positive"): "neg_first",
    }
    if labels not in known:
        raise RuntimeError(
            f"Demo label layout {labels} is not one of the 4 handled orders - "
            "extend DEMO_ORDERS / _ordered_demos."
        )
    return known[labels]


@lru_cache(maxsize=None)
def _parts(strategy: str) -> dict[str, Any]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"prompt_sensitivity supports {SUPPORTED_STRATEGIES}, got {strategy!r}."
        )
    instruction, format_line, demos, user_template = _load_baseline_parts(strategy)
    return {
        "instruction": instruction,
        "format_line": format_line,
        "demos": demos,
        "user_template": user_template,
        "baseline_demo_order": _detect_demo_order(demos),
    }


# --------------------------------------------------------------------------
# Axis 1: instruction phrasing
# --------------------------------------------------------------------------
# "p1" is the verbatim instruction block, shared by every SUPPORTED_STRATEGIES
# entry (checked in assert_baseline_matches). p2-p4 are full paraphrases: every
# definition, example snippet and rule is kept, only wording / structure /
# ordering changes.
_P2 = (
    "You are labeling the sentiment of IMDB movie reviews - informal notes "
    "written by ordinary viewers. Assign each review one of two labels.\n\n"
    "Positive: the reviewer's overall take on the film is favorable - they "
    "recommend it, praise its acting, writing, direction, atmosphere, or "
    "emotional impact, or simply say they enjoyed it (e.g., \"the pacing drags "
    "a bit, but the ending floored me\", \"an underrated gem I'll rewatch\").\n\n"
    "Negative: the reviewer's overall take is unfavorable - they pan the "
    "acting, writing, pacing, or coherence, describe being bored or let down, "
    "or tell other people to stay away (e.g., \"two hours I'll never get "
    "back\", \"great premise, wasted on a lazy script\").\n\n"
    "Base the label on the reviewer's bottom-line verdict, not on the mood of "
    "the story being summarized: a review of a bleak or violent film may still "
    "be Positive, and a review of a feel-good film may be Negative. When a "
    "review is mixed and cuts both ways, go with the side the reviewer's final "
    "recommendation favors. Treat sarcasm and irony as meaning the reverse of "
    "what the words literally say.\n\n"
    "Reply with one JSON object per review: \"label\" is either \"positive\" "
    "or \"negative\", \"confidence\" is a value between 0.0 and 1.0, and "
    "\"evidence\" is a single concise sentence that points to what in the "
    "review backs the label - stated outright, not worked out step by step."
)
_P3 = (
    "Task: label each IMDB movie review - these are casual, user-written film "
    "reviews - as positive or negative.\n\n"
    "positive = the reviewer comes down in favor of the film overall: "
    "recommends it, praises the acting, writing, direction, atmosphere, or "
    "emotional impact, or says they had a good time (e.g., \"the pacing drags "
    "a bit, but the ending floored me\", \"an underrated gem I'll rewatch\").\n"
    "negative = the reviewer comes down against it overall: faults the acting, "
    "writing, pacing, or coherence, reports boredom or disappointment, or "
    "warns others off (e.g., \"two hours I'll never get back\", \"great "
    "premise, wasted on a lazy script\").\n\n"
    "What matters is the reviewer's overall judgment of the film, not the "
    "emotional tone of the plot they recount - a tragic or violent film can "
    "earn a positive review, and a cheerful one a negative review. For a mixed "
    "review, follow whichever way the reviewer's final recommendation tips. "
    "Read sarcasm and irony as the opposite of their literal wording.\n\n"
    "Answer with a single JSON object: \"label\" is \"positive\" or "
    "\"negative\", \"confidence\" is a number from 0.0 to 1.0, and "
    "\"evidence\" is one concise sentence naming what in the review supports "
    "the label, stated directly rather than reasoned out."
)
_P4 = (
    "You are a sentiment annotator working through IMDB movie reviews, which "
    "are informal and written by everyday users. Every review gets exactly one "
    "of two sentiments.\n\n"
    "Positive: overall, the reviewer is favorable toward the film - "
    "recommending it, praising elements like acting, writing, direction, "
    "atmosphere, or emotional impact, or reporting that they enjoyed it (for "
    "instance, \"the pacing drags a bit, but the ending floored me\" or \"an "
    "underrated gem I'll rewatch\").\n\n"
    "Negative: overall, the reviewer is unfavorable toward the film - "
    "criticizing elements like acting, writing, pacing, or coherence, "
    "reporting boredom or disappointment, or advising others to avoid it (for "
    "instance, \"two hours I'll never get back\" or \"great premise, wasted on "
    "a lazy script\").\n\n"
    "Decide from the reviewer's overall verdict on the film rather than the "
    "emotional register of the plot they describe: a tragic or violent film "
    "can still be reviewed positively, and an upbeat film negatively. If the "
    "review is genuinely mixed, side with the direction of the reviewer's "
    "closing recommendation. Sarcasm and irony should be read as the opposite "
    "of the literal words.\n\n"
    "For each review, return one JSON object in which \"label\" holds "
    "\"positive\" or \"negative\", \"confidence\" holds a number from 0.0 to "
    "1.0, and \"evidence\" holds one concise sentence stating what in the "
    "review supports the label, given directly instead of reasoned out."
)
# p1 is filled in per call from the target strategy's file; the module-level
# dict carries the default strategy's p1 for callers that read PHRASINGS
# directly (all SUPPORTED_STRATEGIES share it).
PHRASINGS: dict[str, str] = {
    "p1": _parts(DEFAULT_STRATEGY)["instruction"], "p2": _P2, "p3": _P3, "p4": _P4,
}


def phrasings_for(strategy: str) -> dict[str, str]:
    """PHRASINGS with `p1` bound to this strategy's own instruction block."""
    return {**PHRASINGS, "p1": _parts(strategy)["instruction"]}


# --------------------------------------------------------------------------
# Axis 2: demo order
# --------------------------------------------------------------------------
DEMO_ORDERS: dict[str, str] = {
    "alternating": "positive, negative, positive, negative",
    "neg_alternating": "negative, positive, negative, positive",
    "pos_first": "both positives, then both negatives",
    "neg_first": "both negatives, then both positives",
}

# --------------------------------------------------------------------------
# Axis 3: output schema (field order / field names)
# --------------------------------------------------------------------------
_SCHEMA_PLACEHOLDERS = {
    "label": '"<positive|negative>"',
    "confidence": "<0.0-1.0>",
    "evidence": '"<one sentence>"',
}
SCHEMA_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "order": ["label", "confidence", "evidence"],
        "names": {"label": "label", "confidence": "confidence", "evidence": "evidence"},
    },
    "reordered": {
        "order": ["confidence", "evidence", "label"],
        "names": {"label": "label", "confidence": "confidence", "evidence": "evidence"},
    },
    "renamed": {
        "order": ["label", "confidence", "evidence"],
        "names": {"label": "sentiment", "confidence": "certainty", "evidence": "justification"},
        "key_map": {"label": "sentiment", "confidence": "certainty", "evidence": "justification"},
    },
}


def baseline_for(strategy: str = DEFAULT_STRATEGY) -> dict[str, str]:
    """The one (phrasing, demo_order, schema) combo that must reproduce the
    strategy's prompt file exactly - `p1`, the file's own demo order, and the
    unmodified schema."""
    return {
        "phrasing": "p1",
        "demo_order": _parts(strategy)["baseline_demo_order"],
        "schema": "baseline",
    }


# Back-compat alias for the default strategy.
BASELINE: dict[str, str] = baseline_for(DEFAULT_STRATEGY)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def _rename_fields(text: str, schema: str) -> str:
    """Swap the quoted canonical field names for a renamed schema's names,
    everywhere they appear (format line, demo answers, and the instruction
    prose that spells the schema out in words). Only quoted tokens are touched -
    never `"positive"`/`"negative"`, never the bare word "label" in prose."""
    key_map = SCHEMA_VARIANTS[schema].get("key_map")
    if not key_map:
        return text
    for canon, new in key_map.items():
        text = text.replace(f'"{canon}"', f'"{new}"')
    return text


def _format_line(schema: str) -> str:
    spec = SCHEMA_VARIANTS[schema]
    parts = [f'"{spec["names"][f]}": {_SCHEMA_PLACEHOLDERS[f]}' for f in spec["order"]]
    return "{" + ", ".join(parts) + "}"


def _demo_answer(demo: dict[str, Any], schema: str) -> str:
    spec = SCHEMA_VARIANTS[schema]
    obj = {spec["names"][f]: demo[f] for f in spec["order"]}
    return json.dumps(obj, ensure_ascii=False)


def _ordered_demos(strategy: str, demo_order: str) -> list[dict[str, Any]]:
    demos = _parts(strategy)["demos"]
    pos = [d for d in demos if d["label"] == "positive"]
    neg = [d for d in demos if d["label"] == "negative"]
    if demo_order == "pos_first":
        return pos + neg
    if demo_order == "neg_first":
        return neg + pos
    if demo_order == "alternating":
        return [x for pair in zip(pos, neg) for x in pair]
    if demo_order == "neg_alternating":
        return [x for pair in zip(neg, pos) for x in pair]
    raise ValueError(f"unknown demo_order: {demo_order!r}")


def build_system_prompt(strategy: str, phrasing: str, demo_order: str, schema: str) -> str:
    """The system message for one (strategy, phrasing, demo_order, schema) combo."""
    instruction = _rename_fields(phrasings_for(strategy)[phrasing], schema)
    demo_block = "\n\n".join(
        f"Review: {d['text']}\nJSON: {_demo_answer(d, schema)}"
        for d in _ordered_demos(strategy, demo_order)
    )
    return (
        f"{instruction}\n\n"
        f"Output format:\n{_format_line(schema)}\n\n"
        f"Examples:\n\n{demo_block}"
    )


def build_messages(
    strategy: str, phrasing: str, demo_order: str, schema: str, text: str
) -> list[dict[str, str]]:
    """Chat messages for one review under one prompt variant."""
    return [
        {"role": "system", "content": build_system_prompt(strategy, phrasing, demo_order, schema)},
        {"role": "user", "content": _parts(strategy)["user_template"].format(text=text)},
    ]


def assert_baseline_matches(strategy: str = DEFAULT_STRATEGY) -> None:
    """The baseline combination must rebuild the strategy's .txt file exactly,
    and its instruction block must equal PHRASINGS['p1'] (so p2-p4 are valid
    paraphrases of it). A silent drift here would mean the "baseline" column is
    not actually the strategy under test."""
    system, user_template = prompts2._load_template(strategy)
    rebuilt = build_system_prompt(strategy, **baseline_for(strategy))
    if rebuilt != system:
        raise AssertionError(
            f"prompt_sensitivity baseline does not match "
            f"prompts/{prompts2.STRATEGIES[strategy]} - the module and the prompt "
            "file have drifted. Re-sync _load_baseline_parts / the paraphrases."
        )
    if user_template.strip() != _parts(strategy)["user_template"]:
        raise AssertionError("prompt_sensitivity user template drift.")
    if _parts(strategy)["instruction"] != PHRASINGS["p1"]:
        raise AssertionError(
            f"{strategy}'s instruction block differs from PHRASINGS['p1'] - the "
            "hand-written paraphrases p2-p4 were written for the shared block and "
            "may no longer be faithful. Handle this strategy separately."
        )


# --------------------------------------------------------------------------
# Variant enumeration
# --------------------------------------------------------------------------
def build_variants(strategy: str = DEFAULT_STRATEGY, only_axis: str | None = None) -> list[dict[str, str]]:
    """The baseline plus one-at-a-time perturbations of each axis.

    Each entry: {axis, variant, label, phrasing, demo_order, schema}. `label`
    is the output-subfolder name. `only_axis` restricts to one axis (the
    baseline is always included as the reference point).
    """
    base = baseline_for(strategy)
    variants = [{"axis": "baseline", "variant": "baseline", "label": "baseline", **base}]

    axes = {
        "phrasing": [k for k in PHRASINGS if k != base["phrasing"]],
        "demo_order": [k for k in DEMO_ORDERS if k != base["demo_order"]],
        "schema": [k for k in SCHEMA_VARIANTS if k != base["schema"]],
    }
    for axis, names in axes.items():
        if only_axis and axis != only_axis:
            continue
        for name in names:
            variants.append({
                "axis": axis,
                "variant": name,
                "label": f"{axis}__{name}",
                **{**base, axis: name},
            })
    return variants


# --------------------------------------------------------------------------
# Parsing: map a renamed-schema reply back to the canonical 3 fields
# --------------------------------------------------------------------------
_FEWSHOT_KEYS = {"label", "confidence", "evidence"}


def _finish_fewshot(obj: dict[str, Any], strategy: str | None, raw: str) -> json_parser.FewShotResult:
    """The tail of json_parser.parse_fewshot, applied to an already-extracted
    (and key-remapped) object. Kept in sync with that function on purpose - it
    is short and stable, and reusing it lets the renamed-schema variant keep
    the real extraction strategy instead of a re-serialised 'strict'."""
    extra_keys = [k for k in obj if k not in _FEWSHOT_KEYS]

    label = json_parser.normalize_label(obj.get("label"))
    if label is None:
        return json_parser.FewShotResult(
            False, error="invalid_label", raw=raw, strategy=strategy, extra_keys=extra_keys
        )

    raw_conf = obj.get("confidence")
    conf_valid = isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool)
    try:
        confidence = max(0.0, min(1.0, float(raw_conf)))
    except (TypeError, ValueError):
        confidence, conf_valid = None, False

    evidence = obj.get("evidence")
    evidence = evidence.strip() if isinstance(evidence, str) and evidence.strip() else None

    return json_parser.FewShotResult(
        True, label=label, confidence=confidence, confidence_valid=conf_valid,
        evidence=evidence, strategy=strategy, extra_keys=extra_keys, raw=raw,
    )


def parse_variant(raw: str, schema: str) -> json_parser.FewShotResult:
    """Parse a reply, undoing any field renaming first so label quality and
    calibration stay comparable across schema variants. Strategy-independent."""
    key_map = SCHEMA_VARIANTS[schema].get("key_map")
    if not key_map:
        return json_parser.parse_fewshot(raw)

    text = (raw or "").strip()
    if not text:
        return json_parser.FewShotResult(False, error="empty_output", raw=raw)

    obj, strat = json_parser._extract_object(text)
    if obj is None:
        return json_parser.FewShotResult(False, error="no_json_found", raw=raw)

    inverse = {alt: canon for canon, alt in key_map.items()}
    remapped = {inverse.get(k, k): v for k, v in obj.items()}
    return _finish_fewshot(remapped, strat, raw)


def variant_metadata(strategy: str, v: dict[str, str]) -> dict[str, Any]:
    """Snapshot of one variant's prompt, saved alongside its results."""
    return {
        "strategy": strategy,
        "axis": v["axis"],
        "variant": v["variant"],
        "label": v["label"],
        "phrasing": v["phrasing"],
        "instruction": _rename_fields(phrasings_for(strategy)[v["phrasing"]], v["schema"]),
        "demo_order": v["demo_order"],
        "demo_order_desc": DEMO_ORDERS[v["demo_order"]],
        "schema": v["schema"],
        "schema_format_line": _format_line(v["schema"]),
        "system_prompt": build_system_prompt(strategy, v["phrasing"], v["demo_order"], v["schema"]),
        "user_template": _parts(strategy)["user_template"],
        "demo_label_order": [d["label"] for d in _ordered_demos(strategy, v["demo_order"])],
    }
