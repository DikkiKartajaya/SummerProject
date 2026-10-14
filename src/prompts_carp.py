"""CARP (Clue And Reasoning Prompting) prompt construction for Task 1.

An ablation of the plain zero-shot prompt in `prompts.py`, for comparing the
two prompting methods head-to-head: same label definitions and
disambiguation rules as `prompts.py` (so any accuracy difference reflects the
prompting method, not the task definition), but the model is required to work
through three explicit steps - Clue, then Reasoning, then Label - before
answering, instead of jumping straight to a bare label. This is the standard
CARP structure (Sun et al., 2023, "Text Classification via Large Language
Models"): surface-level clues first, a diagnostic reasoning process second,
the label decided last and required to match that reasoning.

The prompt text lives in prompts/task1_carp.txt, not here - this module only
loads it and builds the chat messages, exactly like `prompts.py`. The reply
schema adds two keys (`clues`, `reasoning`) around the same `label` key Task 1
already parses, so `json_parser.parse()` needs no changes: `clues`/`reasoning`
just show up as `extra_keys` on an otherwise normal parse.
"""

from __future__ import annotations

from .utils import PROJECT_ROOT

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER TEMPLATE ==="


def _load_template() -> tuple[str, str]:
    """Split prompts/task1_carp.txt into (system_text, user_template)."""
    path = PROJECT_ROOT / "prompts" / "task1_carp.txt"
    text = path.read_text(encoding="utf-8")
    _, rest = text.split(_SYSTEM_MARKER, 1)
    system, user_template = rest.split(_USER_MARKER, 1)
    return system.strip(), user_template.strip()


SYSTEM_PROMPT, USER_TEMPLATE = _load_template()


# Appended on retries when the previous attempt produced unparsable output.
# Attempt 1 still asks for the full CARP structure; attempt 2 gives up on
# recovering the reasoning and just salvages a bare label, same philosophy
# as prompts.py's escalation.
RETRY_SUFFIXES = [
    "",  # attempt 0: the plain CARP prompt
    '\n\nYour previous reply was not valid JSON. Follow the same clue -> reasoning -> label '
    'process, but reply with the JSON object only, starting with { and ending with }.',
    '\n\nFINAL ATTEMPT. Skip the clue and reasoning fields. Output exactly one of these two '
    'strings and nothing else:\n{"label": "positive"}\n{"label": "negative"}',
]


def build_messages(text: str, attempt: int = 0) -> list[dict[str, str]]:
    """Return the chat messages for one review.

    `attempt` selects the retry suffix; 0 is the pristine CARP prompt used for
    the headline numbers. Same contract as `prompts.build_messages`, so
    `inference.run()` can drive either one interchangeably.
    """
    suffix = RETRY_SUFFIXES[min(attempt, len(RETRY_SUFFIXES) - 1)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text) + suffix},
    ]


def prompt_metadata() -> dict[str, object]:
    """Snapshot of the prompt, stored alongside results for reproducibility."""
    return {
        "method": "carp",
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "retry_suffixes": RETRY_SUFFIXES,
        "n_examples_in_prompt": 0,  # zero-shot CARP: no input/label pairs shown
    }
