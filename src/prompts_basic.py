"""Basic prompt construction for Task 1 - the minimal baseline.

A third point of comparison alongside `prompts.py` (structured zero-shot:
label definitions, disambiguation rules, JSON schema) and `prompts_carp.py`
(the same rules plus a Clue -> Reasoning -> Label chain): this one has none
of that prompt engineering. No label definitions, no rules, no JSON schema,
no reasoning step - just a three-line instruction, the review, and a
"Sentiment:" cue, asking the model to answer with a single bare word. It
exists to measure what the other two prompts' structure is actually buying,
if anything.

The prompt text lives in prompts/task1_basic.txt, not here - this module only
loads it and builds the chat messages, exactly like `prompts.py`. The reply
is a bare "POSITIVE"/"NEGATIVE" word rather than JSON, but that needs no
parser changes either: `json_parser.parse()`'s last, most lenient rung
(`bare_label`) exists precisely to recover a single label word with no JSON
around it - this prompt just always lands there, rather than occasionally.
"""

from __future__ import annotations

from .utils import PROJECT_ROOT

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER TEMPLATE ==="


def _load_template() -> tuple[str, str]:
    """Split prompts/task1_basic.txt into (system_text, user_template)."""
    path = PROJECT_ROOT / "prompts" / "task1_basic.txt"
    text = path.read_text(encoding="utf-8")
    _, rest = text.split(_SYSTEM_MARKER, 1)
    system, user_template = rest.split(_USER_MARKER, 1)
    return system.strip(), user_template.strip()


SYSTEM_PROMPT, USER_TEMPLATE = _load_template()


# Appended on retries when the previous attempt produced no recoverable label.
# There is no JSON structure to remind the model to follow here - just a
# blunter restatement of "one word, nothing else".
RETRY_SUFFIXES = [
    "",  # attempt 0: the plain basic prompt
    "\n\nYour previous reply did not contain exactly one label. "
    "Reply with exactly one word: POSITIVE or NEGATIVE.",
    "\n\nFINAL ATTEMPT. Output exactly one word and nothing else: "
    "POSITIVE or NEGATIVE.",
]


def build_messages(text: str, attempt: int = 0) -> list[dict[str, str]]:
    """Return the chat messages for one review.

    `attempt` selects the retry suffix; 0 is the pristine basic prompt used
    for the headline numbers. Same contract as `prompts.build_messages`, so
    `inference.run()` can drive any of the three Task 1 prompts interchangeably.
    """
    suffix = RETRY_SUFFIXES[min(attempt, len(RETRY_SUFFIXES) - 1)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text) + suffix},
    ]


def prompt_metadata() -> dict[str, object]:
    """Snapshot of the prompt, stored alongside results for reproducibility."""
    return {
        "method": "basic",
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "retry_suffixes": RETRY_SUFFIXES,
        "n_examples_in_prompt": 0,
    }
