"""Zero-shot prompt construction.

Design constraints from the task sheet:
  - a system message and a user message
  - explicit definitions for positive / negative
  - NO task examples anywhere (that is what makes it zero-shot)
  - the output must be constrained to {"label": "positive|negative"}

The prompt text itself lives in prompts/task1_zeroshot.txt, not here - this
module only loads it and builds the chat messages. The JSON skeleton shown in
the system prompt is a *format* specification, not a labelled example: it
never pairs an input review with an output label, so the zero-shot condition
holds.
"""

from __future__ import annotations

from .utils import PROJECT_ROOT

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER TEMPLATE ==="


def _load_template() -> tuple[str, str]:
    """Split prompts/task1_zeroshot.txt into (system_text, user_template)."""
    path = PROJECT_ROOT / "prompts" / "task1_zeroshot.txt"
    text = path.read_text(encoding="utf-8")
    _, rest = text.split(_SYSTEM_MARKER, 1)
    system, user_template = rest.split(_USER_MARKER, 1)
    return system.strip(), user_template.strip()


SYSTEM_PROMPT, USER_TEMPLATE = _load_template()


# Appended on retries when the previous attempt produced unparsable output.
# Escalates in strictness rather than repeating the same instruction verbatim.
RETRY_SUFFIXES = [
    "",  # attempt 0: the plain prompt
    '\n\nYour previous reply was not valid JSON. Reply with the JSON object only, '
    'starting with { and ending with }.',
    '\n\nFINAL ATTEMPT. Output exactly one of these two strings and nothing else:\n'
    '{"label": "positive"}\n{"label": "negative"}',
]


def build_messages(text: str, attempt: int = 0) -> list[dict[str, str]]:
    """Return the chat messages for one review.

    `attempt` selects the retry suffix; 0 is the pristine zero-shot prompt used
    for the headline numbers.
    """
    suffix = RETRY_SUFFIXES[min(attempt, len(RETRY_SUFFIXES) - 1)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text) + suffix},
    ]


def prompt_metadata() -> dict[str, object]:
    """Snapshot of the prompt, stored alongside results for reproducibility."""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "retry_suffixes": RETRY_SUFFIXES,
        "n_examples_in_prompt": 0,  # zero-shot: no input/label pairs shown
    }
