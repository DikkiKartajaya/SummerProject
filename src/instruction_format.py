"""Turn a (review, label) row into an instruction-formatted training example.

Task 3 fine-tunes with the model's own chat template, not a bespoke prompt
string:

    system    : label definitions + the JSON output format
    user      : "Movie review:\\n<review>\\n\\nSentiment:"
    assistant : {"label": "positive"}          <- the supervised target

The assistant turn is byte-identical to the answer Task 1's zero-shot prompt
asks for, so a fine-tuned model is scored by the *same* `json_parser.parse`
and `metrics.compute` as the zero-/few-shot baselines - the only thing that
changed is the weights.

`build_supervised_example()` returns `input_ids` + `labels` with the prompt
masked to -100, so the loss is computed on the answer tokens only. Long
reviews are truncated to `max_review_tokens` *before* templating, so the
system prompt and the answer are never cut.

The prompt text lives in `prompts/task3_instruction.txt` (same SYSTEM /
USER TEMPLATE / ASSISTANT TEMPLATE marker convention as the other prompt
files); this module only loads it and assembles messages.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .utils import LABELS, PROJECT_ROOT

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER TEMPLATE ==="
_ASSISTANT_MARKER = "=== ASSISTANT TEMPLATE ==="

DEFAULT_PROMPT_FILE = "prompts/task3_instruction.txt"


@lru_cache(maxsize=4)
def _load_template(prompt_file: str) -> tuple[str, str, str]:
    """Split the prompt file into (system, user_template, assistant_template)."""
    text = (PROJECT_ROOT / prompt_file).read_text(encoding="utf-8")
    _, rest = text.split(_SYSTEM_MARKER, 1)
    system, rest = rest.split(_USER_MARKER, 1)
    user_template, assistant_template = rest.split(_ASSISTANT_MARKER, 1)
    return system.strip(), user_template.strip(), assistant_template.strip()


def target_text(label: str, prompt_file: str = DEFAULT_PROMPT_FILE) -> str:
    """The assistant turn for a gold label, e.g. `{"label": "positive"}`."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    _, _, assistant_template = _load_template(prompt_file)
    return assistant_template.format(label=label)


def build_messages(
    text: str, label: str | None = None, prompt_file: str = DEFAULT_PROMPT_FILE
) -> list[dict[str, str]]:
    """Chat messages for one review.

    `label=None` -> just system + user (an inference prompt). Pass a gold
    label to also append the assistant turn (a training example).
    """
    system, user_template, _ = _load_template(prompt_file)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(text=text)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": target_text(label, prompt_file)})
    return messages


def _truncate_review(text: str, tokenizer: Any, max_review_tokens: int) -> str:
    """Clip a review to `max_review_tokens` tokens, keeping the start.

    Sentiment is almost always judgeable from the opening of a review, and
    keeping the *start* (rather than a middle window) matches how Task 1
    truncates. Done on the raw review text before templating so the system
    prompt and the answer are always intact.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_review_tokens:
        return text
    return tokenizer.decode(ids[:max_review_tokens], skip_special_tokens=True)


def _template_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    """`apply_chat_template` -> a plain list[int] of token ids (version-tolerant)."""
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
    )
    ids = out["input_ids"]
    # Some versions nest a single conversation one level deep.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_prompt_ids(
    text: str,
    tokenizer: Any,
    max_review_tokens: int = 896,
    prompt_file: str = DEFAULT_PROMPT_FILE,
) -> list[int]:
    """Tokenised inference prompt (system + user + `assistant` header), ready
    for `model.generate`. No answer, no EOS."""
    review = _truncate_review(text, tokenizer, max_review_tokens)
    return _template_ids(
        tokenizer, build_messages(review, label=None, prompt_file=prompt_file),
        add_generation_prompt=True,
    )


def build_supervised_example(
    text: str,
    label: str,
    tokenizer: Any,
    max_review_tokens: int = 896,
    max_seq_len: int = 1024,
    prompt_file: str = DEFAULT_PROMPT_FILE,
) -> dict[str, list[int]] | None:
    """One tokenised training example: `{input_ids, labels}` with the prompt
    tokens masked to -100.

    Returns None if the full example exceeds `max_seq_len` even after the
    review was truncated (dropped rather than cutting the answer off).
    """
    review = _truncate_review(text, tokenizer, max_review_tokens)

    prompt_ids = _template_ids(
        tokenizer, build_messages(review, label=None, prompt_file=prompt_file),
        add_generation_prompt=True,
    )
    full_ids = _template_ids(
        tokenizer, build_messages(review, label=label, prompt_file=prompt_file),
        add_generation_prompt=False,
    )
    if len(full_ids) > max_seq_len:
        return None

    # The assistant header tokens are identical in both encodings; the answer
    # only starts where they diverge. Masking by the common-prefix length
    # (rather than assuming `prompt_ids` is a literal prefix) is robust to a
    # BPE merge at the prompt/answer boundary - at worst one straddling token
    # is left unmasked, which is the standard SFT convention.
    n_common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        n_common += 1
    if n_common >= len(full_ids):
        raise RuntimeError("instruction masking produced an empty target span")

    labels = [-100] * n_common + full_ids[n_common:]
    return {"input_ids": full_ids, "labels": labels}


def prompt_metadata(prompt_file: str = DEFAULT_PROMPT_FILE) -> dict[str, Any]:
    """Snapshot of the instruction format, stored alongside results."""
    system, user_template, assistant_template = _load_template(prompt_file)
    return {
        "prompt_file": prompt_file,
        "system_prompt": system,
        "user_template": user_template,
        "assistant_template": assistant_template,
        "target_examples": {lab: target_text(lab, prompt_file) for lab in LABELS},
        "loss_on": "assistant tokens only (prompt masked to -100)",
    }
