"""Task 2 few-shot prompt construction.

Loads the static prompt files under `prompts/` and builds the chat messages
for one review. The prompt text itself lives in the .txt files, not here -
this module only splits each file into its system/user halves and
substitutes {text}. `curated`, `basic`, `mixed`, and `contrastive` all share
the same system prompt as `curated` (`basic` strips it down instead), so
`basic` isolates the system prompt's rules/definitions as the only
variable, not example choice; `mixed` keeps `curated`'s system prompt but
swaps in 8 demos (4 per class) drawn from
`data/processed/mixed_sentiment_samples.csv`, reviews that contain both
positive and negative sentences, to test whether the model still follows
the reviewer's overall verdict when the examples themselves are
sentiment-mixed; `contrastive` instead swaps in 8 demos (4 per class, as 4
positive/negative pairs) drawn from `data/processed/contrastive_samples.csv`,
near-duplicate reviews (paired by high lexical/Jaccard overlap) that land on
opposite labels, to test whether the model anchors on surface wording or
genuinely judges the reviewer's verdict. `basic_mixed` and
`basic_contrastive` reuse `mixed`/`contrastive`'s demos *verbatim* under
`basic`'s stripped-down system prompt, isolating the rules/definitions the
same way `basic` relates to `curated`.

`random` and `basic_random` isolate example *selection* instead: their .txt
files carry the same rules as `curated` and `basic` respectively
(`random` = curated's rules + unfiltered examples; `basic_random` = basic's
bare-minimum rules + the *same* unfiltered examples, so it isolates example
selection the same way while stripped of the rules/definitions, mirroring
how `basic` relates to `curated`). Their 4 demo *reviews* (2 positive + 2
negative, one fixed unfiltered draw from `train.csv`, `random_state=42` - no
length/quality curation, which is the whole point of this ablation) are
baked directly into `task2_fewshot_random.txt`/`task2_fewshot_basic_random.txt`
as literal text - exactly like `curated`'s/`basic`'s demos - not templated in
at runtime. `data/processed/random_fewshot_samples.csv`
(`scripts/10_select_random_fewshot_samples.py`) is the record of that draw:
it is what the two .txt files' example text was generated from, the same
role `selected_sample.csv` plays for `curated`/`basic`'s hand-picked demos.
Regenerating the CSV with a different seed does not change the prompt files
by itself - the baked-in text has to be regenerated from it too (see that
script's docstring). Only the *review* is unfiltered; each demo's JSON
answer (confidence, evidence) is hand-authored the same way `curated`'s/
`basic`'s are, citing what the review actually says rather than a generic
templated sentence - the whole comparison isolates which reviews got
selected, not the quality of the demonstrated answer.

`foundational`, `structured`, and `comprehensive` isolate the *instruction*
instead, as three steps of one ladder rather than a pair of extremes. All
three reuse `curated`'s hand-picked demos verbatim, so example choice is
held fixed and only the instruction above them grows: `foundational` gives
the bare task, the dataset in one sentence, and a numbered sentiment list,
with no definition of either label; `structured` adds a definition per
label plus the tie-break rule for reviews that praise and criticise at
once; `comprehensive` adds a background section on how the IMDB dataset was
built (star-rating labels, the discarded 5-6 band, at most 30 reviews per
film) and on how its reviews are written (plot summary before verdict,
<br /> artifacts, quoted star ratings, sarcasm). They are adapted from the
three-stage prompt ladder in kc6699c/LLM4OralHistoryAnalysis
(Sentiment_Prompts.md), rewritten for IMDB's two labels and for the
label+confidence+evidence schema the rest of Task 2 uses. Against `basic`
(no rules) and `curated` (rules written as prose) they place the same
question - how much instruction the demos actually need - on a graded
scale.

Because the two files hold ordinary static text like every other strategy,
`build_messages` treats all of `STRATEGIES` identically - no special-casing.
`prompt_metadata` still tags `random`/`basic_random` with `fewshot_seed` and
`fewshot_example_uids`, read from the CSV, purely as reproducibility
metadata (which rows the baked-in text came from), not because anything is
computed from it at request time.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .utils import PROJECT_ROOT

STRATEGIES = {
    "curated": "task2_fewshot.txt",
    # Same rules as "curated", but the 2 examples are an unfiltered
    # precomputed draw instead of hand-picked. See module docstring.
    "random": "task2_fewshot_random.txt",
    # Same 6 curated demos as "curated" (3 per class), but a bare-minimum
    # system prompt: no label definitions, no disambiguation rules - just the
    # task, the JSON schema, and the examples. The Task 2 counterpart of Task
    # 1's basic prompt (prompts_basic.py), adapted to keep the
    # label+confidence+evidence schema Task 2's quality metrics need.
    "basic": "task2_fewshot_basic.txt",
    # "basic"'s bare-minimum rules + the *same* unfiltered examples as
    # "random". See module docstring.
    "basic_random": "task2_fewshot_basic_random.txt",
    # Same rules as "random"/"curated", but the 4 demos (2 per class) are the
    # reviews closest to their class centroid in
    # sentence-transformers/all-MiniLM-L6-v2 embedding space - the "prototype"
    # reviews - instead of an unfiltered random draw. Pairs with
    # "basic_representative" to isolate example *selection* (random vs
    # representative) the same way "random"/"basic_random" isolate the rules.
    # data/processed/representative_example.csv
    # (scripts/11_select_representative_samples.py) is the record of that draw.
    "representative": "task2_fewshot_representative.txt",
    # "basic"'s bare-minimum rules + the *same* representative examples as
    # "representative".
    "basic_representative": "task2_fewshot_basic_representative.txt",
    # Same system prompt as "curated", but 8 demos instead of 6 (4 per
    # class), drawn from data/processed/mixed_sentiment_samples.csv: reviews
    # that mix praise and criticism yet still land clearly
    # positive/negative. Isolates whether demoing mixed-sentiment reviews
    # (vs. curated's clear-cut ones) changes how well the model resolves
    # mixed reviews at inference time.
    "mixed": "task2_fewshot_mixed.txt",
    # Same system prompt as "curated", but 8 demos instead of 6 (4 per class,
    # as 4 positive/negative pairs), drawn from
    # data/processed/contrastive_samples.csv: reviews paired by high lexical
    # overlap (same movie/topic, similar wording) yet opposite labels.
    # Isolates whether demoing near-duplicate contrastive pairs sharpens the
    # model's reliance on the reviewer's actual verdict rather than surface
    # phrasing.
    "contrastive": "task2_fewshot_contrastive.txt",
    # "basic"'s bare-minimum system prompt (no label definitions, no
    # disambiguation rules) + the *same* mixed-sentiment demos as "mixed",
    # unchanged. Pairs with "mixed" to isolate the system prompt's
    # rules/definitions the same way "basic" relates to "curated".
    "basic_mixed": "task2_fewshot_basic_mixed.txt",
    # "basic"'s bare-minimum system prompt + the *same* contrastive
    # pair demos as "contrastive", unchanged. Pairs with "contrastive" to
    # isolate the system prompt's rules/definitions.
    "basic_contrastive": "task2_fewshot_basic_contrastive.txt",
    # Step 1 of the instruction ladder (see module docstring): `curated`'s
    # demos, unchanged, under the bare task + a one-sentence description of
    # the dataset + a numbered sentiment list, with neither label defined.
    "foundational": "task2_fewshot_foundational.txt",
    # Step 2: "foundational" + a definition per label + the tie-break rule
    # for reviews that praise and criticise at once. Same demos.
    "structured": "task2_fewshot_structured.txt",
    # Step 3: "structured" + a background section on how the IMDB dataset
    # was built and how its reviews are written. Same demos.
    "comprehensive": "task2_fewshot_comprehensive.txt",
}

_SYSTEM_MARKER = "=== SYSTEM ==="
_USER_MARKER = "=== USER TEMPLATE ==="


def _load_template(strategy: str) -> tuple[str, str]:
    """Split a prompt file into (system_text, user_template)."""
    path = PROJECT_ROOT / "prompts" / STRATEGIES[strategy]
    text = path.read_text(encoding="utf-8")
    _, rest = text.split(_SYSTEM_MARKER, 1)
    system, user_template = rest.split(_USER_MARKER, 1)
    return system.strip(), user_template.strip()


_TEMPLATES = {name: _load_template(name) for name in STRATEGIES}


def build_messages(strategy: str, text: str) -> list[dict[str, str]]:
    """Chat messages for one review, under the given few-shot strategy. Every
    strategy's prompt text (examples included) is static - loaded once at
    import time - so this just fills in the review text."""
    system, user_template = _TEMPLATES[strategy]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(text=text)},
    ]


# --------------------------------------------------------------------------
# "random" / "basic_random" reproducibility metadata: which train.csv rows
# the two .txt files' baked-in examples came from (see module docstring +
# scripts/10_select_random_fewshot_samples.py). Not used to build the
# prompt - the prompt text is already static - only to record provenance
# alongside the results.
# --------------------------------------------------------------------------
RANDOM_STRATEGIES = {"random", "basic_random"}
_RANDOM_SAMPLES_CSV = PROJECT_ROOT / "data" / "processed" / "random_fewshot_samples.csv"


@lru_cache(maxsize=1)
def _load_random_samples() -> pd.DataFrame:
    """The single precomputed draw (2 positive + 2 negative rows) that
    `task2_fewshot_random.txt`/`task2_fewshot_basic_random.txt`'s baked-in
    examples were generated from. The file must carry exactly one seed."""
    if not _RANDOM_SAMPLES_CSV.exists():
        raise FileNotFoundError(
            f"{_RANDOM_SAMPLES_CSV} not found. Run "
            "`python scripts/10_select_random_fewshot_samples.py` to "
            "regenerate the record of the 'random'/'basic_random' demo draw."
        )
    df = pd.read_csv(_RANDOM_SAMPLES_CSV)
    seeds = df["seed"].unique().tolist()
    if len(seeds) != 1:
        raise ValueError(
            f"{_RANDOM_SAMPLES_CSV} must contain rows for exactly one seed, "
            f"found {seeds}. Re-run scripts/10_select_random_fewshot_samples.py."
        )
    return df


def random_fewshot_seed() -> int:
    """The one seed random_fewshot_samples.csv was drawn with, read from the
    file itself rather than duplicated as a constant here."""
    return int(_load_random_samples()["seed"].iloc[0])


def prompt_metadata(strategy: str) -> dict[str, object]:
    """Snapshot of one strategy's prompt, stored alongside its results for
    reproducibility - the Task 2 counterpart of `prompts.prompt_metadata()`.

    For strategies in `RANDOM_STRATEGIES`, also records `fewshot_seed` and
    `fewshot_example_uids` - which train.csv rows the baked-in examples were
    generated from (see module docstring), so the draw can be audited later.
    """
    system, user_template = _TEMPLATES[strategy]
    extra: dict[str, object] = {}
    if strategy in RANDOM_STRATEGIES:
        extra = {
            "fewshot_seed": random_fewshot_seed(),
            "fewshot_example_uids": _load_random_samples()["uid"].tolist(),
        }
    return {
        "strategy": strategy,
        "source_file": STRATEGIES[strategy],
        "system_prompt": system,
        "user_template": user_template,
        "n_examples_in_prompt": system.count("\nReview:"),
        **extra,
    }
