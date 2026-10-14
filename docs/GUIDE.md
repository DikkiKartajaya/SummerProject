# Tasks 1–3 — Zero-Shot / Few-Shot / QLoRA Sentiment Analysis

Tasks 1–2 use **Llama-3.2-3B-Instruct**; Task 3 (QLoRA) fine-tunes
**Llama-3.1-8B-Instruct**. All three share the frozen IMDB split.

A reproducible, no-training baseline for binary sentiment classification on the
HuggingFace **[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)**
movie review dataset, built to run locally on a single **RTX 5080 (16 GB)**.

**Task 1 (zero-shot)** — no examples in the prompt at all:

- loads the dataset straight from the Hub (`stanfordnlp/imdb`, binary
  `positive` / `negative`, no relabeling needed) and creates a train/val/test
  split **exactly once** (reserved for the later fine-tuning tasks — Task 1
  itself scores the full pool); EDA is a separate, re-runnable step
- prompts an instruction-tuned LLM with a zero-shot system+user prompt, no examples
- forces a constrained JSON answer, parses and validates it, and retries failures
- reports accuracy, macro-F1, per-class precision/recall/F1, and a confusion
  matrix, plus invalid-output rate, inference latency, token usage and peak
  GPU memory

**Task 2 (few-shot)** — compares four prompting strategies:

- **curated** (4 hand-picked, balanced examples, 2 per class + rules) vs.
  **random** (4 unfiltered examples, 2 per class + the same rules) vs.
  **basic** (the curated examples, rules stripped out) vs. **basic_random**
  (`random`'s exact same 4 examples, rules stripped out), tested on `val.csv`
  before picking a winner
- the model must answer with a label, a confidence score, and one sentence of
  supporting evidence — no chain-of-thought
- adds three quality dimensions beyond label accuracy: evidence relevance,
  confidence calibration, and format compliance
- reuses Task 1's model loading, metrics, and reporting; see §11 below

Both tasks record the exact environment (Python, PyTorch, CUDA, Transformers,
PEFT, GPU, OS).

---

## 1. Requirements

| | |
|---|---|
| GPU | NVIDIA RTX 5080 (16 GB) or any CUDA GPU with ≥ 8 GB |
| Driver | 570+ (needed for CUDA 12.8) |
| Python | 3.10 – 3.12 (3.11 recommended) |
| Disk | ~10 GB (model weights + venv) |

> **RTX 50-series note.** Blackwell GPUs are compute capability `sm_120`. Only
> PyTorch **≥ 2.7 built against CUDA 12.8** ships kernels for it. A default
> `pip install torch` may give you a CUDA 12.4 build that imports fine and then
> fails at the first matmul. `setup.sh` installs the right wheel and smoke-tests it.

---

## 2. Setup

### Linux / WSL2

```bash
git clone <this-repo> llm-sentiment && cd llm-sentiment
bash setup.sh                    # venv + CUDA 12.8 PyTorch + creates .env
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
cd llm-sentiment
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

<details>
<summary>Manual setup, if you prefer</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate                       # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
</details>

Verify the GPU is really usable:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect something like: 2.7.0+cu128 NVIDIA GeForce RTX 5080 (12, 0)
```

---

## 3. Get the dataset

Nothing to download by hand. `scripts/01_prepare_data.py` pulls
[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) straight
from the Hub the first time it runs, via `datasets.load_dataset()`, and caches
it locally under `~/.cache/huggingface`. The dataset is public, but the loader
still accepts the same `HF_TOKEN` used for the gated model checkpoint (§4) —
one credential, one source of truth.

### What each HF split is for

The dataset ships three splits; the design mirrors the original two-CSV setup
one-for-one:

```
train (25k, labeled) ──split 80/10/10──> train   → Tasks 2-4 learn from this
                                         val     → pick checkpoints, tune
                                         test    → the final reported score
test  (25k, labeled) ───────────────────> holdout → independent cross-check,
                                                    never split, never trained on
unsupervised (50k, label = -1) ─────────> not used anywhere in this project
```

| Role | Comes from | Used by |
|---|---|---|
| `train` | HF `train` split (80%) | Task 3 QLoRA fine-tuning |
| `val` | HF `train` split (10%) | checkpoint selection, early stopping |
| `test` | HF `train` split (10%) | the headline number for a fine-tuned model |
| `all` (pool) | all of the HF `train` split | **Task 1** samples from here |
| `holdout` | HF `test` split | sanity-check that a score generalises |

**Why keep a separate holdout?** Your `test` set comes from the same HF split as
`train`, so it shares the same distribution. A good score there does not prove
the model generalises — it might just have learned that split's quirks. The
holdout is IMDB's own labeled `test` split and never influences any decision,
so comparing the two scores tells you which it is. Similar scores → real
generalisation. Holdout much lower → overfitting.

Task 1 is unaffected by all of this: with no training there is nothing to leak,
so it samples from the full pool. You can still score the holdout explicitly:

```bash
python scripts/03_run_task1.py --source holdout --n-samples all --run-id zs_holdout
```

**Overlap protection.** The two HF splits are meant to be disjoint, but any row
appearing in both is removed from the holdout automatically, and the count is
logged. A row that sits in `train` *and* in the "untouched" holdout would
defeat the whole point.

### Label handling

IMDB is natively binary — `positive` / `negative`, mapped from the dataset's
`0`/`1` integer `label` column. There is no third class and no remapping step:
every row that carries a real label (i.e. not the `unsupervised` split's `-1`
placeholder) is used as-is. The reviews also keep literal `<br />` line breaks
from the source HTML; the cleaning step collapses those to spaces before any
prompt or figure sees the text.

---

## 4. Credentials (access token)

`meta-llama/Llama-3.2-3B-Instruct` is a **gated** repo, so it needs a
HuggingFace access token.

**Step 1 — get a token.** Go to <https://huggingface.co/settings/tokens> and
create one with **Read** access.

**Step 2 — accept the licence** at
<https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>. A valid token is not
enough on its own; the licence has to be accepted by your account.

**Step 3 — put the token in `.env`.** `setup.sh` already created one from the
template; otherwise `cp .env.example .env`. Then edit it:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

That's it — every script loads `.env` on demand and reads `HF_TOKEN` from the
environment. Weights (~6.5 GB) download on first use and are cached in
`~/.cache/huggingface`.

You can also pass a token explicitly with `--hf-token`, but avoid it in
practice: it ends up in your shell history. `.env` is the recommended path for
a local machine; a real environment variable (`export HF_TOKEN=...`) works too
and takes precedence over `.env` if both are set.

### Token safety

- `.env` is gitignored (`.env.example` is the committed template) — do not
  commit it, and `setup.sh` sets it to `chmod 600`
- no script ever prints, logs, or writes the token value anywhere; the only
  thing recorded is whether one is set (`env_record.json`'s `hf_token_set`
  boolean), so that file is safe to attach to a report
- if a token leaks, revoke it at <https://huggingface.co/settings/tokens>

> **Prefer no gate at all?** Set `model.checkpoint` in `configs/task1.yaml` to an
> ungated instruct model such as `Qwen/Qwen2.5-3B-Instruct`. It then runs with no
> token; nothing else changes.

### Offline / already cached

Once the weights are cached you can skip the Hub entirely:

```bash
HF_HUB_OFFLINE=1 python scripts/03_run_task1.py
```

---

## 5. Run

Four steps, in order.

```bash
# 0) record the environment
python scripts/00_record_env.py

# 1) download + clean + create the frozen split (run ONCE)
python scripts/01_prepare_data.py

# 2) EDA on the cleaned pool (safe to re-run any time after step 1)
python scripts/02_run_eda.py

# 3) the zero-shot baseline
python scripts/03_run_task1.py

# 4) (optional) compare every run you have done so far
python scripts/04_summarize_runs.py
```

Before a long GPU job, the offline tests take about two seconds and catch most
wiring mistakes:

```bash
python tests/test_offline.py         # parser, prompt, metrics, dataset, env
python tests/test_pipeline_mock.py   # retry loop + checkpoint, with a fake model
```

### Useful flags

```bash
# more examples (spec minimum and default are both 200)
python scripts/03_run_task1.py --n-samples 1000

# the whole cleaned pool (~25k rows — expect a while)
python scripts/03_run_task1.py --n-samples all --run-id full_pool

# 4-bit ablation, same prompt and seed
python scripts/03_run_task1.py --precision nf4 --run-id ablation_nf4

# larger batch if VRAM allows; smaller if you hit OOM
python scripts/03_run_task1.py --batch-size 16

# score a specific partition instead of the full pool
python scripts/03_run_task1.py --source test

# score the independent holdout (all ~25k rows of the HF `test` split)
python scripts/03_run_task1.py --source holdout --n-samples all --run-id zs_holdout

# skip the batch-size-1 latency probe
python scripts/03_run_task1.py --latency-samples 0

# run from the local cache with no Hub calls at all
HF_HUB_OFFLINE=1 python scripts/03_run_task1.py

# a different prompting strategy (see §12): zero_shot (default) | carp | basic
python scripts/03_run_task1.py --strategy carp
```

A run is resumable. While it's in progress, `outputs/task1/<run_id>/predictions.jsonl`
holds a crash-safe checkpoint (one line per scored row); if the run is
interrupted, re-running with the **same `--run-id`** picks up where it left
off instead of starting over. Use `--no-resume` to start clean. Once a run
finishes successfully, that checkpoint is deleted - every row it held is
already in `predictions_task1.csv`, so keeping both would just duplicate data.

---

## 6. What you get

```
outputs/
├── env_record.json                 # environment snapshot
├── eda/
│   ├── eda_stats.json
│   ├── eda_label_distribution.png
│   ├── eda_length_distribution.png
│   └── eda_top_tokens.png
└── task1/
    ├── LATEST_results.md           # copy of the most recent run's summary
    └── <run_id>/
        ├── run_config.json         # the exact config used
        ├── env_record.json         # environment at run time
        ├── prompt.json             # system prompt + template + retry suffixes
        ├── predictions_task1.csv   # every scored row: text, gold, predicted, parse strategy, raw output...
        ├── errors_task1.csv        # misclassified rows only
        ├── invalid_outputs_task1.csv # rows that never produced a parseable label
        ├── metrics_task1.json      # accuracy, macro-F1, per-class, confusion matrix
        ├── results_task1.md        # the readable summary table
        ├── run_stats.json          # latency, tokens, GPU memory
        ├── run_meta.json
        └── confusion_matrix_task1.png

data/processed/                     # the frozen split — created once
├── clean_pool.csv                  # all of the HF `train` split, cleaned
├── train.csv / val.csv / test.csv
├── holdout.csv                     # HF `test` split, de-overlapped
└── split_manifest.json             # sizes, label counts
```

`results_task1.md` is the file to read first. It reports accuracy, macro-F1,
per-class precision/recall/F1, a confusion matrix, the invalid-output rate,
and performance (latency, tokens generated, peak GPU memory).

---

## 7. The split is created only once

`scripts/01_prepare_data.py` checks whether the split files already exist in
`data/processed/`:

- files exist → reused as-is, nothing is rewritten
- files missing → the split is created

That keeps Tasks 2–4 training and evaluating on exactly the same partitions
without an accidental re-split. `--force` discards the existing split and
creates a new one — only do that if you deliberately changed the cleaning
rules, the ratios, or the seed, and mean it.

Task 1 does not use the split at all: it evaluates a stratified sample of the
**full cleaned pool**, because a zero-shot baseline involves no training and has
nothing to leak.

The holdout is refreshed on every run (it is never partitioned), so a later
change to `data.holdout_split` just needs a re-run of `01_prepare_data.py` —
the holdout appears without touching an already-frozen split.

---

## 8. Tuning for your GPU

| Setting | Where | Effect |
|---|---|---|
| `model.precision` | config / `--precision` | `bf16` ≈ 6.5 GB, `nf4` ≈ 2.5 GB, `int8` ≈ 3.5 GB |
| `task1.batch_size` | config / `--batch-size` | linear VRAM cost; 8 is safe on 16 GB, but IMDB's longer reviews cost more per-row than tweets did — lower it first if you hit OOM |
| `model.max_input_tokens` | config | truncation length for the full templated prompt (system message + review), measured with the tokenizer, not estimated from word counts; 1536 clears p99.5 (~1429 tokens) with room to spare — a truncated prompt can cut off the trailing `JSON:` cue and produce garbage instead of a clean parse failure, so raise it if `invalid_outputs_task1.csv` shows garbled output on long reviews |
| `decoding.max_new_tokens` | config | 24 is plenty for `{"label":"positive"}` |

If you hit `CUDA out of memory`: lower `--batch-size` first, then switch to
`--precision nf4`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution` | Wrong PyTorch build. Reinstall from the `cu128` index (step 2). |
| A gated-repo / 401 error loading the model | Accept the licence on the model page **and** check `HF_TOKEN` in `.env`. Run `python scripts/00_record_env.py` to confirm a token is set. |
| Token in `.env` seems ignored | A non-empty `HF_TOKEN` exported in your shell wins over the file. Check with `echo $HF_TOKEN`. |
| Dataset fails to download / times out | `stanfordnlp/imdb` is fetched from the Hub on first use — check network access, or set `HF_HUB_OFFLINE=1` once it's cached. |
| `holdout.csv not found` when using `--source holdout` | Confirm `data.holdout_split: "test"` is set in the config and re-run `01_prepare_data.py` (no `--force` needed). |
| `CUDA out of memory` | `--batch-size 4`, or `--precision nf4`. |
| High invalid-output rate | Check `invalid_outputs_task1.csv` — it has the raw model output and parse error for every row that never produced a valid label. |

---

## 10. Reproducibility

Everything needed to reproduce a number is written next to it:

- `run_config.json` — model checkpoint + revision, precision, decoding params, seed
- `run_meta.json` — latency, token usage, peak GPU memory, and whether a
  credential source was an explicit argument or the environment (never the
  token itself)
- `prompt.json` — the exact prompt text and retry suffixes, with a version tag
- `env_record.json` — Python, PyTorch, CUDA, Transformers, PEFT, bitsandbytes,
  driver, GPU model and VRAM, OS, and whether an HF token is set (never the value)
- `eval_subset.csv` — the exact rows that were scored
- `predictions.jsonl` — the raw model output for every example

Seeds are set for `random`, `numpy` and `torch`; cuDNN runs in deterministic mode
and the main pass uses greedy decoding, so a re-run on the same hardware and
software reproduces the same predictions.

---

## 11. Task 2 — Few-shot prompt comparison

Same model, same dataset, same split as Task 1 (sections 1–4) — nothing extra
to install or configure. Just run `01_prepare_data.py` first if you haven't
already. Where Task 1 asks for a bare `{"label": ...}`, Task 2 asks for
`{"label": ..., "confidence": <0.0-1.0>, "evidence": "<one sentence>"}` and
compares several **few-shot prompting strategies**, all built the same way
Task 1's basic/zero-shot/CARP ablations are (§12) — one dict entry in
`src/prompts2.STRATEGIES`, everything else shared. The four below are the
core ablation pairs; `src/prompts2.STRATEGIES` also carries
`representative` / `basic_representative` (demos closest to each class
centroid in embedding space — see `scripts/11_select_representative_samples.py`),
and the matched pairs `mixed` / `basic_mixed` and `contrastive` /
`basic_contrastive` (each `basic_*` reuses the other's demos verbatim under
`basic`'s stripped-down system prompt, isolating the rules the same way
`basic` relates to `curated`), plus the instruction ladder `foundational` /
`structured` / `comprehensive` (below). On the final **test** comparison
(`outputs/task2/final_all/`) **`curated`** wins (macro-F1 ≈ 0.94, invalid
≈ 0.3%, schema-compliance ≈ 99.5%), and it is the prompt §13's sensitivity
check perturbs. (`basic_representative` leads on `val` but not on `test`.)

- **curated** (`prompts/task2_fewshot.txt`) — 4 balanced examples (2 per
  class), hand-picked from `data/processed/selected_sample.csv` for being
  short, clear, and unambiguous, alongside label definitions and 4
  disambiguation rules (the same ones Task 1's zero-shot prompt uses).
- **random** (`prompts/task2_fewshot_random.txt`) — the same rules as
  `curated`, but its 4 examples (2 positive + 2 negative) are an unfiltered,
  precomputed draw from `train.csv` instead of hand-picked ones, to see
  whether curation of the *examples* actually helps.
- **basic** (`prompts/task2_fewshot_basic.txt`) — the *same* 4 curated demos
  as `curated`, but with the rules and label definitions stripped out: just
  the task, the JSON schema, and the examples. Isolates the *rules'*
  contribution instead of the examples' — `curated` vs `basic` differ only in
  how much instruction surrounds the same two demonstrations.
- **basic_random** (`prompts/task2_fewshot_basic_random.txt`) — `basic`'s
  bare-minimum rules (no label definitions, no disambiguation rules), paired
  with `random`'s *exact same* 4 unfiltered demos. Isolates example selection
  the same way `random` does, but stripped of the rules, mirroring how
  `basic` relates to `curated`.

`random` and `basic_random` are a matched pair: their `.txt` files bake in
the identical 4 example blocks (same reviews, same JSON answers) as literal
text, so any difference between them is attributable only to the
rules/definitions in the system prompt, not to a different example draw.

#### The instruction ladder — `foundational` / `structured` / `comprehensive`

`basic` and `curated` put the same demos under *no* rules and under *full*
rules. These three grade that same question instead of taking it at its two
extremes: all three reuse `curated`'s 4 hand-picked demos **verbatim**, and
carry the identical output schema and user template as every other strategy,
so example choice is held fixed and only the instruction above the demos
grows.

- **foundational** (`prompts/task2_fewshot_foundational.txt`) — the bare
  task, the dataset in one sentence, and a numbered sentiment list. Neither
  label is defined. Sits between `basic` (no framing at all) and `structured`.
- **structured** (`prompts/task2_fewshot_structured.txt`) — `foundational`
  plus a definition per label and the tie-break rule for reviews that praise
  and criticise at once ("select the strongest one: the verdict the
  reviewer's final recommendation leans toward").
- **comprehensive** (`prompts/task2_fewshot_comprehensive.txt`) —
  `structured` plus a background section on how the IMDB dataset was built
  (labels derived from the reviewer's own star rating, the 5–6 band
  discarded as neutral, at most 30 reviews per film) and on how its reviews
  are written (plot summary before the verdict, `<br />` artifacts, quoted
  star ratings such as "2/10", sarcasm and mock praise).

The ladder is adapted from the three-stage prompt progression in
[kc6699c/LLM4OralHistoryAnalysis](https://github.com/kc6699c/LLM4OralHistoryAnalysis/blob/main/Sentiment_Prompts.md),
which grades Foundational → Structured → Comprehensive prompts over Densho
oral-history testimony. Rewritten here for IMDB's two labels and for Task 2's
`label` + `confidence` + `evidence` schema.

> **Token budget.** `comprehensive`'s system prompt is ~1,150 tokens against
> `curated`'s ~690, so with a long review it can exceed
> `model.max_input_tokens: 2048` in `configs/task2.yaml`: 66/2490 `val` rows
> (2.7%) and 75/2491 `test` rows (3.0%) overflow, versus 3 and 2 rows (0.1%)
> under `curated`. Tokenization truncates on the *right*, which cuts the
> trailing `JSON:` cue rather than the review, so an overflowing row returns
> an unparseable answer rather than a verdict on a shortened review. The
> longest row in either split needs 3,255 tokens under `comprehensive`, so
> raise `max_input_tokens` to 3,328 before reading anything into this
> strategy's invalid-output rate — note that doing so also changes
> `random`'s and `basic_random`'s existing numbers if those are re-run.

#### Where the random examples come from

`data/processed/random_fewshot_samples.csv` holds one fixed, reproducible
*record* of the draw — 2 positive + 2 negative reviews sampled unfiltered
from `train.csv` with `random_state=42` — generated once by:

```bash
python scripts/10_select_random_fewshot_samples.py
```

This file is not read at prompt-build time. Like `curated`'s/`basic`'s demos
(hand-picked from `selected_sample.csv`), `random`'s/`basic_random`'s 4
example blocks are baked directly into
`prompts/task2_fewshot_random.txt`/`prompts/task2_fewshot_basic_random.txt`
as static text — `src/prompts2.py` loads all six strategy files the same
uniform way, with no runtime templating for any of them. Only the *review*
text is unfiltered/random; each demo's JSON answer (confidence, evidence) is
still hand-authored to cite what that specific review actually says, the
same way `curated`'s/`basic`'s are — the ablation isolates which reviews got
selected, not the quality of the demonstrated answer. The CSV is what that
baked-in text was generated from and stays useful as an audit trail
(`fewshot_seed`/`fewshot_example_uids` in `random`'s/`basic_random`'s
`prompt.json`); re-running the script with a different `--seed` produces a
new draw but does **not** by itself change either `.txt` file — the baked-in
text has to be regenerated from it as a separate, deliberate step.

Beyond final-label accuracy/macro-F1 (identical scoring to Task 1, via the
shared `metrics.py`), each strategy is also scored on three quality
dimensions specific to the richer schema (`src/eval_task2.py`, all heuristic —
no extra model calls, no hidden judge):

| Dimension | What it measures | Key numbers |
|---|---|---|
| **Evidence relevance** | Is the evidence sentence actually grounded in the review, or generic/fabricated? Content-word overlap between evidence and review text. | `mean_relevance`, `evidence_present_rate`, `label_echo_rate` (evidence that just repeats the label word instead of citing the review) |
| **Confidence calibration** | When the model says "0.9 confident", is it right ~90% of the time? | Expected Calibration Error (`ece`), `brier_score`, a 10-bin reliability table |
| **Format compliance** | How often is the reply *exactly* the required 3-key JSON (not just recoverable via the lenient ladder)? | `schema_compliance_rate`, `strategy_breakdown`, `extra_keys_rate` |

### Data used

| File | Role |
|---|---|
| `data/processed/random_fewshot_samples.csv` | audit record of the fixed 4-row draw (2 positive + 2 negative, seed 42) that `random`'s/`basic_random`'s baked-in example blocks were generated from — see `scripts/10_select_random_fewshot_samples.py` |
| `data/processed/selected_sample.csv` | 12 hand-picked candidates (6 per class) — the pool the 4 demos shared by `curated` and `basic` were chosen from |
| `data/processed/val.csv` | used to compare the prompt strategies |
| `data/processed/test.csv` | used for the final 200-example headline number, once a strategy is picked |

### Run it

```bash
# optional: regenerate the audit-record CSV for random/basic_random's draw
# (the strategies themselves need no setup step — their example text is
# already baked into the .txt files)
python scripts/10_select_random_fewshot_samples.py

# compare all four strategies on val.csv (default: 200 stratified examples)
python scripts/05_run_task2_fewshot.py

# just one strategy, or a quick smaller check
python scripts/05_run_task2_fewshot.py --strategy basic --n-samples 50

# final comparison once a strategy wins, on test.csv (200 examples)
python scripts/05_run_task2_fewshot.py --source test --n-samples 200 --run-id final --strategy curated
```

Config: `configs/task2.yaml` — same model as Task 1, but `max_new_tokens` is
bumped to 80 (label + confidence + evidence needs more room than a bare
label) and `max_input_tokens` to 2048 (the few-shot demos are full reviews
stacked on top of a target review that can itself run to ~900 words).

### What you get

```
outputs/task2/<run_id>/
├── run_config.json
├── env_record.json
├── curated/
│   ├── prompt.json                 # this strategy's exact system prompt + user template
│   ├── predictions_task2.csv       # label, confidence, evidence per row
│   ├── errors_task2.csv
│   ├── invalid_outputs_task2.csv
│   ├── metrics_task2.json          # final-label quality: accuracy, macro-F1, confusion matrix
│   ├── quality_task2.json          # evidence relevance, calibration, format compliance
│   ├── results_task2.md            # both of the above, one readable report
│   ├── confusion_matrix_task2.png
│   └── run_meta.json
├── random/
│   └── (same files, plus fewshot_seed/fewshot_example_uids in prompt.json)
├── basic/
│   └── (same files)
└── basic_random/
    └── (same files, plus fewshot_seed/fewshot_example_uids in prompt.json)
```

The console prints three side-by-side tables at the end of the run: label
quality (accuracy/macro-F1/invalid rate), the quality dimensions
(schema-compliance/evidence-relevance/ECE), and performance (latency/tokens/VRAM).
`scripts/06_summarize_task2_runs.py` folds all of this — including the quality
columns — into one CSV across every run you've done.

**No retry ladder here**, unlike Task 1 — a bad reply is scored as `unparsed`
on the first attempt. This script exists to compare prompt strategies quickly,
not to run a hardened final pipeline; if a strategy's invalid rate turns out
high, the same retry mechanism Task 1 uses can be added the same way.

---

## 12. Task 1 prompt ablations — basic vs. structured zero-shot vs. CARP

Three prompts, one task, so the numbers isolate what prompt engineering is
actually buying. All three share the same model, data, split, retry-aware
`inference.run()` loop, and metrics — only `build_messages` changes:

| Prompt | Command / config | What it asks for |
|---|---|---|
| **basic** | `scripts/03_run_task1.py --strategy basic` / `configs/task1_basic.yaml` | a bare `POSITIVE`/`NEGATIVE` word — no label definitions, no rules, no JSON schema |
| **zero-shot** (the Task 1 baseline, §5) | `scripts/03_run_task1.py` (default `--strategy zero_shot`) / `configs/task1.yaml` | `{"label": ...}`, with label definitions and 4 disambiguation rules |
| **CARP** | `scripts/03_run_task1.py --strategy carp` / `configs/task1_carp.yaml` | `{"clue": ..., "reasoning": ..., "label": ...}` — the same rules as zero-shot, plus a required Clue → Reasoning → Label chain (Sun et al., 2023) |

`--strategy` picks the prompt module *and* its matching config (each has its
own decoding budget — CARP's reasoning chain needs far more `max_new_tokens`
than a bare label does); pass `--config` to override that default. All three
write into the **same** `outputs/task1/` directory, each tagged with its own
`"prompt_method"` in `run_meta.json` (`basic` / `zero_shot` / `carp`), so
`scripts/04_summarize_runs.py` lists all of them in one table via its
`prompt_method` column — no extra flags needed to compare.

```bash
python scripts/03_run_task1.py --strategy basic --n-samples 200 --run-id basic_200
python scripts/03_run_task1.py                  --n-samples 200 --run-id zeroshot_200
python scripts/03_run_task1.py --strategy carp  --n-samples 200 --run-id carp_200

python scripts/04_summarize_runs.py   # basic / zero_shot / carp, side by side
```

`json_parser.parse()` needed **zero changes** to support any of this: the
basic prompt's bare word lands on the parser's most lenient rung
(`bare_label`, built to recover exactly this case) on every row instead of
occasionally; CARP's extra `clue`/`reasoning` keys just show up as
`extra_keys` on an otherwise normal parse. Only the prompt text and the
decoding/input token budgets differ between the three configs.

### CARP in detail — with vs. without reasoning

`scripts/03_run_task1.py --strategy carp` reruns Task 1 with a **CARP** prompt
(Clue And Reasoning Prompting - Sun et al., 2023) instead of the plain
zero-shot one: the model must work through three explicit steps - **C**lue
(surface sentiment words/phrases), **A**nd **R**easoning (step-by-step,
weighing conflicting clues), then **P** the label - instead of answering with
a bare `{"label": ...}` straight away.

### Run it

```bash
python scripts/03_run_task1.py --strategy carp                          # default 200 samples
python scripts/03_run_task1.py --strategy carp --n-samples 50 --run-id carp_50

# then compare with the plain zero-shot baseline, side by side:
python scripts/04_summarize_runs.py
```

CARP runs are written to `outputs/task1/<run_id>/` — **the same directory**
Task 1's plain runs use — tagged `"prompt_method": "carp"` in `run_meta.json`
(plain runs are tagged `"zero_shot"`). `scripts/04_summarize_runs.py` already
reads every subfolder there, so it lists both methods in one table
(`prompt_method` column) without any extra flags. `results_task1.md` also
shows the prompt method on the CARP runs.

### What's different from plain Task 1

- **Prompt**: `prompts/task1_carp.txt` / `src/prompts_carp.py`, reusing the
  identical label definitions and disambiguation rules as
  `prompts/task1_zeroshot.txt` - only the CARP structure and the JSON schema
  (`clue`, `reasoning`, `label` instead of just `label`) are new.
- **Decoding budget**: `max_new_tokens: 448` (vs. 24) - a bare label needs
  ~12 tokens, a clue list plus several sentences of reasoning needs much more
  room; too small a budget truncates the JSON before the model reaches the
  `label` field, which fails to parse the same way a badly-formed reply would.
- **Input budget**: `max_input_tokens: 2048` (vs. 1536) - CARP's system
  prompt is itself ~404 tokens (vs. ~229 for the plain prompt), stacked on
  top of the same reviews, so it needs more headroom for the same p99 review
  length to avoid truncating the input (see §8 below on why this matters).
- **No other code changes.** `inference.run()`/`latency_probe()` take a
  pluggable `build_messages` function (default: the plain zero-shot one);
  `--strategy carp` just passes `prompts_carp.build_messages` instead.
  `json_parser.parse()` needed no changes at all - it already reports any
  keys besides `label` as `extra_keys`, so `clue`/`reasoning` show up there
  for free on every successfully parsed CARP row.

### A note on `max_input_tokens`

Measuring the CARP prompt's real token length surfaced a latent bug in Task
1's own config: `max_input_tokens: 1024` was sized from an EDA word-count
estimate (`words_p95 ≈ 593`) without accounting for the ~229-token system
prompt stacked on top of the review - on the real tokenizer, **4.45% of the
pool's reviews exceeded 1024 tokens once the system prompt was included**,
silently truncating the end of the prompt (including the trailing `JSON:`
cue) instead of failing loudly. This is consistent with the `full_pool` run's
measured `invalid_output_rate` of 4.41%. `configs/task1.yaml` now uses
`max_input_tokens: 1536` (p99.5 ≈ 1429), measured the same way. If you have
an existing `full_pool` (or similarly large) Task 1 run from before this fix,
consider re-running it — the corrected budget should reduce the invalid-output
rate and may shift accuracy/macro-F1 slightly.

### Basic prompt in detail — the floor the other two are measured against

`scripts/03_run_task1.py --strategy basic` (`prompts/task1_basic.txt` /
`src/prompts_basic.py`) is the opposite move from CARP: instead of adding
structure, it strips everything back to three lines -

```
Classify the sentiment of the following movie review as POSITIVE or NEGATIVE.

Output exactly one label: POSITIVE or NEGATIVE.

Movie review:
{text}

Sentiment:
```

- no label definitions, no disambiguation rules, no JSON schema, no
  reasoning step
- the model answers with a bare word, not JSON — that word is recovered by
  `json_parser.parse()`'s existing `bare_label` rung, so every successfully
  parsed row here reads `parse_strategy: bare_label` (expected, not a
  fallback — this prompt never asks for JSON in the first place)
- `decoding.max_new_tokens` is kept at 24, matching the structured zero-shot
  prompt, so the comparison is about the prompt's structure, not a bigger
  token budget

Run it the same way as the other two:

```bash
python scripts/03_run_task1.py --strategy basic --n-samples 200 --run-id basic_200
python scripts/04_summarize_runs.py
```

---

## 13. Task 2 — prompt sensitivity (best & worst strategy)

Task 2 (§11) compares few-shot *strategies* on the **test** split
(`outputs/task2/final_all/`): **`curated`** ("curated concise") won,
**`random`** ("random concise") lost. That comparison changes big things (which
examples, how many rules). This step asks a different question: *how much of a
strategy's score survives changes that shouldn't matter?*

`scripts/07_run_prompt_sensitivity.py` takes **one** strategy — `--strategy`,
default `curated`; `--strategy random` for the loser — and perturbs its prompt
along three axes, one at a time, holding everything else (model, decoding, that
strategy's 4 demo reviews + hand-authored answers, the eval rows) fixed:

| Axis | Variants | What it probes |
|---|---|---|
| **instruction phrasing** | 4 full paraphrases (`p1`–`p4`, `p1` = verbatim) of the instruction block — context + both label definitions + the judgment rules + the JSON-answer instruction — each keeping every definition, example snippet and rule, changing only wording and structure. `curated` and `random` share the identical instruction block, so one paraphrase set covers both. | does the exact wording of the instructions matter, or only their content? |
| **demo order** | the same 4 demos in each of `pos,neg,pos,neg` / `pos,pos,neg,neg` / `neg,neg,pos,pos` / `neg,pos,neg,pos` — the strategy's own file order is the baseline (`curated`: alternating, `random`: pos-first) | is the model anchoring on the last demo's label, or on recency/primacy? |
| **output schema** | the 3 fields reordered (`confidence,evidence,label`); and the 3 fields renamed (`label→sentiment`, `confidence→certainty`, `evidence→justification`) everywhere they appear — format line, demos, and the instruction prose | does field order / naming change label quality or just format compliance? |

The **baseline** combination reproduces the strategy's prompt file byte for
byte — `src/prompt_sensitivity.assert_baseline_matches(strategy)` enforces this
(and that its instruction block equals the shared `PHRASINGS["p1"]`), and the
demos + instruction are read back out of the `.txt` rather than duplicated. It
is scored here too, as the reference every axis is measured against.

> A one-character typo in `prompts/task2_fewshot_random.txt` (a demo starting
> `eview:` instead of `Review:`) was fixed as part of this — the existing
> `outputs/task2/final_all/random` number predates the fix; re-run
> `scripts/05` for `random` if you need it to match exactly (the effect is
> tiny, and `random` is the loser either way).

```bash
# build every variant's prompt and eyeball it — no GPU, no model load
python scripts/07_run_prompt_sensitivity.py --dry-run
python scripts/07_run_prompt_sensitivity.py --dry-run --strategy random

# full run on the test split (1000 stratified rows) — winner, then loser
python scripts/07_run_prompt_sensitivity.py
python scripts/07_run_prompt_sensitivity.py --strategy random

# 200 test rows, all of test.csv, or just one axis
python scripts/07_run_prompt_sensitivity.py --strategy random --n-samples 200
python scripts/07_run_prompt_sensitivity.py --n-samples 3000 --run-id curated_full
python scripts/07_run_prompt_sensitivity.py --only-axis demo_order
```

Config is shared with Task 2 (`configs/task2.yaml`); `--source` defaults to
`test` here (not `val`). Output lands under
`outputs/task2_sensitivity/<run_id>/` (run-id defaults to
`<strategy>_<timestamp>`, so the two strategies don't collide):

```
outputs/task2_sensitivity/<run_id>/          # run_id e.g. curated_20260830_… or random_…
├── run_config.json
├── env_record.json
├── eval_sample.json
├── variants.json                  # every variant's full prompt text
├── baseline/                      # reproduces the strategy's prompt exactly
│   ├── prompt.json / prompt.txt
│   ├── predictions_task2.csv / errors_task2.csv / invalid_outputs_task2.csv
│   ├── metrics_task2.json / quality_task2.json / results_task2.md
│   ├── confusion_matrix_task2.png
│   └── run_meta.json
├── phrasing__p2/  phrasing__p3/  phrasing__p4/
├── demo_order__…/  (the 3 orders that aren't this strategy's baseline)
├── schema__reordered/  schema__renamed/
├── sensitivity_summary.csv        # one row per variant, with deltas vs. baseline
└── sensitivity_report.md          # per-axis spread (range / std) + a verdict
```

`sensitivity_report.md` is the file to read: for each axis it tabulates every
variant's accuracy / macro-F1 / invalid rate / schema-compliance / evidence
relevance / ECE, then the **range** and population **std** of those numbers
across `{baseline + that axis}`. A small range means the strategy's score is
robust; a large range means it leans on a wording choice a reasonable person
could have made differently. Running both strategies lets you ask whether the
*loser* is just more prompt-fragile than the winner.

> The `renamed` schema variant is scored *after* mapping
> `sentiment`/`certainty`/`justification` back to `label`/`confidence`/`evidence`
> (`prompt_sensitivity.parse_variant`), so its label/calibration numbers stay
> comparable; its `schema_compliance_rate` then means "structurally a clean
> 3-field object", not "used the baseline field names".

Offline tests cover the harness for **both** strategies (baseline rebuilds the
`.txt` byte-for-byte + shares `p1`, one-axis-at-a-time variant enumeration, the
demo-order permutations, the renamed-schema parse round-trip, and a content
check that each hand-written paraphrase keeps every example snippet and rule)
in `tests/test_offline.py`.

---

## 14. Task 2 — self-consistency (majority voting)

`scripts/08_run_self_consistency.py` runs Task 2's best and worst test-set
strategies — **`curated`** ("curated concise") and **`random`** ("random
concise") — and optionally replaces the single greedy prediction per review
with **N stochastic predictions + a majority vote** (Wang et al. 2022).

```bash
# greedy baseline (unchanged), both strategies, 200 test rows
python scripts/08_run_self_consistency.py --n-samples 200

# the requested experiment: 3 samples per review, majority vote
python scripts/08_run_self_consistency.py --n-samples 200 --self-consistency 3

# one strategy, or more votes
python scripts/08_run_self_consistency.py --strategy curated --self-consistency 5
```

| `--self-consistency` | Decoding | Per review |
|---|---|---|
| `1` (default) | greedy, `temperature=0` | one reply — **byte-identical to `scripts/05`'s inference**, the greedy path is not touched |
| `N > 1` | `temperature=0.7`, `top_p=0.95`, seeded from `seed: 42` | N replies in **one** `generate()` call (`num_return_sequences=N` — transformers' equivalent of vLLM `SamplingParams(n=N)`), then majority vote |

**Vote rules** (`src/self_consistency.py`): a row is valid if ≥ 1 candidate
parses; `label` = majority over the valid candidates; ties go to the
highest-confidence candidate; `confidence` = mean confidence of the candidates
that voted for the winner; `evidence` = the first winner-agreeing candidate's.

The output dataframe keeps every column `scripts/05` writes (so
`eval_task2.evaluate()` is unaffected) and adds exactly two: **`vote_margin`**
(winning votes ÷ valid votes) and **`n_valid_votes`**. `run_meta.json` records
`self_consistency_n` and `temperature`.

> ⚠️ **`invalid_output_rate` is not apple-to-apple across modes.** With `N > 1`
> a row is valid if *any* of the N candidates parses — a weaker bar than
> greedy's "the one reply parses" — so the self-consistency invalid rate can
> only be ≤ the single-sample baseline's. Compare **accuracy / macro-F1**
> across modes; read the invalid rate only within self-consistency runs. The
> scripts, `run_meta.json`, and `results_task2.md` all repeat this caveat.

Output lands under `outputs/task2_selfconsistency/<run_id>/{curated,random}/`
(same files as `scripts/05`), plus `self_consistency_summary.csv` at the run
root. Config and the frozen split are shared with Task 2 (`configs/task2.yaml`);
`--source` defaults to `test`.

Offline tests cover the vote aggregation (clean majority, invalid-candidate
exclusion, tie-break, all-invalid row, column compatibility with `parse_fewshot`)
in `tests/test_offline.py`.

---

## 15. Task 3 — QLoRA fine-tuning (Llama-3.1-8B-Instruct)

Fine-tunes a **LoRA adapter** on top of a **4-bit NF4** base model, trains
**only on `data/processed/train.csv`** (the frozen 80% split), and selects
the checkpoint by **validation macro-F1**. Same JSON answer schema, same
`json_parser` / `metrics.py` as Tasks 1–2, so the fine-tuned number is
directly comparable to the zero-/few-shot baselines.

### Extra credential step

`meta-llama/Llama-3.1-8B-Instruct` is a **separately gated** repo. The token
in `.env` is the same one Tasks 1–2 use, but you must also accept the **Llama
3.1** licence at
<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> (accepting 3.2 does
not cover 3.1). Weights (~16 GB) download on first use and are quantized to
4-bit at load time (~5.5 GB VRAM).

### How each task-sheet requirement is met

| # | Requirement | Where |
|---|---|---|
| 1 | 4-bit NF4 + double quant, bf16 compute if supported else fp16 | `src/qlora.load_base_model` (`resolve_compute_dtype` picks bf16/fp16) |
| 2 | `prepare_model_for_kbit_training`, gradient checkpointing, LoRA targets | `src/qlora.prepare_for_training` (non-reentrant GC) + `build_lora_config` |
| 3 | Report r / alpha / dropout / target modules / trainable % | printed each seed + `seed_<seed>/lora_summary.json` |
| 4 | Instruction format, train on train split only, select by val macro-F1 | `src/instruction_format` + `MacroF1Trainer` (`load_best_model_at_end`, `metric_for_best_model="eval_macro_f1"`) |
| 5 | Log train/val loss, LR, time, peak GPU memory; ≥2 seeds + diagrams | `training_log.csv/json`, `run_meta.json`, `summary.*`, and the `.png` diagrams |

Default LoRA config (`configs/task3.yaml`): **r=16, alpha=32, dropout=0.05**,
targets `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
(≈ 0.5 % of parameters trainable for the 8B model).

### Instruction format

Each `(review, label)` pair is rendered with the model's own chat template —
a system message with the label definitions
(`prompts/task3_instruction.txt`), a user message with the review, and an
assistant turn that is exactly `{"label": "positive"}`. The loss is computed
on the assistant tokens only (the prompt is masked to `-100`). Long reviews
are truncated to `instruction.max_review_tokens` *before* templating so the
answer is never cut.

### Run it

```bash
# prerequisite: the frozen split must exist
python scripts/01_prepare_data.py

# default: 2 seeds (42, 1337), 3000 train samples, 1 epoch  (~1 GPU-hour/seed)
python scripts/12_run_task3_qlora.py

# quick smoke (few minutes)
python scripts/12_run_task3_qlora.py --seeds 42 --max-train-samples 200 --max-steps 20

# the full-data run
python scripts/12_run_task3_qlora.py --max-train-samples all --epochs 2 --run-id full

# more seeds / a single seed
python scripts/12_run_task3_qlora.py --seeds 42 1337 2024

# compare every Task 3 run in one table
python scripts/13_summarize_task3_runs.py
```

### What you get

```
outputs/task3/<run_id>/
├── run_config.json / env_record.json
├── summary.json / summary.md            # mean ± std across seeds
├── curves_across_seeds.png              # train/val loss + val macro-F1 over steps
├── macro_f1_by_seed.png                 # val/test macro-F1 per seed, mean ± std
├── resources_by_seed.png                # training time + peak GPU memory per seed
└── seed_<seed>/
    ├── lora_summary.json                # r / alpha / dropout / targets / trainable %
    ├── training_log.csv / training_log.json
    ├── training_curves_seed<seed>.png   # loss / macro-F1 / learning-rate
    ├── metrics_{val,test}_task3.json
    ├── predictions_{val,test}_task3.csv
    ├── confusion_matrix_{val,test}_task3.png
    ├── adapter/                          # the selected LoRA adapter (load with PeftModel)
    ├── run_meta.json / results_task3.md
    └── checkpoints/                      # best + last (Trainer, save_total_limit=2)
```

Checkpoint selection is driven by **generation** macro-F1, not token loss:
`MacroF1Trainer.evaluate()` decodes the JSON answer on a stratified
validation subset (`eval.select_max_samples`), parses it with the shared
`json_parser`, scores it with `metrics.compute`, and logs `eval_macro_f1`;
`load_best_model_at_end` then restores that adapter before the final full
val/test scoring. `eval_loss` is still logged as a monitoring curve.

### Tuning for your GPU

| Setting | Where | Effect |
|---|---|---|
| `train.per_device_train_batch_size` / `gradient_accumulation_steps` | config | effective batch = product; lower the accum if training is slow, raise it for a smoother curve |
| `instruction.max_seq_len` / `max_review_tokens` | config | truncation length; the dominant VRAM knob during training |
| `eval.batch_size` | config / | generation-eval batch; lower first if the eval step OOMs |
| `train.optim` | config | `paged_adamw_8bit` (default) pages optimizer state to host RAM on a spike |

If you hit `CUDA out of memory`: lower `max_seq_len`, then `eval.batch_size`,
then set `gradient_accumulation_steps` higher (batch size 1 is already the
minimum).

---

See `REPORT.md` for the architecture and the reasoning behind the design choices.
