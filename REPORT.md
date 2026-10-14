# Architecture Report — Task 1 (Zero-Shot) & Task 2 (Few-Shot) Sentiment Analysis

**Model:** `meta-llama/Llama-3.2-3B-Instruct`
**Dataset:** Kaggle — Twitter Entity Sentiment Analysis
**Hardware target:** single NVIDIA RTX 5080, 16 GB, local
**Scope:** zero-shot (Task 1) and few-shot (Task 2) classification, no
training or fine-tuning of any kind

---

## 1. Design goals

The task sheet asks for a *reproducible no-training baseline*. Four goals follow
from that word, and they drove every structural decision:

1. **Reproducible** — the same command on the same machine yields the same
   numbers. Greedy decoding, seeds everywhere, and the config/prompt/environment
   frozen into each run directory.
2. **Auditable** — every number can be traced back to the raw model output that
   produced it. Nothing is thrown away.
3. **Robust** — an LLM occasionally ignores the output format. That must degrade
   into a measured failure rate, not a crashed run.
4. **Extensible** — Tasks 2–4 (few-shot, QLoRA, distillation) reuse the same
   data, split, parser and metrics without modification.

Goal 4 is why the split module exists at all: a zero-shot baseline needs no
split, but the project it belongs to does, and creating it once up front is what
makes the later comparisons valid. Task 2 (§8) is this goal put into practice:
`model.py`, `metrics.py`, `dataset.py` and the frozen split carried over with
zero changes; only the prompt and the parser's schema needed a second version.

---

## 2. Repository layout

```
llm-sentiment/
├── configs/task1.yaml        single source of truth for Task 1's parameters
├── configs/task1_carp.yaml   Task 1 with the CARP prompt: mirrors task1.yaml, bigger token budgets
├── configs/task1_basic.yaml  Task 1 with the basic prompt: mirrors task1.yaml, same token budgets
├── configs/task2.yaml        same model + split, Task 2's decoding/eval settings
├── prompts/                  prompt text as plain files, not Python string literals
│   ├── task1_zeroshot.txt    Task 1's system message + user template
│   ├── task1_carp.txt        Task 1's CARP variant: clue -> reasoning -> label
│   ├── task1_basic.txt       Task 1's basic variant: no rules, no schema, one word
│   ├── task2_fewshot.txt     Task 2, curated-examples strategy
│   ├── task2_fewshot_random.txt   Task 2, random-examples strategy
│   └── task2_fewshot_basic.txt    Task 2, same curated examples, rules stripped out
├── .env.example              committed template; .env holds the real tokens
├── src/                      library code — importable, no side effects
│   ├── utils.py              config, seeding, logging, JSON/JSONL I/O, env/token
│   ├── env_record.py         environment capture
│   ├── dataset.py            cleaning + the frozen split
│   ├── eda.py                statistics and figures
│   ├── model.py              loading + deterministic batched generation
│   ├── prompts.py            loads prompts/task1_zeroshot.txt, retry escalations
│   ├── prompts_carp.py       loads prompts/task1_carp.txt, same build_messages contract as prompts.py
│   ├── prompts_basic.py      loads prompts/task1_basic.txt, same build_messages contract as prompts.py
│   ├── prompts2.py           loads Task 2's prompt files (STRATEGIES dict)
│   ├── prompt_sensitivity.py builds a Task 2 strategy's prompt-perturbation variants (curated/random, §8.8)
│   ├── self_consistency.py   majority-vote aggregation for Task 2 N-sample inference (§8.9)
│   ├── json_parser.py        parse → validate ladder (Task 1) + parse_fewshot (Task 2)
│   ├── inference.py          the retry-aware scoring loop (Task 1 and its prompt ablations)
│   ├── eval_task2.py         Task 2 quality metrics: evidence relevance, calibration, format compliance
│   ├── gpu_monitor.py        VRAM measurement
│   ├── plot_style.py         shared minimalist theme for every figure
│   └── metrics.py            scores, figures, reports (shared by both tasks)
├── scripts/                  thin CLI entry points — orchestration only
│   ├── 00_record_env.py
│   ├── 01_prepare_data.py    download + clean + freeze the split (no EDA)
│   ├── 02_run_eda.py         EDA on the cleaned pool — separate, re-runnable step
│   ├── 03_run_task1.py       zero-shot baseline + CARP + basic-prompt ablations, via --strategy
│   ├── 04_summarize_runs.py
│   ├── 05_run_task2_fewshot.py
│   ├── 06_summarize_task2_runs.py
│   ├── 07_run_prompt_sensitivity.py  perturb the winning Task 2 prompt, score on test (§8.8)
│   └── 08_run_self_consistency.py    curated + random, greedy or N-sample majority vote (§8.9)
├── tests/                    offline checks, no GPU required
├── data/raw | data/processed
└── outputs/
```

The separation is deliberate: `src/` contains logic and is import-safe;
`scripts/` contains sequencing and argument parsing and nothing else. A reviewer
can read any one module without the rest, and Task 2 can `import src.prompts`
without dragging in a CLI.

---

## 3. Pipeline

```
        configs/task1.yaml
               │
      ┌────────┴─────────┐
      │                  │
  00_record_env      01_prepare_data ──► data/processed/
      │                  │                 ├─ clean_pool.csv       (Task 1)
 env_record.json         │                 ├─ train/val/test       (Tasks 2-4)
      │                  │                 └─ holdout.csv          (cross-check)
      │                  ▼
      │            02_run_eda ──► outputs/eda/*
      │                  │
      └──────────────► 03_run_task1 ◄──── data/processed/clean_pool.csv
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
    model.load        inference.run         metrics.compute
   (bf16 / nf4)     prompt → generate      accuracy, macro-F1,
                     → parse → retry       per-class, confusion matrix
                            │
                    outputs/task1/<run_id>/
```

### 3.1 Data layer (`src/dataset.py`)

`load_clean()` turns the raw CSV into a canonical frame with columns
`uid, entity, text, label`:

| Step | Reason |
|---|---|
| attach a header | the Kaggle export ships without one |
| lowercase labels | `Positive` / `positive` must not be different classes |
| remap `irrelevant` → `neutral` | these rows mention the entity without taking a stance toward it, which is what `neutral` already means; folding them in keeps the task 3-class while using 100% of the labelled data |
| drop empty / < 3 chars | nothing to classify |
| dedupe on `(entity, text)` | the dataset repeats tweets; duplicates inflate whichever class they belong to |
| `uid` = SHA1 of `entity‖text` | a stable row identity, so checkpoints and resumes survive a reordering |

The remapping is applied **before** the `keep_labels` filter, and it is not
free: `neutral` goes from the smallest of the three raw classes to the largest,
roughly 42% of the pool against ~30% negative and ~28% positive. Two consequences
follow, and both are handled rather than ignored:

- **Metric choice.** A model that over-predicts `neutral` now scores well on
  accuracy for the wrong reason. Macro-F1 weights all three classes equally and
  does not reward that, which is why it is the headline metric (§3.8).
- **Prompt wording.** `neutral` now spans two different phenomena — "no clear
  stance" and "not about this entity at all". The prompt has to say so
  explicitly, or the model will read a stance into off-topic posts and lose
  precisely the rows the remapping added (§3.4).

The remapping is a config entry (`data.label_map`), not hard-coded, so the
drop-instead variant is a one-line change and the count of remapped rows is
logged on every run.

One subtlety worth naming: under newer pandas string dtypes a bare
`.astype(str)` leaves `NaN` as a float rather than the string `"nan"`, so a
single missing tweet leaks a float into a text column and crashes downstream
string operations. The code goes through `.astype(object).fillna("")` first.

**The two Kaggle files.** The download contains `twitter_training.csv` (~74k)
and `twitter_validation.csv` (~1k). The second file's name is misleading: it is
not the validation split, just a smaller second file. The roles assigned here:

| Partition | Source | Purpose |
|---|---|---|
| `train` / `val` / `test` | training.csv, split 80/10/10 | Tasks 2–4: learn, select, report |
| `all` (pool) | all of training.csv | Task 1 samples from here |
| `holdout` | validation.csv | independent cross-check |

The val split is carved out of the pool rather than taken from the Kaggle
"validation" file, because model selection needs a slice drawn from the same
distribution as training — that is what makes an early-stopping decision
meaningful. The second file is more valuable as something else entirely: a set
that was never split, never trained on, and never consulted for any decision.

The reasoning is that `test` shares a file, and therefore a distribution, with
`train`. A strong `test` score is consistent with two very different worlds —
the model learned sentiment, or it learned that file's idiosyncrasies. One
number cannot distinguish them; two can. Agreement between `test` and `holdout`
is evidence of generalisation, and a gap localises the problem.

`load_holdout()` applies the identical cleaning function (`_read_and_clean`), so
the two sets are never accidentally compared under different preprocessing, and
then removes any row whose `uid` also appears in the pool. The files are
supposed to be disjoint, but a shared row would sit in `train` *and* in the
"untouched" holdout — precisely the leak the holdout exists to rule out. The
removed count is logged rather than silently applied.

The holdout is refreshed on every invocation, including the reuse path: it is
never partitioned, so it cannot invalidate a split. The practical payoff is that
`twitter_validation.csv` can be added to a project whose split was frozen weeks
earlier, and `01_prepare_data.py` will write `holdout.csv` without touching
train/val/test. A missing second file degrades to a warning — the holdout is a
bonus, never a requirement.

`build_split()` performs a two-stage stratified split (80/10/10) and writes a
manifest with sizes, ratios, and label counts per partition. On any later
invocation:

- split files already on disk → reuse, do nothing
- nothing on disk → create it
- `--force` → overwrite, explicitly requested

This is deliberately just an `if not exists` check, not a fingerprint over the
data and config: "split only once" only needs to survive a plain re-run of
`01_prepare_data.py`. If you change the cleaning rules, the ratios, or the seed
and want a new split, pass `--force` yourself — the script does not try to
detect that for you.

**Task 1 does not consume the split.** A zero-shot baseline has no training
phase and therefore no leakage risk, so restricting it to a 10% test set would
only add variance. It samples from the full cleaned pool via `sample_for_eval()`,
stratified so the label proportions match the pool, with a warning if `n < 200`.

### 3.2 EDA (`src/eda.py`)

Purely descriptive, and it feeds two concrete decisions rather than just
producing pictures:

- **imbalance ratio** (max class ÷ min class) — justifies reporting macro-F1
  alongside accuracy
- **word-count p95/p99** — sets `model.max_input_tokens`, so truncation never
  silently cuts a tweet in half

Figures: label distribution, length distributions, top entities, and top content
words per label (the last one is a cheap check that the labels carry real
signal). Everything is also written to `eda_stats.json`.

### 3.2b Credentials (`src/utils.py`)

The model is gated, so the pipeline needs an access token. This is kept
deliberately minimal rather than built out as its own module: `load_env()`
reads `KEY=value` lines from `.env` into `os.environ` (skipping comments and
blank lines; an existing environment variable always wins over the file), and
`get_hf_token()` calls it and returns `os.environ.get("HF_TOKEN")`. `model.load()`
uses it whenever no `--hf-token` argument is given.

A token value is never printed, logged, or written to any output file.
`env_record.json` records only whether one is set (`hf_token_set: bool`), never
the value or its source — safe to hand in with the assignment. `.env` is
gitignored, `.env.example` is the committed template, and `setup.sh` chmods the
real file to `600`. The `--hf-token` CLI flag exists but its help text warns
that a token on the command line lands in shell history.

There is no separate access-preflight step. An earlier version called
`model_info()` before loading anything, to turn a gated-repo problem into a
two-second failure instead of a failure after a multi-GB download — that logic
was cut in favour of the simplest thing that works: if the token is missing or
wrong, `AutoModelForCausalLM.from_pretrained()` raises its own clear error
(`GatedRepoError`, a 401, etc.) a few seconds into the run anyway. One fewer
module to maintain, at the cost of that error surfacing slightly later.

The whole mechanism is about fifteen lines with **no third-party dependency**
— no `python-dotenv` for two functions.

### 3.3 Model layer (`src/model.py`)

Precision is a config switch, all four paths sharing one code path:

| Precision | VRAM (3B) | Use |
|---|---|---|
| `bf16` | ~6.5 GB | default; ample headroom on 16 GB |
| `fp16` | ~6.5 GB | hardware without bf16 |
| `nf4` | ~2.5 GB | 4-bit; identical setup to the QLoRA task, so it doubles as an ablation |
| `int8` | ~3.5 GB | rarely worth it — slower than nf4 in practice |

Two details matter more than they look:

- **Left padding.** Llama has no pad token, so `pad_token = eos_token`, and
  `padding_side = "left"`. With right padding on a decoder-only model, pads sit
  between the prompt and the first generated token and corrupt the output. Left
  padding also means every row in a batch starts generating at the same index,
  so one slice offset strips the prompt for the whole batch.
- **`do_sample=False` means greedy.** `temperature` and `top_p` are then
  inactive, and passing them only emits warnings — so they are omitted rather
  than passed and ignored. Sampling is switched on *only* on retry attempts,
  which keeps the headline comparison fully deterministic.

`transformers` is imported inside `load()` rather than at module scope: it costs
seconds to import, and keeping it lazy lets the offline tests exercise the retry
loop without the full HF stack.

### 3.4 Prompt (`src/prompts.py`, text in `prompts/task1_zeroshot.txt`)

A system message and a user message, per the spec. The wording itself lives in
`prompts/task1_zeroshot.txt` (split into a `=== SYSTEM ===` and a
`=== USER TEMPLATE ===` section); `prompts.py` just loads it and substitutes
`{entity}`/`{text}`. Task 2's prompt files follow the identical convention
(§8), so every prompt in the project is reviewable as plain text without
reading Python.

The system message defines all three labels in terms of *what the author
expresses*. `neutral` is spelled out as four distinct cases, because after the
`irrelevant` → `neutral` remapping it genuinely covers four:

| Case | Origin |
|---|---|
| (a) factual, informational, or a question | raw `Neutral` |
| (b) mixed or too ambiguous to lean | raw `Neutral` |
| (c) entity mentioned in passing, opinion aimed elsewhere | raw `Irrelevant` |
| (d) off-topic — the name appears but the post is not about it | raw `Irrelevant` |

Cases (c) and (d) are the ones a model gets wrong if the prompt stays silent
about them: it finds *some* emotion in the text and assigns `positive` or
`negative`, which is exactly the failure mode the remapping would otherwise
introduce. Naming them costs two sentences and recovers the whole added class.

Five disambiguation rules follow, each targeting a failure mode this dataset
actually produces:

1. sentiment is toward the **target entity**, not the post in general — this
   dataset is entity-level, and a post can praise one thing while criticising
   another. The rule also states that strong emotion *not* aimed at the entity
   is `neutral`, which is the (c)/(d) case restated as an instruction
2. attitude, not subject matter — a calm report of bad news is neutral
3. profanity is not a label signal — this corpus is full of enthusiastic swearing
4. non-English posts are still classified
5. always pick one of the three; never refuse, never invent a label

**Zero-shot is preserved.** The JSON skeleton
`{"label": "<positive|negative|neutral>"}` appears in the system prompt, but it
is a *format* specification: no tweet is ever paired with a label anywhere in the
prompt. `prompt_metadata()` records `n_examples_in_prompt: 0` explicitly.

Retries escalate in strictness rather than repeating the same words:

| Attempt | Suffix | Temperature |
|---|---|---|
| 0 | none — the pristine zero-shot prompt | 0.0 (greedy) |
| 1 | "Your previous reply was not valid JSON…" | 0.2 |
| 2 | "FINAL ATTEMPT" + the three literal strings | 0.4 |

The small temperature bump exists because greedy decoding is deterministic: if
attempt 0 produced garbage, re-running the same prompt greedily produces the
*same* garbage. A nudge lets the model escape the degenerate path. Attempt 0 is
never sampled, so the primary result stays reproducible.

### 3.5 JSON parsing (`src/json_parser.py`)

A ladder from strictest to most permissive, recording which rung succeeded:

| # | Strategy | Handles |
|---|---|---|
| 1 | `strict` | the whole reply is a JSON object |
| 2 | `fenced` | ` ```json { … } ``` ` |
| 3 | `substring` | `Sure! {"label": "positive"}` |
| 4 | `regex_kv` | malformed JSON, readable key/value |
| 5 | `bare_label` | no JSON, exactly one label word present |

Only rung 5 is genuinely lenient, and it is counted separately so a reviewer can
discount it. Two guards keep it honest: a reply containing *more than one* label
word is rejected as `ambiguous_multiple_labels` rather than guessed, and a small
alias table normalises near-misses (`pos`, `positve`, `Neutral`).

`validate(strict_schema=True)` is available as an ablation: it accepts only
clean, single-key JSON, which measures raw format-following ability rather than
end-to-end classification accuracy.

Which strategy resolved each reply is kept per row in `predictions_task1.csv`
(`parse_strategy` column) for manual inspection, even though the aggregate
report only surfaces the single `invalid_output_rate` number (§3.8).

### 3.6 Inference loop (`src/inference.py`)

Per batch: build prompts → generate → parse each reply → collect the failures →
**retry only those**, at the next escalation level. Rows that still fail after
`max_attempts` receive the configured fallback (`unparsed` by default, so they
are visible in the metrics rather than hidden inside `neutral`).

Retrying only the failures matters at scale: with a ~2% failure rate, re-running
whole batches would cost ~50× more than re-running the individual rows.

Every batch is appended to a `predictions.jsonl` checkpoint (with an `fsync`)
under the run directory, so an interrupted run resumes by `uid` instead of
starting over - that is what makes an overnight full-pool run practical. The
checkpoint is transient: once a run finishes successfully its rows are already
in `predictions_task1.csv`, so `scripts/03_run_task1.py` deletes it rather than
leave a duplicate on disk.

### 3.7 Measurement (`src/gpu_monitor.py`, `src/inference.py`)

**GPU memory** is reported two ways, because they answer different questions:

- `peak_allocated` / `peak_reserved` from the torch allocator — what *this
  process* used
- `nvidia_smi_used` — what the *device* shows, including the CUDA context

The second is the number that decides whether a workload fits in 16 GB; the
first is the number to compare between precisions.

**Latency** is likewise reported two ways, because a single number would be
misleading:

- *batched* — wall time ÷ examples, i.e. throughput, the number that matters for
  scoring a dataset
- *single-request* (`latency_probe`) — true batch-size-1 end-to-end time with
  mean/median/p90/p95, the number that matters for serving

The probe discards the first few iterations: CUDA kernel autotuning inflates
them, and including them would overstate the median by a wide margin. All timings
are taken around explicit `torch.cuda.synchronize()` calls — CUDA is
asynchronous, and timing without them measures queue-submission speed, not work.

### 3.8 Metrics (`src/metrics.py`)

Deliberately narrow scope: accuracy, macro-F1, per-class precision/recall/F1,
and a confusion matrix for classification quality; invalid-output rate for
pipeline reliability; latency, token usage and peak GPU memory (via
`run_meta`/`run_stats.json`) for performance. Unparsed rows count as wrong in
accuracy/macro-F1 — the honest end-to-end number — and additionally get their
own `unparsed` column in the confusion matrix so they stay visible rather than
folding invisibly into whichever class is most common.

The headline metric is **macro-F1**, not accuracy: the classes are imbalanced,
and `neutral` is the class an LLM most often collapses. Accuracy hides that;
macro-F1 does not.

The confusion matrix is rectangular rather than square: gold is always one of
the three real labels by construction (the dataset only has three, and the
model's answer is never used as ground truth), so only the predicted side can
be `unparsed`. A square matrix would carry a permanently all-zero "unparsed"
gold row, which is not a real category — just clutter.

Artefacts: `metrics_task1.json`, a readable `results_task1.md`, a two-panel
confusion matrix (counts + row-normalised recall), `errors_task1.csv` — the
misclassified rows alone, and `invalid_outputs_task1.csv` — just the rows where
`predicted` is `unparsed` (a strict subset of the errors file, isolated because
those rows are a pipeline failure, not a classification one: the model never
returned an answer to be right or wrong about). `predictions_task1.csv` retains
every per-row detail (parse strategy, attempts used, raw model output) that
does not make it into the aggregate report, for anyone who wants to dig into
the pipeline mechanics rather than just the classification numbers.

---

## 4. Reproducibility contract

| Element | Where it is recorded |
|---|---|
| model checkpoint + revision | `run_config.json`, `run_meta.json` |
| exact prompt text (system + user template + retry suffixes) | `prompt.json` |
| decoding parameters | `run_config.json`, `run_meta.json` |
| seed | `run_config.json` |
| Python / PyTorch / CUDA / Transformers / PEFT / bitsandbytes | `env_record.json` |
| whether an HF token was set (never the token) | `env_record.json` |
| GPU model, VRAM, driver, OS | `env_record.json` |
| exact rows scored + raw model output per row | `predictions_task1.csv` (`raw_output` column) |
| data split sizes + label counts | `data/processed/split_manifest.json` |
| holdout size + overlap removed | `data/processed/split_manifest.json` (`holdout` block) |

`set_seed()` seeds `random`, `numpy` and `torch`, enables deterministic cuDNN,
and sets `CUBLAS_WORKSPACE_CONFIG` before CUDA initialises (it has no effect
afterwards). Combined with greedy decoding on attempt 0, a re-run on identical
hardware and software reproduces the predictions.

Honest caveat: bit-exact reproducibility does **not** hold across different GPU
architectures, PyTorch versions, or `batch_size` values — batched matmul
reduction order changes the floating-point result, which can flip a near-tie
argmax. This is why `batch_size` is recorded with the results.

---

## 5. Testing

| File | Covers | Needs GPU |
|---|---|---|
| `tests/test_offline.py` | Task 1's parser ladder (12 positive + 3 rejection cases) and prompt construction; Task 2's `parse_fewshot` (strategy/extra-key tracking, confidence clamping, invalid label/JSON rejection), both `prompts2` strategies, and `eval_task2`'s evidence-relevance/calibration/format-compliance metrics against a hand-computed fixture; the `prompt_sensitivity` harness for both `curated` and `random` (baseline rebuilds the strategy's prompt byte-for-byte + shares `p1`, one-axis-at-a-time variant enumeration, demo-order permutations, renamed-schema parse round-trip, and each hand-written paraphrase keeping every example snippet + rule — §8.8); `self_consistency.aggregate` (clean majority, invalid-candidate exclusion from the vote, tie-break by confidence, all-invalid row, column compatibility with `parse_fewshot` + the 2 new vote columns — §8.9); classification-metric arithmetic; dataset cleaning/dedup/holdout-overlap against a mocked HF loader; `.env` parsing + HF token lookup | no |
| `tests/test_pipeline_mock.py` | Task 1's retry loop, JSONL checkpoint + resume, metric/figure writers, driven by a fake model that deliberately emits fences, refusals and prose | no |

The mock model's output distribution is tuned so the retry and fallback branches
are actually taken — a mock that always returns clean JSON would leave the most
failure-prone code path untested. Both suites run in seconds and are worth
running before any long GPU job.

---

## 6. Design decisions, and what was given up

| Decision | Why | Cost |
|---|---|---|
| Remap `irrelevant` → `neutral` | semantically the same thing (no stance toward the entity); keeps the task 3-class and uses 100% of the labelled data | `neutral` becomes the majority class (~42%), so accuracy alone would be misleading — macro-F1 is reported instead, and the prompt had to name the off-topic case |
| Evaluate on the full pool, not the test split | zero-shot has no training, so no leakage; more rows = lower variance | Task 1 numbers are not directly comparable to a fine-tuned model scored on `test` — run `--source test` for that comparison |
| `twitter_validation.csv` as an independent holdout, not as the val split | model selection needs a slice from the training distribution; a second file is worth more as a generalisation check | one extra evaluation run to get the second number |
| Holdout refreshed on every run, not frozen with the split | it is never partitioned, so it can be added later without invalidating an existing split | a small special case in `build_split` |
| bf16 by default, not 4-bit | 3B in bf16 fits 16 GB with room to spare; no quantization error in the baseline | ~4 GB more VRAM than nf4 |
| Retry with a temperature bump | greedy retries reproduce the same failure | attempts ≥ 1 are not deterministic — recorded per row |
| Lenient `bare_label` parsing | recovers a real slice of otherwise-lost predictions | arguably generous; the per-row strategy is kept in `predictions_task1.csv` so it can be filtered out |
| Fallback label `unparsed`, not `neutral` | keeps failures visible in the metrics | adds a column to the confusion matrix |
| Split reused by a plain exists-check, not a data/config fingerprint | "split only once" only needs to survive a re-run; simpler to reason about | changing the cleaning rules or seed does not auto-invalidate an old split — `--force` is on you |
| JSONL checkpoint per batch | makes long runs interruptible | one `fsync` per batch (negligible next to generation) |
| `.env` read by two functions in `utils.py`, no dedicated module | one token mechanism for laptop and CI; no extra package | no access preflight, no masking helper — a bad token surfaces as `from_pretrained`'s own error instead of a custom message |
| Metrics report scoped to accuracy/macro-F1/per-class/confusion matrix + invalid-output rate/latency/tokens/GPU memory | matches exactly what needs reporting; a shorter report is easier to read and to grade | weighted-F1, MCC, Cohen's κ, and the parse-strategy/retry breakdown are no longer computed — the per-row detail (attempts, strategy) still lives in `predictions_task1.csv` if needed |

---

## 7. Measured results

From a 500-example stratified run (`bf16`, greedy, seed 42 — see
`outputs/task1/LATEST_results.md` for the current numbers):

| Metric | Value |
|---|---|
| Accuracy | 0.51 |
| Macro F1 | 0.484 |
| Invalid-output rate | 0.00% |

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.535 | 0.616 | 0.572 | 138 |
| negative | 0.482 | 0.895 | 0.627 | 152 |
| neutral | 0.576 | **0.162** | 0.253 | 210 |

Accuracy and macro-F1 diverge for exactly the reason §3.8 predicts: the gap is
concentrated in `neutral`. The model over-predicts `negative` (282 of 500
predictions, against 152 actually `negative`) and under-predicts `neutral` (59
predictions against 210 gold). Sampling the `neutral`→`negative` errors shows a
consistent pattern — the model reacts to profanity or general negativity in a
tweet even when it is not directed at the target entity (rule 1, §3.4), and
even though the prompt explicitly says profanity alone should not decide the
label (rule 3). Both are exactly the `irrelevant`→`neutral` failure modes named
in §3.1 and §3.4.

The row-normalised confusion matrix shows this at a glance: the `neutral` row
has a short diagonal cell with mass spread mostly toward `negative`. A useful
diagnostic when the numbers disappoint: `errors_task1.csv` retains the `entity`
column, so filtering it for rows whose text barely mentions the entity isolates
the remapped `Irrelevant` population and shows whether the loss is concentrated
there or spread across all of `neutral`.

An invalid-output rate at or near 0% is expected from a well-behaved instruct
model with a constrained-output prompt; substantially higher suggests a prompt
or chat-template problem rather than a model limitation.

---

## 8. Task 2: Few-shot prompt comparison

Task 1 established that the pipeline (data, model, parser, metrics) is
reusable without modification; Task 2 is the proof. It reuses Task 1's model
loading, data layer, and classification metrics wholesale, and adds a second
prompt-loading module, a richer 3-key parser, a second runner script, and a
new module for the quality dimensions that a bare label can't capture.

### 8.1 What's new vs. what's reused

| Reused unchanged | New |
|---|---|
| `model.py`, `metrics.py`, `dataset.py`, `env_record.py`, `gpu_monitor.py` | `prompts2.py` (loads Task 2's prompt files) |
| `json_parser`'s `_FENCE_RE`/`_OBJ_RE`/`normalize_label` | `json_parser.parse_fewshot()` (3-key schema, now with strategy/extra-key tracking) |
| the frozen `train`/`val`/`test` split from Task 1 | `configs/task2.yaml`, `scripts/05_run_task2_fewshot.py` |
| the label definitions and disambiguation rules from Task 1's system prompt | `prompts/task2_fewshot.txt`, `prompts/task2_fewshot_random.txt`, `src/eval_task2.py` |

### 8.2 Example selection

`data/processed/selected_sample.csv` — 12 rows, 6 per class — is a **hand-curated**
candidate pool, not something this codebase generates; it is an external input
the pipeline documents, not the output of any script here. `task2_fewshot.txt`
embeds 4 of these 12 rows (2 per class), chosen for being short (25-55 words)
and unambiguous; the CSV's `used_in_curated_prompt` column only flags 2 of
those 4 (1 per class) — a stale/incomplete marker from an earlier version of
the prompt with fewer demos, not a reliable list of which rows are actually
embedded today. Every candidate comes from `train.csv` (not `val`/`test`), so
no few-shot demonstration ever leaks eval data into the prompt.

### 8.3 Four prompting strategies (`prompts/task2_fewshot*.txt`)

All four files require the same 3-key JSON reply — `label`, `confidence`
(0.0-1.0), `evidence` (one sentence, stated directly, no chain-of-thought) —
and are registered in `src/prompts2.STRATEGIES`, a plain `{name: filename}`
dict. Adding a strategy is a two-line change (one dict entry, one new .txt
file): `scripts/05_run_task2_fewshot.py` iterates `STRATEGIES` whenever
`--strategy` is omitted, so a new entry runs alongside the others with no
other code change, exactly the same mechanism Task 1's three prompts (§9)
share via `outputs/task1/`'s common directory.

| Strategy | File | Rules + definitions? | Examples |
|---|---|---|---|
| `curated` | `task2_fewshot.txt` | yes (same as Task 1's zero-shot, §3.4) | 4 balanced (2/class), hand-picked from `selected_sample.csv` |
| `random` | `task2_fewshot_random.txt` | yes | 4 unfiltered (2 positive + 2 negative), one fixed precomputed draw - see §8.3.1 |
| `basic` | `task2_fewshot_basic.txt` | **no** - task + JSON schema only | the *same* 4 curated demos as `curated` |
| `basic_random` | `task2_fewshot_basic_random.txt` | **no** - task + JSON schema only | the *exact same* 4 rows as `random` |

Each pair isolates one variable. `curated` vs `random` differ only in where
the examples come from, not in count - both are 4 demos, 2 per class, so any
gap reflects curation quality rather than a difference in how much few-shot
context the model sees. `random` vs `basic_random` isolate the opposite
variable: identical 4 demo rows, with only the system prompt's rules and
label definitions differing - the same relationship `curated` has to
`basic`, but on the unfiltered example set.

All four files bake their example text in as literal, static content -
`src/prompts2.py` loads every strategy's `.txt` file the same uniform way
(split into system/user halves, substitute `{text}`) with no runtime
templating for any of them, `random`/`basic_random` included. That wasn't
always true: an earlier version of `random`/`basic_random` left a
`__RANDOM_FEWSHOT_EXAMPLES__` placeholder in the `.txt` file and filled it in
at prompt-build time from `data/processed/random_fewshot_samples.csv` on
every call. Baking the text in instead removes that indirection - the
prompt actually sent to the model is now visible by opening the `.txt` file,
exactly like `curated`/`basic`, and `build_messages()` no longer needs a
special case for any strategy at all.

The dataset has no gold confidence or evidence text, so every strategy's
demonstration answers are **hand-authored** - including `random`'s/
`basic_random`'s, even though the *reviews* they cite are an unfiltered
draw. Confidence is varied by how genuinely clear each example is (0.65-0.98
across all four files) rather than maxed out everywhere - a prompt that only
ever demonstrates 0.95+ confidence teaches the model to output 0.95+
regardless of how uncertain it actually should be, which would flatten the
calibration metric in §8.4 before the model even gets a chance to be honest.
An earlier version of `random`/`basic_random` used a fixed, generic
templated evidence sentence per row (`"The reviewer's overall tone and word
choice support a <label> verdict."`) instead, reasoning that inventing a
plausible-sounding, human-authored justification for an unfiltered example
would reintroduce curation through the back door. In practice that produced
evidence that would score poorly under §8.4's `mean_relevance` metric (it
shares almost no content words with the review it's attached to) and
modeled exactly the shortcut that metric exists to catch - so the demo
answers were rewritten to cite specific wording from each review instead
(e.g. `"Assigns an explicit *1/2 out of **** and faults the 'hoked-up
melodrama' plus an ill-matched lead pairing."`), the same standard
`curated`/`basic`'s demos are held to. The ablation isolates which *reviews*
get selected (curated vs. unfiltered), not whether the demonstrated answer
is a good one - a generic answer was never necessary to keep that isolation
clean, and a well-grounded one is a strictly better demonstration for the
model to learn the schema from either way.

#### 8.3.1 Where the random examples come from (`scripts/10_select_random_fewshot_samples.py`)

`random` and `basic_random` must use the *exact same* 4 demo reviews for the
pairing to isolate only the system prompt's rules, the same way `curated`
and `basic` do. `scripts/10_select_random_fewshot_samples.py` draws 2
positive + 2 negative rows once (`DataFrame.sample(random_state=42)`,
unfiltered - no length/quality curation, which is the whole point of this
ablation) and writes them to `data/processed/random_fewshot_samples.csv`.
That file is a *record* of the draw, not something read at runtime: the
actual example blocks (review text + hand-authored JSON answer, §8.3) are
baked into both `.txt` files as a separate, deliberate authoring step, the
same relationship `selected_sample.csv` has to `curated`'s/`basic`'s baked-in
demos. Re-running the script with a different `--seed` produces a new draw
but does not, by itself, change either `.txt` file. An earlier version of
this pipeline instead drew three different seeds and ran `random`/
`basic_random` once per seed (to average out a lucky-or-unlucky draw),
writing `<strategy>_seed<seed>/` subfolders - that added a whole extra
dimension (seed) to every run directory and report table for what is
fundamentally a single fixed-example-set comparison, so it was replaced with
the one-seed, one-file, 4-row design described here. The seed is still
recorded as metadata - `prompts2.random_fewshot_seed()` reads it back from
the CSV itself (rather than duplicating it as a second constant) purely for
provenance, and it appears as `fewshot_seed` in `random`'s/`basic_random`'s
`prompt.json` alongside `fewshot_example_uids`, so the draw stays fully
auditable even though it plays no role in building the prompt itself.

### 8.4 Parsing and quality metrics (`json_parser.parse_fewshot`, `src/eval_task2.py`)

`parse_fewshot` reuses Task 1's JSON-extraction helpers (whole-string →
fenced → first `{...}` substring) but drops the two most lenient rungs of the
ladder (`regex_kv`, `bare_label`): those exist in Task 1 to recover a label
from badly-malformed output, but there is no way to recover a fabricated
`confidence`/`evidence` from a bare word, so a reply that doesn't parse as
proper JSON is simply invalid here. Unlike Task 1's `ParseResult`, it also
records *which* extraction stage succeeded and which extra keys were present
- Task 1 only needed this to gate `strict_schema` mode, but Task 2 needs it
as a first-class metric (format compliance, below). `confidence` is coerced
to a clamped `[0.0, 1.0]` float on a best-effort basis - a non-numeric value
degrades to `None` (and `confidence_valid=False`) rather than failing the
row, since the label is what the classification metrics need.

`metrics.compute()` (shared with Task 1) covers final-label quality -
accuracy, macro-F1, per-class, confusion matrix, invalid-output rate.
`eval_task2.evaluate()` covers the three dimensions the richer schema adds,
all heuristic and deterministic (no extra model calls, no LLM-as-judge,
matching the project's no-training baseline philosophy):

- **Evidence relevance** - fraction of the evidence sentence's content words
  (stopwords and words ≤2 chars dropped) that also appear in the review text.
  A fabricated or generic evidence sentence ("the label is negative") scores
  near zero; a genuine citation of the review's own wording scores high.
  `label_echo_rate` separately flags evidence that just restates the label
  word instead of citing anything.
- **Confidence calibration** - Expected Calibration Error (10 confidence
  bins; ECE = Σ (bin weight) × |bin accuracy − bin mean confidence|) and
  Brier score, both computed against the model's own correctness (`gold ==
  predicted`), over the rows that returned a numeric confidence.
- **Format compliance** - `schema_compliance_rate` = the fraction of rows
  that are strict JSON, contain exactly `label`/`confidence`/`evidence` and
  nothing else, with a numeric confidence and a non-empty evidence string.
  This is stricter than `invalid_output_rate` (which only requires a
  recoverable label): a reply extracted via `fenced`/`substring`, or missing
  `confidence`, still counts as a valid *label* but not as schema-compliant.

### 8.5 Inference (`scripts/05_run_task2_fewshot.py`)

The opposite trade-off from Task 1's inference loop (§3.6): **no retry
ladder**. This script exists to compare prompt strategies quickly, not to
run a hardened final pipeline — every failure counts directly against the
strategy being tested on the first attempt, rather than being smoothed over by
retries. If a strategy's invalid rate turns out high on `val`, the same
retry-escalation pattern `inference.py`/`prompts.py` already use could be
ported over before the final `test` run.

Each strategy is scored in its own pass and written to its own subdirectory —
`outputs/task2/<run_id>/{curated,random,basic,basic_random}/` — reusing
`metrics.save_all()` unchanged for the label-quality artefacts
(`predictions_task2.csv`, `invalid_outputs_task2.csv`, confusion matrix PNG),
then appending the `eval_task2` quality block to the same `results_task2.md`
and writing it separately as `quality_task2.json` (kept out of `metrics.py`
so that module stays generic/shared with Task 1). `prompts2.prompt_metadata(strategy)`
is written as that strategy's own `prompt.json` - the Task 2 counterpart of
Task 1's `prompt.json`, but one per strategy since they genuinely differ (for
`random`/`basic_random` it also carries `fewshot_seed`/`fewshot_example_uids`,
§8.3.1). Every strategy - including `random`/`basic_random` - now runs
exactly once per invocation of `scripts/05_run_task2_fewshot.py`, the same as
`curated`/`basic`; there is no seed loop left in the runner. The console
prints three side-by-side tables at the end: label quality, quality
dimensions, and performance.

### 8.6 Data roles

Task 2 reuses Task 1's frozen split (§3.1) but, unlike Task 1 (which scores
the full pool because there is no training to leak into), puts `val`/`test`
to their originally-intended purpose:

| File | Role in Task 2 |
|---|---|
| `random_fewshot_samples.csv` | the fixed 4-row draw (2 positive + 2 negative, seed 42) shared by `random` and `basic_random` - see §8.3.1 |
| `selected_sample.csv` | the 12 curated candidates the 4 demos shared by `curated` and `basic` were chosen from |
| `val.csv` | compares the prompt strategies |
| `test.csv` (200 examples) | the final headline number, once a strategy is picked |

That is the same train/select/report discipline a fine-tuned model (Task 3)
would follow — applied here even though there is no training, so the reported
Task 2 number is chosen the same principled way rather than by eyeballing.

### 8.7 Results

Run `scripts/05_run_task2_fewshot.py` and see
`outputs/task2/<run_id>/<strategy>/results_task2.md` for the current numbers,
or `scripts/06_summarize_task2_runs.py` for a CSV across every run (including
the `schema_compliance_rate`/`mean_evidence_relevance`/`ece` columns). The
strategies are compared on `val` first, then the field is re-scored on `test`
(`outputs/task2/final_all/`) and the winner picked from that: **`curated`**
(macro-F1 ≈ 0.94, invalid-output ≈ 0.3%, schema-compliance ≈ 99.5%).
`basic_representative` leads on `val` but drops on `test` (and its
schema-compliance is far lower, ~73%), which is itself a small cautionary tale
about picking on `val` alone — and the reason §8.8 exists.

### 8.8 Prompt sensitivity of the best & worst strategy (`scripts/07_run_prompt_sensitivity.py`)

§8.7 picks a winner by changing *large* things between strategies — which
examples, how many rules. That leaves a question the strategy comparison
can't answer: how much of a strategy's score is a real property of the prompt,
and how much is luck in wording that a reasonable person could have written
differently? Step 7 answers it on the **test** split by perturbing one
strategy's prompt — `--strategy`, default `curated` (the winner), or
`--strategy random` (the loser) — along three axes that are meant to be
semantically inert, one at a time, with everything else (model, decoding, that
strategy's 4 demo reviews and their hand-authored answers, the exact eval rows)
held fixed. Running both answers a sharper question: is `random` *worse*, or
just *more fragile*?

| Axis | Variants | Isolates |
|---|---|---|
| **instruction phrasing** | 4 full paraphrases (`p1`–`p4`; `p1` is verbatim) of the whole instruction block — context, both label definitions with their example snippets, the "judge the overall verdict / mixed / sarcasm" rules, and the JSON-answer instruction — each keeping every definition and rule, changing only wording, sentence structure and order. `curated` and `random` share the *identical* instruction block (they differ only in demos), so one paraphrase set is faithful to both — `assert_baseline_matches` checks that per strategy. | whether the model keys off the instruction's exact surface form or only its content |
| **demo order** | the same 4 demos in each of `pos,neg,pos,neg` / `pos,pos,neg,neg` / `neg,neg,pos,pos` / `neg,pos,neg,pos`; the strategy's own file layout is the baseline (`curated`: alternating, `random`: pos-first) and the other three are the perturbations | recency/primacy effects — is the model biased toward the last demo's label? |
| **output schema** | the 3 fields reordered (`confidence,evidence,label`); and renamed (`label→sentiment`, `confidence→certainty`, `evidence→justification`) everywhere they appear — the format line, every demo answer, and the instruction prose that spells the schema out in words | whether field order/naming touches label quality or only format compliance |

**The baseline is not re-implemented.** `src/prompt_sensitivity.py` reads each
strategy's instruction and 4 demos (review text + hand-authored answer) back
out of its `.txt`, and `assert_baseline_matches(strategy)` fails the run if
`build_system_prompt(strategy, **baseline_for(strategy))` does not reproduce
that file byte for byte *and* the instruction block does not equal the shared
`PHRASINGS["p1"]` — so the "baseline" column is provably the real strategy, a
later `.txt` edit can't silently desync the harness, and a future strategy
with a *different* instruction block fails loudly rather than being paraphrased
unfaithfully. `SUPPORTED_STRATEGIES` / `WINNING_STRATEGY` are one-line changes.
The paraphrases `p2`–`p4` *are* hand-written (a multi-paragraph instruction
can't be mechanically reworded), so an offline test checks each one still
carries all four example snippets and the sarcasm/mixed-review rules — a cheap
guard against a paraphrase quietly dropping content.

One byproduct: `prompts/task2_fewshot_random.txt` had a one-character typo (a
demo starting `eview:` instead of `Review:`) that broke the byte-exact
baseline parse; it was fixed here. The `outputs/task2/final_all/random` number
predates the fix — re-run `scripts/05` for `random` to refresh it; the effect
is a single token at the start of one demo and `random` is the loser regardless.

**Parsing the renamed schema.** The `renamed` variant makes the model emit
`{"sentiment": …, "certainty": …, "justification": …}`.
`prompt_sensitivity.parse_variant()` maps those keys back to
`label`/`confidence`/`evidence` *before* the normal `parse_fewshot` validation,
reusing `json_parser._extract_object` so the real extraction strategy
(`strict`/`repaired`/`substring`) is preserved rather than a re-serialised
`strict`. Label accuracy, macro-F1 and calibration therefore stay directly
comparable across schema variants; `schema_compliance_rate` for the renamed
variant reads as "structurally a clean 3-field object", not "used the baseline
field names", and the report says so in a footnote. The `reordered` variant
changes only the key order in the format line and demo answers — the
instruction prose still lists the fields label-first, a deliberate small
inconsistency the report also footnotes.

**No retry ladder**, same as §8.5 — this is a comparison harness, not a
hardened pipeline, so every parse failure counts directly against the variant.
Each variant is scored in its own pass into
`outputs/task2_sensitivity/<run_id>/<axis>__<variant>/`, reusing
`metrics.save_all()` + `eval_task2.evaluate()` unchanged.
`sensitivity_report.md` tabulates, per axis, each variant's
accuracy/macro-F1/invalid-rate/schema-compliance/evidence-relevance/ECE and
then the **range** and population **std** of those columns across
`{baseline + that axis}` — a compact "how brittle is this" readout, plus a
`--dry-run` mode that writes every variant's prompt text without loading the
model.

### 8.9 Self-consistency (`scripts/08_run_self_consistency.py`, `src/self_consistency.py`)

Task 2's inference is greedy — one deterministic reply per review. Self-
consistency (Wang et al. 2022) instead draws N stochastic replies and takes
the majority label; the intuition is that a review the model genuinely
understands lands on the same label across samples, while a borderline call
wobbles, and the vote is more robust than any single draw. Step 8 runs it on
the two ends of the §8.7 test-set spread — `curated` (best) and `random`
(worst) — to see whether voting closes the gap between a good prompt and a bad
one, or just adds cost.

**The greedy path is untouched.** `--self-consistency 1` (the default) calls
the exact same `model.generate_batch(..., temperature=0.0)` +
`json_parser.parse_fewshot` that `scripts/05` uses, and writes the identical
columns — the two new ones (`vote_margin`, `n_valid_votes`) are appended as
constants (`1.0`, `1`). `--self-consistency N` (N > 1) switches to a new
`model.generate_batch_samples()`: one `generate()` call with
`num_return_sequences=N`, `do_sample=True`, `temperature=0.7`, `top_p=0.95`.
The spec was written against vLLM (`SamplingParams(n=N, ...)`); this project
runs on HF transformers, where `num_return_sequences=N` is the direct
equivalent — a single forward loop, not N separate calls. Determinism comes
from `transformers.set_seed(cfg.seed)` once before the batch loop; a re-run on
the same hardware reproduces the same samples (same GPU-arch caveat as §4).

**Vote aggregation** (`self_consistency.aggregate`, the only new logic, kept in
its own module so it is unit-tested offline and touches neither `json_parser`
nor `metrics.py` nor `eval_task2.py`):

| Field | Rule |
|---|---|
| valid row | ≥ 1 of the N candidates parses |
| `label` | majority vote over the valid candidates; a tie → the label of the single highest-confidence valid candidate |
| `confidence` | mean confidence of the candidates that voted for the winning label |
| `evidence` | the first winner-agreeing candidate's evidence |
| `parse_strategy` / `extra_keys` | taken from that same representative candidate, so `eval_task2`'s format-compliance metric still has something sensible to read |
| `vote_margin` (new column) | winning votes ÷ valid votes |
| `n_valid_votes` (new column) | how many of the N candidates parsed |

`run_meta.json` records `self_consistency_n`, `temperature`, `top_p`,
`sample_seed`, and the mean vote margin / valid-vote count.

**Why `invalid_output_rate` stops being comparable.** The single-sample
pipeline marks a row invalid when its one reply fails to parse. Self-
consistency marks a row invalid only when *all N* replies fail — a strictly
weaker bar, so the invalid rate can only go down as N rises, regardless of
whether the labels got any better. The scripts, `run_meta.json`
(`invalid_rate_note`), and the appended `results_task2.md` section all say this
explicitly: compare **accuracy / macro-F1** across modes, and read
`invalid_output_rate` only *within* a set of self-consistency runs. Everything
else in the report (`accuracy`, `macro_f1`, calibration, evidence relevance)
stays directly comparable, because those are computed the same way on the
voted label/confidence/evidence as on a greedy one.

---

## 9. Task 1 prompt ablations: basic vs. structured zero-shot vs. CARP

Three prompts share the same model, data, split, retry-aware inference loop,
and metrics, so the numbers isolate what the prompt's structure is buying:

| Prompt | Asks for | Command / config |
|---|---|---|
| basic (§9.5) | a bare `POSITIVE`/`NEGATIVE` word - no rules, no schema | `03_run_task1.py --strategy basic` / `task1_basic.yaml` |
| zero-shot (§3.4, the Task 1 baseline) | `{"label": ...}`, with label definitions + 4 rules | `03_run_task1.py` (default) / `task1.yaml` |
| CARP (§9.1-9.4) | `{"clue", "reasoning", "label"}`, same rules + a required reasoning chain | `03_run_task1.py --strategy carp` / `task1_carp.yaml` |

All three write into the same `outputs/task1/` directory, tagged by
`run_meta.json`'s `"prompt_method"` (`basic` / `zero_shot` / `carp`), so
`scripts/04_summarize_runs.py` compares all three in one table (§9.5 has the
full mechanism). §9.1-9.4 below cover CARP, the most structurally different
of the three; §9.5 covers the basic prompt, the plainest.

### 9.1 What CARP changes vs. what's reused

`scripts/03_run_task1.py --strategy carp` reruns Task 1 with a **CARP**
prompt (Clue And Reasoning Prompting, Sun et al. 2023, "Text Classification
via Large Language Models") instead of the plain zero-shot one, to compare
the two prompting methods head-to-head under otherwise identical conditions.

Everything except the prompt and the decoding budget is shared with Task 1's
plain pipeline:

| Reused unchanged | New |
|---|---|
| `model.py`, `metrics.py`, `dataset.py`, `env_record.py`, `gpu_monitor.py`, `json_parser.parse()` | `prompts_carp.py` (loads `prompts/task1_carp.txt`) |
| `inference.run()` / `_classify_batch()` / `latency_probe()`'s retry-aware loop, checkpointing, and latency probe | `configs/task1_carp.yaml`, the `carp` entry in `03_run_task1.py`'s `STRATEGIES` dict |
| the label definitions and disambiguation rules from `prompts/task1_zeroshot.txt` | |

`inference.py`'s three public functions now take an optional `build_messages`
callable (default: `prompts.build_messages`), instead of hard-importing
`prompts` internally. That is the entire mechanism that lets
`--strategy carp` reuse the retry ladder, JSONL checkpoint/resume, and
latency probe verbatim - it just passes `prompts_carp.build_messages` through
instead of duplicating the loop.

### 9.2 The CARP prompt (`prompts/task1_carp.txt`)

Same label definitions and four disambiguation rules as the plain zero-shot
prompt (§3.4) - the classification task hasn't changed, only how the model is
asked to arrive at the answer. CARP requires three ordered steps before a
label is allowed:

1. **Clue** - surface-level sentiment-carrying keywords, phrases, tone, and
   any explicit recommendation the review contains.
2. **Reasoning** - a step-by-step diagnostic process from those clues to a
   verdict, explicitly required to say which clue dominates when they
   conflict (rule 2: net judgment, not sentence-counting).
3. **Label** - the conclusion the reasoning actually reached.

This is the opposite instruction from Task 2's `evidence` field, which
explicitly asks the model to "state the conclusion directly - do not show
step-by-step reasoning" (the evidence justifies a decision already made).
CARP's reasoning is meant to influence the decision, not just narrate it
afterward - the prompt says so explicitly ("never write reasoning that
supports one label and then output the other").

The reply schema is `{"clue": ..., "reasoning": ..., "label": ...}` instead
of Task 1's bare `{"label": ...}`. `json_parser.parse()` needed **no code
changes** to handle this: it already extracts `label` from any JSON object
regardless of what else is present, reporting the rest as `extra_keys` - so
`clue`/`reasoning` simply show up as `extra_keys: ["clue", "reasoning"]` on
every successfully parsed CARP row, visible per-row in
`predictions_task1.csv` for qualitative inspection.

### 9.3 Decoding and input budget

CARP's answer is far longer than a bare label, so both budgets in
`configs/task1_carp.yaml` are sized up from `configs/task1.yaml`:

| Setting | Plain Task 1 | CARP | Why |
|---|---|---|---|
| `decoding.max_new_tokens` | 24 | 448 | a bare label needs ~12 tokens; a clue list plus several sentences of reasoning needs much more room. An earlier value of 320, tried first, still truncated the JSON before the `label` field on ~8% of a small smoke sample - the failure showed up as `no_label_found`, indistinguishable from a genuinely malformed reply, until the raw output was inspected and turned out to be a reasoning chain cut off mid-sentence. |
| `model.max_input_tokens` | 1536 | 2048 | CARP's system prompt is ~404 tokens (measured), vs. ~229 for the plain prompt - nearly double the fixed overhead stacked on top of the same reviews. |

### 9.4 A latent bug this surfaced in Task 1 itself

Measuring CARP's real token length (via the tokenizer, not an estimate) is
what surfaced a bug in the *plain* Task 1 config: `max_input_tokens: 1024`
had been sized from an EDA word-count estimate (`words_p95 ≈ 593` words,
scaled by an assumed ~1.3 tokens/word) that never accounted for the ~229
tokens the system prompt itself adds to every request. Measured directly:
review length alone is p95 ≈ 738 / p99 ≈ 1085 tokens, so `system + review` at
p99 is ≈ 1314 tokens - already over the 1024 cap. Across a 2,000-row sample
of the pool, **4.45% of reviews exceeded 1024 tokens** once the system prompt
was included, which silently truncated the *end* of the prompt - including
the trailing `JSON:` cue and the chat template's own generation-prompt
tokens - rather than raising an error. The `full_pool` (24,902-row) Task 1
run's measured `invalid_output_rate` of 4.41% lines up almost exactly with
that 4.45% estimate, which is strong corroborating evidence this was a real,
not just theoretical, source of failures. `configs/task1.yaml` now uses
`max_input_tokens: 1536` (clears p99.5 ≈ 1429 with margin); a full-pool rerun
is expected to lower the invalid-output rate and may shift accuracy/macro-F1
slightly.

### 9.5 The basic prompt: the floor CARP and zero-shot are measured against

`scripts/03_run_task1.py --strategy basic` (`prompts/task1_basic.txt` /
`src/prompts_basic.py`) is the opposite move from CARP: instead of adding
structure on top of the zero-shot prompt, it strips the zero-shot prompt's
structure away entirely - no label definitions, no disambiguation rules, no
JSON schema, no reasoning step. Three lines of instruction, the review, and
a bare `Sentiment:` cue; the model answers with one word.

That word is not JSON, which is why this prompt needed **zero parser
changes**, same as CARP needed none: `json_parser.parse()`'s ladder already
ends in a `bare_label` rung specifically built to recover a single label word
with nothing else around it (originally meant to catch a JSON-prompted model
that occasionally drops the JSON and just answers in prose). The basic
prompt just lands there on essentially every row instead of occasionally, so
`predictions_task1.csv`'s `parse_strategy` column reads `bare_label`
uniformly - expected, not a sign of degraded compliance, since this prompt
never asked for JSON in the first place.

`configs/task1_basic.yaml` keeps `decoding.max_new_tokens` at 24 (matching
the structured zero-shot prompt, not CARP's 448) and `max_input_tokens` at
1536 (matching plain Task 1's corrected value - the basic prompt's system
message is shorter than the zero-shot one, so 1536 is already generous
here). Equalising the budget with the zero-shot prompt keeps the comparison
about the prompt's structure specifically, not a side effect of a bigger
token allowance.

### 9.6 Comparing all three

All three configs write to `outputs/task1/<run_id>/` - **the same
directory** - and tag `run_meta.json` with `"prompt_method"`: `"zero_shot"`
(plain), `"carp"`, or `"basic"` (older runs from before this field existed
default to `"zero_shot"` when read back). `scripts/04_summarize_runs.py`
already walks every subdirectory of `outputs/task1/`, so all three land in
one CSV via its `prompt_method` column with no extra wiring - the comparison
each ablation script's docstring asks for is the existing summarizer, not a
new script per pair.

---

## 10. Extending to Tasks 3–4

The pieces built for Task 1 and reused by Task 2 are meant to keep being
reused as-is:

- **Task 3 (QLoRA):** implemented in `src/qlora.py` + `src/instruction_format.py`
  + `src/train_task3.py`, driven by `scripts/12_run_task3_qlora.py`
  (`configs/task3.yaml`). Fine-tunes a LoRA adapter on **Llama-3.1-8B-Instruct**
  loaded in 4-bit NF4 (double quant, bf16 compute); trains only on `train.csv`,
  selects the checkpoint by **generation** macro-F1 on `val.csv`
  (`MacroF1Trainer` + `load_best_model_at_end`), and reports on `test.csv`
  across ≥ 2 seeds. The supervised target is byte-identical to Task 1's
  `{"label": ...}` answer, so `json_parser` and `metrics.compute` are reused
  unchanged and the fine-tuned number sits in the same table as the
  zero-/few-shot baselines. See README §15.
- **Task 4 (distillation):** the teacher's predictions land in the same
  `predictions.jsonl` format, so the student trains on teacher labels using the
  same loader and is evaluated by the same `metrics.compute()`.

Because the split is created once and reused by every later task, every one of
those numbers is directly comparable to the Task 1 and Task 2 baselines.
