"""Retry-aware inference loop for Task 1.

Flow per batch:
  1. build prompts (attempt 0, greedy)
  2. generate, parse each reply
  3. collect the rows that failed to parse and retry ONLY those, with a
     stricter prompt suffix and a slightly higher temperature
  4. after `max_attempts`, assign the configured fallback label

Predictions are appended to a JSONL checkpoint after every batch, so an
interrupted run resumes instead of starting over.

`build_messages` is pluggable (default: `prompts.build_messages`, the plain
zero-shot prompt) so this same loop - retries, checkpointing, latency probe,
metrics - can drive any Task 1 prompting method with the same contract:
`build_messages(text: str, attempt: int) -> list[dict[str, str]]`. That is how
`scripts/03_run_task1.py --strategy carp` reuses this module for the CARP
ablation instead of duplicating the loop.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from . import json_parser, model as model_mod, prompts
from .gpu_monitor import MemoryTracker, snapshot
from .utils import append_jsonl, get_logger, read_jsonl, resolve

logger = get_logger(__name__)

BuildMessages = Callable[..., list[dict[str, str]]]


# --------------------------------------------------------------------------
# Retry-aware classification of one batch
# --------------------------------------------------------------------------
def _classify_batch(
    lm: model_mod.LoadedModel,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    build_messages: BuildMessages = prompts.build_messages,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Classify a batch, retrying only the rows whose output failed to parse."""
    rcfg = cfg["retry"]
    max_attempts = int(rcfg["max_attempts"])
    temps = list(rcfg["escalate_temperature"])

    # results[i] is filled in as soon as row i parses successfully.
    results: list[dict[str, Any] | None] = [None] * len(rows)
    pending = list(range(len(rows)))
    batch_stats: list[dict[str, float]] = []

    for attempt in range(max_attempts):
        if not pending:
            break

        messages = [
            build_messages(rows[i]["text"], attempt=attempt)
            for i in pending
        ]
        temperature = temps[min(attempt, len(temps) - 1)]
        completions, stats = model_mod.generate_batch(lm, messages, cfg, temperature)
        stats["attempt"] = attempt
        batch_stats.append(stats)

        # Per-sample latency inside a batch is the batch time shared out; the
        # true single-sample latency is measured separately (see latency_probe).
        per_sample = stats["batch_seconds"] / max(len(pending), 1)

        still_pending: list[int] = []
        for idx, raw in zip(pending, completions):
            parsed = json_parser.parse_and_validate(raw)
            if parsed.ok:
                results[idx] = {
                    "predicted": parsed.label,
                    "attempts_used": attempt + 1,
                    "raw_output": raw[:500],
                    "latency_seconds": round(per_sample, 4),
                    "temperature": temperature,
                    **parsed.as_dict(),
                }
            else:
                still_pending.append(idx)
                # Keep the last failure so it can be inspected if all retries fail.
                results[idx] = {
                    "predicted": None,
                    "attempts_used": attempt + 1,
                    "raw_output": raw[:500],
                    "latency_seconds": round(per_sample, 4),
                    "temperature": temperature,
                    **parsed.as_dict(),
                }

        if still_pending:
            logger.debug("Attempt %d: %d/%d rows need a retry",
                         attempt, len(still_pending), len(pending))
        pending = still_pending

    # Everything that never parsed gets the configured fallback.
    fallback = rcfg.get("fallback_label", "unparsed")
    for idx in pending:
        results[idx]["predicted"] = fallback
        results[idx]["fell_back"] = True

    out = []
    for row, res in zip(rows, results):
        res.setdefault("fell_back", False)
        out.append({
            "uid": row["uid"],
            "text": row["text"],
            "gold": row["label"],
            **res,
        })
    return out, batch_stats


# --------------------------------------------------------------------------
# Dedicated single-sample latency probe
# --------------------------------------------------------------------------
def latency_probe(
    lm: model_mod.LoadedModel,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    n: int = 30,
    warmup: int = 3,
    build_messages: BuildMessages = prompts.build_messages,
) -> dict[str, Any]:
    """Measure true batch-size-1 end-to-end latency.

    Batched throughput and single-request latency are different numbers and
    both are worth reporting; this function isolates the latter. The first few
    calls are discarded because CUDA kernel autotuning inflates them.
    """
    sample = rows[: n + warmup]
    timings: list[float] = []

    for i, row in enumerate(sample):
        msgs = [build_messages(row["text"], attempt=0)]
        t0 = time.perf_counter()
        model_mod.generate_batch(lm, msgs, cfg, temperature=0.0)
        dt = time.perf_counter() - t0
        if i >= warmup:  # skip warm-up iterations
            timings.append(dt)

    if not timings:
        return {"error": "not enough samples for a latency probe"}

    ordered = sorted(timings)
    return {
        "n_measured": len(timings),
        "warmup_discarded": warmup,
        "batch_size": 1,
        "mean_seconds": round(statistics.fmean(timings), 4),
        "median_seconds": round(statistics.median(timings), 4),
        "p90_seconds": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        "p95_seconds": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        "min_seconds": round(min(timings), 4),
        "max_seconds": round(max(timings), 4),
        "throughput_samples_per_second": round(len(timings) / sum(timings), 3),
    }


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
def run(
    lm: model_mod.LoadedModel,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    resume: bool = True,
    build_messages: BuildMessages = prompts.build_messages,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every row in `df`. Returns (predictions_df, run_stats)."""
    checkpoint_path = resolve(checkpoint_path)

    # Resume: skip uids already present in the checkpoint file.
    done: dict[str, dict] = {}
    if resume and checkpoint_path.exists():
        done = {r["uid"]: r for r in read_jsonl(checkpoint_path)}
        if done:
            logger.info("Resuming: %d predictions already on disk.", len(done))

    todo = df[~df["uid"].isin(done.keys())].to_dict("records")
    batch_size = int(cfg["task1"]["batch_size"])
    all_batch_stats: list[dict[str, float]] = []

    logger.info("Scoring %d rows (batch size %d)...", len(todo), batch_size)
    wall_start = time.perf_counter()

    with MemoryTracker() as mem:
        for start in tqdm(range(0, len(todo), batch_size), desc="zero-shot", unit="batch"):
            batch = todo[start: start + batch_size]
            records, stats = _classify_batch(lm, batch, cfg, build_messages=build_messages)
            append_jsonl(records, checkpoint_path)  # crash-safe
            for rec in records:
                done[rec["uid"]] = rec
            all_batch_stats.extend(stats)

    wall_seconds = time.perf_counter() - wall_start

    # Reassemble in the dataframe's original order.
    preds = pd.DataFrame([done[uid] for uid in df["uid"] if uid in done])

    latencies = sorted(preds["latency_seconds"].dropna().tolist())
    total_generated = sum(s["generated_tokens"] for s in all_batch_stats)

    stats = {
        "n_scored": len(preds),
        "wall_seconds": round(wall_seconds, 2),
        "batch_size": batch_size,
        "n_generate_calls": len(all_batch_stats),
        "latency_batched": {
            "mean_seconds_per_sample": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "median_seconds_per_sample": round(statistics.median(latencies), 4) if latencies else None,
            "p95_seconds_per_sample": round(latencies[int(0.95 * (len(latencies) - 1))], 4) if latencies else None,
            "throughput_samples_per_second": round(len(preds) / wall_seconds, 3) if wall_seconds else None,
        },
        "tokens": {
            "generated_total": total_generated,
            "generated_per_second": round(total_generated / wall_seconds, 2) if wall_seconds else None,
        },
        "gpu_memory": mem.report,
        "gpu_snapshot_end": snapshot(),
    }
    return preds, stats
