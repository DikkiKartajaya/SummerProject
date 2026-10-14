#!/usr/bin/env python
"""End-to-end pipeline test with a FAKE model - no GPU, no weights downloaded.

Verifies the wiring that the offline unit tests do not touch: the retry loop,
the JSONL checkpoint/resume path, and the metrics/figure writers. The fake
model deliberately emits messy output (markdown fences, refusals, prose) so
the retry and fallback branches are actually exercised.

Usage:
    python tests/test_pipeline_mock.py
"""

from __future__ import annotations

import random
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Make `import torch` succeed on a CPU-only box ---------------------------
# The real run always has torch; this stub only exists so the test is runnable
# anywhere. If torch IS installed, the real one is used.
try:
    import torch  # noqa: F401
except ImportError:
    stub = types.ModuleType("torch")
    # scipy/array-api probes `torch.Tensor`, so the stub must expose it.
    stub.Tensor = type("Tensor", (), {})
    stub.float16 = "float16"
    stub.bfloat16 = "bfloat16"
    stub.inference_mode = lambda: (lambda fn: fn)
    stub.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None,
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda *a: 0, device_count=lambda: 0,
    )
    stub.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(is_available=lambda: False, version=lambda: None,
                                    deterministic=False, benchmark=False)
    )
    stub.version = types.SimpleNamespace(cuda=None)
    stub.manual_seed = lambda s: None
    sys.modules["torch"] = stub

import pandas as pd

from src import inference, metrics as metrics_mod, model as model_mod
from src.utils import load_config, read_jsonl

OUT_DIR = ROOT / "outputs" / "_mock_test"


# --- Fake generation --------------------------------------------------------
_rng = random.Random(0)
_call_log: list[int] = []  # attempts observed, to prove retries happen


def fake_generate_batch(lm, batch_messages, cfg, temperature=None):
    """Stand-in for model.generate_batch: returns messy but plausible replies."""
    # Attempt index is inferred from the retry suffix present in the user turn.
    attempt = 0
    user = batch_messages[0][1]["content"]
    if "FINAL ATTEMPT" in user:
        attempt = 2
    elif "not valid JSON" in user:
        attempt = 1
    _call_log.append(attempt)

    outs = []
    for _ in batch_messages:
        label = _rng.choice(["positive", "negative"])
        if attempt == 0:
            roll = _rng.random()
            if roll < 0.70:
                outs.append(f'{{"label": "{label}"}}')            # clean
            elif roll < 0.82:
                outs.append(f'```json\n{{"label": "{label}"}}\n```')  # fenced
            elif roll < 0.90:
                outs.append(f'Sure. {{"label": "{label}"}}')      # prose + json
            elif roll < 0.96:
                outs.append("I cannot determine this.")           # -> retry
            else:
                outs.append("")                                   # -> retry
        elif attempt == 1:
            outs.append(f'{{"label": "{label}"}}' if _rng.random() < 0.7 else "hmm")
        else:
            # Some rows never recover -> exercises the fallback branch.
            outs.append(f'{{"label": "{label}"}}' if _rng.random() < 0.5 else "???")
    return outs, {
        "batch_seconds": 0.05 * len(batch_messages),
        "batch_size": len(batch_messages),
        "prompt_tokens": 200 * len(batch_messages),
        "generated_tokens": 8 * len(batch_messages),
        "tokens_per_second": 160.0,
    }


def main() -> int:
    cfg = load_config(ROOT / "configs" / "task1.yaml")
    cfg["task1"]["batch_size"] = 8

    # 120 synthetic rows in the shape produced by dataset.load_clean.
    df = pd.DataFrame({
        "uid": [f"u{i:04d}" for i in range(120)],
        "text": [f"synthetic review {i}" for i in range(120)],
        "label": [["positive", "negative"][i % 2] for i in range(120)],
    })

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    model_mod.generate_batch = fake_generate_batch  # monkeypatch
    fake_lm = model_mod.LoadedModel(None, None, "fake/model", "main", "bf16", 0.0, 0, 0.0)

    checks: list[tuple[bool, str]] = []

    # --- 1. full run --------------------------------------------------------
    ckpt = OUT_DIR / "predictions.jsonl"
    preds, stats = inference.run(fake_lm, df, cfg, checkpoint_path=ckpt, resume=False)
    checks.append((len(preds) == 120, f"all rows scored (got {len(preds)})"))
    checks.append((set(preds["uid"]) == set(df["uid"]), "uids preserved"))
    checks.append((max(_call_log) >= 1, "retry path was exercised"))
    checks.append((ckpt.exists() and len(list(read_jsonl(ckpt))) == 120,
                   "checkpoint file complete"))
    checks.append((stats["latency_batched"]["mean_seconds_per_sample"] is not None,
                   "latency recorded"))

    # --- 2. resume: nothing left to do -------------------------------------
    n_calls_before = len(_call_log)
    preds2, _ = inference.run(fake_lm, df, cfg, checkpoint_path=ckpt, resume=True)
    checks.append((len(_call_log) == n_calls_before, "resume skipped finished rows"))
    checks.append((len(preds2) == 120, "resume returned the full set"))

    # --- 3. metrics + artefacts --------------------------------------------
    m = metrics_mod.compute(preds)
    checks.append((m["n"] == 120, "metrics count all rows"))
    checks.append((0.0 <= m["accuracy"] <= 1.0, "accuracy in range"))
    checks.append((0.0 <= m["accuracy_overall"] <= 1.0, "accuracy_overall in range"))
    checks.append((m["accuracy_overall"] <= m["accuracy"], "accuracy_overall <= accuracy (invalid rows only hurt)"))
    checks.append((m["invalid_output_rate"] is not None, "invalid-output rate recorded"))

    paths = metrics_mod.save_all(preds, m, {"checkpoint": "fake", "seed": 42},
                                 OUT_DIR, tag="mock")
    checks.append((all(Path(p).exists() for p in paths.values()),
                   "all artefacts written"))

    for ok, name in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    n_fail = sum(1 for ok, _ in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed.")
    print(f"Mock artefacts: {OUT_DIR}")
    print(f"Sample metrics: acc={m['accuracy']} acc_overall={m['accuracy_overall']} "
          f"macro_f1={m['macro_f1']} invalid_output_rate={m['invalid_output_rate']}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
