#!/usr/bin/env python
"""Step 7b - run the Task 2 prompt-sensitivity check on BOTH strategies in turn.

Thin wrapper over `scripts/07_run_prompt_sensitivity.py`: it invokes that script
once per strategy (default: `curated` then `random`), each in its own fresh
process so the model is loaded cleanly for each. Every knob keeps that script's
own default *except* the sample size, which defaults to 200 here (a fast first
pass) instead of the config's 1000 - still overridable with `--n-samples`.

The underlying script already defaults `--source` to `test`, so these runs
score the test split unless `--source` says otherwise.

Usage:
    # both strategies, test split, 200 rows each
    python scripts/07b_run_prompt_sensitivity_both.py

    # both strategies, full 1000 test rows
    python scripts/07b_run_prompt_sensitivity_both.py --n-samples 1000

    # all of test.csv, one axis only, custom output-folder prefix
    python scripts/07b_run_prompt_sensitivity_both.py --n-samples 100000 \
        --only-axis demo_order --run-id-prefix quick

    # inspect every variant prompt, no GPU
    python scripts/07b_run_prompt_sensitivity_both.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

SCRIPT = Path(__file__).with_name("07_run_prompt_sensitivity.py")
STRATEGIES = ("curated", "random")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run 07_run_prompt_sensitivity.py for curated and random back to back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-samples", type=int, default=200,
                    help="eval rows from the split, per strategy (default: 200; "
                         "a value >= the split size uses all of it)")
    ap.add_argument("--strategies", default=",".join(STRATEGIES),
                    help=f"comma-separated subset/order (default: {','.join(STRATEGIES)})")
    ap.add_argument("--source", default=None,
                    help="split to score on (default: the inner script's default, test)")
    ap.add_argument("--only-axis", default=None,
                    choices=["phrasing", "demo_order", "schema"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-id-prefix", default=None,
                    help="output folder is <prefix>_<strategy> (default: <strategy>_<timestamp>)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and save every variant's prompt, then stop (no model load)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(STRATEGIES)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures: list[str] = []

    for strategy in strategies:
        cmd = [sys.executable, str(SCRIPT), "--strategy", strategy,
               "--n-samples", str(args.n_samples)]
        if args.source is not None:
            cmd += ["--source", args.source]
        if args.only_axis is not None:
            cmd += ["--only-axis", args.only_axis]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.config is not None:
            cmd += ["--config", args.config]
        if args.hf_token is not None:
            cmd += ["--hf-token", args.hf_token]
        if args.dry_run:
            cmd += ["--dry-run"]
        if args.run_id_prefix is not None:
            cmd += ["--run-id", f"{args.run_id_prefix}_{strategy}"]
        else:
            cmd += ["--run-id", f"{strategy}_{stamp}"]

        print("\n" + "#" * 70)
        print(f"# prompt-sensitivity  ->  strategy={strategy}  n={args.n_samples}")
        print("#  " + " ".join(cmd))
        print("#" * 70, flush=True)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(strategy)
            print(f"\n!! {strategy} exited with code {result.returncode}; continuing", flush=True)

    print("\n" + "=" * 70)
    if failures:
        print(f"DONE with failures: {failures}")
        raise SystemExit(1)
    print(f"DONE - {len(strategies)} strateg{'y' if len(strategies) == 1 else 'ies'} finished cleanly")
    print("Summarise both with: python scripts/06_summarize_task2_runs.py "
          "(or read each run's sensitivity_report.md)")


if __name__ == "__main__":
    main()
