#!/usr/bin/env python
"""Step 0 - record the environment.

Writes outputs/env_record.json with Python / PyTorch / CUDA / Transformers /
PEFT versions, GPU VRAM and OS. Run once after setup.

Usage:
    python scripts/00_record_env.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src import env_record


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the runtime environment.")
    ap.add_argument("--out", default="outputs/env_record.json")
    args = ap.parse_args()

    record = env_record.save(args.out)
    print(env_record.summarize(record))
    print(f"\nFull record written to {args.out}")

    # GPU is a hard requirement, checked last so everything above still prints
    gpu = record["gpu"]
    if not gpu.get("cuda_available"):
        raise SystemExit("\nERROR: no CUDA device visible. Check driver + torch install.")

    # string-compare would rank "sm_9" above "sm_12", so parse to int first
    major = int(gpu["devices"][0]["compute_capability"].removeprefix("sm_")[:-1])
    torch_ver = tuple(int(x) for x in record["packages"]["torch"].split(".")[:2]
                      if x.isdigit())
    if major >= 12 and torch_ver < (2, 7):
        print("\nWARNING: an RTX 50-series GPU (sm_120) needs PyTorch >= 2.7 built "
              "for CUDA 12.8. See README.md.")


if __name__ == "__main__":
    main()
