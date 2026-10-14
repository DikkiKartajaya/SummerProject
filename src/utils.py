"""Small shared helpers: config loading, seeding, logging, JSON/JSONL I/O, env."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# root of the repo, so relative paths work from any cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["positive", "negative"]  # IMDB is binary: no neutral class exists
UNPARSED = "unparsed"  # sentinel for rows the model never returned valid JSON for


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    path = resolve(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a possibly-relative path into an absolute one under PROJECT_ROOT."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------
# Env / credentials
# --------------------------------------------------------------------------
def load_env() -> None:
    """Load KEY=value lines from .env into the environment, if the file exists."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_hf_token() -> str | None:
    """HuggingFace token from the environment (loads .env first)."""
    load_env()
    return os.environ.get("HF_TOKEN")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG we touch and force deterministic cuBLAS reductions.

    Called before any model load or generation so a re-run reproduces the
    same outputs bit-for-bit on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs; must be set before CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but keep utils importable alone
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism over raw speed: the workload is generation-bound anyway.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Console logger, optionally mirrored to a file. Safe to call twice."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
                            datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file = resolve(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# JSON / JSONL helpers
# --------------------------------------------------------------------------
def write_json(obj: Any, path: str | Path) -> Path:
    """Dump a dict to pretty-printed UTF-8 JSON, creating parent dirs."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def read_json(path: str | Path) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Append records to a JSONL file. Used for crash-safe checkpointing."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())  # survive a hard kill mid-run


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file, skipping a truncated final line if present."""
    path = resolve(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # partial write from an interrupted run
