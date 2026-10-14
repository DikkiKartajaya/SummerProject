"""Capture the exact software/hardware environment a run happened in.

The task sheet asks for: Python, PyTorch, CUDA, Transformers, PEFT versions,
GPU VRAM and OS. We record those plus a few extras that matter for
reproducing numbers (driver version, bitsandbytes, git commit).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from .utils import PROJECT_ROOT, get_hf_token, write_json


def _pkg_version(name: str) -> str:
    """Installed version of a package, or 'not installed'."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _run(cmd: list[str]) -> str | None:
    """Run a shell command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_info() -> dict[str, Any]:
    """GPU name, total VRAM and compute capability via torch; driver via nvidia-smi."""
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
    except ImportError:
        info["error"] = "torch not installed"
        return info

    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_runtime_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

    if not info["cuda_available"]:
        return info

    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            "index": i,
            "name": props.name,
            "total_vram_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"sm_{props.major}{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    info["devices"] = devices
    info["driver_version"] = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return info


def _cpu_ram_info() -> dict[str, Any]:
    info: dict[str, Any] = {"processor": platform.processor() or platform.machine()}
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    return info


def collect() -> dict[str, Any]:
    """Build the full environment record as a JSON-serialisable dict."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            # Non-empty means we are inside the project venv, as intended.
            "in_venv": sys.prefix != sys.base_prefix,
        },
        "packages": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "accelerate": _pkg_version("accelerate"),
            "bitsandbytes": _pkg_version("bitsandbytes"),
            "tokenizers": _pkg_version("tokenizers"),
            "datasets": _pkg_version("datasets"),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("scikit-learn"),
        },
        "gpu": _gpu_info(),
        "cpu": _cpu_ram_info(),
        "hf_token_set": bool(get_hf_token()),  # presence only, never the value
        "git_commit": _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"]),
    }


def save(path: str = "outputs/env_record.json") -> dict[str, Any]:
    """Collect the record and write it to disk."""
    record = collect()
    write_json(record, path)
    return record


def summarize(record: dict[str, Any]) -> str:
    """Short human-readable block, printed at the top of every run."""
    gpu = record["gpu"]
    if gpu.get("devices"):
        d = gpu["devices"][0]
        gpu_line = f"{d['name']} | {d['total_vram_gb']} GB | {d['compute_capability']}"
    else:
        gpu_line = "no CUDA device"

    pkgs = record["packages"]
    lines = [
        "----- Environment -----",
        f"OS            : {record['os']['system']} {record['os']['release']}",
        f"Python        : {record['python']['version']} (venv={record['python']['in_venv']})",
        f"PyTorch       : {pkgs['torch']}",
        f"CUDA runtime  : {gpu.get('cuda_runtime_version')}  driver {gpu.get('driver_version')}",
        f"Transformers  : {pkgs['transformers']}",
        f"PEFT          : {pkgs['peft']}",
        f"bitsandbytes  : {pkgs['bitsandbytes']}",
        f"GPU           : {gpu_line}",
        f"HF token      : {'set' if record.get('hf_token_set') else 'not set'}",
        "-----------------------",
    ]
    return "\n".join(lines)
