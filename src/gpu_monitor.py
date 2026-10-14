"""GPU memory measurement.

Two complementary numbers, because they answer different questions:
  - torch allocator peaks (allocated / reserved): what THIS process used.
  - nvidia-smi used_mb: what the whole device shows, including the CUDA
    context and any other process. This is the number that matters when
    deciding whether a workload fits in 16 GB.
"""

from __future__ import annotations

import subprocess
from typing import Any

try:
    import torch
except ImportError:  # keeps the module importable for CPU-only offline tests
    torch = None


def _cuda_ready() -> bool:
    return torch is not None and torch.cuda.is_available()


def reset_peak() -> None:
    """Zero the allocator's peak counters before a measured section."""
    if _cuda_ready():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def nvidia_smi_used_mb(index: int = 0) -> float | None:
    """Device-wide memory in use, in MB, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def snapshot(index: int = 0) -> dict[str, Any]:
    """Current + peak memory, in GB."""
    if not _cuda_ready():
        return {"cuda_available": False}

    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(index)
    smi_mb = nvidia_smi_used_mb(index)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "total_vram_gb": round(props.total_memory / 1024**3, 3),
        # Tensors currently alive / peak since the last reset.
        "allocated_gb": round(torch.cuda.memory_allocated(index) / 1024**3, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 1024**3, 3),
        # Caching allocator's pool: closer to what the OS sees.
        "reserved_gb": round(torch.cuda.memory_reserved(index) / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(index) / 1024**3, 3),
        "nvidia_smi_used_gb": round(smi_mb / 1024, 3) if smi_mb is not None else None,
        "utilization_pct": round(
            100 * torch.cuda.max_memory_reserved(index) / props.total_memory, 1
        ),
    }


class MemoryTracker:
    """Context manager wrapping a measured section.

        with MemoryTracker() as mt:
            ...work...
        print(mt.report["peak_reserved_gb"])
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.before: dict[str, Any] = {}
        self.report: dict[str, Any] = {}

    def __enter__(self) -> "MemoryTracker":
        reset_peak()
        self.before = snapshot(self.index)
        return self

    def __exit__(self, *exc: Any) -> None:
        after = snapshot(self.index)
        self.report = {
            "before": self.before,
            "after": after,
            "peak_allocated_gb": after.get("peak_allocated_gb"),
            "peak_reserved_gb": after.get("peak_reserved_gb"),
            "total_vram_gb": after.get("total_vram_gb"),
            "peak_utilization_pct": after.get("utilization_pct"),
        }
