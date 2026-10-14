"""QLoRA base-model loading + LoRA adapter setup for Task 3.

Three steps, matching the task sheet:

1. `load_base_model()` - load the frozen checkpoint in 4-bit NF4 with double
   quantization. `bnb_4bit_compute_dtype` is bfloat16 when the GPU supports
   it (`torch.cuda.is_bf16_supported()`), float16 otherwise.
2. `prepare_for_training()` - `prepare_model_for_kbit_training`, then enable
   (non-reentrant) gradient checkpointing explicitly.
3. `attach_lora()` - build the `LoraConfig` from the config's target modules
   and wrap the model with `get_peft_model`.

`trainable_parameter_summary()` reports r / alpha / dropout / target modules
and the trainable-parameter percentage (task sheet requirement 3).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .utils import get_hf_token, get_logger

logger = get_logger(__name__)

_DTYPE_ALIASES = {
    "auto": None,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
}


@dataclass
class LoadedBase:
    model: Any
    tokenizer: Any
    checkpoint: str
    revision: str
    compute_dtype: str
    load_seconds: float
    n_parameters: int
    weights_vram_gb: float
    # Full memory picture right after the base weights land on the GPU.
    peak_vram_gb: float | None = None      # peak allocated during load
    reserved_vram_gb: float | None = None  # caching-allocator reservation post-load
    token_source: str = "none"


# --------------------------------------------------------------------------
_GIB = 1024 ** 3


def gpu_memory_snapshot(device: Any = None) -> dict[str, float | None]:
    """Point-in-time CUDA memory, in GiB rounded to 3 dp.

    Keys: ``allocated_gb``, ``reserved_gb``, ``peak_allocated_gb``,
    ``peak_reserved_gb``. Every value is ``None`` when CUDA is unavailable or
    the driver refuses the query, so callers never crash on a CPU box.
    """
    empty = {
        "allocated_gb": None, "reserved_gb": None,
        "peak_allocated_gb": None, "peak_reserved_gb": None,
    }
    if not torch.cuda.is_available():
        return empty
    try:
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(device) / _GIB, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(device) / _GIB, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / _GIB, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / _GIB, 3),
        }
    except Exception as exc:  # noqa: BLE001 - defensive: never break on a stats query
        logger.warning("gpu_memory_snapshot unavailable: %s", exc)
        return empty


@contextmanager
def track_gpu_memory(label: str, device: Any = None) -> Iterator[dict[str, Any]]:
    """Measure the VRAM a block of setup code adds.

    Resets the peak counters on entry; on exit fills (and logs) the yielded
    dict with ``label``, ``delta_allocated_gb``, ``peak_allocated_gb`` and
    ``peak_reserved_gb``. The dict is fully populated once the ``with`` block
    exits. A no-GPU environment yields the same dict shape with ``None``s.
    """
    result: dict[str, Any] = {
        "label": label, "delta_allocated_gb": None,
        "peak_allocated_gb": None, "peak_reserved_gb": None,
    }
    if not torch.cuda.is_available():
        yield result
        return

    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reset_peak_memory_stats failed for %r: %s", label, exc)
    before = torch.cuda.memory_allocated(device)
    try:
        yield result
    finally:
        snap = gpu_memory_snapshot(device)
        after = torch.cuda.memory_allocated(device)
        result["delta_allocated_gb"] = round((after - before) / _GIB, 3)
        result["peak_allocated_gb"] = snap["peak_allocated_gb"]
        result["peak_reserved_gb"] = snap["peak_reserved_gb"]
        logger.info(
            "[mem] %s: +%.3f GB allocated | peak %s GB alloc / %s GB reserved",
            label, result["delta_allocated_gb"],
            result["peak_allocated_gb"], result["peak_reserved_gb"],
        )


def resolve_compute_dtype(requested: str = "auto") -> torch.dtype:
    """bf16 if the GPU supports it (and 'auto'/'bf16' asked), else fp16."""
    requested = (requested or "auto").lower()
    if requested not in _DTYPE_ALIASES:
        raise ValueError(f"unknown compute dtype {requested!r}; use auto|bfloat16|float16")

    explicit = _DTYPE_ALIASES[requested]
    if explicit is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("bfloat16 requested but the GPU does not support it - using float16.")
        return torch.float16
    if explicit is not None:
        return explicit

    # "auto"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _bnb_config(qcfg: dict[str, Any], compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _logical_numel(param: Any) -> int:
    """`param.numel()` corrected for bitsandbytes 4-bit packing.

    A `Params4bit` weight is stored as a packed `uint8` buffer with two 4-bit
    values per byte, so its raw `numel()` is half the logical element count.
    """
    n = param.numel()
    if param.__class__.__name__ == "Params4bit":
        n *= 2
    return n


def _raw_param_count(model: Any, *, trainable_only: bool = False) -> int:
    """Logical parameter count for `model`, correcting for 4-bit packing."""
    return sum(
        _logical_numel(p)
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


def load_base_model(cfg: dict[str, Any], hf_token: str | None = None) -> LoadedBase:
    """Load the frozen base model in 4-bit NF4 + double quant."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    checkpoint = mcfg["checkpoint"]
    revision = mcfg.get("revision", "main")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. Task 3 QLoRA training needs a local GPU.")

    if hf_token:
        token, token_source = hf_token, "argument"
    else:
        token = get_hf_token()
        token_source = "env" if token else "none"
    auth_kwargs = {"token": token} if token else {}
    logger.info("HuggingFace token: %s", token_source)

    compute_dtype = resolve_compute_dtype(
        mcfg.get("quantization", {}).get("bnb_4bit_compute_dtype", "auto")
    )
    logger.info("4-bit NF4 + double quant | compute dtype: %s", compute_dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before_gb = torch.cuda.memory_allocated() / _GIB
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, clean_up_tokenization_spaces=False, **auth_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding for training (loss masking); the generation-eval path
    # flips this to "left" per call. See train_task3.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=revision,
        quantization_config=_bnb_config(mcfg.get("quantization", {}), compute_dtype),
        device_map=mcfg.get("device_map", "cuda:0"),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        **auth_kwargs,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    load_seconds = time.perf_counter() - t0
    # Single source of truth for the post-load memory picture; the weights
    # delta is derived from it rather than a second `memory_allocated()` read.
    mem = gpu_memory_snapshot()
    weights_vram_gb = round((mem["allocated_gb"] or 0.0) - vram_before_gb, 3)
    n_params = _raw_param_count(model)

    logger.info(
        "Loaded %s in %.1fs | %.2fB params | %.3f GB weights | "
        "peak %s GB alloc / reserved %s GB",
        checkpoint, load_seconds, n_params / 1e9, weights_vram_gb,
        mem["peak_allocated_gb"], mem["reserved_gb"],
    )
    return LoadedBase(
        model=model, tokenizer=tokenizer, checkpoint=checkpoint, revision=revision,
        compute_dtype=str(compute_dtype).replace("torch.", ""),
        load_seconds=load_seconds, n_parameters=n_params,
        weights_vram_gb=weights_vram_gb,
        peak_vram_gb=mem["peak_allocated_gb"],
        reserved_vram_gb=mem["reserved_gb"],
        token_source=token_source,
    )


# --------------------------------------------------------------------------
def prepare_for_training(model: Any, use_gradient_checkpointing: bool = True) -> Any:
    """`prepare_model_for_kbit_training` + explicit non-reentrant gradient
    checkpointing (task sheet requirement 2)."""
    from peft import prepare_model_for_kbit_training

    # We enable gradient checkpointing ourselves just below (non-reentrant),
    # so tell prepare_* not to also turn it on.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (non-reentrant).")
    return model


def build_lora_config(cfg: dict[str, Any]):
    from peft import LoraConfig

    lcfg = cfg["lora"]
    return LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        bias=lcfg.get("bias", "none"),
        target_modules=list(lcfg["target_modules"]),
        task_type="CAUSAL_LM",
    )


def attach_lora(model: Any, cfg: dict[str, Any]):
    """Wrap the prepared base model with a fresh LoRA adapter."""
    from peft import get_peft_model

    lora_config = build_lora_config(cfg)
    peft_model = get_peft_model(model, lora_config)
    return peft_model, lora_config


# --------------------------------------------------------------------------
def _count_parameters(model: Any) -> tuple[int, int]:
    """(trainable, total) parameter counts, correcting for 4-bit packing.

    bitsandbytes stores a `Linear4bit` weight as a packed `uint8` buffer, so
    `param.numel()` is only half the logical element count. PEFT's own
    `get_nb_trainable_parameters()` already unpacks this, so use it when the
    model exposes it and fall back to the shared `_raw_param_count` helper.
    """
    getter = getattr(model, "get_nb_trainable_parameters", None)
    if callable(getter):
        trainable, total = getter()
        return int(trainable), int(total)

    return _raw_param_count(model, trainable_only=True), _raw_param_count(model)


def trainable_parameter_summary(model: Any, lora_config: Any) -> dict[str, Any]:
    """r / alpha / dropout / target modules + the trainable-parameter share
    (task sheet requirement 3)."""
    trainable, total = _count_parameters(model)
    return {
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_bias": lora_config.bias,
        "target_modules": sorted(lora_config.target_modules)
        if not isinstance(lora_config.target_modules, str)
        else lora_config.target_modules,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        # kept under the old key too so existing run artefacts stay comparable
        "total_parameters_counted": int(total),
        "trainable_percentage": round(100 * trainable / total, 4) if total else None,
    }


def format_trainable_summary(summary: dict[str, Any]) -> str:
    """Human-readable block, printed once per seed before training starts."""
    return "\n".join([
        "----- LoRA adapter -----",
        f"rank r            : {summary['lora_r']}",
        f"alpha             : {summary['lora_alpha']}",
        f"dropout           : {summary['lora_dropout']}",
        f"bias              : {summary['lora_bias']}",
        f"target modules    : {', '.join(summary['target_modules'])}"
        if not isinstance(summary["target_modules"], str)
        else f"target modules    : {summary['target_modules']}",
        f"trainable params  : {summary['trainable_parameters']:,}",
        f"total params      : {summary['total_parameters']:,}",
        f"trainable %        : {summary['trainable_percentage']}%",
        "------------------------",
    ])


def build_memory_summary(
    base: LoadedBase,
    setup_prepare: dict[str, Any] | None = None,
    setup_lora: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collate the base-load snapshot with the two setup-phase deltas."""
    setup_prepare = setup_prepare or {}
    setup_lora = setup_lora or {}
    return {
        "weights_vram_gb": base.weights_vram_gb,
        "peak_vram_gb": base.peak_vram_gb,
        "reserved_vram_gb": base.reserved_vram_gb,
        "prepare_added_gb": setup_prepare.get("delta_allocated_gb"),
        "lora_added_gb": setup_lora.get("delta_allocated_gb"),
        "setup_prepare": setup_prepare,
        "setup_lora": setup_lora,
    }


def format_memory_summary(mem: dict[str, Any]) -> str:
    """VRAM block in the same style as `format_trainable_summary`."""

    def g(key: str) -> str:
        v = mem.get(key)
        return f"{v:.3f} GB" if isinstance(v, (int, float)) else "n/a"

    return "\n".join([
        "----- GPU memory -----",
        f"weights VRAM       : {g('weights_vram_gb')}",
        f"peak VRAM (load)   : {g('peak_vram_gb')}",
        f"reserved VRAM      : {g('reserved_vram_gb')}",
        f"+ k-bit/grad-ckpt  : {g('prepare_added_gb')}",
        f"+ LoRA adapter     : {g('lora_added_gb')}",
        "----------------------",
    ])
