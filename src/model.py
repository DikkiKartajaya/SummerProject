"""Load Llama-3.2-3B-Instruct and run deterministic batched generation.

Precision (config: model.precision): bf16 (default) | fp16 | nf4 | int8.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .utils import get_hf_token, get_logger

# transformers is imported lazily inside load() so offline tests don't need it

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """Model + tokenizer + the metadata we want recorded in the results."""
    model: Any
    tokenizer: Any
    checkpoint: str
    revision: str
    precision: str
    load_seconds: float
    n_parameters: int
    weights_vram_gb: float
    token_source: str = "none"  # where the HF credential came from (never the value)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _quantization_config(precision: str):
    """Build a BitsAndBytesConfig for the 4-/8-bit paths, or None for fp paths."""
    if precision not in ("nf4", "int8"):
        return None

    from transformers import BitsAndBytesConfig

    if precision == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # extra ~0.4 bit/param saved
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load(cfg: dict[str, Any], hf_token: str | None = None) -> LoadedModel:
    """Load tokenizer + model according to the config, and report VRAM used.

    `hf_token` is optional: when omitted it's read from the environment / .env.
    Public checkpoints load fine with no token at all.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    checkpoint, revision = mcfg["checkpoint"], mcfg.get("revision", "main")
    precision = mcfg["precision"].lower()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found. This project targets a local GPU.")

    if hf_token:
        token, token_source = hf_token, "argument"
    else:
        token = get_hf_token()
        token_source = "env" if token else "none"
    auth_kwargs = {"token": token} if token else {}
    logger.info("HuggingFace token: %s", token_source)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, clean_up_tokenization_spaces=False, **auth_kwargs
    )
    # Llama has no dedicated pad token. Left padding is mandatory for batched
    # decoder-only generation, otherwise the pads sit between the prompt and
    # the first generated token and corrupt the output.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_cfg = _quantization_config(precision)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": dtype,
        "attn_implementation": mcfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": True,
        **auth_kwargs,
    }
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
        kwargs["device_map"] = mcfg.get("device_map", "cuda:0")

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    if quant_cfg is None:  # quantized models are already placed by device_map
        model = model.to(mcfg.get("device_map", "cuda:0"))
    model.eval()

    load_seconds = time.perf_counter() - t0
    weights_vram_gb = (torch.cuda.memory_allocated() - vram_before) / 1024**3
    n_params = sum(p.numel() for p in model.parameters())

    logger.info(
        "Loaded %s (%s) in %.1fs | %.2fB params | %.2f GB VRAM for weights",
        checkpoint, precision, load_seconds, n_params / 1e9, weights_vram_gb,
    )
    return LoadedModel(
        model=model, tokenizer=tokenizer, checkpoint=checkpoint, revision=revision,
        precision=precision, load_seconds=load_seconds, n_parameters=n_params,
        weights_vram_gb=round(weights_vram_gb, 3), token_source=token_source,
    )


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def build_generation_kwargs(cfg: dict[str, Any], temperature: float | None = None) -> dict[str, Any]:
    """Assemble generate() kwargs.

    Deterministic (greedy) by default. `temperature` is only honoured on
    retries; passing anything > 0 flips sampling on for that call alone, so the
    main comparison stays fully deterministic.
    """
    dcfg = cfg["decoding"]
    temp = dcfg["temperature"] if temperature is None else temperature
    sample = bool(dcfg["do_sample"]) or temp > 0.0

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(dcfg["max_new_tokens"]),
        "do_sample": sample,
        "repetition_penalty": float(dcfg.get("repetition_penalty", 1.0)),
    }
    if sample:
        kwargs["temperature"] = float(temp)
        kwargs["top_p"] = float(dcfg.get("top_p", 1.0))
    # Passing temperature/top_p with do_sample=False triggers a warning and is
    # ignored, so we simply omit them in the greedy case.
    return kwargs


@torch.inference_mode()
def generate_batch(
    lm: LoadedModel,
    batch_messages: list[list[dict[str, str]]],
    cfg: dict[str, Any],
    temperature: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Generate completions for a batch of chat conversations.

    Returns the decoded continuations (prompt stripped) and timing/token stats.
    """
    tok = lm.tokenizer

    prompts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg["model"]["max_input_tokens"]),
        add_special_tokens=False,  # the chat template already added them
    ).to(lm.model.device)

    gen_kwargs = build_generation_kwargs(cfg, temperature)

    # CUDA is async: synchronise around the call so the latency is real.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = lm.model.generate(
        **enc, **gen_kwargs,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Slice off the prompt: with left padding every row starts generating at
    # the same index, so one offset works for the whole batch.
    prompt_len = enc["input_ids"].shape[1]
    completions = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

    n_new = int((out[:, prompt_len:] != tok.pad_token_id).sum().item())
    stats = {
        "batch_seconds": elapsed,
        "batch_size": len(batch_messages),
        "prompt_tokens": int(enc["input_ids"].numel()),
        "generated_tokens": n_new,
        "tokens_per_second": n_new / elapsed if elapsed > 0 else 0.0,
    }
    return [c.strip() for c in completions], stats


@torch.inference_mode()
def generate_batch_samples(
    lm: LoadedModel,
    batch_messages: list[list[dict[str, str]]],
    cfg: dict[str, Any],
    n: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[list[list[str]], dict[str, float]]:
    """Draw `n` stochastic completions per prompt in ONE generate() call.

    This is the Task 2 self-consistency path (`scripts/08_run_self_consistency.py`).
    vLLM would write this as `SamplingParams(n=n, temperature=..., top_p=...)`;
    on HF transformers the equivalent is `num_return_sequences=n` with
    `do_sample=True` - a single forward loop, not `n` separate calls.

    The greedy `generate_batch` above is untouched: `n == 1` callers keep using
    it. Determinism here comes from the global RNG - call
    `transformers.set_seed(seed)` once before the batch loop.

    Returns a list of per-prompt candidate lists: `out[i]` is the `n` replies
    for `batch_messages[i]`, prompt stripped and whitespace-trimmed.
    """
    if n < 2:
        raise ValueError(f"generate_batch_samples needs n >= 2 (got {n}); use generate_batch for greedy.")

    tok = lm.tokenizer
    prompts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg["model"]["max_input_tokens"]),
        add_special_tokens=False,
    ).to(lm.model.device)

    dcfg = cfg["decoding"]
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(dcfg["max_new_tokens"]),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_return_sequences": int(n),
        "repetition_penalty": float(dcfg.get("repetition_penalty", 1.0)),
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = lm.model.generate(
        **enc, **gen_kwargs,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # generate() returns (batch * n, seq_len) with each prompt's n sequences
    # contiguous: [p0_s0, p0_s1, ..., p0_s{n-1}, p1_s0, ...].
    prompt_len = enc["input_ids"].shape[1]
    cont = out[:, prompt_len:]
    decoded = tok.batch_decode(cont, skip_special_tokens=True)
    bs = len(batch_messages)
    grouped = [[decoded[i * n + j].strip() for j in range(n)] for i in range(bs)]

    n_new = int((cont != tok.pad_token_id).sum().item())
    stats = {
        "batch_seconds": elapsed,
        "batch_size": bs,
        "n_return": int(n),
        "prompt_tokens": int(enc["input_ids"].numel()),
        "generated_tokens": n_new,
        "tokens_per_second": n_new / elapsed if elapsed > 0 else 0.0,
    }
    return grouped, stats
