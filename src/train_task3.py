"""Task 3 QLoRA training engine: one full fine-tune for a single seed.

`train_one_seed()` is the whole cycle:

    load 4-bit base  ->  attach LoRA  ->  build instruction dataset from
    train.csv  ->  Trainer.train()  ->  pick the checkpoint with the best
    validation macro-F1  ->  score that adapter on the full val + test splits

Checkpoint selection (task sheet requirement 4) is driven by *generation*
macro-F1, not token-level loss: `MacroF1Trainer.evaluate()` generates on a
stratified validation subset, parses the JSON answers with the same
`json_parser` the baselines use, and scores them with `metrics.compute`.
`load_best_model_at_end` + `metric_for_best_model="eval_macro_f1"` then
restore the best adapter before the final scoring.

Everything the task sheet asks to log (train/val loss, learning rate,
training time, peak GPU memory) comes out of `trainer.state.log_history` and
the CUDA peak-memory counters; `scripts/12_run_task3_qlora.py` turns it into
CSVs and the diagrams.
"""

from __future__ import annotations

import inspect
import math
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from . import json_parser, instruction_format, metrics as metrics_mod, plot_task3, qlora
from .gpu_monitor import nvidia_smi_used_mb, snapshot
from .utils import LABELS, get_logger, resolve, write_json

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Dataset + collator
# --------------------------------------------------------------------------
class ListDataset(torch.utils.data.Dataset):
    """Thin wrapper so a plain list of tokenised dicts is a torch Dataset."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]


class CausalDataCollator:
    """Right-pad `input_ids` (with `pad_token_id`) and `labels` (with -100),
    building the matching attention mask. Training uses right padding; the
    generation-eval path pads left itself."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids, labs = list(f["input_ids"]), list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_supervised_dataset(
    df: pd.DataFrame, tokenizer: Any, icfg: dict[str, Any], desc: str = "tokenise"
) -> tuple[ListDataset, dict[str, Any]]:
    """Tokenise every (text, label) row into an instruction example."""
    rows, n_dropped = [], 0
    for text, label in tqdm(
        zip(df["text"].tolist(), df["label"].tolist()),
        total=len(df), desc=desc, unit="ex", leave=False,
    ):
        ex = instruction_format.build_supervised_example(
            text, label, tokenizer,
            max_review_tokens=int(icfg["max_review_tokens"]),
            max_seq_len=int(icfg["max_seq_len"]),
            prompt_file=icfg["prompt_file"],
        )
        if ex is None:
            n_dropped += 1
            continue
        rows.append(ex)
    lengths = [len(r["input_ids"]) for r in rows]
    stats = {
        "n_examples": len(rows),
        "n_dropped_too_long": n_dropped,
        "seq_len_mean": round(statistics.fmean(lengths), 1) if lengths else 0,
        "seq_len_max": max(lengths) if lengths else 0,
    }
    logger.info("Supervised dataset: %s", stats)
    return ListDataset(rows), stats


# --------------------------------------------------------------------------
# Generation-based prediction (checkpoint selection + final scoring)
# --------------------------------------------------------------------------
def _set_use_cache(model: Any, value: bool) -> None:
    for obj in (model, getattr(model, "base_model", None)):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            cfg.use_cache = value


def _generation_mode(model: Any, on: bool) -> None:
    """Toggle the model between training layout (gradient checkpointing on,
    KV-cache off) and fast greedy decoding (the reverse)."""
    if on:
        try:
            model.gradient_checkpointing_disable()
        except Exception:  # noqa: BLE001 - not all wrappers expose it
            pass
        _set_use_cache(model, True)
    else:
        _set_use_cache(model, False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


@torch.inference_mode()
def predict_generations(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    prompt_file: str,
    max_review_tokens: int,
    batch_size: int = 16,
    max_new_tokens: int = 16,
    desc: str = "generate",
) -> pd.DataFrame:
    """Greedy-decode a label for every record and parse it.

    `records` need `uid`, `text`, `label`. Returns a frame with
    `gold` / `predicted` (+ `raw_output`) ready for `metrics.compute`.
    """
    was_training = model.training
    prev_padding_side = tokenizer.padding_side
    model.eval()
    _generation_mode(model, on=True)
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    out_rows: list[dict[str, Any]] = []
    try:
        for start in tqdm(
            range(0, len(records), batch_size),
            total=math.ceil(len(records) / batch_size),
            desc=desc, unit="batch", leave=False,
        ):
            batch = records[start: start + batch_size]
            prompt_ids = [
                instruction_format.build_prompt_ids(
                    r["text"], tokenizer, max_review_tokens, prompt_file
                )
                for r in batch
            ]
            max_len = max(len(p) for p in prompt_ids)
            input_ids = torch.tensor(
                [[pad_id] * (max_len - len(p)) + p for p in prompt_ids],
                dtype=torch.long, device=device,
            )
            attn = torch.tensor(
                [[0] * (max_len - len(p)) + [1] * len(p) for p in prompt_ids],
                dtype=torch.long, device=device,
            )
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completions = tokenizer.batch_decode(gen[:, max_len:], skip_special_tokens=True)
            for r, raw in zip(batch, completions):
                parsed = json_parser.parse(raw)
                out_rows.append({
                    "uid": r["uid"],
                    "text": r["text"],
                    "gold": r["label"],
                    "predicted": parsed.label if parsed.ok else "unparsed",
                    "parse_strategy": parsed.strategy,
                    "raw_output": raw.strip()[:200],
                })
    finally:
        tokenizer.padding_side = prev_padding_side
        _generation_mode(model, on=False)
        model.train(was_training)
        torch.cuda.empty_cache()

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------
# Trainer with generation-macro-F1 evaluation
# --------------------------------------------------------------------------
def _make_trainer_class():
    """Build the Trainer subclass lazily (transformers imported on demand)."""
    from transformers import Trainer

    class MacroF1Trainer(Trainer):
        """`evaluate()` reports token-level `eval_loss` AND generation
        `eval_macro_f1`; the latter drives `load_best_model_at_end`."""

        def __init__(self, *args, gen_eval_records=None, gen_eval_cfg=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._gen_eval_records = gen_eval_records or []
            self._gen_eval_cfg = gen_eval_cfg or {}
            self._last_eval_preds: pd.DataFrame | None = None

        def _token_eval_loss(self, eval_dataset=None) -> float:
            dl = self.get_eval_dataloader(eval_dataset)
            model = self.model
            model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for batch in tqdm(dl, desc="eval token-loss", unit="batch", leave=False):
                    batch = self._prepare_inputs(batch)
                    losses.append(float(model(**batch).loss.item()))
            return statistics.fmean(losses) if losses else float("nan")

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            step = int(self.state.global_step)
            logger.info("[step %s] validation: token-loss + generation macro-F1 ...", step)
            eval_loss = self._token_eval_loss(eval_dataset)

            gcfg = self._gen_eval_cfg
            t0 = time.perf_counter()
            preds = predict_generations(
                self.model, self.processing_class, self._gen_eval_records,
                prompt_file=gcfg["prompt_file"],
                max_review_tokens=int(gcfg["max_review_tokens"]),
                batch_size=int(gcfg.get("batch_size", 16)),
                max_new_tokens=int(gcfg.get("max_new_tokens", 16)),
                desc=f"step {step} val-gen",
            )
            self._last_eval_preds = preds
            m = metrics_mod.compute(preds)

            metrics = {
                f"{metric_key_prefix}_loss": round(eval_loss, 6),
                f"{metric_key_prefix}_macro_f1": round(m["macro_f1"], 6),
                f"{metric_key_prefix}_accuracy_overall": m["accuracy_overall"],
                f"{metric_key_prefix}_invalid_output_rate": m["invalid_output_rate"],
                f"{metric_key_prefix}_gen_seconds": round(time.perf_counter() - t0, 2),
            }
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            return metrics

    return MacroF1Trainer


# --------------------------------------------------------------------------
# TrainingArguments (version-tolerant)
# --------------------------------------------------------------------------
def build_training_arguments(
    cfg: dict[str, Any], output_dir: str | Path, seed: int,
    n_train_examples: int, use_bf16: bool,
):
    from transformers import TrainingArguments

    tcfg, ecfg = cfg["train"], cfg["eval"]
    per_dev = int(tcfg["per_device_train_batch_size"])
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    epochs = float(tcfg["num_train_epochs"])
    max_steps = int(tcfg.get("max_steps", -1) or -1)

    steps_per_epoch = max(1, math.ceil(n_train_examples / (per_dev * grad_accum)))
    total_steps = max_steps if max_steps > 0 else max(1, int(steps_per_epoch * epochs))
    warmup_steps = max(1, int(float(tcfg.get("warmup_ratio", 0.0)) * total_steps))

    desired: dict[str, Any] = {
        "output_dir": str(output_dir),
        "seed": seed,
        "data_seed": seed,
        "per_device_train_batch_size": per_dev,
        "per_device_eval_batch_size": int(cfg["eval"].get("loss_batch_size", 4)),
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "max_steps": max_steps if max_steps > 0 else -1,
        "learning_rate": float(tcfg["learning_rate"]),
        "lr_scheduler_type": tcfg.get("lr_scheduler_type", "cosine"),
        "warmup_steps": warmup_steps,
        "weight_decay": float(tcfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(tcfg.get("max_grad_norm", 1.0)),
        "optim": tcfg.get("optim", "paged_adamw_8bit"),
        # Gradient checkpointing is enabled explicitly in qlora.prepare_for_training
        # (non-reentrant); do NOT let TrainingArguments turn it on again.
        "gradient_checkpointing": False,
        "bf16": bool(use_bf16),
        "fp16": not bool(use_bf16),
        "logging_strategy": "steps",
        "logging_steps": int(tcfg.get("logging_steps", 10)),
        "eval_strategy": "steps",
        "eval_steps": int(tcfg["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(tcfg["save_steps"]),
        "save_total_limit": int(tcfg.get("save_total_limit", 2)),
        "load_best_model_at_end": True,
        "metric_for_best_model": ecfg.get("metric_for_best_model", "eval_macro_f1"),
        "greater_is_better": bool(ecfg.get("greater_is_better", True)),
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "label_names": ["labels"],
        "eval_on_start": False,
        "disable_tqdm": False,
    }
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(set(desired) - allowed)
    if dropped:
        logger.info("TrainingArguments: dropping unsupported kwargs %s", dropped)
    return TrainingArguments(**{k: v for k, v in desired.items() if k in allowed})


# --------------------------------------------------------------------------
# Log-history -> tidy tables
# --------------------------------------------------------------------------
def tidy_log_history(log_history: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge Trainer's interleaved train/eval log records into one row per step."""
    by_step: dict[int, dict[str, Any]] = {}
    for rec in log_history:
        step = rec.get("step")
        if step is None:
            continue
        row = by_step.setdefault(step, {"step": step})
        for k, v in rec.items():
            if k in ("step",):
                continue
            if k == "loss":
                row["train_loss"] = v
            else:
                row[k] = v
    df = pd.DataFrame(sorted(by_step.values(), key=lambda r: r["step"]))
    return df


# --------------------------------------------------------------------------
# One seed, end to end
# --------------------------------------------------------------------------
def train_one_seed(
    cfg: dict[str, Any],
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    select_val_df: pd.DataFrame,
    seed_dir: Path,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Full QLoRA cycle for one seed. Writes every artefact under `seed_dir`
    and returns a compact result dict for cross-seed aggregation."""
    from transformers import set_seed as hf_set_seed

    seed_dir = resolve(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    hf_set_seed(seed)

    icfg = cfg["instruction"]
    ecfg = cfg["eval"]

    # --- base model + LoRA ------------------------------------------------
    base = qlora.load_base_model(cfg, hf_token=hf_token)
    with qlora.track_gpu_memory("prepare_for_training") as mem_prepare:
        model = qlora.prepare_for_training(
            base.model,
            use_gradient_checkpointing=bool(cfg["train"].get("gradient_checkpointing", True)),
        )
    with qlora.track_gpu_memory("attach_lora") as mem_lora:
        model, lora_config = qlora.attach_lora(model, cfg)
    use_bf16 = base.compute_dtype == "bfloat16"

    lora_summary = qlora.trainable_parameter_summary(model, lora_config)
    print(qlora.format_trainable_summary(lora_summary))
    write_json(lora_summary, seed_dir / "lora_summary.json")

    memory_summary = qlora.build_memory_summary(base, mem_prepare, mem_lora)
    print(qlora.format_memory_summary(memory_summary))
    write_json(memory_summary, seed_dir / "memory_summary.json")

    # --- datasets -------------------------------------------------------
    tokenizer = base.tokenizer
    logger.info("[seed %s] tokenising %d train rows ...", seed, len(train_df))
    train_ds, train_ds_stats = build_supervised_dataset(
        train_df, tokenizer, icfg, desc=f"seed {seed} tokenise-train"
    )
    # Generation macro-F1 (checkpoint selection) runs on the whole selection
    # subset; the token-level eval_loss is a cheaper monitoring signal, so it
    # uses a smaller slice to keep the eval pauses short on the 8B model.
    loss_max = int(ecfg.get("loss_max_samples") or len(select_val_df))
    select_ds, _ = build_supervised_dataset(
        select_val_df.head(loss_max), tokenizer, icfg, desc=f"seed {seed} tokenise-val"
    )
    gen_eval_records = select_val_df[["uid", "text", "label"]].to_dict("records")

    gen_eval_cfg = {
        "prompt_file": icfg["prompt_file"],
        "max_review_tokens": icfg["max_review_tokens"],
        "batch_size": ecfg.get("batch_size", 16),
        "max_new_tokens": ecfg.get("max_new_tokens", 16),
    }

    args = build_training_arguments(
        cfg, seed_dir / "checkpoints", seed, train_ds_stats["n_examples"], use_bf16
    )

    trainer_cls = _make_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=select_ds,
        data_collator=CausalDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        gen_eval_records=gen_eval_records,
        gen_eval_cfg=gen_eval_cfg,
    )

    # --- train --------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logger.info("[seed %s] training started ...", seed)
    train_result = trainer.train()
    train_seconds = time.perf_counter() - t0
    logger.info("[seed %s] training finished in %.1f s; running post-training scoring ...",
                seed, train_seconds)

    peak_alloc_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    peak_reserved_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    smi_mb = nvidia_smi_used_mb()

    # best checkpoint (by eval_macro_f1)
    best_ckpt = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    best_step = None
    if best_ckpt:
        try:
            best_step = int(Path(best_ckpt).name.split("-")[-1])
        except ValueError:
            best_step = None
    logger.info("Best checkpoint: %s (eval_macro_f1=%s, step=%s)",
                best_ckpt, best_metric, best_step)

    # --- logs --------------------------------------------------------
    logger.info("[seed %s] writing training logs ...", seed)
    log_df = tidy_log_history(trainer.state.log_history)
    log_df.to_csv(seed_dir / "training_log.csv", index=False)
    write_json(trainer.state.log_history, seed_dir / "training_log.json")

    # --- final scoring on FULL val + test with the best adapter ---------
    final_cap = ecfg.get("final_max_samples")
    val_eval_df = val_df if not final_cap else val_df.head(int(final_cap))
    test_eval_df = test_df if not final_cap else test_df.head(int(final_cap))

    splits: dict[str, dict[str, Any]] = {}
    for name, edf in (("val", val_eval_df), ("test", test_eval_df)):
        logger.info("[seed %s] final scoring on %s split (%d rows) ...", seed, name, len(edf))
        preds = predict_generations(
            model, tokenizer, edf[["uid", "text", "label"]].to_dict("records"),
            prompt_file=icfg["prompt_file"],
            max_review_tokens=int(icfg["max_review_tokens"]),
            batch_size=int(gen_eval_cfg["batch_size"]),
            max_new_tokens=int(gen_eval_cfg["max_new_tokens"]),
            desc=f"seed {seed} final-{name}",
        )
        m = metrics_mod.compute(preds)
        write_json(m, seed_dir / f"metrics_{name}_task3.json")
        preds.to_csv(seed_dir / f"predictions_{name}_task3.csv", index=False)
        metrics_mod.plot_confusion(preds, seed_dir, tag=f"{name}_task3")
        splits[name] = m
        logger.info("[seed %s] %s: macro_f1=%.4f acc_overall=%.4f invalid=%.4f",
                    seed, name, m["macro_f1"], m["accuracy_overall"], m["invalid_output_rate"])

    # --- save the selected adapter -------------------------------------
    logger.info("[seed %s] saving selected adapter + tokenizer ...", seed)
    adapter_dir = seed_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # --- per-seed meta + report --------------------------------------
    run_meta = {
        "seed": seed,
        "checkpoint": base.checkpoint,
        "revision": base.revision,
        "quantization": "nf4 + double-quant (4-bit)",
        "compute_dtype": base.compute_dtype,
        "lora": lora_summary,
        "train_dataset": train_ds_stats,
        "n_val_select": len(select_val_df),
        "best_checkpoint": best_ckpt,
        "best_step": best_step,
        "best_eval_macro_f1": best_metric,
        "train_seconds": round(train_seconds, 1),
        "train_runtime_reported": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "peak_gpu_memory": {
            "torch_peak_allocated_gb": peak_alloc_gb,
            "torch_peak_reserved_gb": peak_reserved_gb,
            "nvidia_smi_used_gb": round(smi_mb / 1024, 3) if smi_mb is not None else None,
            "total_vram_gb": snapshot().get("total_vram_gb"),
        },
        "setup_gpu_memory": memory_summary,
        "final_metrics": {
            "val": {k: splits["val"][k] for k in ("macro_f1", "accuracy", "accuracy_overall", "invalid_output_rate")},
            "test": {k: splits["test"][k] for k in ("macro_f1", "accuracy", "accuracy_overall", "invalid_output_rate")},
        },
    }
    logger.info("[seed %s] writing run_meta.json + results_task3.md ...", seed)
    write_json(run_meta, seed_dir / "run_meta.json")
    (seed_dir / "results_task3.md").write_text(
        _seed_report_md(run_meta, cfg), encoding="utf-8"
    )

    # --- per-seed diagrams -------------------------------------------
    logger.info("[seed %s] rendering per-seed diagrams ...", seed)
    plot_task3.plot_seed_curves(log_df, seed, seed_dir)

    # free VRAM before the next seed
    logger.info("[seed %s] releasing model + freeing VRAM ...", seed)
    del trainer, model, base
    torch.cuda.empty_cache()
    logger.info("[seed %s] done.", seed)

    return {
        "seed": seed,
        "best_step": best_step,
        "best_eval_macro_f1": best_metric,
        "train_seconds": round(train_seconds, 1),
        "peak_reserved_gb": peak_reserved_gb,
        "peak_allocated_gb": peak_alloc_gb,
        "val": run_meta["final_metrics"]["val"],
        "test": run_meta["final_metrics"]["test"],
        "log_csv": str(seed_dir / "training_log.csv"),
        "seed_dir": str(seed_dir),
    }


def _seed_report_md(meta: dict[str, Any], cfg: dict[str, Any]) -> str:
    lora = meta["lora"]
    pk = meta["peak_gpu_memory"]
    sm = meta.get("setup_gpu_memory", {})
    v, t = meta["final_metrics"]["val"], meta["final_metrics"]["test"]
    tm = " ".join(lora["target_modules"]) if isinstance(lora["target_modules"], list) else lora["target_modules"]
    return "\n".join([
        f"# Task 3 - QLoRA fine-tune (seed {meta['seed']})",
        "",
        f"- Base: `{meta['checkpoint']}` (revision `{meta['revision']}`)",
        f"- Quantization: {meta['quantization']}, compute dtype `{meta['compute_dtype']}`",
        "",
        "## LoRA adapter",
        "",
        f"- rank r: **{lora['lora_r']}**  |  alpha: **{lora['lora_alpha']}**  |  dropout: **{lora['lora_dropout']}**  |  bias: `{lora['lora_bias']}`",
        f"- target modules: `{tm}`",
        f"- trainable params: **{lora['trainable_parameters']:,}** / {lora['total_parameters']:,} total "
        f"(**{lora['trainable_percentage']}%**)",
        "",
        "## Checkpoint selection (validation macro-F1)",
        "",
        f"- Best checkpoint: `{meta['best_checkpoint']}` at step **{meta['best_step']}**",
        f"- Best validation macro-F1 (selection subset, n={meta['n_val_select']}): **{meta['best_eval_macro_f1']}**",
        "",
        "## Final metrics (best adapter, full splits)",
        "",
        "| Split | Macro-F1 | Accuracy (valid) | Accuracy (overall) | Invalid rate |",
        "|---|---|---|---|---|",
        f"| val  | {v['macro_f1']} | {v['accuracy']} | {v['accuracy_overall']} | {v['invalid_output_rate']} |",
        f"| test | {t['macro_f1']} | {t['accuracy']} | {t['accuracy_overall']} | {t['invalid_output_rate']} |",
        "",
        "## Resources",
        "",
        f"- Training time: **{meta['train_seconds']} s** ({round(meta['train_seconds'] / 60, 1)} min)",
        f"- Peak GPU memory: **{pk['torch_peak_reserved_gb']} GB** reserved "
        f"(torch), {pk['nvidia_smi_used_gb']} GB device-wide of {pk['total_vram_gb']} GB",
        f"- Setup VRAM: {sm.get('weights_vram_gb')} GB weights, {sm.get('peak_vram_gb')} GB peak on load, "
        f"{sm.get('reserved_vram_gb')} GB reserved; +{sm.get('prepare_added_gb')} GB k-bit/grad-ckpt, "
        f"+{sm.get('lora_added_gb')} GB LoRA",
        "",
    ])
