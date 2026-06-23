"""
SpaceLLM Fresh Adapter Training
================================
Load GPT-OSS-20B base model and train a fresh LoRA adapter on combined dataset.

No v1 adapter involved. Simple, direct training pipeline.

Usage:
    python train_spacellm_fresh.py \
        --train_file combined_dataset.json \
        --output_dir ./spacellm_adapter \
        [--epochs 3] [--lr 2e-4] [--lora_r 64]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SpaceLLM.Fresh")

IGNORE_INDEX = -100


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fresh SpaceLLM adapter on combined dataset")
    p.add_argument("--base_model", default="openai/gpt-oss-20b",
                   help="HF id or local path of the base model")
    p.add_argument("--train_file", required=True,
                   help="JSON file with training examples (messages format)")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the trained LoRA adapter")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # Training hyperparams
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Max sequence length")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size per device")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Warmup ratio")
    p.add_argument("--max_grad_norm", type=float, default=0.3,
                   help="Max gradient norm")
    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=128,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.1,
                   help="LoRA dropout")
    p.add_argument("--target_modules", default="lm_head",
                   help="Comma-separated list of module names for LoRA")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


# ── GPU helpers ───────────────────────────────────────────────────────────────
def log_gpu(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        a = torch.cuda.memory_allocated(i) / 1024**3
        r = torch.cuda.memory_reserved(i) / 1024**3
        t = p.total_memory / 1024**3
        log.info(f"  GPU{i} [{p.name}] {label} | alloc={a:.1f}GB res={r:.1f}GB total={t:.1f}GB")


def log_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    log.info(f"  Trainable: {trainable:,} / {total:,}  ({pct:.4f}%)")
    for n, p in model.named_parameters():
        if p.requires_grad:
            log.info(f"    {n}  shape={list(p.shape)}  ({p.numel():,})")


# ── Device-aware cross-entropy loss ──────────────────────────────────────────
def _device_aware_ce_loss(logits, labels, vocab_size=None, **kwargs):
    """Handle device mismatches (MoE models across GPUs)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if shift_labels.device != shift_logits.device:
        shift_labels = shift_labels.to(shift_logits.device)

    V = shift_logits.size(-1)
    oob = (shift_labels != IGNORE_INDEX) & ((shift_labels < 0) | (shift_labels >= V))
    if oob.any():
        shift_labels = shift_labels.clone()
        shift_labels[oob] = IGNORE_INDEX

    loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
        shift_logits.view(-1, V),
        shift_labels.view(-1).long(),
    )
    return loss.to("cuda:0") if torch.cuda.is_available() else loss


def inject_loss(model, label: str = "") -> None:
    """Patch loss_function on model."""
    tag = f" ({label})" if label else ""
    for obj in [model, getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None),
                *list(model.children())]:
        if obj is not None and hasattr(obj, "loss_function"):
            if getattr(obj, "loss_function") is not _device_aware_ce_loss:
                obj.loss_function = _device_aware_ce_loss
                log.info(f"  ✅ loss_function patched on {type(obj).__name__}{tag}")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_examples(train_file: Path) -> list[dict]:
    """Load training JSON in messages format."""
    raw = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {train_file}")

    out, skipped = [], 0
    for rec in raw:
        msgs = rec.get("messages", [])
        asst = [m for m in msgs if m.get("role") == "assistant"]
        if not asst or not (asst[-1].get("content") or "").strip():
            skipped += 1
            continue
        out.append(rec)

    if skipped:
        log.warning(f"  Skipped {skipped}/{len(raw)} records (empty/missing content)")
    if not out:
        raise ValueError("No usable training examples after filtering.")
    log.info(f"  Loaded {len(out)} usable example(s)")
    return out


def tokenise(record: dict, tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize one record. Mask prompt with IGNORE_INDEX."""
    msgs = record.get("messages", [])
    hf_msgs = [
        {"role": "system" if m["role"] == "developer" else m["role"],
         "content": (m.get("content") or "").strip()}
        for m in msgs if (m.get("content") or "").strip()
    ]
    if not hf_msgs:
        return None

    try:
        full_text = tokenizer.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        log.debug(f"  apply_chat_template failed: {e}")
        return None

    full_enc = tokenizer(full_text, truncation=True, max_length=max_seq_len,
                         padding=False, return_tensors=None)
    input_ids = full_enc["input_ids"]
    if len(input_ids) < 4:
        return None

    # Find prefix boundary (everything except last assistant turn)
    prefix_msgs = [m for m in hf_msgs if m["role"] != "assistant"]
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None
    
    prefix_len = len(tokenizer(prefix_text, truncation=True, max_length=max_seq_len,
                                padding=False, return_tensors=None)["input_ids"])

    if prefix_len >= len(input_ids):
        return None

    labels = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels = labels[:len(input_ids)]
    n_active = sum(1 for l in labels if l != IGNORE_INDEX)
    if n_active < 4:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def build_dataset(tokenizer, examples: list[dict], max_seq_len: int):
    """Build training dataset."""
    from datasets import Dataset

    rows, skipped = [], 0
    for rec in examples:
        r = tokenise(rec, tokenizer, max_seq_len)
        if r is None:
            skipped += 1
            continue
        if all(l == IGNORE_INDEX for l in r["labels"]):
            skipped += 1
            continue
        rows.append(r)

    if skipped:
        log.warning(f"  Tokenisation skipped {skipped} examples")
    if not rows:
        raise ValueError("All examples collapsed after tokenisation.")

    lengths = [len(r["input_ids"]) for r in rows]
    log.info(f"  Dataset: {len(rows)} rows | "
             f"seq_len min={min(lengths)} mean={sum(lengths)/len(lengths):.0f} max={max(lengths)}")
    return Dataset.from_list(rows)


# ── NaN guard callback ────────────────────────────────────────────────────────
def make_nan_guard():
    from transformers import TrainerCallback

    class _NaNGuard(TrainerCallback):
        def __init__(self):
            self.triggered = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                log.error(f"NaN/Inf loss at step {state.global_step} — stopping.")
                self.triggered = True
                control.should_training_stop = True
            return control

    return _NaNGuard()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_file = Path(args.train_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    log.info("=" * 70)
    log.info("  SpaceLLM Fresh Adapter Training")
    log.info(f"  base_model : {args.base_model}")
    log.info(f"  train_file : {train_file}")
    log.info(f"  output_dir : {output_dir}")
    log.info(f"  LoRA r={args.lora_r}  alpha={args.lora_alpha}  target={args.target_modules}")
    log.info(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
             f"grad_accum={args.grad_accum}")
    log.info("=" * 70)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Mxfp4Config,
        TrainingArguments, DataCollatorForSeq2Seq, Trainer,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    # ── [1] Load training examples ────────────────────────────────────────
    log.info("\n[1/7] Loading training examples ...")
    examples = load_examples(train_file)

    # ── [2] Load tokenizer ───────────────────────────────────────────────
    log.info(f"\n[2/7] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"
    log.info(f"  tokenizer len={len(tokenizer):,}  "
             f"chat_template={'found' if tokenizer.chat_template else 'MISSING'}")

    # ── [3] Load base model ──────────────────────────────────────────────
    log.info(f"\n[3/7] Loading base model (MXFP4→BF16): {args.base_model}")
    log_gpu("pre-load")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        token=args.hf_token,
    )
    log.info(f"  Loaded in {time.time()-t0:.1f}s | dtype={next(model.parameters()).dtype}")
    model.config.use_cache = False

    # ── [4] Inject loss (pre-PEFT) ────────────────────────────────────────
    log.info("\n[4/7] Injecting device-aware CE loss ...")
    inject_loss(model, "pre-PEFT")

    # ── [5] Attach fresh LoRA ────────────────────────────────────────────
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_cfg)
    log.info(f"\n[5/7] Fresh LoRA attached")
    log.info(f"  r={args.lora_r}  alpha={args.lora_alpha}  target={target_modules}")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("  ✅ enable_input_require_grads + gradient_checkpointing")

    # Freeze non-LoRA params
    log.info("  Freezing non-LoRA parameters ...")
    frozen, lora_count = 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            lora_count += 1
        else:
            param.requires_grad_(False)
            frozen += 1

    leaked = [(n, p.shape) for n, p in model.named_parameters()
              if p.requires_grad and "lora_" not in n]
    if leaked:
        log.error("  ❌ Non-LoRA params still trainable:")
        for n, s in leaked:
            log.error(f"     {n}  {s}")
        return 1

    log.info(f"  Frozen={frozen}  LoRA trainable={lora_count}  ✅")
    inject_loss(model, "post-PEFT")
    log_trainable(model)
    log_gpu("after LoRA init (CPU)")

    # ── [6] GPU dispatch ─────────────────────────────────────────────────
    log.info("\n[6/7] Dispatching model to GPU ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        no_split = list({
            type(m).__name__
            for m in model.modules()
            if (isinstance(m, nn.Module) and type(m) is not nn.Module
                and ("layer" in type(m).__name__.lower() 
                     or "block" in type(m).__name__.lower())
                and sum(p.numel() for p in m.parameters()) > 1_000_000)
        })

        max_memory = {
            i: f"{max(0, int((torch.cuda.mem_get_info(i)[0] - 4*1024**3) / 1024**3))}GiB"
            for i in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = "80GiB"

        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)
        log.info(f"  GPU dispatch done in {time.time()-t1:.1f}s")
        if hasattr(model, "hf_device_map"):
            from collections import Counter
            for dev, cnt in sorted(Counter(str(v) for v in model.hf_device_map.values()).items()):
                log.info(f"    {dev}: {cnt} layer(s)")
    except Exception as e:
        log.warning(f"  dispatch_model failed ({e}) — falling back to cuda:0")
        model = model.to("cuda:0")

    log_gpu("after dispatch")
    inject_loss(model, "post-dispatch")

    # ── Build dataset ────────────────────────────────────────────────────
    log.info("\n  Tokenising training examples ...")
    train_dataset = build_dataset(tokenizer, examples, args.max_seq_len)

    # ── TrainingArguments ────────────────────────────────────────────────
    MAX_GRAD_NORM = args.max_grad_norm
    training_args = TrainingArguments(
        output_dir=str(output_dir / "_ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
    )
    nan_guard = make_nan_guard()

    # ── Custom Trainer ───────────────────────────────────────────────────
    class SpaceLLMTrainer(Trainer):
        def _lm_device(self):
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            dev = self._lm_device()
            if dev and "labels" in inputs and inputs["labels"].device != dev:
                inputs["labels"] = inputs["labels"].to(dev)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = (super().training_step(model, inputs, num_items_in_batch)
                    if num_items_in_batch is not None
                    else super().training_step(model, inputs))
            trainable = [p for p in model.parameters()
                        if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
            return loss

    trainer = SpaceLLMTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[nan_guard],
    )

    inject_loss(trainer.model, "post-Trainer")
    log.info(f"  lm_head device: {trainer._lm_device()}")

    # ── [7] Train ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info(f"  Training | {len(train_dataset)} examples | {args.epochs} epochs")
    log.info(f"  LoRA: r={args.lora_r}  alpha={args.lora_alpha}  target={target_modules}")
    log.info("=" * 70 + "\n")

    t_start = time.time()
    try:
        trainer.train()
    except KeyboardInterrupt:
        log.warning("Interrupted — saving current state ...")
        interrupted = output_dir / "interrupted"
        trainer.save_model(str(interrupted))
        tokenizer.save_pretrained(str(interrupted))
        log.info(f"Partial save → {interrupted}")
        return 0
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        return 1

    elapsed = time.time() - t_start
    log.info(f"\n  Training complete in {elapsed/60:.1f} min")

    if nan_guard.triggered:
        log.error("  NaN/Inf detected — NOT saving adapter.")
        return 1

    # ── Save adapter ─────────────────────────────────────────────────────
    log.info(f"\n  Saving adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model_name": "SpaceLLM_fresh",
        "base_model": args.base_model,
        "strategy": "fresh_lora_only",
        "train_file": str(train_file),
        "examples_used": len(train_dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("\n" + "=" * 70)
    log.info("  SpaceLLM Fresh Adapter — Complete ✅")
    log.info("=" * 70)
    log.info(f"  Adapter saved  → {output_dir}")
    log.info(f"  To load:")
    log.info(f"    from transformers import AutoModelForCausalLM")
    log.info(f"    from peft import PeftModel")
    log.info(f"    base = AutoModelForCausalLM.from_pretrained('{args.base_model}', ...)")
    log.info(f"    model = PeftModel.from_pretrained(base, '{output_dir}')")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

    
