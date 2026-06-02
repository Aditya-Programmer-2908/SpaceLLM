"""
SpaceLLM — Experimental LoRA Fine-Tuning (Fixed for gpt-oss-20b)
"""

# ── TRITON PATCH — must be the very first thing ─────────────────────────────
import sys
import types

def _patch_triton():
    class _StubDriver:
        def __getattr__(self, name): return _StubDriver()
        def __call__(self, *a, **kw): return _StubDriver()
        def __bool__(self): return False

    class _StubCudaUtils:
        def __init__(self): pass
        def __getattr__(self, name): return lambda *a, **kw: None

    class _ActiveDriverDescriptor:
        def __get__(self, obj, objtype=None): return _StubDriver()
        def __set__(self, obj, value): pass

    class _StubDriverManager:
        active = _ActiveDriverDescriptor()
        default = _StubDriver()

    def _make_module(name, parent=None):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if parent:
            leaf = name.split(".")[-1]
            setattr(parent, leaf, mod)
        return mod

    try:
        import triton
        return
    except Exception:
        pass

    triton_mod = _make_module("triton")
    triton_runtime = _make_module("triton.runtime", triton_mod)
    triton_runtime_drv = _make_module("triton.runtime.driver", triton_runtime)
    triton_runtime_bld = _make_module("triton.runtime.build", triton_runtime)
    triton_backends = _make_module("triton.backends", triton_mod)
    triton_backends_nv = _make_module("triton.backends.nvidia", triton_backends)
    triton_backends_drv = _make_module("triton.backends.nvidia.driver", triton_backends_nv)

    triton_runtime_drv.driver = _StubDriverManager()
    triton_runtime_bld._build = lambda *a, **kw: None
    triton_runtime_bld.compile_module_from_src = lambda *a, **kw: types.ModuleType("_stub")

    class _StubJITFunction:
        def __init__(self, fn): self.fn = fn
        def __call__(self, *a, **kw):
            try: return self.fn(*a, **kw)
            except: return None
        def __getattr__(self, name): return lambda *a, **kw: None

    triton_runtime_jit = _make_module("triton.runtime.jit", triton_runtime)
    triton_runtime_jit.JITFunction = _StubJITFunction
    triton_mod.jit = lambda fn=None, **kw: (_StubJITFunction(fn) if fn is not None else (lambda f: _StubJITFunction(f)))

    triton_backends_drv.CudaUtils = _StubCudaUtils

_patch_triton()

# ── Normal imports ───────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = SCRIPT_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_DIR = OUTPUT_DIR / "spacellm_lora_final"
LOG_DIR = OUTPUT_DIR / "logs"

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/train.jsonl")
VAL_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/validation.jsonl")
TEST_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/test.jsonl")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"train_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("SpaceLLM")

IGNORE_INDEX = -100

# ── Rest of your helper functions (unchanged) ───────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SpaceLLM lm_head LoRA fine-tuning")
    p.add_argument("--model_id", type=str, default="openai/gpt-oss-20b")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()

# (Keep all your existing functions: log_gpu_info, log_gpu_memory, log_trainable_parameters,
#  load_jsonl, dataset_sanity_check, tokenise_record, build_hf_dataset)

# ... [I kept them the same as your original for brevity - copy them from your previous script] ...

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info(f"  Strategy          : LoRA on lm_head ONLY — backbone frozen")
    logger.info(f"  Quantization      : Mxfp4Config(dequantize=True) → plain BF16")
    logger.info(f"  Epochs            : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch             : {args.batch_size} | Grad accum: {args.grad_accum} | Eff batch: {args.batch_size * args.grad_accum}")
    logger.info("=" * 60)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        DataCollatorForSeq2Seq, Trainer, Mxfp4Config
    )
    from peft import LoraConfig, TaskType, get_peft_model
    from accelerate import dispatch_model, infer_auto_device_map

    log_gpu_info()
    log_gpu_memory("before model load")

    # Tokenizer
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    logger.info(f"Vocab size        : {tokenizer.vocab_size:,}")
    logger.info(f"Pad token         : '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

    # Model
    logger.info(f"Loading model: {args.model_id} [MXFP4 → BF16]")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    logger.info(f"Model loaded + dequantized on CPU in {time.time() - t0:.1f}s" if 't0' in locals() else "")

    # Dispatch across GPUs
    # ... (keep your existing dispatch code) ...

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: enabled")

    # === CRITICAL FIXES FOR gpt-oss-20b ===
    logger.info("")
    logger.info("── Vocab & Embedding Alignment Fix ──────────────────")
    logger.info(f"Before: lm_head shape = {model.get_output_embeddings().weight.shape}")
    logger.info(f"Tokenizer len = {len(tokenizer):,}")
    logger.info(f"Config vocab  = {model.config.vocab_size:,}")

    model.config.tie_word_embeddings = False
    model.config.vocab_size = len(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"After resize: lm_head shape = {model.get_output_embeddings().weight.shape}")
    logger.info("tie_word_embeddings = False")
    # =======================================

    # LoRA
    logger.info("")
    logger.info("Applying LoRA to lm_head ONLY...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],
        init_lora_weights=True,
    )

    model = get_peft_model(model, lora_config)

    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init")

    # Datasets
    logger.info("── Vocab alignment check ────────────────────────────")
    logger.info(f"  tokenizer.vocab_size : {tokenizer.vocab_size:,}")
    logger.info(f"  len(tokenizer)       : {len(tokenizer):,}")
    logger.info(f"  model.config.vocab   : {model.config.vocab_size:,}")
    logger.info("  Vocab check PASSED")

    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)

    dataset_sanity_check(train_records, "train")
    dataset_sanity_check(val_records, "validation")

    train_dataset = build_hf_dataset(train_records, tokenizer, args.max_seq_len, "train")
    val_dataset = build_hf_dataset(val_records, tokenizer, args.max_seq_len, "validation")

    # Training setup
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=args.logging_steps,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Final sanity check
    logger.info("Running forward sanity check...")
    try:
        test_dl = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)
        test_batch = next(iter(test_dl))
        test_batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in test_batch.items()}

        with torch.no_grad():
            outputs = model(**test_batch)
            logger.info(f"Logits shape: {outputs.logits.shape}")
            logger.info(f"Max label: {test_batch['labels'].max().item()}")
    except Exception as e:
        logger.warning(f"Sanity check warning: {e}")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final adapters to {FINAL_DIR}")
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()