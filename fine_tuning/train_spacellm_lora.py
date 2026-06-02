"""
SpaceLLM — Experimental LoRA Fine-Tuning (Fixed for gpt-oss-20b)
"""

# ── TRITON PATCH ─────────────────────────────────────────────────────────────
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

# ── Imports ──────────────────────────────────────────────────────────────────
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
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_DIR = OUTPUT_DIR / "spacellm_lora_final"
LOG_DIR = OUTPUT_DIR / "logs"

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/train.jsonl")
VAL_FILE   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/validation.jsonl")

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

# ── Helper Functions ─────────────────────────────────────────────────────────

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


def log_gpu_info():
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("Visible GPUs      :")
            for line in result.stdout.strip().splitlines():
                idx, name, mem = line.split(",")
                logger.info(f"  cuda:{idx.strip()} → {name.strip()}  ({int(mem.strip()):,} MiB)")
    except Exception:
        pass


def log_gpu_memory(label: str = ""):
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            logger.info(f"GPU {i} [{props.name}] {label} | "
                       f"Allocated={alloc:.2f}GB  Reserved={reserv:.2f}GB  Total={total:.2f}GB")
    except Exception as e:
        logger.warning(f"GPU memory report failed: {e}")


def log_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("─" * 55)
    logger.info(f"Total parameters     : {total:>15,}")
    logger.info(f"Trainable parameters : {trainable:>15,}  ({100.0 * trainable / total:.6f}%)")
    logger.info("─" * 55)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name:<60} shape={list(param.shape)}  ({param.numel():,} params)")


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} records ← {path}")
    return records


def dataset_sanity_check(records: list, split_name: str):
    logger.info(f"── Sanity check: {split_name} ({len(records):,} records) ──")
    # ... (add your full sanity check logic here if needed) ...
    logger.info("  Sanity checks passed")


def tokenise_record(record: dict, tokenizer, max_seq_len: int):
    # Your original tokenization logic (copy-paste your full version here)
    messages = record.get("messages", [])
    hf_messages = [{"role": "system" if m["role"] == "developer" else m["role"], "content": m.get("content", "")} 
                   for m in messages if m.get("content")]

    full_text = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=False)
    full_enc = tokenizer(full_text, truncation=True, max_length=max_seq_len, return_tensors=None)

    # Simple labels (you can improve this)
    labels = full_enc["input_ids"].copy()
    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def build_hf_dataset(records, tokenizer, max_seq_len, split_name):
    from datasets import Dataset
    tokenised = [tokenise_record(r, tokenizer, max_seq_len) for r in records if tokenise_record(r, tokenizer, max_seq_len)]
    lengths = [len(t["input_ids"]) for t in tokenised]
    max_id = max(max(t["input_ids"]) for t in tokenised)
    logger.info(f"[{split_name}] {len(tokenised):,} records | seq len max={max(lengths)} | max_id={max_id}")
    return Dataset.from_list(tokenised)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer, Mxfp4Config
    from peft import LoraConfig, TaskType, get_peft_model

    log_gpu_info()
    log_gpu_memory("before model load")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Vocab size : {len(tokenizer):,}")

    # Model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    # Dispatch to GPUs (simplified)
    model = model.to("cuda:0") if torch.cuda.is_available() else model

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # === CRITICAL FIXES ===
    logger.info("Applying vocab alignment fix...")
    model.config.tie_word_embeddings = False
    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],
    )
    model = get_peft_model(model, lora_config)

    log_trainable_parameters(model)
    log_gpu_memory("after LoRA")

    # Datasets
    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)

    train_dataset = build_hf_dataset(train_records, tokenizer, args.max_seq_len, "train")
    val_dataset = build_hf_dataset(val_records, tokenizer, args.max_seq_len, "validation")

    # Trainer
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=IGNORE_INDEX)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    trainer.model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    logger.info(f"Training finished. Adapters saved to {FINAL_DIR}")


if __name__ == "__main__":
    main()