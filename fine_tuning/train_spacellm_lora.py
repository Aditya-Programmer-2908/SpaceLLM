"""
SpaceLLM — Experimental LoRA Fine-Tuning
==========================================
Model     : openai/gpt-oss-20b  (MoE, MXFP4 quantized checkpoint)
Phase     : Experimentation
Strategy  : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method    : Standard BF16 LoRA — NOT QLoRA, no bitsandbytes

Launch:
    export CUDA_VISIBLE_DEVICES=1,2
    python fine_tuning/train_spacellm_lora.py

Output layout:
  SpaceLLM/fine_tuning/outputs/
  ├── checkpoints/
  ├── spacellm_lora_final/     ← LOAD THIS for inference / evaluation
  └── logs/

To load after training:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    export CUDA_VISIBLE_DEVICES=1,2
    base  = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                torch_dtype=torch.bfloat16,
                device_map="auto")
    model = PeftModel.from_pretrained(base, "./outputs/spacellm_lora_final")
    tok   = AutoTokenizer.from_pretrained("./outputs/spacellm_lora_final")
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = SCRIPT_DIR / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FINAL_DIR  = OUTPUT_DIR / "spacellm_lora_final"
LOG_DIR    = OUTPUT_DIR / "logs"

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Fixed dataset paths ───────────────────────────────────────────────────────

TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/train.jsonl")
VAL_FILE   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/validation.jsonl")
TEST_FILE  = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/test.jsonl")

# ── Logging ───────────────────────────────────────────────────────────────────

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"train_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("SpaceLLM")

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpaceLLM lm_head LoRA fine-tuning")
    p.add_argument("--model_id",               type=str,   default="openai/gpt-oss-20b")
    p.add_argument("--epochs",                 type=int,   default=3)
    p.add_argument("--batch_size",             type=int,   default=1)
    p.add_argument("--grad_accum",             type=int,   default=16)
    p.add_argument("--lr",                     type=float, default=2e-4)
    p.add_argument("--max_seq_len",            type=int,   default=2048)
    p.add_argument("--warmup_steps",           type=int,   default=100)
    p.add_argument("--save_steps",             type=int,   default=500)
    p.add_argument("--eval_steps",             type=int,   default=500)
    p.add_argument("--logging_steps",          type=int,   default=20)
    p.add_argument("--save_total_limit",       type=int,   default=2)
    p.add_argument("--resume_from_checkpoint", type=str,   default=None)
    return p.parse_args()

# ── GPU diagnostics ───────────────────────────────────────────────────────────

def log_gpu_memory(label: str = ""):
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("No CUDA device — running on CPU")
            return
        for i in range(torch.cuda.device_count()):
            props  = torch.cuda.get_device_properties(i)
            alloc  = torch.cuda.memory_allocated(i)  / 1024**3
            reserv = torch.cuda.memory_reserved(i)   / 1024**3
            total  = props.total_memory               / 1024**3
            logger.info(
                f"GPU {i} [{props.name}] {label} | "
                f"Allocated={alloc:.2f}GB  Reserved={reserv:.2f}GB  Total={total:.2f}GB"
            )
    except Exception as e:
        logger.warning(f"GPU memory report failed: {e}")


def log_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    pct       = 100.0 * trainable / total if total else 0.0
    logger.info("─" * 55)
    logger.info(f"Total parameters     : {total:>15,}")
    logger.info(f"Trainable parameters : {trainable:>15,}  ({pct:.6f}%)")
    logger.info(f"Frozen parameters    : {frozen:>15,}")
    logger.info("─" * 55)
    logger.info("Trainable layers (LoRA weights only):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"  {name:<60}  "
                f"shape={str(list(param.shape)):<20}  "
                f"({param.numel():,} params)"
            )

# ── JSONL loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_no} in {path.name}: {e}")
    logger.info(f"Loaded {len(records):,} records  ←  {path}")
    return records

# ── Dataset sanity check ──────────────────────────────────────────────────────

def dataset_sanity_check(records: list, split_name: str):
    logger.info(f"── Sanity check: {split_name} ({len(records):,} records) ──")
    issues, no_assistant = 0, 0
    org_dist, diff_dist  = defaultdict(int), defaultdict(int)
    chain_ids            = set()

    for i, r in enumerate(records):
        for field in ("sample_id", "source_id", "mission_name", "organization",
                      "aspect", "difficulty", "chain_id", "messages"):
            if field not in r:
                logger.warning(f"  Record {i}: missing '{field}'")
                issues += 1
                break
        roles = [m.get("role") for m in r.get("messages", [])]
        if "assistant" not in roles:
            no_assistant += 1
        org_dist[r.get("organization", "?")]  += 1
        diff_dist[r.get("difficulty",   "?")] += 1
        chain_ids.add(r.get("chain_id", ""))

    logger.info(f"  Unique chains     : {len(chain_ids)}")
    logger.info(f"  Structural issues : {issues}")
    logger.info(f"  No assistant turn : {no_assistant}")
    logger.info(f"  Organizations     : {dict(sorted(org_dist.items()))}")
    logger.info(f"  Difficulty        : {dict(sorted(diff_dist.items()))}")

# ── Tokenisation with assistant-only loss masking ─────────────────────────────

IGNORE_INDEX = -100


def tokenise_record(record: dict, tokenizer, max_seq_len: int):
    messages = record.get("messages", [])

    hf_messages = []
    for msg in messages:
        role    = "system" if msg["role"] == "developer" else msg["role"]
        content = msg.get("content", "").strip()
        if content:
            hf_messages.append({"role": role, "content": content})

    if not hf_messages:
        return None

    try:
        full_text = tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e} — skipping")
        return None

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )
    input_ids = full_enc["input_ids"]

    if len(input_ids) < 4:
        return None

    prefix_msgs = [m for m in hf_messages if m["role"] != "assistant"]
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None

    prefix_enc = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )
    prefix_len = len(prefix_enc["input_ids"])

    labels = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels = labels[:len(input_ids)]

    if all(lbl == IGNORE_INDEX for lbl in labels):
        return None

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
    }


def build_hf_dataset(records: list, tokenizer, max_seq_len: int, split_name: str):
    from datasets import Dataset

    tokenised, skipped = [], 0
    for record in records:
        result = tokenise_record(record, tokenizer, max_seq_len)
        if result is None:
            skipped += 1
        else:
            tokenised.append(result)

    if skipped:
        logger.warning(f"[{split_name}] Skipped {skipped} records")
    if not tokenised:
        logger.error(f"[{split_name}] Zero usable records — aborting")
        sys.exit(1)

    lengths = [len(t["input_ids"]) for t in tokenised]
    logger.info(
        f"[{split_name}] {len(tokenised):,} records | "
        f"seq len  min={min(lengths)}  max={max(lengths)}  "
        f"mean={sum(lengths)/len(lengths):.0f}"
    )
    return Dataset.from_list(tokenised)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Detect available GPUs from CUDA_VISIBLE_DEVICES ──────────────────
    # With CUDA_VISIBLE_DEVICES=1,2 → PyTorch sees them as cuda:0 and cuda:1
    # We load the model onto cuda:0 (physical GPU 1) and let device_map
    # spill overflow layers to cuda:1 (physical GPU 2) if needed
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info(f"  Strategy          : LoRA on lm_head ONLY — backbone frozen")
    logger.info(f"  Epochs            : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch             : {args.batch_size}  |  Grad accum: {args.grad_accum}"
                f"  |  Eff batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  CUDA_VISIBLE_DEVS : {visible if visible else 'all'}")
    logger.info(f"  Log               : {LOG_FILE}")
    logger.info("=" * 60)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            DataCollatorForSeq2Seq,
            Trainer,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: uv pip install -r requirements.txt")
        sys.exit(1)

    # Detect how many GPUs are visible
    n_gpus = torch.cuda.device_count()
    logger.info(f"Visible GPUs      : {n_gpus}")
    for i in range(n_gpus):
        logger.info(f"  cuda:{i} → {torch.cuda.get_device_name(i)}")

    # ── Dataset file check ────────────────────────────────────────────────
    logger.info("")
    logger.info("── Dataset files ────────────────────────────────────")
    for label, path in [("Train", TRAIN_FILE), ("Val", VAL_FILE), ("Test", TEST_FILE)]:
        if path.exists():
            kb = path.stat().st_size / 1024
            logger.info(f"  {label:<6}: OK  ({kb:.1f} KB)  →  {path}")
        else:
            logger.info(f"  {label:<6}: NOT FOUND  →  {path}")

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        logger.error("Train or validation file missing — cannot proceed")
        sys.exit(1)

    log_gpu_memory("before model load")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"Loading tokenizer: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token")

    tokenizer.padding_side = "right"
    logger.info(f"Vocab size        : {tokenizer.vocab_size:,}")
    logger.info(f"Pad token         : '{tokenizer.pad_token}'  (id={tokenizer.pad_token_id})")

    if tokenizer.chat_template is not None:
        logger.info("Chat template     : found (harmony format)")
    else:
        logger.warning("Chat template     : NOT FOUND")

    # ── Model loading strategy ────────────────────────────────────────────
    # With CUDA_VISIBLE_DEVICES=1,2:
    #   - PyTorch remaps → cuda:0 = physical GPU1, cuda:1 = physical GPU2
    #   - We build a max_memory dict to control how model layers are split
    #   - lm_head is kept on cuda:0 so LoRA backprop stays on one device
    #   - Transformer layers spill to cuda:1 if needed
    logger.info("")
    logger.info(f"Loading model: {args.model_id}  [BF16, {n_gpus} GPU(s)]")

    if n_gpus >= 2:
        # Split: put lm_head (last layer) on cuda:0 by leaving it most memory
        # Give cuda:1 enough for transformer body, cuda:0 for lm_head + LoRA
        gpu0_mem = torch.cuda.get_device_properties(0).total_memory
        gpu1_mem = torch.cuda.get_device_properties(1).total_memory
        # Reserve 2GB headroom on each
        max_memory = {
            0: f"{int((gpu0_mem / 1024**3) - 2)}GiB",
            1: f"{int((gpu1_mem / 1024**3) - 2)}GiB",
        }
        logger.info(f"Max memory map    : {max_memory}")
        device_map_arg = "auto"
    else:
        # Single GPU — put everything on cuda:0
        max_memory = None
        device_map_arg = "cuda:0"

    logger.info(f"device_map        : {device_map_arg}")

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map_arg,
            max_memory=max_memory,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,   # handles MXFP4 → BF16 conversion
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        sys.exit(1)

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    logger.info(f"Model dtype       : {next(model.parameters()).dtype}")
    if hasattr(model, "hf_device_map"):
        # Show layer → device assignment
        device_counts = defaultdict(int)
        for layer, dev in model.hf_device_map.items():
            device_counts[str(dev)] += 1
        logger.info("Layer distribution:")
        for dev, count in sorted(device_counts.items()):
            logger.info(f"  {dev} : {count} layer(s)")
        # Log specifically where lm_head landed — critical for LoRA
        lm_head_device = model.hf_device_map.get("lm_head", "unknown")
        logger.info(f"lm_head device    : {lm_head_device}  ← LoRA will be on this device")

    log_gpu_memory("after model load")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: enabled")

    # ── LoRA on lm_head ONLY ──────────────────────────────────────────────
    # LoRA adapters are automatically placed on the same device as lm_head
    # With our max_memory setup, lm_head stays on cuda:0
    logger.info("")
    logger.info("Applying LoRA to lm_head ONLY — backbone stays frozen ...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],
    )
    model = get_peft_model(model, lora_config)

    # Verify LoRA weights are on same device as lm_head — prevents backprop error
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"LoRA weight '{name}' on device: {param.device}")

    logger.info("")
    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init")

    # ── Datasets ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Loading datasets ─────────────────────────────────")
    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)

    dataset_sanity_check(train_records, "train")
    dataset_sanity_check(val_records,   "validation")

    logger.info("")
    logger.info("── Tokenising ───────────────────────────────────────")
    train_dataset = build_hf_dataset(train_records, tokenizer, args.max_seq_len, "train")
    val_dataset   = build_hf_dataset(val_records,   tokenizer, args.max_seq_len, "validation")

    # ── Training arguments ────────────────────────────────────────────────
    logger.info("")
    logger.info("── Training configuration ───────────────────────────")
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
        fp16=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    for k, v in {
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  args.batch_size * args.grad_accum,
        "warmup_steps":     args.warmup_steps,
        "optimizer":        "adamw_torch",
        "scheduler":        "cosine",
        "bf16":             True,
        "max_seq_len":      args.max_seq_len,
        "visible_gpus":     visible if visible else "all",
        "n_gpus_used":      n_gpus,
    }.items():
        logger.info(f"  {k:<25}: {v}")

    # ── Data collator ─────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Starting training ...")
    logger.info("=" * 60)

    resume = args.resume_from_checkpoint
    if resume:
        if Path(resume).exists():
            logger.info(f"Resuming from: {resume}")
        else:
            logger.warning(f"Checkpoint not found: {resume} — starting fresh")
            resume = None

    t_start = time.time()
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving current state ...")
        interrupted_dir = CKPT_DIR / "interrupted"
        trainer.save_model(str(interrupted_dir))
        tokenizer.save_pretrained(str(interrupted_dir))
        logger.info(f"Saved to: {interrupted_dir}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min  ({elapsed:.0f}s)")

    # ── Metrics ───────────────────────────────────────────────────────────
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    train_metrics_file = LOG_DIR / f"train_metrics_{RUN_ID}.json"
    with train_metrics_file.open("w") as f:
        json.dump(train_metrics, f, indent=2)
    logger.info(f"Train metrics → {train_metrics_file}")

    logger.info("Running final validation evaluation ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)

    eval_metrics_file = LOG_DIR / f"eval_metrics_{RUN_ID}.json"
    with eval_metrics_file.open("w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Eval metrics  → {eval_metrics_file}")

    # ── Save final LoRA adapters + tokenizer ──────────────────────────────
    logger.info("")
    logger.info(f"Saving final LoRA adapters → {FINAL_DIR}")
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    adapter_info = {
        "run_id":                RUN_ID,
        "base_model":            args.model_id,
        "strategy":              "LoRA on lm_head ONLY — backbone frozen — BF16",
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_dropout":          0.05,
        "target_modules":        ["lm_head"],
        "epochs":                args.epochs,
        "learning_rate":         args.lr,
        "batch_size":            args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size":  args.batch_size * args.grad_accum,
        "max_seq_len":           args.max_seq_len,
        "train_samples":         len(train_dataset),
        "val_samples":           len(val_dataset),
        "train_metrics":         train_metrics,
        "eval_metrics":          eval_metrics,
        "final_adapter_dir":     str(FINAL_DIR),
        "checkpoints_dir":       str(CKPT_DIR),
        "log_file":              str(LOG_FILE),
    }
    with (FINAL_DIR / "adapter_info.json").open("w") as f:
        json.dump(adapter_info, f, indent=2)

    log_gpu_memory("after training")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  SpaceLLM lm_head LoRA — Training Complete")
    logger.info("=" * 60)
    logger.info(f"  Final adapters   →  {FINAL_DIR}")
    logger.info(f"  Checkpoints      →  {CKPT_DIR}")
    logger.info(f"  Logs             →  {LOG_DIR}")
    logger.info("")
    logger.info("  To load for evaluation:")
    logger.info("  export CUDA_VISIBLE_DEVICES=1,2")
    logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info("  from peft import PeftModel")
    logger.info("  import torch")
    logger.info(f"  base  = AutoModelForCausalLM.from_pretrained(")
    logger.info(f"              '{args.model_id}',")
    logger.info(f"              torch_dtype=torch.bfloat16, device_map='auto')")
    logger.info(f"  model = PeftModel.from_pretrained(base, '{FINAL_DIR}')")
    logger.info(f"  tok   = AutoTokenizer.from_pretrained('{FINAL_DIR}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()