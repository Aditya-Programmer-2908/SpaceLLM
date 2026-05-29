"""
SpaceLLM — Experimental LoRA Fine-Tuning
==========================================
Model     : openai/gpt-oss-20b
Phase     : Experimentation
Strategy  : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method    : Standard BF16 LoRA — NOT QLoRA, no 4-bit, no bitsandbytes

File      : /home/aditya/SpaceLLM/fine_tuning/train_spacellm_lora.py

Output layout after training:
  SpaceLLM/fine_tuning/outputs/
  ├── checkpoints/                ← intermediate saves
  ├── spacellm_lora_final/        ← LOAD THIS for inference / evaluation
  │   ├── adapter_config.json
  │   ├── adapter_model.safetensors
  │   ├── tokenizer.json
  │   ├── tokenizer_config.json
  │   └── adapter_info.json
  └── logs/
      ├── train_YYYYMMDD_HHMMSS.log
      ├── train_metrics_*.json
      └── eval_metrics_*.json

To load after training:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

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
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent      # fine_tuning/
PROJECT_ROOT = SCRIPT_DIR.parent                    # SpaceLLM/

OUTPUT_DIR = SCRIPT_DIR / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FINAL_DIR  = OUTPUT_DIR / "spacellm_lora_final"
LOG_DIR    = OUTPUT_DIR / "logs"

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Fixed dataset paths — never reshuffle, chain-aware split ─────────────────

TRAIN_FILE = Path("/home/aditya/SpaceLLM/data_processing/DatasetA_core_QA/train.jsonl")
VAL_FILE   = Path("/home/aditya/SpaceLLM/data_processing/DatasetA_core_QA/validation.jsonl")
TEST_FILE  = Path("/home/aditya/SpaceLLM/data_processing/DatasetA_core_QA/test.jsonl")

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
    p.add_argument("--warmup_ratio",           type=float, default=0.05)
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
            logger.warning("No CUDA device — running on CPU (very slow for 20B)")
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

IGNORE_INDEX = -100   # standard PyTorch cross-entropy ignore label


def tokenise_record(record: dict, tokenizer, max_seq_len: int) -> dict | None:
    """
    Build input_ids / attention_mask / labels for one record.

    gpt-oss-20b uses the harmony response format via its built-in chat template.
    tokenizer.apply_chat_template() handles this automatically.

    Loss masking:
      - developer (system) tokens → IGNORE_INDEX
      - user tokens               → IGNORE_INDEX
      - assistant tokens          → real ids  ← loss computed here ONLY
    """
    messages = record.get("messages", [])

    # Map 'developer' → 'system' so gpt-oss chat template recognises it
    hf_messages = []
    for msg in messages:
        role    = "system" if msg["role"] == "developer" else msg["role"]
        content = msg.get("content", "").strip()
        if content:
            hf_messages.append({"role": role, "content": content})

    if not hf_messages:
        return None

    # ── Full conversation (system + user + assistant) ─────────────────────
    # tokenize=False → we get a string back, tokenise manually for control
    try:
        full_text = tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e} — skipping record")
        return None

    full_enc  = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )
    input_ids = full_enc["input_ids"]

    if len(input_ids) < 4:
        return None

    # ── Prefix only (system + user) → find where assistant response begins ─
    prefix_msgs = [m for m in hf_messages if m["role"] != "assistant"]

    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs,
            tokenize=False,
            add_generation_prompt=True,   # adds the assistant turn opener token(s)
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

    # ── Build labels: mask prefix, keep assistant tokens ──────────────────
    labels = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels = labels[:len(input_ids)]   # trim if truncated

    # Skip records where assistant portion is entirely masked
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
        logger.warning(f"[{split_name}] Skipped {skipped} records during tokenisation")
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

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA  (lm_head only, BF16)")
    logger.info(f"  Run ID   : {RUN_ID}")
    logger.info(f"  Model    : {args.model_id}")
    logger.info(f"  Strategy : LoRA on lm_head ONLY — backbone fully frozen")
    logger.info(f"  Epochs   : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch    : {args.batch_size}  |  Grad accum: {args.grad_accum}  "
                f"|  Eff batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  Log      : {LOG_FILE}")
    logger.info("=" * 60)

    # ── Deferred imports ──────────────────────────────────────────────────
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
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Dataset file verification ─────────────────────────────────────────
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

    # gpt-oss-20b may not have a pad token — set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token")

    tokenizer.padding_side = "right"   # required for causal LM training
    logger.info(f"Vocab size       : {tokenizer.vocab_size:,}")
    logger.info(f"Pad token        : '{tokenizer.pad_token}'  (id={tokenizer.pad_token_id})")

    # Confirm chat template is available
    if tokenizer.chat_template is not None:
        logger.info("Chat template    : found (harmony format applied automatically)")
    else:
        logger.warning("Chat template    : NOT FOUND — formatting may be incorrect")

    # ── Model in BF16 — no quantization ───────────────────────────────────
    logger.info("")
    logger.info(f"Loading model in BF16: {args.model_id}")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",            # spread across available GPUs automatically
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        sys.exit(1)

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    logger.info(f"Model dtype      : {next(model.parameters()).dtype}")
    logger.info(f"Model device map : {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    log_gpu_memory("after model load")

    # Required for gradient checkpointing compatibility
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: enabled")

    # ── LoRA on lm_head ONLY ──────────────────────────────────────────────
    logger.info("")
    logger.info("Applying LoRA adapters to lm_head ONLY ...")
    logger.info("  All transformer layers remain frozen.")
    logger.info("  Only lm_head LoRA weights will be trained.")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],       # ← ONLY lm_head, nothing else
    )

    model = get_peft_model(model, lora_config)

    logger.info("")
    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init")

    # ── Load datasets ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Loading datasets ─────────────────────────────────")
    train_records = load_jsonl(TRAIN_FILE)
    val_records   = load_jsonl(VAL_FILE)

    dataset_sanity_check(train_records, "train")
    dataset_sanity_check(val_records,   "validation")

    logger.info("")
    logger.info("── Tokenising datasets ──────────────────────────────")
    train_dataset = build_hf_dataset(train_records, tokenizer, args.max_seq_len, "train")
    val_dataset   = build_hf_dataset(val_records,   tokenizer, args.max_seq_len, "validation")

    # ── Training arguments ────────────────────────────────────────────────
    logger.info("")
    logger.info("── Training configuration ───────────────────────────")
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),

        # Epochs and batching
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Optimizer and LR schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",

        # Precision — BF16 only, no fp16, no quantization
        bf16=True,
        fp16=False,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_dir=str(LOG_DIR),
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        # Performance
        group_by_length=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    for k, v in {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "optimizer": "adamw_torch",
        "scheduler": "cosine",
        "bf16": True,
        "max_seq_len": args.max_seq_len,
    }.items():
        logger.info(f"  {k:<25}: {v}")

    # ── Data collator ─────────────────────────────────────────────────────
    # Dynamic padding; preserves IGNORE_INDEX in labels so loss mask is intact
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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Starting training ...")
    logger.info("=" * 60)

    # Resume from checkpoint if provided
    resume = args.resume_from_checkpoint
    if resume:
        if Path(resume).exists():
            logger.info(f"Resuming from checkpoint: {resume}")
        else:
            logger.warning(f"Checkpoint not found: {resume} — starting fresh")
            resume = None

    t_start = time.time()
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving current state ...")
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

    # ── Save metrics ──────────────────────────────────────────────────────
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    train_metrics_file = LOG_DIR / f"train_metrics_{RUN_ID}.json"
    with train_metrics_file.open("w") as f:
        json.dump(train_metrics, f, indent=2)
    logger.info(f"Train metrics → {train_metrics_file}")

    # Final validation evaluation
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

    # Human-readable run summary
    adapter_info = {
        "run_id":                  RUN_ID,
        "base_model":              args.model_id,
        "strategy":                "LoRA on lm_head ONLY — backbone frozen — BF16",
        "lora_r":                  16,
        "lora_alpha":              32,
        "lora_dropout":            0.05,
        "target_modules":          ["lm_head"],
        "epochs":                  args.epochs,
        "learning_rate":           args.lr,
        "batch_size":              args.batch_size,
        "gradient_accumulation":   args.grad_accum,
        "effective_batch_size":    args.batch_size * args.grad_accum,
        "max_seq_len":             args.max_seq_len,
        "train_samples":           len(train_dataset),
        "val_samples":             len(val_dataset),
        "train_metrics":           train_metrics,
        "eval_metrics":            eval_metrics,
        "final_adapter_dir":       str(FINAL_DIR),
        "checkpoints_dir":         str(CKPT_DIR),
        "log_file":                str(LOG_FILE),
    }
    with (FINAL_DIR / "adapter_info.json").open("w") as f:
        json.dump(adapter_info, f, indent=2)

    log_gpu_memory("after training")

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SpaceLLM lm_head LoRA — Training Complete")
    logger.info("=" * 60)
    logger.info(f"  Final adapters   →  {FINAL_DIR}")
    logger.info(f"  Checkpoints      →  {CKPT_DIR}")
    logger.info(f"  Logs             →  {LOG_DIR}")
    logger.info("")
    logger.info("  To load for evaluation / inference:")
    logger.info("")
    logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info("  from peft import PeftModel")
    logger.info("  import torch")
    logger.info("")
    logger.info(f"  base  = AutoModelForCausalLM.from_pretrained(")
    logger.info(f"              '{args.model_id}',")
    logger.info(f"              torch_dtype=torch.bfloat16, device_map='auto')")
    logger.info(f"  model = PeftModel.from_pretrained(base, '{FINAL_DIR}')")
    logger.info(f"  tok   = AutoTokenizer.from_pretrained('{FINAL_DIR}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()