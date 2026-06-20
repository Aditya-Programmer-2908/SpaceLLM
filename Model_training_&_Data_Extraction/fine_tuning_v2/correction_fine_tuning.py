"""
SpaceLLM :: Continual Correction Fine-Tuning
==============================================
Responsibility: Take a small batch of human-corrected QA pairs (built by
                execute.py's RETRAIN_ADAPTER step) and continue training
                the currently-live LoRA adapter on top of the base model,
                producing an updated adapter ready to push to HuggingFace.

Called by:
    backend/mape_k/execute.py  (RETRAIN_ADAPTER action)

Typical invocation (as run by execute.py):
    python correction_fine_tuning.py \
        --train_file  /mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/correction_train_injection.json \
        --output_dir  /mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/fine_tuning_v2/outputs/spacellm_lora_final \
        --epochs 3 --lr 2e-4 --max_seq_len 2048 \
        --base_adapter AdityaPS/SpaceLLM_v3

Design notes
------------
- CONTINUAL, NOT FROM-SCRATCH. If --base_adapter is given, we load the
  base model + that adapter with is_trainable=True and keep training it.
  Correction batches are small (a handful to a few dozen examples), so
  re-initializing a fresh LoRA from the base model every cycle would
  throw away everything learned in prior retrain cycles. If --base_adapter
  is omitted, a fresh LoRA is initialized (matches your established
  r=16 / alpha=32, targeting "lm_head").

- MXFP4. gpt-oss-20b ships MXFP4-quantized. MXFP4 weights aren't directly
  differentiable, so we load with Mxfp4Config(dequantize=True) when the
  installed transformers version supports it. This is the fix for the
  "MXFP4 quantization blocking" crash noted in your earlier fine-tuning
  work — if you hit a similar block again, the first thing to check is
  whether `dequantize=True` actually took effect for your transformers
  version.

- lm_head-ONLY LoRA. Matches your established config. For an MoE model
  this also sidesteps touching the expert/router weights with a tiny,
  noisy correction batch — a meaningful source of the NaN/gradient-spike
  instability you debugged previously. If you want to expand
  target_modules later, do it deliberately and watch the loss curve.

- NO BLIND resize_token_embeddings(). The CUDA "label out of range" bug
  you hit before came from resizing embeddings unconditionally. Here we
  only touch the pad token id (reusing eos_token if no pad token exists)
  and never call resize_token_embeddings — vocab size is left untouched.

- ASSISTANT-ONLY LOSS. Each training example is a 2-turn
  [user question, assistant corrected-answer] pair. Loss is computed only
  on the assistant tokens (prompt tokens are masked to -100), so the model
  isn't penalized for "predicting" the question back.

- NaN GUARD. A callback watches the logged loss every step; on the first
  NaN/Inf it stops training and the script exits non-zero, so
  execute.py's `if result.returncode != 0: raise RuntimeError(...)` stops
  the pipeline before a broken adapter gets pushed to HuggingFace.

- HF TOKEN. Read from the env var named by $HF_TOKEN (inherited from
  execute.py's subprocess environment). Needed only if --base_adapter
  is a private repo. Never hardcode a token in this file.

Author: SpaceLLM Project
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.correction_finetune")

# ---------------------------------------------------------------------------
# Default paths (overridable via CLI; execute.py passes these explicitly)
# ---------------------------------------------------------------------------

_MAPE_DIR        = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k")
_FINE_TUNING_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/fine_tuning_v2")

DEFAULT_TRAIN_FILE  = _MAPE_DIR / "correction_train_injection.json"
DEFAULT_OUTPUT_DIR  = _FINE_TUNING_DIR / "outputs" / "spacellm_lora_final"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual LoRA correction fine-tuning for SpaceLLM.")

    p.add_argument("--base_model", default="openai/gpt-oss-20b",
                    help="Base model to load (MXFP4-quantized, dequantized for training).")
    p.add_argument("--base_adapter", default=None,
                    help="HF repo id of an existing LoRA adapter to continue training from. "
                         "Omit to initialize a fresh LoRA from --base_model instead.")

    p.add_argument("--train_file", default=str(DEFAULT_TRAIN_FILE),
                    help="JSON file of correction records (list of {... , messages: [...]}).")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Where to save the resulting adapter.")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=2048)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", default="lm_head",
                    help="Comma-separated module names for LoRA. Default matches your "
                         "established config (lm_head only).")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--min_reference_words", type=int, default=3,
                    help="Skip records whose assistant content has fewer words than this "
                         "(filters out near-empty / junk corrections).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_token_env", default="HF_TOKEN",
                    help="Name of the env var holding your HF token (for pulling a private "
                         "--base_adapter). Never pass a literal token here.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_correction_examples(train_file: Path, min_reference_words: int) -> list[dict]:
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    raw = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of records in {train_file}, got {type(raw)}.")

    examples = []
    skipped = 0
    for rec in raw:
        messages = rec.get("messages")
        if not messages or len(messages) < 2:
            skipped += 1
            continue
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_turns:
            skipped += 1
            continue
        last_assistant = assistant_turns[-1].get("content", "") or ""
        if len(last_assistant.split()) < min_reference_words:
            skipped += 1
            continue
        examples.append(rec)

    if skipped:
        log.warning("Skipped %d/%d record(s) (missing/short assistant content).",
                    skipped, len(raw))
    if not examples:
        raise ValueError("No usable training examples after filtering.")

    log.info("Loaded %d usable correction example(s) from %s.", len(examples), train_file)
    return examples


def build_supervised_example(tokenizer, messages: list[dict], max_seq_len: int) -> dict[str, list[int]]:
    """
    Tokenize a [user, assistant] (or longer) conversation and mask every
    token except the final assistant turn with -100, so loss is only
    computed on the corrected answer.
    """
    prompt_messages = messages[:-1]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_seq_len, add_special_tokens=False,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_seq_len, add_special_tokens=False,
    )["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_dataset(tokenizer, examples: list[dict], max_seq_len: int):
    from datasets import Dataset

    rows = [build_supervised_example(tokenizer, ex["messages"], max_seq_len) for ex in examples]
    # Drop any example that collapsed to an empty/fully-masked sequence.
    rows = [r for r in rows if any(l != -100 for l in r["labels"])]
    if not rows:
        raise ValueError("All examples collapsed to empty/fully-masked sequences after tokenization.")
    return Dataset.from_list(rows)


class PaddingCollator:
    """Pads input_ids/attention_mask/labels to the longest sequence in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []

        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        # Reuse eos as pad — do NOT resize embeddings for this.
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    try:
        from transformers import Mxfp4Config
        quantization_config = Mxfp4Config(dequantize=True)
        log.info("Using Mxfp4Config(dequantize=True) for trainable MXFP4 weights.")
    except ImportError:
        log.warning(
            "Mxfp4Config not available in this transformers version. Loading without "
            "explicit MXFP4 dequantization — if you hit a 'quantized tensor has no "
            "grad_fn' style error, upgrade transformers."
        )

    load_kwargs: dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    log.info("Loading base model %s ...", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    base_model.config.use_cache = False  # required alongside gradient checkpointing

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    if args.base_adapter:
        hf_token = os.environ.get(args.hf_token_env)
        log.info("Loading existing adapter for continual training: %s", args.base_adapter)
        model = PeftModel.from_pretrained(
            base_model, args.base_adapter, is_trainable=True, token=hf_token,
        )
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        log.info("No --base_adapter given — initializing fresh LoRA (r=%d, alpha=%d, targets=%s).",
                  args.lora_r, args.lora_alpha, target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# NaN guard
# ---------------------------------------------------------------------------

class NaNGuardCallback:
    """
    transformers.TrainerCallback subclass built lazily (after transformers
    is imported) — see make_nan_guard_callback() below. Stops training the
    moment a NaN/Inf loss is logged, so we never save or push a broken
    adapter.
    """
    nan_detected: bool = False


def make_nan_guard_callback():
    from transformers import TrainerCallback

    class _NaNGuard(TrainerCallback, NaNGuardCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return control
            loss = logs["loss"]
            if loss is None or (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))):
                log.error("NaN/Inf loss detected at step %s — stopping training.", state.global_step)
                self.nan_detected = True
                control.should_training_stop = True
            return control

    return _NaNGuard()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    import torch
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(args.seed)

    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_correction_examples(train_file, args.min_reference_words)
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = build_dataset(tokenizer, examples, args.max_seq_len)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    nan_guard = make_nan_guard_callback()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_ckpts"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[nan_guard],
    )

    log.info("Starting training: %d example(s), %d epoch(s), lr=%s, max_seq_len=%d",
              len(dataset), args.epochs, args.lr, args.max_seq_len)
    trainer.train()

    if nan_guard.nan_detected:
        log.error("Training aborted due to NaN/Inf loss. Adapter NOT saved.")
        return 1

    log.info("Saving adapter to %s ...", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.base_model,
        "base_adapter": args.base_adapter,
        "examples_used": len(dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )

    log.info("Done. Adapter saved → %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
