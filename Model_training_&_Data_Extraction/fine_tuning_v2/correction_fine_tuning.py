"""
SpaceLLM Fresh Adapter Training  v2
=====================================
Model    : openai/gpt-oss-20b  (MoE, MXFP4 quantized checkpoint)
Strategy : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method   : Standard BF16 LoRA — NOT QLoRA, no bitsandbytes

This script is the OFFLINE RETRAINING step in the v2 continual-learning
pipeline.  It trains a brand-new lm_head LoRA adapter on the combined
dataset produced by execute.py (DATASET_EXPANSION actions), then saves
the adapter to --output_dir.  The new adapter replaces the previous one
entirely — no stacking, no merging.

Lifecycle
─────────
    GPT-OSS-20B + SpaceLLM_vN
         ↓  (MAPE-K collects corrections → execute.py grows combined_dataset.json)
    python train_spacellm_fresh.py \\
        --train_file combined_dataset.json \\
        --output_dir ./spacellm_v_next_adapter
         ↓
    GPT-OSS-20B + SpaceLLM_v(N+1)   ← vN retired

Key engineering decisions (carried over from lora_finetuning_v4.py)
────────────────────────────────────────────────────────────────────
  [CRITICAL] lm_head weight untied (detach+clone) BEFORE get_peft_model().
             With tie_word_embeddings=True the lm_head and embed_tokens
             share the same tensor; PEFT wraps lm_head but autograd sees
             the weight as belonging to the frozen embed_tokens path and
             cuts gradients to lora_A. detach().clone() materialises
             lm_head as an independent Parameter before PEFT touches it.

  [CRITICAL] resize_token_embeddings tie-re-introduction guard: after
             resize we check id() equality and re-untie if needed.

  [CRITICAL] Device-aware CE loss: MoE models shard across GPUs; lm_head
             may land on cuda:1/2 while Trainer places labels on cuda:0.
             Loss moves labels to match logits then returns to cuda:0.

  [NICE]     Triton stub patch applied before any import so the MoE router
             (which imports triton) doesn't crash on machines without triton.

  [NICE]     Gradient checkpointing + fused AdamW + cosine LR.

  [NICE]     NaN/Inf guard callback — stops training and skips save.

  [NICE]     Atomic adapter save with training_metadata.json sidecar.

Usage
─────
    python train_spacellm_fresh.py \\
        --train_file /path/to/combined_dataset.json \\
        --output_dir ./spacellm_v2_adapter

    # Full options:
    python train_spacellm_fresh.py \\
        --train_file combined_dataset.json \\
        --output_dir ./spacellm_v2_adapter \\
        --base_model openai/gpt-oss-20b \\
        --epochs 15 \\
        --lr 2e-4 \\
        --lora_r 32 \\
        --lora_alpha 128 \\
        --max_seq_len 2048 \\
        --hf_token $HF_TOKEN

Output layout
─────────────
    <output_dir>/
    ├── adapter_config.json          ← PEFT adapter config
    ├── adapter_model.safetensors    ← trained LoRA weights
    ├── tokenizer files              ← tokenizer snapshot
    ├── training_metadata.json       ← hyperparams + dataset provenance
    └── _ckpts/                      ← intermediate checkpoints (if any)
"""

# ── TRITON PATCH — must be the very first thing before any other import ───────
import sys
import types


def _patch_triton():
    """
    Stub out triton if it is not installed.  The GPT-OSS-20B MoE router
    imports triton at model-load time; without this stub the import chain
    crashes on machines that don't have triton installed.
    """

    class _StubDriver:
        def __getattr__(self, name):
            return _StubDriver()
        def __call__(self, *a, **kw):
            return _StubDriver()
        def __bool__(self):
            return False

    class _StubCudaUtils:
        def __getattr__(self, name):
            return lambda *a, **kw: None

    class _ActiveDriverDescriptor:
        def __get__(self, obj, objtype=None):
            return _StubDriver()
        def __set__(self, obj, value):
            pass

    class _StubDriverManager:
        active  = _ActiveDriverDescriptor()
        default = _StubDriver()
        def __init__(self):
            self._active  = _StubDriver()
            self._default = _StubDriver()

    def _make_module(name, parent=None):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if parent:
            setattr(parent, name.split(".")[-1], mod)
        return mod

    try:
        import triton  # noqa: F401
        return  # already installed — nothing to do
    except Exception:
        pass

    triton_mod          = _make_module("triton")
    triton_runtime      = _make_module("triton.runtime",                triton_mod)
    triton_runtime_drv  = _make_module("triton.runtime.driver",         triton_runtime)
    triton_runtime_bld  = _make_module("triton.runtime.build",          triton_runtime)
    triton_runtime_jit  = _make_module("triton.runtime.jit",            triton_runtime)
    triton_backends     = _make_module("triton.backends",               triton_mod)
    triton_backends_nv  = _make_module("triton.backends.nvidia",        triton_backends)
    triton_backends_drv = _make_module("triton.backends.nvidia.driver", triton_backends_nv)

    drv_singleton = _StubDriverManager()
    triton_runtime_drv.driver = drv_singleton

    triton_runtime_bld._build                  = lambda *a, **kw: None
    triton_runtime_bld.compile_module_from_src = lambda *a, **kw: types.ModuleType("_stub")
    triton_runtime_bld.load_module             = lambda *a, **kw: types.ModuleType("_stub")

    class _StubJITFunction:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *a, **kw):
            try:
                return self.fn(*a, **kw)
            except Exception:
                return None
        def __getattr__(self, name):
            return lambda *a, **kw: None

    triton_runtime_jit.JITFunction = _StubJITFunction
    triton_mod.jit = lambda fn=None, **kw: (
        _StubJITFunction(fn) if fn is not None
        else (lambda f: _StubJITFunction(f))
    )

    triton_backends_drv.CudaUtils = _StubCudaUtils
    triton_mod.runtime  = triton_runtime
    triton_mod.backends = triton_backends


_patch_triton()

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ── Logging ───────────────────────────────────────────────────────────────────

RUN_ID   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_FILE = Path(f"train_fresh_{RUN_ID}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("SpaceLLM.Fresh")

IGNORE_INDEX = -100

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a fresh SpaceLLM lm_head LoRA adapter on combined_dataset.json"
    )
    # Paths
    p.add_argument("--base_model",  default="openai/gpt-oss-20b",
                   help="HuggingFace model id or local path of the base model")
    p.add_argument("--train_file",  required=True,
                   help="Path to combined_dataset.json (output of execute.py)")
    p.add_argument("--output_dir",  required=True,
                   help="Directory to save the trained LoRA adapter")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # Training hyperparams
    p.add_argument("--epochs",        type=int,   default=15,
                   help="Training epochs (default 15)")
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--max_seq_len",   type=int,   default=2048)
    p.add_argument("--batch_size",    type=int,   default=1,
                   help="Per-device batch size")
    p.add_argument("--grad_accum",    type=int,   default=32,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_steps",  type=int,   default=200)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--logging_steps", type=int,   default=10)
    p.add_argument("--save_steps",    type=int,   default=500,
                   help="Save intermediate checkpoint every N steps (0 = never)")
    # LoRA hyperparams
    p.add_argument("--lora_r",        type=int,   default=32)
    p.add_argument("--lora_alpha",    type=int,   default=128,
                   help="LoRA alpha (default 4×r = 128)")
    p.add_argument("--lora_dropout",  type=float, default=0.1)
    p.add_argument("--target_modules",default="lm_head",
                   help="Comma-separated list of module names for LoRA")
    # Misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_gpu_info() -> None:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            log.info("Visible GPUs:")
            for line in r.stdout.strip().splitlines():
                idx, name, mem = line.split(",")
                log.info("  cuda:%s → %s  (%s MiB)",
                         idx.strip(), name.strip(), f"{int(mem.strip()):,}")
    except Exception:
        pass


def log_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        props  = torch.cuda.get_device_properties(i)
        alloc  = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserv = torch.cuda.memory_reserved(i)  / 1024 ** 3
        total  = props.total_memory              / 1024 ** 3
        log.info("  GPU%d [%s] %s | alloc=%.1fGB  res=%.1fGB  total=%.1fGB",
                 i, props.name, label, alloc, reserv, total)


def log_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100.0 * trainable / total if total else 0.0
    log.info("  Trainable : %s / %s  (%.6f%%)", f"{trainable:,}", f"{total:,}", pct)
    for n, p in model.named_parameters():
        if p.requires_grad:
            log.info("    %-60s  shape=%-20s  (%s params)",
                     n, str(list(p.shape)), f"{p.numel():,}")


# ─────────────────────────────────────────────────────────────────────────────
# Device-aware CE loss
# ─────────────────────────────────────────────────────────────────────────────

def _make_device_aware_ce_loss():
    """
    MoE models shard across GPUs.  lm_head may land on cuda:1 or cuda:2
    while Trainer places labels on cuda:0.  This loss:
      1. Moves labels to match logits device.
      2. Clamps any out-of-vocab label ids to IGNORE_INDEX.
      3. Returns scalar loss to cuda:0 so Trainer can accumulate it.
    """
    def _loss(logits, labels, vocab_size=None, **kwargs):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        dev = shift_logits.device
        if shift_labels.device != dev:
            shift_labels = shift_labels.to(dev)

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

    return _loss


_DEVICE_AWARE_CE_LOSS = _make_device_aware_ce_loss()


def inject_loss(model, label: str = "") -> None:
    """Patch loss_function on model and its PEFT wrappers."""
    tag = f" ({label})" if label else ""
    candidates = [model,
                  getattr(model, "base_model", None),
                  getattr(getattr(model, "base_model", None), "model", None),
                  *list(model.children())]
    for obj in candidates:
        if obj is not None and hasattr(obj, "loss_function"):
            if getattr(obj, "loss_function") is not _DEVICE_AWARE_CE_LOSS:
                obj.loss_function = _DEVICE_AWARE_CE_LOSS
                log.info("  ✅ loss_function patched on %s%s", type(obj).__name__, tag)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_examples(train_file: Path) -> list[dict]:
    """
    Load combined_dataset.json.  Expected format:
        [ { "messages": [ {role, content}, ... ] }, ... ]

    Records without a non-empty assistant turn are skipped with a warning.
    """
    raw = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {train_file}, got {type(raw).__name__}")

    out, skipped = [], 0
    for rec in raw:
        msgs = rec.get("messages", [])
        asst = [m for m in msgs if m.get("role") == "assistant"
                and (m.get("content") or "").strip()]
        if not asst:
            skipped += 1
            continue
        out.append(rec)

    if skipped:
        log.warning("  Skipped %d / %d records (missing / empty assistant turn)",
                    skipped, len(raw))
    if not out:
        raise ValueError("No usable training examples after filtering.")

    log.info("  Loaded %d usable example(s) from %s", len(out), train_file)
    return out


def dataset_sanity_check(records: list[dict], split_name: str) -> None:
    """Log structural stats matching the v4 sanity check."""
    log.info("── Sanity check: %s (%d records) ──", split_name, len(records))
    issues, no_assistant = 0, 0
    org_dist: dict = defaultdict(int)
    diff_dist: dict = defaultdict(int)
    chain_ids: set  = set()

    for i, r in enumerate(records):
        for field in ("sample_id", "source_id", "mission_name", "organization",
                      "aspect", "difficulty", "chain_id", "messages"):
            if field not in r:
                log.warning("  Record %d: missing '%s'", i, field)
                issues += 1
                break
        roles = [m.get("role") for m in r.get("messages", [])]
        if "assistant" not in roles:
            no_assistant += 1
        org_dist[r.get("organization", "?")]  += 1
        diff_dist[r.get("difficulty",   "?")] += 1
        chain_ids.add(r.get("chain_id", ""))

    log.info("  Unique chains     : %d", len(chain_ids))
    log.info("  Structural issues : %d", issues)
    log.info("  No assistant turn : %d", no_assistant)
    log.info("  Organizations     : %s", dict(sorted(org_dist.items())))
    log.info("  Difficulty        : %s", dict(sorted(diff_dist.items())))


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_record(record: dict, tokenizer, max_seq_len: int,
                    debug: bool = False) -> dict | None:
    """
    Tokenise one record.  Assistant tokens are kept as labels; everything
    before (developer / user turns) is masked with IGNORE_INDEX.
    Returns None for records that should be skipped.
    """
    msgs = record.get("messages", [])

    hf_msgs = []
    for m in msgs:
        role    = "system" if m["role"] == "developer" else m["role"]
        content = (m.get("content") or "").strip()
        if content:
            hf_msgs.append({"role": role, "content": content})

    if not hf_msgs:
        return None

    try:
        full_text = tokenizer.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        log.debug("apply_chat_template failed: %s", e)
        return None

    full_enc  = tokenizer(full_text, truncation=True, max_length=max_seq_len,
                          padding=False, return_tensors=None)
    input_ids = full_enc["input_ids"]
    if len(input_ids) < 4:
        return None

    # Build prefix (everything except the last assistant turn) to find boundary
    prefix_msgs = [m for m in hf_msgs if m["role"] != "assistant"]
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

    prefix_len = len(tokenizer(
        prefix_text, truncation=True, max_length=max_seq_len,
        padding=False, return_tensors=None)["input_ids"])

    if prefix_len >= len(input_ids):
        return None

    labels   = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels   = labels[:len(input_ids)]
    n_active = sum(1 for l in labels if l != IGNORE_INDEX)

    if n_active < 4:
        return None

    if debug:
        log.info("    prefix_len=%d  total=%d  active=%d", prefix_len, len(input_ids), n_active)
        log.info("    First active (loss) tokens:")
        shown = 0
        for tok_id, lbl in zip(input_ids, labels):
            if lbl != IGNORE_INDEX and shown < 8:
                log.info("      %r", tokenizer.decode([tok_id]))
                shown += 1

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
        "n_active":       n_active,
    }


def build_dataset(tokenizer, examples: list[dict],
                  max_seq_len: int, vocab_size: int | None = None):
    """Tokenise all examples and return a HuggingFace Dataset."""
    from datasets import Dataset

    rows, skipped, clamped = [], 0, 0

    for i, rec in enumerate(examples):
        result = tokenise_record(rec, tokenizer, max_seq_len, debug=(i < 3))
        if result is None:
            skipped += 1
            continue

        result.pop("n_active")

        if vocab_size is not None:
            new_labels, had_oob = [], False
            for lbl in result["labels"]:
                if lbl != IGNORE_INDEX and (lbl < 0 or lbl >= vocab_size):
                    new_labels.append(IGNORE_INDEX)
                    had_oob = True
                else:
                    new_labels.append(lbl)
            if had_oob:
                result["labels"] = new_labels
                clamped += 1

        if all(l == IGNORE_INDEX for l in result["labels"]):
            skipped += 1
            continue

        rows.append(result)

    if skipped:
        log.warning("  Tokenisation skipped %d examples", skipped)
    if clamped:
        log.warning("  Clamped OOV labels in %d records", clamped)
    if not rows:
        raise ValueError("All examples collapsed after tokenisation.")

    lengths     = [len(r["input_ids"]) for r in rows]
    active_lens = []
    for r in rows:
        active_lens.append(sum(1 for l in r["labels"] if l != IGNORE_INDEX))

    log.info(
        "  Dataset: %d rows | seq_len min=%d mean=%.0f max=%d | "
        "active min=%d mean=%.1f max=%d",
        len(rows),
        min(lengths), sum(lengths) / len(lengths), max(lengths),
        min(active_lens), sum(active_lens) / len(active_lens), max(active_lens),
    )

    if sum(active_lens) / len(active_lens) < 10:
        log.warning("  ⚠️  Mean active tokens < 10 — label masking may be too aggressive")

    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def make_nan_guard():
    """Stop training and flag the run if loss goes NaN/Inf."""
    from transformers import TrainerCallback

    class _NaNGuard(TrainerCallback):
        def __init__(self):
            self.triggered = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                log.error("NaN/Inf loss at step %d — stopping.", state.global_step)
                self.triggered = True
                control.should_training_stop = True
            return control

    return _NaNGuard()


def make_history_callback():
    """Collect loss / LR / grad-norm history for post-training summary."""
    from transformers import TrainerCallback

    class _History(TrainerCallback):
        def __init__(self):
            self.train_loss: list[tuple[int, float]] = []
            self.lr_history:  list[tuple[int, float]] = []
            self.grad_norms:  list[tuple[int, float]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            if "loss"          in logs: self.train_loss.append((step, logs["loss"]))
            if "learning_rate" in logs: self.lr_history.append((step, logs["learning_rate"]))
            if "grad_norm"     in logs: self.grad_norms.append((step, logs["grad_norm"]))

    return _History()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    train_file = Path(args.train_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    log.info("=" * 70)
    log.info("  SpaceLLM Fresh Adapter Training  v2")
    log.info("  Run ID      : %s", RUN_ID)
    log.info("  base_model  : %s", args.base_model)
    log.info("  train_file  : %s", train_file)
    log.info("  output_dir  : %s", output_dir)
    log.info("  epochs      : %d  |  lr: %g  |  batch: %d  |  grad_accum: %d",
             args.epochs, args.lr, args.batch_size, args.grad_accum)
    log.info("  LoRA        : r=%d  alpha=%d  dropout=%.2f  target=%s",
             args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)
    log.info("  eff_batch   : %d  |  max_seq_len: %d",
             args.batch_size * args.grad_accum, args.max_seq_len)
    log.info("  CUDA_VISIBLE_DEVICES : %s",
             os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    log.info("  log file    : %s", LOG_FILE)
    log.info("=" * 70)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Mxfp4Config,
        TrainingArguments, DataCollatorForSeq2Seq, Trainer,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    log_gpu_info()

    # ── [1] Load training examples ────────────────────────────────────────
    log.info("\n[1/8] Loading training examples ...")
    examples = load_examples(train_file)
    dataset_sanity_check(examples, "combined_dataset")

    # ── [2] Load tokenizer ───────────────────────────────────────────────
    log.info("\n[2/8] Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"
    log.info("  vocab_size=%s  len(tokenizer)=%s  chat_template=%s",
             f"{tokenizer.vocab_size:,}", f"{len(tokenizer):,}",
             "found" if tokenizer.chat_template else "MISSING")

    # ── [3] Load base model to CPU ───────────────────────────────────────
    log.info("\n[3/8] Loading base model (MXFP4 → BF16) to CPU: %s", args.base_model)
    log_gpu_memory("pre-load")
    t0    = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        token=args.hf_token,
        ignore_mismatched_sizes=True,
    )
    log.info("  Loaded in %.1fs | dtype=%s", time.time() - t0,
             next(model.parameters()).dtype)
    model.config.use_cache = False

    # ── [4] Vocab alignment + CRITICAL lm_head untie ─────────────────────
    log.info("\n[4/8] Vocab alignment & lm_head untie ...")

    # 4a. Disable weight-tying in config (flag only — does NOT break live tie)
    model.config.tie_word_embeddings = False

    # 4b. CRITICAL: physically materialise lm_head as its own independent
    #     tensor BEFORE get_peft_model().  With tie_word_embeddings=True,
    #     lm_head.weight and embed_tokens.weight are the SAME object;
    #     autograd cuts gradients to lora_A through the frozen embed_tokens
    #     path.  detach().clone() breaks that link.
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    log.info("  ✅ lm_head weight untied (detach+clone) before PEFT")

    # 4c. Resize vocab to tokenizer length (padded to multiple of 64)
    _current_vocab = model.get_output_embeddings().weight.shape[0]
    model.resize_token_embeddings(_current_vocab, pad_to_multiple_of=64)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = actual_vocab

    assert model.get_input_embeddings().weight.shape[0]  == actual_vocab
    assert model.get_output_embeddings().weight.shape[0] == actual_vocab
    log.info("  Vocab aligned: %s (padded to multiple of 64)", f"{actual_vocab:,}")

    # 4d. Guard: resize_token_embeddings can silently re-tie the weights
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        log.warning("  ⚠️  resize re-tied lm_head — untying again")
        lm_head = model.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        log.info("  ✅ lm_head re-untied after resize")
    else:
        log.info("  ✅ lm_head still independent after resize")

    # ── [5] Loss injection (pre-PEFT) ────────────────────────────────────
    log.info("\n[5/8] Injecting device-aware CE loss (pre-PEFT) ...")
    inject_loss(model, "pre-PEFT")

    # ── [6] Apply fresh LoRA on CPU ──────────────────────────────────────
    log.info("\n[6/8] Applying fresh LoRA on CPU ...")
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
    log.info("  ✅ get_peft_model() applied (lm_head already untied)")
    log.info("  target_modules : %s", target_modules)

    # Enable input grads + gradient checkpointing (after PEFT)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("  ✅ enable_input_require_grads + gradient_checkpointing")

    # Explicit freeze: only lora_ params trainable
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
            log.error("     %-60s  %s", n, s)
        return 1
    log.info("  Frozen=%d  LoRA trainable=%d  ✅", frozen, lora_count)

    inject_loss(model, "post-PEFT pre-dispatch")
    log_trainable(model)
    log_gpu_memory("after LoRA init (CPU)")

    # ── GPU dispatch ─────────────────────────────────────────────────────
    log.info("\n  Dispatching PEFT model across GPUs ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        # Identify no-split classes (transformer layers / MoE blocks)
        no_split: list[str] = []
        for _, module in model.named_modules():
            cls      = type(module)
            cls_name = cls.__name__.lower()
            if (issubclass(cls, nn.Module) and cls is not nn.Module
                    and ("layer" in cls_name or "block" in cls_name)
                    and cls.__name__ not in no_split
                    and sum(p.numel() for p in module.parameters()) > 1_000_000):
                no_split.append(cls.__name__)
        no_split = list(dict.fromkeys(no_split))
        log.info("  no_split_module_classes : %s", no_split)

        n_gpus     = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            free = torch.cuda.mem_get_info(i)[0]
            max_memory[i] = f"{max(0, int((free - 4 * 1024 ** 3) / 1024 ** 3))}GiB"
        max_memory["cpu"] = "80GiB"
        log.info("  max_memory : %s", max_memory)

        device_map = infer_auto_device_map(
            model, max_memory=max_memory,
            no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)
        log.info("  GPU dispatch done in %.1fs", time.time() - t1)

        if hasattr(model, "hf_device_map"):
            from collections import Counter
            for dev, cnt in sorted(
                Counter(str(v) for v in model.hf_device_map.values()).items()
            ):
                log.info("    %s : %d layer(s)", dev, cnt)

    except Exception as e:
        log.warning("  dispatch_model failed (%s) — falling back to cuda:0", e)
        model = model.to("cuda:0")

    log_gpu_memory("after dispatch")
    inject_loss(model, "post-dispatch")

    # Confirm vocab post-dispatch
    _post_peft_vocab = model.get_output_embeddings().weight.shape[0]
    if _post_peft_vocab != model.config.vocab_size:
        log.warning("  lm_head vocab (%d) != config (%d) — fixing",
                    _post_peft_vocab, model.config.vocab_size)
        model.config.vocab_size = _post_peft_vocab
    log.info("  lm_head vocab = %s  ✅", f"{_post_peft_vocab:,}")

    # ── [7] Tokenise dataset ─────────────────────────────────────────────
    log.info("\n[7/8] Tokenising training examples ...")
    train_dataset = build_dataset(
        tokenizer, examples, args.max_seq_len, vocab_size=_post_peft_vocab)

    # ── TrainingArguments ────────────────────────────────────────────────
    MAX_GRAD_NORM = args.max_grad_norm
    ckpt_dir      = output_dir / "_ckpts"

    save_strategy = "steps" if args.save_steps > 0 else "no"
    save_steps    = args.save_steps if args.save_steps > 0 else 500

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=args.warmup_steps,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch_fused",
        weight_decay=args.weight_decay,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="no",          # no validation split in fresh training
        load_best_model_at_end=False,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=True,
    )

    log.info("\n  Training configuration:")
    for k, v in {
        "epochs":          args.epochs,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "grad_accum":      args.grad_accum,
        "eff_batch":       args.batch_size * args.grad_accum,
        "max_grad_norm":   MAX_GRAD_NORM,
        "weight_decay":    args.weight_decay,
        "warmup_steps":    args.warmup_steps,
        "lora_r":          args.lora_r,
        "lora_alpha":      args.lora_alpha,
        "scheduler":       "cosine_with_restarts",
        "optimizer":       "adamw_torch_fused",
        "train_samples":   len(train_dataset),
    }.items():
        log.info("    %-25s: %s", k, v)

    # ── Data collator ─────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    nan_guard       = make_nan_guard()
    history_cb      = make_history_callback()

    # ── Custom Trainer ────────────────────────────────────────────────────
    class SpaceLLMTrainer(Trainer):
        """
        Overrides two methods:
          _prepare_inputs  — moves labels to lm_head device (MoE multi-GPU)
          training_step    — adds explicit clip_grad_norm_ on LoRA params
        """
        def _get_lm_head_device(self):
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            inputs    = super()._prepare_inputs(inputs)
            lm_device = self._get_lm_head_device()
            if lm_device and "labels" in inputs \
                    and inputs["labels"].device != lm_device:
                inputs["labels"] = inputs["labels"].to(lm_device)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = (
                super().training_step(model, inputs, num_items_in_batch)
                if num_items_in_batch is not None
                else super().training_step(model, inputs)
            )
            # Belt-and-suspenders grad clip on LoRA params specifically
            trainable = [p for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
            return loss

    trainer = SpaceLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[nan_guard, history_cb],
    )

    inject_loss(trainer.model, "post-Trainer")
    log.info("  lm_head device : %s", trainer._get_lm_head_device())

    # ── [8] Train ────────────────────────────────────────────────────────
    log.info("\n[8/8] Training ...")
    log.info("=" * 70)
    log.info("  %d examples  ×  %d epochs  =  ~%d steps per epoch",
             len(train_dataset), args.epochs,
             max(1, len(train_dataset) // (args.batch_size * args.grad_accum)))
    log.info("=" * 70 + "\n")

    t_start = time.time()
    resume  = args.resume_from_checkpoint
    if resume and not Path(resume).exists():
        log.warning("  Checkpoint not found: %s — starting fresh", resume)
        resume = None

    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        log.warning("Interrupted — saving partial adapter ...")
        interrupted = output_dir / "interrupted"
        trainer.save_model(str(interrupted))
        tokenizer.save_pretrained(str(interrupted))
        log.info("Partial save → %s", interrupted)
        return 0
    except Exception as e:
        log.error("Training failed: %s", e, exc_info=True)
        return 1

    elapsed = time.time() - t_start
    log.info("\n  Training complete in %.1f min", elapsed / 60)

    if nan_guard.triggered:
        log.error("  NaN/Inf detected during training — NOT saving adapter.")
        return 1

    # ── Training summary ──────────────────────────────────────────────────
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)

    if history_cb.train_loss:
        final_loss = history_cb.train_loss[-1][1]
        best_loss  = min(v for _, v in history_cb.train_loss)
        log.info("  Final train loss : %.4f  |  Best : %.4f", final_loss, best_loss)

    # ── Save adapter ──────────────────────────────────────────────────────
    log.info("\n  Saving adapter → %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "run_id":           RUN_ID,
        "model_name":       "SpaceLLM_fresh",
        "base_model":       args.base_model,
        "strategy":         "fresh_lora_only — lm_head LoRA, backbone frozen, BF16",
        "train_file":       str(train_file),
        "examples_used":    len(train_dataset),
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  args.batch_size * args.grad_accum,
        "warmup_steps":     args.warmup_steps,
        "max_grad_norm":    MAX_GRAD_NORM,
        "weight_decay":     args.weight_decay,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "lora_dropout":     args.lora_dropout,
        "target_modules":   target_modules,
        "train_metrics":    train_metrics,
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "log_file":         str(LOG_FILE),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Save training graphs ──────────────────────────────────────────────
    _save_training_graphs(history_cb, output_dir, RUN_ID)

    log.info("\n" + "=" * 70)
    log.info("  SpaceLLM Fresh Adapter  — Complete ✅")
    log.info("=" * 70)
    log.info("  Adapter saved  → %s", output_dir)
    log.info("  Metadata       → %s/training_metadata.json", output_dir)
    log.info("  Log            → %s", LOG_FILE)
    log.info("")
    log.info("  To load:")
    log.info("    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config")
    log.info("    from peft import PeftModel")
    log.info("    base  = AutoModelForCausalLM.from_pretrained('%s',", args.base_model)
    log.info("                quantization_config=Mxfp4Config(dequantize=True), device_map='auto')")
    log.info("    model = PeftModel.from_pretrained(base, '%s')", output_dir)
    log.info("    tok   = AutoTokenizer.from_pretrained('%s')", output_dir)
    log.info("=" * 70)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Training graph helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_training_graphs(history, output_dir: Path, run_id: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not available — skipping graph saving")
        return

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    def _plot(xy_list, title, ylabel, color, path):
        if not xy_list:
            return
        steps, vals = zip(*xy_list)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("  Graph → %s", path)

    _plot(history.train_loss, f"Training Loss [{run_id}]",
          "Loss",      "#e74c3c", graphs_dir / f"train_loss_{run_id}.png")
    _plot(history.lr_history,  f"LR Schedule [{run_id}]",
          "LR",        "#27ae60", graphs_dir / f"lr_schedule_{run_id}.png")
    _plot(history.grad_norms,  f"Gradient Norm [{run_id}]",
          "Grad Norm", "#8e44ad", graphs_dir / f"grad_norm_{run_id}.png")

    # Overview grid
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"SpaceLLM Fresh Adapter — Training Overview  [{run_id}]",
                     fontsize=13, fontweight="bold")
        for ax, (xy_list, title, color, ylabel) in zip(axes, [
            (history.train_loss, "Training Loss",  "#e74c3c", "Loss"),
            (history.lr_history, "LR Schedule",    "#27ae60", "LR"),
            (history.grad_norms, "Gradient Norm",  "#8e44ad", "Grad Norm"),
        ]):
            if xy_list:
                steps, vals = zip(*xy_list)
                ax.plot(steps, vals, color=color, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        overview = graphs_dir / f"overview_{run_id}.png"
        fig.savefig(overview, dpi=150)
        plt.close(fig)
        log.info("  Overview graph → %s", overview)
    except Exception as e:
        log.warning("Overview graph failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
