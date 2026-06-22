"""
SpaceLLM :: Continual Correction Fine-Tuning
==============================================
Responsibility: Take human-corrected QA pairs from correction_train_injection.json
                and CONTINUE TRAINING the already fine-tuned SpaceLLM adapter,
                producing an updated adapter at fine_tuning_v2/outputs/spacellm_lora_final/.

This is CONTINUAL LEARNING — not training from scratch.
    Cycle 0 : base_adapter = AdityaPS/SpaceLLM_v1   (HF fine-tuned adapter, vocab=200064)
    Cycle N : base_adapter = local adapter saved from cycle N-1 (vocab=201088)

KEY FIX — adapter loading order:
    HF adapters (SpaceLLM_v1) were saved at vocab=200064.
    The tokenizer after resize lands at vocab=201088 (pad_to_multiple_of=64).
    Loading PEFT after resize causes a shape mismatch crash.

    Correct order for HF adapters:
        1. Load base model  (vocab=200064, no resize yet)
        2. Untie lm_head
        3. Loss injection pre-PEFT
        4. PeftModel.from_pretrained  (shapes match: 200064 == 200064)
        5. merge_and_unload()         (LoRA baked into base weights)
        6. Re-untie lm_head post-merge
        7. resize_token_embeddings    (200064 → 201088, safe now)
        8. Re-untie guard post-resize
        9. Fresh correction LoRA via get_peft_model()

    Correct order for local adapters (cycle 1+, saved at vocab=201088):
        1. Load base model  (vocab=200064)
        2. Untie lm_head
        3. resize_token_embeddings    (200064 → 201088, matches saved adapter)
        4. Re-untie guard post-resize
        5. Loss injection pre-PEFT
        6. PeftModel.from_pretrained  (shapes match: 201088 == 201088)
        7. merge_and_unload()
        8. Re-untie lm_head post-merge
        9. Fresh correction LoRA via get_peft_model()

Called by:
    backend/mape_k/execute.py  (RETRAIN_ADAPTER action)

Invocation:
    python correction_fine_tuning.py \
        --train_file  .../backend/mape_k/correction_train_injection.json \
        --output_dir  .../fine_tuning_v2/outputs/spacellm_lora_final \
        --base_adapter AdityaPS/SpaceLLM_v1 \
        --lora_r 64 --lora_alpha 128

Author: SpaceLLM Project
"""

from __future__ import annotations

# ── TRITON PATCH — must be the very first thing ───────────────────────────────
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
        active  = _ActiveDriverDescriptor()
        default = _StubDriver()
        def __init__(self):
            self._active  = _StubDriver()
            self._default = _StubDriver()

    def _make_module(name, parent=None):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if parent:
            leaf = name.split(".")[-1]
            setattr(parent, leaf, mod)
        return mod

    try:
        import triton  # noqa: F401
        return
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
        def __init__(self, fn): self.fn = fn
        def __call__(self, *a, **kw):
            try: return self.fn(*a, **kw)
            except Exception: return None
        def __getattr__(self, name): return lambda *a, **kw: None

    triton_runtime_jit.JITFunction = _StubJITFunction
    triton_mod.jit = lambda fn=None, **kw: (
        _StubJITFunction(fn) if fn is not None else (lambda f: _StubJITFunction(f))
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
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────

_MAPE_DIR        = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k")
_FINE_TUNING_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/fine_tuning_v2")

DEFAULT_TRAIN_FILE = _MAPE_DIR / "correction_train_injection.json"
DEFAULT_OUTPUT_DIR = _FINE_TUNING_DIR / "outputs" / "spacellm_lora_final"

OUTPUT_DIR = _FINE_TUNING_DIR / "outputs"
LOG_DIR    = OUTPUT_DIR / "logs"
GRAPH_DIR  = OUTPUT_DIR / "graphs"
for _d in (LOG_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"correction_finetune_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("SpaceLLM.correction")

# ── Adapter source detection ──────────────────────────────────────────────────

def _is_hf_adapter(adapter_path: str) -> bool:
    """
    Returns True for HF repo ids like "AdityaPS/SpaceLLM_v1".
    Returns False for local filesystem paths (cycle 1+ adapters).

    HF adapters were saved at vocab=200064 (pre-resize).
    Local adapters were saved at vocab=201088 (post-resize, pad_to_multiple_of=64).
    The resize order must differ between the two cases to avoid PEFT shape
    mismatch errors on load.
    """
    p = Path(adapter_path)
    # A local path either exists on disk or starts with / or ./
    if p.exists():
        return False
    if adapter_path.startswith("/") or adapter_path.startswith("./"):
        return False
    # HF repo ids look like "owner/repo" with no path separators beyond one /
    parts = adapter_path.split("/")
    return len(parts) == 2 and not parts[0].startswith(".")


# ── Device-aware CE loss (copied exactly from lora_finetuning_v4.py) ──────────

def _make_device_aware_ce_loss():
    """
    MoE models shard across GPUs. lm_head may land on cuda:1 or cuda:2
    while Trainer places labels on cuda:0. Moves labels to match logits
    before computing CE, then returns loss to cuda:0.
    """
    def _device_aware_ce_loss(logits, labels, vocab_size=None, **kwargs):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_device = shift_logits.device
        if shift_labels.device != logits_device:
            shift_labels = shift_labels.to(logits_device)

        vocab_size = shift_logits.size(-1)

        oob_mask = (shift_labels != -100) & (
            (shift_labels < 0) | (shift_labels >= vocab_size)
        )
        if oob_mask.any():
            shift_labels = shift_labels.clone()
            shift_labels[oob_mask] = -100

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1).long(),
        )
        return loss.to("cuda:0")

    return _device_aware_ce_loss


_DEVICE_AWARE_CE_LOSS = _make_device_aware_ce_loss()


def _inject_loss_function(model, label=""):
    replaced = False
    candidates = [model]
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    for child in list(model.children()):
        candidates.append(child)

    for obj in candidates:
        if obj is not None and hasattr(obj, "loss_function"):
            if getattr(obj, "loss_function") is not _DEVICE_AWARE_CE_LOSS:
                setattr(obj, "loss_function", _DEVICE_AWARE_CE_LOSS)
                replaced = True
                logger.info(
                    f"  ✅ Replaced loss_function on {type(obj).__name__}"
                    + (f" ({label})" if label else "")
                )
    return replaced


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpaceLLM continual correction fine-tuning")
    p.add_argument("--base_model",    default="openai/gpt-oss-20b",
                   help="Base foundation model (MXFP4 checkpoint)")
    p.add_argument("--base_adapter",  default=None,
                   help="HF repo id OR local path of existing LoRA adapter to continue from. "
                        "First cycle: AdityaPS/SpaceLLM_v1. Later cycles: local output path.")
    p.add_argument("--train_file",    default=str(DEFAULT_TRAIN_FILE))
    p.add_argument("--output_dir",    default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--max_seq_len",   type=int,   default=2048)
    p.add_argument("--lora_r",        type=int,   default=64)
    p.add_argument("--lora_alpha",    type=int,   default=128)
    p.add_argument("--lora_dropout",  type=float, default=0.1)
    p.add_argument("--target_modules",default="lm_head")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--warmup_ratio",  type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--min_reference_words", type=int, default=3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--hf_token_env",  default="HF_TOKEN")
    return p.parse_args()


# ── GPU helpers ───────────────────────────────────────────────────────────────

def log_gpu_memory(label: str = ""):
    try:
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
    pct       = 100.0 * trainable / total if total else 0.0
    logger.info("─" * 55)
    logger.info(f"Total parameters     : {total:>15,}")
    logger.info(f"Trainable parameters : {trainable:>15,}  ({pct:.6f}%)")
    logger.info(f"Frozen parameters    : {total - trainable:>15,}")
    logger.info("─" * 55)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"  {name:<60}  shape={str(list(param.shape)):<20}  ({param.numel():,} params)"
            )


# ── Data loading ──────────────────────────────────────────────────────────────

IGNORE_INDEX = -100


def load_correction_examples(train_file: Path, min_reference_words: int) -> list[dict]:
    """
    Load correction_train_injection.json written by execute.py.
    Expected format — list of:
        {
          "messages": [
              {"role": "user",      "content": "<question>"},
              {"role": "assistant", "content": "<human_correction>"}
          ],
          "feedback_id": "...",
          "bertscore":   0.87
        }
    """
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    raw = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {train_file}, got {type(raw)}")

    examples, skipped = [], 0
    for rec in raw:
        messages = rec.get("messages")
        if not messages or len(messages) < 2:
            skipped += 1
            continue
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_turns:
            skipped += 1
            continue
        last_asst = (assistant_turns[-1].get("content") or "").strip()
        if len(last_asst.split()) < min_reference_words:
            skipped += 1
            continue
        examples.append(rec)

    if skipped:
        logger.warning(f"Skipped {skipped}/{len(raw)} record(s) (missing/short assistant content).")
    if not examples:
        raise ValueError("No usable training examples after filtering.")

    logger.info(f"Loaded {len(examples)} usable correction example(s) from {train_file}.")
    return examples


def tokenise_record(record: dict, tokenizer, max_seq_len: int) -> dict | None:
    """
    Tokenize one correction pair. Masks all prompt tokens with IGNORE_INDEX
    so loss is computed only on the corrected assistant answer.
    """
    messages = record.get("messages", [])
    hf_messages = []
    for msg in messages:
        role    = "system" if msg.get("role") == "developer" else msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if content:
            hf_messages.append({"role": role, "content": content})

    if not hf_messages:
        return None

    try:
        full_text = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e} — skipping")
        return None

    full_enc  = tokenizer(full_text, truncation=True, max_length=max_seq_len,
                          padding=False, return_tensors=None)
    input_ids = full_enc["input_ids"]
    if len(input_ids) < 4:
        return None

    prefix_msgs = [m for m in hf_messages if m["role"] != "assistant"]
    try:
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

    prefix_enc = tokenizer(prefix_text, truncation=True, max_length=max_seq_len,
                           padding=False, return_tensors=None)
    prefix_len = len(prefix_enc["input_ids"])

    if prefix_len >= len(input_ids):
        return None

    labels   = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels   = labels[:len(input_ids)]
    n_active = sum(1 for l in labels if l != IGNORE_INDEX)

    if n_active < 4:
        return None

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
        "n_active":       n_active,
    }


def build_dataset(tokenizer, examples: list[dict], max_seq_len: int, vocab_size: int):
    from datasets import Dataset

    tokenised, skipped, clamped = [], 0, 0
    active_counts = []

    for rec in examples:
        result = tokenise_record(rec, tokenizer, max_seq_len)
        if result is None:
            skipped += 1
            continue

        n_active = result.pop("n_active")
        active_counts.append(n_active)

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

        if all(lbl == IGNORE_INDEX for lbl in result["labels"]):
            skipped += 1
            continue

        tokenised.append(result)

    if skipped:
        logger.warning(f"Tokenisation: skipped {skipped}/{len(examples)} examples.")
    if clamped:
        logger.warning(f"Tokenisation: clamped OOV labels in {clamped} examples.")
    if not tokenised:
        raise ValueError("All examples collapsed after tokenisation — cannot train.")

    lengths      = [len(r["input_ids"]) for r in tokenised]
    mean_active  = sum(active_counts) / len(active_counts) if active_counts else 0
    logger.info(
        f"Dataset: {len(tokenised)} rows | "
        f"seq_len min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.0f} | "
        f"active_tokens min={min(active_counts)} mean={mean_active:.1f} max={max(active_counts)}"
    )

    if mean_active < 10:
        logger.warning(f"⚠️  mean active tokens={mean_active:.1f} — label masking may be too aggressive.")

    return Dataset.from_list(tokenised)


# ── NaN guard ─────────────────────────────────────────────────────────────────

def make_nan_guard_callback():
    from transformers import TrainerCallback

    class _NaNGuard(TrainerCallback):
        def __init__(self):
            self.nan_detected = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return control
            loss = logs["loss"]
            if loss is None or (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))):
                logger.error(f"NaN/Inf loss at step {state.global_step} — stopping training.")
                self.nan_detected = True
                control.should_training_stop = True
            return control

    return _NaNGuard()


# ── Vocab resize helper ───────────────────────────────────────────────────────

def _resize_and_realign(model, tokenizer, label: str = "") -> int:
    """
    Resize token embeddings to match tokenizer, pad to multiple of 64,
    re-untie lm_head if resize re-tied it, and return the actual vocab size.
    Always call this on a plain (non-PEFT) model — never on a PeftModel.
    """
    tag = f" ({label})" if label else ""
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = actual_vocab

    # Guard: resize_token_embeddings sometimes re-ties the weights
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        logger.warning(f"  resize re-tied lm_head{tag} — re-untying")
        lm_head = model.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        logger.info(f"  ✅ Re-untied lm_head post-resize{tag}")
    else:
        logger.info(f"  ✅ lm_head independent after resize{tag}")

    assert model.get_input_embeddings().weight.shape[0]  == actual_vocab, \
        f"embed_tokens vocab mismatch after resize{tag}"
    assert model.get_output_embeddings().weight.shape[0] == actual_vocab, \
        f"lm_head vocab mismatch after resize{tag}"
    logger.info(f"  Vocab alignment PASSED{tag}  (vocab={actual_vocab:,}  padded to multiple of 64)")
    return actual_vocab


def _untie_lm_head(model, label: str = "") -> None:
    """Unconditionally materialise lm_head as an independent Parameter."""
    tag = f" ({label})" if label else ""
    model.config.tie_word_embeddings = False
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info(f"✅ lm_head untied and cloned{tag}")


def _post_merge_realign(model, label: str = "") -> None:
    """Re-untie lm_head after merge_and_unload() in case PEFT re-tied it."""
    tag = f" ({label})" if label else ""
    model.config.tie_word_embeddings = False
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        lm_head = model.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        logger.info(f"  Re-untied lm_head post-merge{tag}")
    else:
        logger.info(f"  ✅ lm_head already independent post-merge{tag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Detect adapter source so we can choose the correct resize order
    hf_adapter = _is_hf_adapter(args.base_adapter) if args.base_adapter else False

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Continual Correction Fine-Tuning")
    logger.info(f"  Run ID        : {RUN_ID}")
    logger.info(f"  Base model    : {args.base_model}")
    logger.info(f"  Base adapter  : {args.base_adapter or 'None — fresh LoRA (not recommended)'}")
    logger.info(f"  Adapter source: {'HuggingFace Hub (vocab=200064 → resize AFTER merge)' if hf_adapter else 'local path (vocab=201088 → resize BEFORE load)'}")
    logger.info(f"  Train file    : {args.train_file}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info(f"  LoRA r={args.lora_r}  alpha={args.lora_alpha}  target={args.target_modules}")
    logger.info(f"  Epochs={args.epochs}  LR={args.lr}  batch={args.batch_size}  grad_accum={args.grad_accum}")
    logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    logger.info("=" * 60)

    torch.manual_seed(args.seed)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        DataCollatorForSeq2Seq, Trainer, Mxfp4Config,
    )
    from peft import LoraConfig, TaskType, PeftModel, get_peft_model

    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load correction examples ───────────────────────────────────────────
    examples = load_correction_examples(train_file, args.min_reference_words)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token")
    tokenizer.padding_side = "right"
    logger.info(f"Vocab size: {tokenizer.vocab_size:,}  len(tokenizer): {len(tokenizer):,}")
    logger.info(f"Chat template: {'found' if tokenizer.chat_template else 'NOT FOUND'}")

    # =========================================================================
    # MODEL LOADING PIPELINE
    #
    # The resize ORDER depends on where the adapter was saved:
    #
    #   HF adapter  (SpaceLLM_v1, vocab=200064):
    #       Load base → untie → loss-pre-PEFT → merge(no resize yet) →
    #       post-merge realign → RESIZE → loss-post-merge → fresh LoRA
    #
    #   Local adapter (cycle 1+, vocab=201088):
    #       Load base → untie → RESIZE → loss-pre-PEFT →
    #       merge → post-merge realign → loss-post-merge → fresh LoRA
    #
    # In both cases the final fresh LoRA always operates at vocab=201088.
    # =========================================================================

    # ── Step 1: Load base model to CPU ────────────────────────────────────
    logger.info(f"\nLoading base model to CPU: {args.base_model}  [MXFP4 → BF16 dequantize]")
    log_gpu_memory("before model load")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s  |  dtype: {next(model.parameters()).dtype}")
    model.config.use_cache = False

    # ── Step 2: Untie lm_head (always, before anything else) ──────────────
    logger.info("\n── lm_head untie ────────────────────────────────────")
    _untie_lm_head(model, label="initial")

    # ── Step 3 (local adapter only): resize BEFORE loading adapter ─────────
    # Local adapters were saved at vocab=201088. We must resize the base
    # model to that vocab before PeftModel.from_pretrained() so shapes match.
    if args.base_adapter and not hf_adapter:
        logger.info("\n── Vocab resize (pre-PEFT, local adapter path) ──────")
        _resize_and_realign(model, tokenizer, label="pre-PEFT local")

    # ── Step 4: Loss injection pre-PEFT ───────────────────────────────────
    logger.info("\n── Injecting CE loss (pre-PEFT) ─────────────────────")
    _inject_loss_function(model, label="pre-PEFT")

    # ── Step 5: Load adapter and merge into base weights ───────────────────
    #
    # We always use merge_and_unload() so the fresh correction LoRA
    # (step 8) does NOT stack on top of a live PEFT adapter.
    # Stacking two LoRA adapters on lm_head at different ranks produced
    # NaN/Inf logits (confirmed by the stage-4 diagnostic).
    #
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    if args.base_adapter:
        hf_token = os.environ.get(args.hf_token_env)
        logger.info(f"\n── Loading adapter for merge: {args.base_adapter} ──────")

        with warnings.catch_warnings():
            # Suppress the tie_word_embeddings UserWarning from PEFT —
            # we handle untying manually both before and after merge.
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
            peft_for_merge = PeftModel.from_pretrained(
                model,
                args.base_adapter,
                is_trainable=False,   # read-only — we're about to merge
                token=hf_token,
            )

        logger.info("  PeftModel loaded — merging into base weights ...")
        model = peft_for_merge.merge_and_unload()
        logger.info("✅ merge_and_unload() complete — LoRA baked into base weights")

        # Re-untie lm_head: merge_and_unload() can re-introduce weight tying
        _post_merge_realign(model, label="post-merge")
        _inject_loss_function(model, label="post-merge")

    # ── Step 6 (HF adapter only): resize AFTER merge ──────────────────────
    # HF adapter was at vocab=200064. Now that it is merged and unloaded,
    # the model is a plain AutoModelForCausalLM and we can safely resize.
    # For local adapters this was already done in step 3.
    if hf_adapter or not args.base_adapter:
        logger.info("\n── Vocab resize (post-merge / no-adapter path) ──────")
        _resize_and_realign(model, tokenizer, label="post-merge HF")
        _inject_loss_function(model, label="post-resize")

    # Record the final vocab size — used for dataset OOV clamping
    actual_vocab = model.get_output_embeddings().weight.shape[0]
    logger.info(f"\n  Final vocab: {actual_vocab:,}")

    # ── Step 7: Apply fresh correction LoRA ───────────────────────────────
    # The base model now has SpaceLLM_v1 knowledge merged in.
    # We attach a brand-new LoRA (no stacking, no shape conflicts).
    logger.info("\n── Applying fresh correction LoRA ───────────────────")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    logger.info("✅ get_peft_model() — fresh correction LoRA applied")

    # ── Step 8: enable grads + gradient checkpointing ─────────────────────
    model.enable_input_require_grads()
    logger.info("✅ enable_input_require_grads()")

    model.gradient_checkpointing_enable()
    logger.info("✅ gradient_checkpointing_enable() (post-PEFT)")

    # ── Step 9: Explicit freeze of all non-LoRA params ────────────────────
    logger.info("\n── Explicit parameter freeze ────────────────────────")
    frozen_count, lora_count = 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            lora_count += 1
        else:
            param.requires_grad_(False)
            frozen_count += 1
    logger.info(f"  Frozen : {frozen_count} tensors")
    logger.info(f"  LoRA   : {lora_count} tensors (requires_grad=True)")

    leaked = [(n, p.shape) for n, p in model.named_parameters()
              if p.requires_grad and "lora_" not in n]
    if leaked:
        logger.error("  ❌ Non-LoRA params still trainable — aborting:")
        for n, s in leaked:
            logger.error(f"     {n}  {s}")
        return 1
    logger.info("  ✅ No non-LoRA params leaked as trainable")

    # Loss injection post-PEFT pre-dispatch
    logger.info("\n── CE loss injection (post-PEFT, pre-dispatch) ──────")
    _inject_loss_function(model, label="post-PEFT pre-dispatch")

    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init (CPU)")

    # ── Step 10: GPU dispatch via accelerate ──────────────────────────────
    logger.info("\nDispatching PEFT model across GPUs ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        no_split = []
        for name, module in model.named_modules():
            cls      = type(module)
            cls_name = cls.__name__.lower()
            if (
                issubclass(cls, nn.Module) and cls is not nn.Module
                and ("layer" in cls_name or "block" in cls_name)
                and cls.__name__ not in no_split
                and sum(p.numel() for p in module.parameters()) > 1_000_000
            ):
                no_split.append(cls.__name__)
        no_split = list(dict.fromkeys(no_split))
        logger.info(f"no_split_module_classes : {no_split}")

        n_gpus     = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            free  = torch.cuda.mem_get_info(i)[0]
            alloc = max(0, free - 4 * 1024**3)
            max_memory[i] = f"{int(alloc / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"max_memory per device  : {max_memory}")

        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)

    except Exception as e:
        logger.warning(f"dispatch_model failed ({e}) — falling back to cuda:0")
        model = model.to("cuda:0")

    logger.info(f"GPU dispatch done in {time.time() - t1:.1f}s")
    if hasattr(model, "hf_device_map"):
        from collections import Counter
        dev_counts = Counter(str(v) for v in model.hf_device_map.values())
        for dev, count in sorted(dev_counts.items()):
            logger.info(f"  {dev} : {count} layers")
    log_gpu_memory("after dispatch")

    # ── Step 11: Loss injection post-dispatch ─────────────────────────────
    logger.info("\n── CE loss injection (post-dispatch) ────────────────")
    _inject_loss_function(model, label="post-dispatch")

    # Vocab check post-dispatch
    logger.info("\n── Vocab alignment (post-dispatch) ──────────────────")
    _post_peft_vocab = model.get_output_embeddings().weight.shape[0]
    if _post_peft_vocab != model.config.vocab_size:
        logger.warning(f"  lm_head vocab ({_post_peft_vocab}) != config ({model.config.vocab_size}) — fixing")
        model.config.vocab_size = _post_peft_vocab
    logger.info(f"  lm_head vocab = {_post_peft_vocab:,}  ✅")

    # ── Build dataset ──────────────────────────────────────────────────────
    logger.info("\n── Tokenising correction examples ───────────────────")
    train_dataset = build_dataset(tokenizer, examples, args.max_seq_len, _post_peft_vocab)

    # ── Training arguments ─────────────────────────────────────────────────
    MAX_GRAD_NORM = args.max_grad_norm
    logger.info("\n── Training configuration ───────────────────────────")
    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "_trainer_ckpts"),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = args.warmup_ratio,
        max_grad_norm               = MAX_GRAD_NORM,
        optim                       = "adamw_torch_fused",
        weight_decay                = 0.01,
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 1,
        logging_first_step          = True,
        save_strategy               = "no",
        eval_strategy               = "no",
        report_to                   = [],
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
        seed                        = args.seed,
        gradient_checkpointing      = True,
    )
    logger.info(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
                f"grad_accum={args.grad_accum}  eff_batch={args.batch_size * args.grad_accum}")
    logger.info(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}  "
                f"max_grad_norm={MAX_GRAD_NORM}  optimizer=adamw_torch_fused")

    # ── Data collator ──────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
    )

    # ── NaN guard ──────────────────────────────────────────────────────────
    nan_guard = make_nan_guard_callback()

    # ── DeviceAwareTrainer ─────────────────────────────────────────────────
    class DeviceAwareTrainer(Trainer):
        """
        1. _prepare_inputs() moves labels to lm_head device each step.
        2. training_step() adds explicit clip_grad_norm_ on trainable params.
        """
        def _get_lm_head_device(self):
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            inputs    = super()._prepare_inputs(inputs)
            lm_device = self._get_lm_head_device()
            if lm_device is None:
                return inputs
            if "labels" in inputs and inputs["labels"].device != lm_device:
                inputs["labels"] = inputs["labels"].to(lm_device)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch=None):
            if num_items_in_batch is not None:
                loss = super().training_step(model, inputs, num_items_in_batch)
            else:
                loss = super().training_step(model, inputs)

            trainable_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)

            return loss

    trainer = DeviceAwareTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        processing_class = tokenizer,
        data_collator    = data_collator,
        callbacks        = [nan_guard],
    )

    # Final loss patch post-Trainer init
    logger.info("\n── Final CE loss patch (post-Trainer init) ──────────")
    _inject_loss_function(trainer.model, label="post-Trainer")
    logger.info(f"  lm_head device: {trainer._get_lm_head_device()}")

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Starting correction training ({len(train_dataset)} examples, {args.epochs} epochs)")
    logger.info(f"  Base: {args.base_adapter or 'fresh LoRA'}  →  merged into base weights")
    logger.info(f"  Fresh correction LoRA: r={args.lora_r}  alpha={args.lora_alpha}")
    logger.info("=" * 60)

    t_start = time.time()
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving current state ...")
        interrupted_dir = output_dir / "interrupted"
        trainer.save_model(str(interrupted_dir))
        tokenizer.save_pretrained(str(interrupted_dir))
        logger.info(f"Saved to: {interrupted_dir}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min")

    if nan_guard.nan_detected:
        logger.error("NaN/Inf loss detected — adapter NOT saved to avoid corrupting SpaceLLM.")
        return 1

    # ── Save adapter ────────────────────────────────────────────────────────
    logger.info(f"\nSaving updated adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "run_id":             RUN_ID,
        "base_model":         args.base_model,
        "base_adapter":       args.base_adapter,
        "base_adapter_source": "hf_hub" if hf_adapter else "local",
        "merge_strategy":     "merge_and_unload + fresh_lora",
        "output_dir":         str(output_dir),
        "train_file":         str(train_file),
        "examples_used":      len(train_dataset),
        "epochs":             args.epochs,
        "lr":                 args.lr,
        "max_seq_len":        args.max_seq_len,
        "lora_r":             args.lora_r,
        "lora_alpha":         args.lora_alpha,
        "target_modules":     args.target_modules,
        "vocab_size":         _post_peft_vocab,
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "continual_learning": True,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("  SpaceLLM Correction Fine-Tuning — Complete")
    logger.info("=" * 60)
    logger.info(f"  Adapter saved  →  {output_dir}")
    logger.info(f"  Log            →  {LOG_FILE}")
    logger.info(f"  Next cycle use →  --base_adapter {output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
