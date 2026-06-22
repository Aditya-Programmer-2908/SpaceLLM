"""
SpaceLLM :: Continual Correction Fine-Tuning
==============================================
KEY BUG FIXED (this version):
    _resize_and_realign() was calling resize_token_embeddings() on an untied
    lm_head. Because lm_head was already detached as an independent Parameter
    BEFORE resize, resize_token_embeddings() only grew embed_tokens and left
    lm_head at vocab=200,064. The fresh correction LoRA was then wrapped around
    the 200,064-dim lm_head, saving lora_B with shape [200064, 64].
    At inference, detect_adapter_vocab_size() correctly read 200,064 from lora_B
    and resized the base model to 200,064 — but the embed_tokens was now at
    200,064 too, so all new-token embeddings (rows 200064..201087) were missing,
    causing token-id-0 degenerate output ("!!!...").

    FIX: resize_token_embeddings() must be called BEFORE untying lm_head.
    The function internally copies the tied weight to both embed_tokens and
    lm_head when they share the same tensor. After resize, THEN we untie.
    This ensures both tables grow to 201,088 before lm_head is detached.

    Correct order (both HF and local adapter paths):
        1. Load base model (weights tied by default)
        2. resize_token_embeddings()  ← both embed_tokens AND lm_head grow
        3. _untie_lm_head()           ← now safe to detach as independent param
        4. loss injection
        5. PeftModel / merge / fresh LoRA  (all at vocab=201,088)
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
    p = Path(adapter_path)
    if p.exists():
        return False
    if adapter_path.startswith("/") or adapter_path.startswith("./"):
        return False
    parts = adapter_path.split("/")
    return len(parts) == 2 and not parts[0].startswith(".")


# ── Device-aware CE loss ──────────────────────────────────────────────────────

def _make_device_aware_ce_loss():
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
    p.add_argument("--base_model",    default="openai/gpt-oss-20b")
    p.add_argument("--base_adapter",  default=None)
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


# ── Vocab helpers ─────────────────────────────────────────────────────────────

def _resize_then_untie(model, tokenizer, label: str = "") -> int:
    """
    *** THE CORRECT ORDER ***

    resize_token_embeddings() must be called while lm_head is STILL TIED to
    embed_tokens (i.e. they share the same tensor object). Only then does the
    resize grow BOTH tables atomically. Untying afterwards is safe because both
    tables are already at the new size.

    Calling untie BEFORE resize leaves lm_head at the old vocab size, which is
    exactly the bug that caused lora_B to be saved as [200064, 64] instead of
    [201088, 64].
    """
    tag = f" ({label})" if label else ""

    # ── 1. Verify tied state before resize ────────────────────────────────
    tied_before = (
        id(model.get_input_embeddings().weight) ==
        id(model.get_output_embeddings().weight)
    )
    logger.info(f"  lm_head tied before resize{tag}: {tied_before}")
    if not tied_before:
        # Already untied — resize only grows embed_tokens, not lm_head.
        # We must manually grow lm_head too.
        logger.warning(
            f"  ⚠️  lm_head already untied before resize{tag} — "
            "will resize lm_head manually after embed_tokens resize."
        )

    # ── 2. Resize (grows both tables if tied; only embed_tokens if not) ───
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = actual_vocab

    # ── 3. If was untied, manually resize lm_head to match embed_tokens ──
    if not tied_before:
        embed_vocab = model.get_input_embeddings().weight.shape[0]
        lm_head     = model.get_output_embeddings()
        if lm_head.weight.shape[0] != embed_vocab:
            logger.warning(
                f"  lm_head vocab {lm_head.weight.shape[0]:,} != "
                f"embed_tokens vocab {embed_vocab:,} — manually resizing lm_head"
            )
            old_weight = lm_head.weight.data
            new_weight = torch.zeros(
                embed_vocab, old_weight.shape[1],
                dtype=old_weight.dtype, device=old_weight.device
            )
            copy_rows = min(old_weight.shape[0], embed_vocab)
            new_weight[:copy_rows] = old_weight[:copy_rows]
            lm_head.weight = nn.Parameter(new_weight)
            actual_vocab = embed_vocab
            model.config.vocab_size = actual_vocab
            logger.info(f"  ✅ lm_head manually resized to {actual_vocab:,}")

    # ── 4. NOW untie lm_head (both tables are at the same size) ──────────
    model.config.tie_word_embeddings = False
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        lm_head = model.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        logger.info(f"  ✅ lm_head untied post-resize{tag}")
    else:
        logger.info(f"  ✅ lm_head already independent post-resize{tag}")

    # ── 5. Verify both tables match ───────────────────────────────────────
    embed_v = model.get_input_embeddings().weight.shape[0]
    lmhd_v  = model.get_output_embeddings().weight.shape[0]
    assert embed_v == actual_vocab, f"embed_tokens mismatch: {embed_v} != {actual_vocab}"
    assert lmhd_v  == actual_vocab, f"lm_head mismatch: {lmhd_v} != {actual_vocab}"
    logger.info(
        f"  ✅ Vocab alignment PASSED{tag}  "
        f"embed_tokens={embed_v:,}  lm_head={lmhd_v:,}  "
        f"(padded to multiple of 64)"
    )
    return actual_vocab


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


# ── Data loading ──────────────────────────────────────────────────────────────

IGNORE_INDEX = -100


def load_correction_examples(train_file: Path, min_reference_words: int) -> list[dict]:
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
    lengths     = [len(r["input_ids"]) for r in tokenised]
    mean_active = sum(active_counts) / len(active_counts) if active_counts else 0
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    hf_adapter = _is_hf_adapter(args.base_adapter) if args.base_adapter else False

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Continual Correction Fine-Tuning (FIXED)")
    logger.info(f"  Run ID        : {RUN_ID}")
    logger.info(f"  Base model    : {args.base_model}")
    logger.info(f"  Base adapter  : {args.base_adapter or 'None'}")
    logger.info(f"  Adapter source: {'HuggingFace Hub' if hf_adapter else 'local path'}")
    logger.info(f"  Train file    : {args.train_file}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info(f"  LoRA r={args.lora_r}  alpha={args.lora_alpha}  target={args.target_modules}")
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

    examples = load_correction_examples(train_file, args.min_reference_words)

    logger.info(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    logger.info(f"Tokenizer: vocab={tokenizer.vocab_size:,}  len={len(tokenizer):,}")

    # =========================================================================
    # MODEL LOADING PIPELINE
    #
    # CRITICAL RULE: resize_token_embeddings() must be called while lm_head
    # is STILL TIED (shares tensor with embed_tokens). Untie AFTER resize.
    #
    # HF adapter (SpaceLLM_v1, saved at vocab=200064):
    #   Load base → PeftModel.from_pretrained (200064==200064) →
    #   merge_and_unload → post-merge check → RESIZE+UNTIE →
    #   loss inject → fresh LoRA
    #
    # Local adapter (cycle 1+, saved at vocab=201088):
    #   Load base → RESIZE+UNTIE → loss inject →
    #   PeftModel.from_pretrained (201088==201088) →
    #   merge_and_unload → post-merge check → fresh LoRA
    # =========================================================================

    logger.info(f"\nLoading base model to CPU: {args.base_model}")
    log_gpu_memory("before model load")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s  |  dtype: {next(model.parameters()).dtype}")
    model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    hf_token = os.environ.get(args.hf_token_env)

    # ── HF adapter path ───────────────────────────────────────────────────
    if args.base_adapter and hf_adapter:
        logger.info("\n── [HF adapter] Loading PEFT at base vocab (200064) ────")
        _inject_loss_function(model, label="pre-PEFT HF")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
            peft_for_merge = PeftModel.from_pretrained(
                model, args.base_adapter, is_trainable=False, token=hf_token)

        logger.info("  Merging into base weights ...")
        model = peft_for_merge.merge_and_unload()
        logger.info("✅ merge_and_unload() complete")
        _post_merge_realign(model, label="post-merge HF")
        _inject_loss_function(model, label="post-merge HF")

        logger.info("\n── [HF adapter] Resize+Untie (post-merge) ───────────────")
        actual_vocab = _resize_then_untie(model, tokenizer, label="post-merge HF")
        _inject_loss_function(model, label="post-resize HF")

    # ── Local adapter path ────────────────────────────────────────────────
    elif args.base_adapter and not hf_adapter:
        logger.info("\n── [Local adapter] Resize+Untie BEFORE loading PEFT ────")
        actual_vocab = _resize_then_untie(model, tokenizer, label="pre-PEFT local")
        _inject_loss_function(model, label="pre-PEFT local")

        logger.info(f"\n── [Local adapter] Loading PEFT at vocab={actual_vocab:,} ──")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
            peft_for_merge = PeftModel.from_pretrained(
                model, args.base_adapter, is_trainable=False, token=hf_token)

        logger.info("  Merging into base weights ...")
        model = peft_for_merge.merge_and_unload()
        logger.info("✅ merge_and_unload() complete")
        _post_merge_realign(model, label="post-merge local")
        _inject_loss_function(model, label="post-merge local")

    # ── No adapter (fresh LoRA only) ──────────────────────────────────────
    else:
        logger.info("\n── [No adapter] Resize+Untie ────────────────────────────")
        actual_vocab = _resize_then_untie(model, tokenizer, label="no-adapter")
        _inject_loss_function(model, label="no-adapter")

    # ── Verify final vocab before LoRA ────────────────────────────────────
    actual_vocab = model.get_output_embeddings().weight.shape[0]
    embed_vocab  = model.get_input_embeddings().weight.shape[0]
    logger.info(f"\n  Pre-LoRA vocab check: embed_tokens={embed_vocab:,}  lm_head={actual_vocab:,}")
    assert embed_vocab == actual_vocab, (
        f"FATAL: embed_tokens ({embed_vocab:,}) != lm_head ({actual_vocab:,}) before LoRA — "
        "this would save lora_B at the wrong vocab size!"
    )
    logger.info(f"  ✅ Both tables at {actual_vocab:,} — safe to apply LoRA")

    # ── Fresh correction LoRA ─────────────────────────────────────────────
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

    # ── Verify LoRA was created with the right vocab ───────────────────────
    for name, param in model.named_parameters():
        if "lm_head" in name and "lora_B" in name:
            logger.info(f"  lora_B shape: {list(param.shape)}  (expect [{actual_vocab}, {args.lora_r}])")
            assert param.shape[0] == actual_vocab, (
                f"lora_B vocab mismatch: {param.shape[0]} != {actual_vocab}"
            )
            logger.info(f"  ✅ lora_B vocab confirmed at {actual_vocab:,}")
            break

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    logger.info("\n── Explicit parameter freeze ────────────────────────")
    frozen_count, lora_count = 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            lora_count += 1
        else:
            param.requires_grad_(False)
            frozen_count += 1
    logger.info(f"  Frozen: {frozen_count}  LoRA: {lora_count}")

    leaked = [(n, p.shape) for n, p in model.named_parameters()
              if p.requires_grad and "lora_" not in n]
    if leaked:
        logger.error("  ❌ Non-LoRA params still trainable — aborting")
        for n, s in leaked:
            logger.error(f"     {n}  {s}")
        return 1
    logger.info("  ✅ No non-LoRA params leaked")

    _inject_loss_function(model, label="post-PEFT pre-dispatch")
    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init (CPU)")

    # ── GPU dispatch ──────────────────────────────────────────────────────
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
        logger.info(f"no_split_module_classes: {no_split}")
        n_gpus     = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            free  = torch.cuda.mem_get_info(i)[0]
            alloc = max(0, free - 4 * 1024**3)
            max_memory[i] = f"{int(alloc / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"max_memory: {max_memory}")
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)
    except Exception as e:
        logger.warning(f"dispatch_model failed ({e}) — falling back to cuda:0")
        model = model.to("cuda:0")

    logger.info(f"GPU dispatch done in {time.time()-t1:.1f}s")
    if hasattr(model, "hf_device_map"):
        from collections import Counter
        dev_counts = Counter(str(v) for v in model.hf_device_map.values())
        for dev, count in sorted(dev_counts.items()):
            logger.info(f"  {dev}: {count} layers")
    log_gpu_memory("after dispatch")

    _inject_loss_function(model, label="post-dispatch")

    _post_peft_vocab = model.get_output_embeddings().weight.shape[0]
    if _post_peft_vocab != model.config.vocab_size:
        logger.warning(f"  lm_head vocab ({_post_peft_vocab}) != config ({model.config.vocab_size}) — fixing")
        model.config.vocab_size = _post_peft_vocab
    logger.info(f"  lm_head vocab post-dispatch = {_post_peft_vocab:,}  ✅")

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("\n── Tokenising correction examples ───────────────────")
    train_dataset = build_dataset(tokenizer, examples, args.max_seq_len, _post_peft_vocab)

    MAX_GRAD_NORM = args.max_grad_norm
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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        pad_to_multiple_of=64, label_pad_token_id=IGNORE_INDEX,
    )
    nan_guard = make_nan_guard_callback()

    class DeviceAwareTrainer(Trainer):
        def _get_lm_head_device(self):
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            inputs    = super()._prepare_inputs(inputs)
            lm_device = self._get_lm_head_device()
            if lm_device and "labels" in inputs and inputs["labels"].device != lm_device:
                inputs["labels"] = inputs["labels"].to(lm_device)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = (
                super().training_step(model, inputs, num_items_in_batch)
                if num_items_in_batch is not None
                else super().training_step(model, inputs)
            )
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
            return loss

    trainer = DeviceAwareTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer, data_collator=data_collator, callbacks=[nan_guard],
    )
    _inject_loss_function(trainer.model, label="post-Trainer")
    logger.info(f"  lm_head device: {trainer._get_lm_head_device()}")

    logger.info("=" * 60)
    logger.info(f"  Starting training ({len(train_dataset)} examples, {args.epochs} epochs)")
    logger.info(f"  Fresh correction LoRA: r={args.lora_r}  alpha={args.lora_alpha}")
    logger.info(f"  lora_B vocab = {actual_vocab:,}  ← this must appear in detect_adapter_vocab_size()")
    logger.info("=" * 60)

    t_start = time.time()
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving ...")
        interrupted_dir = output_dir / "interrupted"
        trainer.save_model(str(interrupted_dir))
        tokenizer.save_pretrained(str(interrupted_dir))
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    logger.info(f"Training complete in {(time.time()-t_start)/60:.1f} min")

    if nan_guard.nan_detected:
        logger.error("NaN/Inf detected — NOT saving adapter.")
        return 1

    logger.info(f"\nSaving adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "run_id":              RUN_ID,
        "base_model":          args.base_model,
        "base_adapter":        args.base_adapter,
        "base_adapter_source": "hf_hub" if hf_adapter else "local",
        "merge_strategy":      "merge_and_unload + fresh_lora",
        "output_dir":          str(output_dir),
        "train_file":          str(train_file),
        "examples_used":       len(train_dataset),
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "max_seq_len":         args.max_seq_len,
        "lora_r":              args.lora_r,
        "lora_alpha":          args.lora_alpha,
        "target_modules":      args.target_modules,
        "vocab_size":          actual_vocab,   # ← correct: set before LoRA, verified by assertion
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "continual_learning":  True,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("  SpaceLLM Correction Fine-Tuning — Complete")
    logger.info(f"  Adapter saved  →  {output_dir}")
    logger.info(f"  vocab_size in metadata = {actual_vocab:,}  ✅")
    logger.info(f"  Next cycle:  --base_adapter {output_dir}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
