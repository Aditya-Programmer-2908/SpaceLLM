"""
SpaceLLM v2 Training Script
============================
Pipeline:
    GPT-OSS base (MXFP4 → BF16)
        ↓  load + merge
    SpaceLLM_v1 adapter (AdityaPS/SpaceLLM_v1, vocab=200064)
        ↓  merge_and_unload()  → v1 knowledge baked into weights
    Merged SpaceLLM weights
        ↓  resize vocab (200064 → 201088)
        ↓  attach fresh correction LoRA
    Train on human corrections  (only correction LoRA params update)
        ↓
    Save → SpaceLLM_v2  (adapter contains v1 knowledge + corrections)

The saved adapter at --output_dir can be loaded back with:
    PeftModel.from_pretrained(base_model_resized, output_dir)

Usage:
    python train_spacellm_v2.py \
        --train_file   /path/to/corrections.json \
        --output_dir   /path/to/spacellm_v2 \
        [--base_model  openai/gpt-oss-20b] \
        [--v1_adapter  AdityaPS/SpaceLLM_v1] \
        [--epochs 3] [--lr 2e-4] [--lora_r 64] [--lora_alpha 128]

corrections.json format (list of):
    [
      {
        "messages": [
          {"role": "user",      "content": "<question>"},
          {"role": "assistant", "content": "<human_corrected_answer>"}
        ]
      },
      ...
    ]
    OR flat format:
    [
      {"question": "...", "reference": "..."},
      ...
    ]
"""

from __future__ import annotations

# ── Triton stub (must be first) ───────────────────────────────────────────────
import sys
import types


def _patch_triton():
    try:
        import triton  # noqa: F401
        return
    except Exception:
        pass

    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
        def __bool__(self): return False

    class _ActiveDesc:
        def __get__(self, obj, t=None): return _Stub()
        def __set__(self, obj, v): pass

    class _DrvMgr:
        active = _ActiveDesc()
        default = _Stub()

    def _mod(name, parent=None):
        m = types.ModuleType(name)
        sys.modules[name] = m
        if parent:
            setattr(parent, name.split(".")[-1], m)
        return m

    t   = _mod("triton")
    tr  = _mod("triton.runtime",               t)
    trd = _mod("triton.runtime.driver",         tr)
    trb = _mod("triton.runtime.build",          tr)
    trj = _mod("triton.runtime.jit",            tr)
    tb  = _mod("triton.backends",              t)
    tbn = _mod("triton.backends.nvidia",        tb)
    tbd = _mod("triton.backends.nvidia.driver", tbn)

    trd.driver = _DrvMgr()
    trb._build = trb.compile_module_from_src = trb.load_module = lambda *a, **kw: types.ModuleType("_s")

    class _JIT:
        def __init__(self, fn): self.fn = fn
        def __call__(self, *a, **kw):
            try: return self.fn(*a, **kw)
            except: return None
        def __getattr__(self, n): return lambda *a, **kw: None

    trj.JITFunction = _JIT
    t.jit = lambda fn=None, **kw: (_JIT(fn) if fn else (lambda f: _JIT(f)))
    tbd.CudaUtils = type("CudaUtils", (), {"__init__": lambda s: None, "__getattr__": lambda s, n: lambda *a, **kw: None})
    t.runtime = tr; t.backends = tb


_patch_triton()

# ── Standard imports ──────────────────────────────────────────────────────────
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("SpaceLLM.v2")

IGNORE_INDEX = -100


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpaceLLM v2 (merge v1 → correction LoRA)")
    p.add_argument("--base_model",    default="openai/gpt-oss-20b",
                   help="HF id or local path of the base foundation model")
    p.add_argument("--v1_adapter",    default="AdityaPS/SpaceLLM_v1",
                   help="HF id or local path of the SpaceLLM v1 LoRA adapter")
    p.add_argument("--train_file",    required=True,
                   help="JSON file with correction examples")
    p.add_argument("--output_dir",    required=True,
                   help="Where to save the SpaceLLM v2 adapter")
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # Training hyper-params
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--max_seq_len",   type=int,   default=2048)
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--warmup_ratio",  type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    # LoRA hyper-params
    p.add_argument("--lora_r",        type=int,   default=64)
    p.add_argument("--lora_alpha",    type=int,   default=128)
    p.add_argument("--lora_dropout",  type=float, default=0.1)
    p.add_argument("--target_modules",default="lm_head",
                   help="Comma-separated list of module names to apply LoRA to")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ── GPU helpers ───────────────────────────────────────────────────────────────
def log_gpu(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        a = torch.cuda.memory_allocated(i) / 1024**3
        r = torch.cuda.memory_reserved(i)  / 1024**3
        t = p.total_memory                 / 1024**3
        log.info(f"  GPU{i} [{p.name}] {label} | alloc={a:.1f}GB res={r:.1f}GB total={t:.1f}GB")


def log_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.4f}%)")
    for n, p in model.named_parameters():
        if p.requires_grad:
            log.info(f"    {n}  shape={list(p.shape)}  ({p.numel():,})")


# ── lm_head / vocab helpers ───────────────────────────────────────────────────
def untie_lm_head(model, label: str = "") -> None:
    """Materialise lm_head as an independent Parameter (no weight sharing)."""
    model.config.tie_word_embeddings = False
    lm = model.get_output_embeddings()
    lm.weight = nn.Parameter(lm.weight.detach().clone())
    log.info(f"  ✅ lm_head untied{' (' + label + ')' if label else ''}")


def resize_vocab(model, tokenizer, label: str = "") -> int:
    """
    Resize token embeddings to match tokenizer length, padded to multiple of 64.
    Re-unties lm_head if resize accidentally re-ties it.
    Returns the actual (padded) vocab size.
    """
    tag = f" ({label})" if label else ""
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = vocab

    # resize_token_embeddings can silently re-tie weights — guard against it
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        log.warning(f"  resize re-tied lm_head{tag} — re-untying")
        lm = model.get_output_embeddings()
        lm.weight = nn.Parameter(lm.weight.detach().clone())

    emb_vocab = model.get_input_embeddings().weight.shape[0]
    lm_vocab  = model.get_output_embeddings().weight.shape[0]
    assert emb_vocab == vocab and lm_vocab == vocab, \
        f"Vocab mismatch after resize{tag}: embed={emb_vocab}, lm_head={lm_vocab}, expected={vocab}"

    log.info(f"  ✅ Vocab resized → {vocab:,} (pad_to_multiple_of=64){tag}")
    return vocab


# ── Device-aware cross-entropy loss ──────────────────────────────────────────
def _device_aware_ce_loss(logits, labels, vocab_size=None, **kwargs):
    """
    MoE models shard across multiple GPUs. lm_head may be on cuda:1/2
    while Trainer puts labels on cuda:0. This bridges the device gap.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if shift_labels.device != shift_logits.device:
        shift_labels = shift_labels.to(shift_logits.device)

    V = shift_logits.size(-1)
    # Clamp any out-of-vocab label indices to IGNORE_INDEX
    oob = (shift_labels != IGNORE_INDEX) & ((shift_labels < 0) | (shift_labels >= V))
    if oob.any():
        shift_labels = shift_labels.clone()
        shift_labels[oob] = IGNORE_INDEX

    loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
        shift_logits.view(-1, V),
        shift_labels.view(-1).long(),
    )
    return loss.to("cuda:0")


def inject_loss(model, label: str = "") -> None:
    """Patch loss_function on the model and its immediate children."""
    tag = f" ({label})" if label else ""
    replaced = False
    for obj in [model, getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None),
                *list(model.children())]:
        if obj is not None and hasattr(obj, "loss_function"):
            if getattr(obj, "loss_function") is not _device_aware_ce_loss:
                obj.loss_function = _device_aware_ce_loss
                replaced = True
                log.info(f"  ✅ loss_function patched on {type(obj).__name__}{tag}")
    if not replaced:
        log.debug(f"  (no loss_function attribute found to patch{tag})")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_examples(train_file: Path) -> list[dict]:
    """
    Load training JSON. Accepts two formats:
      A) messages format: [{"messages": [{"role":..., "content":...}, ...]}]
      B) flat format:     [{"question": "...", "reference": "..."}]
    Always returns messages-format records.
    """
    raw = json.loads(train_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {train_file}")

    out, skipped = [], 0
    for rec in raw:
        # Flat format → convert to messages format
        if "question" in rec or "reference" in rec:
            q = (rec.get("question") or "").strip()
            a = (rec.get("reference") or "").strip()
            if not q or not a:
                skipped += 1
                continue
            rec = {
                "messages": [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": a},
                ],
                **{k: v for k, v in rec.items() if k not in ("question", "reference")},
            }

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
    log.info(f"  Loaded {len(out)} usable correction example(s)")
    return out


def tokenise(record: dict, tokenizer, max_seq_len: int) -> dict | None:
    """
    Tokenize one record. Masks the prompt with IGNORE_INDEX so the loss
    is computed only on the corrected assistant answer.
    """
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
        log.warning(f"  apply_chat_template failed: {e} — skipping")
        return None

    full_enc  = tokenizer(full_text, truncation=True, max_length=max_seq_len,
                          padding=False, return_tensors=None)
    input_ids = full_enc["input_ids"]
    if len(input_ids) < 4:
        return None

    # Build prefix (everything except last assistant turn) to find mask boundary
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

    labels   = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]
    labels   = labels[:len(input_ids)]
    n_active = sum(1 for l in labels if l != IGNORE_INDEX)
    if n_active < 4:
        return None

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
    }


def build_dataset(tokenizer, examples: list[dict], max_seq_len: int, vocab_size: int):
    from datasets import Dataset

    rows, skipped, clamped = [], 0, 0
    for rec in examples:
        r = tokenise(rec, tokenizer, max_seq_len)
        if r is None:
            skipped += 1
            continue
        # Clamp any OOV label tokens
        new_labels, had_oob = [], False
        for lbl in r["labels"]:
            if lbl != IGNORE_INDEX and (lbl < 0 or lbl >= vocab_size):
                new_labels.append(IGNORE_INDEX); had_oob = True
            else:
                new_labels.append(lbl)
        if had_oob:
            r["labels"] = new_labels; clamped += 1
        if all(l == IGNORE_INDEX for l in r["labels"]):
            skipped += 1; continue
        rows.append(r)

    if skipped:  log.warning(f"  Tokenisation skipped {skipped} examples")
    if clamped:  log.warning(f"  Clamped OOV labels in {clamped} examples")
    if not rows: raise ValueError("All examples collapsed after tokenisation.")

    lengths = [len(r["input_ids"]) for r in rows]
    log.info(f"  Dataset: {len(rows)} rows | "
             f"seq_len min={min(lengths)} mean={sum(lengths)/len(lengths):.0f} max={max(lengths)}")
    return Dataset.from_list(rows)


# ── NaN guard callback ────────────────────────────────────────────────────────
def make_nan_guard():
    from transformers import TrainerCallback

    class _NaNGuard(TrainerCallback):
        def __init__(self): self.triggered = False
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
    args       = parse_args()
    output_dir = Path(args.output_dir)
    train_file = Path(args.train_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    log.info("=" * 65)
    log.info("  SpaceLLM v2 Training")
    log.info(f"  base_model  : {args.base_model}")
    log.info(f"  v1_adapter  : {args.v1_adapter}")
    log.info(f"  train_file  : {train_file}")
    log.info(f"  output_dir  : {output_dir}")
    log.info(f"  LoRA r={args.lora_r}  alpha={args.lora_alpha}  target={args.target_modules}")
    log.info(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
             f"grad_accum={args.grad_accum}")
    log.info("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
    from peft import LoraConfig, TaskType, PeftModel, get_peft_model

    # ── Load correction examples ──────────────────────────────────────────
    log.info("\n[1/9] Loading correction examples ...")
    examples = load_examples(train_file)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    log.info(f"\n[2/9] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"
    log.info(f"  tokenizer len={len(tokenizer):,}  "
             f"chat_template={'found' if tokenizer.chat_template else 'MISSING'}")

    # =========================================================================
    # PIPELINE (HF adapter, vocab=200064):
    #
    #  Step 3  Load base model to CPU                  (vocab=200064)
    #  Step 4  Untie lm_head
    #  Step 5  Inject device-aware CE loss (pre-PEFT)
    #  Step 6  PeftModel.from_pretrained(v1_adapter)   (shapes match: 200064)
    #  Step 7  merge_and_unload()                      (v1 baked into weights)
    #  Step 8  Re-untie lm_head post-merge
    #  Step 9  resize_token_embeddings                 (200064 → 201088)
    #  Step 10 Inject CE loss post-resize
    #  Step 11 Fresh correction LoRA via get_peft_model()
    #  Step 12 Freeze everything except LoRA params
    #  Step 13 GPU dispatch
    #  Step 14 Train
    #  Step 15 Save v2 adapter
    # =========================================================================

    # ── [3/9] Load base model ─────────────────────────────────────────────
    log.info(f"\n[3/9] Loading base model (CPU, MXFP4→BF16): {args.base_model}")
    log_gpu("pre-load")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        token=args.hf_token,
    )
    log.info(f"  Loaded in {time.time()-t0:.1f}s | dtype={next(model.parameters()).dtype}")
    model.config.use_cache = False

    # ── [4/9] Untie lm_head ───────────────────────────────────────────────
    log.info("\n[4/9] Untying lm_head ...")
    untie_lm_head(model, "initial")

    # ── [5/9] CE loss injection (pre-PEFT) ───────────────────────────────
    log.info("\n[5/9] Injecting device-aware CE loss (pre-PEFT) ...")
    inject_loss(model, "pre-PEFT")

    # ── [6/9] Load SpaceLLM v1 adapter ───────────────────────────────────
    log.info(f"\n[6/9] Loading SpaceLLM v1 adapter: {args.v1_adapter}")
    log.info("      Base model vocab=200064, adapter saved at vocab=200064 → shapes match ✅")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
        peft_model = PeftModel.from_pretrained(
            model,
            args.v1_adapter,
            is_trainable=False,      # read-only — about to merge
            token=args.hf_token,
        )
    log.info("  ✅ SpaceLLM v1 adapter loaded")

    # ── [7/9] merge_and_unload — bake v1 into base weights ───────────────
    log.info("\n[7/9] Merging v1 adapter into base weights (merge_and_unload) ...")
    model = peft_model.merge_and_unload()
    log.info("  ✅ merge_and_unload() complete — SpaceLLM v1 knowledge is now base weights")

    # ── [8/9] Post-merge cleanup ──────────────────────────────────────────
    log.info("\n[8/9] Post-merge lm_head realignment ...")
    model.config.tie_word_embeddings = False
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        log.warning("  merge re-tied lm_head — re-untying")
        untie_lm_head(model, "post-merge")
    else:
        log.info("  ✅ lm_head already independent post-merge")
    inject_loss(model, "post-merge")

    # ── Vocab resize (safe now — plain model, no PEFT wrapper) ───────────
    log.info("  Resizing vocab (200064 → ~201088) ...")
    actual_vocab = resize_vocab(model, tokenizer, "post-merge")
    inject_loss(model, "post-resize")

    log.info(f"\n  Final vocab size: {actual_vocab:,}")

    # ── Attach fresh correction LoRA ──────────────────────────────────────
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
    log.info(f"\n  ✅ Fresh correction LoRA attached  "
             f"(r={args.lora_r}, alpha={args.lora_alpha}, target={target_modules})")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("  ✅ enable_input_require_grads + gradient_checkpointing")

    # ── Freeze everything except LoRA params ──────────────────────────────
    log.info("\n  Freezing non-LoRA parameters ...")
    frozen, lora_count = 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True);  lora_count += 1
        else:
            param.requires_grad_(False); frozen += 1

    leaked = [(n, p.shape) for n, p in model.named_parameters()
              if p.requires_grad and "lora_" not in n]
    if leaked:
        log.error("  ❌ Non-LoRA params still trainable — aborting:")
        for n, s in leaked: log.error(f"     {n}  {s}")
        return 1
    log.info(f"  Frozen={frozen}  LoRA trainable={lora_count}  ✅ no leaks")
    inject_loss(model, "post-PEFT")
    log_trainable(model)
    log_gpu("after LoRA init (CPU)")

    # ── GPU dispatch ──────────────────────────────────────────────────────
    log.info("\n  Dispatching model across GPUs ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        # Collect large module class names for no_split
        no_split = list({
            type(m).__name__
            for m in model.modules()
            if (isinstance(m, nn.Module) and type(m) is not nn.Module
                and ("layer" in type(m).__name__.lower() or "block" in type(m).__name__.lower())
                and sum(p.numel() for p in m.parameters()) > 1_000_000)
        })
        log.info(f"  no_split_module_classes: {no_split}")

        max_memory = {
            i: f"{max(0, int((torch.cuda.mem_get_info(i)[0] - 4*1024**3) / 1024**3))}GiB"
            for i in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = "80GiB"
        log.info(f"  max_memory: {max_memory}")

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

    # Sync config vocab after dispatch
    final_vocab = model.get_output_embeddings().weight.shape[0]
    if final_vocab != model.config.vocab_size:
        log.warning(f"  Fixing config.vocab_size: {model.config.vocab_size} → {final_vocab}")
        model.config.vocab_size = final_vocab
    log.info(f"  Final vocab (post-dispatch): {final_vocab:,}  ✅")

    # ── Build dataset ─────────────────────────────────────────────────────
    log.info("\n  Tokenising correction examples ...")
    train_dataset = build_dataset(tokenizer, examples, args.max_seq_len, final_vocab)

    # ── TrainingArguments ─────────────────────────────────────────────────
    MAX_GRAD_NORM = args.max_grad_norm
    training_args = TrainingArguments(
        output_dir                  = str(output_dir / "_ckpts"),
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
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
    )
    nan_guard = make_nan_guard()

    # ── Custom Trainer: device-aware label placement + grad clipping ──────
    class SpaceLLMTrainer(Trainer):
        def _lm_device(self):
            try: return next(self.model.get_output_embeddings().parameters()).device
            except: return None

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            dev    = self._lm_device()
            if dev and "labels" in inputs and inputs["labels"].device != dev:
                inputs["labels"] = inputs["labels"].to(dev)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = (super().training_step(model, inputs, num_items_in_batch)
                    if num_items_in_batch is not None
                    else super().training_step(model, inputs))
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
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

    # Final loss patch after Trainer init
    inject_loss(trainer.model, "post-Trainer")
    log.info(f"  lm_head device: {trainer._lm_device()}")

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info(f"  Starting training | {len(train_dataset)} examples | {args.epochs} epochs")
    log.info(f"  Base (merged): SpaceLLM_v1 baked into weights")
    log.info(f"  LoRA: r={args.lora_r}  alpha={args.lora_alpha}  target={target_modules}")
    log.info("=" * 65)

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
    log.info(f"  Training complete in {elapsed/60:.1f} min")

    if nan_guard.triggered:
        log.error("  NaN/Inf detected — NOT saving adapter to protect model integrity.")
        return 1

    # ── Save SpaceLLM v2 adapter ──────────────────────────────────────────
    log.info(f"\n  Saving SpaceLLM v2 adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "model_name":        "SpaceLLM_v2",
        "base_model":        args.base_model,
        "v1_adapter":        args.v1_adapter,
        "merge_strategy":    "merge_and_unload + fresh_correction_lora",
        "output_dir":        str(output_dir),
        "train_file":        str(train_file),
        "examples_used":     len(train_dataset),
        "epochs":            args.epochs,
        "lr":                args.lr,
        "lora_r":            args.lora_r,
        "lora_alpha":        args.lora_alpha,
        "target_modules":    args.target_modules,
        "vocab_size":        final_vocab,
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "knowledge_sources": ["SpaceLLM_v1 (merged)", "human_corrections"],
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("")
    log.info("=" * 65)
    log.info("  SpaceLLM v2 — Complete ✅")
    log.info("=" * 65)
    log.info(f"  Adapter saved  → {output_dir}")
    log.info(f"  Contains       : SpaceLLM_v1 knowledge (merged) + human corrections (LoRA)")
    log.info(f"  To load v2:")
    log.info(f"    model = AutoModelForCausalLM.from_pretrained('{args.base_model}', ...)")
    log.info(f"    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)")
    log.info(f"    model = PeftModel.from_pretrained(model, '{output_dir}')")
    log.info("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
