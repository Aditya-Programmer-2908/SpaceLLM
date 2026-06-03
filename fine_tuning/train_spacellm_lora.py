"""
SpaceLLM — Experimental LoRA Fine-Tuning
==========================================
Model     : openai/gpt-oss-20b  (MoE, MXFP4 quantized checkpoint)
Phase     : Experimentation
Strategy  : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method    : Standard BF16 LoRA — NOT QLoRA, no bitsandbytes

Triton     : Available (python3.12-dev installed).
             MXFP4 weights are dequantized to BF16 via Mxfp4Config(dequantize=True)
             so the model is fully training-compatible.

Launch:
    export CUDA_VISIBLE_DEVICES=0,1,2   # all three GPUs
    python fine_tuning/train_spacellm_lora.py

Output layout:
  SpaceLLM/fine_tuning/outputs/
  ├── checkpoints/
  ├── spacellm_lora_final/     ← LOAD THIS for inference / evaluation
  └── logs/

To load after training:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel
    import torch
    base  = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                quantization_config=Mxfp4Config(dequantize=True),
                device_map="auto")
    model = PeftModel.from_pretrained(base, "./outputs/spacellm_lora_final")
    tok   = AutoTokenizer.from_pretrained("./outputs/spacellm_lora_final")
"""

# ── TRITON PATCH — must be the very first thing before any other import ───────
import sys
import types


def _patch_triton():
    """
    Install a lightweight stub for triton.backends.nvidia.driver before
    the real triton package loads.
    """

    class _StubDriver:
        def __getattr__(self, name):
            return _StubDriver()
        def __call__(self, *a, **kw):
            return _StubDriver()
        def __bool__(self):
            return False

    class _StubCudaUtils:
        def __init__(self):
            pass
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

    def _stub_build(*a, **kw):
        return None
    triton_runtime_bld._build                  = _stub_build
    triton_runtime_bld.compile_module_from_src = lambda *a, **kw: types.ModuleType("_stub_cuda_utils")
    triton_runtime_bld.load_module             = lambda *a, **kw: types.ModuleType("_stub_cuda_utils")

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

# ── Now safe to import everything else ───────────────────────────────────────

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

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
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("SpaceLLM")

# ── Standard CE loss replacement ─────────────────────────────────────────────
# CRITICAL FIX: gpt-oss-20b stores a custom loss_function on the model and
# calls it as self.loss_function(logits, labels, ...) during forward().
# Setting it to None causes 'NoneType object is not callable'.
# We replace it with this standard shifted cross-entropy callable instead.

def _make_device_aware_ce_loss():
    """
    Returns a device-aware cross-entropy loss function.

    ROOT CAUSE THIS FIXES:
      gpt-oss-20b is a MoE model sharded across multiple GPUs via device_map.
      The lm_head (producing logits) lands on e.g. cuda:1, while HuggingFace
      Trainer always places the batch (including labels) on cuda:0.
      PyTorch's nll_loss then crashes with:
        "Expected all tensors to be on the same device, but found target on
         cuda:0 and other tensors on cuda:1"
      The fix: always move labels onto whatever device logits are on, BEFORE
      computing loss. This is safe because logits are the authoritative output
      device in a sharded model.
    """
    def _device_aware_ce_loss(logits, labels, vocab_size=None, **kwargs):
        # ── Step 1: shift for causal LM ──────────────────────────────────
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # ── Step 2: device alignment — THE CORE FIX ──────────────────────
        # logits live on the lm_head device (may be cuda:1 in sharded model).
        # labels come from the Trainer and live on cuda:0.
        # Move labels to match logits — never the other way, since logits
        # are the larger tensor and moving them would break the shard layout.
        logits_device = shift_logits.device
        if shift_labels.device != logits_device:
            shift_labels = shift_labels.to(logits_device)

        # ── Step 3: vocab size ────────────────────────────────────────────
        if vocab_size is None:
            vocab_size = shift_logits.size(-1)

        # ── Step 4: standard CE ───────────────────────────────────────────
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1).long(),
        )

        # Move loss back to cuda:0 so Trainer can accumulate it correctly
        # (Trainer always expects loss on its primary device)
        return loss.to("cuda:0")

    return _device_aware_ce_loss


# Module-level singleton — created once, injected everywhere
_DEVICE_AWARE_CE_LOSS = _make_device_aware_ce_loss()


def _inject_loss_function(model, loss_fn=None, label=""):
    """
    Walk all likely locations where gpt-oss stores its loss_function
    and replace every one we find with loss_fn.
    Defaults to the device-aware CE loss if loss_fn is not provided.
    Returns True if at least one replacement was made.
    """
    if loss_fn is None:
        loss_fn = _DEVICE_AWARE_CE_LOSS

    replaced = False
    candidates = [model]
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    # Also walk top-level children one level deep
    for child in list(model.children()):
        candidates.append(child)

    for obj in candidates:
        if obj is not None and hasattr(obj, "loss_function"):
            old = getattr(obj, "loss_function")
            if old is not loss_fn:          # avoid double-patching
                setattr(obj, "loss_function", loss_fn)
                replaced = True
                logger.info(
                    f"  ✅ Replaced loss_function on {type(obj).__name__}"
                    + (f" ({label})" if label else "")
                )
    return replaced


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

def log_gpu_info():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info("Visible GPUs      :")
            for line in result.stdout.strip().splitlines():
                idx, name, mem = line.split(",")
                logger.info(f"  cuda:{idx.strip()} → {name.strip()}  ({int(mem.strip()):,} MiB)")
    except Exception:
        pass


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
        raise SystemExit(1)
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
        raise SystemExit(1)

    lengths = [len(t["input_ids"]) for t in tokenised]
    max_input_id = max(max(t["input_ids"]) for t in tokenised)
    max_label_id = max(
        max((lbl for lbl in t["labels"] if lbl != IGNORE_INDEX), default=0)
        for t in tokenised
    )
    logger.info(
        f"[{split_name}] {len(tokenised):,} records | "
        f"seq len  min={min(lengths)}  max={max(lengths)}  "
        f"mean={sum(lengths)/len(lengths):.0f} | "
        f"max_input_id={max_input_id}  max_label_id={max_label_id}"
    )
    return Dataset.from_list(tokenised)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info(f"  Strategy          : LoRA on lm_head ONLY — backbone frozen")
    logger.info(f"  Quantization      : Mxfp4Config(dequantize=True)  →  plain BF16")
    logger.info(f"  Epochs            : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch             : {args.batch_size}  |  Grad accum: {args.grad_accum}"
                f"  |  Eff batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  CUDA_VISIBLE_DEVS : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    logger.info(f"  Log               : {LOG_FILE}")
    logger.info("=" * 60)

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            DataCollatorForSeq2Seq,
            Trainer,
            Mxfp4Config,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: uv pip install -r requirements.txt")
        raise SystemExit(1)

    log_gpu_info()

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
        raise SystemExit(1)

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
        raise SystemExit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token")

    tokenizer.padding_side = "right"
    logger.info(f"Vocab size        : {tokenizer.vocab_size:,}")
    logger.info(f"len(tokenizer)    : {len(tokenizer):,}")
    logger.info(f"Pad token         : '{tokenizer.pad_token}'  (id={tokenizer.pad_token_id})")

    if tokenizer.chat_template is not None:
        logger.info("Chat template     : found (harmony format)")
    else:
        logger.warning("Chat template     : NOT FOUND")

    # ── Model load — dequantize MXFP4 → BF16 on CPU ──────────────────────
    logger.info("")
    logger.info(f"Loading model: {args.model_id}  [MXFP4 → BF16 dequantize on CPU]")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=Mxfp4Config(dequantize=True),
            device_map="cpu",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise SystemExit(1)

    logger.info(f"Model loaded + dequantized on CPU in {time.time() - t0:.1f}s")
    logger.info(f"Model dtype       : {next(model.parameters()).dtype}")

    # ── Vocab alignment — resize BEFORE LoRA wrapping ────────────────────
    logger.info("")
    logger.info("── Vocab & lm_head alignment ────────────────────────")
    logger.info(f"  tokenizer vocab  : {len(tokenizer):,}")
    logger.info(f"  model config vocab: {model.config.vocab_size:,}")
    logger.info(f"  lm_head shape    : {model.get_output_embeddings().weight.shape}")

    model.config.tie_word_embeddings = False
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"  Resizing embeddings: {model.config.vocab_size:,} → {len(tokenizer):,}")
        model.config.vocab_size = len(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.config.vocab_size = len(tokenizer)
        logger.info("  Vocab sizes already aligned — no resize needed")

    logger.info(f"  lm_head shape after resize: {model.get_output_embeddings().weight.shape}")

    # ── CRITICAL FIX: Replace custom loss_function BEFORE GPU dispatch ────
    # gpt-oss-20b has a custom loss_function attribute that its forward()
    # calls directly. Setting it to None (previous approach) causes
    # "NoneType object is not callable". We replace it with a standard
    # shifted cross-entropy that matches the expected call signature.
    logger.info("")
    logger.info("── Injecting standard CE loss (pre-dispatch) ────────")
    found = _inject_loss_function(model, label="pre-dispatch")
    if not found:
        logger.warning("  loss_function attribute not found at this stage — will retry after dispatch")

    # ── Dispatch model across GPUs ────────────────────────────────────────
    logger.info("")
    logger.info("Dispatching model across GPUs ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        # Auto-discover decoder layer class names (must not be split across GPUs)
        no_split = []
        for name, module in model.named_modules():
            cls      = type(module)
            cls_name = cls.__name__.lower()
            if (
                issubclass(cls, nn.Module)
                and cls is not nn.Module
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
            alloc = max(0, free - 4 * 1024**3)      # 4 GB headroom per GPU
            max_memory[i] = f"{int(alloc / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"max_memory per device  : {max_memory}")

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
        )
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
    log_gpu_memory("after model load")

    # ── Re-inject loss function after dispatch (device_map can rewrap model) ─
    logger.info("")
    logger.info("── Re-injecting standard CE loss (post-dispatch) ────")
    _inject_loss_function(model, label="post-dispatch")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: enabled")

    # ── LoRA on lm_head ONLY ──────────────────────────────────────────────
    logger.info("")
    logger.info("Applying LoRA to lm_head ONLY — backbone stays frozen ...")

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

    # ── Re-inject loss function a final time after PEFT wrapping ─────────
    # PEFT wraps the model in PeftModel which adds another layer of indirection.
    # We must patch all accessible paths again.
    logger.info("")
    logger.info("── Re-injecting standard CE loss (post-PEFT) ────────")
    _inject_loss_function(model, label="post-PEFT")

    try:
        lm_head = model.get_output_embeddings()
        logger.info(f"Final lm_head shape after LoRA: {lm_head.weight.shape}")
    except Exception:
        logger.info("Could not inspect final lm_head shape")

    logger.info("")
    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init")

    # ── Final vocab alignment check ───────────────────────────────────────
    logger.info("")
    logger.info("── Vocab alignment (final check) ────────────────────")
    logger.info(f"  tokenizer.vocab_size : {tokenizer.vocab_size:,}")
    logger.info(f"  len(tokenizer)       : {len(tokenizer):,}")
    logger.info(f"  model.config.vocab   : {model.config.vocab_size:,}")
    logger.info(f"  pad_token_id         : {tokenizer.pad_token_id}")
    logger.info(f"  eos_token_id         : {tokenizer.eos_token_id}")

    if len(tokenizer) > model.config.vocab_size:
        logger.warning("Vocab mismatch detected post-LoRA — resizing again")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    logger.info("  ✅ Vocab alignment PASSED")

    # ── Load datasets ─────────────────────────────────────────────────────
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
        "device_map":       "cpu→GPU dispatch (accelerate)",
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
    # ── Device-aware Trainer subclass ─────────────────────────────────────
    # SECOND LINE OF DEFENCE against the cross-device loss crash.
    #
    # Even with the device-aware loss function, HuggingFace Trainer moves the
    # entire batch to self.args.device (cuda:0) before calling model(**inputs).
    # In a sharded MoE model the lm_head lives on cuda:1, so logits come back
    # on cuda:1 while labels are still on cuda:0.
    #
    # We override _prepare_inputs to detect where the lm_head actually lives
    # and move labels there, so they are co-located with logits when the loss
    # function is called.

    class DeviceAwareTrainer(Trainer):
        """Trainer that moves labels to the lm_head device before forward."""

        def _get_lm_head_device(self):
            """Return the device the lm_head (output embedding) lives on."""
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            # Let the base class do its normal preparation (moves to cuda:0)
            inputs = super()._prepare_inputs(inputs)

            lm_device = self._get_lm_head_device()
            if lm_device is None:
                return inputs

            # Move labels to lm_head device if they differ
            if "labels" in inputs:
                current = inputs["labels"].device
                if current != lm_device:
                    inputs["labels"] = inputs["labels"].to(lm_device)

            return inputs

    trainer = DeviceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Final loss_function patch after Trainer init ──────────────────────
    logger.info("")
    logger.info("── Final loss_function patch (post-Trainer init) ────")
    _inject_loss_function(trainer.model, label="post-Trainer")
    logger.info(f"  lm_head device   : {trainer._get_lm_head_device()}")

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
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise SystemExit(1)

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
        "triton":                "stubbed (dequantize path used)",
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
    logger.info("  from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config")
    logger.info("  from peft import PeftModel")
    logger.info(f"  base  = AutoModelForCausalLM.from_pretrained(")
    logger.info(f"              '{args.model_id}',")
    logger.info(f"              quantization_config=Mxfp4Config(dequantize=True), device_map='auto')")
    logger.info(f"  model = PeftModel.from_pretrained(base, '{FINAL_DIR}')")
    logger.info(f"  tok   = AutoTokenizer.from_pretrained('{FINAL_DIR}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()