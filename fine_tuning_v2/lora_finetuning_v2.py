"""
SpaceLLM — Experimental LoRA Fine-Tuning  v2
=============================================
Model     : openai/gpt-oss-20b  (MoE, MXFP4 quantized checkpoint)
Phase     : Experimentation
Strategy  : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method    : Standard BF16 LoRA — NOT QLoRA, no bitsandbytes

Triton     : Available (python3.12-dev installed).
             MXFP4 weights are dequantized to BF16 via Mxfp4Config(dequantize=True)
             so the model is fully training-compatible.

Launch:
    export CUDA_VISIBLE_DEVICES=0,1,2   # all three GPUs
    python fine_tuning_v2/lora_finetuning_v2.py

Output layout:
  SpaceLLM/fine_tuning_v2/outputs/
  ├── checkpoints/
  ├── spacellm_lora_final/     ← LOAD THIS for inference / evaluation
  ├── graphs/                  ← training + eval loss curves, test results
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

CHANGES FROM v1:
  - Updated dataset paths  →  DatasetA_core_QA_v2  (.json instead of .jsonl)
  - Output dir             →  fine_tuning_v2/outputs/
  - Added graph saving     →  training loss, eval loss, LR schedule curves
  - Added test evaluation  →  uses fine-tuned model; saves metrics + per-sample
                               predictions to outputs/graphs/test_results.json
  - Fixed max_grad_norm inconsistency (was logged as 1.0, actual was 0.5 → now
    consistently 0.5 everywhere)
  - CustomCallback now records loss/lr at every logging step for smooth curves
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
GRAPH_DIR  = OUTPUT_DIR / "graphs"             # ← NEW: all plots saved here

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Fixed dataset paths  (v2 — .json) ────────────────────────────────────────

TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/train.json")
VAL_FILE   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/validation.json")
TEST_FILE  = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/test.json")

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

def _make_device_aware_ce_loss():
    """
    Returns a device-aware cross-entropy loss function.

    ROOT CAUSE THIS FIXES:
      gpt-oss-20b is a MoE model sharded across multiple GPUs via device_map.
      The lm_head lands on e.g. cuda:1, while HuggingFace Trainer always places
      the batch (including labels) on cuda:0.
      FIX: move labels onto whatever device logits are on before computing loss.
    """
    def _device_aware_ce_loss(logits, labels, vocab_size=None, **kwargs):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_device = shift_logits.device
        if shift_labels.device != logits_device:
            shift_labels = shift_labels.to(logits_device)

        # Always read vocab size from actual logits shape (not config)
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


def _inject_loss_function(model, loss_fn=None, label=""):
    if loss_fn is None:
        loss_fn = _DEVICE_AWARE_CE_LOSS

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
            old = getattr(obj, "loss_function")
            if old is not loss_fn:
                setattr(obj, "loss_function", loss_fn)
                replaced = True
                logger.info(
                    f"  ✅ Replaced loss_function on {type(obj).__name__}"
                    + (f" ({label})" if label else "")
                )
    return replaced


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpaceLLM lm_head LoRA fine-tuning v2")
    p.add_argument("--model_id",               type=str,   default="openai/gpt-oss-20b")
    p.add_argument("--epochs",                 type=int,   default=3)
    p.add_argument("--batch_size",             type=int,   default=1)
    p.add_argument("--grad_accum",             type=int,   default=16)
    p.add_argument("--lr",                     type=float, default=5e-6)
    p.add_argument("--max_seq_len",            type=int,   default=2048)
    p.add_argument("--warmup_steps",           type=int,   default=350)
    p.add_argument("--save_steps",             type=int,   default=500)
    p.add_argument("--eval_steps",             type=int,   default=500)
    p.add_argument("--logging_steps",          type=int,   default=20)
    p.add_argument("--save_total_limit",       type=int,   default=2)
    p.add_argument("--max_test_samples",       type=int,   default=200,
                   help="Max samples to run generation on during test eval (cost control)")
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

# ── JSON loading (.json — list of records or newline-delimited) ───────────────

def load_json(path: Path) -> list:
    """
    Supports both a JSON array  [{ ... }, { ... }]
    and newline-delimited JSONL  { ... }\n{ ... }
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise SystemExit(1)

    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()

    # Try array first
    if raw.startswith("["):
        try:
            records = json.loads(raw)
            logger.info(f"Loaded {len(records):,} records (JSON array)  ←  {path}")
            return records
        except json.JSONDecodeError:
            pass

    # Fall back to JSONL
    records = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed line {line_no} in {path.name}: {e}")

    logger.info(f"Loaded {len(records):,} records (JSONL)  ←  {path}")
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


def build_hf_dataset(records: list, tokenizer, max_seq_len: int, split_name: str,
                      vocab_size: int = None):
    from datasets import Dataset

    tokenised, skipped, clamped_records = [], 0, 0
    for record in records:
        result = tokenise_record(record, tokenizer, max_seq_len)
        if result is None:
            skipped += 1
            continue

        if vocab_size is not None:
            new_labels = []
            had_oob = False
            for lbl in result["labels"]:
                if lbl != IGNORE_INDEX and (lbl < 0 or lbl >= vocab_size):
                    new_labels.append(IGNORE_INDEX)
                    had_oob = True
                else:
                    new_labels.append(lbl)
            if had_oob:
                result["labels"] = new_labels
                clamped_records += 1

        if all(lbl == IGNORE_INDEX for lbl in result["labels"]):
            skipped += 1
            continue

        tokenised.append(result)

    if skipped:
        logger.warning(f"[{split_name}] Skipped {skipped} records (no valid labels)")
    if clamped_records:
        logger.warning(
            f"[{split_name}] Clamped out-of-vocab labels in {clamped_records} records "
            f"(vocab_size={vocab_size:,})"
        )
    if not tokenised:
        logger.error(f"[{split_name}] Zero usable records — aborting")
        raise SystemExit(1)

    lengths      = [len(t["input_ids"]) for t in tokenised]
    max_input_id = max(max(t["input_ids"]) for t in tokenised)
    max_label_id = max(
        max((lbl for lbl in t["labels"] if lbl != IGNORE_INDEX), default=0)
        for t in tokenised
    )

    if vocab_size is not None and max_label_id >= vocab_size:
        logger.error(
            f"[{split_name}] FATAL: max_label_id={max_label_id} >= vocab_size={vocab_size} "
            f"after clamping — this WILL cause CUDA assertion failure"
        )
        raise SystemExit(1)

    logger.info(
        f"[{split_name}] {len(tokenised):,} records | "
        f"seq len  min={min(lengths)}  max={max(lengths)}  "
        f"mean={sum(lengths)/len(lengths):.0f} | "
        f"max_input_id={max_input_id}  max_label_id={max_label_id} | "
        f"vocab_size={vocab_size if vocab_size else 'unchecked'}"
    )
    return Dataset.from_list(tokenised)

# ── Training history callback ─────────────────────────────────────────────────

class HistoryCallback:
    """
    Attached to Trainer to collect loss, eval_loss, and learning-rate at
    every logging/eval step so we can plot smooth curves after training.
    """
    from transformers import TrainerCallback

    def __init__(self):
        self.train_loss  : list = []   # [(step, loss), ...]
        self.eval_loss   : list = []   # [(step, eval_loss), ...]
        self.lr_history  : list = []   # [(step, lr), ...]

    # We inherit from TrainerCallback dynamically to avoid import-time issues
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_loss.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_loss.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.lr_history.append((step, logs["learning_rate"]))


# ── Graph saving ──────────────────────────────────────────────────────────────

def save_training_graphs(history: HistoryCallback, run_id: str):
    """
    Saves three figures to GRAPH_DIR:
      1. training_loss_{run_id}.png
      2. eval_loss_{run_id}.png
      3. lr_schedule_{run_id}.png
    Also saves a combined overview figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not available — skipping graph saving")
        return

    def _plot(xy_list, title, xlabel, ylabel, color, path):
        if not xy_list:
            logger.warning(f"No data for '{title}' — skipping")
            return
        steps, vals = zip(*xy_list)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, vals, color=color, linewidth=1.5, label=ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"  Graph saved → {path}")

    # Individual plots
    _plot(history.train_loss, "Training Loss",       "Step", "Loss",          "#e74c3c",
          GRAPH_DIR / f"training_loss_{run_id}.png")
    _plot(history.eval_loss,  "Validation Eval Loss","Step", "Eval Loss",     "#2980b9",
          GRAPH_DIR / f"eval_loss_{run_id}.png")
    _plot(history.lr_history, "Learning Rate Schedule","Step","Learning Rate","#27ae60",
          GRAPH_DIR / f"lr_schedule_{run_id}.png")

    # Combined 2×2 overview
    try:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        fig.suptitle(f"SpaceLLM LoRA v2 — Training Overview  [{run_id}]",
                     fontsize=13, fontweight="bold")

        for ax, (xy_list, title, color, ylabel) in zip(axes, [
            (history.train_loss, "Training Loss",        "#e74c3c", "Loss"),
            (history.eval_loss,  "Validation Eval Loss", "#2980b9", "Eval Loss"),
            (history.lr_history, "LR Schedule",          "#27ae60", "Learning Rate"),
        ]):
            if xy_list:
                steps, vals = zip(*xy_list)
                ax.plot(steps, vals, color=color, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        overview_path = GRAPH_DIR / f"training_overview_{run_id}.png"
        fig.savefig(overview_path, dpi=150)
        plt.close(fig)
        logger.info(f"  Overview graph → {overview_path}")
    except Exception as e:
        logger.warning(f"Overview graph failed: {e}")


def save_test_loss_graph(test_losses: list, run_id: str):
    """Per-sample test loss bar chart (up to 200 samples for readability)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not test_losses:
        return

    display = test_losses[:200]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Test Set Evaluation — SpaceLLM LoRA v2  [{run_id}]",
                 fontsize=13, fontweight="bold")

    # Per-sample loss bar chart
    axes[0].bar(range(len(display)), display, color="#8e44ad", alpha=0.7, width=1.0)
    axes[0].set_title(f"Per-Sample Loss (first {len(display)} samples)")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Loss")
    axes[0].axhline(sum(display) / len(display), color="red",
                    linestyle="--", label=f"Mean={sum(display)/len(display):.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss histogram
    axes[1].hist(test_losses, bins=40, color="#8e44ad", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"Loss Distribution (all {len(test_losses)} samples)")
    axes[1].set_xlabel("Loss")
    axes[1].set_ylabel("Count")
    axes[1].axvline(sum(test_losses) / len(test_losses), color="red",
                    linestyle="--", label=f"Mean={sum(test_losses)/len(test_losses):.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = GRAPH_DIR / f"test_loss_{run_id}.png"
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    logger.info(f"  Test loss graph → {path}")


# ── Test evaluation using fine-tuned model ────────────────────────────────────

def run_test_evaluation(model, tokenizer, test_records: list,
                        max_seq_len: int, vocab_size: int,
                        max_samples: int, run_id: str):
    """
    Runs two complementary evaluations on the test set:

    1. LOSS EVALUATION  — teacher-forced cross-entropy on every test record,
                          identical to how val loss is computed during training.
                          Fast; covers all test records.

    2. GENERATION EVALUATION — greedy decode the first `max_samples` records,
                               compare generated text to the reference answer.
                               Computes exact-match and token-overlap (ROUGE-L proxy).

    Results are saved to:
      outputs/graphs/test_results_{run_id}.json
      outputs/graphs/test_loss_{run_id}.png
    """
    from transformers import DataCollatorForSeq2Seq
    import torch.nn.functional as F

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Test Evaluation")
    logger.info("=" * 60)

    model.eval()

    # ── 1. Loss evaluation (all records) ─────────────────────────────────
    logger.info(f"── Phase 1: Loss evaluation on {len(test_records):,} records ──")

    per_sample_losses = []
    skipped = 0

    with torch.no_grad():
        for i, record in enumerate(test_records):
            result = tokenise_record(record, tokenizer, max_seq_len)
            if result is None:
                skipped += 1
                continue

            # Label clamping
            new_labels = []
            for lbl in result["labels"]:
                if lbl != IGNORE_INDEX and (lbl < 0 or lbl >= vocab_size):
                    new_labels.append(IGNORE_INDEX)
                else:
                    new_labels.append(lbl)
            result["labels"] = new_labels

            if all(lbl == IGNORE_INDEX for lbl in result["labels"]):
                skipped += 1
                continue

            input_ids = torch.tensor([result["input_ids"]],      dtype=torch.long)
            attn_mask = torch.tensor([result["attention_mask"]], dtype=torch.long)
            labels    = torch.tensor([result["labels"]],          dtype=torch.long)

            # Move to first GPU
            device = next(model.parameters()).device
            try:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)
            except Exception:
                pass

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                loss_val = outputs.loss
                if loss_val is not None and not torch.isnan(loss_val):
                    per_sample_losses.append(float(loss_val.item()))
            except Exception as e:
                logger.warning(f"  Loss eval failed for record {i}: {e}")
                skipped += 1

            if (i + 1) % 50 == 0:
                completed = len(per_sample_losses)
                mean_so_far = sum(per_sample_losses) / completed if completed else float("nan")
                logger.info(f"  [{i+1}/{len(test_records)}]  "
                            f"valid={completed}  mean_loss={mean_so_far:.4f}")

    mean_test_loss = sum(per_sample_losses) / len(per_sample_losses) if per_sample_losses else float("nan")
    logger.info(f"  Loss eval done: {len(per_sample_losses):,} samples  "
                f"(skipped={skipped})  mean_loss={mean_test_loss:.4f}")

    # Save loss graph
    save_test_loss_graph(per_sample_losses, run_id)

    # ── 2. Generation evaluation (first max_samples records) ─────────────
    logger.info("")
    logger.info(f"── Phase 2: Generation evaluation on first {max_samples} records ──")

    gen_records   = test_records[:max_samples]
    predictions   = []
    exact_matches = 0
    token_overlaps= []

    with torch.no_grad():
        for i, record in enumerate(gen_records):
            messages = record.get("messages", [])
            hf_messages = []
            for msg in messages:
                role    = "system" if msg["role"] == "developer" else msg["role"]
                content = msg.get("content", "").strip()
                if content and role != "assistant":
                    hf_messages.append({"role": role, "content": content})

            # Reference answer
            ref_answer = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    ref_answer = msg.get("content", "").strip()
                    break

            if not hf_messages or not ref_answer:
                continue

            try:
                prompt_text = tokenizer.apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                continue

            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len - 256,   # leave room for generation
            )

            device = next(model.parameters()).device
            try:
                enc = {k: v.to(device) for k, v in enc.items()}
            except Exception:
                pass

            try:
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,        # greedy
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # Decode only the newly generated tokens
                new_ids    = gen_ids[0][enc["input_ids"].shape[1]:]
                generated  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception as e:
                logger.warning(f"  Generation failed for record {i}: {e}")
                continue

            # Exact match (case-insensitive, stripped)
            em = int(generated.lower() == ref_answer.lower())
            exact_matches += em

            # Token overlap  (simple F1 proxy — no external lib needed)
            ref_toks = set(ref_answer.lower().split())
            gen_toks = set(generated.lower().split())
            if ref_toks or gen_toks:
                overlap = len(ref_toks & gen_toks)
                prec    = overlap / len(gen_toks)  if gen_toks  else 0.0
                rec     = overlap / len(ref_toks)  if ref_toks  else 0.0
                f1      = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0
            token_overlaps.append(f1)

            predictions.append({
                "sample_id":  record.get("sample_id", i),
                "reference":  ref_answer,
                "generated":  generated,
                "exact_match": em,
                "token_f1":   round(f1, 4),
            })

            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{len(gen_records)}]  "
                            f"EM so far={exact_matches}/{len(predictions)}  "
                            f"mean_F1={sum(token_overlaps)/len(token_overlaps):.4f}")

    n_gen = len(predictions)
    em_rate    = exact_matches / n_gen                   if n_gen else float("nan")
    mean_f1    = sum(token_overlaps) / len(token_overlaps) if token_overlaps else float("nan")

    logger.info("")
    logger.info("── Test Results ─────────────────────────────────────")
    logger.info(f"  Loss eval samples  : {len(per_sample_losses):,}")
    logger.info(f"  Mean test loss     : {mean_test_loss:.4f}")
    logger.info(f"  Generation samples : {n_gen}")
    logger.info(f"  Exact match        : {exact_matches}/{n_gen}  ({em_rate*100:.2f}%)")
    logger.info(f"  Mean token F1      : {mean_f1:.4f}")

    # ── Save generation quality graph ────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if token_overlaps:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Test Generation Metrics — SpaceLLM LoRA v2  [{run_id}]",
                         fontsize=13, fontweight="bold")

            # Token F1 histogram
            axes[0].hist(token_overlaps, bins=30, color="#16a085", alpha=0.8, edgecolor="white")
            axes[0].set_title("Token F1 Distribution")
            axes[0].set_xlabel("Token F1")
            axes[0].set_ylabel("Count")
            axes[0].axvline(mean_f1, color="red", linestyle="--",
                            label=f"Mean={mean_f1:.3f}")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Exact match bar
            axes[1].bar(["Exact Match", "No Match"],
                        [exact_matches, n_gen - exact_matches],
                        color=["#27ae60", "#e74c3c"], alpha=0.8)
            axes[1].set_title(f"Exact Match  ({em_rate*100:.1f}%)")
            axes[1].set_ylabel("Count")
            for bar_val, bar_obj in zip(
                [exact_matches, n_gen - exact_matches], axes[1].patches
            ):
                axes[1].text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 0.5,
                    str(bar_val), ha="center", va="bottom", fontsize=12
                )
            axes[1].grid(True, alpha=0.3, axis="y")

            fig.tight_layout()
            gen_graph_path = GRAPH_DIR / f"test_generation_{run_id}.png"
            fig.savefig(gen_graph_path, dpi=150)
            plt.close(fig)
            logger.info(f"  Generation graph → {gen_graph_path}")
    except Exception as e:
        logger.warning(f"Generation graph failed: {e}")

    # ── Consolidate and save all test results ─────────────────────────────
    test_results = {
        "run_id":              run_id,
        "total_test_records":  len(test_records),
        "loss_eval": {
            "samples_evaluated": len(per_sample_losses),
            "samples_skipped":   skipped,
            "mean_test_loss":    round(mean_test_loss, 6) if not isinstance(mean_test_loss, str) else mean_test_loss,
        },
        "generation_eval": {
            "samples_evaluated": n_gen,
            "exact_matches":     exact_matches,
            "exact_match_rate":  round(em_rate, 6)  if n_gen else None,
            "mean_token_f1":     round(mean_f1, 6)  if n_gen else None,
        },
        "per_sample_predictions": predictions,
    }

    results_path = GRAPH_DIR / f"test_results_{run_id}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Full test results → {results_path}")

    return test_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Experimental LoRA v2  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info(f"  Strategy          : LoRA on lm_head ONLY — backbone frozen")
    logger.info(f"  Quantization      : Mxfp4Config(dequantize=True)  →  plain BF16")
    logger.info(f"  Epochs            : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch             : {args.batch_size}  |  Grad accum: {args.grad_accum}"
                f"  |  Eff batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  CUDA_VISIBLE_DEVS : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    logger.info(f"  Log               : {LOG_FILE}")
    logger.info(f"  Graphs            : {GRAPH_DIR}")
    logger.info("=" * 60)

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
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

    # ── Model load ────────────────────────────────────────────────────────
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

    # ── Vocab alignment ───────────────────────────────────────────────────
    logger.info("")
    logger.info("── Vocab & lm_head alignment ────────────────────────")
    logger.info(f"  tokenizer.vocab_size : {tokenizer.vocab_size:,}")
    logger.info(f"  len(tokenizer)       : {len(tokenizer):,}")
    logger.info(f"  model.config.vocab   : {model.config.vocab_size:,}")
    logger.info(f"  embed_tokens shape   : {model.get_input_embeddings().weight.shape}")
    logger.info(f"  lm_head shape        : {model.get_output_embeddings().weight.shape}")
    logger.info(f"  tie_word_embeddings  : {model.config.tie_word_embeddings}")

    model.config.tie_word_embeddings = False
    _current_vocab = model.get_output_embeddings().weight.shape[0]
    model.resize_token_embeddings(_current_vocab, pad_to_multiple_of=64)
    logger.info("  Embeddings untied (tie_word_embeddings=False)")

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    actual_vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = actual_vocab

    logger.info(f"  target (len tokenizer)    : {len(tokenizer):,}")
    logger.info(f"  actual (padded, step=64)  : {actual_vocab:,}")
    logger.info(f"  embed_tokens shape after  : {model.get_input_embeddings().weight.shape}")
    logger.info(f"  lm_head shape after       : {model.get_output_embeddings().weight.shape}")
    logger.info(f"  model.config.vocab_size   : {model.config.vocab_size:,}")

    assert model.get_input_embeddings().weight.shape[0] == actual_vocab
    assert model.get_output_embeddings().weight.shape[0] == actual_vocab
    assert model.config.vocab_size == actual_vocab
    logger.info(f"  Vocab alignment PASSED (vocab={actual_vocab:,})")

    # ── Inject loss (pre-dispatch) ────────────────────────────────────────
    logger.info("")
    logger.info("── Injecting standard CE loss (pre-dispatch) ────────")
    found = _inject_loss_function(model, label="pre-dispatch")
    if not found:
        logger.warning("  loss_function not found pre-dispatch — will retry after dispatch")

    # ── GPU dispatch ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("Dispatching model across GPUs ...")
    t1 = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

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
            alloc = max(0, free - 4 * 1024**3)
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

    # ── Re-inject after dispatch ──────────────────────────────────────────
    logger.info("")
    logger.info("── Re-injecting standard CE loss (post-dispatch) ────")
    _inject_loss_function(model, label="post-dispatch")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing: enabled")

    # ── LoRA on lm_head ───────────────────────────────────────────────────
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

    # ── Post-PEFT vocab check ─────────────────────────────────────────────
    logger.info("")
    logger.info("── Vocab alignment (post-PEFT final check) ──────────")
    _post_peft_vocab = model.get_output_embeddings().weight.shape[0]
    logger.info(f"  tokenizer.vocab_size : {tokenizer.vocab_size:,}")
    logger.info(f"  len(tokenizer)       : {len(tokenizer):,}")
    logger.info(f"  model.config.vocab   : {model.config.vocab_size:,}")
    logger.info(f"  lm_head actual vocab : {_post_peft_vocab:,}")
    logger.info(f"  pad_token_id         : {tokenizer.pad_token_id}")
    logger.info(f"  eos_token_id         : {tokenizer.eos_token_id}")

    if _post_peft_vocab != model.config.vocab_size:
        logger.warning(
            f"  lm_head vocab ({_post_peft_vocab}) != config.vocab_size "
            f"({model.config.vocab_size}) after PEFT — fixing config"
        )
        model.config.vocab_size = _post_peft_vocab

    _max_valid_label = _post_peft_vocab - 1
    logger.info(f"  Max valid label ID   : {_max_valid_label:,}")
    logger.info("  Vocab alignment PASSED (post-PEFT)")

    # ── Load datasets ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Loading datasets ─────────────────────────────────")
    train_records = load_json(TRAIN_FILE)
    val_records   = load_json(VAL_FILE)
    test_records  = load_json(TEST_FILE) if TEST_FILE.exists() else []

    dataset_sanity_check(train_records, "train")
    dataset_sanity_check(val_records,   "validation")
    if test_records:
        dataset_sanity_check(test_records, "test")
    else:
        logger.warning("Test file not found — test evaluation will be skipped")

    logger.info("")
    logger.info("── Tokenising ───────────────────────────────────────")
    train_dataset = build_hf_dataset(
        train_records, tokenizer, args.max_seq_len, "train",
        vocab_size=_post_peft_vocab,
    )
    val_dataset   = build_hf_dataset(
        val_records, tokenizer, args.max_seq_len, "validation",
        vocab_size=_post_peft_vocab,
    )

    # ── Training arguments ────────────────────────────────────────────────
    MAX_GRAD_NORM = 0.5     # single source of truth — used in args AND logs

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
        max_grad_norm=MAX_GRAD_NORM,
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
        "max_grad_norm":    MAX_GRAD_NORM,
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
        pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
    )

    # ── Training history callback ─────────────────────────────────────────
    history = HistoryCallback()

    # Attach TrainerCallback interface dynamically
    class _TrainingHistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            history.on_log(args, state, control, logs=logs, **kwargs)

    history_callback = _TrainingHistoryCallback()

    # ── Device-aware Trainer ──────────────────────────────────────────────
    class DeviceAwareTrainer(Trainer):
        def _get_lm_head_device(self):
            try:
                return next(self.model.get_output_embeddings().parameters()).device
            except Exception:
                return None

        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            lm_device = self._get_lm_head_device()
            if lm_device is None:
                return inputs
            if "labels" in inputs:
                if inputs["labels"].device != lm_device:
                    inputs["labels"] = inputs["labels"].to(lm_device)
            return inputs

    trainer = DeviceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[history_callback],
    )

    # ── Final loss patch after Trainer init ──────────────────────────────
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
        # Still save graphs on interrupt
        logger.info("Saving training graphs (partial) ...")
        save_training_graphs(history, RUN_ID)
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise SystemExit(1)

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min  ({elapsed:.0f}s)")

    # ── Save training graphs ──────────────────────────────────────────────
    logger.info("")
    logger.info("── Saving training graphs ───────────────────────────")
    save_training_graphs(history, RUN_ID)

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

    # ── Save final LoRA adapters ──────────────────────────────────────────
    logger.info("")
    logger.info(f"Saving final LoRA adapters → {FINAL_DIR}")
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    # ── Test evaluation ───────────────────────────────────────────────────
    test_results = None
    if test_records:
        test_results = run_test_evaluation(
            model=trainer.model,
            tokenizer=tokenizer,
            test_records=test_records,
            max_seq_len=args.max_seq_len,
            vocab_size=_post_peft_vocab,
            max_samples=args.max_test_samples,
            run_id=RUN_ID,
        )
    else:
        logger.warning("Skipping test evaluation — test.json not found")

    # ── Adapter info JSON ─────────────────────────────────────────────────
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
        "max_grad_norm":         MAX_GRAD_NORM,
        "batch_size":            args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size":  args.batch_size * args.grad_accum,
        "max_seq_len":           args.max_seq_len,
        "train_samples":         len(train_dataset),
        "val_samples":           len(val_dataset),
        "test_samples":          len(test_records),
        "train_metrics":         train_metrics,
        "eval_metrics":          eval_metrics,
        "test_results_summary":  (
            {
                "mean_test_loss":     test_results["loss_eval"]["mean_test_loss"],
                "exact_match_rate":   test_results["generation_eval"]["exact_match_rate"],
                "mean_token_f1":      test_results["generation_eval"]["mean_token_f1"],
            } if test_results else None
        ),
        "final_adapter_dir":     str(FINAL_DIR),
        "checkpoints_dir":       str(CKPT_DIR),
        "graphs_dir":            str(GRAPH_DIR),
        "log_file":              str(LOG_FILE),
    }
    with (FINAL_DIR / "adapter_info.json").open("w") as f:
        json.dump(adapter_info, f, indent=2)

    log_gpu_memory("after training")

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SpaceLLM lm_head LoRA v2 — Training Complete")
    logger.info("=" * 60)
    logger.info(f"  Final adapters   →  {FINAL_DIR}")
    logger.info(f"  Checkpoints      →  {CKPT_DIR}")
    logger.info(f"  Graphs           →  {GRAPH_DIR}")
    logger.info(f"  Logs             →  {LOG_DIR}")
    if test_results:
        le = test_results["loss_eval"]
        ge = test_results["generation_eval"]
        logger.info("")
        logger.info("  Test Results Summary:")
        logger.info(f"    Mean test loss    : {le['mean_test_loss']:.4f}  "
                    f"({le['samples_evaluated']:,} samples)")
        if ge["exact_match_rate"] is not None:
            logger.info(f"    Exact match       : {ge['exact_match_rate']*100:.2f}%  "
                        f"({ge['exact_matches']}/{ge['samples_evaluated']})")
            logger.info(f"    Mean token F1     : {ge['mean_token_f1']:.4f}")
    logger.info("")
    logger.info("  To load for evaluation:")
    logger.info("    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config")
    logger.info("    from peft import PeftModel")
    logger.info(f"    base  = AutoModelForCausalLM.from_pretrained(")
    logger.info(f"                '{args.model_id}',")
    logger.info(f"                quantization_config=Mxfp4Config(dequantize=True), device_map='auto')")
    logger.info(f"    model = PeftModel.from_pretrained(base, '{FINAL_DIR}')")
    logger.info(f"    tok   = AutoTokenizer.from_pretrained('{FINAL_DIR}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()