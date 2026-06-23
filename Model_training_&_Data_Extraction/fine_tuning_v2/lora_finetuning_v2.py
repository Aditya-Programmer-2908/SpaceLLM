"""
SpaceLLM — Optimized LoRA Fine-Tuning  v4
==========================================
Model     : openai/gpt-oss-20b  (MoE, MXFP4 quantized checkpoint)
Phase     : Experimentation
Strategy  : Freeze full transformer backbone, apply LoRA ONLY to lm_head
Method    : Standard BF16 LoRA — NOT QLoRA, no bitsandbytes

FIXES vs v3-fixed:
  [CRITICAL] 1. lm_head weight untied and cloned BEFORE get_peft_model()
                 → model.config.tie_word_embeddings = False only updates the
                   config flag — it does NOT break the live tensor tie.
                   With tie_word_embeddings=True, lm_head.weight and
                   embed_tokens.weight are the SAME tensor object in memory.
                   PEFT wraps lm_head, but autograd sees the weight as belonging
                   to the frozen embed_tokens path and cuts gradients to lora_A.
                   detach().clone() materializes lm_head as its own independent
                   parameter before PEFT ever touches it.

  [NICE]     2. LoRA r raised 16→32, lora_alpha raised 64→128
                 → Stronger learning signal for lm_head-only fine-tuning.

  [NICE]     3. grad_accum raised 16→32, lr raised 1e-4→2e-4
                 → Larger effective batch, faster convergence.

  [NICE]     4. max_grad_norm tightened 0.5→0.3, weight_decay=0.01 added.

  [NICE]     5. EarlyStoppingCallback added (patience=8).

  [NICE]     6. lr_scheduler_type changed cosine→cosine_with_restarts.

  [NICE]     7. optim changed adamw_torch→adamw_torch_fused (faster on GPU).

Launch:
    export CUDA_VISIBLE_DEVICES=1,2
    python fine_tuning_v2/lora_finetuning_v4.py

    # Override args:
    python fine_tuning_v2/lora_finetuning_v4.py --lr 2e-4 --epochs 5

Output layout:
  SpaceLLM/fine_tuning_v2/outputs/
  ├── checkpoints/
  ├── spacellm_lora_final/
  ├── graphs/
  └── logs/
"""

# ── TRITON PATCH — must be the very first thing before any other import ───────
import sys
import types


def _patch_triton():
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

    triton_runtime_bld._build                  = lambda *a, **kw: None
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

# ── Imports ───────────────────────────────────────────────────────────────────

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
GRAPH_DIR  = OUTPUT_DIR / "graphs"

for _d in (CKPT_DIR, FINAL_DIR, LOG_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset paths ─────────────────────────────────────────────────────────────

TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/train.json")
VAL_FILE   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/validate.json")
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

# ── Device-aware CE loss ──────────────────────────────────────────────────────

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
            if getattr(obj, "loss_function") is not loss_fn:
                setattr(obj, "loss_function", loss_fn)
                replaced = True
                logger.info(
                    f"  ✅ Replaced loss_function on {type(obj).__name__}"
                    + (f" ({label})" if label else "")
                )
    return replaced


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpaceLLM lm_head LoRA fine-tuning v4")
    p.add_argument("--model_id",               type=str,   default="openai/gpt-oss-20b")
    p.add_argument("--epochs",                 type=int,   default=5)
    p.add_argument("--batch_size",             type=int,   default=1)
    p.add_argument("--grad_accum",             type=int,   default=32)
    p.add_argument("--lr",                     type=float, default=2e-4)
    p.add_argument("--max_seq_len",            type=int,   default=2048)
    p.add_argument("--warmup_steps",           type=int,   default=200)
    p.add_argument("--save_steps",             type=int,   default=300)
    p.add_argument("--eval_steps",             type=int,   default=300)
    p.add_argument("--logging_steps",          type=int,   default=10)
    p.add_argument("--save_total_limit",       type=int,   default=3)
    p.add_argument("--max_test_samples",       type=int,   default=300)
    p.add_argument("--patience",               type=int,   default=8,
                   help="Early stopping patience (eval steps)")
    p.add_argument("--resume_from_checkpoint", type=str,   default=None)
    p.add_argument("--skip_grad_verify",       action="store_true")
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




# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise SystemExit(1)

    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        try:
            records = json.loads(raw)
            logger.info(f"Loaded {len(records):,} records (JSON array)  ←  {path}")
            return records
        except json.JSONDecodeError:
            pass

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


# ── Tokenisation ──────────────────────────────────────────────────────────────

IGNORE_INDEX = -100


def tokenise_record(record: dict, tokenizer, max_seq_len: int, debug: bool = False):
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

    # Extra safety: assistant response must have at least 4 real tokens
    if n_active < 4:
        return None

    if debug:
        logger.info(f"    prefix_len={prefix_len}  total={len(input_ids)}  active={n_active}")
        logger.info("    Active (loss) tokens sample:")
        shown = 0
        for tok_id, lbl in zip(input_ids, labels):
            if lbl != IGNORE_INDEX and shown < 10:
                logger.info(f"      {repr(tokenizer.decode([tok_id]))}")
                shown += 1

    return {
        "input_ids":      input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
        "n_active":       n_active,
    }


def build_hf_dataset(records: list, tokenizer, max_seq_len: int, split_name: str,
                     vocab_size: int = None, debug_first_n: int = 3):
    from datasets import Dataset

    tokenised, skipped, clamped_records = [], 0, 0
    active_counts = []

    for i, record in enumerate(records):
        result = tokenise_record(
            record, tokenizer, max_seq_len,
            debug=(i < debug_first_n and split_name == "train"),
        )
        if result is None:
            skipped += 1
            continue

        n_active = result.pop("n_active")
        active_counts.append(n_active)

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
                clamped_records += 1

        if all(lbl == IGNORE_INDEX for lbl in result["labels"]):
            skipped += 1
            continue

        tokenised.append(result)

    if skipped:
        logger.warning(f"[{split_name}] Skipped {skipped} records")
    if clamped_records:
        logger.warning(f"[{split_name}] Clamped OOV labels in {clamped_records} records")
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
        logger.error(f"[{split_name}] FATAL: max_label_id={max_label_id} >= vocab_size={vocab_size}")
        raise SystemExit(1)

    mean_active = sum(active_counts) / len(active_counts) if active_counts else 0
    min_active  = min(active_counts) if active_counts else 0
    max_active  = max(active_counts) if active_counts else 0

    logger.info(
        f"[{split_name}] {len(tokenised):,} records | "
        f"seq len  min={min(lengths)}  max={max(lengths)}  mean={sum(lengths)/len(lengths):.0f} | "
        f"active tokens  min={min_active}  mean={mean_active:.1f}  max={max_active} | "
        f"max_input_id={max_input_id}  max_label_id={max_label_id}"
    )

    if mean_active < 10:
        logger.warning(
            f"[{split_name}] ⚠️  mean active tokens={mean_active:.1f} — "
            f"label masking may be too aggressive."
        )

    return Dataset.from_list(tokenised)


# ── Training history callback ─────────────────────────────────────────────────

class HistoryCallback:
    def __init__(self):
        self.train_loss : list = []
        self.eval_loss  : list = []
        self.lr_history : list = []
        self.grad_norms : list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss"          in logs: self.train_loss.append((step, logs["loss"]))
        if "eval_loss"     in logs: self.eval_loss.append((step, logs["eval_loss"]))
        if "learning_rate" in logs: self.lr_history.append((step, logs["learning_rate"]))
        if "grad_norm"     in logs: self.grad_norms.append((step, logs["grad_norm"]))


# ── Graph saving ──────────────────────────────────────────────────────────────

def save_training_graphs(history: HistoryCallback, run_id: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not available — skipping graph saving")
        return

    def _plot(xy_list, title, xlabel, ylabel, color, path):
        if not xy_list:
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

    _plot(history.train_loss, "Training Loss",         "Step", "Loss",         "#e74c3c",
          GRAPH_DIR / f"training_loss_{run_id}.png")
    _plot(history.eval_loss,  "Validation Eval Loss",  "Step", "Eval Loss",    "#2980b9",
          GRAPH_DIR / f"eval_loss_{run_id}.png")
    _plot(history.lr_history, "Learning Rate Schedule","Step", "LR",           "#27ae60",
          GRAPH_DIR / f"lr_schedule_{run_id}.png")
    _plot(history.grad_norms, "Gradient Norm",         "Step", "Grad Norm",    "#8e44ad",
          GRAPH_DIR / f"grad_norm_{run_id}.png")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(f"SpaceLLM LoRA v4 — Training Overview  [{run_id}]",
                     fontsize=13, fontweight="bold")
        panels = [
            (history.train_loss, "Training Loss",        "#e74c3c", "Loss"),
            (history.eval_loss,  "Validation Eval Loss", "#2980b9", "Eval Loss"),
            (history.lr_history, "LR Schedule",          "#27ae60", "LR"),
            (history.grad_norms, "Gradient Norm",        "#8e44ad", "Grad Norm"),
        ]
        for ax, (xy_list, title, color, ylabel) in zip(axes.flat, panels):
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
    fig.suptitle(f"Test Set Evaluation — SpaceLLM LoRA v4  [{run_id}]",
                 fontsize=13, fontweight="bold")

    axes[0].bar(range(len(display)), display, color="#8e44ad", alpha=0.7, width=1.0)
    axes[0].set_title(f"Per-Sample Loss (first {len(display)} samples)")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Loss")
    mean_v = sum(display) / len(display)
    axes[0].axhline(mean_v, color="red", linestyle="--", label=f"Mean={mean_v:.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(test_losses, bins=40, color="#8e44ad", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"Loss Distribution (all {len(test_losses)} samples)")
    axes[1].set_xlabel("Loss")
    axes[1].set_ylabel("Count")
    mean_all = sum(test_losses) / len(test_losses)
    axes[1].axvline(mean_all, color="red", linestyle="--", label=f"Mean={mean_all:.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = GRAPH_DIR / f"test_loss_{run_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Test loss graph → {path}")


# ── Test evaluation ───────────────────────────────────────────────────────────

def run_test_evaluation(model, tokenizer, test_records: list,
                        max_seq_len: int, vocab_size: int,
                        max_samples: int, run_id: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Test Evaluation")
    logger.info("=" * 60)

    model.eval()

    # Phase 1: Loss
    logger.info(f"── Phase 1: Loss on {len(test_records):,} records ──")
    per_sample_losses, skipped = [], 0

    with torch.no_grad():
        for i, record in enumerate(test_records):
            result = tokenise_record(record, tokenizer, max_seq_len)
            if result is None:
                skipped += 1
                continue
            result.pop("n_active", None)

            new_labels = [
                IGNORE_INDEX if (lbl != IGNORE_INDEX and (lbl < 0 or lbl >= vocab_size))
                else lbl
                for lbl in result["labels"]
            ]
            result["labels"] = new_labels

            if all(lbl == IGNORE_INDEX for lbl in result["labels"]):
                skipped += 1
                continue

            input_ids = torch.tensor([result["input_ids"]],      dtype=torch.long)
            attn_mask = torch.tensor([result["attention_mask"]], dtype=torch.long)
            labels    = torch.tensor([result["labels"]],          dtype=torch.long)

            device = next(model.parameters()).device
            try:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)
            except Exception:
                pass

            try:
                outputs  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss_val = outputs.loss
                if loss_val is not None and not torch.isnan(loss_val):
                    per_sample_losses.append(float(loss_val.item()))
            except Exception as e:
                logger.warning(f"  Loss eval failed for record {i}: {e}")
                skipped += 1

            if (i + 1) % 50 == 0:
                completed   = len(per_sample_losses)
                mean_so_far = sum(per_sample_losses) / completed if completed else float("nan")
                logger.info(f"  [{i+1}/{len(test_records)}]  valid={completed}  mean_loss={mean_so_far:.4f}")

    mean_test_loss = sum(per_sample_losses) / len(per_sample_losses) if per_sample_losses else float("nan")
    logger.info(f"  Loss eval: {len(per_sample_losses):,} samples (skipped={skipped})  mean_loss={mean_test_loss:.4f}")
    save_test_loss_graph(per_sample_losses, run_id)

    # Phase 2: Generation
    logger.info("")
    logger.info(f"── Phase 2: Generation on first {max_samples} records ──")

    gen_records    = test_records[:max_samples]
    predictions    = []
    exact_matches  = 0
    token_overlaps = []

    with torch.no_grad():
        for i, record in enumerate(gen_records):
            messages    = record.get("messages", [])
            hf_messages = []
            for msg in messages:
                role    = "system" if msg["role"] == "developer" else msg["role"]
                content = msg.get("content", "").strip()
                if content and role != "assistant":
                    hf_messages.append({"role": role, "content": content})

            ref_answer = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    ref_answer = msg.get("content", "").strip()
                    break

            if not hf_messages or not ref_answer:
                continue

            try:
                prompt_text = tokenizer.apply_chat_template(
                    hf_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                continue

            enc = tokenizer(prompt_text, return_tensors="pt",
                            truncation=True, max_length=max_seq_len - 256)
            device = next(model.parameters()).device
            try:
                enc = {k: v.to(device) for k, v in enc.items()}
            except Exception:
                pass

            try:
                gen_ids   = model.generate(
                    **enc, max_new_tokens=256, do_sample=False, temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_ids   = gen_ids[0][enc["input_ids"].shape[1]:]
                generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception as e:
                logger.warning(f"  Generation failed for record {i}: {e}")
                continue

            em = int(generated.lower() == ref_answer.lower())
            exact_matches += em

            ref_toks = set(ref_answer.lower().split())
            gen_toks = set(generated.lower().split())
            if ref_toks or gen_toks:
                overlap = len(ref_toks & gen_toks)
                prec    = overlap / len(gen_toks) if gen_toks else 0.0
                rec     = overlap / len(ref_toks) if ref_toks else 0.0
                f1      = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0
            token_overlaps.append(f1)

            predictions.append({
                "sample_id":   record.get("sample_id", i),
                "reference":   ref_answer,
                "generated":   generated,
                "exact_match": em,
                "token_f1":    round(f1, 4),
            })

            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{len(gen_records)}]  "
                            f"EM={exact_matches}/{len(predictions)}  "
                            f"mean_F1={sum(token_overlaps)/len(token_overlaps):.4f}")

    n_gen   = len(predictions)
    em_rate = exact_matches / n_gen                       if n_gen          else float("nan")
    mean_f1 = sum(token_overlaps) / len(token_overlaps)   if token_overlaps else float("nan")

    logger.info("")
    logger.info("── Test Results ─────────────────────────────────────")
    logger.info(f"  Mean test loss     : {mean_test_loss:.4f}  ({len(per_sample_losses):,} samples)")
    logger.info(f"  Exact match        : {exact_matches}/{n_gen}  ({em_rate*100:.2f}%)")
    logger.info(f"  Mean token F1      : {mean_f1:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if token_overlaps:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Test Generation Metrics — SpaceLLM LoRA v4  [{run_id}]",
                         fontsize=13, fontweight="bold")
            axes[0].hist(token_overlaps, bins=30, color="#16a085", alpha=0.8, edgecolor="white")
            axes[0].set_title("Token F1 Distribution")
            axes[0].set_xlabel("Token F1")
            axes[0].set_ylabel("Count")
            axes[0].axvline(mean_f1, color="red", linestyle="--", label=f"Mean={mean_f1:.3f}")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[1].bar(["Exact Match", "No Match"], [exact_matches, n_gen - exact_matches],
                        color=["#27ae60", "#e74c3c"], alpha=0.8)
            axes[1].set_title(f"Exact Match  ({em_rate*100:.1f}%)")
            axes[1].set_ylabel("Count")
            axes[1].grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            gen_graph_path = GRAPH_DIR / f"test_generation_{run_id}.png"
            fig.savefig(gen_graph_path, dpi=150)
            plt.close(fig)
            logger.info(f"  Generation graph → {gen_graph_path}")
    except Exception as e:
        logger.warning(f"Generation graph failed: {e}")

    test_results = {
        "run_id":             run_id,
        "total_test_records": len(test_records),
        "loss_eval": {
            "samples_evaluated": len(per_sample_losses),
            "samples_skipped":   skipped,
            "mean_test_loss":    round(mean_test_loss, 6),
        },
        "generation_eval": {
            "samples_evaluated": n_gen,
            "exact_matches":     exact_matches,
            "exact_match_rate":  round(em_rate, 6) if n_gen else None,
            "mean_token_f1":     round(mean_f1, 6) if n_gen else None,
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
    logger.info("  SpaceLLM — LoRA v4  (lm_head only, BF16)")
    logger.info(f"  Run ID            : {RUN_ID}")
    logger.info(f"  Model             : {args.model_id}")
    logger.info(f"  Strategy          : LoRA on lm_head ONLY — backbone frozen")
    logger.info(f"  Key fix           : lm_head untied BEFORE get_peft_model()")
    logger.info(f"  Epochs            : {args.epochs}  |  LR: {args.lr}")
    logger.info(f"  Batch             : {args.batch_size}  |  Grad accum: {args.grad_accum}"
                f"  |  Eff batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  CUDA_VISIBLE_DEVS : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    logger.info(f"  Log               : {LOG_FILE}")
    logger.info("=" * 60)

    try:
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            DataCollatorForSeq2Seq, Trainer, TrainerCallback, Mxfp4Config,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
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
    logger.info(f"Chat template     : {'found' if tokenizer.chat_template else 'NOT FOUND'}")

    # ── Model load to CPU ─────────────────────────────────────────────────
    logger.info("")
    logger.info(f"Loading model to CPU: {args.model_id}  [MXFP4 → BF16 dequantize]")
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

    logger.info(f"Model loaded in {time.time() - t0:.1f}s  |  dtype: {next(model.parameters()).dtype}")

    # ── Vocab alignment ───────────────────────────────────────────────────
    logger.info("")
    logger.info("── Vocab & lm_head alignment ────────────────────────")

    # Step 1: Disable weight tying in config
    model.config.tie_word_embeddings = False

    # ===========================================================
    # CRITICAL FIX: Physically untie lm_head from embed_tokens.
    #
    # Setting tie_word_embeddings=False in the config only updates
    # the config flag — it does NOT break the live tensor tie.
    # With tie_word_embeddings=True, lm_head.weight and
    # embed_tokens.weight point to the SAME tensor object in memory.
    # PEFT wraps lm_head, but autograd sees the underlying weight
    # as belonging to the frozen embed_tokens path, so gradients
    # to lora_A.weight are cut off at the tied tensor.
    # detach().clone() materializes lm_head as its own independent
    # Parameter BEFORE get_peft_model() ever touches the model.
    # ===========================================================
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info("✅ lm_head weight untied and cloned as independent tensor")

    # Step 2: Resize vocab
    _current_vocab = model.get_output_embeddings().weight.shape[0]
    model.resize_token_embeddings(_current_vocab, pad_to_multiple_of=64)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    actual_vocab = model.get_output_embeddings().weight.shape[0]
    model.config.vocab_size = actual_vocab

    assert model.get_input_embeddings().weight.shape[0]  == actual_vocab
    assert model.get_output_embeddings().weight.shape[0] == actual_vocab
    logger.info(f"  Vocab alignment PASSED  (vocab={actual_vocab:,}  padded to multiple of 64)")

    # Verify untie was not undone by resize
    embed_id  = id(model.get_input_embeddings().weight)
    lm_head_id = id(model.get_output_embeddings().weight)
    if embed_id == lm_head_id:
        logger.error("  ❌ FATAL: resize_token_embeddings re-tied lm_head to embed_tokens!")
        logger.error("  Call lm_head.weight = nn.Parameter(...) again after resize.")
        lm_head = model.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        logger.info("  ✅ Re-untied lm_head after resize.")
    else:
        logger.info("  ✅ lm_head still independent after resize (tie not re-introduced)")

    model.config.use_cache = False

    # ── Inject loss pre-PEFT ──────────────────────────────────────────────
    logger.info("")
    logger.info("── Injecting CE loss (pre-PEFT) ─────────────────────")
    _inject_loss_function(model, label="pre-PEFT")

    # =========================================================================
    # CORRECT ORDER (v4):
    #   1. Load model to CPU                     ← done
    #   2. Vocab alignment + untie lm_head       ← done  *** THE KEY FIX ***
    #   3. get_peft_model() on raw CPU model     ← PEFT wraps untied lm_head
    #   4. enable_input_require_grads()          ← hook on correct forward()
    #   5. gradient_checkpointing_enable()       ← after PEFT
    #   6. Explicit freeze of all non-LoRA params
    #   7. dispatch_model() to GPUs              ← dispatch wraps PEFT model
    #   8. _inject_loss_function() post-dispatch
    #   9. verify_gradient_flow()                ← must pass
    # =========================================================================

    # ── Step 3: Apply LoRA on CPU ─────────────────────────────────────────
    logger.info("")
    logger.info("Applying LoRA to lm_head ONLY (on CPU, before dispatch) ...")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,       # 4×r
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    logger.info("✅ get_peft_model() applied on raw CPU model (lm_head already untied)")

    # ── Step 4 & 5: enable grads + gradient checkpointing ────────────────
    model.enable_input_require_grads()
    logger.info("✅ enable_input_require_grads() called")

    model.gradient_checkpointing_enable()
    logger.info("✅ gradient_checkpointing_enable() called (post-PEFT)")

    # ── Step 6: Explicit freeze ───────────────────────────────────────────
    logger.info("")
    logger.info("── Explicit parameter freeze (pre-dispatch) ─────────")
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
        logger.error("  ❌ Non-LoRA params still trainable:")
        for n, s in leaked:
            logger.error(f"     {n}  {s}")
        raise SystemExit(1)
    logger.info("  ✅ No non-LoRA params leaked as trainable")

    # Re-inject loss post-PEFT
    logger.info("")
    logger.info("── Re-injecting CE loss (post-PEFT, pre-dispatch) ───")
    _inject_loss_function(model, label="post-PEFT pre-dispatch")

    log_trainable_parameters(model)
    log_gpu_memory("after LoRA init (CPU)")

    # ── Step 7: GPU dispatch ──────────────────────────────────────────────
    logger.info("")
    logger.info("Dispatching PEFT model across GPUs ...")
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

    # ── Step 8: Re-inject loss post-dispatch ─────────────────────────────
    logger.info("")
    logger.info("── Re-injecting CE loss (post-dispatch) ─────────────")
    _inject_loss_function(model, label="post-dispatch")

    # ── Vocab check post-dispatch ─────────────────────────────────────────
    logger.info("")
    logger.info("── Vocab alignment (post-dispatch) ──────────────────")
    _post_peft_vocab = model.get_output_embeddings().weight.shape[0]
    if _post_peft_vocab != model.config.vocab_size:
        logger.warning(f"  lm_head vocab ({_post_peft_vocab}) != config ({model.config.vocab_size}) — fixing")
        model.config.vocab_size = _post_peft_vocab
    logger.info(f"  lm_head vocab = {_post_peft_vocab:,}  ✅")

  
  

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
        vocab_size=_post_peft_vocab, debug_first_n=3,
    )
    val_dataset = build_hf_dataset(
        val_records, tokenizer, args.max_seq_len, "validation",
        vocab_size=_post_peft_vocab, debug_first_n=0,
    )

    # ── Training arguments ────────────────────────────────────────────────
    MAX_GRAD_NORM = 0.3

    logger.info("")
    logger.info("── Training configuration ───────────────────────────")
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=args.warmup_steps,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch_fused",
        weight_decay=0.01,
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
        "weight_decay":     0.01,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  args.batch_size * args.grad_accum,
        "warmup_steps":     args.warmup_steps,
        "lora_r":           32,
        "lora_alpha":       128,
        "optimizer":        "adamw_torch_fused",
        "scheduler":        "cosine_with_restarts",
        "bf16":             True,
        "max_seq_len":      args.max_seq_len,
        "early_stop_patce": args.patience,
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

    # ── Callbacks ─────────────────────────────────────────────────────────
    history = HistoryCallback()

    class _TrainingHistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            history.on_log(args, state, control, logs=logs, **kwargs)

    history_callback    = _TrainingHistoryCallback()
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=args.patience)

    # ── Trainer ───────────────────────────────────────────────────────────
    class DeviceAwareTrainer(Trainer):
        """
        1. _prepare_inputs() moves labels to lm_head device.
        2. training_step() adds explicit clip_grad_norm_ on trainable params
           as belt-and-suspenders alongside max_grad_norm in TrainingArguments.
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

            # Explicit clip — guards against multi-GPU/BF16 scaler clip misses
            trainable_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)

            return loss

    trainer = DeviceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[history_callback, early_stop_callback],
    )

    # Final loss patch post-Trainer init
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
        save_training_graphs(history, RUN_ID)
        logger.info(f"Saved to: {interrupted_dir}")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise SystemExit(1)

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min")

    # ── Save graphs ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Saving training graphs ───────────────────────────")
    save_training_graphs(history, RUN_ID)

    # ── Metrics ───────────────────────────────────────────────────────────
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    with (LOG_DIR / f"train_metrics_{RUN_ID}.json").open("w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info("Running final validation evaluation ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)

    with (LOG_DIR / f"eval_metrics_{RUN_ID}.json").open("w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ── Save final LoRA adapters ──────────────────────────────────────────
    logger.info("")
    logger.info(f"Saving final LoRA adapters → {FINAL_DIR}")
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    # ── Test evaluation ───────────────────────────────────────────────────
    test_results = None
    if test_records:
        test_results = run_test_evaluation(
            model=trainer.model, tokenizer=tokenizer,
            test_records=test_records, max_seq_len=args.max_seq_len,
            vocab_size=_post_peft_vocab, max_samples=args.max_test_samples,
            run_id=RUN_ID,
        )
    else:
        logger.warning("Skipping test evaluation — test.json not found")

    # ── Adapter info ──────────────────────────────────────────────────────
    adapter_info = {
        "run_id":          RUN_ID,
        "version":         "v4",
        "base_model":      args.model_id,
        "strategy":        "LoRA on lm_head ONLY — backbone frozen — BF16",
        "key_fixes_vs_v3_fixed": [
            "lm_head weight untied (detach+clone) BEFORE get_peft_model()",
            "resize_token_embeddings tie-re-introduction guard added",
            "lora_r raised 16→32, lora_alpha raised 64→128",
            "grad_accum raised 16→32, lr raised 1e-4→2e-4",
            "max_grad_norm tightened 0.5→0.3, weight_decay=0.01 added",
            "EarlyStoppingCallback added",
            "scheduler changed cosine→cosine_with_restarts",
            "optimizer changed adamw_torch→adamw_torch_fused",
        ],
        "lora_r":                32,
        "lora_alpha":            128,
        "lora_dropout":          0.1,
        "target_modules":        ["lm_head"],
        "epochs":                args.epochs,
        "learning_rate":         args.lr,
        "max_grad_norm":         MAX_GRAD_NORM,
        "weight_decay":          0.01,
        "batch_size":            args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size":  args.batch_size * args.grad_accum,
        "max_seq_len":           args.max_seq_len,
        "train_samples":         len(train_dataset),
        "val_samples":           len(val_dataset),
        "test_samples":          len(test_records),
        "train_metrics":         train_metrics,
        "eval_metrics":          eval_metrics,
        "test_results_summary": (
            {
                "mean_test_loss":   test_results["loss_eval"]["mean_test_loss"],
                "exact_match_rate": test_results["generation_eval"]["exact_match_rate"],
                "mean_token_f1":    test_results["generation_eval"]["mean_token_f1"],
            } if test_results else None
        ),
        "final_adapter_dir": str(FINAL_DIR),
        "checkpoints_dir":   str(CKPT_DIR),
        "graphs_dir":        str(GRAPH_DIR),
        "log_file":          str(LOG_FILE),
    }
    with (FINAL_DIR / "adapter_info.json").open("w") as f:
        json.dump(adapter_info, f, indent=2)

    log_gpu_memory("after training")

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SpaceLLM lm_head LoRA v4 — Training Complete")
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
        logger.info(f"    Mean test loss    : {le['mean_test_loss']:.4f}  ({le['samples_evaluated']:,} samples)")
        if ge["exact_match_rate"] is not None:
            logger.info(f"    Exact match       : {ge['exact_match_rate']*100:.2f}%")
            logger.info(f"    Mean token F1     : {ge['mean_token_f1']:.4f}")
    logger.info("")
    logger.info("  To load for evaluation:")
    logger.info("    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config")
    logger.info("    from peft import PeftModel")
    logger.info(f"    base  = AutoModelForCausalLM.from_pretrained('{args.model_id}',")
    logger.info(f"                quantization_config=Mxfp4Config(dequantize=True), device_map='auto')")
    logger.info(f"    model = PeftModel.from_pretrained(base, '{FINAL_DIR}')")
    logger.info(f"    tok   = AutoTokenizer.from_pretrained('{FINAL_DIR}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
