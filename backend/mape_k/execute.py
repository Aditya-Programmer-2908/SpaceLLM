"""
SpaceLLM MAPE-K :: Executor Component  (v3 — Dataset-Expansion + Merge + Auto-Training)
==========================================================================================
Changes from v2:
  1. DatasetExporter now MERGES core train.json (DatasetA_core_QA_v2/train.json) into
     combined_dataset.json on first write, so correction fine-tuning always trains on
     the full corpus, not just the small set of MAPE-K corrections.
  2. CORRECTION_FINE_TUNING_SCRIPT path corrected to train_spacellm_fresh.py.
  3. Auto-training CLI flags aligned with train_spacellm_fresh.py's argparse spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.executor")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR             = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR             = BASE_DIR / "mape_k"

PLAN_ACTIONS_LOG     = MAPE_DIR / "plan_actions.jsonl"
EXECUTION_LOG        = MAPE_DIR / "execution_log.jsonl"
FEEDBACK_LOG         = BASE_DIR / "feedback_log.jsonl"
FRONTEND_PATCH_FILE  = BASE_DIR / "frontend_patch.md"
TOPIC_GUARDRAIL_FILE = MAPE_DIR / "topic_guardrail.json"
REVIEW_QUEUE_FILE    = MAPE_DIR / "human_review_queue.jsonl"
REVIEW_STATS_FILE    = MAPE_DIR / "review_stats.json"
STATE_FILE           = MAPE_DIR / ".executor_state.json"
FAILED_LOG           = MAPE_DIR / "executor_failed.jsonl"

# ── v2/v3: dataset expansion ──────────────────────────────────────────────
COMBINED_DATASET_FILE = BASE_DIR / "combined_dataset.json"
DATASET_BACKUP_DIR    = MAPE_DIR / "dataset_backups"

# ── FIX 1: Core training dataset to merge into combined_dataset.json ──────
# This is the original SpaceLLM training corpus. Every time execute.py
# expands the dataset, corrections are merged ON TOP of this base corpus so
# that fine-tuning always sees the full dataset, not just the small correction
# batch.
CORE_TRAIN_FILE = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction"
    "/data_processing/DatasetA_core_QA_v2/train.json"
)

# ── Knowledge repository (inference-time RAG) ─────────────────────────────
KNOWLEDGE_REPO_FILE  = MAPE_DIR / "knowledge_repository.json"
KNOWLEDGE_BACKUP_DIR = MAPE_DIR / "knowledge_repository_backups"

# ── FIX 2: Correct training script filename ───────────────────────────────
# The script is train_spacellm_fresh.py, NOT correction_fine_tuning.py.
CORRECTION_FINE_TUNING_SCRIPT = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction"
    "/fine_tuning_v2/train_spacellm_fresh.py"
)
TRAINING_OUTPUT_DIR = BASE_DIR / "spacellm_adapters"

MAPE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
DATASET_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPACELLM_SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "spacecraft technology, and scientific discoveries. Answer questions clearly "
    "and accurately based on your knowledge of space exploration. Tailor the "
    "depth of your response to the complexity of the question."
)

_SAMPLE_ID_PREFIX = "SPC"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutorConfig:
    patch_file_max_bytes:       int   = 500_000
    retrieval_top_k:            int   = 3
    retrieval_min_similarity:   float = 0.55
    dedup_similarity_threshold: float = 0.97
    # ── Auto-training ─────────────────────────────────────────────────────
    auto_train:                 bool  = True
    """Automatically trigger train_spacellm_fresh.py after DATASET_EXPANSION"""
    auto_train_epochs:          int   = 15
    auto_train_lr:              float = 2e-4
    auto_train_batch_size:      int   = 1
    auto_train_grad_accum:      int   = 32
    auto_train_lora_r:          int   = 32
    auto_train_lora_alpha:      int   = 128
    auto_train_max_seq_len:     int   = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionRecord:
    execution_id: str
    action_id:    str
    action_type:  str
    plan_id:      str
    status:       str            # "EXECUTED" | "FAILED"
    started_at:   str
    finished_at:  str
    duration_s:   float
    result:       dict[str, Any] = field(default_factory=dict)
    error:        str | None     = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic-write helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDER = None
_EMBEDDER_BACKEND: str | None = None
_HASH_DIM = 384


def _get_embedder():
    global _EMBEDDER, _EMBEDDER_BACKEND
    if _EMBEDDER_BACKEND is not None:
        return _EMBEDDER, _EMBEDDER_BACKEND
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get(
            "SPACELLM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_BACKEND = "sentence-transformers"
        log.info("Embedding backend: sentence-transformers (%s)", model_name)
    except Exception as exc:
        log.warning(
            "sentence-transformers unavailable (%s) — falling back to hashed "
            "bag-of-words embeddings.", exc,
        )
        _EMBEDDER = None
        _EMBEDDER_BACKEND = "hashing_fallback"
    return _EMBEDDER, _EMBEDDER_BACKEND


def _hashing_embed(text: str) -> list[float]:
    vec = np.zeros(_HASH_DIM, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % _HASH_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def embed_text(text: str) -> list[float]:
    model, backend = _get_embedder()
    if backend == "sentence-transformers":
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    return _hashing_embed(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Repository
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeRepository:
    """Inference-time RAG; bridges gap between retrains."""

    def __init__(self, path: Path = KNOWLEDGE_REPO_FILE,
                 dedup_threshold: float = 0.97) -> None:
        self.path = path
        self.dedup_threshold = dedup_threshold
        self._entries: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:
            log.warning("Could not parse %s (%s) — starting with empty repository.", self.path, exc)
            return []

    def _save(self) -> None:
        _atomic_write_json(self.path, self._entries)

    def _backup(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = KNOWLEDGE_BACKUP_DIR / f"knowledge_repository_{ts}.json"
        backup_path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return backup_path

    def add(self, question: str, correct_answer: str, *,
            feedback_id: str | None = None,
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        embedding = embed_text(question)
        entry = {
            "id":             str(uuid.uuid4()),
            "question":       question,
            "correct_answer": correct_answer,
            "embedding":      embedding,
            "feedback_id":    feedback_id,
            "created_at":     datetime.now(timezone.utc).isoformat(),
            "metadata":       metadata or {},
        }
        for i, existing in enumerate(self._entries):
            if cosine_similarity(existing["embedding"], embedding) >= self.dedup_threshold:
                entry["id"] = existing["id"]
                self._entries[i] = entry
                self._save()
                return entry
        self._entries.append(entry)
        self._save()
        return entry

    def query(self, question: str, top_k: int = 3,
              min_similarity: float = 0.0) -> list[dict[str, Any]]:
        if not self._entries:
            return []
        q_emb = embed_text(question)
        scored = []
        for entry in self._entries:
            sim = cosine_similarity(q_emb, entry["embedding"])
            if sim >= min_similarity:
                scored.append({**entry, "similarity": sim})
        scored.sort(key=lambda e: e["similarity"], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._entries)


def build_augmented_prompt(
    query: str,
    repository: "KnowledgeRepository | None" = None,
    top_k: int = 3,
    min_similarity: float = 0.55,
) -> str:
    """Inference-time entry point for RAG augmentation."""
    repository = repository or KnowledgeRepository()
    hits = repository.query(query, top_k=top_k, min_similarity=min_similarity)
    if not hits:
        return query

    lines = [
        "# Relevant prior human-verified corrections "
        "(use as grounding context; do not quote verbatim unless directly applicable):"
    ]
    for i, hit in enumerate(hits, 1):
        lines.append(
            f"{i}. Q: {hit['question']}\n   Verified correct answer: {hit['correct_answer']}"
        )
    lines.append("\n# User question:")
    lines.append(query)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataset Loader  [FIX 1 — NEW]
# ─────────────────────────────────────────────────────────────────────────────

def _load_core_train_records() -> list[dict[str, Any]]:
    """
    Load DatasetA_core_QA_v2/train.json.

    The file is expected to be a JSON list of records that already use the
    combined_dataset schema (i.e. each record has a "messages" list with
    developer/user/assistant turns).  Records that don't match the schema are
    skipped with a warning so a corrupt core file never blocks a MAPE-K cycle.
    """
    if not CORE_TRAIN_FILE.exists():
        log.warning(
            "Core train file not found: %s — combined_dataset will contain "
            "only MAPE-K corrections this cycle.", CORE_TRAIN_FILE
        )
        return []

    try:
        raw = json.loads(CORE_TRAIN_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to parse core train file %s: %s", CORE_TRAIN_FILE, exc)
        return []

    if not isinstance(raw, list):
        log.error(
            "Core train file %s is not a JSON list (got %s) — skipping.",
            CORE_TRAIN_FILE, type(raw).__name__
        )
        return []

    valid, skipped = [], 0
    for rec in raw:
        msgs = rec.get("messages", [])
        has_user = any(m.get("role") in ("user",) for m in msgs)
        has_asst = any(
            m.get("role") == "assistant" and (m.get("content") or "").strip()
            for m in msgs
        )
        if not (has_user and has_asst):
            skipped += 1
            continue
        valid.append(rec)

    if skipped:
        log.warning(
            "Core train file: skipped %d / %d records (missing user or assistant turn).",
            skipped, len(raw)
        )
    log.info(
        "Loaded %d core training records from %s.", len(valid), CORE_TRAIN_FILE
    )
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Exporter  [FIX 1 — UPDATED to merge core + corrections]
# ─────────────────────────────────────────────────────────────────────────────

class DatasetExporter:
    """
    Manages COMBINED_DATASET_FILE — unified training corpus.

    On first use (or whenever combined_dataset.json is absent / empty) this
    class loads the core train.json and seeds combined_dataset.json with it.
    Subsequent DATASET_EXPANSION calls then append/overwrite only the
    correction records ON TOP of the already-present core records, so the
    fine-tuning script always receives the full corpus.
    """

    def __init__(self, path: Path = COMBINED_DATASET_FILE) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = self._load_or_seed()
        self._next_id: int = self._infer_next_id()

    def _load_or_seed(self) -> list[dict[str, Any]]:
        """
        Load combined_dataset.json if it exists and is non-empty.
        Otherwise seed it with the core train.json records.
        """
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    log.info(
                        "Loaded %d existing records from %s.", len(data), self.path
                    )
                    return data
                log.info(
                    "combined_dataset.json exists but is empty — seeding from core train file."
                )
            except Exception as exc:
                log.warning(
                    "Could not parse %s (%s) — re-seeding from core train file.", self.path, exc
                )
        else:
            log.info(
                "combined_dataset.json not found at %s — seeding from core train file.", self.path
            )

        # Seed from core dataset
        core_records = _load_core_train_records()
        if core_records:
            _atomic_write_json(self.path, core_records)
            log.info(
                "Seeded combined_dataset.json with %d core records from %s.",
                len(core_records), CORE_TRAIN_FILE
            )
        return core_records

    def _save(self) -> None:
        _atomic_write_json(self.path, self._records)

    def backup(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = DATASET_BACKUP_DIR / f"combined_dataset_{ts}.json"
        backup_path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Dataset backed up → %s", backup_path)
        return backup_path

    def _infer_next_id(self) -> int:
        max_id = 0
        for rec in self._records:
            sid = rec.get("sample_id", "")
            if sid.startswith(f"{_SAMPLE_ID_PREFIX}_"):
                try:
                    max_id = max(max_id, int(sid.split("_", 1)[1]))
                except ValueError:
                    pass
        return max_id + 1

    def _next_sample_id(self) -> str:
        sid = f"{_SAMPLE_ID_PREFIX}_{self._next_id:06d}"
        self._next_id += 1
        return sid

    def _build_training_record(
        self,
        question: str,
        correct_answer: str,
        *,
        feedback_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = metadata or {}
        fid  = feedback_id or str(uuid.uuid4())

        source_id  = f"mapek_correction_{fid}"
        chain_id   = source_id
        sample_id  = self._next_sample_id()

        mission_name = (
            meta.get("mission_name")
            or meta.get("mission")
            or _extract_mission_hint(question)
            or "SpaceLLM MAPE-K Correction"
        )
        organization = meta.get("organization", "NASA")
        aspect       = meta.get("aspect", "CORRECTION").upper()
        difficulty   = meta.get("difficulty", "basic")
        source_url   = meta.get("source_url", "")

        return {
            "sample_id":    sample_id,
            "source_id":    source_id,
            "mission_name": mission_name,
            "organization": organization,
            "aspect":       aspect,
            "difficulty":   difficulty,
            "chain_id":     chain_id,
            "source_url":   source_url,
            "messages": [
                {"role": "developer", "content": SPACELLM_SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": correct_answer},
            ],
        }

    def append_corrections(
        self,
        examples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Merge correction examples into self._records.

        Matching is done by normalised question text.  If a question already
        exists (either from a prior correction or from the core corpus) the
        record is overwritten in-place so the corrected answer takes priority.
        """
        added:       list[dict[str, Any]] = []
        overwritten: int = 0

        existing_by_question: dict[str, int] = {
            _norm(self._extract_question(r)): i
            for i, r in enumerate(self._records)
        }

        for ex in examples:
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid       = ex.get("feedback_id") or str(uuid.uuid4())

            if not question or not reference:
                log.warning("Skipping example %s — empty question or reference.", fid)
                continue

            meta = {k: ex[k] for k in (
                "mission_name", "organization", "aspect",
                "difficulty", "source_url", "bertscore",
            ) if k in ex}
            meta["feedback_id"] = fid

            new_rec = self._build_training_record(
                question, reference, feedback_id=fid, metadata=meta
            )

            norm_q = _norm(question)
            if norm_q in existing_by_question:
                idx = existing_by_question[norm_q]
                # Preserve the original sample_id so training history is stable
                new_rec["sample_id"] = self._records[idx]["sample_id"]
                self._records[idx] = new_rec
                overwritten += 1
                log.info(
                    "  Overwrote existing record for question: %.80s…", question
                )
            else:
                self._records.append(new_rec)
                existing_by_question[norm_q] = len(self._records) - 1

            added.append(new_rec)

        if added:
            self._save()

        return added, overwritten

    @staticmethod
    def _extract_question(record: dict) -> str:
        for msg in record.get("messages", []):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    def __len__(self) -> int:
        return len(self._records)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_mission_hint(question: str) -> str | None:
    patterns = [
        r"(?:the\s+)?([A-Z][A-Za-z0-9 \-]+?)\s+[Mm]ission",
        r"[Mm]ission\s+([A-Z][A-Za-z0-9 \-]+)",
        r"(?:aboard|on board|on|from)\s+([A-Z][A-Za-z0-9 \-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, question)
        if m:
            candidate = m.group(1).strip()
            if 2 < len(candidate) < 60:
                return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Training Trigger  [FIX 2 — CLI flags aligned with train_spacellm_fresh.py]
# ─────────────────────────────────────────────────────────────────────────────

def trigger_auto_training(config: ExecutorConfig) -> dict[str, Any]:
    """
    Trigger train_spacellm_fresh.py after a successful DATASET_EXPANSION.

    All CLI flags are taken from train_spacellm_fresh.py's argparse spec:
        --train_file, --output_dir, --epochs, --lr, --batch_size,
        --grad_accum, --lora_r, --lora_alpha, --max_seq_len, --hf_token

    Returns a status dict with keys:
        triggered, status, reason, training_output_dir, training_log,
        training_return_code
    """
    if not config.auto_train:
        return {
            "triggered": False,
            "status": "skipped",
            "reason": "auto_train=False in ExecutorConfig",
        }

    if not CORRECTION_FINE_TUNING_SCRIPT.exists():
        return {
            "triggered": False,
            "status": "failed",
            "reason": (
                f"Training script not found: {CORRECTION_FINE_TUNING_SCRIPT}\n"
                f"Expected: train_spacellm_fresh.py"
            ),
        }

    if not COMBINED_DATASET_FILE.exists():
        return {
            "triggered": False,
            "status": "failed",
            "reason": f"combined_dataset.json not found: {COMBINED_DATASET_FILE}",
        }

    # Verify the combined dataset has actual content
    try:
        ds = json.loads(COMBINED_DATASET_FILE.read_text(encoding="utf-8"))
        if not isinstance(ds, list) or len(ds) == 0:
            return {
                "triggered": False,
                "status": "failed",
                "reason": "combined_dataset.json is empty — nothing to train on.",
            }
        log.info("  Training dataset: %d records in %s", len(ds), COMBINED_DATASET_FILE)
    except Exception as exc:
        return {
            "triggered": False,
            "status": "failed",
            "reason": f"Could not read combined_dataset.json: {exc}",
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = TRAINING_OUTPUT_DIR / f"spacellm_adapter_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("  AUTO-TRAINING: Triggering %s", CORRECTION_FINE_TUNING_SCRIPT.name)
    log.info("=" * 70)
    log.info("  Train file    : %s", COMBINED_DATASET_FILE)
    log.info("  Output dir    : %s", output_dir)
    log.info("  Epochs        : %d", config.auto_train_epochs)
    log.info("  Learning rate : %g", config.auto_train_lr)
    log.info("  Batch size    : %d", config.auto_train_batch_size)
    log.info("  Grad accum    : %d", config.auto_train_grad_accum)
    log.info("  LoRA r        : %d", config.auto_train_lora_r)
    log.info("  LoRA alpha    : %d", config.auto_train_lora_alpha)
    log.info("  Max seq len   : %d", config.auto_train_max_seq_len)
    log.info("=" * 70)

    # FIX 2: CLI flags match train_spacellm_fresh.py's argparse exactly
    cmd = [
        sys.executable,
        str(CORRECTION_FINE_TUNING_SCRIPT),
        "--train_file",  str(COMBINED_DATASET_FILE),
        "--output_dir",  str(output_dir),
        "--epochs",      str(config.auto_train_epochs),
        "--lr",          str(config.auto_train_lr),
        "--batch_size",  str(config.auto_train_batch_size),
        "--grad_accum",  str(config.auto_train_grad_accum),
        "--lora_r",      str(config.auto_train_lora_r),
        "--lora_alpha",  str(config.auto_train_lora_alpha),
        "--max_seq_len", str(config.auto_train_max_seq_len),
    ]

    # Pass HF token via CLI (train_spacellm_fresh.py reads --hf_token)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        cmd.extend(["--hf_token", hf_token])

    training_log_file = MAPE_DIR / f"auto_training_{ts}.log"
    log.info("  Training log  : %s", training_log_file)

    try:
        with training_log_file.open("w", encoding="utf-8") as logfh:
            result = subprocess.run(
                cmd,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                timeout=None,   # training can take hours
                text=True,
            )

        if result.returncode == 0:
            log.info("✓ Auto-training completed successfully")
            log.info("  Adapter saved to: %s", output_dir)
            return {
                "triggered":            True,
                "status":               "success",
                "training_output_dir":  str(output_dir),
                "training_log":         str(training_log_file),
                "training_return_code": result.returncode,
            }
        else:
            log.error("✗ Auto-training failed (return code %d)", result.returncode)
            log.error("  Check log: %s", training_log_file)
            return {
                "triggered":            True,
                "status":               "failed",
                "reason":               f"Training subprocess returned {result.returncode}",
                "training_log":         str(training_log_file),
                "training_return_code": result.returncode,
            }

    except subprocess.TimeoutExpired:
        return {
            "triggered":    True,
            "status":       "failed",
            "reason":       "Training subprocess timed out",
            "training_log": str(training_log_file),
        }
    except Exception as exc:
        log.error("✗ Failed to launch auto-training: %s", exc, exc_info=True)
        return {
            "triggered": True,
            "status":    "failed",
            "reason":    str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class Executor:

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config  = config or ExecutorConfig()
        self._state  = self._load_state()
        self._repo   = KnowledgeRepository(
            dedup_threshold=self.config.dedup_similarity_threshold
        )
        self._export = DatasetExporter()
        self._dataset_expansion_executed = False
        log.info(
            "Executor initialised. "
            "executed_action_ids=%d  "
            "knowledge_repo_entries=%d  "
            "combined_dataset_size=%d  "
            "auto_train=%s",
            len(self._state.get("executed_ids", [])),
            len(self._repo),
            len(self._export),
            self.config.auto_train,
        )

    # ── Public entry point ────────────────────────────────────────────────

    def run(self) -> list[ExecutionRecord]:
        """
        Full execution cycle:
          1. Load PENDING / auto_approved actions from plan_actions.jsonl.
          2. Sort by priority descending (CRITICAL first).
          3. Execute each in isolation — one failure cannot poison the batch.
          4. Persist records, update statuses, save state.
          5. If any DATASET_EXPANSION succeeded and auto_train=True,
             trigger train_spacellm_fresh.py on the merged combined_dataset.
        """
        log.info("=" * 60)
        log.info("Executor cycle starting.")

        pending = self._load_pending_actions()
        if not pending:
            log.info("No new PENDING actions to execute.")
            return []

        pending.sort(key=lambda a: a.get("priority", 0), reverse=True)
        log.info("Executing %d pending action(s).", len(pending))

        records:     list[ExecutionRecord] = []
        updated_ids: dict[str, str]        = {}

        for action in pending:
            action_id   = action.get("action_id", str(uuid.uuid4()))
            action_type = action.get("action_type", "UNKNOWN")
            plan_id     = action.get("plan_id", "unknown")
            started_at  = datetime.now(timezone.utc).isoformat()

            log.info("→ [%s] action_id=%s  priority=%s",
                     action_type, action_id[:8], action.get("priority"))

            try:
                result      = self._dispatch(action)
                finished_at = datetime.now(timezone.utc).isoformat()
                duration_s  = (
                    datetime.fromisoformat(finished_at) -
                    datetime.fromisoformat(started_at)
                ).total_seconds()
                rec = ExecutionRecord(
                    execution_id = str(uuid.uuid4()),
                    action_id    = action_id,
                    action_type  = action_type,
                    plan_id      = plan_id,
                    status       = "EXECUTED",
                    started_at   = started_at,
                    finished_at  = finished_at,
                    duration_s   = round(duration_s, 3),
                    result       = result,
                )
                updated_ids[action_id] = "EXECUTED"
                log.info("  ✓ EXECUTED in %.3fs", duration_s)

                if action_type in ("DATASET_EXPANSION", "RETRIEVAL_MEMORY_UPDATE", "RETRAIN_ADAPTER"):
                    self._dataset_expansion_executed = True

            except Exception as exc:
                finished_at = datetime.now(timezone.utc).isoformat()
                duration_s  = (
                    datetime.fromisoformat(finished_at) -
                    datetime.fromisoformat(started_at)
                ).total_seconds()
                log.error("  ✗ FAILED action %s (%s): %s",
                          action_id[:8], action_type, exc, exc_info=True)
                self._log_failed_action(action, exc)
                rec = ExecutionRecord(
                    execution_id = str(uuid.uuid4()),
                    action_id    = action_id,
                    action_type  = action_type,
                    plan_id      = plan_id,
                    status       = "FAILED",
                    started_at   = started_at,
                    finished_at  = finished_at,
                    duration_s   = round(duration_s, 3),
                    error        = str(exc),
                )
                updated_ids[action_id] = "FAILED"

            self._append_execution_log(rec)
            records.append(rec)
            self._state.setdefault("executed_ids", []).append(action_id)

        self._update_action_statuses(updated_ids)

        self._state["dataset_next_sample_id"] = self._export._next_id
        self._save_state()

        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        log.info("Cycle complete. %d EXECUTED, %d FAILED.", executed, failed)

        # Trigger auto-training if any dataset expansion succeeded
        training_status: dict[str, Any] = {}
        if self._dataset_expansion_executed and self.config.auto_train:
            training_status = trigger_auto_training(self.config)
            self._state["last_auto_training"] = training_status
            self._save_state()

        log.info("=" * 60)
        return records

    # ── Dispatcher ────────────────────────────────────────────────────────

    def _dispatch(self, action: dict) -> dict[str, Any]:
        action_type = action.get("action_type")
        payload     = action.get("payload", {})
        handler = {
            "DATASET_EXPANSION":       self._execute_dataset_expansion,
            "RETRIEVAL_MEMORY_UPDATE": self._execute_dataset_expansion,
            "RETRAIN_ADAPTER":         self._execute_dataset_expansion,
            "PROMPT_PATCH":            self._execute_prompt_patch,
            "TOPIC_GUARDRAIL":         self._execute_topic_guardrail,
            "FLAG_FOR_REVIEW":         self._execute_flag_for_review,
            "NO_ACTION":               self._execute_no_action,
        }.get(action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action_type!r}")
        return handler(payload, action)

    # ── Handler: DATASET_EXPANSION ────────────────────────────────────────

    def _execute_dataset_expansion(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        examples = payload.get("training_examples", [])
        if not examples:
            raise ValueError(
                "DATASET_EXPANSION payload has no training_examples. "
                "Expected key: 'training_examples' with list of "
                "{question, reference[, feedback_id, mission_name, ...]} dicts."
            )

        backup_path = self._export.backup()
        added_records, overwritten = self._export.append_corrections(examples)

        if not added_records:
            raise ValueError("All training examples were invalid — nothing written to dataset.")

        log.info(
            "Dataset expansion: %d record(s) written (%d new, %d overwrote existing). "
            "combined_dataset.json now has %d records total (core + corrections).",
            len(added_records),
            len(added_records) - overwritten,
            overwritten,
            len(self._export),
        )

        repo_backup = self._repo._backup()
        repo_added: list[str] = []
        valid_feedback_ids: list[str] = []

        for ex in examples:
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid       = ex.get("feedback_id", "")

            if not question or not reference:
                continue

            entry = self._repo.add(
                question, reference,
                feedback_id=fid,
                metadata={"bertscore": ex.get("bertscore")},
            )
            repo_added.append(entry["id"])
            if fid:
                valid_feedback_ids.append(fid)

        marked = self._mark_used_in_training(set(valid_feedback_ids))

        self._state["last_dataset_expansion"] = {
            "records_written":  len(added_records),
            "overwritten":      overwritten,
            "dataset_size":     len(self._export),
            "repo_size":        len(self._repo),
            "backup_path":      str(backup_path),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

        return {
            "combined_dataset_file":    str(COMBINED_DATASET_FILE),
            "dataset_backup_path":      str(backup_path),
            "core_train_file":          str(CORE_TRAIN_FILE),
            "records_written":          len(added_records),
            "records_overwritten":      overwritten,
            "dataset_size_after":       len(self._export),
            "added_sample_ids":         [r["sample_id"] for r in added_records],
            "knowledge_repo_file":      str(KNOWLEDGE_REPO_FILE),
            "knowledge_backup_path":    str(repo_backup),
            "repo_entries_after":       len(self._repo),
            "marked_in_feedback":       marked,
            "embedding_backend":        _EMBEDDER_BACKEND or "uninitialised",
            "next_step": (
                "(Auto-training will be triggered automatically — "
                f"trains on full merged corpus of {len(self._export)} records)"
                if self.config.auto_train
                else f"Run: python {CORRECTION_FINE_TUNING_SCRIPT} "
                     f"--train_file {COMBINED_DATASET_FILE} "
                     f"--output_dir ./spacellm_v_next_adapter"
            ),
        }

    def _mark_used_in_training(self, feedback_ids: set[str]) -> int:
        if not FEEDBACK_LOG.exists() or not feedback_ids:
            return 0
        updated:   int         = 0
        new_lines: list[str]   = []
        with FEEDBACK_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    if record.get("feedback_id") in feedback_ids:
                        record["used_in_training"] = True
                        updated += 1
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(stripped)
        tmp = FEEDBACK_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(FEEDBACK_LOG)
        return updated

    # ── Handler: PROMPT_PATCH ─────────────────────────────────────────────

    def _execute_prompt_patch(self, payload: dict, action: dict) -> dict[str, Any]:
        patch_key  = payload.get("patch_key", "unknown_patch")
        patch_text = payload.get("patch_text", "").strip()
        target     = payload.get("target", "system_prompt")

        if not patch_text:
            raise ValueError(f"PROMPT_PATCH has empty patch_text for key '{patch_key}'.")

        current_size = FRONTEND_PATCH_FILE.stat().st_size if FRONTEND_PATCH_FILE.exists() else 0
        if current_size > self.config.patch_file_max_bytes:
            log.warning(
                "frontend_patch.md is %.1f KB — consider archiving old patches.",
                current_size / 1024,
            )

        applied_at = datetime.now(timezone.utc).isoformat()
        block = (
            f"\n<!-- PATCH_START patch_key={patch_key} applied_at={applied_at} -->\n"
            f"{patch_text}\n"
            f"<!-- PATCH_END patch_key={patch_key} -->\n"
        )
        with FRONTEND_PATCH_FILE.open("a", encoding="utf-8") as fh:
            fh.write(block)

        log.info("Patch '%s' appended to %s", patch_key, FRONTEND_PATCH_FILE)
        return {
            "patch_key":   patch_key,
            "target":      target,
            "applied_at":  applied_at,
            "patch_bytes": len(patch_text),
        }

    # ── Handler: TOPIC_GUARDRAIL ──────────────────────────────────────────

    def _execute_topic_guardrail(self, payload: dict, action: dict) -> dict[str, Any]:
        topics           = payload.get("topics_reported", [])
        suggested_action = payload.get("suggested_action", "")
        topic_specific   = payload.get("topic_specific", False)
        now              = datetime.now(timezone.utc).isoformat()

        existing: dict[str, Any] = {}
        if TOPIC_GUARDRAIL_FILE.exists():
            try:
                existing = json.loads(TOPIC_GUARDRAIL_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Could not parse topic_guardrail.json (%s) — overwriting.", exc)

        history: list[dict] = existing.get("history", [])
        history.append({
            "applied_at":       now,
            "topics_reported":  topics,
            "suggested_action": suggested_action,
        })

        _atomic_write_json(TOPIC_GUARDRAIL_FILE, {
            "updated_at":       now,
            "topic_specific":   topic_specific,
            "topics_reported":  topics,
            "suggested_action": suggested_action,
            "history":          history,
        })

        log.info("Topic guardrail updated. topics=%s", topics)
        return {
            "guardrail_file":  str(TOPIC_GUARDRAIL_FILE),
            "topics_reported": topics,
            "topic_specific":  topic_specific,
            "history_entries": len(history),
        }

    # ── Handler: FLAG_FOR_REVIEW ──────────────────────────────────────────

    def _execute_flag_for_review(self, payload: dict, action: dict) -> dict[str, Any]:
        now      = datetime.now(timezone.utc).isoformat()
        appended = 0

        stats: dict[str, Any] = {}
        if REVIEW_STATS_FILE.exists():
            try:
                stats = json.loads(REVIEW_STATS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        stats.setdefault("total_flagged", 0)
        stats.setdefault("by_reason", {})

        with REVIEW_QUEUE_FILE.open("a", encoding="utf-8") as fh:

            if "flagged_questions" in payload:
                for item in payload.get("flagged_questions", []):
                    fh.write(json.dumps({
                        "review_id":  str(uuid.uuid4()),
                        "flagged_at": now,
                        "action_id":  action.get("action_id"),
                        "plan_id":    action.get("plan_id"),
                        "category":   "repeated_failure",
                        "status":     "OPEN",
                        **item,
                    }) + "\n")
                    appended += 1
                stats["total_flagged"] += appended
                stats["by_reason"]["repeated_failure"] = (
                    stats["by_reason"].get("repeated_failure", 0) + appended
                )

            elif "reason" in payload:
                reason = payload.get("reason", "unknown")
                fh.write(json.dumps({
                    "review_id":     str(uuid.uuid4()),
                    "flagged_at":    now,
                    "action_id":     action.get("action_id"),
                    "plan_id":       action.get("plan_id"),
                    "category":      reason,
                    "status":        "OPEN",
                    "elapsed_hours": payload.get("elapsed_hours"),
                    "reasoning":     action.get("reasoning", []),
                }) + "\n")
                appended += 1
                stats["total_flagged"] += 1
                stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1

        stats["last_updated"] = now
        _atomic_write_json(REVIEW_STATS_FILE, stats)

        log.info("Flagged %d item(s) for human review → %s", appended, REVIEW_QUEUE_FILE)
        return {
            "review_queue_file": str(REVIEW_QUEUE_FILE),
            "items_appended":    appended,
            "total_open":        stats["total_flagged"],
        }

    # ── Handler: NO_ACTION ────────────────────────────────────────────────

    def _execute_no_action(self, payload: dict, action: dict) -> dict[str, Any]:
        log.info("NO_ACTION — all signals within normal thresholds this cycle.")
        return {"note": "No intervention required this cycle."}

    # ── plan_actions.jsonl I/O ────────────────────────────────────────────

    def _load_pending_actions(self) -> list[dict]:
        if not PLAN_ACTIONS_LOG.exists():
            log.warning("plan_actions.jsonl not found: %s", PLAN_ACTIONS_LOG)
            return []

        executed_ids = set(self._state.get("executed_ids", []))
        pending: list[dict] = []
        with PLAN_ACTIONS_LOG.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    action = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    log.warning("plan_actions.jsonl line %d: malformed JSON (%s)", lineno, exc)
                    continue
                action_id = action.get("action_id")
                if not action_id:
                    continue
                if action_id in executed_ids:
                    continue
                if action.get("status") != "PENDING":
                    continue
                if not action.get("auto_approved", False):
                    log.info("Action %s needs manual approval — skipping.", action_id[:8])
                    continue
                pending.append(action)

        log.info("Loaded %d PENDING action(s).", len(pending))
        return pending

    def _update_action_statuses(self, updated_ids: dict[str, str]) -> None:
        if not PLAN_ACTIONS_LOG.exists() or not updated_ids:
            return
        new_lines: list[str] = []
        with PLAN_ACTIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    action = json.loads(stripped)
                    if action.get("action_id") in updated_ids:
                        action["status"]      = updated_ids[action["action_id"]]
                        action["executed_at"] = datetime.now(timezone.utc).isoformat()
                    new_lines.append(json.dumps(action))
                except json.JSONDecodeError:
                    new_lines.append(stripped)
        tmp = PLAN_ACTIONS_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(PLAN_ACTIONS_LOG)
        log.info("Updated statuses for %d action(s) in plan_actions.jsonl.", len(updated_ids))

    # ── Execution log ─────────────────────────────────────────────────────

    def _append_execution_log(self, rec: ExecutionRecord) -> None:
        try:
            with EXECUTION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as exc:
            log.error("Failed to write execution log: %s", exc)

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                return state
            except Exception as exc:
                log.warning("Could not load executor state (%s). Starting fresh.", exc)
        return {"executed_ids": []}

    def _save_state(self) -> None:
        _atomic_write_json(STATE_FILE, self._state)

    # ── Failed-action logging ─────────────────────────────────────────────

    def _log_failed_action(self, action: dict, exc: Exception) -> None:
        try:
            with FAILED_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "action_id":   action.get("action_id"),
                    "action_type": action.get("action_type"),
                    "error":       str(exc),
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except OSError as write_exc:
            log.error("Could not write to executor_failed.jsonl: %s", write_exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Can disable auto-training via env var: SPACELLM_AUTO_TRAIN=false
    auto_train = os.environ.get("SPACELLM_AUTO_TRAIN", "true").lower() in ("true", "1", "yes")

    config   = ExecutorConfig(auto_train=auto_train)
    executor = Executor(config)
    records  = executor.run()

    print(f"\n{'='*60}")
    print(f"  SpaceLLM Executor v3 — Cycle Complete")
    print(f"{'='*60}")
    if not records:
        print("  No actions were executed this cycle.")
    else:
        for rec in records:
            icon = "✓" if rec.status == "EXECUTED" else "✗"
            print(
                f"  {icon} [{rec.action_type:<26}] "
                f"status={rec.status:<8}  "
                f"duration={rec.duration_s:.3f}s  "
                f"action_id={rec.action_id[:8]}"
            )
        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        print(f"\n  Total: {executed} EXECUTED, {failed} FAILED")
        print(f"\n  Execution log      → {EXECUTION_LOG}")
        print(f"  Knowledge repo     → {KNOWLEDGE_REPO_FILE}")
        print(f"  Combined dataset   → {COMBINED_DATASET_FILE}")
        print(f"  Core train file    → {CORE_TRAIN_FILE}")
        if config.auto_train:
            print(f"\n  ⚡ AUTO-TRAINING enabled")
            print(f"     Script    : {CORRECTION_FINE_TUNING_SCRIPT}")
            print(f"     Output dir: {TRAINING_OUTPUT_DIR}")
            last_training = executor._state.get("last_auto_training", {})
            if last_training:
                print(f"     Status    : {last_training.get('status')}")
                if last_training.get("training_log"):
                    print(f"     Log file  : {last_training.get('training_log')}")
                if last_training.get("training_output_dir"):
                    print(f"     Adapter   : {last_training.get('training_output_dir')}")
    print(f"{'='*60}\n")
    sys.exit(0 if not any(r.status == "FAILED" for r in records) else 1)
