"""
SpaceLLM MAPE-K :: Executor Component  (Retrieval-Augmented Continual Learning)
================================================================================
Responsibility: Read Planner output → execute each action → update status
                PENDING → EXECUTED | FAILED → write execution_log.jsonl.

Pipeline position:
    Monitor (monitor_events.jsonl)
        ↓
    Analyser (analysis_report.json)
        ↓
    Planner (plan_actions.jsonl)
        ↓
    Executor  ← YOU ARE HERE
        (execution_log.jsonl, updated plan_actions.jsonl)

Action handlers
---------------
RETRIEVAL_MEMORY_UPDATE   (replaces the old RETRAIN_ADAPTER / LoRA flow)
    Takes the training_examples list from the Planner payload (question +
    human-corrected reference answer), embeds each question, and writes
    {question, correct_answer, embedding} records into a JSON-based
    knowledge repository (KNOWLEDGE_REPO_FILE). No subprocess, no
    fine-tuning run, no LoRA weights to merge/stack.

    WHY: a previous attempt at continual learning via stacked LoRA
    adapters (trained on ~10-15 examples per cycle) caused generation
    collapse (degenerate repeated-token outputs) at inference time —
    a classic small-batch/over-fit LoRA failure mode. Retrieval-based
    memory sidesteps this entirely: corrections are stored verbatim and
    only ever *injected into the prompt* at inference time, so the base
    model's weights (SpaceLLM_v1) are never touched and can never
    collapse. This also means corrections are available immediately,
    with no training latency — useful for a live demo.

    Back-compat: action_type "RETRAIN_ADAPTER" is aliased to the same
    handler, so any actions already queued by an older Planner build
    still execute correctly.

PROMPT_PATCH
    Appends a dated patch block to frontend_patch.md. The controller /
    FastAPI startup code reads this file and injects the active patches into
    the system prompt.

TOPIC_GUARDRAIL
    Writes / merges a JSON entry into mape_k/topic_guardrail.json. The
    FastAPI core reads this on startup to tighten uncertainty language.

FLAG_FOR_REVIEW
    Appends structured records to mape_k/human_review_queue.jsonl and
    increments counters in mape_k/review_stats.json.

NO_ACTION
    Marked EXECUTED immediately.

Inference-time retrieval (for controller.py / FastAPI request handler)
------------------------------------------------------------------------
    from execute import KnowledgeRepository, build_augmented_prompt

    repo = KnowledgeRepository()                 # loads existing JSON repo
    prompt = build_augmented_prompt(user_query, repo, top_k=3)
    # ... send `prompt` to GPT-OSS-20B + SpaceLLM_v1 as usual ...

Design principles
-----------------
- Atomic JSON/JSONL writes via tmp-file + rename.
- Poison-pill guard per action — one failure can't block the rest.
- Persisted state in .executor_state.json across scheduler cycles.
- No subprocess / wall-clock training timeout needed anymore — retrieval
  updates are pure in-process JSON writes and complete in milliseconds.

Author: SpaceLLM Project
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.executor")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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

# Retrieval-based continual learning
KNOWLEDGE_REPO_FILE  = MAPE_DIR / "knowledge_repository.json"
KNOWLEDGE_BACKUP_DIR = MAPE_DIR / "knowledge_repository_backups"

MAPE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    patch_file_max_bytes:    int = 500_000
    retrieval_top_k:         int = 3
    retrieval_min_similarity: float = 0.55
    dedup_similarity_threshold: float = 0.97   # near-identical question → overwrite, not duplicate


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Atomic-write helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
#
# Primary backend: sentence-transformers (good semantic quality, CPU-friendly,
# tiny model). Falls back to a deterministic hashed bag-of-words embedding if
# the package / model can't be loaded (e.g. no network access to download
# weights) — this keeps the system fully functional for a live demo even in
# a constrained environment, at the cost of weaker semantic matching.

_EMBEDDER = None
_EMBEDDER_BACKEND: str | None = None
_HASH_DIM = 384  # matches all-MiniLM-L6-v2 dim, so vectors are interchangeable in size


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
            "bag-of-words embeddings. Install `sentence-transformers` for "
            "better retrieval quality.", exc,
        )
        _EMBEDDER = None
        _EMBEDDER_BACKEND = "hashing_fallback"
    return _EMBEDDER, _EMBEDDER_BACKEND


def _hashing_embed(text: str) -> list[float]:
    """Deterministic fallback embedding — hashed token counts, L2-normalised."""
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
    """Embed a single string. Used both at write-time (storing corrections)
    and at read-time (embedding the incoming user query)."""
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


# ---------------------------------------------------------------------------
# Knowledge repository (JSON-backed retrieval memory)
# ---------------------------------------------------------------------------

class KnowledgeRepository:
    """
    Flat JSON list of correction records:

        [
          {
            "id": "<uuid>",
            "question": "...",
            "correct_answer": "...",
            "embedding": [...],
            "feedback_id": "...",
            "created_at": "...",
            "metadata": {...}
          },
          ...
        ]

    Loaded fully into memory (fine for thousands of corrections; if the
    repo grows into the tens-of-thousands, swap the linear `query()` scan
    for an ANN index such as FAISS/usearch — the on-disk schema doesn't
    need to change).
    """

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
            log.warning(
                "Could not parse %s (%s) — starting with an empty repository.",
                self.path, exc,
            )
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
        """Embed `question` and upsert a correction record. If a
        near-duplicate question already exists (cosine similarity above
        `dedup_threshold`), it is overwritten in place rather than added
        again, so repeated corrections to the same question don't bloat
        the repository or create competing retrieval hits."""
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
        """Embed `question` and return the top-k most similar stored
        corrections (each with an added "similarity" field), filtered by
        `min_similarity`."""
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
    """
    INFERENCE-TIME entry point. Call this from controller.py / the FastAPI
    request handler *before* sending the prompt to GPT-OSS-20B + SpaceLLM_v1.

        repo = KnowledgeRepository()
        prompt = build_augmented_prompt(user_message, repo)
        response = model.generate(prompt)

    If no sufficiently similar correction exists, the original query is
    returned unchanged (no-op) — this keeps normal questions untouched and
    only intervenes where a relevant human correction exists.
    """
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


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config or ExecutorConfig()
        self._state = self._load_state()
        self._repo  = KnowledgeRepository(dedup_threshold=self.config.dedup_similarity_threshold)
        log.info(
            "Executor initialised. executed_action_ids=%d  knowledge_repo_entries=%d",
            len(self._state.get("executed_ids", [])),
            len(self._repo),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[ExecutionRecord]:
        """
        Full execution cycle:
          1. Load PENDING / auto_approved actions from plan_actions.jsonl.
          2. Sort by priority descending (CRITICAL first).
          3. Execute each in isolation — one failure can't poison the batch.
          4. Persist records, update statuses, save state.
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
        self._save_state()

        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        log.info("Cycle complete. %d EXECUTED, %d FAILED.", executed, failed)
        log.info("=" * 60)
        return records

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: dict) -> dict[str, Any]:
        action_type = action.get("action_type")
        payload     = action.get("payload", {})
        handler = {
            "RETRIEVAL_MEMORY_UPDATE": self._execute_retrieval_update,
            "RETRAIN_ADAPTER":         self._execute_retrieval_update,  # back-compat alias
            "PROMPT_PATCH":            self._execute_prompt_patch,
            "TOPIC_GUARDRAIL":         self._execute_topic_guardrail,
            "FLAG_FOR_REVIEW":         self._execute_flag_for_review,
            "NO_ACTION":               self._execute_no_action,
        }.get(action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action_type!r}")
        return handler(payload, action)

    # ------------------------------------------------------------------
    # Handler: RETRIEVAL_MEMORY_UPDATE  (replaces RETRAIN_ADAPTER)
    # ------------------------------------------------------------------

    def _execute_retrieval_update(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Store each {question, corrected answer} pair from the Planner's
        training_examples directly into the JSON knowledge repository.
        No LoRA, no subprocess, no base-model weight changes — the
        correction is retrievable on the very next request.
        """
        examples = payload.get("training_examples", [])
        if not examples:
            raise ValueError("RETRIEVAL_MEMORY_UPDATE payload has no training_examples.")

        # Snapshot before mutating, for audit / rollback.
        backup_path = self._repo._backup()
        log.info("Knowledge repository backed up to %s", backup_path)

        added_entries:      list[dict[str, Any]] = []
        valid_feedback_ids: list[str]             = []

        for ex in examples:
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid       = ex.get("feedback_id", "")

            if not question or not reference:
                log.warning("Skipping example %s — empty question or reference.", fid)
                continue

            entry = self._repo.add(
                question, reference,
                feedback_id=fid,
                metadata={"bertscore": ex.get("bertscore")},
            )
            added_entries.append(entry)
            if fid:
                valid_feedback_ids.append(fid)

        if not added_entries:
            raise ValueError("All training examples were invalid — nothing stored.")

        marked = self._mark_used_in_training(set(valid_feedback_ids))
        log.info(
            "Stored %d correction(s) in knowledge repository (%s). Repo size now %d.",
            len(added_entries), KNOWLEDGE_REPO_FILE, len(self._repo),
        )

        self._state["last_retrieval_update"] = {
            "entries_added": len(added_entries),
            "repo_size":     len(self._repo),
            "backup_path":   str(backup_path),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        return {
            "knowledge_repo_file": str(KNOWLEDGE_REPO_FILE),
            "backup_path":         str(backup_path),
            "entries_added":       len(added_entries),
            "added_ids":           [e["id"] for e in added_entries],
            "repo_size_after":     len(self._repo),
            "marked_in_feedback":  marked,
            "embedding_backend":   _EMBEDDER_BACKEND or "uninitialised",
        }

    def _mark_used_in_training(self, feedback_ids: set[str]) -> int:
        """Atomically rewrite feedback_log.jsonl flipping used_in_training=True."""
        if not FEEDBACK_LOG.exists() or not feedback_ids:
            return 0
        updated   = 0
        new_lines: list[str] = []
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

    # ------------------------------------------------------------------
    # Handler: PROMPT_PATCH
    # ------------------------------------------------------------------

    def _execute_prompt_patch(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Append a versioned patch block to frontend_patch.md.

        Block format:
            <!-- PATCH_START patch_key=<key> applied_at=<iso> -->
            <patch_text>
            <!-- PATCH_END patch_key=<key> -->

        controller.py's get_system_prompt() parses these blocks and layers
        them over the base system prompt. Last block per key wins (dedup).
        """
        patch_key  = payload.get("patch_key", "unknown_patch")
        patch_text = payload.get("patch_text", "").strip()
        target     = payload.get("target", "system_prompt")

        if not patch_text:
            raise ValueError(f"PROMPT_PATCH has empty patch_text for key '{patch_key}'.")

        current_size = FRONTEND_PATCH_FILE.stat().st_size if FRONTEND_PATCH_FILE.exists() else 0
        if current_size > self.config.patch_file_max_bytes:
            log.warning("frontend_patch.md is %.1f KB — consider archiving old patches.",
                        current_size / 1024)

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

    # ------------------------------------------------------------------
    # Handler: TOPIC_GUARDRAIL
    # ------------------------------------------------------------------

    def _execute_topic_guardrail(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Merge guardrail directive into mape_k/topic_guardrail.json.
        FastAPI core reads this on startup to adjust the system prompt.
        """
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

    # ------------------------------------------------------------------
    # Handler: FLAG_FOR_REVIEW
    # ------------------------------------------------------------------

    def _execute_flag_for_review(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Append flagged items to human_review_queue.jsonl.

        Two payload shapes (both emitted by plan.py):
          Shape A — repeated failures:
            { "flagged_questions": [{question_key, negative_count, feedback_id}, ...] }
          Shape B — retrain cooldown:
            { "reason": "retrain_cooldown_active", "elapsed_hours": float }
        """
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
                    "review_id":    str(uuid.uuid4()),
                    "flagged_at":   now,
                    "action_id":    action.get("action_id"),
                    "plan_id":      action.get("plan_id"),
                    "category":     reason,
                    "status":       "OPEN",
                    "elapsed_hours": payload.get("elapsed_hours"),
                    "reasoning":    action.get("reasoning", []),
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

    # ------------------------------------------------------------------
    # Handler: NO_ACTION
    # ------------------------------------------------------------------

    def _execute_no_action(self, payload: dict, action: dict) -> dict[str, Any]:
        log.info("NO_ACTION — all signals within normal thresholds this cycle.")
        return {"note": "No intervention required this cycle."}

    # ------------------------------------------------------------------
    # plan_actions.jsonl I/O
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Execution log
    # ------------------------------------------------------------------

    def _append_execution_log(self, rec: ExecutionRecord) -> None:
        try:
            with EXECUTION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as exc:
            log.error("Failed to write execution log: %s", exc)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Could not load executor state (%s). Starting fresh.", exc)
        return {"executed_ids": []}

    def _save_state(self) -> None:
        _atomic_write_json(STATE_FILE, self._state)

    # ------------------------------------------------------------------
    # Failed-action logging
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor = Executor()
    records  = executor.run()

    print(f"\n{'='*60}")
    print(f"  SpaceLLM Executor — Cycle Complete")
    print(f"{'='*60}")
    if not records:
        print("  No actions were executed this cycle.")
    else:
        for rec in records:
            icon = "✓" if rec.status == "EXECUTED" else "✗"
            print(
                f"  {icon} [{rec.action_type:<24}] "
                f"status={rec.status:<8}  "
                f"duration={rec.duration_s:.3f}s  "
                f"action_id={rec.action_id[:8]}"
            )
        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        print(f"\n  Total: {executed} EXECUTED, {failed} FAILED")
        print(f"\n  Execution log     → {EXECUTION_LOG}")
        print(f"  Knowledge repo    → {KNOWLEDGE_REPO_FILE}")
    print(f"{'='*60}\n")
