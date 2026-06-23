"""
SpaceLLM MAPE-K :: Executor Component (Dataset Expansion Strategy)
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

Architecture Philosophy (v2):
-----------------------------
This executor implements the DATASET EXPANSION strategy for continual learning:

1. NO adapter stacking - ever
2. NO retrieval-augmented generation
3. Validated corrections are appended to the master training dataset
4. Periodic offline retraining produces SpaceLLM_v(N+1) which REPLACES v(N)
5. GPT-OSS-20B base model remains unchanged

Lifecycle:
    GPT-OSS-20B + SpaceLLM_v1
        ↓ Collect Feedback
        ↓ Append to Training Dataset
        ↓ Retrain New LM-head Adapter (offline)
    GPT-OSS-20B + SpaceLLM_v2 (replaces v1)

Action handlers
---------------
APPEND_TO_TRAINING_DATASET
    Takes validated corrections from the Planner payload and appends them
    to the master training dataset in the exact SpaceLLM format:
    {sample_id, source_id, mission_name, organization, aspect, difficulty,
     chain_id, source_url, messages: [{role, content}, ...]}

PROMPT_PATCH
    Appends a dated patch block to frontend_patch.md.

TOPIC_GUARDRAIL
    Writes / merges a JSON entry into mape_k/topic_guardrail.json.

FLAG_FOR_REVIEW
    Appends structured records to mape_k/human_review_queue.jsonl.

SCHEDULE_RETRAINING
    Marks dataset as ready for offline retraining, writes retraining manifest.

NO_ACTION
    Marked EXECUTED immediately.

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

BASE_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR = BASE_DIR / "mape_k"
DATA_DIR = BASE_DIR.parent / "Model_training_&_Data_Extraction" / "data_processing"

PLAN_ACTIONS_LOG = MAPE_DIR / "plan_actions.jsonl"
EXECUTION_LOG = MAPE_DIR / "execution_log.jsonl"
FEEDBACK_LOG = BASE_DIR / "feedback_log.jsonl"
FRONTEND_PATCH_FILE = BASE_DIR / "frontend_patch.md"
TOPIC_GUARDRAIL_FILE = MAPE_DIR / "topic_guardrail.json"
REVIEW_QUEUE_FILE = MAPE_DIR / "human_review_queue.jsonl"
REVIEW_STATS_FILE = MAPE_DIR / "review_stats.json"
STATE_FILE = MAPE_DIR / ".executor_state.json"
FAILED_LOG = MAPE_DIR / "executor_failed.jsonl"

# Dataset expansion paths
MASTER_TRAINING_DATASET = DATA_DIR / "DatasetA_core_QA_v2" / "train.json"
CORRECTIONS_DATASET = MAPE_DIR / "corrections_dataset.json"
CORRECTIONS_STAGING = MAPE_DIR / "corrections_staging.jsonl"
RETRAINING_MANIFEST = MAPE_DIR / "retraining_manifest.json"
DATASET_BACKUP_DIR = MAPE_DIR / "dataset_backups"

MAPE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# System prompt for SpaceLLM
SPACELLM_SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "spacecraft technology, and scientific discoveries. Answer questions clearly "
    "and accurately based on your knowledge of space exploration. Tailor the depth "
    "of your response to the complexity of the question."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    patch_file_max_bytes: int = 500_000
    min_corrections_for_retrain: int = 50  # Minimum corrections before suggesting retrain
    auto_schedule_retrain_threshold: int = 200  # Auto-schedule at this many corrections


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    execution_id: str
    action_id: str
    action_type: str
    plan_id: str
    status: str  # "EXECUTED" | "FAILED"
    started_at: str
    finished_at: str
    duration_s: float
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingExample:
    """Represents a single training example in SpaceLLM format."""
    sample_id: str
    source_id: str
    mission_name: str
    organization: str
    aspect: str
    difficulty: str
    chain_id: str
    source_url: str
    messages: list[dict[str, str]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_correction(
        cls,
        question: str,
        correct_answer: str,
        feedback_id: str,
        metadata: dict[str, Any] | None = None
    ) -> "TrainingExample":
        """
        Create a TrainingExample from a user correction/feedback.
        
        Maps feedback fields to the SpaceLLM training format.
        """
        metadata = metadata or {}
        
        # Generate deterministic sample_id from feedback_id
        sample_id = f"CORR_{feedback_id[:12].upper()}" if feedback_id else f"CORR_{uuid.uuid4().hex[:12].upper()}"
        
        # Extract metadata or use defaults
        mission_name = metadata.get("mission_name", "User Correction")
        organization = metadata.get("organization", "MAPE-K Feedback")
        topic = metadata.get("topic", "general")
        
        # Determine aspect from question type or metadata
        aspect = cls._infer_aspect(question, metadata)
        
        # Determine difficulty from answer complexity
        difficulty = cls._infer_difficulty(correct_answer, metadata)
        
        # Build source_id and chain_id
        source_id = f"correction_{topic}_{aspect}".lower().replace(" ", "_")
        chain_id = f"{source_id}_{sample_id[-6:]}"
        
        # Source URL - could be feedback portal or original query context
        source_url = metadata.get("source_url", "mape-k://feedback/corrections")
        
        messages = [
            {
                "role": "developer",
                "content": SPACELLM_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question.strip()
            },
            {
                "role": "assistant",
                "content": correct_answer.strip()
            }
        ]
        
        return cls(
            sample_id=sample_id,
            source_id=source_id,
            mission_name=mission_name,
            organization=organization,
            aspect=aspect,
            difficulty=difficulty,
            chain_id=chain_id,
            source_url=source_url,
            messages=messages
        )
    
    @staticmethod
    def _infer_aspect(question: str, metadata: dict) -> str:
        """Infer the aspect (OBJECTIVE, TECHNICAL, DISCOVERY, etc.) from question."""
        if "aspect" in metadata:
            return metadata["aspect"].upper()
        
        q_lower = question.lower()
        
        if any(kw in q_lower for kw in ["objective", "goal", "purpose", "aim", "mission"]):
            return "OBJECTIVE"
        elif any(kw in q_lower for kw in ["how", "technical", "mechanism", "process", "work"]):
            return "TECHNICAL"
        elif any(kw in q_lower for kw in ["discover", "found", "observed", "detected"]):
            return "DISCOVERY"
        elif any(kw in q_lower for kw in ["when", "date", "timeline", "schedule"]):
            return "TIMELINE"
        elif any(kw in q_lower for kw in ["who", "team", "scientist", "astronaut"]):
            return "PERSONNEL"
        elif any(kw in q_lower for kw in ["where", "location", "orbit", "destination"]):
            return "LOCATION"
        else:
            return "GENERAL"
    
    @staticmethod
    def _infer_difficulty(answer: str, metadata: dict) -> str:
        """Infer difficulty from answer complexity."""
        if "difficulty" in metadata:
            return metadata["difficulty"].lower()
        
        word_count = len(answer.split())
        
        if word_count < 30:
            return "basic"
        elif word_count < 80:
            return "intermediate"
        else:
            return "advanced"


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


def _load_json_safe(path: Path, default: Any = None) -> Any:
    """Load JSON file safely, returning default on failure."""
    if not path.exists():
        return default if default is not None else []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse %s (%s) — using default.", path, exc)
        return default if default is not None else []


# ---------------------------------------------------------------------------
# Corrections Dataset Manager
# ---------------------------------------------------------------------------

class CorrectionsDatasetManager:
    """
    Manages the corrections dataset that will be merged with the master
    training dataset for periodic retraining.
    
    Corrections are stored in two places:
    1. corrections_staging.jsonl - Raw corrections as they come in
    2. corrections_dataset.json - Validated, formatted corrections ready for training
    """
    
    def __init__(self, config: ExecutorConfig | None = None):
        self.config = config or ExecutorConfig()
        self._corrections: list[dict] = self._load_corrections()
        self._sample_ids: set[str] = {c["sample_id"] for c in self._corrections}
    
    def _load_corrections(self) -> list[dict]:
        """Load existing corrections dataset."""
        return _load_json_safe(CORRECTIONS_DATASET, [])
    
    def _save_corrections(self) -> None:
        """Save corrections dataset atomically."""
        _atomic_write_json(CORRECTIONS_DATASET, self._corrections)
    
    def _backup(self) -> Path:
        """Create timestamped backup of corrections dataset."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = DATASET_BACKUP_DIR / f"corrections_dataset_{ts}.json"
        backup_path.write_text(
            json.dumps(self._corrections, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return backup_path
    
    def add_correction(
        self,
        question: str,
        correct_answer: str,
        feedback_id: str,
        metadata: dict[str, Any] | None = None
    ) -> TrainingExample:
        """
        Add a validated correction to the dataset.
        
        Deduplicates by checking for near-identical questions.
        """
        # Create training example in SpaceLLM format
        example = TrainingExample.from_correction(
            question=question,
            correct_answer=correct_answer,
            feedback_id=feedback_id,
            metadata=metadata
        )
        
        # Check for duplicate sample_id (shouldn't happen but safety check)
        if example.sample_id in self._sample_ids:
            # Regenerate with new UUID suffix
            example.sample_id = f"CORR_{uuid.uuid4().hex[:12].upper()}"
        
        # Check for near-duplicate questions (simple text matching)
        normalized_q = self._normalize_text(question)
        for existing in self._corrections:
            existing_q = self._normalize_text(
                existing.get("messages", [{}])[-2].get("content", "")
                if len(existing.get("messages", [])) >= 2 else ""
            )
            if self._text_similarity(normalized_q, existing_q) > 0.95:
                # Update existing instead of adding duplicate
                log.info("Updating existing correction (similar question found)")
                existing["messages"][-1]["content"] = correct_answer.strip()
                existing["_updated_at"] = datetime.now(timezone.utc).isoformat()
                existing["_feedback_id"] = feedback_id
                self._save_corrections()
                return TrainingExample(**{k: v for k, v in existing.items() if not k.startswith("_")})
        
        # Add new correction
        example_dict = example.to_dict()
        example_dict["_created_at"] = datetime.now(timezone.utc).isoformat()
        example_dict["_feedback_id"] = feedback_id
        example_dict["_metadata"] = metadata or {}
        
        self._corrections.append(example_dict)
        self._sample_ids.add(example.sample_id)
        self._save_corrections()
        
        # Also append to staging log for audit trail
        self._append_staging(example_dict)
        
        return example
    
    def _append_staging(self, example_dict: dict) -> None:
        """Append to staging JSONL for audit trail."""
        try:
            with CORRECTIONS_STAGING.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(example_dict) + "\n")
        except OSError as exc:
            log.warning("Could not append to staging: %s", exc)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity for deduplication."""
        if not a or not b:
            return 0.0
        words_a = set(a.split())
        words_b = set(b.split())
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the corrections dataset."""
        return {
            "total_corrections": len(self._corrections),
            "ready_for_retrain": len(self._corrections) >= self.config.min_corrections_for_retrain,
            "auto_retrain_threshold": self.config.auto_schedule_retrain_threshold,
            "should_auto_schedule": len(self._corrections) >= self.config.auto_schedule_retrain_threshold,
        }
    
    def export_for_training(self) -> list[dict]:
        """
        Export corrections in clean SpaceLLM format for training.
        
        Strips internal metadata fields (prefixed with _).
        """
        clean = []
        for c in self._corrections:
            clean.append({k: v for k, v in c.items() if not k.startswith("_")})
        return clean
    
    def merge_with_master_dataset(self) -> tuple[Path, int, int]:
        """
        Merge corrections with master training dataset.
        
        Returns (merged_path, original_count, corrections_count)
        """
        # Load master dataset
        master = _load_json_safe(MASTER_TRAINING_DATASET, [])
        original_count = len(master)
        
        # Get clean corrections
        corrections = self.export_for_training()
        corrections_count = len(corrections)
        
        # Create merged dataset
        merged = master + corrections
        
        # Write to new file (don't overwrite master)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        merged_path = MAPE_DIR / f"merged_training_dataset_{ts}.json"
        _atomic_write_json(merged_path, merged)
        
        log.info(
            "Merged dataset created: %d original + %d corrections = %d total",
            original_count, corrections_count, len(merged)
        )
        
        return merged_path, original_count, corrections_count
    
    def __len__(self) -> int:
        return len(self._corrections)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    
    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config or ExecutorConfig()
        self._state = self._load_state()
        self._corrections_mgr = CorrectionsDatasetManager(self.config)
        log.info(
            "Executor initialised. executed_action_ids=%d  corrections_count=%d",
            len(self._state.get("executed_ids", [])),
            len(self._corrections_mgr),
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
        
        records: list[ExecutionRecord] = []
        updated_ids: dict[str, str] = {}
        
        for action in pending:
            action_id = action.get("action_id", str(uuid.uuid4()))
            action_type = action.get("action_type", "UNKNOWN")
            plan_id = action.get("plan_id", "unknown")
            started_at = datetime.now(timezone.utc).isoformat()
            
            log.info(
                "→ [%s] action_id=%s  priority=%s",
                action_type, action_id[:8], action.get("priority")
            )
            
            try:
                result = self._dispatch(action)
                finished_at = datetime.now(timezone.utc).isoformat()
                duration_s = (
                    datetime.fromisoformat(finished_at) -
                    datetime.fromisoformat(started_at)
                ).total_seconds()
                rec = ExecutionRecord(
                    execution_id=str(uuid.uuid4()),
                    action_id=action_id,
                    action_type=action_type,
                    plan_id=plan_id,
                    status="EXECUTED",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_s=round(duration_s, 3),
                    result=result,
                )
                updated_ids[action_id] = "EXECUTED"
                log.info("  ✓ EXECUTED in %.3fs", duration_s)
            
            except Exception as exc:
                finished_at = datetime.now(timezone.utc).isoformat()
                duration_s = (
                    datetime.fromisoformat(finished_at) -
                    datetime.fromisoformat(started_at)
                ).total_seconds()
                log.error(
                    "  ✗ FAILED action %s (%s): %s",
                    action_id[:8], action_type, exc, exc_info=True
                )
                self._log_failed_action(action, exc)
                rec = ExecutionRecord(
                    execution_id=str(uuid.uuid4()),
                    action_id=action_id,
                    action_type=action_type,
                    plan_id=plan_id,
                    status="FAILED",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_s=round(duration_s, 3),
                    error=str(exc),
                )
                updated_ids[action_id] = "FAILED"
            
            self._append_execution_log(rec)
            records.append(rec)
            self._state.setdefault("executed_ids", []).append(action_id)
        
        self._update_action_statuses(updated_ids)
        self._save_state()
        
        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed = sum(1 for r in records if r.status == "FAILED")
        log.info("Cycle complete. %d EXECUTED, %d FAILED.", executed, failed)
        log.info("=" * 60)
        return records
    
    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    
    def _dispatch(self, action: dict) -> dict[str, Any]:
        action_type = action.get("action_type")
        payload = action.get("payload", {})
        handler = {
            "APPEND_TO_TRAINING_DATASET": self._execute_append_to_dataset,
            # Legacy aliases for backward compatibility
            "RETRIEVAL_MEMORY_UPDATE": self._execute_append_to_dataset,
            "RETRAIN_ADAPTER": self._execute_append_to_dataset,
            # Other handlers
            "PROMPT_PATCH": self._execute_prompt_patch,
            "TOPIC_GUARDRAIL": self._execute_topic_guardrail,
            "FLAG_FOR_REVIEW": self._execute_flag_for_review,
            "SCHEDULE_RETRAINING": self._execute_schedule_retraining,
            "NO_ACTION": self._execute_no_action,
        }.get(action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action_type!r}")
        return handler(payload, action)
    
    # ------------------------------------------------------------------
    # Handler: APPEND_TO_TRAINING_DATASET
    # ------------------------------------------------------------------
    
    def _execute_append_to_dataset(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        """
        Append validated corrections to the training dataset.
        
        This is the core of the dataset expansion strategy:
        - Each correction becomes a properly formatted training example
        - Examples are stored and will be merged with master dataset
        - Periodic offline retraining produces new adapter version
        """
        examples = payload.get("training_examples", [])
        if not examples:
            raise ValueError("APPEND_TO_TRAINING_DATASET payload has no training_examples.")
        
        # Backup before mutating
        backup_path = self._corrections_mgr._backup()
        log.info("Corrections dataset backed up to %s", backup_path)
        
        added_examples: list[TrainingExample] = []
        valid_feedback_ids: list[str] = []
        
        for ex in examples:
            question = (ex.get("question") or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid = ex.get("feedback_id", "")
            
            if not question or not reference:
                log.warning("Skipping example %s — empty question or reference.", fid)
                continue
            
            # Extract metadata for proper categorization
            metadata = {
                "bertscore": ex.get("bertscore"),
                "topic": ex.get("topic", "general"),
                "mission_name": ex.get("mission_name"),
                "organization": ex.get("organization"),
                "aspect": ex.get("aspect"),
                "difficulty": ex.get("difficulty"),
                "source_url": ex.get("source_url"),
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            example = self._corrections_mgr.add_correction(
                question=question,
                correct_answer=reference,
                feedback_id=fid,
                metadata=metadata
            )
            added_examples.append(example)
            if fid:
                valid_feedback_ids.append(fid)
        
        if not added_examples:
            raise ValueError("All training examples were invalid — nothing stored.")
        
        # Mark used in feedback log
        marked = self._mark_used_in_training(set(valid_feedback_ids))
        
        # Get stats
        stats = self._corrections_mgr.get_stats()
        
        log.info(
            "Added %d correction(s) to training dataset. Total corrections: %d",
            len(added_examples), stats["total_corrections"]
        )
        
        # Check if we should auto-schedule retraining
        auto_scheduled = False
        if stats["should_auto_schedule"]:
            log.info("Auto-scheduling retraining: %d corrections accumulated", stats["total_corrections"])
            self._create_retraining_manifest(trigger="auto_threshold")
            auto_scheduled = True
        
        self._state["last_dataset_update"] = {
            "examples_added": len(added_examples),
            "total_corrections": stats["total_corrections"],
            "backup_path": str(backup_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        return {
            "corrections_dataset_file": str(CORRECTIONS_DATASET),
            "backup_path": str(backup_path),
            "examples_added": len(added_examples),
            "sample_ids": [e.sample_id for e in added_examples],
            "total_corrections": stats["total_corrections"],
            "marked_in_feedback": marked,
            "ready_for_retrain": stats["ready_for_retrain"],
            "auto_scheduled_retrain": auto_scheduled,
        }
    
    def _mark_used_in_training(self, feedback_ids: set[str]) -> int:
        """Atomically rewrite feedback_log.jsonl flipping used_in_training=True."""
        if not FEEDBACK_LOG.exists() or not feedback_ids:
            return 0
        updated = 0
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
                        record["added_to_dataset_at"] = datetime.now(timezone.utc).isoformat()
                        updated += 1
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(stripped)
        tmp = FEEDBACK_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(FEEDBACK_LOG)
        return updated
    
    # ------------------------------------------------------------------
    # Handler: SCHEDULE_RETRAINING
    # ------------------------------------------------------------------
    
    def _execute_schedule_retraining(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        """
        Schedule offline retraining by creating a manifest.
        
        The actual training is done offline by train_spacellm_fresh.py
        using the merged dataset.
        """
        trigger = payload.get("trigger", "manual")
        
        merged_path, original_count, corrections_count = \
            self._corrections_mgr.merge_with_master_dataset()
        
        manifest = self._create_retraining_manifest(
            trigger=trigger,
            merged_dataset_path=str(merged_path),
            original_count=original_count,
            corrections_count=corrections_count
        )
        
        log.info("Retraining scheduled. Manifest: %s", RETRAINING_MANIFEST)
        
        return {
            "manifest_file": str(RETRAINING_MANIFEST),
            "merged_dataset": str(merged_path),
            "original_examples": original_count,
            "correction_examples": corrections_count,
            "total_examples": original_count + corrections_count,
            "trigger": trigger,
        }
    
    def _create_retraining_manifest(
        self,
        trigger: str = "manual",
        merged_dataset_path: str | None = None,
        original_count: int | None = None,
        corrections_count: int | None = None
    ) -> dict:
        """Create retraining manifest for offline training script."""
        stats = self._corrections_mgr.get_stats()
        
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "status": "PENDING",
            "corrections_count": stats["total_corrections"],
            "corrections_dataset": str(CORRECTIONS_DATASET),
            "master_dataset": str(MASTER_TRAINING_DATASET),
            "merged_dataset": merged_dataset_path,
            "original_count": original_count,
            "training_command": (
                f"python train_spacellm_fresh.py "
                f"--train_file {merged_dataset_path or 'MERGED_DATASET_PATH'} "
                f"--output_dir ./spacellm_v{{N+1}} "
                f"--epochs 3 --lr 2e-4 --lora_r 64"
            ),
            "notes": [
                "After training completes:",
                "1. Validate new adapter on held-out test set",
                "2. Replace SpaceLLM_v{N} with SpaceLLM_v{N+1}",
                "3. Update RETRAINING_MANIFEST status to COMPLETED",
                "4. Archive corrections_dataset.json",
            ]
        }
        
        _atomic_write_json(RETRAINING_MANIFEST, manifest)
        return manifest
    
    # ------------------------------------------------------------------
    # Handler: PROMPT_PATCH
    # ------------------------------------------------------------------
    
    def _execute_prompt_patch(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        """Append a versioned patch block to frontend_patch.md."""
        patch_key = payload.get("patch_key", "unknown_patch")
        patch_text = payload.get("patch_text", "").strip()
        target = payload.get("target", "system_prompt")
        
        if not patch_text:
            raise ValueError(f"PROMPT_PATCH has empty patch_text for key '{patch_key}'.")
        
        current_size = FRONTEND_PATCH_FILE.stat().st_size if FRONTEND_PATCH_FILE.exists() else 0
        if current_size > self.config.patch_file_max_bytes:
            log.warning(
                "frontend_patch.md is %.1f KB — consider archiving old patches.",
                current_size / 1024
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
            "patch_key": patch_key,
            "target": target,
            "applied_at": applied_at,
            "patch_bytes": len(patch_text),
        }
    
    # ------------------------------------------------------------------
    # Handler: TOPIC_GUARDRAIL
    # ------------------------------------------------------------------
    
    def _execute_topic_guardrail(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        """Merge guardrail directive into mape_k/topic_guardrail.json."""
        topics = payload.get("topics_reported", [])
        suggested_action = payload.get("suggested_action", "")
        topic_specific = payload.get("topic_specific", False)
        now = datetime.now(timezone.utc).isoformat()
        
        existing: dict[str, Any] = _load_json_safe(TOPIC_GUARDRAIL_FILE, {})
        
        history: list[dict] = existing.get("history", [])
        history.append({
            "applied_at": now,
            "topics_reported": topics,
            "suggested_action": suggested_action,
        })
        
        _atomic_write_json(TOPIC_GUARDRAIL_FILE, {
            "updated_at": now,
            "topic_specific": topic_specific,
            "topics_reported": topics,
            "suggested_action": suggested_action,
            "history": history,
        })
        
        log.info("Topic guardrail updated. topics=%s", topics)
        return {
            "guardrail_file": str(TOPIC_GUARDRAIL_FILE),
            "topics_reported": topics,
            "topic_specific": topic_specific,
            "history_entries": len(history),
        }
    
    # ------------------------------------------------------------------
    # Handler: FLAG_FOR_REVIEW
    # ------------------------------------------------------------------
    
    def _execute_flag_for_review(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
        """Append flagged items to human_review_queue.jsonl."""
        now = datetime.now(timezone.utc).isoformat()
        appended = 0
        
        stats: dict[str, Any] = _load_json_safe(REVIEW_STATS_FILE, {})
        stats.setdefault("total_flagged", 0)
        stats.setdefault("by_reason", {})
        
        with REVIEW_QUEUE_FILE.open("a", encoding="utf-8") as fh:
            if "flagged_questions" in payload:
                for item in payload.get("flagged_questions", []):
                    fh.write(json.dumps({
                        "review_id": str(uuid.uuid4()),
                        "flagged_at": now,
                        "action_id": action.get("action_id"),
                        "plan_id": action.get("plan_id"),
                        "category": "repeated_failure",
                        "status": "OPEN",
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
                    "review_id": str(uuid.uuid4()),
                    "flagged_at": now,
                    "action_id": action.get("action_id"),
                    "plan_id": action.get("plan_id"),
                    "category": reason,
                    "status": "OPEN",
                    "elapsed_hours": payload.get("elapsed_hours"),
                    "reasoning": action.get("reasoning", []),
                }) + "\n")
                appended += 1
                stats["total_flagged"] += 1
                stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        
        stats["last_updated"] = now
        _atomic_write_json(REVIEW_STATS_FILE, stats)
        
        log.info("Flagged %d item(s) for human review → %s", appended, REVIEW_QUEUE_FILE)
        return {
            "review_queue_file": str(REVIEW_QUEUE_FILE),
            "items_appended": appended,
            "total_open": stats["total_flagged"],
        }
    
    # ------------------------------------------------------------------
    # Handler: NO_ACTION
    # ------------------------------------------------------------------
    
    def _execute_no_action(
        self, payload: dict, action: dict
    ) -> dict[str, Any]:
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
                        action["status"] = updated_ids[action["action_id"]]
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
        return _load_json_safe(STATE_FILE, {"executed_ids": []})
    
    def _save_state(self) -> None:
        _atomic_write_json(STATE_FILE, self._state)
    
    # ------------------------------------------------------------------
    # Failed-action logging
    # ------------------------------------------------------------------
    
    def _log_failed_action(self, action: dict, exc: Exception) -> None:
        try:
            with FAILED_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "action_id": action.get("action_id"),
                    "action_type": action.get("action_type"),
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except OSError as write_exc:
            log.error("Could not write to executor_failed.jsonl: %s", write_exc)


# ---------------------------------------------------------------------------
# Utility functions for external use
# ---------------------------------------------------------------------------

def get_corrections_stats() -> dict[str, Any]:
    """Get current corrections dataset statistics."""
    mgr = CorrectionsDatasetManager()
    return mgr.get_stats()


def export_corrections_for_training() -> list[dict]:
    """Export all corrections in clean training format."""
    mgr = CorrectionsDatasetManager()
    return mgr.export_for_training()


def create_merged_dataset() -> tuple[Path, int, int]:
    """Create merged dataset for training."""
    mgr = CorrectionsDatasetManager()
    return mgr.merge_with_master_dataset()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SpaceLLM MAPE-K Executor")
    parser.add_argument(
        "--stats", action="store_true",
        help="Show corrections dataset statistics and exit"
    )
    parser.add_argument(
        "--export", type=str, metavar="PATH",
        help="Export corrections to specified JSON file and exit"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Create merged training dataset and exit"
    )
    args = parser.parse_args()
    
    if args.stats:
        stats = get_corrections_stats()
        print(json.dumps(stats, indent=2))
    elif args.export:
        corrections = export_corrections_for_training()
        Path(args.export).write_text(
            json.dumps(corrections, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Exported {len(corrections)} corrections to {args.export}")
    elif args.merge:
        merged_path, orig, corr = create_merged_dataset()
        print(f"Merged dataset created: {merged_path}")
        print(f"  Original examples: {orig}")
        print(f"  Corrections: {corr}")
        print(f"  Total: {orig + corr}")
    else:
        executor = Executor()
        records = executor.run()
        
        print(f"\n{'=' * 60}")
        print("  SpaceLLM Executor — Cycle Complete")
        print(f"{'=' * 60}")
        if not records:
            print("  No actions were executed this cycle.")
        else:
            for rec in records:
                icon = "✓" if rec.status == "EXECUTED" else "✗"
                print(
                    f"  {icon} [{rec.action_type:<28}] "
                    f"status={rec.status:<8}  "
                    f"duration={rec.duration_s:.3f}s  "
                    f"action_id={rec.action_id[:8]}"
                )
            executed = sum(1 for r in records if r.status == "EXECUTED")
            failed = sum(1 for r in records if r.status == "FAILED")
            print(f"\n  Total: {executed} EXECUTED, {failed} FAILED")
        
        stats = get_corrections_stats()
        print(f"\n  Corrections Dataset Stats:")
        print(f"    Total corrections: {stats['total_corrections']}")
        print(f"    Ready for retrain: {stats['ready_for_retrain']}")
        print(f"\n  Output Files:")
        print(f"    Execution log     → {EXECUTION_LOG}")
        print(f"    Corrections       → {CORRECTIONS_DATASET}")
        print(f"{'=' * 60}\n")
