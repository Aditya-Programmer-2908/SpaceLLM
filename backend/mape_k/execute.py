"""
SpaceLLM MAPE-K :: Executor Component
=======================================
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
RETRAIN_ADAPTER
    Converts the training_examples list from the Planner payload into the
    messages format expected by correction_fine_tuning.py, writes them to
    correction_train_injection.json (+ a timestamped backup), then launches
    correction_fine_tuning.py via subprocess.

    CONTINUAL LEARNING: On the very first cycle the script continues from
    AdityaPS/SpaceLLM_v1 (the HF-hosted fine-tuned adapter). After each
    successful retrain the local adapter path is stored in
    .executor_state.json["last_adapter"]["adapter_path"] and used as the
    --base_adapter on the next cycle, so every correction run builds on the
    previous one rather than resetting to raw gpt-oss-20b weights.

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

Design principles
-----------------
- Atomic JSON/JSONL writes via tmp-file + rename.
- Poison-pill guard per action — one failure can't block the rest.
- Persisted state in .executor_state.json across scheduler cycles.
- Subprocess stdout/stderr streamed live via threads.
- Hard wall-clock timeout (ExecutorConfig.retrain_timeout_seconds).

Author: SpaceLLM Project
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
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

BASE_DIR             = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR             = BASE_DIR / "mape_k"
FINE_TUNING_DIR      = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/fine_tuning_v2")

PLAN_ACTIONS_LOG     = MAPE_DIR / "plan_actions.jsonl"
EXECUTION_LOG        = MAPE_DIR / "execution_log.jsonl"
FEEDBACK_LOG         = BASE_DIR / "feedback_log.jsonl"
FRONTEND_PATCH_FILE  = BASE_DIR / "frontend_patch.md"
TOPIC_GUARDRAIL_FILE = MAPE_DIR / "topic_guardrail.json"
REVIEW_QUEUE_FILE    = MAPE_DIR / "human_review_queue.jsonl"
REVIEW_STATS_FILE    = MAPE_DIR / "review_stats.json"
STATE_FILE           = MAPE_DIR / ".executor_state.json"
FAILED_LOG           = MAPE_DIR / "executor_failed.jsonl"

# Fine-tuning
FINETUNE_SCRIPT       = FINE_TUNING_DIR / "correction_fine_tuning.py"
FINETUNE_DATASET_DIR  = MAPE_DIR / "retrain_datasets"
CORRECTION_TRAIN_FILE = MAPE_DIR / "correction_train_injection.json"
ADAPTER_OUTPUT_DIR    = FINE_TUNING_DIR / "outputs"

# The HF repo of the original fine-tuned adapter.
# Used as --base_adapter on the FIRST correction cycle so we always build
# on top of SpaceLLM_v1, never on raw gpt-oss-20b weights.
INITIAL_BASE_ADAPTER = "AdityaPS/SpaceLLM_v1"

MAPE_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    retrain_timeout_seconds: int = 7200           # 2-hour hard cap
    python_executable:       str = sys.executable  # same venv as execute.py
    dataset_file_prefix:     str = "retrain_dataset"
    patch_file_max_bytes:    int = 500_000


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
# Executor
# ---------------------------------------------------------------------------

class Executor:

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config or ExecutorConfig()
        self._state = self._load_state()
        log.info(
            "Executor initialised. executed_action_ids=%d  last_adapter=%s",
            len(self._state.get("executed_ids", [])),
            self._state.get("last_adapter", {}).get("adapter_path", "none"),
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
                log.info("  ✓ EXECUTED in %.1fs", duration_s)

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
            "RETRAIN_ADAPTER": self._execute_retrain,
            "PROMPT_PATCH":    self._execute_prompt_patch,
            "TOPIC_GUARDRAIL": self._execute_topic_guardrail,
            "FLAG_FOR_REVIEW": self._execute_flag_for_review,
            "NO_ACTION":       self._execute_no_action,
        }.get(action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action_type!r}")
        return handler(payload, action)

    # ------------------------------------------------------------------
    # Handler: RETRAIN_ADAPTER
    # ------------------------------------------------------------------

    def _execute_retrain(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        CONTINUAL LEARNING flow
        -----------------------
        Cycle 0  (no previous local adapter):
            --base_adapter AdityaPS/SpaceLLM_v1   ← your HF fine-tuned model
        Cycle 1+  (previous local adapter exists):
            --base_adapter <local path saved from last cycle>

        This means every correction run stacks on top of the previous one
        instead of resetting to raw gpt-oss-20b weights each time.
        """
        examples     = payload.get("training_examples", [])
        target_label = payload.get("target_adapter_label", "unknown")
        base_version = payload.get("base_model_version", "unknown")

        if not examples:
            raise ValueError("RETRAIN_ADAPTER payload has no training_examples.")

        # ── Build training dataset in messages format ──────────────────
        ts               = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path      = FINETUNE_DATASET_DIR / f"{self.config.dataset_file_prefix}_{ts}.json"
        valid_ids:       list[str]  = []
        records_for_training: list[dict] = []

        for ex in examples:
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid       = ex.get("feedback_id", "")

            if not question or not reference:
                log.warning("Skipping example %s — empty question or reference.", fid)
                continue

            # correction_fine_tuning.py expects messages format:
            #   [{"role": "user", "content": <question>},
            #    {"role": "assistant", "content": <human_correction>}]
            # Loss is masked so only the assistant turn is trained on.
            records_for_training.append({
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": reference},
                ],
                "feedback_id": fid,
                "bertscore":   ex.get("bertscore"),
            })
            valid_ids.append(fid)

        if not valid_ids:
            raise ValueError("All training examples were invalid — nothing to train on.")

        # Write to the default path correction_fine_tuning.py reads
        CORRECTION_TRAIN_FILE.write_text(
            json.dumps(records_for_training, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Timestamped backup for audit trail
        backup_path.write_text(
            json.dumps(records_for_training, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Dataset written: %s  (%d examples)", CORRECTION_TRAIN_FILE, len(valid_ids))
        log.info("Backup written:  %s", backup_path)

        # ── Resolve base_adapter for continual learning ────────────────
        #
        # Priority order:
        #   1. Local adapter saved from a previous correction cycle
        #   2. AdityaPS/SpaceLLM_v1 (the original HF fine-tuned adapter)
        #
        # We never fall back to no adapter — that would train on raw
        # gpt-oss-20b and throw away all prior fine-tuning.
        last_adapter_path = self._state.get("last_adapter", {}).get("adapter_path")

        if last_adapter_path and Path(last_adapter_path).exists():
            base_adapter = last_adapter_path
            log.info("Continual learning: continuing from local adapter: %s", base_adapter)
        else:
            base_adapter = INITIAL_BASE_ADAPTER
            log.info("Continual learning: no local adapter found — starting from HF adapter: %s",
                     base_adapter)

        # ── Output directory ───────────────────────────────────────────
        adapter_dir = ADAPTER_OUTPUT_DIR / "spacellm_lora_final"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # ── Build subprocess command ───────────────────────────────────
        cmd = [
            self.config.python_executable,
            str(FINETUNE_SCRIPT),
            "--train_file",   str(CORRECTION_TRAIN_FILE),
            "--output_dir",   str(adapter_dir),
            "--base_adapter", base_adapter,   # always set — continual learning
            "--lora_r",       "64",
            "--lora_alpha",   "128",
        ]

        log.info("Launching fine-tuning subprocess ...")
        log.info("  cmd : %s", " ".join(cmd))
        log.info("  timeout: %ds", self.config.retrain_timeout_seconds)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ},   # inherit CUDA_VISIBLE_DEVICES, HF_HOME, HF_TOKEN
            )

            def _drain(stream, store: list[str], label: str):
                for line in stream:
                    line = line.rstrip()
                    store.append(line)
                    log.info("[finetune %s] %s", label, line)

            t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines, "STDOUT"))
            t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, "STDERR"))
            t_out.start()
            t_err.start()

            try:
                returncode = proc.wait(timeout=self.config.retrain_timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                t_out.join(timeout=5)
                t_err.join(timeout=5)
                raise RuntimeError(
                    f"Fine-tuning timed out after {self.config.retrain_timeout_seconds}s."
                )

            t_out.join()
            t_err.join()

        except FileNotFoundError:
            raise RuntimeError(
                f"Fine-tuning script not found: {FINETUNE_SCRIPT}"
            )

        if returncode != 0:
            raise RuntimeError(
                f"Fine-tuning exited with code {returncode}.\n"
                f"Last stderr:\n" + "\n".join(stderr_lines[-30:])
            )

        # ── Mark feedback records as used ──────────────────────────────
        marked = self._mark_used_in_training(set(valid_ids))
        log.info("Marked %d/%d records as used_in_training=True.", marked, len(valid_ids))

        # ── Persist adapter path for next cycle ────────────────────────
        # Next cycle will use this local path as --base_adapter so
        # corrections stack cumulatively: v1 → cycle1 → cycle2 → ...
        self._state["last_adapter"] = {
            "adapter_path":  str(adapter_dir),
            "hf_repo_id":    None,           # local from here on
            "target_label":  target_label,
            "base_used":     base_adapter,
            "example_count": len(valid_ids),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }
        log.info("Saved adapter path for next cycle: %s", adapter_dir)

        return {
            "target_adapter_label": target_label,
            "adapter_path":         str(adapter_dir),
            "base_adapter_used":    base_adapter,
            "train_file":           str(CORRECTION_TRAIN_FILE),
            "backup_path":          str(backup_path),
            "examples_used":        len(valid_ids),
            "returncode":           returncode,
            "stdout_tail":          stdout_lines[-20:],
            "stderr_tail":          stderr_lines[-10:],
            "marked_in_feedback":   marked,
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
                f"  {icon} [{rec.action_type:<20}] "
                f"status={rec.status:<8}  "
                f"duration={rec.duration_s:.1f}s  "
                f"action_id={rec.action_id[:8]}"
            )
        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        print(f"\n  Total: {executed} EXECUTED, {failed} FAILED")
        print(f"\n  Execution log → {EXECUTION_LOG}")
    print(f"{'='*60}\n")
