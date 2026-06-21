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
    JSONL format expected by correction_fine_tuning.py / lora_finetuning_v2.py
    (instruction-tuning pairs with the human correction as the target), writes
    them to a timestamped dataset file, then launches the fine-tuning script
    via subprocess.  The process is captured line-by-line so progress is
    visible in execution_log.jsonl and a hard wall-clock timeout
    (ExecutorConfig.retrain_timeout_seconds) prevents runaway jobs.
    On success the new adapter path is recorded and feedback_log.jsonl is
    updated to flip used_in_training=True for every example that went into
    the run.

PROMPT_PATCH
    Appends a dated patch block to frontend_patch.md.  The controller /
    FastAPI startup code is expected to read this file and inject the active
    patches into its system prompt — the Executor never touches running
    inference processes directly.  Each patch block is tagged with the
    patch_key so duplicates can be detected and de-duped by the reader.

TOPIC_GUARDRAIL
    Writes (or merges) a JSON entry into mape_k/topic_guardrail.json.  The
    FastAPI core should read this file on startup (and optionally poll it)
    to decide whether to tighten uncertainty language or add a topic caveat
    to the system prompt.

FLAG_FOR_REVIEW
    Appends structured records to mape_k/human_review_queue.jsonl and
    increments a per-category counter in mape_k/review_stats.json so a
    dashboard can surface the backlog without reading every line.

NO_ACTION
    Marked EXECUTED immediately; logged at INFO level.

Design principles (matching the rest of the MAPE-K codebase)
-------------------------------------------------------------
- Atomic JSON/JSONL writes via tmp-file + rename — a killed process never
  leaves a half-written file.
- Poison-pill guard per action — one bad action can't block the rest of the
  batch; failures are logged to executor_failed.jsonl and the action is
  marked FAILED in plan_actions.jsonl.
- Persisted, version-aware state in .executor_state.json so the Executor
  knows which action_ids it has already processed across scheduler cycles.
- The retrain subprocess is non-blocking from the MAPE loop's perspective
  only in the sense that the Executor waits for it (within the timeout) and
  records the outcome — this keeps the MAPE loop honest about whether the
  retrain actually succeeded before it can be counted in `last_retrain_at`.

Author: SpaceLLM Project
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
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
# Paths  (mirror the layout used by monitor / analyser / planner)
# ---------------------------------------------------------------------------

BASE_DIR              = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR              = BASE_DIR / "mape_k"
FINE_TUNING_DIR       = BASE_DIR.parent / "fine_tuning_v2"  # ../fine_tuning_v2

PLAN_ACTIONS_LOG      = MAPE_DIR / "plan_actions.jsonl"
EXECUTION_LOG         = MAPE_DIR / "execution_log.jsonl"
FEEDBACK_LOG          = BASE_DIR / "feedback_log.jsonl"
FRONTEND_PATCH_FILE   = BASE_DIR / "frontend_patch.md"
TOPIC_GUARDRAIL_FILE  = MAPE_DIR / "topic_guardrail.json"
REVIEW_QUEUE_FILE     = MAPE_DIR / "human_review_queue.jsonl"
REVIEW_STATS_FILE     = MAPE_DIR / "review_stats.json"
STATE_FILE            = MAPE_DIR / ".executor_state.json"
FAILED_LOG            = MAPE_DIR / "executor_failed.jsonl"

# Fine-tuning script & dataset output directory
FINETUNE_SCRIPT         = FINE_TUNING_DIR / "correction_fine_tuning.py"
FINETUNE_DATASET_DIR    = MAPE_DIR / "retrain_datasets"
CORRECTION_TRAIN_FILE   = MAPE_DIR / "correction_train_injection.json"   # hardcoded default in the script
ADAPTER_OUTPUT_DIR      = FINE_TUNING_DIR / "outputs"

MAPE_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    # Retrain subprocess
    retrain_timeout_seconds: int  = 7200          # 2 hours hard cap
    python_executable:       str  = sys.executable # same venv that runs execute.py

    # Dataset naming
    dataset_file_prefix: str = "retrain_dataset"

    # Prompt patch
    patch_file_max_bytes: int = 500_000            # guard against unbounded growth


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    execution_id:  str
    action_id:     str
    action_type:   str
    plan_id:       str
    status:        str            # "EXECUTED" | "FAILED" | "SKIPPED"
    started_at:    str
    finished_at:   str
    duration_s:    float
    result:        dict[str, Any] = field(default_factory=dict)
    error:         str | None     = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Atomic-write helper (matches the pattern in monitor / analyser / planner)
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
            "Executor initialised. executed_action_ids=%d",
            len(self._state.get("executed_ids", [])),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[ExecutionRecord]:
        """
        Full execution cycle:
          1. Load all PENDING actions from plan_actions.jsonl (unseen only).
          2. Sort by priority descending so CRITICAL actions go first.
          3. Execute each in isolation — one failure can't poison the batch.
          4. Persist execution records and update action statuses.
        Returns all ExecutionRecords produced this cycle.
        """
        log.info("=" * 60)
        log.info("Executor cycle starting.")

        pending = self._load_pending_actions()
        if not pending:
            log.info("No new PENDING actions to execute.")
            return []

        # Highest priority first
        pending.sort(key=lambda a: a.get("priority", 0), reverse=True)
        log.info("Executing %d pending action(s).", len(pending))

        records: list[ExecutionRecord] = []
        updated_ids: dict[str, str] = {}   # action_id -> new status

        for action in pending:
            action_id   = action.get("action_id", str(uuid.uuid4()))
            action_type = action.get("action_type", "UNKNOWN")
            plan_id     = action.get("plan_id", "unknown")

            started_at = datetime.now(timezone.utc).isoformat()
            log.info("→ [%s] action_id=%s  priority=%s",
                     action_type, action_id[:8], action.get("priority"))

            try:
                result = self._dispatch(action)
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

                log.error(
                    "  ✗ FAILED action %s (%s): %s",
                    action_id[:8], action_type, exc, exc_info=True,
                )
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

        # Rewrite plan_actions.jsonl with updated statuses
        self._update_action_statuses(updated_ids)
        self._save_state()

        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        log.info("Cycle complete. %d EXECUTED, %d FAILED.", executed, failed)
        log.info("=" * 60)
        return records

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: dict) -> dict[str, Any]:
        action_type = action.get("action_type")
        payload     = action.get("payload", {})

        dispatch_map = {
            "RETRAIN_ADAPTER": self._execute_retrain,
            "PROMPT_PATCH":    self._execute_prompt_patch,
            "TOPIC_GUARDRAIL": self._execute_topic_guardrail,
            "FLAG_FOR_REVIEW": self._execute_flag_for_review,
            "NO_ACTION":       self._execute_no_action,
        }

        handler = dispatch_map.get(action_type)
        if handler is None:
            raise ValueError(f"Unknown action_type: {action_type!r}")

        return handler(payload, action)

    # ------------------------------------------------------------------
    # Handler: RETRAIN_ADAPTER
    # ------------------------------------------------------------------

    def _execute_retrain(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        1. Validate and write training examples to a timestamped JSONL dataset.
        2. Launch correction_fine_tuning.py as a subprocess.
        3. Stream stdout/stderr into the execution result.
        4. On success, mark used_in_training=True in feedback_log.jsonl.

        The dataset format is one JSON object per line with keys:
            instruction  (the question)
            input        (the model's original, flawed answer)
            output       (the human correction — the learning target)
            feedback_id  (traceability back to feedback_log.jsonl)
            bertscore    (float or null — can be used for loss weighting)

        This matches the SFT format that correction_fine_tuning.py expects
        (instruction / input / output triplets, where the LoRA model is
        trained to produce `output` given `instruction + input`).
        """
        examples       = payload.get("training_examples", [])
        base_version   = payload.get("base_model_version", "unknown")
        target_label   = payload.get("target_adapter_label", "unknown")

        if not examples:
            raise ValueError("RETRAIN_ADAPTER payload has no training_examples.")

        # Write dataset
        ts           = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Also write a timestamped backup for traceability
        backup_path  = FINETUNE_DATASET_DIR / f"{self.config.dataset_file_prefix}_{ts}.json"
        valid_ids: list[str] = []
        records_for_training: list[dict] = []

        for ex in examples:
            question  = (ex.get("question")  or "").strip()
            candidate = (ex.get("candidate") or "").strip()
            reference = (ex.get("reference") or "").strip()
            fid       = ex.get("feedback_id", "")

            if not question or not reference:
                log.warning("Skipping training example %s — empty question or reference.", fid)
                continue

            # correction_fine_tuning.py expects:
            # { "messages": [ {"role": "user", "content": <question>},
            #                  {"role": "assistant", "content": <reference>} ],
            #   "feedback_id": ..., "bertscore": ... }
            # The candidate (original bad answer) is intentionally excluded —
            # the script masks prompt tokens and trains only on the assistant turn.
            record = {
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": reference},
                ],
                "feedback_id": fid,
                "bertscore":   ex.get("bertscore"),
            }
            records_for_training.append(record)
            valid_ids.append(fid)

        # Write to the path the script reads by default (--train_file default)
        import json as _json
        CORRECTION_TRAIN_FILE.write_text(
            _json.dumps(records_for_training, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Timestamped backup
        backup_path.write_text(
            _json.dumps(records_for_training, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Dataset written: %s  (%d examples)", CORRECTION_TRAIN_FILE, len(valid_ids))
        log.info("Backup written:  %s", backup_path)

        if not valid_ids:
            raise ValueError("All training examples were invalid — nothing written to dataset.")

        # Build subprocess command matching correction_fine_tuning.py's actual CLI:
        #   --train_file   path to the JSON file we just wrote
        #   --output_dir   directory where the adapter will be saved
        #   --base_adapter existing HF adapter repo to continue training from (optional)
        #   --epochs / --lr / --lora_r / --lora_alpha passed through for full control
        adapter_dir = ADAPTER_OUTPUT_DIR / "spacellm_lora_final"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # If a previous adapter exists locally, continue from it (continual learning)
        base_adapter = self._state.get("last_adapter", {}).get("hf_repo_id") or None

        cmd = [
            self.config.python_executable,
            str(FINETUNE_SCRIPT),
            "--train_file",  str(CORRECTION_TRAIN_FILE),
            "--output_dir",  str(adapter_dir),
            "--lora_r",      "64",
            "--lora_alpha",  "128",
        ]
        if base_adapter:
            cmd += ["--base_adapter", base_adapter]
            log.info("Continuing from existing adapter: %s", base_adapter)

        log.info("Launching fine-tuning: %s", " ".join(cmd))
        log.info("Timeout: %ds", self.config.retrain_timeout_seconds)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        returncode = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ},  # inherit CUDA_VISIBLE_DEVICES, HF_HOME, etc.
            )

            # Stream output so the execution log reflects real-time progress
            import threading

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
                    f"Fine-tuning process timed out after {self.config.retrain_timeout_seconds}s "
                    f"and was killed."
                )

            t_out.join()
            t_err.join()

        except FileNotFoundError:
            raise RuntimeError(
                f"Fine-tuning script not found at {FINETUNE_SCRIPT}. "
                "Check FINE_TUNING_DIR path in execute.py."
            )

        if returncode != 0:
            tail_stderr = "\n".join(stderr_lines[-30:])
            raise RuntimeError(
                f"Fine-tuning exited with code {returncode}.\n"
                f"Last stderr:\n{tail_stderr}"
            )

        # Mark examples as used in feedback_log.jsonl
        marked = self._mark_used_in_training(set(valid_ids))
        log.info("Marked %d/%d feedback records as used_in_training=True.",
                 marked, len(valid_ids))

        # Persist the new adapter path in executor state for reference
        self._state["last_adapter"] = {
            "target_label":  target_label,
            "adapter_path":  str(adapter_dir),
            "dataset_path":  str(dataset_path),
            "example_count": len(valid_ids),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        return {
            "target_adapter_label": target_label,
            "adapter_path":         str(adapter_dir),
            "train_file":           str(CORRECTION_TRAIN_FILE),
            "backup_path":          str(backup_path),
            "examples_used":        len(valid_ids),
            "returncode":           returncode,
            "stdout_tail":          stdout_lines[-20:],
            "stderr_tail":          stderr_lines[-10:],
            "marked_in_feedback":   marked,
        }

    def _mark_used_in_training(self, feedback_ids: set[str]) -> int:
        """
        Atomically rewrite feedback_log.jsonl, flipping used_in_training=True
        for every record whose feedback_id is in the provided set.
        Returns the count of records actually updated.
        """
        if not FEEDBACK_LOG.exists() or not feedback_ids:
            return 0

        updated  = 0
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
                    new_lines.append(stripped)  # preserve malformed lines as-is

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

        The file is read by the FastAPI server (or controller.py) at startup
        to build the system prompt.  Format is Markdown so it's human-readable
        and a simple parser can extract patch blocks by the sentinel lines.

        Each block:
            <!-- PATCH_START patch_key=<key> applied_at=<iso> -->
            <patch_text>
            <!-- PATCH_END patch_key=<key> -->

        A reader that assembles the system prompt should collect all PATCH
        blocks in file order and append their text to the base system prompt,
        deduplicating by patch_key (last-write wins) so re-issuing a patch
        after a cooldown cleanly replaces the old wording.
        """
        patch_key  = payload.get("patch_key", "unknown_patch")
        patch_text = payload.get("patch_text", "").strip()
        target     = payload.get("target", "system_prompt")

        if not patch_text:
            raise ValueError(f"PROMPT_PATCH payload has empty patch_text for key '{patch_key}'.")

        # Guard file size
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

    # ------------------------------------------------------------------
    # Handler: TOPIC_GUARDRAIL
    # ------------------------------------------------------------------

    def _execute_topic_guardrail(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Merge the guardrail directive into mape_k/topic_guardrail.json.

        Schema of topic_guardrail.json:
        {
          "updated_at": "<ISO>",
          "topic_specific": bool,
          "topics_reported": [...],
          "suggested_action": "...",
          "history": [
            { "applied_at": ..., "topics_reported": [...], "suggested_action": ... },
            ...
          ]
        }

        The FastAPI core or controller.py should read this file at startup
        (and optionally poll it) to apply the suggested instructions.  The
        executor doesn't reach into the running server — it writes a file
        and the server is responsible for picking it up.
        """
        topics           = payload.get("topics_reported", [])
        suggested_action = payload.get("suggested_action", "")
        topic_specific   = payload.get("topic_specific", False)
        now              = datetime.now(timezone.utc).isoformat()

        # Load existing state
        existing: dict[str, Any] = {}
        if TOPIC_GUARDRAIL_FILE.exists():
            try:
                existing = json.loads(TOPIC_GUARDRAIL_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Could not parse topic_guardrail.json (%s) — overwriting.", exc)

        history: list[dict] = existing.get("history", [])
        history.append({
            "applied_at":      now,
            "topics_reported": topics,
            "suggested_action": suggested_action,
        })

        guardrail = {
            "updated_at":      now,
            "topic_specific":  topic_specific,
            "topics_reported": topics,
            "suggested_action": suggested_action,
            "history":         history,
        }
        _atomic_write_json(TOPIC_GUARDRAIL_FILE, guardrail)

        log.info("Topic guardrail updated. topics=%s  topic_specific=%s", topics, topic_specific)

        return {
            "guardrail_file":   str(TOPIC_GUARDRAIL_FILE),
            "topics_reported":  topics,
            "topic_specific":   topic_specific,
            "history_entries":  len(history),
        }

    # ------------------------------------------------------------------
    # Handler: FLAG_FOR_REVIEW
    # ------------------------------------------------------------------

    def _execute_flag_for_review(self, payload: dict, action: dict) -> dict[str, Any]:
        """
        Append flagged items to mape_k/human_review_queue.jsonl and
        update aggregate counters in mape_k/review_stats.json.

        Two payload shapes are supported (as emitted by plan.py):
          Shape A — repeated failures:
            { "flagged_questions": [ { "question_key", "negative_count", "feedback_id" }, ... ] }
          Shape B — retrain cooldown:
            { "reason": "retrain_cooldown_active", "elapsed_hours": float }
        """
        now     = datetime.now(timezone.utc).isoformat()
        appended = 0

        # Load review stats
        stats: dict[str, Any] = {}
        if REVIEW_STATS_FILE.exists():
            try:
                stats = json.loads(REVIEW_STATS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        stats.setdefault("total_flagged", 0)
        stats.setdefault("by_reason", {})

        with REVIEW_QUEUE_FILE.open("a", encoding="utf-8") as fh:

            # Shape A: repeated-failure questions
            if "flagged_questions" in payload:
                for item in payload.get("flagged_questions", []):
                    record = {
                        "review_id":      str(uuid.uuid4()),
                        "flagged_at":     now,
                        "action_id":      action.get("action_id"),
                        "plan_id":        action.get("plan_id"),
                        "category":       "repeated_failure",
                        "status":         "OPEN",
                        **item,
                    }
                    fh.write(json.dumps(record) + "\n")
                    appended += 1
                stats["total_flagged"]                        += appended
                stats["by_reason"]["repeated_failure"]         = (
                    stats["by_reason"].get("repeated_failure", 0) + appended
                )

            # Shape B: retrain cooldown
            elif "reason" in payload:
                reason = payload.get("reason", "unknown")
                record = {
                    "review_id":    str(uuid.uuid4()),
                    "flagged_at":   now,
                    "action_id":    action.get("action_id"),
                    "plan_id":      action.get("plan_id"),
                    "category":     reason,
                    "status":       "OPEN",
                    "elapsed_hours": payload.get("elapsed_hours"),
                    "reasoning":    action.get("reasoning", []),
                }
                fh.write(json.dumps(record) + "\n")
                appended += 1
                stats["total_flagged"]            += 1
                stats["by_reason"][reason]         = stats["by_reason"].get(reason, 0) + 1

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
        """
        Read plan_actions.jsonl and return actions that are:
          - status == "PENDING"
          - auto_approved == True
          - action_id not already in _state["executed_ids"]
        """
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
                    log.warning("plan_actions.jsonl line %d: missing action_id — skipping.", lineno)
                    continue
                if action_id in executed_ids:
                    continue
                if action.get("status") != "PENDING":
                    continue
                if not action.get("auto_approved", False):
                    log.info("Action %s requires manual approval — skipping.", action_id[:8])
                    continue
                pending.append(action)

        log.info("Loaded %d PENDING action(s).", len(pending))
        return pending

    def _update_action_statuses(self, updated_ids: dict[str, str]) -> None:
        """
        Atomically rewrite plan_actions.jsonl, updating the `status` field
        for every action_id in `updated_ids`.
        """
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
                    aid    = action.get("action_id")
                    if aid in updated_ids:
                        action["status"]      = updated_ids[aid]
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
            log.error("Failed to write execution log entry: %s", exc)

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
    # Failed-action logging (matches poison-pill pattern in the other components)
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
            status_icon = "✓" if rec.status == "EXECUTED" else "✗"
            print(
                f"  {status_icon} [{rec.action_type:<20}] "
                f"status={rec.status:<8}  "
                f"duration={rec.duration_s:.1f}s  "
                f"action_id={rec.action_id[:8]}"
            )
        executed = sum(1 for r in records if r.status == "EXECUTED")
        failed   = sum(1 for r in records if r.status == "FAILED")
        print(f"\n  Total: {executed} EXECUTED, {failed} FAILED")
        print(f"\n  Execution log → {EXECUTION_LOG}")
    print(f"{'='*60}\n")
