"""
SpaceLLM MAPE-K :: Executor Component
=======================================
Responsibility: Read Planner actions → execute them → update statuses.

Actions handled:
  RETRAIN_ADAPTER  → build training data from corrections → run
                     lora_finetuning_v4.py → push adapter to HuggingFace
                     → hot-reload backend → mark corrections used
  PROMPT_PATCH     → patch system prompt in main.py → restart uvicorn
  TOPIC_GUARDRAIL  → log + write to human_review_queue.jsonl
  FLAG_FOR_REVIEW  → write to human_review_queue.jsonl
  NO_ACTION        → log and skip

Pipeline position:
    Planner (plan_actions.jsonl)
        ↓
    Executor  ← YOU ARE HERE
        ↓
    SpaceLLM_v(N+1) live on GPU

Author: SpaceLLM Project
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
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

BASE_DIR            = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR            = BASE_DIR / "mape_k"
FINE_TUNING_DIR     = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/fine_tuning_v2")
ADAPTER_OUTPUT_DIR  = FINE_TUNING_DIR / "outputs" / "spacellm_lora_final"
TRAIN_DATA_DIR      = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2")
CORRECTION_TRAIN_FILE = MAPE_DIR / "correction_train_injection.json"

PLAN_ACTIONS_LOG    = MAPE_DIR / "plan_actions.jsonl"
PLAN_REPORT_PATH    = MAPE_DIR / "plan_report.json"
FEEDBACK_LOG        = BASE_DIR / "feedback_log.jsonl"
MAIN_PY_PATH        = BASE_DIR / "main.py"
HUMAN_REVIEW_QUEUE  = MAPE_DIR / "human_review_queue.jsonl"
EXECUTION_LOG       = MAPE_DIR / "execution_log.jsonl"
STATE_FILE          = MAPE_DIR / ".executor_state.json"

MAPE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    # HuggingFace
    hf_repo_id:              str   = "AdityaPS/SpaceLLM_v1"   # updated per version
    hf_token_env:            str   = "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"               # env var name

    # Fine-tuning script
    finetune_script:         str   = str(FINE_TUNING_DIR / "lora_finetuning_v4.py")
    cuda_visible_devices:    str   = "1"
    finetune_epochs:         int   = 3
    finetune_lr:             float = 2e-4
    finetune_max_seq_len:    int   = 2048

    # Backend restart
    uvicorn_host:            str   = "localhost"
    uvicorn_port:            int   = 8000
    uvicorn_startup_wait_s:  int   = 300   # wait up to 5 min for model to load

    # Prompt patch
    system_prompt_marker:    str   = "SYSTEM_PROMPT = ("


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    action_id:    str
    action_type:  str
    status:       str       # "EXECUTED" | "FAILED" | "SKIPPED"
    timestamp:    str
    duration_s:   float
    details:      dict = field(default_factory=dict)
    error:        str  = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config or ExecutorConfig()
        self._state = self._load_state()
        log.info("Executor initialised.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[ExecutionResult]:
        log.info("=" * 60)
        log.info("Executor cycle starting.")

        actions = self._load_pending_actions()
        if not actions:
            log.info("No PENDING actions found.")
            return []

        log.info("Found %d PENDING action(s).", len(actions))
        # Sort by priority descending — highest priority executes first
        actions.sort(key=lambda a: a.get("priority", 0), reverse=True)

        results: list[ExecutionResult] = []
        for action in actions:
            result = self._execute_action(action)
            results.append(result)
            _append_jsonl(EXECUTION_LOG, result.to_dict())
            self._update_action_status(action["action_id"], result.status, result.error)

        self._save_state()
        log.info("Executor cycle complete. %d action(s) processed.", len(results))
        log.info("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _execute_action(self, action: dict) -> ExecutionResult:
        action_type = action.get("action_type", "UNKNOWN")
        action_id   = action.get("action_id", "?")
        t0          = time.time()

        log.info("Executing [%s] action_id=%s ...", action_type, action_id[:8])

        try:
            if action_type == "RETRAIN_ADAPTER":
                details = self._execute_retrain(action)
            elif action_type == "PROMPT_PATCH":
                details = self._execute_prompt_patch(action)
            elif action_type == "TOPIC_GUARDRAIL":
                details = self._execute_guardrail(action)
            elif action_type == "FLAG_FOR_REVIEW":
                details = self._execute_flag_for_review(action)
            elif action_type == "NO_ACTION":
                details = {"message": "No action required this cycle."}
            else:
                raise ValueError(f"Unknown action_type: {action_type}")

            return ExecutionResult(
                action_id   = action_id,
                action_type = action_type,
                status      = "EXECUTED",
                timestamp   = datetime.now(timezone.utc).isoformat(),
                duration_s  = round(time.time() - t0, 2),
                details     = details,
            )

        except Exception as exc:
            log.error("Action [%s] %s FAILED: %s", action_type, action_id[:8], exc, exc_info=True)
            return ExecutionResult(
                action_id   = action_id,
                action_type = action_type,
                status      = "FAILED",
                timestamp   = datetime.now(timezone.utc).isoformat(),
                duration_s  = round(time.time() - t0, 2),
                error       = str(exc),
            )

    # ------------------------------------------------------------------
    # RETRAIN_ADAPTER
    # ------------------------------------------------------------------

    def _execute_retrain(self, action: dict) -> dict:
        payload        = action.get("payload", {})
        base_version   = payload.get("base_model_version", "SpaceLLM_v1")
        target_label   = payload.get("target_adapter_label", "SpaceLLM_v2")
        examples       = payload.get("training_examples", [])

        if not examples:
            raise ValueError("No training examples in RETRAIN_ADAPTER payload.")

        log.info("Retraining: %s → %s  (%d examples)",
                 base_version, target_label, len(examples))

        # Step 1 — Build correction training file
        self._build_correction_train_file(examples)

        # Step 2 — Run fine-tuning script
        self._run_finetuning(target_label)

        # Step 3 — Push new adapter to HuggingFace
        new_repo_id = self._push_to_hf(target_label)

        # Step 4 — Update ADAPTER_MODEL_ID in main.py
        self._update_adapter_in_main_py(new_repo_id, target_label)

        # Step 5 — Mark corrections as used_in_training in feedback_log
        feedback_ids = [e.get("feedback_id") for e in examples if e.get("feedback_id")]
        self._mark_corrections_used(feedback_ids)

        # Step 6 — Restart uvicorn with new adapter
        self._restart_backend()

        return {
            "base_version":     base_version,
            "new_version":      target_label,
            "new_repo_id":      new_repo_id,
            "examples_used":    len(examples),
            "feedback_ids_marked": len(feedback_ids),
        }

    def _build_correction_train_file(self, examples: list[dict]) -> None:
        """
        Convert correction pairs into the training JSON format expected by
        lora_finetuning_v4.py (same schema as DatasetA_core_QA_v2).
        Each example becomes a multi-turn conversation:
          user: [original question]
          assistant: [human correction / reference answer]
        """
        records = []
        for i, ex in enumerate(examples):
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            if not question or not reference:
                continue

            records.append({
                "sample_id":    f"correction_{i:04d}",
                "source_id":    ex.get("feedback_id", f"fb_{i}"),
                "mission_name": "MAPE-K Correction",
                "organization": "SpaceLLM",
                "aspect":       "correction",
                "difficulty":   "medium",
                "chain_id":     f"chain_correction_{i:04d}",
                "bertscore":    ex.get("bertscore"),
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": reference},
                ],
            })

        if not records:
            raise ValueError("No valid correction examples after filtering.")

        CORRECTION_TRAIN_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Correction training file written: %d records → %s",
                 len(records), CORRECTION_TRAIN_FILE)

    def _run_finetuning(self, target_label: str) -> None:
        """Run lora_finetuning_v4.py with the correction data injected."""
        script = self.config.finetune_script
        if not Path(script).exists():
            raise FileNotFoundError(f"Fine-tuning script not found: {script}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        cmd = [
            sys.executable, script,
            "--epochs",      str(self.config.finetune_epochs),
            "--lr",          str(self.config.finetune_lr),
            "--max_seq_len", str(self.config.finetune_max_seq_len),
        ]

        log.info("Running fine-tuning: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            env     = env,
            capture_output = False,   # stream output to terminal
            check   = False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Fine-tuning script exited with code {result.returncode}."
            )
        log.info("Fine-tuning complete for %s.", target_label)

    def _push_to_hf(self, target_label: str) -> str:
        """
        Push the newly trained adapter to HuggingFace Hub.
        Returns the new repo_id (e.g. AdityaPS/SpaceLLM_v2).
        """
        from huggingface_hub import HfApi

        hf_token = os.environ.get(self.config.hf_token_env)
        if not hf_token:
            raise EnvironmentError(
                f"HuggingFace token not found. "
                f"Set env var: export {self.config.hf_token_env}=<your_token>"
            )

        # Derive new repo_id from target_label
        # e.g. "SpaceLLM_v2" → "AdityaPS/SpaceLLM_v2"
        base_org  = self.config.hf_repo_id.split("/")[0]
        new_repo  = f"{base_org}/{target_label}"

        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=new_repo, repo_type="model", exist_ok=True)
            log.info("HF repo ready: %s", new_repo)
        except Exception as exc:
            raise RuntimeError(f"Failed to create HF repo {new_repo}: {exc}")

        # Upload adapter folder
        if not ADAPTER_OUTPUT_DIR.exists():
            raise FileNotFoundError(
                f"Adapter output directory not found: {ADAPTER_OUTPUT_DIR}"
            )

        log.info("Uploading adapter from %s → %s ...", ADAPTER_OUTPUT_DIR, new_repo)
        api.upload_folder(
            folder_path = str(ADAPTER_OUTPUT_DIR),
            repo_id     = new_repo,
            repo_type   = "model",
            commit_message = f"MAPE-K auto-retrain: {target_label}",
        )
        log.info("Adapter pushed to HuggingFace: https://huggingface.co/%s", new_repo)
        return new_repo

    def _update_adapter_in_main_py(self, new_repo_id: str, target_label: str) -> None:
        """Update ADAPTER_MODEL_ID in main.py to point at the new adapter."""
        if not MAIN_PY_PATH.exists():
            raise FileNotFoundError(f"main.py not found: {MAIN_PY_PATH}")

        content = MAIN_PY_PATH.read_text(encoding="utf-8")

        # Replace ADAPTER_MODEL_ID = "..." line
        new_content = re.sub(
            r'ADAPTER_MODEL_ID\s*=\s*"[^"]+"',
            f'ADAPTER_MODEL_ID    = "{new_repo_id}"',
            content,
        )
        if new_content == content:
            raise ValueError("Could not find ADAPTER_MODEL_ID in main.py to update.")

        MAIN_PY_PATH.write_text(new_content, encoding="utf-8")
        log.info("main.py updated: ADAPTER_MODEL_ID → %s", new_repo_id)

    def _mark_corrections_used(self, feedback_ids: list[str]) -> None:
        """Flip used_in_training=True for all feedback_ids in feedback_log.jsonl."""
        if not FEEDBACK_LOG.exists() or not feedback_ids:
            return

        id_set    = set(feedback_ids)
        updated   = 0
        new_lines = []

        with FEEDBACK_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("feedback_id") in id_set:
                        record["used_in_training"] = True
                        updated += 1
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(line)

        tmp = FEEDBACK_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(FEEDBACK_LOG)
        log.info("Marked %d feedback record(s) as used_in_training=True.", updated)

    def _restart_backend(self) -> None:
        """
        Kill the running uvicorn process and restart it with the updated main.py.
        Waits until /health returns ready=true before returning.
        """
        import signal as _signal

        # Find uvicorn PID
        try:
            result = subprocess.run(
                ["pgrep", "-f", "uvicorn main:app"],
                capture_output=True, text=True,
            )
            pids = [int(p) for p in result.stdout.strip().split() if p.isdigit()]
        except Exception:
            pids = []

        for pid in pids:
            try:
                os.kill(pid, _signal.SIGTERM)
                log.info("Sent SIGTERM to uvicorn PID %d.", pid)
            except ProcessLookupError:
                pass

        time.sleep(5)   # give uvicorn time to shut down cleanly

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", self.config.uvicorn_host,
            "--port", str(self.config.uvicorn_port),
        ]
        log.info("Starting uvicorn: %s", " ".join(cmd))
        subprocess.Popen(cmd, env=env, cwd=str(BASE_DIR))

        # Poll /health until ready
        import urllib.request
        health_url = f"http://{self.config.uvicorn_host}:{self.config.uvicorn_port}/health"
        deadline   = time.time() + self.config.uvicorn_startup_wait_s
        log.info("Waiting for backend to become ready (up to %ds) ...",
                 self.config.uvicorn_startup_wait_s)

        while time.time() < deadline:
            time.sleep(10)
            try:
                with urllib.request.urlopen(health_url, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("ready"):
                        log.info("Backend is ready. New adapter loaded.")
                        return
            except Exception:
                pass

        raise TimeoutError(
            f"Backend did not become ready within {self.config.uvicorn_startup_wait_s}s."
        )

    # ------------------------------------------------------------------
    # PROMPT_PATCH
    # ------------------------------------------------------------------

    def _execute_prompt_patch(self, action: dict) -> dict:
        payload    = action.get("payload", {})
        patch_text = payload.get("patch_text", "").strip()
        patch_key  = payload.get("patch_key", "unknown")

        if not patch_text:
            raise ValueError("patch_text is empty in PROMPT_PATCH payload.")

        if not MAIN_PY_PATH.exists():
            raise FileNotFoundError(f"main.py not found: {MAIN_PY_PATH}")

        content = MAIN_PY_PATH.read_text(encoding="utf-8")

        # Find the SYSTEM_PROMPT block and append the patch as a new sentence
        marker = self.config.system_prompt_marker
        idx    = content.find(marker)
        if idx == -1:
            raise ValueError(f"Could not find '{marker}' in main.py.")

        # Find the closing paren of the SYSTEM_PROMPT tuple
        close_idx = content.find("\n)", idx)
        if close_idx == -1:
            raise ValueError("Could not find closing paren of SYSTEM_PROMPT in main.py.")

        # Insert patch as last line of SYSTEM_PROMPT before closing paren
        patch_line = f'    "{patch_text}"'
        new_content = (
            content[:close_idx]
            + "\n"
            + patch_line
            + content[close_idx:]
        )

        MAIN_PY_PATH.write_text(new_content, encoding="utf-8")
        log.info("Prompt patch '%s' applied to main.py.", patch_key)

        # Restart backend to pick up new prompt
        self._restart_backend()

        return {
            "patch_key":  patch_key,
            "patch_text": patch_text,
            "applied_to": str(MAIN_PY_PATH),
        }

    # ------------------------------------------------------------------
    # TOPIC_GUARDRAIL
    # ------------------------------------------------------------------

    def _execute_guardrail(self, action: dict) -> dict:
        payload = action.get("payload", {})
        record  = {
            "type":           "TOPIC_GUARDRAIL",
            "action_id":      action.get("action_id"),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "topics":         payload.get("topics_reported", []),
            "topic_specific": payload.get("topic_specific", False),
            "suggestion":     payload.get("suggested_action", ""),
            "reasoning":      action.get("reasoning", []),
        }
        _append_jsonl(HUMAN_REVIEW_QUEUE, record)
        log.info("Topic guardrail logged to human_review_queue.jsonl.")
        return {"written_to": str(HUMAN_REVIEW_QUEUE)}

    # ------------------------------------------------------------------
    # FLAG_FOR_REVIEW
    # ------------------------------------------------------------------

    def _execute_flag_for_review(self, action: dict) -> dict:
        payload = action.get("payload", {})
        record  = {
            "type":              "FLAG_FOR_REVIEW",
            "action_id":         action.get("action_id"),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "flagged_questions": payload.get("flagged_questions", []),
            "reasoning":         action.get("reasoning", []),
        }
        _append_jsonl(HUMAN_REVIEW_QUEUE, record)
        log.info("Flagged %d question(s) for human review.",
                 len(payload.get("flagged_questions", [])))
        return {
            "flagged_count": len(payload.get("flagged_questions", [])),
            "written_to":    str(HUMAN_REVIEW_QUEUE),
        }

    # ------------------------------------------------------------------
    # plan_actions.jsonl management
    # ------------------------------------------------------------------

    def _load_pending_actions(self) -> list[dict]:
        if not PLAN_ACTIONS_LOG.exists():
            return []

        pending = []
        seen    = set()
        with PLAN_ACTIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    action = json.loads(line)
                except json.JSONDecodeError:
                    continue

                aid = action.get("action_id")
                if not aid or aid in seen:
                    continue
                seen.add(aid)

                if action.get("status") == "PENDING":
                    pending.append(action)

        return pending

    def _update_action_status(
        self, action_id: str, status: str, error: str = ""
    ) -> None:
        """Rewrite plan_actions.jsonl updating the status of one action."""
        if not PLAN_ACTIONS_LOG.exists():
            return

        new_lines = []
        with PLAN_ACTIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("action_id") == action_id:
                        record["status"]      = status
                        record["executed_at"] = datetime.now(timezone.utc).isoformat()
                        if error:
                            record["error"] = error
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(line)

        tmp = PLAN_ACTIONS_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(PLAN_ACTIONS_LOG)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception as exc:
                log.warning("Could not load executor state (%s). Starting fresh.", exc)
        return {}

    def _save_state(self) -> None:
        _atomic_write_json(STATE_FILE, self._state)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor = Executor()
    results  = executor.run()

    print(f"\n{'='*60}")
    print(f"  SpaceLLM Executor — Cycle Complete")
    print(f"{'='*60}")
    if not results:
        print("  No actions executed.")
    else:
        for r in results:
            icon = "✓" if r.status == "EXECUTED" else "✗"
            print(f"  {icon} [{r.action_type:<16}] {r.status:<8}  ({r.duration_s:.1f}s)")
            if r.error:
                print(f"    ERROR: {r.error}")
    print(f"\n  Execution log → {EXECUTION_LOG}")
    print(f"  Review queue  → {HUMAN_REVIEW_QUEUE}")
    print(f"{'='*60}\n")
