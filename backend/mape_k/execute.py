"""
SpaceLLM MAPE-K :: Executor Component
=======================================
Responsibility: Read Planner actions → execute them → update statuses.

Actions handled:
  RETRAIN_ADAPTER  → build correction_train_injection.json
                   → run lora_finetuning_v5.py (delta training on corrections only)
                   → push new adapter to HuggingFace
                   → update ADAPTER_MODEL_ID in main.py
                   → mark corrections as used_in_training=True in feedback_log.jsonl
                   → restart uvicorn and wait for /health ready=true

  PROMPT_PATCH     → inject patch_text into SYSTEM_PROMPT in main.py → restart

  TOPIC_GUARDRAIL  → log to human_review_queue.jsonl

  FLAG_FOR_REVIEW  → log to human_review_queue.jsonl

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
import subprocess
import sys
import time
import urllib.request
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
FINE_TUNING_DIR      = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/fine_tuning_v2")

# v5 adapter output dir  (matches FINAL_DIR in lora_finetuning_v5.py)
ADAPTER_OUTPUT_DIR   = FINE_TUNING_DIR / "outputs_v5" / "spacellm_lora_final"

# Original train data for replay buffer (used by v5)
ORIGINAL_TRAIN_FILE  = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/train.json"
)
VAL_FILE             = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/validate.json"
)

# MAPE-K files
CORRECTION_FILE      = MAPE_DIR / "correction_train_injection.json"
PLAN_ACTIONS_LOG     = MAPE_DIR / "plan_actions.jsonl"
FEEDBACK_LOG         = BASE_DIR / "feedback_log.jsonl"
MAIN_PY_PATH         = BASE_DIR / "main.py"
HUMAN_REVIEW_QUEUE   = MAPE_DIR / "human_review_queue.jsonl"
EXECUTION_LOG        = MAPE_DIR / "execution_log.jsonl"
STATE_FILE           = MAPE_DIR / ".executor_state.json"

MAPE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    # Fine-tuning script — v5 does delta training (corrections + replay only)
    finetune_script:          str   = str(FINE_TUNING_DIR / "lora_finetuning_v5.py")
    cuda_visible_devices:     str   = "1"

    # v5 hyperparams (smaller than v4 since dataset is tiny)
    finetune_epochs:          int   = 3
    finetune_lr:              float = 5e-5
    finetune_replay_ratio:    float = 0.30   # 30% replay buffer
    finetune_max_seq_len:     int   = 2048

    # HuggingFace push
    hf_token_env:             str   = "HF_TOKEN"
    hf_base_org:              str   = "AdityaPS"

    # Backend restart
    uvicorn_host:             str   = "localhost"
    uvicorn_port:             int   = 8000
    uvicorn_startup_wait_s:   int   = 360    # 6 min for model to load

    # Prompt patch
    system_prompt_marker:     str   = "SYSTEM_PROMPT = ("
    system_prompt_close:      str   = "\n)"


# ---------------------------------------------------------------------------
# Helpers
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
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    action_id:   str
    action_type: str
    status:      str       # "EXECUTED" | "FAILED" | "SKIPPED"
    timestamp:   str
    duration_s:  float
    details:     dict = field(default_factory=dict)
    error:       str  = ""

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

        # Highest priority first
        actions.sort(key=lambda a: a.get("priority", 0), reverse=True)
        log.info("Found %d PENDING action(s).", len(actions))

        results: list[ExecutionResult] = []
        for action in actions:
            result = self._execute_action(action)
            results.append(result)
            _append_jsonl(EXECUTION_LOG, result.to_dict())
            self._update_action_status(
                action["action_id"], result.status, result.error
            )

        self._save_state()
        log.info("Executor cycle complete. %d action(s) processed.", len(results))
        log.info("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _execute_action(self, action: dict) -> ExecutionResult:
        atype = action.get("action_type", "UNKNOWN")
        aid   = action.get("action_id", "?")
        t0    = time.time()
        log.info("Executing [%s] action_id=%s ...", atype, aid[:8])

        try:
            if atype == "RETRAIN_ADAPTER":
                details = self._execute_retrain(action)
            elif atype == "PROMPT_PATCH":
                details = self._execute_prompt_patch(action)
            elif atype == "TOPIC_GUARDRAIL":
                details = self._execute_guardrail(action)
            elif atype == "FLAG_FOR_REVIEW":
                details = self._execute_flag_for_review(action)
            elif atype == "NO_ACTION":
                details = {"message": "No action required this cycle."}
                log.info("NO_ACTION — skipping.")
            else:
                raise ValueError(f"Unknown action_type: {atype}")

            return ExecutionResult(
                action_id   = aid,
                action_type = atype,
                status      = "EXECUTED",
                timestamp   = datetime.now(timezone.utc).isoformat(),
                duration_s  = round(time.time() - t0, 2),
                details     = details,
            )

        except Exception as exc:
            log.error("Action [%s] %s FAILED: %s", atype, aid[:8], exc, exc_info=True)
            return ExecutionResult(
                action_id   = aid,
                action_type = atype,
                status      = "FAILED",
                timestamp   = datetime.now(timezone.utc).isoformat(),
                duration_s  = round(time.time() - t0, 2),
                error       = str(exc),
            )

    # ------------------------------------------------------------------
    # RETRAIN_ADAPTER
    # ------------------------------------------------------------------

    def _execute_retrain(self, action: dict) -> dict:
        payload       = action.get("payload", {})
        base_version  = payload.get("base_model_version", "SpaceLLM_v1")
        target_label  = payload.get("target_adapter_label", "SpaceLLM_v2")
        examples      = payload.get("training_examples", [])

        if not examples:
            raise ValueError("RETRAIN_ADAPTER payload has no training_examples.")

        log.info("Retraining: %s -> %s  (%d correction examples)",
                 base_version, target_label, len(examples))

        # 1. Write correction_train_injection.json
        self._write_correction_file(examples)

        # 2. Derive current adapter repo from main.py so v5 continues from it
        current_adapter = self._read_current_adapter_from_main()
        log.info("Continuing from adapter: %s", current_adapter)

        # 3. Run lora_finetuning_v5.py (delta training)
        self._run_finetuning_v5(current_adapter)

        # 4. Push new adapter to HuggingFace
        new_repo_id = self._push_to_hf(target_label)

        # 5. Update ADAPTER_MODEL_ID in main.py
        self._update_adapter_in_main_py(new_repo_id)

        # 6. Mark corrections as used_in_training=True
        feedback_ids = [e.get("feedback_id") for e in examples if e.get("feedback_id")]
        self._mark_corrections_used(feedback_ids)

        # 7. Restart backend
        self._restart_backend()

        return {
            "base_version":          base_version,
            "new_version":           target_label,
            "new_repo_id":           new_repo_id,
            "continued_from":        current_adapter,
            "examples_used":         len(examples),
            "feedback_ids_marked":   len(feedback_ids),
        }

    def _write_correction_file(self, examples: list[dict]) -> None:
        """
        Convert Planner correction examples into lora_finetuning_v5.py's
        expected format: each example is a record with a `messages` list
        containing a user turn (question) and an assistant turn (reference).
        """
        records = []
        for i, ex in enumerate(examples):
            question  = (ex.get("question")  or "").strip()
            reference = (ex.get("reference") or "").strip()
            if not question or not reference:
                log.debug("Skipping example %d — missing question or reference.", i)
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
            raise ValueError(
                "No valid correction records after filtering "
                "(all had empty question or reference)."
            )

        CORRECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        CORRECTION_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Correction file written: %d records -> %s",
                 len(records), CORRECTION_FILE)

    def _read_current_adapter_from_main(self) -> str:
        """Read the current ADAPTER_MODEL_ID value from main.py."""
        if not MAIN_PY_PATH.exists():
            return "AdityaPS/SpaceLLM_v1"
        content = MAIN_PY_PATH.read_text(encoding="utf-8")
        match = re.search(r'ADAPTER_MODEL_ID\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
        log.warning("Could not parse ADAPTER_MODEL_ID from main.py — using default.")
        return "AdityaPS/SpaceLLM_v1"

    def _run_finetuning_v5(self, current_adapter: str) -> None:
        """
        Run lora_finetuning_v5.py with:
          --correction_file  correction_train_injection.json
          --adapter_base     <current live adapter>   (continue from it, not scratch)
          --original_train   DatasetA_core_QA_v2/train.json  (replay buffer pool)
          --replay_ratio     0.30
          --epochs / --lr    from config
        """
        script = self.config.finetune_script
        if not Path(script).exists():
            raise FileNotFoundError(f"v5 fine-tuning script not found: {script}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        cmd = [
            sys.executable, script,
            "--correction_file", str(CORRECTION_FILE),
            "--adapter_base",    current_adapter,
            "--original_train",  str(ORIGINAL_TRAIN_FILE),
            "--val_file",        str(VAL_FILE),
            "--replay_ratio",    str(self.config.finetune_replay_ratio),
            "--epochs",          str(self.config.finetune_epochs),
            "--lr",              str(self.config.finetune_lr),
            "--max_seq_len",     str(self.config.finetune_max_seq_len),
        ]

        log.info("Running v5 fine-tuning: %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"lora_finetuning_v5.py exited with code {result.returncode}."
            )
        log.info("v5 fine-tuning complete.")

    def _push_to_hf(self, target_label: str) -> str:
        """
        Push adapter from ADAPTER_OUTPUT_DIR to HuggingFace Hub.
        Returns new repo_id e.g. 'AdityaPS/SpaceLLM_v2'.
        """
        from huggingface_hub import HfApi

        hf_token = os.environ.get(self.config.hf_token_env)
        if not hf_token:
            raise EnvironmentError(
                f"HF token not set. Run: export {self.config.hf_token_env}=<your_token>"
            )

        new_repo = f"{self.config.hf_base_org}/{target_label}"
        api      = HfApi(token=hf_token)

        # Create repo (idempotent)
        api.create_repo(repo_id=new_repo, repo_type="model", exist_ok=True)
        log.info("HF repo ready: %s", new_repo)

        if not ADAPTER_OUTPUT_DIR.exists():
            raise FileNotFoundError(
                f"Adapter output directory not found: {ADAPTER_OUTPUT_DIR}\n"
                f"Ensure lora_finetuning_v5.py completed successfully."
            )

        log.info("Uploading adapter %s -> %s ...", ADAPTER_OUTPUT_DIR, new_repo)
        api.upload_folder(
            folder_path    = str(ADAPTER_OUTPUT_DIR),
            repo_id        = new_repo,
            repo_type      = "model",
            commit_message = f"MAPE-K auto-retrain: {target_label}",
        )
        log.info("Adapter pushed: https://huggingface.co/%s", new_repo)
        return new_repo

    def _update_adapter_in_main_py(self, new_repo_id: str) -> None:
        """Patch ADAPTER_MODEL_ID in main.py to point at the new adapter."""
        if not MAIN_PY_PATH.exists():
            raise FileNotFoundError(f"main.py not found: {MAIN_PY_PATH}")

        content = MAIN_PY_PATH.read_text(encoding="utf-8")
        new_content = re.sub(
            r'(ADAPTER_MODEL_ID\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{new_repo_id}"',
            content,
        )
        if new_content == content:
            raise ValueError("Could not find ADAPTER_MODEL_ID in main.py.")

        MAIN_PY_PATH.write_text(new_content, encoding="utf-8")
        log.info("main.py updated: ADAPTER_MODEL_ID = %s", new_repo_id)

    def _mark_corrections_used(self, feedback_ids: list[str]) -> None:
        """Flip used_in_training=True for all used feedback_ids."""
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
        """Kill uvicorn, restart it, wait for /health ready=true."""
        import signal as _signal

        # Find and kill existing uvicorn
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

        time.sleep(5)   # let uvicorn shut down

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", self.config.uvicorn_host,
            "--port", str(self.config.uvicorn_port),
        ]
        log.info("Starting uvicorn: %s", " ".join(cmd))
        subprocess.Popen(cmd, env=env, cwd=str(BASE_DIR))

        # Poll /health
        health_url = (
            f"http://{self.config.uvicorn_host}:{self.config.uvicorn_port}/health"
        )
        deadline = time.time() + self.config.uvicorn_startup_wait_s
        log.info("Waiting for backend (up to %ds) ...",
                 self.config.uvicorn_startup_wait_s)

        while time.time() < deadline:
            time.sleep(15)
            try:
                with urllib.request.urlopen(health_url, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("ready"):
                        log.info("Backend ready — new adapter loaded successfully.")
                        return
                    log.info("Backend starting... (ready=%s)", data.get("ready"))
            except Exception as exc:
                log.debug("Health check: %s", exc)

        raise TimeoutError(
            f"Backend did not become ready within {self.config.uvicorn_startup_wait_s}s."
        )

    # ------------------------------------------------------------------
    # PROMPT_PATCH
    # ------------------------------------------------------------------

    def _execute_prompt_patch(self, action: dict) -> dict:
        payload    = action.get("payload", {})
        patch_text = (payload.get("patch_text") or "").strip()
        patch_key  = payload.get("patch_key", "unknown")

        if not patch_text:
            raise ValueError("patch_text is empty in PROMPT_PATCH payload.")
        if not MAIN_PY_PATH.exists():
            raise FileNotFoundError(f"main.py not found: {MAIN_PY_PATH}")

        content = MAIN_PY_PATH.read_text(encoding="utf-8")

        marker    = self.config.system_prompt_marker
        start_idx = content.find(marker)
        if start_idx == -1:
            raise ValueError(f"Could not find '{marker}' in main.py.")

        # Find the closing paren of the SYSTEM_PROMPT tuple
        close_idx = content.find("\n)", start_idx)
        if close_idx == -1:
            raise ValueError("Could not find closing ')' of SYSTEM_PROMPT in main.py.")

        # Append patch as a new concatenated string before the closing paren
        patch_line  = f'\n    "{patch_text}"'
        new_content = content[:close_idx] + patch_line + content[close_idx:]

        MAIN_PY_PATH.write_text(new_content, encoding="utf-8")
        log.info("Prompt patch '%s' applied.", patch_key)

        # Restart backend to pick up new system prompt
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
        log.info("Topic guardrail written to human_review_queue.jsonl.")
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
        n = len(payload.get("flagged_questions", []))
        log.info("Flagged %d question(s) for human review.", n)
        return {"flagged_count": n, "written_to": str(HUMAN_REVIEW_QUEUE)}

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
    # State
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
                print(f"      ERROR: {r.error}")
    print(f"\n  Execution log  -> {EXECUTION_LOG}")
    print(f"  Review queue   -> {HUMAN_REVIEW_QUEUE}")
    print(f"{'='*60}\n")
