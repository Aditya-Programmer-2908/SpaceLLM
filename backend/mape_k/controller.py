"""
SpaceLLM MAPE-K :: Controller
================================
Orchestrates the full loop:  Monitor → Analyser → Planner → Executor

Run once (single cycle):
    python controller.py

Run on a schedule (e.g. every 30 minutes):
    python controller.py --interval 1800

The controller also reads frontend_patch.md and topic_guardrail.json to
expose the *current effective system prompt* — useful for the FastAPI
server to call get_system_prompt() at startup without duplicating the
patch-file parsing logic.

Author: SpaceLLM Project
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.controller")

# Base path — same across all MAPE-K modules
BASE_DIR            = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR            = BASE_DIR / "mape_k"
FRONTEND_PATCH_FILE = BASE_DIR / "frontend_patch.md"
TOPIC_GUARDRAIL_FILE = MAPE_DIR / "topic_guardrail.json"


# ---------------------------------------------------------------------------
# System-prompt assembly  (used by FastAPI / inference core)
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert assistant specialising in space exploration, "
    "satellite systems, launch vehicles, and astronomy. "
    "Answer questions accurately, cite uncertainty when relevant, and provide "
    "structured content (tables, timelines, step-by-step lists) when the user "
    "asks for it."
)


def _parse_patch_blocks(patch_file: Path) -> dict[str, str]:
    """
    Parse frontend_patch.md for PATCH_START / PATCH_END sentinel blocks.
    Returns {patch_key: patch_text} — last block wins for each key (dedup).
    """
    if not patch_file.exists():
        return {}

    patches: dict[str, str] = {}
    current_key:  str | None = None
    current_lines: list[str] = []

    for line in patch_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("<!-- PATCH_START"):
            # e.g. <!-- PATCH_START patch_key=incomplete_answer_structural_dodge applied_at=... -->
            for part in line.split():
                if part.startswith("patch_key="):
                    current_key = part.split("=", 1)[1].rstrip(" -->")
            current_lines = []
        elif line.startswith("<!-- PATCH_END") and current_key:
            patches[current_key] = "\n".join(current_lines).strip()
            current_key = None
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    return patches


def get_system_prompt() -> str:
    """
    Assemble the live system prompt by layering active patches over the base.
    Call this from the FastAPI server at startup (and optionally poll it).
    """
    patches = _parse_patch_blocks(FRONTEND_PATCH_FILE)
    parts   = [BASE_SYSTEM_PROMPT]

    if patches:
        parts.append("\n\n# Active behaviour patches\n")
        for key, text in patches.items():
            parts.append(f"[{key}] {text}")
        log.info("System prompt assembled with %d active patch(es): %s",
                 len(patches), list(patches.keys()))
    else:
        log.info("No active patches — using base system prompt.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:

    def __init__(self) -> None:
        # Lazy imports so each component's own logging fires after basicConfig
        from monitor import Monitor
        from analyze import Analyser
        from plan    import Planner
        from execute import Executor

        self.monitor  = Monitor()
        self.analyser = Analyser()
        self.planner  = Planner()
        self.executor = Executor()

    def run_cycle(self) -> dict[str, Any]:
        """Execute one full MAPE-K cycle and return a summary dict."""
        started = datetime.now(timezone.utc)
        log.info("╔══════════════════════════════════╗")
        log.info("║  MAPE-K Cycle  %s  ║", started.strftime("%Y-%m-%d %H:%M:%S UTC"))
        log.info("╚══════════════════════════════════╝")

        # ── Monitor ───────────────────────────────────────────────────────────
        log.info("── MONITOR ──")
        try:
            monitor_events = self.monitor.run()
        except Exception as exc:
            log.error("Monitor failed: %s", exc, exc_info=True)
            monitor_events = []

        # ── Analyser ──────────────────────────────────────────────────────────
        log.info("── ANALYSER ──")
        try:
            report = self.analyser.run()
        except Exception as exc:
            log.error("Analyser failed: %s", exc, exc_info=True)
            report = None

        # ── Planner ───────────────────────────────────────────────────────────
        log.info("── PLANNER ──")
        try:
            plan = self.planner.run()
        except Exception as exc:
            log.error("Planner failed: %s", exc, exc_info=True)
            plan = None

        # ── Executor ──────────────────────────────────────────────────────────
        log.info("── EXECUTOR ──")
        try:
            exec_records = self.executor.run()
        except Exception as exc:
            log.error("Executor failed: %s", exc, exc_info=True)
            exec_records = []

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()

        summary = {
            "timestamp":        started.isoformat(),
            "elapsed_s":        round(elapsed, 1),
            "monitor_events":   len(monitor_events),
            "severity":         report.severity      if report else "N/A",
            "should_retrain":   report.should_retrain if report else False,
            "plan_actions":     len(plan.actions)    if plan   else 0,
            "exec_executed":    sum(1 for r in exec_records if r.status == "EXECUTED"),
            "exec_failed":      sum(1 for r in exec_records if r.status == "FAILED"),
        }

        log.info("Cycle complete in %.1fs. Summary: %s", elapsed, summary)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SpaceLLM MAPE-K Controller")
    parser.add_argument(
        "--interval", type=int, default=0,
        help="Seconds between cycles. 0 = run once (default).",
    )
    args = parser.parse_args()

    controller = Controller()

    if args.interval <= 0:
        controller.run_cycle()
    else:
        log.info("Scheduler mode: running every %ds. Ctrl-C to stop.", args.interval)
        while True:
            try:
                controller.run_cycle()
            except KeyboardInterrupt:
                log.info("Interrupted — shutting down.")
                break
            except Exception as exc:
                log.error("Unexpected error in cycle: %s", exc, exc_info=True)
            log.info("Sleeping %ds until next cycle...", args.interval)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
