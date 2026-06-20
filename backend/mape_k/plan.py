"""
SpaceLLM MAPE-K :: Planner Component
=======================================
Responsibility: Read Analyser output (+ recent Monitor events for detail)
                -> decide what to DO about it -> emit an action queue for
                the Executor.

Pipeline position:
    Monitor (monitor_events.jsonl)
        ↓
    Analyser (analysis_report.json)
        ↓
    Planner   ← YOU ARE HERE
        ↓
    Executor (reads plan_actions.jsonl, flips status PENDING -> EXECUTED)

Design notes
------------
- analysis_report.json is a SNAPSHOT (overwritten every Analyser run), not
  a log. The Planner therefore tracks the last-processed `run_id` in its
  own state file so it never silently re-plans (or silently misses) a
  report.
- The report only carries aggregate counts (e.g. incomplete_answer_count).
  For anything that needs qualitative detail (which dodge phrase, which
  question_key keeps failing) the Planner also reads the tail of
  monitor_events.jsonl, sized to exactly the number of events the report
  says it processed (`metadata.events_processed`), so it's looking at the
  same window the Analyser scored.
- Retraining is fully automatic per product decision, but still cooldown-
  gated (PlannerConfig.retrain_cooldown_hours) so a noisy stretch of
  negative feedback can't trigger repeated retrains back-to-back.
- Prompt-patch actions are dedup'd by a stable key (action_type + topic)
  across planning cycles (`prompt_patch_cooldown_cycles`) so an unresolved
  issue doesn't spam an identical patch suggestion every single run.
- TOPIC_GUARDRAIL is honest about classify_topic() currently returning
  "Other" for everything (see monitor.py CHANGELOG #6) - the payload is
  marked `topic_specific: false` rather than pretending otherwise. Once
  real topic tagging exists upstream, this naturally becomes per-topic
  with no changes needed here.

Author: SpaceLLM Project
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
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
log = logging.getLogger("spacellm.planner")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR         = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR         = BASE_DIR / "mape_k"

REPORT_PATH      = MAPE_DIR / "analysis_report.json"
EVENTS_LOG       = MAPE_DIR / "monitor_events.jsonl"
PLAN_REPORT_PATH = MAPE_DIR / "plan_report.json"
PLAN_ACTIONS_LOG = MAPE_DIR / "plan_actions.jsonl"
STATE_FILE       = MAPE_DIR / ".planner_state.json"
FAILED_LOG       = MAPE_DIR / "planner_failed.jsonl"

MAPE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config — tunable thresholds, centralised for ablation
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    # Retrain gating
    retrain_cooldown_hours:          float = 6.0
    max_training_examples:           int   = 500
    min_reference_words_for_training: int  = 6     # filters junk refs like "blank response"

    # Prompt-patch triggers (counts are from THIS cycle's new events,
    # mirroring how the Analyser's signals are scoped)
    incomplete_answer_patch_threshold: int = 2
    hallucination_patch_threshold:     int = 2
    prompt_patch_cooldown_cycles:      int = 3   # don't reissue same patch key for N cycles

    # Topic guardrail / review flagging
    domain_drift_guardrail_threshold: int = 1
    repeated_failure_flag_threshold:  int = 1

# ---------------------------------------------------------------------------
# Priority weights (numeric, used for sorting/escalation by the Executor)
# ---------------------------------------------------------------------------

SEVERITY_PRIORITY = {"CRITICAL": 100, "HIGH": 70, "MEDIUM": 40, "LOW": 10}

ACTION_TYPES = {
    "RETRAIN_ADAPTER",
    "PROMPT_PATCH",
    "TOPIC_GUARDRAIL",
    "FLAG_FOR_REVIEW",
    "NO_ACTION",
}


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PlanAction:
    action_id:     str
    action_type:   str
    priority:      int
    status:        str                     # "PENDING" | "EXECUTED" | "FAILED"
    auto_approved: bool
    created_at:    str
    reasoning:     list[str]
    payload:       dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    plan_id:       str
    timestamp:     str
    source_run_id: str          # analysis_report.json's run_id this plan came from
    model_version: str
    severity:      str
    actions:       list[PlanAction]
    metadata:      dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self._state = self._load_state()
        log.info(
            "Planner initialised. last_processed_run_id=%s last_retrain_at=%s",
            self._state.get("last_processed_run_id"), self._state.get("last_retrain_at"),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Plan | None:
        log.info("=" * 60)
        log.info("Planner cycle starting.")

        report = self._load_report()
        if report is None:
            log.warning("No analysis_report.json found — nothing to plan.")
            return None

        if report.get("run_id") == self._state.get("last_processed_run_id"):
            log.info("No new analysis report since last planning cycle. Skipping.")
            return None

        events = self._load_recent_events(report)
        log.info("Planning against report run_id=%s severity=%s (events_in_window=%d)",
                  report.get("run_id"), report.get("severity"), len(events))

        actions: list[PlanAction] = []

        try:
            retrain_action = self._decide_retrain(report)
            if retrain_action:
                actions.append(retrain_action)
        except Exception as exc:
            self._log_failed("RETRAIN_ADAPTER", exc)

        try:
            actions.extend(self._decide_prompt_patches(report, events))
        except Exception as exc:
            self._log_failed("PROMPT_PATCH", exc)

        try:
            guardrail_action = self._decide_topic_guardrail(report, events)
            if guardrail_action:
                actions.append(guardrail_action)
        except Exception as exc:
            self._log_failed("TOPIC_GUARDRAIL", exc)

        try:
            actions.extend(self._decide_flag_for_review(report, events))
        except Exception as exc:
            self._log_failed("FLAG_FOR_REVIEW", exc)

        if not actions:
            actions.append(self._make_action(
                action_type = "NO_ACTION",
                priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
                reasoning   = ["No triggers fired this cycle.", *report.get("retrain_trigger_reasons", [])],
                payload     = {},
            ))

        plan = Plan(
            plan_id       = str(uuid.uuid4()),
            timestamp     = datetime.now(timezone.utc).isoformat(),
            source_run_id = report.get("run_id", "unknown"),
            model_version = report.get("model_version", "unknown"),
            severity      = report.get("severity", "LOW"),
            actions       = actions,
            metadata      = {"events_in_window": len(events)},
        )

        self._save_plan(plan)

        self._state["last_processed_run_id"] = report.get("run_id")
        if any(a.action_type == "RETRAIN_ADAPTER" for a in actions):
            self._state["last_retrain_at"] = plan.timestamp
        self._save_state()

        log.info("Plan %s written. %d action(s): %s",
                  plan.plan_id[:8], len(actions),
                  Counter(a.action_type for a in actions))
        log.info("=" * 60)
        return plan

    # ------------------------------------------------------------------
    # Decision: retrain
    # ------------------------------------------------------------------

    def _decide_retrain(self, report: dict) -> PlanAction | None:
        if not report.get("should_retrain"):
            return None

        last_retrain_at = self._state.get("last_retrain_at")
        if last_retrain_at:
            elapsed_h = (
                datetime.now(timezone.utc) - datetime.fromisoformat(last_retrain_at)
            ).total_seconds() / 3600
            if elapsed_h < self.config.retrain_cooldown_hours:
                log.info(
                    "Retrain suppressed by cooldown (%.1fh elapsed < %.1fh required).",
                    elapsed_h, self.config.retrain_cooldown_hours,
                )
                return self._make_action(
                    action_type = "FLAG_FOR_REVIEW",
                    priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
                    reasoning   = [
                        "should_retrain=true but suppressed by cooldown — surfaced for "
                        "visibility instead of triggering another retrain.",
                        f"{elapsed_h:.1f}h since last retrain, "
                        f"cooldown is {self.config.retrain_cooldown_hours}h.",
                    ],
                    payload = {"reason": "retrain_cooldown_active", "elapsed_hours": round(elapsed_h, 2)},
                )

        raw_corrections = report.get("corrections_for_training", [])
        cleaned, dropped = [], 0
        for c in raw_corrections:
            ref = (c.get("reference") or "").strip()
            if len(ref.split()) >= self.config.min_reference_words_for_training:
                cleaned.append(c)
            else:
                dropped += 1

        cleaned = cleaned[: self.config.max_training_examples]
        cleaned, enriched = self._backfill_bertscore(cleaned, report)

        reasoning = list(report.get("retrain_trigger_reasons", []))
        reasoning.append(
            f"{dropped} correction(s) excluded from training set — reference too short "
            f"(<{self.config.min_reference_words_for_training} words) to be a usable "
            f"target; likely a feedback comment rather than an actual correction."
        )
        reasoning.append(
            f"{enriched}/{len(cleaned)} training example(s) backfilled with their real "
            f"BERTScore (joined from analysis_report.low_bertscore_pairs by feedback_id) "
            f"— corrections_for_training reads bertscore pre-backfill so it's null there."
        )

        payload = {
            "base_model_version":   report.get("model_version", "unknown"),
            "target_adapter_label": self._next_version_label(report.get("model_version", "unknown")),
            "training_examples":    cleaned,
            "training_examples_count":   len(cleaned),
            "training_examples_dropped": dropped,
        }

        return self._make_action(
            action_type = "RETRAIN_ADAPTER",
            priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10) + 10,
            reasoning   = reasoning,
            payload     = payload,
        )

    def _backfill_bertscore(
        self, examples: list[dict], report: dict,
    ) -> tuple[list[dict], int]:
        """
        corrections_for_training in analysis_report.json reads the feedback
        record's `bertscore` field, which the Analyser only backfills to
        disk AFTER that list is already built — so every entry shows
        bertscore=null even when the Analyser computed a real score for it
        the same run. Join against `low_bertscore_pairs` (which DOES carry
        the freshly-computed bertscore_f1) by feedback_id to recover it,
        so the retrain payload can actually be weighted by how bad each
        original answer was instead of treating every example as unscored.

        Note: low_bertscore_pairs only contains pairs that fell BELOW
        bertscore_low_threshold, so an example with a healthy score won't
        be in it — that's expected, not a miss. We only backfill what's
        available; anything not found stays null (it may simply be a
        higher-scoring example, or its score will land in the next cycle's
        backfilled feedback_log.jsonl).
        """
        score_by_fid = {
            p.get("feedback_id"): p.get("bertscore_f1")
            for p in report.get("low_bertscore_pairs", [])
            if p.get("feedback_id")
        }
        enriched = 0
        out = []
        for ex in examples:
            ex = dict(ex)
            fid = ex.get("feedback_id")
            if ex.get("bertscore") is None and fid in score_by_fid:
                ex["bertscore"] = score_by_fid[fid]
                enriched += 1
            out.append(ex)
        return out, enriched

    # ------------------------------------------------------------------
    # Decision: prompt patches
    # ------------------------------------------------------------------

    def _decide_prompt_patches(self, report: dict, events: list[dict]) -> list[PlanAction]:
        actions: list[PlanAction] = []
        signals = report.get("signals", {})

        if signals.get("incomplete_answer_count", 0) >= self.config.incomplete_answer_patch_threshold:
            patch_key = "incomplete_answer_structural_dodge"
            if self._should_issue_patch(patch_key):
                dodge_events = [e for e in events if e.get("event_type") == "INCOMPLETE_ANSWER"]
                keywords = sorted({
                    e.get("metadata", {}).get("request_keyword", "")
                    for e in dodge_events if e.get("metadata", {}).get("request_keyword")
                })
                actions.append(self._make_action(
                    action_type = "PROMPT_PATCH",
                    priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
                    reasoning   = [
                        f"incomplete_answer_count={signals.get('incomplete_answer_count')} "
                        f">= threshold {self.config.incomplete_answer_patch_threshold}.",
                        f"Observed structural request keywords: {keywords or 'n/a'}.",
                    ],
                    payload = {
                        "patch_key":  patch_key,
                        "target":     "system_prompt",
                        "patch_text": (
                            "When the user asks for a list, table, timeline, schedule, or "
                            "steps, render the structured content directly in the same "
                            "response. Never say 'see below', 'as follows', or 'will be "
                            "covered' without immediately including the full content."
                        ),
                        "observed_keywords": keywords,
                    },
                ))

        if signals.get("hallucination_count", 0) >= self.config.hallucination_patch_threshold:
            patch_key = "hallucination_factual_grounding"
            if self._should_issue_patch(patch_key):
                hall_events = [e for e in events if e.get("event_type") == "POSSIBLE_HALLUCINATION"]
                kw_hits = Counter(
                    kw for e in hall_events
                    for kw in e.get("metadata", {}).get("signal_keywords", [])
                )
                actions.append(self._make_action(
                    action_type = "PROMPT_PATCH",
                    priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
                    reasoning   = [
                        f"hallucination_count={signals.get('hallucination_count')} "
                        f">= threshold {self.config.hallucination_patch_threshold}.",
                        f"Top user-reported signal phrases: {dict(kw_hits.most_common(5)) or 'n/a'}.",
                    ],
                    payload = {
                        "patch_key":  patch_key,
                        "target":     "system_prompt",
                        "patch_text": (
                            "For factual claims about specific dates, mission names, "
                            "partner agencies, or numeric figures, only state details you "
                            "are confident are correct; if uncertain, say so explicitly "
                            "rather than inventing specifics."
                        ),
                    },
                ))

        return actions

    def _should_issue_patch(self, patch_key: str) -> bool:
        """Dedup: don't reissue the same patch suggestion for N consecutive cycles."""
        issued = self._state.setdefault("issued_patches", {})
        cycle  = self._state.get("cycle_count", 0)
        last   = issued.get(patch_key)
        if last is not None and (cycle - last) < self.config.prompt_patch_cooldown_cycles:
            log.info("Patch '%s' suppressed (issued %d cycle(s) ago).", patch_key, cycle - last)
            return False
        issued[patch_key] = cycle
        return True

    # ------------------------------------------------------------------
    # Decision: topic guardrail
    # ------------------------------------------------------------------

    def _decide_topic_guardrail(self, report: dict, events: list[dict]) -> PlanAction | None:
        signals = report.get("signals", {})
        if signals.get("domain_drift_count", 0) < self.config.domain_drift_guardrail_threshold:
            return None

        drift_events = [e for e in events if e.get("event_type") == "DOMAIN_DRIFT"]
        topics = sorted({e.get("topic", "Other") for e in drift_events})

        return self._make_action(
            action_type = "TOPIC_GUARDRAIL",
            priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
            reasoning   = [
                f"domain_drift_count={signals.get('domain_drift_count')} for topics: {topics}.",
                "NOTE: classify_topic() currently returns 'Other' for all input "
                "(see monitor.py CHANGELOG #6), so this drift signal is effectively "
                "global rather than per-domain until real topic tagging is added.",
            ],
            payload = {
                "topic_specific": False,
                "topics_reported": topics,
                "suggested_action": (
                    "Negative-feedback rate is elevated across overall traffic. "
                    "Consider tightening the system prompt's accuracy/uncertainty "
                    "instructions broadly rather than per-topic until topic "
                    "classification is implemented."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Decision: flag recurring failures for human review
    # ------------------------------------------------------------------

    def _decide_flag_for_review(self, report: dict, events: list[dict]) -> list[PlanAction]:
        repeat_events = [e for e in events if e.get("event_type") == "REPEATED_FAILURE"]
        if len(repeat_events) < self.config.repeated_failure_flag_threshold:
            return []

        flagged = [
            {
                "question_key":   e.get("metadata", {}).get("question_key"),
                "negative_count": e.get("metadata", {}).get("negative_count"),
                "feedback_id":    e.get("feedback_id"),
            }
            for e in repeat_events
        ]

        return [self._make_action(
            action_type = "FLAG_FOR_REVIEW",
            priority    = SEVERITY_PRIORITY.get(report.get("severity", "LOW"), 10),
            reasoning   = [f"{len(flagged)} question(s) crossed the repeated-failure threshold."],
            payload     = {"flagged_questions": flagged},
        )]

    # ------------------------------------------------------------------
    # Action / plan construction & persistence
    # ------------------------------------------------------------------

    def _make_action(
        self, action_type: str, priority: int, reasoning: list[str], payload: dict,
    ) -> PlanAction:
        assert action_type in ACTION_TYPES, f"Unknown action_type: {action_type}"
        return PlanAction(
            action_id     = str(uuid.uuid4()),
            action_type   = action_type,
            priority      = priority,
            status        = "PENDING",
            auto_approved = True,
            created_at    = datetime.now(timezone.utc).isoformat(),
            reasoning     = reasoning,
            payload       = payload,
        )

    def _save_plan(self, plan: Plan) -> None:
        _atomic_write_json(PLAN_REPORT_PATH, plan.to_dict())
        try:
            with PLAN_ACTIONS_LOG.open("a", encoding="utf-8") as fh:
                for action in plan.actions:
                    record = action.to_dict()
                    record["plan_id"] = plan.plan_id
                    record["source_run_id"] = plan.source_run_id
                    fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.error("Failed to append to plan_actions.jsonl: %s", exc)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_report(self) -> dict | None:
        if not REPORT_PATH.exists():
            return None
        try:
            return json.loads(REPORT_PATH.read_text())
        except Exception as exc:
            log.error("Could not parse analysis_report.json: %s", exc)
            return None

    def _load_recent_events(self, report: dict) -> list[dict]:
        """
        Read the tail of monitor_events.jsonl, sized to exactly the number
        of events the report says it processed this cycle, so the Planner
        looks at the same window the Analyser scored (events.jsonl is an
        ever-growing append log shared across cycles, not a snapshot).
        """
        n = report.get("metadata", {}).get("events_processed", 0)
        if n <= 0 or not EVENTS_LOG.exists():
            return []
        lines = EVENTS_LOG.read_text().splitlines()
        tail  = lines[-n:] if n <= len(lines) else lines
        events = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                state["cycle_count"] = state.get("cycle_count", 0) + 1
                return state
            except Exception as exc:
                log.warning("Could not load planner state (%s). Starting fresh.", exc)
        return {"cycle_count": 0}

    def _save_state(self) -> None:
        _atomic_write_json(STATE_FILE, self._state)

    def _log_failed(self, action_type: str, exc: Exception) -> None:
        try:
            with FAILED_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "action_type": action_type,
                    "error":       str(exc),
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except OSError as write_exc:
            log.error("Could not write to planner_failed.jsonl: %s", write_exc)
        log.error("Decision step '%s' failed - skipped, logged. (%s)", action_type, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _next_version_label(model_version: str) -> str:
        """SpaceLLM_v1 -> SpaceLLM_v2; falls back to a timestamp suffix if
        the version string doesn't end in a parseable integer."""
        match = re.match(r"^(.*?)(\d+)$", model_version)
        if match:
            prefix, num = match.groups()
            return f"{prefix}{int(num) + 1}"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{model_version}-adapter-{ts}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    planner = Planner()
    plan    = planner.run()

    print(f"\n{'='*60}")
    print(f"  SpaceLLM Planner — Cycle Complete")
    print(f"{'='*60}")
    if plan is None:
        print("  No new analysis report to plan against.")
    else:
        print(f"  Plan ID   : {plan.plan_id}")
        print(f"  Severity  : {plan.severity}")
        print(f"  Actions   : {len(plan.actions)}")
        for a in plan.actions:
            print(f"    [{a.action_type:<16}] priority={a.priority:>3}  status={a.status}")
        print(f"\n  Plan      → {PLAN_REPORT_PATH}")
        print(f"  Actions   → {PLAN_ACTIONS_LOG}")
    print(f"{'='*60}\n")
