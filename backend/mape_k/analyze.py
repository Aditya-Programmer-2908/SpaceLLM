"""
SpaceLLM MAPE-K :: Analyser Component
=======================================
Responsibility: Read Monitor output → compute BERTScore → aggregate signals
                → decide severity → backfill feedback_log → write analysis_report.json

Pipeline position:
    Monitor (monitor_events.jsonl)
        ↓
    Analyser  ← YOU ARE HERE
        ↓
    Planner (analysis_report.json)
        ↓
    Executor

Run:
    python analyse.py

Author: SpaceLLM Project
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter, defaultdict
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
log = logging.getLogger("spacellm.analyser")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend")
MAPE_DIR      = BASE_DIR / "mape_k"

FEEDBACK_LOG  = BASE_DIR / "feedback_log.jsonl"
EVENTS_LOG    = MAPE_DIR / "monitor_events.jsonl"
REPORT_PATH   = MAPE_DIR / "analysis_report.json"
SEEN_IDS_FILE = MAPE_DIR / ".analyser_seen_event_ids.json"
FAILED_LOG    = MAPE_DIR / "analyser_failed.jsonl"

MAPE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thresholds — edit here for ablation studies
# ---------------------------------------------------------------------------

@dataclass
class AnalyserConfig:
    # BERTScore
    bertscore_model:            str   = "distilbert-base-uncased"
    bertscore_lang:             str   = "en"
    bertscore_low_threshold:    float = 0.75   # below this = concern
    bertscore_critical:         float = 0.60   # below this = CRITICAL

    # Negative rate
    neg_rate_medium:            float = 0.20
    neg_rate_high:              float = 0.35
    neg_rate_critical:          float = 0.50

    # Correction count triggers
    corrections_medium:         int   = 10
    corrections_high:           int   = 30
    corrections_critical:       int   = 60

    # Hallucination count triggers
    hallucination_medium:       int   = 3
    hallucination_high:         int   = 8
    hallucination_critical:     int   = 15

    # Repeated failure triggers
    repeated_failure_medium:    int   = 2
    repeated_failure_high:      int   = 5
    repeated_failure_critical:  int   = 10

    # Min corrections needed to recommend retraining
    min_corrections_to_retrain: int   = 5


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

def max_severity(*levels: str) -> str:
    return max(levels, key=lambda s: SEVERITY_RANK.get(s, 0))


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CorrectionPair:
    feedback_id:  str
    question:     str
    candidate:    str
    reference:    str
    bertscore_f1: float | None = None


@dataclass
class SignalSummary:
    total_feedback:           int   = 0
    positive_count:           int   = 0
    negative_count:           int   = 0
    negative_rate:            float = 0.0
    correction_pairs:         int   = 0
    corrections_unused:       int   = 0
    mean_bertscore:           float | None = None
    min_bertscore:            float | None = None
    low_bertscore_count:      int   = 0
    hallucination_count:      int   = 0
    incomplete_answer_count:  int   = 0
    repeated_failure_count:   int   = 0
    domain_drift_count:       int   = 0
    event_type_breakdown:     dict  = field(default_factory=dict)


@dataclass
class AnalysisReport:
    run_id:                   str
    timestamp:                str
    model_version:            str
    severity:                 str
    should_retrain:           bool
    retrain_trigger_reasons:  list[str]
    signals:                  SignalSummary
    corrections_for_training: list[dict]
    low_bertscore_pairs:      list[dict]
    metadata:                 dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = asdict(self.signals)
        return d


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class Analyser:

    def __init__(self, config: AnalyserConfig | None = None) -> None:
        self.config       = config or AnalyserConfig()
        self._seen_ids:   set[str] = self._load_seen_ids()
        self._bert_scorer = None
        log.info("Analyser initialised. seen_event_ids=%d", len(self._seen_ids))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> AnalysisReport:
        log.info("=" * 60)
        log.info("Analyser cycle starting.")

        new_events   = self._load_new_events()
        all_feedback = self._load_all_feedback()
        log.info("New events: %d | Total feedback records: %d",
                 len(new_events), len(all_feedback))

        correction_pairs = self._collect_correction_pairs(new_events, all_feedback)
        log.info("Correction pairs for BERTScore: %d", len(correction_pairs))

        if correction_pairs:
            correction_pairs = self._compute_bertscore(correction_pairs)

        signals  = self._aggregate_signals(new_events, all_feedback, correction_pairs)
        severity, trigger_reasons = self._decide_severity(signals)
        model_version             = self._dominant_model_version(all_feedback)
        corrections_for_training  = self._get_corrections_for_training(all_feedback)

        low_bertscore_pairs = [
            {
                "feedback_id":  p.feedback_id,
                "question":     p.question,
                "candidate":    p.candidate[:300],
                "reference":    p.reference[:300],
                "bertscore_f1": p.bertscore_f1,
            }
            for p in correction_pairs
            if p.bertscore_f1 is not None
            and p.bertscore_f1 < self.config.bertscore_low_threshold
        ]

        report = AnalysisReport(
            run_id                   = str(uuid.uuid4()),
            timestamp                = datetime.now(timezone.utc).isoformat(),
            model_version            = model_version,
            severity                 = severity,
            should_retrain           = (
                severity in ("HIGH", "CRITICAL")
                and signals.corrections_unused >= self.config.min_corrections_to_retrain
            ),
            retrain_trigger_reasons  = trigger_reasons,
            signals                  = signals,
            corrections_for_training = corrections_for_training,
            low_bertscore_pairs      = low_bertscore_pairs,
            metadata                 = {
                "events_processed": len(new_events),
                "bertscore_model":  self.config.bertscore_model,
            },
        )

        if correction_pairs:
            self._backfill_bertscore(correction_pairs)

        _atomic_write_json(REPORT_PATH, report.to_dict())
        log.info("Report written → %s", REPORT_PATH)

        for event in new_events:
            self._seen_ids.add(event["event_id"])
        _atomic_write_json(SEEN_IDS_FILE, list(self._seen_ids))

        log.info("=" * 60)
        log.info("SEVERITY      : %s", severity)
        log.info("SHOULD RETRAIN: %s", report.should_retrain)
        log.info("Neg rate      : %.1f%%", signals.negative_rate * 100)
        log.info("BERTScore     : %s",
                 f"{signals.mean_bertscore:.4f}" if signals.mean_bertscore else "N/A")
        log.info("Corrections   : %d total / %d unused",
                 signals.correction_pairs, signals.corrections_unused)
        log.info("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Load new monitor events
    # ------------------------------------------------------------------

    def _load_new_events(self) -> list[dict]:
        if not EVENTS_LOG.exists():
            log.warning("monitor_events.jsonl not found: %s", EVENTS_LOG)
            return []

        events = []
        with EVENTS_LOG.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Line %d: malformed JSON (%s)", lineno, exc)
                    continue
                event_id = event.get("event_id")
                if not event_id or event_id in self._seen_ids:
                    continue
                events.append(event)

        log.info("Loaded %d new monitor events.", len(events))
        return events

    # ------------------------------------------------------------------
    # Load all feedback records
    # ------------------------------------------------------------------

    def _load_all_feedback(self) -> list[dict]:
        if not FEEDBACK_LOG.exists():
            log.warning("feedback_log.jsonl not found: %s", FEEDBACK_LOG)
            return []

        records = []
        with FEEDBACK_LOG.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Line %d: malformed JSON (%s)", lineno, exc)
        return records

    # ------------------------------------------------------------------
    # Collect correction pairs from HUMAN_CORRECTION events
    # ------------------------------------------------------------------

    def _collect_correction_pairs(
        self, events: list[dict], feedback: list[dict]
    ) -> list[CorrectionPair]:
        fb_index = {r["feedback_id"]: r for r in feedback if "feedback_id" in r}
        pairs    = []

        for event in events:
            if event.get("event_type") != "HUMAN_CORRECTION":
                continue
            fid    = event.get("feedback_id")
            record = fb_index.get(fid) if fid else None
            if not record:
                log.warning("HUMAN_CORRECTION event: feedback_id %s not in log.", fid)
                continue

            candidate = record.get("candidate", "").strip()
            reference = record.get("reference", "").strip()
            if not candidate or not reference:
                continue

            pairs.append(CorrectionPair(
                feedback_id = fid,
                question    = record.get("question", ""),
                candidate   = candidate,
                reference   = reference,
            ))

        return pairs

    # ------------------------------------------------------------------
    # BERTScore computation
    # ------------------------------------------------------------------

    def _get_bert_scorer(self):
        if self._bert_scorer is None:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(
                    model_type            = self.config.bertscore_model,
                    lang                  = self.config.bertscore_lang,
                    rescale_with_baseline = True,
                )
                log.info("BERTScorer loaded: %s", self.config.bertscore_model)
            except ImportError:
                log.error("bert-score not installed. Run: pip install bert-score")
                raise
        return self._bert_scorer

    def _compute_bertscore(
        self, pairs: list[CorrectionPair]
    ) -> list[CorrectionPair]:
        try:
            scorer     = self._get_bert_scorer()
            candidates = [p.candidate for p in pairs]
            references = [p.reference for p in pairs]

            log.info("Computing BERTScore for %d pairs ...", len(pairs))
            _, _, f1_scores = scorer.score(candidates, references)

            for pair, f1 in zip(pairs, f1_scores):
                pair.bertscore_f1 = round(float(f1), 6)

            scores = [p.bertscore_f1 for p in pairs]
            log.info("BERTScore done. mean=%.4f  min=%.4f",
                     sum(scores) / len(scores), min(scores))

        except Exception as exc:
            log.warning("BERTScore failed (%s) — falling back to token F1.", exc)
            for pair in pairs:
                pair.bertscore_f1 = self._token_f1(pair.candidate, pair.reference)

        return pairs

    @staticmethod
    def _token_f1(candidate: str, reference: str) -> float:
        cand = set(candidate.lower().split())
        ref  = set(reference.lower().split())
        if not cand or not ref:
            return 0.0
        overlap   = len(cand & ref)
        precision = overlap / len(cand)
        recall    = overlap / len(ref)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 6)

    # ------------------------------------------------------------------
    # Aggregate signals
    # ------------------------------------------------------------------

    def _aggregate_signals(
        self,
        events:           list[dict],
        feedback:         list[dict],
        correction_pairs: list[CorrectionPair],
    ) -> SignalSummary:
        s = SignalSummary()

        s.total_feedback     = len(feedback)
        s.positive_count     = sum(1 for r in feedback if r.get("feedback_type") == "positive")
        s.negative_count     = sum(1 for r in feedback if r.get("feedback_type") == "negative")
        s.negative_rate      = s.negative_count / s.total_feedback if s.total_feedback else 0.0
        s.correction_pairs   = sum(1 for r in feedback if r.get("has_correction"))
        s.corrections_unused = sum(
            1 for r in feedback
            if r.get("has_correction") and not r.get("used_in_training")
        )

        scored = [p for p in correction_pairs if p.bertscore_f1 is not None]
        if scored:
            scores            = [p.bertscore_f1 for p in scored]
            s.mean_bertscore  = round(sum(scores) / len(scores), 6)
            s.min_bertscore   = round(min(scores), 6)
            s.low_bertscore_count = sum(
                1 for sc in scores if sc < self.config.bertscore_low_threshold
            )

        event_types                = [e.get("event_type", "") for e in events]
        s.event_type_breakdown     = dict(Counter(event_types))
        s.hallucination_count      = event_types.count("POSSIBLE_HALLUCINATION")
        s.incomplete_answer_count  = event_types.count("INCOMPLETE_ANSWER")
        s.repeated_failure_count   = event_types.count("REPEATED_FAILURE")
        s.domain_drift_count       = event_types.count("DOMAIN_DRIFT")

        return s

    # ------------------------------------------------------------------
    # Decide severity
    # ------------------------------------------------------------------

    def _decide_severity(
        self, signals: SignalSummary
    ) -> tuple[str, list[str]]:
        cfg     = self.config
        level   = "LOW"
        reasons = []

        # Negative rate
        if signals.negative_rate >= cfg.neg_rate_critical:
            level = max_severity(level, "CRITICAL")
            reasons.append(f"negative_rate {signals.negative_rate*100:.1f}% >= {cfg.neg_rate_critical*100:.0f}% (CRITICAL)")
        elif signals.negative_rate >= cfg.neg_rate_high:
            level = max_severity(level, "HIGH")
            reasons.append(f"negative_rate {signals.negative_rate*100:.1f}% >= {cfg.neg_rate_high*100:.0f}% (HIGH)")
        elif signals.negative_rate >= cfg.neg_rate_medium:
            level = max_severity(level, "MEDIUM")
            reasons.append(f"negative_rate {signals.negative_rate*100:.1f}% >= {cfg.neg_rate_medium*100:.0f}% (MEDIUM)")

        # BERTScore
        if signals.mean_bertscore is not None:
            if signals.mean_bertscore < cfg.bertscore_critical:
                level = max_severity(level, "CRITICAL")
                reasons.append(f"mean_bertscore {signals.mean_bertscore:.4f} < {cfg.bertscore_critical} (CRITICAL)")
            elif signals.mean_bertscore < cfg.bertscore_low_threshold:
                level = max_severity(level, "HIGH")
                reasons.append(f"mean_bertscore {signals.mean_bertscore:.4f} < {cfg.bertscore_low_threshold} (HIGH)")

        # Unused corrections
        if signals.corrections_unused >= cfg.corrections_critical:
            level = max_severity(level, "CRITICAL")
            reasons.append(f"corrections_unused {signals.corrections_unused} >= {cfg.corrections_critical} (CRITICAL)")
        elif signals.corrections_unused >= cfg.corrections_high:
            level = max_severity(level, "HIGH")
            reasons.append(f"corrections_unused {signals.corrections_unused} >= {cfg.corrections_high} (HIGH)")
        elif signals.corrections_unused >= cfg.corrections_medium:
            level = max_severity(level, "MEDIUM")
            reasons.append(f"corrections_unused {signals.corrections_unused} >= {cfg.corrections_medium} (MEDIUM)")

        # Hallucinations
        if signals.hallucination_count >= cfg.hallucination_critical:
            level = max_severity(level, "CRITICAL")
            reasons.append(f"hallucinations {signals.hallucination_count} >= {cfg.hallucination_critical} (CRITICAL)")
        elif signals.hallucination_count >= cfg.hallucination_high:
            level = max_severity(level, "HIGH")
            reasons.append(f"hallucinations {signals.hallucination_count} >= {cfg.hallucination_high} (HIGH)")
        elif signals.hallucination_count >= cfg.hallucination_medium:
            level = max_severity(level, "MEDIUM")
            reasons.append(f"hallucinations {signals.hallucination_count} >= {cfg.hallucination_medium} (MEDIUM)")

        # Repeated failures
        if signals.repeated_failure_count >= cfg.repeated_failure_critical:
            level = max_severity(level, "CRITICAL")
            reasons.append(f"repeated_failures {signals.repeated_failure_count} >= {cfg.repeated_failure_critical} (CRITICAL)")
        elif signals.repeated_failure_count >= cfg.repeated_failure_high:
            level = max_severity(level, "HIGH")
            reasons.append(f"repeated_failures {signals.repeated_failure_count} >= {cfg.repeated_failure_high} (HIGH)")
        elif signals.repeated_failure_count >= cfg.repeated_failure_medium:
            level = max_severity(level, "MEDIUM")
            reasons.append(f"repeated_failures {signals.repeated_failure_count} >= {cfg.repeated_failure_medium} (MEDIUM)")

        # Domain drift
        if signals.domain_drift_count > 0:
            level = max_severity(level, "MEDIUM")
            reasons.append(f"domain_drift detected ({signals.domain_drift_count} events)")

        if not reasons:
            reasons.append("All signals within normal thresholds.")

        return level, reasons

    # ------------------------------------------------------------------
    # Get corrections for training
    # ------------------------------------------------------------------

    def _get_corrections_for_training(self, feedback: list[dict]) -> list[dict]:
        return [
            {
                "feedback_id":   r.get("feedback_id"),
                "question":      r.get("question", ""),
                "candidate":     r.get("candidate", ""),
                "reference":     r.get("reference", ""),
                "bertscore":     r.get("bertscore"),
                "timestamp":     r.get("timestamp"),
                "model_version": r.get("model_version"),
            }
            for r in feedback
            if r.get("has_correction") and not r.get("used_in_training")
        ]

    # ------------------------------------------------------------------
    # Backfill BERTScore into feedback_log.jsonl
    # ------------------------------------------------------------------

    def _backfill_bertscore(self, pairs: list[CorrectionPair]) -> None:
        if not FEEDBACK_LOG.exists():
            return

        score_map = {
            p.feedback_id: p.bertscore_f1
            for p in pairs if p.bertscore_f1 is not None
        }
        if not score_map:
            return

        updated   = 0
        new_lines = []
        with FEEDBACK_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    fid    = record.get("feedback_id")
                    if fid in score_map and record.get("bertscore") is None:
                        record["bertscore"] = score_map[fid]
                        updated += 1
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(line)

        tmp = FEEDBACK_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(FEEDBACK_LOG)
        log.info("Backfilled BERTScore into %d feedback records.", updated)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_seen_ids(self) -> set[str]:
        if SEEN_IDS_FILE.exists():
            try:
                return set(json.loads(SEEN_IDS_FILE.read_text()))
            except Exception as exc:
                log.warning("Could not load seen event IDs (%s). Starting fresh.", exc)
        return set()

    def _dominant_model_version(self, feedback: list[dict]) -> str:
        versions = [r.get("model_version", "unknown") for r in feedback]
        if not versions:
            return "unknown"
        return Counter(versions).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyser = Analyser()
    report   = analyser.run()

    print(f"\n{'='*60}")
    print(f"  SpaceLLM Analyser — Cycle Complete")
    print(f"{'='*60}")
    print(f"  Severity       : {report.severity}")
    print(f"  Should Retrain : {report.should_retrain}")
    print(f"  Model Version  : {report.model_version}")
    print(f"  Neg Rate       : {report.signals.negative_rate*100:.1f}%")
    print(f"  BERTScore      : {report.signals.mean_bertscore or 'N/A'}")
    print(f"  Unused Corr.   : {report.signals.corrections_unused}")
    print(f"\n  Trigger Reasons:")
    for r in report.retrain_trigger_reasons:
        print(f"    • {r}")
    print(f"\n  Report → {REPORT_PATH}")
    print(f"{'='*60}\n")


