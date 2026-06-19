"""
SpaceLLM MAPE-K :: Monitor Component
=====================================
Responsibility: Observe → Extract signals → Generate structured events.
Does NOT retrain, plan, or update the model.

Author : Pratap (AdityaPS / SpaceLLM project)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.monitor")

# ── Paths ──────────────────────────────────────────────────────────────────────

FEEDBACK_LOG   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/feedback_log.jsonl")
EVENTS_LOG     = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/monitor_events.jsonl")
SEEN_IDS_FILE  = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/.seen_feedback_ids.json")

# ── Priority weights ───────────────────────────────────────────────────────────

PRIORITY = {
    "negative_feedback": 5,
    "human_correction":  10,
    "hallucination":     8,
    "incomplete_answer": 4,
    "repeated_failure":  6,
    "domain_drift":      7,
    "positive_feedback": 1,   # low-weight positive signal
}

# ── Topic keyword map ──────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Apollo Missions":   ["apollo", "saturn v", "neil armstrong", "buzz aldrin",
                          "lunar module", "sea of tranquility", "apollo 11",
                          "apollo 13", "apollo program"],
    "Artemis Program":   ["artemis", "sls", "orion capsule", "gateway",
                          "lunar south pole", "artemis i", "artemis ii"],
    "Satellite Launches":["satellite", "geostationary", "leo orbit", "geo orbit",
                          "launch vehicle", "payload", "cubesat", "smallsat",
                          "pslv", "gslv", "falcon 9", "atlas v", "ariane"],
    "Spacecraft":        ["spacecraft", "probe", "voyager", "cassini", "juno",
                          "new horizons", "pioneer", "messenger", "dawn",
                          "perseverance", "curiosity", "opportunity", "spirit",
                          "ingenuity", "hubble", "james webb", "jwst"],
    "Astronomy":         ["black hole", "neutron star", "galaxy", "nebula",
                          "exoplanet", "dark matter", "dark energy", "pulsar",
                          "quasar", "supernova", "cosmology", "redshift",
                          "gravitational wave", "ligo", "telescope"],
    "ISRO":              ["isro", "chandrayaan", "mangalyaan", "gaganyaan",
                          "pslv", "gslv", "vikram", "pragyan", "aditya-l1",
                          "sriharikota", "indian space"],
    "NASA":              ["nasa", "kennedy space center", "jet propulsion",
                          "jpl", "goddard", "johnson space center", "iss",
                          "space shuttle", "hubble", "artemis", "mars 2020"],
    "SpaceX":            ["spacex", "falcon", "starship", "dragon", "crew dragon",
                          "starlink", "elon musk", "raptor engine", "boca chica",
                          "super heavy"],
    "ESA":               ["esa", "european space agency", "ariane", "rosetta",
                          "huygens", "mars express", "envisat", "sentinel",
                          "gaia", "cheops", "juice"],
}

# ── Hallucination signal keywords ──────────────────────────────────────────────

HALLUCINATION_KEYWORDS = [
    "wrong", "false", "hallucinated", "made up", "incorrect", "not true",
    "never happened", "fabricated", "inaccurate", "misleading", "error",
    "factually wrong", "that's not right", "that is not right",
    "that's wrong", "completely wrong", "totally wrong",
]

# ── Incomplete-answer signal patterns ─────────────────────────────────────────

INCOMPLETE_REQUEST_PATTERNS = re.compile(
    r"\b(timeline|list|dates?|table|schedule|steps?|milestones?|events?|missions?|launches?)\b",
    re.IGNORECASE,
)

INCOMPLETE_DODGE_PATTERNS = re.compile(
    r"\b(as follows|listed below|below is|provided below|following table"
    r"|outlined below|are as follows|will be covered|will include)\b",
    re.IGNORECASE,
)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    feedback_id:      str
    message_id:       str
    feedback_type:    str                  # "positive" | "negative"
    model_version:    str
    timestamp:        str
    question:         str
    candidate:        str                  # LLM answer
    reference:        str                  # human correction (may be empty)
    has_correction:   bool
    used_in_training: bool
    bertscore:        float | None
    conversation:     list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FeedbackRecord":
        return FeedbackRecord(
            feedback_id      = d["feedback_id"],
            message_id       = d["message_id"],
            feedback_type    = d["feedback_type"],
            model_version    = d.get("model_version", "unknown"),
            timestamp        = d["timestamp"],
            question         = d.get("question", ""),
            candidate        = d.get("candidate", ""),
            reference        = d.get("reference", ""),
            has_correction   = bool(d.get("has_correction", False)),
            used_in_training = bool(d.get("used_in_training", False)),
            bertscore        = d.get("bertscore"),
            conversation     = d.get("conversation", []),
        )


@dataclass
class MonitorEvent:
    event_id:          str
    feedback_id:       str
    event_type:        str
    topic:             str
    priority:          int
    requires_learning: bool
    timestamp:         str
    metadata:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Monitor ────────────────────────────────────────────────────────────────────

class Monitor:
    """
    MAPE-K Monitor for SpaceLLM.

    Reads feedback_log.jsonl, processes only new entries, detects signal
    events, scores them by priority, and appends structured MonitorEvent
    records to monitor_events.jsonl.

    This class is intentionally read-only with respect to the model — it
    observes and emits; the Analyzer, Planner, and Executor act on its output.
    """

    def __init__(
        self,
        feedback_path: Path = FEEDBACK_LOG,
        events_path:   Path = EVENTS_LOG,
        seen_ids_path: Path = SEEN_IDS_FILE,
    ) -> None:
        self.feedback_path = feedback_path
        self.events_path   = events_path
        self.seen_ids_path = seen_ids_path

        # In-memory state for repeated-failure and domain-drift detection
        # topic → list of feedback_types seen this run
        self._topic_feedback_history: dict[str, list[str]] = defaultdict(list)

        # question_tokens → count of negative feedback (for repeated-failure)
        self._negative_question_index: dict[str, int] = defaultdict(int)

        self._seen_ids: set[str] = self._load_seen_ids()

        EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        log.info("Monitor initialised. Seen IDs loaded: %d", len(self._seen_ids))

    # ── Persistence helpers ────────────────────────────────────────────────────

    def _load_seen_ids(self) -> set[str]:
        if self.seen_ids_path.exists():
            try:
                return set(json.loads(self.seen_ids_path.read_text()))
            except Exception as exc:
                log.warning("Could not load seen IDs (%s). Starting fresh.", exc)
        return set()

    def _save_seen_ids(self) -> None:
        self.seen_ids_path.write_text(json.dumps(list(self._seen_ids)))

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> list[MonitorEvent]:
        """
        Full monitoring cycle:
          1. Load new feedback records.
          2. For each record, detect all applicable events.
          3. Persist events and mark records as seen.
        Returns all events generated in this cycle.
        """
        records = self.load_feedback()
        if not records:
            log.info("No new feedback entries to process.")
            return []

        log.info("Processing %d new feedback records.", len(records))
        all_events: list[MonitorEvent] = []

        for record in records:
            events = self.process_feedback(record)
            for event in events:
                self.save_event(event)
                all_events.append(event)
            self._seen_ids.add(record.feedback_id)

        self._save_seen_ids()
        log.info("Cycle complete. %d events generated.", len(all_events))
        return all_events

    # ── Feedback loading ───────────────────────────────────────────────────────

    def load_feedback(self) -> list[FeedbackRecord]:
        """Read feedback_log.jsonl and return only unseen, valid records."""
        if not self.feedback_path.exists():
            log.error("Feedback log not found: %s", self.feedback_path)
            return []

        records: list[FeedbackRecord] = []
        with self.feedback_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Line %d: malformed JSON — skipping. (%s)", lineno, exc)
                    continue

                fid = data.get("feedback_id")
                if not fid:
                    log.warning("Line %d: missing feedback_id — skipping.", lineno)
                    continue
                if fid in self._seen_ids:
                    continue

                try:
                    records.append(FeedbackRecord.from_dict(data))
                except KeyError as exc:
                    log.warning("Line %d: missing required field %s — skipping.", lineno, exc)
                except Exception as exc:
                    log.warning("Line %d: could not parse record (%s) — skipping.", lineno, exc)

        return records

    # ── Central processing ─────────────────────────────────────────────────────

    def process_feedback(self, record: FeedbackRecord) -> list[MonitorEvent]:
        """
        Run all detectors against a single FeedbackRecord.
        Returns a list of MonitorEvents (one per triggered detector).
        """
        events: list[MonitorEvent] = []
        topic = self.classify_topic(record.question + " " + record.candidate)

        # Update history for cross-record detectors
        self._topic_feedback_history[topic].append(record.feedback_type)
        if record.feedback_type == "negative":
            key = self._question_key(record.question)
            self._negative_question_index[key] += 1

        # ── Positive feedback ──────────────────────────────────────────────────
        if record.feedback_type == "positive":
            events.append(self.generate_event(
                record      = record,
                event_type  = "POSITIVE_FEEDBACK",
                topic       = topic,
                base_scores = {"positive_feedback": 1},
                metadata    = {"question_preview": record.question[:120]},
            ))

        # ── Negative feedback ──────────────────────────────────────────────────
        if record.feedback_type == "negative":
            events.append(self.generate_event(
                record      = record,
                event_type  = "NEGATIVE_FEEDBACK",
                topic       = topic,
                base_scores = {"negative_feedback": 5},
                metadata    = {"question_preview": record.question[:120]},
            ))

        # ── Human correction ───────────────────────────────────────────────────
        if record.has_correction and record.reference.strip():
            events.append(self.generate_event(
                record      = record,
                event_type  = "HUMAN_CORRECTION",
                topic       = topic,
                base_scores = {"human_correction": 10},
                metadata    = {
                    "correction_preview":  record.reference[:200],
                    "candidate_preview":   record.candidate[:200],
                },
            ))

        # ── Incomplete answer ──────────────────────────────────────────────────
        incomplete, incomplete_meta = self.detect_incomplete_answer(record)
        if incomplete:
            events.append(self.generate_event(
                record      = record,
                event_type  = "INCOMPLETE_ANSWER",
                topic       = topic,
                base_scores = {"incomplete_answer": 4},
                metadata    = incomplete_meta,
            ))

        # ── Hallucination signal ───────────────────────────────────────────────
        hallucination, hall_meta = self.detect_hallucination(record)
        if hallucination:
            events.append(self.generate_event(
                record      = record,
                event_type  = "POSSIBLE_HALLUCINATION",
                topic       = topic,
                base_scores = {"hallucination": 8},
                metadata    = hall_meta,
            ))

        # ── Repeated failure ───────────────────────────────────────────────────
        repeated, repeat_meta = self.detect_repeated_failure(record)
        if repeated:
            events.append(self.generate_event(
                record      = record,
                event_type  = "REPEATED_FAILURE",
                topic       = topic,
                base_scores = {"negative_feedback": 5, "repeated_failure": 6},
                metadata    = repeat_meta,
            ))

        # ── Domain drift ───────────────────────────────────────────────────────
        drift, drift_meta = self.detect_domain_drift(topic)
        if drift:
            events.append(self.generate_event(
                record      = record,
                event_type  = "DOMAIN_DRIFT",
                topic       = topic,
                base_scores = {"domain_drift": 7},
                metadata    = drift_meta,
            ))

        return events

    # ── Detectors ──────────────────────────────────────────────────────────────

    def classify_topic(self, text: str) -> str:
        """
        Keyword-based topic classifier.
        Returns the best-matching category or 'Other'.
        """
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for category, keywords in TOPIC_KEYWORDS.items():
            hit = sum(1 for kw in keywords if kw in text_lower)
            if hit:
                scores[category] = hit
        if not scores:
            return "Other"
        return max(scores, key=lambda c: scores[c])

    def detect_incomplete_answer(
        self, record: FeedbackRecord
    ) -> tuple[bool, dict[str, Any]]:
        """
        Fires when:
          - The user question contains a structural-content request
            (timeline, list, table, dates, schedule …)
          AND
          - The candidate answer contains a dodge phrase that promises
            content without delivering it.
        """
        question_match = INCOMPLETE_REQUEST_PATTERNS.search(record.question)
        answer_match   = INCOMPLETE_DODGE_PATTERNS.search(record.candidate)

        if question_match and answer_match:
            return True, {
                "request_keyword": question_match.group(),
                "dodge_phrase":    answer_match.group(),
                "answer_length":   len(record.candidate.split()),
            }
        # Also fire if answer is suspiciously short for a structural request
        if question_match and len(record.candidate.split()) < 60:
            return True, {
                "request_keyword": question_match.group(),
                "reason":          "answer_too_short_for_structural_request",
                "answer_length":   len(record.candidate.split()),
            }
        return False, {}

    def detect_hallucination(
        self, record: FeedbackRecord
    ) -> tuple[bool, dict[str, Any]]:
        """
        Scans user feedback text and the full conversation for hallucination
        signal keywords.
        """
        # Collect all user-turn text from conversation + the reference correction
        search_corpus = record.reference.lower()
        for turn in record.conversation:
            if turn.get("role") == "user":
                search_corpus += " " + turn.get("content", "").lower()

        found = [kw for kw in HALLUCINATION_KEYWORDS if kw in search_corpus]
        if found:
            return True, {
                "signal_keywords":  found,
                "correction_given": bool(record.reference.strip()),
            }
        return False, {}

    def detect_repeated_failure(
        self, record: FeedbackRecord
    ) -> tuple[bool, dict[str, Any]]:
        """
        Fires when the same semantically similar question has attracted
        negative feedback 3 or more times (based on normalised question tokens).
        """
        if record.feedback_type != "negative":
            return False, {}
        key   = self._question_key(record.question)
        count = self._negative_question_index[key]
        if count >= 3:
            return True, {
                "question_key":     key,
                "negative_count":   count,
            }
        return False, {}

    def detect_domain_drift(
        self, topic: str
    ) -> tuple[bool, dict[str, Any]]:
        """
        Fires when a topic has received more than 5 feedback entries and
        the negative rate exceeds 60 %.
        """
        history = self._topic_feedback_history[topic]
        total   = len(history)
        if total < 5:
            return False, {}
        neg_rate = history.count("negative") / total
        if neg_rate > 0.60:
            return True, {
                "topic":              topic,
                "total_feedback":     total,
                "negative_rate_pct":  round(neg_rate * 100, 1),
            }
        return False, {}

    # ── Priority & event construction ──────────────────────────────────────────

    def calculate_priority(self, base_scores: dict[str, int]) -> int:
        """
        Sum weighted scores from the triggered signal components.
        Priority is unbounded upward so compounded events score higher.
        """
        return sum(PRIORITY.get(k, 0) * v for k, v in base_scores.items())

    def generate_event(
        self,
        record:      FeedbackRecord,
        event_type:  str,
        topic:       str,
        base_scores: dict[str, int],
        metadata:    dict[str, Any],
    ) -> MonitorEvent:
        priority = self.calculate_priority(base_scores)
        return MonitorEvent(
            event_id          = str(uuid.uuid4()),
            feedback_id       = record.feedback_id,
            event_type        = event_type,
            topic             = topic,
            priority          = priority,
            requires_learning = priority >= 8,   # threshold: learning-worthy
            timestamp         = datetime.now(timezone.utc).isoformat(),
            metadata          = {
                "model_version": record.model_version,
                "feedback_type": record.feedback_type,
                **metadata,
            },
        )

    def save_event(self, event: MonitorEvent) -> None:
        """Append a MonitorEvent as a JSON line to monitor_events.jsonl."""
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict()) + "\n")
            log.info("Event saved: [%s] type=%s priority=%d topic=%s",
                     event.event_id[:8], event.event_type,
                     event.priority, event.topic)
        except OSError as exc:
            log.error("Failed to save event %s: %s", event.event_id, exc)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _question_key(question: str) -> str:
        """
        Normalise a question into a comparable key for repeated-failure detection.
        Lowercases, strips punctuation, keeps only alpha-numeric tokens,
        removes common stop words, and takes the 6 most informative tokens.
        """
        stop = {"what", "is", "are", "was", "were", "the", "a", "an",
                "of", "in", "on", "at", "to", "for", "and", "or", "me",
                "tell", "explain", "describe", "how", "why", "did", "does"}
        tokens = re.sub(r"[^a-z0-9\s]", "", question.lower()).split()
        key_tokens = [t for t in tokens if t not in stop][:6]
        return " ".join(sorted(key_tokens))


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    monitor = Monitor()
    events  = monitor.run()
    print(f"\n✓ Monitor cycle complete — {len(events)} event(s) generated.")
    for e in events:
        print(f"  [{e.event_type:<25}] priority={e.priority:>3}  topic={e.topic}")
