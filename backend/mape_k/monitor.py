"""
SpaceLLM MAPE-K :: Monitor Component
=====================================
Responsibility: Observe -> Extract signals -> Generate structured events.
Does NOT retrain, plan, or update the model.

Author : Pratap (AdityaPS / SpaceLLM project)

CHANGELOG (research-readiness pass)
------------------------------------
1. PERSISTED, VERSION-AWARE STATE
   `_topic_feedback_history` and `_negative_question_index` used to live
   only in memory and reset every time `Monitor()` was instantiated (i.e.
   every scheduled cycle). That made REPEATED_FAILURE (needs >=3 hits) and
   DOMAIN_DRIFT (needs >=5 samples) effectively dead code, since each
   cycle only ever "saw" its own small batch of new feedback. State is now
   loaded from / saved to `monitor_state.json` (atomic write), and the
   per-topic history is tagged with `model_version` so stats from a
   superseded adapter don't keep tripping alarms against the current one.

2. POISON-PILL / CRASH SAFETY
   A single malformed record used to be able to kill the whole cycle
   before `_seen_ids` was saved, causing it (and everything before it in
   the batch) to be reprocessed forever. Each record is now processed in
   isolation; failures are logged to `monitor_failed.jsonl` and the record
   is still marked seen. State is also checkpointed every N records
   (`MonitorConfig.checkpoint_interval`), not just at the end of a cycle.

3. IDEMPOTENT DOMAIN_DRIFT
   Previously fired on *every* record once a topic crossed the negative-
   rate threshold. Now tracks active/inactive transitions per topic and
   only re-emits after a cooldown, so the Planner doesn't get spammed.

4. TIGHTER INCOMPLETE_ANSWER DETECTION
   The old single keyword set (list/dates/events/missions/launches/...)
   matched almost any space-domain question, so short-but-correct answers
   were routinely flagged. Keywords are now split into STRONG (timeline,
   table, schedule, steps, milestones - unambiguous structural asks) and
   WEAK (list/dates/events/missions/launches - common nouns that only
   count as a structural request when paired with an explicit
   enumeration cue like "all"/"every"/"how many").

5. Centralised tunables in `MonitorConfig` for threshold sweeps/ablations,
   atomic JSON writes to avoid corrupted state on a killed process, and a
   couple of defensive guards (e.g. `None` conversation content).

Public API (`Monitor`, `run()`, `process_feedback()`, detector method
names) is unchanged so existing Analyzer/Planner code keeps working.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- Logging -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm.monitor")

# -- Paths -----------------------------------------------------------------------

FEEDBACK_LOG   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/feedback_log.jsonl")
EVENTS_LOG     = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/monitor_events.jsonl")
SEEN_IDS_FILE  = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/.seen_feedback_ids.json")
STATE_FILE     = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/.monitor_state.json")
FAILED_LOG     = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/backend/mape_k/monitor_failed.jsonl")

# -- Priority weights ------------------------------------------------------------

PRIORITY = {
    "negative_feedback": 5,
    "human_correction":  10,
    "hallucination":     8,
    "incomplete_answer": 4,
    "repeated_failure":  6,
    "domain_drift":      7,
    "positive_feedback": 1,   # low-weight positive signal
}

# -- Topic keyword map -------------------------------------------------------------

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

# -- Hallucination signal keywords ------------------------------------------------

HALLUCINATION_KEYWORDS = [
    "wrong", "false", "hallucinated", "made up", "incorrect", "not true",
    "never happened", "fabricated", "inaccurate", "misleading", "error",
    "factually wrong", "that's not right", "that is not right",
    "that's wrong", "completely wrong", "totally wrong",
]

# -- Incomplete-answer signal patterns ---------------------------------------------
#
# Split into STRONG (unambiguous structural request) and WEAK (common
# domain noun that only counts when paired with an enumeration cue). See
# CHANGELOG #4 for why this replaced the old single-pattern version.

STRUCTURAL_STRONG_PATTERN = re.compile(
    r"\b(timeline|table|schedule|steps?|milestones?)\b", re.IGNORECASE,
)
STRUCTURAL_WEAK_PATTERN = re.compile(
    r"\b(list|dates?|events?|missions?|launches?)\b", re.IGNORECASE,
)
ENUMERATION_CUE_PATTERN = re.compile(
    r"\b(all|every|each|how many|number of|list of|complete list|full list)\b",
    re.IGNORECASE,
)

INCOMPLETE_DODGE_PATTERNS = re.compile(
    r"\b(as follows|listed below|below is|provided below|following table"
    r"|outlined below|are as follows|will be covered|will include)\b",
    re.IGNORECASE,
)


# -- Config ------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    """
    Centralised, tunable thresholds for the Monitor.

    Keeping these in one place makes threshold sweeps / ablation studies
    (a natural part of evaluating a MAPE-K continual-learning loop) a
    one-line change instead of a code edit.
    """
    repeated_failure_threshold:           int   = 3
    drift_min_samples:                    int   = 5
    drift_neg_rate_threshold:             float = 0.60
    drift_window_size:                    int   = 50   # sliding window per topic
    drift_reemit_cooldown_records:        int   = 5     # re-alert cadence once active
    incomplete_short_word_threshold:      int   = 60
    requires_learning_priority_threshold: int   = 8
    checkpoint_interval:                  int   = 20    # save state every N records


# -- Data models ---------------------------------------------------------------------

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


# -- Small IO helper -------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via a temp file + rename so a killed process never leaves
    a half-written, corrupted state file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


# -- Monitor ----------------------------------------------------------------------------

class Monitor:
    """
    MAPE-K Monitor for SpaceLLM.

    Reads feedback_log.jsonl, processes only new entries, detects signal
    events, scores them by priority, and appends structured MonitorEvent
    records to monitor_events.jsonl.

    This class is intentionally read-only with respect to the model - it
    observes and emits; the Analyzer, Planner, and Executor act on its output.

    Cross-cycle statistics (per-topic feedback history, per-question
    negative counts, domain-drift active/inactive state) are persisted to
    `state_path` so they survive the Monitor being re-instantiated on every
    scheduled run.
    """

    def __init__(
        self,
        feedback_path:   Path = FEEDBACK_LOG,
        events_path:     Path = EVENTS_LOG,
        seen_ids_path:   Path = SEEN_IDS_FILE,
        state_path:      Path = STATE_FILE,
        failed_log_path: Path = FAILED_LOG,
        config: MonitorConfig | None = None,
    ) -> None:
        self.feedback_path   = feedback_path
        self.events_path     = events_path
        self.seen_ids_path   = seen_ids_path
        self.state_path      = state_path
        self.failed_log_path = failed_log_path
        self.config          = config or MonitorConfig()

        self._seen_ids: set[str] = self._load_seen_ids()
        self._load_state()

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "Monitor initialised. seen_ids=%d topics_tracked=%d negative_keys=%d",
            len(self._seen_ids), len(self._topic_history), len(self._negative_question_index),
        )

    # -- Persistence helpers ----------------------------------------------------------

    def _load_seen_ids(self) -> set[str]:
        if self.seen_ids_path.exists():
            try:
                return set(json.loads(self.seen_ids_path.read_text()))
            except Exception as exc:
                log.warning("Could not load seen IDs (%s). Starting fresh.", exc)
        return set()

    def _save_seen_ids(self) -> None:
        _atomic_write_json(self.seen_ids_path, list(self._seen_ids))

    def _load_state(self) -> None:
        """
        Loads the persisted cross-cycle stats. See CHANGELOG #1 - this is
        what makes REPEATED_FAILURE / DOMAIN_DRIFT actually work across
        scheduled cycles instead of resetting every time.
        """
        self._topic_history: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._drift_state: dict[str, dict[str, Any]] = {}
        self._negative_question_index: dict[str, dict[str, Any]] = {}

        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            for topic, hist in raw.get("topic_history", {}).items():
                self._topic_history[topic] = hist
            self._drift_state = raw.get("drift_state", {})
            self._negative_question_index = raw.get("negative_question_index", {})
        except Exception as exc:
            log.warning("Could not load monitor state (%s). Starting fresh.", exc)

    def _save_state(self) -> None:
        payload = {
            "topic_history":          dict(self._topic_history),
            "drift_state":             self._drift_state,
            "negative_question_index": self._negative_question_index,
        }
        _atomic_write_json(self.state_path, payload)

    def _log_failed_record(self, record: FeedbackRecord, exc: Exception) -> None:
        try:
            with self.failed_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "feedback_id": record.feedback_id,
                    "error":       str(exc),
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except OSError as write_exc:
            log.error("Could not write to failed-record log: %s", write_exc)

    # -- Public entry point -------------------------------------------------------------

    def run(self) -> list[MonitorEvent]:
        """
        Full monitoring cycle:
          1. Load new feedback records (sorted by timestamp for deterministic
             repeated-failure / drift ordering).
          2. For each record, detect all applicable events, in isolation -
             one bad record can't take down the rest of the batch.
          3. Persist events, state, and seen-IDs (checkpointed periodically,
             not just at the end, so a crash mid-batch loses minimal progress).
        Returns all events generated in this cycle.
        """
        records = self.load_feedback()
        if not records:
            log.info("No new feedback entries to process.")
            return []
        records.sort(key=lambda r: r.timestamp)

        log.info("Processing %d new feedback records.", len(records))
        all_events: list[MonitorEvent] = []
        failed = 0

        for i, record in enumerate(records, start=1):
            try:
                events = self.process_feedback(record)
            except Exception as exc:
                failed += 1
                log.error(
                    "Record %s failed processing - skipping (poison-pill guard). (%s)",
                    record.feedback_id, exc, exc_info=True,
                )
                self._log_failed_record(record, exc)
                # Mark as seen regardless so this record can never permanently
                # block the pipeline. It's recoverable via monitor_failed.jsonl.
                self._seen_ids.add(record.feedback_id)
                continue

            for event in events:
                self.save_event(event)
                all_events.append(event)
            self._seen_ids.add(record.feedback_id)

            if i % self.config.checkpoint_interval == 0:
                self._save_seen_ids()
                self._save_state()

        self._save_seen_ids()
        self._save_state()

        if all_events:
            breakdown = dict(Counter(e.event_type for e in all_events))
            log.info("Event breakdown this cycle: %s", breakdown)
        log.info("Cycle complete. %d events generated, %d records failed.",
                  len(all_events), failed)
        return all_events

    # -- Feedback loading -----------------------------------------------------------------

    def load_feedback(self) -> list[FeedbackRecord]:
        """Read feedback_log.jsonl and return only unseen, valid records.

        A malformed last line is treated as a partial write-in-progress from
        a concurrently-running FastAPI process: it's logged and skipped
        without being marked seen, so it's naturally picked up once complete
        on the next cycle.
        """
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
                    log.warning("Line %d: malformed JSON - skipping. (%s)", lineno, exc)
                    continue

                fid = data.get("feedback_id")
                if not fid:
                    log.warning("Line %d: missing feedback_id - skipping.", lineno)
                    continue
                if fid in self._seen_ids:
                    continue

                try:
                    records.append(FeedbackRecord.from_dict(data))
                except KeyError as exc:
                    log.warning("Line %d: missing required field %s - skipping.", lineno, exc)
                except Exception as exc:
                    log.warning("Line %d: could not parse record (%s) - skipping.", lineno, exc)

        return records

    # -- Central processing --------------------------------------------------------------

    def process_feedback(self, record: FeedbackRecord) -> list[MonitorEvent]:
        """
        Run all detectors against a single FeedbackRecord.
        Returns a list of MonitorEvents (one per triggered detector).
        """
        events: list[MonitorEvent] = []
        topic = self.classify_topic(record.question + " " + record.candidate)

        # Persisted, version-tagged sliding window for cross-record detectors.
        hist = self._topic_history[topic]
        hist.append({"v": record.model_version, "t": record.feedback_type})
        if len(hist) > self.config.drift_window_size:
            del hist[: len(hist) - self.config.drift_window_size]

        if record.feedback_type == "negative":
            key = self._question_key(record.question)
            self._bump_negative_index(key, record.model_version)

        # -- Positive feedback --------------------------------------------------------
        if record.feedback_type == "positive":
            events.append(self.generate_event(
                record      = record,
                event_type  = "POSITIVE_FEEDBACK",
                topic       = topic,
                base_scores = {"positive_feedback": 1},
                metadata    = {"question_preview": record.question[:120]},
            ))

        # -- Negative feedback ---------------------------------------------------------
        if record.feedback_type == "negative":
            events.append(self.generate_event(
                record      = record,
                event_type  = "NEGATIVE_FEEDBACK",
                topic       = topic,
                base_scores = {"negative_feedback": 1},
                metadata    = {"question_preview": record.question[:120]},
            ))

        # -- Human correction ------------------------------------------------------------
        if record.has_correction and record.reference.strip():
            events.append(self.generate_event(
                record      = record,
                event_type  = "HUMAN_CORRECTION",
                topic       = topic,
                base_scores = {"human_correction": 1},
                metadata    = {
                    "correction_preview":  record.reference[:200],
                    "candidate_preview":   record.candidate[:200],
                },
            ))

        # -- Incomplete answer ------------------------------------------------------------
        incomplete, incomplete_meta = self.detect_incomplete_answer(record)
        if incomplete:
            events.append(self.generate_event(
                record      = record,
                event_type  = "INCOMPLETE_ANSWER",
                topic       = topic,
                base_scores = {"incomplete_answer": 1},
                metadata    = incomplete_meta,
            ))

        # -- Hallucination signal -----------------------------------------------------------
        hallucination, hall_meta = self.detect_hallucination(record)
        if hallucination:
            events.append(self.generate_event(
                record      = record,
                event_type  = "POSSIBLE_HALLUCINATION",
                topic       = topic,
                base_scores = {"hallucination": 1},
                metadata    = hall_meta,
            ))

        # -- Repeated failure -----------------------------------------------------------------
        repeated, repeat_meta = self.detect_repeated_failure(record)
        if repeated:
            events.append(self.generate_event(
                record      = record,
                event_type  = "REPEATED_FAILURE",
                topic       = topic,
                base_scores = {"negative_feedback": 1, "repeated_failure": 1},
                metadata    = repeat_meta,
            ))

        # -- Domain drift -----------------------------------------------------------------------
        drift, drift_meta = self.detect_domain_drift(topic, record.model_version)
        if drift:
            events.append(self.generate_event(
                record      = record,
                event_type  = "DOMAIN_DRIFT",
                topic       = topic,
                base_scores = {"domain_drift": 1},
                metadata    = drift_meta,
            ))

        return events

    # -- Detectors --------------------------------------------------------------------------

    def classify_topic(self, text: str) -> str:
        """
        Keyword-based topic classifier.
        Returns the best-matching category or 'Other'.
        Note: ties between categories with equal hit counts resolve to
        whichever is defined first in TOPIC_KEYWORDS (insertion order) -
        deterministic, but worth knowing if you see a topic look
        "sticky" in ambiguous, multi-topic questions.
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
        Fires when the question is a genuine structural-content request
        AND the answer either dodges it outright or is suspiciously short.

        Two-tier keyword matching (see CHANGELOG #4):
          - STRONG (timeline/table/schedule/steps/milestones) is an
            unambiguous ask for structured content by itself.
          - WEAK (list/dates/events/missions/launches) is extremely common
            in plain space-domain questions, so it only counts as a
            structural request when paired with an explicit enumeration
            cue ("all", "every", "how many", ...). This is what keeps a
            short, correct answer to "what mission was Apollo 11" from
            being wrongly flagged.
        """
        strong_match = STRUCTURAL_STRONG_PATTERN.search(record.question)
        weak_match   = STRUCTURAL_WEAK_PATTERN.search(record.question)
        cue_match    = ENUMERATION_CUE_PATTERN.search(record.question)

        is_structural_request = bool(strong_match) or bool(weak_match and cue_match)
        if not is_structural_request:
            return False, {}

        keyword = (strong_match or weak_match).group()
        answer_match = INCOMPLETE_DODGE_PATTERNS.search(record.candidate)

        if answer_match:
            return True, {
                "request_keyword": keyword,
                "dodge_phrase":    answer_match.group(),
                "answer_length":   len(record.candidate.split()),
                "confidence":      "high",
            }

        word_count = len(record.candidate.split())
        if word_count < self.config.incomplete_short_word_threshold:
            return True, {
                "request_keyword": keyword,
                "reason":          "answer_too_short_for_structural_request",
                "answer_length":   word_count,
                "confidence":      "medium",
            }
        return False, {}

    def detect_hallucination(
        self, record: FeedbackRecord
    ) -> tuple[bool, dict[str, Any]]:
        """
        Scans user feedback text and the full conversation for hallucination
        signal keywords. Heuristic / lexical only - it catches the user
        *saying* the model was wrong, not the model being wrong with no
        pushback. If you want to catch silent hallucinations later, this is
        the natural place to plug in a fact-verification or entity-check
        callable (e.g. cross-referencing named missions/dates against a
        gazetteer, or your existing BERTScore eval) as an additional signal.
        """
        search_corpus = record.reference.lower()
        for turn in record.conversation:
            if turn.get("role") == "user":
                search_corpus += " " + (turn.get("content") or "").lower()

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
        negative feedback 3+ times *under the current model version*. The
        count resets when a new model_version first appears for that
        question, so a complaint that was actually fixed by a later adapter
        doesn't keep counting against the new model forever.
        """
        if record.feedback_type != "negative":
            return False, {}
        key   = self._question_key(record.question)
        entry = self._negative_question_index.get(key, {})
        count = entry.get("count", 0)
        if count >= self.config.repeated_failure_threshold:
            return True, {
                "question_key":     key,
                "negative_count":   count,
                "model_version":    entry.get("model_version"),
            }
        return False, {}

    def detect_domain_drift(
        self, topic: str, model_version: str
    ) -> tuple[bool, dict[str, Any]]:
        """
        Fires when a topic's negative-feedback rate, computed only over
        feedback collected under the *current* model version (within the
        persisted sliding window), exceeds the configured threshold.

        Idempotent: once a topic enters "drift" state it won't re-fire on
        every subsequent record - only after `drift_reemit_cooldown_records`
        more records have passed, so the Planner sees a periodic re-alert
        rather than a flood. Dropping back under the threshold clears the
        active state.
        """
        window = self._topic_history.get(topic, [])
        scoped = [e for e in window if e["v"] == model_version]
        total = len(scoped)

        state = self._drift_state.setdefault(topic, {"active": False, "records_since_emit": 0})

        if total < self.config.drift_min_samples:
            return False, {}

        neg_rate = sum(1 for e in scoped if e["t"] == "negative") / total

        if neg_rate <= self.config.drift_neg_rate_threshold:
            state["active"] = False
            state["records_since_emit"] = 0
            return False, {}

        if not state["active"]:
            state["active"] = True
            state["records_since_emit"] = 0
            fire = True
        else:
            state["records_since_emit"] += 1
            fire = state["records_since_emit"] >= self.config.drift_reemit_cooldown_records
            if fire:
                state["records_since_emit"] = 0

        if not fire:
            return False, {}

        return True, {
            "topic":                        topic,
            "model_version":                model_version,
            "total_feedback_this_version":  total,
            "negative_rate_pct":            round(neg_rate * 100, 1),
            "window_size":                  len(window),
        }

    # -- Priority & event construction -----------------------------------------------------

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
            requires_learning = priority >= self.config.requires_learning_priority_threshold,
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

    # -- Utilities -------------------------------------------------------------------------

    def _bump_negative_index(self, key: str, model_version: str) -> int:
        """
        Increment the negative-feedback count for a normalised question key,
        scoped to the current model_version. If the version has changed
        since this key was last seen, the count restarts at 1 - treating it
        as a fresh issue against the new model rather than carrying over
        stale signal from a superseded adapter.
        """
        entry = self._negative_question_index.get(key)
        if entry is None or entry.get("model_version") != model_version:
            entry = {"count": 0, "model_version": model_version}
        entry["count"] += 1
        self._negative_question_index[key] = entry
        return entry["count"]

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


# -- CLI entry point --------------------------------------------------------------------

if __name__ == "__main__":
    monitor = Monitor()
    events  = monitor.run()
    print(f"\n✓ Monitor cycle complete — {len(events)} event(s) generated.")
    for e in events:
        print(f"  [{e.event_type:<25}] priority={e.priority:>3}  topic={e.topic}")
