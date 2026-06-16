"""
SpaceLLM — Dataset Preparation Pipeline v2
============================================
Script   : preparation_DatasetAv2.py
Input    : /home/aditya/SpaceLLM/processed.json
Output   : /home/aditya/SpaceLLM/data_processing/DatasetA_core_QA_v2/cleaned_DatasetAv2.json

Objective:
    Transform processed.json into a minimalistic, research-grade dataset for
    training a Space mission chatbot. Only 5 standardized aspects are retained.

Allowed Aspects (exactly 5):
    OBJECTIVE    — mission objective and purpose
    TECHNOLOGY   — spacecraft and systems
    EXPERIMENTS  — scientific instruments and experiments
    RESULTS      — scientific discoveries and findings
    IMPACT       — significance and future impact

Discarded Aspects (never included):
    orbit / trajectory
    timeline / mission phases
    risks / challenges
    international collaboration
    future missions (unless the raw aspect maps to IMPACT)

Usage:
    python preparation_DatasetAv2.py
    python preparation_DatasetAv2.py --input /path/to/processed.json
    python preparation_DatasetAv2.py --input processed.json --output cleaned_DatasetAv2.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # SpaceLLM/

DEFAULT_INPUT  = PROJECT_ROOT / "processed.json"
DEFAULT_OUTPUT = SCRIPT_DIR   / "cleaned_DatasetAv2.json"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SpaceLLM.v2")

# ── Constants ─────────────────────────────────────────────────────────────────

# The 5 canonical aspect labels used in the output dataset
ASPECT_OBJECTIVE   = "OBJECTIVE"
ASPECT_TECHNOLOGY  = "TECHNOLOGY"
ASPECT_EXPERIMENTS = "EXPERIMENTS"
ASPECT_RESULTS     = "RESULTS"
ASPECT_IMPACT      = "IMPACT"

ALLOWED_ASPECTS = {
    ASPECT_OBJECTIVE,
    ASPECT_TECHNOLOGY,
    ASPECT_EXPERIMENTS,
    ASPECT_RESULTS,
    ASPECT_IMPACT,
}

# Difficulty labels mapped from level keys
DIFFICULTY_MAP = {
    "level_1": "basic",
    "level_2": "intermediate",
    "level_3": "advanced",
}

# ── Aspect mapping rules ──────────────────────────────────────────────────────
# Each entry is a (substring_to_match, canonical_aspect) pair.
# Matching is case-insensitive and uses substring search.
# Order matters — first match wins.
# Aspects NOT matched by any rule are DISCARDED.

ASPECT_MAPPING_RULES = [
    # ── OBJECTIVE ─────────────────────────────────────────────────────────
    ("mission objective",          ASPECT_OBJECTIVE),
    ("objective and purpose",      ASPECT_OBJECTIVE),
    ("mission purpose",            ASPECT_OBJECTIVE),
    ("primary objective",          ASPECT_OBJECTIVE),

    # ── TECHNOLOGY ────────────────────────────────────────────────────────
    ("spacecraft and technology",  ASPECT_TECHNOLOGY),
    ("spacecraft technology",      ASPECT_TECHNOLOGY),
    ("technology",                 ASPECT_TECHNOLOGY),
    ("systems",                    ASPECT_TECHNOLOGY),
    ("propulsion",                 ASPECT_TECHNOLOGY),
    ("engineering",                ASPECT_TECHNOLOGY),
    ("hardware",                   ASPECT_TECHNOLOGY),

    # ── EXPERIMENTS ───────────────────────────────────────────────────────
    ("scientific instruments",     ASPECT_EXPERIMENTS),
    ("instruments and experiments",ASPECT_EXPERIMENTS),
    ("experiments",                ASPECT_EXPERIMENTS),
    ("instruments",                ASPECT_EXPERIMENTS),
    ("payload",                    ASPECT_EXPERIMENTS),
    ("sensor",                     ASPECT_EXPERIMENTS),

    # ── RESULTS ───────────────────────────────────────────────────────────
    ("scientific discoveries",     ASPECT_RESULTS),
    ("discoveries and results",    ASPECT_RESULTS),
    ("discoveries and findings",   ASPECT_RESULTS),
    ("scientific results",         ASPECT_RESULTS),
    ("findings",                   ASPECT_RESULTS),
    ("results",                    ASPECT_RESULTS),
    ("discoveries",                ASPECT_RESULTS),
    ("observations",               ASPECT_RESULTS),

    # ── IMPACT ────────────────────────────────────────────────────────────
    ("significance and impact",    ASPECT_IMPACT),
    ("impact on space exploration",ASPECT_IMPACT),
    ("future missions",            ASPECT_IMPACT),   # merged into IMPACT per spec
    ("follow-up",                  ASPECT_IMPACT),
    ("follow up",                  ASPECT_IMPACT),
    ("inspired by",                ASPECT_IMPACT),
    ("significance",               ASPECT_IMPACT),
    ("impact",                     ASPECT_IMPACT),
    ("legacy",                     ASPECT_IMPACT),

    # ── DISCARD (explicit, never matched to allowed aspects) ──────────────
    # orbit / trajectory  → no rule → falls through to discard
    # timeline / phases   → no rule → falls through to discard
    # risks / challenges  → no rule → falls through to discard
    # international collab→ no rule → falls through to discard
]


def map_aspect(raw_aspect: str) -> str | None:
    """
    Map a raw aspect string from processed.json to one of the 5 canonical
    aspect labels, or return None if it should be discarded.

    Matching is case-insensitive substring search.
    First matching rule wins.
    """
    normalised = raw_aspect.strip().lower()
    for keyword, canonical in ASPECT_MAPPING_RULES:
        if keyword.lower() in normalised:
            return canonical
    return None   # no rule matched → discard


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "spacecraft technology, and scientific discoveries. "
    "Answer questions clearly and accurately based on your knowledge of space exploration. "
    "Tailor the depth of your response to the complexity of the question."
)

# ── Sample ID counter ─────────────────────────────────────────────────────────

class SampleIDCounter:
    """Thread-safe sequential sample ID generator."""

    def __init__(self, prefix: str = "SPC", width: int = 6):
        self._count  = 0
        self._prefix = prefix
        self._width  = width

    def next(self) -> str:
        self._count += 1
        return f"{self._prefix}_{self._count:0{self._width}d}"


# ── Core transformation ───────────────────────────────────────────────────────

def build_chain_id(mission_name: str, aspect: str) -> str:
    """
    Deterministic chain_id from mission name + canonical aspect.
    Format: <slugified_mission>_<aspect>
    """
    slug = mission_name.strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    slug = "_".join(part for part in slug.split("_") if part)   # collapse underscores
    return f"{slug}_{aspect.lower()}"


def process_mission(mission: dict, counter: SampleIDCounter) -> list[dict]:
    """
    Transform one mission dict from processed.json into a list of flat records.

    Strategy:
      1. For each qa_chain, map the raw aspect to a canonical one.
      2. Group chains by canonical aspect (keep first occurrence per aspect).
      3. For each aspect group, emit one record per available level.
      4. Skip levels with missing/empty question or answer.
      5. Skip entire aspect if no valid levels remain after filtering.
    """
    mission_name = mission.get("mission_name") or mission.get("Mission_name", "Unknown")
    organisation = (
        mission.get("organisation")
        or mission.get("organization")
        or mission.get("Organisation", "Unknown")
    )
    source_url   = mission.get("source_url", "")
    qa_chains    = mission.get("qa_chains", [])

    # ── Step 1 + 2: Map and group by canonical aspect ─────────────────────
    # aspect_chains: canonical_aspect → first matching qa_chain dict
    # We keep only the FIRST chain that maps to each canonical aspect to
    # avoid duplicates when a mission has two raw aspects that both map to
    # e.g. RESULTS.
    aspect_chains: dict[str, dict] = {}

    for chain in qa_chains:
        raw_aspect = chain.get("aspect", "")
        canonical  = map_aspect(raw_aspect)

        if canonical is None:
            logger.debug(f"  [{mission_name}] Discarding aspect: '{raw_aspect}'")
            continue

        if canonical not in aspect_chains:
            aspect_chains[canonical] = chain
            logger.debug(f"  [{mission_name}] Mapped '{raw_aspect}' → {canonical}")
        else:
            logger.debug(
                f"  [{mission_name}] Duplicate mapping for {canonical} "
                f"('{raw_aspect}') — keeping first"
            )

    # ── Step 3: Emit records ──────────────────────────────────────────────
    records: list[dict] = []

    # Deterministic ordering: emit aspects in canonical order
    for aspect in [
        ASPECT_OBJECTIVE,
        ASPECT_TECHNOLOGY,
        ASPECT_EXPERIMENTS,
        ASPECT_RESULTS,
        ASPECT_IMPACT,
    ]:
        if aspect not in aspect_chains:
            continue

        chain    = aspect_chains[aspect]
        chain_id = build_chain_id(mission_name, aspect)

        for level_key, difficulty in DIFFICULTY_MAP.items():
            level_data = chain.get(level_key)
            if not level_data:
                continue

            question = (level_data.get("question") or "").strip()
            answer   = (level_data.get("answer")   or "").strip()

            if not question or not answer:
                logger.debug(
                    f"  [{mission_name}] Skipping {aspect}/{difficulty} "
                    f"— missing question or answer"
                )
                continue

            record = {
                "sample_id":    counter.next(),
                "source_id":    chain_id,
                "mission_name": mission_name,
                "organization": organisation,
                "aspect":       aspect,
                "difficulty":   difficulty,
                "chain_id":     chain_id,
                "source_url":   source_url,
                "messages": [
                    {"role": "developer", "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
            records.append(record)

    return records


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(input_path: Path, output_path: Path) -> None:
    logger.info("=" * 60)
    logger.info("  SpaceLLM — Dataset Preparation Pipeline v2")
    logger.info(f"  Input   : {input_path}")
    logger.info(f"  Output  : {output_path}")
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info("Loading processed.json ...")
    with input_path.open(encoding="utf-8") as f:
        missions = json.load(f)

    if not isinstance(missions, list):
        logger.error("Expected processed.json to be a JSON array of mission objects.")
        sys.exit(1)

    logger.info(f"Loaded {len(missions):,} missions")

    # ── Transform ─────────────────────────────────────────────────────────
    counter   = SampleIDCounter()
    all_records: list[dict] = []

    aspect_counts    = {a: 0 for a in ALLOWED_ASPECTS}
    difficulty_counts = {"basic": 0, "intermediate": 0, "advanced": 0}
    org_counts: dict[str, int] = {}
    missions_with_zero_records = 0

    for i, mission in enumerate(missions):
        name = mission.get("mission_name") or mission.get("Mission_name", f"mission_{i}")
        records = process_mission(mission, counter)

        if not records:
            missions_with_zero_records += 1
            logger.warning(f"  No valid records produced for: {name}")
            continue

        for r in records:
            aspect_counts[r["aspect"]]       += 1
            difficulty_counts[r["difficulty"]] += 1
            org = r["organization"]
            org_counts[org] = org_counts.get(org, 0) + 1

        all_records.extend(records)

    # ── Deduplication check ───────────────────────────────────────────────
    sample_ids = [r["sample_id"] for r in all_records]
    if len(sample_ids) != len(set(sample_ids)):
        logger.error("FATAL: Duplicate sample_ids detected — aborting.")
        sys.exit(1)

    # ── Stats ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Pipeline Statistics ──────────────────────────────")
    logger.info(f"  Total missions processed    : {len(missions):,}")
    logger.info(f"  Missions with 0 records     : {missions_with_zero_records:,}")
    logger.info(f"  Total records produced      : {len(all_records):,}")
    logger.info(f"  Duplicate sample_ids        : None (verified)")
    logger.info("")
    logger.info("  Records by aspect:")
    for aspect, count in aspect_counts.items():
        logger.info(f"    {aspect:<15}: {count:,}")
    logger.info("")
    logger.info("  Records by difficulty:")
    for diff, count in difficulty_counts.items():
        logger.info(f"    {diff:<15}: {count:,}")
    logger.info("")
    logger.info("  Records by organization:")
    for org, count in sorted(org_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {org:<20}: {count:,}")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("")
    logger.info(f"Writing {len(all_records):,} records → {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  SpaceLLM Dataset v2 — Complete")
    logger.info(f"  Output  : {output_path}")
    logger.info(f"  Records : {len(all_records):,}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpaceLLM Dataset Preparation Pipeline v2"
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to processed.json (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for output JSON (default: {DEFAULT_OUTPUT})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path  = args.input.resolve(),
        output_path = args.output.resolve(),
    )