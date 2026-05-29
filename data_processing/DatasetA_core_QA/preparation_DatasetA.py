"""
DatasetA Core QA Builder
========================
Transforms SpaceLLM/processed.json into a flat, research-grade Q&A dataset
stored at SpaceLLM/data_processing/DatasetA_core_QA/cleaned_datasetA.json.

No LLM is used — pure deterministic transformation.

Usage:
    python preparation_DatasetA.py                          # auto-detect paths
    python preparation_DatasetA.py --input /path/to/processed.json
    python preparation_DatasetA.py --input /path/to/processed.json --output /path/to/cleaned_datasetA.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert assistant for space missions, "
    "astronomy, and scientific research."
)

LEVEL_TO_DIFFICULTY = {
    "level_1": "basic",
    "level_2": "intermediate",
    "level_3": "advanced",
}


# ── Logging ──────────────────────────────────────────────────────────────────

def log(tag: str, msg: str) -> None:
    tag_map = {
        "INFO":  "\033[94m[INFO] \033[0m",
        "OK":    "\033[92m[OK]   \033[0m",
        "WARN":  "\033[93m[WARN] \033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m",
        "STEP":  "\033[96m[STEP] \033[0m",
    }
    prefix = tag_map.get(tag, f"[{tag}]")
    print(f"{prefix} {msg}", flush=True)


# ── Path resolution ──────────────────────────────────────────────────────────

def resolve_paths(args_input: str | None, args_output: str | None):
    """
    Resolve input and output paths without hardcoding directory depth.

    Strategy for input (processed.json):
      1. Use --input arg if provided.
      2. Walk UP from this script's location looking for processed.json.
      3. Fall back to current working directory.

    Strategy for output (cleaned_datasetA.json):
      1. Use --output arg if provided.
      2. Place it in the same directory as this script.
    """
    script_dir = Path(__file__).resolve().parent
    log("INFO", f"Script location  : {script_dir}")

    # ── Input ────────────────────────────────────────────────────────────────
    if args_input:
        input_file = Path(args_input).resolve()
        log("INFO", f"Input (explicit) : {input_file}")
    else:
        # Walk up the directory tree looking for processed.json
        input_file = None
        search_root = script_dir
        log("INFO", "No --input given — searching upward for processed.json ...")
        for parent in [script_dir, *script_dir.parents]:
            candidate = parent / "processed.json"
            log("INFO", f"  Checking: {candidate}")
            if candidate.exists():
                input_file = candidate
                log("OK",   f"  Found   : {input_file}")
                break

        if input_file is None:
            # Last resort: cwd
            candidate = Path.cwd() / "processed.json"
            log("WARN", f"Not found by walking up. Trying cwd: {candidate}")
            if candidate.exists():
                input_file = candidate
                log("OK", f"Found in cwd: {input_file}")
            else:
                log("ERROR", "processed.json not found anywhere.")
                log("ERROR", "Pass the path explicitly:  python preparation_DatasetA.py --input /path/to/processed.json")
                sys.exit(1)

    # ── Output ───────────────────────────────────────────────────────────────
    if args_output:
        output_file = Path(args_output).resolve()
        log("INFO", f"Output (explicit): {output_file}")
    else:
        output_file = script_dir / "cleaned_datasetA.json"
        log("INFO", f"Output (default) : {output_file}")

    return input_file, output_file


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_source_id(org: str, mission: str) -> str:
    """NASA + '3D Tissue Chips' → 'NASA_3DTissueChips'"""
    clean_org     = re.sub(r"[^A-Za-z0-9]", "", org)
    clean_mission = re.sub(r"[^A-Za-z0-9]", "", mission)
    return f"{clean_org}_{clean_mission}"


def make_chain_id(source_id: str, chain_index: int) -> str:
    """'NASA_3DTissueChips' + 0 → 'NASA_3DTissueChips_001'"""
    return f"{source_id}_{chain_index + 1:03d}"


def make_sample_id(counter: int) -> str:
    """1 → 'SPC_000001'"""
    return f"SPC_{counter:06d}"


# ── Core transform ───────────────────────────────────────────────────────────

def transform(missions: list) -> list:
    records   = []
    sample_no = 1
    skipped_missions = 0
    skipped_qa       = 0

    log("STEP", f"Transforming {len(missions)} mission(s) ...")
    print()

    for m_idx, mission in enumerate(missions):
        org          = (mission.get("organisation") or mission.get("organization", "")).strip()
        mission_name = mission.get("mission_name", "").strip()
        source_url   = mission.get("source_url", "").strip()
        qa_chains    = mission.get("qa_chains", [])

        # ── Mission-level validation ──────────────────────────────────────
        if not org or not mission_name:
            log("WARN", f"Mission #{m_idx + 1} — missing org or mission_name. Skipping entire entry.")
            skipped_missions += 1
            continue

        source_id = make_source_id(org, mission_name)
        log("INFO", f"Mission #{m_idx + 1} | {org} — {mission_name}")
        log("INFO", f"  source_id   : {source_id}")
        log("INFO", f"  source_url  : {source_url or '(none)'}")
        log("INFO", f"  qa_chains   : {len(qa_chains)}")

        mission_records = 0

        for chain_idx, chain in enumerate(qa_chains):
            aspect   = chain.get("aspect", "").strip()
            chain_id = make_chain_id(source_id, chain_idx)

            log("INFO", f"  Chain {chain_idx + 1:02d}/{len(qa_chains):02d} | chain_id={chain_id} | aspect='{aspect}'")

            for level_key, difficulty in LEVEL_TO_DIFFICULTY.items():
                level_data = chain.get(level_key)

                if not level_data:
                    log("WARN", f"    {level_key} missing entirely — skipped.")
                    skipped_qa += 1
                    continue

                question = (level_data.get("question") or "").strip()
                answer   = (level_data.get("answer")   or "").strip()

                if not question or not answer:
                    log("WARN", f"    {level_key} has empty Q or A — skipped.")
                    skipped_qa += 1
                    continue

                sample_id = make_sample_id(sample_no)
                log("OK",   f"    {level_key} → {difficulty:<12} | {sample_id}")

                record = {
                    "sample_id":    sample_id,
                    "source_id":    source_id,
                    "mission_name": mission_name,
                    "organization": org,
                    "aspect":       aspect,
                    "difficulty":   difficulty,
                    "chain_id":     chain_id,
                    "source_url":   source_url,
                    "messages": [
                        {
                            "role":    "developer",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role":    "user",
                            "content": question,
                        },
                        {
                            "role":    "assistant",
                            "content": answer,
                        },
                    ],
                }

                records.append(record)
                sample_no      += 1
                mission_records += 1

        log("INFO", f"  → {mission_records} record(s) produced for this mission.")
        print()

    return records, skipped_missions, skipped_qa


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build DatasetA Core QA from processed.json")
    parser.add_argument("--input",  type=str, default=None, help="Path to processed.json")
    parser.add_argument("--output", type=str, default=None, help="Path to cleaned_datasetA.json")
    args = parser.parse_args()

    print()
    log("STEP", "═══════════════════════════════════════════")
    log("STEP", "   DatasetA Core QA Builder — SpaceLLM    ")
    log("STEP", "═══════════════════════════════════════════")
    print()

    # 1. Resolve paths
    input_file, output_file = resolve_paths(args.input, args.output)
    print()

    # 2. Load input
    log("STEP", f"Loading {input_file} ...")
    file_size = input_file.stat().st_size
    log("INFO", f"File size: {file_size / 1024:.1f} KB")

    with input_file.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    # Normalise to list of missions
    if isinstance(raw, list):
        missions = raw
        log("INFO", f"Top-level structure: list ({len(missions)} item(s))")
    elif isinstance(raw, dict):
        for key in ("missions", "data", "results", "records"):
            if key in raw and isinstance(raw[key], list):
                missions = raw[key]
                log("INFO", f"Top-level structure: dict — using key '{key}' ({len(missions)} item(s))")
                break
        else:
            missions = [raw]
            log("WARN", "Top-level structure: single dict — treating as one mission.")
    else:
        log("ERROR", f"Unexpected JSON type at root: {type(raw)}")
        sys.exit(1)

    log("OK", f"{len(missions)} mission(s) loaded.")
    print()

    # 3. Transform
    records, skipped_missions, skipped_qa = transform(missions)

    # 4. Write output
    log("STEP", f"Writing output to {output_file} ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    out_size = output_file.stat().st_size
    log("OK", f"File written — {out_size / 1024:.1f} KB")
    print()

    # 5. Summary report
    log("STEP", "═══════════════ SUMMARY ═══════════════════")
    log("INFO", f"Missions processed   : {len(missions) - skipped_missions}")
    log("INFO", f"Missions skipped     : {skipped_missions}")
    log("INFO", f"Q&A pairs skipped    : {skipped_qa}")
    log("INFO", f"Total records output : {len(records)}")
    print()

    difficulty_counts = Counter(r["difficulty"]   for r in records)
    org_counts        = Counter(r["organization"] for r in records)
    aspect_counts     = Counter(r["aspect"]       for r in records)

    log("INFO", "Difficulty breakdown:")
    for d in ("basic", "intermediate", "advanced"):
        log("INFO", f"  {d:<15}: {difficulty_counts.get(d, 0):>5}")
    print()

    log("INFO", "Records per organisation:")
    for org, cnt in org_counts.most_common():
        log("INFO", f"  {org:<25}: {cnt:>5}")
    print()

    log("INFO", "Records per aspect:")
    for aspect, cnt in aspect_counts.most_common():
        log("INFO", f"  {aspect:<40}: {cnt:>4}")
    print()

    log("OK", f"Output saved → {output_file}")
    log("STEP", "═══════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()