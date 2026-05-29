#!/usr/bin/env python3
"""
Space Mission Q&A Generator — ALL PAIRS edition
For each mission, generates multiple 3-level Q&A chains covering ALL
aspects regardless of how much data is available. The model uses its
space knowledge to fill gaps.

Features:
- Saves after every mission (resume-safe)
- Skips already-processed missions on restart
- Live progress with ETA
"""

import json
import sys
import time
import requests
from pathlib import Path
from datetime import timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
# OLLAMA_URL  = "http://172.16.5.121:11434/api/generate"
OLLAMA_URL  = "http://127.0.0.1:11434/api/generate"
MODEL       = "mistral-small3.1:24b"
OUTPUT_FILE = "processed.json"

# All aspects — every mission gets all of these
ASPECTS = [
    "mission objective and purpose",
    "spacecraft and technology",
    "scientific instruments and experiments",
    "launch date, timeline, and mission phases",
    "orbit and trajectory",
    "scientific discoveries and results",
    "significance and impact on space exploration",
    "challenges and risks",
    "international collaboration and agencies involved",
    "future missions or follow-ups inspired by this mission",
]

SOURCES = [
    {"org": "NASA",  "file": "nasa_mission_scraper/nasa_missions.json"},
    {"org": "ESA",   "file": "ESA_mission_scrapper/esa_missions.json"},
    {"org": "ISRO",  "file": "ISRO_mission_scrapper/isro_missions.json"},
]

# ── Persistence ────────────────────────────────────────────────────────────────
def load_existing(output_path: Path) -> tuple[list, set]:
    if not output_path.exists():
        return [], set()
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        done = {entry["mission_name"] for entry in data}
        print(f"  ↺  Resuming — {len(done)} missions already done, skipping them.")
        return data, done
    except Exception:
        return [], set()


def save(results: list, output_path: Path):
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    tmp.replace(output_path)


# ── Context builder ────────────────────────────────────────────────────────────
def build_context(record: dict) -> str:
    parts = [f"Mission: {record.get('mission_name', 'Unknown')}"]
    structured = record.get("structured", {})
    for key in ("objective", "summary", "launch_date", "end_date",
                "agency", "spacecraft", "orbit", "instruments"):
        val = structured.get(key, "")
        if val:
            parts.append(f"{key.replace('_', ' ').title()}: {val}")
    full_text = record.get("extraction", {}).get("full_text", "")
    if full_text and len(full_text) > 50:
        parts.append(f"Details:\n{full_text[:2000].strip()}")
    return "\n".join(parts)


# ── Ollama call ────────────────────────────────────────────────────────────────
def call_ollama(prompt: str, retries: int = 3) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 1500},
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=240)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            if "<think>" in raw and "</think>" in raw:
                raw = raw[raw.index("</think>") + len("</think>"):].strip()
            return raw
        except Exception as e:
            print(f"      [attempt {attempt}/{retries}] Ollama error: {e}")
            if attempt < retries:
                time.sleep(3)
    return ""


def parse_json(raw: str):
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = raw.find(start_ch)
        end   = raw.rfind(end_ch)
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Generate one 3-level chain for one aspect ─────────────────────────────────
def generate_chain(context: str, aspect: str, mission_name: str) -> dict | None:
    prompt = f"""You are an expert space science educator creating educational Q&A pairs.

Generate a 3-level Q&A chain about the aspect: "{aspect}"
for the space mission described below.

Use the mission data provided. If the data does not explicitly cover this aspect,
use your knowledge of this mission and space exploration in general to provide
accurate, informative answers. Always generate all 3 levels — never skip.

Rules:
- Level 1: A clear foundational question about this aspect and a detailed answer.
- Level 2: A follow-up question that digs deeper into the Level 1 answer, plus its answer.
- Level 3: A follow-up question that digs deeper into the Level 2 answer, plus its answer.
- Questions must be specific to this mission, not generic.
- Answers must be informative and at least 2-3 sentences each.
- Respond ONLY with valid JSON. No preamble, no markdown fences, no explanation.

Required JSON structure:
{{
  "aspect": "{aspect}",
  "level_1": {{"question": "...", "answer": "..."}},
  "level_2": {{"question": "...", "answer": "..."}},
  "level_3": {{"question": "...", "answer": "..."}}
}}

Mission Information:
{context}"""

    raw = call_ollama(prompt)
    if not raw:
        return None

    parsed = parse_json(raw)
    if not isinstance(parsed, dict):
        return None
    if all(k in parsed for k in ("level_1", "level_2", "level_3")):
        parsed["aspect"] = aspect
        return parsed
    return None


# ── Generate ALL chains for a mission ─────────────────────────────────────────
def generate_all_pairs(record: dict, org: str) -> dict:
    mission_name = record.get("mission_name", "Unknown")
    context      = build_context(record)

    chains = []
    for i, aspect in enumerate(ASPECTS, 1):
        print(f"      [{i}/{len(ASPECTS)}] {aspect} …", end=" ", flush=True)

        chain = generate_chain(context, aspect, mission_name)
        if chain:
            chains.append(chain)
            print("✓")
        else:
            # Retry once more with a simplified prompt before giving up
            print("retrying …", end=" ", flush=True)
            chain = generate_chain(context, aspect, mission_name)
            if chain:
                chains.append(chain)
                print("✓")
            else:
                print("✗ (failed after retry)")

    return {
        "organisation": org,
        "mission_name":  mission_name,
        "source_url":    record.get("source_url", ""),
        "scraped_at":    record.get("scraped_at", ""),
        "total_chains":  len(chains),
        "qa_chains":     chains,
    }


# ── ETA helper ────────────────────────────────────────────────────────────────
def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    return str(timedelta(seconds=int(seconds)))[:-3]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    base_dir    = Path(__file__).parent
    output_path = base_dir / OUTPUT_FILE

    print(f"Checking Ollama at {OLLAMA_URL} …")
    try:
        r = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=10)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
        print(f"  Available models: {tags}")
        global MODEL
        if MODEL not in tags:
            matches = [t for t in tags if "deepseek-r1" in t.lower()]
            if matches:
                MODEL = matches[0]
                print(f"  Using model: {MODEL}")
            else:
                print(f"  ⚠  '{MODEL}' not found. Pull it: ollama pull deepseek-r1")
    except Exception as e:
        print(f"  ⚠  Cannot reach Ollama: {e}")
        sys.exit(1)

    all_results, already_done = load_existing(output_path)
    total_processed = 0
    total_failed    = 0
    timing_samples  = []

    for source in SOURCES:
        org  = source["org"]
        path = base_dir / source["file"]

        print(f"\n{'='*60}")
        print(f"Organisation: {org}  |  File: {path}")
        print(f"{'='*60}")

        if not path.exists():
            print(f"  ✗  File not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        records = list(data.values()) if isinstance(data, dict) else data
        print(f"  Found {len(records)} missions.")

        for idx, record in enumerate(records, 1):
            mission_name = record.get("mission_name", f"Mission_{idx}")

            if mission_name in already_done:
                print(f"  [{idx}/{len(records)}] {mission_name} … SKIP (already done)")
                continue

            eta_str = ""
            if timing_samples:
                avg       = sum(timing_samples) / len(timing_samples)
                remaining = len(records) - idx
                eta_str   = f"  ETA ~{fmt_eta(avg * remaining)}"

            print(f"\n  [{idx}/{len(records)}] {mission_name}{eta_str}")

            context = build_context(record)
            if len(context.strip()) < 50:
                print("    SKIP (no content)")
                continue

            t0      = time.time()
            result  = generate_all_pairs(record, org)
            elapsed = time.time() - t0
            timing_samples.append(elapsed)

            if result["total_chains"] > 0:
                all_results.append(result)
                already_done.add(mission_name)
                total_processed += 1
                save(all_results, output_path)
                print(f"    ✓  {result['total_chains']}/{len(ASPECTS)} chains saved  ({elapsed:.1f}s)")
            else:
                total_failed += 1
                print(f"    ✗  No chains generated  ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"Done!  Missions processed: {total_processed}  |  Failed: {total_failed}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
