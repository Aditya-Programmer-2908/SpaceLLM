"""
DatasetA Train / Validation / Test Splitter
============================================
Input  : cleaned_datasetA.json  (same directory as this script)
Output : train.jsonl, validation.jsonl, test.jsonl  (same directory)

Split unit : chain_id  — all 3 records in a chain always stay together
Ratios     : train=85%  validation=10%  test=5%

Guarantees
- No chain is ever broken across splits
- Every mission appears in train
- Organization distribution is preserved proportionally
- Difficulty balance is automatic (every chain carries basic+intermediate+advanced)
- Deterministic — same input always produces the same output
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
TEST_RATIO  = 0.05

SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_FILE  = SCRIPT_DIR / "cleaned_datasetA.json"
TRAIN_FILE  = SCRIPT_DIR / "train.jsonl"
VAL_FILE    = SCRIPT_DIR / "validation.jsonl"
TEST_FILE   = SCRIPT_DIR / "test.jsonl"

# ── Logging ───────────────────────────────────────────────────────────────────

def log(tag, msg):
    colors = {"INFO": "\033[94m", "OK": "\033[92m", "WARN": "\033[93m",
              "ERROR": "\033[91m", "STEP": "\033[96m"}
    print(f"{colors.get(tag,'')  }[{tag}]\033[0m {msg}", flush=True)

def banner(title):
    print()
    log("STEP", "─" * 55)
    log("STEP", f"  {title}")
    log("STEP", "─" * 55)
    print()

# ── Load ──────────────────────────────────────────────────────────────────────

banner("Step 1 — Load cleaned_datasetA.json")

if not INPUT_FILE.exists():
    log("ERROR", f"File not found: {INPUT_FILE}")
    sys.exit(1)

with INPUT_FILE.open(encoding="utf-8") as f:
    records = json.load(f)

log("OK",   f"Loaded {len(records)} records from {INPUT_FILE}")

# ── Group by chain_id ─────────────────────────────────────────────────────────

banner("Step 2 — Group records by chain_id")

chains = defaultdict(list)
for r in records:
    chains[r["chain_id"]].append(r)

chains = dict(chains)
log("OK", f"Total chains : {len(chains)}")

# Report chain completeness
size_counts = defaultdict(int)
for recs in chains.values():
    size_counts[len(recs)] += 1
for size, count in sorted(size_counts.items()):
    flag = "" if size == 3 else "  ← INCOMPLETE"
    log("INFO", f"  {size} record(s)/chain : {count} chain(s){flag}")

# ── Measure distributions ─────────────────────────────────────────────────────

banner("Step 3 — Measure distributions")

org_chains     = defaultdict(list)   # org     -> [chain_ids]
mission_chains = defaultdict(list)   # mission -> [chain_ids]

for cid, recs in chains.items():
    r = recs[0]
    org_chains[r["organization"]].append(cid)
    mission_chains[r["mission_name"]].append(cid)

log("INFO", f"Organizations : {len(org_chains)}")
for org, cids in sorted(org_chains.items()):
    log("INFO", f"  {org:<30} {len(cids):>4} chains")

print()
log("INFO", f"Missions : {len(mission_chains)}")
for m, cids in sorted(mission_chains.items()):
    log("INFO", f"  {m:<40} {len(cids):>4} chains")

# ── Assign chains to splits ───────────────────────────────────────────────────

banner("Step 4 — Assign chains to splits")

assignments = {}   # chain_id -> "train" | "validation" | "test"

# Guarantee: first chain of every mission goes to train
for mission, cids in mission_chains.items():
    # Sort deterministically so the same chain is always picked
    first = sorted(cids)[0]
    assignments[first] = "train"

log("INFO", f"Reserved {len(assignments)} chain(s) for train (1 per mission)")

# Distribute remaining chains per org proportionally
for org in sorted(org_chains.keys()):
    remaining = sorted([c for c in org_chains[org] if c not in assignments])
    n = len(remaining)
    if n == 0:
        continue

    n_train = round(n * TRAIN_RATIO)
    n_val   = round(n * VAL_RATIO)
    n_test  = n - n_train - n_val    # remainder goes to train

    # Clamp to avoid negatives on tiny orgs
    n_test  = max(0, n_test)
    n_val   = max(0, n_val)
    n_train = n - n_val - n_test

    log("INFO", f"  {org:<28}  remaining={n:>3}  →  train={n_train}  val={n_val}  test={n_test}")

    for i, cid in enumerate(remaining):
        if i < n_train:
            assignments[cid] = "train"
        elif i < n_train + n_val:
            assignments[cid] = "validation"
        else:
            assignments[cid] = "test"

# Safety: assign anything missed to train
for cid in chains:
    if cid not in assignments:
        log("WARN", f"Unassigned chain {cid} → defaulting to train")
        assignments[cid] = "train"

# Summary
split_chain_counts = defaultdict(int)
for split in assignments.values():
    split_chain_counts[split] += 1

print()
log("INFO", "Chain assignment result:")
for split in ("train", "validation", "test"):
    n   = split_chain_counts[split]
    pct = 100 * n / len(chains)
    log("INFO", f"  {split:<12} {n:>4} chains  ({pct:.1f}%)")

# ── Collect records per split ─────────────────────────────────────────────────

banner("Step 5 — Collect records")

DIFF_ORDER = {"basic": 0, "intermediate": 1, "advanced": 2}

split_records = {"train": [], "validation": [], "test": []}

for cid, recs in chains.items():
    split = assignments[cid]
    sorted_recs = sorted(recs, key=lambda r: DIFF_ORDER.get(r.get("difficulty", ""), 99))
    split_records[split].extend(sorted_recs)

for split in ("train", "validation", "test"):
    log("OK", f"  {split:<12} {len(split_records[split]):>5} records")

# ── Verify distributions ──────────────────────────────────────────────────────

banner("Step 6 — Verify distributions per split")

for split in ("train", "validation", "test"):
    recs = split_records[split]
    if not recs:
        log("WARN", f"{split} is empty")
        continue

    org_c  = defaultdict(int)
    diff_c = defaultdict(int)
    miss_c = defaultdict(int)

    for r in recs:
        org_c[r["organization"]] += 1
        diff_c[r["difficulty"]] += 1
        miss_c[r["mission_name"]] += 1

    log("INFO", f"── {split.upper()} ({len(recs)} records) ──────────────────────")
    log("INFO", "  Organizations:")
    for k, v in sorted(org_c.items(), key=lambda x: -x[1]):
        log("INFO", f"    {k:<30} {v:>4} records")
    log("INFO", "  Difficulty:")
    for d in ("basic", "intermediate", "advanced"):
        log("INFO", f"    {d:<15} {diff_c.get(d,0):>4} records")
    log("INFO", f"  Missions covered : {len(miss_c)}")
    print()

# ── Write JSONL ───────────────────────────────────────────────────────────────

banner("Step 7 — Write JSONL files")

def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kb = path.stat().st_size / 1024
    log("OK", f"  {path.name:<22} {len(records):>5} records  ({kb:.1f} KB)")

write_jsonl(split_records["train"],      TRAIN_FILE)
write_jsonl(split_records["validation"], VAL_FILE)
write_jsonl(split_records["test"],       TEST_FILE)

# ── Final check ───────────────────────────────────────────────────────────────

banner("Final Summary")

total_out = sum(len(v) for v in split_records.values())
log("INFO", f"Input records    : {len(records)}")
log("INFO", f"Output records   : {total_out}")

if total_out == len(records):
    log("OK", "Record count verified — no records lost or duplicated")
else:
    log("WARN", f"Mismatch! {len(records)} in vs {total_out} out")

log("INFO", f"Total chains     : {len(chains)}")
for split in ("train", "validation", "test"):
    log("INFO", f"  {split:<12} {split_chain_counts[split]:>4} chains")

print()
log("OK", f"train.jsonl      → {TRAIN_FILE}")
log("OK", f"validation.jsonl → {VAL_FILE}")
log("OK", f"test.jsonl       → {TEST_FILE}")
log("STEP", "─" * 55)
print()