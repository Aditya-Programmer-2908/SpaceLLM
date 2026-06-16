"""
training/dataset_builder.py
---------------------------
Utility to export TrainingSamples from the DB into JSONL format for
offline inspection or manual training runs outside the MAPE-K loop.

Usage:
    python -m training.dataset_builder --out dataset.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from database.db import init_db, AsyncSessionLocal
from database import knowledge as kb


async def export_dataset(out_path: str, only_unused: bool = True) -> int:
    await init_db()
    async with AsyncSessionLocal() as db:
        samples = await kb.get_training_samples(
            db, version=None if not only_unused else "unused"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for s in samples:
                f.write(
                    json.dumps({"prompt": s.prompt, "completion": s.completion}) + "\n"
                )
    return len(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",  default="dataset.jsonl")
    parser.add_argument("--all",  action="store_true", help="Include already-used samples")
    args = parser.parse_args()

    n = asyncio.run(export_dataset(args.out, only_unused=not args.all))
    print(f"Exported {n} samples → {args.out}")
