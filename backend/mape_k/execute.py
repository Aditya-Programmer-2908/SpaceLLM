"""
mape_k/execute.py  —  E layer
------------------------------
Orchestrates continual LoRA fine-tuning and HuggingFace adapter push.
Runs in a subprocess-friendly way (blocking; call from a thread or separate
process so it doesn't block the FastAPI event loop).
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import knowledge as kb
from mape_k.plan import RetrainingPlan

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    new_version_tag: str
    hf_repo_id: str
    bertscore: Optional[float]
    error: Optional[str] = None


async def execute(
    db: AsyncSession, plan: RetrainingPlan
) -> ExecutionResult:
    """
    High-level coroutine that:
      1. Builds the JSONL dataset from plan.sample_ids
      2. Runs LoRA training in a thread (blocking)
      3. Evaluates the checkpoint
      4. Pushes the adapter to HuggingFace
      5. Registers the new version in the DB
      6. Hot-swaps the inference engine
    """
    import asyncio

    # Fetch samples
    from sqlalchemy import select
    from database.models import TrainingSample
    result = await db.execute(
        select(TrainingSample).where(TrainingSample.id.in_(plan.sample_ids))
    )
    samples = result.scalars().all()

    if not samples:
        return ExecutionResult(
            success=False,
            new_version_tag=plan.new_version_tag,
            hf_repo_id="",
            bertscore=None,
            error="No training samples found.",
        )

    # Write dataset to a temp JSONL file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for s in samples:
            f.write(
                json.dumps({"prompt": s.prompt, "completion": s.completion}) + "\n"
            )
        dataset_path = f.name

    logger.info(
        "Starting LoRA training — %d samples → %s", len(samples), plan.new_version_tag
    )

    try:
        # Run blocking training in a thread pool
        loop = asyncio.get_event_loop()
        checkpoint_dir, bertscore = await loop.run_in_executor(
            None,
            _run_training,
            dataset_path,
            plan.base_adapter_repo,
            plan.new_version_tag,
        )

        if checkpoint_dir is None:
            raise RuntimeError("Training returned no checkpoint.")

        # Push to HuggingFace
        hf_repo_id = await loop.run_in_executor(
            None, _push_to_hf, checkpoint_dir, plan.new_version_tag
        )

        # Register in DB
        await kb.register_adapter(
            db,
            version_tag=plan.new_version_tag,
            hf_repo_id=hf_repo_id,
            base_version=plan.base_adapter_repo.split("/")[-1],
            bertscore=bertscore,
            train_samples=len(samples),
            notes=f"Auto-trained by MAPE-K. Dataset: {dataset_path}",
        )

        # Mark samples as used
        await kb.mark_samples_used(db, plan.sample_ids, plan.new_version_tag)

        # Hot-swap inference model
        from core.inference import load_model
        await load_model(adapter_repo=hf_repo_id)

        logger.info("Execution complete — new adapter: %s", hf_repo_id)
        return ExecutionResult(
            success=True,
            new_version_tag=plan.new_version_tag,
            hf_repo_id=hf_repo_id,
            bertscore=bertscore,
        )

    except Exception as exc:
        logger.error("Execution failed: %s", exc, exc_info=True)
        return ExecutionResult(
            success=False,
            new_version_tag=plan.new_version_tag,
            hf_repo_id="",
            bertscore=None,
            error=str(exc),
        )
    finally:
        os.unlink(dataset_path)


# ── Blocking helpers (run in thread pool) ────────────────────────────────────

def _run_training(
    dataset_path: str,
    base_adapter_repo: str,
    new_version_tag: str,
) -> tuple[Optional[str], Optional[float]]:
    """Load base + existing LoRA, continue fine-tuning, return checkpoint dir."""
    import torch
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import json

    out_dir = Path(settings.OUTPUT_DIR) / new_version_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(
        settings.BASE_MODEL_ID,
        token=settings.HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load base + previous LoRA adapter
    base = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=settings.DEVICE_MAP,
        trust_remote_code=True,
        token=settings.HF_TOKEN or None,
    )
    model = PeftModel.from_pretrained(
        base, base_adapter_repo,
        is_trainable=True,
        token=settings.HF_TOKEN or None,
    )

    # Load dataset
    records = []
    with open(dataset_path) as f:
        for line in f:
            records.append(json.loads(line))
    ds = Dataset.from_list(records)

    def tokenize(ex):
        full = ex["prompt"] + ex["completion"] + tok.eos_token
        enc  = tok(full, truncation=True, max_length=1024, padding="max_length")
        # Labels: mask the prompt tokens
        prompt_ids = tok(ex["prompt"], truncation=True, max_length=512)["input_ids"]
        labels = [-100] * len(prompt_ids) + enc["input_ids"][len(prompt_ids):]
        labels = labels[:1024] + [-100] * max(0, 1024 - len(labels))
        enc["labels"] = labels
        return enc

    ds = ds.map(tokenize, remove_columns=["prompt", "completion"])
    ds = ds.train_test_split(test_size=0.1, seed=42)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=settings.TRAIN_EPOCHS,
        per_device_train_batch_size=settings.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=settings.GRADIENT_ACCUM,
        learning_rate=settings.LEARNING_RATE,
        warmup_steps=settings.WARMUP_STEPS,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=64),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Quick BERTScore eval on test set
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    hyps, refs = [], []
    for ex in ds["test"].select(range(min(20, len(ds["test"])))):
        hyps.append(str(ex.get("completion", "")))
        refs.append(str(ex.get("prompt", "")))
    if hyps:
        _, _, F1 = scorer.score(hyps, refs)
        bertscore = round(float(F1.mean()), 4)
    else:
        bertscore = None

    return str(out_dir), bertscore


def _push_to_hf(checkpoint_dir: str, version_tag: str) -> str:
    """Push adapter to HuggingFace and return the full repo ID."""
    from huggingface_hub import HfApi
    api = HfApi(token=settings.HF_TOKEN or None)
    repo_id = f"{settings.HF_ORG}/{version_tag}"
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        commit_message=f"Auto-push by SpaceLLM MAPE-K controller — {version_tag}",
    )
    logger.info("Pushed adapter to https://huggingface.co/%s", repo_id)
    return repo_id
