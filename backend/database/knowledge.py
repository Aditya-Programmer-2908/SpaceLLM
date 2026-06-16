"""
database/knowledge.py
---------------------
K layer of MAPE-K — single place for all database I/O.
All functions accept an AsyncSession so callers control transactions.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    Interaction, Feedback, TrainingSample, AdapterVersion, MapeRun,
)


# ── Interactions ─────────────────────────────────────────────────────────────

async def save_interaction(
    db: AsyncSession,
    user_query: str,
    model_response: str,
    model_version: str,
    bertscore: Optional[float] = None,
    latency_ms: Optional[float] = None,
    session_id: Optional[str] = None,
) -> int:
    row = Interaction(
        session_id=session_id,
        user_query=user_query,
        model_response=model_response,
        model_version=model_version,
        bertscore=bertscore,
        latency_ms=latency_ms,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def update_interaction_bertscore(
    db: AsyncSession, interaction_id: int, bertscore: float
) -> None:
    await db.execute(
        update(Interaction)
        .where(Interaction.id == interaction_id)
        .values(bertscore=bertscore)
    )
    await db.commit()


# ── Feedback ─────────────────────────────────────────────────────────────────

async def save_feedback(
    db: AsyncSession,
    interaction_id: int,
    feedback_type: str,
    correction_text: Optional[str] = None,
) -> int:
    row = Feedback(
        interaction_id=interaction_id,
        feedback_type=feedback_type,
        correction_text=correction_text,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def get_unprocessed_negative_feedback(
    db: AsyncSession, limit: int = 200
) -> list[Feedback]:
    result = await db.execute(
        select(Feedback)
        .where(Feedback.feedback_type == "negative")
        .where(Feedback.is_processed == False)  # noqa: E712
        .where(Feedback.correction_text != None)  # noqa: E711
        .limit(limit)
    )
    return list(result.scalars().all())


async def mark_feedback_processed(
    db: AsyncSession, feedback_ids: list[int]
) -> None:
    for fid in feedback_ids:
        await db.execute(
            update(Feedback)
            .where(Feedback.id == fid)
            .values(is_processed=True)
        )
    await db.commit()


async def feedback_stats(db: AsyncSession) -> dict:
    total_q = await db.execute(select(func.count()).select_from(Feedback))
    pos_q   = await db.execute(
        select(func.count()).select_from(Feedback)
        .where(Feedback.feedback_type == "positive")
    )
    neg_q   = await db.execute(
        select(func.count()).select_from(Feedback)
        .where(Feedback.feedback_type == "negative")
    )
    corr_q  = await db.execute(
        select(func.count()).select_from(Feedback)
        .where(Feedback.correction_text != None)  # noqa: E711
        .where(Feedback.is_processed == False)    # noqa: E712
    )
    total = total_q.scalar() or 0
    pos   = pos_q.scalar()   or 0
    neg   = neg_q.scalar()   or 0
    corr  = corr_q.scalar()  or 0
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "pending_corrections": corr,
        "neg_ratio": round(neg / total, 3) if total else 0.0,
    }


# ── Training Samples ──────────────────────────────────────────────────────────

async def save_training_sample(
    db: AsyncSession,
    prompt: str,
    completion: str,
    feedback_id: Optional[int] = None,
    source: str = "human_correction",
    quality_score: Optional[float] = None,
) -> int:
    row = TrainingSample(
        feedback_id=feedback_id,
        prompt=prompt,
        completion=completion,
        source=source,
        quality_score=quality_score,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def get_training_samples(
    db: AsyncSession, version: Optional[str] = None, limit: int = 500
) -> list[TrainingSample]:
    q = select(TrainingSample)
    if version:
        q = q.where(TrainingSample.used_in_version == None)  # noqa: E711
    result = await db.execute(q.limit(limit))
    return list(result.scalars().all())


async def mark_samples_used(
    db: AsyncSession, sample_ids: list[int], version: str
) -> None:
    for sid in sample_ids:
        await db.execute(
            update(TrainingSample)
            .where(TrainingSample.id == sid)
            .values(used_in_version=version)
        )
    await db.commit()


# ── Adapter Versions ──────────────────────────────────────────────────────────

async def register_adapter(
    db: AsyncSession,
    version_tag: str,
    hf_repo_id: str,
    base_version: Optional[str] = None,
    bertscore: Optional[float] = None,
    train_samples: Optional[int] = None,
    notes: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> int:
    row = AdapterVersion(
        version_tag=version_tag,
        hf_repo_id=hf_repo_id,
        base_version=base_version,
        bertscore=bertscore,
        train_samples=train_samples,
        notes=notes,
        extra_meta=extra_meta,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def get_latest_adapter(db: AsyncSession) -> Optional[AdapterVersion]:
    result = await db.execute(
        select(AdapterVersion).order_by(AdapterVersion.pushed_at.desc()).limit(1)
    )
    return result.scalars().first()


async def list_adapters(db: AsyncSession) -> list[AdapterVersion]:
    result = await db.execute(
        select(AdapterVersion).order_by(AdapterVersion.pushed_at.desc())
    )
    return list(result.scalars().all())


# ── MAPE-K Runs ───────────────────────────────────────────────────────────────

async def start_mape_run(
    db: AsyncSession, triggered_by: str = "scheduler"
) -> int:
    row = MapeRun(triggered_by=triggered_by, status="running")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def finish_mape_run(
    db: AsyncSession,
    run_id: int,
    status: str,
    samples_found: int = 0,
    retrain_decided: bool = False,
    new_version: Optional[str] = None,
    log: Optional[str] = None,
) -> None:
    await db.execute(
        update(MapeRun)
        .where(MapeRun.id == run_id)
        .values(
            status=status,
            samples_found=samples_found,
            retrain_decided=retrain_decided,
            new_version=new_version,
            log=log,
            finished_at=datetime.utcnow(),
        )
    )
    await db.commit()
