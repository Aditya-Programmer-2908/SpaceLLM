"""
mape_k/analyze.py  —  A layer
------------------------------
Reads unprocessed negative feedback, converts corrections into training
samples, and returns a summary report consumed by the Plan layer.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import knowledge as kb
from database.models import Feedback, Interaction
from sqlalchemy import select

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    total_feedback: int
    positive: int
    negative: int
    neg_ratio: float
    pending_corrections: int
    new_sample_ids: list[int]        # TrainingSample IDs just created
    quality_issues: list[dict]       # interactions with low BERTScore


async def analyze(db: AsyncSession) -> AnalysisReport:
    """
    1. Gather feedback stats.
    2. Convert pending corrections → TrainingSamples.
    3. Find interactions with BERTScore below threshold.
    """
    stats = await kb.feedback_stats(db)
    logger.info("Feedback stats: %s", stats)

    # ── Step 1: convert corrections to training samples ──────────────────
    pending: list[Feedback] = await kb.get_unprocessed_negative_feedback(db)
    new_ids: list[int] = []

    for fb in pending:
        # Fetch the original query so we can build a prompt/completion pair
        result = await db.execute(
            select(Interaction).where(Interaction.id == fb.interaction_id)
        )
        interaction = result.scalars().first()
        if interaction is None:
            continue

        sid = await kb.save_training_sample(
            db,
            prompt=interaction.user_query,
            completion=fb.correction_text,  # type: ignore[arg-type]
            feedback_id=fb.id,
            source="human_correction",
        )
        new_ids.append(sid)

    if new_ids:
        await kb.mark_feedback_processed(db, [fb.id for fb in pending])
        logger.info("Created %d new training samples from corrections.", len(new_ids))

    # ── Step 2: find low-quality responses ───────────────────────────────
    from sqlalchemy import select as sa_select
    from database.models import Interaction as I
    result = await db.execute(
        sa_select(I)
        .where(I.bertscore != None)  # noqa: E711
        .where(I.bertscore < settings.BERTSCORE_THRESHOLD)
        .order_by(I.bertscore)
        .limit(50)
    )
    low_q = result.scalars().all()
    quality_issues = [
        {"id": r.id, "query": r.user_query[:80], "bertscore": r.bertscore}
        for r in low_q
    ]

    return AnalysisReport(
        total_feedback=stats["total"],
        positive=stats["positive"],
        negative=stats["negative"],
        neg_ratio=stats["neg_ratio"],
        pending_corrections=stats["pending_corrections"],
        new_sample_ids=new_ids,
        quality_issues=quality_issues,
    )
