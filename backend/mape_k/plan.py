"""
mape_k/plan.py  —  P layer
---------------------------
Consumes the AnalysisReport and emits a RetrainingPlan (or None).
Decision logic is intentionally simple and easy to tune via config.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import knowledge as kb
from mape_k.analyze import AnalysisReport

logger = logging.getLogger(__name__)


@dataclass
class RetrainingPlan:
    should_retrain: bool
    reason: str
    sample_ids: list[int] = field(default_factory=list)
    base_adapter_repo: str = ""
    new_version_tag: str = ""


async def plan(db: AsyncSession, report: AnalysisReport) -> RetrainingPlan:
    """
    Decide whether retraining is warranted.

    Rules (all must pass):
      1. At least MIN_FEEDBACK_FOR_RETRAIN samples available.
      2. Negative feedback ratio >= MIN_NEG_RATIO  OR  quality_issues exist.
    """
    # Collect all unused training samples
    samples = await kb.get_training_samples(db)
    sample_ids = [s.id for s in samples]

    enough_data = len(sample_ids) >= settings.MIN_FEEDBACK_FOR_RETRAIN
    quality_signal = (
        report.neg_ratio >= settings.MIN_NEG_RATIO
        or len(report.quality_issues) > 0
    )

    if not enough_data:
        reason = (
            f"Only {len(sample_ids)} training samples available "
            f"(need ≥ {settings.MIN_FEEDBACK_FOR_RETRAIN}). Skipping retrain."
        )
        logger.info(reason)
        return RetrainingPlan(should_retrain=False, reason=reason)

    if not quality_signal:
        reason = (
            f"Quality looks fine (neg_ratio={report.neg_ratio:.2%}, "
            f"low-q interactions={len(report.quality_issues)}). Skipping retrain."
        )
        logger.info(reason)
        return RetrainingPlan(should_retrain=False, reason=reason)

    # Determine next version tag
    latest = await kb.get_latest_adapter(db)
    if latest:
        # SpaceLLM_v2 → SpaceLLM_v3 etc.
        try:
            num = int(latest.version_tag.split("_v")[-1]) + 1
        except ValueError:
            num = 2
        base_repo = latest.hf_repo_id
    else:
        num = 2
        base_repo = settings.HF_REPO_ID

    new_tag = f"SpaceLLM_v{num}"
    reason = (
        f"Retrain triggered: {len(sample_ids)} samples, "
        f"neg_ratio={report.neg_ratio:.2%}, "
        f"low-q={len(report.quality_issues)}"
    )
    logger.info("%s → %s", reason, new_tag)

    return RetrainingPlan(
        should_retrain=True,
        reason=reason,
        sample_ids=sample_ids,
        base_adapter_repo=base_repo,
        new_version_tag=new_tag,
    )
