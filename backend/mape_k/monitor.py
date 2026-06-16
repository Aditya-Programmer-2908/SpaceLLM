"""
mape_k/monitor.py  —  M layer
------------------------------
Persists every interaction and schedules background BERTScore computation.
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from core import bertscore as bs
from database import knowledge as kb

logger = logging.getLogger(__name__)


async def log_interaction(
    db: AsyncSession,
    user_query: str,
    model_response: str,
    model_version: str,
    latency_ms: float,
    session_id: Optional[str] = None,
) -> int:
    """Persist interaction immediately; score asynchronously."""
    interaction_id = await kb.save_interaction(
        db,
        user_query=user_query,
        model_response=model_response,
        model_version=model_version,
        latency_ms=latency_ms,
        session_id=session_id,
    )
    # Fire-and-forget BERTScore computation
    asyncio.create_task(
        _score_and_update(interaction_id, model_response, user_query)
    )
    return interaction_id


async def _score_and_update(
    interaction_id: int, response: str, query: str
) -> None:
    """Compute BERTScore and write it back; runs outside the request lifecycle."""
    from database.db import AsyncSessionLocal
    try:
        f1 = await bs.score(response, query)
        if f1 is not None:
            async with AsyncSessionLocal() as db:
                await kb.update_interaction_bertscore(db, interaction_id, f1)
                logger.debug(
                    "BERTScore %.4f stored for interaction %d", f1, interaction_id
                )
    except Exception as exc:
        logger.warning("Background BERTScore failed for %d: %s", interaction_id, exc)
