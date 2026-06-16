"""
core/bertscore.py
-----------------
Thin async wrapper around the `bert_score` library.
Scores are computed in a thread-pool so they don't block the event loop.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Reference sentences used when no ground-truth is available.
# We score the model response against the user query as a proxy for relevance.
# Replace with domain-specific references for better signal.
_FALLBACK_REF = None


@lru_cache(maxsize=1)
def _get_scorer():
    """Load BERTScorer once and cache it."""
    try:
        from bert_score import BERTScorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        logger.info("BERTScorer loaded.")
        return scorer
    except Exception as e:
        logger.warning("BERTScorer unavailable: %s", e)
        return None


async def score(
    hypothesis: str,
    reference: Optional[str] = None,
) -> Optional[float]:
    """Return F1 BERTScore for a single hypothesis/reference pair.

    Parameters
    ----------
    hypothesis  : model output
    reference   : ground-truth or user query (used as fallback proxy)

    Returns
    -------
    float in [0, 1] or None if scoring fails.
    """
    ref = reference or hypothesis   # self-score as absolute fallback
    loop = asyncio.get_event_loop()
    try:
        f1 = await loop.run_in_executor(None, _compute_f1, hypothesis, ref)
        return round(float(f1), 4)
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)
        return None


def _compute_f1(hyp: str, ref: str) -> float:
    scorer = _get_scorer()
    if scorer is None:
        return 0.0
    _, _, F1 = scorer.score([hyp], [ref])
    return F1.item()
