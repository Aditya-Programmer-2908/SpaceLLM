import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core import inference
from database.db import get_db
from database import knowledge as kb
from mape_k.controller import run_mape_cycle

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference.is_loaded(),
        "model_version": inference.current_version(),
    }


@router.get("/admin/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    """Dashboard stats: feedback counts, adapter history."""
    fb_stats  = await kb.feedback_stats(db)
    adapters  = await kb.list_adapters(db)
    return {
        "feedback": fb_stats,
        "adapters": [
            {
                "version": a.version_tag,
                "hf_repo": a.hf_repo_id,
                "bertscore": a.bertscore,
                "train_samples": a.train_samples,
                "pushed_at": a.pushed_at.isoformat() if a.pushed_at else None,
            }
            for a in adapters
        ],
    }


@router.post("/admin/mape-run")
async def trigger_mape(db: AsyncSession = Depends(get_db)):
    """Manually trigger a MAPE-K cycle (for testing / forced retraining)."""
    # Run in background so the HTTP response returns immediately
    asyncio.create_task(run_mape_cycle(triggered_by="manual"))
    return {"status": "MAPE-K cycle started in background."}


@router.post("/admin/reload-model")
async def reload_model(repo_id: str | None = None):
    """Hot-swap the inference model (useful after a manual HF push)."""
    asyncio.create_task(inference.load_model(adapter_repo=repo_id))
    return {"status": f"Reloading model{f' from {repo_id}' if repo_id else ''} …"}
