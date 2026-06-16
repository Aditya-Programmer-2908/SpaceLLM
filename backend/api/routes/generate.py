from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.generate import GenerateRequest, GenerateResponse
from core import inference
from database.db import get_db
from mape_k import monitor

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(
    req: GenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    if not inference.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    messages = [m.model_dump() for m in req.messages]

    try:
        response_text, latency_ms = await inference.generate(
            messages=messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # M layer — log the interaction (BERTScore computed async in background)
    interaction_id = await monitor.log_interaction(
        db=db,
        user_query=messages[-1]["content"],
        model_response=response_text,
        model_version=inference.current_version(),
        latency_ms=latency_ms,
        session_id=req.session_id,
    )

    return GenerateResponse(
        response=response_text,
        model_version=inference.current_version(),
        interaction_id=interaction_id,
        latency_ms=round(latency_ms, 2),
    )
