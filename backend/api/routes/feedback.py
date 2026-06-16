from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.feedback import FeedbackRequest, FeedbackResponse
from database.db import get_db
from database import knowledge as kb

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    # Validate correction provided for negative feedback
    if req.feedback_type == "negative" and not req.correction_text:
        # Still accept it — will be flagged without a training sample
        pass

    try:
        feedback_id = await kb.save_feedback(
            db,
            interaction_id=req.interaction_id,
            feedback_type=req.feedback_type,
            correction_text=req.correction_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    msg = (
        "Positive feedback logged."
        if req.feedback_type == "positive"
        else "Negative feedback logged. Correction queued for next training cycle."
    )
    return FeedbackResponse(status="ok", feedback_id=feedback_id, message=msg)
