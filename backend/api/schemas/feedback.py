from pydantic import BaseModel, Field
from typing import Optional


class FeedbackRequest(BaseModel):
    interaction_id: int
    feedback_type: str = Field(..., pattern="^(positive|negative)$")
    correction_text: Optional[str] = None   # required when feedback_type == negative
    model_version: str = "SpaceLLM_v1"


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int
    message: str
