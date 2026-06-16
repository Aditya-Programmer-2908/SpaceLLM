from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str                    # "user" | "assistant"
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    session_id: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    response: str
    model_version: str
    bertscore: Optional[float] = None
    interaction_id: int
    latency_ms: float
