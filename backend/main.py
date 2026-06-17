"""
SpaceLLM Backend — FastAPI
Serves:
  POST /generate   → inference via SpaceLLM_v1 (LoRA over gpt-oss-20b)
  POST /feedback   → stores RLHF/correction payloads for MAPE-K pipeline
  GET  /health     → liveness check

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("spacellm")

# ── Paths ─────────────────────────────────────────────────────────────────
FEEDBACK_LOG = Path("feedback_log.jsonl")   # append-only JSONL for MAPE-K

# ── Model IDs ─────────────────────────────────────────────────────────────
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "astronomy, aerospace engineering, and satellite operations. "
    "Provide accurate, concise, technically rigorous answers. "
    "If a question is outside the space domain, politely redirect the user."
)

# ══════════════════════════════════════════════════════════════════════════
# Model loading  (done once at startup)
# ══════════════════════════════════════════════════════════════════════════
log.info("Loading base model %s …", BASE_MODEL_ID)

_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=Mxfp4Config(dequantize=True),  # dequantise → BF16
    device_map="auto",
    trust_remote_code=True,
)

log.info("Attaching LoRA adapter %s …", ADAPTER_MODEL_ID)
model     = PeftModel.from_pretrained(_base, ADAPTER_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID)
model.eval()

log.info("SpaceLLM_v1 ready ✓")

# ══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="SpaceLLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    Literal["system", "user", "assistant"]
    content: str

class GenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ..., description="Conversation history (user/assistant turns)."
    )
    max_new_tokens: int  = Field(512,  ge=1,   le=2048)
    temperature:    float = Field(0.7,  ge=0.0, le=2.0)
    top_p:          float = Field(0.9,  ge=0.0, le=1.0)
    do_sample:      bool  = Field(True)

class GenerateResponse(BaseModel):
    response:      str
    model_version: str = "SpaceLLM_v1"

class FeedbackRequest(BaseModel):
    message_id:    str
    feedback_type: Literal["positive", "negative"]
    correction:    str | None = None
    timestamp:     str | None = None
    model_version: str = "SpaceLLM_v1"
    conversation:  list[ChatMessage] = []

class FeedbackResponse(BaseModel):
    status:      str
    feedback_id: str

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": ADAPTER_MODEL_ID}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Run inference through SpaceLLM_v1.
    Prepends the system prompt if the caller didn't include one.
    """
    msgs = [m.model_dump() for m in req.messages]

    # Ensure system prompt is present
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    # Apply harmony chat template (required by gpt-oss-20b)
    try:
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        log.error("Chat-template error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Template error: {exc}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.do_sample else 1.0,
            top_p=req.top_p            if req.do_sample else 1.0,
            do_sample=req.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    log.info("Generated %d tokens for %d-turn conversation.",
             len(new_tokens), len(req.messages))

    return GenerateResponse(response=response_text)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """
    Persist RLHF feedback to an append-only JSONL file.
    In the full MAPE-K pipeline this triggers the Analyse → Plan → Execute cycle.
    """
    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id":   feedback_id,
        "message_id":    req.message_id,
        "feedback_type": req.feedback_type,
        "correction":    req.correction,
        "model_version": req.model_version,
        "timestamp":     req.timestamp or datetime.now(timezone.utc).isoformat(),
        "conversation":  [m.model_dump() for m in req.conversation],
    }

    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Feedback %s recorded — type=%s", feedback_id, req.feedback_type)

    return FeedbackResponse(status="logged", feedback_id=feedback_id)
