import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"
SYSTEM_PROMPT    = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "astronomy, aerospace engineering, and satellite operations. "
    "Provide accurate, concise, technically rigorous answers. "
    "If a question is outside the space domain, politely redirect the user."
)

model     = None
tokenizer = None
pipe      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, pipe

    log.info("Loading base model %s ...", BASE_MODEL_ID)
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map={"": device},
        trust_remote_code=True,
    )
    log.info("Base model loaded. Attaching LoRA adapter %s ...", ADAPTER_MODEL_ID)

    model = PeftModel.from_pretrained(
        _base,
        ADAPTER_MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map={"": device},
    )

    log.info("SpaceLLM_v1 ready!")
    yield

    log.info("Shutting down.")
    del model, tokenizer, pipe
    torch.cuda.empty_cache()


app = FastAPI(title="SpaceLLM API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role:    Literal["system", "user", "assistant"]
    content: str

class GenerateRequest(BaseModel):
    messages:       list[ChatMessage]
    max_new_tokens: int   = 512
    temperature:    float = 0.7
    top_p:          float = 0.9
    do_sample:      bool  = True

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


@app.get("/health")
def health():
    ready = pipe is not None
    return {"status": "ok" if ready else "loading", "model": ADAPTER_MODEL_ID, "ready": ready}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    msgs = [m.model_dump() for m in req.messages]
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    result = pipe(
        msgs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature if req.do_sample else 1.0,
        top_p=req.top_p if req.do_sample else 1.0,
        do_sample=req.do_sample,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    response_text = result[0]["generated_text"].strip()
    log.info("Response generated (%d chars).", len(response_text))
    return GenerateResponse(response=response_text)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
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
    log.info("Feedback %s logged.", feedback_id)
    return FeedbackResponse(status="logged", feedback_id=feedback_id)
