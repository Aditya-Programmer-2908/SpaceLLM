import json
import logging
import uuid
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ── Login to HuggingFace first, before any other HF imports ──────────────
from huggingface_hub import login
login(token="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")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"
HF_TOKEN         = "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"
SYSTEM_PROMPT    = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "astronomy, aerospace engineering, and satellite operations. "
    "Provide accurate, concise, technically rigorous answers. "
    "If a question is outside the space domain, politely redirect the user."
)

model     = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    log.info("Loading base model %s ...", BASE_MODEL_ID)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    config = AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        config=config,
        quantization_config=bnb_cfg,
        device_map={"": device},
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    log.info("Base model loaded. Attaching adapter %s ...", ADAPTER_MODEL_ID)

    model = PeftModel.from_pretrained(
        _base,
        ADAPTER_MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    log.info("SpaceLLM_v1 ready!")

    yield

    log.info("Shutting down.")
    del model, tokenizer
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
    ready = model is not None and tokenizer is not None
    return {"status": "ok" if ready else "loading", "model": ADAPTER_MODEL_ID, "ready": ready}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    msgs = [m.model_dump() for m in req.messages]
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    try:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Template error: {exc}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.do_sample else 1.0,
            top_p=req.top_p if req.do_sample else 1.0,
            do_sample=req.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens    = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info("Generated %d tokens.", len(new_tokens))
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
