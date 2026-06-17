import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, pipeline

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

    log.info("Loading tokenizer from adapter: %s", ADAPTER_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    log.info("Tokenizer vocab size: %d  |  len(tokenizer): %d",
             tokenizer.vocab_size, len(tokenizer))

    # ── Load base model exactly as fine-tuning script did ────────────────
    # Fine-tuning used Mxfp4Config(dequantize=True) → BF16 weights on CPU,
    # then dispatched to GPU. We mirror that exactly so adapter shapes match.
    log.info("Loading base model %s  [Mxfp4Config dequantize=True → BF16] ...", BASE_MODEL_ID)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="auto",
        trust_remote_code=True,
    )
    log.info("Base model loaded. dtype=%s", next(_base.parameters()).dtype)

    # ── Vocab alignment — mirrors fine-tuning script exactly ─────────────
    # Fine-tuning did:
    #   1. tie_word_embeddings = False
    #   2. detach+clone lm_head weight
    #   3. resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    # We must reproduce the same vocab size so adapter weights match.
    _base.config.tie_word_embeddings = False

    lm_head = _base.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    log.info("lm_head weight untied and cloned.")

    _base.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = _base.get_output_embeddings().weight.shape[0]
    _base.config.vocab_size = actual_vocab
    log.info("Vocab after resize: %d  (padded to multiple of 64)", actual_vocab)

    # Guard: if resize re-tied lm_head, untie again
    if id(_base.get_input_embeddings().weight) == id(_base.get_output_embeddings().weight):
        lm_head = _base.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        log.info("Re-untied lm_head after resize.")

    _base.config.use_cache = False

    # ── Attach LoRA adapter ───────────────────────────────────────────────
    log.info("Attaching LoRA adapter: %s", ADAPTER_MODEL_ID)
    model = PeftModel.from_pretrained(
        _base,
        ADAPTER_MODEL_ID,
        trust_remote_code=True,
        is_trainable=False,
    )
    model.eval()
    log.info("SpaceLLM_v1 ready!")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

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

    response_text = result[0]["generated_text"]
    if isinstance(response_text, list):
        # pipeline returned list of dicts (chat format)
        response_text = response_text[-1].get("content", "")
    response_text = response_text.strip()

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
