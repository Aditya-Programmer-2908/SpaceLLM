import json
import logging
import re
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"

SYSTEM_PROMPT = ("""
You are SpaceLLM, an expert AI assistant for space missions,
astronomy, satellites, rockets and aerospace engineering.

Your goal is to help users understand topics clearly.

Response Style Rules:

1. Match the user's request depth.
   - Simple questions → concise answers.
   - Requests like "explain", "tell me more",
     "detailed", "teach me", "how", "why"
     → provide a complete explanation.

2. Always answer the actual question.
   Never say:
   - "The following information is provided below"
   - "A detailed summary is given below"
   - "The information is as follows"

   Instead provide the information directly.

3. Prefer explanations that are:
   - easy to understand
   - engaging
   - factually accurate
   - complete

4. Structure long answers using:
   - Introduction
   - Main explanation
   - Key facts
   - Significance

5. Never reveal internal reasoning.

6. If uncertain, state uncertainty briefly.

7. If outside the space domain, say:
   "I specialise in space missions and astronomy.
   Please consult a general-purpose assistant for this."
""")

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

    device = "cuda:0"   # CUDA_VISIBLE_DEVICES=1 makes GPU1 appear as cuda:0
    log.info("Loading base model %s [native MXFP4] on %s ...", BASE_MODEL_ID, device)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map={"": device},
        trust_remote_code=True,
    )
    log.info("Base model loaded. dtype=%s", next(_base.parameters()).dtype)

    # Vocab alignment — mirrors fine-tuning script exactly
    _base.config.tie_word_embeddings = False
    lm_head = _base.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    log.info("lm_head weight untied and cloned.")

    _base.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = _base.get_output_embeddings().weight.shape[0]
    _base.config.vocab_size = actual_vocab
    log.info("Vocab after resize: %d (padded to multiple of 64)", actual_vocab)

    # Guard: if resize re-tied lm_head, untie again
    if id(_base.get_input_embeddings().weight) == id(_base.get_output_embeddings().weight):
        lm_head = _base.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        log.info("Re-untied lm_head after resize.")

    _base.config.use_cache = False

    log.info("Attaching LoRA adapter: %s", ADAPTER_MODEL_ID)
    model = PeftModel.from_pretrained(
        _base,
        ADAPTER_MODEL_ID,
        trust_remote_code=True,
        is_trainable=False,
        ignore_mismatched_sizes=True,
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


def clean_response(text: str) -> str:
    """
    gpt-oss-20b outputs a chain-of-thought reasoning block before the actual
    answer. The reasoning always ends with one or more 'final' tokens.
    This function strips the reasoning block and returns only the answer.
    """
    # Step 1: Find the last 'final' token and take everything after it
    lower = text.lower()
    last_final = lower.rfind("final")
    if last_final != -1:
        text = text[last_final + len("final"):].strip()

    # Step 2: Strip any remaining leading 'final' repetitions
    text = re.sub(r"^(final\s*)+", "", text, flags=re.IGNORECASE).strip()

    # Step 3: Remove lines that are reasoning artifacts
    artifact_prefixes = (
        "the assistant",
        "the user",
        "assistant will",
        "assistant should",
        "assistant can",
        "assistant must",
    )
    clean_lines = [
        line for line in text.splitlines()
        if not line.strip().lower().startswith(artifact_prefixes)
    ]
    return "\n".join(clean_lines).strip()


@app.get("/health")
def health():
    ready = pipe is not None
    return {"status": "ok" if ready else "loading", "model": ADAPTER_MODEL_ID, "ready": ready}


# Phrases that indicate the model promised content but didn't deliver it
EMPTY_PROMISE_PATTERNS = [
    "below is", "are provided below", "is provided below",
    "following table", "the following list", "the timeline below",
    "listed below", "as follows", "are as follows",
]

def is_empty_promise(text: str) -> bool:
    """Returns True if response promises content but contains no actual data."""
    lower = text.lower()
    has_promise = any(p in lower for p in EMPTY_PROMISE_PATTERNS)
    # If it promises a list/table but has fewer than 5 lines, it's hollow
    has_content = len([l for l in text.splitlines() if l.strip()]) >= 5
    return has_promise and not has_content


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    msgs = [m.model_dump() for m in req.messages]
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    def run_pipe(messages, temperature):
        result = pipe(
            messages,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=10,
            temperature=temperature,
            top_p=req.top_p if req.do_sample else 1.0,
            do_sample=req.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        raw = result[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")
        return clean_response(raw)

    # First attempt
    response_text = run_pipe(msgs, req.temperature if req.do_sample else 1.0)

    # If response is an empty promise, retry with a direct instruction appended
    if is_empty_promise(response_text):
        log.warning("Empty promise detected — retrying with direct instruction.")
        retry_msgs = msgs + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": "Please provide the actual content now. Do not say 'below' — write it directly here."},
        ]
        response_text = run_pipe(retry_msgs, 0.3)  # lower temp for more focused retry

    log.info("Response generated (%d chars).", len(response_text))
    return GenerateResponse(response=response_text)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    ts   = req.timestamp or datetime.now(timezone.utc).isoformat()
    conv = [m.model_dump() for m in req.conversation]

    # Extract the last user question and last LLM answer from conversation
    last_question   = ""
    last_llm_answer = ""
    for msg in reversed(conv):
        if not last_llm_answer and msg["role"] == "assistant":
            last_llm_answer = msg["content"]
        if not last_question and msg["role"] == "user":
            last_question = msg["content"]
        if last_question and last_llm_answer:
            break

    record = {
        "feedback_id":      feedback_id,
        "message_id":       req.message_id,
        "feedback_type":    req.feedback_type,
        "model_version":    req.model_version,
        "timestamp":        ts,
        "question":         last_question,        # what the user asked
        "candidate":        last_llm_answer,      # LLM answer  (BERTScore candidate)
        "reference":        req.correction or "", # user correction (BERTScore reference)
        "has_correction":   bool(req.correction),
        "used_in_training": False,                # flipped to True after v2 fine-tuning
        "bertscore":        None,                 # filled by Analyse phase
        "conversation":     conv,
    }

    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Feedback %s (%s) saved.", feedback_id, req.feedback_type)
    return FeedbackResponse(status="logged", feedback_id=feedback_id)
