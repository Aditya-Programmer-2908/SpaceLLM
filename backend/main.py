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

SYSTEM_PROMPT = (
    "You are SpaceLLM, a precise AI assistant for space missions, astronomy, and aerospace engineering. "
    "You are fine-tuned on mission data from NASA, ISRO, and ESA. "
    "Answer DIRECTLY and concisely. Never explain your reasoning process. "
    "Never output internal thoughts, plans, or meta-commentary. "
    "If the question is outside the space domain, say: "
    "'I specialise in space missions and astronomy. Please consult a general-purpose assistant for this.' "
    "If uncertain, say so briefly and give your best answer. "
    "Keep answers factual and technically accurate."
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


# gpt-oss models emit the "harmony" response format: a reasoning ("analysis")
# block followed by the real answer in a "final" channel, delimited by
# literal special tokens, e.g.:
#   <|channel|>analysis<|message|> ... reasoning ... <|end|>
#   <|start|>assistant<|channel|>final<|message|> ... answer ... <|return|>
#
# We must extract content using these literal markers. Searching for the
# plain English word "final" (the previous approach) is unsafe: space/launch
# answers routinely contain that word in normal sentences (e.g. "final orbit
# insertion", "final stage burn", "final launch window"). rfind("final")
# would match the LAST such occurrence inside the real answer and discard
# everything before it -- which is exactly why timelines/lists were coming
# back truncated or blank.
HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.IGNORECASE | re.DOTALL,
)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def clean_response(text: str) -> str:
    """
    Extract the final-channel answer from gpt-oss harmony-formatted output,
    stripping the analysis/reasoning block. Falls back to returning the text
    unchanged (minus stray special tokens) if no harmony markers are present
    -- e.g. when the chat pipeline has already parsed the turns for us.
    """
    if not text:
        return ""

    matches = HARMONY_FINAL_RE.findall(text)
    if matches:
        # If the model emitted multiple "final" blocks, the last one wins.
        text = matches[-1]

    # Defensively strip any leftover special tokens (covers both the
    # harmony-marker case above and the "no markers found" fallback case).
    text = SPECIAL_TOKEN_RE.sub("", text).strip()

    # Remove any residual meta-commentary lines that slipped through.
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
        min_new_tokens=50,
        temperature=req.temperature if req.do_sample else 1.0,
        top_p=req.top_p if req.do_sample else 1.0,
        do_sample=req.do_sample,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    raw = result[0]["generated_text"]
    if isinstance(raw, list):
        raw = raw[-1].get("content", "")

    response_text = clean_response(raw)

    # Safety net: if cleaning somehow produced an empty string but the model
    # did generate something, fall back to the raw text (special tokens
    # stripped) rather than silently returning blank to the user.
    if not response_text and raw:
        response_text = SPECIAL_TOKEN_RE.sub("", raw).strip()

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
