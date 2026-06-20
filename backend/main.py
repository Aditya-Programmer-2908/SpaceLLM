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
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"

# GPT-OSS is a native reasoning model: every generation writes a hidden
# "analysis" (chain-of-thought) pass before the visible "final" pass, and both
# consume max_new_tokens. "low" keeps the hidden pass short so the budget is
# spent on the answer instead of being eaten by reasoning tokens.
# Only settable via apply_chat_template — pipeline() does not expose it reliably.
REASONING_EFFORT = "low"

# ── System prompts ─────────────────────────────────────────────────────────────

# Default: short, factual, direct. 50-200 words.
SYSTEM_PROMPT_SHORT = """\
You are SpaceLLM, a precise expert assistant specialising in space missions, \
astronomy, satellites, launch vehicles, planetary science, and aerospace engineering, \
trained on data from NASA, ISRO, and ESA mission archives.

RESPONSE RULES:
- Answer directly and factually. No preamble, no filler.
- Also respond to greetings gracefully in 10 - 30 words.
- Keep responses between 50 and 200 words.
- If the question is outside the space domain, say exactly:
  "I specialise in space missions. Please consult a general-purpose assistant."
- Never reveal your reasoning process in the output.\
"""

# Detailed: chain-of-thought reasoning internally, high-quality answer externally.
SYSTEM_PROMPT_DETAIL = """\
You are SpaceLLM, a deep-knowledge expert assistant specialising in space missions, \
astronomy, satellites, launch vehicles, planetary science, and aerospace engineering, \
trained on data from NASA, ISRO, and ESA mission archives.

THINKING PROCESS (internal — never shown to user):
Before writing your answer, silently work through these steps:
1. Identify the core topic and what the user actually needs to understand.
2. Recall the key facts, figures, dates, and technical parameters relevant to this topic.
3. Identify common misconceptions or gaps a learner might have.
4. Structure the explanation: background → mechanics/technical detail → significance → legacy.
5. Check: is every claim accurate? Are any figures approximate? Note uncertainty if present.

OUTPUT RULES:
- Write a thorough, technically accurate, educational explanation.
- Aim for 300–600 words. Do not pad; stop when the explanation is complete.
- Make sure that the response is complete if not complete make sure it is complete and is abiding the rules of the this program like max tokens = 1024.
- Use flowing paragraphs. No bullet lists.
- Open directly with substance — never with "This overview covers..." or similar.
- Show technical depth: include specific mission names, instrument names, orbital parameters,
  dates, agencies, and scientific outcomes where relevant.
- If the question is outside the space domain, say exactly:
  "I specialise in space missions. Please consult a general-purpose assistant."
- Never reveal your chain-of-thought in the output.\
"""

# Keywords that force the short path regardless of anything else
SHORT_INTENT = {
    "in short", "briefly", "brief", "quick", "summarize", "summary",
    "tldr", "tl;dr", "one line", "one sentence", "short answer",
    "concise", "in brief", "just tell me", "short", "give me a short",
    "keep it short",
}

# Keywords that trigger the detailed path.
# NOTE: bare "how", "why", "about", "tell me" were removed — they're too
# generic and false-positive on everyday short queries/greetings (e.g. "how
# are you", "show me the rover" via substring match on "how"). The phrasal
# versions below already capture genuine detail-seeking intent.
DETAIL_INTENT = {
    "explain", "tell me about", "describe", "detail", "details", "detailed",
    "how does", "how do", "how did", "why did", "why does", "why is",
    "what is", "what are", "what was", "what were",
    "teach me", "elaborate", "compare", "history", "background",
    "overview", "in depth", "deep dive", "walk me through",
}

# Compiled once at import time. \b word-boundary matching prevents partial-word
# hits like "how" matching inside "show"/"somehow", or "is" matching inside "this".
_SHORT_PATTERN  = re.compile(r"\b(" + "|".join(re.escape(k) for k in SHORT_INTENT)  + r")\b")
_DETAIL_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in DETAIL_INTENT) + r")\b")

model     = None
tokenizer = None


# ── Model lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    log.info("Loading tokenizer: %s", ADAPTER_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    log.info("Tokenizer vocab=%d  len=%d", tokenizer.vocab_size, len(tokenizer))

    device = "cuda:0"
    log.info("Loading base model %s on %s", BASE_MODEL_ID, device)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map={"": device},
        trust_remote_code=True,
    )
    log.info("Base loaded. dtype=%s", next(_base.parameters()).dtype)

    _base.config.tie_word_embeddings = False
    lm_head = _base.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    log.info("lm_head untied.")

    _base.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    actual_vocab = _base.get_output_embeddings().weight.shape[0]
    _base.config.vocab_size = actual_vocab
    log.info("Vocab after resize: %d", actual_vocab)

    if id(_base.get_input_embeddings().weight) == id(_base.get_output_embeddings().weight):
        lm_head = _base.get_output_embeddings()
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        log.info("Re-untied lm_head after resize.")

    _base.config.use_cache = False

    log.info("Attaching LoRA adapter: %s", ADAPTER_MODEL_ID)
    model = PeftModel.from_pretrained(
        _base, ADAPTER_MODEL_ID,
        trust_remote_code=True,
        is_trainable=False,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    log.info("SpaceLLM ready.")

    yield

    log.info("Shutting down.")
    del model, tokenizer
    torch.cuda.empty_cache()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="SpaceLLM API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    Literal["system", "user", "assistant"]
    content: str

class GenerateRequest(BaseModel):
    messages:       list[ChatMessage]
    max_new_tokens: int   = 1536   # was 1024 — 600-word detail answers need headroom
    temperature:    float = 0.7
    top_p:          float = 0.9
    do_sample:      bool  = True

class GenerateResponse(BaseModel):
    response:      str
    model_version: str = "SpaceLLM_v1"
    truncated:     bool = False

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


# ── Core logic ─────────────────────────────────────────────────────────────────

def classify_request(messages: list[dict]) -> str:
    """
    Returns 'short' or 'detail' by scanning the last 3 user turns.
    SHORT_INTENT wins if present; then DETAIL_INTENT; else 'short'.
    Uses word-boundary regex matching so phrases only match whole words
    (e.g. "how" in DETAIL_INTENT won't fire on "show" or "somehow").
    """
    text = " ".join(
        m["content"].lower()
        for m in messages[-3:]
        if m["role"] == "user"
    )
    if _SHORT_PATTERN.search(text):
        log.info("classify=short (short-intent keyword) | %.120s", text)
        return "short"
    if _DETAIL_PATTERN.search(text):
        log.info("classify=detail | %.120s", text)
        return "detail"
    log.info("classify=short (default) | %.120s", text)
    return "short"


def build_messages(raw: list[dict], mode: str) -> list[dict]:
    """
    Construct final message list.
    - Strip any client-supplied system messages (we own the system prompt).
    - Inject the correct system prompt at position 0.
    """
    system = SYSTEM_PROMPT_DETAIL if mode == "detail" else SYSTEM_PROMPT_SHORT
    conv   = [m for m in raw if m["role"] != "system"]
    return [{"role": "system", "content": system}] + conv


def clean_response(text: str) -> str:
    """
    Remove GPT-OSS internal reasoning artifacts that leak into output.
    Only strips clearly mechanical prefixes — does not alter actual content.
    """
    if not text:
        return ""
    text = text.strip()

    # Strip leaked reasoning blocks: "Analysis ... Final answer:"
    if text.lower().startswith("analysis"):
        parts = re.split(r"\bfinal\b", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[1].strip()

    # Strip leading "Final " / "Final Answer:" prefixes
    text = re.sub(r"^(final\s*(answer\s*[:\-]?\s*)?)+", "", text, flags=re.IGNORECASE).strip()

    # Strip lines where model talks about itself in third person
    bad_starts = (
        "the assistant", "assistant will", "assistant should",
        "assistant can", "assistant must",
    )
    lines = [l for l in text.splitlines()
             if not l.strip().lower().startswith(bad_starts)]
    return "\n".join(lines).strip()


def run_generate(messages: list[dict], req: GenerateRequest, temperature: float,
                  min_new_tokens: int) -> tuple[str, bool]:
    """
    Calls model.generate() directly (instead of the high-level pipeline) so
    that reasoning_effort can be passed through apply_chat_template — this is
    not reliably exposed via pipeline(). Also detects whether generation hit
    the max_new_tokens ceiling, which is the main signal for truncated output.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=REASONING_EFFORT,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=req.top_p if req.do_sample else 1.0,
            do_sample=req.do_sample,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    truncated  = new_tokens.shape[0] >= req.max_new_tokens
    if truncated:
        log.warning("Hit max_new_tokens=%d before EOS — response likely truncated.",
                     req.max_new_tokens)

    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    log.info("\n=== RAW ===\n%s\n===========", raw)
    return clean_response(raw), truncated


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    ready = model is not None
    return {"status": "ok" if ready else "loading", "model": ADAPTER_MODEL_ID, "ready": ready}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    raw  = [m.model_dump() for m in req.messages]
    mode = classify_request(raw)
    msgs = build_messages(raw, mode)

    temperature = req.temperature if req.do_sample else 1.0

    # Token floors:
    # - short: 30 tokens minimum (enough for 1-2 solid sentences, prevents empty output)
    # - detail: 400 tokens minimum (~300 words), model stops naturally when complete
    min_tok = 400 if mode == "detail" else 30

    response, truncated = run_generate(msgs, req, temperature, min_new_tokens=min_tok)

    log.info("mode=%s | truncated=%s | %d chars | ~%d words",
              mode, truncated, len(response), len(response.split()))
    return GenerateResponse(response=response, truncated=truncated)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    ts   = req.timestamp or datetime.now(timezone.utc).isoformat()
    conv = [m.model_dump() for m in req.conversation]

    last_question = last_answer = ""
    for msg in reversed(conv):
        if not last_answer   and msg["role"] == "assistant": last_answer   = msg["content"]
        if not last_question and msg["role"] == "user":      last_question = msg["content"]
        if last_question and last_answer:
            break

    record = {
        "feedback_id":      feedback_id,
        "message_id":       req.message_id,
        "feedback_type":    req.feedback_type,
        "model_version":    req.model_version,
        "timestamp":        ts,
        "question":         last_question,
        "candidate":        last_answer,
        "reference":        req.correction or "",
        "has_correction":   bool(req.correction),
        "used_in_training": False,
        "bertscore":        None,
        "conversation":     conv,
    }

    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Feedback %s (%s) saved.", feedback_id, req.feedback_type)
    return FeedbackResponse(status="logged", feedback_id=feedback_id)
