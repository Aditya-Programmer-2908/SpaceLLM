import json
import logging
import re
import uuid
from collections import Counter
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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are SpaceLLM, an expert AI assistant for space missions,
astronomy, satellites, rockets, planetary science,
and aerospace engineering.
Your goal is to help users understand space topics clearly.

Response Style Rules:
1. Match the user's desired depth.
   - Short factual questions → concise answer (2-4 sentences).
   - Questions containing: explain, details, tell me about, how, why,
     compare, teach, elaborate, overview, describe, history
     → provide a thorough multi-paragraph explanation.
2. NEVER describe what you are going to write. Write it directly.
   Forbidden opener phrases:
   - "This overview covers..."  "The following discusses..."
   - "Details are provided below..."  "As follows:"  "Listed below:"
3. Thorough answers must contain:
   - Introduction, main explanation, key facts, historical significance.
4. Prefer natural educational language.
5. Never reveal chain of thought.
6. If uncertain, say so briefly.
7. If outside the space domain, say:
   "I specialise in space missions. Please consult a general-purpose assistant."
"""

DETAIL_ADDENDUM = """
ADDITIONAL REQUIREMENT FOR THIS RESPONSE:
The user is asking for a thorough explanation. You must:
- Write at least 500 words.
- Use multiple paragraphs with natural flow.
- Cover background, technical details, key facts, scientific significance,
  historical impact, and legacy.
- Do NOT use bullet lists as a replacement for explanation.
- Do NOT open with any meta-phrase like "This covers..." or "Below is...".
- Start writing the actual explanation immediately.
"""

EXPAND_MSG = {
    "role": "user",
    "content": (
        "Your previous answer is too short or cut off. "
        "Continue and expand substantially — cover background, technical details, "
        "scientific significance, historical impact. "
        "Do not repeat your opening sentence. Write directly."
    ),
}

RETRY_MSG = {
    "role": "user",
    "content": (
        "You described what you would explain but did not explain it. "
        "Write the actual explanation now. No 'below', no 'as follows'. Start immediately."
    ),
}

# ── Detection patterns ─────────────────────────────────────────────────────────

EMPTY_PROMISE_PATTERNS = [
    "below is", "are provided below", "is provided below",
    "following table", "the following list", "the timeline below",
    "listed below", "as follows", "are as follows",
]

INCOMPLETE_PATTERNS = [
    "this overview covers", "this detailed overview covers",
    "the following discusses", "details are provided below",
    "the report covers", "this explanation covers",
    "the following topics", "the following sections",
    "a detailed summary", "the mission involved",
    "the following goals", "the following specific goals",
    "these objectives included", "which were achieved",
    "were as follows", "are listed below",
    "can be summarized", "are outlined below",
    "the following table", "the following points",
]

SUSPICIOUS_ENDINGS = [
    "as follows:", "below:", "the following:", "etc.",
    "including:", "such as:", "namely:", "these include:",
    "which include:", "are:", "follows:",
]

LOOP_PATTERNS = [
    "a significant achievement in the space and the mission",
    "a major achievement in the space and the mission",
    "which is a significant achievement in the space",
    "a major milestone in space exploration, as it demonstrated",
    "the mission's success was a major milestone",
]

# Keywords that signal the user explicitly wants brevity — override detail path
SHORT_INTENT = [
    "in short", "briefly", "brief", "quick", "quickl",
    "summarize", "tldr", "tl;dr", "one line", "one sentence",
    "short answer", "concise", "in brief", "just tell me",
    "short", "give me a short", "keep it short",
]

# Keywords that signal the user wants depth
DETAIL_KEYWORDS = [
    "detail", "details", "detailed", "explain", "tell me about",
    "teach", "how", "why", "compare", "history", "complete", "full",
    "elaborate", "more about", "research", "study", "deep", "in depth",
    "describe", "what is", "what are", "objective", "objectives",
    "overview", "background", "tell me", "about",
]

# A detailed response must have at least this many words; below triggers expansion
DETAIL_MIN_WORDS = 400

model     = None
tokenizer = None
pipe      = None


# ── Model lifecycle ────────────────────────────────────────────────────────────

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

    device = "cuda:0"
    log.info("Loading base model %s on %s ...", BASE_MODEL_ID, device)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map={"": device},
        trust_remote_code=True,
    )
    log.info("Base model loaded. dtype=%s", next(_base.parameters()).dtype)

    _base.config.tie_word_embeddings = False
    lm_head = _base.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    log.info("lm_head weight untied and cloned.")

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
    log.info("SpaceLLM_v1 ready!")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    yield

    log.info("Shutting down.")
    del model, tokenizer, pipe
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
    max_new_tokens: int   = 768
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


# ── Helper functions ───────────────────────────────────────────────────────────

def needs_detailed_response(user_messages: list[dict]) -> bool:
    """
    Check only raw user messages (before system injection).
    Short-intent keywords always override detail keywords.
    """
    text = " ".join(
        m["content"].lower() for m in user_messages[-3:] if m["role"] == "user"
    )
    if any(kw in text for kw in SHORT_INTENT):
        log.info("detail=False (short-intent override) | %.100s", text)
        return False
    result = any(kw in text for kw in DETAIL_KEYWORDS)
    log.info("detail=%s | %.100s", result, text)
    return result


def build_messages(raw_user_msgs: list[dict], detailed: bool) -> list[dict]:
    """
    Construct the final message list:
      [system_prompt (with detail addendum baked in if needed)] + [user conversation]
    The detail addendum lives in the SYSTEM message — never appended mid-conversation.
    """
    system_content = SYSTEM_PROMPT.strip()
    if detailed:
        system_content += "\n\n" + DETAIL_ADDENDUM.strip()

    # Strip any system messages the client sent (we own the system prompt)
    conv = [m for m in raw_user_msgs if m["role"] != "system"]
    return [{"role": "system", "content": system_content}] + conv


def clean_response(text: str) -> str:
    """Strip GPT-OSS reasoning artifacts."""
    if not text:
        return ""
    text = text.strip()
    if text.lower().startswith("analysis"):
        parts = re.split(r"\bfinal\b", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[1].strip()
    text = re.sub(r"^(final\s*)+", "", text, flags=re.IGNORECASE).strip()
    bad_starts = ("the assistant", "assistant will", "assistant should",
                  "assistant can", "assistant must")
    lines = [l for l in text.splitlines()
             if not l.strip().lower().startswith(bad_starts)]
    return "\n".join(lines).strip()


def is_empty_promise(text: str) -> bool:
    """True if the model promised to list content but produced nothing."""
    if not text:
        return True
    lower = text.lower()
    has_promise = any(p in lower for p in EMPTY_PROMISE_PATTERNS)
    real_lines  = [l for l in text.splitlines() if l.strip()]
    return has_promise and len(real_lines) < 6


def looks_incomplete(text: str, detailed: bool) -> bool:
    """True if the response looks cut-off or too thin for what was asked."""
    if not text or not text.strip():
        return True
    lower = text.lower().strip()
    if any(p in lower for p in INCOMPLETE_PATTERNS):
        log.info("looks_incomplete: INCOMPLETE_PATTERN matched")
        return True
    if any(lower.endswith(e) for e in SUSPICIOUS_ENDINGS):
        log.info("looks_incomplete: SUSPICIOUS_ENDING matched")
        return True
    if detailed and len(text.split()) < DETAIL_MIN_WORDS:
        log.info("looks_incomplete: word_count=%d < %d", len(text.split()), DETAIL_MIN_WORDS)
        return True
    return False


def is_looping(text: str) -> bool:
    """True if the model is repeating itself."""
    lower = text.lower()
    if any(p in lower for p in LOOP_PATTERNS):
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]", lower) if len(s.strip()) > 30]
    if sentences:
        freq = Counter(sentences).most_common(1)[0][1]
        if freq >= 3:
            return True
    return False


def truncate_at_loop(text: str) -> str:
    """Cut text at the first sentence that repeats."""
    parts  = re.split(r"(?<=[.!?])\s+", text)
    seen   = set()
    result = []
    for sent in parts:
        key = sent.strip().lower()
        if key in seen and len(key) > 40:
            log.info("truncate_at_loop: cut at repeated sentence")
            break
        seen.add(key)
        result.append(sent)
    return " ".join(result).strip()


def run_pipe(
    messages:           list[dict],
    req:                GenerateRequest,
    temperature:        float,
    min_new_tokens:     int   = 10,
    repetition_penalty: float = 1.15,
) -> str:
    result = pipe(
        messages,
        max_new_tokens=max(req.max_new_tokens, 1024),  # no upper cap
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=req.top_p if req.do_sample else 1.0,
        do_sample=req.do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    raw = result[0]["generated_text"]
    if isinstance(raw, list):
        raw = raw[-1].get("content", "")
    log.info("\n=== RAW ===\n%s\n===========", raw)
    return clean_response(raw)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    ready = pipe is not None
    return {"status": "ok" if ready else "loading", "model": ADAPTER_MODEL_ID, "ready": ready}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    # Work on raw user messages BEFORE any system injection
    raw = [m.model_dump() for m in req.messages]
    detailed = needs_detailed_response(raw)

    # Build final message list with correct system prompt
    msgs = build_messages(raw, detailed)

    temperature = req.temperature if req.do_sample else 1.0

    # Detailed responses get a minimum token floor so the model can't bail early;
    # default responses have no artificial floor (model stops when done naturally).
    min_tok = 300 if detailed else 10

    # ── Pass 1: initial generation ────────────────────────────────────────────
    response = run_pipe(msgs, req, temperature, min_new_tokens=min_tok)

    if is_looping(response):
        log.warning("Pass 1: loop detected — truncating")
        response = truncate_at_loop(response)

    # ── Pass 2: expand if still too short / incomplete ────────────────────────
    if looks_incomplete(response, detailed):
        log.warning("Pass 2: incomplete — expanding")
        expand_msgs = msgs + [
            {"role": "assistant", "content": response},
            EXPAND_MSG,
        ]
        continuation = run_pipe(expand_msgs, req, temperature=0.4,
                                min_new_tokens=300)
        if is_looping(continuation):
            continuation = truncate_at_loop(continuation)

        # If pass 1 was pure meta-description, replace; otherwise append
        if any(p in response.lower() for p in INCOMPLETE_PATTERNS):
            response = continuation
        else:
            response = response.rstrip() + "\n\n" + continuation

    # ── Pass 3: retry if model made an empty promise ──────────────────────────
    if is_empty_promise(response):
        log.warning("Pass 3: empty promise — retrying")
        retry_msgs = msgs + [
            {"role": "assistant", "content": response},
            RETRY_MSG,
        ]
        response = run_pipe(retry_msgs, req, temperature=0.3,
                            min_new_tokens=300)
        if is_looping(response):
            response = truncate_at_loop(response)

    log.info("Final response: %d chars, ~%d words", len(response), len(response.split()))
    return GenerateResponse(response=response)


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
