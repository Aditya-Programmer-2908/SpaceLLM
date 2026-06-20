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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")
BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"

# ── System prompts ─────────────────────────────────────────────────────────────

# Default: short, factual, direct. 50-200 words.
SYSTEM_PROMPT_SHORT = """\
You are SpaceLLM, a precise expert assistant specialising in space missions,
astronomy, satellites, launch vehicles, planetary science, and aerospace engineering.

Your knowledge is grounded in information from NASA, ISRO, ESA, JAXA, Roscosmos,
and other reputable space agencies.

RESPONSE RULES:
- Answer directly and factually.
- No preamble, filler, roleplay, or unnecessary commentary.
- Also respond to greetings gracefully in 10–30 words (e.g. hi, hello, thanks, sorry).
- Keep responses between 50 and 200 words.
- Accuracy is more important than completeness.
- Never invent mission names, spacecraft, satellites, instruments, payloads,
  launch vehicles, dates, technical specifications, or scientific results.
- Do not guess.
- If a detail is uncertain, explicitly say:
  "I am not certain about that detail."
- Prefer omitting uncertain information rather than speculating.
- Do not transfer instruments or payloads between different missions.
- Verify internally that mission names, payloads, launch vehicles, and dates
  are consistent before responding.

FINAL FACT CHECK (internal only):
Before responding, verify:
- mission names
- spacecraft names
- satellite names
- instrument names
- launch vehicles
- dates
- numerical values

Remove any claim that is uncertain.

If the question is outside the space domain, say exactly:
"I specialise in space missions. Please consult a general-purpose assistant."

Never reveal your reasoning process or fact-checking process.
"""

# Detailed: chain-of-thought reasoning internally, high-quality answer externally.
SYSTEM_PROMPT_DETAIL = """\
You are SpaceLLM, a deep-knowledge expert assistant specialising in space missions,
astronomy, satellites, launch vehicles, planetary science, astrophysics,
and aerospace engineering.

Your knowledge is grounded in information from NASA, ISRO, ESA, JAXA,
Roscosmos, CNSA, scientific mission archives, and peer-reviewed space science sources.

INTERNAL THINKING PROCESS (never shown to user):

Before writing an answer:

1. Identify the mission, spacecraft, satellite, celestial body, or concept being discussed.
2. Determine what the user actually wants to know.
3. Recall relevant facts, dates, agencies, spacecraft, instruments,
   orbital parameters, scientific findings, and mission outcomes.
4. Identify common misconceptions associated with the topic.
5. Organize the explanation logically:
   background → technical details → scientific objectives →
   mission results → significance.
6. Verify internally that all mission names, spacecraft names,
   payloads, instruments, launch vehicles, and dates are consistent.
7. Remove any claim that is uncertain.

FACTUAL RELIABILITY RULES:

- Accuracy is more important than completeness.
- Never invent:
  - mission names
  - spacecraft names
  - satellite names
  - payloads
  - instruments
  - launch vehicles
  - launch dates
  - orbital parameters
  - scientific discoveries
  - numerical values

- Never transfer payloads or instruments between missions.
- Never infer facts solely from similar missions.
- If a fact is uncertain, explicitly state uncertainty.
- Prefer omission over speculation.
- Avoid confident statements unless the fact is well established.

SPACE-DOMAIN VALIDATION RULES:

Before mentioning an instrument, confirm it belongs to the mission.

Before mentioning a launch vehicle, confirm it launched the mission.

Before mentioning a date, confirm the event and date correspond.

Before mentioning a scientific discovery, confirm the mission actually made it.

Examples of mistakes to avoid:
- Assigning MOXIE to Mangalyaan.
- Assigning Perseverance instruments to Curiosity.
- Assigning Aditya-L1 instruments that do not exist.
- Confusing landing dates with launch dates.
- Confusing orbital insertion dates with landing dates.
- Confusing developmental and operational launch vehicle flights.

OUTPUT RULES:

- Write a technically accurate educational explanation.
- Aim for 150–500 words.
- Stop when the explanation is complete.
- Never add details merely to increase length.
- Use flowing paragraphs.
- No bullet lists unless explicitly requested.
- Open directly with substance.
- Include specific mission names, spacecraft, agencies,
  instruments, dates, and scientific outcomes when relevant.
- Explain significance and context where useful.
- Clearly distinguish confirmed facts from uncertain information.

FINAL FACT CHECK (internal only):

Review every:
- mission name
- spacecraft name
- satellite name
- instrument name
- payload name
- launch vehicle
- date
- numerical value
- scientific result

Remove or revise any statement that is uncertain.

If the question is outside the space domain, say exactly:
"I specialise in space missions. Please consult a general-purpose assistant."

Never reveal chain-of-thought, internal reasoning, verification steps,
or fact-checking procedures.
"""

# Keywords that force the short path regardless of anything else
SHORT_INTENT = {
    "in short", "briefly", "brief", "quick", "summarize", "summary",
    "tldr", "tl;dr", "one line", "one sentence", "short answer",
    "concise", "in brief", "just tell me", "short", "give me a short",
    "keep it short",
}

# Keywords that trigger the detailed path
DETAIL_INTENT = {
    "explain", "tell me about", "describe", "detail", "details", "detailed",
    "how does", "how do", "how did", "why did", "why does", "why is",
    "what is", "what are", "what was", "what were",
    "teach me", "elaborate", "compare", "history", "background",
    "overview", "in depth", "deep dive", "walk me through", "about",
    "tell me", "how", "why",
}

model     = None
tokenizer = None
pipe      = None


# ── Model lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, pipe

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
    max_new_tokens: int   = 1024
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


# ── Core logic ─────────────────────────────────────────────────────────────────

def classify_request(messages: list[dict]) -> str:
    """
    Returns 'short' or 'detail' by scanning the last 3 user turns.
    SHORT_INTENT wins if present; then DETAIL_INTENT; else 'short'.
    """
    text = " ".join(
        m["content"].lower()
        for m in messages[-3:]
        if m["role"] == "user"
    )
    if any(kw in text for kw in SHORT_INTENT):
        log.info("classify=short (short-intent keyword) | %.120s", text)
        return "short"
    if any(kw in text for kw in DETAIL_INTENT):
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


def run_pipe(messages: list[dict], req: GenerateRequest, temperature: float,
             min_new_tokens: int) -> str:
    result = pipe(
        messages,
        max_new_tokens=req.max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=req.top_p if req.do_sample else 1.0,
        do_sample=req.do_sample,
        repetition_penalty=1.1,
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

    raw  = [m.model_dump() for m in req.messages]
    mode = classify_request(raw)
    msgs = build_messages(raw, mode)

    temperature = req.temperature if req.do_sample else 1.0

    # Token floors:
    # - short: 30 tokens minimum (enough for 1-2 solid sentences, prevents empty output)
    # - detail: 400 tokens minimum (~300 words), model stops naturally when complete
    min_tok = 400 if mode == "detail" else 30

    response = run_pipe(msgs, req, temperature, min_new_tokens=min_tok)

    log.info("mode=%s | %d chars | ~%d words", mode, len(response), len(response.split()))
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
