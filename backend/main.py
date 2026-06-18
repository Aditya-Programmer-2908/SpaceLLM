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
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("spacellm")

FEEDBACK_LOG     = Path("feedback_log.jsonl")

# ── MongoDB ───────────────────────────────────────────────────────────────
MONGO_URI        = "mongodb+srv://adityapratapusingh:Aditya2984@cluster0.ss7p3.mongodb.net/?appName=Cluster0"
MONGO_DB_NAME    = "SpaceLLM"

# Three collections:
#   feedback      — all feedback records (positive + negative)
#   corrections   — only negative feedback with user correction
#                   (question + LLM answer + user correction)
#                   used as candidate/reference pairs for BERTScore
#   conversations — full conversation history per session

try:
    _mongo_client  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    _mongo_client.server_info()
    _mongo_db      = _mongo_client[MONGO_DB_NAME]
    _feedback_col  = _mongo_db["feedback"]
    _correction_col = _mongo_db["corrections"]
    _conv_col      = _mongo_db["conversations"]
    log.info("MongoDB connected: %s", MONGO_DB_NAME)
except Exception as _e:
    log.warning("MongoDB connection failed: %s — falling back to JSONL only", _e)
    _mongo_client   = None
    _feedback_col   = None
    _correction_col = None
    _conv_col       = None

BASE_MODEL_ID    = "openai/gpt-oss-20b"
ADAPTER_MODEL_ID = "AdityaPS/SpaceLLM_v1"
SYSTEM_PROMPT    = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "Answer any questions related to the space missions conducted by ISRO (Indian Space Research Organisation), "
    "NASA (National Aeronautics and Space Administration) and ESA (European Space Agency). "
    "Carefully consider the user's question and provide a detailed, accurate answer based on your extensive knowledge "
    "(you are fine tuned on the data of the missions of the NASA, ISRO and ESA). "
    "Provide accurate, concise, technically rigorous answers. "
    "If a question is outside the space domain, politely redirect the user. "
    "If you are unsure about an answer, clearly state that you are uncertain and provide the best possible information "
    "based on your knowledge. Also politely ask the user to provide feedback if the answer was helpful or not, "
    "so that you can learn and improve over time."
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

    device = "cuda:0"
    log.info("Loading base model %s  [native MXFP4] on %s ...", BASE_MODEL_ID, device)
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
    log.info("Vocab after resize: %d  (padded to multiple of 64)", actual_vocab)

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
    correction:    str | None = None   # user's corrected answer (reference)
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
        response_text = response_text[-1].get("content", "")
    response_text = response_text.strip()

    log.info("Response generated (%d chars).", len(response_text))
    return GenerateResponse(response=response_text)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    ts = req.timestamp or datetime.now(timezone.utc).isoformat()

    # ── Extract question and LLM answer from conversation ─────────────────
    # conversation is the full history: [user, assistant, user, assistant ...]
    # We want the LAST user question and the LAST assistant answer
    # (the one the user is giving feedback on)
    conv = [m.model_dump() for m in req.conversation]

    last_question  = ""
    last_llm_answer = ""
    for msg in reversed(conv):
        if not last_llm_answer and msg["role"] == "assistant":
            last_llm_answer = msg["content"]
        if not last_question and msg["role"] == "user":
            last_question = msg["content"]
        if last_question and last_llm_answer:
            break

    # ── Base feedback record (stored in 'feedback' collection) ────────────
    feedback_record = {
        "feedback_id":   feedback_id,
        "message_id":    req.message_id,
        "feedback_type": req.feedback_type,
        "model_version": req.model_version,
        "timestamp":     ts,
        "question":      last_question,
        "llm_answer":    last_llm_answer,   # candidate answer for BERTScore
        "correction":    req.correction,    # reference answer for BERTScore (if provided)
        "conversation":  conv,
    }

    # ── Correction record (stored in 'corrections' collection) ────────────
    # Only saved when user flagged as negative — this is the MAPE-K training data
    # Schema designed for BERTScore: candidate = llm_answer, reference = correction
    correction_record = None
    if req.feedback_type == "negative" and last_question:
        correction_record = {
            "correction_id":  str(uuid.uuid4()),
            "feedback_id":    feedback_id,
            "timestamp":      ts,
            "model_version":  req.model_version,

            # Core fields for BERTScore evaluation
            "question":       last_question,        # what the user asked
            "candidate":      last_llm_answer,      # what the LLM said (to evaluate)
            "reference":      req.correction or "",  # what the user says it should be

            # Training pipeline fields
            "has_correction": bool(req.correction),  # True = has reference for retraining
            "used_in_training": False,                # flipped to True after v2 fine-tuning
            "bertscore":      None,                   # filled in by Analyse phase
        }

    # ── Save to JSONL backup ──────────────────────────────────────────────
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_record) + "\n")

    # ── Save to MongoDB ───────────────────────────────────────────────────
    mongo_status = "jsonl_only"
    if _feedback_col is not None:
        try:
            _feedback_col.insert_one({**feedback_record})
            mongo_status = "mongodb+jsonl"

            if correction_record is not None:
                _correction_col.insert_one({**correction_record})
                log.info(
                    "Correction saved — question: '%s...' | has_reference: %s",
                    last_question[:60], correction_record["has_correction"]
                )

            log.info("Feedback %s (%s) saved to MongoDB.", feedback_id, req.feedback_type)

        except PyMongoError as e:
            log.warning("MongoDB write failed: %s — JSONL only.", e)
    else:
        log.info("Feedback %s saved to JSONL only (MongoDB unavailable).", feedback_id)

    return FeedbackResponse(status=mongo_status, feedback_id=feedback_id)