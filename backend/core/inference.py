"""
core/inference.py
-----------------
Loads the SpaceLLM LoRA adapter on top of the base model and exposes a single
`generate()` coroutine that the API routes call.

Loading strategy
~~~~~~~~~~~~~~~~
1. Load base model (BF16, device_map=auto for multi-GPU)
2. Pull the active LoRA adapter from HuggingFace
3. Merge weights only when GPU memory allows; otherwise keep PEFT wrapper

The module keeps one global ModelState so the model is loaded once at startup
and reloaded only when a new adapter version is pushed.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from config import settings

logger = logging.getLogger(__name__)

# ── System prompt injected before every conversation ───────────────────────
SYSTEM_PROMPT = (
    "You are SpaceLLM, an expert AI assistant specialising in space missions, "
    "astronomy, and aerospace engineering. Provide accurate, detailed, and "
    "scientifically correct answers. Use equations and technical terms where "
    "appropriate. If you are unsure, say so rather than speculating."
)


@dataclass
class ModelState:
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    current_version: str = "unloaded"
    loading: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_state = ModelState()


# ── Public API ──────────────────────────────────────────────────────────────

async def load_model(adapter_repo: Optional[str] = None) -> None:
    """Load (or hot-swap) the SpaceLLM model + LoRA adapter.

    Parameters
    ----------
    adapter_repo : str, optional
        HuggingFace repo ID of the adapter to load.
        Defaults to ``settings.HF_REPO_ID``.
    """
    async with _state._lock:
        if _state.loading:
            logger.info("Model already loading — skipping duplicate call.")
            return
        _state.loading = True

    repo = adapter_repo or settings.HF_REPO_ID
    logger.info("Loading SpaceLLM — base: %s  adapter: %s", settings.BASE_MODEL_ID, repo)

    try:
        tok = AutoTokenizer.from_pretrained(
            settings.BASE_MODEL_ID,
            token=settings.HF_TOKEN or None,
            trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        load_kwargs: dict = dict(
            pretrained_model_name_or_path=settings.BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=settings.DEVICE_MAP,
            trust_remote_code=True,
            token=settings.HF_TOKEN or None,
        )
        if settings.LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig  # lazy import
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        model = PeftModel.from_pretrained(
            base, repo,
            token=settings.HF_TOKEN or None,
        )
        model.eval()

        _state.tokenizer = tok
        _state.model = model
        _state.current_version = repo.split("/")[-1]  # e.g. "SpaceLLM_v1"
        logger.info("Model ready — version: %s", _state.current_version)

    except Exception as exc:
        logger.error("Failed to load model: %s", exc, exc_info=True)
        raise
    finally:
        _state.loading = False


async def generate(
    messages: list[dict],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> tuple[str, float]:
    """Run inference.

    Parameters
    ----------
    messages : list of {"role": str, "content": str}
    max_new_tokens : override for this call
    temperature : override for this call

    Returns
    -------
    (response_text, latency_ms)
    """
    if _state.model is None or _state.tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    tok = _state.tokenizer
    model = _state.model

    # Build a simple chat prompt ──────────────────────────────────────────
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n"
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        prompt += f"<|{role}|>\n{m['content']}\n"
    prompt += "<|assistant|>\n"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens or settings.MAX_NEW_TOKENS,
        temperature=temperature or settings.TEMPERATURE,
        top_p=settings.TOP_P,
        do_sample=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Decode only the new tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tok.decode(new_ids, skip_special_tokens=True).strip()

    return response, latency_ms


def current_version() -> str:
    return _state.current_version


def is_loaded() -> bool:
    return _state.model is not None
