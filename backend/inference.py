"""
SpaceLLM — Retrieval-Augmented Inference + BERTScore  (OPTIMISED)
==================================================================
Replaces the old "correction adapter" inference script. There is no LoRA
adapter stacked on top of SpaceLLM_v1 anymore — the base model weights are
NEVER modified post-deployment. Corrections live in a JSON knowledge
repository (written by execute.py's RETRIEVAL_MEMORY_UPDATE handler) and
are injected into the prompt at inference time, per-query.

Speed improvements over the original serial script
----------------------------------------------------
Original:  ~3.5 min/sample  →  5291 samples ≈ 12.8 days
Optimised: batched HF (Fix 1) + async RAG prefetch (Fix 3)  →  ~2-4 hrs
           vLLM engine       (Fix 2, --engine vllm)          →  ~45 min-1.5 hrs

Key changes
-----------
1. Batched generation  — tokenizer pads a full batch; model.generate() processes
   N sequences in parallel instead of 1.  GPU utilisation jumps from ~5-10 % to
   80-95 %.  Controlled by --batch-size (default 16).

2. Async RAG prefetch — all retrieval (pure Python/CPU) runs concurrently in a
   ThreadPoolExecutor BEFORE the generation loop so retrieval latency is hidden
   behind GPU compute entirely.

3. vLLM engine (--engine vllm) — uses PagedAttention + continuous batching for
   maximum throughput.  Fires all 5291 prompts in one shot; vLLM schedules them.
   Note: vLLM does not support MXFP4, so weights load in float16/bfloat16.

Usage
-----
    # default: batched HF + async RAG (~2-4 hrs)
    python retrieval_inference_bertscore.py

    # vLLM engine (~45 min-1.5 hrs, requires: pip install vllm)
    python retrieval_inference_bertscore.py --engine vllm

    # baseline, no retrieval
    python retrieval_inference_bertscore.py --no-rag

    # run both RAG and no-RAG, report delta
    python retrieval_inference_bertscore.py --ablation

    # tuning knobs
    python retrieval_inference_bertscore.py --batch-size 32 --top-k 5 --min-similarity 0.5
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")
BACKEND_DIR = BASE_DIR / "backend"
MAPE_DIR    = BACKEND_DIR / "mape_k"

TEST_FILE = BASE_DIR / "Model_training_&_Data_Extraction/data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR   = BASE_DIR / "Model_training_&_Data_Extraction/fine_tuning_v2/outputs/bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"inference_retrieval_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("RAGEval")

sys.path.insert(0, str(MAPE_DIR))

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID             = "openai/gpt-oss-20b"
SPACELLM_V1_ADAPTER  = "AdityaPS/SpaceLLM_v1"
MAX_SEQ_LEN          = 2048
MAX_NEW_TOKENS       = 256
MAX_SAMPLES          = None         # set to int to cap eval set
DEVICE               = "cuda:0"
SMOKE_TEST_SAMPLES   = 5

DEFAULT_TOP_K          = 3
DEFAULT_MIN_SIMILARITY = 0.55
DEFAULT_BATCH_SIZE     = 16         # raise to 32 if VRAM allows; drop to 8 if OOM
RAG_PREFETCH_WORKERS   = 8          # ThreadPoolExecutor threads for async RAG


# ── Helpers ────────────────────────────────────────────────────────────────────

def log_gpu_memory(label: str = ""):
    for i in range(torch.cuda.device_count()):
        props  = torch.cuda.get_device_properties(i)
        alloc  = torch.cuda.memory_allocated(i) / 1024**3
        reserv = torch.cuda.memory_reserved(i)  / 1024**3
        total  = props.total_memory              / 1024**3
        logger.info(
            f"GPU {i} [{props.name}] {label} | "
            f"Alloc={alloc:.2f}GB  Reserved={reserv:.2f}GB  Total={total:.2f}GB"
        )


def load_test_data(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()
    data = (
        json.loads(raw)
        if raw.startswith("[")
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )
    logger.info(f"Loaded {len(data):,} test records from {path}")
    return data


def extract_messages_and_reference(record: dict) -> tuple[list[dict] | None, str | None]:
    """
    Returns the raw HF message list (untemplated) so retrieval augmentation
    can be injected into the correct user turn before chat templating.
    """
    messages    = record.get("messages", [])
    hf_messages = []
    ref_answer  = ""

    for msg in messages:
        role    = "system" if msg["role"] == "developer" else msg["role"]
        content = msg.get("content", "").strip()
        if role == "assistant":
            ref_answer = content
        elif content:
            hf_messages.append({"role": role, "content": content})

    if not hf_messages or not ref_answer:
        return None, None
    return hf_messages, ref_answer


def inject_retrieval_context(
    hf_messages: list[dict],
    repo,
    build_augmented_prompt,
    top_k: int,
    min_similarity: float,
) -> tuple[list[dict], int, str]:
    user_indices = [i for i, m in enumerate(hf_messages) if m["role"] == "user"]
    if not user_indices:
        return hf_messages, 0, ""

    last_user_idx     = user_indices[-1]
    original_question = hf_messages[last_user_idx]["content"]

    augmented     = build_augmented_prompt(original_question, repo, top_k=top_k, min_similarity=min_similarity)
    num_retrieved = 0 if augmented == original_question else top_k

    new_messages                          = [dict(m) for m in hf_messages]
    new_messages[last_user_idx]["content"] = augmented
    return new_messages, num_retrieved, original_question


def detect_adapter_vocab_size(adapter_path: str, hf_token: str | None = None) -> int:
    from safetensors import safe_open

    local_path = Path(adapter_path)
    if local_path.exists() and (local_path / "adapter_model.safetensors").exists():
        safetensors_path = local_path / "adapter_model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        safetensors_path = Path(
            hf_hub_download(repo_id=str(adapter_path), filename="adapter_model.safetensors", token=hf_token)
        )

    candidates = []
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            if "lm_head" in key:
                shape = f.get_slice(key).get_shape()
                candidates.append((key, shape))

    if not candidates:
        raise RuntimeError(f"No lm_head tensors found in {safetensors_path}.")

    key, shape = max(candidates, key=lambda kv: max(kv[1]))
    vocab = max(shape)
    logger.info(f"  Detected required vocab size {vocab:,} from tensor '{key}' shape={list(shape)}")
    return vocab


def untie_lm_head(model, label: str = ""):
    model.config.tie_word_embeddings = False
    lm_head        = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info(f"  ✅ lm_head untied {label}")


def get_input_device(model) -> torch.device:
    try:
        return next(model.get_input_embeddings().parameters()).device
    except Exception:
        return torch.device("cuda:0")


def looks_degenerate(token_ids: torch.Tensor) -> tuple[bool, str]:
    from collections import Counter
    ids = token_ids.tolist()
    if not ids:
        return True, "empty generation"
    counts            = Counter(ids)
    top_id, top_count = counts.most_common(1)[0]
    frac              = top_count / len(ids)
    if frac > 0.6:
        return True, f"token id {top_id} makes up {frac:.0%} of the generation"
    return False, ""


# ── Generation helpers ─────────────────────────────────────────────────────────

def generate_one(model, tokenizer, prompt: str) -> tuple[torch.Tensor, str]:
    """Single-sample generation — used only for smoke test and OOM fallback."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=False,
    )
    input_device = get_input_device(model)
    enc = {k: v.to(input_device) for k, v in enc.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = gen_ids[0][enc["input_ids"].shape[1]:]
    text    = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return new_ids, text


def generate_batch(model, tokenizer, prompts: list[str]) -> list[tuple[torch.Tensor, str]]:
    """
    FIX 1 — Batched generation.
    Pads a list of prompts (left-padding, already set on tokenizer) and runs
    model.generate() once for the whole batch.  GPU utilisation jumps from
    ~5-10 % (serial) to 80-95 % (batched).
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=True,   # tokenizer.padding_side == "left" set at load time
    )
    input_device = get_input_device(model)
    enc = {k: v.to(input_device) for k, v in enc.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for inp_ids, out_ids in zip(enc["input_ids"], gen_ids):
        # strip the (padded) input prefix — new tokens only
        new_ids = out_ids[inp_ids.shape[0]:]
        text    = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append((new_ids, text))
    return results


def flush_batch(
    model,
    tokenizer,
    pending_records: list[dict],
    pending_meta: list[dict],
    batch_size: int,
) -> list[tuple[str, str, dict]]:
    """
    Process one batch of prompts.  Falls back to serial generation per-sample
    if an OOM error occurs (e.g. a pathologically long prompt in the batch).
    Returns list of (reference, generated, prediction_dict).
    """
    prompts = [r["prompt"] for r in pending_records]
    try:
        batch_results = generate_batch(model, tokenizer, prompts)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM on batch of {len(prompts)} — retrying sample-by-sample")
        torch.cuda.empty_cache()
        batch_results = []
        for p in prompts:
            try:
                batch_results.append(generate_one(model, tokenizer, p))
            except Exception as e:
                logger.warning(f"  Serial fallback also failed: {e}")
                batch_results.append((torch.tensor([]), ""))

    out = []
    for (token_ids, generated), meta in zip(batch_results, pending_meta):
        out.append((
            meta["ref"],
            generated,
            {
                "sample_id":             meta["sample_id"],
                "reference":             meta["ref"],
                "generated":             generated,
                "retrieved_corrections": meta["n_retrieved"],
                "top_similarity":        meta["top_sim"],
            },
        ))
    return out


# ── FIX 3: Async RAG prefetch ─────────────────────────────────────────────────

def prefetch_all_rag(
    records: list[dict],
    repo,
    build_augmented_prompt,
    use_rag: bool,
    top_k: int,
    min_similarity: float,
    n_workers: int = RAG_PREFETCH_WORKERS,
) -> list[dict | None]:
    """
    FIX 3 — Async RAG prefetch.
    All retrieval (pure CPU/Python) runs concurrently in a thread pool BEFORE
    the generation loop starts, so retrieval latency is completely hidden behind
    GPU compute.  Returns a list of pre-computed metadata dicts in the same
    order as `records`, or None for records that should be skipped.
    """
    logger.info(f"Prefetching RAG for {len(records):,} records "
                f"({'enabled' if use_rag else 'disabled'}) "
                f"with {n_workers} threads ...")
    t0 = time.time()

    def _process(idx_record):
        idx, record = idx_record
        hf_messages, ref = extract_messages_and_reference(record)
        if hf_messages is None or ref is None:
            return idx, None

        retrieved_hits = []
        if use_rag:
            user_indices = [j for j, m in enumerate(hf_messages) if m["role"] == "user"]
            raw_q = hf_messages[user_indices[-1]]["content"] if user_indices else ""
            if raw_q:
                retrieved_hits = repo.query(raw_q, top_k=top_k, min_similarity=min_similarity)
            hf_messages, _, _ = inject_retrieval_context(
                hf_messages, repo, build_augmented_prompt, top_k, min_similarity
            )

        return idx, {
            "hf_messages":   hf_messages,
            "ref":           ref,
            "n_retrieved":   len(retrieved_hits),
            "top_sim":       round(retrieved_hits[0]["similarity"], 4) if retrieved_hits else None,
            "sample_id":     record.get("sample_id", idx),
        }

    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process, (i, r)): i for i, r in enumerate(records)}
        done = 0
        for future in as_completed(futures):
            idx, meta = future.result()
            results[idx] = meta
            done += 1
            if done % 500 == 0:
                logger.info(f"  RAG prefetch: {done}/{len(records)} done")

    elapsed = time.time() - t0
    valid   = sum(1 for r in results if r is not None)
    logger.info(f"RAG prefetch done — {valid:,} valid records in {elapsed:.1f}s")
    return results


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """
    Load SpaceLLM_v1 (base + adapter) using HuggingFace + PEFT.
    The base checkpoint's native lm_head size can drift from what v1 was
    actually trained against, so we detect v1's expected vocab size from its
    saved tensors and resize the freshly-loaded base model to match BEFORE
    attaching the adapter — otherwise PeftModel.from_pretrained fails with a
    state_dict shape mismatch on lm_head.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Detecting vocab size required by SpaceLLM_v1 adapter ...")
    required_vocab = detect_adapter_vocab_size(SPACELLM_V1_ADAPTER, hf_token=hf_token)

    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        SPACELLM_V1_ADAPTER, trust_remote_code=True, token=hf_token
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None — fix tokenizer config before running.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("  pad_token was unset — falling back to eos_token")
    tokenizer.padding_side = "left"   # required for left-padded batched generation
    logger.info(f"  vocab={tokenizer.vocab_size:,}  eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}")

    logger.info(f"Loading base model {MODEL_ID} to CPU ...")
    t0    = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        token=hf_token,
    )
    logger.info(f"  Base model loaded in {time.time()-t0:.1f}s")
    log_gpu_memory("after base model load (CPU)")

    untie_lm_head(model, "(base, pre-resize)")
    current_vocab = model.get_output_embeddings().weight.shape[0]
    if current_vocab != required_vocab:
        logger.info(f"Resizing base model vocab {current_vocab:,} → {required_vocab:,} ...")
        model.resize_token_embeddings(required_vocab)
        model.config.vocab_size = required_vocab
        if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
            untie_lm_head(model, "(re-untied after resize)")
        actual_vocab = model.get_output_embeddings().weight.shape[0]
        assert actual_vocab == required_vocab, (
            f"Vocab resize mismatch: got {actual_vocab:,}, expected {required_vocab:,}"
        )
        logger.info(f"  ✅ Vocab confirmed at {actual_vocab:,}")
    else:
        logger.info("  Base model vocab already matches SpaceLLM_v1 — no resize needed.")

    logger.info(f"Loading SpaceLLM_v1 adapter from {SPACELLM_V1_ADAPTER} ...")
    model = PeftModel.from_pretrained(
        model, SPACELLM_V1_ADAPTER, adapter_name="spacellm_v1",
        is_trainable=False, token=hf_token,
    )
    model.set_adapter("spacellm_v1")
    logger.info("  ✅ SpaceLLM_v1 adapter loaded (unmodified — no further training)")

    model.eval()
    torch.cuda.empty_cache()

    logger.info("Dispatching model across GPUs ...")
    t_dispatch = time.time()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        no_split = []
        for name, module in model.named_modules():
            cls      = type(module)
            cls_name = cls.__name__.lower()
            if (
                issubclass(cls, nn.Module) and cls is not nn.Module
                and ("layer" in cls_name or "block" in cls_name)
                and cls.__name__ not in no_split
                and sum(p.numel() for p in module.parameters()) > 1_000_000
            ):
                no_split.append(cls.__name__)
        no_split = list(dict.fromkeys(no_split))
        logger.info(f"  no_split_module_classes: {no_split}")

        n_gpus     = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            free       = torch.cuda.mem_get_info(i)[0]
            alloc      = max(0, free - 4 * 1024**3)
            max_memory[i] = f"{int(alloc / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"  max_memory per device: {max_memory}")

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
        model      = dispatch_model(model, device_map=device_map)
        logger.info(f"  dispatch_model done in {time.time()-t_dispatch:.1f}s")

        if hasattr(model, "hf_device_map"):
            from collections import Counter
            dev_counts = Counter(str(v) for v in model.hf_device_map.values())
            for dev, count in sorted(dev_counts.items()):
                logger.info(f"    {dev} : {count} modules")

    except Exception as e:
        logger.warning(f"  dispatch_model failed ({e}) — falling back to single GPU {DEVICE}")
        model = model.to(DEVICE)

    log_gpu_memory("after dispatch")
    return model, tokenizer


# ── vLLM engine (Fix 2) ────────────────────────────────────────────────────────

def run_eval_vllm(
    test_records: list[dict],
    use_rag: bool,
    repo,
    build_augmented_prompt,
    top_k: int,
    min_similarity: float,
    tag: str,
) -> dict:
    """
    FIX 2 — vLLM-backed eval.
    Uses PagedAttention + continuous batching.  Fires all prompts in one call;
    vLLM schedules them with no idle GPU time between sequences.
    Requires:  pip install vllm
    Note: vLLM does not support MXFP4 — loads weights as float16.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Initialising vLLM engine ...")
    llm = LLM(
        model=MODEL_ID,
        enable_lora=True,
        max_lora_rank=64,           # adjust to match your adapter's rank
        gpu_memory_utilization=0.90,
        max_model_len=MAX_SEQ_LEN,
        dtype="float16",            # vLLM does not support MXFP4
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        tokenizer=SPACELLM_V1_ADAPTER,
        trust_remote_code=True,
    )
    lora_req = LoRARequest("spacellm_v1", 1, SPACELLM_V1_ADAPTER)
    sampling  = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    # -- Async RAG prefetch (same as HF path) ---------------------------------
    prefetched = prefetch_all_rag(
        test_records, repo, build_augmented_prompt, use_rag, top_k, min_similarity
    )

    # -- Build tokenizer for chat template ------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        SPACELLM_V1_ADAPTER, trust_remote_code=True, token=hf_token
    )

    all_prompts = []
    valid_meta  = []
    skipped     = 0

    for i, meta in enumerate(prefetched):
        if meta is None:
            skipped += 1
            continue
        prompt = tokenizer.apply_chat_template(
            meta["hf_messages"], tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)
        valid_meta.append(meta)

    logger.info(f"Sending {len(all_prompts):,} prompts to vLLM (skipped={skipped}) ...")
    t_inf   = time.time()
    outputs = llm.generate(all_prompts, sampling, lora_request=lora_req)
    inf_time = time.time() - t_inf
    logger.info(f"vLLM generation done in {inf_time/60:.1f} min")

    references, hypotheses, predictions = [], [], []
    for output, meta in zip(outputs, valid_meta):
        generated = output.outputs[0].text.strip()
        references.append(meta["ref"])
        hypotheses.append(generated)
        predictions.append({
            "sample_id":             meta["sample_id"],
            "reference":             meta["ref"],
            "generated":             generated,
            "retrieved_corrections": meta["n_retrieved"],
            "top_similarity":        meta["top_sim"],
        })

    return _compute_and_save_metrics(
        references, hypotheses, predictions,
        test_records, skipped, inf_time,
        use_rag, top_k, min_similarity,
        repo, tag,
        engine="vllm",
    )


# ── HF engine eval (Fix 1 + Fix 3) ────────────────────────────────────────────

def run_eval(
    model,
    tokenizer,
    test_records: list[dict],
    use_rag: bool,
    repo,
    build_augmented_prompt,
    top_k: int,
    min_similarity: float,
    tag: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    HuggingFace-backed eval with FIX 1 (batched generation) + FIX 3 (async
    RAG prefetch).  Falls back to serial generation on OOM.
    """

    # ── Smoke test (always serial for clear error messages) ──────────────────
    logger.info(f"\n── Smoke test ({SMOKE_TEST_SAMPLES} samples, tag={tag}) ──────────")
    smoke_ok = True
    for i, record in enumerate(test_records[:SMOKE_TEST_SAMPLES]):
        hf_messages, ref = extract_messages_and_reference(record)
        if hf_messages is None:
            continue
        if use_rag:
            hf_messages, _, _ = inject_retrieval_context(
                hf_messages, repo, build_augmented_prompt, top_k, min_similarity
            )
        prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        token_ids, text = generate_one(model, tokenizer, prompt)
        bad, reason = looks_degenerate(token_ids)
        logger.info(f"  [{i}] {text[:100]!r}")
        if bad:
            logger.error(f"  ❌ Sample {i} degenerate: {reason}")
            smoke_ok = False

    if not smoke_ok:
        raise RuntimeError(
            "Smoke test detected degenerate generation. Since SpaceLLM_v1's "
            "weights are never modified, this points to a loading/checkpoint "
            "problem — check the adapter path and tokenizer."
        )
    logger.info("  ✅ Smoke test passed — proceeding to full eval\n")

    # ── FIX 3: Async RAG prefetch for all records ────────────────────────────
    prefetched = prefetch_all_rag(
        test_records, repo, build_augmented_prompt, use_rag, top_k, min_similarity
    )

    # ── FIX 1: Batched generation ────────────────────────────────────────────
    logger.info(
        f"Running batched inference — {len(test_records):,} samples  "
        f"batch_size={batch_size}  RAG={use_rag}  tag={tag}"
    )

    references, hypotheses, predictions = [], [], []
    skipped         = 0
    pending_records = []   # list of {"prompt": str}
    pending_meta    = []   # list of metadata dicts
    t_inf           = time.time()
    n_batches       = 0

    for i, meta in enumerate(prefetched):
        if meta is None:
            skipped += 1
            continue

        prompt = tokenizer.apply_chat_template(
            meta["hf_messages"], tokenize=False, add_generation_prompt=True
        )
        pending_records.append({"prompt": prompt})
        pending_meta.append(meta)

        # Flush when batch is full
        if len(pending_records) >= batch_size:
            for ref_ans, gen, pred in flush_batch(model, tokenizer, pending_records, pending_meta, batch_size):
                references.append(ref_ans)
                hypotheses.append(gen)
                predictions.append(pred)
            n_batches      += 1
            pending_records = []
            pending_meta    = []

            if n_batches % 10 == 0:
                elapsed  = time.time() - t_inf
                done_cnt = len(predictions)
                rate     = done_cnt / elapsed
                eta      = (len(test_records) - i - 1) / rate if rate > 0 else float("inf")
                logger.info(
                    f"  Batch {n_batches} | valid={done_cnt}  skipped={skipped}  "
                    f"speed={rate:.2f} it/s  ETA={eta/60:.1f} min"
                )
            log_gpu_memory(f"batch {n_batches}")

    # Flush remainder
    if pending_records:
        for ref_ans, gen, pred in flush_batch(model, tokenizer, pending_records, pending_meta, batch_size):
            references.append(ref_ans)
            hypotheses.append(gen)
            predictions.append(pred)

    inf_time = time.time() - t_inf
    logger.info(
        f"\nInference done (tag={tag}) — {len(predictions):,} samples  "
        f"skipped={skipped}  time={inf_time/60:.1f} min"
    )

    return _compute_and_save_metrics(
        references, hypotheses, predictions,
        test_records, skipped, inf_time,
        use_rag, top_k, min_similarity,
        repo, tag,
        engine=f"hf_batch{batch_size}",
    )


# ── Shared metrics computation ─────────────────────────────────────────────────

def _compute_and_save_metrics(
    references: list[str],
    hypotheses: list[str],
    predictions: list[dict],
    test_records: list[dict],
    skipped: int,
    inf_time: float,
    use_rag: bool,
    top_k: int,
    min_similarity: float,
    repo,
    tag: str,
    engine: str,
) -> dict:
    if not predictions:
        logger.error("No predictions generated — cannot compute metrics.")
        return {}

    raw_path = OUT_DIR / f"predictions_raw_{tag}_{RUN_ID}.json"
    raw_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Raw predictions saved → {raw_path}")

    # ── Token F1 ──────────────────────────────────────────────────────────────
    token_f1s = []
    for hyp, ref in zip(hypotheses, references):
        ref_toks = set(ref.lower().split())
        gen_toks = set(hyp.lower().split())
        if ref_toks or gen_toks:
            overlap = len(ref_toks & gen_toks)
            prec = overlap / len(gen_toks) if gen_toks else 0.0
            rec  = overlap / len(ref_toks) if ref_toks else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        token_f1s.append(f1)
    mean_token_f1 = sum(token_f1s) / len(token_f1s)
    logger.info(f"Token F1 mean: {mean_token_f1:.4f}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    logger.info("\nComputing BERTScore ...")
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en", model_type="roberta-large",
        batch_size=64,        # higher than default — BERTScore is fast on GPU
        verbose=True,
        device=DEVICE,
    )
    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    f1_list = F1.tolist()
    logger.info(f"BERTScore  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}")

    for i, pred in enumerate(predictions):
        pred["bert_f1"]  = round(f1_list[i], 4)
        pred["token_f1"] = round(token_f1s[i], 4)

    results = {
        "run_id":       RUN_ID,
        "tag":          tag,
        "engine":       engine,
        "base_model":   MODEL_ID,
        "adapter":      SPACELLM_V1_ADAPTER,
        "adapter_note": (
            "SpaceLLM_v1 only — never modified post-deployment. "
            "Corrections applied via retrieval, not retraining."
        ),
        "rag_enabled": use_rag,
        "rag_config": {
            "top_k":          top_k,
            "min_similarity": min_similarity,
            "repo_size":      len(repo) if use_rag else None,
        },
        "config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_seq_len":    MAX_SEQ_LEN,
        },
        "summary": {
            "total_records":      len(test_records),
            "evaluated":          len(predictions),
            "skipped":            skipped,
            "inference_time_min": round(inf_time / 60, 2),
        },
        "bert_score": {
            "model":     "roberta-large",
            "precision": round(mean_p, 4),
            "recall":    round(mean_r, 4),
            "f1":        round(mean_f1, 4),
        },
        "token_f1_mean": round(mean_token_f1, 4),
        "per_sample":    predictions,
    }

    out_path = OUT_DIR / f"bertscore_results_{tag}_{RUN_ID}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Full results saved ({tag}) → {out_path}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpaceLLM retrieval-augmented inference + BERTScore (optimised)")
    parser.add_argument(
        "--engine", choices=["hf", "vllm"], default="hf",
        help="hf = batched HuggingFace (default, ~2-4 hrs);  vllm = vLLM engine (~45 min-1.5 hrs)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for HF engine (default {DEFAULT_BATCH_SIZE}). Ignored when --engine vllm.",
    )
    parser.add_argument("--no-rag",     action="store_true", help="Disable retrieval augmentation (baseline run).")
    parser.add_argument("--ablation",   action="store_true", help="Run both with and without RAG, report the delta.")
    parser.add_argument("--top-k",         type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Retrieval-Augmented Inference + BERTScore (OPTIMISED)")
    logger.info(f"  Run ID      : {RUN_ID}")
    logger.info(f"  Engine      : {args.engine}")
    logger.info(f"  Batch size  : {args.batch_size}  (HF engine only)")
    logger.info(f"  Adapter     : {SPACELLM_V1_ADAPTER}  (unmodified)")
    logger.info(f"  Test file   : {TEST_FILE}")
    logger.info(f"  RAG top_k={args.top_k}  min_similarity={args.min_similarity}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    from execute import KnowledgeRepository, build_augmented_prompt
    repo = KnowledgeRepository()
    logger.info(f"Knowledge repository loaded — {len(repo)} stored correction(s).")

    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    if args.engine == "vllm":
        # vLLM path — no HF model load needed
        if args.ablation:
            baseline = run_eval_vllm(
                test_records, use_rag=False,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity, tag="no_rag",
            )
            rag_run = run_eval_vllm(
                test_records, use_rag=True,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity, tag="rag",
            )
            _log_ablation_delta(baseline, rag_run)
        else:
            use_rag = not args.no_rag
            run_eval_vllm(
                test_records, use_rag=use_rag,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity,
                tag="rag" if use_rag else "no_rag",
            )

    else:
        # HF path — load model once, reuse across ablation runs
        model, tokenizer = load_model_and_tokenizer()

        if args.ablation:
            baseline = run_eval(
                model, tokenizer, test_records, use_rag=False,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity,
                tag="no_rag", batch_size=args.batch_size,
            )
            rag_run = run_eval(
                model, tokenizer, test_records, use_rag=True,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity,
                tag="rag", batch_size=args.batch_size,
            )
            _log_ablation_delta(baseline, rag_run)
        else:
            use_rag = not args.no_rag
            run_eval(
                model, tokenizer, test_records, use_rag=use_rag,
                repo=repo, build_augmented_prompt=build_augmented_prompt,
                top_k=args.top_k, min_similarity=args.min_similarity,
                tag="rag" if use_rag else "no_rag",
                batch_size=args.batch_size,
            )

    logger.info("=" * 60)
    logger.info("  Done.")
    logger.info("=" * 60)


def _log_ablation_delta(baseline: dict, rag_run: dict):
    if not (baseline and rag_run):
        return
    delta_bert  = rag_run["bert_score"]["f1"] - baseline["bert_score"]["f1"]
    delta_token = rag_run["token_f1_mean"]     - baseline["token_f1_mean"]
    logger.info("=" * 60)
    logger.info("  ABLATION RESULT")
    logger.info(
        f"  BERTScore F1   no_rag={baseline['bert_score']['f1']:.4f}  "
        f"rag={rag_run['bert_score']['f1']:.4f}  delta={delta_bert:+.4f}"
    )
    logger.info(
        f"  Token F1       no_rag={baseline['token_f1_mean']:.4f}  "
        f"rag={rag_run['token_f1_mean']:.4f}  delta={delta_token:+.4f}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
