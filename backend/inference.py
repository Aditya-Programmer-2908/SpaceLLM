"""
SpaceLLM — Retrieval-Augmented Inference + BERTScore
=======================================================
Replaces the old "correction adapter" inference script. There is no LoRA
adapter stacked on top of SpaceLLM_v1 anymore — the base model weights are
NEVER modified post-deployment. Corrections live in a JSON knowledge
repository (written by execute.py's RETRIEVAL_MEMORY_UPDATE handler) and
are injected into the prompt at inference time, per-query.

Why this replaces the adapter-stacking script:
    The previous flow trained a new LoRA correction adapter on ~10-15
    examples per cycle and merged it on top of SpaceLLM_v1. On a dataset
    that small, the adapter overfit and caused generation collapse
    (degenerate repeated-token outputs) — see the smoke-test guard that
    used to live in this file. Retrieval-augmented generation sidesteps
    the failure mode entirely: SpaceLLM_v1's weights are loaded once and
    never touched, so there is nothing left to collapse. Corrections are
    available the instant they're written to the repository — no
    training run, no vocab-resize bookkeeping, no adapter-shape conflicts.

Pipeline:
    1. Load base model (openai/gpt-oss-20b, MXFP4 dequantize)
    2. Detect the vocab size SpaceLLM_v1 was trained against from its saved
       adapter tensors, and resize the freshly-loaded base model to match
       if the base checkpoint's native vocab has drifted (e.g. HF re-pads
       gpt-oss-20b's embedding table over time, independent of any LoRA
       work). This is NOT the old adapter-stacking resize hack — it's
       unavoidable bookkeeping any time you attach an adapter to a base
       checkpoint whose shapes may have moved since the adapter was saved.
    3. Load the SpaceLLM_v1 adapter — exactly one adapter, never modified,
       never merged, never retrained
    4. Load the JSON knowledge repository (execute.py's KnowledgeRepository)
    5. For each test record: retrieve top-k similar corrections for the
       raw question, inject them into the user turn, then chat-template
       and generate
    6. Token F1 + BERTScore, same as before
    7. Optional ablation: run the same test set with RAG disabled to get
       a side-by-side delta for the paper

Usage:
    python retrieval_inference_bertscore.py
    python retrieval_inference_bertscore.py --no-rag          # baseline, no retrieval
    python retrieval_inference_bertscore.py --ablation        # runs both, reports delta
    python retrieval_inference_bertscore.py --top-k 5 --min-similarity 0.5
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")
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
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("RAGEval")

# execute.py (KnowledgeRepository, build_augmented_prompt) lives under
# backend/mape_k — make it importable without turning this script into a
# package.
sys.path.insert(0, str(MAPE_DIR))

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID            = "openai/gpt-oss-20b"
SPACELLM_V1_ADAPTER = "AdityaPS/SpaceLLM_v1"   # HF-hosted adapter — never modified
MAX_SEQ_LEN          = 2048
MAX_NEW_TOKENS       = 256
MAX_SAMPLES          = None        # set to an int to cap the eval set
DEVICE               = "cuda:0"
SMOKE_TEST_SAMPLES   = 5

DEFAULT_TOP_K          = 3
DEFAULT_MIN_SIMILARITY = 0.55


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
    Like the old extract_prompt_and_reference, but stops BEFORE chat
    templating and returns the raw HF message list instead. We need the
    untemplated user turn so retrieval augmentation can be injected into
    the right place before the template (and its special tokens) are
    applied.
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
    """
    Finds the last user turn, retrieves similar past corrections for its
    raw text, and replaces that turn's content with the augmented version.
    Returns (possibly-modified messages, num_corrections_retrieved,
    raw_question_text) so callers can log/inspect what was retrieved.
    """
    user_indices = [i for i, m in enumerate(hf_messages) if m["role"] == "user"]
    if not user_indices:
        return hf_messages, 0, ""

    last_user_idx     = user_indices[-1]
    original_question = hf_messages[last_user_idx]["content"]

    augmented = build_augmented_prompt(
        original_question, repo, top_k=top_k, min_similarity=min_similarity
    )
    num_retrieved = 0 if augmented == original_question else top_k  # exact count not returned by build_augmented_prompt

    new_messages = [dict(m) for m in hf_messages]
    new_messages[last_user_idx]["content"] = augmented
    return new_messages, num_retrieved, original_question


def detect_adapter_vocab_size(adapter_path: str, hf_token: str | None = None) -> int:
    """
    Read the vocab size an adapter was trained against, straight from its
    saved tensors — works for a local directory OR an HF repo id (e.g.
    "AdityaPS/SpaceLLM_v1"), downloading just the safetensors file in the
    latter case rather than guessing from tokenizer/config metadata, which
    can drift from the actual checkpoint shapes.

    Looks at every "lm_head" tensor in the adapter (base_layer.weight,
    lora_A.weight, lora_B.weight — whichever are present) and takes the
    largest dimension across all of them, since the vocab dimension is
    always the larger side of an lm_head matrix.
    """
    from safetensors import safe_open

    local_path = Path(adapter_path)
    if local_path.exists() and (local_path / "adapter_model.safetensors").exists():
        safetensors_path = local_path / "adapter_model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        safetensors_path = Path(hf_hub_download(
            repo_id=str(adapter_path), filename="adapter_model.safetensors", token=hf_token,
        ))

    candidates = []
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            if "lm_head" in key:
                shape = f.get_slice(key).get_shape()
                candidates.append((key, shape))

    if not candidates:
        raise RuntimeError(
            f"No lm_head tensors found in {safetensors_path} — cannot determine "
            f"the vocab size this adapter expects."
        )

    key, shape = max(candidates, key=lambda kv: max(kv[1]))
    vocab = max(shape)
    logger.info(f"  Detected required vocab size {vocab:,} from tensor '{key}' shape={list(shape)}")
    return vocab


def untie_lm_head(model, label: str = ""):
    model.config.tie_word_embeddings = False
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info(f"  ✅ lm_head untied {label}")


def get_input_device(model) -> torch.device:
    """Return the device of the embedding layer (first layer inputs must go here)."""
    try:
        return next(model.get_input_embeddings().parameters()).device
    except Exception:
        return torch.device("cuda:0")


def generate_one(model, tokenizer, prompt: str) -> tuple[torch.Tensor, str]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=False,
    )
    # Send inputs to wherever the embedding layer lives (may differ from DEVICE
    # after dispatch_model shards the MoE model across multiple GPUs)
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


def looks_degenerate(token_ids: torch.Tensor) -> tuple[bool, str]:
    """Kept as a cheap sanity check on the live deployment. With retrieval
    replacing adapter retraining, the base model weights are never touched,
    so this should essentially never trigger — if it does, something else
    (e.g. a bad checkpoint swap) is wrong, not a retraining artifact."""
    from collections import Counter
    ids = token_ids.tolist()
    if not ids:
        return True, "empty generation"
    counts             = Counter(ids)
    top_id, top_count  = counts.most_common(1)[0]
    frac = top_count / len(ids)
    if frac > 0.6:
        return True, f"token id {top_id} makes up {frac:.0%} of the generation"
    return False, ""


def load_model_and_tokenizer():
    """Load SpaceLLM_v1. The base checkpoint's native lm_head size can drift
    from what v1 was actually trained against (e.g. HF re-pads gpt-oss-20b's
    vocab over time), so we detect v1's expected vocab size from its saved
    tensors and resize the freshly-loaded base model to match BEFORE
    attaching the adapter — otherwise PeftModel.from_pretrained fails with
    a state_dict shape mismatch on lm_head. This is unrelated to the old
    LoRA-stacking/retraining problem; it's just base-checkpoint drift."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Detecting vocab size required by SpaceLLM_v1 adapter ...")
    required_vocab = detect_adapter_vocab_size(SPACELLM_V1_ADAPTER, hf_token=hf_token)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SPACELLM_V1_ADAPTER, trust_remote_code=True, token=hf_token)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None — fix tokenizer config before running.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("  pad_token was unset — falling back to eos_token")
    tokenizer.padding_side = "left"
    logger.info(f"  vocab={tokenizer.vocab_size:,}  eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}")

    logger.info(f"Loading base model {MODEL_ID} to CPU (will dispatch after PEFT) ...")
    t0 = time.time()
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

    # ── Align vocab with what v1 was trained against ──────────────────────
    untie_lm_head(model, "(base, pre-resize)")
    current_vocab = model.get_output_embeddings().weight.shape[0]
    if current_vocab != required_vocab:
        logger.info(f"Resizing base model vocab {current_vocab:,} → {required_vocab:,} to match SpaceLLM_v1 ...")
        model.resize_token_embeddings(required_vocab)
        model.config.vocab_size = required_vocab
        # resize can re-tie weights — guard against it
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
    logger.info("  ✅ SpaceLLM_v1 adapter loaded and set as active (unmodified — no further training)")

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
            free  = torch.cuda.mem_get_info(i)[0]
            alloc = max(0, free - 4 * 1024**3)
            max_memory[i] = f"{int(alloc / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"  max_memory per device: {max_memory}")

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
        model = dispatch_model(model, device_map=device_map)
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
) -> dict:
    """Run inference + BERTScore over test_records, optionally with
    retrieval augmentation. `tag` distinguishes output filenames when
    running an ablation (e.g. 'rag' vs 'no_rag')."""

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
            "weights are never modified in this setup, this points to a "
            "loading/checkpoint problem, not a retraining artifact — check "
            "the adapter path and tokenizer."
        )
    logger.info("  ✅ Smoke test passed — proceeding to full eval\n")

    logger.info(f"Running inference on {len(test_records):,} samples (RAG={use_rag}) ...")
    references, hypotheses, predictions = [], [], []
    skipped = 0
    retrieved_counts = []
    t_inf = time.time()

    for i, record in enumerate(test_records):
        hf_messages, ref = extract_messages_and_reference(record)
        if hf_messages is None:
            skipped += 1
            continue

        retrieved_hits = []
        if use_rag:
            user_indices = [j for j, m in enumerate(hf_messages) if m["role"] == "user"]
            raw_question = hf_messages[user_indices[-1]]["content"] if user_indices else ""
            retrieved_hits = repo.query(raw_question, top_k=top_k, min_similarity=min_similarity) if raw_question else []
            hf_messages, _, _ = inject_retrieval_context(
                hf_messages, repo, build_augmented_prompt, top_k, min_similarity
            )
        retrieved_counts.append(len(retrieved_hits))

        prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        try:
            _, generated = generate_one(model, tokenizer, prompt)
        except Exception as e:
            logger.warning(f"  Generation failed [{i}]: {e}")
            skipped += 1
            continue

        references.append(ref)
        hypotheses.append(generated)
        predictions.append({
            "sample_id":           record.get("sample_id", i),
            "reference":           ref,
            "generated":           generated,
            "retrieved_corrections": len(retrieved_hits),
            "top_similarity":      round(retrieved_hits[0]["similarity"], 4) if retrieved_hits else None,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_inf
            rate    = (i + 1) / elapsed
            eta     = (len(test_records) - i - 1) / rate
            logger.info(
                f"  [{i+1}/{len(test_records)}] valid={len(predictions)}  skipped={skipped}  "
                f"speed={rate:.2f} it/s  ETA={eta/60:.1f} min"
            )

    inf_time = time.time() - t_inf
    logger.info(f"\nInference done (tag={tag}) — {len(predictions):,} samples  skipped={skipped}  time={inf_time/60:.1f} min")

    if not predictions:
        logger.error("No predictions generated — cannot compute BERTScore.")
        return {}

    raw_path = OUT_DIR / f"predictions_raw_{tag}_{RUN_ID}.json"
    raw_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Raw predictions saved → {raw_path}")

    # ── Token F1 ───────────────────────────────────────────────────────────
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

    # ── BERTScore ──────────────────────────────────────────────────────────
    logger.info("\nComputing BERTScore ...")
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en", model_type="roberta-large",
        batch_size=32, verbose=True, device=DEVICE,
    )
    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    f1_list = F1.tolist()
    logger.info(f"BERTScore  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}")

    for i, pred in enumerate(predictions):
        pred["bert_f1"]  = round(f1_list[i], 4)
        pred["token_f1"] = round(token_f1s[i], 4)

    results = {
        "run_id":          RUN_ID,
        "tag":             tag,
        "base_model":      MODEL_ID,
        "adapter":         SPACELLM_V1_ADAPTER,
        "adapter_note":    "SpaceLLM_v1 only — never modified post-deployment. "
                            "Corrections applied via retrieval, not retraining.",
        "rag_enabled":     use_rag,
        "rag_config": {
            "top_k":          top_k,
            "min_similarity": min_similarity,
            "repo_size":      len(repo) if use_rag else None,
            "mean_retrieved": round(sum(retrieved_counts) / len(retrieved_counts), 3) if retrieved_counts else 0,
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
    parser = argparse.ArgumentParser(description="SpaceLLM retrieval-augmented inference + BERTScore")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval augmentation (baseline run).")
    parser.add_argument("--ablation", action="store_true", help="Run both with and without RAG, report the delta.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — Retrieval-Augmented Inference + BERTScore")
    logger.info(f"  Run ID     : {RUN_ID}")
    logger.info(f"  Adapter    : {SPACELLM_V1_ADAPTER}  (unmodified)")
    logger.info(f"  Test file  : {TEST_FILE}")
    logger.info(f"  RAG top_k={args.top_k}  min_similarity={args.min_similarity}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    from execute import KnowledgeRepository, build_augmented_prompt
    repo = KnowledgeRepository()
    logger.info(f"Knowledge repository loaded — {len(repo)} stored correction(s).")

    model, tokenizer = load_model_and_tokenizer()

    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    if args.ablation:
        baseline = run_eval(
            model, tokenizer, test_records, use_rag=False,
            repo=repo, build_augmented_prompt=build_augmented_prompt,
            top_k=args.top_k, min_similarity=args.min_similarity, tag="no_rag",
        )
        rag_run = run_eval(
            model, tokenizer, test_records, use_rag=True,
            repo=repo, build_augmented_prompt=build_augmented_prompt,
            top_k=args.top_k, min_similarity=args.min_similarity, tag="rag",
        )
        if baseline and rag_run:
            delta_bert  = rag_run["bert_score"]["f1"] - baseline["bert_score"]["f1"]
            delta_token = rag_run["token_f1_mean"] - baseline["token_f1_mean"]
            logger.info("=" * 60)
            logger.info("  ABLATION RESULT")
            logger.info(f"  BERTScore F1   no_rag={baseline['bert_score']['f1']:.4f}  "
                        f"rag={rag_run['bert_score']['f1']:.4f}  delta={delta_bert:+.4f}")
            logger.info(f"  Token F1       no_rag={baseline['token_f1_mean']:.4f}  "
                        f"rag={rag_run['token_f1_mean']:.4f}  delta={delta_token:+.4f}")
            logger.info("=" * 60)
    else:
        use_rag = not args.no_rag
        tag = "rag" if use_rag else "no_rag"
        run_eval(
            model, tokenizer, test_records, use_rag=use_rag,
            repo=repo, build_augmented_prompt=build_augmented_prompt,
            top_k=args.top_k, min_similarity=args.min_similarity, tag=tag,
        )

    logger.info("=" * 60)
    logger.info("  Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
