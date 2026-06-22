"""
SpaceLLM — Correction Adapter Inference + BERTScore
=====================================================
Uses ONLY the locally saved correction adapter at spacellm_lora_final.
That adapter already has SpaceLLM_v1 knowledge baked in via merge_and_unload(),
so there is NO adapter stacking — one adapter, one load, no shape conflicts.

Pipeline:
    1. Detect true vocab size from the adapter checkpoint tensors
    2. Load base model (openai/gpt-oss-20b, MXFP4 dequantize)
    3. Untie lm_head
    4. Resize vocab to match adapter
    5. Load single correction adapter
    6. Smoke test (5 samples) — abort if degenerate
    7. Full inference + BERTScore
"""

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")

CORRECTION_ADAPTER_DIR = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/"
    "fine_tuning_v2/outputs/spacellm_lora_final"
)

TEST_FILE = BASE_DIR / "Model_training_&_Data_Extraction/data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR   = BASE_DIR / "Model_training_&_Data_Extraction/fine_tuning_v2/outputs/bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"inference_correction_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("BERTEval")

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID           = "openai/gpt-oss-20b"
MAX_SEQ_LEN        = 2048
MAX_NEW_TOKENS     = 256
MAX_SAMPLES        = None       # set to an int to cap the eval set
DEVICE             = "cuda:0"
SMOKE_TEST_SAMPLES = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_adapter_vocab_size(adapter_dir: Path) -> int:
    """
    Read the true vocab size directly from the saved adapter tensors.
    lora_B for lm_head has shape [vocab_size, r] — take max(shape).
    Raises RuntimeError if it cannot be determined (never fall back to a guess).
    """
    from safetensors import safe_open

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors not found at {safetensors_path}. "
            "Make sure correction_fine_tuning.py ran successfully first."
        )

    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            if "lm_head" in key and ("lora_B" in key or "lora_b" in key):
                shape = f.get_slice(key).get_shape()
                vocab = max(shape)
                logger.info(f"  Detected vocab size {vocab:,} from tensor '{key}' shape={list(shape)}")
                return vocab

    raise RuntimeError(
        f"Could not find an lm_head lora_B tensor in {safetensors_path}. "
        "Inspect the file manually: `python -c \"from safetensors import safe_open; "
        "f=safe_open('adapter_model.safetensors', framework='pt'); print(list(f.keys()))\"`"
    )


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


def untie_lm_head(model, label: str = ""):
    model.config.tie_word_embeddings = False
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info(f"✅ lm_head untied {label}")


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


def extract_prompt_and_reference(record: dict, tokenizer) -> tuple[str | None, str | None]:
    messages   = record.get("messages", [])
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

    try:
        prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e}")
        return None, None

    return prompt, ref_answer


def generate_one(model, tokenizer, prompt: str, device: str) -> tuple[torch.Tensor, str]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

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
    ids = token_ids.tolist()
    if not ids:
        return True, "empty generation"
    counts       = Counter(ids)
    top_id, top_count = counts.most_common(1)[0]
    frac = top_count / len(ids)
    if frac > 0.6:
        return True, f"token id {top_id} makes up {frac:.0%} of the generation"
    return False, ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  SpaceLLM — Correction Adapter Only (v1 merged in) — Inference + BERTScore")
    logger.info(f"  Run ID     : {RUN_ID}")
    logger.info(f"  Adapter    : {CORRECTION_ADAPTER_DIR}")
    logger.info(f"  Test file  : {TEST_FILE}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")
    log_gpu_memory("before model load")

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    # ── Step 1: detect vocab size from the adapter checkpoint ─────────────
    logger.info("\n── Detecting vocab size from adapter checkpoint ──────")
    TRAINED_VOCAB_SIZE = detect_adapter_vocab_size(CORRECTION_ADAPTER_DIR)
    logger.info(f"  ✅ Vocab size = {TRAINED_VOCAB_SIZE:,}")

    # ── Step 2: tokenizer ─────────────────────────────────────────────────
    logger.info("\nLoading tokenizer...")
    tokenizer_source = (
        str(CORRECTION_ADAPTER_DIR)
        if (CORRECTION_ADAPTER_DIR / "tokenizer_config.json").exists()
        else MODEL_ID
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    logger.info(f"  Tokenizer loaded from: {tokenizer_source}")

    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None — fix tokenizer config before running.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("  pad_token was unset — falling back to eos_token")

    tokenizer.padding_side = "left"
    logger.info(
        f"  vocab={tokenizer.vocab_size:,}  "
        f"eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}"
    )

    # ── Step 3: base model ────────────────────────────────────────────────
    logger.info(f"\nLoading base model {MODEL_ID} to {DEVICE} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map=DEVICE,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    logger.info(f"  Base model loaded in {time.time()-t0:.1f}s")
    log_gpu_memory("after base model load")

    # ── Step 4: untie + resize ────────────────────────────────────────────
    untie_lm_head(model, "(base, pre-resize)")

    logger.info(f"\nResizing vocab to {TRAINED_VOCAB_SIZE:,} ...")
    model.resize_token_embeddings(TRAINED_VOCAB_SIZE)
    model.config.vocab_size = TRAINED_VOCAB_SIZE

    # resize can re-tie weights — guard against it
    if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
        untie_lm_head(model, "(re-untied after resize)")

    actual_vocab = model.get_output_embeddings().weight.shape[0]
    assert actual_vocab == TRAINED_VOCAB_SIZE, (
        f"Vocab resize mismatch: got {actual_vocab:,}, expected {TRAINED_VOCAB_SIZE:,}"
    )
    logger.info(f"  ✅ Vocab confirmed at {actual_vocab:,}")

    # ── Step 5: load the single correction adapter ─────────────────────────
    # This adapter was produced by merge_and_unload(SpaceLLM_v1) + fresh LoRA,
    # so it already encodes v1 knowledge. No stacking required.
    logger.info(f"\nLoading correction adapter from {CORRECTION_ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(
        model,
        str(CORRECTION_ADAPTER_DIR),
        adapter_name="correction",
        is_trainable=False,
    )
    model.set_adapter("correction")
    logger.info("  ✅ Correction adapter loaded and set as active")

    model.eval()
    torch.cuda.empty_cache()
    log_gpu_memory("after adapter load")

    # ── Step 6: smoke test ────────────────────────────────────────────────
    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    logger.info(f"\n── Smoke test ({SMOKE_TEST_SAMPLES} samples) ────────────────────")
    smoke_ok = True
    for i, record in enumerate(test_records[:SMOKE_TEST_SAMPLES]):
        prompt, ref = extract_prompt_and_reference(record, tokenizer)
        if prompt is None:
            continue
        token_ids, text = generate_one(model, tokenizer, prompt, DEVICE)
        bad, reason = looks_degenerate(token_ids)
        logger.info(f"  [{i}] {text[:100]!r}")
        if bad:
            logger.error(f"  ❌ Sample {i} degenerate: {reason}")
            smoke_ok = False

    if not smoke_ok:
        raise RuntimeError(
            "Smoke test detected degenerate generation. Aborting before full eval. "
            "Check vocab size detection and adapter loading above."
        )
    logger.info("  ✅ Smoke test passed — proceeding to full eval\n")

    # ── Step 7: full inference ─────────────────────────────────────────────
    logger.info(f"Running inference on {len(test_records):,} samples ...")
    references, hypotheses, predictions = [], [], []
    skipped = 0
    t_inf = time.time()

    for i, record in enumerate(test_records):
        prompt, ref = extract_prompt_and_reference(record, tokenizer)
        if prompt is None:
            skipped += 1
            continue
        try:
            _, generated = generate_one(model, tokenizer, prompt, DEVICE)
        except Exception as e:
            logger.warning(f"  Generation failed [{i}]: {e}")
            skipped += 1
            continue

        references.append(ref)
        hypotheses.append(generated)
        predictions.append({
            "sample_id": record.get("sample_id", i),
            "reference": ref,
            "generated": generated,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_inf
            rate    = (i + 1) / elapsed
            eta     = (len(test_records) - i - 1) / rate
            logger.info(
                f"  [{i+1}/{len(test_records)}] "
                f"valid={len(predictions)}  skipped={skipped}  "
                f"speed={rate:.2f} it/s  ETA={eta/60:.1f} min"
            )

    inf_time = time.time() - t_inf
    logger.info(
        f"\nInference done — {len(predictions):,} samples  "
        f"skipped={skipped}  time={inf_time/60:.1f} min"
    )

    if not predictions:
        logger.error("No predictions generated — cannot compute BERTScore.")
        return

    # Save raw predictions
    raw_path = OUT_DIR / f"predictions_raw_correction_{RUN_ID}.json"
    raw_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "run_id":             RUN_ID,
        "base_model":         MODEL_ID,
        "adapter":            str(CORRECTION_ADAPTER_DIR),
        "adapter_note":       "SpaceLLM_v1 merged into base via merge_and_unload(); correction LoRA on top — single adapter, no stacking",
        "config": {
            "device":         DEVICE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_seq_len":    MAX_SEQ_LEN,
            "vocab_size":     TRAINED_VOCAB_SIZE,
            "vocab_source":   "detected_from_adapter_checkpoint",
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

    out_path = OUT_DIR / f"bertscore_results_correction_{RUN_ID}.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Full results saved → {out_path}")
    logger.info("=" * 60)
    logger.info("  Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
