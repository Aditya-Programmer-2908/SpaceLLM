"""
SpaceLLM v1 (HF) + Correction Adapter — Full Test Set Inference + BERTScore
============================================================================
PATCHED VERSION — fixes the cause of the "!!!!!!!" degenerate-generation bug.

WHAT WAS BROKEN
----------------
1. TRAINED_VOCAB_SIZE = 200064 was a HARDCODED GUESS (the original script's
   own docstring even said "please sanity-check before running"). If the
   real vocab size used when the v1 / correction adapters were trained
   doesn't exactly match that number, resize_token_embeddings() leaves the
   lm_head rows misaligned with the LoRA deltas being added on top. The
   result is a near-random/degenerate output distribution -> the model
   greedily picks the same low-index token every step -> decodes as
   repeated "!" for the full max_new_tokens budget.
   FIX: read the TRUE vocab size directly out of the adapter checkpoint's
   tensor shapes (ground truth) instead of guessing a constant.

2. model.base_model.set_adapter(["v1", "correction"]) reaches past
   PeftModel into the underlying LoraModel as a workaround for older PEFT
   versions that don't accept a list. This can leave PeftModel's own
   `active_adapter` bookkeeping out of sync with what's actually active
   on the LoRA layers, depending on PEFT version.
   FIX: try the proper PeftModel multi-adapter API first, fall back to the
   base_model workaround only if needed, and EXPLICITLY VERIFY + LOG which
   adapters are actually active before running any inference.

3. No early sanity check. The original script only discovers the problem
   after running BERTScore on the full 5,291-sample test set (~16 hours).
   FIX: run a 5-sample smoke test first; abort loudly if generation looks
   degenerate (repeated single token) before burning the full eval budget.
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

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")

CORRECTION_ADAPTER_DIR = Path(
    "/mnt/DATA/saurabh/aditya/SpaceLLM/Model_training_&_Data_Extraction/"
    "fine_tuning_v2/outputs/spacellm_lora_final"
)

V1_ADAPTER_REPO = "AdityaPS/SpaceLLM_v1"

TEST_FILE = BASE_DIR / "Model_training_&_Data_Extraction/data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR   = BASE_DIR / "Model_training_&_Data_Extraction/fine_tuning_v2/outputs/bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"inference_v1_plus_correction_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("BERTEval")

# ── Config ────────────────────────────────────────────────────────
MODEL_ID       = "openai/gpt-oss-20b"
MAX_SEQ_LEN    = 2048
MAX_NEW_TOKENS = 256
MAX_SAMPLES    = None
DEVICE         = "cuda:0"

SMOKE_TEST_SAMPLES = 5   # run this many samples first and check for collapse


# ── NEW: derive true vocab size from the adapter checkpoint itself ───────
def detect_adapter_vocab_size(adapter_dir_or_repo, hf_token=None) -> int | None:
    """
    Inspect the adapter's saved tensors and return the row count of the
    lm_head LoRA weight (the dimension that encodes vocab size). Works for
    both local directories and HF Hub repos. Returns None if it can't be
    determined (caller should treat that as a hard stop, not a guess).
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        if Path(adapter_dir_or_repo).exists():
            local_path = Path(adapter_dir_or_repo) / "adapter_model.safetensors"
            if not local_path.exists():
                logger.warning(f"  No adapter_model.safetensors found in {adapter_dir_or_repo}")
                return None
        else:
            local_path = hf_hub_download(
                repo_id=str(adapter_dir_or_repo),
                filename="adapter_model.safetensors",
                token=hf_token,
            )

        with safe_open(local_path, framework="pt") as f:
            for key in f.keys():
                if "lm_head" in key and ("lora_B" in key or "lora_b" in key):
                    shape = f.get_slice(key).get_shape()
                    # lora_B for a Linear(out_features=vocab_size) is [vocab_size, r]
                    vocab = max(shape)
                    logger.info(f"  Detected vocab size {vocab:,} from tensor '{key}' shape={shape}")
                    return vocab
        logger.warning(f"  Could not find an lm_head lora_B tensor in {adapter_dir_or_repo}")
        return None
    except Exception as e:
        logger.warning(f"  Vocab-size detection failed for {adapter_dir_or_repo}: {e}")
        return None


def load_test_data(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()
    data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(data):,} test records")
    return data


def extract_prompt_and_reference(record: dict, tokenizer):
    messages, hf_messages, ref_answer = record.get("messages", []), [], ""
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
        prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.warning(f"Template failed: {e}")
        return None, None
    return prompt, ref_answer


def log_gpu_memory(label=""):
    for i in range(torch.cuda.device_count()):
        props  = torch.cuda.get_device_properties(i)
        alloc  = torch.cuda.memory_allocated(i) / 1024**3
        reserv = torch.cuda.memory_reserved(i)  / 1024**3
        total  = props.total_memory              / 1024**3
        logger.info(f"GPU {i} [{props.name}] {label} | Alloc={alloc:.2f}GB Reserved={reserv:.2f}GB Total={total:.2f}GB")


def untie_lm_head(model, label=""):
    model.config.tie_word_embeddings = False
    lm_head = model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info(f"✅ lm_head untied {label}")


def generate_one(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                     max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        gen_ids = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = gen_ids[0][enc["input_ids"].shape[1]:]
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def looks_degenerate(token_ids) -> tuple[bool, str]:
    """Detect the 'repeated single token' collapse signature directly on ids."""
    ids = token_ids.tolist()
    if not ids:
        return True, "empty generation"
    counts = Counter(ids)
    top_id, top_count = counts.most_common(1)[0]
    frac = top_count / len(ids)
    if frac > 0.6:
        return True, f"token id {top_id} makes up {frac:.0%} of the generation"
    return False, ""


# ── Main ──────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  SpaceLLM v1 (HF) + Correction Adapter — Inference + BERTScore [PATCHED]")
    logger.info(f"  Run ID        : {RUN_ID}")
    logger.info(f"  V1 adapter    : {V1_ADAPTER_REPO}")
    logger.info(f"  Correction    : {CORRECTION_ADAPTER_DIR}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")
    log_gpu_memory("before model load")

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    # ── FIX #1: determine TRUE vocab size from the adapters, don't guess ──
    logger.info("\n── Detecting true vocab size from adapter checkpoints ──")
    vocab_v1         = detect_adapter_vocab_size(V1_ADAPTER_REPO, hf_token)
    vocab_correction = detect_adapter_vocab_size(CORRECTION_ADAPTER_DIR, hf_token)

    if vocab_v1 is None and vocab_correction is None:
        raise RuntimeError(
            "Could not determine vocab size from EITHER adapter checkpoint. "
            "Refusing to fall back to a hardcoded guess — that's what caused "
            "the '!!!!' degenerate-output bug last time. Inspect "
            "adapter_model.safetensors manually before proceeding."
        )
    if vocab_v1 and vocab_correction and vocab_v1 != vocab_correction:
        logger.error(
            f"  ❌ Vocab size MISMATCH between adapters: v1={vocab_v1:,}  "
            f"correction={vocab_correction:,}. These adapters were trained "
            f"with different vocab sizes and CANNOT be stacked safely. "
            f"Re-train the correction adapter with --base_adapter pointing "
            f"at v1 so they share one vocab, or fix the resize step."
        )
        raise RuntimeError("Adapter vocab size mismatch — see log for details.")

    TRAINED_VOCAB_SIZE = vocab_v1 or vocab_correction
    logger.info(f"  ✅ Using detected vocab size: {TRAINED_VOCAB_SIZE:,} (no hardcoded constant)")

    # ── Tokenizer ──────────────────────────────────────────────────
    logger.info("\nLoading tokenizer...")
    if (CORRECTION_ADAPTER_DIR / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(CORRECTION_ADAPTER_DIR), trust_remote_code=True)
        logger.info("Tokenizer loaded from correction adapter dir")
    else:
        tokenizer = AutoTokenizer.from_pretrained(V1_ADAPTER_REPO, trust_remote_code=True)
        logger.info(f"Tokenizer loaded from {V1_ADAPTER_REPO}")

    # ── FIX #3 (partial): don't silently conflate pad/eos without checking
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None — fix the tokenizer config before running inference.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(
            "  pad_token was unset, falling back to eos_token. This is fine for "
            "generation, but verify eos_token_id actually matches the model's "
            "real chat-template stop token (check model.generation_config.eos_token_id)."
        )
    tokenizer.padding_side = "left"
    logger.info(f"Tokenizer vocab={tokenizer.vocab_size:,}  eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}")

    # ── Base model ───────────────────────────────────────────────
    logger.info(f"\nLoading base model {MODEL_ID} to {DEVICE}...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=Mxfp4Config(dequantize=True),
        device_map=DEVICE, trust_remote_code=True, ignore_mismatched_sizes=True,
    )
    logger.info(f"Base model loaded in {time.time()-t0:.1f}s")
    log_gpu_memory("after base model load")

    untie_lm_head(base_model, "(base)")

    logger.info(f"Resizing vocab to detected size {TRAINED_VOCAB_SIZE:,}...")
    base_model.resize_token_embeddings(TRAINED_VOCAB_SIZE)
    base_model.config.vocab_size = TRAINED_VOCAB_SIZE

    if id(base_model.get_input_embeddings().weight) == id(base_model.get_output_embeddings().weight):
        untie_lm_head(base_model, "(re-untied after resize)")

    actual_vocab = base_model.get_output_embeddings().weight.shape[0]
    assert actual_vocab == TRAINED_VOCAB_SIZE, f"Resize mismatch: got {actual_vocab}, expected {TRAINED_VOCAB_SIZE}"
    logger.info(f"✅ Vocab = {actual_vocab:,}")

    # ── Load adapters ────────────────────────────────────────────
    logger.info(f"\nLoading v1 adapter from {V1_ADAPTER_REPO}...")
    model = PeftModel.from_pretrained(base_model, V1_ADAPTER_REPO, adapter_name="v1", is_trainable=False)
    torch.cuda.empty_cache()
    logger.info("✅ v1 adapter loaded (unmerged)")

    logger.info(f"\nLoading correction adapter from {CORRECTION_ADAPTER_DIR}...")
    model.load_adapter(str(CORRECTION_ADAPTER_DIR), adapter_name="correction", is_trainable=False)
    torch.cuda.empty_cache()

    # ── FIX #2: proper multi-adapter activation + explicit verification ──
    try:
        model.set_adapter(["v1", "correction"])
        logger.info("✅ Activated both adapters via PeftModel.set_adapter() (preferred path)")
    except TypeError:
        logger.warning("  PeftModel.set_adapter() doesn't accept a list on this PEFT version — "
                        "falling back to base_model.set_adapter(). Verifying state below.")
        model.base_model.set_adapter(["v1", "correction"])

    # Verify — don't just assume the workaround did what we think.
    active = getattr(model, "active_adapters", None) or [getattr(model, "active_adapter", None)]
    logger.info(f"  active_adapters reported by PeftModel: {active}")
    lora_active = set()
    for name, module in model.named_modules():
        if hasattr(module, "active_adapter"):
            aa = module.active_adapter
            lora_active.update(aa if isinstance(aa, (list, set)) else [aa])
    logger.info(f"  active_adapter(s) actually set on LoRA layers: {lora_active}")
    if not {"v1", "correction"}.issubset(lora_active):
        raise RuntimeError(
            f"Expected both 'v1' and 'correction' active on LoRA layers, got {lora_active}. "
            f"Multi-adapter activation did not take effect — fix this before running inference."
        )

    model.eval()
    log_gpu_memory("after both adapters active")

    cpu_params = [(n, p.device) for n, p in model.named_parameters() if p.device.type == "cpu"]
    if cpu_params:
        logger.warning(f"  {len(cpu_params)} params still on CPU — moving to {DEVICE}")
        model = model.to(DEVICE)

    # ── Load test data ──────────────────────────────────────────
    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    # ── FIX #3: smoke test BEFORE the full run ───────────────────
    logger.info(f"\n── Smoke test on {SMOKE_TEST_SAMPLES} samples before full eval ──")
    smoke_ok = True
    for i, record in enumerate(test_records[:SMOKE_TEST_SAMPLES]):
        prompt, ref = extract_prompt_and_reference(record, tokenizer)
        if prompt is None:
            continue
        token_ids, text = generate_one(model, tokenizer, prompt, DEVICE)
        bad, reason = looks_degenerate(token_ids)
        logger.info(f"  [{i}] generated={text[:80]!r}")
        if bad:
            logger.error(f"  ❌ Sample {i} looks degenerate: {reason}")
            smoke_ok = False

    if not smoke_ok:
        raise RuntimeError(
            "Smoke test detected degenerate generation (repeated single token). "
            "ABORTING before the full 5,000+ sample / multi-hour BERTScore run. "
            "Root-cause this first — see the vocab-size and adapter-activation "
            "checks above for likely culprits."
        )
    logger.info("✅ Smoke test passed — generations look like real text. Proceeding to full eval.\n")

    # ── Full inference ────────────────────────────────────────────
    logger.info(f"Running inference on {len(test_records):,} samples...")
    references, hypotheses, predictions, skipped = [], [], [], 0
    t_inf = time.time()

    for i, record in enumerate(test_records):
        prompt, ref = extract_prompt_and_reference(record, tokenizer)
        if prompt is None:
            skipped += 1
            continue
        try:
            _, generated = generate_one(model, tokenizer, prompt, DEVICE)
        except Exception as e:
            logger.warning(f"Generation failed [{i}]: {e}")
            skipped += 1
            continue

        references.append(ref)
        hypotheses.append(generated)
        predictions.append({"sample_id": record.get("sample_id", i), "reference": ref, "generated": generated})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_inf
            rate = (i + 1) / elapsed
            eta = (len(test_records) - i - 1) / rate
            logger.info(f"  [{i+1}/{len(test_records)}] valid={len(predictions)} skipped={skipped} "
                        f"speed={rate:.2f} it/s ETA={eta/60:.1f} min")

    inf_time = time.time() - t_inf
    logger.info(f"\nInference done — {len(predictions):,} samples, skipped={skipped}, time={inf_time/60:.1f} min")

    if not predictions:
        logger.error("No predictions generated — cannot compute BERTScore")
        return

    raw_path = OUT_DIR / f"predictions_raw_v1_plus_correction_{RUN_ID}.json"
    raw_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Raw predictions saved → {raw_path}")

    # ── Token F1 ──────────────────────────────────────────────────
    token_f1s = []
    for hyp, ref in zip(hypotheses, references):
        ref_toks, gen_toks = set(ref.lower().split()), set(hyp.lower().split())
        if ref_toks or gen_toks:
            overlap = len(ref_toks & gen_toks)
            prec = overlap / len(gen_toks) if gen_toks else 0.0
            rec  = overlap / len(ref_toks) if ref_toks else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        token_f1s.append(f1)
    mean_token_f1 = sum(token_f1s) / len(token_f1s)
    logger.info(f"Token F1 mean: {mean_token_f1:.4f}")

    # ── BERTScore ─────────────────────────────────────────────────
    from bert_score import score as bert_score
    P, R, F1 = bert_score(hypotheses, references, lang="en", model_type="roberta-large",
                           batch_size=32, verbose=True, device=DEVICE)
    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    f1_list = F1.tolist()

    logger.info(f"BERTScore P={mean_p:.4f} R={mean_r:.4f} F1={mean_f1:.4f}")

    for i, p in enumerate(predictions):
        p["bert_f1"]  = round(f1_list[i], 4)
        p["token_f1"] = round(token_f1s[i], 4)

    results = {
        "run_id": RUN_ID, "base_model": MODEL_ID, "v1_adapter": V1_ADAPTER_REPO,
        "correction_adapter": str(CORRECTION_ADAPTER_DIR),
        "config": {"device": DEVICE, "max_new_tokens": MAX_NEW_TOKENS,
                   "max_seq_len": MAX_SEQ_LEN, "vocab_size": TRAINED_VOCAB_SIZE,
                   "vocab_size_source": "detected_from_adapter_checkpoint"},
        "summary": {"total_records": len(test_records), "evaluated": len(predictions),
                    "skipped": skipped, "inference_time_min": round(inf_time / 60, 2)},
        "bert_score": {"model": "roberta-large", "precision": round(mean_p, 4),
                       "recall": round(mean_r, 4), "f1": round(mean_f1, 4)},
        "token_f1_mean": round(mean_token_f1, 4),
        "per_sample": predictions,
    }
    out_path = OUT_DIR / f"bertscore_results_v1_plus_correction_{RUN_ID}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Full results saved → {out_path}")


if __name__ == "__main__":
    main()
