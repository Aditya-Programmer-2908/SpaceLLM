"""
SpaceLLM LoRA v4 — Full Test Set Inference + BERTScore
=======================================================
Loads fine-tuned LoRA adapters, runs inference on full test set,
computes BERTScore F1 for all samples.
"""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime

import torch

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")
ADAPTER_DIR = BASE_DIR / "fine_tuning_v2/outputs/spacellm_lora_final"
TEST_FILE   = BASE_DIR / "data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR     = BASE_DIR / "fine_tuning_v2/outputs/bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"inference_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("BERTEval")

# ── Config ─────────────────────────────────────────────────────────
MODEL_ID        = "openai/gpt-oss-20b"
MAX_SEQ_LEN     = 2048
MAX_NEW_TOKENS  = 256
BATCH_SIZE      = 1          # increase if VRAM allows
MAX_SAMPLES     = None       # None = full test set


def load_test_data(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(data):,} test records")
    return data


def extract_prompt_and_reference(record: dict, tokenizer, max_seq_len: int):
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

    try:
        prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.warning(f"Template failed: {e}")
        return None, None

    return prompt, ref_answer


def main():
    logger.info("=" * 60)
    logger.info("  SpaceLLM LoRA v4 — Inference + BERTScore")
    logger.info(f"  Run ID   : {RUN_ID}")
    logger.info(f"  Adapter  : {ADAPTER_DIR}")
    logger.info(f"  Test set : {TEST_FILE}")
    logger.info("=" * 60)

    # ── Check GPU ─────────────────────────────────────────────────
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    # ── Load model ─────────────────────────────────────────────────
    logger.info("\nLoading base model + LoRA adapters...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # better for generation
    logger.info(f"Tokenizer loaded  vocab={tokenizer.vocab_size:,}")

    import torch.nn as nn

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="auto",
        trust_remote_code=True,
    )

    # Step 1 — Untie lm_head (same as training script)
    base_model.config.tie_word_embeddings = False
    lm_head = base_model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info("✅ lm_head untied")

    # Step 2 — Resize to match training vocab (200064)
    TRAINED_VOCAB_SIZE = 200064  # from error message
    base_model.resize_token_embeddings(TRAINED_VOCAB_SIZE)
    base_model.config.vocab_size = TRAINED_VOCAB_SIZE
    logger.info(f"✅ Vocab resized to {TRAINED_VOCAB_SIZE}")

    # Step 3 — Re-untie after resize (resize can re-tie)
    lm_head = base_model.get_output_embeddings()
    lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
    logger.info("✅ lm_head re-untied after resize")

    # Step 4 — Now load adapter
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    logger.info("✅ LoRA adapter loaded successfully")

    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # ── Load test data ─────────────────────────────────────────────
    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]
        logger.info(f"Limiting to {MAX_SAMPLES} samples")

    # ── Run inference ──────────────────────────────────────────────
    logger.info(f"\nRunning inference on {len(test_records):,} samples...")
    references   = []
    hypotheses   = []
    predictions  = []
    skipped      = 0

    t_inf = time.time()
    with torch.no_grad():
        for i, record in enumerate(test_records):
            prompt, ref = extract_prompt_and_reference(
                record, tokenizer, MAX_SEQ_LEN)

            if prompt is None:
                skipped += 1
                continue

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
                padding=False,
            )

            device = next(model.parameters()).device
            try:
                enc = {k: v.to(device) for k, v in enc.items()}
            except Exception:
                pass

            try:
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_ids   = gen_ids[0][enc["input_ids"].shape[1]:]
                generated = tokenizer.decode(
                    new_ids, skip_special_tokens=True).strip()
            except Exception as e:
                logger.warning(f"Generation failed [{i}]: {e}")
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
                    f"  [{i+1}/{len(test_records)}]  "
                    f"skipped={skipped}  "
                    f"speed={rate:.1f} it/s  "
                    f"ETA={eta/60:.1f} min"
                )

    logger.info(f"\nInference done — {len(predictions):,} samples  "
                f"skipped={skipped}  "
                f"time={( time.time()-t_inf)/60:.1f} min")

    # ── BERTScore ──────────────────────────────────────────────────
    logger.info("\nComputing BERTScore...")
    from bert_score import score as bert_score

    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang="en",
        model_type="roberta-large",
        batch_size=32,
        verbose=True,
    )

    mean_p   = P.mean().item()
    mean_r   = R.mean().item()
    mean_f1  = F1.mean().item()
    f1_list  = F1.tolist()

    logger.info("")
    logger.info("=" * 50)
    logger.info(f"  BERTScore Precision : {mean_p:.4f}")
    logger.info(f"  BERTScore Recall    : {mean_r:.4f}")
    logger.info(f"  BERTScore F1        : {mean_f1:.4f}")
    logger.info("=" * 50)

    # ── Token F1 for comparison ────────────────────────────────────
    token_f1s = []
    for pred, ref in zip(hypotheses, references):
        ref_toks = set(ref.lower().split())
        gen_toks = set(pred.lower().split())
        if ref_toks or gen_toks:
            overlap = len(ref_toks & gen_toks)
            prec = overlap / len(gen_toks) if gen_toks else 0.0
            rec  = overlap / len(ref_toks) if ref_toks else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        token_f1s.append(f1)

    mean_token_f1 = sum(token_f1s) / len(token_f1s)

    logger.info("\n── Metric Comparison ─────────────────────────────")
    logger.info(f"  Token F1     : {mean_token_f1:.4f}  ← word overlap only")
    logger.info(f"  BERTScore F1 : {mean_f1:.4f}  ← semantic similarity")
    logger.info("=" * 50)

    # ── Add scores to predictions ──────────────────────────────────
    for i, p in enumerate(predictions):
        p["bert_f1"]  = round(f1_list[i],    4)
        p["token_f1"] = round(token_f1s[i],  4)

    # ── Best and worst samples ─────────────────────────────────────
    worst = sorted(predictions, key=lambda x: x["bert_f1"])[:10]
    best  = sorted(predictions, key=lambda x: x["bert_f1"], reverse=True)[:10]

    logger.info("\n── Top 10 WORST (lowest BERTScore) ───────────────")
    for s in worst:
        logger.info(f"  [{s['sample_id']}]  BERTScore={s['bert_f1']}  TokenF1={s['token_f1']}")
        logger.info(f"    REF: {s['reference'][:120]}")
        logger.info(f"    GEN: {s['generated'][:120]}")
        logger.info("")

    logger.info("\n── Top 10 BEST (highest BERTScore) ───────────────")
    for s in best:
        logger.info(f"  [{s['sample_id']}]  BERTScore={s['bert_f1']}  TokenF1={s['token_f1']}")
        logger.info(f"    REF: {s['reference'][:120]}")
        logger.info(f"    GEN: {s['generated'][:120]}")
        logger.info("")

    # ── Save full results ──────────────────────────────────────────
    results = {
        "run_id":   RUN_ID,
        "adapter":  str(ADAPTER_DIR),
        "samples":  len(predictions),
        "skipped":  skipped,
        "bert_score": {
            "model":     "roberta-large",
            "precision": round(mean_p,         4),
            "recall":    round(mean_r,         4),
            "f1":        round(mean_f1,        4),
        },
        "token_f1_mean": round(mean_token_f1, 4),
        "per_sample":    predictions,
    }

    out_path = OUT_DIR / f"bertscore_results_{RUN_ID}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nFull results saved → {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()