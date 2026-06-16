"""
SpaceLLM — Baseline GPT-OSS-20B Inference + BERTScore
=======================================================
Runs the RAW base model (no LoRA) on the full test set
and computes BERTScore to compare against fine-tuned model.

FIXES:
  - Load to CPU first, then force ALL tensors to GPU
  - force_model_to_device() handles MoE expert weights on CPU
  - with torch.cuda.device(DEVICE) context for generate()

Fine-tuned result (for reference):
  BERTScore Precision : 0.8736
  BERTScore Recall    : 0.8857
  BERTScore F1        : 0.8795
"""

import json
import logging
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR  = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")
TEST_FILE = BASE_DIR / "data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR   = BASE_DIR / "fine_tuning_v2/outputs/baseline_bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"baseline_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("Baseline")

# ── Config ────────────────────────────────────────────────────────
MODEL_ID       = "openai/gpt-oss-20b"
MAX_SEQ_LEN    = 2048
MAX_NEW_TOKENS = 256
MAX_SAMPLES    = None
DEVICE         = "cuda:0"

FINETUNED_SCORES = {
    "precision": 0.8736,
    "recall":    0.8857,
    "f1":        0.8795,
}


def force_model_to_device(model, device):
    """
    Aggressively moves every parameter and buffer to device.
    Required for MoE models where expert weights stay on CPU.
    """
    logger.info(f"Force-moving ALL tensors to {device}...")
    moved_params, moved_buffers, skipped = 0, 0, 0

    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            try:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
                moved_params += 1
            except Exception as e:
                logger.warning(f"  Could not move param {name}: {e}")
                skipped += 1

    for name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            try:
                parts = name.split(".")
                mod   = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                setattr(mod, parts[-1], buf.to(device))
                moved_buffers += 1
            except Exception as e:
                logger.warning(f"  Could not move buffer {name}: {e}")
                skipped += 1

    logger.info(f"  Moved {moved_params} params + {moved_buffers} buffers | skipped {skipped}")

    cpu_params = [n for n, p in model.named_parameters() if p.device.type == "cpu"]
    if cpu_params:
        logger.warning(f"  ⚠️  {len(cpu_params)} params still on CPU")
        for n in cpu_params[:5]:
            logger.warning(f"     {n}")
    else:
        logger.info("  ✅ All parameters on GPU")

    return model


def load_test_data(path):
    with path.open(encoding="utf-8") as f:
        raw = f.read().strip()
    data = json.loads(raw) if raw.startswith("[") else \
           [json.loads(l) for l in raw.splitlines() if l.strip()]
    logger.info(f"Loaded {len(data):,} test records")
    return data


def extract_prompt_and_reference(record, tokenizer):
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
        prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.warning(f"Template failed: {e}")
        return None, None
    return prompt, ref_answer


def log_gpu_memory(label=""):
    for i in range(torch.cuda.device_count()):
        props  = torch.cuda.get_device_properties(i)
        alloc  = torch.cuda.memory_allocated(i)  / 1024**3
        total  = props.total_memory               / 1024**3
        logger.info(f"GPU {i} [{props.name}] {label} | Alloc={alloc:.2f}GB / {total:.2f}GB")


def main():
    logger.info("=" * 60)
    logger.info("  GPT-OSS-20B Baseline — Inference + BERTScore")
    logger.info(f"  Run ID : {RUN_ID}")
    logger.info(f"  Model  : {MODEL_ID}  (NO LoRA)")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    logger.info(f"Tokenizer loaded  vocab={tokenizer.vocab_size:,}")

    # ── Load to CPU, then force GPU ───────────────────────────────
    logger.info(f"\nLoading base model to CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    logger.info(f"Loaded to CPU in {time.time()-t0:.1f}s")

    model = force_model_to_device(model, DEVICE)
    model.eval()
    log_gpu_memory("after force GPU")

    # ── Test data ─────────────────────────────────────────────────
    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    # ── Inference ─────────────────────────────────────────────────
    logger.info(f"\nRunning inference on {len(test_records):,} samples...")
    references, hypotheses, predictions = [], [], []
    skipped = 0
    t_inf   = time.time()

    with torch.no_grad():
        for i, record in enumerate(test_records):
            prompt, ref = extract_prompt_and_reference(record, tokenizer)
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
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            try:
                with torch.cuda.device(DEVICE):
                    gen_ids = model.generate(
                        **enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_ids   = gen_ids[0][enc["input_ids"].shape[1]:]
                generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
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
                    f"  [{i+1}/{len(test_records)}]  valid={len(predictions)}  "
                    f"skipped={skipped}  speed={rate:.2f} it/s  ETA={eta/60:.1f} min"
                )

    inf_time = time.time() - t_inf
    logger.info(f"\nInference done — {len(predictions):,} samples  "
                f"skipped={skipped}  time={inf_time/60:.1f} min")

    if not predictions:
        logger.error("No predictions — aborting")
        return

    # Save raw predictions immediately
    raw_path = OUT_DIR / f"predictions_raw_{RUN_ID}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw predictions saved → {raw_path}")

    # ── Token F1 ──────────────────────────────────────────────────
    token_f1s = []
    for hyp, ref in zip(hypotheses, references):
        ref_toks = set(ref.lower().split())
        gen_toks = set(hyp.lower().split())
        if ref_toks or gen_toks:
            overlap = len(ref_toks & gen_toks)
            prec    = overlap / len(gen_toks) if gen_toks else 0.0
            rec     = overlap / len(ref_toks) if ref_toks else 0.0
            f1      = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        token_f1s.append(f1)
    mean_token_f1 = sum(token_f1s) / len(token_f1s)

    # ── BERTScore ─────────────────────────────────────────────────
    logger.info("\nComputing BERTScore...")
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en", model_type="roberta-large",
        batch_size=32, verbose=True, device=DEVICE,
    )
    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    f1_list = F1.tolist()

    # ── Print comparison ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BASELINE vs FINE-TUNED COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  {'Metric':<20} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}")
    logger.info(f"  {'-'*56}")
    for metric, ft_val in FINETUNED_SCORES.items():
        base_val = {"precision": mean_p, "recall": mean_r, "f1": mean_f1}[metric]
        delta    = ft_val - base_val
        sign     = "+" if delta >= 0 else ""
        logger.info(f"  {metric.capitalize():<20} {base_val:>12.4f} {ft_val:>12.4f} {sign}{delta:>9.4f}")
    delta_f1   = FINETUNED_SCORES["f1"] - mean_f1
    pct_change = (delta_f1 / mean_f1) * 100
    logger.info("=" * 60)
    if delta_f1 > 0:
        logger.info(f"  ✅ Fine-tuning improved F1 by {delta_f1:.4f} ({pct_change:.2f}%)")
    else:
        logger.info(f"  ⚠️  Baseline stronger by {abs(delta_f1):.4f} ({abs(pct_change):.2f}%)")

    # ── Save results ──────────────────────────────────────────────
    for i, p in enumerate(predictions):
        p["bert_f1"]  = round(f1_list[i],   4)
        p["token_f1"] = round(token_f1s[i], 4)

    results = {
        "run_id": RUN_ID, "model": MODEL_ID,
        "note": "RAW base model — NO LoRA",
        "summary": {
            "total_records": len(test_records), "evaluated": len(predictions),
            "skipped": skipped, "inference_time_min": round(inf_time/60, 2),
        },
        "bert_score": {
            "model": "roberta-large",
            "precision": round(mean_p, 4), "recall": round(mean_r, 4), "f1": round(mean_f1, 4),
        },
        "token_f1_mean": round(mean_token_f1, 4),
        "comparison_vs_finetuned": {
            "baseline_f1": round(mean_f1, 4),
            "finetuned_f1": FINETUNED_SCORES["f1"],
            "delta_f1": round(delta_f1, 4),
            "pct_change": round(pct_change, 2),
        },
        "per_sample": predictions,
    }

    out_path = OUT_DIR / f"baseline_results_{RUN_ID}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {out_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()