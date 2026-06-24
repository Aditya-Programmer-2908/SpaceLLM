"""
SpaceLLM — BERTScore Eval with spacellm_v2_adapter
===================================================
Loads gpt-oss-20b + spacellm_v2_adapter (local), runs batched inference
on the test set, computes BERTScore + Token F1, saves results to JSON.

Usage
-----
    python bertscore_v2.py
    python bertscore_v2.py --batch-size 32
    python bertscore_v2.py --engine vllm
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM")
ADAPTER_PATH = BASE_DIR / "backend/spacellm_adapters/spacellm_v2_adapter"

TEST_FILE  = BASE_DIR / "Model_training_&_Data_Extraction/data_processing/DatasetA_core_QA_v2/test.json"
OUT_DIR    = BASE_DIR / "Model_training_&_Data_Extraction/fine_tuning_v2/outputs/bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_DIR / f"inference_v2_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("BERTScoreV2")

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID         = "openai/gpt-oss-20b"
MAX_SEQ_LEN      = 2048
MAX_NEW_TOKENS   = 256
MAX_SAMPLES      = None        # set to int to cap eval set
DEVICE           = "cuda:0"
SMOKE_TEST_N     = 5
DEFAULT_BATCH    = 16


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


def looks_degenerate(token_ids: torch.Tensor) -> tuple[bool, str]:
    ids = token_ids.tolist()
    if not ids:
        return True, "empty generation"
    counts            = Counter(ids)
    top_id, top_count = counts.most_common(1)[0]
    frac              = top_count / len(ids)
    if frac > 0.6:
        return True, f"token id {top_id} makes up {frac:.0%} of generation"
    return False, ""


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


def detect_adapter_vocab_size(adapter_path: Path) -> int:
    from safetensors import safe_open
    sf_path = adapter_path / "adapter_model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found at {sf_path}")

    candidates = []
    with safe_open(str(sf_path), framework="pt") as f:
        for key in f.keys():
            if "lm_head" in key:
                shape = f.get_slice(key).get_shape()
                candidates.append((key, shape))

    if not candidates:
        raise RuntimeError(f"No lm_head tensors found in {sf_path}")

    key, shape = max(candidates, key=lambda kv: max(kv[1]))
    vocab = max(shape)
    logger.info(f"  Detected vocab size {vocab:,} from '{key}' shape={list(shape)}")
    return vocab


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    logger.info(f"Adapter path : {ADAPTER_PATH}")
    required_vocab = detect_adapter_vocab_size(ADAPTER_PATH)

    logger.info("Loading tokenizer from adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH), trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("  pad_token unset — falling back to eos_token")
    tokenizer.padding_side = "left"
    logger.info(f"  vocab={tokenizer.vocab_size:,}  eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}")

    logger.info(f"Loading base model {MODEL_ID} ...")
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

    untie_lm_head(model, "(base, pre-resize)")
    current_vocab = model.get_output_embeddings().weight.shape[0]
    if current_vocab != required_vocab:
        logger.info(f"  Resizing vocab {current_vocab:,} → {required_vocab:,} ...")
        model.resize_token_embeddings(required_vocab)
        model.config.vocab_size = required_vocab
        if id(model.get_input_embeddings().weight) == id(model.get_output_embeddings().weight):
            untie_lm_head(model, "(re-untied after resize)")
        logger.info(f"  ✅ Vocab confirmed at {model.get_output_embeddings().weight.shape[0]:,}")
    else:
        logger.info("  Vocab already matches — no resize needed.")

    logger.info(f"Attaching spacellm_v2_adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(
        model, str(ADAPTER_PATH),
        adapter_name="spacellm_v2",
        is_trainable=False,
        token=hf_token,
    )
    model.set_adapter("spacellm_v2")
    logger.info("  ✅ spacellm_v2_adapter attached")

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

        n_gpus     = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            free          = torch.cuda.mem_get_info(i)[0]
            max_memory[i] = f"{int(max(0, free - 4 * 1024**3) / 1024**3)}GiB"
        max_memory["cpu"] = "80GiB"
        logger.info(f"  max_memory: {max_memory}")

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
        model      = dispatch_model(model, device_map=device_map)
        logger.info(f"  Dispatched in {time.time()-t_dispatch:.1f}s")

    except Exception as e:
        logger.warning(f"  dispatch_model failed ({e}) — falling back to {DEVICE}")
        model = model.to(DEVICE)

    log_gpu_memory("after dispatch")
    return model, tokenizer


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_one(model, tokenizer, prompt: str) -> tuple[torch.Tensor, str]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=False,
    )
    enc = {k: v.to(get_input_device(model)) for k, v in enc.items()}
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
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, prompts: list[str]) -> list[tuple[torch.Tensor, str]]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=True,
    )
    enc = {k: v.to(get_input_device(model)) for k, v in enc.items()}
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
        new_ids = out_ids[inp_ids.shape[0]:]
        results.append((new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()))
    return results


def flush_batch(model, tokenizer, pending_prompts, pending_meta):
    try:
        batch_results = generate_batch(model, tokenizer, pending_prompts)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM on batch of {len(pending_prompts)} — retrying serially")
        torch.cuda.empty_cache()
        batch_results = []
        for p in pending_prompts:
            try:
                batch_results.append(generate_one(model, tokenizer, p))
            except Exception as e:
                logger.warning(f"  Serial fallback failed: {e}")
                batch_results.append((torch.tensor([]), ""))

    out = []
    for (token_ids, generated), meta in zip(batch_results, pending_meta):
        out.append((meta["ref"], generated, {
            "sample_id": meta["sample_id"],
            "reference": meta["ref"],
            "generated": generated,
        }))
    return out


# ── Eval loop ──────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, test_records: list[dict], batch_size: int) -> dict:

    # Smoke test
    logger.info(f"\n── Smoke test ({SMOKE_TEST_N} samples) ──")
    for i, record in enumerate(test_records[:SMOKE_TEST_N]):
        hf_messages, ref = extract_messages_and_reference(record)
        if hf_messages is None:
            continue
        prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        token_ids, text = generate_one(model, tokenizer, prompt)
        bad, reason = looks_degenerate(token_ids)
        logger.info(f"  [{i}] {text[:120]!r}")
        if bad:
            raise RuntimeError(f"Smoke test failed on sample {i}: {reason}")
    logger.info("  ✅ Smoke test passed\n")

    logger.info(f"Running batched inference — {len(test_records):,} samples  batch_size={batch_size}")
    references, hypotheses, predictions = [], [], []
    pending_prompts, pending_meta = [], []
    skipped   = 0
    n_batches = 0
    t_inf     = time.time()

    for i, record in enumerate(test_records):
        hf_messages, ref = extract_messages_and_reference(record)
        if hf_messages is None:
            skipped += 1
            continue

        prompt = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        pending_prompts.append(prompt)
        pending_meta.append({"ref": ref, "sample_id": record.get("sample_id", i)})

        if len(pending_prompts) >= batch_size:
            for ref_ans, gen, pred in flush_batch(model, tokenizer, pending_prompts, pending_meta):
                references.append(ref_ans)
                hypotheses.append(gen)
                predictions.append(pred)
            n_batches += 1
            pending_prompts, pending_meta = [], []

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

    if pending_prompts:
        for ref_ans, gen, pred in flush_batch(model, tokenizer, pending_prompts, pending_meta):
            references.append(ref_ans)
            hypotheses.append(gen)
            predictions.append(pred)

    inf_time = time.time() - t_inf
    logger.info(f"Inference done — {len(predictions):,} samples  skipped={skipped}  time={inf_time/60:.1f} min")

    # Token F1
    token_f1s = []
    for hyp, ref in zip(hypotheses, references):
        ref_t = set(ref.lower().split())
        gen_t = set(hyp.lower().split())
        if ref_t or gen_t:
            overlap = len(ref_t & gen_t)
            prec = overlap / len(gen_t) if gen_t else 0.0
            rec  = overlap / len(ref_t) if ref_t else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        token_f1s.append(f1)
    mean_token_f1 = sum(token_f1s) / len(token_f1s)
    logger.info(f"Token F1 mean: {mean_token_f1:.4f}")

    # BERTScore
    logger.info("Computing BERTScore ...")
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en", model_type="roberta-large",
        batch_size=64, verbose=True, device=DEVICE,
    )
    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    f1_list = F1.tolist()
    logger.info(f"BERTScore  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}")

    for i, pred in enumerate(predictions):
        pred["bert_f1"]  = round(f1_list[i], 4)
        pred["token_f1"] = round(token_f1s[i], 4)

    results = {
        "run_id":       RUN_ID,
        "base_model":   MODEL_ID,
        "adapter":      str(ADAPTER_PATH),
        "adapter_name": "spacellm_v2_adapter",
        "config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_seq_len":    MAX_SEQ_LEN,
            "batch_size":     batch_size,
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

    raw_path = OUT_DIR / f"predictions_v2_{RUN_ID}.json"
    out_path = OUT_DIR / f"bertscore_v2_{RUN_ID}.json"
    raw_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path.write_text(json.dumps(results,     indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved → {out_path}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpaceLLM v2 BERTScore eval")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SpaceLLM — BERTScore Eval  [spacellm_v2_adapter]")
    logger.info(f"  Run ID   : {RUN_ID}")
    logger.info(f"  Adapter  : {ADAPTER_PATH}")
    logger.info(f"  Test     : {TEST_FILE}")
    logger.info(f"  Batch    : {args.batch_size}")
    logger.info("=" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    test_records = load_test_data(TEST_FILE)
    if MAX_SAMPLES:
        test_records = test_records[:MAX_SAMPLES]

    model, tokenizer = load_model_and_tokenizer()
    results = run_eval(model, tokenizer, test_records, batch_size=args.batch_size)

    logger.info("=" * 60)
    logger.info(f"  BERTScore F1  : {results['bert_score']['f1']:.4f}")
    logger.info(f"  Token F1      : {results['token_f1_mean']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
