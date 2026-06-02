# ================= SPACE LLM TRAINING (SAFE VERSION) =================

import sys, types, json, logging, os, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import dispatch_model, infer_auto_device_map

IGNORE_INDEX = -100

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpaceLLM")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ================= PATHS =================
TRAIN_FILE = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/train.jsonl")
VAL_FILE   = Path("/mnt/DATA/saurabh/aditya/SpaceLLM/data_processing/DatasetA_core_QA/validation.jsonl")

OUTPUT_DIR = Path("./outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FINAL_DIR  = OUTPUT_DIR / "final"

for d in [CKPT_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================= LOAD JSONL =================
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ================= SAFE TOKENISATION =================
def build_chat(record, tokenizer):
    msgs = record["messages"]

    chat = []
    for m in msgs:
        role = "system" if m["role"] == "developer" else m["role"]
        chat.append({"role": role, "content": m["content"]})

    return chat


def tokenize(record, tokenizer, max_len=2048):
    chat = build_chat(record, tokenizer)

    # FULL TEXT (single source of truth)
    text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
    )

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if len(input_ids) < 4:
        return None

    # ================= FIX: SAFE LABEL CREATION =================
    labels = input_ids.copy()

    # mask prompt (everything except assistant completion)
    # safer method: detect last assistant block by template fallback
    sep = tokenizer.eos_token_id

    # DO NOT trust template offsets → instead:
    # only train on last chunk heuristic
    cutoff = len(input_ids) // 2

    labels[:cutoff] = [IGNORE_INDEX] * cutoff

    # HARD SAFETY FILTER
    labels = [
        t if t == IGNORE_INDEX or (0 <= t < len(tokenizer))
        else IGNORE_INDEX
        for t in labels
    ]

    # ensure at least some learning signal
    if all(x == IGNORE_INDEX for x in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(records, tokenizer):
    out = []
    for r in records:
        x = tokenize(r, tokenizer)
        if x:
            out.append(x)
    return Dataset.from_list(out)


# ================= MODEL LOAD =================
def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    return model


# ================= GPU DISPATCH =================
def dispatch(model):
    n_gpus = torch.cuda.device_count()

    max_memory = {i: "40GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "80GiB"

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=[],
    )

    return dispatch_model(model, device_map=device_map)


# ================= TRAIN =================
def main():

    model_id = "openai/gpt-oss-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = load_model(model_id)

    logger.info("Dispatching model...")
    model = dispatch(model)

    # ================= FIX VOCAB =================
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ================= DATA =================
    train_data = load_jsonl(TRAIN_FILE)
    val_data   = load_jsonl(VAL_FILE)

    train_ds = build_dataset(train_data, tokenizer)
    val_ds   = build_dataset(val_data, tokenizer)

    # ================= LoRA =================
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["lm_head"],
    )

    model = get_peft_model(model, lora)

    # ================= COLLATOR (IMPORTANT FIX) =================
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ================= TRAINING ARGS =================
    args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ================= TRAIN =================
    logger.info("START TRAINING")

    trainer.train()

    # ================= SAVE =================
    trainer.model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    logger.info("DONE → saved to final dir")


if __name__ == "__main__":
    main()