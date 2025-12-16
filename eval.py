import math
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from config import EvalConfig


def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [ex["attention_mask"] for ex in batch], dtype=torch.long
    )
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate_model(model, name: str, dataloader, device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    return {"name": name, "loss": mean_loss, "ppl": ppl}


@validate_call
def main(conf: EvalConfig) -> None:
    dtype = torch.bfloat16 if conf.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(conf.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(conf.dataset_name, split=conf.eval_split)
    if conf.max_eval_samples:
        raw_eval = raw_dataset.select(range(conf.max_eval_samples))
    else:
        raw_eval = raw_dataset

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            max_length=conf.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    eval_dataset = raw_eval.map(format_example, remove_columns=raw_eval.column_names)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=conf.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load models
    print("Loading teacher...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.teacher_model_name,
        torch_dtype=dtype,
        device_map=conf.device_map,
        trust_remote_code=True,
    )

    print("Loading PTQ student (4-bit, no finetune)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    ptq_student = AutoModelForCausalLM.from_pretrained(
        conf.ptq_student_name,
        quantization_config=bnb_config,
        device_map=conf.device_map,
        trust_remote_code=True,
    )

    print("Loading KD (off-policy) student...")
    kd_student = AutoModelForCausalLM.from_pretrained(
        str(conf.kd_student_dir),
        torch_dtype=dtype,
        device_map=conf.device_map,
        trust_remote_code=True,
    )

    print("Loading on-policy KD student...")
    onpolicy_student = AutoModelForCausalLM.from_pretrained(
        str(conf.onpolicy_student_dir),
        torch_dtype=dtype,
        device_map=conf.device_map,
        trust_remote_code=True,
    )

    device = list(teacher_model.parameters())[0].device

    results: List[Dict[str, Any]] = []
    results.append(evaluate_model(teacher_model, "teacher_fp", eval_loader, device))
    results.append(evaluate_model(ptq_student, "student_ptq_4bit", eval_loader, device))
    results.append(
        evaluate_model(kd_student, "student_kd_offpolicy", eval_loader, device)
    )
    results.append(
        evaluate_model(onpolicy_student, "student_kd_onpolicy", eval_loader, device)
    )

    print("\n=== Perplexity comparison on Tulu-3 eval slice ===")
    print(f"{'model':30s} | {'loss':10s} | {'ppl':10s}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:30s} | {r['loss']:.4f}     | {r['ppl']:.2f}")


if __name__ == "__main__":
    main(**parse_argv())
