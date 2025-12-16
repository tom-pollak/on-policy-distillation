import os
import random

os.environ["WANDB_PROJECT"] = "on-policy-distillation"

import torch
import torch.nn.functional as F
from datasets import load_dataset
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from config import KDBaselineConfig


@validate_call
def main(conf: KDBaselineConfig = KDBaselineConfig()) -> None:
    # -----------------------------
    # Seeding
    # -----------------------------
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(conf.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Dataset
    # -----------------------------
    raw_dataset = load_dataset(conf.dataset_name, split="train")

    if conf.max_train_samples is not None:
        raw_train = raw_dataset.select(range(conf.max_train_samples))
    else:
        raw_train = raw_dataset

    if conf.max_eval_samples is not None:
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

    train_dataset = raw_train.map(format_example, remove_columns=raw_train.column_names)
    eval_dataset = raw_eval.map(format_example, remove_columns=raw_eval.column_names)

    # -----------------------------
    # Models
    # -----------------------------
    device_map = conf.device_map
    dtype = torch.bfloat16 if conf.mixed_precision == "bf16" else torch.float16

    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.teacher_model_name,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        conf.student_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # -----------------------------
    # KD Trainer
    # -----------------------------
    class KDTrainer(Trainer):
        def __init__(self, teacher_model=None, kd_alpha=0.5, temperature=1.0, **kwargs):
            super().__init__(**kwargs)
            self.teacher_model = teacher_model
            self.kd_alpha = kd_alpha
            self.temperature = temperature

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=None,
            )
            logits_s = outputs_student.logits / self.temperature

            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                logits_t = outputs_teacher.logits / self.temperature

            loss_ce = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            log_probs_s = F.log_softmax(logits_s, dim=-1)
            probs_t = F.softmax(logits_t, dim=-1)
            loss_kl = F.kl_div(
                log_probs_s,
                probs_t,
                reduction="batchmean",
                log_target=False,
            )

            loss = (
                conf.kd_alpha * loss_ce
                + (1.0 - conf.kd_alpha) * (conf.temperature**2) * loss_kl
            )
            return (loss, outputs_student) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(conf.output_dir),
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.per_device_eval_batch_size,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        learning_rate=conf.learning_rate,
        num_train_epochs=conf.num_train_epochs,
        weight_decay=conf.weight_decay,
        warmup_ratio=conf.warmup_ratio,
        logging_steps=conf.logging_steps,
        eval_steps=conf.eval_steps,
        save_steps=conf.save_steps,
        save_total_limit=2,
        bf16=conf.mixed_precision == "bf16",
        fp16=conf.mixed_precision == "fp16",
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name="baseline",
        torch_compile=conf.dynamo_backend != "no",
        torch_compile_backend=conf.dynamo_backend
        if conf.dynamo_backend != "no"
        else None,
    )

    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        kd_alpha=conf.kd_alpha,
        temperature=conf.temperature,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(conf.output_dir))
    tokenizer.save_pretrained(str(conf.output_dir))


if __name__ == "__main__":
    main(**parse_argv())
