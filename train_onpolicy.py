import os
import random

os.environ["WANDB_PROJECT"] = "on-policy-distillation"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.gkd import GKDConfig, GKDTrainer

from config import OnPolicyKDConfig


@validate_call
def main(conf: OnPolicyKDConfig = OnPolicyKDConfig()) -> None:
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)

    dtype = torch.bfloat16 if conf.mixed_precision == "bf16" else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset - extract prompts (drop final assistant turn)
    raw_dataset = load_dataset(conf.dataset_name, split="train")
    if conf.max_train_samples is not None:
        raw_dataset = raw_dataset.select(range(conf.max_train_samples))

    def extract_prompt(example):
        messages = example["messages"]
        # Drop last assistant/model message if present
        if len(messages) >= 2 and messages[-1]["role"] in ("assistant", "model"):
            messages = messages[:-1]
        return {"messages": messages}

    raw_dataset = raw_dataset.map(extract_prompt)

    # Teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.teacher_model_name,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Student model (quantized + LoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        conf.student_model_name,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
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

    # GKD training config
    training_args = GKDConfig(
        output_dir=str(conf.output_dir),
        per_device_train_batch_size=conf.per_device_train_batch_size,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        learning_rate=conf.learning_rate,
        max_steps=conf.max_train_steps,
        warmup_ratio=conf.warmup_ratio,
        logging_steps=50,
        save_steps=500,
        bf16=conf.mixed_precision == "bf16",
        fp16=conf.mixed_precision == "fp16",
        report_to=["wandb"],
        run_name="on_policy_gkd",
        lmbda=1.0,  # 1.0 = pure on-policy (generate from student)
        beta=1.0,  # 1.0 = reverse KL (mode-seeking), 0.0 = forward KL
        temperature=1.0,
        max_new_tokens=conf.max_new_tokens,
    )

    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=raw_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(conf.output_dir))
    tokenizer.save_pretrained(str(conf.output_dir))


if __name__ == "__main__":
    main(**parse_argv())
