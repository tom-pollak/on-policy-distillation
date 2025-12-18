import os

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

from config import TrainConfig, Tee


def filter_dataset(dataset, tokenizer, max_length, min_response_tokens=32):
    # Skip samples with no conversation turns
    def more_than_one_message(example):
        return len(example.get("messages", [])) > 0

    # Skip samples with empty message content (whitespace-only).
    # Empty prompts cause IndexError in model.generate() when it tries to check
    # inputs_tensor[:, -1] but dimension 1 has size 0.
    def non_empty_message(example):
        return all(m.get("content", "").strip() for m in example["messages"])

    # Skip samples where prompt is too long to leave room for response tokens.
    # GKDTrainer.compute_loss does: logits[:, prompt_len - 1 : -1, :]
    # If prompt_len >= seq_len, this produces an empty tensor -> IndexError
    def has_room_for_response(example):
        prompt_msgs = [m for m in example["messages"] if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        return prompt_len < max_length - min_response_tokens

    filters = [more_than_one_message, non_empty_message, has_room_for_response]
    return dataset.filter(lambda x: all(f(x) for f in filters))


@validate_call
def main(cfg: TrainConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    Tee.redirect_stdout_stderr(cfg.output_dir / "train.log")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_JOB_TYPE"] = "train"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset_name, split="train")
    dataset = filter_dataset(dataset, tokenizer, max_length=cfg.max_length)

    # Models
    teacher = cfg.load_model()
    student = cfg.load_quant_model("qat")
    if cfg.use_lora:
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
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    training_args = GKDConfig(
        bf16=cfg.mixed_precision == "bf16",
        fp16=cfg.mixed_precision == "fp16",
        # torch_compile=True,
        # torch_compile_backend="inductor",
        report_to=["wandb"],
        run_name=cfg.output_dir.stem,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        **cfg.trainer_kwargs(),
    )

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))


if __name__ == "__main__":
    main(TrainConfig(**parse_argv()))
