import os
import random

os.environ["WANDB_PROJECT"] = "on-policy-distillation"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pydantic import validate_call
from pydantic_config import parse_argv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
from torchao.quantization import quantize_

from config import TrainConfig


@validate_call
def main(conf: TrainConfig = TrainConfig()) -> None:
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)

    dtype = torch.bfloat16 if conf.mixed_precision == "bf16" else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(conf.dataset_name, split="train")

    # Teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Student model (quantized + LoRA)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    student_model = AutoModelForCausalLM.from_pretrained(
        conf.model_name,
        torch_dtype=dtype,
        device_map={"": local_rank},
    )
    quantize_(student_model, conf.get_quant_config())

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

    training_args = GKDConfig(
        bf16=conf.mixed_precision == "bf16",
        fp16=conf.mixed_precision == "fp16",
        torch_compile=True,
        torch_compile_backend="inductor",
        # torch_compile_mode="max-autotune",
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        run_name=conf.output_dir.stem,
        # TODO: revisit gradient checkpointing - DDP + LoRA + checkpointing causes
        # "parameter marked ready twice" error even with use_reentrant=False.
        gradient_checkpointing=False,
        **conf.trainer_kwargs(),
    )

    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(conf.output_dir))
    tokenizer.save_pretrained(str(conf.output_dir))


if __name__ == "__main__":
    main(TrainConfig(**parse_argv()))
