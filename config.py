import os
import sys
from pathlib import Path
from typing import Literal

import torch
from pydantic import field_validator, model_validator
from pydantic_config import BaseConfig
from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.qat import QATConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class SharedConfig(BaseConfig):
    """Base config with shared model and quantization settings."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    mixed_precision: str = "bf16"
    # torchao: int4, nvfp4 (B200 only) | bitsandbytes: bnb_fp4, bnb_nf4
    quant_type: Literal["int4", "nvfp4", "bnb_fp4", "bnb_nf4"] = "int4"
    wandb_project: str = "on-policy-distillation"

    @property
    def dtype(self):
        return torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16

    @property
    def quant_backend(self) -> Literal["torchao", "bitsandbytes"]:
        return "bitsandbytes" if self.quant_type.startswith("bnb_") else "torchao"

    def _get_bnb_config(self):
        assert self.quant_backend == "bitsandbytes"
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type=self.quant_type.removeprefix("bnb_"),
        )

    def _get_torchao_config(self):
        """Get torchao quantization config."""
        assert self.quant_backend == "torchao"
        match self.quant_type:
            case "int4":
                return Int4WeightOnlyConfig()
            case "nvfp4":
                return NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True)
            case _:
                raise ValueError(f"No torchao config for {self.quant_type}")

    def get_quant_config(self):
        if self.quant_backend == "bitsandbytes":
            return self._get_bnb_config()
        else:
            return self._get_torchao_config()

    def get_qat_config(self):
        if self.quant_backend == "bitsandbytes":
            return self._get_bnb_config()
        else:
            return QATConfig(self._get_torchao_config(), step="prepare")

    def load_model(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype, device_map={"": local_rank}
        )

    def load_quant_model(self, method: Literal["qat", "ptq"] = "qat"):
        quant_config = (
            self.get_qat_config() if method == "qat" else self.get_quant_config()
        )
        if isinstance(quant_config, BitsAndBytesConfig):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map={"": local_rank},
            )
        else:  # torchao
            model = self.load_model()
            quantize_(model, quant_config)
        return model


class TrainConfig(SharedConfig):
    """Config for on-policy knowledge distillation training."""

    # data
    dataset_name: str = "allenai/tulu-3-sft-mixture"

    # misc
    seed: int = 42

    use_lora: bool = True
    resume: bool = True

    # trainer
    max_steps: int = 1000
    max_length: int = 1024

    # batching / optimisation
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.2

    # logging
    logging_steps: int = 1
    save_steps: int = 100

    # sampling
    max_new_tokens: int = 128
    min_new_tokens: int = 16
    include_tokens_per_second: bool = True

    output_dir: Path | None = None  # auto-generated from hyperparams if not set

    # GKD params
    lmbda: float = 1.0  # 0.0 = off-policy (dataset), 1.0 = on-policy (student rollouts)
    beta: float = 1.0  # 0.0 = forward KL, 1.0 = reverse KL
    seq_kd: bool = (
        False  # sequence-level KD (supervised FT on teacher-generated output)
    )

    # wandb
    tags: list[str] = ["train"]

    # eval
    do_eval: bool = True

    # perplexity eval during training
    perplexity_dataset: str | None = "wikitext"  # None to disable
    eval_strategy: str = "steps"
    eval_steps: int = 250

    @model_validator(mode="after")
    def set_output_dir(self):
        if self.output_dir is not None:
            return self

        # Format: lmbda{λ}_lr{lr}_beta{β}
        # Only include non-default values to keep names short
        parts = [f"lmbda{self.lmbda:g}"]

        # Learning rate: format as 1e-5 style
        lr = self.learning_rate
        exp = int(f"{lr:e}".split("e")[1])
        mantissa = lr / (10**exp)
        if mantissa == 1.0:
            parts.append(f"lr1e{exp}")
        else:
            parts.append(f"lr{mantissa:g}e{exp}")

        # Beta if not default (1.0)
        if self.beta != 1.0:
            parts.append(f"beta{self.beta:g}")

        self.output_dir = Path("dump") / "_".join(parts)
        return self

    def trainer_kwargs(self):
        return self.model_dump(
            exclude={
                "model_name",
                "quant_type",
                "wandb_project",
                "dataset_name",
                "mixed_precision",
                "seed",
                "use_lora",
                "resume",
                "tags",
                "do_eval",
                "eval_strategy",  # set dynamically based on perplexity_dataset
            }
        )


class EvalConfig(SharedConfig):
    """Config for model evaluation."""

    lora_paths: list[Path] = []
    eval_teacher: bool = True
    # wandb
    tags: list[str] = ["eval"]
    tasks: list[str] = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "mmlu",
    ]
    # perplexity eval
    perplexity_dataset: str | None = "wikitext"  # None to skip

    @field_validator("lora_paths", "tasks", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return [v] if not isinstance(v, list) else v


class QuantEvalConfig(SharedConfig):
    """Config for GPTQ/AWQ quantization evaluation."""

    # calibration
    dataset_name: str = "allenai/tulu-3-sft-mixture"
    num_calibration_samples: int = 512
    max_seq_length: int = 2048

    output_dir: Path = Path("./qwen_4b_quant")

    # wandb
    tags: list[str] = ["quant-eval"]

    tasks: list[str] = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "mmlu",
    ]
    # perplexity eval
    perplexity_dataset: str | None = "wikitext"  # None to skip

    @field_validator("tasks", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return [v] if not isinstance(v, list) else v


class Tee:
    def __init__(self, file_path, stream, main_only=False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.active = not main_only or local_rank == 0
        if self.active:
            self.stream = stream
            self.file = open(file_path, "a")
        else:
            devnull = open(os.devnull, "w")
            self.stream = self.file = devnull

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    @classmethod
    def redirect_stdout_stderr(cls, log_path, main_only=False):
        sys.stdout = cls(log_path, sys.stdout, main_only)
        sys.stderr = cls(log_path, sys.stderr, main_only)
