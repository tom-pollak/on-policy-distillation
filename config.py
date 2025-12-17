from pathlib import Path
from typing import Literal

from pydantic_config import BaseConfig
from torchao.quantization import Int4WeightOnlyConfig
from torchao.quantization.qat import QATConfig
from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig
from transformers import BitsAndBytesConfig


class SharedConfig(BaseConfig):
    """Base config with shared model and quantization settings."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # torchao: int4, nvfp4 (B200 only) | bitsandbytes: bnb_fp4, bnb_nf4
    quant_type: Literal["int4", "nvfp4", "bnb_fp4", "bnb_nf4"] = "bnb_nf4"
    wandb_project: str = "on-policy-distillation"

    @property
    def quant_backend(self) -> Literal["torchao", "bitsandbytes"]:
        return "bitsandbytes" if self.quant_type.startswith("bnb_") else "torchao"

    def get_bnb_config(self, dtype):
        assert self.quant_backend == "bitsandbytes"
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=self.quant_type.removeprefix("bnb_"),
        )

    def get_torchao_config(self):
        """Get torchao quantization config."""
        assert self.quant_backend == "torchao"
        match self.quant_type:
            case "int4":
                return Int4WeightOnlyConfig()
            case "nvfp4":
                return NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True)
            case _:
                raise ValueError(f"No torchao config for {self.quant_type}")

    def get_torchao_qat_config(self):
        """Get QAT config for training with fake quantization."""
        return QATConfig(self.get_torchao_config(), step="prepare")


class TrainConfig(SharedConfig):
    """Config for on-policy knowledge distillation training."""

    # data
    dataset_name: str = "allenai/tulu-3-sft-mixture"

    # precision
    mixed_precision: str = "bf16"

    # misc
    seed: int = 42

    # trainer
    max_steps: int = 500

    # batching / optimisation
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # logging
    logging_steps: int = 1
    save_steps: int = 100

    # sampling
    temperature: float = 1.0
    max_new_tokens: int = 128

    output_dir: Path = Path("./qwen_onpolicy_kd")

    # GKD params
    lmbda: float = 1.0  # 0.0 = off-policy (dataset), 1.0 = on-policy (student rollouts)
    beta: float = 0.0  # 0.0 = forward KL, 1.0 = reverse KL

    def trainer_kwargs(self):
        return self.model_dump(
            exclude=[
                "model_name",
                "dataset_name",
                "mixed_precision",
                "dynamo_backend",
                "seed",
                "quant_type",
                "wandb_project",
            ]
        )


class EvalConfig(SharedConfig):
    """Config for model evaluation."""

    lora_paths: list[Path] = [Path("./qwen_kd_baseline"), Path("./qwen_onpolicy_kd")]
    tasks: list[str] = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"]
