from pathlib import Path
from pydantic_config import BaseConfig


class OnPolicyKDConfig(BaseConfig):
    # model & data
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "allenai/tulu-3-sft-mixture"

    max_train_steps: int = 100

    # precision
    mixed_precision: str = "bf16"
    dynamo_backend: str = "inductor"

    # batching / optimisation
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # logging
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100

    # sampling
    temperature: float = 1.0
    max_new_tokens: int = 128

    # misc
    seed: int = 42

    output_dir: Path = Path("./qwen_onpolicy_kd")

    # GKD params
    lmbda: float = 1.0  # 0.0 = off-policy (dataset), 1.0 = on-policy (student rollouts)
    beta: float = 0.0  # 0.0 = forward KL, 1.0 = reverse KL


class EvalConfig(BaseConfig):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    kd_student_dir: Path = Path("./qwen_kd_baseline")
    onpolicy_student_dir: Path = Path("./qwen_onpolicy_merged")  # merged + re-quantized
