from pathlib import Path
from pydantic_config import BaseConfig


class CommonModelConfig(BaseConfig):
    # ---- Models / data ----
    teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "allenai/tulu-3-sft-mixture"

    # sequence / precision
    max_seq_length: int = 2048
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    dynamo_backend: str = "inductor"  # "no", "inductor", etc.

    # batching / optimisation (shared defaults)
    per_device_train_batch_size: int = 4  # H100s can handle much more
    per_device_eval_batch_size: int = 18
    gradient_accumulation_steps: int = 1  # reduce since batch is larger
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # logging / ckpts (shared)
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100

    # KD shared
    temperature: float = 1.0

    # misc
    seed: int = 42


class OnPolicyKDConfig(CommonModelConfig):
    # we reuse:
    # - per_device_train_batch_size as rollout batch_size
    # - gradient_accumulation_steps, learning_rate, warmup_ratio, seed, etc.

    max_train_steps: int = 100
    max_new_tokens: int = 128
    output_dir: Path = Path("./qwen_onpolicy_kd")

    # GKD params
    lmbda: float = 1.0  # 0.0 = off-policy (dataset), 1.0 = on-policy (student rollouts)
    beta: float = 0.0  # 0.0 = forward KL, 1.0 = reverse KL


class MergeConfig(BaseConfig):
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_dir: Path = Path("./qwen_onpolicy_kd")
    output_dir: Path = Path("./qwen_onpolicy_merged")
    bf16: bool = True


class EvalConfig(CommonModelConfig):
    # which student checkpoints to compare
    ptq_student_name: str = "Qwen/Qwen2.5-7B-Instruct"
    kd_student_dir: Path = Path("./qwen_kd_baseline")
    onpolicy_student_dir: Path = Path("./qwen_onpolicy_merged")  # merged + re-quantized

    eval_split: str = "train"  # Tulu-3 is a single split; we just slice
    # per_device_eval_batch_size, max_eval_samples, max_seq_length, etc.
    # are inherited from CommonModelConfig
