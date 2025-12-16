from pathlib import Path
from pydantic_config import BaseConfig


class CommonModelConfig(BaseConfig):
    # ---- Models / data ----
    teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "allenai/tulu-3-sft-mixture"

    # sequence / precision
    max_seq_length: int = 2048
    bf16: bool = True
    device_map: str = (
        "auto"  # used in baseline/eval; on-policy uses single-GPU by default
    )

    # data slicing (train/eval)
    max_train_samples: int | None = 100_000
    max_eval_samples: int | None = 2_000

    # batching / optimisation (shared defaults)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    # misc
    seed: int = 42


class KDBaselineConfig(CommonModelConfig):
    # training length
    num_train_epochs: float = 1.0

    # logging / ckpts
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: Path = Path("./qwen_kd_baseline")

    # KD-specific
    kd_alpha: float = 0.5  # weight on supervised CE; 1 - alpha on KL
    temperature: float = 1.0


class OnPolicyKDConfig(CommonModelConfig):
    # we reuse:
    # - per_device_train_batch_size as rollout batch_size
    # - gradient_accumulation_steps, learning_rate, warmup_ratio, seed, etc.

    max_train_steps: int = 10_000
    max_new_tokens: int = 128
    output_dir: Path = Path("./qwen_onpolicy_kd")


class EvalConfig(CommonModelConfig):
    # which student checkpoints to compare
    ptq_student_name: str = "Qwen/Qwen2.5-7B-Instruct"
    kd_student_dir: Path = Path("./qwen_kd_baseline")
    onpolicy_student_dir: Path = Path("./qwen_onpolicy_kd")

    eval_split: str = "train"  # Tulu-3 is a single split; we just slice
    # per_device_eval_batch_size, max_eval_samples, max_seq_length, etc.
    # are inherited from CommonModelConfig
