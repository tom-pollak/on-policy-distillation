# On-Policy Distillation

## Setup

Install dependencies and lock the environment with `uv`:

```bash
uv sync
```

torch transformers datasets accelerate bitsandbytes peft scipy pydantic pydantic_config

## Training

```
k8s/train.sh qwen_onpolicy_4b_int4 --lmbda 1 --quant_type int4

k8s/train.sh qwen_kd_baseline --lmbda 0 --quant_type int4

k8s/eval.sh eval --lora_paths qwen_kd_baseline qwen_onpolicy_4b_int4
```

### 1. Off-policy KD baseline

```bash
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant_type int4
```

### 2. On-policy KD

```bash
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant_type int4
```

### 3. Perplexity comparison (teacher vs PTQ vs KD vs on-policy KD)

```bash
uv run python eval.py
```

## Results

```
model                     | hellaswa | arc_easy | arc_chal | winogran |     mmlu
teacher                   |   0.5262 |   0.8308 |   0.5572 |   0.6835 |   0.7067
teacher_ptq               |          |          |   0.5410 |          |
```
