# On-Policy Distillation

## Setup

Install dependencies and lock the environment with `uv`:

```bash
uv sync
```

torch transformers datasets accelerate bitsandbytes peft scipy pydantic pydantic_config

## Training

### 1. Off-policy KD baseline

```bash
uv run accelerate launch train_baseline.py
```

### 2. On-policy KD

```bash
uv run accelerate launch train_onpolicy.py
```

### 3. Perplexity comparison (teacher vs PTQ vs KD vs on-policy KD)

```bash
uv run python eval_compare_qwen.py
```

## Results
