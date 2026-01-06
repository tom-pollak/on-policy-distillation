# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project exploring knowledge distillation from FP32 teacher models to INT4 student models using on-policy (student-generated) data. Compares on-policy vs off-policy distillation for quantization - finding they perform equivalently (~0.3% difference).

## Commands

```bash
# Install dependencies
uv sync

# Training (single run)
uv run accelerate launch train.py @configs/offpolicy.toml --output-dir dump/offpolicy
uv run accelerate launch train.py @configs/onpolicy.toml --output-dir dump/onpolicy

# Evaluation
uv run python eval.py --lora-paths dump/offpolicy --model-name Qwen/Qwen3-4B-Instruct-2507

# GPTQ/AWQ comparison
uv run python gptq_awq.py

# Hyperparameter sweep (K8s)
bash k8s/sweep.sh sweeps/lr_sweep_offpolicy.yaml 4
```

## Architecture

### Configuration System (`config.py`)
- Pydantic v2 with `pydantic_config` for TOML/CLI parsing
- **SharedConfig**: Model loading, quantization backends (torchao INT4/NVFP4, bitsandbytes FP4/NF4)
- **TrainConfig**: GKD params (λ, β), LoRA settings, batch sizes
- **EvalConfig**: lm-eval tasks + perplexity evaluation
- Config files use `@configs/file.toml` syntax on CLI

### Training Pipeline (`train.py`)
- Loads teacher (FP16) and student (INT4 QAT + LoRA)
- Uses `GKDTrainer` from TRL for on-policy distillation
- Dataset: `allenai/tulu-3-sft-mixture` with quality filtering
- Periodic perplexity evaluation during training

### Key Hyperparameters
- **λ (lambda)**: 0=off-policy (dataset only), 1=on-policy (student-generated)
- **β (beta)**: 0=forward KL, 1=reverse KL
- Optimal: β=0 for off-policy, β=1 for on-policy

### Evaluation (`eval.py`)
- Benchmarks: HellaSwag, ARC-Easy/Challenge, WinoGrande, MMLU
- Perplexity: Sliding window on wikitext/c4
- Merges LoRA adapters before final quantization

### Custom Trainer (`trainer.py`)
- Extends TRL's `GKDTrainer` to add `min_new_tokens` support
