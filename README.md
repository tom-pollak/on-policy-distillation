# On-Policy Distillation

Knowledge distillation from FP32 teacher to MXFP4 student using on-policy (student-generated) data rather than static datasets.

Standard KD trains on fixed datasets, causing distribution mismatch—the student never learns from its own mistakes. On-policy distillation fixes this by having the student generate sequences and learning from teacher feedback on those generations. Based on [GKD](https://arxiv.org/abs/2306.13649), we use `GKDTrainer` with `λ` controlling the interpolation: `λ=0` is off-policy (dataset only), `λ=1` is fully on-policy (student generations only).

## Setup

```bash
uv sync
```

## Training

```bash
# Off-policy KD baseline (λ=0)
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant-type int4 --do-eval

# On-policy KD (λ=1)
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant-type int4 --do-eval
```
