# On-Policy Distillation

Knowledge distillation from FP32 teacher to MXFP4 student using on-policy (student-generated) data rather than static datasets.

## Setup

```bash
uv sync
```

## Training

```bash
# Off-policy KD baseline (λ=0)
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant-type int4

# On-policy KD (λ=1)
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant-type int4
```

## Evaluation

```bash
uv run python eval.py
```
