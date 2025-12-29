# On-Policy Quantization

Knowledge distillation from FP32 teacher to INT4 student using on-policy (student-generated) data rather than static datasets.

Standard KD trains on fixed datasets, causing distribution mismatch—the student never learns from its own mistakes. On-policy distillation fixes this by having the student generate sequences and learning from teacher feedback on those generations. Inspired by [GKD](https://arxiv.org/abs/2306.13649) and [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).

## TL;DR: Null result

> On-policy distillation doesn't provide significant benefits over off-policy distillation. Performance differences are noise (~0.5%)

- Distribution mismatch is probably not a big issue when student teacher share same weights in a different precision.
- On-policy rollouts are much more compute intensive than simple off-policy SFT, which is another disadvantage

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

## Results

`λ` controls data source interpolation:

- `λ=0`: Off-policy (dataset sequences only)
- `λ=1`: Fully on-policy (student-generated sequences only)

### Baseline

| Model          | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande | MMLU      |
| -------------- | --------- | -------- | ------------- | ---------- | --------- |
| Teacher (FP32) | 0.526     | 0.831    | 0.557         | 0.684      | **0.707** |

### On-Policy (λ=1) vs Off-Policy (λ=0)

Best runs from each approach (default hyperparameters, 10k steps):

| Model            | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande | MMLU  |
| ---------------- | --------- | -------- | ------------- | ---------- | ----- |
| Off-policy (λ=0) | 0.515     | 0.819    | 0.538         | 0.678      | 0.681 |
| On-policy (λ=1)  | 0.512     | 0.819    | 0.533         | 0.676      | 0.686 |
| Mixed (λ=0.5)    | 0.516     | 0.819    | 0.533         | 0.680      | 0.686 |

### Learning Rate Sweep (λ=0, off-policy)

| Learning Rate | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande | MMLU      |
| ------------- | --------- | -------- | ------------- | ---------- | --------- |
| 5e-6          | 0.515     | 0.817    | 0.538         | **0.688**  | **0.682** |
| 1e-5          | 0.515     | 0.818    | 0.536         | 0.678      | 0.681     |
| 2e-5          | **0.518** | 0.816    | **0.549**     | 0.672      | 0.680     |
| 5e-5          | 0.514     | 0.815    | 0.534         | 0.673      | 0.677     |
| 1e-4          | 0.513     | 0.811    | 0.521         | 0.670      | 0.674     |

### Beta (KL coefficient) Sweep (λ=0)

| Beta        | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU  |
| ----------- | --------- | --------- | ------------- | ---------- | ----- |
| 0.0 (no KL) | **0.521** | **0.823** | **0.540**     | 0.678      | 0.681 |
| 0.5         | 0.517     | 0.822     | 0.537         | 0.679      | 0.681 |
| 1.0         | 0.515     | 0.820     | 0.538         | **0.684**  | 0.679 |

### On-Policy Hyperparameter Sweeps (λ=1)

**Batch size:**

| Batch Size | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU      |
| ---------- | --------- | --------- | ------------- | ---------- | --------- |
| 16         | 0.513     | 0.817     | **0.543**     | 0.673      | **0.685** |
| 32         | 0.512     | **0.822** | 0.540         | 0.671      | 0.684     |

**Rollout length (max new tokens):**

| Tokens | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU      |
| ------ | --------- | --------- | ------------- | ---------- | --------- |
| 256    | 0.513     | **0.818** | 0.530         | **0.681**  | 0.682     |
| 512    | **0.514** | 0.815     | **0.536**     | 0.676      | **0.685** |
