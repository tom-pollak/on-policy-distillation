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

### On-Policy (λ=1) vs Off-Policy (λ=0)

![Benchmark Accuracy by Quantization Method](figures/main_comparison.png)

Distillation recovers most of the accuracy lost by naive PTQ INT4, but on-policy (λ=1) and off-policy (λ=0) perform equivalently. The ~0.5% differences between them are within noise.

### Learning Rate Sweep (λ=0, off-policy)

![LR Sweep](figures/lr_sweep.png)

Higher learning rates achieve lower training loss but worse eval accuracy. Lower LRs (5e-6, 1e-5) generalize better despite not fitting the training data as tightly.

### Extended Training (λ=0, 1k vs 20k steps)

| Steps | Train Loss | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU      |
| ----- | ---------- | --------- | --------- | ------------- | ---------- | --------- |
| 1k    | 0.118      | 0.514     | 0.815     | 0.534         | **0.673**  | **0.677** |
| 20k   | 0.068      | 0.513     | **0.820** | **0.546**     | 0.664      | 0.674     |

20x compute reduces training loss by 42% but eval accuracy is mixed. Suggests we've hit the accuracy ceiling for INT4 quantization.

### Beta sweep (λ=0)

- `beta=0`: forward KL
- `beta=1`: reverse KL
- `beta` between 0-1 interpolates between the two.

For standard distillation, forward KL is often used, however [on policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/#loss-function-reverse-kl) recommends reverse KL. I find similar results for λ=0 (off-policy distillation) forward KL performs better.

| Beta | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU  |
| ---- | --------- | --------- | ------------- | ---------- | ----- |
| 0.0  | **0.521** | **0.823** | **0.540**     | 0.678      | 0.681 |
| 0.5  | 0.517     | 0.822     | 0.537         | 0.679      | 0.681 |
| 1.0  | 0.515     | 0.820     | 0.538         | **0.684**  | 0.679 |

### On-Policy Hyperparameter Sweeps (λ=1)

#### Beta

| Beta | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande | MMLU  |
| ---- | --------- | -------- | ------------- | ---------- | ----- |
| 0.0  | \*\*      | \*\*     | \*\*          | \*\*       | \*\*  |
| 0.5  | **0.515** | 0.819    | **0.538**     | 0.675      | 0.686 |
| 1.0  | 0.512     | 0.819    | 0.533         | **0.676**  | 0.686 |

#### Increasing compute

From limited sweeps, it seems that batch size and rollout length are not sensitive params.

> Note I do the same number of steps, so doubling batch size / rollout lenght doubles the compute budget! Suggesting we've saturated the quantization accuracy of the model.

##### Batch size

| Batch Size | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU      |
| ---------- | --------- | --------- | ------------- | ---------- | --------- |
| 16         | 0.513     | 0.817     | **0.543**     | 0.673      | **0.685** |
| 32         | 0.512     | **0.822** | 0.540         | 0.671      | 0.684     |

##### Rollout length

| Tokens | HellaSwag | ARC-Easy  | ARC-Challenge | WinoGrande | MMLU      |
| ------ | --------- | --------- | ------------- | ---------- | --------- |
| 128    | 0.512     | **0.819** | 0.533         | 0.676      | **0.686** |
| 256    | 0.513     | 0.818     | 0.530         | **0.681**  | 0.682     |
| 512    | **0.514** | 0.815     | **0.536**     | 0.676      | 0.685     |
