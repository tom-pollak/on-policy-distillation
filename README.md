# On-Policy Distillation

## Setup

Install dependencies and lock the environment with `uv`:

```bash
uv sync
```

torch transformers datasets accelerate bitsandbytes peft scipy pydantic pydantic_config

## Training

```bash
k8s/train.sh lmbda_1_int4 --lmbda 1 --quant-type int4 --max-new-tokens 128

k8s/train.sh lmbda_0_int4 --lmbda 0 --quant-type int4

k8s/train.sh lmbda_05_int4 --lmbda 0.5 --quant-type int4 --max-new-tokens 128

k8s/train.sh lmbda_1_bnb_fp4 --lmbda 1 --quant-type bnb_fp4 --max-new-tokens 128

k8s/train.sh lmbda_0_bnb_fp4 --lmbda 0 --quant-type bnb_fp4

```

## SWEEPS

### λ=0 Runs (Fast HP Discovery)

```bash
# LR sweep (4 runs)
k8s/train.sh lmbda0_lr5e6 --lmbda 0 --learning-rate 5e-6 --quant-type int4 --tags sweep --tags lr
k8s/train.sh lmbda0_lr1e5 --lmbda 0 --learning-rate 1e-5 --quant-type int4 --tags sweep --tags lr
k8s/train.sh lmbda0_lr2e5 --lmbda 0 --learning-rate 2e-5 --quant-type int4 --tags sweep --tags lr
k8s/train.sh lmbda0_lr5e5 --lmbda 0 --learning-rate 5e-5 --quant-type int4 --tags sweep --tags lr
k8s/train.sh lmbda0_lr1e4 --lmbda 0 --learning-rate 1e-4 --quant-type int4 --tags sweep --tags lr

# Beta sweep (3 runs)
k8s/train.sh lmbda0_beta1 --lmbda 0 --beta 1 --quant-type int4 --tags sweep --tags beta
k8s/train.sh lmbda0_beta0 --lmbda 0 --beta 0 --quant-type int4 --tags sweep --tags beta
k8s/train.sh lmbda0_beta05 --lmbda 0 --beta 0.5 --quant-type int4 --tags sweep --tags beta --per-device-train-batch-size 2 --gradient-accumulation-steps 4

# Batch size (2 runs)
k8s/train.sh lmbda0_bs16 --lmbda 0 --gradient-accumulation-steps 4 --quant-type int4 --tags sweep --tags batch
k8s/train.sh lmbda0_bs32 --lmbda 0 --gradient-accumulation-steps 8 --quant-type int4 --tags sweep --tags batch

# Weight decay (1 run)
k8s/train.sh lmbda0_wd01 --lmbda 0 --weight-decay 0.01 --quant-type int4 --tags sweep --tags wd

# LR × Batch cross - higher batch often needs higher LR (3 runs)
k8s/train.sh lmbda0_bs16_lr2e5 --lmbda 0 --gradient-accumulation-steps 4 --learning-rate 2e-5 --quant-type int4 --tags sweep --tags lr-batch
k8s/train.sh lmbda0_bs16_lr5e5 --lmbda 0 --gradient-accumulation-steps 4 --learning-rate 5e-5 --quant-type int4 --tags sweep --tags lr-batch
k8s/train.sh lmbda0_bs32_lr5e5 --lmbda 0 --gradient-accumulation-steps 8 --learning-rate 5e-5 --quant-type int4 --tags sweep --tags lr-batch

# Warmup ratio (2 runs)
k8s/train.sh lmbda0_warmup05 --lmbda 0 --warmup-ratio 0.05 --quant-type int4 --tags sweep --tags warmup
k8s/train.sh lmbda0_warmup10 --lmbda 0 --warmup-ratio 0.1 --quant-type int4 --tags sweep --tags warmup

# Longer
k8s/train.sh lmbda0_lr5e5_steps20k --lmbda 0 --learning-rate 5e-5 --quant-type int4 --tags sweep --tags lr --max-steps 20000
```

### λ=1 Runs (Validate on On-Policy)

```bash
# LR sweep
k8s/train.sh lmbda1_lr5e5 --lmbda 1 --learning-rate 5e-5 --quant-type int4 --tags sweep --tags lr
k8s/train.sh lmbda1_lr1e4 --lmbda 1 --learning-rate 1e-4 --quant-type int4 --tags sweep --tags lr

# Rollout length sweep (2 runs)
k8s/train.sh lmbda1_tok256 --lmbda 1 --quant-type int4 --max-new-tokens 256 --tags sweep --tags rollout --per-device-train_batch-size 2 --gradient-accumulation-steps 4
k8s/train.sh lmbda1_tok512 --lmbda 1 --quant-type int4 --max-new-tokens 512 --tags sweep --tags rollout --per-device-train_batch-size 1 --gradient-accumulation-steps 8

# Batch size sweep (2 runs)
k8s/train.sh lmbda1_bs16 --lmbda 1 --gradient-accumulation-steps 4 --quant-type int4 --max-new-tokens 256 --tags sweep --tags batch
k8s/train.sh lmbda1_bs32 --lmbda 1 --gradient-accumulation-steps 8 --quant-type int4 --max-new-tokens 256 --tags sweep --tags batch

# Beta sanity check (1 run)
k8s/train.sh lmbda1_beta05 --lmbda 1 --beta 0.5 --quant-type int4 --max-new-tokens 256 --tags sweep --tags beta  --per-device-train_batch-size 2 --gradient-accumulation-steps 4
```

```bash
k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_0_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --lora_paths dump/lmbda_0_bnb_fp4



k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_0_int4 --lora_paths dump/lmbda_0_bnb_fp4

# tomorrow
k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --no-eval-teacher
```

### 1. Off-policy KD baseline

```bash
uv run accelerate launch train.py --lmbda 0 --output_dir qwen_kd_baseline --quant-type int4
```

### 2. On-policy KD

```bash
uv run accelerate launch train.py --lmbda 1 --output_dir qwen_onpolicy_4b_int4 --quant-type int4
```

### 3. Perplexity comparison (teacher vs PTQ vs KD vs on-policy KD)

```bash
uv run python eval.py
```

## Results

```
model                     | hellaswa | arc_easy | arc_chal | winogran |     mmlu
teacher                   |   0.5262 |   0.8308 |   0.5572 |   0.6835 |   0.7067
ptq_bnb_nf4               |          |          |   0.5410 |          |
ptq_int4                                        |   0.5265
int4_lmbda_0                                    |   0.5375
int4_lmbda_1                                    |   0.5384
int4_lmbda_05                                   |   0.5427
bnb_fp4_lmbda_1                                 |   0.5392



lmbda_0_int4/1000         |   0.5159 |   0.8186 |   0.5307 |   0.6772 |   0.6805
```


