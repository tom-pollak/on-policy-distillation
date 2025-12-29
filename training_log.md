# Training Log

## Training Commands

```bash
k8s/train.sh lmbda_1_int4 --lmbda 1 --quant-type int4 --max-new-tokens 128
k8s/train.sh lmbda_0_int4 --lmbda 0 --quant-type int4
k8s/train.sh lmbda_05_int4 --lmbda 0.5 --quant-type int4 --max-new-tokens 128
k8s/train.sh lmbda_1_bnb_fp4 --lmbda 1 --quant-type bnb_fp4 --max-new-tokens 128
k8s/train.sh lmbda_0_bnb_fp4 --lmbda 0 --quant-type bnb_fp4
```

## Sweeps

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

# Beta
k8s/train.sh lmbda1_beta05 --lmbda 1 --beta 0.5 --quant-type int4 --max-new-tokens 256 --tags sweep --tags beta  --per-device-train_batch-size 2 --gradient-accumulation-steps 4
```

### Extended

```bash
k8s/train.sh lmbda0_lr5e6_beta0 --lmbda 0 --learning-rate 5e-6 --beta 0.0 --quant-type int4 --tags sweep --tags combined

k8s/train.sh lmbda1_beta0 --lmbda 1 --beta 0.0 --quant-type int4 --max-new-tokens 256 --tags sweep --tags beta  --per-device-train_batch-size 2 --gradient-accumulation-steps 4

k8s/train.sh lmbda1_lr5e6 --lmbda 1 --learning-rate 5e-6 --quant-type int4 --tags sweep --tags lr --tags lmbda1
k8s/train.sh lmbda1_lr1e5 --lmbda 1 --learning-rate 1e-5 --quant-type int4  --tags sweep --tags lr --tags lmbda1
k8s/train.sh lmbda05_beta0 --lmbda 0.5 --beta 0.0 --quant-type int4  --tags sweep --tags beta --tags lmbda05

```

## Eval Commands

```bash
k8s/eval.sh eval --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_0_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --lora_paths dump/lmbda_0_bnb_fp4

k8s/eval.sh eval-base --quant-type int4 --lora-paths dump/lmbda_1_int4 --lora_paths dump/lmbda_05_int4 --lora_paths dump/lmbda_1_bnb_fp4 --no-eval-teacher --tags eval --tags qtypes

k8s/eval.sh eval-lmbda1-batch --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda1_bs16 --lora-paths dump/lmbda1_bs32 --tags lmbda1 --tags batch
k8s/eval.sh eval-lmbda1-rollout --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda1_tok256 --lora-paths dump/lmbda1_tok512 --tags rollout --tags lmbda1
k8s/eval.sh eval-lmbda1-lr --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda1_lr5e5 --lora-paths dump/lmbda1_lr1e4 --tags lr --tags lmbda1
k8s/eval.sh eval-lmbda1-beta --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda1_beta05 --tags beta --tags lmbda1
k8s/eval.sh eval-lmbda0-beta --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda0_beta1 --lora-paths dump/lmbda0_beta05 --lora-paths dump/lmbda0_beta0 --tags beta --tags lmbda0
k8s/eval.sh eval-lmbda0-20k --quant-type int4 --tags eval --no-eval-teacher --lora-paths dump/lmbda0_lr5e5_steps20k --tags 20k --tags lmbda0
k8s/eval.sh eval-lmbda0-lr --quant-type int4 --tags eval --lora-paths dump/lmbda0_lr5e6 --lora-paths dump/lmbda0_lr1e5 --lora-paths dump/lmbda0_lr2e5 --lora-paths dump/lmbda0_lr5e5 --lora-paths dump/lmbda0_lr1e4 --tags lr --no-eval-teacher
```

## Results

```
model                     | hellaswa | arc_easy | arc_chal | winogran |     mmlu
teacher                   |   0.5262 |   0.8308 |   0.5572 |   0.6835 |   0.7067

lmbda0_lr5e5_steps20k     |   0.5126 |   0.8199 |   0.5461 |   0.6638 |   0.6735

lmbda1_bs32               |   0.5119 |   0.8224 |   0.5401 |   0.6709 |   0.6839
lmbda1_bs16               |   0.5131 |   0.8173 |   0.5427 |   0.6732 |   0.6847

lmbda0_lr5e6              |   0.5149 |   0.8165 |   0.5375 |   0.6875 |   0.6822
lmbda0_lr1e5              |   0.5153 |   0.8178 |   0.5358 |   0.6780 |   0.6807
lmbda0_lr2e5              |   0.5177 |   0.8157 |   0.5486 |   0.6717 |   0.6797
lmbda0_lr5e5              |   0.5137 |   0.8148 |   0.5341 |   0.6725 |   0.6773
lmbda0_lr1e4              |   0.5127 |   0.8106 |   0.5213 |   0.6701 |   0.6742


lmbda0_beta1              |   0.5145 |   0.8203 |   0.5384 |   0.6835 |   0.6788
lmbda0_beta05             |   0.5172 |   0.8220 |   0.5367 |   0.6788 |   0.6811
lmbda0_beta0              |   0.5213 |   0.8232 |   0.5401 |   0.6780 |   0.6806

lmbda1_beta05             |   0.5150 |   0.8194 |   0.5384 |   0.6748 |   0.6864

lmbda1_lr5e5              |   0.5118 |   0.8123 |   0.5256 |   0.6646 |   0.6776
lmbda1_lr1e4              |   0.5105 |   0.8157 |   0.5367 |   0.6701 |   0.6769

lmbda1_tok256             |   0.5127 |   0.8178 |   0.5299 |   0.6811 |   0.6824
lmbda1_tok512             |   0.5141 |   0.8148 |   0.5358 |   0.6756 |   0.6845

lmbda_1_int4              |   0.5121 |   0.8186 |   0.5333 |   0.6764 |   0.6855
lmbda_05_int4             |   0.5160 |   0.8194 |   0.5333 |   0.6803 |   0.6857
lmbda_1_int4              |   0.5121 |   0.8186 |   0.5333 |   0.6764 |   0.6855
```
