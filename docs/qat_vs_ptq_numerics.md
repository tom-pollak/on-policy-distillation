# QAT vs PTQ Quantization Numerics

This document explains the differences between QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization) in torchao.

## QAT: Fake Quantization (Training)

QAT uses `Int4WeightFakeQuantizeConfig` which simulates int4 quantization during training while keeping weights in FP16/BF16. The forward pass applies fake quantization noise so the model learns to be robust to quantization.

```python
from torchao.quantization.qat import QATConfig
from torchao.quantization import Int4WeightOnlyConfig, quantize_

# Training setup
quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="prepare"))
# Model now has FakeQuantizedLinear layers
```

### QAT Numerics (FBGEMM-style)

From `Int4WeightFakeQuantizeConfig._bf16_activations_forward`:

```python
# Asymmetric, unsigned int4 (0-15)
qmin, qmax = 0, 15
group_size = 128  # Hardcoded

# Per-group scale and zero_point
max_val = torch.amax(w_grouped, dim=-1)
min_val = torch.amin(w_grouped, dim=-1)
scale = (max_val - min_val) / qmax
zero_point = min_val + scale * 8  # Shift point

# Fake quantize
fq = round((w - min_val) / scale).clamp(0, 15)
fq = (fq - 8) * scale + zero_point  # Shift to symmetric around zero_point
```

Key characteristics:

- **Unsigned int4**: Values 0-15, then shifted by 8
- **Scale formula**: `(max - min) / 15`
- **Zero point**: `min + scale * 8`
- **Group size**: Hardcoded to 128
- **Purpose**: Match FBGEMM kernel numerics for deployment

## PTQ: Actual Int4 Storage (Inference)

PTQ uses `Int4WeightOnlyConfig` which actually quantizes weights to int4 storage using `Int4Tensor`.

```python
from torchao.quantization import Int4WeightOnlyConfig, quantize_

# Inference setup
quantize_(model, Int4WeightOnlyConfig())
# Model weights are now Int4Tensor with qdata, scale, zero_point
```

### PTQ Numerics (torchao native)

The `Int4Tensor` format:

- `qdata`: Packed int8 tensor (2 int4 values per byte, low/high nibbles)
- `scale`: Per-group scales, shape `(n_groups, out_features)`
- `zero_point`: Per-group zero points, shape `(n_groups, out_features)`

```python
# Dequantization formula
# Signed int4: -8 to 7
low = (qdata & 0x0F).to(torch.int8)
high = ((qdata >> 4) & 0x0F).to(torch.int8)
unpacked = torch.stack([low, high], dim=-1)

# Convert unsigned (0-15) to signed (-8 to 7)
signed = torch.where(unpacked > 7, unpacked - 16, unpacked)

# Dequantize
dequant = (signed - zero_point) * scale
```

Key characteristics:

- **Signed int4**: Values -8 to 7
- **Different scale/zp computation**: Not the same as FBGEMM
- **Configurable group size**: Default 128, but can vary

## Incompatiblities

| Aspect        | QAT (FBGEMM)             | PTQ (torchao native)  |
| ------------- | ------------------------ | --------------------- |
| Int4 range    | 0-15 (unsigned, shifted) | -8 to 7 (signed)      |
| Scale formula | `(max - min) / 15`       | Different computation |
| Zero point    | `min + scale * 8`        | Different computation |
| Storage       | FP16 with noise          | Actual packed int4    |

## LoRA Merge Semantics

When training with QAT + LoRA:

```
y = Q(W) @ x + ΔW @ x     where Q = fake_quant, ΔW = B @ A
```

When we call `merge_and_unload()`:

```
y = Q(W + ΔW) @ x
```

These are not equivalent: `Q(W) + ΔW ≠ Q(W + ΔW)`. The LoRA learned to correct `Q(W)`, not to be merged before quantization.

With `requantize_after_lora=False`, `Q()` is skipped entirely: `y = (W + ΔW) @ x`. Test with and without lora quantization error.
