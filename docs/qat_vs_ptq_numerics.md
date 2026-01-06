# QAT vs PTQ Quantization Numerics

This document explains the differences between QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization) in torchao, and why LoRA adapters trained with QAT cannot be directly applied to PTQ-quantized weights.

## Summary

**The LoRA adapters trained with QAT fake quantization are NOT compatible with PTQ-quantized weights.** They use fundamentally different quantization numerics.

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

## Why They're Incompatible

| Aspect        | QAT (FBGEMM)             | PTQ (torchao native)  |
| ------------- | ------------------------ | --------------------- |
| Int4 range    | 0-15 (unsigned, shifted) | -8 to 7 (signed)      |
| Scale formula | `(max - min) / 15`       | Different computation |
| Zero point    | `min + scale * 8`        | Different computation |
| Storage       | FP16 with noise          | Actual packed int4    |

The quantization error introduced by QAT is **different** from PTQ:

- QAT error: ~0.0005 mean absolute error
- PTQ error: ~0.001 mean absolute error
- QAT vs PTQ difference: ~0.001 mean absolute error

While these numbers seem small, the LoRA adapters learned to correct the **specific** QAT noise pattern. When applied to PTQ-quantized weights (with different noise), the corrections don't align.

## Experimental Evidence

When we tried to match training by:

1. Load model
2. PTQ quantize (introduces PTQ quantization error)
3. Dequantize back to FP16 (preserving PTQ error)
4. Apply LoRA (trained with QAT error)
5. Re-quantize

**Results were broken**: Random accuracy (~0.5 on WinoGrande), perplexity ~71 million.

## Correct Evaluation Approaches

### Option 1: Standard PTQ (Simple, but loses LoRA benefit)

Apply LoRA to FP16 weights, merge, then PTQ quantize:

```python
model = load_model()  # FP16
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
quantize_(model, Int4WeightOnlyConfig())  # PTQ at the end
```

**Problem**: The LoRA was trained to correct the *specific* QAT noise pattern. Applying it to clean FP16 weights, then adding *different* PTQ noise, means the LoRA corrections don't align with the quantization error. This may still improve results, but doesn't match training.

### Option 2: QAT Fake Quantization (Matches Training)

Use QAT fake quantization for evaluation to match training exactly:

```python
model = load_model()
quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="prepare"))
# Model now has FakeQuantizedLinear with QAT noise
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
# Keep as fake-quantized (BF16 storage with QAT noise in forward pass)
```

**Pros**: Matches training numerics exactly - LoRA corrections align with quantization error.
**Cons**: Weights are still BF16 (not actual int4), so no memory savings at inference.

### Option 3: QAT Prepare → Convert (Recommended)

Use QAT's two-step process: prepare (fake quant) → convert (real int4):

```python
model = load_model()
# Step 1: Prepare - adds fake quantization (same as training)
quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="prepare"))
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
# Step 2: Convert - converts FakeQuantizedLinear to real int4
quantize_(model, QATConfig(Int4WeightOnlyConfig(), step="convert"))
```

**Pros**:
- Matches training numerics during LoRA application
- Final model uses real int4 storage (memory savings)
- QAT convert uses FBGEMM-compatible int4 format

This is the correct approach because:
1. The LoRA sees the same QAT noise it was trained with
2. The final conversion preserves the numerics (FBGEMM-style quantization throughout)

## Files

- `dequantize.py`: Manual dequantization for **PTQ** Int4Tensor (uses PTQ numerics, NOT QAT)
  - Useful for debugging/analysis of PTQ quantization
  - **Do not use for eval** - PTQ numerics don't match training
- `compare_qat_ptq.py`: Script comparing QAT vs PTQ numerics
- `eval.py`: Evaluation script - should use Option 3 (QAT prepare → convert)

## Implementation Notes

### Why we don't need manual dequantization for eval

The broken approach was:
```
PTQ quantize → manual dequantize → apply LoRA
```
This fails because PTQ uses different numerics than QAT training.

The correct approach with QAT:
```
QAT prepare → apply LoRA → (optionally) QAT convert
```

With QAT `step="prepare"`:
- Linear layers become `FakeQuantizedLinear`
- Weights stay in BF16 (not actually quantized)
- Forward pass applies fake quant noise matching training
- LoRA can be applied directly to BF16 weights
- No manual dequantization needed!

The QAT convert step (`step="convert"`) handles the transition to real int4 using matched FBGEMM numerics.

### What about dequantize.py?

`dequantize.py` was created for the PTQ flow and uses PTQ numerics. It's still useful for:
- Debugging/analyzing PTQ quantization error
- Understanding the Int4Tensor format

But it's **not needed** for the eval pipeline when using QAT (Option 2/3).

## References

- torchao QAT: `torchao/quantization/qat/_linear.py`
- torchao Int4: `torchao/dtypes/nf4tensor.py` and related
- FBGEMM int4 kernels: Used by QAT for deployment compatibility
