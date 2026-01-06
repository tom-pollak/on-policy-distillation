"""PTQ dequantization utilities for torchao Int4Tensor.

WARNING: These functions use PTQ numerics which are DIFFERENT from QAT training.
Do not use for eval - use QAT prepare/convert instead.
See docs/qat_vs_ptq_numerics.md for details.
"""

import torch
from torch import nn


def dequantize_int4(weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a PTQ Int4Tensor back to float.

    WARNING: This uses PTQ numerics which are DIFFERENT from QAT training.
    Do not use for eval - use QAT prepare/convert instead.

    Args:
        weight: A torchao Int4Tensor with qdata, scale, and zero_point attributes.

    Returns:
        Dequantized tensor in the original dtype (typically bf16).
    """
    if not hasattr(weight, "qdata"):
        raise ValueError("Weight does not have qdata attribute - not an Int4Tensor?")

    qdata = weight.qdata
    scale = weight.scale
    zero_point = weight.zero_point

    # Unpack int8 -> two int4 values (low and high nibbles)
    low = (qdata & 0x0F).to(torch.int8)
    high = ((qdata >> 4) & 0x0F).to(torch.int8)
    unpacked = torch.stack([low, high], dim=-1).reshape(
        qdata.shape[0], qdata.shape[1] * 2
    )

    # Convert unsigned int4 (0-15) to signed (-8 to 7)
    unpacked = torch.where(unpacked > 7, unpacked - 16, unpacked)

    # Dequantize with group-wise scale/zero_point
    # scale/zp are (n_groups, out_features)
    n_groups = scale.shape[0]
    group_size = unpacked.shape[1] // n_groups

    # Reshape for group-wise ops: (out, in) -> (out, n_groups, group_size)
    unpacked_grouped = unpacked.reshape(unpacked.shape[0], n_groups, group_size)
    scale_bc = scale.T.unsqueeze(-1)  # (out, n_groups, 1)
    zp_bc = zero_point.T.unsqueeze(-1)

    dequant = ((unpacked_grouped.to(scale.dtype) - zp_bc) * scale_bc).reshape(
        unpacked.shape[0], unpacked.shape[1]
    )

    return dequant


def dequantize_model_(model: nn.Module) -> None:
    """Dequantize all PTQ Int4 quantized Linear layers in a model (in-place).

    WARNING: This uses PTQ numerics which are DIFFERENT from QAT training.
    Do not use for eval - use QAT prepare/convert instead.

    Args:
        model: Model with quantized weights to dequantize.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                if not hasattr(w, "qdata"):
                    continue
                dequant = dequantize_int4(w)
                module.weight = nn.Parameter(dequant, requires_grad=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    # Test with a realistic layer size
    in_features, out_features = 4096, 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(
        f"Testing PTQ dequantization on {out_features}x{in_features} layer ({device})"
    )

    # Create layer and store original weights
    layer = nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)
    layer.weight.data = torch.randn_like(layer.weight.data)
    original = layer.weight.data.clone()

    # Quantize with PTQ
    quantize_(layer, Int4WeightOnlyConfig())
    print(f"Quantized weight type: {type(layer.weight.data)}")

    # Dequantize
    dequant = dequantize_int4(layer.weight.data)

    # Compute diff
    diff = (original - dequant).float()
    abs_diff = diff.abs()

    print("\nPTQ Quantization Error Statistics:")
    print(f"  Max absolute error:  {abs_diff.max().item():.6f}")
    print(f"  Mean absolute error: {abs_diff.mean().item():.6f}")
    print(f"  Std absolute error:  {abs_diff.std().item():.6f}")
    print(
        f"  Relative error:      {(abs_diff / original.abs().float()).mean().item():.4%}"
    )

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of errors
    ax = axes[0]
    diff_flat = diff.cpu().flatten().numpy()
    ax.hist(diff_flat, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Quantization Error")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of PTQ Quantization Errors")
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # Heatmap of absolute errors (subsample for visibility)
    ax = axes[1]
    subsample = 64
    abs_diff_sub = abs_diff[:: out_features // subsample, :: in_features // subsample]
    im = ax.imshow(abs_diff_sub.cpu().numpy(), aspect="auto", cmap="hot")
    ax.set_xlabel("In Features (subsampled)")
    ax.set_ylabel("Out Features (subsampled)")
    ax.set_title("Absolute Error Heatmap")
    plt.colorbar(im, ax=ax)

    # Original vs dequantized scatter (subsample)
    ax = axes[2]
    n_points = 10000
    idx = torch.randperm(original.numel())[:n_points]
    orig_flat = original.float().cpu().flatten()[idx].numpy()
    deq_flat = dequant.float().cpu().flatten()[idx].numpy()
    ax.scatter(orig_flat, deq_flat, alpha=0.1, s=1)
    ax.plot(
        [orig_flat.min(), orig_flat.max()],
        [orig_flat.min(), orig_flat.max()],
        "r--",
        alpha=0.5,
    )
    ax.set_xlabel("Original Weight")
    ax.set_ylabel("Dequantized Weight")
    ax.set_title("Original vs Dequantized")

    plt.tight_layout()
    plt.savefig("figures/quantization_error.png", dpi=150)
    print("\nVisualization saved to figures/quantization_error.png")
    plt.show()
