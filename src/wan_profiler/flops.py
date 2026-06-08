"""FLOPs counting utilities for profiling.

Estimates floating-point operations per module based on
layer type and tensor dimensions. These are analytical estimates,
not measured counts.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def count_linear_flops(
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    seq_len: int = 1,
) -> int:
    """Count FLOPs for a linear (fully connected) layer.

    A linear layer computes y = xW + b.
    FLOPs = 2 * in_features * out_features * batch_size * seq_len
    (multiply-accumulate = 2 ops per element)

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        batch_size: Batch size.
        seq_len: Sequence length (for transformer contexts).

    Returns:
        Estimated FLOPs.
    """
    return 2 * in_features * out_features * batch_size * seq_len


def count_attention_flops(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int = 1,
) -> int:
    """Count FLOPs for multi-head self-attention.

    Includes QKV projection, attention scores, softmax (approximate),
    attention-weighted values, and output projection.

    Args:
        seq_len: Sequence length.
        hidden_dim: Hidden dimension (d_model).
        num_heads: Number of attention heads.
        batch_size: Batch size.

    Returns:
        Estimated FLOPs.
    """
    head_dim = hidden_dim // num_heads

    # QKV projections: 3 linear layers
    qkv_flops = 3 * count_linear_flops(hidden_dim, hidden_dim, batch_size, seq_len)

    # Attention scores: Q @ K^T -> (batch, heads, seq, seq)
    score_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2

    # Softmax (approximate: ~5 ops per element)
    softmax_flops = batch_size * num_heads * seq_len * seq_len * 5

    # Attention @ V -> (batch, heads, seq, head_dim)
    av_flops = batch_size * num_heads * seq_len * head_dim * seq_len * 2

    # Output projection
    out_flops = count_linear_flops(hidden_dim, hidden_dim, batch_size, seq_len)

    total = qkv_flops + score_flops + softmax_flops + av_flops + out_flops
    return total


def count_layernorm_flops(
    hidden_dim: int,
    batch_size: int = 1,
    seq_len: int = 1,
) -> int:
    """Count FLOPs for layer normalization.

    Includes mean, variance, normalization, and affine transform.

    Args:
        hidden_dim: Dimension being normalized.
        batch_size: Batch size.
        seq_len: Sequence length.

    Returns:
        Estimated FLOPs.
    """
    num_elements = batch_size * seq_len
    # Mean: sum + divide
    mean_flops = hidden_dim * num_elements
    # Variance: subtract mean, square, sum, divide
    var_flops = hidden_dim * num_elements * 3
    # Normalize: subtract, divide by sqrt(var + eps)
    norm_flops = hidden_dim * num_elements * 2
    # Affine: multiply by gamma, add beta
    affine_flops = hidden_dim * num_elements * 2

    return mean_flops + var_flops + norm_flops + affine_flops


def count_activation_flops(
    num_elements: int,
    activation: str = "gelu",
) -> int:
    """Count FLOPs for an activation function.

    Args:
        num_elements: Number of elements to activate.
        activation: Activation function name.

    Returns:
        Estimated FLOPs.
    """
    ops_per_element = {
        "relu": 1,
        "gelu": 8,  # Approximate (involves erf or tanh)
        "silu": 4,  # x * sigmoid(x)
        "swish": 4,
        "sigmoid": 4,
        "tanh": 5,
    }
    return num_elements * ops_per_element.get(activation.lower(), 4)


def estimate_module_flops(
    module: Any,
    input_shape: Tuple[int, ...],
) -> int:
    """Estimate FLOPs for a single module based on its type and input shape.

    Args:
        module: PyTorch module.
        input_shape: Shape of the input tensor.

    Returns:
        Estimated FLOPs. Returns 0 if the module type is not recognized.
    """
    import torch.nn as nn

    module_type = type(module).__name__

    try:
        if isinstance(module, nn.Linear):
            batch_seq = 1
            for dim in input_shape[:-1]:
                batch_seq *= dim
            return count_linear_flops(
                module.in_features,
                module.out_features,
                batch_size=1,
                seq_len=batch_seq,
            )

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            num_elements = 1
            for dim in input_shape:
                num_elements *= dim
            normalized_shape = input_shape[-1] if isinstance(module, nn.LayerNorm) else input_shape[1]
            return count_layernorm_flops(normalized_shape, batch_size=num_elements // normalized_shape)

        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            num_elements = 1
            for dim in input_shape:
                num_elements *= dim
            activation_name = module_type.lower()
            return count_activation_flops(num_elements, activation_name)

        elif isinstance(module, nn.Conv2d):
            batch = input_shape[0] if len(input_shape) >= 4 else 1
            out_h = input_shape[-2] // (module.stride[0] if hasattr(module.stride, '__getitem__') else module.stride)
            out_w = input_shape[-1] // (module.stride[1] if hasattr(module.stride, '__getitem__') else module.stride)
            flops_per_output = module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 2
            return batch * module.out_channels * out_h * out_w * flops_per_output

    except Exception as e:
        logger.debug("Could not estimate FLOPs for %s: %s", module_type, str(e))

    return 0


def categorize_module(module_name: str, module_type: str) -> str:
    """Categorize a module into a high-level operation type.

    Args:
        module_name: Fully qualified module name.
        module_type: Class name of the module.

    Returns:
        Category string: "attention", "ffn", "normalization",
        "activation", "embedding", "temporal", or "other".
    """
    name_lower = module_name.lower()
    type_lower = module_type.lower()

    if any(kw in name_lower for kw in ["attn", "attention", "self_attn", "cross_attn"]):
        return "attention"

    if any(kw in name_lower for kw in ["ffn", "feed_forward", "mlp", "fc"]):
        return "ffn"

    if any(kw in type_lower for kw in ["layernorm", "groupnorm", "batchnorm", "rmsnorm"]):
        return "normalization"

    if any(kw in type_lower for kw in ["relu", "gelu", "silu", "sigmoid", "tanh"]):
        return "activation"

    if any(kw in name_lower for kw in ["embed", "patch_embed", "pos_embed"]):
        return "embedding"

    if any(kw in name_lower for kw in ["temporal", "time", "t_block"]):
        return "temporal"

    return "other"
