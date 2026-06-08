"""Memory-efficient model loading for Wan 1.3B.

Handles downloading and loading the model with special care for
memory-constrained environments (16GB RAM target).
"""

import gc
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# HuggingFace model identifiers for Wan variants
MODEL_REGISTRY: Dict[str, str] = {
    "wan-1.3b": "Wan-AI/Wan2.1-T2V-1.3B",
}


def get_model_id(model_name: str) -> str:
    """Resolve a model name to its HuggingFace model ID.

    Args:
        model_name: Short name (e.g., "wan-1.3b") or full HF ID.

    Returns:
        HuggingFace model identifier string.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    # Assume it's a direct HuggingFace ID
    if "/" in model_name:
        return model_name
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Known models: {list(MODEL_REGISTRY.keys())}. "
        f"Or provide a full HuggingFace model ID (e.g., 'org/model')."
    )


def download_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Download model weights from HuggingFace if not already cached.

    Args:
        model_name: Model name or HuggingFace ID.
        cache_dir: Optional directory to cache the download.

    Returns:
        Local path to the downloaded model files.
    """
    from huggingface_hub import snapshot_download

    model_id = get_model_id(model_name)
    logger.info("Downloading model '%s' from HuggingFace...", model_id)

    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
    )

    logger.info("Model downloaded to: %s", local_path)
    return local_path


def check_memory_budget(required_gb: float) -> bool:
    """Check if enough RAM is available for model loading.

    Args:
        required_gb: Estimated memory required in gigabytes.

    Returns:
        True if enough memory is available, False otherwise.
    """
    import psutil

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    logger.info(
        "Memory check: %.1f GB available, %.1f GB required",
        available_gb,
        required_gb,
    )
    return available_gb >= required_gb


def estimate_model_memory(model_name: str, dtype: str = "float16") -> float:
    """Estimate memory required to load the model.

    Args:
        model_name: Model name.
        dtype: Data type for loading ("float16", "bfloat16", "float32").

    Returns:
        Estimated memory in gigabytes.
    """
    # Approximate parameter counts (in billions)
    param_counts = {
        "wan-1.3b": 1.3,
    }

    bytes_per_param = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }

    model_key = model_name.lower()
    if model_key not in param_counts:
        logger.warning(
            "Unknown model size for '%s', defaulting to 1.3B params", model_name
        )
        params = 1.3
    else:
        params = param_counts[model_key]

    bpp = bytes_per_param.get(dtype, 2)
    # Model weights + ~30% overhead for buffers, optimizer states, etc.
    estimated_gb = (params * 1e9 * bpp) / (1024 ** 3) * 1.3
    return round(estimated_gb, 2)


def load_model(
    model_name: str,
    model_path: Optional[str] = None,
    dtype: str = "float16",
    low_memory: bool = True,
) -> Any:
    """Load a Wan model for profiling.

    Args:
        model_name: Model name (e.g., "wan-1.3b").
        model_path: Local path to weights. Downloads if None.
        dtype: Data type for loading.
        low_memory: If True, use memory-efficient loading strategies.

    Returns:
        The loaded PyTorch model in eval mode.

    Raises:
        MemoryError: If insufficient RAM is available.
        RuntimeError: If model loading fails.
    """
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Check memory budget
    estimated_mem = estimate_model_memory(model_name, dtype)
    logger.info("Estimated memory for %s (%s): %.2f GB", model_name, dtype, estimated_mem)

    if not check_memory_budget(estimated_mem):
        if not low_memory:
            raise MemoryError(
                f"Insufficient RAM. Need ~{estimated_mem:.1f} GB. "
                f"Try --low-memory mode or use float16."
            )
        logger.warning(
            "Tight memory. Proceeding with aggressive memory optimization."
        )

    # Resolve model path
    if model_path is None:
        model_path = download_model(model_name)

    logger.info("Loading model from: %s", model_path)

    # Force garbage collection before loading
    gc.collect()

    try:
        from transformers import AutoModel

        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if low_memory:
            load_kwargs["low_cpu_mem_usage"] = True
            logger.info("Using low_cpu_mem_usage for memory-efficient loading")

        model = AutoModel.from_pretrained(**load_kwargs)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded. Parameters: %s total (%.2f B), %s trainable",
            f"{total_params:,}",
            total_params / 1e9,
            f"{trainable_params:,}",
        )

        return model

    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        gc.collect()
        raise RuntimeError(f"Model loading failed: {e}") from e


def get_module_tree(model: Any) -> Dict[str, Dict[str, Any]]:
    """Extract the module hierarchy from a loaded model.

    Args:
        model: A PyTorch model.

    Returns:
        Dictionary mapping module names to their info (type, param count, etc.).
    """
    tree = {}
    for name, module in model.named_modules():
        if name == "":
            name = "(root)"
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        tree[name] = {
            "type": type(module).__name__,
            "param_count": param_count,
            "has_children": len(list(module.children())) > 0,
        }
    return tree
