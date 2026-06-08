"""Memory tracking utilities for profiling.

Provides tools to measure memory usage at various granularities:
per-module, per-operation, and peak usage during forward passes.
"""

import gc
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    """Get current process memory usage in megabytes.

    Returns:
        Current RSS (Resident Set Size) in MB.
    """
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_system_memory_info() -> Dict[str, float]:
    """Get system memory information.

    Returns:
        Dictionary with total, available, used, and percent memory (in GB/%).
    """
    import psutil

    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "used_gb": round(mem.used / (1024 ** 3), 2),
        "percent": mem.percent,
    }


def get_torch_memory_stats() -> Dict[str, float]:
    """Get PyTorch memory allocation statistics.

    Works for both CPU and CUDA. On CPU, falls back to process-level
    memory tracking since PyTorch doesn't track CPU allocations as precisely.

    Returns:
        Dictionary with current and peak memory in MB.
    """
    import torch

    stats = {
        "current_mb": get_process_memory_mb(),
        "peak_mb": 0.0,
    }

    if torch.cuda.is_available():
        stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
        stats["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return stats


class MemoryTracker:
    """Track memory usage per module during forward passes.

    Uses forward hooks to snapshot memory before and after each module.
    On CPU, this uses process-level RSS tracking which includes all allocations.

    Args:
        track_peak: Whether to track peak memory (slightly more overhead).
    """

    def __init__(self, track_peak: bool = True) -> None:
        self._track_peak = track_peak
        self._hooks: list = []
        self._memory_before: Dict[str, float] = {}
        self._memory_delta: Dict[str, list] = {}
        self._peak_memory_mb: float = 0.0

    def _make_pre_hook(self, name: str) -> Any:
        """Create a pre-hook that snapshots memory before the module runs.

        Args:
            name: Module name.

        Returns:
            Hook function.
        """
        def hook(module: Any, inputs: Any) -> None:
            gc.collect()
            self._memory_before[name] = get_process_memory_mb()
        return hook

    def _make_post_hook(self, name: str) -> Any:
        """Create a post-hook that measures memory delta after the module runs.

        Args:
            name: Module name.

        Returns:
            Hook function.
        """
        def hook(module: Any, inputs: Any, output: Any) -> None:
            current = get_process_memory_mb()
            if name in self._memory_before:
                delta = current - self._memory_before[name]
                if name not in self._memory_delta:
                    self._memory_delta[name] = []
                self._memory_delta[name].append(delta)
                del self._memory_before[name]

            if self._track_peak and current > self._peak_memory_mb:
                self._peak_memory_mb = current
        return hook

    def register_hooks(self, model: Any, max_depth: Optional[int] = 3) -> int:
        """Register memory tracking hooks on model modules.

        Args:
            model: PyTorch model.
            max_depth: Maximum module depth to track. None = all depths.

        Returns:
            Number of modules instrumented.
        """
        count = 0
        for name, module in model.named_modules():
            if name == "":
                name = "(root)"

            depth = name.count(".") + 1 if name != "(root)" else 0
            if max_depth is not None and depth > max_depth:
                continue

            pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
            post_hook = module.register_forward_hook(self._make_post_hook(name))
            self._hooks.extend([pre_hook, post_hook])
            count += 1

        logger.info("Registered memory hooks on %d modules", count)
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_memory_deltas(self) -> Dict[str, float]:
        """Get average memory delta per module.

        Returns:
            Dictionary mapping module names to average memory delta in MB.
        """
        result = {}
        for name, deltas in self._memory_delta.items():
            if deltas:
                result[name] = sum(deltas) / len(deltas)
        return result

    def get_peak_memory_mb(self) -> float:
        """Get peak process memory observed during profiling.

        Returns:
            Peak memory in MB.
        """
        return self._peak_memory_mb

    def reset(self) -> None:
        """Clear all collected memory data."""
        self._memory_before.clear()
        self._memory_delta.clear()
        self._peak_memory_mb = 0.0


def measure_model_footprint(model: Any) -> Dict[str, Any]:
    """Measure the static memory footprint of a loaded model.

    Counts parameter memory and buffer memory separately.

    Args:
        model: A loaded PyTorch model.

    Returns:
        Dictionary with parameter and buffer memory details.
    """
    import torch

    total_param_bytes = 0
    total_buffer_bytes = 0
    param_counts = {}

    for name, param in model.named_parameters():
        size_bytes = param.numel() * param.element_size()
        total_param_bytes += size_bytes
        # Group by top-level module
        top_module = name.split(".")[0] if "." in name else name
        param_counts[top_module] = param_counts.get(top_module, 0) + size_bytes

    for name, buffer in model.named_buffers():
        total_buffer_bytes += buffer.numel() * buffer.element_size()

    return {
        "total_param_mb": round(total_param_bytes / (1024 ** 2), 2),
        "total_buffer_mb": round(total_buffer_bytes / (1024 ** 2), 2),
        "total_mb": round((total_param_bytes + total_buffer_bytes) / (1024 ** 2), 2),
        "per_module_mb": {
            k: round(v / (1024 ** 2), 2)
            for k, v in sorted(param_counts.items(), key=lambda x: -x[1])
        },
    }
