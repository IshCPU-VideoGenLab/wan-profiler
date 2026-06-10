"""Core profiling engine for Wan 1.3B.

Uses PyTorch forward hooks to measure per-module execution time
without modifying the model's source code.
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModuleProfile:
    """Profiling data for a single module.

    Args:
        name: Fully qualified module name.
        module_type: Class name of the module.
        param_count: Number of parameters (non-recursive).
        times_ms: List of measured forward pass times in milliseconds.
        flops: Estimated FLOPs for one forward pass.
        memory_mb: Peak memory usage in megabytes.
    """

    name: str
    module_type: str
    param_count: int = 0
    times_ms: List[float] = field(default_factory=list)
    flops: int = 0
    memory_mb: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average forward pass time in milliseconds."""
        if not self.times_ms:
            return 0.0
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def std_time_ms(self) -> float:
        """Standard deviation of forward pass time."""
        if len(self.times_ms) < 2:
            return 0.0
        avg = self.avg_time_ms
        variance = sum((t - avg) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)
        return variance ** 0.5


@dataclass
class ProfileResults:
    """Complete profiling results for a model.

    Args:
        model_name: Name of the profiled model.
        hardware: Hardware info dictionary.
        modules: Per-module profiling data.
        total_time_ms: Total forward pass time (averaged).
        total_flops: Total FLOPs for one forward pass.
        peak_memory_mb: Peak memory during profiling.
        num_steps: Number of profiling iterations.
        config: Configuration used for profiling.
    """

    model_name: str
    hardware: Dict[str, Any]
    modules: Dict[str, ModuleProfile]
    total_time_ms: float = 0.0
    total_flops: int = 0
    peak_memory_mb: float = 0.0
    num_steps: int = 0
    config: Optional[Dict[str, Any]] = None


class TimeProfiler:
    """Hook-based time profiler for PyTorch models.

    Registers forward pre-hooks and hooks on target modules to measure
    wall-clock execution time per module.

    Args:
        target_depth: Maximum module depth to profile. None = all depths.
        leaf_only: If True, only profile leaf modules (no children).
    """

    def __init__(
        self,
        target_depth: Optional[int] = None,
        leaf_only: bool = False,
    ) -> None:
        self._target_depth = target_depth
        self._leaf_only = leaf_only
        self._hooks: List[Any] = []
        self._timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def _get_depth(self, name: str) -> int:
        """Get the depth of a module from its dotted name."""
        if name == "" or name == "(root)":
            return 0
        return name.count(".") + 1

    def _should_profile(self, name: str, module: Any) -> bool:
        """Determine if a module should be profiled.

        Args:
            name: Module name.
            module: The module instance.

        Returns:
            True if the module should be profiled.
        """
        if self._leaf_only and len(list(module.children())) > 0:
            return False
        if self._target_depth is not None:
            if self._get_depth(name) > self._target_depth:
                return False
        return True

    def _make_pre_hook(self, name: str) -> Callable:
        """Create a forward pre-hook that records the start time.

        Args:
            name: Module name for identification.

        Returns:
            Hook function.
        """
        def hook(module: Any, inputs: Any) -> None:
            self._start_times[name] = time.perf_counter()
        return hook

    def _make_post_hook(self, name: str) -> Callable:
        """Create a forward hook that records the elapsed time.

        Args:
            name: Module name for identification.

        Returns:
            Hook function.
        """
        def hook(module: Any, inputs: Any, output: Any) -> None:
            if name in self._start_times:
                elapsed = (time.perf_counter() - self._start_times[name]) * 1000
                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(elapsed)
                del self._start_times[name]
        return hook

    def register_hooks(self, model: Any) -> int:
        """Register timing hooks on all target modules.

        Args:
            model: PyTorch model to instrument.

        Returns:
            Number of modules instrumented.
        """
        count = 0
        for name, module in model.named_modules():
            if name == "":
                name = "(root)"
            if not self._should_profile(name, module):
                continue

            pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
            post_hook = module.register_forward_hook(self._make_post_hook(name))
            self._hooks.extend([pre_hook, post_hook])
            count += 1

        logger.info("Registered timing hooks on %d modules", count)
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("All timing hooks removed")

    def get_timings(self) -> Dict[str, List[float]]:
        """Get collected timing data.

        Returns:
            Dictionary mapping module names to lists of elapsed times (ms).
        """
        return dict(self._timings)

    def reset(self) -> None:
        """Clear all collected timing data."""
        self._timings.clear()
        self._start_times.clear()


def create_dummy_input(
    num_frames: int = 8,
    resolution: Tuple[int, int] = (256, 256),
    dtype_str: str = "float16",
) -> Dict[str, Any]:
    """Create a dummy input for the Wan diffusion transformer (DiT) forward pass.

    Matches ``WanTransformer3DModel.forward(hidden_states, timestep,
    encoder_hidden_states)``. The text embedding is a DUMMY tensor of the
    correct shape (text_dim = 4096), so we never need to load the ~11 GB T5
    text encoder just to profile where the DiT spends compute.

    Args:
        num_frames: Number of pixel-space video frames.
        resolution: (height, width) of each pixel-space frame.
        dtype_str: Data type string.

    Returns:
        Dictionary of input tensors keyed by the DiT's forward argument names.
    """
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    h, w = resolution
    # Wan VAE compresses 8x spatially and ~4x temporally; the DiT latent has
    # 16 channels. Keeping the token count modest (<= rope_max_seq_len=1024)
    # is enough to exercise every module for profiling.
    latent_h = h // 8
    latent_w = w // 8
    latent_frames = max(1, num_frames // 4)
    latent_channels = 16
    text_seq_len = 512   # dummy prompt length
    text_dim = 4096      # umt5-xxl embedding dim (WanTransformer3DModel.text_dim)

    dummy_input = {
        "hidden_states": torch.randn(
            1, latent_channels, latent_frames, latent_h, latent_w, dtype=dtype,
        ),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, text_seq_len, text_dim, dtype=dtype),
    }

    logger.info(
        "Created dummy DiT input: hidden_states %s, encoder_hidden_states %s, dtype %s",
        list(dummy_input["hidden_states"].shape),
        list(dummy_input["encoder_hidden_states"].shape),
        dtype,
    )
    return dummy_input


def profile_model(config: Any) -> ProfileResults:
    """Run a complete profiling session on the model.

    This is the main entry point for profiling. It loads the model,
    registers hooks, runs forward passes, and collects results.

    Args:
        config: A ProfileConfig instance.

    Returns:
        ProfileResults with all collected measurements.
    """
    from wan_profiler.loader import load_model, get_module_tree

    logger.info("Starting profiling run for '%s'", config.model_name)

    # Load model
    logger.info("Loading model...")
    model = load_model(
        model_name=config.model_name,
        model_path=config.model_path,
        dtype=config.dtype,
        low_memory=config.low_memory,
    )

    # Get module tree
    module_tree = get_module_tree(model)
    logger.info("Model has %d modules", len(module_tree))

    # Setup profiler
    time_profiler = TimeProfiler(target_depth=3, leaf_only=False)
    time_profiler.register_hooks(model)

    # Create dummy input
    dummy_input = create_dummy_input(
        num_frames=config.input_frames,
        resolution=config.input_resolution,
        dtype_str=config.dtype,
    )

    import torch

    # Warmup
    logger.info("Running %d warmup steps...", config.num_warmup_steps)
    with torch.no_grad():
        for i in range(config.num_warmup_steps):
            try:
                model(**dummy_input)
            except Exception as e:
                logger.warning("Warmup step %d failed: %s", i, str(e))
                logger.info("Trying alternative forward pass...")
                try:
                    model(dummy_input["hidden_states"], dummy_input["timestep"], dummy_input["encoder_hidden_states"])
                except Exception as e2:
                    logger.error("Alternative forward pass also failed: %s", str(e2))
                    break

    # Reset timings after warmup
    time_profiler.reset()

    # Profile
    logger.info("Running %d profiling steps...", config.num_profile_steps)
    total_times = []
    with torch.no_grad():
        for i in range(config.num_profile_steps):
            gc.collect()
            start = time.perf_counter()
            try:
                model(**dummy_input)
            except Exception:
                try:
                    model(dummy_input["hidden_states"], dummy_input["timestep"], dummy_input["encoder_hidden_states"])
                except Exception as e:
                    logger.error("Profiling step %d failed: %s", i, str(e))
                    continue
            elapsed = (time.perf_counter() - start) * 1000
            total_times.append(elapsed)
            if config.verbose:
                logger.info("Step %d/%d: %.1f ms", i + 1, config.num_profile_steps, elapsed)

    # Collect results
    timings = time_profiler.get_timings()
    time_profiler.remove_hooks()

    # Build module profiles
    modules = {}
    for name, info in module_tree.items():
        profile = ModuleProfile(
            name=name,
            module_type=info["type"],
            param_count=info["param_count"],
        )
        if name in timings:
            profile.times_ms = timings[name]
        modules[name] = profile

    # Build results
    avg_total = sum(total_times) / len(total_times) if total_times else 0.0

    results = ProfileResults(
        model_name=config.model_name,
        hardware={
            "cpu": config.hardware.cpu_name,
            "cores": config.hardware.cpu_cores,
            "threads": config.hardware.cpu_threads,
            "ram_gb": config.hardware.ram_gb,
            "has_cuda": config.hardware.has_cuda,
        },
        modules=modules,
        total_time_ms=avg_total,
        num_steps=config.num_profile_steps,
        config={
            "dtype": config.dtype,
            "low_memory": config.low_memory,
            "input_frames": config.input_frames,
            "input_resolution": config.input_resolution,
        },
    )

    logger.info(
        "Profiling complete. Average total time: %.1f ms across %d steps",
        avg_total,
        len(total_times),
    )

    # Cleanup
    del model
    gc.collect()

    return results
