"""Configuration for wan-profiler runs."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareInfo:
    """Information about the hardware being profiled on.

    Auto-detected at runtime, but can be overridden for testing.
    """

    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_gb: float = 0.0
    has_cuda: bool = False

    def __post_init__(self) -> None:
        """Auto-detect hardware if not provided."""
        if not self.cpu_name:
            self._detect()

    def _detect(self) -> None:
        """Detect hardware specs from the system."""
        import psutil

        try:
            import platform
            self.cpu_name = platform.processor() or "Unknown CPU"
        except Exception:
            self.cpu_name = "Unknown CPU"

        self.cpu_cores = psutil.cpu_count(logical=False) or 1
        self.cpu_threads = psutil.cpu_count(logical=True) or 1
        self.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
        except ImportError:
            self.has_cuda = False


@dataclass
class ProfileConfig:
    """Configuration for a profiling run.

    Args:
        model_name: Name/path of the model to profile (e.g., "wan-1.3b").
        model_path: Local path to model weights. If None, downloads from HuggingFace.
        output_dir: Directory to write results to.
        low_memory: Enable memory-efficient loading (layer-by-layer).
        dtype: Data type for model loading ("float16", "bfloat16", "float32").
        profile_time: Whether to profile wall-clock time per module.
        profile_flops: Whether to estimate FLOPs per module.
        profile_memory: Whether to track memory per module.
        num_warmup_steps: Number of warmup iterations before measurement.
        num_profile_steps: Number of iterations to average measurements over.
        input_frames: Number of video frames for the dummy input.
        input_resolution: Resolution (height, width) for the dummy input.
        verbose: Whether to print progress to stdout.
    """

    model_name: str = "wan-1.3b"
    model_path: Optional[str] = None
    output_dir: str = "results"
    low_memory: bool = True
    dtype: str = "float16"
    profile_time: bool = True
    profile_flops: bool = False
    profile_memory: bool = False
    num_warmup_steps: int = 2
    num_profile_steps: int = 5
    input_frames: int = 8
    input_resolution: tuple = (256, 256)
    verbose: bool = True
    hardware: HardwareInfo = field(default_factory=HardwareInfo)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        os.makedirs(self.output_dir, exist_ok=True)

        valid_dtypes = {"float16", "bfloat16", "float32"}
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Must be one of: {valid_dtypes}"
            )

        if self.num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative")

        if self.num_profile_steps < 1:
            raise ValueError("num_profile_steps must be at least 1")
