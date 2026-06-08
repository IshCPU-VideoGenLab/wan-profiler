"""Tests for wan_profiler.config module."""

import os
import pytest
from wan_profiler.config import ProfileConfig, HardwareInfo


class TestHardwareInfo:
    """Tests for HardwareInfo dataclass."""

    def test_auto_detection(self) -> None:
        """HardwareInfo should auto-detect CPU info."""
        hw = HardwareInfo()
        assert hw.cpu_cores >= 1
        assert hw.cpu_threads >= 1
        assert hw.ram_gb > 0

    def test_manual_override(self) -> None:
        """Manual values should override auto-detection."""
        hw = HardwareInfo(
            cpu_name="Test CPU",
            cpu_cores=4,
            cpu_threads=8,
            ram_gb=32.0,
            has_cuda=False,
        )
        assert hw.cpu_name == "Test CPU"
        assert hw.cpu_cores == 4
        assert hw.cpu_threads == 8
        assert hw.ram_gb == 32.0
        assert hw.has_cuda is False


class TestProfileConfig:
    """Tests for ProfileConfig dataclass."""

    def test_default_config(self, tmp_path: str) -> None:
        """Default config should be valid."""
        config = ProfileConfig(output_dir=str(tmp_path))
        assert config.model_name == "wan-1.3b"
        assert config.dtype == "float16"
        assert config.low_memory is True
        assert config.num_warmup_steps == 2
        assert config.num_profile_steps == 5

    def test_output_dir_created(self, tmp_path: str) -> None:
        """Output directory should be created on init."""
        output_dir = os.path.join(str(tmp_path), "new_dir")
        config = ProfileConfig(output_dir=output_dir)
        assert os.path.isdir(output_dir)

    def test_invalid_dtype_raises(self, tmp_path: str) -> None:
        """Invalid dtype should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            ProfileConfig(output_dir=str(tmp_path), dtype="int8")

    def test_negative_warmup_raises(self, tmp_path: str) -> None:
        """Negative warmup steps should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ProfileConfig(output_dir=str(tmp_path), num_warmup_steps=-1)

    def test_zero_profile_steps_raises(self, tmp_path: str) -> None:
        """Zero profile steps should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            ProfileConfig(output_dir=str(tmp_path), num_profile_steps=0)

    def test_custom_resolution(self, tmp_path: str) -> None:
        """Custom resolution should be stored correctly."""
        config = ProfileConfig(
            output_dir=str(tmp_path),
            input_resolution=(128, 128),
        )
        assert config.input_resolution == (128, 128)
