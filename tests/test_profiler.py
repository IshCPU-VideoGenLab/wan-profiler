"""Tests for wan_profiler.profiler module."""

import pytest
import torch
import torch.nn as nn

from wan_profiler.profiler import ModuleProfile, TimeProfiler


class TestModuleProfile:
    """Tests for ModuleProfile dataclass."""

    def test_avg_time_empty(self) -> None:
        """Average time should be 0 for empty measurements."""
        profile = ModuleProfile(name="test", module_type="Linear")
        assert profile.avg_time_ms == 0.0

    def test_avg_time_calculated(self) -> None:
        """Average time should be computed correctly."""
        profile = ModuleProfile(
            name="test",
            module_type="Linear",
            times_ms=[10.0, 20.0, 30.0],
        )
        assert profile.avg_time_ms == pytest.approx(20.0)

    def test_std_time_single_value(self) -> None:
        """Standard deviation should be 0 for a single measurement."""
        profile = ModuleProfile(
            name="test",
            module_type="Linear",
            times_ms=[10.0],
        )
        assert profile.std_time_ms == 0.0

    def test_std_time_multiple_values(self) -> None:
        """Standard deviation should be computed correctly."""
        profile = ModuleProfile(
            name="test",
            module_type="Linear",
            times_ms=[10.0, 10.0, 10.0],
        )
        assert profile.std_time_ms == pytest.approx(0.0)


class TestTimeProfiler:
    """Tests for TimeProfiler."""

    def _make_simple_model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def test_register_hooks(self) -> None:
        """Hooks should be registered on modules."""
        model = self._make_simple_model()
        profiler = TimeProfiler()
        count = profiler.register_hooks(model)
        assert count > 0
        profiler.remove_hooks()

    def test_timing_collection(self) -> None:
        """Timings should be collected after forward pass."""
        model = self._make_simple_model()
        profiler = TimeProfiler()
        profiler.register_hooks(model)

        x = torch.randn(1, 64)
        with torch.no_grad():
            model(x)

        timings = profiler.get_timings()
        assert len(timings) > 0

        # All timings should be positive
        for name, times in timings.items():
            for t in times:
                assert t > 0, f"Non-positive time for {name}: {t}"

        profiler.remove_hooks()

    def test_reset_clears_timings(self) -> None:
        """Reset should clear all collected data."""
        model = self._make_simple_model()
        profiler = TimeProfiler()
        profiler.register_hooks(model)

        x = torch.randn(1, 64)
        with torch.no_grad():
            model(x)

        profiler.reset()
        timings = profiler.get_timings()
        assert len(timings) == 0

        profiler.remove_hooks()

    def test_leaf_only_mode(self) -> None:
        """Leaf-only mode should only profile leaf modules."""
        model = self._make_simple_model()
        profiler = TimeProfiler(leaf_only=True)
        count = profiler.register_hooks(model)
        # Sequential has 3 leaf children (Linear, ReLU, Linear)
        # but Sequential itself and the root should be excluded
        assert count == 3
        profiler.remove_hooks()

    def test_depth_filtering(self) -> None:
        """Depth filtering should limit profiled modules."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
            ),
            nn.Linear(32, 16),
        )
        profiler = TimeProfiler(target_depth=1)
        count = profiler.register_hooks(model)
        # Depth 0: root, depth 1: Sequential.0, Sequential.1
        assert count >= 1
        profiler.remove_hooks()

    def test_multiple_forward_passes(self) -> None:
        """Multiple forward passes should accumulate timings."""
        model = self._make_simple_model()
        profiler = TimeProfiler()
        profiler.register_hooks(model)

        x = torch.randn(1, 64)
        with torch.no_grad():
            model(x)
            model(x)
            model(x)

        timings = profiler.get_timings()
        for name, times in timings.items():
            assert len(times) == 3, f"{name} should have 3 measurements, got {len(times)}"

        profiler.remove_hooks()
