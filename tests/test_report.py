"""Tests for wan_profiler.report module."""

import json
import os
import pytest

from wan_profiler.profiler import ModuleProfile, ProfileResults
from wan_profiler.report import (
    format_summary_table,
    results_to_dict,
    save_json_report,
    save_csv_report,
)


@pytest.fixture
def sample_results() -> ProfileResults:
    """Create sample profiling results for testing."""
    modules = {
        "blocks.0.attention": ModuleProfile(
            name="blocks.0.attention",
            module_type="MultiHeadAttention",
            param_count=1000000,
            times_ms=[100.0, 110.0, 105.0],
        ),
        "blocks.0.ffn": ModuleProfile(
            name="blocks.0.ffn",
            module_type="FeedForward",
            param_count=2000000,
            times_ms=[50.0, 55.0, 52.0],
        ),
        "blocks.0.norm": ModuleProfile(
            name="blocks.0.norm",
            module_type="LayerNorm",
            param_count=1000,
            times_ms=[2.0, 2.1, 1.9],
        ),
    }

    return ProfileResults(
        model_name="wan-1.3b",
        hardware={
            "cpu": "Test CPU",
            "cores": 2,
            "threads": 4,
            "ram_gb": 16.0,
            "has_cuda": False,
        },
        modules=modules,
        total_time_ms=160.0,
        num_steps=3,
        config={
            "dtype": "float16",
            "low_memory": True,
            "input_frames": 8,
            "input_resolution": (256, 256),
        },
    )


class TestResultsToDict:
    """Tests for results_to_dict conversion."""

    def test_basic_structure(self, sample_results: ProfileResults) -> None:
        """Output should have required top-level keys."""
        data = results_to_dict(sample_results)
        assert "meta" in data
        assert "hardware" in data
        assert "summary" in data
        assert "category_breakdown" in data
        assert "per_module" in data

    def test_modules_sorted_by_time(self, sample_results: ProfileResults) -> None:
        """Modules should be sorted by time descending."""
        data = results_to_dict(sample_results)
        modules = data["per_module"]
        times = [m["avg_time_ms"] for m in modules]
        assert times == sorted(times, reverse=True)

    def test_time_percentages(self, sample_results: ProfileResults) -> None:
        """Time percentages should be calculated correctly."""
        data = results_to_dict(sample_results)
        for module in data["per_module"]:
            expected_pct = (module["avg_time_ms"] / 160.0) * 100
            assert module["time_pct"] == pytest.approx(expected_pct, abs=0.1)

    def test_meta_fields(self, sample_results: ProfileResults) -> None:
        """Meta section should contain tool info."""
        data = results_to_dict(sample_results)
        assert data["meta"]["tool"] == "wan-profiler"
        assert data["meta"]["model"] == "wan-1.3b"
        assert "timestamp" in data["meta"]


class TestSaveReports:
    """Tests for saving report files."""

    def test_save_json(self, sample_results: ProfileResults, tmp_path: str) -> None:
        """JSON report should be valid and readable."""
        output_path = save_json_report(sample_results, str(tmp_path))
        assert os.path.exists(output_path)

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["meta"]["model"] == "wan-1.3b"
        assert len(data["per_module"]) == 3

    def test_save_csv(self, sample_results: ProfileResults, tmp_path: str) -> None:
        """CSV report should be created with correct headers."""
        output_path = save_csv_report(sample_results, str(tmp_path))
        assert os.path.exists(output_path)

        with open(output_path, "r") as f:
            header = f.readline().strip()

        assert "name" in header
        assert "avg_time_ms" in header
        assert "category" in header


class TestFormatSummaryTable:
    """Tests for human-readable summary formatting."""

    def test_contains_model_name(self, sample_results: ProfileResults) -> None:
        """Summary should contain the model name."""
        table = format_summary_table(sample_results)
        assert "wan-1.3b" in table

    def test_contains_hardware(self, sample_results: ProfileResults) -> None:
        """Summary should contain hardware info."""
        table = format_summary_table(sample_results)
        assert "Test CPU" in table

    def test_contains_modules(self, sample_results: ProfileResults) -> None:
        """Summary should list top modules."""
        table = format_summary_table(sample_results)
        assert "attention" in table.lower()
