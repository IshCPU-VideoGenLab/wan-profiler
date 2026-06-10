"""Report generation for profiling results.

Produces structured JSON output (machine-readable) and human-readable
summary tables for stdout. The JSON format is designed to be stable
and suitable for inclusion in research papers.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from wan_profiler.flops import categorize_module

logger = logging.getLogger(__name__)


def results_to_dict(results: Any) -> Dict[str, Any]:
    """Convert ProfileResults to a serializable dictionary.

    Args:
        results: A ProfileResults instance.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    per_module = []
    for name, profile in results.modules.items():
        if profile.avg_time_ms == 0 and profile.param_count == 0:
            continue

        entry = {
            "name": name,
            "type": profile.module_type,
            "category": categorize_module(name, profile.module_type),
            "param_count": profile.param_count,
            "avg_time_ms": round(profile.avg_time_ms, 3),
            "std_time_ms": round(profile.std_time_ms, 3),
            "time_pct": 0.0,
            "flops": profile.flops,
            "memory_mb": round(profile.memory_mb, 2),
        }

        if results.total_time_ms > 0 and profile.avg_time_ms > 0:
            entry["time_pct"] = round(
                (profile.avg_time_ms / results.total_time_ms) * 100, 2
            )

        per_module.append(entry)

    # Sort by time descending
    per_module.sort(key=lambda x: x["avg_time_ms"], reverse=True)

    # Category summary via EXCLUSIVE ("self") time, robust to untimed
    # intermediate containers (e.g. nn.ModuleList, whose forward hook never
    # fires). Each timed module's time is attributed to its NEAREST TIMED
    # ANCESTOR; self_time = time - sum(times of its timed descendants that
    # have no closer timed ancestor). This makes categories sum to ~100%
    # (previously ~300% from double-counting parents) and keeps attention's
    # scaled-dot-product compute (done inside the attention module, not a
    # child Linear) correctly in the "attention" bucket.
    ROOT_NAMES = {"", "(root)"}
    timed = {e["name"] for e in per_module if e["avg_time_ms"] > 0}
    root_name = next((e["name"] for e in per_module if e["name"] in ROOT_NAMES), None)

    def _nearest_timed_ancestor(name: str) -> Optional[str]:
        if name in ROOT_NAMES:
            return None
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            anc = ".".join(parts[:i])
            if anc in timed:
                return anc
        return root_name  # fall back to the model root

    child_time: Dict[str, float] = {}
    for entry in per_module:
        if entry["avg_time_ms"] <= 0:
            continue
        anc = _nearest_timed_ancestor(entry["name"])
        if anc is not None:
            child_time[anc] = child_time.get(anc, 0.0) + entry["avg_time_ms"]

    category_times: Dict[str, float] = {}
    for entry in per_module:
        self_time = max(0.0, entry["avg_time_ms"] - child_time.get(entry["name"], 0.0))
        cat = entry["category"]
        category_times[cat] = category_times.get(cat, 0) + self_time

    category_summary = []
    for cat, total_ms in sorted(category_times.items(), key=lambda x: -x[1]):
        pct = (total_ms / results.total_time_ms * 100) if results.total_time_ms > 0 else 0
        category_summary.append({
            "category": cat,
            "total_time_ms": round(total_ms, 3),
            "time_pct": round(pct, 2),
        })

    output = {
        "meta": {
            "tool": "wan-profiler",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "model": results.model_name,
        },
        "hardware": results.hardware,
        "config": results.config or {},
        "summary": {
            "total_time_ms": round(results.total_time_ms, 3),
            "total_flops": results.total_flops,
            "peak_memory_mb": round(results.peak_memory_mb, 2),
            "num_profiled_modules": len(per_module),
            "num_profile_steps": results.num_steps,
        },
        "category_breakdown": category_summary,
        "per_module": per_module,
    }

    return output


def save_json_report(
    results: Any,
    output_dir: str,
    filename: str = "profile_results.json",
) -> str:
    """Save profiling results as a JSON file.

    Args:
        results: A ProfileResults instance.
        output_dir: Directory to write the file to.
        filename: Output filename.

    Returns:
        Full path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    data = results_to_dict(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("JSON report saved to: %s", output_path)
    return output_path


def save_csv_report(
    results: Any,
    output_dir: str,
    filename: str = "profile_results.csv",
) -> str:
    """Save per-module profiling results as a CSV file.

    Args:
        results: A ProfileResults instance.
        output_dir: Directory to write the file to.
        filename: Output filename.

    Returns:
        Full path to the written file.
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    data = results_to_dict(results)
    modules = data["per_module"]

    if not modules:
        logger.warning("No module data to write to CSV")
        return output_path

    fieldnames = ["name", "type", "category", "param_count", "avg_time_ms",
                  "std_time_ms", "time_pct", "flops", "memory_mb"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for module in modules:
            writer.writerow({k: module.get(k, "") for k in fieldnames})

    logger.info("CSV report saved to: %s", output_path)
    return output_path


def format_summary_table(results: Any, top_n: int = 15) -> str:
    """Format a human-readable summary table for stdout.

    Args:
        results: A ProfileResults instance.
        top_n: Number of top modules to show.

    Returns:
        Formatted string for printing.
    """
    data = results_to_dict(results)
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 80)
    lines.append("  wan-profiler Results")
    lines.append("=" * 80)
    lines.append("")

    # Hardware
    hw = data["hardware"]
    lines.append(f"  Hardware: {hw.get('cpu', 'Unknown')} "
                 f"({hw.get('cores', '?')} cores, {hw.get('ram_gb', '?')} GB RAM)")
    lines.append(f"  Model:    {data['meta']['model']}")
    lines.append(f"  Config:   dtype={data['config'].get('dtype', '?')}, "
                 f"frames={data['config'].get('input_frames', '?')}, "
                 f"resolution={data['config'].get('input_resolution', '?')}")
    lines.append("")

    # Summary
    summary = data["summary"]
    lines.append(f"  Total forward pass time: {summary['total_time_ms']:.1f} ms")
    lines.append(f"  Profiled modules: {summary['num_profiled_modules']}")
    lines.append(f"  Profile steps: {summary['num_profile_steps']}")
    lines.append("")

    # Category breakdown
    lines.append("  Category Breakdown:")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Category':<20} {'Time (ms)':>12} {'Percent':>10}")
    lines.append("  " + "-" * 50)
    for cat in data["category_breakdown"]:
        lines.append(
            f"  {cat['category']:<20} {cat['total_time_ms']:>12.1f} {cat['time_pct']:>9.1f}%"
        )
    lines.append("  " + "-" * 50)
    lines.append("")

    # Top modules
    modules = data["per_module"][:top_n]
    if modules:
        lines.append(f"  Top {min(top_n, len(modules))} Modules by Time:")
        lines.append("  " + "-" * 76)
        lines.append(
            f"  {'#':<4} {'Module':<35} {'Type':<15} {'Time (ms)':>10} {'Pct':>7}"
        )
        lines.append("  " + "-" * 76)
        for i, mod in enumerate(modules, 1):
            name = mod["name"]
            if len(name) > 33:
                name = "..." + name[-30:]
            lines.append(
                f"  {i:<4} {name:<35} {mod['type']:<15} "
                f"{mod['avg_time_ms']:>10.1f} {mod['time_pct']:>6.1f}%"
            )
        lines.append("  " + "-" * 76)

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def generate_report(
    results: Any,
    output_dir: str = "results",
    save_json: bool = True,
    save_csv: bool = True,
    print_summary: bool = True,
) -> Dict[str, str]:
    """Generate all report outputs from profiling results.

    Args:
        results: A ProfileResults instance.
        output_dir: Directory to write files to.
        save_json: Whether to save JSON report.
        save_csv: Whether to save CSV report.
        print_summary: Whether to print summary to stdout.

    Returns:
        Dictionary of output file paths.
    """
    outputs = {}

    if save_json:
        outputs["json"] = save_json_report(results, output_dir)

    if save_csv:
        outputs["csv"] = save_csv_report(results, output_dir)

    if print_summary:
        summary = format_summary_table(results)
        print(summary)

    logger.info("Report generation complete. Files: %s", list(outputs.values()))
    return outputs
