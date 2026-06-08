#!/usr/bin/env python
"""Generate visualizations from wan-profiler results.

Reads a profile_results.json file and produces bar charts showing
time, FLOPs, and memory distribution across modules and categories.

Usage:
    python scripts/visualize.py --input results/profile_results.json --output results/
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_results(input_path: str) -> Dict[str, Any]:
    """Load profiling results from a JSON file.

    Args:
        input_path: Path to the profile_results.json file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_category_breakdown(
    data: Dict[str, Any],
    output_dir: str,
) -> str:
    """Plot time breakdown by operation category.

    Args:
        data: Profiling results dictionary.
        output_dir: Directory to save the chart.

    Returns:
        Path to the saved chart image.
    """
    import matplotlib.pyplot as plt

    categories = data.get("category_breakdown", [])
    if not categories:
        logger.warning("No category data to plot")
        return ""

    names = [c["category"] for c in categories]
    times = [c["total_time_ms"] for c in categories]
    pcts = [c["time_pct"] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, times, color="#2196F3", edgecolor="white")

    # Add percentage labels
    for bar_item, pct in zip(bars, pcts):
        width = bar_item.get_width()
        ax.text(
            width + max(times) * 0.01,
            bar_item.get_y() + bar_item.get_height() / 2,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Wan 1.3B — Time by Operation Category")
    ax.invert_yaxis()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "category_breakdown.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Category chart saved to: %s", output_path)
    return output_path


def plot_top_modules(
    data: Dict[str, Any],
    output_dir: str,
    top_n: int = 15,
) -> str:
    """Plot time for top N most expensive modules.

    Args:
        data: Profiling results dictionary.
        output_dir: Directory to save the chart.
        top_n: Number of modules to show.

    Returns:
        Path to the saved chart image.
    """
    import matplotlib.pyplot as plt

    modules = data.get("per_module", [])[:top_n]
    if not modules:
        logger.warning("No module data to plot")
        return ""

    names = []
    for m in modules:
        name = m["name"]
        if len(name) > 40:
            name = "..." + name[-37:]
        names.append(name)

    times = [m["avg_time_ms"] for m in modules]

    # Color by category
    category_colors = {
        "attention": "#F44336",
        "ffn": "#FF9800",
        "normalization": "#4CAF50",
        "activation": "#9C27B0",
        "embedding": "#00BCD4",
        "temporal": "#3F51B5",
        "other": "#9E9E9E",
    }
    colors = [category_colors.get(m.get("category", "other"), "#9E9E9E") for m in modules]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(names, times, color=colors, edgecolor="white")

    # Add time labels
    for bar_item, t in zip(bars, times):
        width = bar_item.get_width()
        ax.text(
            width + max(times) * 0.01,
            bar_item.get_y() + bar_item.get_height() / 2,
            f"{t:.1f} ms",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Wan 1.3B — Top {len(modules)} Modules by Time")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=cat)
        for cat, color in category_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "top_modules.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Top modules chart saved to: %s", output_path)
    return output_path


def plot_parameter_distribution(
    data: Dict[str, Any],
    output_dir: str,
) -> str:
    """Plot parameter count distribution by category.

    Args:
        data: Profiling results dictionary.
        output_dir: Directory to save the chart.

    Returns:
        Path to the saved chart image.
    """
    import matplotlib.pyplot as plt

    modules = data.get("per_module", [])
    if not modules:
        return ""

    # Aggregate params by category
    category_params: Dict[str, int] = {}
    for m in modules:
        cat = m.get("category", "other")
        category_params[cat] = category_params.get(cat, 0) + m.get("param_count", 0)

    # Filter out zero-param categories
    category_params = {k: v for k, v in category_params.items() if v > 0}
    if not category_params:
        return ""

    sorted_items = sorted(category_params.items(), key=lambda x: -x[1])
    names = [item[0] for item in sorted_items]
    params = [item[1] / 1e6 for item in sorted_items]  # In millions

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, params, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Parameters (millions)")
    ax.set_title("Wan 1.3B — Parameter Distribution by Category")
    ax.invert_yaxis()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "param_distribution.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Parameter chart saved to: %s", output_path)
    return output_path


def main() -> int:
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate charts from wan-profiler results."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to profile_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for charts (default: results/)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top modules to show (default: 15)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        return 1

    os.makedirs(args.output, exist_ok=True)

    data = load_results(args.input)

    charts = []
    charts.append(plot_category_breakdown(data, args.output))
    charts.append(plot_top_modules(data, args.output, top_n=args.top_n))
    charts.append(plot_parameter_distribution(data, args.output))

    generated = [c for c in charts if c]
    logger.info("Generated %d charts in: %s", len(generated), args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
