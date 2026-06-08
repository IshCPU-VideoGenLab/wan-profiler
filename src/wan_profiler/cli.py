"""Command-line interface for wan-profiler.

Provides the main entry point for running profiling from the terminal.
Can be invoked as: python -m wan_profiler or via the `wan-profiler` console script.
"""

import argparse
import logging
import sys
from typing import List, Optional

from wan_profiler.config import ProfileConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="wan-profiler",
        description="Profile Wan 1.3B compute distribution for CPU-native video generation research.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="wan-1.3b",
        help="Model name or HuggingFace ID (default: wan-1.3b)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to model weights (downloads from HF if not set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model loading (default: float16)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        default=True,
        help="Enable memory-efficient loading (default: True)",
    )
    parser.add_argument(
        "--no-low-memory",
        action="store_false",
        dest="low_memory",
        help="Disable memory-efficient loading",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full profiling suite (time + FLOPs + memory)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of profiling iterations (default: 5)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Number of input video frames (default: 8)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Input resolution height width (default: 256 256)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


def setup_logging(debug: bool = False, quiet: bool = False) -> None:
    """Configure logging for the profiler.

    Args:
        debug: Enable debug-level logging.
        quiet: Suppress info-level logging.
    """
    level = logging.DEBUG if debug else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for wan-profiler CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args(argv)
    setup_logging(debug=args.debug, quiet=args.quiet)

    logger = logging.getLogger(__name__)
    logger.info("wan-profiler starting")

    try:
        config = ProfileConfig(
            model_name=args.model,
            model_path=args.model_path,
            output_dir=args.output,
            low_memory=args.low_memory,
            dtype=args.dtype,
            profile_time=True,
            profile_flops=args.full,
            profile_memory=args.full,
            num_warmup_steps=args.warmup,
            num_profile_steps=args.steps,
            input_frames=args.frames,
            input_resolution=tuple(args.resolution),
            verbose=not args.quiet,
        )

        logger.info("Configuration: %s", config)

        from wan_profiler.profiler import profile_model
        from wan_profiler.report import generate_report

        results = profile_model(config)
        generate_report(
            results,
            output_dir=config.output_dir,
            save_json=True,
            save_csv=True,
            print_summary=not args.quiet,
        )

        logger.info("Profiling complete. Results saved to: %s", config.output_dir)
        return 0

    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
        return 1
    except MemoryError as e:
        logger.error("Out of memory: %s", str(e))
        logger.error("Try: --low-memory --dtype float16 --frames 4 --resolution 128 128")
        return 1
    except Exception as e:
        logger.error("Profiling failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
