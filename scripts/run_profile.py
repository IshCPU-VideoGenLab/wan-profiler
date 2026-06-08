#!/usr/bin/env python
"""Run Wan 1.3B profiling.

Convenience script for running profiling from the scripts/ directory.
For full CLI options, see: python -m wan_profiler --help

Usage:
    python scripts/run_profile.py --model wan-1.3b --output results/
    python scripts/run_profile.py --model wan-1.3b --output results/ --low-memory
    python scripts/run_profile.py --model wan-1.3b --output results/ --full
"""

import sys
import os

# Add src/ to path so we can import wan_profiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wan_profiler.cli import main

if __name__ == "__main__":
    sys.exit(main())
