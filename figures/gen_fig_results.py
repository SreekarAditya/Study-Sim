#!/usr/bin/env python3
"""Regenerate all numerical result figures from the emitted CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from demo_sim.plotting import generate_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    args = parser.parse_args()
    generate_figures(Path(args.results_dir))


if __name__ == "__main__":
    main()

