"""Command-line entry point."""

from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results", help="Result directory (default: results)")
    args = parser.parse_args()
    output = run_pipeline(args.output)
    print(f"Synthetic demonstration complete: {output}")
    print(f"Report: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()

