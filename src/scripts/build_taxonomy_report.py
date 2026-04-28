#!/usr/bin/env python3
"""Build taxonomy overlap reports without rewriting manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.taxonomy import (  # noqa: E402
    DEFAULT_MACRO_LOG_CSV,
    DEFAULT_MACRO_ROOT,
    DEFAULT_MICRO_CONGO_ROOT,
    DEFAULT_OUTPUT_REPORT_DIR,
    DEFAULT_XYLOMAMI_SPECIES_CSV,
    build_and_write_reports,
    print_terminal_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build XyloMaMi-Bench taxonomy normalization and overlap reports."
    )
    parser.add_argument(
        "--macro_root",
        type=Path,
        default=DEFAULT_MACRO_ROOT,
        help="Processed macro dataset root.",
    )
    parser.add_argument(
        "--macro_log_csv",
        type=Path,
        default=DEFAULT_MACRO_LOG_CSV,
        help="Optional preprocess log CSV used to enrich macro taxonomy metadata.",
    )
    parser.add_argument(
        "--micro_root",
        type=Path,
        default=DEFAULT_MICRO_CONGO_ROOT,
        help="Micro Congo dataset root.",
    )
    parser.add_argument(
        "--xylomami_species_csv",
        type=Path,
        default=DEFAULT_XYLOMAMI_SPECIES_CSV,
        help="Final 100-species XyloMaMi-Bench CSV.",
    )
    parser.add_argument(
        "--taxonomy_correction_csv",
        type=Path,
        default=None,
        help="Optional CSV that extends or overrides the built-in taxonomy correction map.",
    )
    parser.add_argument(
        "--macro_output_image_root",
        type=Path,
        default=None,
        help="Optional path prefix to use when building report artifacts from macro records.",
    )
    parser.add_argument(
        "--micro_output_image_root",
        type=Path,
        default=None,
        help="Optional path prefix to use when building report artifacts from micro records.",
    )
    parser.add_argument(
        "--output_report_dir",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_DIR,
        help="Directory where taxonomy reports will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts, outputs = build_and_write_reports(
        macro_root=args.macro_root,
        micro_root=args.micro_root,
        xylomami_species_csv=args.xylomami_species_csv,
        correction_csv=args.taxonomy_correction_csv,
        macro_log_csv=args.macro_log_csv,
        output_report_dir=args.output_report_dir,
        macro_output_image_root=args.macro_output_image_root,
        micro_output_image_root=args.micro_output_image_root,
    )
    print_terminal_summary(artifacts)
    print("\nWrote files:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
