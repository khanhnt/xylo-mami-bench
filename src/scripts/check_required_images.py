#!/usr/bin/env python3
"""Check whether split CSV images are available under a local image root."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = Path("data/processed/splits/p3_val.csv")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT,
        help="Split CSV to check. Default: data/processed/splits/p3_val.csv.",
    )
    parser.add_argument(
        "--macro_image_root",
        type=Path,
        required=True,
        help="Root directory that contains paths like Phase1/... from image_rel_path.",
    )
    parser.add_argument(
        "--modality",
        default="macro",
        choices=("macro", "micro", "all"),
        help="Which modality to check. Default: macro.",
    )
    parser.add_argument(
        "--missing_csv",
        type=Path,
        default=None,
        help="Optional CSV path for missing rows.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of missing paths to print. Default: 20.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_rows(split_csv: Path, modality: str) -> list[dict[str, str]]:
    with split_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")
        required = {"image_rel_path", "image_path", "modality", "species"}
        missing_columns = sorted(required - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"Split CSV is missing required columns: {missing_columns}")
        rows = [
            row
            for row in reader
            if modality == "all" or (row.get("modality") or "").strip().lower() == modality
        ]
    if not rows:
        raise ValueError(f"No rows matched modality={modality!r} in {split_csv}")
    return rows


def checked_path(root: Path, row: dict[str, str]) -> Path:
    rel_path = (row.get("image_rel_path") or "").strip()
    if rel_path:
        return root / rel_path
    return Path((row.get("image_path") or "").strip())


def write_missing_csv(path: Path, missing_rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checked_path",
        "image_rel_path",
        "image_path",
        "modality",
        "species",
        "specimen_id",
        "source_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in missing_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_csv = resolve_repo_path(args.split)
    image_root = args.macro_image_root.expanduser()
    missing_csv = resolve_repo_path(args.missing_csv) if args.missing_csv is not None else None

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")

    rows = read_rows(split_csv, args.modality)
    missing_rows: list[dict[str, str]] = []
    present_count = 0
    total_bytes = 0
    species_counter: Counter[str] = Counter()
    missing_species_counter: Counter[str] = Counter()

    for row in rows:
        species = (row.get("species") or "").strip()
        species_counter[species] += 1
        path = checked_path(image_root, row)
        if path.exists() and path.is_file():
            present_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            continue
        missing_species_counter[species] += 1
        missing_row = dict(row)
        missing_row["checked_path"] = str(path)
        missing_rows.append(missing_row)

    total = len(rows)
    missing_count = len(missing_rows)
    present_pct = 100.0 * present_count / total
    print("Image Availability Check")
    print(f"split_csv       : {split_csv}")
    print(f"image_root      : {image_root}")
    print(f"modality        : {args.modality}")
    print(f"required images : {total}")
    print(f"present         : {present_count} ({present_pct:.2f}%)")
    print(f"missing         : {missing_count} ({100.0 - present_pct:.2f}%)")
    print(f"present size    : {total_bytes / (1024 ** 3):.2f} GiB")
    print(f"species checked : {len(species_counter)}")
    print(f"species missing : {len(missing_species_counter)}")

    if missing_rows:
        print(f"\nFirst {min(args.sample, missing_count)} missing paths:")
        for row in missing_rows[: max(0, args.sample)]:
            print(row["checked_path"])
        if missing_csv is not None:
            write_missing_csv(missing_csv, missing_rows)
            print(f"\nMissing CSV written to: {missing_csv}")
        return 1

    print("\nOK: all required images are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
