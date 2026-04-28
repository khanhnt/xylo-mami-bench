#!/usr/bin/env python3
"""Preprocess the XyloMaMi-Bench final-100 macro wood subset into a JPEG-only dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageFile, UnidentifiedImageError
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    Image = None
    ImageFile = None

    class UnidentifiedImageError(Exception):
        """Placeholder used when Pillow is unavailable."""


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PHASE1_ROOT = Path(
    os.environ.get(
        "XYLOMAMI_PHASE1_ROOT",
        str(REPO_ROOT / "data" / "source" / "macro_phase1"),
    )
)
DEFAULT_PHASE2_ROOT = Path(
    os.environ.get("XYLOMAMI_PHASE2_ROOT", str(REPO_ROOT / "data" / "source" / "macro_phase2"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "XYLOMAMI_PREPROCESS_OUTPUT_ROOT",
        str(REPO_ROOT / "data" / "images" / "macro"),
    )
)
DEFAULT_XYLOMAMI_SPECIES_CSV = REPO_ROOT / "data" / "xylomami_final_100_species_list.csv"

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
SPECIES_FOLDER_PATTERN = re.compile(r"^\s*(?P<prefix>\d+)\s*\.\s*(?P<species>.+?)\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")

DEFAULT_TAXONOMY_CORRECTIONS = {
    "Albizia zyia": "Albizia zygia",
    "Bagassa guiamensis": "Bagassa guianensis",
    "Hevea brasilensis": "Hevea brasiliensis",
    "Burckella obovati": "Burckella obovata",
    "Heritiera littoralin": "Heritiera littoralis",
    "Entandrophragma anolens": "Entandrophragma angolense",
}

TAXONOMY_CORRECTION_SOURCE_COLUMNS = (
    "raw_species_name",
    "raw_name",
    "raw",
    "typo",
    "source",
    "incorrect_name",
    "input_species",
)
TAXONOMY_CORRECTION_TARGET_COLUMNS = (
    "corrected_species_name",
    "corrected_name",
    "corrected",
    "canonical_name",
    "canonical",
    "target",
    "species",
    "matched_species_name",
)

LOG_COLUMNS = [
    "original_path",
    "processed_path",
    "original_width",
    "original_height",
    "new_width",
    "new_height",
    "original_bytes",
    "new_bytes",
    "status",
    "error_message",
    "phase",
    "raw_species_folder_name",
    "raw_species_name",
    "corrected_species_name",
    "in_xylomami_final_100",
    "protocol_group",
    "overlap_type",
    "protocol_usage",
    "matched_species_name",
    "taxonomy_match_status",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def require_pillow() -> None:
    if Image is None or ImageFile is None:
        raise RuntimeError(
            "Pillow is required for this script. Install it with `pip install pillow`."
        )
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def get_lanczos_filter() -> int:
    require_pillow()
    assert Image is not None
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip(" .")


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def resolve_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


@dataclass(frozen=True)
class FinalSpeciesRecord:
    species: str
    genus: str
    group: str
    overlap_type: str
    protocol_usage: str
    group_order: str


@dataclass(frozen=True)
class TaxonomyAnnotation:
    raw_species_folder_name: str
    raw_species_name: str
    corrected_species_name: str
    in_xylomami_final_100: int
    protocol_group: str
    overlap_type: str
    protocol_usage: str
    matched_species_name: str
    taxonomy_match_status: str


@dataclass(frozen=True)
class FileTask:
    source_path: Path
    output_path: Path
    relative_path: Path
    phase: str
    annotation: TaxonomyAnnotation


@dataclass
class ProcessingResult:
    original_path: str
    processed_path: str
    original_width: int | None
    original_height: int | None
    new_width: int | None
    new_height: int | None
    original_bytes: int | None
    new_bytes: int | None
    status: str
    error_message: str
    phase: str
    raw_species_folder_name: str
    raw_species_name: str
    corrected_species_name: str
    in_xylomami_final_100: int
    protocol_group: str
    overlap_type: str
    protocol_usage: str
    matched_species_name: str
    taxonomy_match_status: str

    def to_csv_row(self) -> dict[str, str | int]:
        row: dict[str, str | int] = {
            "original_path": self.original_path,
            "processed_path": self.processed_path,
            "original_width": "" if self.original_width is None else self.original_width,
            "original_height": "" if self.original_height is None else self.original_height,
            "new_width": "" if self.new_width is None else self.new_width,
            "new_height": "" if self.new_height is None else self.new_height,
            "original_bytes": "" if self.original_bytes is None else self.original_bytes,
            "new_bytes": "" if self.new_bytes is None else self.new_bytes,
            "status": self.status,
            "error_message": self.error_message,
            "phase": self.phase,
            "raw_species_folder_name": self.raw_species_folder_name,
            "raw_species_name": self.raw_species_name,
            "corrected_species_name": self.corrected_species_name,
            "in_xylomami_final_100": self.in_xylomami_final_100,
            "protocol_group": self.protocol_group,
            "overlap_type": self.overlap_type,
            "protocol_usage": self.protocol_usage,
            "matched_species_name": self.matched_species_name,
            "taxonomy_match_status": self.taxonomy_match_status,
        }
        return row


@dataclass
class RunStats:
    files_found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_original_bytes: int = 0
    total_output_bytes: int = 0

    def __post_init__(self) -> None:
        self.phase_counts: Counter[str] = Counter()
        self.xylomami_counts: Counter[str] = Counter()
        self.group_counts: Counter[str] = Counter()

    def update(self, result: ProcessingResult) -> None:
        self.files_found += 1
        self.phase_counts[result.phase] += 1
        self.xylomami_counts[str(result.in_xylomami_final_100)] += 1
        self.group_counts[result.protocol_group or "unmatched"] += 1

        if result.status == "processed":
            self.processed += 1
        elif result.status.startswith("skipped"):
            self.skipped += 1
        else:
            self.failed += 1

        if result.original_bytes is not None:
            self.total_original_bytes += result.original_bytes
        if result.new_bytes is not None:
            self.total_output_bytes += result.new_bytes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan both macro roots, keep only files that match the "
            "XyloMaMi-Bench final 100 species list, and export aspect-ratio-preserving "
            "JPEGs plus a detailed CSV log."
        )
    )
    parser.add_argument("--phase1_root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--phase2_root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--xylomami_species_csv",
        type=Path,
        default=DEFAULT_XYLOMAMI_SPECIES_CSV,
    )
    parser.add_argument(
        "--taxonomy_correction_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV that extends or overrides the built-in taxonomy correction map. "
            "Preferred columns: raw_species_name, corrected_species_name."
        ),
    )
    parser.add_argument("--long_side", type=int, default=1280)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads to use for image conversion.",
    )
    parser.add_argument(
        "--log_csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <output_root>/preprocess_macro_dataset_log.csv.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.phase1_root = resolve_path(args.phase1_root)
    args.phase2_root = resolve_path(args.phase2_root)
    args.output_root = resolve_path(args.output_root)
    args.xylomami_species_csv = resolve_path(args.xylomami_species_csv)
    args.taxonomy_correction_csv = (
        resolve_path(args.taxonomy_correction_csv)
        if args.taxonomy_correction_csv is not None
        else None
    )
    args.log_csv = (
        resolve_path(args.log_csv)
        if args.log_csv is not None
        else args.output_root / "preprocess_macro_dataset_log.csv"
    )

    if args.long_side <= 0:
        raise ValueError("--long_side must be greater than 0.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be greater than 0.")
    if not args.phase1_root.is_dir():
        raise FileNotFoundError(f"Phase 1 root not found: {args.phase1_root}")
    if not args.phase2_root.is_dir():
        raise FileNotFoundError(f"Phase 2 root not found: {args.phase2_root}")
    if not args.xylomami_species_csv.is_file():
        raise FileNotFoundError(
            f"XyloMaMi-Bench species CSV not found: {args.xylomami_species_csv}"
        )
    if args.taxonomy_correction_csv is not None and not args.taxonomy_correction_csv.is_file():
        raise FileNotFoundError(
            f"Taxonomy correction CSV not found: {args.taxonomy_correction_csv}"
        )
    return args


def load_final_species_records(path: Path) -> dict[str, FinalSpeciesRecord]:
    required_columns = {"species", "genus", "group", "overlap_type", "protocol_usage"}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"XyloMaMi-Bench species CSV is missing required columns: {missing}")

        records: dict[str, FinalSpeciesRecord] = {}
        for row in reader:
            species_name = (row.get("species") or "").strip()
            if not species_name:
                continue

            key = normalize_name(species_name)
            if key in records:
                logging.warning(
                    "Duplicate XyloMaMi-Bench species entry for '%s'; keeping the first record.",
                    species_name,
                )
                continue

            records[key] = FinalSpeciesRecord(
                species=species_name,
                genus=(row.get("genus") or "").strip(),
                group=(row.get("group") or "").strip(),
                overlap_type=(row.get("overlap_type") or "").strip(),
                protocol_usage=(row.get("protocol_usage") or "").strip(),
                group_order=(row.get("group_order") or "").strip(),
            )

    return records


def load_taxonomy_corrections(path: Path | None) -> dict[str, str]:
    corrections = {
        normalize_name(raw_name): corrected_name
        for raw_name, corrected_name in DEFAULT_TAXONOMY_CORRECTIONS.items()
    }
    if path is None:
        return corrections

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return corrections

    header = [normalize_name(value) for value in rows[0]]
    source_index = next(
        (index for index, name in enumerate(header) if name in TAXONOMY_CORRECTION_SOURCE_COLUMNS),
        None,
    )
    target_index = next(
        (index for index, name in enumerate(header) if name in TAXONOMY_CORRECTION_TARGET_COLUMNS),
        None,
    )

    start_index = 0
    if source_index is not None and target_index is not None:
        start_index = 1
    elif len(rows[0]) >= 2:
        source_index = 0
        target_index = 1
    else:
        raise ValueError(
            "Taxonomy correction CSV must either provide recognized header columns "
            "or at least two columns per row."
        )

    assert source_index is not None
    assert target_index is not None

    for row in rows[start_index:]:
        if len(row) <= max(source_index, target_index):
            continue

        raw_name = row[source_index].strip()
        corrected_name = row[target_index].strip()
        if not raw_name or not corrected_name:
            continue

        corrections[normalize_name(raw_name)] = corrected_name

    return corrections


def iter_image_files(root: Path) -> Iterable[Path]:
    def onerror(error: OSError) -> None:
        logging.warning("Directory walk error under %s: %s", root, error)

    for current_root, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames.sort()
        filenames.sort()
        current_root_path = Path(current_root)
        for filename in filenames:
            candidate = current_root_path / filename
            if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                yield candidate


def parse_species_folder(relative_path: Path) -> tuple[str, str]:
    directory_parts = relative_path.parts[:-1]
    # Prefer the outermost matching species folder. Some datasets include
    # specimen-level subfolders such as "3333. Bagassa guianensis.3" inside the
    # canonical species folder "3333.Bagassa guianensis"; scanning from the root
    # side preserves the true species name.
    for part in directory_parts:
        match = SPECIES_FOLDER_PATTERN.match(part)
        if match:
            return part.strip(), match.group("species").strip()
    return "", ""


def build_taxonomy_annotation(
    raw_species_folder_name: str,
    raw_species_name: str,
    corrections: dict[str, str],
    final_species_records: dict[str, FinalSpeciesRecord],
) -> TaxonomyAnnotation:
    if not raw_species_name:
        return TaxonomyAnnotation(
            raw_species_folder_name=raw_species_folder_name,
            raw_species_name="",
            corrected_species_name="",
            in_xylomami_final_100=0,
            protocol_group="",
            overlap_type="",
            protocol_usage="",
            matched_species_name="",
            taxonomy_match_status="missing_species_folder",
        )

    normalized_raw = normalize_name(raw_species_name)
    raw_match = final_species_records.get(normalized_raw)

    corrected_species_name = corrections.get(normalized_raw, raw_species_name)
    normalized_corrected = normalize_name(corrected_species_name)
    corrected_match = final_species_records.get(normalized_corrected)

    if raw_match is not None:
        matched_record = raw_match
        taxonomy_match_status = "matched_raw"
    elif corrected_match is not None:
        matched_record = corrected_match
        taxonomy_match_status = "matched_corrected"
    else:
        matched_record = None
        taxonomy_match_status = "not_in_xylomami"

    return TaxonomyAnnotation(
        raw_species_folder_name=raw_species_folder_name,
        raw_species_name=raw_species_name,
        corrected_species_name=corrected_species_name,
        in_xylomami_final_100=1 if matched_record is not None else 0,
        protocol_group="" if matched_record is None else matched_record.group,
        overlap_type="" if matched_record is None else matched_record.overlap_type,
        protocol_usage="" if matched_record is None else matched_record.protocol_usage,
        matched_species_name="" if matched_record is None else matched_record.species,
        taxonomy_match_status=taxonomy_match_status,
    )


@dataclass(frozen=True)
class ScannedFile:
    source_path: Path
    relative_path: Path
    tentative_output_relative: Path
    phase: str
    annotation: TaxonomyAnnotation


@dataclass(frozen=True)
class ScanPhaseResult:
    tasks: list[FileTask]
    scanned_files: int
    selected_files: int
    excluded_not_in_xylomami: int


@dataclass(frozen=True)
class BuildTasksResult:
    tasks: list[FileTask]
    scanned_files: int
    selected_files: int
    excluded_not_in_xylomami: int


def scan_phase(
    root: Path,
    phase: str,
    output_root: Path,
    corrections: dict[str, str],
    final_species_records: dict[str, FinalSpeciesRecord],
) -> ScanPhaseResult:
    scanned_files: list[ScannedFile] = []
    tentative_counts: Counter[str] = Counter()
    scanned_count = 0
    excluded_not_in_xylomami = 0

    for source_path in iter_image_files(root):
        scanned_count += 1
        relative_path = source_path.relative_to(root)
        raw_species_folder_name, raw_species_name = parse_species_folder(relative_path)
        annotation = build_taxonomy_annotation(
            raw_species_folder_name=raw_species_folder_name,
            raw_species_name=raw_species_name,
            corrections=corrections,
            final_species_records=final_species_records,
        )
        if annotation.in_xylomami_final_100 != 1:
            excluded_not_in_xylomami += 1
            continue

        tentative_output_relative = relative_path.with_suffix(".jpg")
        scanned_files.append(
            ScannedFile(
                source_path=source_path,
                relative_path=relative_path,
                tentative_output_relative=tentative_output_relative,
                phase=phase,
                annotation=annotation,
            )
        )
        tentative_counts[tentative_output_relative.as_posix()] += 1

    tasks: list[FileTask] = []
    assigned_output_paths: Counter[str] = Counter()

    for scanned_file in scanned_files:
        output_relative = scanned_file.tentative_output_relative
        tentative_key = output_relative.as_posix()
        if tentative_counts[tentative_key] > 1:
            extension_token = scanned_file.source_path.suffix.lower().lstrip(".") or "img"
            output_relative = scanned_file.relative_path.with_name(
                f"{scanned_file.source_path.stem}__from_{extension_token}.jpg"
            )

        candidate_key = output_relative.as_posix()
        assigned_output_paths[candidate_key] += 1
        if assigned_output_paths[candidate_key] > 1:
            output_relative = output_relative.with_name(
                f"{output_relative.stem}__dup{assigned_output_paths[candidate_key]}.jpg"
            )

        tasks.append(
            FileTask(
                source_path=scanned_file.source_path,
                output_path=output_root / phase / output_relative,
                relative_path=scanned_file.relative_path,
                phase=phase,
                annotation=scanned_file.annotation,
            )
        )

    return ScanPhaseResult(
        tasks=tasks,
        scanned_files=scanned_count,
        selected_files=len(tasks),
        excluded_not_in_xylomami=excluded_not_in_xylomami,
    )


def compute_resized_dimensions(width: int, height: int, long_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    current_long_side = max(width, height)
    if current_long_side == long_side:
        return width, height

    scale = long_side / float(current_long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height


def read_valid_output_metadata(
    path: Path,
    expected_long_side: int,
) -> tuple[int, int, int] | None:
    require_pillow()
    assert Image is not None

    if not path.is_file():
        return None

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            image_format = (image.format or "").upper()

        new_bytes = path.stat().st_size
        if image_format != "JPEG":
            return None
        if width <= 0 or height <= 0 or new_bytes <= 0:
            return None
        if max(width, height) != expected_long_side:
            return None
        return width, height, new_bytes
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def result_from_task(
    task: FileTask,
    *,
    original_width: int | None = None,
    original_height: int | None = None,
    new_width: int | None = None,
    new_height: int | None = None,
    original_bytes: int | None = None,
    new_bytes: int | None = None,
    status: str,
    error_message: str = "",
) -> ProcessingResult:
    return ProcessingResult(
        original_path=str(task.source_path),
        processed_path=str(task.output_path),
        original_width=original_width,
        original_height=original_height,
        new_width=new_width,
        new_height=new_height,
        original_bytes=original_bytes,
        new_bytes=new_bytes,
        status=status,
        error_message=error_message,
        phase=task.phase,
        raw_species_folder_name=task.annotation.raw_species_folder_name,
        raw_species_name=task.annotation.raw_species_name,
        corrected_species_name=task.annotation.corrected_species_name,
        in_xylomami_final_100=task.annotation.in_xylomami_final_100,
        protocol_group=task.annotation.protocol_group,
        overlap_type=task.annotation.overlap_type,
        protocol_usage=task.annotation.protocol_usage,
        matched_species_name=task.annotation.matched_species_name,
        taxonomy_match_status=task.annotation.taxonomy_match_status,
    )


def process_task(
    task: FileTask,
    *,
    overwrite: bool,
    long_side: int,
    jpeg_quality: int,
) -> ProcessingResult:
    require_pillow()
    assert Image is not None

    original_bytes = safe_size(task.source_path)

    if not overwrite:
        existing_output = read_valid_output_metadata(task.output_path, long_side)
        if existing_output is not None:
            new_width, new_height, new_bytes = existing_output
            return result_from_task(
                task,
                new_width=new_width,
                new_height=new_height,
                original_bytes=original_bytes,
                new_bytes=new_bytes,
                status="skipped_exists",
            )

    try:
        with Image.open(task.source_path) as image:
            image.load()
            original_width, original_height = image.size
            processed_image = image if image.mode == "RGB" else image.convert("RGB")
            if processed_image is image:
                processed_image = image.copy()

        new_width, new_height = compute_resized_dimensions(
            original_width,
            original_height,
            long_side,
        )

        if (new_width, new_height) != processed_image.size:
            processed_image = processed_image.resize(
                (new_width, new_height),
                resample=get_lanczos_filter(),
            )

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_image.save(
            task.output_path,
            format="JPEG",
            quality=jpeg_quality,
        )
        new_bytes = safe_size(task.output_path)

        return result_from_task(
            task,
            original_width=original_width,
            original_height=original_height,
            new_width=new_width,
            new_height=new_height,
            original_bytes=original_bytes,
            new_bytes=new_bytes,
            status="processed",
        )
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
        return result_from_task(
            task,
            original_bytes=original_bytes,
            status="failed",
            error_message=str(exc),
        )


def write_results_csv(log_csv_path: Path, results: Iterable[ProcessingResult]) -> RunStats:
    stats = RunStats()
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with log_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())
            stats.update(result)

    return stats


def process_tasks(
    tasks: Sequence[FileTask],
    *,
    overwrite: bool,
    long_side: int,
    jpeg_quality: int,
    num_workers: int,
    log_csv_path: Path,
) -> RunStats:
    total_tasks = len(tasks)
    progress_interval = max(1, min(250, total_tasks // 20 or 1))

    def log_progress(index: int) -> None:
        if index % progress_interval == 0 or index == total_tasks:
            logging.info("Progress: %d/%d files processed.", index, total_tasks)

    if num_workers == 1:
        def iter_results() -> Iterable[ProcessingResult]:
            for index, task in enumerate(tasks, start=1):
                yield process_task(
                    task,
                    overwrite=overwrite,
                    long_side=long_side,
                    jpeg_quality=jpeg_quality,
                )
                log_progress(index)

        return write_results_csv(log_csv_path, iter_results())

    def iter_parallel_results() -> Iterable[ProcessingResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map: dict[Future[ProcessingResult], FileTask] = {
                executor.submit(
                    process_task,
                    task,
                    overwrite=overwrite,
                    long_side=long_side,
                    jpeg_quality=jpeg_quality,
                ): task
                for task in tasks
            }

            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive cleanup path.
                    result = result_from_task(
                        task,
                        original_bytes=safe_size(task.source_path),
                        status="failed",
                        error_message=f"Unhandled worker exception: {exc}",
                    )
                yield result
                log_progress(index)

    return write_results_csv(log_csv_path, iter_parallel_results())


def log_summary(
    *,
    stats: RunStats,
    elapsed_seconds: float,
    log_csv_path: Path,
) -> None:
    logging.info("Files selected for processing: %d", stats.files_found)
    logging.info("Processed successfully: %d", stats.processed)
    logging.info("Skipped: %d", stats.skipped)
    logging.info("Failed: %d", stats.failed)
    logging.info("Processing time: %.2f seconds", elapsed_seconds)
    logging.info("Total input size: %s", format_bytes(stats.total_original_bytes))
    logging.info("Total output size: %s", format_bytes(stats.total_output_bytes))

    phase_summary = ", ".join(
        f"{phase}={count}" for phase, count in sorted(stats.phase_counts.items())
    )
    xylomami_summary = ", ".join(
        f"{flag}={count}" for flag, count in sorted(stats.xylomami_counts.items())
    )
    group_summary = ", ".join(
        f"{group}={count}" for group, count in sorted(stats.group_counts.items())
    )

    logging.info("Counts by phase: %s", phase_summary or "n/a")
    logging.info("Counts by in_xylomami_final_100: %s", xylomami_summary or "n/a")
    logging.info("Counts by protocol_group: %s", group_summary or "n/a")
    logging.info("CSV log written to: %s", log_csv_path)


def build_tasks(
    *,
    phase1_root: Path,
    phase2_root: Path,
    output_root: Path,
    corrections: dict[str, str],
    final_species_records: dict[str, FinalSpeciesRecord],
) -> BuildTasksResult:
    phase1_result = scan_phase(
        root=phase1_root,
        phase="Phase1",
        output_root=output_root,
        corrections=corrections,
        final_species_records=final_species_records,
    )
    phase2_result = scan_phase(
        root=phase2_root,
        phase="Phase2",
        output_root=output_root,
        corrections=corrections,
        final_species_records=final_species_records,
    )
    return BuildTasksResult(
        tasks=phase1_result.tasks + phase2_result.tasks,
        scanned_files=phase1_result.scanned_files + phase2_result.scanned_files,
        selected_files=phase1_result.selected_files + phase2_result.selected_files,
        excluded_not_in_xylomami=(
            phase1_result.excluded_not_in_xylomami + phase2_result.excluded_not_in_xylomami
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    start_time = time.perf_counter()

    try:
        args = validate_args(parse_args(argv))
        final_species_records = load_final_species_records(args.xylomami_species_csv)
        corrections = load_taxonomy_corrections(args.taxonomy_correction_csv)
        build_result = build_tasks(
            phase1_root=args.phase1_root,
            phase2_root=args.phase2_root,
            output_root=args.output_root,
            corrections=corrections,
            final_species_records=final_species_records,
        )

        logging.info("Phase 1 root: %s", args.phase1_root)
        logging.info("Phase 2 root: %s", args.phase2_root)
        logging.info("Output root: %s", args.output_root)
        logging.info("XyloMaMi-Bench species CSV: %s", args.xylomami_species_csv)
        logging.info(
            "Taxonomy correction source: %s",
            args.taxonomy_correction_csv or "built-in defaults only",
        )
        logging.info("Files scanned: %d", build_result.scanned_files)
        logging.info("Files selected from XyloMaMi-Bench final 100: %d", build_result.selected_files)
        logging.info(
            "Files excluded because they are not in XyloMaMi-Bench final 100: %d",
            build_result.excluded_not_in_xylomami,
        )

        stats = process_tasks(
            build_result.tasks,
            overwrite=args.overwrite,
            long_side=args.long_side,
            jpeg_quality=args.jpeg_quality,
            num_workers=args.num_workers,
            log_csv_path=args.log_csv,
        )
        elapsed_seconds = time.perf_counter() - start_time
        log_summary(stats=stats, elapsed_seconds=elapsed_seconds, log_csv_path=args.log_csv)

        if stats.failed:
            logging.warning(
                "Processing completed with %d failed files. Inspect the CSV log for details.",
                stats.failed,
            )
        return 0
    except Exception as exc:
        logging.error("Preprocessing failed before completion: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
