"""Taxonomy normalization and manifest-building utilities for XyloMaMi-Bench."""

from __future__ import annotations

import csv
import difflib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MACRO_ROOT = Path(
    os.environ.get(
        "XYLOMAMI_MACRO_ROOT",
        str(REPO_ROOT / "data" / "images" / "macro"),
    )
)
DEFAULT_MACRO_LOG_CSV = Path(
    os.environ.get(
        "XYLOMAMI_MACRO_LOG_CSV",
        str(DEFAULT_MACRO_ROOT / "preprocess_macro_dataset_log.csv"),
    )
)
DEFAULT_MICRO_CONGO_ROOT = Path(
    os.environ.get(
        "XYLOMAMI_MICRO_CONGO_ROOT",
        str(REPO_ROOT / "data" / "images" / "micro"),
    )
)
DEFAULT_XYLOMAMI_SPECIES_CSV = REPO_ROOT / "data" / "xylomami_final_100_species_list.csv"
DEFAULT_OUTPUT_MANIFEST_DIR = REPO_ROOT / "data" / "processed" / "manifests"
DEFAULT_OUTPUT_REPORT_DIR = REPO_ROOT / "data" / "processed" / "reports"

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIES_FOLDER_PATTERN = re.compile(r"^\s*(?P<prefix>\d+)\s*\.\s*(?P<species>.+?)\s*$")
SPECIMEN_TOKEN_PATTERN = re.compile(r"^(?P<token>[A-Za-z]{1,8}\d{2,}[A-Za-z0-9-]*)[_-].*$")
GROUP_SORT_ORDER = {"A": 0, "B": 1, "C": 2}

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

MANIFEST_COLUMNS = [
    "image_path",
    "image_rel_path",
    "dataset_name",
    "modality",
    "view",
    "species",
    "genus",
    "family",
    "specimen_id",
    "source_id",
    "phase",
    "raw_species_name",
    "normalized_species_name",
    "protocol_group",
    "overlap_type",
    "protocol_usage",
    "matched_species_name",
    "taxonomy_match_status",
    "taxonomy_suggestions",
]

OVERLAP_SUMMARY_COLUMNS = [
    "species",
    "genus",
    "family",
    "protocol_group",
    "overlap_type",
    "protocol_usage",
    "macro_image_count",
    "micro_exact_image_count",
    "micro_genus_related_species_count",
    "micro_genus_related_image_count",
    "micro_genus_related_species",
    "macro_present",
    "micro_exact_present",
]

UNCERTAIN_MATCH_COLUMNS = [
    "dataset_name",
    "modality",
    "raw_species_name",
    "normalized_species_name",
    "genus",
    "protocol_group",
    "overlap_type",
    "protocol_usage",
    "taxonomy_match_status",
    "matched_species_name",
    "taxonomy_suggestions",
    "image_count",
]


@dataclass(frozen=True)
class XyloMamiSpeciesRecord:
    species: str
    genus: str
    family: str
    group: str
    overlap_type: str
    protocol_usage: str
    group_order: int


@dataclass(frozen=True)
class TaxonomyMatch:
    raw_species_name: str
    normalized_species_name: str
    genus: str
    family: str
    matched_species_name: str
    protocol_group: str
    overlap_type: str
    protocol_usage: str
    match_status: str
    taxonomy_suggestions: tuple[str, ...]


@dataclass(frozen=True)
class MacroLogEntry:
    phase: str
    raw_species_folder_name: str
    raw_species_name: str
    corrected_species_name: str
    matched_species_name: str
    protocol_group: str
    overlap_type: str
    protocol_usage: str


@dataclass
class MacroLogIndex:
    by_absolute_path: dict[str, MacroLogEntry]
    by_relative_path: dict[str, MacroLogEntry]

    def lookup(self, absolute_path: Path, relative_path: Path) -> MacroLogEntry | None:
        return self.by_absolute_path.get(normalize_path_key(absolute_path)) or self.by_relative_path.get(
            relative_path.as_posix()
        )


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    image_rel_path: str
    dataset_name: str
    modality: str
    view: str
    species: str
    genus: str
    family: str
    specimen_id: str
    source_id: str
    phase: str
    raw_species_name: str
    normalized_species_name: str
    protocol_group: str
    overlap_type: str
    protocol_usage: str
    matched_species_name: str
    taxonomy_match_status: str
    taxonomy_suggestions: tuple[str, ...]

    @property
    def protocol_tokens(self) -> set[str]:
        return parse_protocol_usage(self.protocol_usage)

    def to_row(self) -> dict[str, str]:
        row = {
            "image_path": self.image_path,
            "image_rel_path": self.image_rel_path,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "view": self.view,
            "species": self.species,
            "genus": self.genus,
            "family": self.family,
            "specimen_id": self.specimen_id,
            "source_id": self.source_id,
            "phase": self.phase,
            "raw_species_name": self.raw_species_name,
            "normalized_species_name": self.normalized_species_name,
            "protocol_group": self.protocol_group,
            "overlap_type": self.overlap_type,
            "protocol_usage": self.protocol_usage,
            "matched_species_name": self.matched_species_name,
            "taxonomy_match_status": self.taxonomy_match_status,
            "taxonomy_suggestions": "|".join(self.taxonomy_suggestions),
        }
        return row


@dataclass(frozen=True)
class XyloMamiIndex:
    records: tuple[XyloMamiSpeciesRecord, ...]
    by_species_key: dict[str, XyloMamiSpeciesRecord]
    genus_to_records: dict[str, tuple[XyloMamiSpeciesRecord, ...]]

    @property
    def species_names(self) -> list[str]:
        return [record.species for record in self.records]

    @property
    def species_set(self) -> set[str]:
        return {record.species for record in self.records}

    @property
    def genus_set(self) -> set[str]:
        return {record.genus for record in self.records}

    def lookup_species(self, value: str) -> XyloMamiSpeciesRecord | None:
        return self.by_species_key.get(normalized_key(value))

    def lookup_genus_records(self, genus: str) -> tuple[XyloMamiSpeciesRecord, ...]:
        return self.genus_to_records.get(normalized_key(genus), ())

    def family_for_genus(self, genus: str) -> str:
        families = {record.family for record in self.lookup_genus_records(genus) if record.family}
        return next(iter(families)) if len(families) == 1 else ""


@dataclass
class BuildArtifacts:
    macro_records: list[ManifestRecord]
    micro_records: list[ManifestRecord]
    overlap_rows: list[dict[str, str | int]]
    uncertain_rows: list[dict[str, str | int]]
    summary: dict[str, Any]

    @property
    def combined_records(self) -> list[ManifestRecord]:
        return self.macro_records + self.micro_records


def normalize_path_key(path: Path) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def sanitize_raw_species_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).replace("_", " ")
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.strip(" .")


def sanitize_metadata_value(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def canonicalize_species_name(value: str) -> str:
    cleaned = sanitize_raw_species_name(value)
    if not cleaned:
        return ""
    parts = cleaned.split(" ")
    genus = parts[0][:1].upper() + parts[0][1:].lower()
    remainder = [part.lower() for part in parts[1:]]
    return " ".join([genus, *remainder]).strip()


def normalized_key(value: str) -> str:
    return canonicalize_species_name(value).lower()


def extract_genus(species_name: str) -> str:
    if not species_name:
        return ""
    return canonicalize_species_name(species_name).split(" ", 1)[0]


def parse_protocol_usage(protocol_usage: str) -> set[str]:
    return {
        token.strip()
        for token in protocol_usage.split(",")
        if token.strip()
    }


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)


def load_taxonomy_corrections(
    correction_csv: Path | None,
    *,
    include_defaults: bool = True,
) -> dict[str, str]:
    corrections: dict[str, str] = {}
    if include_defaults:
        for raw_name, corrected_name in DEFAULT_TAXONOMY_CORRECTIONS.items():
            corrections[normalized_key(raw_name)] = canonicalize_species_name(corrected_name)

    if correction_csv is None:
        return corrections
    if not correction_csv.exists():
        raise FileNotFoundError(f"Taxonomy correction CSV not found: {correction_csv}")

    with correction_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return corrections

        source_column = next(
            (column for column in TAXONOMY_CORRECTION_SOURCE_COLUMNS if column in reader.fieldnames),
            None,
        )
        target_column = next(
            (column for column in TAXONOMY_CORRECTION_TARGET_COLUMNS if column in reader.fieldnames),
            None,
        )
        if source_column is None or target_column is None:
            raise ValueError(
                "Taxonomy correction CSV must contain one source column and one target column."
            )

        for row in reader:
            raw_name = sanitize_raw_species_name(row.get(source_column, ""))
            corrected_name = canonicalize_species_name(row.get(target_column, ""))
            if raw_name and corrected_name:
                corrections[normalized_key(raw_name)] = corrected_name
    return corrections


def load_xylomami_index(path: Path) -> XyloMamiIndex:
    records: list[XyloMamiSpeciesRecord] = []
    by_species_key: dict[str, XyloMamiSpeciesRecord] = {}
    genus_to_records: defaultdict[str, list[XyloMamiSpeciesRecord]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            species = canonicalize_species_name(row.get("species", ""))
            if not species:
                continue
            record = XyloMamiSpeciesRecord(
                species=species,
                genus=canonicalize_species_name(row.get("genus", "") or extract_genus(species)),
                family=sanitize_metadata_value(row.get("family", "")),
                group=sanitize_metadata_value(row.get("group", "")),
                overlap_type=sanitize_metadata_value(row.get("overlap_type", "")),
                protocol_usage=sanitize_metadata_value(row.get("protocol_usage", "")),
                group_order=int((row.get("group_order") or "0").strip() or "0"),
            )
            records.append(record)
            by_species_key[normalized_key(record.species)] = record
            genus_to_records[normalized_key(record.genus)].append(record)

    records.sort(
        key=lambda record: (
            GROUP_SORT_ORDER.get(record.group, 999),
            record.group_order,
            record.species,
        )
    )
    frozen_genus_to_records = {
        genus_key: tuple(sorted(items, key=lambda record: (record.group_order, record.species)))
        for genus_key, items in genus_to_records.items()
    }
    return XyloMamiIndex(
        records=tuple(records),
        by_species_key=by_species_key,
        genus_to_records=frozen_genus_to_records,
    )


def load_macro_log_index(log_csv: Path | None, macro_root: Path) -> MacroLogIndex:
    if log_csv is None or not log_csv.exists():
        return MacroLogIndex(by_absolute_path={}, by_relative_path={})

    by_absolute_path: dict[str, MacroLogEntry] = {}
    by_relative_path: dict[str, MacroLogEntry] = {}
    macro_root_key = normalize_path_key(macro_root)

    with log_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            processed_path_text = (row.get("processed_path") or "").strip()
            if not processed_path_text:
                continue
            entry = MacroLogEntry(
                phase=sanitize_metadata_value(row.get("phase", "")),
                raw_species_folder_name=sanitize_raw_species_name(
                    row.get("raw_species_folder_name", "")
                ),
                raw_species_name=sanitize_raw_species_name(row.get("raw_species_name", "")),
                corrected_species_name=canonicalize_species_name(
                    row.get("corrected_species_name", "")
                ),
                matched_species_name=canonicalize_species_name(
                    row.get("matched_species_name", "")
                ),
                protocol_group=sanitize_metadata_value(row.get("protocol_group", "")),
                overlap_type=sanitize_metadata_value(row.get("overlap_type", "")),
                protocol_usage=sanitize_metadata_value(row.get("protocol_usage", "")),
            )
            processed_path = Path(processed_path_text)
            by_absolute_path[normalize_path_key(processed_path)] = entry

            processed_key = normalize_path_key(processed_path)
            if processed_key.startswith(macro_root_key + os.sep):
                relative_key = Path(processed_key).relative_to(macro_root_key).as_posix()
                by_relative_path[relative_key] = entry
                continue
            try:
                relative_key = processed_path.relative_to(macro_root).as_posix()
            except ValueError:
                continue
            by_relative_path[relative_key] = entry

    return MacroLogIndex(
        by_absolute_path=by_absolute_path,
        by_relative_path=by_relative_path,
    )


def collect_image_paths(root: Path) -> list[Path]:
    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda item: item.as_posix())


def parse_macro_species_folder(folder_name: str) -> tuple[str, str] | None:
    match = SPECIES_FOLDER_PATTERN.match(folder_name)
    if match is None:
        return None
    species = sanitize_raw_species_name(match.group("species"))
    if not species:
        return None
    return folder_name, species


def find_macro_species_folder(relative_path: Path) -> tuple[str, str] | None:
    directory_parts = relative_path.parts[:-1]
    for part in directory_parts:
        parsed = parse_macro_species_folder(part)
        if parsed is not None:
            return parsed
    return None


def derive_phase(relative_path: Path, fallback_phase: str = "") -> str:
    if relative_path.parts:
        first_part = relative_path.parts[0]
        if first_part in {"Phase1", "Phase2"}:
            return first_part
    return fallback_phase


def derive_specimen_id(relative_path: Path, species_folder_name: str) -> str:
    parts = list(relative_path.parts)
    species_index = None
    for index, part in enumerate(parts[:-1]):
        if part == species_folder_name:
            species_index = index
            break

    if species_index is not None:
        specimen_parts = parts[species_index + 1 : -1]
        if specimen_parts:
            return "/".join(specimen_parts)

    stem = relative_path.stem
    match = SPECIMEN_TOKEN_PATTERN.match(stem)
    if match is not None:
        return match.group("token")
    return stem


def derive_source_id(dataset_name: str, relative_path: Path) -> str:
    return f"{dataset_name}:{relative_path.with_suffix('').as_posix()}"


def resolve_output_image_path(source_root: Path, output_root: Path | None, relative_path: Path) -> str:
    resolved_root = output_root if output_root is not None else source_root
    return str(resolved_root / relative_path)


def suggest_species_candidates(species_name: str, xylomami_index: XyloMamiIndex) -> tuple[str, ...]:
    canonical_name = canonicalize_species_name(species_name)
    suggestions: list[str] = []

    for candidate in difflib.get_close_matches(
        canonical_name,
        xylomami_index.species_names,
        n=3,
        cutoff=0.88,
    ):
        if candidate not in suggestions:
            suggestions.append(candidate)

    genus = extract_genus(canonical_name)
    genus_candidates = [record.species for record in xylomami_index.lookup_genus_records(genus)]
    for candidate in difflib.get_close_matches(
        canonical_name,
        genus_candidates,
        n=3,
        cutoff=0.72,
    ):
        if candidate not in suggestions:
            suggestions.append(candidate)
    return tuple(suggestions[:3])


def resolve_taxonomy_match(
    raw_species_name: str,
    *,
    xylomami_index: XyloMamiIndex,
    corrections: Mapping[str, str],
    explicit_normalized_name: str = "",
    allow_genus_only_inference: bool = False,
) -> TaxonomyMatch:
    raw_species = sanitize_raw_species_name(raw_species_name)
    raw_key = normalized_key(raw_species)

    if explicit_normalized_name:
        normalized_species = canonicalize_species_name(explicit_normalized_name)
    else:
        normalized_species = corrections.get(raw_key, canonicalize_species_name(raw_species))

    normalized_species = canonicalize_species_name(normalized_species)
    matched_record = xylomami_index.lookup_species(normalized_species)
    if matched_record is not None:
        match_status = (
            "exact_species_match"
            if normalized_key(normalized_species) == raw_key
            else "exact_species_match_via_normalization"
        )
        return TaxonomyMatch(
            raw_species_name=raw_species,
            normalized_species_name=matched_record.species,
            genus=matched_record.genus,
            family=matched_record.family,
            matched_species_name=matched_record.species,
            protocol_group=matched_record.group,
            overlap_type=matched_record.overlap_type,
            protocol_usage=matched_record.protocol_usage,
            match_status=match_status,
            taxonomy_suggestions=(),
        )

    genus = extract_genus(normalized_species)
    genus_records = xylomami_index.lookup_genus_records(genus)
    if allow_genus_only_inference and genus_records:
        taxonomy_suggestions = tuple(record.species for record in genus_records[:3])
        return TaxonomyMatch(
            raw_species_name=raw_species,
            normalized_species_name=normalized_species,
            genus=genus,
            family=xylomami_index.family_for_genus(genus),
            matched_species_name="",
            protocol_group="B",
            overlap_type="genus_overlap_only",
            protocol_usage="P2,P3",
            match_status="genus_match_only",
            taxonomy_suggestions=taxonomy_suggestions,
        )

    suggestions = suggest_species_candidates(normalized_species, xylomami_index)
    return TaxonomyMatch(
        raw_species_name=raw_species,
        normalized_species_name=normalized_species,
        genus=genus,
        family=xylomami_index.family_for_genus(genus),
        matched_species_name="",
        protocol_group="",
        overlap_type="",
        protocol_usage="",
        match_status="unmatched_with_close_candidates" if suggestions else "unmatched",
        taxonomy_suggestions=suggestions,
    )


def build_macro_manifest(
    macro_root: Path,
    *,
    xylomami_index: XyloMamiIndex,
    corrections: Mapping[str, str],
    macro_log_csv: Path | None = None,
    output_image_root: Path | None = None,
) -> list[ManifestRecord]:
    log_index = load_macro_log_index(macro_log_csv, macro_root)
    records: list[ManifestRecord] = []

    for image_path in collect_image_paths(macro_root):
        relative_path = image_path.relative_to(macro_root)
        log_entry = log_index.lookup(image_path, relative_path)
        parsed_folder = find_macro_species_folder(relative_path)
        raw_species_folder_name = parsed_folder[0] if parsed_folder is not None else ""
        parsed_raw_species_name = parsed_folder[1] if parsed_folder is not None else ""

        raw_species_name = (
            log_entry.raw_species_name
            if log_entry is not None and log_entry.raw_species_name
            else parsed_raw_species_name
        )
        if not raw_species_name:
            raw_species_name = sanitize_raw_species_name(relative_path.parent.name)

        explicit_normalized_name = (
            log_entry.corrected_species_name
            if log_entry is not None and log_entry.corrected_species_name
            else ""
        )
        match = resolve_taxonomy_match(
            raw_species_name,
            xylomami_index=xylomami_index,
            corrections=corrections,
            explicit_normalized_name=explicit_normalized_name,
            allow_genus_only_inference=False,
        )

        species_folder_label = (
            raw_species_folder_name
            or (log_entry.raw_species_folder_name if log_entry is not None else "")
            or relative_path.parent.name
        )

        records.append(
            ManifestRecord(
                image_path=resolve_output_image_path(macro_root, output_image_root, relative_path),
                image_rel_path=relative_path.as_posix(),
                dataset_name="xylomami_macro",
                modality="macro",
                view="cross_section_end_grain",
                species=match.normalized_species_name,
                genus=match.genus,
                family=match.family,
                specimen_id=derive_specimen_id(relative_path, species_folder_label),
                source_id=derive_source_id("xylomami_macro", relative_path),
                phase=derive_phase(relative_path, fallback_phase=log_entry.phase if log_entry else ""),
                raw_species_name=match.raw_species_name,
                normalized_species_name=match.normalized_species_name,
                protocol_group=match.protocol_group,
                overlap_type=match.overlap_type,
                protocol_usage=match.protocol_usage,
                matched_species_name=match.matched_species_name,
                taxonomy_match_status=match.match_status,
                taxonomy_suggestions=match.taxonomy_suggestions,
            )
        )

    return records


def build_micro_manifest(
    micro_root: Path,
    *,
    xylomami_index: XyloMamiIndex,
    corrections: Mapping[str, str],
    output_image_root: Path | None = None,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []

    for image_path in collect_image_paths(micro_root):
        relative_path = image_path.relative_to(micro_root)
        if not relative_path.parts:
            continue
        species_folder_name = relative_path.parts[0]
        raw_species_name = sanitize_raw_species_name(species_folder_name)
        match = resolve_taxonomy_match(
            raw_species_name,
            xylomami_index=xylomami_index,
            corrections=corrections,
            allow_genus_only_inference=True,
        )
        records.append(
            ManifestRecord(
                image_path=resolve_output_image_path(micro_root, output_image_root, relative_path),
                image_rel_path=relative_path.as_posix(),
                dataset_name="micro_congo",
                modality="micro",
                view="transversal_semi_thin_section",
                species=match.normalized_species_name,
                genus=match.genus,
                family=match.family,
                specimen_id=derive_specimen_id(relative_path, species_folder_name),
                source_id=derive_source_id("micro_congo", relative_path),
                phase="",
                raw_species_name=match.raw_species_name,
                normalized_species_name=match.normalized_species_name,
                protocol_group=match.protocol_group,
                overlap_type=match.overlap_type,
                protocol_usage=match.protocol_usage,
                matched_species_name=match.matched_species_name,
                taxonomy_match_status=match.match_status,
                taxonomy_suggestions=match.taxonomy_suggestions,
            )
        )

    return records


def sort_manifest_records(records: Sequence[ManifestRecord]) -> list[ManifestRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.dataset_name,
            record.modality,
            record.phase,
            record.protocol_group,
            record.species,
            record.image_path,
        ),
    )


def build_candidate_manifest(records: Sequence[ManifestRecord], protocol: str) -> list[ManifestRecord]:
    return [
        record
        for record in sort_manifest_records(records)
        if protocol in record.protocol_tokens
    ]


def build_uncertain_rows(records: Sequence[ManifestRecord]) -> list[dict[str, str | int]]:
    grouped_counts: Counter[tuple[str, ...]] = Counter()
    metadata: dict[tuple[str, ...], dict[str, str]] = {}

    for record in records:
        key = (
            record.dataset_name,
            record.modality,
            record.raw_species_name,
            record.normalized_species_name,
            record.genus,
            record.protocol_group,
            record.overlap_type,
            record.protocol_usage,
            record.taxonomy_match_status,
            record.matched_species_name,
            "|".join(record.taxonomy_suggestions),
        )
        grouped_counts[key] += 1
        metadata[key] = {
            "dataset_name": record.dataset_name,
            "modality": record.modality,
            "raw_species_name": record.raw_species_name,
            "normalized_species_name": record.normalized_species_name,
            "genus": record.genus,
            "protocol_group": record.protocol_group,
            "overlap_type": record.overlap_type,
            "protocol_usage": record.protocol_usage,
            "taxonomy_match_status": record.taxonomy_match_status,
            "matched_species_name": record.matched_species_name,
            "taxonomy_suggestions": "|".join(record.taxonomy_suggestions),
        }

    rows: list[dict[str, str | int]] = []
    for key, count in grouped_counts.items():
        row = metadata[key]
        status = row["taxonomy_match_status"]
        suggestions = row["taxonomy_suggestions"]
        if status not in {"genus_match_only", "unmatched_with_close_candidates"} and not suggestions:
            continue
        rows.append({**row, "image_count": count})
    rows.sort(
        key=lambda row: (
            str(row["dataset_name"]),
            str(row["taxonomy_match_status"]),
            str(row["normalized_species_name"]),
        )
    )
    return rows


def build_overlap_rows(
    macro_records: Sequence[ManifestRecord],
    micro_records: Sequence[ManifestRecord],
    xylomami_index: XyloMamiIndex,
) -> list[dict[str, str | int]]:
    macro_counts = Counter(
        record.matched_species_name
        for record in macro_records
        if record.matched_species_name
    )
    micro_exact_counts = Counter(
        record.matched_species_name
        for record in micro_records
        if record.matched_species_name
    )
    micro_genus_image_counts = Counter(
        record.genus
        for record in micro_records
        if record.taxonomy_match_status == "genus_match_only"
    )
    micro_genus_species: defaultdict[str, set[str]] = defaultdict(set)
    for record in micro_records:
        if record.taxonomy_match_status == "genus_match_only":
            micro_genus_species[record.genus].add(record.species)

    rows: list[dict[str, str | int]] = []
    for species_record in xylomami_index.records:
        genus_related_species = sorted(micro_genus_species.get(species_record.genus, set()))
        rows.append(
            {
                "species": species_record.species,
                "genus": species_record.genus,
                "family": species_record.family,
                "protocol_group": species_record.group,
                "overlap_type": species_record.overlap_type,
                "protocol_usage": species_record.protocol_usage,
                "macro_image_count": macro_counts.get(species_record.species, 0),
                "micro_exact_image_count": micro_exact_counts.get(species_record.species, 0),
                "micro_genus_related_species_count": len(genus_related_species),
                "micro_genus_related_image_count": micro_genus_image_counts.get(
                    species_record.genus, 0
                ),
                "micro_genus_related_species": "|".join(genus_related_species),
                "macro_present": 1 if macro_counts.get(species_record.species, 0) else 0,
                "micro_exact_present": 1 if micro_exact_counts.get(species_record.species, 0) else 0,
            }
        )
    return rows


def summarize_manifest_records(
    macro_records: Sequence[ManifestRecord],
    micro_records: Sequence[ManifestRecord],
    xylomami_index: XyloMamiIndex,
) -> dict[str, Any]:
    macro_raw_species = sorted({record.raw_species_name for record in macro_records})
    macro_normalized_species = sorted({record.normalized_species_name for record in macro_records})
    micro_raw_species = sorted({record.raw_species_name for record in micro_records})
    micro_normalized_species = sorted({record.normalized_species_name for record in micro_records})

    macro_exact_species = sorted(
        {
            record.matched_species_name
            for record in macro_records
            if record.matched_species_name
        }
    )
    micro_exact_species = sorted(
        {
            record.matched_species_name
            for record in micro_records
            if record.matched_species_name
        }
    )
    micro_genus_only_species = sorted(
        {
            record.normalized_species_name
            for record in micro_records
            if record.taxonomy_match_status == "genus_match_only"
        }
    )

    macro_group_image_counts = Counter(
        record.protocol_group
        for record in macro_records
        if record.protocol_group
    )

    macro_extra_species = sorted(
        {
            record.normalized_species_name
            for record in macro_records
            if not record.matched_species_name
        }
    )
    micro_extra_species = sorted(
        {
            record.normalized_species_name
            for record in micro_records
            if not record.protocol_usage
        }
    )
    missing_from_macro = sorted(xylomami_index.species_set - set(macro_exact_species))
    missing_from_micro_exact = sorted(xylomami_index.species_set - set(micro_exact_species))
    matched_to_final100 = sorted(set(macro_exact_species) | set(micro_exact_species))

    combined_records = list(macro_records) + list(micro_records)
    candidate_counts = {
        "P1": sum(1 for record in combined_records if "P1" in record.protocol_tokens),
        "P2": sum(1 for record in combined_records if "P2" in record.protocol_tokens),
        "P3": sum(1 for record in combined_records if "P3" in record.protocol_tokens),
    }

    summary = {
        "macro": {
            "image_count": len(macro_records),
            "species_before_normalization": len(macro_raw_species),
            "species_after_normalization": len(macro_normalized_species),
            "species_matched_to_final100": len(macro_exact_species),
            "group_image_counts": dict(sorted(macro_group_image_counts.items())),
            "extra_species_not_used_count": len(macro_extra_species),
            "extra_species_not_used": macro_extra_species,
            "missing_final100_species_count": len(missing_from_macro),
            "missing_final100_species": missing_from_macro,
        },
        "micro": {
            "image_count": len(micro_records),
            "species_before_normalization": len(micro_raw_species),
            "species_after_normalization": len(micro_normalized_species),
            "exact_species_matched_to_final100": len(micro_exact_species),
            "genus_only_species_count": len(micro_genus_only_species),
            "genus_only_species": micro_genus_only_species,
            "extra_species_not_used_count": len(micro_extra_species),
            "extra_species_not_used": micro_extra_species,
            "missing_final100_species_count_exact_only": len(missing_from_micro_exact),
            "missing_final100_species_exact_only": missing_from_micro_exact,
        },
        "xylomami": {
            "final_species_count": len(xylomami_index.records),
            "species_matched_to_final100_count": len(matched_to_final100),
            "species_matched_to_final100": matched_to_final100,
            "missing_final_list_species_count_after_macro_match": len(missing_from_macro),
            "missing_final_list_species_after_macro_match": missing_from_macro,
        },
        "protocol_candidate_image_counts": candidate_counts,
    }
    return summary


def build_artifacts(
    *,
    macro_root: Path,
    micro_root: Path,
    xylomami_species_csv: Path,
    correction_csv: Path | None = None,
    macro_log_csv: Path | None = None,
    macro_output_image_root: Path | None = None,
    micro_output_image_root: Path | None = None,
) -> BuildArtifacts:
    xylomami_index = load_xylomami_index(xylomami_species_csv)
    corrections = load_taxonomy_corrections(correction_csv)
    macro_records = sort_manifest_records(
        build_macro_manifest(
            macro_root,
            xylomami_index=xylomami_index,
            corrections=corrections,
            macro_log_csv=macro_log_csv,
            output_image_root=macro_output_image_root,
        )
    )
    micro_records = sort_manifest_records(
        build_micro_manifest(
            micro_root,
            xylomami_index=xylomami_index,
            corrections=corrections,
            output_image_root=micro_output_image_root,
        )
    )
    overlap_rows = build_overlap_rows(macro_records, micro_records, xylomami_index)
    uncertain_rows = build_uncertain_rows(macro_records + micro_records)
    summary = summarize_manifest_records(macro_records, micro_records, xylomami_index)
    return BuildArtifacts(
        macro_records=macro_records,
        micro_records=micro_records,
        overlap_rows=overlap_rows,
        uncertain_rows=uncertain_rows,
        summary=summary,
    )


def write_manifest_bundle(
    artifacts: BuildArtifacts,
    *,
    output_manifest_dir: Path,
    output_report_dir: Path,
) -> dict[str, Path]:
    ensure_directory(output_manifest_dir)
    ensure_directory(output_report_dir)

    macro_manifest_path = output_manifest_dir / "macro_all.csv"
    micro_manifest_path = output_manifest_dir / "micro_congo_all.csv"
    p1_manifest_path = output_manifest_dir / "xylomami_p1_candidates.csv"
    p2_manifest_path = output_manifest_dir / "xylomami_p2_candidates.csv"
    p3_manifest_path = output_manifest_dir / "xylomami_p3_candidates.csv"

    write_csv(macro_manifest_path, MANIFEST_COLUMNS, (record.to_row() for record in artifacts.macro_records))
    write_csv(micro_manifest_path, MANIFEST_COLUMNS, (record.to_row() for record in artifacts.micro_records))
    write_csv(
        p1_manifest_path,
        MANIFEST_COLUMNS,
        (record.to_row() for record in build_candidate_manifest(artifacts.combined_records, "P1")),
    )
    write_csv(
        p2_manifest_path,
        MANIFEST_COLUMNS,
        (record.to_row() for record in build_candidate_manifest(artifacts.combined_records, "P2")),
    )
    write_csv(
        p3_manifest_path,
        MANIFEST_COLUMNS,
        (record.to_row() for record in build_candidate_manifest(artifacts.combined_records, "P3")),
    )

    taxonomy_summary_path = output_report_dir / "taxonomy_summary.json"
    overlap_summary_path = output_report_dir / "overlap_summary.csv"
    uncertain_matches_path = output_report_dir / "uncertain_taxonomy_matches.csv"

    write_json(taxonomy_summary_path, artifacts.summary)
    write_csv(overlap_summary_path, OVERLAP_SUMMARY_COLUMNS, artifacts.overlap_rows)
    write_csv(uncertain_matches_path, UNCERTAIN_MATCH_COLUMNS, artifacts.uncertain_rows)

    return {
        "macro_manifest": macro_manifest_path,
        "micro_manifest": micro_manifest_path,
        "xylomami_p1_candidates": p1_manifest_path,
        "xylomami_p2_candidates": p2_manifest_path,
        "xylomami_p3_candidates": p3_manifest_path,
        "taxonomy_summary": taxonomy_summary_path,
        "overlap_summary": overlap_summary_path,
        "uncertain_taxonomy_matches": uncertain_matches_path,
    }


def write_report_bundle(
    artifacts: BuildArtifacts,
    *,
    output_report_dir: Path,
) -> dict[str, Path]:
    ensure_directory(output_report_dir)
    taxonomy_summary_path = output_report_dir / "taxonomy_summary.json"
    overlap_summary_path = output_report_dir / "overlap_summary.csv"
    uncertain_matches_path = output_report_dir / "uncertain_taxonomy_matches.csv"

    write_json(taxonomy_summary_path, artifacts.summary)
    write_csv(overlap_summary_path, OVERLAP_SUMMARY_COLUMNS, artifacts.overlap_rows)
    write_csv(uncertain_matches_path, UNCERTAIN_MATCH_COLUMNS, artifacts.uncertain_rows)
    return {
        "taxonomy_summary": taxonomy_summary_path,
        "overlap_summary": overlap_summary_path,
        "uncertain_taxonomy_matches": uncertain_matches_path,
    }


def print_terminal_summary(artifacts: BuildArtifacts) -> None:
    summary = artifacts.summary
    macro = summary["macro"]
    micro = summary["micro"]
    xylomami = summary["xylomami"]

    print("Taxonomy Summary")
    print(f"  Macro species before normalization: {macro['species_before_normalization']}")
    print(f"  Macro species after normalization:  {macro['species_after_normalization']}")
    print(f"  Micro species found:               {micro['species_before_normalization']}")
    print(
        f"  Species matched to final100:       {xylomami['species_matched_to_final100_count']}"
    )
    print(
        f"  Missing final-list species:        {xylomami['missing_final_list_species_count_after_macro_match']}"
    )
    print(
        f"  Extra macro species not used:      {macro['extra_species_not_used_count']}"
    )
    print(
        f"  Extra micro species not used:      {micro['extra_species_not_used_count']}"
    )

    group_counts = macro["group_image_counts"]
    print("  Macro group image counts:")
    print(f"    A={group_counts.get('A', 0)}")
    print(f"    B={group_counts.get('B', 0)}")
    print(f"    C={group_counts.get('C', 0)}")

    missing_species = xylomami["missing_final_list_species_after_macro_match"]
    print("  Missing final-list species names:")
    print(f"    {', '.join(missing_species) if missing_species else 'None'}")

    extra_macro_species = macro["extra_species_not_used"]
    print("  Extra macro species not used:")
    print(f"    {', '.join(extra_macro_species) if extra_macro_species else 'None'}")

    extra_micro_species = micro["extra_species_not_used"]
    print("  Extra micro species not used:")
    print(f"    {', '.join(extra_micro_species) if extra_micro_species else 'None'}")

    print("Per-species XyloMaMi-Bench overlap counts (macro vs micro)")
    print("species\tgroup\tmacro_images\tmicro_exact_images\tmicro_genus_related_images")
    for row in artifacts.overlap_rows:
        print(
            f"{row['species']}\t{row['protocol_group']}\t{row['macro_image_count']}\t"
            f"{row['micro_exact_image_count']}\t{row['micro_genus_related_image_count']}"
        )


def build_and_write_manifests(
    *,
    macro_root: Path,
    micro_root: Path,
    xylomami_species_csv: Path,
    correction_csv: Path | None,
    macro_log_csv: Path | None,
    output_manifest_dir: Path,
    output_report_dir: Path,
    macro_output_image_root: Path | None = None,
    micro_output_image_root: Path | None = None,
) -> tuple[BuildArtifacts, dict[str, Path]]:
    artifacts = build_artifacts(
        macro_root=macro_root,
        micro_root=micro_root,
        xylomami_species_csv=xylomami_species_csv,
        correction_csv=correction_csv,
        macro_log_csv=macro_log_csv,
        macro_output_image_root=macro_output_image_root,
        micro_output_image_root=micro_output_image_root,
    )
    outputs = write_manifest_bundle(
        artifacts,
        output_manifest_dir=output_manifest_dir,
        output_report_dir=output_report_dir,
    )
    return artifacts, outputs


def build_and_write_reports(
    *,
    macro_root: Path,
    micro_root: Path,
    xylomami_species_csv: Path,
    correction_csv: Path | None,
    macro_log_csv: Path | None,
    output_report_dir: Path,
    macro_output_image_root: Path | None = None,
    micro_output_image_root: Path | None = None,
) -> tuple[BuildArtifacts, dict[str, Path]]:
    artifacts = build_artifacts(
        macro_root=macro_root,
        micro_root=micro_root,
        xylomami_species_csv=xylomami_species_csv,
        correction_csv=correction_csv,
        macro_log_csv=macro_log_csv,
        macro_output_image_root=macro_output_image_root,
        micro_output_image_root=micro_output_image_root,
    )
    outputs = write_report_bundle(
        artifacts,
        output_report_dir=output_report_dir,
    )
    return artifacts, outputs
