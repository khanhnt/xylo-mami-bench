"""Manifest-backed dataset utilities for XyloMaMi-Bench experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.taxonomy import extract_genus, parse_protocol_usage

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_MODES = frozenset({"macro_classification", "micro_classification", "joint_alignment"})
MODALITIES = frozenset({"macro", "micro"})

TransformFn = Callable[[Image.Image], Tensor]


@dataclass(frozen=True)
class LabelMapping:
    """Stable mapping between species names and class indices."""

    species_to_index: dict[str, int]
    index_to_species: tuple[str, ...]
    label_space_name: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.index_to_species)

    def index_for(self, species: str) -> int:
        return self.species_to_index[species]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_space_name": self.label_space_name,
            "species_to_index": dict(self.species_to_index),
            "index_to_species": list(self.index_to_species),
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> LabelMapping:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            species_to_index={str(key): int(value) for key, value in payload["species_to_index"].items()},
            index_to_species=tuple(str(item) for item in payload["index_to_species"]),
            label_space_name=str(payload.get("label_space_name", "")),
        )


@dataclass(frozen=True)
class ManifestSample:
    """Typed in-memory representation of a manifest row."""

    row_index: int
    image_path: Path
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


@dataclass(frozen=True)
class DatasetSummary:
    """Lightweight dataset counts for quick inspection and logging."""

    sample_count: int
    species_count: int
    modality_counts: dict[str, int]
    protocol_group_counts: dict[str, int]


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in DATASET_MODES:
        raise ValueError(
            f"Unsupported dataset mode '{mode}'. Expected one of {sorted(DATASET_MODES)}."
        )
    return normalized


def _normalize_modalities(
    modality: str | Sequence[str] | None,
    *,
    mode: str,
) -> tuple[str, ...]:
    if modality is None:
        if mode == "macro_classification":
            return ("macro",)
        if mode == "micro_classification":
            return ("micro",)
        return ("macro", "micro")

    if isinstance(modality, str):
        normalized = modality.strip().lower()
        if normalized == "both":
            values = ("macro", "micro")
        else:
            values = (normalized,)
    else:
        values = tuple(sorted({item.strip().lower() for item in modality if item.strip()}))

    invalid = [value for value in values if value not in MODALITIES]
    if invalid:
        raise ValueError(f"Unsupported modality filter(s): {invalid}.")
    if mode == "macro_classification" and values != ("macro",):
        raise ValueError("macro_classification mode only supports modality='macro'.")
    if mode == "micro_classification" and values != ("micro",):
        raise ValueError("micro_classification mode only supports modality='micro'.")
    return values


def _normalize_protocol_filter(protocol_filter: str | Sequence[str] | None) -> set[str]:
    if protocol_filter is None:
        return set()
    if isinstance(protocol_filter, str):
        tokens = [protocol_filter]
    else:
        tokens = list(protocol_filter)
    return {token.strip().upper() for token in tokens if token and token.strip()}


def _parse_taxonomy_suggestions(raw_value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw_value.split("|") if item.strip())


def _coerce_label_mapping(
    label_mapping: LabelMapping | Mapping[str, int],
    *,
    label_space_name: str = "",
) -> LabelMapping:
    if isinstance(label_mapping, LabelMapping):
        return label_mapping

    ordered_items = sorted(label_mapping.items(), key=lambda item: item[1])
    index_to_species = tuple(species for species, _ in ordered_items)
    return LabelMapping(
        species_to_index={str(species): int(index) for species, index in ordered_items},
        index_to_species=index_to_species,
        label_space_name=label_space_name,
    )


def _normalize_image_root_override(
    image_root_override: str | Path | Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    if image_root_override is None:
        return {}
    if isinstance(image_root_override, (str, Path)):
        root = Path(image_root_override)
        return {"macro": root, "micro": root}
    overrides: dict[str, Path] = {}
    for modality, root in image_root_override.items():
        normalized_modality = modality.strip().lower()
        if normalized_modality not in MODALITIES:
            raise ValueError(
                f"Unsupported image root override modality '{modality}'. "
                f"Expected one of {sorted(MODALITIES)}."
            )
        overrides[normalized_modality] = Path(root)
    return overrides


def build_label_mapping_from_species(
    species_names: Iterable[str],
    *,
    label_space_name: str = "",
) -> LabelMapping:
    ordered_species = tuple(sorted({species.strip() for species in species_names if species.strip()}))
    species_to_index = {species: index for index, species in enumerate(ordered_species)}
    return LabelMapping(
        species_to_index=species_to_index,
        index_to_species=ordered_species,
        label_space_name=label_space_name,
    )


def build_label_mapping_from_samples(
    samples: Sequence[ManifestSample],
    *,
    label_space_name: str = "",
) -> LabelMapping:
    return build_label_mapping_from_species(
        (sample.species for sample in samples),
        label_space_name=label_space_name,
    )


def build_label_mapping_from_csvs(
    manifest_paths: Sequence[str | Path],
    *,
    mode: str = "joint_alignment",
    modality: str | Sequence[str] | None = None,
    protocol_filter: str | Sequence[str] | None = None,
    label_space_name: str = "",
) -> LabelMapping:
    samples: list[ManifestSample] = []
    for manifest_path in manifest_paths:
        samples.extend(
            load_manifest_samples(
                manifest_path,
                mode=mode,
                modality=modality,
                protocol_filter=protocol_filter,
            )
        )
    return build_label_mapping_from_samples(samples, label_space_name=label_space_name)


def save_label_mapping(path: str | Path, label_mapping: LabelMapping | Mapping[str, int]) -> None:
    """Persist a label mapping JSON file."""

    _coerce_label_mapping(label_mapping).save(path)


def load_label_mapping(path: str | Path) -> LabelMapping:
    """Load a label mapping JSON file."""

    return LabelMapping.load(path)


def load_manifest_samples(
    manifest_path: str | Path,
    *,
    mode: str = "joint_alignment",
    modality: str | Sequence[str] | None = None,
    protocol_filter: str | Sequence[str] | None = None,
) -> list[ManifestSample]:
    """Load typed manifest rows with modality/protocol filtering and validation."""

    resolved_mode = _normalize_mode(mode)
    allowed_modalities = set(_normalize_modalities(modality, mode=resolved_mode))
    allowed_protocols = _normalize_protocol_filter(protocol_filter)
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_file}")

    with manifest_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest '{manifest_file}' is missing a CSV header row.")
        samples: list[ManifestSample] = []
        for row_index, row in enumerate(reader):
            sample_modality = (row.get("modality") or "").strip().lower()
            if sample_modality not in allowed_modalities:
                continue

            protocol_tokens = parse_protocol_usage(row.get("protocol_usage", ""))
            if allowed_protocols and not (protocol_tokens & allowed_protocols):
                continue

            species = (row.get("species") or "").strip()
            if not species:
                raise ValueError(
                    f"Manifest '{manifest_file}' row {row_index + 2} is missing a species value."
                )
            image_path = Path((row.get("image_path") or "").strip())
            image_rel_path = (row.get("image_rel_path") or "").strip()
            if not str(image_path) and not image_rel_path:
                raise ValueError(
                    f"Manifest '{manifest_file}' row {row_index + 2} is missing both "
                    "'image_path' and 'image_rel_path'."
                )
            genus = (row.get("genus") or "").strip() or extract_genus(species)
            samples.append(
                ManifestSample(
                    row_index=row_index,
                    image_path=image_path,
                    image_rel_path=image_rel_path,
                    dataset_name=(row.get("dataset_name") or "").strip(),
                    modality=sample_modality,
                    view=(row.get("view") or "").strip(),
                    species=species,
                    genus=genus,
                    family=(row.get("family") or "").strip(),
                    specimen_id=(row.get("specimen_id") or "").strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    phase=(row.get("phase") or "").strip(),
                    raw_species_name=(row.get("raw_species_name") or "").strip(),
                    normalized_species_name=(row.get("normalized_species_name") or "").strip(),
                    protocol_group=(row.get("protocol_group") or "").strip(),
                    overlap_type=(row.get("overlap_type") or "").strip(),
                    protocol_usage=(row.get("protocol_usage") or "").strip(),
                    matched_species_name=(row.get("matched_species_name") or "").strip(),
                    taxonomy_match_status=(row.get("taxonomy_match_status") or "").strip(),
                    taxonomy_suggestions=_parse_taxonomy_suggestions(
                        row.get("taxonomy_suggestions", "")
                    ),
                )
            )

    if not samples:
        raise ValueError(
            f"No samples were loaded from manifest '{manifest_file}' with mode={resolved_mode} "
            f"and modalities={sorted(allowed_modalities)}."
        )
    return samples


def pil_image_to_tensor(image: Image.Image) -> Tensor:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    image_bytes = rgb_image.tobytes()
    if hasattr(torch, "frombuffer"):
        tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
    else:  # pragma: no cover - compatibility path for older torch builds
        tensor = torch.tensor(bytearray(image_bytes), dtype=torch.uint8)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1).contiguous()
    return tensor.float().div(255.0)


class ManifestDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset backed by XyloMaMi-Bench split CSV manifests."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        mode: str = "joint_alignment",
        modality: str | Sequence[str] | None = None,
        protocol_filter: str | Sequence[str] | None = None,
        transform: TransformFn | Mapping[str, TransformFn] | None = None,
        label_mapping: LabelMapping | Mapping[str, int] | None = None,
        label_space_name: str = "",
        image_root_override: str | Path | Mapping[str, str | Path] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.mode = _normalize_mode(mode)
        self.modalities = _normalize_modalities(modality, mode=self.mode)
        self.protocol_filter = _normalize_protocol_filter(protocol_filter)
        self.samples = load_manifest_samples(
            self.manifest_path,
            mode=self.mode,
            modality=self.modalities,
            protocol_filter=self.protocol_filter,
        )
        self.transform = transform
        self.image_root_override = _normalize_image_root_override(image_root_override)
        self.label_mapping = (
            build_label_mapping_from_samples(self.samples, label_space_name=label_space_name)
            if label_mapping is None
            else _coerce_label_mapping(label_mapping, label_space_name=label_space_name)
        )

        missing_species = sorted(
            {
                sample.species
                for sample in self.samples
                if sample.species not in self.label_mapping.species_to_index
            }
        )
        if missing_species:
            raise ValueError(
                "Label mapping does not cover all dataset species. Missing species: "
                + ", ".join(missing_species)
            )

        self.indices_by_species: dict[str, list[int]] = defaultdict(list)
        self.indices_by_modality: dict[str, list[int]] = defaultdict(list)
        self.indices_by_species_and_modality: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.species_counts: Counter[str] = Counter()
        self.modality_counts: Counter[str] = Counter()
        self.protocol_group_counts: Counter[str] = Counter()

        for index, sample in enumerate(self.samples):
            self.indices_by_species[sample.species].append(index)
            self.indices_by_modality[sample.modality].append(index)
            self.indices_by_species_and_modality[sample.species][sample.modality].append(index)
            self.species_counts[sample.species] += 1
            self.modality_counts[sample.modality] += 1
            if sample.protocol_group:
                self.protocol_group_counts[sample.protocol_group] += 1

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return self.label_mapping.num_classes

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            sample_count=len(self.samples),
            species_count=len(self.indices_by_species),
            modality_counts=dict(sorted(self.modality_counts.items())),
            protocol_group_counts=dict(sorted(self.protocol_group_counts.items())),
        )

    def resolve_transform(self, modality: str) -> TransformFn | None:
        if self.transform is None:
            return None
        if callable(self.transform):
            return self.transform
        if modality not in self.transform:
            raise KeyError(f"No transform configured for modality '{modality}'.")
        return self.transform[modality]

    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

    def resolve_image_path(self, sample: ManifestSample) -> Path:
        override_root = self.image_root_override.get(sample.modality)
        if override_root is not None and sample.image_rel_path:
            return override_root / Path(sample.image_rel_path)
        return sample.image_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        resolved_image_path = self.resolve_image_path(sample)
        image = self._load_image(resolved_image_path)
        transform = self.resolve_transform(sample.modality)
        image_tensor = transform(image) if transform is not None else pil_image_to_tensor(image)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                f"Expected transform to return a torch.Tensor, but received {type(image_tensor)!r}."
            )

        return {
            "image": image_tensor,
            "label": self.label_mapping.index_for(sample.species),
            "species": sample.species,
            "genus": sample.genus,
            "modality": sample.modality,
            "view": sample.view,
            "phase": sample.phase,
            "protocol_group": sample.protocol_group,
            "overlap_type": sample.overlap_type,
            "specimen_id": sample.specimen_id,
            "source_id": sample.source_id,
            "image_path": str(resolved_image_path),
            "manifest_image_path": str(sample.image_path),
            "image_rel_path": sample.image_rel_path,
            "dataset_name": sample.dataset_name,
            "protocol_usage": sample.protocol_usage,
            "raw_species_name": sample.raw_species_name,
            "normalized_species_name": sample.normalized_species_name,
            "matched_species_name": sample.matched_species_name,
            "taxonomy_match_status": sample.taxonomy_match_status,
        }
