#!/usr/bin/env python3
"""Generate deterministic grouped train/val/test splits for XyloMaMi-Bench protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.taxonomy import (  # noqa: E402
    DEFAULT_OUTPUT_MANIFEST_DIR,
    DEFAULT_OUTPUT_REPORT_DIR,
    ensure_directory,
)

DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "processed" / "splits"
SPLIT_NAMES = ("train", "val", "test")
PROTOCOL_FILES = {
    "p1": "xylomami_p1_candidates.csv",
    "p2": "xylomami_p2_candidates.csv",
    "p3": "xylomami_p3_candidates.csv",
}
INVALID_GROUP_VALUES = {"", "na", "n/a", "none", "null", "nan", "missing", "unknown"}
REPORT_CLASS_COUNT_COLUMNS = [
    "protocol",
    "split",
    "species",
    "protocol_group",
    "total_samples",
    "groups_in_split",
    "macro_samples",
    "micro_samples",
    "phase1_macro_samples",
    "phase2_macro_samples",
    "present_in_split",
]


@dataclass(frozen=True)
class ManifestRecord:
    row_index: int
    row: dict[str, str]
    species: str
    modality: str
    phase: str
    protocol_group: str
    specimen_id: str
    source_id: str
    group_token: str
    group_key: str


@dataclass
class GroupUnit:
    key: str
    species: str
    modality: str
    protocol_group: str
    rows: list[ManifestRecord] = field(default_factory=list)
    phase_counts: Counter[str] = field(default_factory=Counter)
    size: int = 0

    def add_record(self, record: ManifestRecord) -> None:
        self.rows.append(record)
        self.size += 1
        if record.phase:
            self.phase_counts[record.phase] += 1


@dataclass
class ProtocolSplitArtifacts:
    protocol: str
    fieldnames: list[str]
    split_rows: dict[str, list[dict[str, str]]]
    summary: dict[str, Any]
    class_count_rows: list[dict[str, Any]]


@dataclass
class AssignmentState:
    split_groups: dict[str, list[GroupUnit]]
    total_assigned: Counter[str] = field(default_factory=Counter)
    modality_assigned: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    phase_assigned: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    species_presence: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    species_count_per_split: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic grouped train/val/test splits for XyloMaMi-Bench."
    )
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST_DIR,
        help="Directory containing xylomami_p1/p2/p3 candidate manifests.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory where split CSVs will be written.",
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_DIR,
        help="Directory where split reports will be written.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for tie-breaking.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive.")
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, but received {total:.6f}."
        )
    return ratios


def stable_hash_int(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(f"{seed}|{'|'.join(parts)}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def normalize_optional_id(value: str) -> str:
    normalized = (value or "").strip()
    if normalized.lower() in INVALID_GROUP_VALUES:
        return ""
    return normalized


def read_manifest(manifest_path: Path) -> tuple[list[str], list[ManifestRecord]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        records: list[ManifestRecord] = []
        for row_index, row in enumerate(reader):
            species = (row.get("species") or "").strip()
            modality = (row.get("modality") or "").strip()
            phase = (row.get("phase") or "").strip()
            protocol_group = (row.get("protocol_group") or "").strip()
            specimen_id = normalize_optional_id(row.get("specimen_id", ""))
            source_id = normalize_optional_id(row.get("source_id", ""))
            group_token = specimen_id if specimen_id else (source_id or row.get("image_path", ""))
            group_kind = "specimen" if specimen_id else "source"
            group_key = (
                f"{row.get('dataset_name', '').strip()}|{modality}|{species}|"
                f"{group_kind}:{group_token}"
            )
            records.append(
                ManifestRecord(
                    row_index=row_index,
                    row=row,
                    species=species,
                    modality=modality,
                    phase=phase,
                    protocol_group=protocol_group,
                    specimen_id=specimen_id,
                    source_id=source_id,
                    group_token=group_token,
                    group_key=group_key,
                )
            )
    return fieldnames, records


def build_groups(records: Sequence[ManifestRecord]) -> dict[str, list[GroupUnit]]:
    groups_by_key: dict[str, GroupUnit] = {}
    for record in records:
        group = groups_by_key.get(record.group_key)
        if group is None:
            group = GroupUnit(
                key=record.group_key,
                species=record.species,
                modality=record.modality,
                protocol_group=record.protocol_group,
            )
            groups_by_key[record.group_key] = group
        group.add_record(record)

    groups_by_species: defaultdict[str, list[GroupUnit]] = defaultdict(list)
    for group in groups_by_key.values():
        groups_by_species[group.species].append(group)
    return dict(groups_by_species)


def sort_species_for_assignment(
    groups_by_species: Mapping[str, Sequence[GroupUnit]],
    seed: int,
) -> list[str]:
    def species_key(species: str) -> tuple[int, int, int]:
        groups = groups_by_species[species]
        total_samples = sum(group.size for group in groups)
        return (
            len(groups),
            -total_samples,
            stable_hash_int(seed, species),
        )

    return sorted(groups_by_species, key=species_key)


def sort_groups_for_assignment(groups: Sequence[GroupUnit], seed: int) -> list[GroupUnit]:
    return sorted(
        groups,
        key=lambda group: (-group.size, stable_hash_int(seed, group.key)),
    )


def initialize_assignment_state() -> AssignmentState:
    return AssignmentState(split_groups={split_name: [] for split_name in SPLIT_NAMES})


def compute_targets(total_count: int, ratios: Mapping[str, float]) -> dict[str, float]:
    return {split_name: total_count * ratios[split_name] for split_name in SPLIT_NAMES}


def compute_modality_targets(
    records: Sequence[ManifestRecord],
    ratios: Mapping[str, float],
) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    modality_totals = Counter(record.modality for record in records)
    modality_targets = {
        modality: compute_targets(total_count, ratios)
        for modality, total_count in modality_totals.items()
    }
    return dict(modality_totals), modality_targets


def compute_phase_targets(
    records: Sequence[ManifestRecord],
    ratios: Mapping[str, float],
) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    phase_totals = Counter(record.phase for record in records if record.phase)
    phase_targets = {
        phase_name: compute_targets(total_count, ratios)
        for phase_name, total_count in phase_totals.items()
    }
    return dict(phase_totals), phase_targets


def register_group_assignment(state: AssignmentState, group: GroupUnit, split_name: str) -> None:
    state.split_groups[split_name].append(group)
    state.total_assigned[split_name] += group.size
    state.modality_assigned[group.modality][split_name] += group.size
    for phase_name, phase_count in group.phase_counts.items():
        state.phase_assigned[phase_name][split_name] += phase_count
    if state.species_presence[group.species][split_name] == 0:
        state.species_count_per_split[split_name] += 1
    state.species_presence[group.species][split_name] += 1


def remove_group_assignment(state: AssignmentState, group: GroupUnit, split_name: str) -> None:
    state.split_groups[split_name].remove(group)
    state.total_assigned[split_name] -= group.size
    state.modality_assigned[group.modality][split_name] -= group.size
    for phase_name, phase_count in group.phase_counts.items():
        state.phase_assigned[phase_name][split_name] -= phase_count
    state.species_presence[group.species][split_name] -= 1
    if state.species_presence[group.species][split_name] <= 0:
        state.species_count_per_split[split_name] -= 1
        state.species_presence[group.species].pop(split_name, None)


def average_phase_need(
    group: GroupUnit,
    split_name: str,
    *,
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    state: AssignmentState,
) -> float:
    phase_components: list[float] = []
    for phase_name, phase_count in group.phase_counts.items():
        phase_total = phase_totals.get(phase_name, 0)
        if phase_total <= 0 or phase_count <= 0:
            continue
        phase_remaining = (
            phase_targets[phase_name][split_name] - state.phase_assigned[phase_name][split_name]
        )
        phase_components.append(phase_remaining / max(1.0, phase_total))
    return mean(phase_components) if phase_components else 0.0


def choose_two_group_holdout(
    entity_key: str,
    *,
    state: AssignmentState,
    total_targets: Mapping[str, float],
    seed: int,
) -> str:
    return min(
        ("val", "test"),
        key=lambda split_name: (
            state.species_count_per_split[split_name],
            state.total_assigned[split_name] / max(1.0, total_targets[split_name]),
            stable_hash_int(seed, entity_key, split_name),
        ),
    )


def determine_required_splits(
    entity_key: str,
    groups: Sequence[GroupUnit],
    *,
    state: AssignmentState,
    total_targets: Mapping[str, float],
    seed: int,
) -> tuple[str, ...]:
    group_count = len(groups)
    if group_count >= len(SPLIT_NAMES):
        return SPLIT_NAMES
    if group_count == 2:
        return ("train", choose_two_group_holdout(
            entity_key,
            state=state,
            total_targets=total_targets,
            seed=seed,
        ))
    return ("train",)


def sort_holdout_splits_by_need(
    split_names: Sequence[str],
    *,
    state: AssignmentState,
    total_targets: Mapping[str, float],
    seed: int,
    entity_key: str,
    preferred_splits: set[str] | None = None,
) -> list[str]:
    preferred = preferred_splits or set()
    return sorted(
        split_names,
        key=lambda split_name: (
            0 if split_name in preferred else 1,
            state.species_count_per_split[split_name],
            state.total_assigned[split_name] / max(1.0, total_targets[split_name]),
            stable_hash_int(seed, entity_key, split_name),
        ),
    )


def assign_seed_group(
    candidate_groups: Sequence[GroupUnit],
    split_name: str,
    *,
    remaining_groups: list[GroupUnit],
    species_assigned: Counter[str],
    state: AssignmentState,
    prefer_small: bool,
    total_samples: int,
    total_targets: Mapping[str, float],
    modality_totals: Mapping[str, int],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    seed: int,
) -> GroupUnit:
    selected_group = select_seed_group(
        candidate_groups,
        split_name,
        prefer_small=prefer_small,
        total_samples=total_samples,
        total_targets=total_targets,
        modality_totals=modality_totals,
        modality_targets=modality_targets,
        phase_totals=phase_totals,
        phase_targets=phase_targets,
        state=state,
        seed=seed,
    )
    remaining_groups.remove(selected_group)
    register_group_assignment(state, selected_group, split_name)
    species_assigned[split_name] += selected_group.size
    return selected_group


def seed_group_score(
    group: GroupUnit,
    split_name: str,
    *,
    prefer_small: bool,
    total_samples: int,
    total_targets: Mapping[str, float],
    modality_totals: Mapping[str, int],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    state: AssignmentState,
    seed: int,
) -> tuple[int, int, float, float, int, int]:
    total_remaining = (
        total_targets[split_name] - state.total_assigned[split_name]
    ) / max(1.0, total_samples)
    modality_remaining = (
        modality_targets[group.modality][split_name] - state.modality_assigned[group.modality][split_name]
    ) / max(1.0, modality_totals[group.modality])
    phase_need = average_phase_need(
        group,
        split_name,
        phase_totals=phase_totals,
        phase_targets=phase_targets,
        state=state,
    )
    size_component = -group.size if prefer_small else group.size
    return (
        1 if total_remaining > 0 else 0,
        1 if modality_remaining > 0 else 0,
        phase_need,
        total_remaining,
        modality_remaining,
        size_component,
        -stable_hash_int(seed, group.key, split_name),
    )


def select_seed_group(
    remaining_groups: Sequence[GroupUnit],
    split_name: str,
    *,
    prefer_small: bool,
    total_samples: int,
    total_targets: Mapping[str, float],
    modality_totals: Mapping[str, int],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    state: AssignmentState,
    seed: int,
) -> GroupUnit:
    return max(
        remaining_groups,
        key=lambda group: seed_group_score(
            group,
            split_name,
            prefer_small=prefer_small,
            total_samples=total_samples,
            total_targets=total_targets,
            modality_totals=modality_totals,
            modality_targets=modality_targets,
            phase_totals=phase_totals,
            phase_targets=phase_targets,
            state=state,
            seed=seed,
        ),
    )


def seed_species_assignments(
    species: str,
    groups: Sequence[GroupUnit],
    required_splits: Sequence[str],
    *,
    state: AssignmentState,
    total_samples: int,
    total_targets: Mapping[str, float],
    modality_totals: Mapping[str, int],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    seed: int,
) -> tuple[list[GroupUnit], Counter[str], dict[str, tuple[str, ...]]]:
    remaining_groups = list(groups)
    species_assigned: Counter[str] = Counter()
    modality_required_splits: dict[str, tuple[str, ...]] = {}
    modality_groups: dict[str, list[GroupUnit]] = defaultdict(list)
    species_present_splits: set[str] = set()

    for group in remaining_groups:
        modality_groups[group.modality].append(group)

    modality_order = sorted(
        modality_groups,
        key=lambda modality: (
            len(modality_groups[modality]),
            0 if modality == "micro" else 1,
            modality,
        ),
    )

    for modality in modality_order:
        modality_required_splits[modality] = determine_required_splits(
            f"{species}|{modality}",
            modality_groups[modality],
            state=state,
            total_targets=total_targets,
            seed=seed,
        )

    for modality in modality_order:
        if "train" not in modality_required_splits[modality]:
            continue
        candidates = [group for group in remaining_groups if group.modality == modality]
        if not candidates:
            continue
        assign_seed_group(
            candidates,
            "train",
            remaining_groups=remaining_groups,
            species_assigned=species_assigned,
            state=state,
            prefer_small=False,
            total_samples=total_samples,
            total_targets=total_targets,
            modality_totals=modality_totals,
            modality_targets=modality_targets,
            phase_totals=phase_totals,
            phase_targets=phase_targets,
            seed=seed,
        )
        species_present_splits.add("train")

    for modality in modality_order:
        holdout_splits = [
            split_name for split_name in modality_required_splits[modality] if split_name != "train"
        ]
        ordered_holdout_splits = sort_holdout_splits_by_need(
            holdout_splits,
            state=state,
            total_targets=total_targets,
            seed=seed,
            entity_key=f"{species}|{modality}",
            preferred_splits=set(required_splits) - species_present_splits,
        )
        for split_name in ordered_holdout_splits:
            candidates = [group for group in remaining_groups if group.modality == modality]
            if not candidates:
                continue
            assign_seed_group(
                candidates,
                split_name,
                remaining_groups=remaining_groups,
                species_assigned=species_assigned,
                state=state,
                prefer_small=True,
                total_samples=total_samples,
                total_targets=total_targets,
                modality_totals=modality_totals,
                modality_targets=modality_targets,
                phase_totals=phase_totals,
                phase_targets=phase_targets,
                seed=seed,
            )
            species_present_splits.add(split_name)

    missing_species_splits = [split_name for split_name in required_splits if split_name not in species_present_splits]
    ordered_missing_species_splits = sort_holdout_splits_by_need(
        missing_species_splits,
        state=state,
        total_targets=total_targets,
        seed=seed,
        entity_key=species,
    )
    for split_name in ordered_missing_species_splits:
        if not remaining_groups:
            break
        assign_seed_group(
            remaining_groups,
            split_name,
            remaining_groups=remaining_groups,
            species_assigned=species_assigned,
            state=state,
            prefer_small=(split_name != "train"),
            total_samples=total_samples,
            total_targets=total_targets,
            modality_totals=modality_totals,
            modality_targets=modality_targets,
            phase_totals=phase_totals,
            phase_targets=phase_targets,
            seed=seed,
        )
        species_present_splits.add(split_name)

    return remaining_groups, species_assigned, modality_required_splits


def assignment_score(
    group: GroupUnit,
    split_name: str,
    *,
    species_total: int,
    species_targets: Mapping[str, float],
    species_assigned: Mapping[str, int],
    total_samples: int,
    total_targets: Mapping[str, float],
    modality_totals: Mapping[str, int],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_totals: Mapping[str, int],
    phase_targets: Mapping[str, Mapping[str, float]],
    state: AssignmentState,
    seed: int,
) -> tuple[int, int, int, float, float, float, float, int]:
    species_remaining = species_targets[split_name] - species_assigned[split_name]
    total_remaining = total_targets[split_name] - state.total_assigned[split_name]
    modality_remaining = (
        modality_targets[group.modality][split_name] - state.modality_assigned[group.modality][split_name]
    )
    phase_need = average_phase_need(
        group,
        split_name,
        phase_totals=phase_totals,
        phase_targets=phase_targets,
        state=state,
    )

    after_species_residual = abs(
        (species_assigned[split_name] + group.size) - species_targets[split_name]
    ) / max(1.0, species_total)
    after_total_residual = abs(
        (state.total_assigned[split_name] + group.size) - total_targets[split_name]
    ) / max(1.0, total_samples)
    after_modality_residual = abs(
        (state.modality_assigned[group.modality][split_name] + group.size)
        - modality_targets[group.modality][split_name]
    ) / max(1.0, modality_totals[group.modality])

    weighted_remaining = (
        0.45 * (species_remaining / max(1.0, species_total))
        + 0.25 * (total_remaining / max(1.0, total_samples))
        + 0.20 * (modality_remaining / max(1.0, modality_totals[group.modality]))
        + 0.10 * phase_need
    )

    return (
        1 if species_remaining > 0 else 0,
        1 if total_remaining > 0 else 0,
        1 if modality_remaining > 0 else 0,
        weighted_remaining,
        -after_species_residual,
        -after_total_residual,
        -after_modality_residual,
        -stable_hash_int(seed, group.key, split_name),
    )


def build_distribution_counters(
    split_groups: Mapping[str, Sequence[GroupUnit]],
) -> tuple[
    dict[str, int],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[tuple[str, str], Counter[str]],
]:
    total_counts: Counter[str] = Counter()
    modality_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    phase_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    species_presence: defaultdict[str, Counter[str]] = defaultdict(Counter)
    modality_presence: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for split_name, groups in split_groups.items():
        for group in groups:
            total_counts[split_name] += group.size
            modality_counts[group.modality][split_name] += group.size
            for phase_name, phase_count in group.phase_counts.items():
                phase_counts[phase_name][split_name] += phase_count
            species_presence[group.species][split_name] += 1
            modality_presence[(group.species, group.modality)][split_name] += 1

    return (
        dict(total_counts),
        dict(modality_counts),
        dict(phase_counts),
        dict(species_presence),
        dict(modality_presence),
    )


def distribution_error(
    total_counts: Mapping[str, int],
    total_targets: Mapping[str, float],
    modality_counts: Mapping[str, Mapping[str, int]],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_counts: Mapping[str, Mapping[str, int]],
    phase_targets: Mapping[str, Mapping[str, float]],
) -> float:
    total_scale = max(1.0, sum(total_targets.values()))
    total_error = sum(
        abs(total_counts.get(split_name, 0) - total_targets[split_name])
        for split_name in SPLIT_NAMES
    ) / total_scale

    modality_error = 0.0
    for modality, targets in modality_targets.items():
        modality_scale = max(1.0, sum(targets.values()))
        modality_error += sum(
            abs(modality_counts.get(modality, {}).get(split_name, 0) - targets[split_name])
            for split_name in SPLIT_NAMES
        ) / modality_scale

    phase_error = 0.0
    for phase_name, targets in phase_targets.items():
        phase_scale = max(1.0, sum(targets.values()))
        phase_error += sum(
            abs(phase_counts.get(phase_name, {}).get(split_name, 0) - targets[split_name])
            for split_name in SPLIT_NAMES
        ) / phase_scale

    return total_error + (0.35 * modality_error) + (0.20 * phase_error)


def rebalance_group_assignments(
    split_groups: dict[str, list[GroupUnit]],
    *,
    total_targets: Mapping[str, float],
    modality_targets: Mapping[str, Mapping[str, float]],
    phase_targets: Mapping[str, Mapping[str, float]],
    required_species_presence: Mapping[str, Sequence[str]],
    required_modality_presence: Mapping[tuple[str, str], Sequence[str]],
    seed: int,
) -> None:
    counts, modality_counts, phase_counts, species_presence, modality_presence = (
        build_distribution_counters(split_groups)
    )

    max_iterations = sum(len(groups) for groups in split_groups.values()) * 4
    for _ in range(max_iterations):
        current_error = distribution_error(
            counts,
            total_targets,
            modality_counts,
            modality_targets,
            phase_counts,
            phase_targets,
        )
        best_move: tuple[float, int, int] | None = None
        best_group: GroupUnit | None = None
        best_donor = ""
        best_receiver = ""
        best_index = -1

        for donor in SPLIT_NAMES:
            if counts.get(donor, 0) <= total_targets[donor]:
                continue
            for receiver in SPLIT_NAMES:
                if donor == receiver or counts.get(receiver, 0) >= total_targets[receiver]:
                    continue
                for index, group in enumerate(split_groups[donor]):
                    if (
                        donor in required_species_presence.get(group.species, ())
                        and species_presence.get(group.species, {}).get(donor, 0) <= 1
                    ):
                        continue
                    if (
                        donor in required_modality_presence.get((group.species, group.modality), ())
                        and modality_presence.get((group.species, group.modality), {}).get(donor, 0) <= 1
                    ):
                        continue

                    new_counts = dict(counts)
                    new_counts[donor] = new_counts.get(donor, 0) - group.size
                    new_counts[receiver] = new_counts.get(receiver, 0) + group.size

                    new_modality_counts = {
                        modality: Counter(split_counts)
                        for modality, split_counts in modality_counts.items()
                    }
                    new_modality_counts[group.modality][donor] -= group.size
                    new_modality_counts[group.modality][receiver] += group.size

                    new_phase_counts = {
                        phase_name: Counter(split_counts)
                        for phase_name, split_counts in phase_counts.items()
                    }
                    for phase_name, phase_count in group.phase_counts.items():
                        new_phase_counts.setdefault(phase_name, Counter())
                        new_phase_counts[phase_name][donor] -= phase_count
                        new_phase_counts[phase_name][receiver] += phase_count

                    new_error = distribution_error(
                        new_counts,
                        total_targets,
                        new_modality_counts,
                        modality_targets,
                        new_phase_counts,
                        phase_targets,
                    )
                    if new_error >= current_error:
                        continue

                    move_key = (
                        new_error,
                        group.size,
                        stable_hash_int(seed, donor, receiver, group.key),
                    )
                    if best_move is None or move_key < best_move:
                        best_move = move_key
                        best_group = group
                        best_donor = donor
                        best_receiver = receiver
                        best_index = index

        if best_move is None or best_group is None:
            break

        moved_group = split_groups[best_donor].pop(best_index)
        split_groups[best_receiver].append(moved_group)
        counts[best_donor] -= moved_group.size
        counts[best_receiver] += moved_group.size
        modality_counts[moved_group.modality][best_donor] -= moved_group.size
        modality_counts[moved_group.modality][best_receiver] += moved_group.size
        for phase_name, phase_count in moved_group.phase_counts.items():
            phase_counts[phase_name][best_donor] -= phase_count
            phase_counts[phase_name][best_receiver] += phase_count
        species_presence[moved_group.species][best_donor] -= 1
        species_presence[moved_group.species][best_receiver] += 1
        modality_presence[(moved_group.species, moved_group.modality)][best_donor] -= 1
        modality_presence[(moved_group.species, moved_group.modality)][best_receiver] += 1


def split_protocol_records(
    records: Sequence[ManifestRecord],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, list[ManifestRecord]]:
    split_records: dict[str, list[ManifestRecord]] = {split_name: [] for split_name in SPLIT_NAMES}
    if not records:
        return split_records

    groups_by_species = build_groups(records)
    species_order = sort_species_for_assignment(groups_by_species, seed)
    total_samples = len(records)
    total_targets = compute_targets(total_samples, ratios)
    modality_totals, modality_targets = compute_modality_targets(records, ratios)
    phase_totals, phase_targets = compute_phase_targets(records, ratios)
    state = initialize_assignment_state()
    required_species_presence: dict[str, tuple[str, ...]] = {}
    required_modality_presence: dict[tuple[str, str], tuple[str, ...]] = {}

    for species in species_order:
        ordered_groups = list(sort_groups_for_assignment(groups_by_species[species], seed))
        required_splits = determine_required_splits(
            species,
            ordered_groups,
            state=state,
            total_targets=total_targets,
            seed=seed,
        )
        required_species_presence[species] = required_splits
        remaining_groups, species_assigned, modality_required_splits = seed_species_assignments(
            species,
            ordered_groups,
            required_splits,
            state=state,
            total_samples=total_samples,
            total_targets=total_targets,
            modality_totals=modality_totals,
            modality_targets=modality_targets,
            phase_totals=phase_totals,
            phase_targets=phase_targets,
            seed=seed,
        )
        for modality, modality_splits in modality_required_splits.items():
            required_modality_presence[(species, modality)] = modality_splits

        species_total = sum(group.size for group in ordered_groups)
        species_targets = compute_targets(species_total, ratios)
        while remaining_groups:
            group = remaining_groups.pop(0)
            split_name = max(
                SPLIT_NAMES,
                key=lambda candidate_split: assignment_score(
                    group,
                    candidate_split,
                    species_total=species_total,
                    species_targets=species_targets,
                    species_assigned=species_assigned,
                    total_samples=total_samples,
                    total_targets=total_targets,
                    modality_totals=modality_totals,
                    modality_targets=modality_targets,
                    phase_totals=phase_totals,
                    phase_targets=phase_targets,
                    state=state,
                    seed=seed,
                ),
            )
            register_group_assignment(state, group, split_name)
            species_assigned[split_name] += group.size

    rebalance_group_assignments(
        state.split_groups,
        total_targets=total_targets,
        modality_targets=modality_targets,
        phase_targets=phase_targets,
        required_species_presence=required_species_presence,
        required_modality_presence=required_modality_presence,
        seed=seed,
    )

    for split_name, groups in state.split_groups.items():
        for group in groups:
            split_records[split_name].extend(group.rows)
    return split_records


def sort_split_records(records: Sequence[ManifestRecord]) -> list[ManifestRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.species,
            record.modality,
            record.phase,
            record.row.get("image_path", ""),
        ),
    )


def class_balance_stats(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    species_counts = Counter(row["species"] for row in rows)
    values = sorted(species_counts.values())
    if not values:
        return {
            "present_species_count": 0,
            "min_samples_per_species": 0,
            "max_samples_per_species": 0,
            "mean_samples_per_species": 0.0,
            "median_samples_per_species": 0.0,
        }
    return {
        "present_species_count": len(values),
        "min_samples_per_species": min(values),
        "max_samples_per_species": max(values),
        "mean_samples_per_species": round(mean(values), 4),
        "median_samples_per_species": round(median(values), 4),
    }


def build_class_count_rows(
    protocol: str,
    split_rows: Mapping[str, Sequence[dict[str, str]]],
    all_species: Sequence[str],
    protocol_group_by_species: Mapping[str, str],
    groups_by_species: Mapping[str, Sequence[GroupUnit]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        species_split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        species_group_membership: defaultdict[str, set[str]] = defaultdict(set)
        for row in split_rows[split_name]:
            species = row["species"]
            species_split_counts[species]["total"] += 1
            species_split_counts[species][row["modality"]] += 1
            if row["modality"] == "macro":
                if row["phase"] == "Phase1":
                    species_split_counts[species]["phase1_macro"] += 1
                elif row["phase"] == "Phase2":
                    species_split_counts[species]["phase2_macro"] += 1
            group_identifier = row.get("specimen_id") or row.get("source_id") or row.get("image_path")
            species_group_membership[species].add(group_identifier)

        for species in all_species:
            counts = species_split_counts.get(species, Counter())
            rows.append(
                {
                    "protocol": protocol,
                    "split": split_name,
                    "species": species,
                    "protocol_group": protocol_group_by_species.get(species, ""),
                    "total_samples": counts.get("total", 0),
                    "groups_in_split": len(species_group_membership.get(species, set())),
                    "macro_samples": counts.get("macro", 0),
                    "micro_samples": counts.get("micro", 0),
                    "phase1_macro_samples": counts.get("phase1_macro", 0),
                    "phase2_macro_samples": counts.get("phase2_macro", 0),
                    "present_in_split": 1 if counts.get("total", 0) else 0,
                }
            )
    return rows


def build_summary(
    protocol: str,
    *,
    records: Sequence[ManifestRecord],
    split_rows: Mapping[str, Sequence[dict[str, str]]],
    groups_by_species: Mapping[str, Sequence[GroupUnit]],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, Any]:
    total_samples = len(records)
    target_counts = {split_name: total_samples * ratios[split_name] for split_name in SPLIT_NAMES}
    protocol_group_by_species = {
        record.species: record.protocol_group for record in records if record.protocol_group
    }

    split_summaries: dict[str, Any] = {}
    actual_ratios: dict[str, float] = {}
    ratio_warnings: list[dict[str, Any]] = []
    modality_warnings: list[str] = []

    species_presence: defaultdict[str, set[str]] = defaultdict(set)
    for split_name, rows in split_rows.items():
        actual_count = len(rows)
        actual_ratio = actual_count / total_samples if total_samples else 0.0
        actual_ratios[split_name] = round(actual_ratio, 6)
        ratio_delta = actual_ratio - ratios[split_name]
        if abs(ratio_delta) > 0.01:
            ratio_warnings.append(
                {
                    "split": split_name,
                    "target_ratio": round(ratios[split_name], 6),
                    "actual_ratio": round(actual_ratio, 6),
                    "delta_ratio": round(ratio_delta, 6),
                    "target_samples": round(target_counts[split_name], 4),
                    "actual_samples": actual_count,
                }
            )

        modality_counts = Counter(row["modality"] for row in rows)
        if not modality_counts.get("macro"):
            modality_warnings.append(f"{split_name} has no macro samples.")
        if not modality_counts.get("micro"):
            modality_warnings.append(f"{split_name} has no micro samples.")

        macro_phase_counts = Counter(
            row["phase"] for row in rows if row["modality"] == "macro" and row["phase"]
        )
        species_in_split = sorted({row["species"] for row in rows})
        for species in species_in_split:
            species_presence[species].add(split_name)

        split_summaries[split_name] = {
            "sample_count": actual_count,
            "target_sample_count": round(target_counts[split_name], 4),
            "actual_ratio": round(actual_ratio, 6),
            "modality_counts": dict(sorted(modality_counts.items())),
            "species_count": len(species_in_split),
            "macro_phase_counts": dict(sorted(macro_phase_counts.items())),
            "class_balance": class_balance_stats(rows),
        }

    group_counts_by_species = {species: len(groups) for species, groups in groups_by_species.items()}
    sample_counts_by_species = {
        species: sum(group.size for group in groups) for species, groups in groups_by_species.items()
    }
    group_threshold = len(SPLIT_NAMES)
    min_holdout_samples = max(math.ceil(1 / ratios["val"]), math.ceil(1 / ratios["test"]))

    too_few_groups = sorted(
        species for species, group_count in group_counts_by_species.items() if group_count < group_threshold
    )
    too_few_samples = sorted(
        species for species, sample_count in sample_counts_by_species.items() if sample_count < min_holdout_samples
    )
    missing_from_val = sorted(
        species
        for species, splits in species_presence.items()
        if "train" in splits and "val" not in splits
    )
    missing_from_test = sorted(
        species
        for species, splits in species_presence.items()
        if "train" in splits and "test" not in splits
    )

    return {
        "protocol": protocol,
        "seed": seed,
        "ratios": dict(ratios),
        "total_samples": total_samples,
        "group_count": sum(len(groups) for groups in groups_by_species.values()),
        "species_count": len(groups_by_species),
        "split_summaries": split_summaries,
        "warnings": {
            "classes_with_too_few_groups_for_full_coverage": too_few_groups,
            "classes_with_too_few_samples_for_nonzero_holdout": too_few_samples,
            "classes_in_train_but_missing_from_val": missing_from_val,
            "classes_in_train_but_missing_from_test": missing_from_test,
            "ratio_imperfections": ratio_warnings,
            "modality_warnings": modality_warnings,
        },
        "actual_ratios": actual_ratios,
        "protocol_group_by_species": protocol_group_by_species,
    }


def split_protocol_manifest(
    protocol: str,
    manifest_path: Path,
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> ProtocolSplitArtifacts:
    fieldnames, records = read_manifest(manifest_path)
    groups_by_species = build_groups(records)
    split_records = split_protocol_records(
        records,
        ratios=ratios,
        seed=seed,
    )

    split_rows = {
        split_name: [record.row for record in sort_split_records(split_records[split_name])]
        for split_name in SPLIT_NAMES
    }
    summary = build_summary(
        protocol,
        records=records,
        split_rows=split_rows,
        groups_by_species=groups_by_species,
        ratios=ratios,
        seed=seed,
    )
    all_species = sorted(groups_by_species)
    protocol_group_by_species = summary["protocol_group_by_species"]
    class_count_rows = build_class_count_rows(
        protocol,
        split_rows,
        all_species,
        protocol_group_by_species,
        groups_by_species,
    )
    summary.pop("protocol_group_by_species", None)
    return ProtocolSplitArtifacts(
        protocol=protocol,
        fieldnames=fieldnames,
        split_rows=split_rows,
        summary=summary,
        class_count_rows=class_count_rows,
    )


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


def write_protocol_outputs(
    artifacts: ProtocolSplitArtifacts,
    *,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Path]:
    split_paths = {
        split_name: output_dir / f"{artifacts.protocol}_{split_name}.csv"
        for split_name in SPLIT_NAMES
    }
    for split_name, path in split_paths.items():
        write_csv(path, artifacts.fieldnames, artifacts.split_rows[split_name])

    summary_path = report_dir / f"{artifacts.protocol}_split_summary.json"
    class_counts_path = report_dir / f"{artifacts.protocol}_split_class_counts.csv"
    write_json(summary_path, artifacts.summary)
    write_csv(class_counts_path, REPORT_CLASS_COUNT_COLUMNS, artifacts.class_count_rows)

    return {
        **split_paths,
        "summary": summary_path,
        "class_counts": class_counts_path,
    }


def print_protocol_summary(artifacts: ProtocolSplitArtifacts) -> None:
    summary = artifacts.summary
    print(f"\nProtocol {artifacts.protocol.upper()}")
    print(f"  Total samples: {summary['total_samples']}")
    for split_name in SPLIT_NAMES:
        split_summary = summary["split_summaries"][split_name]
        class_balance = split_summary["class_balance"]
        print(
            f"  {split_name}: samples={split_summary['sample_count']} "
            f"ratio={split_summary['actual_ratio']:.4f} "
            f"species={split_summary['species_count']} "
            f"modality={split_summary['modality_counts']} "
            f"macro_phase={split_summary['macro_phase_counts']}"
        )
        print(
            f"    class_balance: min={class_balance['min_samples_per_species']} "
            f"max={class_balance['max_samples_per_species']} "
            f"mean={class_balance['mean_samples_per_species']} "
            f"median={class_balance['median_samples_per_species']}"
        )

    warnings = summary["warnings"]
    print(
        f"  Warnings: too_few_groups={len(warnings['classes_with_too_few_groups_for_full_coverage'])}, "
        f"too_few_samples={len(warnings['classes_with_too_few_samples_for_nonzero_holdout'])}, "
        f"missing_val={len(warnings['classes_in_train_but_missing_from_val'])}, "
        f"missing_test={len(warnings['classes_in_train_but_missing_from_test'])}, "
        f"ratio_imperfections={len(warnings['ratio_imperfections'])}"
    )


def main() -> int:
    args = parse_args()
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    ensure_directory(args.output_dir)
    ensure_directory(args.report_dir)

    outputs: dict[str, dict[str, Path]] = {}
    for protocol, filename in PROTOCOL_FILES.items():
        manifest_path = args.manifest_dir / filename
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for {protocol}: {manifest_path}")

        artifacts = split_protocol_manifest(
            protocol,
            manifest_path,
            ratios=ratios,
            seed=args.seed,
        )
        outputs[protocol] = write_protocol_outputs(
            artifacts,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
        )
        print_protocol_summary(artifacts)

    print("\nWrote split files:")
    for protocol in ("p1", "p2", "p3"):
        protocol_outputs = outputs[protocol]
        print(f"  {protocol}_train: {protocol_outputs['train']}")
        print(f"  {protocol}_val:   {protocol_outputs['val']}")
        print(f"  {protocol}_test:  {protocol_outputs['test']}")
        print(f"  {protocol}_summary: {protocol_outputs['summary']}")
        print(f"  {protocol}_class_counts: {protocol_outputs['class_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
