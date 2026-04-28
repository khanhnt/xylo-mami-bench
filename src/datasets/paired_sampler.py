"""Class-aware modality-balanced sampling utilities for XyloMaMi-Bench alignment training."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from torch.utils.data import Sampler

from src.datasets.manifest_dataset import ManifestDataset


def _stable_hash_int(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(f"{seed}|{'|'.join(parts)}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True)
class AlignmentBatch:
    indices: list[int]
    modality_counts: dict[str, int]
    species_counts: dict[str, int]


@dataclass
class _PoolState:
    base_indices: tuple[int, ...]
    order: list[int]
    position: int
    cycle: int


class AlignmentBatchSampler(Sampler[list[int]]):
    """Batch sampler for joint macro-micro alignment experiments.

    The sampler builds species-aware batches with three goals:

    1. Prefer overlap species that have both macro and micro samples.
    2. Keep batch modality composition reasonably balanced.
    3. Mitigate class imbalance via inverse-frequency species sampling.
    """

    def __init__(
        self,
        dataset: ManifestDataset,
        batch_size: int,
        *,
        classes_per_batch: int | None = None,
        paired_species_per_batch: int | None = None,
        min_cross_modal_pairs: int = 1,
        target_macro_fraction: float = 0.5,
        inverse_frequency_power: float = 0.5,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= target_macro_fraction <= 1.0:
            raise ValueError("target_macro_fraction must be between 0 and 1.")
        if inverse_frequency_power < 0:
            raise ValueError("inverse_frequency_power must be non-negative.")
        if min_cross_modal_pairs < 0:
            raise ValueError("min_cross_modal_pairs must be non-negative.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.inverse_frequency_power = inverse_frequency_power
        self.min_cross_modal_pairs = min_cross_modal_pairs

        self.all_species = tuple(sorted(dataset.indices_by_species))
        self.species_weights = {
            species: 1.0 / (max(1, dataset.species_counts[species]) ** inverse_frequency_power)
            for species in self.all_species
        }
        self.species_by_modality = {
            modality: tuple(
                sorted(
                    species
                    for species, modality_indices in dataset.indices_by_species_and_modality.items()
                    if modality_indices.get(modality)
                )
            )
            for modality in ("macro", "micro")
        }
        self.paired_species = tuple(
            sorted(
                species
                for species, modality_indices in dataset.indices_by_species_and_modality.items()
                if modality_indices.get("macro") and modality_indices.get("micro")
            )
        )

        default_classes_per_batch = max(1, min(len(self.all_species), batch_size // 2))
        self.classes_per_batch = classes_per_batch or default_classes_per_batch
        default_pair_count = max(min_cross_modal_pairs, batch_size // 4)
        self.paired_species_per_batch = paired_species_per_batch or default_pair_count
        self.paired_species_per_batch = min(self.paired_species_per_batch, batch_size // 2)

        if dataset.modality_counts.get("macro") and dataset.modality_counts.get("micro"):
            self.target_macro_fraction = target_macro_fraction
        elif dataset.modality_counts.get("macro"):
            self.target_macro_fraction = 1.0
        else:
            self.target_macro_fraction = 0.0

        self.steps_per_epoch = (
            len(dataset) // batch_size if drop_last else math.ceil(len(dataset) / batch_size)
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _make_rng(self, *parts: str) -> random.Random:
        return random.Random(_stable_hash_int(self.seed + self.epoch, *parts))

    def _shuffle_indices(self, indices: Sequence[int], key: str, cycle: int) -> list[int]:
        shuffled = list(indices)
        self._make_rng(key, str(cycle)).shuffle(shuffled)
        return shuffled

    def _init_epoch_state(self) -> dict[tuple[str, str], _PoolState]:
        state: dict[tuple[str, str], _PoolState] = {}
        for species, modality_indices in self.dataset.indices_by_species_and_modality.items():
            for modality, indices in modality_indices.items():
                key = (species, modality)
                state[key] = _PoolState(
                    base_indices=tuple(indices),
                    order=self._shuffle_indices(indices, f"{species}|{modality}", 0),
                    position=0,
                    cycle=0,
                )
        return state

    def _draw_index(
        self,
        species: str,
        modality: str,
        *,
        epoch_state: dict[tuple[str, str], _PoolState],
        used_in_batch: set[int],
    ) -> int | None:
        key = (species, modality)
        pool = epoch_state.get(key)
        if pool is None or not pool.base_indices:
            return None

        attempts = 0
        max_attempts = max(1, len(pool.base_indices) * 2)
        while attempts < max_attempts:
            if pool.position >= len(pool.order):
                pool.cycle += 1
                pool.order = self._shuffle_indices(pool.base_indices, f"{species}|{modality}", pool.cycle)
                pool.position = 0

            index = pool.order[pool.position]
            pool.position += 1
            attempts += 1
            if index not in used_in_batch:
                return index
        return None

    def _weighted_choice(
        self,
        candidates: Sequence[str],
        *,
        rng: random.Random,
        excluded_species: set[str] | None = None,
    ) -> str | None:
        filtered = [species for species in candidates if not excluded_species or species not in excluded_species]
        if not filtered:
            return None
        weights = [self.species_weights[species] for species in filtered]
        return rng.choices(filtered, weights=weights, k=1)[0]

    def _weighted_distinct_species(
        self,
        candidates: Sequence[str],
        *,
        count: int,
        rng: random.Random,
    ) -> list[str]:
        remaining = list(candidates)
        selected: list[str] = []
        while remaining and len(selected) < count:
            weights = [self.species_weights[species] for species in remaining]
            chosen = rng.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _target_modality_counts(self) -> tuple[int, int]:
        target_macro = int(round(self.batch_size * self.target_macro_fraction))
        target_micro = self.batch_size - target_macro
        return target_macro, target_micro

    def _preferred_fill_modality(
        self,
        modality_counts: Counter[str],
        *,
        target_macro: int,
        target_micro: int,
    ) -> str:
        if target_micro <= 0:
            return "macro"
        if target_macro <= 0:
            return "micro"
        if modality_counts["macro"] < target_macro and modality_counts["micro"] >= target_micro:
            return "macro"
        if modality_counts["micro"] < target_micro and modality_counts["macro"] >= target_macro:
            return "micro"
        macro_gap = target_macro - modality_counts["macro"]
        micro_gap = target_micro - modality_counts["micro"]
        return "macro" if macro_gap >= micro_gap else "micro"

    def _candidate_species_for_modality(self, modality: str) -> Sequence[str]:
        candidates = self.species_by_modality.get(modality, ())
        return candidates if candidates else self.all_species

    def _fill_single_sample(
        self,
        *,
        rng: random.Random,
        epoch_state: dict[tuple[str, str], _PoolState],
        batch: list[int],
        used_in_batch: set[int],
        batch_species: Counter[str],
        modality_counts: Counter[str],
        target_macro: int,
        target_micro: int,
    ) -> bool:
        desired_modality = self._preferred_fill_modality(
            modality_counts,
            target_macro=target_macro,
            target_micro=target_micro,
        )

        candidate_species = self._candidate_species_for_modality(desired_modality)
        distinct_species = set(batch_species)
        excluded_species = distinct_species if len(distinct_species) < self.classes_per_batch else None

        species = self._weighted_choice(
            candidate_species,
            rng=rng,
            excluded_species=excluded_species,
        )
        if species is None:
            species = self._weighted_choice(candidate_species, rng=rng)
        if species is None:
            return False

        index = self._draw_index(
            species,
            desired_modality,
            epoch_state=epoch_state,
            used_in_batch=used_in_batch,
        )
        if index is None:
            alternate_modality = "micro" if desired_modality == "macro" else "macro"
            index = self._draw_index(
                species,
                alternate_modality,
                epoch_state=epoch_state,
                used_in_batch=used_in_batch,
            )
            if index is None:
                return False
            desired_modality = alternate_modality

        batch.append(index)
        used_in_batch.add(index)
        batch_species[species] += 1
        modality_counts[desired_modality] += 1
        return True

    def _build_batch(
        self,
        *,
        rng: random.Random,
        epoch_state: dict[tuple[str, str], _PoolState],
    ) -> list[int]:
        batch: list[int] = []
        used_in_batch: set[int] = set()
        batch_species: Counter[str] = Counter()
        modality_counts: Counter[str] = Counter()
        target_macro, target_micro = self._target_modality_counts()

        pair_species_target = min(
            len(self.paired_species),
            self.paired_species_per_batch,
            self.batch_size // 2,
        )
        pair_species_target = max(
            0 if not self.paired_species else min(pair_species_target, self.batch_size // 2),
            min(self.min_cross_modal_pairs, len(self.paired_species), self.batch_size // 2),
        )
        paired_species = self._weighted_distinct_species(
            self.paired_species,
            count=pair_species_target,
            rng=rng,
        )

        for species in paired_species:
            macro_index = self._draw_index(
                species,
                "macro",
                epoch_state=epoch_state,
                used_in_batch=used_in_batch,
            )
            micro_index = self._draw_index(
                species,
                "micro",
                epoch_state=epoch_state,
                used_in_batch=used_in_batch,
            )
            if macro_index is None or micro_index is None:
                continue
            batch.extend((macro_index, micro_index))
            used_in_batch.update((macro_index, micro_index))
            batch_species[species] += 2
            modality_counts["macro"] += 1
            modality_counts["micro"] += 1
            if len(batch) >= self.batch_size:
                break

        attempts_without_progress = 0
        while len(batch) < self.batch_size:
            filled = self._fill_single_sample(
                rng=rng,
                epoch_state=epoch_state,
                batch=batch,
                used_in_batch=used_in_batch,
                batch_species=batch_species,
                modality_counts=modality_counts,
                target_macro=target_macro,
                target_micro=target_micro,
            )
            if filled:
                attempts_without_progress = 0
                continue

            attempts_without_progress += 1
            if attempts_without_progress > max(8, len(self.all_species)):
                break

        if len(batch) > self.batch_size:
            batch = batch[: self.batch_size]
        rng.shuffle(batch)
        return batch

    def __iter__(self) -> Iterable[list[int]]:
        epoch_state = self._init_epoch_state()
        for batch_index in range(self.steps_per_epoch):
            rng = self._make_rng("batch", str(batch_index))
            batch = self._build_batch(rng=rng, epoch_state=epoch_state)
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                yield batch

