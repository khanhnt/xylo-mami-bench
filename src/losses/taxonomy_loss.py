"""Taxonomy-aware relation utilities for XyloMaMi-Bench dual-encoder training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SUPPORTED_RELATION_POLICIES = frozenset(
    {
        "exact_only",
        "exact_plus_genus_soft",
        "exact_plus_genus_relaxed_negative",
        "genus_only",
        "shuffled_exact_control",
        "hard_negative_control",
    }
)


@dataclass(frozen=True)
class TaxonomyRelationMasks:
    """Pairwise cross-modal taxonomy relation masks."""

    cross_modal_mask: Tensor
    exact_species_mask: Tensor
    genus_soft_mask: Tensor
    unrelated_mask: Tensor

    @property
    def exact_pair_count(self) -> int:
        return int(self.exact_species_mask.sum().item() // 2)

    @property
    def genus_pair_count(self) -> int:
        return int(self.genus_soft_mask.sum().item() // 2)


def build_taxonomy_relation_masks(
    species_ids: Tensor,
    genus_ids: Tensor,
    modality_ids: Tensor,
) -> TaxonomyRelationMasks:
    """Build exact-species and same-genus-different-species masks."""

    if species_ids.ndim != 1 or genus_ids.ndim != 1 or modality_ids.ndim != 1:
        raise ValueError("species_ids, genus_ids, and modality_ids must all be 1D tensors.")
    if not (species_ids.shape[0] == genus_ids.shape[0] == modality_ids.shape[0]):
        raise ValueError("species_ids, genus_ids, and modality_ids must have matching lengths.")

    batch_size = int(species_ids.shape[0])
    device = species_ids.device
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)

    same_species = species_ids[:, None] == species_ids[None, :]
    same_genus = genus_ids[:, None] == genus_ids[None, :]
    same_modality = modality_ids[:, None] == modality_ids[None, :]

    cross_modal_mask = (~same_modality) & (~diagonal)
    exact_species_mask = cross_modal_mask & same_species
    genus_soft_mask = cross_modal_mask & same_genus & (~same_species)
    unrelated_mask = cross_modal_mask & (~same_genus)

    return TaxonomyRelationMasks(
        cross_modal_mask=cross_modal_mask,
        exact_species_mask=exact_species_mask,
        genus_soft_mask=genus_soft_mask,
        unrelated_mask=unrelated_mask,
    )


class TaxonomyRelationLoss(nn.Module):
    """Encourage same-genus cross-modal pairs to remain moderately close.

    This is intentionally softer than exact-species supervised contrast. It
    applies only to same-genus, different-species cross-modal pairs.

    In XyloMaMi-Bench v2 this module is used in two ways:
    1. `exact_plus_genus_soft`: a weak same-genus attraction term.
    2. `exact_plus_genus_relaxed_negative`: an optional light floor regularizer
       while same-genus pairs remain excluded from the exact-positive mask.
    """

    def __init__(
        self,
        *,
        similarity_floor: float = 0.35,
        normalize_embeddings: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not -1.0 <= similarity_floor <= 1.0:
            raise ValueError("similarity_floor must be in [-1, 1].")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.similarity_floor = float(similarity_floor)
        self.normalize_embeddings = normalize_embeddings
        self.eps = float(eps)

    def forward(
        self,
        embeddings: Tensor,
        genus_soft_mask: Tensor,
        *,
        pair_weights: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D with shape [batch, dim], got {tuple(embeddings.shape)}."
            )
        batch_size = int(embeddings.shape[0])
        if genus_soft_mask.ndim != 2 or genus_soft_mask.shape != (batch_size, batch_size):
            raise ValueError(
                "genus_soft_mask must have shape "
                f"{(batch_size, batch_size)}, got {tuple(genus_soft_mask.shape)}."
            )
        if batch_size <= 1:
            return embeddings.new_zeros(())

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2.0, dim=1)

        similarity = torch.matmul(embeddings, embeddings.T)
        upper_triangular = torch.triu(genus_soft_mask.to(dtype=torch.bool, device=embeddings.device), diagonal=1)
        if not upper_triangular.any():
            return embeddings.new_zeros(())

        penalties = F.relu(self.similarity_floor - similarity[upper_triangular])
        if pair_weights is not None:
            if pair_weights.ndim != 2 or pair_weights.shape != (batch_size, batch_size):
                raise ValueError(
                    "pair_weights must have shape "
                    f"{(batch_size, batch_size)}, got {tuple(pair_weights.shape)}."
                )
            weights = pair_weights.to(device=embeddings.device, dtype=penalties.dtype)[upper_triangular]
            return (penalties * weights).sum() / weights.sum().clamp_min(self.eps)
        return penalties.mean()


def build_relaxed_negative_weights(
    masks: TaxonomyRelationMasks,
    *,
    genus_negative_weight: float,
) -> Tensor:
    """Build denominator weights for relaxed-negative SupCon policies.

    Exact-species positives keep full weight, unrelated pairs keep full
    negative repulsion, and same-genus different-species cross-modal pairs are
    downweighted to reduce repulsion without promoting them to positives.
    """

    if not 0.0 <= genus_negative_weight <= 1.0:
        raise ValueError("genus_negative_weight must be in [0, 1].")

    weights = masks.cross_modal_mask.to(dtype=torch.float32)
    weights = weights + (~masks.cross_modal_mask).to(dtype=torch.float32)
    weights[masks.genus_soft_mask] = genus_negative_weight
    return weights
