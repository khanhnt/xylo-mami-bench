"""Supervised contrastive loss for XyloMaMi-Bench alignment training.

This implementation is intentionally trainer-driven: the trainer supplies the
positive mask and optional negative weights so XyloMaMi-Bench can switch between exact
alignment, genus-soft alignment, and relaxed-negative policies without changing
the model itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_pair_mask(mask: Tensor, batch_size: int) -> Tensor:
    if mask.ndim != 2 or mask.shape != (batch_size, batch_size):
        raise ValueError(
            f"positive_mask must have shape {(batch_size, batch_size)}, got {tuple(mask.shape)}."
        )
    return mask.to(dtype=torch.float32)


def _validate_pair_weights(weights: Tensor, batch_size: int, *, name: str) -> Tensor:
    if weights.ndim != 2 or weights.shape != (batch_size, batch_size):
        raise ValueError(
            f"{name} must have shape {(batch_size, batch_size)}, got {tuple(weights.shape)}."
        )
    if torch.any(weights < 0):
        raise ValueError(f"{name} must be non-negative.")
    return weights.to(dtype=torch.float32)


class SupConLoss(nn.Module):
    """Supervised contrastive loss with weighted positive masks.

    This implementation accepts a precomputed positive mask so the trainer can
    control which pairs are considered positives. For XyloMaMi-Bench we use strong
    cross-modal exact-species positives by default.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        normalize_embeddings: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if base_temperature <= 0:
            raise ValueError("base_temperature must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.normalize_embeddings = normalize_embeddings
        self.eps = float(eps)

    def forward(
        self,
        embeddings: Tensor,
        positive_mask: Tensor,
        *,
        anchor_mask: Tensor | None = None,
        sample_weights: Tensor | None = None,
        negative_weights: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D with shape [batch, dim], got {tuple(embeddings.shape)}."
            )
        batch_size = int(embeddings.shape[0])
        if batch_size <= 1:
            return embeddings.new_zeros(())

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2.0, dim=1)

        positive_mask = _validate_pair_mask(positive_mask, batch_size).to(embeddings.device)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        logits_mask = (~self_mask).to(dtype=embeddings.dtype)
        positive_mask = positive_mask * logits_mask

        denominator_weights = logits_mask
        if negative_weights is not None:
            validated_negative_weights = _validate_pair_weights(
                negative_weights,
                batch_size,
                name="negative_weights",
            ).to(device=embeddings.device, dtype=embeddings.dtype)
            denominator_weights = torch.where(
                positive_mask > 0,
                torch.ones_like(validated_negative_weights, dtype=embeddings.dtype),
                validated_negative_weights,
            )
            denominator_weights = denominator_weights * logits_mask

        exp_logits = torch.exp(logits) * denominator_weights
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_weights = positive_mask.sum(dim=1)
        valid_anchors = positive_weights > 0
        if anchor_mask is not None:
            if anchor_mask.ndim != 1 or anchor_mask.shape[0] != batch_size:
                raise ValueError(
                    f"anchor_mask must have shape {(batch_size,)}, got {tuple(anchor_mask.shape)}."
                )
            valid_anchors = valid_anchors & anchor_mask.to(device=embeddings.device, dtype=torch.bool)
        if not valid_anchors.any():
            return embeddings.new_zeros(())

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_weights.clamp_min(self.eps)
        losses = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_anchors]

        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.shape[0] != batch_size:
                raise ValueError(
                    f"sample_weights must have shape {(batch_size,)}, got {tuple(sample_weights.shape)}."
                )
            weights = sample_weights.to(device=embeddings.device, dtype=losses.dtype)[valid_anchors]
            return (losses * weights).sum() / weights.sum().clamp_min(self.eps)
        return losses.mean()
