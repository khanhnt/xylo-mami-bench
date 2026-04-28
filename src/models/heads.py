"""Reusable pooling, projection, and classifier heads for XyloMaMi-Bench models."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SUPPORTED_POOL_TYPES = frozenset({"avg", "mean", "token"})


def _normalize_pool_type(pool_type: str) -> str:
    normalized = pool_type.strip().lower()
    if normalized not in SUPPORTED_POOL_TYPES:
        raise ValueError(
            f"Unsupported pool_type '{pool_type}'. Expected one of {sorted(SUPPORTED_POOL_TYPES)}."
        )
    return normalized


def _coerce_class_weights(class_weights: Tensor | Sequence[float] | None) -> Tensor | None:
    if class_weights is None:
        return None
    if isinstance(class_weights, Tensor):
        return class_weights.float()
    return torch.tensor(list(class_weights), dtype=torch.float32)


class GlobalFeaturePooling(nn.Module):
    """Pool backbone features into a `[batch, channels]` representation."""

    def __init__(self, pool_type: str = "avg") -> None:
        super().__init__()
        self.pool_type = _normalize_pool_type(pool_type)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim == 2:
            return features
        if features.ndim == 4:
            pooled = self.spatial_pool(features)
            return pooled.flatten(1)
        if features.ndim == 3:
            if self.pool_type == "token":
                return features[:, 0]
            return features.mean(dim=1)
        raise ValueError(
            f"Expected 2D, 3D, or 4D feature tensors, but received shape {tuple(features.shape)}."
        )


class LinearClassifierHead(nn.Module):
    """Dropout + linear projection head for branch-specific classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, pooled_features: Tensor) -> Tensor:
        return self.classifier(self.dropout(pooled_features))


class ProjectionMLP(nn.Module):
    """Projection MLP used by the dual-encoder alignment model.

    The default ConvNeXt-friendly configuration is `768 -> 512 -> 256`, while
    smaller backbones can safely use `feature_dim -> feature_dim -> 256`.
    """

    def __init__(
        self,
        in_features: int,
        *,
        hidden_dims: Sequence[int] | None = None,
        out_features: int = 256,
        dropout: float = 0.0,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        normalized_hidden_dims = tuple(int(dim) for dim in (hidden_dims or ()))
        if any(dim <= 0 for dim in normalized_hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")

        layers: list[nn.Module] = []
        last_dim = in_features
        for hidden_dim in normalized_hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_features))

        self.in_features = in_features
        self.hidden_dims = normalized_hidden_dims
        self.out_features = out_features
        self.dropout_probability = dropout
        self.l2_normalize = l2_normalize
        self.network = nn.Sequential(*layers)

    def forward(self, pooled_features: Tensor) -> Tensor:
        projected = self.network(pooled_features)
        if self.l2_normalize:
            projected = F.normalize(projected, p=2.0, dim=1)
        return projected


def default_projection_hidden_dims(feature_dim: int) -> tuple[int, ...]:
    """Choose a sensible default hidden shape for alignment projection heads."""

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if feature_dim >= 512:
        return (512,)
    return (feature_dim,)


def build_classification_loss(
    *,
    class_weights: Tensor | Sequence[float] | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build a CrossEntropyLoss compatible with class weights and label smoothing."""

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the range [0, 1).")

    return nn.CrossEntropyLoss(
        weight=_coerce_class_weights(class_weights),
        label_smoothing=label_smoothing,
    )


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    class_weights: Tensor | Sequence[float] | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Functional helper for trainer-side compatibility."""

    weight = _coerce_class_weights(class_weights)
    return F.cross_entropy(
        logits,
        targets,
        weight=weight.to(logits.device) if weight is not None else None,
        label_smoothing=label_smoothing,
    )


def iter_trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Yield parameters that currently require gradients."""

    for parameter in module.parameters():
        if parameter.requires_grad:
            yield parameter
