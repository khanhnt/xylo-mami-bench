"""Reusable RGB+gray fusion utilities for XyloMaMi-Bench models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from src.models.backbones import (
    ImageSizeArg,
    SUPPORTED_BACKBONES,
    build_backbone,
)
from src.models.heads import GlobalFeaturePooling

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RGB_TO_GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)


def _normalize_patterns(patterns: Sequence[str] | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    normalized = tuple(pattern.strip().lower() for pattern in patterns if pattern.strip())
    return tuple(dict.fromkeys(normalized))


def _resolve_backbone_name(backbone_name: str) -> str:
    normalized = backbone_name.strip()
    if normalized not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Expected one of {sorted(SUPPORTED_BACKBONES)}."
        )
    return normalized


def normalized_rgb_to_grayscale_triplet(images: Tensor) -> Tensor:
    """Convert normalized RGB tensors into normalized grayscale 3-channel tensors."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "normalized_rgb_to_grayscale_triplet expects a [batch, 3, height, width] tensor, "
            f"got {tuple(images.shape)}."
        )

    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = torch.clamp(images * std + mean, min=0.0, max=1.0)

    weights = images.new_tensor(RGB_TO_GRAY_WEIGHTS).view(1, 3, 1, 1)
    gray = (rgb * weights).sum(dim=1, keepdim=True)
    gray_triplet = gray.repeat(1, 3, 1, 1)
    return (gray_triplet - mean) / std


@dataclass(frozen=True)
class DualViewFeatureOutput:
    rgb_backbone_features: Tensor
    gray_backbone_features: Tensor
    rgb_pooled_features: Tensor
    gray_pooled_features: Tensor
    fused_pooled_features: Tensor


@dataclass(frozen=True)
class FusionOutput:
    fused_features: Tensor
    gate: Tensor | None = None


class SingleViewFeatureEncoder(nn.Module):
    """Backbone + pooling wrapper for one RGB or gray view."""

    def __init__(
        self,
        *,
        backbone_name: str,
        pretrained: bool = True,
        image_size: ImageSizeArg | None = None,
        pool_type: str | None = None,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        resolved_backbone_name = _resolve_backbone_name(backbone_name)
        self.backbone = build_backbone(
            resolved_backbone_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.pool = GlobalFeaturePooling(pool_type or self.backbone.default_pool_type)

        self.backbone_name = self.backbone.name
        self.image_size = self.backbone.image_size
        self.feature_dim = self.backbone.feature_dim
        self.configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    def forward_backbone(self, images: Tensor) -> Tensor:
        return self.backbone.forward_features(images)

    def pool_features(self, backbone_features: Tensor) -> Tensor:
        return self.pool(backbone_features)

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        backbone_features = self.forward_backbone(images)
        pooled_features = self.pool_features(backbone_features)
        return backbone_features, pooled_features

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def configure_backbone_trainability(
        self,
        *,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        patterns = _normalize_patterns(trainable_backbone_patterns)
        if patterns:
            self.freeze_backbone()
            matched = False
            for name, parameter in self.backbone.named_parameters():
                lowered_name = name.lower()
                if any(pattern in lowered_name for pattern in patterns):
                    parameter.requires_grad = True
                    matched = True
            if not matched:
                raise ValueError(
                    "No backbone parameters matched trainable_backbone_patterns="
                    f"{list(patterns)}."
                )
            return

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()


class ResidualFeatureFusion(nn.Module):
    """Late fusion block supporting residual or gated RGB-gray fusion."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        mode: str = "residual",
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in {"residual", "gated"}:
            raise ValueError("mode must be either 'residual' or 'gated'.")

        resolved_hidden_dim = hidden_dim or feature_dim
        layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, resolved_hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(resolved_hidden_dim, feature_dim))
        self.network = nn.Sequential(*layers)
        self.gate_network = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, resolved_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(resolved_hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.feature_dim = feature_dim
        self.hidden_dim = resolved_hidden_dim
        self.dropout_probability = dropout
        self.residual_scale = float(residual_scale)
        self.mode = resolved_mode

    def forward(self, rgb_features: Tensor, gray_features: Tensor) -> FusionOutput:
        if rgb_features.shape != gray_features.shape:
            raise ValueError(
                "RGB and gray features must have matching shapes, got "
                f"{tuple(rgb_features.shape)} vs {tuple(gray_features.shape)}."
            )
        concatenated = torch.cat([rgb_features, gray_features], dim=1)
        if self.mode == "gated":
            gate = self.gate_network(concatenated)
            fused = (gate * rgb_features) + ((1.0 - gate) * gray_features)
            return FusionOutput(fused_features=fused, gate=gate)

        delta = self.network(concatenated)
        fused = rgb_features + self.residual_scale * delta
        return FusionOutput(fused_features=fused, gate=None)
