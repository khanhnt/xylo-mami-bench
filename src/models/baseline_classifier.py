"""Baseline classifier for XyloMaMi-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from torch import Tensor, nn

from src.models.backbones import (
    DEFAULT_BACKBONE,
    SUPPORTED_BACKBONES,
    ImageSizeArg,
    TimmBackbone,
    build_backbone,
)
from src.models.heads import (
    GlobalFeaturePooling,
    LinearClassifierHead,
    build_classification_loss,
)
from src.models.rgbgray_fusion import (
    ResidualFeatureFusion,
    SingleViewFeatureEncoder,
    normalized_rgb_to_grayscale_triplet,
)


@dataclass(frozen=True)
class BaselineClassifierOutput:
    """Structured output for forward passes that need features and logits."""

    logits: Tensor
    pooled_features: Tensor
    backbone_features: Tensor
    rgb_pooled_features: Tensor | None = None
    gray_pooled_features: Tensor | None = None
    rgb_backbone_features: Tensor | None = None
    gray_backbone_features: Tensor | None = None
    fusion_gate: Tensor | None = None


def _normalize_patterns(patterns: Sequence[str] | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    normalized = tuple(pattern.strip().lower() for pattern in patterns if pattern.strip())
    return tuple(dict.fromkeys(normalized))


class BaselineClassifier(nn.Module):
    """Strong single-branch baseline for XyloMaMi-Bench macro or micro classification."""

    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_BACKBONE,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
        pool_type: str | None = None,
        image_size: ImageSizeArg | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. Expected one of {sorted(SUPPORTED_BACKBONES)}."
            )

        self.backbone = build_backbone(
            backbone_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.pool = GlobalFeaturePooling(pool_type or self.backbone.default_pool_type)
        self.head = LinearClassifierHead(
            in_features=self.backbone.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.backbone_name = self.backbone.name
        self.image_size = self.backbone.image_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def forward_backbone(self, images: Tensor) -> Tensor:
        return self.backbone.forward_features(images)

    def pool_features(self, backbone_features: Tensor) -> Tensor:
        return self.pool(backbone_features)

    def forward_head(self, pooled_features: Tensor) -> Tensor:
        return self.head(pooled_features)

    def extract_features(self, images: Tensor, *, pooled: bool = True) -> Tensor:
        backbone_features = self.forward_backbone(images)
        if pooled:
            return self.pool_features(backbone_features)
        return backbone_features

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

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

    def build_loss(
        self,
        *,
        class_weights: Tensor | Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        return build_classification_loss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        images: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | BaselineClassifierOutput:
        backbone_features = self.forward_backbone(images)
        pooled_features = self.pool_features(backbone_features)
        logits = self.forward_head(pooled_features)
        if return_features:
            return BaselineClassifierOutput(
                logits=logits,
                pooled_features=pooled_features,
                backbone_features=backbone_features,
            )
        return logits


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return `(total_params, trainable_params)` for a module."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def is_backbone_frozen(backbone: TimmBackbone) -> bool:
    """Return True when all backbone parameters are frozen."""

    return all(not parameter.requires_grad for parameter in backbone.parameters())


class RGBGrayFusionClassifier(nn.Module):
    """Single-branch classifier with RGB + grayscale late fusion."""

    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_BACKBONE,
        gray_backbone_name: str | None = None,
        num_classes: int,
        pretrained: bool = True,
        gray_pretrained: bool | None = None,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
        pool_type: str | None = None,
        image_size: ImageSizeArg | None = None,
        fusion_hidden_dim: int | None = None,
        fusion_dropout: float = 0.0,
        fusion_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        resolved_gray_backbone_name = gray_backbone_name or backbone_name
        resolved_gray_pretrained = pretrained if gray_pretrained is None else gray_pretrained

        self.rgb_encoder = SingleViewFeatureEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            image_size=image_size,
            pool_type=pool_type,
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )
        self.gray_encoder = SingleViewFeatureEncoder(
            backbone_name=resolved_gray_backbone_name,
            pretrained=resolved_gray_pretrained,
            image_size=image_size,
            pool_type=pool_type,
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )
        if self.rgb_encoder.feature_dim != self.gray_encoder.feature_dim:
            raise ValueError(
                "RGB and gray encoders must expose the same feature_dim, got "
                f"{self.rgb_encoder.feature_dim} and {self.gray_encoder.feature_dim}."
            )

        self.backbone = nn.ModuleList([self.rgb_encoder.backbone, self.gray_encoder.backbone])
        self.fusion = ResidualFeatureFusion(
            self.rgb_encoder.feature_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            residual_scale=fusion_residual_scale,
        )
        self.head = LinearClassifierHead(
            in_features=self.rgb_encoder.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.backbone_name = self.rgb_encoder.backbone_name
        self.gray_backbone_name = self.gray_encoder.backbone_name
        self.image_size = self.rgb_encoder.image_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.fusion_hidden_dim = self.fusion.hidden_dim
        self.fusion_dropout = fusion_dropout
        self.fusion_residual_scale = fusion_residual_scale

    @property
    def feature_dim(self) -> int:
        return self.rgb_encoder.feature_dim

    def _build_gray_view(self, images: Tensor) -> Tensor:
        return normalized_rgb_to_grayscale_triplet(images)

    def extract_features(self, images: Tensor, *, pooled: bool = True) -> Tensor:
        rgb_backbone_features, rgb_pooled_features = self.rgb_encoder.encode(images)
        gray_backbone_features, gray_pooled_features = self.gray_encoder.encode(
            self._build_gray_view(images)
        )
        if pooled:
            return self.fusion(rgb_pooled_features, gray_pooled_features).fused_features
        return torch.stack([rgb_backbone_features, gray_backbone_features], dim=1)

    def build_loss(
        self,
        *,
        class_weights: Tensor | Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        return build_classification_loss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        images: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | BaselineClassifierOutput:
        rgb_backbone_features, rgb_pooled_features = self.rgb_encoder.encode(images)
        gray_images = self._build_gray_view(images)
        gray_backbone_features, gray_pooled_features = self.gray_encoder.encode(gray_images)
        fusion_output = self.fusion(rgb_pooled_features, gray_pooled_features)
        fused_pooled_features = fusion_output.fused_features
        logits = self.head(fused_pooled_features)
        if return_features:
            return BaselineClassifierOutput(
                logits=logits,
                pooled_features=fused_pooled_features,
                backbone_features=rgb_backbone_features,
                rgb_pooled_features=rgb_pooled_features,
                gray_pooled_features=gray_pooled_features,
                rgb_backbone_features=rgb_backbone_features,
                gray_backbone_features=gray_backbone_features,
                fusion_gate=fusion_output.gate,
            )
        return logits
