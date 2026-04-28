"""Dual-encoder alignment model for XyloMaMi-Bench macro-micro representation learning."""

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
    ProjectionMLP,
    build_classification_loss,
    default_projection_hidden_dims,
)
from src.models.rgbgray_fusion import (
    ResidualFeatureFusion,
    SingleViewFeatureEncoder,
    normalized_rgb_to_grayscale_triplet,
)


@dataclass(frozen=True)
class EncoderBranchOutput:
    """Structured output of one modality branch."""

    logits: Tensor
    embeddings: Tensor
    pooled_features: Tensor
    backbone_features: Tensor
    rgb_pooled_features: Tensor | None = None
    gray_pooled_features: Tensor | None = None
    rgb_backbone_features: Tensor | None = None
    gray_backbone_features: Tensor | None = None
    fusion_gate: Tensor | None = None


@dataclass(frozen=True)
class DualEncoderAlignOutput:
    """Structured output of the XyloMaMi-Bench dual-encoder model."""

    logits_macro: Tensor | None
    logits_micro: Tensor | None
    embeddings_macro: Tensor | None
    embeddings_micro: Tensor | None
    pooled_features_macro: Tensor | None
    pooled_features_micro: Tensor | None
    backbone_features_macro: Tensor | None
    backbone_features_micro: Tensor | None
    rgb_pooled_features_macro: Tensor | None
    gray_pooled_features_macro: Tensor | None
    rgb_pooled_features_micro: Tensor | None
    gray_pooled_features_micro: Tensor | None
    fusion_gate_macro: Tensor | None
    fusion_gate_micro: Tensor | None


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


class AlignmentEncoderBranch(nn.Module):
    """One modality branch of the XyloMaMi-Bench dual-encoder alignment model."""

    def __init__(
        self,
        *,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        image_size: ImageSizeArg | None = None,
        pool_type: str | None = None,
        projection_hidden_dims: Sequence[int] | None = None,
        projection_dim: int = 256,
        projection_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        resolved_backbone_name = _resolve_backbone_name(backbone_name)
        self.backbone = build_backbone(
            resolved_backbone_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.pool = GlobalFeaturePooling(pool_type or self.backbone.default_pool_type)
        hidden_dims = tuple(projection_hidden_dims) if projection_hidden_dims is not None else default_projection_hidden_dims(self.backbone.feature_dim)
        self.projection = ProjectionMLP(
            self.backbone.feature_dim,
            hidden_dims=hidden_dims,
            out_features=projection_dim,
            dropout=projection_dropout,
            l2_normalize=True,
        )
        self.classifier = LinearClassifierHead(
            in_features=self.backbone.feature_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        self.backbone_name = self.backbone.name
        self.num_classes = num_classes
        self.image_size = self.backbone.image_size
        self.feature_dim = self.backbone.feature_dim
        self.embedding_dim = projection_dim
        self.projection_hidden_dims = hidden_dims
        self.projection_dropout = projection_dropout
        self.classifier_dropout = classifier_dropout

        self.configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    def forward_backbone(self, images: Tensor) -> Tensor:
        return self.backbone.forward_features(images)

    def pool_features(self, backbone_features: Tensor) -> Tensor:
        return self.pool(backbone_features)

    def project_features(self, pooled_features: Tensor) -> Tensor:
        return self.projection(pooled_features)

    def classify_features(self, pooled_features: Tensor) -> Tensor:
        return self.classifier(pooled_features)

    def encode(self, images: Tensor) -> EncoderBranchOutput:
        backbone_features = self.forward_backbone(images)
        pooled_features = self.pool_features(backbone_features)
        embeddings = self.project_features(pooled_features)
        logits = self.classify_features(pooled_features)
        return EncoderBranchOutput(
            logits=logits,
            embeddings=embeddings,
            pooled_features=pooled_features,
            backbone_features=backbone_features,
        )

    def extract_embeddings(self, images: Tensor) -> Tensor:
        return self.encode(images).embeddings

    def build_classification_loss(
        self,
        *,
        class_weights: Tensor | Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        return build_classification_loss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    def projection_parameters(self) -> Iterable[nn.Parameter]:
        return self.projection.parameters()

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.classifier.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.projection.parameters()
        yield from self.classifier.parameters()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def freeze_branch(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_branch(self) -> None:
        for parameter in self.parameters():
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

    def parameter_collections(self, *, trainable_only: bool = True) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return branch parameter groups for later optimizer construction."""

        collections = {
            "backbone": tuple(self.backbone.parameters()),
            "projection": tuple(self.projection.parameters()),
            "classifier": tuple(self.classifier.parameters()),
        }
        if not trainable_only:
            return collections
        return {
            name: tuple(parameter for parameter in parameters if parameter.requires_grad)
            for name, parameters in collections.items()
        }

    def forward(self, images: Tensor) -> EncoderBranchOutput:
        return self.encode(images)


class DualEncoderAlign(nn.Module):
    """Macro-micro dual encoder for XyloMaMi-Bench."""

    def __init__(
        self,
        *,
        macro_num_classes: int,
        micro_num_classes: int,
        macro_backbone_name: str = DEFAULT_BACKBONE,
        micro_backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        macro_pretrained: bool | None = None,
        micro_pretrained: bool | None = None,
        image_size: ImageSizeArg | None = None,
        macro_image_size: ImageSizeArg | None = None,
        micro_image_size: ImageSizeArg | None = None,
        macro_pool_type: str | None = None,
        micro_pool_type: str | None = None,
        macro_projection_hidden_dims: Sequence[int] | None = None,
        micro_projection_hidden_dims: Sequence[int] | None = None,
        macro_projection_dim: int = 256,
        micro_projection_dim: int = 256,
        projection_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        freeze_macro_backbone: bool = False,
        freeze_micro_backbone: bool = False,
        macro_trainable_backbone_patterns: Sequence[str] | None = None,
        micro_trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if macro_num_classes <= 0:
            raise ValueError("macro_num_classes must be positive.")
        if micro_num_classes <= 0:
            raise ValueError("micro_num_classes must be positive.")

        resolved_macro_pretrained = pretrained if macro_pretrained is None else macro_pretrained
        resolved_micro_pretrained = pretrained if micro_pretrained is None else micro_pretrained
        resolved_macro_image_size = macro_image_size if macro_image_size is not None else image_size
        resolved_micro_image_size = micro_image_size if micro_image_size is not None else image_size

        self.macro_branch = AlignmentEncoderBranch(
            backbone_name=macro_backbone_name,
            num_classes=macro_num_classes,
            pretrained=resolved_macro_pretrained,
            image_size=resolved_macro_image_size,
            pool_type=macro_pool_type,
            projection_hidden_dims=macro_projection_hidden_dims,
            projection_dim=macro_projection_dim,
            projection_dropout=projection_dropout,
            classifier_dropout=classifier_dropout,
            freeze_backbone=freeze_macro_backbone,
            trainable_backbone_patterns=macro_trainable_backbone_patterns,
        )
        self.micro_branch = AlignmentEncoderBranch(
            backbone_name=micro_backbone_name,
            num_classes=micro_num_classes,
            pretrained=resolved_micro_pretrained,
            image_size=resolved_micro_image_size,
            pool_type=micro_pool_type,
            projection_hidden_dims=micro_projection_hidden_dims,
            projection_dim=micro_projection_dim,
            projection_dropout=projection_dropout,
            classifier_dropout=classifier_dropout,
            freeze_backbone=freeze_micro_backbone,
            trainable_backbone_patterns=micro_trainable_backbone_patterns,
        )

    @property
    def macro_feature_dim(self) -> int:
        return self.macro_branch.feature_dim

    @property
    def micro_feature_dim(self) -> int:
        return self.micro_branch.feature_dim

    @property
    def macro_embedding_dim(self) -> int:
        return self.macro_branch.embedding_dim

    @property
    def micro_embedding_dim(self) -> int:
        return self.micro_branch.embedding_dim

    def forward_macro(self, images: Tensor) -> EncoderBranchOutput:
        return self.macro_branch(images)

    def forward_micro(self, images: Tensor) -> EncoderBranchOutput:
        return self.micro_branch(images)

    def macro_backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.backbone_parameters()

    def micro_backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.backbone_parameters()

    def macro_projection_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.projection_parameters()

    def micro_projection_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.projection_parameters()

    def macro_classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.classifier_parameters()

    def micro_classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.classifier_parameters()

    def freeze_branch(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).freeze_branch()

    def unfreeze_branch(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).unfreeze_branch()

    def freeze_branch_backbone(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).freeze_backbone()

    def unfreeze_branch_backbone(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).unfreeze_backbone()

    def configure_branch_backbone_trainability(
        self,
        branch_name: str,
        *,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        self._resolve_branch(branch_name).configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    def parameter_collections(self, *, trainable_only: bool = True) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return optimizer-friendly parameter collections for each branch component."""

        grouped: dict[str, tuple[nn.Parameter, ...]] = {}
        for branch_name, branch in (("macro", self.macro_branch), ("micro", self.micro_branch)):
            for component_name, parameters in branch.parameter_collections(trainable_only=trainable_only).items():
                grouped[f"{branch_name}_{component_name}"] = parameters
        return grouped

    def forward(
        self,
        *,
        macro_images: Tensor | None = None,
        micro_images: Tensor | None = None,
    ) -> DualEncoderAlignOutput:
        if macro_images is None and micro_images is None:
            raise ValueError("At least one of macro_images or micro_images must be provided.")

        macro_output = self.macro_branch(macro_images) if macro_images is not None else None
        micro_output = self.micro_branch(micro_images) if micro_images is not None else None
        return DualEncoderAlignOutput(
            logits_macro=macro_output.logits if macro_output is not None else None,
            logits_micro=micro_output.logits if micro_output is not None else None,
            embeddings_macro=macro_output.embeddings if macro_output is not None else None,
            embeddings_micro=micro_output.embeddings if micro_output is not None else None,
            pooled_features_macro=macro_output.pooled_features if macro_output is not None else None,
            pooled_features_micro=micro_output.pooled_features if micro_output is not None else None,
            backbone_features_macro=macro_output.backbone_features if macro_output is not None else None,
            backbone_features_micro=micro_output.backbone_features if micro_output is not None else None,
            rgb_pooled_features_macro=(
                macro_output.rgb_pooled_features if macro_output is not None else None
            ),
            gray_pooled_features_macro=(
                macro_output.gray_pooled_features if macro_output is not None else None
            ),
            rgb_pooled_features_micro=(
                micro_output.rgb_pooled_features if micro_output is not None else None
            ),
            gray_pooled_features_micro=(
                micro_output.gray_pooled_features if micro_output is not None else None
            ),
            fusion_gate_macro=None,
            fusion_gate_micro=None,
        )

    def _resolve_branch(self, branch_name: str) -> AlignmentEncoderBranch:
        normalized = branch_name.strip().lower()
        if normalized == "macro":
            return self.macro_branch
        if normalized == "micro":
            return self.micro_branch
        raise ValueError("branch_name must be either 'macro' or 'micro'.")


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return `(total_params, trainable_params)` for a module."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def is_backbone_frozen(backbone: nn.Module) -> bool:
    """Return True when all backbone parameters are frozen."""

    return all(not parameter.requires_grad for parameter in backbone.parameters())


class RGBGrayFusionAlignmentEncoderBranch(nn.Module):
    """One modality branch with RGB + grayscale late fusion before classification/alignment."""

    def __init__(
        self,
        *,
        backbone_name: str,
        gray_backbone_name: str | None = None,
        num_classes: int,
        pretrained: bool = True,
        gray_pretrained: bool | None = None,
        image_size: ImageSizeArg | None = None,
        pool_type: str | None = None,
        projection_hidden_dims: Sequence[int] | None = None,
        projection_dim: int = 256,
        projection_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        fusion_hidden_dim: int | None = None,
        fusion_dropout: float = 0.0,
        fusion_residual_scale: float = 0.1,
        fusion_mode: str = "residual",
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

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

        hidden_dims = (
            tuple(projection_hidden_dims)
            if projection_hidden_dims is not None
            else default_projection_hidden_dims(self.rgb_encoder.feature_dim)
        )
        self.backbone = nn.ModuleList([self.rgb_encoder.backbone, self.gray_encoder.backbone])
        self.fusion = ResidualFeatureFusion(
            self.rgb_encoder.feature_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            residual_scale=fusion_residual_scale,
            mode=fusion_mode,
        )
        self.projection = ProjectionMLP(
            self.rgb_encoder.feature_dim,
            hidden_dims=hidden_dims,
            out_features=projection_dim,
            dropout=projection_dropout,
            l2_normalize=True,
        )
        self.classifier = LinearClassifierHead(
            in_features=self.rgb_encoder.feature_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        self.backbone_name = self.rgb_encoder.backbone_name
        self.gray_backbone_name = self.gray_encoder.backbone_name
        self.num_classes = num_classes
        self.image_size = self.rgb_encoder.image_size
        self.feature_dim = self.rgb_encoder.feature_dim
        self.embedding_dim = projection_dim
        self.projection_hidden_dims = hidden_dims
        self.projection_dropout = projection_dropout
        self.classifier_dropout = classifier_dropout
        self.fusion_hidden_dim = self.fusion.hidden_dim
        self.fusion_dropout = fusion_dropout
        self.fusion_residual_scale = fusion_residual_scale
        self.fusion_mode = fusion_mode

    def _build_gray_view(self, images: Tensor) -> Tensor:
        return normalized_rgb_to_grayscale_triplet(images)

    def forward_backbone(self, images: Tensor) -> Tensor:
        return self.rgb_encoder.forward_backbone(images)

    def pool_features(self, backbone_features: Tensor) -> Tensor:
        return self.rgb_encoder.pool_features(backbone_features)

    def project_features(self, pooled_features: Tensor) -> Tensor:
        return self.projection(pooled_features)

    def classify_features(self, pooled_features: Tensor) -> Tensor:
        return self.classifier(pooled_features)

    def encode(self, images: Tensor) -> EncoderBranchOutput:
        rgb_backbone_features, rgb_pooled_features = self.rgb_encoder.encode(images)
        gray_images = self._build_gray_view(images)
        gray_backbone_features, gray_pooled_features = self.gray_encoder.encode(gray_images)
        fusion_output = self.fusion(rgb_pooled_features, gray_pooled_features)
        fused_pooled_features = fusion_output.fused_features
        embeddings = self.project_features(fused_pooled_features)
        logits = self.classify_features(fused_pooled_features)
        return EncoderBranchOutput(
            logits=logits,
            embeddings=embeddings,
            pooled_features=fused_pooled_features,
            backbone_features=rgb_backbone_features,
            rgb_pooled_features=rgb_pooled_features,
            gray_pooled_features=gray_pooled_features,
            rgb_backbone_features=rgb_backbone_features,
            gray_backbone_features=gray_backbone_features,
            fusion_gate=fusion_output.gate,
        )

    def extract_embeddings(self, images: Tensor) -> Tensor:
        return self.encode(images).embeddings

    def build_classification_loss(
        self,
        *,
        class_weights: Tensor | Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        return build_classification_loss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.rgb_encoder.parameters()
        yield from self.gray_encoder.parameters()

    def projection_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.fusion.parameters()
        yield from self.projection.parameters()

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.classifier.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.fusion.parameters()
        yield from self.projection.parameters()
        yield from self.classifier.parameters()

    def freeze_backbone(self) -> None:
        self.rgb_encoder.freeze_backbone()
        self.gray_encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        self.rgb_encoder.unfreeze_backbone()
        self.gray_encoder.unfreeze_backbone()

    def freeze_branch(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_branch(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def configure_backbone_trainability(
        self,
        *,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        self.rgb_encoder.configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )
        self.gray_encoder.configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    def parameter_collections(self, *, trainable_only: bool = True) -> dict[str, tuple[nn.Parameter, ...]]:
        collections = {
            "backbone": tuple(self.backbone_parameters()),
            "projection": tuple(self.projection_parameters()),
            "classifier": tuple(self.classifier.parameters()),
        }
        if not trainable_only:
            return collections
        return {
            name: tuple(parameter for parameter in parameters if parameter.requires_grad)
            for name, parameters in collections.items()
        }

    def forward(self, images: Tensor) -> EncoderBranchOutput:
        return self.encode(images)


class RGBGrayFusionDualEncoderAlign(nn.Module):
    """Dual encoder align model with per-modality RGB + grayscale late fusion."""

    def __init__(
        self,
        *,
        macro_num_classes: int,
        micro_num_classes: int,
        macro_backbone_name: str = DEFAULT_BACKBONE,
        micro_backbone_name: str = DEFAULT_BACKBONE,
        macro_gray_backbone_name: str | None = None,
        micro_gray_backbone_name: str | None = None,
        pretrained: bool = True,
        macro_pretrained: bool | None = None,
        micro_pretrained: bool | None = None,
        macro_gray_pretrained: bool | None = None,
        micro_gray_pretrained: bool | None = None,
        image_size: ImageSizeArg | None = None,
        macro_image_size: ImageSizeArg | None = None,
        micro_image_size: ImageSizeArg | None = None,
        macro_pool_type: str | None = None,
        micro_pool_type: str | None = None,
        macro_projection_hidden_dims: Sequence[int] | None = None,
        micro_projection_hidden_dims: Sequence[int] | None = None,
        macro_projection_dim: int = 256,
        micro_projection_dim: int = 256,
        projection_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        macro_fusion_hidden_dim: int | None = None,
        micro_fusion_hidden_dim: int | None = None,
        fusion_dropout: float = 0.0,
        fusion_residual_scale: float = 0.1,
        fusion_mode: str = "residual",
        freeze_macro_backbone: bool = False,
        freeze_micro_backbone: bool = False,
        macro_trainable_backbone_patterns: Sequence[str] | None = None,
        micro_trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if macro_num_classes <= 0:
            raise ValueError("macro_num_classes must be positive.")
        if micro_num_classes <= 0:
            raise ValueError("micro_num_classes must be positive.")

        resolved_macro_pretrained = pretrained if macro_pretrained is None else macro_pretrained
        resolved_micro_pretrained = pretrained if micro_pretrained is None else micro_pretrained
        resolved_macro_image_size = macro_image_size if macro_image_size is not None else image_size
        resolved_micro_image_size = micro_image_size if micro_image_size is not None else image_size

        self.macro_branch = RGBGrayFusionAlignmentEncoderBranch(
            backbone_name=macro_backbone_name,
            gray_backbone_name=macro_gray_backbone_name,
            num_classes=macro_num_classes,
            pretrained=resolved_macro_pretrained,
            gray_pretrained=macro_gray_pretrained,
            image_size=resolved_macro_image_size,
            pool_type=macro_pool_type,
            projection_hidden_dims=macro_projection_hidden_dims,
            projection_dim=macro_projection_dim,
            projection_dropout=projection_dropout,
            classifier_dropout=classifier_dropout,
            fusion_hidden_dim=macro_fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            fusion_residual_scale=fusion_residual_scale,
            fusion_mode=fusion_mode,
            freeze_backbone=freeze_macro_backbone,
            trainable_backbone_patterns=macro_trainable_backbone_patterns,
        )
        self.micro_branch = RGBGrayFusionAlignmentEncoderBranch(
            backbone_name=micro_backbone_name,
            gray_backbone_name=micro_gray_backbone_name,
            num_classes=micro_num_classes,
            pretrained=resolved_micro_pretrained,
            gray_pretrained=micro_gray_pretrained,
            image_size=resolved_micro_image_size,
            pool_type=micro_pool_type,
            projection_hidden_dims=micro_projection_hidden_dims,
            projection_dim=micro_projection_dim,
            projection_dropout=projection_dropout,
            classifier_dropout=classifier_dropout,
            fusion_hidden_dim=micro_fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            fusion_residual_scale=fusion_residual_scale,
            fusion_mode=fusion_mode,
            freeze_backbone=freeze_micro_backbone,
            trainable_backbone_patterns=micro_trainable_backbone_patterns,
        )

    @property
    def macro_feature_dim(self) -> int:
        return self.macro_branch.feature_dim

    @property
    def micro_feature_dim(self) -> int:
        return self.micro_branch.feature_dim

    @property
    def macro_embedding_dim(self) -> int:
        return self.macro_branch.embedding_dim

    @property
    def micro_embedding_dim(self) -> int:
        return self.micro_branch.embedding_dim

    def forward_macro(self, images: Tensor) -> EncoderBranchOutput:
        return self.macro_branch(images)

    def forward_micro(self, images: Tensor) -> EncoderBranchOutput:
        return self.micro_branch(images)

    def macro_backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.backbone_parameters()

    def micro_backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.backbone_parameters()

    def macro_projection_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.projection_parameters()

    def micro_projection_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.projection_parameters()

    def macro_classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.macro_branch.classifier_parameters()

    def micro_classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.micro_branch.classifier_parameters()

    def freeze_branch(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).freeze_branch()

    def unfreeze_branch(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).unfreeze_branch()

    def freeze_branch_backbone(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).freeze_backbone()

    def unfreeze_branch_backbone(self, branch_name: str) -> None:
        self._resolve_branch(branch_name).unfreeze_backbone()

    def configure_branch_backbone_trainability(
        self,
        branch_name: str,
        *,
        freeze_backbone: bool = False,
        trainable_backbone_patterns: Sequence[str] | None = None,
    ) -> None:
        self._resolve_branch(branch_name).configure_backbone_trainability(
            freeze_backbone=freeze_backbone,
            trainable_backbone_patterns=trainable_backbone_patterns,
        )

    def parameter_collections(self, *, trainable_only: bool = True) -> dict[str, tuple[nn.Parameter, ...]]:
        grouped: dict[str, tuple[nn.Parameter, ...]] = {}
        for branch_name, branch in (("macro", self.macro_branch), ("micro", self.micro_branch)):
            for component_name, parameters in branch.parameter_collections(trainable_only=trainable_only).items():
                grouped[f"{branch_name}_{component_name}"] = parameters
        return grouped

    def forward(
        self,
        *,
        macro_images: Tensor | None = None,
        micro_images: Tensor | None = None,
    ) -> DualEncoderAlignOutput:
        if macro_images is None and micro_images is None:
            raise ValueError("At least one of macro_images or micro_images must be provided.")

        macro_output = self.macro_branch(macro_images) if macro_images is not None else None
        micro_output = self.micro_branch(micro_images) if micro_images is not None else None
        return DualEncoderAlignOutput(
            logits_macro=macro_output.logits if macro_output is not None else None,
            logits_micro=micro_output.logits if micro_output is not None else None,
            embeddings_macro=macro_output.embeddings if macro_output is not None else None,
            embeddings_micro=micro_output.embeddings if micro_output is not None else None,
            pooled_features_macro=macro_output.pooled_features if macro_output is not None else None,
            pooled_features_micro=micro_output.pooled_features if micro_output is not None else None,
            backbone_features_macro=macro_output.backbone_features if macro_output is not None else None,
            backbone_features_micro=micro_output.backbone_features if micro_output is not None else None,
            rgb_pooled_features_macro=(
                macro_output.rgb_pooled_features if macro_output is not None else None
            ),
            gray_pooled_features_macro=(
                macro_output.gray_pooled_features if macro_output is not None else None
            ),
            rgb_pooled_features_micro=(
                micro_output.rgb_pooled_features if micro_output is not None else None
            ),
            gray_pooled_features_micro=(
                micro_output.gray_pooled_features if micro_output is not None else None
            ),
            fusion_gate_macro=(
                macro_output.fusion_gate if macro_output is not None else None
            ),
            fusion_gate_micro=(
                micro_output.fusion_gate if micro_output is not None else None
            ),
        )

    def _resolve_branch(
        self,
        branch_name: str,
    ) -> RGBGrayFusionAlignmentEncoderBranch:
        normalized = branch_name.strip().lower()
        if normalized == "macro":
            return self.macro_branch
        if normalized == "micro":
            return self.micro_branch
        raise ValueError("branch_name must be either 'macro' or 'micro'.")
