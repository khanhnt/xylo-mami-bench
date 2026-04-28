"""Backbone wrappers for XyloMaMi-Bench baseline classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor, nn

try:
    import timm
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    raise ModuleNotFoundError(
        "timm is required for XyloMaMi-Bench model backbones. "
        "Install it with `pip install timm`."
    ) from exc

SUPPORTED_BACKBONES = frozenset(
    {
        "convnext_small",
        "convnext_base",
        "densenet121",
        "resnet50",
        "vit_small_patch16_224",
    }
)
SEQUENCE_BACKBONES = frozenset({"vit_small_patch16_224"})
DEFAULT_BACKBONE = "convnext_small"
ImageSizeArg = int | tuple[int, int]


@dataclass(frozen=True)
class BackboneMetadata:
    """Static metadata describing a backbone's output contract."""

    name: str
    feature_dim: int
    feature_layout: str
    default_pool_type: str


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip()
    if normalized not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Expected one of {sorted(SUPPORTED_BACKBONES)}."
        )
    return normalized


def _infer_feature_dim(model: nn.Module) -> int:
    num_features = getattr(model, "num_features", None)
    if isinstance(num_features, int) and num_features > 0:
        return num_features

    feature_info = getattr(model, "feature_info", None)
    if feature_info is not None and hasattr(feature_info, "channels"):
        channels = feature_info.channels()
        if channels:
            return int(channels[-1])

    raise ValueError(
        "Unable to infer backbone feature_dim from the timm model. "
        "Expected `num_features` or `feature_info.channels()` to be available."
    )


def _build_metadata(name: str, model: nn.Module) -> BackboneMetadata:
    feature_layout = "sequence" if name in SEQUENCE_BACKBONES else "spatial"
    default_pool_type = "token" if feature_layout == "sequence" else "avg"
    return BackboneMetadata(
        name=name,
        feature_dim=_infer_feature_dim(model),
        feature_layout=feature_layout,
        default_pool_type=default_pool_type,
    )


def _normalize_image_size(image_size: ImageSizeArg | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size tuples must contain exactly two integers.")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive.")
    return (height, width)


class TimmBackbone(nn.Module):
    """Thin wrapper around a timm model with a stable feature interface."""

    def __init__(
        self,
        name: str = DEFAULT_BACKBONE,
        *,
        pretrained: bool = True,
        image_size: ImageSizeArg | None = None,
        timm_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        resolved_name = _normalize_backbone_name(name)
        model_kwargs = dict(timm_kwargs or {})
        model_kwargs.setdefault("num_classes", 0)
        model_kwargs.setdefault("global_pool", "")
        normalized_image_size = _normalize_image_size(image_size)
        if normalized_image_size is not None and resolved_name in SEQUENCE_BACKBONES:
            model_kwargs.setdefault("img_size", normalized_image_size)

        self.model = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            **model_kwargs,
        )
        self.metadata = _build_metadata(resolved_name, self.model)
        self.name = self.metadata.name
        self.pretrained = pretrained
        self.image_size = normalized_image_size
        self.feature_dim = self.metadata.feature_dim
        self.feature_layout = self.metadata.feature_layout
        self.default_pool_type = self.metadata.default_pool_type

    def forward_features(self, images: Tensor) -> Tensor:
        return self.model.forward_features(images)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_features(images)


def build_backbone(
    name: str = DEFAULT_BACKBONE,
    *,
    pretrained: bool = True,
    image_size: ImageSizeArg | None = None,
    timm_kwargs: Mapping[str, Any] | None = None,
) -> TimmBackbone:
    """Build a supported timm backbone for XyloMaMi-Bench."""

    return TimmBackbone(
        name=name,
        pretrained=pretrained,
        image_size=image_size,
        timm_kwargs=timm_kwargs,
    )
