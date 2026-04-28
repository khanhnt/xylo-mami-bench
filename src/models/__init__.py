"""Model utilities for XyloMaMi-Bench.

This package uses lazy imports so metadata-only tooling does not eagerly import
heavy model dependencies such as timm.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_BACKBONE",
    "SUPPORTED_BACKBONES",
    "BackboneMetadata",
    "TimmBackbone",
    "build_backbone",
    "BaselineClassifier",
    "BaselineClassifierOutput",
    "AlignmentEncoderBranch",
    "DualEncoderAlign",
    "DualEncoderAlignOutput",
    "EncoderBranchOutput",
    "count_parameters",
    "GlobalFeaturePooling",
    "LinearClassifierHead",
    "ProjectionMLP",
    "build_classification_loss",
    "cross_entropy_loss",
    "default_projection_hidden_dims",
]

_LAZY_IMPORTS = {
    "DEFAULT_BACKBONE": ("src.models.backbones", "DEFAULT_BACKBONE"),
    "SUPPORTED_BACKBONES": ("src.models.backbones", "SUPPORTED_BACKBONES"),
    "BackboneMetadata": ("src.models.backbones", "BackboneMetadata"),
    "TimmBackbone": ("src.models.backbones", "TimmBackbone"),
    "build_backbone": ("src.models.backbones", "build_backbone"),
    "BaselineClassifier": ("src.models.baseline_classifier", "BaselineClassifier"),
    "BaselineClassifierOutput": (
        "src.models.baseline_classifier",
        "BaselineClassifierOutput",
    ),
    "count_parameters": ("src.models.baseline_classifier", "count_parameters"),
    "AlignmentEncoderBranch": ("src.models.dual_encoder_align", "AlignmentEncoderBranch"),
    "DualEncoderAlign": ("src.models.dual_encoder_align", "DualEncoderAlign"),
    "DualEncoderAlignOutput": (
        "src.models.dual_encoder_align",
        "DualEncoderAlignOutput",
    ),
    "EncoderBranchOutput": ("src.models.dual_encoder_align", "EncoderBranchOutput"),
    "GlobalFeaturePooling": ("src.models.heads", "GlobalFeaturePooling"),
    "LinearClassifierHead": ("src.models.heads", "LinearClassifierHead"),
    "ProjectionMLP": ("src.models.heads", "ProjectionMLP"),
    "build_classification_loss": ("src.models.heads", "build_classification_loss"),
    "cross_entropy_loss": ("src.models.heads", "cross_entropy_loss"),
    "default_projection_hidden_dims": ("src.models.heads", "default_projection_hidden_dims"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
