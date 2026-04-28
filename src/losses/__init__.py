"""Loss functions for XyloMaMi-Bench baseline and alignment experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SupConLoss",
    "SUPPORTED_RELATION_POLICIES",
    "TaxonomyRelationLoss",
    "TaxonomyRelationMasks",
    "build_relaxed_negative_weights",
    "build_taxonomy_relation_masks",
]

_LAZY_IMPORTS = {
    "SupConLoss": ("src.losses.supcon", "SupConLoss"),
    "SUPPORTED_RELATION_POLICIES": (
        "src.losses.taxonomy_loss",
        "SUPPORTED_RELATION_POLICIES",
    ),
    "TaxonomyRelationLoss": ("src.losses.taxonomy_loss", "TaxonomyRelationLoss"),
    "TaxonomyRelationMasks": ("src.losses.taxonomy_loss", "TaxonomyRelationMasks"),
    "build_relaxed_negative_weights": (
        "src.losses.taxonomy_loss",
        "build_relaxed_negative_weights",
    ),
    "build_taxonomy_relation_masks": (
        "src.losses.taxonomy_loss",
        "build_taxonomy_relation_masks",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
