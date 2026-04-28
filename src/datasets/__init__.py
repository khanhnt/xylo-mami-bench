"""Dataset utilities for XyloMaMi-Bench.

This package intentionally avoids importing heavy deep-learning dependencies at
module import time so metadata-only scripts such as taxonomy audits and split
generation can run in lightweight environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AlignmentBatchSampler",
    "DATASET_MODES",
    "LabelMapping",
    "ManifestDataset",
    "build_label_mapping_from_csvs",
    "build_label_mapping_from_samples",
    "build_transform_bundle",
    "build_transforms",
    "load_label_mapping",
    "save_label_mapping",
]


_LAZY_IMPORTS = {
    "AlignmentBatchSampler": ("src.datasets.paired_sampler", "AlignmentBatchSampler"),
    "DATASET_MODES": ("src.datasets.manifest_dataset", "DATASET_MODES"),
    "LabelMapping": ("src.datasets.manifest_dataset", "LabelMapping"),
    "ManifestDataset": ("src.datasets.manifest_dataset", "ManifestDataset"),
    "build_label_mapping_from_csvs": (
        "src.datasets.manifest_dataset",
        "build_label_mapping_from_csvs",
    ),
    "build_label_mapping_from_samples": (
        "src.datasets.manifest_dataset",
        "build_label_mapping_from_samples",
    ),
    "load_label_mapping": ("src.datasets.manifest_dataset", "load_label_mapping"),
    "save_label_mapping": ("src.datasets.manifest_dataset", "save_label_mapping"),
    "build_transform_bundle": ("src.datasets.transforms", "build_transform_bundle"),
    "build_transforms": ("src.datasets.transforms", "build_transforms"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

