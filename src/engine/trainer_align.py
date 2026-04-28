"""Training and evaluation utilities for the XyloMaMi-Bench dual-encoder model.

This module implements the XyloMaMi-Bench v2 alignment trainer with:

1. Branch-wise warm-start loading from baseline checkpoints.
2. Config-driven staged training.
3. Branch-aware optimizer groups.
4. Configurable checkpoint selection metrics.
5. Exact-only, genus-soft, relaxed-negative, and causal-control relation policies.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import (
    LabelMapping,
    ManifestDataset,
    build_label_mapping_from_csvs,
    build_label_mapping_from_species,
)
from src.datasets.paired_sampler import AlignmentBatchSampler
from src.datasets.transforms import build_transforms
from src.engine.metrics import ClassificationMetricAccumulator, ClassificationMetricsSummary
from src.losses.supcon import SupConLoss
from src.losses.taxonomy_loss import (
    SUPPORTED_RELATION_POLICIES,
    TaxonomyRelationLoss,
    TaxonomyRelationMasks,
    build_relaxed_negative_weights,
    build_taxonomy_relation_masks,
)
from src.models.dual_encoder_align import (
    AlignmentEncoderBranch,
    DualEncoderAlign,
    RGBGrayFusionAlignmentEncoderBranch,
    RGBGrayFusionDualEncoderAlign,
)
from src.models.heads import LinearClassifierHead, build_classification_loss
from src.utils.checkpoint_io import atomic_torch_save
from src.utils.seeding import set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_METRICS = frozenset(
    {
        "mean_balanced_accuracy",
        "weighted_mean_balanced_accuracy",
        "macro_balanced_accuracy",
        "micro_balanced_accuracy",
        "mean_top1_accuracy",
        "macro_top1_accuracy",
        "micro_top1_accuracy",
        "total_loss",
    }
)


@dataclass(frozen=True)
class SamplerConfig:
    classes_per_batch: int | None
    paired_species_per_batch: int | None
    min_cross_modal_pairs: int
    target_macro_fraction: float
    inverse_frequency_power: float
    drop_last: bool


@dataclass(frozen=True)
class DatasetConfig:
    train_split_csv: Path
    val_split_csv: Path
    test_split_csv: Path | None
    image_size: int
    input_mode: str
    macro_input_mode: str | None
    micro_input_mode: str | None
    batch_size: int
    eval_batch_size: int
    num_workers: int
    image_root_override: dict[str, Path]
    sampler: SamplerConfig


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    macro_backbone_name: str
    micro_backbone_name: str
    macro_gray_backbone_name: str | None
    micro_gray_backbone_name: str | None
    macro_num_classes: int
    micro_num_classes: int
    pretrained: bool
    macro_pretrained: bool | None
    micro_pretrained: bool | None
    macro_gray_pretrained: bool | None
    micro_gray_pretrained: bool | None
    macro_projection_hidden_dims: tuple[int, ...]
    micro_projection_hidden_dims: tuple[int, ...]
    macro_projection_dim: int
    micro_projection_dim: int
    projection_dropout: float
    classifier_dropout: float
    macro_fusion_hidden_dim: int | None
    micro_fusion_hidden_dim: int | None
    fusion_dropout: float
    fusion_residual_scale: float
    fusion_mode: str
    freeze_macro_backbone: bool
    freeze_micro_backbone: bool
    macro_trainable_backbone_patterns: tuple[str, ...]
    micro_trainable_backbone_patterns: tuple[str, ...]


@dataclass(frozen=True)
class WarmstartConfig:
    macro_checkpoint: Path | None
    micro_checkpoint: Path | None
    macro_rgb_checkpoint: Path | None
    macro_gray_checkpoint: Path | None
    micro_rgb_checkpoint: Path | None
    micro_gray_checkpoint: Path | None
    strict: bool


@dataclass(frozen=True)
class StageConfig:
    name: str
    epochs: int
    lambda_macro_ce: float
    lambda_micro_ce: float
    lambda_supcon: float
    lambda_tax: float
    relation_policy: str
    freeze_macro_encoder_epochs: int = 0
    freeze_macro_classifier_epochs: int = 0


@dataclass(frozen=True)
class OptimizerConfig:
    macro_encoder_lr: float
    micro_encoder_lr: float
    macro_projection_lr: float
    micro_projection_lr: float
    macro_classifier_lr: float
    micro_classifier_lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    min_lr_ratio: float
    warmup_steps: int | None
    warmup_epochs: int | None


@dataclass(frozen=True)
class MetricWeightConfig:
    macro: float
    micro: float

    def normalized(self, *, macro_available: bool, micro_available: bool) -> tuple[float, float]:
        weights = [
            self.macro if macro_available else 0.0,
            self.micro if micro_available else 0.0,
        ]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            return (0.0, 0.0)
        return (weights[0] / weight_sum, weights[1] / weight_sum)


TRADEOFF_CHECKPOINT_WEIGHTS = MetricWeightConfig(macro=0.8, micro=0.2)


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    max_steps: int | None
    amp: bool
    grad_clip_norm: float | None
    log_every_n_steps: int
    validate_every_n_epochs: int
    save_every_n_epochs: int
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    best_metric: str
    best_metric_mode: str
    best_metric_weights: MetricWeightConfig
    max_eval_batches: int | None
    resume_from: Path | None


@dataclass(frozen=True)
class LossConfig:
    macro_label_smoothing: float
    micro_label_smoothing: float
    macro_class_weights: str | tuple[float, ...] | None
    micro_class_weights: str | tuple[float, ...] | None
    supcon_temperature: float
    supcon_base_temperature: float
    taxonomy_similarity_floor: float
    taxonomy_pair_weight: float
    relaxed_negative_weight: float
    rgbgray_consistency_weight: float


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: Path | None
    split_csv: Path | None
    split_name: str
    dump_embeddings: bool


@dataclass(frozen=True)
class GenusAuxConfig:
    enabled: bool
    loss_weight: float
    macro_weight: float
    micro_weight: float
    dropout: float
    label_smoothing: float


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    output_dir: Path
    device: str
    dataset: DatasetConfig
    model: ModelConfig
    warmstart: WarmstartConfig
    stages: tuple[StageConfig, ...]
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    genus_aux: GenusAuxConfig
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class AlignExperimentPaths:
    output_dir: Path
    checkpoint_dir: Path
    report_dir: Path
    embedding_dir: Path
    log_path: Path
    metrics_history_path: Path
    macro_label_mapping_path: Path
    micro_label_mapping_path: Path
    joint_label_mapping_path: Path
    resolved_config_path: Path
    best_checkpoint_path: Path
    best_macro_checkpoint_path: Path
    best_tradeoff_checkpoint_path: Path
    last_checkpoint_path: Path


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    best_macro_metric: float | None = None
    best_macro_epoch: int | None = None
    best_tradeoff_metric: float | None = None
    best_tradeoff_epoch: int | None = None
    epochs_without_improvement: int = 0


@dataclass(frozen=True)
class AlignmentLossSummary:
    split_name: str
    stage_name: str
    total_loss: float | None
    macro_ce_loss: float | None
    micro_ce_loss: float | None
    supcon_loss: float | None
    taxonomy_loss: float | None
    rgbgray_consistency_loss: float | None
    macro_rgbgray_consistency_loss: float | None
    micro_rgbgray_consistency_loss: float | None
    genus_aux_loss: float | None
    macro_genus_aux_loss: float | None
    micro_genus_aux_loss: float | None
    macro_sample_count: int
    micro_sample_count: int
    exact_positive_pairs: int
    genus_soft_pairs: int
    exact_positive_anchors: int
    active_positive_pairs: int
    active_positive_anchors: int
    batch_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentEvaluationSummary:
    split_name: str
    stage_name: str
    macro: ClassificationMetricsSummary | None
    micro: ClassificationMetricsSummary | None
    alignment: AlignmentLossSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "stage_name": self.stage_name,
            "macro": self.macro.to_dict() if self.macro is not None else None,
            "micro": self.micro.to_dict() if self.micro is not None else None,
            "alignment": self.alignment.to_dict(),
        }


@dataclass(frozen=True)
class AlignmentStepOutput:
    total_loss: Tensor
    macro_ce_loss: Tensor
    micro_ce_loss: Tensor
    supcon_loss: Tensor
    taxonomy_loss: Tensor
    rgbgray_consistency_loss: Tensor
    macro_rgbgray_consistency_loss: Tensor
    micro_rgbgray_consistency_loss: Tensor
    genus_aux_loss: Tensor
    macro_genus_aux_loss: Tensor
    micro_genus_aux_loss: Tensor
    macro_sample_count: int
    micro_sample_count: int
    exact_positive_pairs: int
    genus_soft_pairs: int
    exact_positive_anchors: int
    active_positive_pairs: int
    active_positive_anchors: int


@dataclass(frozen=True)
class AlignmentPolicyOutput:
    """Policy-specific positive mask bundle for one alignment batch."""

    positive_mask: Tensor
    negative_weights: Tensor | None
    active_positive_pairs: int
    active_positive_anchors: int


@dataclass(frozen=True)
class AlignmentEvaluationArtifacts:
    summary: AlignmentEvaluationSummary
    macro_confusion_matrix_path: Path | None
    macro_per_class_report_path: Path | None
    macro_summary_json_path: Path | None
    micro_confusion_matrix_path: Path | None
    micro_per_class_report_path: Path | None
    micro_summary_json_path: Path | None
    alignment_summary_json_path: Path
    embeddings_dump_path: Path | None


@dataclass(frozen=True)
class WarmstartLoadResult:
    branch_name: str
    checkpoint_path: Path
    loaded_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]


def _categorize_state_keys(keys: Iterable[str]) -> dict[str, int]:
    counts = {
        "backbone": 0,
        "pool": 0,
        "projection": 0,
        "classifier": 0,
        "rgb_encoder": 0,
        "gray_encoder": 0,
        "fusion": 0,
        "other": 0,
    }
    for key in keys:
        component = key.split(".", 1)[0]
        if component in counts:
            counts[component] += 1
        else:
            counts["other"] += 1
    return counts


def serialize_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("raw_config", None)
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value in {None, ""}:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_required_project_path(path_value: str | Path | None, *, field_name: str) -> Path:
    resolved = _resolve_project_path(path_value)
    if resolved is None:
        raise ValueError(f"{field_name} must be set to a non-empty path.")
    return resolved


def _coerce_image_root_override(payload: Mapping[str, Any]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for modality in ("macro", "micro"):
        raw_value = payload.get(modality)
        resolved = _resolve_project_path(raw_value)
        if resolved is not None:
            overrides[modality] = resolved
    return overrides


def _coerce_class_weights(value: Any) -> str | tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none"}:
            return None
        if normalized != "balanced":
            raise ValueError("class weights must be null, 'balanced', or a numeric list.")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    raise ValueError("class weights must be null, 'balanced', or a numeric list.")


def _normalize_hidden_dims(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("projection hidden dims must be a list of integers.")
    dims = tuple(int(item) for item in value)
    if any(dim <= 0 for dim in dims):
        raise ValueError("projection hidden dims must contain only positive integers.")
    return dims


def _coerce_metric_weights(value: Any) -> MetricWeightConfig:
    if value is None:
        return MetricWeightConfig(macro=0.5, micro=0.5)
    if isinstance(value, Mapping):
        return MetricWeightConfig(
            macro=float(value.get("macro", 0.5)),
            micro=float(value.get("micro", 0.5)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return MetricWeightConfig(macro=float(value[0]), micro=float(value[1]))
    raise ValueError(
        "training.best_metric_weights must be null, a {macro, micro} mapping, or a two-item list."
    )


def _default_stage_relation_policy(loss_payload: Mapping[str, Any]) -> str:
    if bool(loss_payload.get("enable_taxonomy_soft_relations", False)):
        return "exact_plus_genus_soft"
    return "exact_only"


def _parse_stage_configs(
    raw_config: Mapping[str, Any],
    *,
    training_payload: Mapping[str, Any],
    loss_payload: Mapping[str, Any],
) -> tuple[StageConfig, ...]:
    default_relation_policy = _default_stage_relation_policy(loss_payload)
    default_stage_epochs = int(training_payload.get("max_epochs", training_payload.get("epochs", 30)))
    stages_payload = raw_config.get("stages")
    if stages_payload is None:
        return (
            StageConfig(
                name="main",
                epochs=max(1, default_stage_epochs),
                lambda_macro_ce=float(loss_payload.get("weight_macro_ce", 1.0)),
                lambda_micro_ce=float(loss_payload.get("weight_micro_ce", 1.0)),
                lambda_supcon=float(loss_payload.get("weight_supcon", 1.0)),
                lambda_tax=float(loss_payload.get("weight_taxonomy", 0.0)),
                relation_policy=default_relation_policy,
                freeze_macro_encoder_epochs=0,
                freeze_macro_classifier_epochs=0,
            ),
        )
    if not isinstance(stages_payload, Sequence) or isinstance(stages_payload, (str, bytes)):
        raise ValueError("stages must be a list of stage mappings.")

    stages: list[StageConfig] = []
    for index, entry in enumerate(stages_payload):
        if not isinstance(entry, Mapping):
            raise ValueError(f"stages[{index}] must be a mapping.")
        relation_policy = str(entry.get("relation_policy", default_relation_policy)).strip().lower()
        stages.append(
            StageConfig(
                name=str(entry.get("name", f"stage_{index + 1}")).strip() or f"stage_{index + 1}",
                epochs=int(entry.get("epochs", 0)),
                lambda_macro_ce=float(entry.get("lambda_macro_ce", loss_payload.get("weight_macro_ce", 1.0))),
                lambda_micro_ce=float(entry.get("lambda_micro_ce", loss_payload.get("weight_micro_ce", 1.0))),
                lambda_supcon=float(entry.get("lambda_supcon", loss_payload.get("weight_supcon", 1.0))),
                lambda_tax=float(entry.get("lambda_tax", loss_payload.get("weight_taxonomy", 0.0))),
                relation_policy=relation_policy,
                freeze_macro_encoder_epochs=int(entry.get("freeze_macro_encoder_epochs", 0)),
                freeze_macro_classifier_epochs=int(entry.get("freeze_macro_classifier_epochs", 0)),
            )
        )
    return tuple(stages)


def parse_experiment_config(raw_config: Mapping[str, Any]) -> ExperimentConfig:
    """Resolve a raw YAML config into a typed XyloMaMi-Bench alignment experiment config."""

    dataset_payload = _require_mapping(raw_config, "dataset")
    model_payload = _require_mapping(raw_config, "model")
    warmstart_payload = _require_mapping(raw_config, "warmstart")
    optimizer_payload = _require_mapping(raw_config, "optimizer")
    scheduler_payload = _require_mapping(raw_config, "scheduler")
    training_payload = _require_mapping(raw_config, "training")
    loss_payload = _require_mapping(raw_config, "loss")
    evaluation_payload = _require_mapping(raw_config, "evaluation")
    genus_aux_payload = _require_mapping(raw_config, "genus_aux")
    sampler_payload = _require_mapping(dataset_payload, "sampler")
    stages = _parse_stage_configs(raw_config, training_payload=training_payload, loss_payload=loss_payload)

    best_metric = str(training_payload.get("best_metric", "weighted_mean_balanced_accuracy")).strip()
    if best_metric not in BEST_METRICS:
        raise ValueError(
            f"training.best_metric must be one of {sorted(BEST_METRICS)}, got '{best_metric}'."
        )
    best_metric_mode = str(training_payload.get("best_metric_mode", "max")).strip().lower()
    if best_metric_mode not in {"max", "min"}:
        raise ValueError("training.best_metric_mode must be 'max' or 'min'.")

    raw_config_copy = json.loads(json.dumps(raw_config))
    raw_max_epochs = training_payload.get("max_epochs", training_payload.get("epochs"))
    resolved_max_epochs = (
        int(raw_max_epochs)
        if raw_max_epochs is not None
        else sum(stage.epochs for stage in stages)
    )

    config = ExperimentConfig(
        experiment_name=str(raw_config.get("experiment_name", "align_experiment")).strip(),
        seed=int(raw_config.get("seed", 42)),
        output_dir=_resolve_project_path(raw_config.get("output_dir")) or (PROJECT_ROOT / "outputs"),
        device=str(raw_config.get("device", "auto")).strip().lower(),
        dataset=DatasetConfig(
            train_split_csv=_resolve_required_project_path(
                dataset_payload.get("train_split_csv"),
                field_name="dataset.train_split_csv",
            ),
            val_split_csv=_resolve_required_project_path(
                dataset_payload.get("val_split_csv"),
                field_name="dataset.val_split_csv",
            ),
            test_split_csv=_resolve_project_path(dataset_payload.get("test_split_csv")),
            image_size=int(dataset_payload.get("image_size", 384)),
            input_mode=str(dataset_payload.get("input_mode", "rgb")).strip().lower(),
            macro_input_mode=(
                str(dataset_payload["macro_input_mode"]).strip().lower()
                if dataset_payload.get("macro_input_mode") is not None
                else None
            ),
            micro_input_mode=(
                str(dataset_payload["micro_input_mode"]).strip().lower()
                if dataset_payload.get("micro_input_mode") is not None
                else None
            ),
            batch_size=int(dataset_payload.get("batch_size", 16)),
            eval_batch_size=int(dataset_payload.get("eval_batch_size", dataset_payload.get("batch_size", 16))),
            num_workers=int(dataset_payload.get("num_workers", 4)),
            image_root_override=_coerce_image_root_override(
                _require_mapping(dataset_payload, "image_root_override")
            ),
            sampler=SamplerConfig(
                classes_per_batch=(
                    int(sampler_payload["classes_per_batch"])
                    if sampler_payload.get("classes_per_batch") is not None
                    else None
                ),
                paired_species_per_batch=(
                    int(sampler_payload["paired_species_per_batch"])
                    if sampler_payload.get("paired_species_per_batch") is not None
                    else None
                ),
                min_cross_modal_pairs=int(sampler_payload.get("min_cross_modal_pairs", 1)),
                target_macro_fraction=float(sampler_payload.get("target_macro_fraction", 0.5)),
                inverse_frequency_power=float(sampler_payload.get("inverse_frequency_power", 0.5)),
                drop_last=bool(sampler_payload.get("drop_last", False)),
            ),
        ),
        model=ModelConfig(
            architecture=str(model_payload.get("architecture", "standard")).strip().lower(),
            macro_backbone_name=str(model_payload.get("macro_backbone_name", "convnext_small")).strip(),
            micro_backbone_name=str(model_payload.get("micro_backbone_name", "convnext_small")).strip(),
            macro_gray_backbone_name=(
                str(model_payload["macro_gray_backbone_name"]).strip()
                if model_payload.get("macro_gray_backbone_name") is not None
                else None
            ),
            micro_gray_backbone_name=(
                str(model_payload["micro_gray_backbone_name"]).strip()
                if model_payload.get("micro_gray_backbone_name") is not None
                else None
            ),
            macro_num_classes=int(model_payload.get("macro_num_classes", 0)),
            micro_num_classes=int(model_payload.get("micro_num_classes", 0)),
            pretrained=bool(model_payload.get("pretrained", True)),
            macro_pretrained=(
                bool(model_payload["macro_pretrained"])
                if model_payload.get("macro_pretrained") is not None
                else None
            ),
            micro_pretrained=(
                bool(model_payload["micro_pretrained"])
                if model_payload.get("micro_pretrained") is not None
                else None
            ),
            macro_gray_pretrained=(
                bool(model_payload["macro_gray_pretrained"])
                if model_payload.get("macro_gray_pretrained") is not None
                else None
            ),
            micro_gray_pretrained=(
                bool(model_payload["micro_gray_pretrained"])
                if model_payload.get("micro_gray_pretrained") is not None
                else None
            ),
            macro_projection_hidden_dims=_normalize_hidden_dims(
                model_payload.get("macro_projection_hidden_dims", model_payload.get("projection_hidden_dims"))
            ),
            micro_projection_hidden_dims=_normalize_hidden_dims(
                model_payload.get("micro_projection_hidden_dims", model_payload.get("projection_hidden_dims"))
            ),
            macro_projection_dim=int(model_payload.get("macro_projection_dim", model_payload.get("projection_dim", 256))),
            micro_projection_dim=int(model_payload.get("micro_projection_dim", model_payload.get("projection_dim", 256))),
            projection_dropout=float(model_payload.get("projection_dropout", 0.1)),
            classifier_dropout=float(model_payload.get("classifier_dropout", 0.2)),
            macro_fusion_hidden_dim=(
                int(model_payload["macro_fusion_hidden_dim"])
                if model_payload.get("macro_fusion_hidden_dim") is not None
                else (
                    int(model_payload["fusion_hidden_dim"])
                    if model_payload.get("fusion_hidden_dim") is not None
                    else None
                )
            ),
            micro_fusion_hidden_dim=(
                int(model_payload["micro_fusion_hidden_dim"])
                if model_payload.get("micro_fusion_hidden_dim") is not None
                else (
                    int(model_payload["fusion_hidden_dim"])
                    if model_payload.get("fusion_hidden_dim") is not None
                    else None
                )
            ),
            fusion_dropout=float(model_payload.get("fusion_dropout", 0.0)),
            fusion_residual_scale=float(model_payload.get("fusion_residual_scale", 0.1)),
            fusion_mode=str(model_payload.get("fusion_mode", "residual")).strip().lower(),
            freeze_macro_backbone=bool(model_payload.get("freeze_macro_backbone", False)),
            freeze_micro_backbone=bool(model_payload.get("freeze_micro_backbone", False)),
            macro_trainable_backbone_patterns=tuple(
                str(item).strip()
                for item in model_payload.get("macro_trainable_backbone_patterns", [])
                if str(item).strip()
            ),
            micro_trainable_backbone_patterns=tuple(
                str(item).strip()
                for item in model_payload.get("micro_trainable_backbone_patterns", [])
                if str(item).strip()
            ),
        ),
        warmstart=WarmstartConfig(
            macro_checkpoint=_resolve_project_path(warmstart_payload.get("macro_checkpoint")),
            micro_checkpoint=_resolve_project_path(warmstart_payload.get("micro_checkpoint")),
            macro_rgb_checkpoint=_resolve_project_path(warmstart_payload.get("macro_rgb_checkpoint")),
            macro_gray_checkpoint=_resolve_project_path(warmstart_payload.get("macro_gray_checkpoint")),
            micro_rgb_checkpoint=_resolve_project_path(warmstart_payload.get("micro_rgb_checkpoint")),
            micro_gray_checkpoint=_resolve_project_path(warmstart_payload.get("micro_gray_checkpoint")),
            strict=bool(warmstart_payload.get("strict", False)),
        ),
        stages=stages,
        optimizer=OptimizerConfig(
            macro_encoder_lr=float(
                optimizer_payload.get(
                    "macro_encoder_lr",
                    optimizer_payload.get("backbone_lr", 5e-5),
                )
            ),
            micro_encoder_lr=float(
                optimizer_payload.get(
                    "micro_encoder_lr",
                    optimizer_payload.get("backbone_lr", 5e-5),
                )
            ),
            macro_projection_lr=float(
                optimizer_payload.get(
                    "macro_projection_lr",
                    optimizer_payload.get(
                        "projection_lr",
                        optimizer_payload.get("head_lr", 5e-4),
                    ),
                )
            ),
            micro_projection_lr=float(
                optimizer_payload.get(
                    "micro_projection_lr",
                    optimizer_payload.get(
                        "projection_lr",
                        optimizer_payload.get("head_lr", 5e-4),
                    ),
                )
            ),
            macro_classifier_lr=float(
                optimizer_payload.get(
                    "macro_classifier_lr",
                    optimizer_payload.get(
                        "classifier_lr",
                        optimizer_payload.get("head_lr", 5e-4),
                    ),
                )
            ),
            micro_classifier_lr=float(
                optimizer_payload.get(
                    "micro_classifier_lr",
                    optimizer_payload.get(
                        "classifier_lr",
                        optimizer_payload.get("head_lr", 5e-4),
                    ),
                )
            ),
            betas=tuple(float(item) for item in optimizer_payload.get("betas", (0.9, 0.999))),
            eps=float(optimizer_payload.get("eps", 1e-8)),
            weight_decay=float(optimizer_payload.get("weight_decay", 0.05)),
        ),
        scheduler=SchedulerConfig(
            min_lr_ratio=float(scheduler_payload.get("min_lr_ratio", 0.01)),
            warmup_steps=(
                int(scheduler_payload["warmup_steps"])
                if scheduler_payload.get("warmup_steps") is not None
                else None
            ),
            warmup_epochs=(
                int(scheduler_payload["warmup_epochs"])
                if scheduler_payload.get("warmup_epochs") is not None
                else None
            ),
        ),
        training=TrainingConfig(
            max_epochs=resolved_max_epochs,
            max_steps=(
                int(training_payload["max_steps"])
                if training_payload.get("max_steps") is not None
                else None
            ),
            amp=bool(training_payload.get("amp", True)),
            grad_clip_norm=(
                float(training_payload["grad_clip_norm"])
                if training_payload.get("grad_clip_norm") is not None
                else None
            ),
            log_every_n_steps=max(1, int(training_payload.get("log_every_n_steps", 20))),
            validate_every_n_epochs=max(1, int(training_payload.get("validate_every_n_epochs", 1))),
            save_every_n_epochs=max(1, int(training_payload.get("save_every_n_epochs", 1))),
            early_stopping_patience=(
                int(training_payload["early_stopping_patience"])
                if training_payload.get("early_stopping_patience") is not None
                else None
            ),
            early_stopping_min_delta=float(training_payload.get("early_stopping_min_delta", 0.0)),
            best_metric=best_metric,
            best_metric_mode=best_metric_mode,
            best_metric_weights=_coerce_metric_weights(training_payload.get("best_metric_weights")),
            max_eval_batches=(
                int(training_payload["max_eval_batches"])
                if training_payload.get("max_eval_batches") is not None
                else None
            ),
            resume_from=_resolve_project_path(training_payload.get("resume_from")),
        ),
        loss=LossConfig(
            macro_label_smoothing=float(loss_payload.get("macro_label_smoothing", 0.0)),
            micro_label_smoothing=float(loss_payload.get("micro_label_smoothing", 0.0)),
            macro_class_weights=_coerce_class_weights(loss_payload.get("macro_class_weights")),
            micro_class_weights=_coerce_class_weights(loss_payload.get("micro_class_weights")),
            supcon_temperature=float(loss_payload.get("supcon_temperature", 0.07)),
            supcon_base_temperature=float(loss_payload.get("supcon_base_temperature", 0.07)),
            taxonomy_similarity_floor=float(loss_payload.get("taxonomy_similarity_floor", 0.35)),
            taxonomy_pair_weight=float(loss_payload.get("taxonomy_pair_weight", 1.0)),
            relaxed_negative_weight=float(loss_payload.get("relaxed_negative_weight", 0.35)),
            rgbgray_consistency_weight=float(loss_payload.get("rgbgray_consistency_weight", 0.0)),
        ),
        evaluation=EvaluationConfig(
            checkpoint_path=_resolve_project_path(evaluation_payload.get("checkpoint_path")),
            split_csv=_resolve_project_path(evaluation_payload.get("split_csv")),
            split_name=str(evaluation_payload.get("split_name", "test")).strip().lower(),
            dump_embeddings=bool(evaluation_payload.get("dump_embeddings", False)),
        ),
        genus_aux=GenusAuxConfig(
            enabled=bool(genus_aux_payload.get("enabled", False)),
            loss_weight=float(genus_aux_payload.get("loss_weight", 0.0)),
            macro_weight=float(genus_aux_payload.get("macro_weight", 1.0)),
            micro_weight=float(genus_aux_payload.get("micro_weight", 1.0)),
            dropout=float(genus_aux_payload.get("dropout", 0.0)),
            label_smoothing=float(genus_aux_payload.get("label_smoothing", 0.0)),
        ),
        raw_config=raw_config_copy,
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate the resolved align config before any heavy runtime work starts."""

    if not config.experiment_name:
        raise ValueError("experiment_name must not be empty.")
    if config.dataset.image_size <= 0:
        raise ValueError("dataset.image_size must be positive.")
    if config.dataset.batch_size <= 0 or config.dataset.eval_batch_size <= 0:
        raise ValueError("dataset batch sizes must be positive.")
    if config.dataset.num_workers < 0:
        raise ValueError("dataset.num_workers must be >= 0.")
    for modality, root_path in config.dataset.image_root_override.items():
        if not root_path.exists():
            raise FileNotFoundError(
                f"dataset.image_root_override.{modality} does not exist: {root_path}"
            )
    if (
        config.dataset.sampler.classes_per_batch is not None
        and config.dataset.sampler.classes_per_batch <= 0
    ):
        raise ValueError("dataset.sampler.classes_per_batch must be positive when provided.")
    if (
        config.dataset.sampler.paired_species_per_batch is not None
        and config.dataset.sampler.paired_species_per_batch <= 0
    ):
        raise ValueError("dataset.sampler.paired_species_per_batch must be positive when provided.")
    if config.dataset.sampler.min_cross_modal_pairs < 0:
        raise ValueError("dataset.sampler.min_cross_modal_pairs must be >= 0.")
    if not 0.0 <= config.dataset.sampler.target_macro_fraction <= 1.0:
        raise ValueError("dataset.sampler.target_macro_fraction must be in [0, 1].")
    if config.dataset.sampler.inverse_frequency_power < 0:
        raise ValueError("dataset.sampler.inverse_frequency_power must be >= 0.")

    if config.model.architecture not in {"standard", "rgbgray_late_fusion"}:
        raise ValueError(
            "model.architecture must be 'standard' or 'rgbgray_late_fusion', "
            f"got '{config.model.architecture}'."
        )
    if config.model.macro_num_classes <= 0 or config.model.micro_num_classes <= 0:
        raise ValueError("model macro/micro num_classes must be positive.")
    if config.model.macro_projection_dim <= 0 or config.model.micro_projection_dim <= 0:
        raise ValueError("projection dims must be positive.")
    if config.model.macro_projection_dim != config.model.micro_projection_dim:
        raise ValueError(
            "Alignment training requires equal macro and micro projection dims, but got "
            f"{config.model.macro_projection_dim} and {config.model.micro_projection_dim}."
        )
    if not 0.0 <= config.model.projection_dropout < 1.0:
        raise ValueError("model.projection_dropout must be in [0, 1).")
    if not 0.0 <= config.model.classifier_dropout < 1.0:
        raise ValueError("model.classifier_dropout must be in [0, 1).")
    if config.model.macro_fusion_hidden_dim is not None and config.model.macro_fusion_hidden_dim <= 0:
        raise ValueError("model.macro_fusion_hidden_dim must be positive when provided.")
    if config.model.micro_fusion_hidden_dim is not None and config.model.micro_fusion_hidden_dim <= 0:
        raise ValueError("model.micro_fusion_hidden_dim must be positive when provided.")
    if not 0.0 <= config.model.fusion_dropout < 1.0:
        raise ValueError("model.fusion_dropout must be in [0, 1).")
    if config.model.fusion_residual_scale < 0:
        raise ValueError("model.fusion_residual_scale must be >= 0.")
    if config.model.fusion_mode not in {"residual", "gated"}:
        raise ValueError("model.fusion_mode must be either 'residual' or 'gated'.")

    if config.warmstart.macro_checkpoint is not None and not config.warmstart.macro_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.macro_checkpoint does not exist: {config.warmstart.macro_checkpoint}"
        )
    if config.warmstart.micro_checkpoint is not None and not config.warmstart.micro_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.micro_checkpoint does not exist: {config.warmstart.micro_checkpoint}"
        )
    if config.warmstart.macro_rgb_checkpoint is not None and not config.warmstart.macro_rgb_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.macro_rgb_checkpoint does not exist: {config.warmstart.macro_rgb_checkpoint}"
        )
    if config.warmstart.macro_gray_checkpoint is not None and not config.warmstart.macro_gray_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.macro_gray_checkpoint does not exist: {config.warmstart.macro_gray_checkpoint}"
        )
    if config.warmstart.micro_rgb_checkpoint is not None and not config.warmstart.micro_rgb_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.micro_rgb_checkpoint does not exist: {config.warmstart.micro_rgb_checkpoint}"
        )
    if config.warmstart.micro_gray_checkpoint is not None and not config.warmstart.micro_gray_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.micro_gray_checkpoint does not exist: {config.warmstart.micro_gray_checkpoint}"
        )

    if not config.stages:
        raise ValueError("At least one stage must be configured.")
    seen_stage_names: set[str] = set()
    for stage in config.stages:
        if not stage.name:
            raise ValueError("Each stage must have a non-empty name.")
        if stage.name in seen_stage_names:
            raise ValueError(f"Duplicate stage name: '{stage.name}'.")
        seen_stage_names.add(stage.name)
        if stage.epochs <= 0:
            raise ValueError(f"Stage '{stage.name}' must have a positive epoch count.")
        if stage.freeze_macro_encoder_epochs < 0:
            raise ValueError(
                f"Stage '{stage.name}' has negative freeze_macro_encoder_epochs."
            )
        if stage.freeze_macro_classifier_epochs < 0:
            raise ValueError(
                f"Stage '{stage.name}' has negative freeze_macro_classifier_epochs."
            )
        if stage.freeze_macro_encoder_epochs > stage.epochs:
            raise ValueError(
                f"Stage '{stage.name}' freeze_macro_encoder_epochs="
                f"{stage.freeze_macro_encoder_epochs} exceeds stage epochs={stage.epochs}."
            )
        if stage.freeze_macro_classifier_epochs > stage.epochs:
            raise ValueError(
                f"Stage '{stage.name}' freeze_macro_classifier_epochs="
                f"{stage.freeze_macro_classifier_epochs} exceeds stage epochs={stage.epochs}."
            )
        if stage.relation_policy not in SUPPORTED_RELATION_POLICIES:
            raise ValueError(
                f"Stage '{stage.name}' uses unsupported relation_policy '{stage.relation_policy}'. "
                f"Expected one of {sorted(SUPPORTED_RELATION_POLICIES)}."
            )
        for weight_name, weight_value in (
            ("lambda_macro_ce", stage.lambda_macro_ce),
            ("lambda_micro_ce", stage.lambda_micro_ce),
            ("lambda_supcon", stage.lambda_supcon),
            ("lambda_tax", stage.lambda_tax),
        ):
            if weight_value < 0:
                raise ValueError(f"Stage '{stage.name}' has negative {weight_name}.")
        if (
            stage.lambda_macro_ce
            + stage.lambda_micro_ce
            + stage.lambda_supcon
            + stage.lambda_tax
            <= 0
        ):
            raise ValueError(f"Stage '{stage.name}' has no active loss terms.")
        if stage.relation_policy == "exact_only" and stage.lambda_tax > 0:
            raise ValueError(
                f"Stage '{stage.name}' uses relation_policy=exact_only, so lambda_tax must be 0."
            )

    if (
        config.training.best_metric_weights.macro < 0
        or config.training.best_metric_weights.micro < 0
        or (
            config.training.best_metric_weights.macro
            + config.training.best_metric_weights.micro
            <= 0
        )
    ):
        raise ValueError("training.best_metric_weights must be non-negative and sum to > 0.")

    if config.optimizer.macro_encoder_lr <= 0 or config.optimizer.micro_encoder_lr <= 0:
        raise ValueError("optimizer encoder learning rates must be positive.")
    if (
        config.optimizer.macro_projection_lr <= 0
        or config.optimizer.micro_projection_lr <= 0
        or config.optimizer.macro_classifier_lr <= 0
        or config.optimizer.micro_classifier_lr <= 0
    ):
        raise ValueError(
            "optimizer projection/classifier learning rates must be positive for both branches."
        )
    if config.optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be >= 0.")
    if len(config.optimizer.betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values.")
    if any(beta < 0 or beta >= 1 for beta in config.optimizer.betas):
        raise ValueError("optimizer.betas values must be in [0, 1).")
    if config.optimizer.eps <= 0:
        raise ValueError("optimizer.eps must be positive.")

    if not 0.0 <= config.scheduler.min_lr_ratio <= 1.0:
        raise ValueError("scheduler.min_lr_ratio must be in [0, 1].")
    if config.scheduler.warmup_steps is not None and config.scheduler.warmup_steps < 0:
        raise ValueError("scheduler.warmup_steps must be >= 0.")
    if config.scheduler.warmup_epochs is not None and config.scheduler.warmup_epochs < 0:
        raise ValueError("scheduler.warmup_epochs must be >= 0.")
    if config.scheduler.warmup_steps is not None and config.scheduler.warmup_epochs is not None:
        raise ValueError("Specify only one of scheduler.warmup_steps or scheduler.warmup_epochs.")

    if config.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive.")
    if config.training.max_steps is not None and config.training.max_steps <= 0:
        raise ValueError("training.max_steps must be positive when provided.")
    if config.training.grad_clip_norm is not None and config.training.grad_clip_norm <= 0:
        raise ValueError("training.grad_clip_norm must be positive when provided.")
    if (
        config.training.early_stopping_patience is not None
        and config.training.early_stopping_patience < 0
    ):
        raise ValueError("training.early_stopping_patience must be >= 0.")
    if config.training.max_eval_batches is not None and config.training.max_eval_batches <= 0:
        raise ValueError("training.max_eval_batches must be positive when provided.")

    if not 0.0 <= config.loss.macro_label_smoothing < 1.0:
        raise ValueError("loss.macro_label_smoothing must be in [0, 1).")
    if not 0.0 <= config.loss.micro_label_smoothing < 1.0:
        raise ValueError("loss.micro_label_smoothing must be in [0, 1).")
    if config.loss.supcon_temperature <= 0 or config.loss.supcon_base_temperature <= 0:
        raise ValueError("SupCon temperatures must be positive.")
    if not -1.0 <= config.loss.taxonomy_similarity_floor <= 1.0:
        raise ValueError("loss.taxonomy_similarity_floor must be in [-1, 1].")
    if config.loss.taxonomy_pair_weight < 0:
        raise ValueError("loss.taxonomy_pair_weight must be >= 0.")
    if not 0.0 <= config.loss.relaxed_negative_weight <= 1.0:
        raise ValueError("loss.relaxed_negative_weight must be in [0, 1].")

    for field_name, split_path in (
        ("dataset.train_split_csv", config.dataset.train_split_csv),
        ("dataset.val_split_csv", config.dataset.val_split_csv),
    ):
        if not split_path.exists():
            raise FileNotFoundError(f"{field_name} does not exist: {split_path}")
    if config.dataset.test_split_csv is not None and not config.dataset.test_split_csv.exists():
        raise FileNotFoundError(
            f"dataset.test_split_csv does not exist: {config.dataset.test_split_csv}"
        )
    if config.training.resume_from is not None and not config.training.resume_from.exists():
        raise FileNotFoundError(
            f"training.resume_from does not exist: {config.training.resume_from}"
        )
    if (
        config.evaluation.checkpoint_path is not None
        and not config.evaluation.checkpoint_path.exists()
    ):
        raise FileNotFoundError(
            f"evaluation.checkpoint_path does not exist: {config.evaluation.checkpoint_path}"
        )
    if config.evaluation.split_csv is not None and not config.evaluation.split_csv.exists():
        raise FileNotFoundError(
            f"evaluation.split_csv does not exist: {config.evaluation.split_csv}"
        )
    if config.evaluation.split_name not in {"train", "val", "test"}:
        raise ValueError("evaluation.split_name must be one of: train, val, test.")

    if config.genus_aux.loss_weight < 0:
        raise ValueError("genus_aux.loss_weight must be >= 0.")
    if config.genus_aux.macro_weight < 0 or config.genus_aux.micro_weight < 0:
        raise ValueError("genus_aux.{macro_weight,micro_weight} must be >= 0.")
    if config.genus_aux.enabled and config.genus_aux.loss_weight <= 0:
        raise ValueError("genus_aux.enabled=true requires genus_aux.loss_weight > 0.")
    if not 0.0 <= config.genus_aux.dropout < 1.0:
        raise ValueError("genus_aux.dropout must be in [0, 1).")
    if not 0.0 <= config.genus_aux.label_smoothing < 1.0:
        raise ValueError("genus_aux.label_smoothing must be in [0, 1).")
    if config.loss.rgbgray_consistency_weight < 0:
        raise ValueError("loss.rgbgray_consistency_weight must be >= 0.")


def seed_everything(seed: int) -> dict[str, Any]:
    return set_global_seed(seed)


def _seed_metadata(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _summary_with_seed(payload: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    return {**_seed_metadata(seed), **dict(payload)}


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(normalized)


def prepare_experiment_paths(output_dir: Path) -> AlignExperimentPaths:
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    log_dir = output_dir / "logs"
    embedding_dir = output_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    return AlignExperimentPaths(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        report_dir=report_dir,
        embedding_dir=embedding_dir,
        log_path=log_dir / "train.log",
        metrics_history_path=report_dir / "metrics_history.jsonl",
        macro_label_mapping_path=output_dir / "macro_label_mapping.json",
        micro_label_mapping_path=output_dir / "micro_label_mapping.json",
        joint_label_mapping_path=output_dir / "joint_label_mapping.json",
        resolved_config_path=output_dir / "resolved_config.yaml",
        best_checkpoint_path=checkpoint_dir / "best.ckpt",
        best_macro_checkpoint_path=checkpoint_dir / "best_macro.ckpt",
        best_tradeoff_checkpoint_path=checkpoint_dir / "best_tradeoff.ckpt",
        last_checkpoint_path=checkpoint_dir / "last.ckpt",
    )


def setup_logger(log_path: Path, *, name: str = "xylomami_align") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_checkpoint_file(path: str | Path) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _autocast_context(device: torch.device, *, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _build_grad_scaler(*, device: torch.device, enabled: bool):
    scaler_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    if device.type != "cuda":
        enabled = False
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(scaler_device, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def _is_no_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    lowered_name = name.lower()
    return (
        parameter.ndim <= 1
        or lowered_name.endswith(".bias")
        or "norm" in lowered_name
        or ".bn" in lowered_name
    )


class WarmupCosineScheduler:
    """Step-based warmup + cosine scheduler with checkpointable state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0

    def _scale_for_step(self, step: int) -> float:
        clamped_step = min(max(step, 0), self.total_steps)
        if self.warmup_steps > 0 and clamped_step < self.warmup_steps:
            return max(
                1e-8,
                min(1.0, float(clamped_step + 1) / float(self.warmup_steps)),
            )
        if self.total_steps <= self.warmup_steps:
            # Very short max_steps runs can be entirely consumed by warmup.
            # In that case we should end at the base LR instead of collapsing to
            # the cosine floor on the final scheduled step.
            return 1.0
        progress = float(clamped_step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, step: int) -> None:
        self.last_step = step
        scale = self._scale_for_step(step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lrs(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.step(int(state_dict.get("last_step", 0)))


class AlignmentLossAccumulator:
    """Track classification and alignment losses across one split."""

    def __init__(self, *, split_name: str, stage_name: str) -> None:
        self.split_name = split_name
        self.stage_name = stage_name
        self.total_loss_sum = 0.0
        self.total_loss_weight = 0
        self.macro_ce_sum = 0.0
        self.macro_ce_weight = 0
        self.micro_ce_sum = 0.0
        self.micro_ce_weight = 0
        self.supcon_sum = 0.0
        self.supcon_weight = 0
        self.taxonomy_sum = 0.0
        self.taxonomy_weight = 0
        self.rgbgray_consistency_sum = 0.0
        self.rgbgray_consistency_weight = 0
        self.macro_rgbgray_consistency_sum = 0.0
        self.macro_rgbgray_consistency_weight = 0
        self.micro_rgbgray_consistency_sum = 0.0
        self.micro_rgbgray_consistency_weight = 0
        self.genus_aux_sum = 0.0
        self.genus_aux_weight = 0
        self.macro_genus_aux_sum = 0.0
        self.macro_genus_aux_weight = 0
        self.micro_genus_aux_sum = 0.0
        self.micro_genus_aux_weight = 0
        self.macro_sample_count = 0
        self.micro_sample_count = 0
        self.exact_positive_pairs = 0
        self.genus_soft_pairs = 0
        self.exact_positive_anchors = 0
        self.active_positive_pairs = 0
        self.active_positive_anchors = 0
        self.batch_count = 0

    def update(self, step: AlignmentStepOutput) -> None:
        batch_weight = step.macro_sample_count + step.micro_sample_count
        self.total_loss_sum += float(step.total_loss.item()) * batch_weight
        self.total_loss_weight += batch_weight
        self.macro_ce_sum += float(step.macro_ce_loss.item()) * step.macro_sample_count
        self.macro_ce_weight += step.macro_sample_count
        self.micro_ce_sum += float(step.micro_ce_loss.item()) * step.micro_sample_count
        self.micro_ce_weight += step.micro_sample_count
        self.supcon_sum += float(step.supcon_loss.item()) * max(1, step.active_positive_anchors)
        self.supcon_weight += max(0, step.active_positive_anchors)
        self.taxonomy_sum += float(step.taxonomy_loss.item()) * max(1, step.genus_soft_pairs)
        self.taxonomy_weight += max(0, step.genus_soft_pairs)
        self.rgbgray_consistency_sum += float(step.rgbgray_consistency_loss.item()) * batch_weight
        self.rgbgray_consistency_weight += batch_weight
        self.macro_rgbgray_consistency_sum += (
            float(step.macro_rgbgray_consistency_loss.item()) * step.macro_sample_count
        )
        self.macro_rgbgray_consistency_weight += step.macro_sample_count
        self.micro_rgbgray_consistency_sum += (
            float(step.micro_rgbgray_consistency_loss.item()) * step.micro_sample_count
        )
        self.micro_rgbgray_consistency_weight += step.micro_sample_count
        self.genus_aux_sum += float(step.genus_aux_loss.item()) * batch_weight
        self.genus_aux_weight += batch_weight
        self.macro_genus_aux_sum += float(step.macro_genus_aux_loss.item()) * step.macro_sample_count
        self.macro_genus_aux_weight += step.macro_sample_count
        self.micro_genus_aux_sum += float(step.micro_genus_aux_loss.item()) * step.micro_sample_count
        self.micro_genus_aux_weight += step.micro_sample_count
        self.macro_sample_count += step.macro_sample_count
        self.micro_sample_count += step.micro_sample_count
        self.exact_positive_pairs += step.exact_positive_pairs
        self.genus_soft_pairs += step.genus_soft_pairs
        self.exact_positive_anchors += step.exact_positive_anchors
        self.active_positive_pairs += step.active_positive_pairs
        self.active_positive_anchors += step.active_positive_anchors
        self.batch_count += 1

    def summary(self) -> AlignmentLossSummary:
        def _safe_mean(total: float, weight: int) -> float | None:
            return total / weight if weight > 0 else None

        return AlignmentLossSummary(
            split_name=self.split_name,
            stage_name=self.stage_name,
            total_loss=_safe_mean(self.total_loss_sum, self.total_loss_weight),
            macro_ce_loss=_safe_mean(self.macro_ce_sum, self.macro_ce_weight),
            micro_ce_loss=_safe_mean(self.micro_ce_sum, self.micro_ce_weight),
            supcon_loss=_safe_mean(self.supcon_sum, self.supcon_weight),
            taxonomy_loss=_safe_mean(self.taxonomy_sum, self.taxonomy_weight),
            rgbgray_consistency_loss=_safe_mean(
                self.rgbgray_consistency_sum,
                self.rgbgray_consistency_weight,
            ),
            macro_rgbgray_consistency_loss=_safe_mean(
                self.macro_rgbgray_consistency_sum,
                self.macro_rgbgray_consistency_weight,
            ),
            micro_rgbgray_consistency_loss=_safe_mean(
                self.micro_rgbgray_consistency_sum,
                self.micro_rgbgray_consistency_weight,
            ),
            genus_aux_loss=_safe_mean(self.genus_aux_sum, self.genus_aux_weight),
            macro_genus_aux_loss=_safe_mean(self.macro_genus_aux_sum, self.macro_genus_aux_weight),
            micro_genus_aux_loss=_safe_mean(self.micro_genus_aux_sum, self.micro_genus_aux_weight),
            macro_sample_count=self.macro_sample_count,
            micro_sample_count=self.micro_sample_count,
            exact_positive_pairs=self.exact_positive_pairs,
            genus_soft_pairs=self.genus_soft_pairs,
            exact_positive_anchors=self.exact_positive_anchors,
            active_positive_pairs=self.active_positive_pairs,
            active_positive_anchors=self.active_positive_anchors,
            batch_count=self.batch_count,
        )


def _encode_string_ids(values: Sequence[str]) -> Tensor:
    mapping: dict[str, int] = {}
    encoded: list[int] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return torch.tensor(encoded, dtype=torch.long)


def _rgbgray_consistency_loss(rgb_features: Tensor, gray_features: Tensor) -> Tensor:
    if rgb_features.shape != gray_features.shape:
        raise ValueError(
            "RGB and gray features must have matching shapes for consistency loss, got "
            f"{tuple(rgb_features.shape)} vs {tuple(gray_features.shape)}."
        )
    if rgb_features.ndim != 2:
        raise ValueError(
            "_rgbgray_consistency_loss expects [batch, feature_dim] tensors, got "
            f"{tuple(rgb_features.shape)}."
        )
    rgb_normalized = F.normalize(rgb_features, dim=1)
    gray_normalized = F.normalize(gray_features, dim=1)
    cosine_similarity = (rgb_normalized * gray_normalized).sum(dim=1)
    return (1.0 - cosine_similarity).mean()


def _square_cross_modal_mask_from_block(
    positive_block: Tensor,
    *,
    macro_count: int,
    micro_count: int,
) -> Tensor:
    if positive_block.ndim != 2 or positive_block.shape != (macro_count, micro_count):
        raise ValueError(
            "positive_block must have shape "
            f"{(macro_count, micro_count)}, got {tuple(positive_block.shape)}."
        )
    full_mask = positive_block.new_zeros((macro_count + micro_count, macro_count + micro_count))
    full_mask[:macro_count, macro_count:] = positive_block
    full_mask[macro_count:, :macro_count] = positive_block.T
    return full_mask


def _count_positive_pairs(positive_mask: Tensor) -> int:
    if positive_mask.ndim != 2 or positive_mask.shape[0] != positive_mask.shape[1]:
        raise ValueError("positive_mask must be square when counting active positive pairs.")
    return int((positive_mask > 0).sum().item() // 2)


def _count_positive_anchors(positive_mask: Tensor) -> int:
    if positive_mask.ndim != 2 or positive_mask.shape[0] != positive_mask.shape[1]:
        raise ValueError("positive_mask must be square when counting active positive anchors.")
    return int(((positive_mask > 0).sum(dim=1) > 0).sum().item())


def _build_species_neutral_negative_weights(
    masks: TaxonomyRelationMasks,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Ignore exact-species pairs while genus-only pairs act as positives."""

    weights = torch.ones_like(masks.cross_modal_mask, dtype=dtype)
    weights[masks.exact_species_mask] = 0.0
    return weights


def _find_forbidden_position_permutation(
    *,
    position_categories: Sequence[int],
    sample_categories: Sequence[int],
    seed: int,
) -> Tensor | None:
    """Assign each sample column to a source position with a forbidden-category constraint.

    The returned permutation `perm` is indexed by actual sample column, and
    `source_block[:, perm]` yields a structurally matched but semantically
    perturbed positive mask. We keep the row/column positive counts fixed while
    changing which micro sample receives each assignment.
    """

    position_values = tuple(int(value) for value in position_categories)
    sample_values = tuple(int(value) for value in sample_categories)
    if len(position_values) != len(sample_values):
        raise ValueError("position_categories and sample_categories must have matching lengths.")

    count = len(position_values)
    if count <= 1:
        return None

    candidates_by_sample: list[list[int]] = []
    for sample_index, sample_value in enumerate(sample_values):
        candidates = [
            position_index
            for position_index, position_value in enumerate(position_values)
            if position_value != sample_value
        ]
        if not candidates:
            return None
        candidates_by_sample.append(candidates)

    rng = random.Random(seed)
    sample_order = list(range(count))
    rng.shuffle(sample_order)
    sample_order.sort(key=lambda sample_index: len(candidates_by_sample[sample_index]))

    used_positions: set[int] = set()
    assignment: list[int] = [-1] * count

    def _backtrack(order_index: int) -> bool:
        if order_index >= count:
            return True
        sample_index = sample_order[order_index]
        candidates = [candidate for candidate in candidates_by_sample[sample_index] if candidate not in used_positions]
        rng.shuffle(candidates)
        for candidate in candidates:
            assignment[sample_index] = candidate
            used_positions.add(candidate)
            if _backtrack(order_index + 1):
                return True
            used_positions.remove(candidate)
            assignment[sample_index] = -1
        return False

    if not _backtrack(0):
        return None
    return torch.tensor(assignment, dtype=torch.long)


def _count_species_for_modality(dataset: ManifestDataset, *, modality: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for species, modality_to_indices in dataset.indices_by_species_and_modality.items():
        counts[species] = len(modality_to_indices.get(modality, ()))
    return counts


def _count_paired_species(dataset: ManifestDataset) -> int:
    return sum(
        1
        for modality_to_indices in dataset.indices_by_species_and_modality.values()
        if modality_to_indices.get("macro") and modality_to_indices.get("micro")
    )


def _count_cross_modal_shared_genera(dataset: ManifestDataset) -> int:
    macro_genera = {
        sample.genus for sample in dataset.samples if sample.modality == "macro" and sample.genus
    }
    micro_genera = {
        sample.genus for sample in dataset.samples if sample.modality == "micro" and sample.genus
    }
    return len(macro_genera & micro_genera)


def _build_genus_label_mapping(
    dataset: ManifestDataset,
    *,
    modality: str,
    label_space_name: str,
) -> LabelMapping:
    genera = [
        sample.genus
        for sample in dataset.samples
        if sample.modality == modality and sample.genus
    ]
    if not genera:
        raise ValueError(
            f"Cannot build a genus auxiliary mapping for modality '{modality}' because no genera were found."
        )
    return build_label_mapping_from_species(
        genera,
        label_space_name=label_space_name,
    )


def _resolve_branch_class_weights(
    *,
    configured_weights: str | tuple[float, ...] | None,
    label_mapping: LabelMapping,
    species_counts: Mapping[str, int],
    device: torch.device,
) -> Tensor | None:
    if configured_weights is None:
        return None
    if configured_weights == "balanced":
        counts = torch.zeros(label_mapping.num_classes, dtype=torch.float32)
        for species, index in label_mapping.species_to_index.items():
            counts[index] = float(species_counts.get(species, 0))
        total = float(counts.sum().item())
        weights = total / (label_mapping.num_classes * counts.clamp_min(1.0))
        weights[counts <= 0] = 0.0
        return weights.to(device)
    weights = torch.tensor(configured_weights, dtype=torch.float32)
    if weights.numel() != label_mapping.num_classes:
        raise ValueError(
            f"Configured {weights.numel()} class weights but expected {label_mapping.num_classes}."
        )
    return weights.to(device)


def build_joint_label_mapping(config: ExperimentConfig) -> LabelMapping:
    return build_label_mapping_from_csvs(
        [config.dataset.train_split_csv],
        mode="joint_alignment",
        label_space_name=f"{config.experiment_name}_joint_alignment",
    )


def build_branch_label_mappings(config: ExperimentConfig) -> tuple[LabelMapping, LabelMapping]:
    macro_mapping = build_label_mapping_from_csvs(
        [config.dataset.train_split_csv],
        mode="macro_classification",
        label_space_name=f"{config.experiment_name}_macro",
    )
    micro_mapping = build_label_mapping_from_csvs(
        [config.dataset.train_split_csv],
        mode="micro_classification",
        label_space_name=f"{config.experiment_name}_micro",
    )
    return macro_mapping, micro_mapping


def _build_dataset(
    split_csv: Path,
    *,
    split_name: str,
    config: ExperimentConfig,
    joint_label_mapping: LabelMapping,
) -> ManifestDataset:
    transform_map = build_transforms(
        split_name,
        config.dataset.image_size,
        input_mode=config.dataset.input_mode,
        macro_input_mode=config.dataset.macro_input_mode,
        micro_input_mode=config.dataset.micro_input_mode,
    )
    return ManifestDataset(
        split_csv,
        mode="joint_alignment",
        transform=transform_map,
        label_mapping=joint_label_mapping,
        label_space_name=joint_label_mapping.label_space_name,
        image_root_override=config.dataset.image_root_override,
    )


def build_dataloaders(
    config: ExperimentConfig,
    *,
    joint_label_mapping: LabelMapping,
) -> tuple[
    ManifestDataset,
    ManifestDataset,
    ManifestDataset | None,
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any] | None,
]:
    train_dataset = _build_dataset(
        config.dataset.train_split_csv,
        split_name="train",
        config=config,
        joint_label_mapping=joint_label_mapping,
    )
    val_dataset = _build_dataset(
        config.dataset.val_split_csv,
        split_name="val",
        config=config,
        joint_label_mapping=joint_label_mapping,
    )
    test_dataset = (
        _build_dataset(
            config.dataset.test_split_csv,
            split_name="test",
            config=config,
            joint_label_mapping=joint_label_mapping,
        )
        if config.dataset.test_split_csv is not None
        else None
    )

    dataloader_kwargs = {
        "num_workers": config.dataset.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.dataset.num_workers > 0,
    }

    train_batch_sampler = AlignmentBatchSampler(
        train_dataset,
        batch_size=config.dataset.batch_size,
        classes_per_batch=config.dataset.sampler.classes_per_batch,
        paired_species_per_batch=config.dataset.sampler.paired_species_per_batch,
        min_cross_modal_pairs=config.dataset.sampler.min_cross_modal_pairs,
        target_macro_fraction=config.dataset.sampler.target_macro_fraction,
        inverse_frequency_power=config.dataset.sampler.inverse_frequency_power,
        drop_last=config.dataset.sampler.drop_last,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        worker_init_fn=partial(_seed_worker, base_seed=config.seed),
        **dataloader_kwargs,
    )

    eval_kwargs = dict(dataloader_kwargs)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.seed + 10_000)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.eval_batch_size,
        shuffle=False,
        drop_last=False,
        generator=val_generator,
        worker_init_fn=partial(_seed_worker, base_seed=config.seed + 10_000),
        **eval_kwargs,
    )
    test_loader = None
    if test_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(config.seed + 20_000)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.dataset.eval_batch_size,
            shuffle=False,
            drop_last=False,
            generator=test_generator,
            worker_init_fn=partial(_seed_worker, base_seed=config.seed + 20_000),
            **eval_kwargs,
        )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(
    config: ExperimentConfig,
    *,
    pretrained_override: bool | None = None,
) -> DualEncoderAlign | RGBGrayFusionDualEncoderAlign:
    shared_pretrained = config.model.pretrained if pretrained_override is None else pretrained_override
    if config.model.architecture == "rgbgray_late_fusion":
        return RGBGrayFusionDualEncoderAlign(
            macro_num_classes=config.model.macro_num_classes,
            micro_num_classes=config.model.micro_num_classes,
            macro_backbone_name=config.model.macro_backbone_name,
            micro_backbone_name=config.model.micro_backbone_name,
            macro_gray_backbone_name=config.model.macro_gray_backbone_name,
            micro_gray_backbone_name=config.model.micro_gray_backbone_name,
            pretrained=shared_pretrained,
            macro_pretrained=config.model.macro_pretrained,
            micro_pretrained=config.model.micro_pretrained,
            macro_gray_pretrained=config.model.macro_gray_pretrained,
            micro_gray_pretrained=config.model.micro_gray_pretrained,
            image_size=config.dataset.image_size,
            macro_projection_hidden_dims=config.model.macro_projection_hidden_dims or None,
            micro_projection_hidden_dims=config.model.micro_projection_hidden_dims or None,
            macro_projection_dim=config.model.macro_projection_dim,
            micro_projection_dim=config.model.micro_projection_dim,
            projection_dropout=config.model.projection_dropout,
            classifier_dropout=config.model.classifier_dropout,
            macro_fusion_hidden_dim=config.model.macro_fusion_hidden_dim,
            micro_fusion_hidden_dim=config.model.micro_fusion_hidden_dim,
            fusion_dropout=config.model.fusion_dropout,
            fusion_residual_scale=config.model.fusion_residual_scale,
            fusion_mode=config.model.fusion_mode,
            freeze_macro_backbone=config.model.freeze_macro_backbone,
            freeze_micro_backbone=config.model.freeze_micro_backbone,
            macro_trainable_backbone_patterns=config.model.macro_trainable_backbone_patterns,
            micro_trainable_backbone_patterns=config.model.micro_trainable_backbone_patterns,
        )
    return DualEncoderAlign(
        macro_num_classes=config.model.macro_num_classes,
        micro_num_classes=config.model.micro_num_classes,
        macro_backbone_name=config.model.macro_backbone_name,
        micro_backbone_name=config.model.micro_backbone_name,
        pretrained=shared_pretrained,
        macro_pretrained=config.model.macro_pretrained,
        micro_pretrained=config.model.micro_pretrained,
        image_size=config.dataset.image_size,
        macro_projection_hidden_dims=config.model.macro_projection_hidden_dims or None,
        micro_projection_hidden_dims=config.model.micro_projection_hidden_dims or None,
        macro_projection_dim=config.model.macro_projection_dim,
        micro_projection_dim=config.model.micro_projection_dim,
        projection_dropout=config.model.projection_dropout,
        classifier_dropout=config.model.classifier_dropout,
        freeze_macro_backbone=config.model.freeze_macro_backbone,
        freeze_micro_backbone=config.model.freeze_micro_backbone,
        macro_trainable_backbone_patterns=config.model.macro_trainable_backbone_patterns,
        micro_trainable_backbone_patterns=config.model.micro_trainable_backbone_patterns,
    )


def _optimizer_component_for_parameter(name: str) -> str:
    if (
        name.startswith("macro_branch.backbone.")
        or name.startswith("macro_branch.pool.")
        or name.startswith("macro_branch.rgb_encoder.")
        or name.startswith("macro_branch.gray_encoder.")
    ):
        return "macro_encoder"
    if (
        name.startswith("micro_branch.backbone.")
        or name.startswith("micro_branch.pool.")
        or name.startswith("micro_branch.rgb_encoder.")
        or name.startswith("micro_branch.gray_encoder.")
    ):
        return "micro_encoder"
    if name.startswith("macro_branch.projection.") or name.startswith("macro_branch.fusion."):
        return "macro_projection"
    if name.startswith("micro_branch.projection.") or name.startswith("micro_branch.fusion."):
        return "micro_projection"
    if name.startswith("macro_branch.classifier."):
        return "macro_classifier"
    if name.startswith("micro_branch.classifier."):
        return "micro_classifier"
    if name.startswith("macro_genus_head."):
        return "macro_classifier"
    if name.startswith("micro_genus_head."):
        return "micro_classifier"
    raise ValueError(f"Unable to assign optimizer component for parameter '{name}'.")


def build_optimizer(
    model: DualEncoderAlign | RGBGrayFusionDualEncoderAlign,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    """Build AdamW with branch-aware encoder, projection, and classifier groups."""

    component_to_lr = {
        "macro_encoder": config.optimizer.macro_encoder_lr,
        "micro_encoder": config.optimizer.micro_encoder_lr,
        "macro_projection": config.optimizer.macro_projection_lr,
        "micro_projection": config.optimizer.micro_projection_lr,
        "macro_classifier": config.optimizer.macro_classifier_lr,
        "micro_classifier": config.optimizer.micro_classifier_lr,
    }
    parameter_groups: dict[tuple[str, bool], dict[str, Any]] = {}
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)

        component = _optimizer_component_for_parameter(name)
        use_decay = not _is_no_decay_parameter(name, parameter)
        key = (component, use_decay)
        if key not in parameter_groups:
            parameter_groups[key] = {
                "params": [],
                "lr": component_to_lr[component],
                "weight_decay": config.optimizer.weight_decay if use_decay else 0.0,
                "group_name": f"{component}_{'decay' if use_decay else 'no_decay'}",
                "component_name": component,
            }
        parameter_groups[key]["params"].append(parameter)

    if not parameter_groups:
        raise ValueError("No trainable parameters found when building optimizer.")

    return torch.optim.AdamW(
        list(parameter_groups.values()),
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )


def _extract_model_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    model_state = checkpoint.get("model_state")
    if isinstance(model_state, Mapping):
        return model_state
    if all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint  # pragma: no cover - compatibility path
    raise ValueError("Checkpoint does not contain a model_state mapping.")


def _validate_warmstart_checkpoint_for_branch(
    checkpoint: Mapping[str, Any],
    *,
    branch_name: str,
    expected_backbone_name: str,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        return
    checkpoint_model = checkpoint_config.get("model")
    if not isinstance(checkpoint_model, Mapping):
        return

    candidate_backbone_names = [
        checkpoint_model.get("backbone_name"),
        checkpoint_model.get(f"{branch_name}_backbone_name"),
    ]
    for candidate_backbone_name in candidate_backbone_names:
        if candidate_backbone_name is None:
            continue
        if str(candidate_backbone_name).strip() != expected_backbone_name:
            raise ValueError(
                f"Warm-start checkpoint backbone mismatch for branch '{branch_name}': "
                f"checkpoint uses '{candidate_backbone_name}' but the branch expects "
                f"'{expected_backbone_name}'."
            )


def _extract_branch_state_for_warmstart(
    checkpoint: Mapping[str, Any],
    *,
    branch_name: str,
) -> dict[str, Tensor]:
    source_state = _extract_model_state_from_checkpoint(checkpoint)
    branch_prefix = f"{branch_name}_branch."
    if any(str(key).startswith(branch_prefix) for key in source_state):
        extracted = {
            str(key)[len(branch_prefix):]: value
            for key, value in source_state.items()
            if str(key).startswith(branch_prefix)
        }
    else:
        extracted = {str(key): value for key, value in source_state.items()}

    normalized: dict[str, Tensor] = {}
    for key, value in extracted.items():
        normalized_key = key
        if normalized_key.startswith("head."):
            normalized_key = "classifier." + normalized_key[len("head."):]
        normalized[normalized_key] = value
    return normalized


def _load_branch_warmstart(
    *,
    branch_name: str,
    branch: AlignmentEncoderBranch,
    checkpoint_path: Path,
    strict: bool,
) -> WarmstartLoadResult:
    """Load one branch from a checkpoint with shape-safe compatibility checks."""

    checkpoint = load_checkpoint_file(checkpoint_path)
    _validate_warmstart_checkpoint_for_branch(
        checkpoint,
        branch_name=branch_name,
        expected_backbone_name=branch.backbone_name,
    )
    source_state = _extract_branch_state_for_warmstart(checkpoint, branch_name=branch_name)
    target_state = branch.state_dict()

    loadable: dict[str, Tensor] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape.append(
                f"{key} source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
            )
            continue
        loadable[key] = value

    if not loadable:
        raise ValueError(
            f"No compatible tensors were found when warm-starting branch '{branch_name}' "
            f"from checkpoint {checkpoint_path}."
        )
    if strict and (skipped_missing or skipped_shape):
        raise ValueError(
            f"Warm-start strict loading failed for branch '{branch_name}'. "
            f"Missing={skipped_missing}, shape_mismatch={skipped_shape}"
        )

    branch.load_state_dict(loadable, strict=False)
    return WarmstartLoadResult(
        branch_name=branch_name,
        checkpoint_path=checkpoint_path,
        loaded_keys=tuple(sorted(loadable)),
        skipped_missing_keys=tuple(sorted(skipped_missing)),
        skipped_shape_keys=tuple(sorted(skipped_shape)),
    )


def _load_rgbgray_fusion_warmstart(
    *,
    branch_name: str,
    branch: RGBGrayFusionAlignmentEncoderBranch,
    checkpoint_path: Path,
    source_view: str,
    load_classifier: bool,
    strict: bool,
) -> WarmstartLoadResult:
    checkpoint = load_checkpoint_file(checkpoint_path)
    expected_backbone_name = (
        branch.rgb_encoder.backbone_name if source_view == "rgb" else branch.gray_encoder.backbone_name
    )
    _validate_warmstart_checkpoint_for_branch(
        checkpoint,
        branch_name=branch_name,
        expected_backbone_name=expected_backbone_name,
    )
    source_state = _extract_branch_state_for_warmstart(checkpoint, branch_name=branch_name)
    target_state = branch.state_dict()
    target_prefix = "rgb_encoder." if source_view == "rgb" else "gray_encoder."

    loadable: dict[str, Tensor] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in source_state.items():
        target_key: str | None = None
        if key.startswith("backbone.") or key.startswith("pool."):
            target_key = target_prefix + key
        elif load_classifier and key.startswith("classifier."):
            target_key = key
        else:
            continue

        if target_key not in target_state:
            skipped_missing.append(target_key)
            continue
        if tuple(target_state[target_key].shape) != tuple(value.shape):
            skipped_shape.append(
                f"{target_key} source={tuple(value.shape)} target={tuple(target_state[target_key].shape)}"
            )
            continue
        loadable[target_key] = value

    if not loadable:
        raise ValueError(
            f"No compatible tensors were found when warm-starting fusion branch '{branch_name}' "
            f"({source_view}) from checkpoint {checkpoint_path}."
        )
    if strict and (skipped_missing or skipped_shape):
        raise ValueError(
            f"Warm-start strict loading failed for fusion branch '{branch_name}' ({source_view}). "
            f"Missing={skipped_missing}, shape_mismatch={skipped_shape}"
        )

    branch.load_state_dict(loadable, strict=False)
    return WarmstartLoadResult(
        branch_name=f"{branch_name}_{source_view}",
        checkpoint_path=checkpoint_path,
        loaded_keys=tuple(sorted(loadable)),
        skipped_missing_keys=tuple(sorted(skipped_missing)),
        skipped_shape_keys=tuple(sorted(skipped_shape)),
    )


def _validate_checkpoint_for_config(
    checkpoint: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    macro_label_mapping: LabelMapping,
    micro_label_mapping: LabelMapping,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, Mapping):
        checkpoint_model = checkpoint_config.get("model")
        if isinstance(checkpoint_model, Mapping):
            expected_backbones = {
                "macro_backbone_name": config.model.macro_backbone_name,
                "micro_backbone_name": config.model.micro_backbone_name,
                "macro_num_classes": config.model.macro_num_classes,
                "micro_num_classes": config.model.micro_num_classes,
            }
            for key, expected in expected_backbones.items():
                value = checkpoint_model.get(key)
                if value is not None and value != expected:
                    raise ValueError(
                        f"Checkpoint mismatch for {key}: checkpoint has {value!r} but config requests {expected!r}."
                    )
            expected_architecture = getattr(config.model, "architecture", None)
            checkpoint_architecture = checkpoint_model.get("architecture")
            if (
                expected_architecture is not None
                and checkpoint_architecture is not None
                and checkpoint_architecture != expected_architecture
            ):
                raise ValueError(
                    "Checkpoint mismatch for model architecture: checkpoint has "
                    f"{checkpoint_architecture!r} but config requests {expected_architecture!r}."
                )
            expected_fusion_mode = getattr(config.model, "fusion_mode", None)
            checkpoint_fusion_mode = checkpoint_model.get("fusion_mode")
            if (
                expected_fusion_mode is not None
                and checkpoint_fusion_mode is not None
                and checkpoint_fusion_mode != expected_fusion_mode
            ):
                raise ValueError(
                    "Checkpoint mismatch for fusion_mode: checkpoint has "
                    f"{checkpoint_fusion_mode!r} but config requests {expected_fusion_mode!r}."
                )
    if macro_label_mapping.num_classes != config.model.macro_num_classes:
        raise ValueError(
            f"Macro label mapping has {macro_label_mapping.num_classes} classes but config requests "
            f"{config.model.macro_num_classes}."
        )
    if micro_label_mapping.num_classes != config.model.micro_num_classes:
        raise ValueError(
            f"Micro label mapping has {micro_label_mapping.num_classes} classes but config requests "
            f"{config.model.micro_num_classes}."
        )


def _load_model_state_with_gate_compatibility(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    allow_optional_gate_mismatch: bool,
    logger: logging.Logger | None = None,
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    optional_gate_fragment = ".fusion.gate_network."

    missing_gate = [
        key for key in incompatible.missing_keys if optional_gate_fragment in key
    ]
    unexpected_gate = [
        key for key in incompatible.unexpected_keys if optional_gate_fragment in key
    ]
    missing_other = [
        key for key in incompatible.missing_keys if optional_gate_fragment not in key
    ]
    unexpected_other = [
        key for key in incompatible.unexpected_keys if optional_gate_fragment not in key
    ]

    if missing_other or unexpected_other:
        details: list[str] = []
        if missing_other:
            details.append(f"missing={missing_other}")
        if unexpected_other:
            details.append(f"unexpected={unexpected_other}")
        raise RuntimeError("Error(s) in loading state_dict: " + "; ".join(details))

    if (missing_gate or unexpected_gate) and not allow_optional_gate_mismatch:
        details = []
        if missing_gate:
            details.append(f"missing={missing_gate}")
        if unexpected_gate:
            details.append(f"unexpected={unexpected_gate}")
        raise RuntimeError(
            "Error(s) in loading state_dict for gated fusion parameters: " + "; ".join(details)
        )

    if logger is not None and (missing_gate or unexpected_gate):
        logger.info(
            "Checkpoint/model compatibility | ignored optional RGB-gray gate keys | "
            "missing=%s | unexpected=%s",
            missing_gate,
            unexpected_gate,
        )


class DualEncoderTrainer:
    """Trainer for the XyloMaMi-Bench dual-encoder model."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        logger: logging.Logger,
        paths: AlignExperimentPaths,
    ) -> None:
        self.config = config
        self.logger = logger
        self.paths = paths
        self.device = resolve_device(config.device)
        self.use_amp = bool(config.training.amp and self.device.type == "cuda")
        self.state = TrainerState()
        self.warmstart_reports: tuple[WarmstartLoadResult, ...] = ()

        seed_info = seed_everything(config.seed)
        self.logger.info(
            "Seed control | seed=%d | cudnn_deterministic=%s | cudnn_benchmark=%s",
            seed_info["seed"],
            seed_info["cudnn_deterministic"],
            seed_info["cudnn_benchmark"],
        )
        self.joint_label_mapping = build_joint_label_mapping(config)
        self.macro_label_mapping, self.micro_label_mapping = build_branch_label_mappings(config)
        if self.macro_label_mapping.num_classes != config.model.macro_num_classes:
            raise ValueError(
                f"Config macro_num_classes={config.model.macro_num_classes} but macro train split "
                f"resolves to {self.macro_label_mapping.num_classes} classes."
            )
        if self.micro_label_mapping.num_classes != config.model.micro_num_classes:
            raise ValueError(
                f"Config micro_num_classes={config.model.micro_num_classes} but micro train split "
                f"resolves to {self.micro_label_mapping.num_classes} classes."
            )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = build_dataloaders(
            config,
            joint_label_mapping=self.joint_label_mapping,
        )
        self._validate_alignment_dataset_compatibility()

        self.model = build_model(
            config,
            pretrained_override=False if config.training.resume_from is not None else None,
        ).to(self.device)
        self.macro_genus_label_mapping: LabelMapping | None = None
        self.micro_genus_label_mapping: LabelMapping | None = None
        self.macro_genus_criterion: nn.Module | None = None
        self.micro_genus_criterion: nn.Module | None = None
        self._initialize_genus_auxiliary()
        if config.training.resume_from is None:
            self._apply_warmstart()
        elif config.warmstart.macro_checkpoint is not None or config.warmstart.micro_checkpoint is not None:
            self.logger.info(
                "Ignoring warm-start checkpoints because training.resume_from is set to %s",
                config.training.resume_from,
            )

        self.optimizer = build_optimizer(self.model, config)
        self.total_training_steps = self._compute_total_training_steps()
        warmup_steps = (
            config.scheduler.warmup_steps
            if config.scheduler.warmup_steps is not None
            else (config.scheduler.warmup_epochs or 0) * max(1, len(self.train_loader))
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_steps=self.total_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=config.scheduler.min_lr_ratio,
        )
        self.scheduler.step(0)
        if self.total_training_steps <= self.scheduler.warmup_steps:
            self.logger.warning(
                "Scheduler is operating in warmup-only mode for this run | total_steps=%d | "
                "warmup_steps=%d | this is expected for very short max_steps runs.",
                self.total_training_steps,
                self.scheduler.warmup_steps,
            )
        self.scaler = _build_grad_scaler(device=self.device, enabled=self.use_amp)

        macro_species_counts = _count_species_for_modality(self.train_dataset, modality="macro")
        micro_species_counts = _count_species_for_modality(self.train_dataset, modality="micro")
        self.macro_class_weights = _resolve_branch_class_weights(
            configured_weights=config.loss.macro_class_weights,
            label_mapping=self.macro_label_mapping,
            species_counts=macro_species_counts,
            device=self.device,
        )
        self.micro_class_weights = _resolve_branch_class_weights(
            configured_weights=config.loss.micro_class_weights,
            label_mapping=self.micro_label_mapping,
            species_counts=micro_species_counts,
            device=self.device,
        )
        self.macro_criterion = self.model.macro_branch.build_classification_loss(
            class_weights=self.macro_class_weights,
            label_smoothing=config.loss.macro_label_smoothing,
        ).to(self.device)
        self.micro_criterion = self.model.micro_branch.build_classification_loss(
            class_weights=self.micro_class_weights,
            label_smoothing=config.loss.micro_label_smoothing,
        ).to(self.device)
        self.supcon_loss = SupConLoss(
            temperature=config.loss.supcon_temperature,
            base_temperature=config.loss.supcon_base_temperature,
            normalize_embeddings=False,
        )
        self.taxonomy_loss = TaxonomyRelationLoss(
            similarity_floor=config.loss.taxonomy_similarity_floor,
            normalize_embeddings=False,
        )

        self.joint_label_mapping.save(self.paths.joint_label_mapping_path)
        self.macro_label_mapping.save(self.paths.macro_label_mapping_path)
        self.micro_label_mapping.save(self.paths.micro_label_mapping_path)

        if config.training.resume_from is not None:
            self.load_checkpoint(config.training.resume_from)

    def _initialize_genus_auxiliary(self) -> None:
        if not self.config.genus_aux.enabled:
            return
        self.macro_genus_label_mapping = _build_genus_label_mapping(
            self.train_dataset,
            modality="macro",
            label_space_name=f"{self.config.experiment_name}_macro_genus",
        )
        self.micro_genus_label_mapping = _build_genus_label_mapping(
            self.train_dataset,
            modality="micro",
            label_space_name=f"{self.config.experiment_name}_micro_genus",
        )
        self.model.macro_genus_head = LinearClassifierHead(
            in_features=self.model.macro_branch.feature_dim,
            num_classes=self.macro_genus_label_mapping.num_classes,
            dropout=self.config.genus_aux.dropout,
        ).to(self.device)
        self.model.micro_genus_head = LinearClassifierHead(
            in_features=self.model.micro_branch.feature_dim,
            num_classes=self.micro_genus_label_mapping.num_classes,
            dropout=self.config.genus_aux.dropout,
        ).to(self.device)
        self.macro_genus_criterion = build_classification_loss(
            label_smoothing=self.config.genus_aux.label_smoothing,
        ).to(self.device)
        self.micro_genus_criterion = build_classification_loss(
            label_smoothing=self.config.genus_aux.label_smoothing,
        ).to(self.device)
        self.logger.info(
            "Genus auxiliary enabled | loss_weight=%.4f | weights(macro=%.3f,micro=%.3f) | "
            "macro_genera=%d | micro_genera=%d | dropout=%.3f",
            self.config.genus_aux.loss_weight,
            self.config.genus_aux.macro_weight,
            self.config.genus_aux.micro_weight,
            self.macro_genus_label_mapping.num_classes,
            self.micro_genus_label_mapping.num_classes,
            self.config.genus_aux.dropout,
        )

    def _apply_warmstart(self) -> None:
        reports: list[WarmstartLoadResult] = []
        if self.config.model.architecture == "rgbgray_late_fusion":
            if not isinstance(self.model.macro_branch, RGBGrayFusionAlignmentEncoderBranch):
                raise TypeError(
                    "rgbgray_late_fusion architecture requires RGBGrayFusionAlignmentEncoderBranch branches."
                )
            if self.config.warmstart.macro_rgb_checkpoint is not None:
                self.logger.info(
                    "Warm-start requested for macro RGB path from %s",
                    self.config.warmstart.macro_rgb_checkpoint,
                )
                reports.append(
                    _load_rgbgray_fusion_warmstart(
                        branch_name="macro",
                        branch=self.model.macro_branch,
                        checkpoint_path=self.config.warmstart.macro_rgb_checkpoint,
                        source_view="rgb",
                        load_classifier=True,
                        strict=self.config.warmstart.strict,
                    )
                )
            if self.config.warmstart.macro_gray_checkpoint is not None:
                self.logger.info(
                    "Warm-start requested for macro gray path from %s",
                    self.config.warmstart.macro_gray_checkpoint,
                )
                reports.append(
                    _load_rgbgray_fusion_warmstart(
                        branch_name="macro",
                        branch=self.model.macro_branch,
                        checkpoint_path=self.config.warmstart.macro_gray_checkpoint,
                        source_view="gray",
                        load_classifier=False,
                        strict=self.config.warmstart.strict,
                    )
                )
            if self.config.warmstart.micro_rgb_checkpoint is not None:
                self.logger.info(
                    "Warm-start requested for micro RGB path from %s",
                    self.config.warmstart.micro_rgb_checkpoint,
                )
                reports.append(
                    _load_rgbgray_fusion_warmstart(
                        branch_name="micro",
                        branch=self.model.micro_branch,
                        checkpoint_path=self.config.warmstart.micro_rgb_checkpoint,
                        source_view="rgb",
                        load_classifier=True,
                        strict=self.config.warmstart.strict,
                    )
                )
            if self.config.warmstart.micro_gray_checkpoint is not None:
                self.logger.info(
                    "Warm-start requested for micro gray path from %s",
                    self.config.warmstart.micro_gray_checkpoint,
                )
                reports.append(
                    _load_rgbgray_fusion_warmstart(
                        branch_name="micro",
                        branch=self.model.micro_branch,
                        checkpoint_path=self.config.warmstart.micro_gray_checkpoint,
                        source_view="gray",
                        load_classifier=False,
                        strict=self.config.warmstart.strict,
                    )
                )
            self.warmstart_reports = tuple(reports)
            for report in reports:
                loaded_counts = _categorize_state_keys(report.loaded_keys)
                skipped_missing_counts = _categorize_state_keys(report.skipped_missing_keys)
                skipped_shape_counts = _categorize_state_keys(
                    key.split(" source=", 1)[0] for key in report.skipped_shape_keys
                )
                self.logger.info(
                    "Warm-started %s from %s | loaded=%d | skipped_missing=%d | skipped_shape=%d",
                    report.branch_name,
                    report.checkpoint_path,
                    len(report.loaded_keys),
                    len(report.skipped_missing_keys),
                    len(report.skipped_shape_keys),
                )
                self.logger.info(
                    "Warm-start %s component summary | loaded=%s | skipped_missing=%s | skipped_shape=%s",
                    report.branch_name,
                    loaded_counts,
                    skipped_missing_counts,
                    skipped_shape_counts,
                )
            return

        if self.config.warmstart.macro_checkpoint is not None:
            self.logger.info(
                "Warm-start requested for macro branch from %s",
                self.config.warmstart.macro_checkpoint,
            )
            reports.append(
                _load_branch_warmstart(
                    branch_name="macro",
                    branch=self.model.macro_branch,
                    checkpoint_path=self.config.warmstart.macro_checkpoint,
                    strict=self.config.warmstart.strict,
                )
            )
        if self.config.warmstart.micro_checkpoint is not None:
            self.logger.info(
                "Warm-start requested for micro branch from %s",
                self.config.warmstart.micro_checkpoint,
            )
            reports.append(
                _load_branch_warmstart(
                    branch_name="micro",
                    branch=self.model.micro_branch,
                    checkpoint_path=self.config.warmstart.micro_checkpoint,
                    strict=self.config.warmstart.strict,
                )
            )
        self.warmstart_reports = tuple(reports)
        for report in reports:
            loaded_counts = _categorize_state_keys(report.loaded_keys)
            skipped_missing_counts = _categorize_state_keys(report.skipped_missing_keys)
            skipped_shape_counts = _categorize_state_keys(
                key.split(" source=", 1)[0] for key in report.skipped_shape_keys
            )
            self.logger.info(
                "Warm-started %s branch from %s | loaded=%d | skipped_missing=%d | skipped_shape=%d",
                report.branch_name,
                report.checkpoint_path,
                len(report.loaded_keys),
                len(report.skipped_missing_keys),
                len(report.skipped_shape_keys),
            )
            self.logger.info(
                "Warm-start %s component summary | loaded=%s | skipped_missing=%s | skipped_shape=%s",
                report.branch_name,
                loaded_counts,
                skipped_missing_counts,
                skipped_shape_counts,
            )
            self.logger.info(
                "Warm-start %s loaded tensors: %s",
                report.branch_name,
                ", ".join(report.loaded_keys) if report.loaded_keys else "none",
            )
            if report.skipped_missing_keys:
                self.logger.warning(
                    "Warm-start %s skipped missing tensors: %s",
                    report.branch_name,
                    ", ".join(report.skipped_missing_keys),
                )
            if report.skipped_shape_keys:
                self.logger.warning(
                    "Warm-start %s skipped shape-mismatched tensors: %s",
                    report.branch_name,
                    ", ".join(report.skipped_shape_keys),
                )

    def _validate_alignment_dataset_compatibility(self) -> None:
        train_macro_count = int(self.train_dataset.modality_counts.get("macro", 0))
        train_micro_count = int(self.train_dataset.modality_counts.get("micro", 0))
        if train_macro_count == 0 or train_micro_count == 0:
            raise ValueError(
                "Alignment training requires both macro and micro samples in the train split, but "
                f"found macro={train_macro_count}, micro={train_micro_count} in "
                f"{self.config.dataset.train_split_csv}."
            )
        if len(self.train_loader) <= 0:
            raise ValueError(
                "The train dataloader has zero batches. Check dataset.batch_size and "
                "dataset.sampler.drop_last for this split."
            )

        paired_species_count = _count_paired_species(self.train_dataset)
        shared_genera_count = _count_cross_modal_shared_genera(self.train_dataset)
        if any(stage.lambda_supcon > 0 for stage in self.config.stages) and paired_species_count == 0:
            raise ValueError(
                "At least one stage enables SupCon, but the train split has no species with both "
                "macro and micro samples."
            )
        if any(stage.relation_policy in {"shuffled_exact_control", "hard_negative_control"} for stage in self.config.stages) and paired_species_count == 0:
            raise ValueError(
                "Control relation policies that permute exact assignments require at least one "
                "cross-modal paired species in the train split."
            )
        if any(stage.relation_policy == "genus_only" for stage in self.config.stages) and shared_genera_count == 0:
            raise ValueError(
                "The genus_only relation policy requires at least one genus shared across macro "
                "and micro samples in the train split."
            )
        if any(
            stage.lambda_tax > 0 and stage.relation_policy != "exact_only"
            for stage in self.config.stages
        ) and shared_genera_count == 0:
            raise ValueError(
                "At least one stage enables taxonomy-aware genus relations, but the train split has "
                "no genera shared across macro and micro samples."
            )
        if any(
            stage.relation_policy == "exact_plus_genus_relaxed_negative"
            for stage in self.config.stages
        ) and shared_genera_count == 0:
            self.logger.warning(
                "Relaxed-negative relation policies are configured but the train split has no shared "
                "cross-modal genera. Those stages will behave like exact-only alignment."
            )

        self.logger.info(
            "Alignment train compatibility | macro_samples=%d | micro_samples=%d | "
            "paired_species=%d | shared_genera=%d",
            train_macro_count,
            train_micro_count,
            paired_species_count,
            shared_genera_count,
        )

    def _planned_stage_epochs(self) -> int:
        return sum(stage.epochs for stage in self.config.stages)

    def _effective_max_epochs(self) -> int:
        return min(self.config.training.max_epochs, self._planned_stage_epochs())

    def _compute_total_training_steps(self) -> int:
        if self.config.training.max_steps is not None:
            return self.config.training.max_steps
        return self._effective_max_epochs() * max(1, len(self.train_loader))

    def _stage_position_for_epoch(self, epoch: int) -> tuple[int, StageConfig, int]:
        """Return `(stage_index, stage_config, stage_epoch_index)` for a global epoch."""
        remaining = epoch
        for stage_index, stage in enumerate(self.config.stages):
            if remaining < stage.epochs:
                return stage_index, stage, remaining
            remaining -= stage.epochs
        return len(self.config.stages) - 1, self.config.stages[-1], max(0, self.config.stages[-1].epochs - 1)

    def _stage_for_epoch(self, epoch: int) -> tuple[int, StageConfig]:
        stage_index, stage, _ = self._stage_position_for_epoch(epoch)
        return stage_index, stage

    def _stage_epoch_bounds(self, stage_index: int) -> tuple[int, int]:
        start_epoch = 0
        for index, stage in enumerate(self.config.stages):
            end_epoch = start_epoch + stage.epochs - 1
            if index == stage_index:
                return start_epoch, end_epoch
            start_epoch = end_epoch + 1
        return (0, self.config.stages[-1].epochs - 1)

    def _restore_configured_backbone_trainability(self) -> None:
        """Restore the per-branch backbone trainability requested by the base config."""
        self.model.configure_branch_backbone_trainability(
            "macro",
            freeze_backbone=self.config.model.freeze_macro_backbone,
            trainable_backbone_patterns=self.config.model.macro_trainable_backbone_patterns,
        )
        self.model.configure_branch_backbone_trainability(
            "micro",
            freeze_backbone=self.config.model.freeze_micro_backbone,
            trainable_backbone_patterns=self.config.model.micro_trainable_backbone_patterns,
        )

    def _restore_default_head_trainability(self) -> None:
        for parameter in self.model.macro_branch.projection.parameters():
            parameter.requires_grad = True
        for parameter in self.model.micro_branch.projection.parameters():
            parameter.requires_grad = True
        for parameter in self.model.macro_branch.classifier.parameters():
            parameter.requires_grad = True
        for parameter in self.model.micro_branch.classifier.parameters():
            parameter.requires_grad = True
        if hasattr(self.model, "macro_genus_head"):
            for parameter in self.model.macro_genus_head.parameters():
                parameter.requires_grad = True
        if hasattr(self.model, "micro_genus_head"):
            for parameter in self.model.micro_genus_head.parameters():
                parameter.requires_grad = True

    def _is_macro_encoder_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for parameter in self.model.macro_branch.backbone.parameters()
        )

    def _is_macro_classifier_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for parameter in self.model.macro_branch.classifier.parameters()
        )

    def _apply_stage_trainability(
        self,
        *,
        stage: StageConfig,
        stage_epoch_index: int,
    ) -> tuple[bool, bool]:
        """Apply stage-local freezing on top of the base config trainability."""
        self._restore_configured_backbone_trainability()
        self._restore_default_head_trainability()
        if stage_epoch_index < stage.freeze_macro_encoder_epochs:
            self.model.freeze_branch_backbone("macro")
        if stage_epoch_index < stage.freeze_macro_classifier_epochs:
            for parameter in self.model.macro_branch.classifier.parameters():
                parameter.requires_grad = False
        return self._is_macro_encoder_frozen(), self._is_macro_classifier_frozen()

    def _metric_improved_against(
        self,
        metric_value: float,
        best_value: float | None,
        *,
        mode: str | None = None,
    ) -> bool:
        if best_value is None:
            return True
        delta = metric_value - best_value
        resolved_mode = mode or self.config.training.best_metric_mode
        if resolved_mode == "min":
            delta = -delta
        return delta > self.config.training.early_stopping_min_delta

    def _metric_improved(self, metric_value: float) -> bool:
        return self._metric_improved_against(metric_value, self.state.best_metric)

    def _current_group_lrs(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for group in self.optimizer.param_groups:
            component_name = str(group.get("component_name", "unknown"))
            grouped.setdefault(component_name, []).append(float(group["lr"]))
        return {name: sum(values) / max(1, len(values)) for name, values in grouped.items()}

    def _append_metrics_history(self, payload: Mapping[str, Any]) -> None:
        with self.paths.metrics_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    def _log_parameter_groups(self) -> None:
        for index, group in enumerate(self.optimizer.param_groups):
            self.logger.info(
                "Optimizer group %d (%s) | lr=%.8f | weight_decay=%.6f | param_count=%d",
                index,
                group.get("group_name", "unnamed"),
                float(group["lr"]),
                float(group["weight_decay"]),
                sum(parameter.numel() for parameter in group["params"]),
            )

    def _extract_modality_batch(
        self,
        batch: Mapping[str, Any],
        *,
        modality: str,
        label_mapping: LabelMapping,
    ) -> tuple[Tensor | None, Tensor | None, list[str], list[str], list[str]]:
        modalities = [str(value) for value in batch["modality"]]
        indices = [index for index, value in enumerate(modalities) if value == modality]
        if not indices:
            return None, None, [], [], []
        images = batch["image"][indices].to(self.device, non_blocking=True)
        species = [str(batch["species"][index]) for index in indices]
        genera = [str(batch["genus"][index]) for index in indices]
        image_paths = [str(batch["image_path"][index]) for index in indices]
        targets = torch.tensor(
            [label_mapping.index_for(species_name) for species_name in species],
            dtype=torch.long,
            device=self.device,
        )
        return images, targets, species, genera, image_paths

    def _policy_seed(
        self,
        *,
        split_name: str,
        batch_index: int,
        macro_count: int,
        micro_count: int,
    ) -> int:
        split_offset = {"train": 0, "val": 1, "test": 2}.get(split_name, 3)
        return (
            (self.config.seed * 1_000_003)
            + (split_offset * 97_409)
            + (batch_index * 1_009)
            + (macro_count * 53)
            + (micro_count * 97)
        )

    def _build_relation_tensors(
        self,
        *,
        macro_species: Sequence[str],
        micro_species: Sequence[str],
        macro_genera: Sequence[str],
        micro_genera: Sequence[str],
        stage: StageConfig,
        embeddings: Tensor,
        split_name: str,
        batch_index: int,
    ) -> tuple[TaxonomyRelationMasks, AlignmentPolicyOutput]:
        species_ids = _encode_string_ids([*macro_species, *micro_species]).to(self.device)
        genus_ids = _encode_string_ids([*macro_genera, *micro_genera]).to(self.device)
        modality_ids = torch.tensor(
            [0] * len(macro_species) + [1] * len(micro_species),
            dtype=torch.long,
            device=self.device,
        )
        masks = build_taxonomy_relation_masks(species_ids, genus_ids, modality_ids)
        macro_count = len(macro_species)
        micro_count = len(micro_species)
        positive_mask = masks.exact_species_mask.to(dtype=embeddings.dtype)
        negative_weights = None
        if macro_count > 0 and micro_count > 0:
            exact_block = masks.exact_species_mask[:macro_count, macro_count:].to(dtype=embeddings.dtype)
            genus_block = masks.genus_soft_mask[:macro_count, macro_count:].to(dtype=embeddings.dtype)
            micro_species_ids = species_ids[macro_count:].detach().cpu().tolist()
            micro_genus_ids = genus_ids[macro_count:].detach().cpu().tolist()
            policy_seed = self._policy_seed(
                split_name=split_name,
                batch_index=batch_index,
                macro_count=macro_count,
                micro_count=micro_count,
            )

            if stage.relation_policy == "genus_only":
                positive_mask = _square_cross_modal_mask_from_block(
                    genus_block,
                    macro_count=macro_count,
                    micro_count=micro_count,
                )
                negative_weights = _build_species_neutral_negative_weights(
                    masks,
                    dtype=embeddings.dtype,
                ).to(device=embeddings.device, dtype=embeddings.dtype)
            elif stage.relation_policy == "shuffled_exact_control":
                permutation = _find_forbidden_position_permutation(
                    position_categories=micro_species_ids,
                    sample_categories=micro_species_ids,
                    seed=policy_seed,
                )
                if permutation is None:
                    positive_mask = embeddings.new_zeros((embeddings.shape[0], embeddings.shape[0]))
                else:
                    shuffled_block = exact_block.index_select(
                        dim=1,
                        index=permutation.to(device=embeddings.device),
                    )
                    positive_mask = _square_cross_modal_mask_from_block(
                        shuffled_block,
                        macro_count=macro_count,
                        micro_count=micro_count,
                    )
            elif stage.relation_policy == "hard_negative_control":
                permutation = _find_forbidden_position_permutation(
                    position_categories=micro_genus_ids,
                    sample_categories=micro_genus_ids,
                    seed=policy_seed,
                )
                if permutation is None:
                    positive_mask = embeddings.new_zeros((embeddings.shape[0], embeddings.shape[0]))
                else:
                    hard_negative_block = exact_block.index_select(
                        dim=1,
                        index=permutation.to(device=embeddings.device),
                    )
                    positive_mask = _square_cross_modal_mask_from_block(
                        hard_negative_block,
                        macro_count=macro_count,
                        micro_count=micro_count,
                    )
        if stage.relation_policy == "exact_plus_genus_relaxed_negative":
            negative_weights = build_relaxed_negative_weights(
                masks,
                genus_negative_weight=self.config.loss.relaxed_negative_weight,
            ).to(device=embeddings.device, dtype=embeddings.dtype)
        policy_output = AlignmentPolicyOutput(
            positive_mask=positive_mask,
            negative_weights=negative_weights,
            active_positive_pairs=_count_positive_pairs(positive_mask),
            active_positive_anchors=_count_positive_anchors(positive_mask),
        )
        return masks, policy_output

    def _compute_alignment_step(
        self,
        batch: Mapping[str, Any],
        *,
        stage: StageConfig,
        split_name: str,
        batch_index: int,
    ) -> tuple[AlignmentStepOutput, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        macro_images, macro_targets, macro_species, macro_genera, _ = self._extract_modality_batch(
            batch,
            modality="macro",
            label_mapping=self.macro_label_mapping,
        )
        micro_images, micro_targets, micro_species, micro_genera, _ = self._extract_modality_batch(
            batch,
            modality="micro",
            label_mapping=self.micro_label_mapping,
        )

        output = self.model(macro_images=macro_images, micro_images=micro_images)

        macro_ce_loss = torch.zeros((), device=self.device)
        if output.logits_macro is not None and macro_targets is not None:
            macro_ce_loss = self.macro_criterion(output.logits_macro, macro_targets)

        micro_ce_loss = torch.zeros((), device=self.device)
        if output.logits_micro is not None and micro_targets is not None:
            micro_ce_loss = self.micro_criterion(output.logits_micro, micro_targets)

        supcon_loss = torch.zeros((), device=self.device)
        taxonomy_loss = torch.zeros((), device=self.device)
        rgbgray_consistency_loss = torch.zeros((), device=self.device)
        macro_rgbgray_consistency_loss = torch.zeros((), device=self.device)
        micro_rgbgray_consistency_loss = torch.zeros((), device=self.device)
        macro_genus_aux_loss = torch.zeros((), device=self.device)
        micro_genus_aux_loss = torch.zeros((), device=self.device)
        genus_aux_loss = torch.zeros((), device=self.device)
        exact_positive_pairs = 0
        genus_soft_pairs = 0
        exact_positive_anchors = 0
        active_positive_pairs = 0
        active_positive_anchors = 0

        if (
            self.config.genus_aux.enabled
            and output.pooled_features_macro is not None
            and macro_genera
            and self.macro_genus_label_mapping is not None
            and self.macro_genus_criterion is not None
            and hasattr(self.model, "macro_genus_head")
        ):
            macro_genus_targets = torch.tensor(
                [self.macro_genus_label_mapping.index_for(genus_name) for genus_name in macro_genera],
                dtype=torch.long,
                device=self.device,
            )
            macro_genus_logits = self.model.macro_genus_head(output.pooled_features_macro)
            macro_genus_aux_loss = self.macro_genus_criterion(macro_genus_logits, macro_genus_targets)

        if (
            self.config.genus_aux.enabled
            and output.pooled_features_micro is not None
            and micro_genera
            and self.micro_genus_label_mapping is not None
            and self.micro_genus_criterion is not None
            and hasattr(self.model, "micro_genus_head")
        ):
            micro_genus_targets = torch.tensor(
                [self.micro_genus_label_mapping.index_for(genus_name) for genus_name in micro_genera],
                dtype=torch.long,
                device=self.device,
            )
            micro_genus_logits = self.model.micro_genus_head(output.pooled_features_micro)
            micro_genus_aux_loss = self.micro_genus_criterion(micro_genus_logits, micro_genus_targets)

        if self.config.genus_aux.enabled:
            genus_aux_loss = (
                self.config.genus_aux.macro_weight * macro_genus_aux_loss
                + self.config.genus_aux.micro_weight * micro_genus_aux_loss
            )

        if (
            self.config.loss.rgbgray_consistency_weight > 0
            and output.rgb_pooled_features_macro is not None
            and output.gray_pooled_features_macro is not None
        ):
            macro_rgbgray_consistency_loss = _rgbgray_consistency_loss(
                output.rgb_pooled_features_macro,
                output.gray_pooled_features_macro,
            )
        if (
            self.config.loss.rgbgray_consistency_weight > 0
            and output.rgb_pooled_features_micro is not None
            and output.gray_pooled_features_micro is not None
        ):
            micro_rgbgray_consistency_loss = _rgbgray_consistency_loss(
                output.rgb_pooled_features_micro,
                output.gray_pooled_features_micro,
            )

        consistency_batch_weight = len(macro_species) + len(micro_species)
        if consistency_batch_weight > 0:
            rgbgray_consistency_loss = (
                (macro_rgbgray_consistency_loss * len(macro_species))
                + (micro_rgbgray_consistency_loss * len(micro_species))
            ) / consistency_batch_weight

        if output.embeddings_macro is not None and output.embeddings_micro is not None:
            embeddings = torch.cat([output.embeddings_macro, output.embeddings_micro], dim=0)
            masks, policy_output = self._build_relation_tensors(
                macro_species=macro_species,
                micro_species=micro_species,
                macro_genera=macro_genera,
                micro_genera=micro_genera,
                stage=stage,
                embeddings=embeddings,
                split_name=split_name,
                batch_index=batch_index,
            )
            exact_positive_pairs = masks.exact_pair_count
            genus_soft_pairs = masks.genus_pair_count
            exact_positive_anchors = int((masks.exact_species_mask.sum(dim=1) > 0).sum().item())
            active_positive_pairs = policy_output.active_positive_pairs
            active_positive_anchors = policy_output.active_positive_anchors
            if stage.lambda_supcon > 0:
                supcon_loss = self.supcon_loss(
                    embeddings,
                    policy_output.positive_mask,
                    negative_weights=policy_output.negative_weights,
                )
            if (
                stage.lambda_tax > 0
                and stage.relation_policy != "exact_only"
                and genus_soft_pairs > 0
            ):
                pair_weights = (
                    masks.genus_soft_mask.to(dtype=embeddings.dtype)
                    * self.config.loss.taxonomy_pair_weight
                )
                taxonomy_loss = self.taxonomy_loss(
                    embeddings,
                    masks.genus_soft_mask,
                    pair_weights=pair_weights,
                )

        total_loss = (
            stage.lambda_macro_ce * macro_ce_loss
            + stage.lambda_micro_ce * micro_ce_loss
            + stage.lambda_supcon * supcon_loss
            + stage.lambda_tax * taxonomy_loss
            + self.config.loss.rgbgray_consistency_weight * rgbgray_consistency_loss
            + self.config.genus_aux.loss_weight * genus_aux_loss
        )

        return (
            AlignmentStepOutput(
                total_loss=total_loss,
                macro_ce_loss=macro_ce_loss,
                micro_ce_loss=micro_ce_loss,
                supcon_loss=supcon_loss,
                taxonomy_loss=taxonomy_loss,
                rgbgray_consistency_loss=rgbgray_consistency_loss,
                macro_rgbgray_consistency_loss=macro_rgbgray_consistency_loss,
                micro_rgbgray_consistency_loss=micro_rgbgray_consistency_loss,
                genus_aux_loss=genus_aux_loss,
                macro_genus_aux_loss=macro_genus_aux_loss,
                micro_genus_aux_loss=micro_genus_aux_loss,
                macro_sample_count=len(macro_species),
                micro_sample_count=len(micro_species),
                exact_positive_pairs=exact_positive_pairs,
                genus_soft_pairs=genus_soft_pairs,
                exact_positive_anchors=exact_positive_anchors,
                active_positive_pairs=active_positive_pairs,
                active_positive_anchors=active_positive_anchors,
            ),
            output.logits_macro,
            macro_targets,
            output.logits_micro,
            micro_targets,
        )

    def _build_epoch_summary(
        self,
        *,
        split_name: str,
        stage_name: str,
        macro_accumulator: ClassificationMetricAccumulator,
        micro_accumulator: ClassificationMetricAccumulator,
        loss_accumulator: AlignmentLossAccumulator,
    ) -> AlignmentEvaluationSummary:
        macro_summary = (
            macro_accumulator.compute() if macro_accumulator.total_samples > 0 else None
        )
        micro_summary = (
            micro_accumulator.compute() if micro_accumulator.total_samples > 0 else None
        )
        return AlignmentEvaluationSummary(
            split_name=split_name,
            stage_name=stage_name,
            macro=macro_summary,
            micro=micro_summary,
            alignment=loss_accumulator.summary(),
        )

    def _weighted_branch_metric(
        self,
        macro_value: float | None,
        micro_value: float | None,
        *,
        weights: MetricWeightConfig | None = None,
    ) -> float:
        resolved_weights = weights or self.config.training.best_metric_weights
        macro_weight, micro_weight = resolved_weights.normalized(
            macro_available=macro_value is not None,
            micro_available=micro_value is not None,
        )
        return (
            (macro_weight * (macro_value or 0.0))
            + (micro_weight * (micro_value or 0.0))
        )

    def _tradeoff_metric_value(self, summary: AlignmentEvaluationSummary) -> float:
        macro_bal = summary.macro.balanced_accuracy if summary.macro is not None else None
        micro_bal = summary.micro.balanced_accuracy if summary.micro is not None else None
        return float(
            self._weighted_branch_metric(
                macro_bal,
                micro_bal,
                weights=TRADEOFF_CHECKPOINT_WEIGHTS,
            )
        )

    def _summary_metric_value(self, summary: AlignmentEvaluationSummary, metric_name: str) -> float:
        def _branch_value(branch: ClassificationMetricsSummary | None, attribute: str) -> float | None:
            if branch is None:
                return None
            return float(getattr(branch, attribute))

        macro_bal = _branch_value(summary.macro, "balanced_accuracy")
        micro_bal = _branch_value(summary.micro, "balanced_accuracy")
        macro_top1 = _branch_value(summary.macro, "top1_accuracy")
        micro_top1 = _branch_value(summary.micro, "top1_accuracy")

        if metric_name == "total_loss":
            return float(summary.alignment.total_loss or 0.0)
        if metric_name == "macro_balanced_accuracy":
            return float(macro_bal or 0.0)
        if metric_name == "micro_balanced_accuracy":
            return float(micro_bal or 0.0)
        if metric_name == "macro_top1_accuracy":
            return float(macro_top1 or 0.0)
        if metric_name == "micro_top1_accuracy":
            return float(micro_top1 or 0.0)
        if metric_name == "mean_balanced_accuracy":
            values = [value for value in (macro_bal, micro_bal) if value is not None]
            return float(sum(values) / max(1, len(values)))
        if metric_name == "weighted_mean_balanced_accuracy":
            return float(self._weighted_branch_metric(macro_bal, micro_bal))
        if metric_name == "mean_top1_accuracy":
            values = [value for value in (macro_top1, micro_top1) if value is not None]
            return float(sum(values) / max(1, len(values)))
        raise KeyError(f"Unsupported validation metric '{metric_name}'.")

    def _flatten_summary_metrics(self, summary: AlignmentEvaluationSummary) -> dict[str, float | None]:
        macro_summary = summary.macro
        micro_summary = summary.micro
        macro_bal = macro_summary.balanced_accuracy if macro_summary is not None else None
        micro_bal = micro_summary.balanced_accuracy if micro_summary is not None else None
        return {
            "macro_top1": macro_summary.top1_accuracy if macro_summary is not None else None,
            "micro_top1": micro_summary.top1_accuracy if micro_summary is not None else None,
            "macro_macro_f1": macro_summary.macro_f1 if macro_summary is not None else None,
            "micro_macro_f1": micro_summary.macro_f1 if micro_summary is not None else None,
            "macro_balanced_accuracy": macro_bal,
            "micro_balanced_accuracy": micro_bal,
            "mean_balanced_accuracy": (
                (sum(value for value in (macro_bal, micro_bal) if value is not None))
                / max(1, sum(value is not None for value in (macro_bal, micro_bal)))
            ),
            "weighted_mean_balanced_accuracy": self._weighted_branch_metric(macro_bal, micro_bal),
            "tradeoff_weighted_mean_balanced_accuracy": self._weighted_branch_metric(
                macro_bal,
                micro_bal,
                weights=TRADEOFF_CHECKPOINT_WEIGHTS,
            ),
            "total_loss": summary.alignment.total_loss,
            "macro_ce": summary.alignment.macro_ce_loss,
            "micro_ce": summary.alignment.micro_ce_loss,
            "supcon": summary.alignment.supcon_loss,
            "taxonomy": summary.alignment.taxonomy_loss,
            "rgbgray_consistency": summary.alignment.rgbgray_consistency_loss,
            "macro_rgbgray_consistency": summary.alignment.macro_rgbgray_consistency_loss,
            "micro_rgbgray_consistency": summary.alignment.micro_rgbgray_consistency_loss,
            "genus_aux": summary.alignment.genus_aux_loss,
            "macro_genus_aux": summary.alignment.macro_genus_aux_loss,
            "micro_genus_aux": summary.alignment.micro_genus_aux_loss,
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        is_best: bool,
        last_val_summary: AlignmentEvaluationSummary | None,
    ) -> None:
        stage_index, stage = self._stage_for_epoch(epoch)
        checkpoint = {
            "experiment_name": self.config.experiment_name,
            **_seed_metadata(self.config.seed),
            "epoch": epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "best_macro_metric": self.state.best_macro_metric,
            "best_macro_epoch": self.state.best_macro_epoch,
            "best_tradeoff_metric": self.state.best_tradeoff_metric,
            "best_tradeoff_epoch": self.state.best_tradeoff_epoch,
            "epochs_without_improvement": self.state.epochs_without_improvement,
            "stage_index": stage_index,
            "stage_name": stage.name,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "macro_label_mapping": self.macro_label_mapping.to_dict(),
            "micro_label_mapping": self.micro_label_mapping.to_dict(),
            "joint_label_mapping": self.joint_label_mapping.to_dict(),
            "config": serialize_experiment_config(self.config),
            "last_val_summary": last_val_summary.to_dict() if last_val_summary is not None else None,
            "is_best": is_best,
            "rng_state": {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        atomic_torch_save(checkpoint, path)

    def load_checkpoint(self, path: Path, *, restore_training_state: bool = True) -> None:
        action = "Resuming from" if restore_training_state else "Loading model weights from"
        self.logger.info("%s checkpoint: %s", action, path)
        checkpoint = load_checkpoint_file(path)
        _validate_checkpoint_for_config(
            checkpoint,
            self.config,
            macro_label_mapping=self.macro_label_mapping,
            micro_label_mapping=self.micro_label_mapping,
        )
        allow_optional_gate_mismatch = (
            self.config.model.architecture == "rgbgray_late_fusion"
            and self.config.model.fusion_mode == "residual"
        )
        _load_model_state_with_gate_compatibility(
            self.model,
            checkpoint["model_state"],
            allow_optional_gate_mismatch=allow_optional_gate_mismatch,
            logger=self.logger,
        )
        if not restore_training_state:
            return

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.use_amp and checkpoint.get("scaler_state") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, Mapping):
            if rng_state.get("python_random") is not None:
                random.setstate(rng_state["python_random"])
            if rng_state.get("numpy_random") is not None:
                np.random.set_state(rng_state["numpy_random"])
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        self.state.epoch = int(checkpoint.get("epoch", 0)) + 1
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.best_metric = (
            float(checkpoint["best_metric"]) if checkpoint.get("best_metric") is not None else None
        )
        self.state.best_epoch = (
            int(checkpoint["best_epoch"]) if checkpoint.get("best_epoch") is not None else None
        )
        self.state.best_macro_metric = (
            float(checkpoint["best_macro_metric"])
            if checkpoint.get("best_macro_metric") is not None
            else None
        )
        self.state.best_macro_epoch = (
            int(checkpoint["best_macro_epoch"])
            if checkpoint.get("best_macro_epoch") is not None
            else None
        )
        self.state.best_tradeoff_metric = (
            float(checkpoint["best_tradeoff_metric"])
            if checkpoint.get("best_tradeoff_metric") is not None
            else None
        )
        self.state.best_tradeoff_epoch = (
            int(checkpoint["best_tradeoff_epoch"])
            if checkpoint.get("best_tradeoff_epoch") is not None
            else None
        )
        self.state.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        self.scheduler.step(self.state.global_step)

    def _log_stage_plan(self) -> None:
        total_stage_epochs = self._planned_stage_epochs()
        effective_max_epochs = self._effective_max_epochs()
        if self.config.training.max_epochs != effective_max_epochs:
            self.logger.info(
                "Training epochs clipped by stage schedule | requested_max_epochs=%d | "
                "stage_total_epochs=%d | effective_max_epochs=%d",
                self.config.training.max_epochs,
                total_stage_epochs,
                effective_max_epochs,
            )
        for stage_index, stage in enumerate(self.config.stages):
            start_epoch, end_epoch = self._stage_epoch_bounds(stage_index)
            self.logger.info(
                "Stage plan %d/%d | name=%s | epochs=%d (global epochs %d-%d) | "
                "relation_policy=%s | lambda_macro_ce=%.3f | lambda_micro_ce=%.3f | "
                "lambda_supcon=%.3f | lambda_tax=%.3f | freeze_macro_encoder_epochs=%d | "
                "freeze_macro_classifier_epochs=%d",
                stage_index + 1,
                len(self.config.stages),
                stage.name,
                stage.epochs,
                start_epoch,
                end_epoch,
                stage.relation_policy,
                stage.lambda_macro_ce,
                stage.lambda_micro_ce,
                stage.lambda_supcon,
                stage.lambda_tax,
                stage.freeze_macro_encoder_epochs,
                stage.freeze_macro_classifier_epochs,
            )

    def _log_stage_transition(
        self,
        *,
        epoch: int,
        stage_index: int,
        stage: StageConfig,
        stage_epoch_index: int,
        macro_encoder_frozen: bool,
        macro_classifier_frozen: bool,
    ) -> None:
        start_epoch, end_epoch = self._stage_epoch_bounds(stage_index)
        self.logger.info(
            "Entering stage %d/%d | name=%s | current_epoch=%d | stage_epoch_window=%d-%d | "
            "relation_policy=%s | lambda_macro_ce=%.3f | lambda_micro_ce=%.3f | "
            "lambda_supcon=%.3f | lambda_tax=%.3f | freeze_macro_encoder_epochs=%d | "
            "freeze_macro_classifier_epochs=%d | stage_epoch_index=%d | "
            "macro_encoder_frozen=%s | macro_classifier_frozen=%s",
            stage_index + 1,
            len(self.config.stages),
            stage.name,
            epoch,
            start_epoch,
            end_epoch,
            stage.relation_policy,
            stage.lambda_macro_ce,
            stage.lambda_micro_ce,
            stage.lambda_supcon,
            stage.lambda_tax,
            stage.freeze_macro_encoder_epochs,
            stage.freeze_macro_classifier_epochs,
            stage_epoch_index,
            macro_encoder_frozen,
            macro_classifier_frozen,
        )

    def _log_stage_trainability(
        self,
        *,
        epoch: int,
        stage: StageConfig,
        stage_epoch_index: int,
        macro_encoder_frozen: bool,
        macro_classifier_frozen: bool,
    ) -> None:
        self.logger.info(
            "Stage trainability | epoch=%d | stage=%s | stage_epoch_index=%d/%d | "
            "freeze_macro_encoder_epochs=%d | freeze_macro_classifier_epochs=%d | "
            "macro_encoder_frozen=%s | macro_classifier_frozen=%s",
            epoch,
            stage.name,
            stage_epoch_index,
            max(0, stage.epochs - 1),
            stage.freeze_macro_encoder_epochs,
            stage.freeze_macro_classifier_epochs,
            macro_encoder_frozen,
            macro_classifier_frozen,
        )

    def _log_epoch_summary(
        self,
        *,
        epoch: int,
        split_name: str,
        stage: StageConfig,
        summary: AlignmentEvaluationSummary,
    ) -> None:
        metrics = self._flatten_summary_metrics(summary)
        self.logger.info(
            "Epoch %d %s | stage=%s | relation_policy=%s | total=%.6f | "
            "macro_top1=%.4f | micro_top1=%.4f | macro_f1=%.4f | micro_f1=%.4f | "
            "macro_bal_acc=%.4f | micro_bal_acc=%.4f | mean_bal_acc=%.4f | "
            "weighted_mean_bal_acc=%.4f | tradeoff_weighted_mean_bal_acc=%.4f | "
            "macro_ce=%.6f | micro_ce=%.6f | supcon=%.6f | taxonomy=%.6f | "
            "rgbgray_consistency=%.6f | genus_aux=%.6f | "
            "active_pairs=%d | exact_pairs=%d | genus_pairs=%d | "
            "active_anchors=%d | exact_anchors=%d",
            epoch,
            split_name,
            stage.name,
            stage.relation_policy,
            metrics["total_loss"] or 0.0,
            metrics["macro_top1"] or 0.0,
            metrics["micro_top1"] or 0.0,
            metrics["macro_macro_f1"] or 0.0,
            metrics["micro_macro_f1"] or 0.0,
            metrics["macro_balanced_accuracy"] or 0.0,
            metrics["micro_balanced_accuracy"] or 0.0,
            metrics["mean_balanced_accuracy"] or 0.0,
            metrics["weighted_mean_balanced_accuracy"] or 0.0,
            metrics["tradeoff_weighted_mean_balanced_accuracy"] or 0.0,
            metrics["macro_ce"] or 0.0,
            metrics["micro_ce"] or 0.0,
            metrics["supcon"] or 0.0,
            metrics["taxonomy"] or 0.0,
            metrics["rgbgray_consistency"] or 0.0,
            metrics["genus_aux"] or 0.0,
            summary.alignment.active_positive_pairs,
            summary.alignment.exact_positive_pairs,
            summary.alignment.genus_soft_pairs,
            summary.alignment.active_positive_anchors,
            summary.alignment.exact_positive_anchors,
        )

    def fit(self) -> dict[str, Any]:
        self.logger.info("Starting experiment: %s", self.config.experiment_name)
        self.logger.info("Device: %s | AMP: %s", self.device, self.use_amp)
        self.logger.info(
            "Train samples=%d | Val samples=%d | Test samples=%s | Macro classes=%d | Micro classes=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset) if self.test_dataset is not None else "n/a",
            self.macro_label_mapping.num_classes,
            self.micro_label_mapping.num_classes,
        )
        self.logger.info(
            "Input modes | shared=%s | macro=%s | micro=%s",
            self.config.dataset.input_mode,
            self.config.dataset.macro_input_mode or self.config.dataset.input_mode,
            self.config.dataset.micro_input_mode or self.config.dataset.input_mode,
        )
        self.logger.info(
            "Model architecture=%s | fusion_mode=%s | fusion_dropout=%.3f | "
            "fusion_residual_scale=%.3f | macro_fusion_hidden_dim=%s | micro_fusion_hidden_dim=%s",
            self.config.model.architecture,
            self.config.model.fusion_mode,
            self.config.model.fusion_dropout,
            self.config.model.fusion_residual_scale,
            self.config.model.macro_fusion_hidden_dim,
            self.config.model.micro_fusion_hidden_dim,
        )
        self.logger.info(
            "Auxiliary regularization | rgbgray_consistency_weight=%.4f | genus_aux_enabled=%s",
            self.config.loss.rgbgray_consistency_weight,
            self.config.genus_aux.enabled,
        )
        self.logger.info(
            "Best checkpoint metric=%s | mode=%s | weights(macro=%.3f,micro=%.3f)",
            self.config.training.best_metric,
            self.config.training.best_metric_mode,
            self.config.training.best_metric_weights.macro,
            self.config.training.best_metric_weights.micro,
        )
        self.logger.info(
            "Secondary tradeoff checkpoint | metric=weighted_mean_balanced_accuracy | "
            "weights(macro=%.3f,micro=%.3f) | path=%s",
            TRADEOFF_CHECKPOINT_WEIGHTS.macro,
            TRADEOFF_CHECKPOINT_WEIGHTS.micro,
            self.paths.best_tradeoff_checkpoint_path,
        )
        if self.warmstart_reports:
            for report in self.warmstart_reports:
                loaded_counts = _categorize_state_keys(report.loaded_keys)
                skipped_missing_counts = _categorize_state_keys(report.skipped_missing_keys)
                skipped_shape_counts = _categorize_state_keys(
                    key.split(" source=", 1)[0] for key in report.skipped_shape_keys
                )
                self.logger.info(
                    "Warm-start summary | branch=%s | checkpoint=%s | loaded=%d | skipped_missing=%d | "
                    "skipped_shape=%d | classifier_loaded=%d | projection_loaded=%d | "
                    "classifier_skipped=%d | projection_skipped=%d",
                    report.branch_name,
                    report.checkpoint_path,
                    len(report.loaded_keys),
                    len(report.skipped_missing_keys),
                    len(report.skipped_shape_keys),
                    loaded_counts["classifier"],
                    loaded_counts["projection"],
                    skipped_missing_counts["classifier"] + skipped_shape_counts["classifier"],
                    skipped_missing_counts["projection"] + skipped_shape_counts["projection"],
                )
        self._log_stage_plan()
        self._log_parameter_groups()
        initial_stage_index, initial_stage, initial_stage_epoch_index = self._stage_position_for_epoch(
            self.state.epoch
        )
        initial_macro_encoder_frozen, initial_macro_classifier_frozen = self._apply_stage_trainability(
            stage=initial_stage,
            stage_epoch_index=initial_stage_epoch_index,
        )
        initial_lrs = self._current_group_lrs()
        self.logger.info(
            "Scheduler start | mode=warmup_cosine | total_steps=%d | warmup_steps=%d | "
            "active_stage=%s (%d/%d) | stage_epoch_index=%d | macro_encoder_frozen=%s | "
            "macro_classifier_frozen=%s | lr_macro_encoder=%.8f | lr_micro_encoder=%.8f | "
            "lr_macro_projection=%.8f | lr_micro_projection=%.8f | "
            "lr_macro_classifier=%.8f | lr_micro_classifier=%.8f | genus_aux_enabled=%s",
            self.total_training_steps,
            self.scheduler.warmup_steps,
            initial_stage.name,
            initial_stage_index + 1,
            len(self.config.stages),
            initial_stage_epoch_index,
            initial_macro_encoder_frozen,
            initial_macro_classifier_frozen,
            initial_lrs.get("macro_encoder", 0.0),
            initial_lrs.get("micro_encoder", 0.0),
            initial_lrs.get("macro_projection", 0.0),
            initial_lrs.get("micro_projection", 0.0),
            initial_lrs.get("macro_classifier", 0.0),
            initial_lrs.get("micro_classifier", 0.0),
            self.config.genus_aux.enabled,
        )

        train_start_time = time.time()
        last_val_summary: AlignmentEvaluationSummary | None = None
        stop_training = False
        current_stage_index: int | None = None
        current_macro_encoder_frozen: bool | None = None
        current_macro_classifier_frozen: bool | None = None
        effective_max_epochs = self._effective_max_epochs()

        for epoch in range(self.state.epoch, effective_max_epochs):
            stage_index, stage, stage_epoch_index = self._stage_position_for_epoch(epoch)
            macro_encoder_frozen, macro_classifier_frozen = self._apply_stage_trainability(
                stage=stage,
                stage_epoch_index=stage_epoch_index,
            )
            if current_stage_index != stage_index:
                self._log_stage_transition(
                    epoch=epoch,
                    stage_index=stage_index,
                    stage=stage,
                    stage_epoch_index=stage_epoch_index,
                    macro_encoder_frozen=macro_encoder_frozen,
                    macro_classifier_frozen=macro_classifier_frozen,
                )
                current_stage_index = stage_index
                current_macro_encoder_frozen = macro_encoder_frozen
                current_macro_classifier_frozen = macro_classifier_frozen
            elif (
                current_macro_encoder_frozen != macro_encoder_frozen
                or current_macro_classifier_frozen != macro_classifier_frozen
            ):
                self._log_stage_trainability(
                    epoch=epoch,
                    stage=stage,
                    stage_epoch_index=stage_epoch_index,
                    macro_encoder_frozen=macro_encoder_frozen,
                    macro_classifier_frozen=macro_classifier_frozen,
                )
                current_macro_encoder_frozen = macro_encoder_frozen
                current_macro_classifier_frozen = macro_classifier_frozen

            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            epoch_start = time.time()
            train_summary = self._run_one_epoch(
                self.train_loader,
                split_name="train",
                training=True,
                stage=stage,
            )
            self._log_epoch_summary(epoch=epoch, split_name="train", stage=stage, summary=train_summary)

            val_summary: AlignmentEvaluationSummary | None = None
            if (epoch + 1) % self.config.training.validate_every_n_epochs == 0:
                val_summary = self._run_one_epoch(
                    self.val_loader,
                    split_name="val",
                    training=False,
                    stage=stage,
                )
                last_val_summary = val_summary
                self._log_epoch_summary(epoch=epoch, split_name="val", stage=stage, summary=val_summary)
                primary_metric_value = self._summary_metric_value(
                    val_summary,
                    self.config.training.best_metric,
                )
                macro_metric_value = self._summary_metric_value(
                    val_summary,
                    "macro_balanced_accuracy",
                )
                tradeoff_metric_value = self._tradeoff_metric_value(val_summary)
                self.logger.info(
                    "Validation checkpoint metrics | primary_%s=%.6f | macro_balanced_accuracy=%.6f | "
                    "tradeoff_weighted_mean_balanced_accuracy=%.6f | tradeoff_weights(macro=%.3f,micro=%.3f)",
                    self.config.training.best_metric,
                    primary_metric_value,
                    macro_metric_value,
                    tradeoff_metric_value,
                    TRADEOFF_CHECKPOINT_WEIGHTS.macro,
                    TRADEOFF_CHECKPOINT_WEIGHTS.micro,
                )
                if self._metric_improved(primary_metric_value):
                    self.state.best_metric = primary_metric_value
                    self.state.best_epoch = epoch
                    self.state.epochs_without_improvement = 0
                    self._save_checkpoint(
                        self.paths.best_checkpoint_path,
                        epoch=epoch,
                        is_best=True,
                        last_val_summary=val_summary,
                    )
                    self.logger.info(
                        "Saved new best checkpoint to %s (metric=%s, value=%.6f)",
                        self.paths.best_checkpoint_path,
                        self.config.training.best_metric,
                        primary_metric_value,
                    )
                else:
                    self.state.epochs_without_improvement += 1
                if self._metric_improved_against(
                    macro_metric_value,
                    self.state.best_macro_metric,
                    mode="max",
                ):
                    self.state.best_macro_metric = macro_metric_value
                    self.state.best_macro_epoch = epoch
                    self._save_checkpoint(
                        self.paths.best_macro_checkpoint_path,
                        epoch=epoch,
                        is_best=True,
                        last_val_summary=val_summary,
                    )
                    self.logger.info(
                        "Saved new macro-best checkpoint to %s (metric=macro_balanced_accuracy, value=%.6f)",
                        self.paths.best_macro_checkpoint_path,
                        macro_metric_value,
                    )
                if self._metric_improved_against(
                    tradeoff_metric_value,
                    self.state.best_tradeoff_metric,
                    mode="max",
                ):
                    self.state.best_tradeoff_metric = tradeoff_metric_value
                    self.state.best_tradeoff_epoch = epoch
                    self._save_checkpoint(
                        self.paths.best_tradeoff_checkpoint_path,
                        epoch=epoch,
                        is_best=True,
                        last_val_summary=val_summary,
                    )
                    self.logger.info(
                        "Saved new tradeoff-best checkpoint to %s "
                        "(metric=weighted_mean_balanced_accuracy, value=%.6f, weights macro=%.3f micro=%.3f)",
                        self.paths.best_tradeoff_checkpoint_path,
                        tradeoff_metric_value,
                        TRADEOFF_CHECKPOINT_WEIGHTS.macro,
                        TRADEOFF_CHECKPOINT_WEIGHTS.micro,
                    )

            self._save_checkpoint(
                self.paths.last_checkpoint_path,
                epoch=epoch,
                is_best=False,
                last_val_summary=last_val_summary,
            )
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint(
                    self.paths.checkpoint_dir / f"epoch_{epoch:03d}.ckpt",
                    epoch=epoch,
                    is_best=False,
                    last_val_summary=last_val_summary,
                )

            history_payload = {
                **_seed_metadata(self.config.seed),
                "epoch": epoch,
                "stage_index": stage_index,
                "stage_name": stage.name,
                "relation_policy": stage.relation_policy,
                "stage_weights": {
                    "lambda_macro_ce": stage.lambda_macro_ce,
                    "lambda_micro_ce": stage.lambda_micro_ce,
                    "lambda_supcon": stage.lambda_supcon,
                    "lambda_tax": stage.lambda_tax,
                    "freeze_macro_encoder_epochs": stage.freeze_macro_encoder_epochs,
                    "freeze_macro_classifier_epochs": stage.freeze_macro_classifier_epochs,
                },
                "stage_epoch_index": stage_epoch_index,
                "macro_encoder_frozen": macro_encoder_frozen,
                "macro_classifier_frozen": macro_classifier_frozen,
                "global_step": self.state.global_step,
                "elapsed_seconds": time.time() - epoch_start,
                "learning_rates": self._current_group_lrs(),
                "train": train_summary.to_dict(),
                "train_metrics": self._flatten_summary_metrics(train_summary),
                "val": val_summary.to_dict() if val_summary is not None else None,
                "val_metrics": self._flatten_summary_metrics(val_summary) if val_summary is not None else None,
                "best_metric_name": self.config.training.best_metric,
                "best_metric_value": self.state.best_metric,
                "best_epoch": self.state.best_epoch,
                "best_macro_metric_name": "macro_balanced_accuracy",
                "best_macro_metric_value": self.state.best_macro_metric,
                "best_macro_epoch": self.state.best_macro_epoch,
                "best_tradeoff_metric_name": "weighted_mean_balanced_accuracy",
                "best_tradeoff_metric_value": self.state.best_tradeoff_metric,
                "best_tradeoff_epoch": self.state.best_tradeoff_epoch,
                "best_tradeoff_metric_weights": {
                    "macro": TRADEOFF_CHECKPOINT_WEIGHTS.macro,
                    "micro": TRADEOFF_CHECKPOINT_WEIGHTS.micro,
                },
            }
            self._append_metrics_history(history_payload)

            if (
                self.config.training.max_steps is not None
                and self.state.global_step >= self.config.training.max_steps
            ):
                self.logger.info("Reached max_steps=%d. Stopping training.", self.config.training.max_steps)
                stop_training = True
            if (
                self.config.training.early_stopping_patience is not None
                and self.state.epochs_without_improvement >= self.config.training.early_stopping_patience
            ):
                self.logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.state.epochs_without_improvement,
                )
                stop_training = True
            if stop_training:
                break

        total_elapsed = time.time() - train_start_time
        self.logger.info("Training finished in %.2f minutes.", total_elapsed / 60.0)

        best_checkpoint = (
            self.paths.best_checkpoint_path
            if self.paths.best_checkpoint_path.exists()
            else self.paths.last_checkpoint_path
        )
        self.load_checkpoint(best_checkpoint, restore_training_state=False)
        best_epoch = self.state.best_epoch if self.state.best_epoch is not None else 0
        _, best_stage = self._stage_for_epoch(best_epoch)

        final_payload: dict[str, Any] = {
            **_seed_metadata(self.config.seed),
            "best_metric_name": self.config.training.best_metric,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "best_macro_metric_name": "macro_balanced_accuracy",
            "best_macro_metric": self.state.best_macro_metric,
            "best_macro_epoch": self.state.best_macro_epoch,
            "best_tradeoff_metric_name": "weighted_mean_balanced_accuracy",
            "best_tradeoff_metric": self.state.best_tradeoff_metric,
            "best_tradeoff_epoch": self.state.best_tradeoff_epoch,
            "best_tradeoff_metric_weights": {
                "macro": TRADEOFF_CHECKPOINT_WEIGHTS.macro,
                "micro": TRADEOFF_CHECKPOINT_WEIGHTS.micro,
            },
            "global_step": self.state.global_step,
            "output_dir": str(self.paths.output_dir),
            "best_checkpoint": str(best_checkpoint),
            "best_macro_checkpoint": str(self.paths.best_macro_checkpoint_path),
            "best_tradeoff_checkpoint": str(self.paths.best_tradeoff_checkpoint_path),
            "stage_schedule": [asdict(stage) for stage in self.config.stages],
        }

        val_artifacts = self.evaluate_and_export(
            split_name="val",
            dataloader=self.val_loader,
            output_prefix="best_val",
            dump_embeddings=self.config.evaluation.dump_embeddings,
            stage=best_stage,
        )
        final_payload["val"] = val_artifacts.summary.to_dict()
        final_payload["val_metrics"] = self._flatten_summary_metrics(val_artifacts.summary)

        if self.test_loader is not None:
            test_artifacts = self.evaluate_and_export(
                split_name="test",
                dataloader=self.test_loader,
                output_prefix="test",
                dump_embeddings=self.config.evaluation.dump_embeddings,
                stage=best_stage,
            )
            final_payload["test"] = test_artifacts.summary.to_dict()
            final_payload["test_metrics"] = self._flatten_summary_metrics(test_artifacts.summary)

        summary_path = self.paths.report_dir / "training_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=2, ensure_ascii=False)
        self.logger.info("Wrote training summary to %s", summary_path)
        return final_payload

    def _run_one_epoch(
        self,
        dataloader: DataLoader[Any],
        *,
        split_name: str,
        training: bool,
        stage: StageConfig,
    ) -> AlignmentEvaluationSummary:
        self.model.train(mode=training)
        macro_accumulator = ClassificationMetricAccumulator(
            self.macro_label_mapping.index_to_species,
            split_name=f"{split_name}_macro",
        )
        micro_accumulator = ClassificationMetricAccumulator(
            self.micro_label_mapping.index_to_species,
            split_name=f"{split_name}_micro",
        )
        loss_accumulator = AlignmentLossAccumulator(split_name=split_name, stage_name=stage.name)

        context = nullcontext() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(dataloader):
                if (
                    training
                    and self.config.training.max_steps is not None
                    and self.state.global_step >= self.config.training.max_steps
                ):
                    break
                if (
                    not training
                    and self.config.training.max_eval_batches is not None
                    and batch_index >= self.config.training.max_eval_batches
                ):
                    break

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with _autocast_context(self.device, enabled=self.use_amp):
                    step_output, macro_logits, macro_targets, micro_logits, micro_targets = (
                        self._compute_alignment_step(
                            batch,
                            stage=stage,
                            split_name=split_name,
                            batch_index=batch_index,
                        )
                    )

                if training:
                    if step_output.total_loss.requires_grad:
                        self.scaler.scale(step_output.total_loss).backward()
                        if self.config.training.grad_clip_norm is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.config.training.grad_clip_norm,
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.state.global_step += 1
                    self.scheduler.step(self.state.global_step)

                loss_accumulator.update(step_output)

                if macro_logits is not None and macro_targets is not None:
                    macro_accumulator.update(
                        macro_logits.detach(),
                        macro_targets.detach(),
                        loss=step_output.macro_ce_loss.detach(),
                    )
                if micro_logits is not None and micro_targets is not None:
                    micro_accumulator.update(
                        micro_logits.detach(),
                        micro_targets.detach(),
                        loss=step_output.micro_ce_loss.detach(),
                    )

                if training and (
                    self.state.global_step == 1
                    or self.state.global_step % self.config.training.log_every_n_steps == 0
                ):
                    current_lrs = self._current_group_lrs()
                    macro_summary = macro_accumulator.compute() if macro_accumulator.total_samples > 0 else None
                    micro_summary = micro_accumulator.compute() if micro_accumulator.total_samples > 0 else None
                    self.logger.info(
                        "Step %d | stage=%s | batch=%d | total=%.6f | macro_top1=%.4f | micro_top1=%.4f | "
                        "supcon=%.6f | taxonomy=%.6f | rgbgray_consistency=%.6f | genus_aux=%.6f | "
                        "active_pairs=%d | exact_pairs=%d | "
                        "genus_pairs=%d | active_anchors=%d | exact_anchors=%d | "
                        "lr_macro_encoder=%.8f | lr_micro_encoder=%.8f | "
                        "lr_macro_projection=%.8f | lr_micro_projection=%.8f | "
                        "lr_macro_classifier=%.8f | lr_micro_classifier=%.8f",
                        self.state.global_step,
                        stage.name,
                        batch_index,
                        step_output.total_loss.item(),
                        macro_summary.top1_accuracy if macro_summary is not None else 0.0,
                        micro_summary.top1_accuracy if micro_summary is not None else 0.0,
                        step_output.supcon_loss.item(),
                        step_output.taxonomy_loss.item(),
                        step_output.rgbgray_consistency_loss.item(),
                        step_output.genus_aux_loss.item(),
                        step_output.active_positive_pairs,
                        step_output.exact_positive_pairs,
                        step_output.genus_soft_pairs,
                        step_output.active_positive_anchors,
                        step_output.exact_positive_anchors,
                        current_lrs.get("macro_encoder", 0.0),
                        current_lrs.get("micro_encoder", 0.0),
                        current_lrs.get("macro_projection", 0.0),
                        current_lrs.get("micro_projection", 0.0),
                        current_lrs.get("macro_classifier", 0.0),
                        current_lrs.get("micro_classifier", 0.0),
                    )

        return self._build_epoch_summary(
            split_name=split_name,
            stage_name=stage.name,
            macro_accumulator=macro_accumulator,
            micro_accumulator=micro_accumulator,
            loss_accumulator=loss_accumulator,
        )

    def _dump_embeddings(
        self,
        dataloader: DataLoader[Any],
        *,
        split_name: str,
        output_path: Path,
    ) -> Path:
        records: list[dict[str, Any]] = []
        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if (
                    self.config.training.max_eval_batches is not None
                    and batch_index >= self.config.training.max_eval_batches
                ):
                    break
                macro_images, _, macro_species, macro_genera, macro_paths = self._extract_modality_batch(
                    batch,
                    modality="macro",
                    label_mapping=self.macro_label_mapping,
                )
                micro_images, _, micro_species, micro_genera, micro_paths = self._extract_modality_batch(
                    batch,
                    modality="micro",
                    label_mapping=self.micro_label_mapping,
                )
                output = self.model(macro_images=macro_images, micro_images=micro_images)
                if output.embeddings_macro is not None:
                    projected_macro = self.model.macro_branch.projection.network(
                        output.pooled_features_macro
                    ).detach().cpu() if output.pooled_features_macro is not None else None
                    for index, embedding in enumerate(output.embeddings_macro.cpu()):
                        records.append(
                            {
                                "split": split_name,
                                "modality": "macro",
                                "species": macro_species[index],
                                "genus": macro_genera[index],
                                "image_path": macro_paths[index],
                                "embedding": embedding,
                                "projected_feature": (
                                    projected_macro[index] if projected_macro is not None else None
                                ),
                                "pooled_feature": (
                                    output.pooled_features_macro[index].detach().cpu()
                                    if output.pooled_features_macro is not None
                                    else None
                                ),
                                "rgb_pooled_feature": (
                                    output.rgb_pooled_features_macro[index].detach().cpu()
                                    if output.rgb_pooled_features_macro is not None
                                    else None
                                ),
                                "gray_pooled_feature": (
                                    output.gray_pooled_features_macro[index].detach().cpu()
                                    if output.gray_pooled_features_macro is not None
                                    else None
                                ),
                            }
                        )
                if output.embeddings_micro is not None:
                    projected_micro = self.model.micro_branch.projection.network(
                        output.pooled_features_micro
                    ).detach().cpu() if output.pooled_features_micro is not None else None
                    for index, embedding in enumerate(output.embeddings_micro.cpu()):
                        records.append(
                            {
                                "split": split_name,
                                "modality": "micro",
                                "species": micro_species[index],
                                "genus": micro_genera[index],
                                "image_path": micro_paths[index],
                                "embedding": embedding,
                                "projected_feature": (
                                    projected_micro[index] if projected_micro is not None else None
                                ),
                                "pooled_feature": (
                                    output.pooled_features_micro[index].detach().cpu()
                                    if output.pooled_features_micro is not None
                                    else None
                                ),
                                "rgb_pooled_feature": (
                                    output.rgb_pooled_features_micro[index].detach().cpu()
                                    if output.rgb_pooled_features_micro is not None
                                    else None
                                ),
                                "gray_pooled_feature": (
                                    output.gray_pooled_features_micro[index].detach().cpu()
                                    if output.gray_pooled_features_micro is not None
                                    else None
                                ),
                            }
                        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, output_path)
        return output_path

    def evaluate_and_export(
        self,
        *,
        split_name: str,
        dataloader: DataLoader[Any],
        output_prefix: str,
        dump_embeddings: bool,
        stage: StageConfig,
    ) -> AlignmentEvaluationArtifacts:
        summary = self._run_one_epoch(
            dataloader,
            split_name=split_name,
            training=False,
            stage=stage,
        )
        macro_confusion_matrix_path = None
        macro_per_class_report_path = None
        macro_summary_json_path = None
        if summary.macro is not None:
            macro_summary_json_path = self.paths.report_dir / f"{output_prefix}_macro_metrics.json"
            macro_confusion_matrix_path = self.paths.report_dir / f"{output_prefix}_macro_confusion_matrix.csv"
            macro_per_class_report_path = self.paths.report_dir / f"{output_prefix}_macro_per_class_report.csv"
            macro_accumulator = self._export_branch_reports(
                dataloader,
                modality="macro",
                label_mapping=self.macro_label_mapping,
                criterion=self.macro_criterion,
                split_name=split_name,
                confusion_matrix_path=macro_confusion_matrix_path,
                per_class_report_path=macro_per_class_report_path,
            )
            with macro_summary_json_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    _summary_with_seed(summary.macro.to_dict(), seed=self.config.seed),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

        micro_confusion_matrix_path = None
        micro_per_class_report_path = None
        micro_summary_json_path = None
        if summary.micro is not None:
            micro_summary_json_path = self.paths.report_dir / f"{output_prefix}_micro_metrics.json"
            micro_confusion_matrix_path = self.paths.report_dir / f"{output_prefix}_micro_confusion_matrix.csv"
            micro_per_class_report_path = self.paths.report_dir / f"{output_prefix}_micro_per_class_report.csv"
            micro_accumulator = self._export_branch_reports(
                dataloader,
                modality="micro",
                label_mapping=self.micro_label_mapping,
                criterion=self.micro_criterion,
                split_name=split_name,
                confusion_matrix_path=micro_confusion_matrix_path,
                per_class_report_path=micro_per_class_report_path,
            )
            with micro_summary_json_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    _summary_with_seed(summary.micro.to_dict(), seed=self.config.seed),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

        alignment_summary_json_path = self.paths.report_dir / f"{output_prefix}_alignment_metrics.json"
        with alignment_summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **_seed_metadata(self.config.seed),
                    **summary.alignment.to_dict(),
                    "combined_metrics": self._flatten_summary_metrics(summary),
                    "relation_policy": stage.relation_policy,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        embeddings_dump_path = None
        if dump_embeddings:
            embeddings_dump_path = self._dump_embeddings(
                dataloader,
                split_name=split_name,
                output_path=self.paths.embedding_dir / f"{output_prefix}_embeddings.pt",
            )

        self.logger.info(
            "Exported %s metrics to %s",
            split_name,
            self.paths.report_dir,
        )
        return AlignmentEvaluationArtifacts(
            summary=summary,
            macro_confusion_matrix_path=macro_confusion_matrix_path,
            macro_per_class_report_path=macro_per_class_report_path,
            macro_summary_json_path=macro_summary_json_path,
            micro_confusion_matrix_path=micro_confusion_matrix_path,
            micro_per_class_report_path=micro_per_class_report_path,
            micro_summary_json_path=micro_summary_json_path,
            alignment_summary_json_path=alignment_summary_json_path,
            embeddings_dump_path=embeddings_dump_path,
        )

    def _export_branch_reports(
        self,
        dataloader: DataLoader[Any],
        *,
        modality: str,
        label_mapping: LabelMapping,
        criterion: nn.Module,
        split_name: str,
        confusion_matrix_path: Path,
        per_class_report_path: Path,
    ) -> ClassificationMetricAccumulator:
        self.model.eval()
        accumulator = ClassificationMetricAccumulator(
            label_mapping.index_to_species,
            split_name=f"{split_name}_{modality}",
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if (
                    self.config.training.max_eval_batches is not None
                    and batch_index >= self.config.training.max_eval_batches
                ):
                    break
                images, targets, _, _, _ = self._extract_modality_batch(
                    batch,
                    modality=modality,
                    label_mapping=label_mapping,
                )
                if images is None or targets is None:
                    continue
                with _autocast_context(self.device, enabled=self.use_amp):
                    if modality == "macro":
                        logits = self.model.forward_macro(images).logits
                    else:
                        logits = self.model.forward_micro(images).logits
                    loss = criterion(logits, targets)
                accumulator.update(logits.detach(), targets.detach(), loss=loss.detach())
        accumulator.export_confusion_matrix_csv(confusion_matrix_path)
        accumulator.export_per_class_report_csv(per_class_report_path)
        return accumulator
