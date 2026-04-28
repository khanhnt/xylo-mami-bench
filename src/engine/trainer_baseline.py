"""Training and evaluation utilities for XyloMaMi-Bench baseline classifiers."""

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import (
    DATASET_MODES,
    LabelMapping,
    ManifestDataset,
    build_label_mapping_from_csvs,
)
from src.datasets.transforms import build_transforms
from src.engine.metrics import ClassificationMetricAccumulator, ClassificationMetricsSummary
from src.models.baseline_classifier import BaselineClassifier, RGBGrayFusionClassifier
from src.utils.checkpoint_io import atomic_torch_save
from src.utils.seeding import set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_MODES = frozenset({"macro_classification", "micro_classification"})
BEST_METRICS = frozenset({"top1_accuracy", "macro_f1", "balanced_accuracy"})


@dataclass(frozen=True)
class DatasetConfig:
    mode: str
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


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    backbone_name: str
    gray_backbone_name: str | None
    num_classes: int
    pretrained: bool
    gray_pretrained: bool | None
    dropout: float
    fusion_hidden_dim: int | None
    fusion_dropout: float
    fusion_residual_scale: float
    freeze_backbone: bool
    trainable_backbone_patterns: tuple[str, ...]


@dataclass(frozen=True)
class WarmstartConfig:
    rgb_checkpoint: Path | None
    gray_checkpoint: Path | None
    strict: bool


@dataclass(frozen=True)
class OptimizerConfig:
    backbone_lr: float
    head_lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    min_lr_ratio: float
    warmup_steps: int | None
    warmup_epochs: int | None


@dataclass(frozen=True)
class TrainingConfig:
    """Trainer-side optimization and checkpoint behavior."""

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
    max_eval_batches: int | None
    resume_from: Path | None


@dataclass(frozen=True)
class LossConfig:
    label_smoothing: float
    class_weights: str | tuple[float, ...] | None


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: Path | None
    split_csv: Path | None
    split_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration."""

    experiment_name: str
    seed: int
    output_dir: Path
    device: str
    dataset: DatasetConfig
    model: ModelConfig
    warmstart: WarmstartConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path
    checkpoint_dir: Path
    report_dir: Path
    log_path: Path
    metrics_history_path: Path
    label_mapping_path: Path
    resolved_config_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0


@dataclass(frozen=True)
class EvaluationArtifacts:
    summary: ClassificationMetricsSummary
    confusion_matrix_path: Path
    per_class_report_path: Path
    summary_json_path: Path


@dataclass(frozen=True)
class WarmstartLoadResult:
    target_name: str
    checkpoint_path: Path
    loaded_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]


def serialize_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an experiment config into a plain serializable mapping."""

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
        if normalized not in {"balanced"}:
            raise ValueError(
                "loss.class_weights must be null, 'balanced', or a numeric list."
            )
        return normalized
    if isinstance(value, Sequence):
        weights = tuple(float(item) for item in value)
        return weights
    raise ValueError("loss.class_weights must be null, 'balanced', or a numeric list.")


def load_checkpoint_file(path: str | Path) -> Mapping[str, Any]:
    """Load a training checkpoint with explicit compatibility for newer PyTorch defaults."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def parse_experiment_config(raw_config: Mapping[str, Any]) -> ExperimentConfig:
    """Parse and validate the baseline experiment config tree."""

    dataset_payload = _require_mapping(raw_config, "dataset")
    mode = str(dataset_payload.get("mode", "")).strip().lower()
    if mode not in CLASSIFICATION_MODES:
        raise ValueError(
            f"Baseline training only supports {sorted(CLASSIFICATION_MODES)}, not '{mode}'."
        )
    if mode not in DATASET_MODES:
        raise ValueError(f"Unsupported dataset mode '{mode}'.")

    model_payload = _require_mapping(raw_config, "model")
    warmstart_payload = _require_mapping(raw_config, "warmstart")
    optimizer_payload = _require_mapping(raw_config, "optimizer")
    scheduler_payload = _require_mapping(raw_config, "scheduler")
    training_payload = _require_mapping(raw_config, "training")
    loss_payload = _require_mapping(raw_config, "loss")
    evaluation_payload = _require_mapping(raw_config, "evaluation")

    best_metric = str(training_payload.get("best_metric", "balanced_accuracy")).strip()
    if best_metric not in BEST_METRICS:
        raise ValueError(
            f"training.best_metric must be one of {sorted(BEST_METRICS)}, got '{best_metric}'."
        )
    best_metric_mode = str(training_payload.get("best_metric_mode", "max")).strip().lower()
    if best_metric_mode not in {"max", "min"}:
        raise ValueError("training.best_metric_mode must be 'max' or 'min'.")

    raw_config_copy = json.loads(json.dumps(raw_config))
    config = ExperimentConfig(
        experiment_name=str(raw_config.get("experiment_name", "baseline_experiment")).strip(),
        seed=int(raw_config.get("seed", 42)),
        output_dir=_resolve_project_path(raw_config.get("output_dir")) or (PROJECT_ROOT / "outputs"),
        device=str(raw_config.get("device", "auto")).strip().lower(),
        dataset=DatasetConfig(
            mode=mode,
            train_split_csv=_resolve_project_path(dataset_payload.get("train_split_csv")) or Path(),
            val_split_csv=_resolve_project_path(dataset_payload.get("val_split_csv")) or Path(),
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
            batch_size=int(dataset_payload.get("batch_size", 32)),
            eval_batch_size=int(dataset_payload.get("eval_batch_size", dataset_payload.get("batch_size", 32))),
            num_workers=int(dataset_payload.get("num_workers", 4)),
            image_root_override=_coerce_image_root_override(
                _require_mapping(dataset_payload, "image_root_override")
            ),
        ),
        model=ModelConfig(
            architecture=str(model_payload.get("architecture", "standard")).strip().lower(),
            backbone_name=str(model_payload.get("backbone_name", "convnext_small")).strip(),
            gray_backbone_name=(
                str(model_payload["gray_backbone_name"]).strip()
                if model_payload.get("gray_backbone_name") is not None
                else None
            ),
            num_classes=int(model_payload.get("num_classes", 0)),
            pretrained=bool(model_payload.get("pretrained", True)),
            gray_pretrained=(
                bool(model_payload["gray_pretrained"])
                if model_payload.get("gray_pretrained") is not None
                else None
            ),
            dropout=float(model_payload.get("dropout", 0.0)),
            fusion_hidden_dim=(
                int(model_payload["fusion_hidden_dim"])
                if model_payload.get("fusion_hidden_dim") is not None
                else None
            ),
            fusion_dropout=float(model_payload.get("fusion_dropout", 0.0)),
            fusion_residual_scale=float(model_payload.get("fusion_residual_scale", 0.1)),
            freeze_backbone=bool(model_payload.get("freeze_backbone", False)),
            trainable_backbone_patterns=tuple(
                str(item).strip()
                for item in model_payload.get("trainable_backbone_patterns", [])
                if str(item).strip()
            ),
        ),
        warmstart=WarmstartConfig(
            rgb_checkpoint=_resolve_project_path(warmstart_payload.get("rgb_checkpoint")),
            gray_checkpoint=_resolve_project_path(warmstart_payload.get("gray_checkpoint")),
            strict=bool(warmstart_payload.get("strict", False)),
        ),
        optimizer=OptimizerConfig(
            backbone_lr=float(optimizer_payload.get("backbone_lr", 1e-4)),
            head_lr=float(optimizer_payload.get("head_lr", 5e-4)),
            betas=tuple(float(item) for item in optimizer_payload.get("betas", (0.9, 0.999))),  # type: ignore[arg-type]
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
            max_epochs=int(
                training_payload.get(
                    "max_epochs",
                    training_payload.get("epochs", 30),
                )
            ),
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
            max_eval_batches=(
                int(training_payload["max_eval_batches"])
                if training_payload.get("max_eval_batches") is not None
                else None
            ),
            resume_from=_resolve_project_path(training_payload.get("resume_from")),
        ),
        loss=LossConfig(
            label_smoothing=float(loss_payload.get("label_smoothing", 0.0)),
            class_weights=_coerce_class_weights(loss_payload.get("class_weights")),
        ),
        evaluation=EvaluationConfig(
            checkpoint_path=_resolve_project_path(evaluation_payload.get("checkpoint_path")),
            split_csv=_resolve_project_path(evaluation_payload.get("split_csv")),
            split_name=str(evaluation_payload.get("split_name", "test")).strip().lower(),
        ),
        raw_config=raw_config_copy,
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Fail early on invalid or mismatched experiment settings."""

    if not config.experiment_name:
        raise ValueError("experiment_name must not be empty.")
    if config.dataset.image_size <= 0:
        raise ValueError("dataset.image_size must be positive.")
    if config.dataset.batch_size <= 0:
        raise ValueError("dataset.batch_size must be positive.")
    if config.dataset.eval_batch_size <= 0:
        raise ValueError("dataset.eval_batch_size must be positive.")
    if config.dataset.num_workers < 0:
        raise ValueError("dataset.num_workers must be >= 0.")
    if config.model.architecture not in {"standard", "rgbgray_late_fusion"}:
        raise ValueError(
            "model.architecture must be 'standard' or 'rgbgray_late_fusion', "
            f"got '{config.model.architecture}'."
        )
    if config.model.num_classes <= 0:
        raise ValueError("model.num_classes must be positive.")
    if config.model.fusion_hidden_dim is not None and config.model.fusion_hidden_dim <= 0:
        raise ValueError("model.fusion_hidden_dim must be positive when provided.")
    if not 0.0 <= config.model.fusion_dropout < 1.0:
        raise ValueError("model.fusion_dropout must be in [0, 1).")
    if config.model.fusion_residual_scale < 0:
        raise ValueError("model.fusion_residual_scale must be >= 0.")
    if config.optimizer.backbone_lr <= 0 or config.optimizer.head_lr <= 0:
        raise ValueError("optimizer learning rates must be positive.")
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
    if not 0.0 <= config.loss.label_smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1).")
    if config.scheduler.warmup_steps is not None and config.scheduler.warmup_steps < 0:
        raise ValueError("scheduler.warmup_steps must be >= 0.")
    if config.scheduler.warmup_epochs is not None and config.scheduler.warmup_epochs < 0:
        raise ValueError("scheduler.warmup_epochs must be >= 0.")
    if (
        config.scheduler.warmup_steps is not None
        and config.scheduler.warmup_epochs is not None
    ):
        raise ValueError(
            "Specify only one of scheduler.warmup_steps or scheduler.warmup_epochs, not both."
        )

    for field_name, split_path in (
        ("dataset.train_split_csv", config.dataset.train_split_csv),
        ("dataset.val_split_csv", config.dataset.val_split_csv),
    ):
        if not str(split_path):
            raise ValueError(f"{field_name} must be set.")
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
    if config.warmstart.rgb_checkpoint is not None and not config.warmstart.rgb_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.rgb_checkpoint does not exist: {config.warmstart.rgb_checkpoint}"
        )
    if config.warmstart.gray_checkpoint is not None and not config.warmstart.gray_checkpoint.exists():
        raise FileNotFoundError(
            f"warmstart.gray_checkpoint does not exist: {config.warmstart.gray_checkpoint}"
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


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed global RNGs and cuDNN switches for reproducible experiments."""

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
    """Resolve the requested runtime device."""

    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(normalized)


def prepare_experiment_paths(output_dir: Path) -> ExperimentPaths:
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        report_dir=report_dir,
        log_path=log_dir / "train.log",
        metrics_history_path=report_dir / "metrics_history.jsonl",
        label_mapping_path=output_dir / "label_mapping.json",
        resolved_config_path=output_dir / "resolved_config.yaml",
        best_checkpoint_path=checkpoint_dir / "best.ckpt",
        last_checkpoint_path=checkpoint_dir / "last.ckpt",
    )


def setup_logger(log_path: Path, *, name: str = "xylomami_baseline") -> logging.Logger:
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


def build_label_mapping_for_experiment(config: ExperimentConfig) -> LabelMapping:
    """Build the branch-specific label mapping from the training split."""

    return build_label_mapping_from_csvs(
        [config.dataset.train_split_csv],
        mode=config.dataset.mode,
        label_space_name=f"{config.experiment_name}_{config.dataset.mode}",
    )


def _build_dataset(
    split_csv: Path,
    *,
    split_name: str,
    config: ExperimentConfig,
    label_mapping: LabelMapping,
) -> ManifestDataset:
    """Build one manifest-backed dataset with the requested transform preset."""

    transform_map = build_transforms(
        split_name,
        config.dataset.image_size,
        input_mode=config.dataset.input_mode,
        macro_input_mode=config.dataset.macro_input_mode,
        micro_input_mode=config.dataset.micro_input_mode,
    )
    return ManifestDataset(
        split_csv,
        mode=config.dataset.mode,
        transform=transform_map,
        label_mapping=label_mapping,
        label_space_name=label_mapping.label_space_name,
        image_root_override=config.dataset.image_root_override,
    )


def build_dataloaders(
    config: ExperimentConfig,
    *,
    label_mapping: LabelMapping,
) -> tuple[ManifestDataset, ManifestDataset, ManifestDataset | None, DataLoader[Any], DataLoader[Any], DataLoader[Any] | None]:
    """Build train/val/test datasets and dataloaders for baseline classification."""

    train_dataset = _build_dataset(
        config.dataset.train_split_csv,
        split_name="train",
        config=config,
        label_mapping=label_mapping,
    )
    val_dataset = _build_dataset(
        config.dataset.val_split_csv,
        split_name="val",
        config=config,
        label_mapping=label_mapping,
    )
    test_dataset = (
        _build_dataset(
            config.dataset.test_split_csv,
            split_name="test",
            config=config,
            label_mapping=label_mapping,
        )
        if config.dataset.test_split_csv is not None
        else None
    )

    if train_dataset.num_classes != config.model.num_classes:
        raise ValueError(
            f"Config num_classes={config.model.num_classes} but train label mapping has "
            f"{train_dataset.num_classes} classes."
        )

    dataloader_kwargs = {
        "num_workers": config.dataset.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.dataset.num_workers > 0,
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
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


def build_eval_dataloader(
    config: ExperimentConfig,
    *,
    split_csv: Path,
    split_name: str,
    label_mapping: LabelMapping,
    batch_size: int | None = None,
    transform_split: str = "val",
) -> tuple[ManifestDataset, DataLoader[Any]]:
    """Build a deterministic evaluation dataloader for an arbitrary split CSV."""

    dataset = _build_dataset(
        split_csv,
        split_name=transform_split,
        config=config,
        label_mapping=label_mapping,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed + 30_000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size or config.dataset.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.dataset.num_workers > 0,
        generator=generator,
        worker_init_fn=partial(_seed_worker, base_seed=config.seed + 30_000),
    )
    return dataset, dataloader


def label_mapping_from_checkpoint(checkpoint: Mapping[str, Any]) -> LabelMapping | None:
    """Recover the stored label mapping from a checkpoint when available."""

    payload = checkpoint.get("label_mapping")
    if not isinstance(payload, Mapping):
        return None
    return LabelMapping(
        species_to_index={str(key): int(value) for key, value in payload["species_to_index"].items()},
        index_to_species=tuple(str(item) for item in payload["index_to_species"]),
        label_space_name=str(payload.get("label_space_name", "")),
    )


def _extract_model_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    model_state = checkpoint.get("model_state")
    if isinstance(model_state, Mapping):
        return model_state  # type: ignore[return-value]
    if isinstance(checkpoint, Mapping):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint did not contain a model_state mapping.")


def _extract_warmstart_state(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    source_state = _extract_model_state_from_checkpoint(checkpoint)
    normalized: dict[str, Tensor] = {}
    for key, value in source_state.items():
        normalized_key = str(key)
        if normalized_key.startswith("head."):
            normalized_key = "classifier." + normalized_key[len("head."):]
        normalized[normalized_key] = value
    return normalized


def _categorize_state_keys(keys: Sequence[str]) -> dict[str, int]:
    counts = {
        "backbone": 0,
        "pool": 0,
        "classifier": 0,
        "rgb_encoder": 0,
        "gray_encoder": 0,
        "fusion": 0,
        "other": 0,
    }
    for key in keys:
        component = key.split(".", 1)[0]
        if component == "head":
            component = "classifier"
        if component in counts:
            counts[component] += 1
        else:
            counts["other"] += 1
    return counts


def _load_module_subset_warmstart(
    *,
    target_name: str,
    target_module: nn.Module,
    checkpoint_path: Path,
    strict: bool,
    key_mapper: Callable[[str], str | None],
) -> WarmstartLoadResult:
    checkpoint = load_checkpoint_file(checkpoint_path)
    source_state = _extract_warmstart_state(checkpoint)
    target_state = target_module.state_dict()

    loadable: dict[str, Tensor] = {}
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in source_state.items():
        target_key = key_mapper(key)
        if target_key is None:
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
            f"No compatible tensors were found when warm-starting '{target_name}' "
            f"from checkpoint {checkpoint_path}."
        )
    if strict and (skipped_missing or skipped_shape):
        raise ValueError(
            f"Warm-start strict loading failed for '{target_name}'. "
            f"Missing={skipped_missing}, shape_mismatch={skipped_shape}"
        )

    target_module.load_state_dict(loadable, strict=False)
    return WarmstartLoadResult(
        target_name=target_name,
        checkpoint_path=checkpoint_path,
        loaded_keys=tuple(sorted(loadable)),
        skipped_missing_keys=tuple(sorted(skipped_missing)),
        skipped_shape_keys=tuple(sorted(skipped_shape)),
    )


def build_model(
    config: ExperimentConfig,
    *,
    pretrained_override: bool | None = None,
) -> BaselineClassifier | RGBGrayFusionClassifier:
    """Instantiate the baseline classifier from experiment config."""

    shared_pretrained = (
        config.model.pretrained if pretrained_override is None else pretrained_override
    )
    if config.model.architecture == "rgbgray_late_fusion":
        return RGBGrayFusionClassifier(
            backbone_name=config.model.backbone_name,
            gray_backbone_name=config.model.gray_backbone_name,
            num_classes=config.model.num_classes,
            pretrained=shared_pretrained,
            gray_pretrained=config.model.gray_pretrained,
            dropout=config.model.dropout,
            freeze_backbone=config.model.freeze_backbone,
            trainable_backbone_patterns=config.model.trainable_backbone_patterns,
            image_size=config.dataset.image_size,
            fusion_hidden_dim=config.model.fusion_hidden_dim,
            fusion_dropout=config.model.fusion_dropout,
            fusion_residual_scale=config.model.fusion_residual_scale,
        )
    return BaselineClassifier(
        backbone_name=config.model.backbone_name,
        num_classes=config.model.num_classes,
        pretrained=shared_pretrained,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
        trainable_backbone_patterns=config.model.trainable_backbone_patterns,
        image_size=config.dataset.image_size,
    )


def _is_no_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    lowered_name = name.lower()
    return (
        parameter.ndim <= 1
        or lowered_name.endswith(".bias")
        or "norm" in lowered_name
        or ".bn" in lowered_name
    )


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    """Seed one dataloader worker deterministically."""

    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def build_optimizer(
    model: BaselineClassifier | RGBGrayFusionClassifier,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    parameter_groups: dict[tuple[str, bool], dict[str, Any]] = {}
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if (
            name.startswith("backbone.")
            or name.startswith("rgb_encoder.")
            or name.startswith("gray_encoder.")
        ):
            branch = "backbone"
        else:
            branch = "head"
        use_decay = not _is_no_decay_parameter(name, parameter)
        key = (branch, use_decay)
        if key not in parameter_groups:
            parameter_groups[key] = {
                "params": [],
                "lr": config.optimizer.backbone_lr if branch == "backbone" else config.optimizer.head_lr,
                "weight_decay": config.optimizer.weight_decay if use_decay else 0.0,
                "group_name": f"{branch}_{'decay' if use_decay else 'no_decay'}",
            }
        parameter_groups[key]["params"].append(parameter)

    if not parameter_groups:
        raise ValueError("No trainable parameters found when building optimizer.")

    return torch.optim.AdamW(
        list(parameter_groups.values()),
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )


def resolve_class_weights(
    config: ExperimentConfig,
    train_dataset: ManifestDataset,
    *,
    device: torch.device,
) -> Tensor | None:
    class_weights = config.loss.class_weights
    if class_weights is None:
        return None
    if class_weights == "balanced":
        counts = torch.zeros(train_dataset.num_classes, dtype=torch.float32)
        for species, count in train_dataset.species_counts.items():
            counts[train_dataset.label_mapping.index_for(species)] = float(count)
        total = float(counts.sum().item())
        weights = total / (train_dataset.num_classes * counts.clamp_min(1.0))
        weights[counts <= 0] = 0.0
        return weights.to(device)

    weights = torch.tensor(class_weights, dtype=torch.float32)
    if weights.numel() != train_dataset.num_classes:
        raise ValueError(
            f"Configured {weights.numel()} class weights but expected {train_dataset.num_classes}."
        )
    return weights.to(device)


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
            return max(1e-8, float(clamped_step + 1) / float(self.warmup_steps))

        if self.total_steps == self.warmup_steps:
            return self.min_lr_ratio

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


def _autocast_context(device: torch.device, *, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _build_grad_scaler(*, device: torch.device, enabled: bool):
    """Create a GradScaler compatible with multiple recent PyTorch versions."""

    if device.type != "cuda":
        enabled = False
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer state tensors onto the active device after checkpoint resume."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _validate_checkpoint_for_config(
    checkpoint: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    label_mapping: LabelMapping | None = None,
) -> None:
    """Validate checkpoint metadata against the active config."""

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, Mapping):
        checkpoint_model = checkpoint_config.get("model")
        if isinstance(checkpoint_model, Mapping):
            checkpoint_backbone = str(checkpoint_model.get("backbone_name", "")).strip()
            if checkpoint_backbone and checkpoint_backbone != config.model.backbone_name:
                raise ValueError(
                    "Checkpoint backbone mismatch: "
                    f"checkpoint uses '{checkpoint_backbone}' but config requests "
                    f"'{config.model.backbone_name}'."
                )
            checkpoint_num_classes = checkpoint_model.get("num_classes")
            if checkpoint_num_classes is not None and int(checkpoint_num_classes) != config.model.num_classes:
                raise ValueError(
                    "Checkpoint num_classes mismatch: "
                    f"checkpoint has {int(checkpoint_num_classes)} but config requests "
                    f"{config.model.num_classes}."
                )
    if label_mapping is not None and label_mapping.num_classes != config.model.num_classes:
        raise ValueError(
            f"Label mapping has {label_mapping.num_classes} classes but config requests "
            f"{config.model.num_classes}."
        )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    criterion: nn.Module,
    class_names: Sequence[str],
    split_name: str,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> tuple[ClassificationMetricsSummary, ClassificationMetricAccumulator]:
    model.eval()
    accumulator = ClassificationMetricAccumulator(class_names, split_name=split_name)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            with _autocast_context(device, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            accumulator.update(logits.detach(), targets.detach(), loss=loss.detach())
    return accumulator.compute(), accumulator


class BaselineTrainer:
    """Trainer for XyloMaMi-Bench baseline classification experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        logger: logging.Logger,
        paths: ExperimentPaths,
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
        self.label_mapping = build_label_mapping_for_experiment(config)
        self.train_dataset, self.val_dataset, self.test_dataset, self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config,
            label_mapping=self.label_mapping,
        )
        self.model = build_model(
            config,
            pretrained_override=False if config.training.resume_from is not None else None,
        ).to(self.device)
        if config.training.resume_from is None:
            self._apply_warmstart()
        self.class_weights = resolve_class_weights(config, self.train_dataset, device=self.device)
        self.criterion = self.model.build_loss(
            class_weights=self.class_weights,
            label_smoothing=config.loss.label_smoothing,
        ).to(self.device)
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
        self.scaler = _build_grad_scaler(device=self.device, enabled=self.use_amp)

        self.label_mapping.save(self.paths.label_mapping_path)
        if config.training.resume_from is not None:
            self.load_checkpoint(config.training.resume_from)

    def _apply_warmstart(self) -> None:
        if self.config.model.architecture != "rgbgray_late_fusion":
            return
        if not isinstance(self.model, RGBGrayFusionClassifier):
            raise TypeError(
                "rgbgray_late_fusion architecture requires an RGBGrayFusionClassifier instance."
            )

        reports: list[WarmstartLoadResult] = []
        if self.config.warmstart.rgb_checkpoint is not None:
            reports.append(
                _load_module_subset_warmstart(
                    target_name="rgb_path",
                    target_module=self.model,
                    checkpoint_path=self.config.warmstart.rgb_checkpoint,
                    strict=self.config.warmstart.strict,
                    key_mapper=lambda key: (
                        f"rgb_encoder.{key}"
                        if key.startswith("backbone.") or key.startswith("pool.")
                        else ("head." + key[len("classifier."):] if key.startswith("classifier.") else None)
                    ),
                )
            )
        if self.config.warmstart.gray_checkpoint is not None:
            reports.append(
                _load_module_subset_warmstart(
                    target_name="gray_path",
                    target_module=self.model,
                    checkpoint_path=self.config.warmstart.gray_checkpoint,
                    strict=self.config.warmstart.strict,
                    key_mapper=lambda key: (
                        f"gray_encoder.{key}"
                        if key.startswith("backbone.") or key.startswith("pool.")
                        else None
                    ),
                )
            )
        self.warmstart_reports = tuple(reports)

    def _compute_total_training_steps(self) -> int:
        if self.config.training.max_steps is not None:
            return self.config.training.max_steps
        return self.config.training.max_epochs * max(1, len(self.train_loader))

    def _metric_improved(self, metric_value: float) -> bool:
        if self.state.best_metric is None:
            return True
        delta = metric_value - self.state.best_metric
        if self.config.training.best_metric_mode == "min":
            delta = -delta
        return delta > self.config.training.early_stopping_min_delta

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        is_best: bool,
        last_val_summary: ClassificationMetricsSummary | None,
    ) -> None:
        checkpoint = {
            "experiment_name": self.config.experiment_name,
            **_seed_metadata(self.config.seed),
            "epoch": epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "epochs_without_improvement": self.state.epochs_without_improvement,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "label_mapping": self.label_mapping.to_dict(),
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
        _validate_checkpoint_for_config(checkpoint, self.config, label_mapping=self.label_mapping)
        self.model.load_state_dict(checkpoint["model_state"])
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
        self.state.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        self.scheduler.step(self.state.global_step)

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

    def _current_group_lrs(self) -> tuple[float, float]:
        backbone_lrs = [
            float(group["lr"])
            for group in self.optimizer.param_groups
            if str(group.get("group_name", "")).startswith("backbone_")
        ]
        head_lrs = [
            float(group["lr"])
            for group in self.optimizer.param_groups
            if str(group.get("group_name", "")).startswith("head_")
        ]
        backbone_lr = min(backbone_lrs) if backbone_lrs else 0.0
        head_lr = max(head_lrs) if head_lrs else 0.0
        return backbone_lr, head_lr

    def fit(self) -> dict[str, Any]:
        self.logger.info("Starting experiment: %s", self.config.experiment_name)
        self.logger.info("Device: %s | AMP: %s", self.device, self.use_amp)
        self.logger.info(
            "Train samples=%d | Val samples=%d | Test samples=%s | Num classes=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset) if self.test_dataset is not None else "n/a",
            self.train_dataset.num_classes,
        )
        self.logger.info(
            "Input modes | shared=%s | macro=%s | micro=%s",
            self.config.dataset.input_mode,
            self.config.dataset.macro_input_mode or self.config.dataset.input_mode,
            self.config.dataset.micro_input_mode or self.config.dataset.input_mode,
        )
        self.logger.info(
            "Model architecture=%s | fusion_hidden_dim=%s | fusion_dropout=%.3f | "
            "fusion_residual_scale=%.3f",
            self.config.model.architecture,
            self.config.model.fusion_hidden_dim,
            self.config.model.fusion_dropout,
            self.config.model.fusion_residual_scale,
        )
        for report in self.warmstart_reports:
            loaded_counts = _categorize_state_keys(report.loaded_keys)
            skipped_missing_counts = _categorize_state_keys(list(report.skipped_missing_keys))
            skipped_shape_counts = _categorize_state_keys(
                [key.split(" source=", 1)[0] for key in report.skipped_shape_keys]
            )
            self.logger.info(
                "Warm-start summary | target=%s | checkpoint=%s | loaded=%d | skipped_missing=%d | "
                "skipped_shape=%d | rgb_loaded=%d | gray_loaded=%d | fusion_loaded=%d | classifier_loaded=%d",
                report.target_name,
                report.checkpoint_path,
                len(report.loaded_keys),
                len(report.skipped_missing_keys),
                len(report.skipped_shape_keys),
                loaded_counts["rgb_encoder"],
                loaded_counts["gray_encoder"],
                loaded_counts["fusion"],
                loaded_counts["classifier"],
            )
            self.logger.info(
                "Warm-start skipped summary | target=%s | skipped_missing=%s | skipped_shape=%s",
                report.target_name,
                skipped_missing_counts,
                skipped_shape_counts,
            )
        self._log_parameter_groups()

        train_start_time = time.time()
        last_val_summary: ClassificationMetricsSummary | None = None
        stop_training = False

        for epoch in range(self.state.epoch, self.config.training.max_epochs):
            epoch_start = time.time()
            train_summary = self._train_one_epoch(epoch)
            self.logger.info(
                "Epoch %d train | loss=%.6f | top1=%.4f | macro_f1=%.4f | balanced_acc=%.4f | step=%d",
                epoch,
                train_summary.loss or 0.0,
                train_summary.top1_accuracy,
                train_summary.macro_f1,
                train_summary.balanced_accuracy,
                self.state.global_step,
            )

            val_summary: ClassificationMetricsSummary | None = None
            if (epoch + 1) % self.config.training.validate_every_n_epochs == 0:
                val_summary, _ = evaluate_model(
                    self.model,
                    self.val_loader,
                    device=self.device,
                    criterion=self.criterion,
                    class_names=self.label_mapping.index_to_species,
                    split_name="val",
                    amp_enabled=self.use_amp,
                    max_batches=self.config.training.max_eval_batches,
                )
                last_val_summary = val_summary
                self.logger.info(
                    "Epoch %d val | loss=%.6f | top1=%.4f | macro_f1=%.4f | balanced_acc=%.4f",
                    epoch,
                    val_summary.loss or 0.0,
                    val_summary.top1_accuracy,
                    val_summary.macro_f1,
                    val_summary.balanced_accuracy,
                )

                metric_value = float(getattr(val_summary, self.config.training.best_metric))
                if self._metric_improved(metric_value):
                    self.state.best_metric = metric_value
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
                        metric_value,
                    )
                else:
                    self.state.epochs_without_improvement += 1

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
                "global_step": self.state.global_step,
                "elapsed_seconds": time.time() - epoch_start,
                "learning_rates": self.scheduler.get_last_lrs(),
                "train": train_summary.to_dict(),
                "val": val_summary.to_dict() if val_summary is not None else None,
                "best_metric": self.state.best_metric,
                "best_epoch": self.state.best_epoch,
            }
            self._append_metrics_history(history_payload)

            if self.config.training.max_steps is not None and self.state.global_step >= self.config.training.max_steps:
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

        final_payload: dict[str, Any] = {
            **_seed_metadata(self.config.seed),
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "global_step": self.state.global_step,
            "output_dir": str(self.paths.output_dir),
            "best_checkpoint": str(self.paths.best_checkpoint_path if self.paths.best_checkpoint_path.exists() else self.paths.last_checkpoint_path),
        }

        best_checkpoint = (
            self.paths.best_checkpoint_path
            if self.paths.best_checkpoint_path.exists()
            else self.paths.last_checkpoint_path
        )
        self.load_checkpoint(best_checkpoint, restore_training_state=False)

        val_artifacts = self.evaluate_and_export(
            split_name="val",
            dataloader=self.val_loader,
            output_prefix="best_val",
        )
        final_payload["val"] = val_artifacts.summary.to_dict()

        if self.test_loader is not None:
            test_artifacts = self.evaluate_and_export(
                split_name="test",
                dataloader=self.test_loader,
                output_prefix="test",
            )
            final_payload["test"] = test_artifacts.summary.to_dict()

        summary_path = self.paths.report_dir / "training_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=2, ensure_ascii=False)
        self.logger.info("Wrote training summary to %s", summary_path)
        return final_payload

    def _train_one_epoch(self, epoch: int) -> ClassificationMetricsSummary:
        del epoch  # currently unused for scheduling beyond global step
        self.model.train()
        accumulator = ClassificationMetricAccumulator(
            self.label_mapping.index_to_species,
            split_name="train",
        )

        for batch_index, batch in enumerate(self.train_loader):
            if self.config.training.max_steps is not None and self.state.global_step >= self.config.training.max_steps:
                break

            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with _autocast_context(self.device, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.config.training.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.grad_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            accumulator.update(logits.detach(), targets.detach(), loss=loss.detach())
            self.state.global_step += 1
            self.scheduler.step(self.state.global_step)

            if (
                self.state.global_step == 1
                or self.state.global_step % self.config.training.log_every_n_steps == 0
            ):
                running_summary = accumulator.compute()
                backbone_lr, head_lr = self._current_group_lrs()
                self.logger.info(
                    "Step %d | batch=%d | loss=%.6f | top1=%.4f | lr_backbone=%.8f | lr_head=%.8f",
                    self.state.global_step,
                    batch_index,
                    running_summary.loss or 0.0,
                    running_summary.top1_accuracy,
                    backbone_lr,
                    head_lr,
                )

        return accumulator.compute()

    def evaluate_and_export(
        self,
        *,
        split_name: str,
        dataloader: DataLoader[Any],
        output_prefix: str,
    ) -> EvaluationArtifacts:
        summary, accumulator = evaluate_model(
            self.model,
            dataloader,
            device=self.device,
            criterion=self.criterion,
            class_names=self.label_mapping.index_to_species,
            split_name=split_name,
            amp_enabled=self.use_amp,
            max_batches=self.config.training.max_eval_batches,
        )
        confusion_matrix_path = self.paths.report_dir / f"{output_prefix}_confusion_matrix.csv"
        per_class_report_path = self.paths.report_dir / f"{output_prefix}_per_class_report.csv"
        summary_json_path = self.paths.report_dir / f"{output_prefix}_metrics.json"
        accumulator.export_confusion_matrix_csv(confusion_matrix_path)
        accumulator.export_per_class_report_csv(per_class_report_path, summary=summary)
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                _summary_with_seed(summary.to_dict(), seed=self.config.seed),
                handle,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info(
            "Exported %s metrics to %s, %s, %s",
            split_name,
            summary_json_path,
            confusion_matrix_path,
            per_class_report_path,
        )
        return EvaluationArtifacts(
            summary=summary,
            confusion_matrix_path=confusion_matrix_path,
            per_class_report_path=per_class_report_path,
            summary_json_path=summary_json_path,
        )
