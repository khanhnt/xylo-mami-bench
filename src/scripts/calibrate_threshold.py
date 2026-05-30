#!/usr/bin/env python3
"""Calibrate confidence thresholds for the P3 macro screening-confirmation rule."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence


def find_repo_root() -> Path:
    env_root = os.environ.get("WOODID_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEFAULT_CONFIG = Path("configs/experiment/p3_align_v2.yaml")
DEFAULT_CHECKPOINT = Path("best.ckpt")
DEFAULT_SPLIT = Path("data/processed/splits/p3_val.csv")
DEFAULT_OUTPUT_DIR = Path("results/calibration/p3_rgb_align_seed42")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the best macro checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT,
        help="Path to the validation split CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where calibration artifacts will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for inference.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for calibration. Default: 0 to avoid macOS/Python spawn pickling issues.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--macro_image_root",
        type=Path,
        default=None,
        help="Optional root for resolving macro image_rel_path values, e.g. a Google Drive VN100 folder.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("calibrate_threshold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "calibrate_threshold.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_macro_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
    seed: int,
    macro_image_root: Path | None,
    logger: logging.Logger,
) -> tuple[Any, Any, Any, Any, torch.device]:
    from src.engine.trainer_align import (
        _load_model_state_with_gate_compatibility,
        _validate_checkpoint_for_config,
        build_branch_label_mappings,
        build_model,
        load_checkpoint_file,
        parse_experiment_config,
        resolve_device,
    )
    from src.utils.config import load_config, set_nested_value

    raw_config = load_config(config_path)
    set_nested_value(raw_config, "seed", int(seed))
    set_nested_value(raw_config, "device", device_name)
    set_nested_value(raw_config, "dataset.eval_batch_size", int(batch_size))
    set_nested_value(raw_config, "warmstart.macro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_gray_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_gray_checkpoint", None)
    if macro_image_root is not None:
        set_nested_value(raw_config, "dataset.image_root_override.macro", str(macro_image_root))

    config = parse_experiment_config(raw_config)
    device = resolve_device(config.device)
    macro_label_mapping, micro_label_mapping = build_branch_label_mappings(config)
    checkpoint = load_checkpoint_file(checkpoint_path)
    _validate_checkpoint_for_config(
        checkpoint,
        config,
        macro_label_mapping=macro_label_mapping,
        micro_label_mapping=micro_label_mapping,
    )
    model = build_model(config, pretrained_override=False).to(device)
    allow_optional_gate_mismatch = (
        config.model.architecture == "rgbgray_late_fusion"
        and config.model.fusion_mode == "residual"
    )
    _load_model_state_with_gate_compatibility(
        model,
        checkpoint["model_state"],
        allow_optional_gate_mismatch=allow_optional_gate_mismatch,
        logger=logger,
    )
    model.eval()

    logger.info("Model loaded | checkpoint=%s | device=%s", checkpoint_path, device)
    return model, config, macro_label_mapping, checkpoint, device


def build_macro_val_loader(
    *,
    config: Any,
    split_csv: Path,
    label_mapping: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[Any, DataLoader[Any]]:
    from src.datasets.manifest_dataset import ManifestDataset
    from src.datasets.transforms import build_transforms

    transform_map = build_transforms(
        "val",
        config.dataset.image_size,
        input_mode=config.dataset.input_mode,
        macro_input_mode=config.dataset.macro_input_mode,
        micro_input_mode=config.dataset.micro_input_mode,
    )
    dataset = ManifestDataset(
        split_csv,
        mode="macro_classification",
        modality="macro",
        transform=transform_map["macro"],
        label_mapping=label_mapping,
        label_space_name=label_mapping.label_space_name,
        image_root_override=config.dataset.image_root_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(config.device.startswith("cuda") and torch.cuda.is_available()),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def run_inference(
    *,
    model: Any,
    dataloader: DataLoader[Any],
    label_mapping: Any,
    device: torch.device,
    logger: logging.Logger,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Inference", unit="batch")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            true_species = [str(value) for value in batch["species"]]
            true_labels = [label_mapping.index_for(species_name) for species_name in true_species]
            targets = torch.tensor(true_labels, dtype=torch.long, device=device)

            logits = model.forward_macro(images).logits
            if logits.ndim != 2 or logits.shape[1] != label_mapping.num_classes:
                raise ValueError(
                    f"Expected macro logits with shape [B, {label_mapping.num_classes}], "
                    f"received {tuple(logits.shape)}."
                )
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            correct = predicted.eq(targets)

            for index, image_path in enumerate(batch["image_path"]):
                pred_index = int(predicted[index].item())
                records.append(
                    {
                        "image_path": str(image_path),
                        "true_label": int(targets[index].item()),
                        "pred_label": pred_index,
                        "confidence": float(confidence[index].item()),
                        "correct": bool(correct[index].item()),
                    }
                )

    logger.info("Inference complete | samples=%d", len(records))
    return pd.DataFrame.from_records(
        records,
        columns=["image_path", "true_label", "pred_label", "confidence", "correct"],
    )


def calibrate_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    confidences = predictions["confidence"].to_numpy(dtype=np.float64)
    correct = predictions["correct"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for tau in np.linspace(0.0, 1.0, 201):
        high_conf_mask = confidences >= tau
        high_conf_count = int(high_conf_mask.sum())
        precision = float(correct[high_conf_mask].mean()) if high_conf_count else float("nan")
        rows.append(
            {
                "tau": float(tau),
                "coverage": float(high_conf_mask.mean()),
                "precision": precision,
                "n_flagged": int((~high_conf_mask).sum()),
            }
        )
    return pd.DataFrame.from_records(rows, columns=["tau", "coverage", "precision", "n_flagged"])


def select_threshold(calibration: pd.DataFrame, target_precision: float) -> dict[str, Any] | None:
    eligible = calibration[calibration["precision"] >= target_precision]
    if eligible.empty:
        return None
    return eligible.sort_values("tau", kind="stable").iloc[0].to_dict()


def threshold_value(row: dict[str, Any] | None, key: str) -> float | int | None:
    if row is None:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    if key == "n_flagged":
        return int(value)
    return float(value)


def build_summary(
    *,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    tau_95_row: dict[str, Any] | None,
    tau_99_row: dict[str, Any] | None,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    total = int(len(predictions))
    overall_accuracy = float(predictions["correct"].mean()) if total else float("nan")

    def pct_flagged(row: dict[str, Any] | None) -> float | None:
        if row is None or total == 0:
            return None
        return float(int(row["n_flagged"]) / total)

    return {
        "tau_95": threshold_value(tau_95_row, "tau"),
        "coverage_at_tau_95": threshold_value(tau_95_row, "coverage"),
        "precision_at_tau_95": threshold_value(tau_95_row, "precision"),
        "n_flagged_at_tau_95": threshold_value(tau_95_row, "n_flagged"),
        "pct_flagged_at_tau_95": pct_flagged(tau_95_row),
        "tau_99": threshold_value(tau_99_row, "tau"),
        "coverage_at_tau_99": threshold_value(tau_99_row, "coverage"),
        "precision_at_tau_99": threshold_value(tau_99_row, "precision"),
        "n_flagged_at_tau_99": threshold_value(tau_99_row, "n_flagged"),
        "pct_flagged_at_tau_99": pct_flagged(tau_99_row),
        "total_val_samples": total,
        "overall_accuracy": overall_accuracy,
        "model_config": str(config_path),
        "checkpoint": str(checkpoint_path),
    }


def annotate_threshold(ax: plt.Axes, row: dict[str, Any] | None, *, y: float) -> None:
    if row is None:
        return
    tau = float(row["tau"])
    coverage = float(row["coverage"])
    ax.axvline(tau, color="tab:red", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.annotate(
        f"τ={tau:.2f} (cov={coverage:.0%})",
        xy=(tau, y),
        xytext=(6, 0),
        textcoords="offset points",
        rotation=90,
        va="center",
        ha="left",
        fontsize=9,
        color="tab:red",
    )


def save_calibration_figure(
    *,
    calibration: pd.DataFrame,
    tau_95_row: dict[str, Any] | None,
    tau_99_row: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    fig, (precision_ax, coverage_ax) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Confidence threshold calibration — P3 macro branch")

    precision_ax.plot(calibration["tau"], calibration["precision"], color="tab:blue", linewidth=2)
    precision_ax.axhline(0.95, color="0.35", linestyle="--", linewidth=1)
    precision_ax.axhline(0.99, color="0.35", linestyle="--", linewidth=1)
    precision_ax.set_ylabel("Precision")
    precision_ax.set_ylim(0.5, 1.0)
    precision_ax.grid(True, alpha=0.25)
    annotate_threshold(precision_ax, tau_95_row, y=0.72)
    annotate_threshold(precision_ax, tau_99_row, y=0.84)

    coverage_ax.plot(calibration["tau"], calibration["coverage"], color="tab:green", linewidth=2)
    coverage_ax.set_xlabel("Confidence threshold τ")
    coverage_ax.set_ylabel("Coverage")
    coverage_ax.set_ylim(0.0, 1.0)
    coverage_ax.grid(True, alpha=0.25)
    annotate_threshold(coverage_ax, tau_95_row, y=0.30)
    annotate_threshold(coverage_ax, tau_99_row, y=0.45)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_dir / "calibration_curve.png", dpi=300)
    fig.savefig(output_dir / "calibration_curve.pdf", dpi=300)
    plt.close(fig)


def format_threshold_line(name: str, row: dict[str, Any] | None, total: int) -> str:
    if row is None:
        return f"║  {name} = n/a   │ coverage = n/a   │ flagged = n/a/{total:<4} ║"
    tau = float(row["tau"])
    coverage = float(row["coverage"]) * 100.0
    n_flagged = int(row["n_flagged"])
    return f"║  {name} = {tau:>4.2f}  │ coverage = {coverage:>5.1f}% │ flagged = {n_flagged:>3}/{total:<4} ║"


def log_summary_box(
    *,
    logger: logging.Logger,
    summary: dict[str, Any],
    tau_95_row: dict[str, Any] | None,
    tau_99_row: dict[str, Any] | None,
) -> None:
    total = int(summary["total_val_samples"])
    accuracy = float(summary["overall_accuracy"]) * 100.0
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║          Threshold Calibration Summary               ║",
        "╠══════════════════════════════════════════════════════╣",
        f"║  Overall val accuracy : {accuracy:>6.2f}%                       ║",
        f"║  Total val samples    : {total:<4}                         ║",
        "╠══════════════════════════════════════════════════════╣",
        format_threshold_line("τ_95", tau_95_row, total),
        format_threshold_line("τ_99", tau_99_row, total),
        "╚══════════════════════════════════════════════════════╝",
    ]
    logger.info("\n%s", "\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_repo_path(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    split_csv = resolve_repo_path(args.split)
    output_dir = resolve_repo_path(args.output_dir)
    logger = setup_logger(output_dir)

    from src.utils.seeding import set_global_seed

    seed_info = set_global_seed(args.seed)
    logger.info(
        "Seed control | seed=%d | cudnn_deterministic=%s | cudnn_benchmark=%s",
        seed_info["seed"],
        seed_info["cudnn_deterministic"],
        seed_info["cudnn_benchmark"],
    )

    model, config, macro_label_mapping, checkpoint, device = load_macro_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device_name=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        macro_image_root=resolve_repo_path(args.macro_image_root) if args.macro_image_root is not None else None,
        logger=logger,
    )
    checkpoint_mapping = checkpoint.get("macro_label_mapping")
    if isinstance(checkpoint_mapping, dict) and "index_to_species" in checkpoint_mapping:
        checkpoint_species = tuple(str(item) for item in checkpoint_mapping["index_to_species"])
        if checkpoint_species != macro_label_mapping.index_to_species:
            raise ValueError("Checkpoint macro label mapping does not match the training split mapping.")

    macro_dataset, macro_loader = build_macro_val_loader(
        config=config,
        split_csv=split_csv,
        label_mapping=macro_label_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Running macro-only calibration inference | split=%s | samples=%d | classes=%d",
        split_csv,
        len(macro_dataset),
        macro_label_mapping.num_classes,
    )
    logger.info("DataLoader workers=%d", args.num_workers)

    predictions = run_inference(
        model=model,
        dataloader=macro_loader,
        label_mapping=macro_label_mapping,
        device=device,
        logger=logger,
    )
    calibration = calibrate_thresholds(predictions)
    tau_95_row = select_threshold(calibration, 0.95)
    tau_99_row = select_threshold(calibration, 0.99)
    logger.info("Threshold tau_95=%s", "n/a" if tau_95_row is None else f"{tau_95_row['tau']:.4f}")
    logger.info("Threshold tau_99=%s", "n/a" if tau_99_row is None else f"{tau_99_row['tau']:.4f}")

    predictions.to_csv(output_dir / "raw_predictions.csv", index=False)
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)
    summary = build_summary(
        predictions=predictions,
        calibration=calibration,
        tau_95_row=tau_95_row,
        tau_99_row=tau_99_row,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    with (output_dir / "threshold_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_calibration_figure(
        calibration=calibration,
        tau_95_row=tau_95_row,
        tau_99_row=tau_99_row,
        output_dir=output_dir,
    )
    logger.info("Calibration artifacts written to %s", output_dir)
    log_summary_box(logger=logger, summary=summary, tau_95_row=tau_95_row, tau_99_row=tau_99_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
