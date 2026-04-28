#!/usr/bin/env python3
"""Evaluate a trained XyloMaMi-Bench baseline or alignment checkpoint on a split CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _append_seed_suffix(path_value: str | Path, seed: int) -> str:
    path = Path(path_value)
    suffix = f"_seed{int(seed)}"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(f"{path.name}{suffix}"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a XyloMaMi-Bench baseline or alignment checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to evaluation.checkpoint_path or output_dir/checkpoints/best.ckpt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=("train", "val", "test"),
        help="Named split to evaluate. Defaults to evaluation.split_name from config.",
    )
    parser.add_argument(
        "--split_csv",
        type=Path,
        default=None,
        help="Optional direct split CSV override.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional override for the evaluation report directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch size override.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional dataset.num_workers override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed override. When --output_dir is not provided, the resolved "
            "config output_dir and evaluation report directory are suffixed with _seed<seed>."
        ),
    )
    parser.add_argument(
        "--macro_image_root",
        type=Path,
        default=None,
        help="Optional override root for macro image_rel_path resolution.",
    )
    parser.add_argument(
        "--micro_image_root",
        type=Path,
        default=None,
        help="Optional override root for micro image_rel_path resolution.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in the form key=value.",
    )
    return parser.parse_args(argv)


def resolve_split_csv(
    config: Any,
    split_name: str,
    explicit_split_csv: Path | None,
) -> Path:
    """Resolve which split CSV should be evaluated."""

    if explicit_split_csv is not None:
        if explicit_split_csv.is_absolute():
            return explicit_split_csv
        return (REPO_ROOT / explicit_split_csv).resolve()
    if split_name == "train":
        return config.dataset.train_split_csv
    if split_name == "val":
        return config.dataset.val_split_csv
    if split_name == "test":
        if config.dataset.test_split_csv is None:
            raise ValueError("No dataset.test_split_csv configured.")
        return config.dataset.test_split_csv
    raise ValueError(f"Unsupported split '{split_name}'.")


def _is_baseline_config(raw_config: Mapping[str, Any]) -> bool:
    dataset_payload = raw_config.get("dataset", {})
    if not isinstance(dataset_payload, Mapping):
        return False
    mode = str(dataset_payload.get("mode", "")).strip().lower()
    return bool(mode)


def _resolve_checkpoint_path(
    config_output_dir: Path,
    checkpoint: Path | None,
    configured_checkpoint: Path | None,
) -> Path:
    checkpoint_path = checkpoint or configured_checkpoint or (config_output_dir / "checkpoints" / "best.ckpt")
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_eval_output_dir(config_output_dir: Path, output_dir: Path | None, split_name: str) -> Path:
    eval_output_dir = output_dir or (config_output_dir / "eval" / split_name)
    if not eval_output_dir.is_absolute():
        eval_output_dir = (REPO_ROOT / eval_output_dir).resolve()
    return eval_output_dir


def _resolve_seeded_eval_output_dir(
    config_output_dir: Path,
    output_dir: Path | None,
    split_name: str,
    seed: int | None,
) -> Path:
    seeded_split_name = f"{split_name}_seed{int(seed)}" if seed is not None else split_name
    return _resolve_eval_output_dir(config_output_dir, output_dir, seeded_split_name)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from src.utils.config import apply_overrides, dump_config, load_config, set_nested_value

    raw_config = load_config(args.config)
    raw_config = apply_overrides(raw_config, args.overrides)
    if args.seed is not None:
        set_nested_value(raw_config, "seed", int(args.seed))
        if args.output_dir is None:
            set_nested_value(
                raw_config,
                "output_dir",
                _append_seed_suffix(raw_config.get("output_dir", "outputs"), args.seed),
            )
    split_name_override = args.split
    configured_split_name = split_name_override or str(
        raw_config.get("evaluation", {}).get("split_name", "test")
    ).strip().lower() or "test"
    if args.device is not None:
        set_nested_value(raw_config, "device", args.device)
    if args.num_workers is not None:
        set_nested_value(raw_config, "dataset.num_workers", args.num_workers)
    if args.batch_size is not None:
        set_nested_value(raw_config, "dataset.eval_batch_size", args.batch_size)
    if args.macro_image_root is not None:
        set_nested_value(
            raw_config,
            "dataset.image_root_override.macro",
            str(args.macro_image_root),
        )
    if args.micro_image_root is not None:
        set_nested_value(
            raw_config,
            "dataset.image_root_override.micro",
            str(args.micro_image_root),
        )
    if split_name_override is not None:
        set_nested_value(raw_config, "evaluation.split_name", split_name_override)
    if args.split_csv is not None:
        set_nested_value(raw_config, f"dataset.{configured_split_name}_split_csv", str(args.split_csv))

    if _is_baseline_config(raw_config):
        from src.datasets.manifest_dataset import build_label_mapping_from_csvs
        from src.engine.trainer_baseline import (
            build_eval_dataloader,
            build_model,
            evaluate_model,
            label_mapping_from_checkpoint,
            load_checkpoint_file,
            parse_experiment_config,
            prepare_experiment_paths,
            resolve_class_weights,
            resolve_device,
            serialize_experiment_config,
            setup_logger,
            _validate_checkpoint_for_config,
        )

        config = parse_experiment_config(raw_config)
        split_name = args.split or config.evaluation.split_name or "test"
        split_csv = resolve_split_csv(config, split_name, args.split_csv or config.evaluation.split_csv)
        checkpoint_path = _resolve_checkpoint_path(
            config.output_dir,
            args.checkpoint,
            config.evaluation.checkpoint_path,
        )
        eval_output_dir = _resolve_seeded_eval_output_dir(
            config.output_dir,
            args.output_dir,
            split_name,
            args.seed,
        )
        paths = prepare_experiment_paths(eval_output_dir)
        dump_config(serialize_experiment_config(config), paths.resolved_config_path)
        logger = setup_logger(paths.log_path, name=f"eval_{config.experiment_name}_{split_name}")
        from src.utils.seeding import set_global_seed

        seed_info = set_global_seed(config.seed)
        logger.info(
            "Seed control | seed=%d | cudnn_deterministic=%s | cudnn_benchmark=%s",
            seed_info["seed"],
            seed_info["cudnn_deterministic"],
            seed_info["cudnn_benchmark"],
        )
        logger.info("Evaluating split '%s' from %s", split_name, split_csv)
        logger.info("Loading checkpoint from %s", checkpoint_path)

        checkpoint = load_checkpoint_file(checkpoint_path)
        label_mapping = label_mapping_from_checkpoint(checkpoint)
        if label_mapping is None:
            label_mapping = build_label_mapping_from_csvs(
                [config.dataset.train_split_csv],
                mode=config.dataset.mode,
                label_space_name=f"{config.experiment_name}_{config.dataset.mode}",
            )
        _validate_checkpoint_for_config(checkpoint, config, label_mapping=label_mapping)

        model = build_model(config, pretrained_override=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        device = resolve_device(config.device)
        model = model.to(device)
        model.eval()

        train_dataset_for_weights, _ = build_eval_dataloader(
            config,
            split_csv=config.dataset.train_split_csv,
            split_name="train",
            label_mapping=label_mapping,
            batch_size=args.batch_size or config.dataset.eval_batch_size,
            transform_split="val",
        )
        class_weights = resolve_class_weights(config, train_dataset_for_weights, device=device)
        criterion = model.build_loss(
            class_weights=class_weights,
            label_smoothing=config.loss.label_smoothing,
        ).to(device)

        eval_dataset, eval_loader = build_eval_dataloader(
            config,
            split_csv=split_csv,
            split_name=split_name,
            label_mapping=label_mapping,
            batch_size=args.batch_size or config.dataset.eval_batch_size,
            transform_split=split_name,
        )
        if eval_dataset.num_classes != config.model.num_classes:
            raise ValueError(
                f"Checkpoint/model expect {config.model.num_classes} classes but split resolves to "
                f"{eval_dataset.num_classes} classes."
            )

        summary, accumulator = evaluate_model(
            model,
            eval_loader,
            device=device,
            criterion=criterion,
            class_names=label_mapping.index_to_species,
            split_name=split_name,
            amp_enabled=(config.training.amp and device.type == "cuda"),
            max_batches=config.training.max_eval_batches,
        )

        confusion_matrix_path = paths.report_dir / f"{split_name}_confusion_matrix.csv"
        per_class_report_path = paths.report_dir / f"{split_name}_per_class_report.csv"
        summary_json_path = paths.report_dir / f"{split_name}_metrics.json"
        accumulator.export_confusion_matrix_csv(confusion_matrix_path)
        accumulator.export_per_class_report_csv(per_class_report_path, summary=summary)
        summary_payload = {**seed_info, **summary.to_dict()}
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
        label_mapping.save(paths.label_mapping_path)

        logger.info(
            "%s metrics | loss=%.6f | top1=%.4f | macro_f1=%.4f | balanced_acc=%.4f",
            split_name,
            summary.loss or 0.0,
            summary.top1_accuracy,
            summary.macro_f1,
            summary.balanced_accuracy,
        )
        logger.info("Reports written to %s", paths.report_dir)
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
        return 0

    from src.engine.trainer_align import (
        DualEncoderTrainer,
        load_checkpoint_file,
        parse_experiment_config,
        prepare_experiment_paths,
        serialize_experiment_config,
        setup_logger,
    )

    # Evaluation should reflect only the provided checkpoint, not re-run baseline warm-start.
    set_nested_value(raw_config, "warmstart.macro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_gray_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_gray_checkpoint", None)
    config = parse_experiment_config(raw_config)
    split_name = args.split or config.evaluation.split_name or "test"
    checkpoint_path = _resolve_checkpoint_path(
        config.output_dir,
        args.checkpoint,
        config.evaluation.checkpoint_path,
    )
    eval_output_dir = _resolve_seeded_eval_output_dir(
        config.output_dir,
        args.output_dir,
        split_name,
        args.seed,
    )
    paths = prepare_experiment_paths(eval_output_dir)
    dump_config(serialize_experiment_config(config), paths.resolved_config_path)
    logger = setup_logger(paths.log_path, name=f"eval_{config.experiment_name}_{split_name}")
    logger.info("Evaluating alignment split '%s'", split_name)
    logger.info("Loading checkpoint from %s", checkpoint_path)

    trainer = DualEncoderTrainer(config, logger=logger, paths=paths)
    checkpoint = load_checkpoint_file(checkpoint_path)
    trainer.load_checkpoint(checkpoint_path, restore_training_state=False)

    if split_name == "train":
        dataloader = trainer.train_loader
    elif split_name == "val":
        dataloader = trainer.val_loader
    elif split_name == "test":
        if trainer.test_loader is None:
            raise ValueError("No dataset.test_split_csv configured for this alignment experiment.")
        dataloader = trainer.test_loader
    else:
        raise ValueError(f"Unsupported split '{split_name}'.")

    stage_index = int(checkpoint.get("stage_index", max(0, len(config.stages) - 1)))
    stage_index = max(0, min(stage_index, len(config.stages) - 1))
    stage = config.stages[stage_index]
    logger.info(
        "Using checkpoint stage for evaluation | stage_index=%d | stage_name=%s | relation_policy=%s",
        stage_index,
        stage.name,
        stage.relation_policy,
    )

    artifacts = trainer.evaluate_and_export(
        split_name=split_name,
        dataloader=dataloader,
        output_prefix=split_name,
        dump_embeddings=config.evaluation.dump_embeddings,
        stage=stage,
    )
    logger.info("Reports written to %s", paths.report_dir)
    print(json.dumps(artifacts.summary.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
