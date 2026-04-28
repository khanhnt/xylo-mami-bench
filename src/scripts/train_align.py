#!/usr/bin/env python3
"""Train the XyloMaMi-Bench dual-encoder alignment model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
    parser = argparse.ArgumentParser(description="Train a XyloMaMi-Bench dual-encoder alignment model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional override for output_dir.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed override. When --output_dir is not provided, the resolved "
            "output_dir is suffixed with _seed<seed> so checkpoints are seed-specific."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional override for training.max_steps to run a short validation pass.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Optional override for training.max_epochs for short trial runs.",
    )
    parser.add_argument(
        "--quick_check",
        action="store_true",
        help=(
            "Enable a short validation mode. When set, defaults to a short run with "
            "max_steps=10, max_epochs=1, num_workers=0, max_eval_batches=2, and "
            "log_every_n_steps=1 unless those values are explicitly overridden."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for dataset.num_workers.",
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
        "--macro_warmstart_checkpoint",
        type=Path,
        default=None,
        help="Optional baseline checkpoint for macro-branch warm-start loading.",
    )
    parser.add_argument(
        "--micro_warmstart_checkpoint",
        type=Path,
        default=None,
        help="Optional baseline checkpoint for micro-branch warm-start loading.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in the form key=value.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from src.engine.trainer_align import (
        DualEncoderTrainer,
        parse_experiment_config,
        prepare_experiment_paths,
        serialize_experiment_config,
        setup_logger,
    )
    from src.utils.config import apply_overrides, dump_config, load_config, set_nested_value

    raw_config = load_config(args.config)
    raw_config = apply_overrides(raw_config, args.overrides)

    if args.seed is not None:
        set_nested_value(raw_config, "seed", int(args.seed))
    if args.output_dir is not None:
        set_nested_value(raw_config, "output_dir", str(args.output_dir))
    elif args.seed is not None:
        set_nested_value(
            raw_config,
            "output_dir",
            _append_seed_suffix(raw_config.get("output_dir", "outputs"), args.seed),
        )
    if args.resume is not None:
        set_nested_value(raw_config, "training.resume_from", str(args.resume))
    if args.device is not None:
        set_nested_value(raw_config, "device", args.device)
    if args.max_steps is not None:
        set_nested_value(raw_config, "training.max_steps", args.max_steps)
    if args.max_epochs is not None:
        set_nested_value(raw_config, "training.max_epochs", args.max_epochs)
    if args.quick_check:
        if args.max_steps is None:
            set_nested_value(raw_config, "training.max_steps", 10)
        if args.max_epochs is None:
            set_nested_value(raw_config, "training.max_epochs", 1)
        if args.num_workers is None:
            set_nested_value(raw_config, "dataset.num_workers", 0)
        set_nested_value(raw_config, "training.max_eval_batches", 2)
        set_nested_value(raw_config, "training.log_every_n_steps", 1)
    if args.num_workers is not None:
        set_nested_value(raw_config, "dataset.num_workers", args.num_workers)
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
    if args.macro_warmstart_checkpoint is not None:
        set_nested_value(
            raw_config,
            "warmstart.macro_checkpoint",
            str(args.macro_warmstart_checkpoint),
        )
    if args.micro_warmstart_checkpoint is not None:
        set_nested_value(
            raw_config,
            "warmstart.micro_checkpoint",
            str(args.micro_warmstart_checkpoint),
        )

    config = parse_experiment_config(raw_config)
    paths = prepare_experiment_paths(config.output_dir)
    dump_config(serialize_experiment_config(config), paths.resolved_config_path)
    logger = setup_logger(paths.log_path, name=f"train_{config.experiment_name}")
    logger.info("Loaded config from %s", args.config)
    logger.info("Resolved config written to %s", paths.resolved_config_path)
    if args.quick_check:
        logger.info(
            "Short validation mode enabled | max_steps=%s | max_epochs=%s | num_workers=%s | max_eval_batches=%s",
            config.training.max_steps,
            config.training.max_epochs,
            config.dataset.num_workers,
            config.training.max_eval_batches,
        )
    if any(
        path is not None
        for path in (
            config.warmstart.macro_checkpoint,
            config.warmstart.micro_checkpoint,
            config.warmstart.macro_rgb_checkpoint,
            config.warmstart.macro_gray_checkpoint,
            config.warmstart.micro_rgb_checkpoint,
            config.warmstart.micro_gray_checkpoint,
        )
    ):
        logger.info(
            "Warm-start config | macro=%s | micro=%s | macro_rgb=%s | macro_gray=%s | "
            "micro_rgb=%s | micro_gray=%s | strict=%s",
            config.warmstart.macro_checkpoint,
            config.warmstart.micro_checkpoint,
            config.warmstart.macro_rgb_checkpoint,
            config.warmstart.macro_gray_checkpoint,
            config.warmstart.micro_rgb_checkpoint,
            config.warmstart.micro_gray_checkpoint,
            config.warmstart.strict,
        )

    trainer = DualEncoderTrainer(config, logger=logger, paths=paths)
    trainer.fit()
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
