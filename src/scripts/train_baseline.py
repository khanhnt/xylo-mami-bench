#!/usr/bin/env python3
"""Train XyloMaMi-Bench baseline classification experiments."""

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
    parser = argparse.ArgumentParser(description="Train a XyloMaMi-Bench baseline classifier.")
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
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in the form key=value.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from src.engine.trainer_baseline import (
        BaselineTrainer,
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

    config = parse_experiment_config(raw_config)
    paths = prepare_experiment_paths(config.output_dir)
    dump_config(serialize_experiment_config(config), paths.resolved_config_path)
    logger = setup_logger(paths.log_path, name=f"train_{config.experiment_name}")
    logger.info("Loaded config from %s", args.config)
    logger.info("Resolved config written to %s", paths.resolved_config_path)
    if config.warmstart.rgb_checkpoint is not None or config.warmstart.gray_checkpoint is not None:
        logger.info(
            "Warm-start config | rgb=%s | gray=%s | strict=%s",
            config.warmstart.rgb_checkpoint,
            config.warmstart.gray_checkpoint,
            config.warmstart.strict,
        )

    trainer = BaselineTrainer(config, logger=logger, paths=paths)
    trainer.fit()
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
