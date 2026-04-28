#!/usr/bin/env python3
"""Run P3 macro-priority robustness experiments across multiple seeds."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEEDS = (42, 2025, 3407)
DEFAULT_MODELS = ("p3_rgb_align", "p3_gray_align", "p3_rgb_gray_gated_v3")
DEFAULT_OUTPUT_ROOT = Path("results/p3_multiseed")
CHECKPOINT_RULE = "macro_priority_best_ckpt"
FEATURE_SOURCE = "embedding"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: Path


MODEL_SPECS: dict[str, ModelSpec] = {
    "p3_rgb_align": ModelSpec(
        name="p3_rgb_align",
        config_path=Path("configs/experiment/p3_align_v2.yaml"),
    ),
    "p3_gray_align": ModelSpec(
        name="p3_gray_align",
        config_path=Path("configs/experiment/p3_align_v2_grayscale.yaml"),
    ),
    "p3_rgb_gray_gated_v3": ModelSpec(
        name="p3_rgb_gray_gated_v3",
        config_path=Path("configs/experiment/p3_align_v3_rgbgray_gated.yaml"),
    ),
}

METRIC_COLUMNS = (
    "model",
    "seed",
    "config_path",
    "run_name",
    "run_dir",
    "checkpoint_path",
    "checkpoint_rule",
    "feature_source",
    "best_epoch",
    "best_metric",
    "best_macro_epoch",
    "best_macro_metric",
    "best_tradeoff_epoch",
    "best_tradeoff_metric",
    "macro_top1_accuracy",
    "macro_balanced_accuracy",
    "macro_macro_f1",
    "micro_top1_accuracy",
    "micro_balanced_accuracy",
    "micro_macro_f1",
    "mean_balanced_accuracy",
    "weighted_mean_balanced_accuracy",
    "tradeoff_score",
    "retrieval_stage_name",
    "retrieval_stage_relation_policy",
    "macro_to_micro_exact_r1",
    "macro_to_micro_exact_r5",
    "macro_to_micro_genus_r1",
    "macro_to_micro_genus_r5",
    "micro_to_macro_exact_r1",
    "micro_to_macro_exact_r5",
    "micro_to_macro_genus_r1",
    "micro_to_macro_genus_r5",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P3 RGB, gray, and RGB-gray gated v3 experiments across multiple seeds."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seed values to run. Default: 42 2025 3407.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(DEFAULT_MODELS),
        help="Subset of P3 experiment aliases to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned output paths without executing or writing metrics.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually execute training/evaluation commands. Without --execute, the runner "
            "defaults to dry-run mode as a safety guard."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed stages and resume incomplete training from last.ckpt when available.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for multi-seed runs and summaries.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def command_text(command: Sequence[str | Path]) -> str:
    return shlex.join(str(item) for item in command)


def run_command(command: Sequence[str | Path], *, dry_run: bool) -> None:
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(item) for item in command], cwd=REPO_ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)!r}.")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def validate_model_config(spec: ModelSpec) -> None:
    from src.utils.config import load_config

    config = load_config(spec.config_path)
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError(f"{spec.config_path} does not define a training mapping.")
    best_metric = str(training.get("best_metric", "")).strip()
    if best_metric != "macro_balanced_accuracy":
        raise ValueError(
            f"{spec.name} must use macro-priority checkpointing, but "
            f"{spec.config_path} sets training.best_metric={best_metric!r}."
        )


def run_name(model_name: str, seed: int) -> str:
    return f"{model_name}_seed{seed}"


def training_complete(run_dir: Path) -> bool:
    return (
        (run_dir / "reports" / "training_summary.json").exists()
        and (run_dir / "checkpoints" / "best.ckpt").exists()
    )


def classification_complete(run_dir: Path) -> bool:
    report_dir = run_dir / "eval" / "test" / "reports"
    return (
        (report_dir / "test_macro_metrics.json").exists()
        and (report_dir / "test_micro_metrics.json").exists()
        and (report_dir / "test_alignment_metrics.json").exists()
    )


def retrieval_complete(run_dir: Path) -> bool:
    retrieval_dir = run_dir / "retrieval"
    return (
        (retrieval_dir / "test_retrieval_summary_embedding.json").exists()
        and (retrieval_dir / "test_retrieval_summary_embedding.csv").exists()
    )


def build_train_command(spec: ModelSpec, seed: int, run_dir: Path, *, resume: bool) -> list[str | Path]:
    command: list[str | Path] = [
        sys.executable,
        "src/scripts/train_align.py",
        "--config",
        spec.config_path,
        "--seed",
        str(seed),
        "--output_dir",
        run_dir,
    ]
    last_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if resume and last_checkpoint.exists() and not training_complete(run_dir):
        command.extend(["--resume", last_checkpoint])
    return command


def build_eval_command(spec: ModelSpec, seed: int, run_dir: Path, checkpoint_path: Path) -> list[str | Path]:
    return [
        sys.executable,
        "src/scripts/eval_model.py",
        "--config",
        spec.config_path,
        "--checkpoint",
        checkpoint_path,
        "--split",
        "test",
        "--output_dir",
        run_dir / "eval" / "test",
        "--seed",
        str(seed),
        "--set",
        f"output_dir={run_dir}",
    ]


def build_retrieval_command(
    spec: ModelSpec,
    seed: int,
    run_dir: Path,
    checkpoint_path: Path,
) -> list[str | Path]:
    name = run_name(spec.name, seed)
    return [
        sys.executable,
        "src/scripts/eval_align_retrieval.py",
        "--run",
        f"{name}::{spec.config_path}::{checkpoint_path}",
        "--split",
        "test",
        "--output_dir",
        run_dir / "retrieval",
        "--feature_source",
        FEATURE_SOURCE,
        "--seed",
        str(seed),
    ]


def require_checkpoint(run_dir: Path) -> Path:
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected macro-priority checkpoint not found: {checkpoint_path}. "
            "Training must finish before evaluation/retrieval."
        )
    return checkpoint_path


def get_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def summarize_run(spec: ModelSpec, seed: int, run_dir: Path) -> dict[str, Any]:
    checkpoint_path = require_checkpoint(run_dir)
    training_summary = load_json(run_dir / "reports" / "training_summary.json")
    macro_metrics = load_json(run_dir / "eval" / "test" / "reports" / "test_macro_metrics.json")
    micro_metrics = load_json(run_dir / "eval" / "test" / "reports" / "test_micro_metrics.json")
    alignment_metrics = load_json(run_dir / "eval" / "test" / "reports" / "test_alignment_metrics.json")
    retrieval_summary = load_json(run_dir / "retrieval" / "test_retrieval_summary_embedding.json")

    retrieval_runs = retrieval_summary.get("runs", [])
    if not isinstance(retrieval_runs, list) or len(retrieval_runs) != 1:
        raise ValueError(
            f"Expected exactly one retrieval run in {run_dir / 'retrieval'}, got {len(retrieval_runs)}."
        )
    retrieval_run = retrieval_runs[0]
    if not isinstance(retrieval_run, Mapping):
        raise TypeError("Retrieval run summary must be a JSON object.")

    combined_metrics = alignment_metrics.get("combined_metrics", {})
    if not isinstance(combined_metrics, Mapping):
        combined_metrics = {}

    row = {
        "model": spec.name,
        "seed": seed,
        "config_path": str(resolve_repo_path(spec.config_path)),
        "run_name": run_name(spec.name, seed),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_rule": CHECKPOINT_RULE,
        "feature_source": FEATURE_SOURCE,
        "best_epoch": training_summary.get("best_epoch"),
        "best_metric": training_summary.get("best_metric"),
        "best_macro_epoch": training_summary.get("best_macro_epoch"),
        "best_macro_metric": training_summary.get("best_macro_metric"),
        "best_tradeoff_epoch": training_summary.get("best_tradeoff_epoch"),
        "best_tradeoff_metric": training_summary.get("best_tradeoff_metric"),
        "macro_top1_accuracy": macro_metrics.get("top1_accuracy"),
        "macro_balanced_accuracy": macro_metrics.get("balanced_accuracy"),
        "macro_macro_f1": macro_metrics.get("macro_f1"),
        "micro_top1_accuracy": micro_metrics.get("top1_accuracy"),
        "micro_balanced_accuracy": micro_metrics.get("balanced_accuracy"),
        "micro_macro_f1": micro_metrics.get("macro_f1"),
        "mean_balanced_accuracy": combined_metrics.get("mean_balanced_accuracy"),
        "weighted_mean_balanced_accuracy": combined_metrics.get("weighted_mean_balanced_accuracy"),
        "tradeoff_score": combined_metrics.get("tradeoff_weighted_mean_balanced_accuracy"),
        "retrieval_stage_name": retrieval_run.get("stage_name"),
        "retrieval_stage_relation_policy": retrieval_run.get("stage_relation_policy"),
        "macro_to_micro_exact_r1": get_nested(retrieval_run, "macro_to_micro", "exact_recall_at_1"),
        "macro_to_micro_exact_r5": get_nested(retrieval_run, "macro_to_micro", "exact_recall_at_5"),
        "macro_to_micro_genus_r1": get_nested(retrieval_run, "macro_to_micro", "genus_recall_at_1"),
        "macro_to_micro_genus_r5": get_nested(retrieval_run, "macro_to_micro", "genus_recall_at_5"),
        "micro_to_macro_exact_r1": get_nested(retrieval_run, "micro_to_macro", "exact_recall_at_1"),
        "micro_to_macro_exact_r5": get_nested(retrieval_run, "micro_to_macro", "exact_recall_at_5"),
        "micro_to_macro_genus_r1": get_nested(retrieval_run, "micro_to_macro", "genus_recall_at_1"),
        "micro_to_macro_genus_r5": get_nested(retrieval_run, "micro_to_macro", "genus_recall_at_5"),
    }
    write_json(run_dir / "per_seed_metrics.json", row)
    write_csv(run_dir / "per_seed_metrics.csv", [row])
    return row


def execute_one(spec: ModelSpec, seed: int, output_root: Path, *, dry_run: bool, resume: bool) -> dict[str, Any] | None:
    run_dir = output_root / "runs" / run_name(spec.name, seed)
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"

    if run_dir.exists() and not resume and not dry_run:
        raise FileExistsError(
            f"Run directory already exists and --resume was not passed: {run_dir}. "
            "Refusing to overwrite an existing run."
        )

    train_command = build_train_command(spec, seed, run_dir, resume=resume)
    eval_command = build_eval_command(spec, seed, run_dir, checkpoint_path)
    retrieval_command = build_retrieval_command(spec, seed, run_dir, checkpoint_path)

    if dry_run:
        print(f"\n# {run_name(spec.name, seed)}")
        run_command(train_command, dry_run=True)
        run_command(eval_command, dry_run=True)
        run_command(retrieval_command, dry_run=True)
        return None

    if resume and training_complete(run_dir):
        print(f"[skip] training complete: {run_dir}")
    else:
        run_command(train_command, dry_run=False)

    checkpoint_path = require_checkpoint(run_dir)

    if resume and classification_complete(run_dir):
        print(f"[skip] classification evaluation complete: {run_dir / 'eval' / 'test'}")
    else:
        run_command(build_eval_command(spec, seed, run_dir, checkpoint_path), dry_run=False)

    if resume and retrieval_complete(run_dir):
        print(f"[skip] retrieval evaluation complete: {run_dir / 'retrieval'}")
    else:
        run_command(build_retrieval_command(spec, seed, run_dir, checkpoint_path), dry_run=False)

    return summarize_run(spec, seed, run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run and args.execute:
        raise ValueError("Use either --dry-run or --execute, not both.")
    dry_run = bool(args.dry_run or not args.execute)
    output_root = resolve_repo_path(args.output_root)

    specs = [MODEL_SPECS[name] for name in args.models]
    for spec in specs:
        validate_model_config(spec)

    if dry_run:
        if not args.dry_run:
            print("# Dry run only. Pass --execute to start training/evaluation.")
        print(f"# Dry run only. Planned output root: {output_root}")
    else:
        print(f"# EXECUTE mode. Output root: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in args.seeds:
            row = execute_one(
                spec,
                int(seed),
                output_root,
                dry_run=dry_run,
                resume=bool(args.resume),
            )
            if row is not None:
                rows.append(row)
                write_csv(output_root / "per_seed_metrics.csv", rows)
                write_json(output_root / "per_seed_metrics.json", {"runs": rows})

    if dry_run:
        return 0

    write_csv(output_root / "per_seed_metrics.csv", rows)
    write_json(output_root / "per_seed_metrics.json", {"runs": rows})
    print(f"Wrote aggregate metrics to {output_root / 'per_seed_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
