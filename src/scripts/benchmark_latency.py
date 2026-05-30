#!/usr/bin/env python3
"""Benchmark model-forward inference latency for Paper 1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure model-forward inference latency from an experiment config and checkpoint. "
            "Defaults match the manuscript deployment benchmark: batch size 1, "
            "100 warm-up runs, and 100 measured runs."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to evaluation.checkpoint_path or output_dir/checkpoints/best.ckpt.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/latency/<config>_<timestamp>.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0.")
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail if the resolved device is not CUDA.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=100,
        help="Untimed warm-up forward passes before every repeat.",
    )
    parser.add_argument(
        "--measured_runs",
        type=int,
        default=100,
        help="Timed forward passes per repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the whole warm-up + measured-run block for a more robust estimate.",
    )
    parser.add_argument(
        "--branch",
        choices=("macro", "micro", "both"),
        default="macro",
        help=(
            "For alignment checkpoints, benchmark the macro branch, micro branch, or both branches "
            "in one forward pass. Baseline checkpoints ignore this option."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Synthetic input height. Defaults to dataset.image_size.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Synthetic input width. Defaults to dataset.image_size.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Synthetic input channels. Keep 3 for RGB and grayscale-triplet models.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Force CUDA automatic mixed precision during timing.",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA automatic mixed precision during timing.",
    )
    parser.add_argument(
        "--benchmark_cudnn",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark during latency measurement.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in the form key=value.",
    )
    return parser.parse_args(argv)


def _is_baseline_config(raw_config: Mapping[str, Any]) -> bool:
    dataset_payload = raw_config.get("dataset", {})
    if not isinstance(dataset_payload, Mapping):
        return False
    return bool(str(dataset_payload.get("mode", "")).strip())


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


def _default_output_dir(config_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "latency" / f"{config_path.stem}_{timestamp}"


def _load_model_from_config(
    *,
    raw_config: dict[str, Any],
    config_path: Path,
    checkpoint_arg: Path | None,
    device_arg: str | None,
    seed_arg: int | None,
    overrides: Sequence[str],
) -> tuple[Any, Any, Path, str]:
    from src.utils.config import apply_overrides, load_config, set_nested_value

    raw_config = apply_overrides(raw_config, overrides)
    if seed_arg is not None:
        set_nested_value(raw_config, "seed", int(seed_arg))
    if device_arg is not None:
        set_nested_value(raw_config, "device", device_arg)

    if _is_baseline_config(raw_config):
        from src.engine.trainer_baseline import (
            _extract_model_state_from_checkpoint,
            build_model,
            load_checkpoint_file,
            parse_experiment_config,
            resolve_device,
        )

        config = parse_experiment_config(raw_config)
        checkpoint_path = _resolve_checkpoint_path(
            config.output_dir,
            checkpoint_arg,
            config.evaluation.checkpoint_path,
        )
        checkpoint = load_checkpoint_file(checkpoint_path)
        model = build_model(config, pretrained_override=False)
        model.load_state_dict(_extract_model_state_from_checkpoint(checkpoint), strict=True)
        device = resolve_device(config.device)
        return model.to(device).eval(), config, checkpoint_path, "baseline"

    from src.engine.trainer_align import (
        _extract_model_state_from_checkpoint,
        build_model,
        load_checkpoint_file,
        parse_experiment_config,
        resolve_device,
    )

    # Latency measurement should reflect only the target checkpoint, not baseline warm-start files.
    for key in (
        "warmstart.macro_checkpoint",
        "warmstart.micro_checkpoint",
        "warmstart.macro_rgb_checkpoint",
        "warmstart.macro_gray_checkpoint",
        "warmstart.micro_rgb_checkpoint",
        "warmstart.micro_gray_checkpoint",
    ):
        set_nested_value(raw_config, key, None)
    config = parse_experiment_config(raw_config)
    checkpoint_path = _resolve_checkpoint_path(
        config.output_dir,
        checkpoint_arg,
        config.evaluation.checkpoint_path,
    )
    checkpoint = load_checkpoint_file(checkpoint_path)
    model = build_model(config, pretrained_override=False)
    model.load_state_dict(_extract_model_state_from_checkpoint(checkpoint), strict=True)
    device = resolve_device(config.device)
    return model.to(device).eval(), config, checkpoint_path, "alignment"


def _sync(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_forward_fn(
    *,
    model: Any,
    model_kind: str,
    branch: str,
    input_tensor: Any,
    amp_enabled: bool,
    device: Any,
) -> Callable[[], None]:
    import torch

    def baseline_forward() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            _ = model(input_tensor)

    def alignment_forward() -> None:
        kwargs: dict[str, Any]
        if branch == "macro":
            kwargs = {"macro_images": input_tensor}
        elif branch == "micro":
            kwargs = {"micro_images": input_tensor}
        else:
            kwargs = {"macro_images": input_tensor, "micro_images": input_tensor}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            _ = model(**kwargs)

    return baseline_forward if model_kind == "baseline" else alignment_forward


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile for an empty sequence.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summarize(values_ms: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values_ms)
    return {
        "mean_ms": float(statistics.fmean(ordered)),
        "std_ms": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
        "median_ms": float(statistics.median(ordered)),
        "p05_ms": _percentile(ordered, 0.05),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _hardware_payload(device: Any) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        payload.update(
            {
                "cuda_device_index": int(device_index),
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_mb": round(props.total_memory / (1024**2), 2),
            }
        )
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Inference Latency Benchmark",
        "",
        f"- Config: `{payload['config_path']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        f"- Model kind: `{payload['model_kind']}`",
        f"- Branch: `{payload['branch']}`",
        f"- Device: `{payload['hardware'].get('cuda_device_name', payload['hardware']['device'])}`",
        f"- Input: batch={payload['batch_size']}, shape={payload['input_shape']}",
        f"- Warm-up / measured / repeats: {payload['warmup_runs']} / {payload['measured_runs']} / {payload['repeats']}",
        f"- AMP: `{payload['amp_enabled']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean latency (ms) | {payload['mean_ms']:.4f} |",
        f"| Std latency (ms) | {payload['std_ms']:.4f} |",
        f"| Median latency (ms) | {payload['median_ms']:.4f} |",
        f"| P95 latency (ms) | {payload['p95_ms']:.4f} |",
        f"| Min latency (ms) | {payload['min_ms']:.4f} |",
        f"| Max latency (ms) | {payload['max_ms']:.4f} |",
        f"| Peak CUDA memory (MB) | {payload.get('peak_cuda_memory_mb', 0.0):.2f} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be non-negative.")
    if args.measured_runs <= 0:
        raise ValueError("--measured_runs must be positive.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.channels <= 0:
        raise ValueError("--channels must be positive.")

    import torch

    from src.utils.config import dump_config, load_config, to_serializable
    from src.utils.seeding import set_global_seed

    raw_config = load_config(args.config)
    model, config, checkpoint_path, model_kind = _load_model_from_config(
        raw_config=raw_config,
        config_path=args.config,
        checkpoint_arg=args.checkpoint,
        device_arg=args.device,
        seed_arg=args.seed,
        overrides=args.overrides,
    )
    device = next(model.parameters()).device
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(f"--require_cuda was set, but resolved device is {device}.")

    seed_info = set_global_seed(int(getattr(config, "seed", 42)))
    torch.backends.cudnn.benchmark = bool(args.benchmark_cudnn)

    image_size = int(config.dataset.image_size)
    height = args.height or image_size
    width = args.width or image_size
    input_shape = (args.batch_size, args.channels, height, width)
    input_tensor = torch.randn(input_shape, device=device)
    amp_enabled = bool(config.training.amp if args.amp is None else args.amp)
    forward_once = _make_forward_fn(
        model=model,
        model_kind=model_kind,
        branch=args.branch,
        input_tensor=input_tensor,
        amp_enabled=amp_enabled,
        device=device,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(args.config)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_config(to_serializable(getattr(config, "raw_config", {})), output_dir / "resolved_config.yaml")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_rows: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []
    for repeat_index in range(args.repeats):
        for _ in range(args.warmup_runs):
            forward_once()
        _sync(device)

        for run_index in range(args.measured_runs):
            _sync(device)
            start = time.perf_counter()
            forward_once()
            _sync(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            all_latencies_ms.append(elapsed_ms)
            run_rows.append(
                {
                    "repeat": repeat_index + 1,
                    "run": run_index + 1,
                    "latency_ms": f"{elapsed_ms:.8f}",
                }
            )

    summary = _summarize(all_latencies_ms)
    peak_cuda_memory_mb = 0.0
    if device.type == "cuda":
        peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    payload: dict[str, Any] = {
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "model_kind": model_kind,
        "experiment_name": str(getattr(config, "experiment_name", "")),
        "branch": args.branch if model_kind == "alignment" else str(config.dataset.mode),
        "seed": int(getattr(config, "seed", 42)),
        "seed_control": seed_info,
        "batch_size": args.batch_size,
        "input_shape": list(input_shape),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "repeats": args.repeats,
        "total_measured_runs": len(all_latencies_ms),
        "amp_enabled": amp_enabled and device.type == "cuda",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "hardware": _hardware_payload(device),
        **summary,
    }
    summary_row = {
        key: payload[key]
        for key in (
            "experiment_name",
            "model_kind",
            "branch",
            "batch_size",
            "warmup_runs",
            "measured_runs",
            "repeats",
            "total_measured_runs",
            "amp_enabled",
            "cudnn_benchmark",
            "mean_ms",
            "std_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "peak_cuda_memory_mb",
        )
    }
    summary_row["device_name"] = payload["hardware"].get("cuda_device_name", payload["hardware"]["device"])
    summary_row["checkpoint_path"] = payload["checkpoint_path"]

    _write_json(output_dir / "latency_summary.json", payload)
    _write_csv(output_dir / "latency_summary.csv", [summary_row])
    _write_csv(output_dir / "latency_runs.csv", run_rows)
    _write_markdown(output_dir / "latency_summary.md", payload)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
