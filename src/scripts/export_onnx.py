#!/usr/bin/env python3
"""Export the P3 RGB alignment macro branch to ONNX for mobile latency tests."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the macro branch of a P3 RGB alignment checkpoint to ONNX."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the P3 RGB alignment YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained alignment checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/macro_branch_p3_rgb.onnx"),
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=384,
        help="Square input resolution used for the dummy export input.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for PyTorch export. ONNX Runtime verification always runs on CPU.",
    )
    parser.add_argument(
        "--external_data",
        action="store_true",
        help="Store ONNX weights in external data files. By default, weights are embedded in the .onnx file for mobile deployment.",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the newer torch.export-based ONNX exporter. The default legacy exporter is preferred for opset-17 mobile compatibility.",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path, *, must_exist: bool = False) -> Path:
    resolved = path if path.is_absolute() else (REPO_ROOT / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def _clear_warmstart_paths(raw_config: dict[str, Any]) -> None:
    from src.utils.config import set_nested_value

    for key in (
        "warmstart.macro_checkpoint",
        "warmstart.micro_checkpoint",
        "warmstart.macro_rgb_checkpoint",
        "warmstart.macro_gray_checkpoint",
        "warmstart.micro_rgb_checkpoint",
        "warmstart.micro_gray_checkpoint",
    ):
        set_nested_value(raw_config, key, None)


def _extract_macro_branch_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    from src.engine.trainer_align import _extract_model_state_from_checkpoint

    state = _extract_model_state_from_checkpoint(checkpoint)
    prefix = "macro_branch."
    macro_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not macro_state:
        raise ValueError("Checkpoint does not contain any macro_branch.* weights.")
    return macro_state


def _build_macro_branch(config_path: Path, checkpoint_path: Path, device_name: str) -> Any:
    import torch

    from src.engine.trainer_align import load_checkpoint_file, parse_experiment_config
    from src.models.dual_encoder_align import (
        AlignmentEncoderBranch,
        RGBGrayFusionAlignmentEncoderBranch,
    )
    from src.utils.config import load_config, set_nested_value

    raw_config = load_config(config_path)
    _clear_warmstart_paths(raw_config)
    set_nested_value(raw_config, "device", device_name)
    config = parse_experiment_config(raw_config)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is False.")

    checkpoint = load_checkpoint_file(checkpoint_path)
    if config.model.architecture == "rgbgray_late_fusion":
        macro_branch = RGBGrayFusionAlignmentEncoderBranch(
            backbone_name=config.model.macro_backbone_name,
            gray_backbone_name=config.model.macro_gray_backbone_name,
            num_classes=config.model.macro_num_classes,
            pretrained=False,
            gray_pretrained=False,
            image_size=config.dataset.image_size,
            projection_hidden_dims=config.model.macro_projection_hidden_dims or None,
            projection_dim=config.model.macro_projection_dim,
            projection_dropout=config.model.projection_dropout,
            classifier_dropout=config.model.classifier_dropout,
            fusion_hidden_dim=config.model.macro_fusion_hidden_dim,
            fusion_dropout=config.model.fusion_dropout,
            fusion_residual_scale=config.model.fusion_residual_scale,
            fusion_mode=config.model.fusion_mode,
            freeze_backbone=config.model.freeze_macro_backbone,
            trainable_backbone_patterns=config.model.macro_trainable_backbone_patterns,
        )
    else:
        macro_branch = AlignmentEncoderBranch(
            backbone_name=config.model.macro_backbone_name,
            num_classes=config.model.macro_num_classes,
            pretrained=False,
            image_size=config.dataset.image_size,
            projection_hidden_dims=config.model.macro_projection_hidden_dims or None,
            projection_dim=config.model.macro_projection_dim,
            projection_dropout=config.model.projection_dropout,
            classifier_dropout=config.model.classifier_dropout,
            freeze_backbone=config.model.freeze_macro_backbone,
            trainable_backbone_patterns=config.model.macro_trainable_backbone_patterns,
        )
    load_result = macro_branch.load_state_dict(_extract_macro_branch_state(checkpoint), strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected macro branch load result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    macro_branch.to(device)
    macro_branch.eval()
    return macro_branch


def _shape_list(array: Any) -> list[int]:
    return [int(dim) for dim in array.shape]


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def _actual_opsets(onnx_model: Any) -> dict[str, int]:
    opsets: dict[str, int] = {}
    for opset in onnx_model.opset_import:
        domain = opset.domain or "ai.onnx"
        opsets[domain] = int(opset.version)
    return opsets


def _external_data_files(onnx_model: Any, output_path: Path) -> list[Path]:
    files: list[Path] = []
    for initializer in onnx_model.graph.initializer:
        for item in initializer.external_data:
            if item.key == "location" and item.value:
                candidate = (output_path.parent / item.value).resolve()
                if candidate.exists() and candidate not in files:
                    files.append(candidate)
    return files


def _onnx_total_size_mb(output_path: Path, external_files: Sequence[Path]) -> float:
    total_bytes = output_path.stat().st_size
    total_bytes += sum(path.stat().st_size for path in external_files)
    return total_bytes / (1024.0 * 1024.0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import numpy as np
    import torch
    from torch import Tensor, nn

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Missing dependency: onnx. Install it before exporting.") from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Missing dependency: onnxruntime. Install onnxruntime CPU for verification.") from exc

    config_path = _resolve_path(args.config, must_exist=True)
    checkpoint_path = _resolve_path(args.checkpoint, must_exist=True)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_size <= 0:
        raise ValueError("--input_size must be positive.")

    device = torch.device(args.device)
    macro_branch = _build_macro_branch(config_path, checkpoint_path, args.device)

    class MacroBranchONNXWrapper(nn.Module):
        def __init__(self, branch: nn.Module) -> None:
            super().__init__()
            self.branch = branch

        def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
            output = self.branch(image)
            return output.logits, output.embeddings

    wrapper = MacroBranchONNXWrapper(macro_branch).to(device).eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    export_kwargs: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["logits", "embedding"],
        "dynamic_axes": {"image": {0: "batch_size"}},
        "opset_version": args.opset,
        "do_constant_folding": True,
    }
    export_signature = inspect.signature(torch.onnx.export)
    if "dynamo" in export_signature.parameters:
        export_kwargs["dynamo"] = bool(args.dynamo)
    if "external_data" in export_signature.parameters:
        export_kwargs["external_data"] = bool(args.external_data)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            **export_kwargs,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    actual_opsets = _actual_opsets(onnx_model)
    external_files = _external_data_files(onnx_model, output_path)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    cpu_input = dummy.detach().cpu().numpy().astype(np.float32, copy=False)
    outputs = session.run(None, {"image": cpu_input})
    output_names = [output.name for output in session.get_outputs()]
    if len(output_names) != len(outputs):
        raise RuntimeError(
            f"ONNX Runtime returned {len(outputs)} outputs for {len(output_names)} output names."
        )
    output_shapes = {
        name: _shape_list(value)
        for name, value in zip(output_names, outputs)
    }

    expected_outputs = {"logits", "embedding"}
    if set(output_shapes) != expected_outputs:
        raise RuntimeError(f"Unexpected ONNX outputs: {sorted(output_shapes)}")

    onnx_file_size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    size_mb = _onnx_total_size_mb(output_path, external_files)
    input_shape = [1, 3, int(args.input_size), int(args.input_size)]
    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "input_shape": input_shape,
        "output_shapes": output_shapes,
        "requested_opset": int(args.opset),
        "actual_opsets": actual_opsets,
        "exporter": "dynamo" if args.dynamo else "legacy",
        "external_data_requested": bool(args.external_data),
        "onnx_size_mb": round(float(size_mb), 4),
        "onnx_file_size_mb": round(float(onnx_file_size_mb), 4),
        "external_data_files": [str(path) for path in external_files],
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = _manifest_path(output_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Exported ONNX: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Model size: {size_mb:.2f} MB")
    if external_files:
        print(f"External data files: {[str(path) for path in external_files]}")
    print(f"Exporter: {'dynamo' if args.dynamo else 'legacy'}")
    print(f"External data requested: {bool(args.external_data)}")
    print(f"Requested opset: {args.opset}")
    print(f"Actual opsets: {actual_opsets}")
    print(f"Input shape: {input_shape}")
    print(f"Output shapes: {output_shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
