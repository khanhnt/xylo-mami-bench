"""Modality-aware transform presets for XyloMaMi-Bench data loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    raise ModuleNotFoundError(
        "torchvision is required for XyloMaMi-Bench transforms. "
        "Install it with `pip install torchvision`."
    ) from exc

SUPPORTED_IMAGE_SIZES = frozenset({384, 448})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_INPUT_MODES = frozenset({"rgb", "grayscale"})

TransformFactory = Callable[[int], transforms.Compose]


@dataclass(frozen=True)
class TransformBundle:
    macro: transforms.Compose
    micro: transforms.Compose

    def as_dict(self) -> dict[str, transforms.Compose]:
        return {"macro": self.macro, "micro": self.micro}


def _validate_image_size(image_size: int) -> int:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported image_size={image_size}. Supported values: {sorted(SUPPORTED_IMAGE_SIZES)}."
        )
    return image_size


def _eval_resize_size(image_size: int) -> int:
    return int(round(image_size / 0.875))


def _normalize_input_mode(input_mode: str) -> str:
    normalized = input_mode.strip().lower()
    if normalized not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode='{input_mode}'. Expected one of {sorted(SUPPORTED_INPUT_MODES)}."
        )
    return normalized


def _color_conversion_layers(input_mode: str) -> list[transforms.Transform]:
    resolved_mode = _normalize_input_mode(input_mode)
    if resolved_mode == "grayscale":
        return [transforms.Grayscale(num_output_channels=3)]
    return []


def _rgb_jitter_layers(*, brightness: float, contrast: float, saturation: float, hue: float) -> list[transforms.Transform]:
    return [
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    ]


def _grayscale_jitter_layers(*, brightness: float, contrast: float) -> list[transforms.Transform]:
    return [
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=0.0,
            hue=0.0,
        )
    ]


def _normalize_layers() -> list[transforms.Transform]:
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_macro_train_transform(image_size: int, *, input_mode: str = "rgb") -> transforms.Compose:
    size = _validate_image_size(image_size)
    resolved_mode = _normalize_input_mode(input_mode)
    jitter_layers = (
        _rgb_jitter_layers(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05,
            hue=0.02,
        )
        if resolved_mode == "rgb"
        else _grayscale_jitter_layers(
            brightness=0.10,
            contrast=0.10,
        )
    )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            *_color_conversion_layers(resolved_mode),
            *jitter_layers,
            *_normalize_layers(),
        ]
    )


def build_micro_train_transform(image_size: int, *, input_mode: str = "rgb") -> transforms.Compose:
    size = _validate_image_size(image_size)
    resolved_mode = _normalize_input_mode(input_mode)
    jitter_layers = (
        _rgb_jitter_layers(
            brightness=0.05,
            contrast=0.05,
            saturation=0.02,
            hue=0.01,
        )
        if resolved_mode == "rgb"
        else _grayscale_jitter_layers(
            brightness=0.05,
            contrast=0.05,
        )
    )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            *_color_conversion_layers(resolved_mode),
            *jitter_layers,
            *_normalize_layers(),
        ]
    )


def build_eval_transform(image_size: int, *, input_mode: str = "rgb") -> transforms.Compose:
    size = _validate_image_size(image_size)
    resize_size = _eval_resize_size(size)
    resolved_mode = _normalize_input_mode(input_mode)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            *_color_conversion_layers(resolved_mode),
            *_normalize_layers(),
        ]
    )


def _resolve_modality_input_modes(
    *,
    input_mode: str = "rgb",
    macro_input_mode: str | None = None,
    micro_input_mode: str | None = None,
) -> tuple[str, str]:
    shared_mode = _normalize_input_mode(input_mode)
    resolved_macro_mode = _normalize_input_mode(macro_input_mode or shared_mode)
    resolved_micro_mode = _normalize_input_mode(micro_input_mode or shared_mode)
    return resolved_macro_mode, resolved_micro_mode


def build_transform_bundle(
    split: str,
    image_size: int,
    *,
    input_mode: str = "rgb",
    macro_input_mode: str | None = None,
    micro_input_mode: str | None = None,
) -> TransformBundle:
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test.")
    resolved_macro_mode, resolved_micro_mode = _resolve_modality_input_modes(
        input_mode=input_mode,
        macro_input_mode=macro_input_mode,
        micro_input_mode=micro_input_mode,
    )

    if normalized_split == "train":
        return TransformBundle(
            macro=build_macro_train_transform(image_size, input_mode=resolved_macro_mode),
            micro=build_micro_train_transform(image_size, input_mode=resolved_micro_mode),
        )

    return TransformBundle(
        macro=build_eval_transform(image_size, input_mode=resolved_macro_mode),
        micro=build_eval_transform(image_size, input_mode=resolved_micro_mode),
    )


def build_transforms(
    split: str,
    image_size: int,
    *,
    input_mode: str = "rgb",
    macro_input_mode: str | None = None,
    micro_input_mode: str | None = None,
) -> dict[str, transforms.Compose]:
    return build_transform_bundle(
        split,
        image_size,
        input_mode=input_mode,
        macro_input_mode=macro_input_mode,
        micro_input_mode=micro_input_mode,
    ).as_dict()
