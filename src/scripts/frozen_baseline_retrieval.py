#!/usr/bin/env python3
"""Evaluate frozen public pretrained baselines on P3 cross-scale retrieval."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import load_manifest_samples
from src.datasets.taxonomy import extract_genus
from src.scripts.bootstrap_retrieval_ci import (
    BootstrapMetric,
    bootstrap_retrieval_metrics,
)
from src.utils.config import load_config


DEFAULT_OUTPUT_DIR = Path("outputs/frozen_baselines/p3")
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_WEIGHTS_LABEL = "openai/clip-vit-base-patch32 (CLIP ViT-B/32)"
OURS_RGB_ALIGN = {
    "M2m_exact_R1": {
        "point_estimate": 0.795,
        "ci_lower": 0.777,
        "ci_upper": 0.815,
    },
    "m2M_exact_R1": {
        "point_estimate": 0.736,
        "ci_lower": 0.649,
        "ci_upper": 0.844,
    },
}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    species: str
    genus: str
    image_id: str


class FrozenImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                rgb_image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image: {record.image_path}") from exc
        image_value: Image.Image | torch.Tensor
        image_value = self.transform(rgb_image) if self.transform is not None else rgb_image
        return {
            "image": image_value,
            "species": record.species,
            "genus": record.genus,
            "image_path": str(record.image_path),
            "image_id": record.image_id,
        }


@dataclass(frozen=True)
class P3Records:
    macro: tuple[ImageRecord, ...]
    micro: tuple[ImageRecord, ...]
    split_csv: Path


@dataclass(frozen=True)
class BaselineEmbeddings:
    macro_emb: np.ndarray
    micro_emb: np.ndarray
    macro_labels: np.ndarray
    micro_labels: np.ndarray
    macro_genus: np.ndarray
    micro_genus: np.ndarray
    macro_ids: np.ndarray
    micro_ids: np.ndarray
    weights: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p3_config",
        type=Path,
        required=True,
        help="Path to the P3 experiment config YAML used only for split paths and image roots.",
    )
    parser.add_argument(
        "--macro_image_root",
        type=Path,
        default=None,
        help="Root folder for macro image_rel_path resolution.",
    )
    parser.add_argument(
        "--micro_image_root",
        type=Path,
        default=None,
        help="Root folder for micro image_rel_path resolution.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reports and cached embeddings will be written.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for frozen encoder inference.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size.")
    parser.add_argument(
        "--run_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or skip CLIP ViT-B/32 image encoder baseline.",
    )
    parser.add_argument(
        "--clip_model_name",
        "--clip_model_id",
        type=str,
        default=DEFAULT_CLIP_MODEL_NAME,
        help=(
            "Hugging Face CLIP model id for transformers. "
            "Default maps CLIP ViT-B/32 to openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument(
        "--run_convnext",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or skip frozen ImageNet ConvNeXt-Small baseline.",
    )
    parser.add_argument(
        "--load_cached_embeddings",
        action="store_true",
        help="Load existing .npy embeddings from output_dir instead of extracting when all cache files exist.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 is safest on macOS and Colab notebooks.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("frozen_baseline_retrieval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "frozen_baseline_retrieval.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_device(device_name: str, logger: logging.Logger) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _config_image_root(raw_config: dict[str, Any], modality: str) -> Path | None:
    image_root_payload = raw_config.get("dataset", {}).get("image_root_override", {})
    if not isinstance(image_root_payload, dict):
        return None
    value = image_root_payload.get(modality)
    return resolve_path(value) if value else None


def load_p3_records(
    *,
    config_path: Path,
    macro_image_root: Path | None,
    micro_image_root: Path | None,
    logger: logging.Logger,
) -> P3Records:
    raw_config = load_config(config_path)
    dataset_payload = raw_config.get("dataset", {})
    split_value = dataset_payload.get("test_split_csv")
    if not split_value:
        raise ValueError(f"No dataset.test_split_csv configured in {config_path}.")
    split_csv = resolve_path(split_value)
    if split_csv is None or not split_csv.exists():
        raise FileNotFoundError(f"P3 test split CSV not found: {split_value}")

    macro_root = resolve_path(macro_image_root) or _config_image_root(raw_config, "macro")
    micro_root = resolve_path(micro_image_root) or _config_image_root(raw_config, "micro")
    samples = load_manifest_samples(split_csv, mode="joint_alignment", modality=("macro", "micro"))
    macro_records: list[ImageRecord] = []
    micro_records: list[ImageRecord] = []
    for sample in samples:
        root = macro_root if sample.modality == "macro" else micro_root
        image_path = root / Path(sample.image_rel_path) if root is not None and sample.image_rel_path else sample.image_path
        genus = sample.genus or extract_genus(sample.species)
        record = ImageRecord(
            image_path=image_path,
            species=sample.species,
            genus=genus,
            image_id=str(image_path),
        )
        if sample.modality == "macro":
            macro_records.append(record)
        elif sample.modality == "micro":
            micro_records.append(record)

    if not macro_records:
        raise ValueError(f"No macro records found in {split_csv}.")
    if not micro_records:
        raise ValueError(f"No micro records found in {split_csv}.")
    logger.info(
        "Loaded P3 test records | split=%s | macro=%d | micro=%d",
        split_csv,
        len(macro_records),
        len(micro_records),
    )
    return P3Records(macro=tuple(macro_records), micro=tuple(micro_records), split_csv=split_csv)


def collate_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = [item["image"] for item in batch]
    if images and isinstance(images[0], torch.Tensor):
        image_payload: torch.Tensor | list[Image.Image] = torch.stack(images, dim=0)
    else:
        image_payload = images
    return {
        "image": image_payload,
        "species": [str(item["species"]) for item in batch],
        "genus": [str(item["genus"]) for item in batch],
        "image_id": [str(item["image_id"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
    }


def build_convnext_transform() -> Callable[[Image.Image], torch.Tensor]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _pool_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        features = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
    elif features.ndim > 2:
        features = features.flatten(1)
    return features


def load_convnext_model(device: torch.device, logger: logging.Logger) -> tuple[torch.nn.Module, str]:
    try:
        import timm

        model = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k_384",
            pretrained=True,
            num_classes=0,
        )
        weights = "ImageNet pretrained (timm convnext_small.fb_in22k_ft_in1k_384)"
        logger.info("Loaded frozen ConvNeXt-Small via timm.")
    except Exception as timm_exc:
        logger.warning("Could not load timm ConvNeXt-Small: %s", timm_exc)
        try:
            from torchvision.models import ConvNeXt_Small_Weights, convnext_small

            model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            model.classifier = torch.nn.Identity()
            weights = "ImageNet pretrained (torchvision ConvNeXt_Small_Weights.IMAGENET1K_V1)"
            logger.info("Loaded frozen ConvNeXt-Small via torchvision fallback.")
        except Exception as torchvision_exc:
            raise RuntimeError(
                "Failed to load ConvNeXt-Small from timm and torchvision."
            ) from torchvision_exc
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights


def extract_torch_model_embeddings(
    *,
    model: torch.nn.Module,
    records: Sequence[ImageRecord],
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = FrozenImageDataset(records, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    all_features: list[np.ndarray] = []
    all_species: list[str] = []
    all_genus: list[str] = []
    all_ids: list[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            features = model(images)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = _pool_features(features.float())
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu().numpy().astype(np.float32))
            all_species.extend(batch["species"])
            all_genus.extend(batch["genus"])
            all_ids.extend(batch["image_id"])
    return (
        np.concatenate(all_features, axis=0),
        np.asarray(all_species, dtype=object),
        np.asarray(all_genus, dtype=object),
        np.asarray(all_ids, dtype=object),
    )


def _load_clip_image_processor(model_name: str, logger: logging.Logger) -> Any:
    try:
        from transformers import CLIPImageProcessor

        return CLIPImageProcessor.from_pretrained(model_name, use_fast=False)
    except Exception as image_processor_exc:
        logger.warning(
            "Could not load CLIPImageProcessor for '%s': %s. Falling back to CLIPProcessor.",
            model_name,
            image_processor_exc,
        )
        from transformers import CLIPProcessor

        try:
            return CLIPProcessor.from_pretrained(model_name, use_fast=False)
        except TypeError:
            return CLIPProcessor.from_pretrained(model_name)


def _coerce_clip_features(output: Any, model: torch.nn.Module) -> torch.Tensor:
    if torch.is_tensor(output):
        return output.float()
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds.float()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        pooled = output.pooler_output
        projection = getattr(model, "visual_projection", None)
        if projection is not None:
            pooled = projection(pooled)
        return pooled.float()
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            try:
                return _coerce_clip_features(item, model)
            except TypeError:
                continue
    raise TypeError(f"Unsupported CLIP image feature output type: {type(output).__name__}")


def _forward_transformers_clip_image(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        output = model.get_image_features(pixel_values=pixel_values)
    else:
        output = model(pixel_values=pixel_values)
    return _coerce_clip_features(output, model)


def try_extract_clip_transformers(
    *,
    macro_records: Sequence[ImageRecord],
    micro_records: Sequence[ImageRecord],
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
) -> BaselineEmbeddings | None:
    try:
        from transformers import CLIPModel, CLIPVisionModelWithProjection
    except Exception as exc:
        logger.warning("transformers CLIP is not available: %s", exc)
        return None

    model_source = "CLIPVisionModelWithProjection"
    try:
        model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    except Exception as exc:
        logger.warning(
            "Could not load CLIPVisionModelWithProjection '%s': %s. Falling back to CLIPModel.",
            model_name,
            exc,
        )
        try:
            model = CLIPModel.from_pretrained(model_name).to(device)
            model_source = "CLIPModel"
        except Exception as clip_model_exc:
            logger.warning("Could not load transformers CLIP model '%s': %s", model_name, clip_model_exc)
            return None

    try:
        processor = _load_clip_image_processor(model_name, logger)
    except Exception as exc:
        logger.warning("Could not load transformers CLIP image processor '%s': %s", model_name, exc)
        return None

    logger.info(
        "Loaded CLIP ViT-B/32 image encoder via transformers %s model_id=%s.",
        model_source,
        model_name,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def extract(records: Sequence[ImageRecord], desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset = FrozenImageDataset(records, transform=None)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        all_features: list[np.ndarray] = []
        all_species: list[str] = []
        all_genus: list[str] = []
        all_ids: list[str] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, unit="batch"):
                inputs = processor(images=batch["image"], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
                features = _forward_transformers_clip_image(model, pixel_values)
                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu().numpy().astype(np.float32))
                all_species.extend(batch["species"])
                all_genus.extend(batch["genus"])
                all_ids.extend(batch["image_id"])
        return (
            np.concatenate(all_features, axis=0),
            np.asarray(all_species, dtype=object),
            np.asarray(all_genus, dtype=object),
            np.asarray(all_ids, dtype=object),
        )

    macro_emb, macro_labels, macro_genus, macro_ids = extract(macro_records, "CLIP macro")
    micro_emb, micro_labels, micro_genus, micro_ids = extract(micro_records, "CLIP micro")
    return BaselineEmbeddings(
        macro_emb=macro_emb,
        micro_emb=micro_emb,
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        macro_genus=macro_genus,
        micro_genus=micro_genus,
        macro_ids=macro_ids,
        micro_ids=micro_ids,
        weights=f"{CLIP_WEIGHTS_LABEL} (transformers: {model_name})",
    )


def try_extract_clip_openai(
    *,
    macro_records: Sequence[ImageRecord],
    micro_records: Sequence[ImageRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
) -> BaselineEmbeddings | None:
    try:
        import clip
    except Exception as exc:
        logger.warning("openai clip package is not available: %s", exc)
        return None
    try:
        model, preprocess = clip.load("ViT-B/32", device=str(device))
    except Exception as exc:
        logger.warning("Could not load openai clip ViT-B/32: %s", exc)
        return None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def extract(records: Sequence[ImageRecord], desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset = FrozenImageDataset(records, transform=preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        all_features: list[np.ndarray] = []
        all_species: list[str] = []
        all_genus: list[str] = []
        all_ids: list[str] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, unit="batch"):
                images = batch["image"].to(device, non_blocking=True)
                features = model.encode_image(images).float()
                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu().numpy().astype(np.float32))
                all_species.extend(batch["species"])
                all_genus.extend(batch["genus"])
                all_ids.extend(batch["image_id"])
        return (
            np.concatenate(all_features, axis=0),
            np.asarray(all_species, dtype=object),
            np.asarray(all_genus, dtype=object),
            np.asarray(all_ids, dtype=object),
        )

    macro_emb, macro_labels, macro_genus, macro_ids = extract(macro_records, "CLIP macro")
    micro_emb, micro_labels, micro_genus, micro_ids = extract(micro_records, "CLIP micro")
    return BaselineEmbeddings(
        macro_emb=macro_emb,
        micro_emb=micro_emb,
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        macro_genus=macro_genus,
        micro_genus=micro_genus,
        macro_ids=macro_ids,
        micro_ids=micro_ids,
        weights=CLIP_WEIGHTS_LABEL,
    )


def cache_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    if prefix == "convnext_frozen":
        macro_name = "convnext_frozen_macro_emb.npy"
        micro_name = "convnext_frozen_micro_emb.npy"
    elif prefix == "clip":
        macro_name = "clip_macro_emb.npy"
        micro_name = "clip_micro_emb.npy"
    else:
        macro_name = f"{prefix}_macro_emb.npy"
        micro_name = f"{prefix}_micro_emb.npy"
    return {
        "macro_emb": output_dir / macro_name,
        "micro_emb": output_dir / micro_name,
        "macro_labels": output_dir / "macro_labels.npy",
        "micro_labels": output_dir / "micro_labels.npy",
        "macro_genus": output_dir / "macro_genus.npy",
        "micro_genus": output_dir / "micro_genus.npy",
        "macro_ids": output_dir / "macro_image_ids.npy",
        "micro_ids": output_dir / "micro_image_ids.npy",
    }


def load_cached_embeddings(output_dir: Path, prefix: str, *, weights: str, logger: logging.Logger) -> BaselineEmbeddings | None:
    paths = cache_paths(output_dir, prefix)
    required = ("macro_emb", "micro_emb", "macro_labels", "micro_labels")
    if not all(paths[key].exists() for key in required):
        return None
    logger.info("Loading cached embeddings for %s from %s", prefix, output_dir)
    macro_labels = np.load(paths["macro_labels"], allow_pickle=True)
    micro_labels = np.load(paths["micro_labels"], allow_pickle=True)
    macro_genus = (
        np.load(paths["macro_genus"], allow_pickle=True)
        if paths["macro_genus"].exists()
        else np.asarray([extract_genus(str(label)) for label in macro_labels], dtype=object)
    )
    micro_genus = (
        np.load(paths["micro_genus"], allow_pickle=True)
        if paths["micro_genus"].exists()
        else np.asarray([extract_genus(str(label)) for label in micro_labels], dtype=object)
    )
    macro_ids = (
        np.load(paths["macro_ids"], allow_pickle=True)
        if paths["macro_ids"].exists()
        else np.asarray([f"macro:{index}" for index in range(len(macro_labels))], dtype=object)
    )
    micro_ids = (
        np.load(paths["micro_ids"], allow_pickle=True)
        if paths["micro_ids"].exists()
        else np.asarray([f"micro:{index}" for index in range(len(micro_labels))], dtype=object)
    )
    return BaselineEmbeddings(
        macro_emb=np.load(paths["macro_emb"]).astype(np.float32),
        micro_emb=np.load(paths["micro_emb"]).astype(np.float32),
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        macro_genus=macro_genus,
        micro_genus=micro_genus,
        macro_ids=macro_ids,
        micro_ids=micro_ids,
        weights=weights,
    )


def save_cached_embeddings(output_dir: Path, prefix: str, embeddings: BaselineEmbeddings, logger: logging.Logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(output_dir, prefix)
    np.save(paths["macro_emb"], embeddings.macro_emb)
    np.save(paths["micro_emb"], embeddings.micro_emb)
    np.save(paths["macro_labels"], embeddings.macro_labels)
    np.save(paths["micro_labels"], embeddings.micro_labels)
    np.save(paths["macro_genus"], embeddings.macro_genus)
    np.save(paths["micro_genus"], embeddings.micro_genus)
    np.save(paths["macro_ids"], embeddings.macro_ids)
    np.save(paths["micro_ids"], embeddings.micro_ids)
    logger.info("Saved cached embeddings for %s to %s", prefix, output_dir)


def extract_convnext_embeddings(
    records: P3Records,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
) -> BaselineEmbeddings:
    model, weights = load_convnext_model(device, logger)
    transform = build_convnext_transform()
    macro_emb, macro_labels, macro_genus, macro_ids = extract_torch_model_embeddings(
        model=model,
        records=records.macro,
        transform=transform,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        desc="ConvNeXt macro",
    )
    micro_emb, micro_labels, micro_genus, micro_ids = extract_torch_model_embeddings(
        model=model,
        records=records.micro,
        transform=transform,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        desc="ConvNeXt micro",
    )
    return BaselineEmbeddings(
        macro_emb=macro_emb,
        micro_emb=micro_emb,
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        macro_genus=macro_genus,
        micro_genus=micro_genus,
        macro_ids=macro_ids,
        micro_ids=micro_ids,
        weights=weights,
    )


def extract_clip_embeddings(
    records: P3Records,
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
) -> BaselineEmbeddings | None:
    embeddings = try_extract_clip_transformers(
        macro_records=records.macro,
        micro_records=records.micro,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger,
    )
    if embeddings is not None:
        return embeddings
    return try_extract_clip_openai(
        macro_records=records.macro,
        micro_records=records.micro,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger,
    )


def metric_to_json(metric: BootstrapMetric) -> dict[str, float]:
    return {
        "point_estimate": float(metric.point_estimate),
        "ci_lower": float(metric.ci_lower),
        "ci_upper": float(metric.ci_upper),
    }


def run_baseline_bootstrap(
    embeddings: BaselineEmbeddings,
    *,
    n_bootstrap: int,
    seed: int,
    logger: logging.Logger,
    baseline_name: str,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    exact_m2m = bootstrap_retrieval_metrics(
        embeddings.macro_emb,
        embeddings.micro_emb,
        embeddings.macro_labels,
        embeddings.micro_labels,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=seed,
        direction="M2m",
        macro_ids=embeddings.macro_ids,
        micro_ids=embeddings.micro_ids,
        match_name="exact",
        dataset_name=f"P3 {baseline_name}",
        logger=logger,
    )
    metrics["M2m_exact_R1"] = metric_to_json(exact_m2m.r1)
    metrics["M2m_exact_R5"] = metric_to_json(exact_m2m.r5)

    exact_m2m_reverse = bootstrap_retrieval_metrics(
        embeddings.macro_emb,
        embeddings.micro_emb,
        embeddings.macro_labels,
        embeddings.micro_labels,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=seed + 100_000,
        direction="m2M",
        macro_ids=embeddings.macro_ids,
        micro_ids=embeddings.micro_ids,
        match_name="exact",
        dataset_name=f"P3 {baseline_name}",
        logger=logger,
    )
    metrics["m2M_exact_R1"] = metric_to_json(exact_m2m_reverse.r1)
    metrics["m2M_exact_R5"] = metric_to_json(exact_m2m_reverse.r5)

    genus_m2m = bootstrap_retrieval_metrics(
        embeddings.macro_emb,
        embeddings.micro_emb,
        embeddings.macro_genus,
        embeddings.micro_genus,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=seed + 200_000,
        direction="M2m",
        macro_ids=embeddings.macro_ids,
        micro_ids=embeddings.micro_ids,
        match_name="genus",
        dataset_name=f"P3 {baseline_name}",
        logger=logger,
    )
    metrics["M2m_genus_R1"] = metric_to_json(genus_m2m.r1)

    genus_m2m_reverse = bootstrap_retrieval_metrics(
        embeddings.macro_emb,
        embeddings.micro_emb,
        embeddings.macro_genus,
        embeddings.micro_genus,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=seed + 300_000,
        direction="m2M",
        macro_ids=embeddings.macro_ids,
        micro_ids=embeddings.micro_ids,
        match_name="genus",
        dataset_name=f"P3 {baseline_name}",
        logger=logger,
    )
    metrics["m2M_genus_R1"] = metric_to_json(genus_m2m_reverse.r1)
    return metrics


def format_pct(value: float | dict[str, float]) -> str:
    point = value["point_estimate"] if isinstance(value, dict) else value
    return f"{point * 100.0:.1f}"


def format_ci(metric: dict[str, float]) -> str:
    return f"({metric['ci_lower'] * 100.0:.1f}--{metric['ci_upper'] * 100.0:.1f})"


def format_stdout_ci(metric: dict[str, float]) -> str:
    return f"({metric['ci_lower'] * 100.0:.1f} - {metric['ci_upper'] * 100.0:.1f}%)"


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_comparison_table(path: Path, results: dict[str, Any]) -> None:
    rows = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & M$\to$m R@1 & 95\% CI & m$\to$M R@1 & 95\% CI \\",
        r"\midrule",
    ]
    row_specs = [
        ("convnext_frozen", "ConvNeXt frozen (ImageNet)"),
        ("clip_vit_b32", "CLIP ViT-B/32 (zero-shot)"),
    ]
    for key, label in row_specs:
        if key not in results:
            continue
        p3 = results[key]["P3"]
        m2m = p3["M2m_exact_R1"]
        reverse = p3["m2M_exact_R1"]
        rows.append(
            f"{label} & {format_pct(m2m)} & {format_ci(m2m)} & "
            f"{format_pct(reverse)} & {format_ci(reverse)} \\\\"
        )
    rows.extend(
        [
            r"\midrule",
            r"\textbf{Ours (RGB align)}  & \textbf{79.5} & (77.7--81.5) & \textbf{73.6} & (64.9--84.4) \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def log_summary(results: dict[str, Any], logger: logging.Logger) -> None:
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║           Frozen Baseline Retrieval Summary                 ║",
        "╠══════════════════════════════════════════════════════════════╣",
    ]
    if "convnext_frozen" in results:
        p3 = results["convnext_frozen"]["P3"]
        lines.append(
            f"║  ConvNeXt frozen  M→m R@1: {format_pct(p3['M2m_exact_R1']):>5}% "
            f"{format_stdout_ci(p3['M2m_exact_R1']):<18} ║"
        )
        lines.append(
            f"║  ConvNeXt frozen  m→M R@1: {format_pct(p3['m2M_exact_R1']):>5}% "
            f"{format_stdout_ci(p3['m2M_exact_R1']):<18} ║"
        )
        lines.append("╠══════════════════════════════════════════════════════════════╣")
    if "clip_vit_b32" in results:
        p3 = results["clip_vit_b32"]["P3"]
        lines.append(
            f"║  CLIP ViT-B/32    M→m R@1: {format_pct(p3['M2m_exact_R1']):>5}% "
            f"{format_stdout_ci(p3['M2m_exact_R1']):<18} ║"
        )
        lines.append(
            f"║  CLIP ViT-B/32    m→M R@1: {format_pct(p3['m2M_exact_R1']):>5}% "
            f"{format_stdout_ci(p3['m2M_exact_R1']):<18} ║"
        )
        lines.append("╠══════════════════════════════════════════════════════════════╣")
    lines.append("║  Ours (RGB align) M→m R@1:  79.5% (77.7 - 81.5%)           ║")
    lines.append("║  Ours (RGB align) m→M R@1:  73.6% (64.9 - 84.4%)           ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    logger.info("\n%s", "\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_bootstrap <= 0:
        raise ValueError("--n_bootstrap must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    output_dir = resolve_path(args.output_dir) or args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    device = resolve_device(args.device, logger)
    logger.info(
        "Frozen baseline retrieval | p3_config=%s | output_dir=%s | n_bootstrap=%d | device=%s",
        args.p3_config,
        output_dir,
        args.n_bootstrap,
        device,
    )

    p3_records: P3Records | None = None

    def get_records() -> P3Records:
        nonlocal p3_records
        if p3_records is None:
            p3_records = load_p3_records(
                config_path=resolve_path(args.p3_config) or args.p3_config,
                macro_image_root=args.macro_image_root,
                micro_image_root=args.micro_image_root,
                logger=logger,
            )
        return p3_records

    results: dict[str, Any] = {}

    if args.run_convnext:
        convnext_embeddings = (
            load_cached_embeddings(
                output_dir,
                "convnext_frozen",
                weights="ImageNet pretrained (timm convnext_small.fb_in22k_ft_in1k_384)",
                logger=logger,
            )
            if args.load_cached_embeddings
            else None
        )
        if convnext_embeddings is None:
            try:
                convnext_embeddings = extract_convnext_embeddings(
                    get_records(),
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    logger=logger,
                )
                save_cached_embeddings(output_dir, "convnext_frozen", convnext_embeddings, logger)
            except Exception as exc:
                logger.warning("Skipping ConvNeXt frozen baseline: %s", exc)
                convnext_embeddings = None
        if convnext_embeddings is not None:
            results["convnext_frozen"] = {
                "weights": convnext_embeddings.weights,
                "P3": run_baseline_bootstrap(
                    convnext_embeddings,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                    logger=logger,
                    baseline_name="ConvNeXt frozen",
                ),
            }

    if args.run_clip:
        clip_embeddings = (
            load_cached_embeddings(
                output_dir,
                "clip",
                weights=CLIP_WEIGHTS_LABEL,
                logger=logger,
            )
            if args.load_cached_embeddings
            else None
        )
        if clip_embeddings is None:
            try:
                clip_embeddings = extract_clip_embeddings(
                    get_records(),
                    model_name=args.clip_model_name,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    logger=logger,
                )
                if clip_embeddings is not None:
                    save_cached_embeddings(output_dir, "clip", clip_embeddings, logger)
            except Exception as exc:
                logger.warning("Skipping CLIP ViT-B/32 baseline: %s", exc)
                clip_embeddings = None
        if clip_embeddings is not None:
            results["clip_vit_b32"] = {
                "weights": clip_embeddings.weights,
                "P3": run_baseline_bootstrap(
                    clip_embeddings,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 500_000,
                    logger=logger,
                    baseline_name="CLIP ViT-B/32",
                ),
            }

    if not results:
        raise RuntimeError("No frozen baselines were evaluated. Check dependencies and cache files.")

    results_payload = {
        **results,
        "ours_rgb_align_reference": {
            "weights": "P3 RGB align, warm-up-guarded macro-priority checkpoint protocol",
            "P3": OURS_RGB_ALIGN,
            "note": "Reference row from the current manuscript; frozen baseline rows are computed by this script.",
        },
    }
    write_results_json(output_dir / "frozen_baseline_results.json", results_payload)
    write_comparison_table(output_dir / "comparison_table.txt", results)
    logger.info("Wrote frozen baseline results to %s", output_dir / "frozen_baseline_results.json")
    logger.info("Wrote LaTeX comparison table to %s", output_dir / "comparison_table.txt")
    log_summary(results, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
