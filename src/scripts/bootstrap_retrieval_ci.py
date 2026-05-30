#!/usr/bin/env python3
"""Bootstrap confidence intervals for cross-scale retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


FEATURE_KEYS = {
    "embedding": "embedding",
    "projected": "projected_feature",
    "pooled": "pooled_feature",
}


@dataclass(frozen=True)
class EmbeddingBundle:
    macro_emb: np.ndarray
    micro_emb: np.ndarray
    macro_labels: np.ndarray
    micro_labels: np.ndarray
    macro_genus: np.ndarray | None = None
    micro_genus: np.ndarray | None = None
    macro_ids: np.ndarray | None = None
    micro_ids: np.ndarray | None = None
    source: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    r1: float
    r5: float
    r1_hits: np.ndarray
    r5_hits: np.ndarray
    eligible_mask: np.ndarray
    query_labels: np.ndarray


@dataclass(frozen=True)
class BootstrapMetric:
    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    distribution: np.ndarray

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def to_json(self) -> dict[str, float]:
        return {
            "point_estimate": float(self.point_estimate),
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "ci_width": float(self.ci_width),
        }


@dataclass(frozen=True)
class BootstrapResult:
    r1: BootstrapMetric
    r5: BootstrapMetric
    retrieval: RetrievalResult


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% confidence intervals for Paper 1 cross-scale retrieval metrics."
    )
    parser.add_argument(
        "--p1_embeddings",
        type=Path,
        default=None,
        help="Optional path to P1 test embedding dump. Supports repo .pt list records, .npz, .json, or .csv.",
    )
    parser.add_argument(
        "--p3_embeddings",
        type=Path,
        default=None,
        help="Optional path to P3 test embedding dump. Supports repo .pt list records, .npz, .json, or .csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/bootstrap_ci"),
        help="Directory where bootstrap reports and plots will be written.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for percentile intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used only when extracting embeddings from checkpoint/config.",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="embedding",
        choices=tuple(FEATURE_KEYS.keys()),
        help="Feature source to use from repo embedding records.",
    )
    parser.add_argument(
        "--p1_run",
        type=str,
        default=None,
        help="Optional P1 extraction spec: name::config_path::checkpoint_path.",
    )
    parser.add_argument(
        "--p3_run",
        type=str,
        default=None,
        help="Optional P3 extraction spec: name::config_path::checkpoint_path.",
    )
    parser.add_argument(
        "--p1_config",
        type=Path,
        default=None,
        help="Optional P1 config path for on-the-fly embedding extraction.",
    )
    parser.add_argument(
        "--p1_checkpoint",
        type=Path,
        default=None,
        help="Optional P1 checkpoint path for on-the-fly embedding extraction.",
    )
    parser.add_argument(
        "--p3_config",
        type=Path,
        default=None,
        help="Optional P3 config path for on-the-fly embedding extraction.",
    )
    parser.add_argument(
        "--p3_checkpoint",
        type=Path,
        default=None,
        help="Optional P3 checkpoint path for on-the-fly embedding extraction.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Named split used when extracting embeddings on-the-fly.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch size override for on-the-fly extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional num_workers override for on-the-fly extraction.",
    )
    parser.add_argument(
        "--macro_image_root",
        type=Path,
        default=None,
        help="Optional macro image root override for on-the-fly extraction.",
    )
    parser.add_argument(
        "--micro_image_root",
        type=Path,
        default=None,
        help="Optional micro image root override for on-the-fly extraction.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional cap for quick extraction sanity checks.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in key=value form for on-the-fly extraction.",
    )
    return parser.parse_args(argv)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bootstrap_retrieval_ci")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "bootstrap_retrieval_ci.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_run_spec(spec: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in spec.split("::")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid run spec '{spec}'. Expected name::config_path::checkpoint_path.")
    name, config_path, checkpoint_path = parts
    return name, Path(config_path), Path(checkpoint_path)


def _append_seed_suffix(value: str | Path, seed: int) -> str:
    path = Path(value)
    suffix = f"_seed{int(seed)}"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(f"{path.name}{suffix}"))


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _as_str_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def _derive_genus(labels: Sequence[Any]) -> np.ndarray:
    try:
        from src.datasets.taxonomy import extract_genus

        return np.asarray([extract_genus(str(label)) for label in labels], dtype=object)
    except Exception:
        genera = []
        for label in labels:
            text = str(label).strip()
            genera.append(text.split(" ", 1)[0] if text else "")
        return np.asarray(genera, dtype=object)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _records_to_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_source: str,
    source: str,
) -> EmbeddingBundle:
    feature_key = FEATURE_KEYS[feature_source]
    macro_records = [record for record in records if str(record.get("modality")) == "macro"]
    micro_records = [record for record in records if str(record.get("modality")) == "micro"]
    if not macro_records:
        raise ValueError(f"No macro records found in {source}.")
    if not micro_records:
        raise ValueError(f"No micro records found in {source}.")

    missing_macro = sum(1 for record in macro_records if record.get(feature_key) is None)
    missing_micro = sum(1 for record in micro_records if record.get(feature_key) is None)
    if missing_macro or missing_micro:
        raise ValueError(
            f"Feature source '{feature_source}' requires '{feature_key}' in every record, "
            f"but missing macro={missing_macro}, micro={missing_micro} in {source}."
        )

    macro_emb = np.stack([_as_numpy(record[feature_key]).astype(np.float32) for record in macro_records], axis=0)
    micro_emb = np.stack([_as_numpy(record[feature_key]).astype(np.float32) for record in micro_records], axis=0)
    macro_labels = _as_str_array([record["species"] for record in macro_records])
    micro_labels = _as_str_array([record["species"] for record in micro_records])
    macro_genus_raw = [record.get("genus", "") for record in macro_records]
    micro_genus_raw = [record.get("genus", "") for record in micro_records]
    macro_genus = _as_str_array(macro_genus_raw) if any(str(value).strip() for value in macro_genus_raw) else None
    micro_genus = _as_str_array(micro_genus_raw) if any(str(value).strip() for value in micro_genus_raw) else None
    macro_ids = _as_str_array([record.get("image_path", f"macro:{index}") for index, record in enumerate(macro_records)])
    micro_ids = _as_str_array([record.get("image_path", f"micro:{index}") for index, record in enumerate(micro_records)])
    return validate_bundle(
        EmbeddingBundle(
            macro_emb=macro_emb,
            micro_emb=micro_emb,
            macro_labels=macro_labels,
            micro_labels=micro_labels,
            macro_genus=macro_genus,
            micro_genus=micro_genus,
            macro_ids=macro_ids,
            micro_ids=micro_ids,
            source=source,
        )
    )


def _mapping_to_bundle(mapping: Mapping[str, Any], *, feature_source: str, source: str) -> EmbeddingBundle:
    feature_suffix = "" if feature_source == "embedding" else f"_{feature_source}"
    macro_emb = _first_present(
        mapping,
        (
            f"macro_embeddings{feature_suffix}",
            "macro_embeddings",
            "macro_emb",
            "macro",
            "z_macro",
            "macro_z",
        ),
    )
    micro_emb = _first_present(
        mapping,
        (
            f"micro_embeddings{feature_suffix}",
            "micro_embeddings",
            "micro_emb",
            "micro",
            "z_micro",
            "micro_z",
        ),
    )
    macro_labels = _first_present(mapping, ("macro_labels", "macro_species", "species_macro", "labels_macro"))
    micro_labels = _first_present(mapping, ("micro_labels", "micro_species", "species_micro", "labels_micro"))
    if macro_emb is None or micro_emb is None or macro_labels is None or micro_labels is None:
        raise ValueError(
            f"Could not find required macro/micro embeddings and labels in {source}. "
            "Expected keys such as macro_embeddings, micro_embeddings, macro_labels, micro_labels."
        )

    macro_genus = _first_present(mapping, ("macro_genus", "macro_genera", "genus_macro"))
    micro_genus = _first_present(mapping, ("micro_genus", "micro_genera", "genus_micro"))
    macro_ids = _first_present(mapping, ("macro_image_paths", "macro_paths", "image_paths_macro", "macro_ids"))
    micro_ids = _first_present(mapping, ("micro_image_paths", "micro_paths", "image_paths_micro", "micro_ids"))
    return validate_bundle(
        EmbeddingBundle(
            macro_emb=_as_numpy(macro_emb).astype(np.float32),
            micro_emb=_as_numpy(micro_emb).astype(np.float32),
            macro_labels=_as_str_array(macro_labels),
            micro_labels=_as_str_array(micro_labels),
            macro_genus=_as_str_array(macro_genus) if macro_genus is not None else None,
            micro_genus=_as_str_array(micro_genus) if micro_genus is not None else None,
            macro_ids=_as_str_array(macro_ids) if macro_ids is not None else None,
            micro_ids=_as_str_array(micro_ids) if micro_ids is not None else None,
            source=source,
        )
    )


def _csv_to_bundle(path: Path, *, feature_source: str) -> EmbeddingBundle:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV embedding file is empty: {path}")

    embedding_columns = [
        column
        for column in rows[0]
        if column.startswith("embedding_") or column.startswith("feature_") or column.startswith("dim_")
    ]
    if not embedding_columns:
        numeric_columns = []
        for column in rows[0]:
            if column in {"split", "modality", "species", "genus", "image_path"}:
                continue
            try:
                float(rows[0][column])
            except (TypeError, ValueError):
                continue
            numeric_columns.append(column)
        embedding_columns = numeric_columns
    if not embedding_columns:
        raise ValueError(f"Could not infer embedding columns in CSV file: {path}")

    records: list[dict[str, Any]] = []
    for row in rows:
        vector = np.asarray([float(row[column]) for column in embedding_columns], dtype=np.float32)
        records.append(
            {
                "modality": row.get("modality", ""),
                "species": row.get("species", row.get("label", "")),
                "genus": row.get("genus", ""),
                "image_path": row.get("image_path", row.get("path", "")),
                FEATURE_KEYS[feature_source]: vector,
            }
        )
    return _records_to_bundle(records, feature_source=feature_source, source=str(path))


def validate_bundle(bundle: EmbeddingBundle) -> EmbeddingBundle:
    if bundle.macro_emb.ndim != 2 or bundle.micro_emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D matrices in {bundle.source}.")
    if bundle.macro_emb.shape[0] != len(bundle.macro_labels):
        raise ValueError(f"Macro embeddings/labels length mismatch in {bundle.source}.")
    if bundle.micro_emb.shape[0] != len(bundle.micro_labels):
        raise ValueError(f"Micro embeddings/labels length mismatch in {bundle.source}.")
    if bundle.macro_emb.shape[1] != bundle.micro_emb.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ in {bundle.source}: "
            f"macro={bundle.macro_emb.shape[1]}, micro={bundle.micro_emb.shape[1]}."
        )

    macro_genus = bundle.macro_genus
    micro_genus = bundle.micro_genus
    if macro_genus is None or len(macro_genus) != len(bundle.macro_labels):
        macro_genus = _derive_genus(bundle.macro_labels)
    if micro_genus is None or len(micro_genus) != len(bundle.micro_labels):
        micro_genus = _derive_genus(bundle.micro_labels)

    macro_ids = bundle.macro_ids
    micro_ids = bundle.micro_ids
    if macro_ids is not None and len(macro_ids) != len(bundle.macro_labels):
        macro_ids = None
    if micro_ids is not None and len(micro_ids) != len(bundle.micro_labels):
        micro_ids = None

    return EmbeddingBundle(
        macro_emb=np.asarray(bundle.macro_emb, dtype=np.float32),
        micro_emb=np.asarray(bundle.micro_emb, dtype=np.float32),
        macro_labels=bundle.macro_labels,
        micro_labels=bundle.micro_labels,
        macro_genus=macro_genus,
        micro_genus=micro_genus,
        macro_ids=macro_ids,
        micro_ids=micro_ids,
        source=bundle.source,
    )


def load_embedding_bundle(path: Path, *, feature_source: str) -> EmbeddingBundle:
    resolved_path = resolve_path(path)
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    suffix = resolved_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        import torch

        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
        if isinstance(payload, list):
            return _records_to_bundle(payload, feature_source=feature_source, source=str(resolved_path))
        if isinstance(payload, Mapping):
            records = payload.get("records")
            if isinstance(records, list):
                return _records_to_bundle(records, feature_source=feature_source, source=str(resolved_path))
            return _mapping_to_bundle(payload, feature_source=feature_source, source=str(resolved_path))
        raise TypeError(f"Unsupported torch embedding payload in {resolved_path}: {type(payload)!r}")

    if suffix == ".npz":
        payload = np.load(resolved_path, allow_pickle=True)
        return _mapping_to_bundle({key: payload[key] for key in payload.files}, feature_source=feature_source, source=str(resolved_path))

    if suffix == ".json":
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return _records_to_bundle(payload, feature_source=feature_source, source=str(resolved_path))
        if isinstance(payload, Mapping):
            records = payload.get("records")
            if isinstance(records, list):
                return _records_to_bundle(records, feature_source=feature_source, source=str(resolved_path))
            return _mapping_to_bundle(payload, feature_source=feature_source, source=str(resolved_path))
        raise TypeError(f"Unsupported JSON embedding payload in {resolved_path}: {type(payload)!r}")

    if suffix == ".csv":
        return _csv_to_bundle(resolved_path, feature_source=feature_source)

    raise ValueError(f"Unsupported embedding file extension for {resolved_path}.")


def extract_embedding_bundle_from_run(
    run_spec: str,
    *,
    dataset_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> EmbeddingBundle:
    from src.engine.trainer_align import (
        DualEncoderTrainer,
        load_checkpoint_file,
        parse_experiment_config,
        prepare_experiment_paths,
        serialize_experiment_config,
        setup_logger,
    )
    from src.utils.config import apply_overrides, dump_config, load_config, set_nested_value
    from src.utils.seeding import set_global_seed

    name, config_path, checkpoint_path = _parse_run_spec(run_spec)
    config_path = resolve_path(config_path) or config_path
    checkpoint_path = resolve_path(checkpoint_path) or checkpoint_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found for {dataset_name}: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {dataset_name}: {checkpoint_path}")

    raw_config = load_config(config_path)
    raw_config = apply_overrides(raw_config, args.overrides)
    set_nested_value(raw_config, "seed", int(args.seed))
    set_nested_value(raw_config, "device", args.device)
    set_nested_value(raw_config, "evaluation.split_name", args.split)
    set_nested_value(raw_config, "evaluation.dump_embeddings", True)
    set_nested_value(raw_config, "warmstart.macro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.macro_gray_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_rgb_checkpoint", None)
    set_nested_value(raw_config, "warmstart.micro_gray_checkpoint", None)
    if args.batch_size is not None:
        set_nested_value(raw_config, "dataset.eval_batch_size", args.batch_size)
    if args.num_workers is not None:
        set_nested_value(raw_config, "dataset.num_workers", args.num_workers)
    if args.macro_image_root is not None:
        set_nested_value(raw_config, "dataset.image_root_override.macro", str(args.macro_image_root))
    if args.micro_image_root is not None:
        set_nested_value(raw_config, "dataset.image_root_override.micro", str(args.micro_image_root))
    if args.max_eval_batches is not None:
        set_nested_value(raw_config, "training.max_eval_batches", args.max_eval_batches)

    run_output_dir = output_dir / "extracted_embeddings" / _append_seed_suffix(f"{dataset_name}_{name}", args.seed)
    set_nested_value(raw_config, "output_dir", str(run_output_dir))
    config = parse_experiment_config(raw_config)
    paths = prepare_experiment_paths(config.output_dir)
    dump_config(serialize_experiment_config(config), paths.resolved_config_path)
    extraction_logger = setup_logger(paths.log_path, name=f"bootstrap_extract_{dataset_name}_{name}")
    seed_info = set_global_seed(config.seed)
    extraction_logger.info(
        "Seed control | seed=%d | cudnn_deterministic=%s | cudnn_benchmark=%s",
        seed_info["seed"],
        seed_info["cudnn_deterministic"],
        seed_info["cudnn_benchmark"],
    )
    extraction_logger.info("Extracting retrieval embeddings | dataset=%s | run=%s", dataset_name, name)
    extraction_logger.info("Using config=%s", config_path)
    extraction_logger.info("Using checkpoint=%s", checkpoint_path)

    trainer = DualEncoderTrainer(config, logger=extraction_logger, paths=paths)
    checkpoint = load_checkpoint_file(checkpoint_path)
    trainer.load_checkpoint(checkpoint_path, restore_training_state=False)
    if args.split == "train":
        dataloader = trainer.train_loader
    elif args.split == "val":
        dataloader = trainer.val_loader
    else:
        if trainer.test_loader is None:
            raise ValueError(f"No dataset.test_split_csv configured for {dataset_name} run '{name}'.")
        dataloader = trainer.test_loader

    stage_index = int(checkpoint.get("stage_index", max(0, len(config.stages) - 1)))
    stage_index = max(0, min(stage_index, len(config.stages) - 1))
    stage = config.stages[stage_index]
    artifacts = trainer.evaluate_and_export(
        split_name=args.split,
        dataloader=dataloader,
        output_prefix=args.split,
        dump_embeddings=True,
        stage=stage,
    )
    if artifacts.embeddings_dump_path is None:
        raise RuntimeError(f"Embedding dump failed for {dataset_name} run '{name}'.")

    logger.info("Extracted %s embeddings to %s", dataset_name, artifacts.embeddings_dump_path)
    return load_embedding_bundle(artifacts.embeddings_dump_path, feature_source=args.feature_source)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def compute_retrieval(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    query_ids: np.ndarray | None = None,
    gallery_ids: np.ndarray | None = None,
    match_name: str = "exact",
) -> RetrievalResult:
    if query_emb.shape[0] != len(query_labels):
        raise ValueError("query_emb/query_labels length mismatch.")
    if gallery_emb.shape[0] != len(gallery_labels):
        raise ValueError("gallery_emb/gallery_labels length mismatch.")
    if gallery_emb.shape[0] == 0:
        raise ValueError("Gallery is empty.")

    query_features = l2_normalize(query_emb.astype(np.float32))
    gallery_features = l2_normalize(gallery_emb.astype(np.float32))
    similarities = query_features @ gallery_features.T
    self_match_matrix = None
    if query_ids is not None and gallery_ids is not None:
        self_match_matrix = query_ids[:, None] == gallery_ids[None, :]
        similarities = similarities.copy()
        similarities[self_match_matrix] = -np.inf

    max_k = min(5, gallery_features.shape[0])
    top_indices = np.argpartition(-similarities, kth=np.arange(max_k), axis=1)[:, :max_k]
    top_scores = np.take_along_axis(similarities, top_indices, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)

    match_matrix = query_labels[:, None] == gallery_labels[None, :]
    if self_match_matrix is not None:
        match_matrix = np.logical_and(match_matrix, ~self_match_matrix)
    eligible_mask = match_matrix.any(axis=1)
    if not eligible_mask.any():
        raise ValueError(f"No eligible queries have a matching {match_name} label in the gallery.")
    top_matches = np.take_along_axis(match_matrix, top_indices, axis=1)
    r1_hits = top_matches[:, :1].any(axis=1).astype(np.float32)
    r5_hits = top_matches[:, :max_k].any(axis=1).astype(np.float32)
    return RetrievalResult(
        r1=float(r1_hits[eligible_mask].mean()),
        r5=float(r5_hits[eligible_mask].mean()),
        r1_hits=r1_hits,
        r5_hits=r5_hits,
        eligible_mask=eligible_mask,
        query_labels=query_labels,
    )


def _bootstrap_hits(
    hits: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    point_estimate: float,
    desc: str,
    logger: logging.Logger,
) -> BootstrapMetric:
    eligible_hits = hits[eligible_mask].astype(np.float32)
    if eligible_hits.size == 0:
        raise ValueError(f"No eligible hits to bootstrap for {desc}.")

    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    log_interval = max(1, n_bootstrap // 10)
    for index in tqdm(range(n_bootstrap), desc=desc, unit="iter"):
        sampled_indices = rng.integers(0, eligible_hits.size, size=eligible_hits.size)
        values[index] = float(eligible_hits[sampled_indices].mean())
        iteration = index + 1
        if iteration % log_interval == 0 or iteration == n_bootstrap:
            logger.info("%s: %d/%d iterations...", desc, iteration, n_bootstrap)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return BootstrapMetric(
        point_estimate=float(point_estimate),
        mean=float(values.mean()),
        std=std,
        ci_lower=float(lower),
        ci_upper=float(upper),
        distribution=values,
    )


def bootstrap_retrieval_metrics(
    macro_emb: np.ndarray,
    micro_emb: np.ndarray,
    macro_labels: np.ndarray,
    micro_labels: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    direction: str = "M2m",
    macro_ids: np.ndarray | None = None,
    micro_ids: np.ndarray | None = None,
    match_name: str = "exact",
    dataset_name: str | None = None,
    logger: logging.Logger | None = None,
) -> BootstrapResult:
    """Bootstrap R@1/R@5 by resampling the query set while keeping the gallery fixed."""
    if logger is None:
        logger = logging.getLogger("bootstrap_retrieval_ci")
    if direction == "M2m":
        query_emb = macro_emb
        gallery_emb = micro_emb
        query_labels = macro_labels
        gallery_labels = micro_labels
        query_ids = macro_ids
        gallery_ids = micro_ids
        direction_label = "M→m"
    elif direction == "m2M":
        query_emb = micro_emb
        gallery_emb = macro_emb
        query_labels = micro_labels
        gallery_labels = macro_labels
        query_ids = micro_ids
        gallery_ids = macro_ids
        direction_label = "m→M"
    else:
        raise ValueError(f"Unsupported direction '{direction}'. Expected 'M2m' or 'm2M'.")

    retrieval = compute_retrieval(
        query_emb,
        gallery_emb,
        query_labels,
        gallery_labels,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        match_name=match_name,
    )
    dataset_prefix = f"{dataset_name} " if dataset_name else ""
    desc_prefix = f"Bootstrap {dataset_prefix}{direction_label} {match_name}"
    r1 = _bootstrap_hits(
        retrieval.r1_hits,
        retrieval.eligible_mask,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        point_estimate=retrieval.r1,
        desc=f"{desc_prefix} R@1",
        logger=logger,
    )
    r5 = _bootstrap_hits(
        retrieval.r5_hits,
        retrieval.eligible_mask,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 10_003,
        point_estimate=retrieval.r5,
        desc=f"{desc_prefix} R@5",
        logger=logger,
    )
    return BootstrapResult(r1=r1, r5=r5, retrieval=retrieval)


def build_metric_payload(result: BootstrapMetric) -> dict[str, float]:
    return result.to_json()


def run_all_bootstraps(
    bundles: Mapping[str, EmbeddingBundle],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, BootstrapMetric]], dict[str, RetrievalResult]]:
    metric_results: dict[str, dict[str, BootstrapMetric]] = {"P1": {}, "P3": {}}
    retrieval_results: dict[str, RetrievalResult] = {}
    tasks = [
        ("P1", "M2m", "exact", "M2m_exact"),
        ("P1", "m2M", "exact", "m2M_exact"),
        ("P3", "M2m", "exact", "M2m_exact"),
        ("P3", "m2M", "exact", "m2M_exact"),
        ("P3", "M2m", "genus", "M2m_genus"),
        ("P3", "m2M", "genus", "m2M_genus"),
    ]
    for task_index, (dataset_name, direction, match_name, key_prefix) in enumerate(tasks):
        if dataset_name not in bundles:
            logger.info("Skipping %s %s %s because no %s embeddings/run were provided.", dataset_name, direction, match_name, dataset_name)
            continue
        bundle = bundles[dataset_name]
        if match_name == "exact":
            macro_labels = bundle.macro_labels
            micro_labels = bundle.micro_labels
        else:
            if bundle.macro_genus is None or bundle.micro_genus is None:
                logger.info("Skipping %s %s genus metrics because genus labels are unavailable.", dataset_name, direction)
                continue
            macro_labels = bundle.macro_genus
            micro_labels = bundle.micro_genus

        logger.info("Bootstrap %s %s %s | n_bootstrap=%d", dataset_name, direction, match_name, n_bootstrap)
        result = bootstrap_retrieval_metrics(
            bundle.macro_emb,
            bundle.micro_emb,
            macro_labels,
            micro_labels,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + task_index * 100_000,
            direction=direction,
            macro_ids=bundle.macro_ids,
            micro_ids=bundle.micro_ids,
            match_name=match_name,
            dataset_name=dataset_name,
            logger=logger,
        )
        metric_results[dataset_name][f"{key_prefix}_R1"] = result.r1
        if not (dataset_name == "P3" and match_name == "genus"):
            metric_results[dataset_name][f"{key_prefix}_R5"] = result.r5
        retrieval_results[f"{dataset_name}_{direction}_{match_name}"] = result.retrieval
    return metric_results, retrieval_results


def write_summary_json(
    path: Path,
    *,
    metrics: Mapping[str, Mapping[str, BootstrapMetric]],
    args: argparse.Namespace,
    bundles: Mapping[str, EmbeddingBundle],
) -> None:
    payload: dict[str, Any] = {
        "config": {
            "n_bootstrap": int(args.n_bootstrap),
            "confidence": float(args.confidence),
            "seed": int(args.seed),
            "feature_source": args.feature_source,
        },
        "inputs": {dataset_name: bundle.source for dataset_name, bundle in bundles.items()},
        "skipped_datasets": [
            dataset_name for dataset_name in ("P1", "P3") if dataset_name not in bundles
        ],
    }
    for dataset_name in ("P1", "P3"):
        payload[dataset_name] = {
            metric_name: build_metric_payload(metric)
            for metric_name, metric in metrics.get(dataset_name, {}).items()
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def plot_bootstrap_distributions(
    path: Path,
    *,
    metrics: Mapping[str, Mapping[str, BootstrapMetric]],
    confidence: float,
) -> None:
    items: list[tuple[str, BootstrapMetric]] = []
    for dataset_name in ("P1", "P3"):
        for metric_name, metric in metrics.get(dataset_name, {}).items():
            items.append((f"{dataset_name} {metric_name}", metric))
    if not items:
        return

    cols = 2
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, 3.0 * rows)), squeeze=False)
    for axis, (name, metric) in zip(axes.ravel(), items):
        values = metric.distribution * 100.0
        lower = metric.ci_lower * 100.0
        upper = metric.ci_upper * 100.0
        point = metric.point_estimate * 100.0
        axis.hist(values, bins=30, color="#4C78A8", alpha=0.75, edgecolor="white")
        axis.axvspan(lower, upper, color="#72B7B2", alpha=0.25)
        axis.axvline(point, color="#D62728", linewidth=2)
        axis.set_title(f"{name}\n{confidence:.0%} CI: [{lower:.1f}, {upper:.1f}]")
        axis.set_xlabel("Recall (%)")
        axis.set_ylabel("Bootstrap count")
    for axis in axes.ravel()[len(items):]:
        axis.axis("off")
    fig.suptitle("Bootstrap retrieval metric distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_per_species_r1(
    path: Path,
    *,
    retrieval_results: Mapping[str, RetrievalResult],
) -> None:
    subplot_specs = [
        ("P1 M→m exact R@1", retrieval_results.get("P1_M2m_exact")),
        ("P3 M→m exact R@1", retrieval_results.get("P3_M2m_exact")),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), squeeze=False)
    for axis, (title, retrieval) in zip(axes.ravel(), subplot_specs):
        if retrieval is None:
            axis.axis("off")
            continue
        labels = retrieval.query_labels[retrieval.eligible_mask]
        hits = retrieval.r1_hits[retrieval.eligible_mask]
        species = sorted(set(str(label) for label in labels), key=lambda label: (float(hits[labels == label].mean()), label))
        data = [hits[labels == species_name] for species_name in species]
        positions = np.arange(1, len(species) + 1)
        axis.boxplot(data, positions=positions, patch_artist=True, showfliers=False)
        means = np.asarray([float(values.mean()) for values in data])
        axis.scatter(positions, means, s=12, color="#1F77B4", zorder=3)
        zero_positions = positions[means == 0.0]
        if zero_positions.size:
            axis.scatter(zero_positions, np.zeros_like(zero_positions), s=26, color="#D62728", zorder=4, label="R@1 = 0")
            axis.legend(loc="lower right")
        axis.set_title(title)
        axis.set_ylabel("Per-query R@1")
        axis.set_ylim(-0.05, 1.05)
        axis.set_xticks(positions)
        axis.set_xticklabels(species, rotation=90, fontsize=6)
        for tick_label, mean_value in zip(axis.get_xticklabels(), means):
            if mean_value == 0.0:
                tick_label.set_color("#D62728")
    fig.suptitle("Per-species R@1 distribution by query species", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_latex_table(path: Path, *, metrics: Mapping[str, Mapping[str, BootstrapMetric]]) -> None:
    rows: list[str] = []
    row_specs = [
        ("P1", "M2m_exact_R1", r"M$\rightarrow$m", "exact R@1"),
        ("P1", "M2m_exact_R5", r"M$\rightarrow$m", "exact R@5"),
        ("P1", "m2M_exact_R1", r"m$\rightarrow$M", "exact R@1"),
        ("P1", "m2M_exact_R5", r"m$\rightarrow$M", "exact R@5"),
        ("P3", "M2m_exact_R1", r"M$\rightarrow$m", "exact R@1"),
        ("P3", "M2m_exact_R5", r"M$\rightarrow$m", "exact R@5"),
        ("P3", "m2M_exact_R1", r"m$\rightarrow$M", "exact R@1"),
        ("P3", "m2M_exact_R5", r"m$\rightarrow$M", "exact R@5"),
        ("P3", "M2m_genus_R1", r"M$\rightarrow$m", "genus R@1"),
        ("P3", "m2M_genus_R1", r"m$\rightarrow$M", "genus R@1"),
    ]
    for dataset_name, metric_name, direction_label, metric_label in row_specs:
        metric = metrics.get(dataset_name, {}).get(metric_name)
        if metric is None:
            continue
        rows.append(
            r"\textbf{%s} & %s & %s & %.1f & (%.1f, %.1f) \\"
            % (
                dataset_name,
                direction_label,
                metric_label,
                metric.point_estimate * 100.0,
                metric.ci_lower * 100.0,
                metric.ci_upper * 100.0,
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_summary_line(dataset_name: str, direction: str, match_name: str, metric: BootstrapMetric) -> str:
    width = metric.ci_width * 100.0
    return (
        f"║  {dataset_name:<2}  {direction:<3}  {match_name:<5}  R@1: "
        f"{metric.point_estimate * 100.0:5.1f}%  "
        f"({metric.ci_lower * 100.0:5.1f} - {metric.ci_upper * 100.0:5.1f}%)  "
        f"width={width:5.1f}%  ║"
    )


def log_stdout_summary(
    metrics: Mapping[str, Mapping[str, BootstrapMetric]],
    *,
    confidence: float,
    logger: logging.Logger,
) -> None:
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║         Bootstrap Retrieval CI Summary ({confidence:.0%} CI)             ║",
        "╠══════════════════════════════════════════════════════════════╣",
    ]
    summary_specs = [
        ("P1", "M2m_exact_R1", "M→m", "exact"),
        ("P1", "m2M_exact_R1", "m→M", "exact"),
        ("P3", "M2m_exact_R1", "M→m", "exact"),
        ("P3", "m2M_exact_R1", "m→M", "exact"),
    ]
    for dataset_name, metric_name, direction_label, match_name in summary_specs:
        metric = metrics.get(dataset_name, {}).get(metric_name)
        if metric is not None:
            lines.append(_format_summary_line(dataset_name, direction_label, match_name, metric))
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("Note: Numbers above are computed from the provided embedding data.")
    logger.info("\n%s", "\n".join(lines))


def load_or_extract_bundle(
    dataset_name: str,
    *,
    embedding_path: Path | None,
    run_spec: str | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> EmbeddingBundle:
    if embedding_path is not None:
        bundle = load_embedding_bundle(embedding_path, feature_source=args.feature_source)
        logger.info(
            "Loaded %s embeddings | source=%s | macro=%s | micro=%s | dim=%d",
            dataset_name,
            bundle.source,
            bundle.macro_emb.shape[0],
            bundle.micro_emb.shape[0],
            bundle.macro_emb.shape[1],
        )
        return bundle
    if config_path is not None or checkpoint_path is not None:
        if config_path is None or checkpoint_path is None:
            raise ValueError(
                f"Provide both --{dataset_name.lower()}_config and "
                f"--{dataset_name.lower()}_checkpoint for {dataset_name} extraction."
            )
        run_spec = f"{dataset_name.lower()}::{config_path}::{checkpoint_path}"
    if run_spec is not None:
        bundle = extract_embedding_bundle_from_run(
            run_spec,
            dataset_name=dataset_name,
            output_dir=output_dir,
            args=args,
            logger=logger,
        )
        logger.info(
            "Loaded extracted %s embeddings | macro=%s | micro=%s | dim=%d",
            dataset_name,
            bundle.macro_emb.shape[0],
            bundle.micro_emb.shape[0],
            bundle.macro_emb.shape[1],
        )
        return bundle
    raise ValueError(
        f"Provide either --{dataset_name.lower()}_embeddings or --{dataset_name.lower()}_run "
        f"for {dataset_name}."
    )


def has_dataset_input(
    *,
    embedding_path: Path | None,
    run_spec: str | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
) -> bool:
    return any(value is not None for value in (embedding_path, run_spec, config_path, checkpoint_path))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_bootstrap <= 0:
        raise ValueError("--n_bootstrap must be positive.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must be between 0 and 1.")

    output_dir = resolve_path(args.output_dir) or args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(
        "Bootstrap retrieval CI | n_bootstrap=%d | confidence=%.3f | seed=%d | feature_source=%s",
        args.n_bootstrap,
        args.confidence,
        args.seed,
        args.feature_source,
    )

    bundles: dict[str, EmbeddingBundle] = {}
    if has_dataset_input(
        embedding_path=args.p1_embeddings,
        run_spec=args.p1_run,
        config_path=args.p1_config,
        checkpoint_path=args.p1_checkpoint,
    ):
        bundles["P1"] = load_or_extract_bundle(
            "P1",
            embedding_path=args.p1_embeddings,
            run_spec=args.p1_run,
            config_path=args.p1_config,
            checkpoint_path=args.p1_checkpoint,
            args=args,
            output_dir=output_dir,
            logger=logger,
        )
    else:
        logger.warning("Skipping P1: no --p1_embeddings or --p1_config/--p1_checkpoint were provided.")

    if has_dataset_input(
        embedding_path=args.p3_embeddings,
        run_spec=args.p3_run,
        config_path=args.p3_config,
        checkpoint_path=args.p3_checkpoint,
    ):
        bundles["P3"] = load_or_extract_bundle(
            "P3",
            embedding_path=args.p3_embeddings,
            run_spec=args.p3_run,
            config_path=args.p3_config,
            checkpoint_path=args.p3_checkpoint,
            args=args,
            output_dir=output_dir,
            logger=logger,
        )
    else:
        logger.warning("Skipping P3: no --p3_embeddings or --p3_config/--p3_checkpoint were provided.")

    if not bundles:
        raise ValueError("No datasets to evaluate. Provide P3 and/or P1 embeddings, or config+checkpoint.")

    metrics, retrieval_results = run_all_bootstraps(
        bundles,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        logger=logger,
    )

    summary_path = output_dir / "bootstrap_ci_summary.json"
    write_summary_json(summary_path, metrics=metrics, args=args, bundles=bundles)
    plot_bootstrap_distributions(
        output_dir / "bootstrap_distributions.png",
        metrics=metrics,
        confidence=args.confidence,
    )
    plot_per_species_r1(output_dir / "per_species_r1.png", retrieval_results=retrieval_results)
    write_latex_table(output_dir / "latex_table.txt", metrics=metrics)
    logger.info("Wrote bootstrap summary to %s", summary_path)
    logger.info("Wrote plots and LaTeX table to %s", output_dir)
    log_stdout_summary(metrics, confidence=args.confidence, logger=logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
