#!/usr/bin/env python3
"""Evaluate cross-modal retrieval for trained XyloMaMi-Bench alignment checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RetrievalDirectionMetrics:
    query_modality: str
    gallery_modality: str
    query_count: int
    gallery_count: int
    eligible_exact_queries: int
    eligible_genus_queries: int
    exact_recall_at_1: float | None
    exact_recall_at_5: float | None
    genus_recall_at_1: float | None
    genus_recall_at_5: float | None
    exact_top1_purity: float | None
    exact_top5_purity: float | None
    genus_top1_purity: float | None
    genus_top5_purity: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalRunSummary:
    run_name: str
    seed: int
    config_path: str
    checkpoint_path: str
    split_name: str
    feature_source: str
    stage_name: str
    stage_relation_policy: str
    macro_to_micro: RetrievalDirectionMetrics
    micro_to_macro: RetrievalDirectionMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "split_name": self.split_name,
            "feature_source": self.feature_source,
            "stage_name": self.stage_name,
            "stage_relation_policy": self.stage_relation_policy,
            "macro_to_micro": self.macro_to_micro.to_dict(),
            "micro_to_macro": self.micro_to_macro.to_dict(),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-modal retrieval for one or more alignment checkpoints."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run spec formatted as name::config_path::checkpoint_path. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/retrieval_analysis"),
        help="Directory where retrieval reports will be written.",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="embedding",
        choices=("embedding", "projected", "pooled"),
        help="Which feature source to use for retrieval similarity.",
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
        help="Optional num_workers override.",
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
            "Optional seed override applied to every run. When --output_dir is left "
            "at its default, the output directory and run names are suffixed with the seed."
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
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional cap for evaluation batches when doing a short validation pass.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional config overrides in the form key=value applied to every run.",
    )
    return parser.parse_args(argv)


def _parse_run_spec(spec: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in spec.split("::")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid --run spec '{spec}'. Expected format name::config_path::checkpoint_path."
        )
    name, config_path_raw, checkpoint_path_raw = parts
    config_path = Path(config_path_raw)
    checkpoint_path = Path(checkpoint_path_raw)
    return name, config_path, checkpoint_path


def _append_seed_suffix(value: str | Path, seed: int) -> str:
    path = Path(value)
    suffix = f"_seed{int(seed)}"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(f"{path.name}{suffix}"))


def _append_seed_to_name(name: str, seed: int) -> str:
    suffix = f"_seed{int(seed)}"
    return name if name.endswith(suffix) else f"{name}{suffix}"


def _encode_labels(values: Sequence[str]) -> torch.Tensor:
    import torch

    mapping: dict[str, int] = {}
    encoded: list[int] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return torch.tensor(encoded, dtype=torch.long)


def _encode_label_pairs(
    query_values: Sequence[str],
    gallery_values: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = _encode_labels([*query_values, *gallery_values])
    query_count = len(query_values)
    return combined[:query_count], combined[query_count:]


def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    valid = values[mask]
    if valid.numel() == 0:
        return None
    return float(valid.float().mean().item())


def _compute_direction_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    query_modality: str,
    gallery_modality: str,
    feature_source: str,
) -> RetrievalDirectionMetrics:
    import torch
    import torch.nn.functional as F

    query_records = [record for record in records if str(record["modality"]) == query_modality]
    gallery_records = [record for record in records if str(record["modality"]) == gallery_modality]
    if not query_records:
        raise ValueError(f"No {query_modality} embeddings found in the retrieval dump.")
    if not gallery_records:
        raise ValueError(f"No {gallery_modality} embeddings found in the retrieval dump.")

    feature_key = {
        "embedding": "embedding",
        "projected": "projected_feature",
        "pooled": "pooled_feature",
    }[feature_source]
    missing_query = sum(1 for record in query_records if record.get(feature_key) is None)
    missing_gallery = sum(1 for record in gallery_records if record.get(feature_key) is None)
    if missing_query or missing_gallery:
        raise ValueError(
            f"Feature source '{feature_source}' requires '{feature_key}' to be present in every record, "
            f"but found missing values for query={missing_query}, gallery={missing_gallery}."
        )

    query_embeddings = F.normalize(
        torch.stack([torch.as_tensor(record[feature_key]).float() for record in query_records], dim=0),
        dim=1,
    )
    gallery_embeddings = F.normalize(
        torch.stack([torch.as_tensor(record[feature_key]).float() for record in gallery_records], dim=0),
        dim=1,
    )
    similarities = query_embeddings @ gallery_embeddings.T

    query_species = [str(record["species"]) for record in query_records]
    query_genera = [str(record["genus"]) for record in query_records]
    gallery_species = [str(record["species"]) for record in gallery_records]
    gallery_genera = [str(record["genus"]) for record in gallery_records]

    query_species_ids, gallery_species_ids = _encode_label_pairs(query_species, gallery_species)
    query_genus_ids, gallery_genus_ids = _encode_label_pairs(query_genera, gallery_genera)

    exact_match_matrix = query_species_ids.unsqueeze(1) == gallery_species_ids.unsqueeze(0)
    genus_match_matrix = query_genus_ids.unsqueeze(1) == gallery_genus_ids.unsqueeze(0)
    eligible_exact = exact_match_matrix.any(dim=1)
    eligible_genus = genus_match_matrix.any(dim=1)

    top1_indices = similarities.topk(k=1, dim=1).indices
    top5_k = min(5, gallery_embeddings.shape[0])
    top5_indices = similarities.topk(k=top5_k, dim=1).indices

    top1_exact = exact_match_matrix.gather(dim=1, index=top1_indices)
    top5_exact = exact_match_matrix.gather(dim=1, index=top5_indices)
    top1_genus = genus_match_matrix.gather(dim=1, index=top1_indices)
    top5_genus = genus_match_matrix.gather(dim=1, index=top5_indices)

    exact_recall_at_1 = _safe_mean(top1_exact.any(dim=1).float(), eligible_exact)
    exact_recall_at_5 = _safe_mean(top5_exact.any(dim=1).float(), eligible_exact)
    genus_recall_at_1 = _safe_mean(top1_genus.any(dim=1).float(), eligible_genus)
    genus_recall_at_5 = _safe_mean(top5_genus.any(dim=1).float(), eligible_genus)

    exact_top1_purity = _safe_mean(top1_exact.float().mean(dim=1), eligible_exact)
    exact_top5_purity = _safe_mean(top5_exact.float().mean(dim=1), eligible_exact)
    genus_top1_purity = _safe_mean(top1_genus.float().mean(dim=1), eligible_genus)
    genus_top5_purity = _safe_mean(top5_genus.float().mean(dim=1), eligible_genus)

    return RetrievalDirectionMetrics(
        query_modality=query_modality,
        gallery_modality=gallery_modality,
        query_count=len(query_records),
        gallery_count=len(gallery_records),
        eligible_exact_queries=int(eligible_exact.sum().item()),
        eligible_genus_queries=int(eligible_genus.sum().item()),
        exact_recall_at_1=exact_recall_at_1,
        exact_recall_at_5=exact_recall_at_5,
        genus_recall_at_1=genus_recall_at_1,
        genus_recall_at_5=genus_recall_at_5,
        exact_top1_purity=exact_top1_purity,
        exact_top5_purity=exact_top5_purity,
        genus_top1_purity=genus_top1_purity,
        genus_top5_purity=genus_top5_purity,
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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


def _write_markdown_summary(path: Path, summaries: Sequence[RetrievalRunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-Modal Retrieval Summary",
        "",
        f"Feature source: `{summaries[0].feature_source}`" if summaries else "",
        "",
        "| Run | Stage | Macro->Micro R@1 | Macro->Micro R@5 | Micro->Macro R@1 | Micro->Macro R@5 | M->m Exact Top5 Purity | m->M Exact Top5 Purity | M->m Genus Top5 Purity | m->M Genus Top5 Purity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        m2m = summary.macro_to_micro
        micro2macro = summary.micro_to_macro
        lines.append(
            "| {run} | {stage} | {m2m_r1} | {m2m_r5} | {m2ma_r1} | {m2ma_r5} | {m2m_exact_p5} | {m2ma_exact_p5} | {m2m_genus_p5} | {m2ma_genus_p5} |".format(
                run=summary.run_name,
                stage=summary.stage_name,
                m2m_r1=_format_metric(m2m.exact_recall_at_1),
                m2m_r5=_format_metric(m2m.exact_recall_at_5),
                m2ma_r1=_format_metric(micro2macro.exact_recall_at_1),
                m2ma_r5=_format_metric(micro2macro.exact_recall_at_5),
                m2m_exact_p5=_format_metric(m2m.exact_top5_purity),
                m2ma_exact_p5=_format_metric(micro2macro.exact_top5_purity),
                m2m_genus_p5=_format_metric(m2m.genus_top5_purity),
                m2ma_genus_p5=_format_metric(micro2macro.genus_top5_purity),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(summary: RetrievalRunSummary) -> dict[str, Any]:
    m2m = summary.macro_to_micro
    micro2macro = summary.micro_to_macro
    return {
        "run_name": summary.run_name,
        "seed": summary.seed,
        "feature_source": summary.feature_source,
        "stage_name": summary.stage_name,
        "stage_relation_policy": summary.stage_relation_policy,
        "macro_to_micro_exact_r1": m2m.exact_recall_at_1,
        "macro_to_micro_exact_r5": m2m.exact_recall_at_5,
        "macro_to_micro_genus_r1": m2m.genus_recall_at_1,
        "macro_to_micro_genus_r5": m2m.genus_recall_at_5,
        "macro_to_micro_exact_top5_purity": m2m.exact_top5_purity,
        "macro_to_micro_genus_top5_purity": m2m.genus_top5_purity,
        "micro_to_macro_exact_r1": micro2macro.exact_recall_at_1,
        "micro_to_macro_exact_r5": micro2macro.exact_recall_at_5,
        "micro_to_macro_genus_r1": micro2macro.genus_recall_at_1,
        "micro_to_macro_genus_r5": micro2macro.genus_recall_at_5,
        "micro_to_macro_exact_top5_purity": micro2macro.exact_top5_purity,
        "micro_to_macro_genus_top5_purity": micro2macro.genus_top5_purity,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from src.engine.trainer_align import (
        DualEncoderTrainer,
        load_checkpoint_file,
        parse_experiment_config,
        prepare_experiment_paths,
        serialize_experiment_config,
        setup_logger,
    )
    from src.utils.config import apply_overrides, dump_config, load_config, set_nested_value

    output_dir = args.output_dir
    if args.seed is not None and output_dir == Path("outputs/retrieval_analysis"):
        output_dir = Path(_append_seed_suffix(output_dir, args.seed))
    output_root = output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[RetrievalRunSummary] = []
    for run_spec in args.runs:
        run_name, config_path, checkpoint_path = _parse_run_spec(run_spec)
        raw_config = load_config(config_path)
        raw_config = apply_overrides(raw_config, args.overrides)
        if args.seed is not None:
            run_name = _append_seed_to_name(run_name, args.seed)
            set_nested_value(raw_config, "seed", int(args.seed))
        set_nested_value(raw_config, "warmstart.macro_checkpoint", None)
        set_nested_value(raw_config, "warmstart.micro_checkpoint", None)
        set_nested_value(raw_config, "warmstart.macro_rgb_checkpoint", None)
        set_nested_value(raw_config, "warmstart.macro_gray_checkpoint", None)
        set_nested_value(raw_config, "warmstart.micro_rgb_checkpoint", None)
        set_nested_value(raw_config, "warmstart.micro_gray_checkpoint", None)
        set_nested_value(raw_config, "evaluation.split_name", args.split)
        set_nested_value(raw_config, "evaluation.dump_embeddings", True)
        if args.device is not None:
            set_nested_value(raw_config, "device", args.device)
        if args.num_workers is not None:
            set_nested_value(raw_config, "dataset.num_workers", args.num_workers)
        if args.batch_size is not None:
            set_nested_value(raw_config, "dataset.eval_batch_size", args.batch_size)
        if args.max_eval_batches is not None:
            set_nested_value(raw_config, "training.max_eval_batches", args.max_eval_batches)
        if args.macro_image_root is not None:
            set_nested_value(raw_config, "dataset.image_root_override.macro", str(args.macro_image_root))
        if args.micro_image_root is not None:
            set_nested_value(raw_config, "dataset.image_root_override.micro", str(args.micro_image_root))

        run_output_dir = output_root / run_name
        set_nested_value(raw_config, "output_dir", str(run_output_dir))
        config = parse_experiment_config(raw_config)
        paths = prepare_experiment_paths(config.output_dir)
        dump_config(serialize_experiment_config(config), paths.resolved_config_path)
        logger = setup_logger(paths.log_path, name=f"retrieval_{config.experiment_name}_{run_name}")
        from src.utils.seeding import set_global_seed

        seed_info = set_global_seed(config.seed)
        logger.info(
            "Seed control | seed=%d | cudnn_deterministic=%s | cudnn_benchmark=%s",
            seed_info["seed"],
            seed_info["cudnn_deterministic"],
            seed_info["cudnn_benchmark"],
        )

        resolved_checkpoint = checkpoint_path if checkpoint_path.is_absolute() else (REPO_ROOT / checkpoint_path).resolve()
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        logger.info("Evaluating retrieval for run=%s", run_name)
        logger.info("Using config=%s", config_path)
        logger.info("Using checkpoint=%s", resolved_checkpoint)
        logger.info("Using feature_source=%s", args.feature_source)

        trainer = DualEncoderTrainer(config, logger=logger, paths=paths)
        checkpoint = load_checkpoint_file(resolved_checkpoint)
        trainer.load_checkpoint(resolved_checkpoint, restore_training_state=False)

        if args.split == "train":
            dataloader = trainer.train_loader
        elif args.split == "val":
            dataloader = trainer.val_loader
        else:
            if trainer.test_loader is None:
                raise ValueError(f"No dataset.test_split_csv configured for run '{run_name}'.")
            dataloader = trainer.test_loader

        stage_index = int(checkpoint.get("stage_index", max(0, len(config.stages) - 1)))
        stage_index = max(0, min(stage_index, len(config.stages) - 1))
        stage = config.stages[stage_index]
        logger.info(
            "Checkpoint stage | stage_index=%d | stage_name=%s | relation_policy=%s",
            stage_index,
            stage.name,
            stage.relation_policy,
        )
        artifacts = trainer.evaluate_and_export(
            split_name=args.split,
            dataloader=dataloader,
            output_prefix=args.split,
            dump_embeddings=True,
            stage=stage,
        )
        if artifacts.embeddings_dump_path is None:
            raise RuntimeError(f"Failed to dump embeddings for run '{run_name}'.")
        records = torch.load(artifacts.embeddings_dump_path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise TypeError(
                f"Expected a list of embedding records in {artifacts.embeddings_dump_path}, got {type(records)!r}."
            )

        macro_to_micro = _compute_direction_metrics(
            records,
            query_modality="macro",
            gallery_modality="micro",
            feature_source=args.feature_source,
        )
        micro_to_macro = _compute_direction_metrics(
            records,
            query_modality="micro",
            gallery_modality="macro",
            feature_source=args.feature_source,
        )
        summary = RetrievalRunSummary(
            run_name=run_name,
            seed=config.seed,
            config_path=str(Path(config_path).resolve()),
            checkpoint_path=str(resolved_checkpoint),
            split_name=args.split,
            feature_source=args.feature_source,
            stage_name=stage.name,
            stage_relation_policy=stage.relation_policy,
            macro_to_micro=macro_to_micro,
            micro_to_macro=micro_to_macro,
        )
        _write_json(
            paths.report_dir / f"{args.split}_retrieval_metrics_{args.feature_source}.json",
            summary.to_dict(),
        )
        logger.info(
            "Retrieval summary | run=%s | feature_source=%s | macro_to_micro exact R@1=%.4f | exact R@5=%.4f | "
            "micro_to_macro exact R@1=%.4f | exact R@5=%.4f",
            run_name,
            args.feature_source,
            summary.macro_to_micro.exact_recall_at_1 or 0.0,
            summary.macro_to_micro.exact_recall_at_5 or 0.0,
            summary.micro_to_macro.exact_recall_at_1 or 0.0,
            summary.micro_to_macro.exact_recall_at_5 or 0.0,
        )
        summaries.append(summary)

    summary_json_path = output_root / f"{args.split}_retrieval_summary_{args.feature_source}.json"
    _write_json(
        summary_json_path,
        {
            "split_name": args.split,
            "seed": int(args.seed) if args.seed is not None else None,
            "feature_source": args.feature_source,
            "runs": [summary.to_dict() for summary in summaries],
        },
    )
    summary_rows = [_summary_row(summary) for summary in summaries]
    _write_csv(output_root / f"{args.split}_retrieval_summary_{args.feature_source}.csv", summary_rows)
    _write_markdown_summary(
        output_root / f"{args.split}_retrieval_summary_{args.feature_source}.md",
        summaries,
    )
    print(
        json.dumps(
            {
                "split_name": args.split,
                "seed": int(args.seed) if args.seed is not None else None,
                "feature_source": args.feature_source,
                "runs": [summary.to_dict() for summary in summaries],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
