"""Classification metrics for XyloMaMi-Bench baseline experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerClassMetrics:
    """Per-class precision/recall/F1 summary."""

    class_index: int
    class_name: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ClassificationMetricsSummary:
    """Aggregate evaluation metrics for one split."""

    split_name: str
    sample_count: int
    loss: float | None
    top1_accuracy: float
    macro_f1: float
    balanced_accuracy: float
    correct_count: int
    num_classes: int
    per_class: tuple[PerClassMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the metrics summary into a JSON-friendly mapping."""

        payload = asdict(self)
        payload["per_class"] = [asdict(item) for item in self.per_class]
        return payload


class ClassificationMetricAccumulator:
    """Accumulate confusion-matrix-based metrics across batches."""

    def __init__(self, class_names: Sequence[str], *, split_name: str = "") -> None:
        if not class_names:
            raise ValueError("class_names must not be empty.")
        self.class_names = tuple(str(name) for name in class_names)
        self.num_classes = len(self.class_names)
        self.split_name = split_name
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )
        self.total_samples = 0
        self.total_loss = 0.0
        self.loss_weight = 0

    def update(self, logits: Tensor, targets: Tensor, *, loss: Tensor | float | None = None) -> None:
        """Accumulate a batch of logits, targets, and optional batch loss."""

        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D, but received shape {tuple(logits.shape)}.")
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1D, but received shape {tuple(targets.shape)}.")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets batch sizes do not match.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits have {logits.shape[1]} classes, expected {self.num_classes}."
            )
        if targets.numel() > 0:
            target_min = int(targets.min().item())
            target_max = int(targets.max().item())
            if target_min < 0 or target_max >= self.num_classes:
                raise ValueError(
                    "targets contain out-of-range class indices: "
                    f"min={target_min}, max={target_max}, expected range "
                    f"[0, {self.num_classes - 1}]."
                )

        predictions = logits.argmax(dim=1)
        encoded = targets.to(torch.int64) * self.num_classes + predictions.to(torch.int64)
        batch_confusion = torch.bincount(encoded.cpu(), minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )
        self.confusion_matrix += batch_confusion
        batch_size = int(targets.shape[0])
        self.total_samples += batch_size

        if loss is not None:
            loss_value = float(loss.item() if isinstance(loss, Tensor) else loss)
            self.total_loss += loss_value * batch_size
            self.loss_weight += batch_size

    def _safe_divide(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        result = torch.zeros_like(numerator, dtype=torch.float64)
        valid = denominator > 0
        result[valid] = numerator[valid].to(torch.float64) / denominator[valid].to(torch.float64)
        return result

    def compute(self) -> ClassificationMetricsSummary:
        """Compute aggregate metrics from the accumulated confusion matrix."""

        support = self.confusion_matrix.sum(dim=1)
        predicted = self.confusion_matrix.sum(dim=0)
        true_positives = self.confusion_matrix.diag()
        false_positives = predicted - true_positives
        false_negatives = support - true_positives

        precision = self._safe_divide(true_positives, predicted)
        recall = self._safe_divide(true_positives, support)
        f1 = self._safe_divide(2 * precision * recall, precision + recall)

        valid_classes = support > 0
        macro_f1 = float(f1[valid_classes].mean().item()) if valid_classes.any() else 0.0
        balanced_accuracy = (
            float(recall[valid_classes].mean().item()) if valid_classes.any() else 0.0
        )
        correct_count = int(true_positives.sum().item())
        top1_accuracy = correct_count / max(1, self.total_samples)
        mean_loss = self.total_loss / self.loss_weight if self.loss_weight > 0 else None

        per_class = tuple(
            PerClassMetrics(
                class_index=index,
                class_name=self.class_names[index],
                support=int(support[index].item()),
                true_positives=int(true_positives[index].item()),
                false_positives=int(false_positives[index].item()),
                false_negatives=int(false_negatives[index].item()),
                precision=float(precision[index].item()),
                recall=float(recall[index].item()),
                f1=float(f1[index].item()),
            )
            for index in range(self.num_classes)
        )

        return ClassificationMetricsSummary(
            split_name=self.split_name,
            sample_count=self.total_samples,
            loss=mean_loss,
            top1_accuracy=float(top1_accuracy),
            macro_f1=float(macro_f1),
            balanced_accuracy=float(balanced_accuracy),
            correct_count=correct_count,
            num_classes=self.num_classes,
            per_class=per_class,
        )

    def export_confusion_matrix_csv(self, path: str | Path) -> None:
        """Export the confusion matrix as a CSV table."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_class", *self.class_names])
            for class_name, row in zip(self.class_names, self.confusion_matrix.tolist()):
                writer.writerow([class_name, *row])

    def export_per_class_report_csv(
        self,
        path: str | Path,
        *,
        summary: ClassificationMetricsSummary | None = None,
    ) -> ClassificationMetricsSummary:
        """Export per-class metrics as CSV."""

        resolved_summary = summary or self.compute()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "class_index",
                    "class_name",
                    "support",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
            for item in resolved_summary.per_class:
                writer.writerow(
                    [
                        item.class_index,
                        item.class_name,
                        item.support,
                        item.true_positives,
                        item.false_positives,
                        item.false_negatives,
                        f"{item.precision:.8f}",
                        f"{item.recall:.8f}",
                        f"{item.f1:.8f}",
                    ]
                )
        return resolved_summary

    def export_summary_json(
        self,
        path: str | Path,
        *,
        summary: ClassificationMetricsSummary | None = None,
    ) -> ClassificationMetricsSummary:
        """Export the aggregate summary as JSON."""

        resolved_summary = summary or self.compute()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                resolved_summary.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
            )
        return resolved_summary
