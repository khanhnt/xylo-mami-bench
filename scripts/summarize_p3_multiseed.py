#!/usr/bin/env python3
"""Summarize P3 multi-seed robustness metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("results/p3_multiseed/per_seed_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("results/p3_multiseed")
EXPECTED_SEEDS = 3


@dataclass(frozen=True)
class MetricSpec:
    output_name: str
    source_name: str


MODEL_DISPLAY_NAMES = {
    "p3_rgb_align": "P3 RGB align",
    "p3_gray_align": "P3 Gray align",
    "p3_rgb_gray_gated_v3": "P3 RGB-gray gated v3",
}

METRICS = (
    MetricSpec("macro_bal_acc", "macro_balanced_accuracy"),
    MetricSpec("micro_bal_acc", "micro_balanced_accuracy"),
    MetricSpec("tradeoff", "tradeoff_score"),
    MetricSpec("emb_M_to_m_exact_R1", "macro_to_micro_exact_r1"),
    MetricSpec("emb_M_to_m_exact_R5", "macro_to_micro_exact_r5"),
    MetricSpec("emb_m_to_M_exact_R1", "micro_to_macro_exact_r1"),
    MetricSpec("emb_m_to_M_exact_R5", "micro_to_macro_exact_r5"),
    MetricSpec("emb_M_to_m_genus_R1", "macro_to_micro_genus_r1"),
    MetricSpec("emb_m_to_M_genus_R1", "micro_to_macro_genus_r1"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input per-seed CSV. Default: results/p3_multiseed/per_seed_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary_mean_std.* files will be written.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def format_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "NA"
    scaled = [value * 100.0 for value in values]
    mean = statistics.mean(scaled)
    std = statistics.stdev(scaled) if len(scaled) > 1 else 0.0
    return f"{mean:.2f} \u00b1 {std:.2f}"


def group_rows(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {model: [] for model in MODEL_DISPLAY_NAMES}
    for row in rows:
        model = (row.get("model") or "").strip()
        if model in grouped:
            grouped[model].append(row)
    return grouped


def completed_seed_count(rows: Sequence[dict[str, str]]) -> int:
    seeds = {(row.get("seed") or "").strip() for row in rows if (row.get("seed") or "").strip()}
    return len(seeds)


def build_summary_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped = group_rows(rows)
    summary_rows: list[dict[str, str]] = []
    for model, display_name in MODEL_DISPLAY_NAMES.items():
        model_rows = grouped[model]
        seed_count = completed_seed_count(model_rows)
        warning = (
            ""
            if seed_count >= EXPECTED_SEEDS
            else f"WARNING: {display_name} has only {seed_count}/{EXPECTED_SEEDS} seeds"
        )
        summary = {
            "model": model,
            "display_name": display_name,
            "n_seeds_completed": str(seed_count),
            "warning": warning,
        }
        for metric in METRICS:
            values = [
                parsed
                for row in model_rows
                if (parsed := parse_float(row.get(metric.source_name))) is not None
            ]
            summary[metric.output_name] = format_mean_std(values)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "display_name", "n_seeds_completed", *[m.output_name for m in METRICS], "warning"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "n", *[metric.output_name for metric in METRICS]]
    lines = [
        "# P3 Multi-seed Summary",
        "",
        "Values are mean \u00b1 standard deviation across completed seeds, reported as percentages.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", *[":---:" for _ in METRICS]]) + " |",
    ]
    for row in rows:
        values = [row["display_name"], row["n_seeds_completed"], *[row[metric.output_name] for metric in METRICS]]
        lines.append("| " + " | ".join(values) + " |")

    warnings = [row["warning"] for row in rows if row.get("warning")]
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in warnings]])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\u00b1": r"$\pm$",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_latex(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "n", *[metric.output_name for metric in METRICS]]
    column_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{tabular}{" + column_spec + r"}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["display_name"], row["n_seeds_completed"], *[row[metric.output_name] for metric in METRICS]]
        lines.append(" & ".join(latex_escape(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"% Values are mean $\pm$ standard deviation across completed seeds, reported as percentages.",
        ]
    )

    warnings = [row["warning"] for row in rows if row.get("warning")]
    if warnings:
        lines.extend([r"% Warnings:", *[f"% {warning}" for warning in warnings]])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    rows = read_rows(input_path)
    summary_rows = build_summary_rows(rows)

    write_csv(output_dir / "summary_mean_std.csv", summary_rows)
    write_markdown(output_dir / "summary_mean_std.md", summary_rows)
    write_latex(output_dir / "summary_latex_table.tex", summary_rows)

    for row in summary_rows:
        if row.get("warning"):
            print(row["warning"], file=sys.stderr)
    print(f"Wrote {output_dir / 'summary_mean_std.csv'}")
    print(f"Wrote {output_dir / 'summary_mean_std.md'}")
    print(f"Wrote {output_dir / 'summary_latex_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
