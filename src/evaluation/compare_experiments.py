"""Combine fresh baseline and transfer-learning metrics from one run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = {
    "Baseline_ClassWeighted",
    "Baseline_Oversampling",
    "Baseline_Augmentation",
    "DenseNet121_Frozen",
    "DenseNet121_FineTuned",
    "ResNet50_Frozen",
    "ResNet50_FineTuned",
}
FIELDS = [
    "model",
    "experiment",
    "run_id",
    "threshold",
    "target_sensitivity",
    "validation_sensitivity",
    "test_loss",
    "accuracy",
    "precision",
    "sensitivity_recall",
    "specificity",
    "f1_score",
    "roc_auc",
    "average_precision",
    "manifest_sha256",
]


def collect_metrics(run_id: str) -> list[dict[str, Any]]:
    baseline_root = PROJECT_ROOT / "results" / "baseline_cnn" / run_id
    transfer_root = PROJECT_ROOT / "results" / "transfer_learning" / run_id
    files = sorted(baseline_root.rglob("metrics.json")) + sorted(
        (transfer_root / "per_model").rglob("metrics.json")
    )
    if not files:
        raise FileNotFoundError(f"No fresh metrics were found for run {run_id!r}.")

    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    names = [record["model"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model metrics found for run {run_id!r}.")
    missing = EXPECTED_MODELS - set(names)
    unexpected = set(names) - EXPECTED_MODELS
    if missing or unexpected:
        raise ValueError(
            f"Expected exactly seven experiment records; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError("A metrics record has a different run ID.")
    manifest_hashes = {record.get("manifest_sha256") for record in records}
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise ValueError("All seven experiments must use the same verified manifest.")
    return sorted(records, key=lambda record: record["model"])


def write_comparison(run_id: str) -> tuple[Path, Path, Path]:
    records = collect_metrics(run_id)
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"model_comparison_{run_id}.csv"
    markdown_path = output_dir / f"model_comparison_{run_id}.md"
    plot_path = PROJECT_ROOT / "results" / "plots" / f"model_comparison_{run_id}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (csv_path, markdown_path, plot_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite comparison outputs: "
            + ", ".join(str(path) for path in existing)
        )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: record.get(field, "") for field in FIELDS} for record in records)

    table_fields = [
        "model",
        "threshold",
        "test_loss",
        "accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]
    lines = [
        f"# Fresh experiment comparison: {run_id}",
        "",
        "Thresholds were selected on validation data to meet the 90% PNEUMONIA sensitivity target,",
        "then held fixed for the test metrics below. All rows use the same manifest SHA-256.",
        "",
        "| " + " | ".join(table_fields) + " |",
        "| " + " | ".join("---" for _ in table_fields) + " |",
    ]
    for record in records:
        cells = []
        for field in table_fields:
            value = record[field]
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics = ["accuracy", "precision", "sensitivity_recall", "specificity", "f1_score", "roc_auc", "average_precision"]
    labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC", "Avg. precision"]
    positions = np.arange(len(metrics))
    width = 0.11
    figure, axis = plt.subplots(figsize=(15, 6))
    for index, record in enumerate(records):
        axis.bar(
            positions + (index - (len(records) - 1) / 2) * width,
            [record[field] for field in metrics],
            width,
            label=record["model"],
        )
    axis.set(
        title=f"Seven-model comparison ({run_id})",
        ylabel="Score",
        ylim=(0, 1),
        xticks=positions,
        xticklabels=labels,
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return csv_path, markdown_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Common run identifier for all seven models.")
    arguments = parser.parse_args()
    paths = write_comparison(arguments.run_id)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
