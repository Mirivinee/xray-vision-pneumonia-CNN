"""Shared validation-threshold selection and held-out binary evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _binary_arrays(
    labels: np.ndarray, pneumonia_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(pneumonia_scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be non-empty arrays of the same length.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must use the binary mapping NORMAL=0, PNEUMONIA=1.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("PNEUMONIA scores must be finite probabilities between 0 and 1.")
    if np.unique(labels).size != 2:
        raise ValueError("Both classes must be present to compute binary evaluation metrics.")
    return labels, scores


def select_threshold_for_sensitivity(
    labels: np.ndarray,
    pneumonia_scores: np.ndarray,
    target_sensitivity: float = 0.90,
) -> dict[str, float]:
    """Choose the most precise validation threshold meeting the recall target."""
    labels, scores = _binary_arrays(labels, pneumonia_scores)
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be greater than 0 and at most 1.")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(recall[:-1] >= target_sensitivity)
    if eligible.size == 0:
        raise RuntimeError(
            f"No validation threshold meets sensitivity target {target_sensitivity:.3f}."
        )

    best_precision = float(np.max(precision[eligible]))
    best = eligible[np.isclose(precision[eligible], best_precision)]
    index = int(best[np.argmax(thresholds[best])])
    return {
        "threshold": float(thresholds[index]),
        "target_sensitivity": float(target_sensitivity),
        "validation_sensitivity": float(recall[index]),
        "validation_precision": float(precision[index]),
    }


def calculate_binary_metrics(
    labels: np.ndarray, pneumonia_scores: np.ndarray, threshold: float
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate metrics at a fixed threshold and return the confusion matrix."""
    labels, scores = _binary_arrays(labels, pneumonia_scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability between 0 and 1.")

    predictions = (scores >= threshold).astype(np.int32)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        ),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    return metrics, matrix


def save_training_history(history: Any, output_path: Path) -> None:
    """Write Keras history values to a CSV file."""
    keys = list(history.history)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *keys])
        writer.writeheader()
        for epoch in range(len(history.epoch)):
            writer.writerow(
                {"epoch": epoch + 1, **{key: history.history[key][epoch] for key in keys}}
            )


def load_training_history(input_path: Path) -> Any:
    """Restore a Keras-like history object from a previously saved CSV."""
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "epoch" not in rows[0]:
        raise ValueError(f"Training history is missing or malformed: {input_path}")
    keys = [key for key in rows[0] if key != "epoch"]
    if not keys:
        raise ValueError(f"Training history has no metrics: {input_path}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"Training history epochs are not contiguous: {input_path}")
    history = {key: [float(row[key]) for row in rows] for key in keys}
    return SimpleNamespace(history=history, epoch=[epoch - 1 for epoch in epochs])


def save_evaluation_artifacts(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    *,
    model_name: str,
    run_id: str,
    experiment: str,
    output_dir: Path,
    manifest_sha256: str,
    test_loss: float,
    operating_point: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Save metrics after validation threshold/model selection is complete."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "predictions.npz",
    )
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    threshold = operating_point["threshold"]
    if not 0 <= threshold <= 1:
        raise ValueError("The validation-selected threshold must be between 0 and 1.")
    validation_metrics, _ = calculate_binary_metrics(
        validation_labels, validation_scores, threshold
    )
    test_metrics, matrix = calculate_binary_metrics(
        test_labels, test_scores, threshold
    )
    test_metrics["test_loss"] = float(test_loss)

    predictions = (test_scores >= threshold).astype(np.int32)
    report = classification_report(
        test_labels,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4,
        zero_division=0,
    )
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    record: dict[str, Any] = {
        "model": model_name,
        "experiment": experiment,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        **operating_point,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **test_metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label", "NORMAL", "PNEUMONIA"])
        writer.writerow(["NORMAL", *matrix[0].tolist()])
        writer.writerow(["PNEUMONIA", *matrix[1].tolist()])
    np.savez_compressed(
        output_dir / "predictions.npz",
        validation_labels=validation_labels,
        validation_scores=validation_scores,
        test_labels=test_labels,
        test_scores=test_scores,
        threshold=np.asarray(threshold),
    )

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=f"{model_name} confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC={test_metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "k--", label="Chance")
    axis.set(title=f"{model_name} ROC curve", xlabel="False positive rate", ylabel="Sensitivity")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"AP={test_metrics['average_precision']:.3f}")
    axis.set(
        title=f"{model_name} precision-recall curve",
        xlabel="PNEUMONIA recall",
        ylabel="Precision",
    )
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=140)
    plt.close(figure)

    details = {
        "matrix": matrix,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }
    return record, details
