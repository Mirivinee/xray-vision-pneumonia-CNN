"""Select validation operating points, then evaluate all seven models on test data."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.evaluation.binary_metrics import (
    calculate_binary_metrics,
    save_evaluation_artifacts,
    select_threshold_for_sensitivity,
)
from src.evaluation.compare_experiments import write_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "seed42_run01"
TARGET_SENSITIVITY = 0.90
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_SPECS = [
    ("Baseline_ClassWeighted", "baseline_class_weighted", "grayscale", "baseline_cnn_class_weighted"),
    ("Baseline_Oversampling", "oversampling", "grayscale", "baseline_cnn_oversampling"),
    ("Baseline_Augmentation", "augmentation", "grayscale", "baseline_cnn_augmentation"),
    ("DenseNet121_Frozen", "transfer_learning", "rgb", "DenseNet121_frozen"),
    ("DenseNet121_FineTuned", "transfer_learning", "rgb", "DenseNet121_fine_tuned"),
    ("ResNet50_Frozen", "transfer_learning", "rgb", "ResNet50_frozen"),
    ("ResNet50_FineTuned", "transfer_learning", "rgb", "ResNet50_fine_tuned"),
]


def resolve_split_dir(split_dir: Path | None) -> Path:
    if split_dir is not None:
        selected = split_dir.resolve()
    else:
        cache_versions = (
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "paultimothymooney"
            / "chest-xray-pneumonia"
            / "versions"
        )
        candidates = [
            PROJECT_ROOT / "data" / "chest_xray_cnn_splits",
            PROJECT_ROOT / "chest_xray_cnn_splits",
            cache_versions / "chest_xray_cnn_splits",
            *sorted(cache_versions.glob("*/chest_xray_cnn_splits")),
        ]
        selected = next(
            (
                path.resolve()
                for path in candidates
                if all((path / split).is_dir() for split in ("train", "val", "test"))
            ),
            None,
        )
        if selected is None:
            raise FileNotFoundError(
                "Set --split-dir to the prepared data matching data/dataset_manifest.csv."
            )
    return selected


def load_split(data_dir: Path, split: str, color_mode: str) -> Any:
    return tf.keras.utils.image_dataset_from_directory(
        data_dir / split,
        image_size=IMAGE_SIZE,
        color_mode=color_mode,
        batch_size=32,
        class_names=CLASS_NAMES,
        label_mode="int",
        shuffle=False,
    )


def get_checkpoint(name: str, stem: str, run_id: str) -> Path:
    if name.startswith("Baseline_"):
        return PROJECT_ROOT / "models" / "baseline" / f"{stem}_{run_id}.keras"
    if name.endswith("_Frozen"):
        architecture = name.split("_")[0]
        return PROJECT_ROOT / "models" / "frozen" / f"{architecture}_{run_id}_frozen.keras"
    architecture = name.split("_")[0]
    return PROJECT_ROOT / "models" / "fine_tuned" / f"{architecture}_{run_id}_fine_tuned.keras"


def load_model(name: str, checkpoint: Path) -> Any:
    custom_objects = None
    if name.startswith("DenseNet121"):
        custom_objects = {"preprocess_input": tf.keras.applications.densenet.preprocess_input}
    elif name.startswith("ResNet50"):
        custom_objects = {"preprocess_input": tf.keras.applications.resnet50.preprocess_input}
    model = tf.keras.models.load_model(
        checkpoint,
        compile=False,
        safe_mode=True,
        custom_objects=custom_objects,
    )
    expected_channels = 1 if name.startswith("Baseline_") else 3
    if model.input_shape != (None, *IMAGE_SIZE, expected_channels) or model.output_shape != (None, 2):
        raise ValueError(
            f"{name}: checkpoint shapes {model.input_shape}->{model.output_shape} are incompatible."
        )
    return model


def predict_positive_scores(model: Any, dataset: Any) -> np.ndarray:
    probabilities = model.predict(dataset, verbose=0)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"Expected two-class probabilities, got shape {probabilities.shape}.")
    return probabilities[:, 1]


def calculate_test_loss(labels: np.ndarray, positive_scores: np.ndarray) -> float:
    probabilities = np.column_stack((1.0 - positive_scores, positive_scores))
    selected = probabilities[np.arange(labels.size), labels]
    return float(np.mean(-np.log(np.clip(selected, np.finfo(np.float64).eps, 1.0))))


def evaluate_run(run_id: str, split_dir: Path | None = None) -> list[dict[str, Any]]:
    data_dir = resolve_split_dir(split_dir)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "data" / "validate_dataset.py"),
            "--split-dir",
            str(data_dir),
        ],
        check=True,
    )
    manifest_summary = json.loads(
        (PROJECT_ROOT / "data" / "dataset_manifest_summary.json").read_text(encoding="utf-8")
    )
    manifest_sha256 = manifest_summary["manifest_sha256"]

    datasets: dict[tuple[str, str], Any] = {}
    labels: dict[tuple[str, str], np.ndarray] = {}
    for color_mode in ("grayscale", "rgb"):
        dataset = load_split(data_dir, "val", color_mode)
        datasets[(color_mode, "val")] = dataset
        labels[(color_mode, "val")] = np.concatenate(
            [batch_labels.numpy().reshape(-1) for _, batch_labels in dataset]
        )

    model_entries = []
    validation_records = []
    for name, experiment, color_mode, stem in MODEL_SPECS:
        checkpoint = get_checkpoint(name, stem, run_id)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {checkpoint}")
        model = load_model(name, checkpoint)
        validation_labels = labels[(color_mode, "val")]
        validation_scores = predict_positive_scores(model, datasets[(color_mode, "val")])
        operating_point = select_threshold_for_sensitivity(
            validation_labels, validation_scores, TARGET_SENSITIVITY
        )
        validation_metrics, _ = calculate_binary_metrics(
            validation_labels, validation_scores, operating_point["threshold"]
        )
        validation_record = {
            "model": name,
            "experiment": experiment,
            "run_id": run_id,
            "manifest_sha256": manifest_sha256,
            **operating_point,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        validation_records.append(validation_record)
        model_entries.append(
            {
                "name": name,
                "experiment": experiment,
                "color_mode": color_mode,
                "checkpoint": checkpoint,
                "model": model,
                "validation_labels": validation_labels,
                "validation_scores": validation_scores,
                "operating_point": operating_point,
            }
        )
        print(
            f"Validation: {name} threshold={operating_point['threshold']:.4f}, "
            f"sensitivity={operating_point['validation_sensitivity']:.4f}, "
            f"precision={operating_point['validation_precision']:.4f}"
        )

    selected = max(
        validation_records,
        key=lambda record: (
            record["validation_precision"],
            record["validation_sensitivity"],
            record["validation_specificity"],
            record["model"],
        ),
    )
    selection_record = {
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "target_sensitivity": TARGET_SENSITIVITY,
        "selection_rule": (
            "Maximize validation precision among models meeting the target PNEUMONIA "
            "sensitivity; tie-break by validation sensitivity, specificity, then model name."
        ),
        "selected_model": selected["model"],
        "validation_metrics": selected,
        "all_validation_metrics": validation_records,
        "test_evaluation_started_after_selection": True,
    }
    selection_path = PROJECT_ROOT / "results" / "metrics" / f"model_selection_{run_id}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    if selection_path.exists():
        raise FileExistsError(f"Refusing to overwrite model-selection results: {selection_path}")
    selection_path.write_text(json.dumps(selection_record, indent=2) + "\n", encoding="utf-8")
    print(f"Validation-only model selection: {selected['model']}")
    print("All validation thresholds and model selection are fixed; beginning held-out test evaluation.")

    for color_mode in ("grayscale", "rgb"):
        dataset = load_split(data_dir, "test", color_mode)
        datasets[(color_mode, "test")] = dataset
        labels[(color_mode, "test")] = np.concatenate(
            [batch_labels.numpy().reshape(-1) for _, batch_labels in dataset]
        )

    results = []
    for entry in model_entries:
        name = entry["name"]
        color_mode = entry["color_mode"]
        test_labels = labels[(color_mode, "test")]
        test_scores = predict_positive_scores(entry["model"], datasets[(color_mode, "test")])
        test_loss = calculate_test_loss(test_labels, test_scores)
        if entry["experiment"] == "transfer_learning":
            output_dir = (
                PROJECT_ROOT / "results" / "transfer_learning" / run_id / "per_model" / name
            )
        else:
            slug = {
                "Baseline_ClassWeighted": "class_weighted",
                "Baseline_Oversampling": "oversampling",
                "Baseline_Augmentation": "augmentation",
            }[name]
            output_dir = PROJECT_ROOT / "results" / "baseline_cnn" / run_id / slug
        record, _ = save_evaluation_artifacts(
            entry["validation_labels"],
            entry["validation_scores"],
            test_labels,
            test_scores,
            model_name=name,
            run_id=run_id,
            experiment=entry["experiment"],
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            test_loss=test_loss,
            operating_point=entry["operating_point"],
        )
        results.append(record)
        print(
            f"Test: {name} loss={test_loss:.4f}, accuracy={record['accuracy']:.4f}, "
            f"sensitivity={record['sensitivity_recall']:.4f}, "
            f"specificity={record['specificity']:.4f}"
        )

    comparison_paths = write_comparison(run_id)
    for path in comparison_paths:
        print("Comparison:", path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--split-dir", type=Path)
    arguments = parser.parse_args()
    evaluate_run(arguments.run_id, arguments.split_dir)


if __name__ == "__main__":
    main()
