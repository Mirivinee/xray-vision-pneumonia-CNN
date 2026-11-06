import numpy as np
import pytest

from src.evaluation.binary_metrics import (
    calculate_binary_metrics,
    load_training_history,
    save_evaluation_artifacts,
    save_training_history,
    select_threshold_for_sensitivity,
)


def test_threshold_meets_target_sensitivity_and_maximizes_precision():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])

    selected = select_threshold_for_sensitivity(labels, scores, 0.90)

    assert selected["threshold"] == pytest.approx(0.4)
    assert selected["validation_sensitivity"] == 1.0
    assert selected["validation_precision"] == pytest.approx(2 / 3)


def test_metrics_use_fixed_threshold_and_correct_class_mapping():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.7, 0.8, 0.9])

    metrics, matrix = calculate_binary_metrics(labels, scores, 0.75)

    assert matrix.tolist() == [[2, 0], [0, 2]]
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_training_history_round_trips(tmp_path):
    history_path = tmp_path / "training_history.csv"
    history = type(
        "History",
        (),
        {"epoch": [0, 1], "history": {"loss": [0.8, 0.4], "val_loss": [0.7, 0.5]}},
    )()

    save_training_history(history, history_path)
    restored = load_training_history(history_path)

    assert restored.epoch == [0, 1]
    assert restored.history == {"loss": [0.8, 0.4], "val_loss": [0.7, 0.5]}


def test_evaluation_artifacts_use_validation_threshold_on_held_out_scores(tmp_path):
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])
    operating_point = select_threshold_for_sensitivity(labels, scores, 0.90)
    metrics, detail = save_evaluation_artifacts(
        labels,
        scores,
        labels,
        scores,
        model_name="Synthetic",
        run_id="test_run",
        experiment="test",
        output_dir=tmp_path,
        manifest_sha256="manifest-hash",
        test_loss=0.25,
        operating_point=operating_point,
    )

    assert metrics["threshold"] == pytest.approx(0.4)
    assert metrics["validation_sensitivity"] == 1.0
    assert metrics["test_loss"] == pytest.approx(0.25)
    assert detail["matrix"].tolist() == [[1, 1], [0, 2]]
    for filename in (
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "predictions.npz",
    ):
        assert (tmp_path / filename).is_file()


@pytest.mark.parametrize(
    ("labels", "scores", "target"),
    [
        ([0, 1], [0.2], 0.9),
        ([0, 2], [0.2, 0.8], 0.9),
        ([0, 1], [0.2, float("nan")], 0.9),
        ([0, 1], [0.2, 0.8], 0.0),
        ([1, 1], [0.2, 0.8], 0.9),
    ],
)
def test_invalid_threshold_inputs_are_rejected(labels, scores, target):
    with pytest.raises(ValueError):
        select_threshold_for_sensitivity(np.array(labels), np.array(scores), target)
