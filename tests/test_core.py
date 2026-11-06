import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="xray_test_"))
os.environ["XRAY_SPLIT_DIR"] = str(TMP / "splits")
os.environ["XRAY_RESULTS"] = str(TMP / "results")
os.environ["XRAY_ARTIFACTS"] = str(TMP / "artifacts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import dataset as D  # noqa: E402
from src import eda  # noqa: E402
from tests.fakedata import make_fake_dataset  # noqa: E402
from src import evaluate as E  # noqa: E402

ROOT = TMP / "chest_xray"


def test_group_id_and_subtype():
    assert D.group_id("person12_bacteria_3.jpeg", 1) == ("PNEUMONIA:person12", False)
    assert D.group_id("person12_virus_1.jpeg", 1)[0] == "PNEUMONIA:person12"
    assert D.group_id("IM-0115-0001.jpeg", 0) == ("NORMAL:im-0115", False)
    assert D.group_id("NORMAL2-IM-0373-0001.jpeg", 0)[0] == "NORMAL:im-0373"
    gid, fb = D.group_id("weird_name.jpeg", 0)
    assert fb and gid == "NORMAL:weird_name"
    assert D.subtype("person1_bacteria_1.jpeg", 1) == "bacterial"
    assert D.subtype("person1_virus_1.jpeg", 1) == "viral"
    assert D.subtype("IM-1.jpeg", 0) == "normal"


def test_manifest_split_and_audit():
    make_fake_dataset(ROOT)
    root = ROOT
    df, audit = D.build_manifest(root, seed=1, n_folds=10)
    assert set(df["split"]) <= {"train", "val", "test", "dropped"}
    assert audit["group_overlap"]["train_val_groups"] == 0
    kept = df[df["split"].isin(["train", "val", "test"])]
    for g, sub in kept[~kept["group_fallback"]].groupby("group_id"):
        s = set(sub["split"])
        assert not ({"train", "val"} <= s)
    test_official = df[df["official_split"] == "test"]
    assert (test_official["split"] == "test").all()
    reasons = set(df.loc[df["split"] == "dropped", "drop_reason"])
    assert "duplicate_of_test" in reasons
    assert "duplicate_in_pool" in reasons
    assert audit["counts"]["val"]["total"] > 0
    assert audit["counts"]["test"]["NORMAL"] == 14
    D.save_manifest(df)
    loaded = D.load_manifest(root)
    assert len(loaded) == len(df) and Path(loaded["path"][0]).exists()
    sp = D.get_splits(df=loaded)
    assert sum(len(v) for v in sp.values()) == len(kept)
    w = D.class_weights(sp["train"])
    n0, n1 = (sp["train"]["label"] == 0).sum(), (sp["train"]["label"] == 1).sum()
    assert abs(w[0] * n0 - w[1] * n1) < 1e-6


def test_metrics_and_threshold():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(400)].astype(int)
    p = np.r_[rng.beta(2, 5, 200), rng.beta(5, 2, 400)]
    m = E.compute_metrics(y, p, 0.5)
    assert m["tn"] + m["fp"] == 200 and m["tp"] + m["fn"] == 400
    assert 0.8 < m["roc_auc"] < 1.0
    for target in (0.9, 0.95, 0.97, 0.99):
        thr, sens, spec = E.choose_threshold(y, p, target)
        mm = E.compute_metrics(y, p, thr)
        assert mm["sensitivity"] >= target - 1e-9
        assert abs(mm["sensitivity"] - sens) < 1e-9 and abs(mm["specificity"] - spec) < 1e-9
        higher = E.compute_metrics(y, p, np.nextafter(thr, 1.0))
        assert higher["sensitivity"] <= mm["sensitivity"]
    tab = E.threshold_table(y, p)
    assert list(tab["target_sensitivity"]) == list(C.SENSITIVITY_TARGETS)
    assert tab["specificity"].is_monotonic_decreasing


def test_bootstrap_mcnemar_auc_diff():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(100), np.ones(150)].astype(int)
    p1 = np.r_[rng.beta(2, 5, 100), rng.beta(5, 2, 150)]
    p2 = np.clip(p1 + rng.normal(0, 0.15, len(p1)), 0, 1)
    ci = E.bootstrap_ci(y, p1, 0.5, n_boot=200)
    m = E.compute_metrics(y, p1, 0.5)
    assert ci["sensitivity"][0] <= m["sensitivity"] <= ci["sensitivity"][1]
    assert ci["roc_auc"][0] <= m["roc_auc"] <= ci["roc_auc"][1]
    d = E.paired_auc_diff(y, p1, p2, n_boot=200)
    assert d["ci"][0] <= d["auc_diff"] + 0.05 and 0 <= d["p_two_sided"] <= 1
    mc = E.mcnemar_exact(y, p1 >= 0.5, p1 >= 0.5)
    assert mc["p_value"] == 1.0
    mc2 = E.mcnemar_exact(y, p1 >= 0.5, p2 >= 0.5)
    assert 0 <= mc2["p_value"] <= 1


def test_screening_calibration_selective():
    t = E.screening_table(0.97, 0.7, prevalences=(0.05, 0.5), n=1000)
    row = t.iloc[1]
    assert abs(row["missed_sick"] - 15) < 1e-9 and abs(row["false_alarms"] - 150) < 1e-9
    assert t.iloc[0]["ppv"] < t.iloc[1]["ppv"]
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    z = np.where(y == 1, 1.0, -1.0) * 2.5 + rng.normal(0, 2, 2000)
    p = 1 / (1 + np.exp(-z))
    T = E.fit_temperature(y, p)
    q = E.apply_temperature(p, T)
    assert E.compute_metrics(y, q)["log_loss"] <= E.compute_metrics(y, p)["log_loss"] + 1e-9
    assert E.ece(y, q) <= E.ece(y, p) + 0.02
    sc = E.selective_curve(y, p, 0.5)
    assert sc["coverage"].iloc[0] == 1.0 and sc["accuracy"].iloc[-1] >= sc["accuracy"].iloc[0]
    ex = E.select_examples(y, p, 0.5, n=3, n_fn=4)
    assert set(ex) == {"TP", "TN", "FN", "FP"} and len(ex["TP"]) == 3 and len(ex["FN"]) <= 4
    if ex["FN"]:
        fn_p = [p[i] for i in ex["FN"]]
        assert fn_p == sorted(fn_p)


def test_pred_io_and_registry():
    from src import utils

    C.ensure_dirs()
    y = np.array([0, 1, 1])
    E.save_preds("r1", "val", y, [0.1, 0.9, 0.8], ["a", "b", "c"])
    d = E.load_preds("r1", "val")
    assert list(d["y"]) == [0, 1, 1] and list(d["rel_paths"]) == ["a", "b", "c"]
    utils.log_run({"run_id": "r1", "x": 1})
    utils.log_run({"run_id": "r2", "x": 2})
    utils.log_run({"run_id": "r1", "x": 3})
    runs = utils.load_runs()
    assert len(runs) == 2 and utils.get_run("r1")["x"] == 3
    tab = E.metrics_table(["r1"], "val")
    assert tab["roc_auc"].iloc[0] == 1.0
    E.log_test_eval("r1")
    E.log_test_eval("r2")
    log = utils.load_json(C.TEST_LOG_JSON)
    assert log["total_evaluations"] == C.LEGACY_TEST_EVALUATIONS + 2


def test_eda_stats_and_checks():
    df = D.load_manifest(ROOT)
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    stats, thumbs = eda.load_or_compute_stats(df)
    assert thumbs.shape == (len(df), eda.THUMB, eda.THUMB)
    assert {"marker_score", "aspect", "is_rgb", "file_bytes"} <= set(stats.columns)
    stats2, _ = eda.load_or_compute_stats(df)
    assert len(stats2) == len(stats)
    meta = eda.metadata_shortcut(stats)
    assert set(meta["split"]) == {"val", "test"}
    adv = eda.adversarial_validation(stats, thumbs)
    assert 0 <= adv["pixels_32x32_auc"].iloc[0] <= 1


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                failed += 1
                import traceback

                traceback.print_exc()
                print("FAIL", name, e)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
