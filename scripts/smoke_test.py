import os
import shutil
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="xray_smoke_"))
os.environ["XRAY_DATA_ROOT"] = str(TMP / "chest_xray")
os.environ["XRAY_SPLIT_DIR"] = str(TMP / "splits")
os.environ["XRAY_RESULTS"] = str(TMP / "results")
os.environ["XRAY_ARTIFACTS"] = str(TMP / "artifacts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fakedata import make_fake_dataset  # noqa: E402

make_fake_dataset(TMP / "chest_xray", n_train=60, n_val=4, n_test=16)

import json  # noqa: E402

import numpy as np  # noqa: E402,F811

from src import config as C  # noqa: E402
from src import dataset as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import gradcam as G  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402
from src.utils import get_run, load_runs, write_env  # noqa: E402

IMG = (64, 64)
C.ensure_dirs()
write_env()

df, audit = D.build_manifest(Path(os.environ["XRAY_DATA_ROOT"]), n_folds=10)
D.save_manifest(df)
sp = D.get_splits()
print({k: len(v) for k, v in sp.items()})

for strategy in T.STRATEGIES:
    T.train_scratch(strategy, 42, epochs=2, patience=2, image_size=IMG, model_kwargs=dict(filters=(8, 16, 32)), tag="smoke")

for arch in M.TL_ARCHS:
    T.train_transfer(arch, 42, head_epochs=1, ft_epochs=1, n_unfrozen=10, image_size=IMG, pretrained=False, tag="smoke")

runs = load_runs()
print(runs[["run_id", "variant", "val_auc", "epochs_trained"]])
assert len(runs) == len(T.STRATEGIES) + 2 * len(M.TL_ARCHS)

scratch_id = "scratch-smoke-oversample-s42"
tl_id = "resnet50-smoke-cw_aug-s42-ft-u10-lr1e-05"
for rid in (scratch_id, tl_id):
    d = E.predict_run(rid, "test")
    m = E.compute_metrics(d["y"], d["p"], 0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0
    E.predict_run(rid, "val", tta=True)

val = E.load_preds(tl_id, "val")
thr, sens, spec = E.choose_threshold(val["y"], val["p"], 0.97)
print("threshold", thr, sens, spec)
E.bootstrap_ci(val["y"], val["p"], thr, n_boot=50)
E.plot_roc({"a": (val["y"], val["p"])}, C.FIG_DIR / "roc.png")
E.plot_pr({"a": (val["y"], val["p"])}, C.FIG_DIR / "pr.png")
E.plot_confusion(val["y"], val["p"], thr, C.FIG_DIR / "cm.png")
E.plot_reliability({"a": (val["y"], val["p"])}, C.FIG_DIR / "rel.png")
E.plot_selective(E.selective_curve(val["y"], val["p"], thr), C.FIG_DIR / "sel.png")
T.plot_transfer_curves("resnet50-smoke-cw_aug-s42-frozen", tl_id, C.FIG_DIR / "tl_curves.png")

test_df = sp["test"]
for rid in (scratch_id, tl_id):
    row = get_run(rid)
    model = M.load_run_model(row)
    img = D.load_image(test_df["path"][0], IMG)
    for fn in (G.gradcam, G.gradcampp):
        cam = fn(model, img)
        assert cam.shape == IMG and np.isfinite(cam).all()
    occ, base = G.occlusion_map(model, img, patch=16, stride=16)
    assert occ.shape == IMG
    print(G.cam_mass_fractions(G.gradcam(model, img)))
    items = [{"path": test_df["path"][i], "title": f"t{i}"} for i in range(2)]
    G.cam_figure(model, items, C.FIG_DIR / f"cam_{rid}.png", methods=("gradcam", "gradcampp", "occlusion"), image_size=IMG)
    p = G.predict_masked(model, test_df, "center", "remove", image_size=IMG)
    assert len(p) == len(test_df)
    rnd = M.build_from_cfg(json.loads(row["model_cfg"]), pretrained=False)
    G.randomization_check(model, rnd, [img])

shutil.rmtree(TMP, ignore_errors=True)
print("SMOKE TEST PASSED")
