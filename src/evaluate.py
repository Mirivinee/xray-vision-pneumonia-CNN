import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as C
from . import dataset as D
from .utils import get_run, load_json, now_iso, save_json

EPS = 1e-7


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tn, fp, fn, tp


def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tn, fp, fn, tp = confusion(y, (p >= thr).astype(int))
    two = len(np.unique(y)) == 2
    return {
        "threshold": float(thr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / len(y),
        "precision": _div(tp, tp + fp),
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def choose_threshold(y, p, target_sens=C.TARGET_SENSITIVITY):
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    i = int(np.argmax(tpr >= target_sens))
    return float(thr[i]), float(tpr[i]), float(1.0 - fpr[i])


def spec_at_sens(y, p, target_sens=C.TARGET_SENSITIVITY):
    return choose_threshold(y, p, target_sens)[2]


def threshold_table(y, p, targets=C.SENSITIVITY_TARGETS):
    rows = []
    for t in targets:
        thr, sens, spec = choose_threshold(y, p, t)
        m = compute_metrics(y, p, thr)
        rows.append(
            {
                "target_sensitivity": t,
                "threshold": thr,
                "sensitivity": sens,
                "specificity": spec,
                "fn": m["fn"],
                "fp": m["fp"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, thr, n_boot=2000, seed=C.SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    keys = ("accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc")
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        pred = pb >= thr
        tp = ((yb == 1) & pred).sum()
        fn = ((yb == 1) & ~pred).sum()
        tn = ((yb == 0) & ~pred).sum()
        fp = ((yb == 0) & pred).sum()
        vals["accuracy"].append((tp + tn) / n)
        vals["sensitivity"].append(tp / (tp + fn))
        vals["specificity"].append(tn / (tn + fp))
        vals["roc_auc"].append(roc_auc_score(yb, pb))
        vals["pr_auc"].append(average_precision_score(yb, pb))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in vals.items()}


def paired_auc_diff(y, p1, p2, n_boot=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = np.asarray(diffs)
    return {
        "auc_diff": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
        "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
        "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
    }


def mcnemar_exact(y, pred1, pred2):
    y = np.asarray(y).astype(int)
    c1 = np.asarray(pred1).astype(int) == y
    c2 = np.asarray(pred2).astype(int) == y
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return {"only_first_correct": b, "only_second_correct": c, "p_value": p}


def screening_table(sens, spec, prevalences=C.PREVALENCES, n=1000):
    rows = []
    for prev in prevalences:
        pos = n * prev
        neg = n - pos
        tp = sens * pos
        fn = pos - tp
        fp = (1 - spec) * neg
        tn = neg - fp
        rows.append(
            {
                "prevalence": prev,
                "screened": n,
                "sick": pos,
                "missed_sick": fn,
                "false_alarms": fp,
                "ppv": _div(tp, tp + fp),
                "npv": _div(tn, tn + fn),
            }
        )
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reliability_table(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(y, p, n_bins=10):
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


def fit_temperature(y, p):
    y = np.asarray(y).astype(float)
    z = _logit(p)

    def nll(log_t):
        q = np.clip(_sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(p, temperature):
    return _sigmoid(_logit(p) / temperature)


def selective_curve(y, p, thr, coverages=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    coverages = np.arange(1.0, 0.449, -0.05) if coverages is None else coverages
    conf = np.abs(_logit(p) - _logit(thr))
    order = np.argsort(-conf)
    rows = []
    for cov in coverages:
        k = int(np.ceil(cov * len(y)))
        idx = order[:k]
        yk, pk = y[idx], p[idx]
        tn, fp, fn, tp = confusion(yk, (pk >= thr).astype(int))
        rows.append(
            {
                "coverage": float(cov),
                "n_kept": k,
                "sensitivity": _div(tp, tp + fn),
                "specificity": _div(tn, tn + fp),
                "accuracy": (tp + tn) / k,
                "missed_sick": fn,
                "false_alarms": fp,
                "referred_sick": int(((y == 1)).sum() - (yk == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_examples(y, p, thr, n=4, n_fn=8, seed=C.SEED):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= thr
    rng = np.random.default_rng(seed)
    tp = np.flatnonzero((y == 1) & pred)
    tn = np.flatnonzero((y == 0) & ~pred)
    fn = np.flatnonzero((y == 1) & ~pred)
    fp = np.flatnonzero((y == 0) & pred)

    def pick(idx, k):
        if len(idx) == 0:
            return []
        return sorted(rng.choice(idx, min(k, len(idx)), replace=False).tolist())

    return {
        "TP": pick(tp, n),
        "TN": pick(tn, n),
        "FN": fn[np.argsort(p[fn])][:n_fn].tolist(),
        "FP": fp[np.argsort(-p[fp])][:n].tolist(),
    }


def ensemble_probs(ps):
    return np.mean([np.asarray(p, dtype=float) for p in ps], axis=0)


def save_preds(name, split, y, p, rel_paths):
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.PRED_DIR / f"{name}_{split}.npz",
        y=np.asarray(y).astype(int),
        p=np.asarray(p, dtype=float),
        rel_paths=np.asarray(rel_paths).astype(str),
    )


def load_preds(name, split):
    z = np.load(C.PRED_DIR / f"{name}_{split}.npz", allow_pickle=False)
    return {"y": z["y"], "p": z["p"], "rel_paths": z["rel_paths"]}


def metrics_table(run_ids, split="val", thresholds=None, include_oracle=True, target_sens=C.TARGET_SENSITIVITY):
    rows = []
    for rid in run_ids:
        d = load_preds(rid, split)
        thr = 0.5 if thresholds is None else thresholds[rid]
        m = compute_metrics(d["y"], d["p"], thr)
        if include_oracle:
            m["spec_at_target_sens"] = spec_at_sens(d["y"], d["p"], target_sens)
        m["run_id"] = rid
        m["split"] = split
        rows.append(m)
    return pd.DataFrame(rows)


def log_test_eval(name, note=""):
    log = load_json(C.TEST_LOG_JSON, {"legacy_evaluations": C.LEGACY_TEST_EVALUATIONS, "events": []})
    log["events"].append({"run": name, "time": now_iso(), "note": note})
    log["total_evaluations"] = log["legacy_evaluations"] + len(log["events"])
    save_json(log, C.TEST_LOG_JSON)


def _zoom_in(x, tf):
    h, w = x.shape[1], x.shape[2]
    return tf.image.resize(tf.image.central_crop(x, 0.95), (h, w))


def _zoom_out(x, tf):
    h, w = x.shape[1], x.shape[2]
    small = tf.image.resize(x, (int(h * 0.95), int(w * 0.95)))
    return tf.image.resize_with_crop_or_pad(small, h, w)


def predict_probs(model, ds, tta=False):
    if not tta:
        return np.asarray(model.predict(ds, verbose=0)).reshape(-1)
    import tensorflow as tf

    out = []
    for x, _ in ds:
        views = [x, _zoom_in(x, tf), _zoom_out(x, tf)]
        out.append(np.mean([np.asarray(model(v, training=False)).reshape(-1) for v in views], axis=0))
    return np.concatenate(out)


def predict_df(model, df, resize="squash", image_size=C.IMAGE_SIZE, tta=False):
    ds, _ = D.make_dataset(df, training=False, resize=resize, image_size=tuple(image_size))
    return predict_probs(model, ds, tta)


def predict_run(rid, split, tta=False, force=False):
    name = f"{rid}-tta" if tta else rid
    if not force and (C.PRED_DIR / f"{name}_{split}.npz").exists():
        return load_preds(name, split)
    from . import models as M

    row = get_run(rid)
    model = M.load_run_model(row)
    df = D.get_splits()[split]
    p = predict_df(model, df, resize=row["resize"], image_size=tuple(json.loads(row["image_size"])), tta=tta)
    save_preds(name, split, df["label"].to_numpy(), p, df["rel_path"].to_numpy())
    if split == "test":
        log_test_eval(name)
    return load_preds(name, split)


def _finish(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_roc(curves, path=None, marks=None, title="ROC"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    for name, (fpr, tpr) in (marks or {}).items():
        ax.scatter([fpr], [tpr], marker="o", s=40, zorder=5)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title=title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_pr(curves, path=None, title="Precision-recall"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.set(xlabel="recall (sensitivity)", ylabel="precision", title=title)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_confusion(y, p, thr, path=None, title="Confusion matrix"):
    tn, fp, fn, tp = confusion(y, (np.asarray(p) >= thr).astype(int))
    cm = np.array([[tn, fp], [fn, tp]])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "counts"), (axes[1], norm, ".1%", "row-normalised")):
        ax.imshow(data, cmap="Blues")
        for i in range(2):
            for j in range(2):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", color="white" if v > data.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, xlabel="predicted", ylabel="true", title=sub)
    fig.suptitle(f"{title} (threshold {thr:.3f})")
    _finish(fig, path)


def plot_reliability(curves, path=None, n_bins=10, title="Reliability diagram"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        t = reliability_table(y, p, n_bins)
        ax.plot(t["mean_pred"], t["frac_pos"], marker="o", label=f"{name} (ECE {ece(y, p, n_bins):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="mean predicted probability", ylabel="observed fraction pneumonia", title=title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_history(hist, path=None, title="", marks=None):
    keys = [("loss", "loss"), ("acc", "accuracy"), ("auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (k, label) in zip(axes, keys):
        if k in hist:
            ax.plot(hist[k], label="train")
        if f"val_{k}" in hist:
            ax.plot(hist[f"val_{k}"], label="val")
        for m in marks or []:
            ax.axvline(m - 0.5, color="gray", ls="--", lw=1)
        ax.set(xlabel="epoch", title=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    _finish(fig, path)


def plot_selective(df, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["coverage"], df["sensitivity"], marker="o", label="sensitivity")
    ax.plot(df["coverage"], df["specificity"], marker="s", label="specificity")
    ax.invert_xaxis()
    ax.set(xlabel="coverage (fraction auto-decided)", ylabel="metric on decided cases", title="Selective prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, path)


def plot_gallery(rel_paths, titles, root, path=None, ncols=4, suptitle=""):
    n = len(rel_paths)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rp, t in zip(axes.ravel(), rel_paths, titles):
        img = Image.open(Path(root) / rp).convert("L")
        img.thumbnail((400, 400))
        ax.imshow(np.asarray(img), cmap="gray")
        ax.set_title(t, fontsize=8)
    fig.suptitle(suptitle)
    _finish(fig, path)
