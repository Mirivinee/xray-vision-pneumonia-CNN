import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .evaluate import _finish

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


THUMB = 128
STAT_COLS = ["rel_path", "split", "label", "subtype", "mode", "width", "height"]
META_FEATURES = ["width", "height", "aspect", "is_rgb", "file_bytes", "mean", "std", "p1", "p99"]


def compute_image_stats(df, thumb=THUMB):
    rows, thumbs = [], []
    ry, rx = np.mgrid[0:thumb, 0:thumb]
    corners = (ry / thumb < 0.25) & ((rx / thumb < 0.25) | (rx / thumb > 0.75))
    for path in tqdm(df["path"], desc="image stats"):
        with Image.open(path) as im:
            im.draft("L", (2 * thumb, 2 * thumb))
            g = im.convert("L").resize((thumb, thumb), Image.BILINEAR)
        a = np.asarray(g, dtype=np.uint8)
        thumbs.append(a)
        rows.append(
            {
                "file_bytes": os.path.getsize(path),
                "mean": float(a.mean()),
                "std": float(a.std()),
                "p1": float(np.percentile(a, 1)),
                "p99": float(np.percentile(a, 99)),
                "marker_score": float((a[corners] >= 235).mean()),
            }
        )
    stats = pd.concat([df[STAT_COLS].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    stats["aspect"] = stats["width"] / stats["height"].clip(lower=1)
    stats["is_rgb"] = (stats["mode"] != "L").astype(int)
    return stats, np.stack(thumbs)


def load_or_compute_stats(df):
    if C.IMAGE_STATS_CSV.exists() and C.THUMB_NPZ.exists():
        stats = pd.read_csv(C.IMAGE_STATS_CSV, keep_default_na=False)
        thumbs = np.load(C.THUMB_NPZ)["thumbs"]
        if len(stats) == len(df) and list(stats["rel_path"]) == list(df["rel_path"]):
            for col in ("width", "height", "label", "file_bytes", "mean", "std", "p1", "p99", "marker_score", "aspect", "is_rgb"):
                stats[col] = pd.to_numeric(stats[col])
            return stats, thumbs
    stats, thumbs = compute_image_stats(df)
    C.IMAGE_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    C.THUMB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(C.IMAGE_STATS_CSV, index=False)
    np.savez_compressed(C.THUMB_NPZ, thumbs=thumbs)
    return stats, thumbs


def plot_counts(counts, path=None):
    c = counts[list(C.CLASS_NAMES)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set(title="Images per class and split", ylabel="count")
    for cont in axes[0].containers:
        axes[0].bar_label(cont, fontsize=8)
    ratio = c["PNEUMONIA"] / c["NORMAL"]
    ratio.plot(kind="bar", ax=axes[1], rot=0, color="gray")
    axes[1].set(title="Pneumonia per normal image", ylabel="ratio")
    axes[1].bar_label(axes[1].containers[0], fmt="%.2f")
    _finish(fig, path)


def plot_intensity_per_class(stats, thumbs, path=None, max_pixels=200_000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name in enumerate(C.CLASS_NAMES):
        px = thumbs[(stats["label"] == label).to_numpy()].ravel()
        px = rng.choice(px, min(max_pixels, len(px)), replace=False)
        ax.hist(px, bins=64, range=(0, 255), density=True, alpha=0.5, label=f"{name} (n={int((stats['label'] == label).sum())} images)")
    ax.set(xlabel="intensity", ylabel="density", title="Pixel-intensity histogram per class")
    ax.legend()
    _finish(fig, path)


def plot_mean_images(stats, thumbs, path=None):
    means = [thumbs[(stats["label"] == l).to_numpy()].mean(axis=0) for l in (0, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, m, t in zip(axes[:2], means, C.CLASS_NAMES):
        ax.imshow(m, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"mean {t}")
    diff = means[1] - means[0]
    v = np.abs(diff).max()
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-v, vmax=v)
    axes[2].set_title("PNEUMONIA - NORMAL")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    _finish(fig, path)


def plot_sizes(stats, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, name in enumerate(C.CLASS_NAMES):
        s = stats[stats["label"] == label]
        axes[0].scatter(s["width"], s["height"], s=6, alpha=0.3, label=name)
        axes[1].hist(s["aspect"], bins=40, alpha=0.5, density=True, label=name)
    axes[0].set(xlabel="width", ylabel="height", title="Image size per class")
    axes[1].set(xlabel="width / height", title="Aspect ratio per class")
    for split in ("train", "val", "test"):
        s = stats[stats["split"] == split]
        axes[2].hist(s["width"], bins=40, alpha=0.4, density=True, label=split)
    axes[2].set(xlabel="width", title="Width per split")
    for ax in axes:
        ax.legend()
    _finish(fig, path)


def plot_marker_gallery(stats, thumbs, path=None, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for label, name in enumerate(C.CLASS_NAMES):
        idx = stats.index[stats["label"] == label]
        top = stats.loc[idx].sort_values("marker_score", ascending=False).index[:n]
        for ax, i in zip(axes[label], top):
            ax.imshow(thumbs[i], cmap="gray")
            ax.set_title(f"{name[:4]} {stats.at[i, 'marker_score']:.2f}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Highest bright-corner score (possible text markers)")
    _finish(fig, path)


def plot_samples(stats, thumbs, path=None, per_cell=3, seed=C.SEED):
    rng = np.random.default_rng(seed)
    splits = ("train", "val", "test")
    fig, axes = plt.subplots(len(splits) * 2, per_cell, figsize=(2.4 * per_cell, 2.4 * len(splits) * 2))
    r = 0
    for split in splits:
        for label, name in enumerate(C.CLASS_NAMES):
            idx = np.flatnonzero(((stats["split"] == split) & (stats["label"] == label)).to_numpy())
            pick = rng.choice(idx, min(per_cell, len(idx)), replace=False)
            for c in range(per_cell):
                ax = axes[r, c]
                ax.axis("off")
                if c < len(pick):
                    ax.imshow(thumbs[pick[c]], cmap="gray")
                    ax.set_title(f"{split}/{name}", fontsize=7)
            r += 1
    _finish(fig, path)


def metadata_shortcut(stats, seed=C.SEED):
    tr = stats[stats["split"] == "train"]
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(tr[META_FEATURES], tr["label"])
    rows = []
    for split in ("val", "test"):
        s = stats[stats["split"] == split]
        p = model.predict_proba(s[META_FEATURES])[:, 1]
        rows.append({"split": split, "metadata_only_auc": float(roc_auc_score(s["label"], p))})
    size_only = ["width", "height", "aspect", "is_rgb", "file_bytes"]
    model2 = GradientBoostingClassifier(random_state=seed).fit(tr[size_only], tr["label"])
    for row, split in zip(rows, ("val", "test")):
        s = stats[stats["split"] == split]
        row["size_format_only_auc"] = float(roc_auc_score(s["label"], model2.predict_proba(s[size_only])[:, 1]))
    return pd.DataFrame(rows)


def adversarial_validation(stats, thumbs, seed=C.SEED, folds=5):
    y = (stats["split"] == "test").astype(int).to_numpy()
    small = thumbs.reshape(len(thumbs), 32, THUMB // 32, 32, THUMB // 32).mean(axis=(2, 4)).reshape(len(thumbs), -1) / 255.0
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    out = {}
    pix = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    out["pixels_32x32_auc"] = float(roc_auc_score(y, cross_val_predict(pix, small, y, cv=cv, method="predict_proba")[:, 1]))
    meta = stats[META_FEATURES].to_numpy(dtype=float)
    gb = GradientBoostingClassifier(random_state=seed)
    out["metadata_auc"] = float(roc_auc_score(y, cross_val_predict(gb, meta, y, cv=cv, method="predict_proba")[:, 1]))
    return pd.DataFrame([out])
