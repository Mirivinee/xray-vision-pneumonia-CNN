import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def _valid_root(p):
    p = Path(p)
    return (p / "train").is_dir() and (p / "test").is_dir()


def _search_root(base):
    base = Path(base)
    cands = [base] + [d for d in base.rglob("*") if d.is_dir() and "__MACOSX" not in d.parts]
    valid = [d for d in cands if _valid_root(d)]
    return min(valid, key=lambda d: len(d.parts)) if valid else None


def find_data_root():
    for cand in (os.environ.get("XRAY_DATA_ROOT"), C.DATA_DIR / "chest_xray", C.DATA_DIR):
        if cand and Path(cand).is_dir():
            root = _search_root(cand)
            if root:
                return root
    import kagglehub

    root = _search_root(Path(kagglehub.dataset_download(C.DATASET_SLUG)))
    if root is None:
        raise FileNotFoundError("Dataset not found. Set XRAY_DATA_ROOT or configure Kaggle credentials.")
    return root


def collect_records(root):
    root = Path(root)
    rows = []
    for split in ("train", "val", "test"):
        for label, cname in enumerate(C.CLASS_NAMES):
            d = root / split / cname
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in C.IMAGE_EXTENSIONS and not p.name.startswith("._"):
                    rows.append(
                        {
                            "rel_path": p.relative_to(root).as_posix(),
                            "filename": p.name,
                            "label": label,
                            "official_split": split,
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    return pd.DataFrame(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_PERSON = re.compile(r"^(person\d+)_", re.I)
_IM = re.compile(r"^(?:NORMAL2-)?(IM-\d+)", re.I)


def group_id(filename, label):
    m = _PERSON.match(filename) or _IM.match(filename)
    prefix = C.CLASS_NAMES[int(label)]
    if m:
        return f"{prefix}:{m.group(1).lower()}", False
    return f"{prefix}:{Path(filename).stem.lower()}", True


def subtype(filename, label):
    if int(label) == 0:
        return "normal"
    low = filename.lower()
    if "bacteria" in low:
        return "bacterial"
    if "virus" in low:
        return "viral"
    return "unknown"


def build_manifest(root, seed=C.SEED, n_folds=C.N_VAL_FOLDS):
    root = Path(root)
    df = collect_records(root)

    meta = []
    for rel in tqdm(df["rel_path"], desc="hashing"):
        p = root / rel
        try:
            sha = sha256_file(p)
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
            meta.append((sha, w, h, mode, True))
        except Exception:
            meta.append(("", 0, 0, "ERR", False))
    meta = pd.DataFrame(meta, columns=["sha256", "width", "height", "mode", "readable"], index=df.index)
    df = pd.concat([df, meta], axis=1)

    df["subtype"] = [subtype(f, l) for f, l in zip(df["filename"], df["label"])]
    gid = [group_id(f, l) for f, l in zip(df["filename"], df["label"])]
    df["group_id"] = [g for g, _ in gid]
    df["group_fallback"] = [fb for _, fb in gid]

    is_test = (df["official_split"] == "test").to_numpy()
    readable = df["readable"].to_numpy()
    df["split"] = np.where(is_test, "test", "pool")
    df["drop_reason"] = ""

    unreadable = ~readable
    df.loc[unreadable, "split"] = "dropped"
    df.loc[unreadable, "drop_reason"] = "unreadable"

    ok = df[readable]
    test_hashes = set(ok.loc[ok["official_split"] == "test", "sha256"])
    n_labels = ok.groupby("sha256")["label"].nunique()
    conflict = set(n_labels[n_labels > 1].index)

    seen = set()
    for i in df.index[(~is_test) & readable]:
        h = df.at[i, "sha256"]
        if h in conflict:
            reason = "label_conflict"
        elif h in test_hashes:
            reason = "duplicate_of_test"
        elif h in seen:
            reason = "duplicate_in_pool"
        else:
            seen.add(h)
            continue
        df.at[i, "split"] = "dropped"
        df.at[i, "drop_reason"] = reason

    pool = df[df["split"] == "pool"]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(pool, pool["label"], pool["group_id"]))
    df.loc[pool.index[tr], "split"] = "train"
    df.loc[pool.index[va], "split"] = "val"

    return df.reset_index(drop=True), audit_manifest(df)


def audit_manifest(df):
    names = C.CLASS_NAMES
    splits = ("train", "val", "test")

    def counts(col, values):
        return {
            s: {n: int(((df[col] == s) & (df["label"] == i)).sum()) for i, n in enumerate(names)} for s in values
        }

    kept = counts("split", splits)
    for s in splits:
        n0, n1 = kept[s]["NORMAL"], kept[s]["PNEUMONIA"]
        kept[s]["total"] = n0 + n1
        kept[s]["pneumonia_fraction"] = n1 / (n0 + n1) if n0 + n1 else float("nan")
        kept[s]["pneumonia_per_normal"] = n1 / n0 if n0 else float("nan")

    dropped = {}
    d = df[df["split"] == "dropped"]
    for reason, g in d.groupby("drop_reason"):
        dropped[reason] = {n: int((g["label"] == i).sum()) for i, n in enumerate(names)}

    keep = df[df["split"].isin(splits)]
    groups = {s: set(keep.loc[(keep["split"] == s) & ~keep["group_fallback"], "group_id"]) for s in splits}
    overlap = {
        "train_val_groups": len(groups["train"] & groups["val"]),
        "train_test_groups": len(groups["train"] & groups["test"]),
        "val_test_groups": len(groups["val"] & groups["test"]),
    }
    affected = {
        s: int((keep["split"].eq(s) & keep["group_id"].isin(groups["test"]) & ~keep["group_fallback"]).sum())
        for s in ("train", "val")
    }
    return {
        "official_counts": counts("official_split", ("train", "val", "test")),
        "counts": kept,
        "dropped": dropped,
        "test_internal_duplicates": int(keep.loc[keep["split"] == "test", "sha256"].duplicated().sum()),
        "group_overlap": overlap,
        "images_sharing_group_with_test": affected,
        "group_fallback": {
            "images": int(keep["group_fallback"].sum()),
            "fraction": float(keep["group_fallback"].mean()),
        },
    }


def save_manifest(df, path=C.MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(root=None, path=C.MANIFEST_PATH):
    df = pd.read_csv(path, keep_default_na=False)
    for col in ("readable", "group_fallback"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    root = Path(root) if root else find_data_root()
    df["path"] = [str(root / r) for r in df["rel_path"]]
    return df


def get_splits(root=None, df=None):
    df = load_manifest(root) if df is None else df
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}


def class_weights(df):
    counts = np.bincount(df["label"].to_numpy().astype(int), minlength=2)
    n = counts.sum()
    return {i: float(n / (2 * c)) for i, c in enumerate(counts)}


@dataclass
class AugConfig:
    rotation: float = 0.02
    zoom: float = 0.05
    translation: float = 0.02
    brightness: float = 0.05
    contrast: float = 0.10
    hflip: bool = False

    def as_dict(self):
        return asdict(self)


def build_augmenter(cfg, seed=C.SEED):
    import tensorflow as tf

    L = tf.keras.layers
    layers = []
    if cfg.rotation:
        layers.append(L.RandomRotation(cfg.rotation, fill_mode="constant", fill_value=0.0, seed=seed))
    if cfg.zoom:
        layers.append(L.RandomZoom((-cfg.zoom, cfg.zoom), fill_mode="constant", fill_value=0.0, seed=seed + 1))
    if cfg.translation:
        layers.append(
            L.RandomTranslation(
                cfg.translation, cfg.translation, fill_mode="constant", fill_value=0.0, seed=seed + 2
            )
        )
    if cfg.brightness:
        layers.append(L.RandomBrightness(cfg.brightness, value_range=(0.0, 255.0), seed=seed + 3))
    if cfg.contrast:
        layers.append(L.RandomContrast(cfg.contrast, seed=seed + 4))
    if cfg.hflip:
        layers.append(L.RandomFlip("horizontal", seed=seed + 5))
    return tf.keras.Sequential(layers, name="augment") if layers else None


def _decode_resize(path, image_size, resize):
    import tensorflow as tf

    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=1, expand_animations=False)
    img = tf.cast(img, tf.float32)
    h, w = image_size
    if resize == "pad":
        img = tf.image.resize_with_pad(img, h, w, antialias=True)
    else:
        img = tf.image.resize(img, (h, w), antialias=True)
    img.set_shape((h, w, 1))
    return img


def load_image(path, image_size=C.IMAGE_SIZE, resize="squash"):
    return _decode_resize(str(path), image_size, resize).numpy()


def make_dataset(
    df,
    *,
    training,
    batch_size=C.BATCH_SIZE,
    image_size=C.IMAGE_SIZE,
    resize="squash",
    oversample=False,
    augment=None,
    seed=C.SEED,
):
    import tensorflow as tf

    auto = tf.data.AUTOTUNE
    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy().astype("int32")
    n = len(df)

    def load(path, label):
        return _decode_resize(path, image_size, resize), label

    if training and oversample:
        parts = []
        for c in (0, 1):
            sel = labels == c
            d = tf.data.Dataset.from_tensor_slices((paths[sel], labels[sel]))
            parts.append(d.shuffle(int(sel.sum()), seed=seed + c).repeat())
        ds = tf.data.Dataset.sample_from_datasets(parts, weights=[0.5, 0.5], seed=seed)
        steps = int(np.ceil(2 * np.bincount(labels, minlength=2).max() / batch_size))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
        steps = int(np.ceil(n / batch_size))

    ds = ds.map(load, num_parallel_calls=auto).batch(batch_size)
    if training and augment is not None:
        aug = build_augmenter(augment, seed)
        if aug is not None:
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=auto)
    return ds.prefetch(auto), steps
