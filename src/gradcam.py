from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from . import config as C
from . import dataset as D
from . import evaluate as E

K = tf.keras


def _call(layer, h):
    try:
        return layer(h, training=False)
    except TypeError:
        return layer(h)


def _forward(model, x, tape):
    h, feats, logit = x, None, None
    for layer in model.layers:
        if isinstance(layer, K.layers.InputLayer):
            continue
        h = _call(layer, h)
        if feats is None and (layer.name == "features" or isinstance(layer, K.Model)):
            feats = h
            tape.watch(feats)
        if layer.name == "logit":
            logit = h
    return feats, logit[:, 0]


def _feats_and_grads(model, image, target):
    x = tf.convert_to_tensor(np.asarray(image, dtype=np.float32)[None])
    with tf.GradientTape() as tape:
        feats, logit = _forward(model, x, tape)
        score = logit if target == "pneumonia" else -logit
    return feats, tape.gradient(score, feats)


def _upsample(cam, size):
    cam = tf.image.resize(cam[..., None], size)[0, ..., 0].numpy()
    return cam / (cam.max() + 1e-8)


def gradcam(model, image, target="pneumonia"):
    feats, grads = _feats_and_grads(model, image, target)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1))
    return _upsample(cam, image.shape[:2])


def gradcampp(model, image, target="pneumonia"):
    feats, grads = _feats_and_grads(model, image, target)
    g2, g3 = grads**2, grads**3
    total = tf.reduce_sum(feats, axis=(1, 2), keepdims=True)
    alpha = g2 / (2.0 * g2 + total * g3 + 1e-8)
    w = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1))
    return _upsample(cam, image.shape[:2])


def occlusion_map(model, image, target="pneumonia", patch=32, stride=16):
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    fill = float(img.mean())
    base = float(np.asarray(model(img[None], training=False)).reshape(-1)[0])
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    batch = []
    for y in ys:
        for x in xs:
            im2 = img.copy()
            im2[y : y + patch, x : x + patch, :] = fill
            batch.append(im2)
    probs = np.asarray(model.predict(np.stack(batch), batch_size=64, verbose=0)).reshape(-1)
    if target != "pneumonia":
        base, probs = 1.0 - base, 1.0 - probs
    drop = (base - probs).reshape(len(ys), len(xs))
    up = tf.image.resize(drop[..., None].astype(np.float32), (h, w))[..., 0].numpy()
    return up, base


def region_masks(shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ry, rx = yy / h, xx / w
    return {
        "center": (ry >= 0.15) & (ry <= 0.85) & (rx >= 0.15) & (rx <= 0.85),
        "border": (ry < 0.08) | (ry > 0.92) | (rx < 0.08) | (rx > 0.92),
        "top_corners": (ry < 0.25) & ((rx < 0.25) | (rx > 0.75)),
    }


def cam_mass_fractions(cam):
    total = float(cam.sum()) + 1e-8
    out = {}
    for name, m in region_masks(cam.shape).items():
        out[f"{name}_mass"] = float(cam[m].sum() / total)
        out[f"{name}_area"] = float(m.mean())
    return out


def predict_masked(model, df, region, mode, resize="squash", image_size=C.IMAGE_SIZE):
    image_size = tuple(image_size)
    m = region_masks(image_size)[region]
    keep = (~m) if mode == "remove" else m
    keep_t = tf.constant(keep[..., None].astype(np.float32))
    ds, _ = D.make_dataset(df, training=False, resize=resize, image_size=image_size)
    ds = ds.map(lambda x, y: (x * keep_t, y))
    return E.predict_probs(model, ds)


def randomization_check(trained_model, random_model, images, target="pneumonia"):
    rhos = []
    for img in images:
        a = gradcam(trained_model, img, target).ravel()
        b = gradcam(random_model, img, target).ravel()
        rhos.append(float(spearmanr(a, b).statistic) if a.std() > 0 and b.std() > 0 else float("nan"))
    return rhos


def cam_figure(model, items, path=None, methods=("gradcam", "gradcampp"), resize="squash", image_size=C.IMAGE_SIZE):
    fns = {"gradcam": gradcam, "gradcampp": gradcampp}
    ncols = 1 + len(methods)
    fig, axes = plt.subplots(len(items), ncols, figsize=(3.2 * ncols, 3.2 * len(items)), squeeze=False)
    for r, item in enumerate(items):
        img = D.load_image(item["path"], image_size, resize)
        axes[r, 0].imshow(img[..., 0], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(item["title"], fontsize=8)
        for c, name in enumerate(methods, start=1):
            if name == "occlusion":
                cam, _ = occlusion_map(model, img, item.get("target", "pneumonia"))
                cam = np.clip(cam, 0, None)
                cam = cam / (cam.max() + 1e-8)
            else:
                cam = fns[name](model, img, item.get("target", "pneumonia"))
            axes[r, c].imshow(img[..., 0], cmap="gray", vmin=0, vmax=255)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[r, c].set_title(name, fontsize=8)
        for c in range(ncols):
            axes[r, c].axis("off")
    E._finish(fig, path)
