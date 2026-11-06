import json

import tensorflow as tf

from . import config as C
from .utils import weights_path

K = tf.keras
L = K.layers

TL_ARCHS = {
    "densenet121": ("DenseNet121", "torch"),
    "resnet50": ("ResNet50", "caffe"),
    "efficientnetb0": ("EfficientNetB0", "none"),
}


class GrayToRGB(L.Layer):
    def call(self, x):
        return tf.repeat(x, 3, axis=-1)


class Preprocess(L.Layer):
    MEAN_BGR = (103.939, 116.779, 123.68)
    TORCH_MEAN = (0.485, 0.456, 0.406)
    TORCH_STD = (0.229, 0.224, 0.225)

    def __init__(self, mode="none", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if self.mode == "torch":
            return (x / 255.0 - tf.constant(self.TORCH_MEAN)) / tf.constant(self.TORCH_STD)
        if self.mode == "caffe":
            return x[..., ::-1] - tf.constant(self.MEAN_BGR)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg["mode"] = self.mode
        return cfg


class BinaryFocalLoss(K.losses.Loss):
    # alpha weights the positive (PNEUMONIA) class; 0.25 up-weights the minority NORMAL class
    def __init__(self, gamma=2.0, alpha=0.25, name="binary_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)
        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * p + (1.0 - y_true) * (1.0 - p)
        at = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        return tf.reduce_mean(-at * tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt), axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(gamma=self.gamma, alpha=self.alpha)
        return cfg


def build_scratch_cnn(
    input_shape=(224, 224, 1),
    filters=(32, 64, 128, 256),
    use_bn=True,
    dropout=0.4,
    dense_units=64,
    pooling="gap",
    name="scratch_cnn",
):
    inputs = K.Input(tuple(input_shape), name="image")
    x = L.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    filters = tuple(filters)
    for i, f in enumerate(filters):
        x = L.Conv2D(f, 3, padding="same", use_bias=not use_bn, name=f"conv{i + 1}")(x)
        if use_bn:
            x = L.BatchNormalization(name=f"bn{i + 1}")(x)
        last = i == len(filters) - 1
        x = L.Activation("relu", name="features" if last else f"relu{i + 1}")(x)
        x = L.MaxPooling2D(name=f"pool{i + 1}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x) if pooling == "gap" else L.Flatten(name="flatten")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    if dense_units:
        x = L.Dense(dense_units, activation="relu", name="fc")(x)
    logit = L.Dense(1, name="logit")(x)
    prob = L.Activation("sigmoid", name="prob")(logit)
    return K.Model(inputs, prob, name=name)


def build_tl_model(arch, input_shape=(224, 224, 1), dropout=0.3, pretrained=True, name=None):
    cls_name, pre_mode = TL_ARCHS[arch]
    input_shape = tuple(input_shape)
    inputs = K.Input(input_shape, name="image")
    x = GrayToRGB(name="gray_to_rgb")(inputs)
    x = Preprocess(pre_mode, name="preprocess")(x)
    backbone = getattr(K.applications, cls_name)(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    x = backbone(x, training=False)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    logit = L.Dense(1, name="logit")(x)
    prob = L.Activation("sigmoid", name="prob")(logit)
    return K.Model(inputs, prob, name=name or arch)


def get_backbone(model):
    return next(l for l in model.layers if isinstance(l, K.Model))


def set_trainable(model, n_unfrozen):
    bb = get_backbone(model)
    if n_unfrozen == 0:
        bb.trainable = False
        return
    bb.trainable = True
    layers = bb.layers
    cutoff = 0 if n_unfrozen is None else max(0, len(layers) - n_unfrozen)
    for i, layer in enumerate(layers):
        layer.trainable = i >= cutoff and not isinstance(layer, L.BatchNormalization)


def compile_model(model, lr, loss="bce", focal_gamma=2.0, focal_alpha=0.25):
    loss_obj = BinaryFocalLoss(focal_gamma, focal_alpha) if loss == "focal" else K.losses.BinaryCrossentropy()
    model.compile(
        optimizer=K.optimizers.Adam(lr),
        loss=loss_obj,
        metrics=[
            K.metrics.BinaryAccuracy(name="acc"),
            K.metrics.AUC(name="auc"),
            K.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_from_cfg(cfg, pretrained=False):
    if cfg["family"] == "scratch":
        return build_scratch_cnn(**cfg["kwargs"])
    return build_tl_model(cfg["family"], pretrained=pretrained, **cfg["kwargs"])


def load_run_model(row):
    cfg = json.loads(row["model_cfg"])
    model = build_from_cfg(cfg, pretrained=False)
    model.load_weights(str(weights_path(row["run_id"])))
    return model
