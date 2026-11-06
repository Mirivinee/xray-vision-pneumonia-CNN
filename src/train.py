import argparse
import json
import time

import tensorflow as tf

from . import config as C
from . import dataset as D
from . import evaluate as E
from . import models as M
from .utils import get_run, is_done, log_run, now_iso, run_id, save_json, set_seed, weights_path

STRATEGIES = {
    "plain": dict(weights=False, oversample=False, focal=False, augment=False),
    "class_weight": dict(weights=True, oversample=False, focal=False, augment=False),
    "oversample": dict(weights=False, oversample=True, focal=False, augment=False),
    "focal": dict(weights=False, oversample=False, focal=True, augment=False),
    "augment": dict(weights=False, oversample=False, focal=False, augment=True),
    "cw_aug": dict(weights=True, oversample=False, focal=False, augment=True),
}


def fit(model, train_ds, val_ds, *, epochs, patience, ckpt_path, class_weight=None, steps_per_epoch=None, monitor="val_auc"):
    K = tf.keras
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        K.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor=monitor, mode="max", save_best_only=True, save_weights_only=True
        ),
        K.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=patience, restore_best_weights=False),
        K.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.3, patience=max(1, patience // 3), min_lr=1e-7
        ),
    ]
    h = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    model.load_weights(str(ckpt_path))
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def _finalize(rid, model, hist, val_df, *, cfg, family, variant, strategy, seed, tag, resize, image_size, params, t0, parent=""):
    p = E.predict_df(model, val_df, resize=resize, image_size=image_size)
    y = val_df["label"].to_numpy()
    E.save_preds(rid, "val", y, p, val_df["rel_path"].to_numpy())
    m = E.compute_metrics(y, p, 0.5)
    best_epoch = int(max(range(len(hist["val_auc"])), key=lambda i: hist["val_auc"][i])) + 1
    save_json(hist, C.HIST_DIR / f"{rid}.json")
    E.plot_history(hist, C.FIG_DIR / f"curves_{rid}.png", title=rid)
    row = {
        "run_id": rid,
        "family": family,
        "variant": variant,
        "strategy": strategy,
        "seed": seed,
        "tag": tag,
        "parent": parent,
        "model_cfg": json.dumps(cfg),
        "resize": resize,
        "image_size": json.dumps(list(image_size)),
        "params": json.dumps(params, default=str),
        "n_params": int(model.count_params()),
        "epochs_trained": len(hist["loss"]),
        "best_epoch": best_epoch,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "val_auc": m["roc_auc"],
        "val_pr_auc": m["pr_auc"],
        "val_acc": m["accuracy"],
        "val_sensitivity": m["sensitivity"],
        "val_specificity": m["specificity"],
        "val_spec_at_target_sens": E.spec_at_sens(y, p, C.TARGET_SENSITIVITY),
        "created": now_iso(),
    }
    log_run(row)
    return row


def train_scratch(
    strategy="class_weight",
    seed=C.SEED,
    *,
    model_kwargs=None,
    aug=None,
    epochs=40,
    patience=8,
    lr=1e-3,
    tag="",
    resize="squash",
    image_size=C.IMAGE_SIZE,
    force=False,
):
    image_size = tuple(image_size)
    rid = run_id("scratch", tag, strategy, f"s{seed}")
    if not force and is_done(rid):
        return get_run(rid)
    aug = aug if aug is not None else D.AugConfig()
    flags = STRATEGIES[strategy]
    mk = dict(model_kwargs or {})
    mk["input_shape"] = [*image_size, 1]
    cfg = {"family": "scratch", "kwargs": mk}

    set_seed(seed)
    sp = D.get_splits()
    train_ds, steps = D.make_dataset(
        sp["train"],
        training=True,
        oversample=flags["oversample"],
        augment=aug if flags["augment"] else None,
        seed=seed,
        resize=resize,
        image_size=image_size,
    )
    val_ds, _ = D.make_dataset(sp["val"], training=False, resize=resize, image_size=image_size)
    model = M.build_scratch_cnn(**mk)
    M.compile_model(model, lr, loss="focal" if flags["focal"] else "bce")
    cw = D.class_weights(sp["train"]) if flags["weights"] else None

    t0 = time.perf_counter()
    hist = fit(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        patience=patience,
        ckpt_path=weights_path(rid),
        class_weight=cw,
        steps_per_epoch=steps if flags["oversample"] else None,
    )
    params = {"lr": lr, "epochs": epochs, "patience": patience, "aug": aug.as_dict() if flags["augment"] else None, "class_weight": cw}
    return _finalize(
        rid, model, hist, sp["val"], cfg=cfg, family="scratch", variant="scratch", strategy=strategy,
        seed=seed, tag=tag, resize=resize, image_size=image_size, params=params, t0=t0,
    )


def train_transfer(
    arch,
    seed=C.SEED,
    *,
    strategy="cw_aug",
    aug=None,
    head_epochs=8,
    ft_epochs=20,
    head_lr=1e-3,
    ft_lr=1e-5,
    n_unfrozen=30,
    head_patience=3,
    ft_patience=5,
    dropout=0.3,
    resize="squash",
    image_size=C.IMAGE_SIZE,
    pretrained=True,
    tag="",
    force=False,
):
    image_size = tuple(image_size)
    aug = aug if aug is not None else D.AugConfig()
    flags = STRATEGIES[strategy]
    base = run_id(arch, tag, strategy, f"s{seed}")
    head_id = run_id(base, "frozen")
    u = "all" if n_unfrozen is None else n_unfrozen
    ft_id = run_id(base, f"ft-u{u}-lr{ft_lr:g}")
    cfg = {"family": arch, "kwargs": {"input_shape": [*image_size, 1], "dropout": dropout}}
    shape = (*image_size, 1)

    sp = D.get_splits()
    train_ds, steps = D.make_dataset(
        sp["train"],
        training=True,
        oversample=flags["oversample"],
        augment=aug if flags["augment"] else None,
        seed=seed,
        resize=resize,
        image_size=image_size,
    )
    val_ds, _ = D.make_dataset(sp["val"], training=False, resize=resize, image_size=image_size)
    cw = D.class_weights(sp["train"]) if flags["weights"] else None
    loss = "focal" if flags["focal"] else "bce"
    spe = steps if flags["oversample"] else None
    common = dict(family=arch, strategy=strategy, seed=seed, tag=tag, resize=resize, image_size=image_size, cfg=cfg)
    aug_params = aug.as_dict() if flags["augment"] else None

    rows = []
    if force or not is_done(head_id):
        set_seed(seed)
        model = M.build_tl_model(arch, input_shape=shape, dropout=dropout, pretrained=pretrained)
        M.set_trainable(model, 0)
        M.compile_model(model, head_lr, loss=loss)
        t0 = time.perf_counter()
        hist = fit(model, train_ds, val_ds, epochs=head_epochs, patience=head_patience, ckpt_path=weights_path(head_id), class_weight=cw, steps_per_epoch=spe)
        rows.append(
            _finalize(
                head_id, model, hist, sp["val"], variant="frozen", t0=t0,
                params={"lr": head_lr, "epochs": head_epochs, "aug": aug_params, "class_weight": cw, "n_unfrozen": 0},
                **common,
            )
        )
    else:
        rows.append(get_run(head_id))

    if force or not is_done(ft_id):
        set_seed(seed)
        model = M.build_tl_model(arch, input_shape=shape, dropout=dropout, pretrained=False)
        model.load_weights(str(weights_path(head_id)))
        M.set_trainable(model, n_unfrozen)
        M.compile_model(model, ft_lr, loss=loss)
        t0 = time.perf_counter()
        hist = fit(model, train_ds, val_ds, epochs=ft_epochs, patience=ft_patience, ckpt_path=weights_path(ft_id), class_weight=cw, steps_per_epoch=spe)
        rows.append(
            _finalize(
                ft_id, model, hist, sp["val"], variant="finetune", t0=t0, parent=head_id,
                params={"lr": ft_lr, "epochs": ft_epochs, "aug": aug_params, "class_weight": cw, "n_unfrozen": u},
                **common,
            )
        )
    else:
        rows.append(get_run(ft_id))
    return rows


def plot_transfer_curves(head_id, ft_id, path=None):
    head = json.load(open(C.HIST_DIR / f"{head_id}.json"))
    ft = json.load(open(C.HIST_DIR / f"{ft_id}.json"))
    merged = {k: head[k] + ft.get(k, []) for k in head if k in ft}
    E.plot_history(merged, path, title=f"{ft_id} (dashed line: fine-tuning starts)", marks=[len(head["loss"]) + 1])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scratch")
    a.add_argument("--strategy", default="class_weight", choices=list(STRATEGIES))
    a.add_argument("--seed", type=int, default=C.SEED)
    a.add_argument("--epochs", type=int, default=40)
    b = sub.add_parser("transfer")
    b.add_argument("--arch", default="densenet121", choices=list(M.TL_ARCHS))
    b.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    if args.cmd == "scratch":
        print(train_scratch(args.strategy, args.seed, epochs=args.epochs))
    else:
        print(train_transfer(args.arch, args.seed))


if __name__ == "__main__":
    main()
