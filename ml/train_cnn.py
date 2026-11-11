"""Train the fire/smoke/none image classifier (MobileNetV2 transfer learning).

    python -m ml.train_cnn --manifest data/fire_manifest.csv --epochs-head 5 --epochs-ft 5
    (CPU is slow: add --max-per-class 2000 --img 160, or train on Colab - see docs/COLAB.md)

Stage 1 trains the new head with the backbone frozen; stage 2 fine-tunes the top layers at a
low learning rate (BatchNorm stays frozen). The fire decision threshold is chosen on the
VALIDATION split (max F1 with recall >= 0.92 if achievable); the test split is reported once.
Model input is raw RGB 0..255 - rescaling is inside the model.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ml.prepare_fire_dataset import CLASSES


def best_threshold(y_bin, p, min_recall=0.92):
    from sklearn.metrics import f1_score, recall_score
    cands = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scored = [(float(t), f1_score(y_bin, p >= t, zero_division=0), recall_score(y_bin, p >= t, zero_division=0))
              for t in cands]
    ok = [s for s in scored if s[2] >= min_recall]
    pool = ok if ok else scored
    return max(pool, key=lambda s: s[1])[0], bool(ok)


def report(y, pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            "per_class": {c: {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(d), 4)}
                          for c, a, b, d in zip(CLASSES, pr, rc, f1)},
            "confusion_matrix(rows=true, cols=pred; none,smoke,fire)":
                confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main(argv=None, quiet=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/fire_manifest.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=5)
    ap.add_argument("--unfreeze", type=int, default=30, help="top backbone layers to fine-tune")
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--max-per-class", type=int, default=0, help="subsample TRAIN per class (CPU)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    log = (lambda *x: None) if quiet else print

    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_fscore_support
    tf.keras.utils.set_random_seed(a.seed)

    df = pd.read_csv(a.manifest)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if a.max_per_class:
        parts["train"] = (parts["train"].groupby("y", group_keys=False)
                          .apply(lambda g: g.sample(min(len(g), a.max_per_class), random_state=a.seed))
                          .reset_index(drop=True))
    counts = {s: {c: int((p["label"] == c).sum()) for c in CLASSES} for s, p in parts.items()}
    log("split counts:", counts)

    def make_ds(p, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((p["path"].to_numpy(), p["y"].to_numpy().astype("int32")))
        if shuffle:
            ds = ds.shuffle(len(p), seed=a.seed, reshuffle_each_iteration=True)

        def load(path, y):
            im = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            im.set_shape([None, None, 3])
            return tf.cast(tf.image.resize(im, (a.img, a.img)), tf.float32), y
        return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(tf.data.AUTOTUNE)

    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(input_shape=(a.img, a.img, 3), include_top=False,
                                             weights=None if a.weights == "none" else "imagenet", pooling="avg")
    base.trainable = False
    model = tf.keras.Sequential([
        L.Input((a.img, a.img, 3)),
        L.RandomFlip("horizontal"), L.RandomRotation(0.05), L.RandomZoom(0.1), L.RandomContrast(0.2),
        L.Rescaling(1 / 127.5, offset=-1.0),
        base, L.Dropout(0.3), L.Dense(len(CLASSES), activation="softmax")])

    cnt = np.bincount(parts["train"]["y"].to_numpy(), minlength=3).astype(float)
    cw = {i: float(cnt.sum() / (3 * max(cnt[i], 1))) for i in range(3)}
    cb = lambda: [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    train_ds, val_ds, test_ds = make_ds(parts["train"], True), make_ds(parts["val"], False), make_ds(parts["test"], False)

    def compile_(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    compile_(1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_head, class_weight=cw, callbacks=cb(), verbose=0 if quiet else 2)
    if a.epochs_ft > 0:
        base.trainable = True
        for layer in base.layers[:-a.unfreeze]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, L.BatchNormalization):
                layer.trainable = False
        compile_(1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_ft, class_weight=cw, callbacks=cb(), verbose=0 if quiet else 2)

    p_val = model.predict(val_ds, verbose=0)
    p_test = model.predict(test_ds, verbose=0)
    yv, yt = parts["val"]["y"].to_numpy(), parts["test"]["y"].to_numpy()
    fire = CLASSES.index("fire")
    tau, feasible = best_threshold((yv == fire).astype(int), p_val[:, fire])
    yb = (yt == fire).astype(int)
    pb = p_test[:, fire] >= tau
    pr, rc, f1, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    auc = float(roc_auc_score(yb, p_test[:, fire])) if 0 < yb.sum() < len(yb) else None
    metrics = {"split_counts": counts,
               "three_class_test": report(yt, p_test.argmax(1)),
               "fire_vs_rest_test": {"threshold": tau, "recall_target_feasible_on_val": feasible,
                                     "precision": round(float(pr), 4), "recall": round(float(rc), 4),
                                     "f1": round(float(f1), 4), "roc_auc": None if auc is None else round(auc, 4),
                                     "target_met_f1>=0.90_and_recall>=0.92": bool(f1 >= 0.90 and rc >= 0.92)}}
    meta = {"class_names": CLASSES, "img_size": a.img, "input": "raw RGB 0..255 (rescaling inside model)",
            "fire_threshold": tau, "seed": a.seed, "weights": a.weights, "tensorflow": tf.__version__}
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    model.save(os.path.join(a.out, "fire_cnn.keras"))
    json.dump(meta, open(os.path.join(a.out, "fire_cnn_meta.json"), "w"), indent=2)
    json.dump({**metrics, "meta": meta}, open(os.path.join(a.out, "metrics", "cnn_metrics.json"), "w"), indent=2)
    if not quiet:
        print(json.dumps(metrics["fire_vs_rest_test"], indent=2))
        print("3-class macro-F1:", metrics["three_class_test"]["macro_f1"])
    return metrics


if __name__ == "__main__":
    main()
