"""Operating-point, calibration and in-domain evaluation of the fire CNN.

    python -m ml.eval_cnn_operating_points --manifest data/fire_manifest.csv
    python -m ml.eval_cnn_operating_points --manifest data/fire_manifest.csv --webcam-neg data/webcam/negatives --webcam-pos data/webcam/fire

Why this exists: train_cnn.py reports one threshold (chosen on validation). The decision engine, however, uses
the fire probability at its own points (WARNING >= 0.40, CRITICAL >= 0.70, forced CRITICAL >= 0.80). This script
reports precision / recall / false-positive rate at EACH of those points, with 95% confidence intervals, plus
calibration (are the probabilities trustworthy as scores?), and optionally measures false alarms on YOUR webcam.

Everything runs the model in inference mode (training=False). The test split is read-only evidence: do not
tune thresholds on it. Choose them on the validation split (--split val) and report them on test.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(c - h, 4), round(c + h, 4))


def point_metrics(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = p >= tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not math.isnan(prec + rec) else float("nan")
    return {"threshold": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "recall_ci95": wilson(tp, tp + fn), "false_positive_rate": round(fp / (fp + tn), 4),
            "false_positive_rate_ci95": wilson(fp, fp + tn)}


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1); rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean()); ece += m.mean() * abs(conf - acc)
        rows.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()), "mean_predicted": round(conf, 3), "observed_fire_rate": round(acc, 3)})
    return {"ece": round(float(ece), 4), "brier": round(float(np.mean((p - y) ** 2)), 4), "reliability": rows}


def runs(mask: np.ndarray) -> int:
    """Number of contiguous True runs (alert episodes)."""
    return int(((mask[1:] & ~mask[:-1]).sum()) + (1 if len(mask) and mask[0] else 0))


def load_runtime(model_dir: str):
    from src.fire_cnn import FireCNN
    cnn = FireCNN(model_dir)
    if not cnn.enabled:
        raise SystemExit(cnn.status)
    return cnn


def folder_probs(cnn, folder: str) -> np.ndarray:
    import cv2
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    if not files:
        raise SystemExit(f"no images in {folder}")
    out = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            out.append(cnn.predict(img)[cnn.class_names.index("fire")])
    return np.asarray(out)


def smoothed(p: np.ndarray, k: int = 3) -> np.ndarray:
    return np.array([p[max(0, i - k + 1): i + 1].mean() for i in range(len(p))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models"); ap.add_argument("--manifest", default="data/fire_manifest.csv")
    ap.add_argument("--split", default="test"); ap.add_argument("--thresholds", default="0.38,0.40,0.70,0.80")
    ap.add_argument("--webcam-neg", default=None, help="folder of consecutive frames with NO fire, in time order")
    ap.add_argument("--webcam-pos", default=None, help="folder of consecutive frames with real (safe, small) fire")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second the system runs the CNN at (0.5 s cycle = 2)")
    ap.add_argument("--smooth", type=int, default=3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models/metrics/cnn_operating_points.json"); ap.add_argument("--csv", default="models/metrics/cnn_predictions.csv")
    a = ap.parse_args(argv)
    taus = [float(t) for t in a.thresholds.split(",")]
    cnn = load_runtime(a.model_dir); names = cnn.class_names; fire = names.index("fire"); size = int(cnn.meta["img_size"])
    report = {"model_dir": a.model_dir, "split": a.split, "thresholds": taus, "tensorflow_note": "inference mode (training=False)"}

    if a.manifest and os.path.exists(a.manifest):
        import pandas as pd, tensorflow as tf
        df = pd.read_csv(a.manifest); df = df[df["split"] == a.split].reset_index(drop=True)
        def load(path):
            im = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False); im.set_shape([None, None, 3])
            return tf.cast(tf.image.resize(im, (size, size)), tf.float32)
        ds = tf.data.Dataset.from_tensor_slices(df["path"].to_numpy()).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(1)
        P = cnn.model.predict(ds, verbose=0)
        y = (df["label"] == "fire").to_numpy().astype(int); p = P[:, fire]
        from sklearn.metrics import average_precision_score, roc_auc_score
        report["fire_vs_rest"] = {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": round(float(roc_auc_score(y, p)), 4),
                                  "average_precision": round(float(average_precision_score(y, p)), 4),
                                  "operating_points": [point_metrics(y, p, t) for t in taus], "calibration": calibration(y, p)}
        pd.DataFrame({"path": df["path"], "label": df["label"], **{f"p_{n}": P[:, i] for i, n in enumerate(names)}}).to_csv(a.csv, index=False)
        # parity of the runtime (cv2) path with the training/evaluation (tf) path on a sample of real images
        import cv2
        idx = np.linspace(0, len(df) - 1, min(200, len(df))).astype(int); diffs = []
        for i in idx:
            img = cv2.imread(df["path"][i])
            if img is not None:
                diffs.append(abs(float(cnn.predict(img)[fire]) - float(p[i])))
        report["runtime_parity"] = {"n": len(diffs), "mean_abs_diff_p_fire": round(float(np.mean(diffs)), 4), "max_abs_diff_p_fire": round(float(np.max(diffs)), 4)}

    if a.webcam_neg:
        pn = folder_probs(cnn, a.webcam_neg); sm = smoothed(pn, a.smooth); hours = len(pn) / a.fps / 3600.0
        report["webcam_negatives"] = {"frames": int(len(pn)), "hours": round(hours, 3), "by_threshold": [
            {"threshold": t, "frames_above": int((sm >= t).sum()), "alert_episodes": runs(sm >= t),
             "episodes_per_hour": round(runs(sm >= t) / hours, 2) if hours else None} for t in taus]}
    if a.webcam_pos:
        pp = folder_probs(cnn, a.webcam_pos); sm = smoothed(pp, a.smooth)
        report["webcam_positives"] = {"frames": int(len(pp)), "by_threshold": [
            {"threshold": t, "frame_recall": round(float((sm >= t).mean()), 4), "detected_at_all": bool((sm >= t).any())} for t in taus]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
