"""Temperature anomaly detection with an LSTM next-value predictor.

NAB (method validation on REAL data, 4 labeled anomaly windows):
    python -m ml.train_temp_lstm --download-nab
    python -m ml.train_temp_lstm --csv data/raw/temperature/machine_temperature_system_failure.csv \
        --windows data/raw/temperature/combined_windows.json \
        --window-key realKnownCause/machine_temperature_system_failure.csv --source NAB

Device (after hardware: retrain on your own logs; the only model the runtime will use):
    python -m ml.train_temp_lstm --csv data/logs/sensor_log.csv --value-col temperature \
        --interval-s 0.5 --source device

Rules: chronological split, scaler/threshold from NORMAL data only, anomaly windows excluded
from training, threshold chosen on validation, test reported once. Baselines: persistence
and rolling z-score, scored with the same smoothing and threshold procedure.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np

NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master/"


def download_nab(dest="data/raw/temperature"):
    os.makedirs(dest, exist_ok=True)
    for rel, name in (("data/realKnownCause/machine_temperature_system_failure.csv",
                       "machine_temperature_system_failure.csv"),
                      ("labels/combined_windows.json", "combined_windows.json")):
        urllib.request.urlretrieve(NAB_BASE + rel, os.path.join(dest, name))
    print("downloaded NAB to", dest)


def load_series(csv, value_col, time_col, windows_path, window_key):
    import pandas as pd
    df = pd.read_csv(csv)
    if "faults" in df.columns:
        df = df[df["faults"].fillna("") == ""].reset_index(drop=True)
    x = df[value_col].astype(float).to_numpy()
    mask = np.zeros(len(x), dtype=bool)
    interval = None
    if windows_path:
        ts = pd.to_datetime(df[time_col])
        interval = float(np.median(np.diff(ts.values).astype("timedelta64[s]").astype(float)))
        for s, e in json.load(open(windows_path))[window_key]:
            mask |= ((ts >= pd.to_datetime(s)) & (ts <= pd.to_datetime(e))).to_numpy()
    return x, mask, interval


def causal_smooth(e, k):
    c = np.cumsum(np.insert(e, 0, 0.0))
    out = np.empty_like(e)
    for i in range(len(e)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def runs(mask):
    """Contiguous True runs -> list of (start, end) inclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def event_eval(flag, mask, seg, interval_s, gap=12):
    a, b = seg
    wins = [(s, e) for s, e in runs(mask) if e >= a and s <= b]
    detected = sum(bool(flag[max(s, a):min(e, b) + 1].any()) for s, e in wins)
    fa_idx = np.flatnonzero(flag[a:b + 1] & ~mask[a:b + 1]) + a
    fa_segments = int(1 + (np.diff(fa_idx) > gap).sum()) if len(fa_idx) else 0
    days = (b - a + 1) * interval_s / 86400.0
    return {"anomaly_windows": len(wins), "detected": int(detected),
            "false_alarm_segments": fa_segments,
            "false_alarms_per_day": round(fa_segments / max(days, 1e-9), 3)}


def point_eval(flag, mask, seg):
    from sklearn.metrics import precision_recall_fscore_support
    a, b = seg
    if not mask[a:b + 1].any():
        return None
    p, r, f, _ = precision_recall_fscore_support(mask[a:b + 1], flag[a:b + 1], average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


def main(argv=None, quiet=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-nab", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--windows")
    ap.add_argument("--window-key")
    ap.add_argument("--source", default="NAB", choices=["NAB", "device"])
    ap.add_argument("--interval-s", type=float)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--smooth", type=int, default=6)
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split", type=float, nargs=2, default=[0.55, 0.70])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models")
    a = ap.parse_args(argv)
    log = (lambda *x: None) if quiet else print
    if a.download_nab:
        download_nab()
        if not a.csv:
            return None
    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)

    x, mask, interval = load_series(a.csv, a.value_col, a.time_col, a.windows, a.window_key)
    interval_s = a.interval_s or interval
    if interval_s is None:
        raise SystemExit("Provide --interval-s (or --windows with timestamps).")
    n, w = len(x), a.window
    c1, c2 = int(a.split[0] * n), int(a.split[1] * n)
    normal_train = ~mask[:c1]
    mean, std = float(x[:c1][normal_train].mean()), float(x[:c1][normal_train].std())
    xs = (x - mean) / std

    sw = np.lib.stride_tricks.sliding_window_view
    W = sw(xs, w)[:-1].astype(np.float32)          # window k predicts xs[k+w]
    Y = xs[w:].astype(np.float32)
    tgt = np.arange(w, n)
    cm = np.concatenate([[0], np.cumsum(mask)])
    clean = (cm[tgt + 1] - cm[tgt - w]) == 0       # no anomaly in inputs or target
    tr = clean & (tgt < c1)
    va = clean & (tgt >= c1) & (tgt < c2)

    model = tf.keras.Sequential([tf.keras.layers.Input((w, 1)), tf.keras.layers.LSTM(64),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    hist = model.fit(W[tr][..., None], Y[tr], validation_data=(W[va][..., None], Y[va]),
                     epochs=a.epochs, batch_size=128, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    epochs_run = len(hist.history["loss"])
    log(f"trained {epochs_run} epochs, val_loss={min(hist.history['val_loss']):.4f}")

    pred = model.predict(W[..., None], batch_size=512, verbose=0).reshape(-1)
    scores = {"lstm": np.abs(pred - Y)}
    scores["persistence"] = np.abs(xs[w:] - xs[w - 1:-1])
    mu = sw(xs, w)[:-1].mean(1)
    sd = sw(xs, w)[:-1].std(1) + 1e-6
    scores["rolling_z"] = np.abs(xs[w:] - mu) / sd

    seg_val, seg_test = (c1, c2 - 1), (c2, n - 1)
    result = {"n_points": n, "interval_s": interval_s, "epochs_run": epochs_run,
              "splits": {"train": [0, c1], "val": [c1, c2], "test": [c2, n]}}
    thr_lstm = None
    for name, sc in scores.items():
        full = np.zeros(n)
        full[w:] = causal_smooth(sc, a.smooth)
        vm = ~mask[c1:c2]
        thr = float(np.quantile(full[c1:c2][vm], a.quantile))
        if name == "lstm":
            thr_lstm = thr
        flag = full > thr
        flag[:w] = False
        entry = {"threshold": round(thr, 5), "event_level_test": event_eval(flag, mask, seg_test, interval_s)}
        pe = point_eval(flag, mask, seg_test)
        if pe:
            entry["point_level_test"] = pe
        result[name] = entry

    meta = {"source": a.source, "interval_s": interval_s, "window": w, "smooth": a.smooth,
            "quantile": a.quantile, "mean": mean, "std": std, "threshold": thr_lstm,
            "epochs_run": epochs_run, "seed": a.seed, "tensorflow": tf.__version__}
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    model.save(os.path.join(a.out, "temp_lstm.keras"))
    json.dump(meta, open(os.path.join(a.out, "temp_lstm_meta.json"), "w"), indent=2)
    json.dump({**result, "meta": meta}, open(os.path.join(a.out, "metrics", f"temp_lstm_{a.source}.json"), "w"), indent=2)
    if not quiet:
        for k in ("lstm", "persistence", "rolling_z"):
            print(k, json.dumps(result[k]))
        if a.source == "NAB":
            print("NOTE: NAB is 5-minute data with 4 labeled anomalies in total (2 in the test split). "
                  "Treat as method validation; the runtime model must be retrained on device logs.")
    return result


if __name__ == "__main__":
    main()
