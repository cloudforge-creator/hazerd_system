"""Runtime temperature-anomaly signal from an LSTM next-value predictor (WBS 5.2.2).

Only enabled for a model trained on THIS DEVICE's own logged data at the runtime sampling
interval. A model trained on NAB (5-minute data) validates the method but must not run on a
0.5 s DHT22 stream, so it is reported as 'not used at runtime'.
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from src import config
from src.decision_engine import Signals


class TempAnomaly:
    def __init__(self, model_dir: str = config.MODEL_DIR, model=None, meta=None):
        self.model, self.meta = model, meta
        self.status = "temperature LSTM off (no device-trained model)"
        if model is None:
            self._load(model_dir)
        if self.model is not None:
            w, k = int(self.meta["window"]), int(self.meta["smooth"])
            self.temps, self.errs = deque(maxlen=w), deque(maxlen=k)
            self.status = "temperature LSTM loaded (device-trained)"

    def _load(self, model_dir: str) -> None:
        mp = os.path.join(model_dir, "temp_lstm.keras")
        jp = os.path.join(model_dir, "temp_lstm_meta.json")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            return
        meta = json.load(open(jp))
        if meta.get("source") != "device" or abs(meta.get("interval_s", 0) - config.CYCLE_SECONDS) > 0.25:
            self.status = (f"temperature LSTM present but trained on '{meta.get('source')}' at "
                           f"{meta.get('interval_s')} s - not used at runtime (retrain on device logs)")
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(mp, compile=False)
            self.meta = meta
        except Exception as exc:
            self.model = None
            self.status = f"temperature LSTM failed to load ({exc.__class__.__name__})"

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def _predict(self, window_scaled: np.ndarray) -> float:
        return float(np.asarray(self.model(window_scaled.reshape(1, -1, 1))).reshape(-1)[0])

    def __call__(self, reading) -> Signals:
        if not self.enabled:
            return Signals()
        if not reading.valid or reading.temperature is None:
            self.temps.clear()
            self.errs.clear()
            return Signals()
        x = (reading.temperature - self.meta["mean"]) / self.meta["std"]
        out = Signals()
        if len(self.temps) == self.temps.maxlen:
            err = abs(self._predict(np.array(self.temps)) - x)
            self.errs.append(err)
            if len(self.errs) == self.errs.maxlen:
                ratio = float(np.mean(self.errs)) / self.meta["threshold"]
                out = Signals(temp_anomaly=float(min(1.0, max(0.0, ratio - 1.0))))
        self.temps.append(x)
        return out
