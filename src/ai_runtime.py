"""Runtime bridge: loads trained ML models and turns readings into Signals.

If the model files are missing or incompatible, ML is disabled and the system runs
rule-based only. It says so explicitly (`status`) - fallback is never silent.
"""
from __future__ import annotations

import os

from src import config
from src.decision_engine import Signals
from src.features import FEATURE_NAMES, FeatureBuffer


class MLSignals:
    def __init__(self, model_dir: str = config.MODEL_DIR):
        self.buffer = FeatureBuffer()
        self.state_model = self.early_model = None
        self.threshold = 0.5
        self._consec = 0
        self.status = "rule-based only (no ML models found)"
        self._load(model_dir)

    def _load(self, model_dir: str) -> None:
        sp = os.path.join(model_dir, "hazard_state.joblib")
        ep = os.path.join(model_dir, "early_warning.joblib")
        if not (os.path.exists(sp) and os.path.exists(ep)):
            return
        try:
            import joblib
            s, e = joblib.load(sp), joblib.load(ep)
            if s["feature_names"] != FEATURE_NAMES or e["feature_names"] != FEATURE_NAMES:
                self.status = "rule-based only (ML feature mismatch - retrain)"
                return
            self.state_model, self.early_model = s["model"], e["model"]
            self.threshold = float(e["threshold"])
            self.status = "ML models loaded"
        except Exception as exc:  # corrupt file, sklearn version mismatch, ...
            self.status = f"rule-based only (ML load failed: {exc.__class__.__name__})"

    @property
    def enabled(self) -> bool:
        return self.state_model is not None

    def __call__(self, reading) -> Signals:
        if not self.enabled:
            return Signals()
        if not reading.valid:            # never feed bad data to the models
            self.buffer.clear()
            self._consec = 0
            return Signals()
        self.buffer.push(reading)
        if not self.buffer.ready:        # 30 s warm-up
            return Signals()
        X = self.buffer.features()
        proba = self.state_model.predict_proba(X)[0]
        hazard = sum(p * config.ML_LEVEL_SCORES[int(c)]
                     for p, c in zip(proba, self.state_model.classes_))
        p_early = float(self.early_model.predict_proba(X)[0, 1])
        self._consec = self._consec + 1 if p_early >= self.threshold else 0
        return Signals(ml_hazard=float(hazard), ml_early_prob=p_early,
                       ml_early_alarm=self._consec >= config.ML_EARLY_CONSEC)
