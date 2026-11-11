"""Early-warning forecaster (WBS 5.3.3 / 5.3.4) - baseline trend model.

Fits a line to the recent fused risk score and projects it forward. This is the
BASELINE; the Phase-5 ML predictor must beat it on early-warning F1 and lead time.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src import config


class RiskForecaster:
    def __init__(self, window_s: float = config.FORECAST_WINDOW_S,
                 horizon_s: float = config.FORECAST_HORIZON_S):
        self.window_s = window_s
        self.horizon_s = horizon_s
        self.buf = deque()

    def update(self, ts: float, score: float):
        """Return (predicted_score, slope_per_second)."""
        self.buf.append((ts, score))
        while self.buf and ts - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        if len(self.buf) < 6 or self.buf[-1][0] - self.buf[0][0] < 2.0:
            return score, 0.0
        t = np.array([p[0] for p in self.buf]) - ts
        s = np.array([p[1] for p in self.buf])
        slope, intercept = np.polyfit(t, s, 1)
        predicted = float(np.clip(intercept + slope * self.horizon_s, 0.0, 1.0))
        return predicted, float(slope)
