"""Window features shared by training and runtime so they can never drift apart.

A window is the last WINDOW readings (30 s at 0.5 s per cycle).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src import config

WINDOW = 60
SHORT = 20

FEATURE_NAMES = [
    "gas_last", "gas_mean20", "gas_std20", "gas_slope20", "gas_slope60",
    "gas_max60", "gas_min60", "gas_range20", "gas_accel", "gas_dev60",
    "temp_last", "temp_slope20", "temp_slope60",
    "hum_last", "hum_slope20", "flame_last", "flame_mean20",
]


def _slope(Y: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares slope per second for each row of Y (n, w)."""
    w = Y.shape[1]
    x = (np.arange(w) - (w - 1) / 2.0) * dt
    return (Y @ x) / (x @ x)


def features_from_windows(G, T, H, F, dt: float = config.CYCLE_SECONDS) -> np.ndarray:
    """G,T,H,F: arrays (n, WINDOW) of gas, temperature, humidity, flame."""
    G, T, H, F = (np.asarray(a, dtype=float) for a in (G, T, H, F))
    g20, t20, h20, f20 = (a[:, -SHORT:] for a in (G, T, H, F))
    gs20, gs60 = _slope(g20, dt), _slope(G, dt)
    cols = [
        G[:, -1], g20.mean(1), g20.std(1), gs20, gs60,
        G.max(1), G.min(1), g20.max(1) - g20.min(1), gs20 - gs60, G[:, -1] - G.mean(1),
        T[:, -1], _slope(t20, dt), _slope(T, dt),
        H[:, -1], _slope(h20, dt), F[:, -1], f20.mean(1),
    ]
    return np.column_stack(cols)


class FeatureBuffer:
    """Rolling buffer of valid readings for runtime inference."""

    def __init__(self, window: int = WINDOW):
        self.window = window
        self.g, self.t, self.h, self.f = (deque(maxlen=window) for _ in range(4))

    def push(self, reading) -> None:
        self.g.append(reading.gas_raw)
        self.t.append(reading.temperature)
        self.h.append(reading.humidity)
        self.f.append(reading.flame)

    def clear(self) -> None:
        for d in (self.g, self.t, self.h, self.f):
            d.clear()

    @property
    def ready(self) -> bool:
        return len(self.g) == self.window

    def features(self) -> np.ndarray:
        return features_from_windows([list(self.g)], [list(self.t)],
                                     [list(self.h)], [list(self.f)])
