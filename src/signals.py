"""Merge several Signals providers (ML state, fire CNN, temperature LSTM) into one."""
from __future__ import annotations

from src.decision_engine import Signals

_MAX_FIELDS = ("gas_risk", "temp_anomaly", "fire_prob", "ml_hazard", "ml_early_prob")


class CombinedSignals:
    def __init__(self, *providers):
        self.providers = [p for p in providers if p is not None]

    def __call__(self, reading) -> Signals:
        out = Signals()
        for p in self.providers:
            s = p(reading)
            for f in _MAX_FIELDS:
                v = getattr(s, f)
                if v is not None:
                    cur = getattr(out, f)
                    setattr(out, f, v if cur is None else max(cur, v))
            if s.people_in_hazard:
                out.people_in_hazard = sorted(set((out.people_in_hazard or []) + list(s.people_in_hazard)))
            if s.ml_early_alarm is not None:
                out.ml_early_alarm = bool(out.ml_early_alarm) or bool(s.ml_early_alarm)
        return out
