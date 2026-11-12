"""M3 decision engine (WBS 6.x): risk scoring, fusion, levels, actions.

Rules of the design
  * Rule-based scores are a safety floor. AI scores can raise, never lower.
  * Physical sensor rules can trigger immediate escalation; the camera CNN alone is
    limited to WARNING until independent gas/temperature/flame evidence corroborates it.
  * Escalation is immediate. De-escalation needs CLEAR_CYCLES calm cycles.
  * Faulty sensors are never read as 'normal'. Usable fields are still used, and
    a persistent fault raises a WARNING.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src import config
from src.predictive import RiskForecaster
from src.sensor_reader import Reading


class Level(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


RANK = {Level.SAFE: 0, Level.WARNING: 1, Level.CRITICAL: 2}


@dataclass
class Signals:
    """Optional AI outputs, each 0..1 (None = model not available)."""
    gas_risk: Optional[float] = None
    temp_anomaly: Optional[float] = None
    fire_prob: Optional[float] = None
    ml_hazard: Optional[float] = None       # expected severity from the ML state model, 0..1
    ml_early_prob: Optional[float] = None   # ML probability of CRITICAL within the horizon
    ml_early_alarm: Optional[bool] = None   # debounced ML early-warning decision
    people_in_hazard: Optional[list] = None  # labels of people inside a hazard zone (camera)


@dataclass
class Decision:
    timestamp: float
    level: Level
    score: float
    gas_score: float
    temp_score: float
    fire_score: float
    predicted_score: float
    predicted_level: Level
    early_warning: bool
    degraded: bool
    reasons: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    ml_score: float = 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def level_for(score: float) -> Level:
    if score >= config.CRITICAL_THRESHOLD:
        return Level.CRITICAL
    if score >= config.WARNING_THRESHOLD:
        return Level.WARNING
    return Level.SAFE


def gas_rule(gas_raw: float) -> float:
    return _clip01((gas_raw - config.GAS_SAFE_MAX) / (config.GAS_CRITICAL - config.GAS_SAFE_MAX))


def temp_rule(temp: float, rate: float) -> float:
    absolute = _clip01((temp - config.TEMP_SAFE_MAX) / (config.TEMP_CRITICAL - config.TEMP_SAFE_MAX))
    rising = _clip01(max(rate, 0.0) / config.TEMP_RATE_CRITICAL)
    return max(absolute, rising)


def _usable(reading: Reading, name: str) -> bool:
    if getattr(reading, name) is None:
        return False
    return not any(f.startswith(name + ":") for f in reading.faults)


def _cnn_solo_cap() -> float:
    """Highest score camera-only evidence may reach. Always strictly below the (live-editable) CRITICAL
    threshold, so lowering that threshold in the dashboard cannot let the CNN alone cause CRITICAL."""
    return min(config.CNN_SOLO_MAX_SCORE, config.CRITICAL_THRESHOLD - 0.01)


def _cnn_corroborated(reading, gas: float, temp: float) -> bool:
    """Independent physical evidence for a CNN-driven CRITICAL. Uses a fixed floor (CNN_CORROBORATION_MIN),
    not the editable CORROBORATION_MIN, and never less than the ordinary corroboration setting."""
    floor = max(config.CNN_CORROBORATION_MIN, config.CORROBORATION_MIN)
    return gas >= floor or temp >= floor or (_usable(reading, "flame") and reading.flame == 1)


class DecisionEngine:
    def __init__(self):
        self.level = Level.SAFE
        self._calm = 0
        self._fault_cycles = 0
        self._trend_early_consec = 0
        self._gas_critical_consec = 0
        self._temps = deque()
        self.forecaster = RiskForecaster()

    # -- helpers ---------------------------------------------------------
    def _temp_rate(self, ts: float, temp: float) -> float:
        self._temps.append((ts, temp))
        while self._temps and ts - self._temps[0][0] > config.RATE_WINDOW_S:
            self._temps.popleft()
        if len(self._temps) < 5 or self._temps[-1][0] - self._temps[0][0] < 2.0:
            return 0.0
        t = np.array([p[0] for p in self._temps]) - ts
        v = np.array([p[1] for p in self._temps])
        return float(np.polyfit(t, v, 1)[0])

    @staticmethod
    def _actions(level: Level, early: bool, fault_alert: bool) -> list:
        if level == Level.CRITICAL:
            return ["LOG_ALERT", "BUZZER_ON", "NOTIFY_ADMIN", "RELAY_SHUTDOWN"]
        if level == Level.WARNING or early or fault_alert:
            return ["LOG_ALERT", "NOTIFY_ADMIN"]
        return []

    # -- main entry ------------------------------------------------------
    def decide(self, reading: Reading, signals: Optional[Signals] = None) -> Decision:
        signals = signals or Signals()
        reasons = []

        degraded = not reading.valid
        self._fault_cycles = self._fault_cycles + 1 if degraded else 0
        fault_alert = self._fault_cycles >= config.FAULT_CYCLES_LIMIT
        if degraded:
            reasons.append("sensor_fault:" + ",".join(reading.faults))

        # sub-scores from usable fields only
        gas = gas_rule(reading.gas_raw) if _usable(reading, "gas_raw") else 0.0
        gas_critical_candidate = _usable(reading, "gas_raw") and gas >= config.CRITICAL_THRESHOLD
        self._gas_critical_consec = (
            self._gas_critical_consec + 1 if gas_critical_candidate else 0
        )
        gas_critical_confirmed = self._gas_critical_consec >= config.GAS_CRITICAL_CONSEC
        # Keep the observed score for diagnostics, but prevent a single gas spike
        # from independently issuing a relay shutdown before persistence is confirmed.
        gas_fusion = gas if gas_critical_confirmed else min(
            gas, config.CRITICAL_THRESHOLD - 0.01
        )
        if gas_critical_candidate and not gas_critical_confirmed:
            reasons.append("gas_critical_pending_confirmation")
        temp = 0.0
        if _usable(reading, "temperature"):
            temp = temp_rule(reading.temperature, self._temp_rate(reading.timestamp, reading.temperature))
        fire = 0.0
        fire_probability = None

        # AI may raise rule scores, but it never lowers rule-based evidence; individual
        # model signals remain subject to the validation and corroboration guards below.
        # A gas reading that has crossed the CRITICAL threshold but has not yet
        # passed the persistence check must not corroborate a CNN shutdown.
        # Sub-critical gas readings remain eligible as physical corroboration.
        physical_gas = (
            gas if not gas_critical_candidate or gas_critical_confirmed
            else min(gas, config.CNN_CORROBORATION_MIN - 0.01)
        )
        physical_temp = temp
        physical_critical = (
            gas_critical_confirmed
            or physical_temp >= config.CRITICAL_THRESHOLD
            or (_usable(reading, "flame") and reading.flame == 1)
        )
        # AI estimates are fused through the weighted ensemble below. Keep these
        # physical-rule channels unchanged so an AI score cannot masquerade as
        # independent sensor corroboration.
        if signals.fire_prob is not None:
            fire_probability = _clip01(signals.fire_prob)
            # The CNN has not yet passed in-domain false-alarm validation. Treat
            # visual evidence as a WARNING-level signal on its own. A CRITICAL
            # override is allowed only when a separate physical sensor corroborates it.
            sensor_corroboration = _cnn_corroborated(reading, physical_gas, physical_temp)
            fire = fire_probability if sensor_corroboration else min(
                fire_probability, _cnn_solo_cap()
            )
            if fire_probability >= config.FIRE_PROB_OVERRIDE and not sensor_corroboration:
                reasons.append("cnn_fire_unconfirmed_warning")

        # The state model reuses the same sensor inputs; it is not an independent
        # safety channel. Its severity cannot cause CRITICAL without a physical
        # sensor rule independently reaching CRITICAL.
        # Weighted CNN/RF/LSTM ensemble. Missing or invalid model outputs are omitted
        # and the remaining weights are renormalized; absent models never count as zero.
        model_values = {
            "cnn": fire_probability,
            "rf": (_clip01(signals.ml_hazard) if signals.ml_hazard is not None else
                   (_clip01(signals.gas_risk) if signals.gas_risk is not None else None)),
            "lstm": _clip01(signals.temp_anomaly) if signals.temp_anomaly is not None else None,
        }
        active_models = [(config.MODEL_WEIGHTS[name], value) for name, value in model_values.items()
                         if value is not None and config.MODEL_WEIGHTS.get(name, 0.0) > 0.0]
        weight_total = sum(weight for weight, _ in active_models)
        model_raw = (sum(weight * value for weight, value in active_models) / weight_total
                     if weight_total > 0 else 0.0)
        ml_cap = min(config.ML_SOLO_MAX_SCORE, config.CRITICAL_THRESHOLD - 0.01)
        # These model outputs are not independent safety channels. Their weighted
        # ensemble may raise WARNING, but cannot cause CRITICAL without physical evidence.
        ml = model_raw if physical_critical else min(model_raw, ml_cap)
        if model_raw >= config.CRITICAL_THRESHOLD and not physical_critical:
            reasons.append("ml_critical_unconfirmed_warning")
        scores = {"gas": gas_fusion, "temp": temp, "fire": fire}
        # ML counts toward the base score but NOT toward corroboration (it re-reads the
        # same physical signals, so it must not double-count as independent evidence).
        cnn_unconfirmed = fire_probability is not None and not sensor_corroboration
        # Camera-only evidence takes no corroboration bonus and stays at its cap, whatever the
        # (live-editable) corroboration setting is; other signals fuse exactly as before.
        fusion_scores = {k: v for k, v in scores.items() if not (cnn_unconfirmed and k == "fire")}
        base = max(max(fusion_scores.values()), ml)
        active = sum(1 for v in fusion_scores.values() if v >= config.CORROBORATION_MIN)
        fused = base + (config.CORROBORATION_BONUS * (active - 1) if active > 1 else 0.0)
        if cnn_unconfirmed:
            fused = max(fused, fire)
        fused = _clip01(fused)
        for name, v in scores.items():
            if v >= config.CORROBORATION_MIN:
                reasons.append(f"{name}_score={v:.2f}")
        if ml >= config.CORROBORATION_MIN:
            reasons.append(f"ml_score={ml:.2f}")

        # immediate overrides
        if _usable(reading, "flame") and reading.flame == 1:
            fused = 1.0
            reasons.append("flame_sensor_override")
        sensor_corroboration = _cnn_corroborated(reading, physical_gas, physical_temp)
        if (fire_probability is not None
                and fire_probability >= config.FIRE_PROB_OVERRIDE
                and sensor_corroboration):
            fused = 1.0
            reasons.append("cnn_fire_corroborated_override")

        # level with hysteresis
        raw_level = level_for(fused)
        if fault_alert and RANK[raw_level] < RANK[Level.WARNING]:
            raw_level = Level.WARNING
            reasons.append("persistent_sensor_fault")
        if RANK[raw_level] >= RANK[self.level]:
            self.level = raw_level
            self._calm = 0
        else:
            self._calm += 1
            if self._calm >= config.CLEAR_CYCLES:
                self.level = raw_level
                self._calm = 0

        # early warning
        predicted, slope = self.forecaster.update(reading.timestamp, fused)
        predicted_level = level_for(predicted)
        trend_candidate = slope >= config.FORECAST_MIN_SLOPE and predicted_level == Level.CRITICAL
        self._trend_early_consec = (self._trend_early_consec + 1 if trend_candidate else 0)
        trend_early = self._trend_early_consec >= config.TREND_EARLY_CONSEC
        ml_early = bool(signals.ml_early_alarm)
        early = (trend_early or ml_early) and self.level != Level.CRITICAL
        if early:
            reasons.append("early_warning")
            if ml_early:
                reasons.append("ml_early_warning")

        actions = self._actions(self.level, early, fault_alert)
        people = list(signals.people_in_hazard or [])
        if people and self.level != Level.SAFE:
            reasons.append("people_in_hazard_zone=" + ", ".join(people))
            actions.append("NOTIFY_EMERGENCY")
        return Decision(
            timestamp=reading.timestamp, level=self.level, score=fused,
            gas_score=gas, temp_score=temp, fire_score=fire,
            predicted_score=predicted, predicted_level=predicted_level,
            early_warning=early, degraded=degraded, reasons=reasons,
            actions=actions,
            ml_score=ml,
        )
