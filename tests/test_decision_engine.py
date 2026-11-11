import pytest

from src import config
from src.decision_engine import DecisionEngine, Level, Signals
from src.simulator import SimulatedSensorReader
from tests.helpers import reading


def run_sim(scenario, seconds=60, dt=0.5):
    sim, eng = SimulatedSensorReader(scenario, dt=dt), DecisionEngine()
    return [eng.decide(sim.read()) for _ in range(int(seconds / dt))]


def test_normal_stays_safe_with_no_actions_or_false_early_warning():
    ds = run_sim("normal", 120)
    assert all(d.level == Level.SAFE for d in ds)
    assert all(not d.actions and not d.early_warning for d in ds)


def test_gas_leak_gives_early_warning_then_warning_then_critical():
    ds = run_sim("gas_leak", 60)
    first = lambda pred: next(i for i, d in enumerate(ds) if pred(d))
    i_early = first(lambda d: d.early_warning)
    i_warn = first(lambda d: d.level == Level.WARNING)
    i_crit = first(lambda d: d.level == Level.CRITICAL)
    assert i_early < i_warn < i_crit
    assert "RELAY_SHUTDOWN" in ds[i_crit].actions


def test_overheat_escalates_to_critical():
    ds = run_sim("overheat", 90)
    levels = [d.level for d in ds]
    assert Level.WARNING in levels and levels[-1] == Level.CRITICAL
    assert levels.index(Level.WARNING) < levels.index(Level.CRITICAL)


def test_fire_scenario_critical_when_flame_appears():
    ds = run_sim("fire", 40, dt=0.5)
    i_crit = next(i for i, d in enumerate(ds) if d.level == Level.CRITICAL)
    assert i_crit <= 41  # flame sensor fires at t=20s (step 40)


def test_flame_sensor_is_immediate_critical():
    d = DecisionEngine().decide(reading(0, flame=1))
    assert d.level == Level.CRITICAL and d.score == 1.0
    assert "flame_sensor_override" in d.reasons


def test_cnn_alone_is_limited_to_warning_until_camera_validation():
    d = DecisionEngine().decide(reading(0), Signals(fire_prob=0.95))
    assert d.level == Level.WARNING
    assert d.score == config.CNN_SOLO_MAX_SCORE
    assert "cnn_fire_unconfirmed_warning" in d.reasons
    assert "RELAY_SHUTDOWN" not in d.actions


def test_cnn_critical_override_requires_independent_sensor_corroboration():
    d = DecisionEngine().decide(reading(0, gas=400), Signals(fire_prob=0.95))
    assert d.level == Level.CRITICAL
    assert "cnn_fire_corroborated_override" in d.reasons
    assert "RELAY_SHUTDOWN" in d.actions


def test_ml_state_model_alone_cannot_trigger_critical_shutdown():
    d = DecisionEngine().decide(reading(0, gas=180), Signals(ml_hazard=0.85))
    assert d.level == Level.WARNING
    assert d.ml_score == config.ML_SOLO_MAX_SCORE
    assert "ml_critical_unconfirmed_warning" in d.reasons
    assert "RELAY_SHUTDOWN" not in d.actions


def test_ml_state_model_preserves_physical_critical_detection():
    d = DecisionEngine().decide(reading(0, gas=700), Signals(ml_hazard=0.0))
    assert d.level == Level.CRITICAL


def test_ai_can_raise_but_never_lower_a_rule_alarm():
    eng = DecisionEngine()
    d = eng.decide(reading(0, gas=700), Signals(gas_risk=0.0))
    assert d.level == Level.CRITICAL
    d2 = DecisionEngine().decide(reading(0, gas=180), Signals(gas_risk=0.9))
    assert d2.level == Level.WARNING
    assert "RELAY_SHUTDOWN" not in d2.actions


def test_corroboration_raises_score():
    solo = DecisionEngine().decide(reading(0, gas=450))          # gas 0.5 only
    both = DecisionEngine().decide(reading(0, gas=450, temp=55.0))
    assert both.score > solo.score


def test_hysteresis_requires_calm_cycles_to_deescalate():
    eng = DecisionEngine()
    eng.decide(reading(0, gas=700))
    levels = [eng.decide(reading(1 + i, gas=180)).level
              for i in range(config.CLEAR_CYCLES + 1)]
    assert levels[0] == Level.CRITICAL
    assert levels[config.CLEAR_CYCLES - 2] == Level.CRITICAL
    assert levels[-1] == Level.SAFE


def test_short_fault_does_not_alarm_but_is_not_called_normal():
    eng = DecisionEngine()
    d = None
    for i in range(config.FAULT_CYCLES_LIMIT - 1):
        d = eng.decide(reading(i, faults=["gas_raw:missing"], gas=None))
    assert d.degraded and d.level == Level.SAFE


def test_persistent_fault_raises_warning():
    eng = DecisionEngine()
    d = None
    for i in range(config.FAULT_CYCLES_LIMIT):
        d = eng.decide(reading(i, faults=["link:no_data"], gas=None, temp=None, hum=None, flame=None))
    assert d.level == Level.WARNING and "persistent_sensor_fault" in d.reasons


def test_usable_sensors_still_protect_when_another_is_faulty():
    d = DecisionEngine().decide(reading(0, gas=700, temp=None, faults=["temperature:missing"]))
    assert d.level == Level.CRITICAL and d.degraded


def test_cnn_guard_cannot_be_bypassed_by_lowering_the_critical_threshold(monkeypatch):
    monkeypatch.setattr(config, "CRITICAL_THRESHOLD", 0.60)
    d = DecisionEngine().decide(reading(0), Signals(fire_prob=0.95))
    assert d.level == Level.WARNING and "RELAY_SHUTDOWN" not in d.actions
    assert d.score < config.CRITICAL_THRESHOLD


def test_cnn_guard_cannot_be_weakened_by_lowering_the_corroboration_setting(monkeypatch):
    monkeypatch.setattr(config, "CORROBORATION_MIN", 0.05)
    d = DecisionEngine().decide(reading(0, gas=320), Signals(fire_prob=0.95))     # gas barely above the 300 baseline
    assert d.level != Level.CRITICAL and "RELAY_SHUTDOWN" not in d.actions


def test_cnn_critical_still_works_with_a_real_gas_rise_even_if_thresholds_are_edited(monkeypatch):
    monkeypatch.setattr(config, "CRITICAL_THRESHOLD", 0.60)
    d = DecisionEngine().decide(reading(0, gas=420), Signals(fire_prob=0.95))
    assert d.level == Level.CRITICAL and "cnn_fire_corroborated_override" in d.reasons


def test_flame_sensor_override_is_unaffected_by_the_cnn_guard():
    d = DecisionEngine().decide(reading(0, flame=1), Signals(fire_prob=0.0))
    assert d.level == Level.CRITICAL and "flame_sensor_override" in d.reasons


def test_model_ensemble_weights_are_cnn_rf_lstm_50_25_25():
    assert config.MODEL_WEIGHTS == {"cnn": 0.50, "rf": 0.25, "lstm": 0.25}
    assert sum(config.MODEL_WEIGHTS.values()) == 1.0


def test_weighted_model_ensemble_uses_available_scores_and_normalizes():
    d = DecisionEngine().decide(
        reading(0, gas=700),
        Signals(fire_prob=0.8, ml_hazard=0.4, temp_anomaly=0.2),
    )
    assert d.ml_score == pytest.approx(0.55)


def test_weighted_model_ensemble_never_causes_critical_by_itself():
    d = DecisionEngine().decide(
        reading(0),
        Signals(fire_prob=1.0, ml_hazard=1.0, temp_anomaly=1.0),
    )
    assert d.level == Level.WARNING
    assert d.ml_score < config.CRITICAL_THRESHOLD
    assert "RELAY_SHUTDOWN" not in d.actions


def test_weighted_model_ensemble_renormalizes_when_models_are_missing():
    d = DecisionEngine().decide(reading(0, gas=700), Signals(ml_hazard=0.4))
    assert d.ml_score == pytest.approx(0.4)


def test_lstm_cannot_physically_corroborate_cnn_or_trigger_shutdown():
    d = DecisionEngine().decide(
        reading(0, gas=180, temp=25.0),
        Signals(fire_prob=0.99, temp_anomaly=1.0),
    )
    assert d.level == Level.WARNING
    assert "cnn_fire_unconfirmed_warning" in d.reasons
    assert "cnn_fire_corroborated_override" not in d.reasons
    assert "RELAY_SHUTDOWN" not in d.actions
