import numpy as np
import pytest

from ml.eval_uci_gas import load_batch
from ml.synth import SCENARIOS, generate_run, generate_runs
from ml.train_ml import main as train_main
from src import config
from src.ai_runtime import MLSignals
from src.decision_engine import DecisionEngine, Level, Signals
from src.features import FEATURE_NAMES, WINDOW, FeatureBuffer, features_from_windows
from src.pipeline import HazardSystem
from src.simulator import SimulatedSensorReader
from src.storage import Storage
from tests.helpers import reading


# ---------- features ----------
def test_slope_features_match_known_ramp():
    ramp = np.tile(np.arange(WINDOW) * 0.5 * 10.0, (1, 1))   # 10 units per second
    X = features_from_windows(ramp, ramp * 0 + 25, ramp * 0 + 50, ramp * 0)
    f = dict(zip(FEATURE_NAMES, X[0]))
    assert f["gas_slope20"] == pytest.approx(10.0) and f["gas_slope60"] == pytest.approx(10.0)
    assert f["temp_slope20"] == pytest.approx(0.0, abs=1e-9)


def test_feature_buffer_warmup_and_shape():
    buf = FeatureBuffer()
    for i in range(WINDOW - 1):
        buf.push(reading(i))
    assert not buf.ready
    buf.push(reading(WINDOW))
    assert buf.ready and buf.features().shape == (1, len(FEATURE_NAMES))


# ---------- generator ----------
def test_generator_is_deterministic_and_valid():
    a = generate_runs(20, 7)
    b = generate_runs(20, 7)
    assert all(np.array_equal(x["gas"], y["gas"]) for x, y in zip(a, b))
    for r in a:
        assert r["gas"].min() >= 0 and r["gas"].max() <= 1023
        assert r["temp"].min() >= -10 and r["temp"].max() <= 80


def test_ground_truth_labels_are_sane():
    rng = np.random.default_rng(1)
    assert generate_run(rng, "normal")["state"].max() == 0
    assert generate_run(rng, "spray")["state"].max() == 0        # benign transient
    assert generate_run(rng, "humid")["state"].max() == 0        # humidity is not a hazard
    fire = generate_run(rng, "fire")
    assert fire["state"][-1] == 2 and fire["flame"][-1] in (0, 1)
    with pytest.raises(ValueError):
        generate_run(rng, "nope")


# ---------- training + runtime (one small shared training run) ----------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    metrics = train_main(runs_n=150, seed=3, out=str(out), quiet=True)
    return str(out), metrics


def test_training_writes_artifacts_and_reasonable_metrics(trained):
    out, m = trained
    import os
    assert os.path.exists(os.path.join(out, "hazard_state.joblib"))
    assert os.path.exists(os.path.join(out, "early_warning.joblib"))
    assert m["meta"]["synthetic"] is True
    assert m["ml_state_test"]["accuracy"] > 0.9
    assert m["ml_state_test"]["macro_f1"] > 0.75
    selected = m["early_threshold_selection_validation"]
    assert 0.10 <= selected["threshold"] <= 0.95
    assert "detection_target_met" in selected and "false_alarm_target_met" in selected
    assert selected["event_report"]["clean_runs"] > 0


def test_no_split_leakage_between_runs():
    from ml.train_ml import split_runs
    runs = generate_runs(120, 5)
    parts = split_runs(runs, 5)
    ids = {k: {id(r) for r in v} for k, v in parts.items()}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in parts.values()) == 120
    assert {r["scenario"] for r in parts["test"]} == set(SCENARIOS)


def test_ml_signals_disabled_when_models_missing(tmp_path):
    ml = MLSignals(str(tmp_path))
    assert not ml.enabled and "rule-based only" in ml.status
    assert ml(reading(0)) == Signals()


def test_ml_signals_reject_invalid_readings(trained):
    ml = MLSignals(trained[0])
    assert ml.enabled
    for i in range(WINDOW):
        ml(reading(i))
    assert ml(reading(99, faults=["gas_raw:missing"], gas=None)) == Signals()
    assert not ml.buffer.ready                      # buffer cleared on bad data


def run_system(models, scenario, seconds):
    sim = SimulatedSensorReader(scenario, dt=0.5)
    sent = []
    sim.send = sent.append
    system = HazardSystem(sim, storage=Storage(), signals_provider=MLSignals(models))
    system.start()
    for _ in range(int(seconds / 0.5)):
        system.step()
    return system, sent


def test_end_to_end_normal_with_ml_has_no_false_alarm(trained):
    system, sent = run_system(trained[0], "normal", 150)
    assert system.storage.count("alerts") == 0 and "RELAY:0" not in sent


@pytest.mark.parametrize("scenario,seconds", [("gas_leak", 70), ("fire", 45), ("overheat", 90)])
def test_end_to_end_hazards_still_shut_down_with_ml(trained, scenario, seconds):
    system, sent = run_system(trained[0], scenario, seconds)
    assert system.last_decision.level == Level.CRITICAL and "RELAY:0" in sent


# ---------- engine + ML signals ----------
def test_ml_hazard_alone_is_capped_below_critical():
    d = DecisionEngine().decide(reading(0, gas=180), Signals(ml_hazard=0.85))
    assert d.level == Level.WARNING
    assert d.ml_score == pytest.approx(0.69)
    assert "ml_critical_unconfirmed_warning" in d.reasons
    assert "RELAY_SHUTDOWN" not in d.actions


def test_ml_hazard_does_not_lower_sustained_physical_critical_evidence():
    eng = DecisionEngine()
    decisions = [
        eng.decide(reading(i * config.CYCLE_SECONDS, gas=700), Signals(ml_hazard=0.0))
        for i in range(config.GAS_CRITICAL_CONSEC)
    ]
    assert decisions[-1].level == Level.CRITICAL
    assert "RELAY_SHUTDOWN" in decisions[-1].actions


def test_ml_does_not_count_as_corroboration():
    solo = DecisionEngine().decide(reading(0, gas=450))
    with_ml = DecisionEngine().decide(reading(0, gas=450), Signals(ml_hazard=0.5))
    assert with_ml.score == pytest.approx(solo.score)


def test_ml_early_alarm_triggers_notify_without_shutdown():
    d = DecisionEngine().decide(reading(0), Signals(ml_early_alarm=True))
    assert d.early_warning and "ml_early_warning" in d.reasons
    assert "NOTIFY_ADMIN" in d.actions and "RELAY_SHUTDOWN" not in d.actions


# ---------- UCI parser ----------
def test_uci_libsvm_parser(tmp_path):
    p = tmp_path / "batch1.dat"
    p.write_text("1;10.0 1:5.5 2:1.5 4:2.0\n3;50.0 1:6.0 3:7.0\n")
    X, y = load_batch(str(p))
    assert X.shape == (2, 4) and list(y) == [1, 3]
    assert X[0, 3] == 2.0 and X[1, 2] == 7.0 and X[1, 1] == 0.0
