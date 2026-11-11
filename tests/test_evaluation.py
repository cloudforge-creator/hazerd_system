import json
import os

import numpy as np
import pytest

from evaluation import suite
from evaluation.suite import ArrayReader, FaultyReader, c1_detection, c2_early_warning, c4_response
from ml.synth import generate_run
from ml.train_ml import main as train_main


def trace(state, levels, early=None, sent=(), executed=None, scenario="gas_leak"):
    n = len(state)
    return {"state": np.array(state), "levels": np.array(levels),
            "early": np.array(early if early is not None else [False] * n), "scenario": scenario,
            "sent": list(sent), "executed": executed or [[] for _ in range(n)], "notifications": 0}


def test_array_reader_records_command_step():
    run = generate_run(np.random.default_rng(1), "normal", steps=10)
    rd = ArrayReader(run)
    rd.read(); rd.read()
    rd.send("RELAY:0")
    assert rd.sent == [(1, "RELAY:0")]


def test_faulty_reader_forced_dropout_and_validity():
    run = generate_run(np.random.default_rng(1), "normal", steps=30)
    fr = FaultyReader(ArrayReader(run), seed=1, p_drop=0, p_nan=0, p_range=0, forced=range(5, 8))
    reads = [fr.read() for _ in range(10)]
    assert [not r.valid for r in reads] == [False] * 5 + [True] * 3 + [False] * 2
    assert reads[5].faults == ["link:no_data"]


def test_c1_perfect_and_late_detection():
    st = [0, 0, 1, 1, 2, 2, 2, 2]
    perfect = c1_detection([trace(st, st)])
    assert perfect["macro_f1"] == 1.0 and perfect["critical_event_detection_rate"] == 1.0
    late = c1_detection([trace(st, [0, 0, 0, 1, 1, 1, 2, 2])])
    assert late["critical_event_detection_rate"] == 1.0
    assert late["median_detection_delay_s(negative=earlier than truth)"] == 1.0   # 2 steps * 0.5 s
    missed = c1_detection([trace(st, [0] * 8)])
    assert missed["critical_event_detection_rate"] == 0.0


def test_c1_counts_false_critical_on_clean_runs():
    clean = trace([0] * 6, [0, 0, 2, 2, 2, 2], scenario="normal")
    assert c1_detection([clean])["false_critical_rate_on_clean_runs"] == 1.0


def test_c2_uses_first_early_flag_before_truth():
    st = [0] * 10 + [2] * 5
    good = trace(st, st, early=[False] * 4 + [True] * 6 + [False] * 5)
    late = trace(st, st, early=[False] * 12 + [True] * 3)
    r = c2_early_warning([good, late])
    assert r["runs_with_critical"] == 2 and r["detected_before_critical"] == 1 and r["median_lead_s"] == 3.0


def test_c4_shutdown_timing_and_false_shutdowns():
    st = [0] * 4 + [2] * 20
    fast = trace(st, st, sent=[(5, "BUZZER:1"), (5, "RELAY:0")], executed=[[]] * 5 + [["NOTIFY_ADMIN"]] + [[]] * 18)
    slow = trace(st, st, sent=[(22, "RELAY:0")])
    none = trace(st, st)
    clean_bad = trace([0] * 10, [0] * 10, sent=[(3, "RELAY:0")], scenario="normal")
    r = c4_response([fast, slow, none, clean_bad])
    g = r["gas_leak"]
    assert g["hazard_runs"] == 3 and g["shutdown_issued_rate"] == round(2 / 3, 4)
    assert g["shutdown_within_5s_of_true_critical"] == round(1 / 3, 4)
    assert g["notification_rate"] == round(1 / 3, 4)
    assert r["ALL_HAZARDS"]["false_shutdown_rate_on_clean_runs"] == 1.0


def test_report_marks_failures_honestly():
    st = [0] * 4 + [2] * 20
    t = [trace(st, [0] * 24)]
    R = {"seed": 1, "quick": True, "ml_enabled": False, "configs": ["rule_only"],
         "rule_only": {"C1": c1_detection(t), "C2": c2_early_warning(t), "C4": c4_response(t)},
         "C3": {"cycle_budget_ms": 500, "rule_only": {"full_cycle(read+decide+respond+DB)": {"p99_ms": 900.0}}},
         "C5": {"soak": {"projected_heap_growth_mb_per_day_at_2hz": 50.0, "errors": 0},
                "http_load": {"latency_p95_ms": 10.0, "errors": 0},
                "stress": {"fault_storm_normal_operation": {"exceptions": 0}}}}
    md = suite.build_report(R)
    assert "**FAIL**" in md and "**PASS**" in md and "RULE-BASED ONLY" in md and "synthetic" in md.lower()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("models"))
    train_main(runs_n=150, seed=3, out=out, quiet=True)
    return out


def test_quick_suite_end_to_end(models, tmp_path):
    out = str(tmp_path / "res")
    R = suite.run_all(quick=True, out_dir=out, seed=5, model_dir=models, progress=lambda *_: None)
    assert R["ml_enabled"] and R["configs"] == ["rule_only", "rule_plus_ml"]
    assert os.path.exists(os.path.join(out, "REPORT.md")) and os.path.exists(os.path.join(out, "results.json"))
    json.load(open(os.path.join(out, "results.json")))
    assert R["C5"]["soak"]["errors"] == 0 and R["C5"]["http_load"]["errors"] == 0
    assert R["C5"]["stress"]["fault_storm_normal_operation"]["exceptions"] == 0
    assert R["C5"]["stress"]["forced_10_cycle_dropout"]["warning_raised_after_cycles"] is not None
    assert R["C3"]["rule_only"]["full_cycle(read+decide+respond+DB)"]["p99_ms"] < 500
    assert "| Measure | Result | Target | Verdict |" in open(os.path.join(out, "REPORT.md"), encoding="utf-8").read()
