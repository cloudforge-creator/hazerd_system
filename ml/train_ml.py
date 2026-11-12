"""Train and evaluate the bootstrap ML models (hazard state + early warning).

    python -m ml.train_ml --runs 700 --seed 42

Splits are by RUN (never by row) and stratified by scenario, so no window of a test run
is ever seen in training. Hyper-parameters are fixed (no tuning on test); the early-warning
threshold is chosen on the validation runs only; test is reported once.
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ml.synth import DT, SCENARIOS, generate_runs
from src import config
from src.decision_engine import DecisionEngine, Level
from src.features import FEATURE_NAMES, WINDOW, features_from_windows
from src.sensor_reader import Reading

HORIZON_STEPS = int(config.FORECAST_HORIZON_S / DT)
LEVEL_IDX = {Level.SAFE: 0, Level.WARNING: 1, Level.CRITICAL: 2}


def split_runs(runs, seed):
    rng = np.random.default_rng(seed)
    assign = [None] * len(runs)
    for sc in SCENARIOS:
        idx = [i for i, r in enumerate(runs) if r["scenario"] == sc]
        rng.shuffle(idx)
        n_tr, n_va = int(round(0.70 * len(idx))), int(round(0.15 * len(idx)))
        for k, i in enumerate(idx):
            assign[i] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    return {s: [r for r, a in zip(runs, assign) if a == s] for s in ("train", "val", "test")}


def run_features(run):
    sw = np.lib.stride_tricks.sliding_window_view
    X = features_from_windows(sw(run["gas"], WINDOW), sw(run["temp"], WINDOW),
                              sw(run["hum"], WINDOW), sw(run["flame"], WINDOW), DT)
    steps = len(run["state"])
    end = np.arange(WINDOW - 1, steps)
    state = run["state"]
    crit = (state == 2).astype(int)
    cs = np.concatenate([[0], np.cumsum(crit)])
    valid = end + HORIZON_STEPS <= steps - 1
    fut = np.zeros(len(end), dtype=bool)
    j = end[valid]
    fut[valid] = (cs[j + HORIZON_STEPS + 1] - cs[j + 1]) > 0
    y_early = fut & (state[end] != 2)
    return X, state[end].astype(int), y_early.astype(int), valid, end


def stack(runs):
    parts = [run_features(r) for r in runs]
    X = np.vstack([p[0] for p in parts])
    ys = np.concatenate([p[1] for p in parts])
    ye = np.concatenate([p[2] for p in parts])
    valid = np.concatenate([p[3] for p in parts])
    return X, ys, ye, valid


def prf(y, p, labels):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {str(l): {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4)}
            for l, a, b, c in zip(labels, pr, rc, f1)}


def state_report(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            "per_class": prf(y, p, [0, 1, 2]),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def baseline_trace(run):
    """Run the rule-based DecisionEngine over a run: per-step level index and early flag."""
    eng, levels, early = DecisionEngine(), [], []
    for i in range(len(run["state"])):
        r = Reading(timestamp=i * DT, gas_raw=int(run["gas"][i]), temperature=float(run["temp"][i]),
                    humidity=float(run["hum"][i]), flame=int(run["flame"][i]), source="synth")
        d = eng.decide(r)
        levels.append(LEVEL_IDX[d.level])
        early.append(d.early_warning)
    return np.array(levels), np.array(early)


def first_alarm(flags):
    idx = np.flatnonzero(flags)
    return int(idx[0]) if len(idx) else None


def consecutive(flags, n):
    ok = np.convolve(flags.astype(int), np.ones(n, dtype=int), mode="full")[: len(flags)] >= n
    return ok


def event_report(runs, alarm_steps):
    """alarm_steps[i] = first alarm step of run i (or None)."""
    det, leads, missed, fa, n_crit, n_clean = 0, [], 0, 0, 0, 0
    by_sc = {}
    for run, a in zip(runs, alarm_steps):
        crit = first_alarm(run["state"] == 2)
        sc = by_sc.setdefault(run["scenario"], {"critical_runs": 0, "detected": 0,
                                                "clean_runs": 0, "false_alarms": 0})
        if crit is None:
            n_clean += 1
            sc["clean_runs"] += 1
            if a is not None:
                fa += 1
                sc["false_alarms"] += 1
        else:
            n_crit += 1
            sc["critical_runs"] += 1
            if a is not None and a < crit:
                det += 1
                sc["detected"] += 1
                leads.append((crit - a) * DT)
            else:
                missed += 1
    return {"runs_with_critical": n_crit, "detected_before_critical": det,
            "detection_rate": round(det / max(n_crit, 1), 4),
            "median_lead_s": round(float(np.median(leads)), 2) if leads else None,
            "p10_lead_s": round(float(np.percentile(leads, 10)), 2) if leads else None,
            "clean_runs": n_clean, "false_alarm_runs": int(fa),
            "false_alarm_run_rate": round(fa / max(n_clean, 1), 4),
            "by_scenario": {k: v for k, v in by_sc.items() if v["critical_runs"] or v["false_alarms"]}}


def main(runs_n=700, seed=42, out="models", quiet=False):
    log = (lambda *a: None) if quiet else print
    runs = generate_runs(runs_n, seed)
    parts = split_runs(runs, seed)
    sizes = {k: len(v) for k, v in parts.items()}
    log("runs per split:", sizes)
    Xtr, ys_tr, ye_tr, vtr = stack(parts["train"])
    Xva, ys_va, ye_va, vva = stack(parts["val"])
    Xte, ys_te, ye_te, vte = stack(parts["test"])

    # The weighted runtime ensemble's RF slot is a real Random Forest model.
    state_model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    state_model.fit(Xtr, ys_tr)

    early_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                 class_weight="balanced", random_state=seed)
    early_model.fit(Xtr[vtr], ye_tr[vtr])

    # Select the early-warning threshold on VALIDATION RUNS using event-level
    # C2 objectives, not window-level F1. A threshold that fires repeatedly in
    # one clean run is costly even if its window F1 looks good.
    validation_candidates = []
    for cand in np.arange(0.10, 0.951, 0.025):
        alarms = []
        for run in parts["val"]:
            X, _, _, _, end = run_features(run)
            flags = early_model.predict_proba(X)[:, 1] >= cand
            a = first_alarm(consecutive(flags, config.ML_EARLY_CONSEC))
            alarms.append(None if a is None else int(end[a]))
        report = event_report(parts["val"], alarms)
        feasible = (report["detection_rate"] >= 0.90
                    and report["false_alarm_run_rate"] <= 0.10)
        shortfall = max(0.0, 0.90 - report["detection_rate"]) + max(
            0.0, report["false_alarm_run_rate"] - 0.10
        )
        key = (int(feasible), -shortfall, report["detection_rate"],
               -report["false_alarm_run_rate"],
               report["median_lead_s"] if report["median_lead_s"] is not None else -1.0)
        validation_candidates.append((key, float(round(cand, 3)), report))
    _, tau, selected_validation = max(validation_candidates, key=lambda item: item[0])

    # ---- test: window level ----
    metrics = {"splits": sizes,
               "early_threshold_selection_validation": {
                   "threshold": tau,
                   "detection_target_met": selected_validation["detection_rate"] >= 0.90,
                   "false_alarm_target_met": selected_validation["false_alarm_run_rate"] <= 0.10,
                   "event_report": selected_validation,
               }}
    metrics["ml_state_test"] = state_report(ys_te, state_model.predict(Xte))

    base_levels = [baseline_trace(r) for r in parts["test"]]
    base_state_pred = np.concatenate([lv[WINDOW - 1:] for lv, _ in base_levels])
    metrics["rule_engine_state_test"] = state_report(ys_te, base_state_pred)

    p_te = early_model.predict_proba(Xte[vte])[:, 1]
    yh = ye_te[vte]
    pr, rc, f1, _ = precision_recall_fscore_support(yh, p_te >= tau, average="binary", zero_division=0)
    metrics["ml_early_window_test"] = {
        "threshold": tau, "precision": round(float(pr), 4), "recall": round(float(rc), 4),
        "f1": round(float(f1), 4), "accuracy": round(float(accuracy_score(yh, p_te >= tau)), 4),
        "positive_rate": round(float(yh.mean()), 4)}

    # ---- test: event level (C2) ----
    ml_alarms = []
    for run in parts["test"]:
        X, _, _, _, end = run_features(run)
        p = early_model.predict_proba(X)[:, 1] >= tau
        a = first_alarm(consecutive(p, config.ML_EARLY_CONSEC))
        ml_alarms.append(None if a is None else int(end[a]))
    base_alarms = [first_alarm(e) for _, e in base_levels]
    metrics["event_level_test"] = {
        "ml_early_warning": event_report(parts["test"], ml_alarms),
        "trend_baseline": event_report(parts["test"], base_alarms)}

    meta = {"seed": seed, "runs": runs_n, "synthetic": True, "sklearn": sklearn.__version__,
            "python": platform.python_version(), "window": WINDOW, "horizon_s": config.FORECAST_HORIZON_S}
    metrics["meta"] = meta
    os.makedirs(os.path.join(out, "metrics"), exist_ok=True)
    joblib.dump({"model": state_model, "feature_names": FEATURE_NAMES, "meta": meta},
                os.path.join(out, "hazard_state.joblib"))
    joblib.dump({"model": early_model, "feature_names": FEATURE_NAMES, "threshold": tau, "meta": meta},
                os.path.join(out, "early_warning.joblib"))
    with open(os.path.join(out, "metrics", "ml_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if not quiet:
        ml, rb = metrics["ml_state_test"], metrics["rule_engine_state_test"]
        print(f"\nSTATE (test)   ML acc={ml['accuracy']} macroF1={ml['macro_f1']} | "
              f"rule engine acc={rb['accuracy']} macroF1={rb['macro_f1']}")
        print("EARLY window   ", metrics["ml_early_window_test"])
        print("EARLY ml event ", metrics["event_level_test"]["ml_early_warning"])
        print("EARLY baseline ", metrics["event_level_test"]["trend_baseline"])
        print("\nNOTE: synthetic data. Retrain on real logged data after hardware arrives.")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    main(a.runs, a.seed, a.out)
