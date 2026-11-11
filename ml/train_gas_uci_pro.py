"""Professional training pipeline for the UCI "Gas Sensor Array Drift" dataset.

    python -m ml.train_gas_uci_pro --data data/raw/gas

Design (time-aware, no leakage across batches)
  * Selection folds  : train on batches 1..k, test on batch k+1 for k+1 in {3, 5, 7}.
                       Model family, preprocessing and hyper-parameters are chosen ONLY here.
  * Final test folds : train on batches 1..k, test on batch k+1 for k+1 in {8, 9, 10}.
                       Touched once, after every choice is frozen.
  * Baseline         : the project's original model (StandardScaler + RandomForest-300).
  * Preprocessing    : "global"    - scaler fitted on the training batches.
                       "per_batch" - each batch standardised by its own statistics (a drift
                                     mitigation; at deployment this needs a short clean-air
                                     baseline window, so it is reported separately).
  * Final deployment model is trained on all 10 batches and saved to models/.
This validates the METHOD on real data. It is not the runtime model (MQ-2 = 1 channel).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.eval_uci_gas import load_batch

GASES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia", 4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
SEL_TEST = [3, 5, 7]
FINAL_TEST = [8, 9, 10]


def families(seed):
    return {
        "RandomForest": lambda **k: RandomForestClassifier(n_estimators=k.get("n", 200), max_features=k.get("mf", "sqrt"), random_state=seed, n_jobs=-1),
        "ExtraTrees": lambda **k: ExtraTreesClassifier(n_estimators=k.get("n", 200), max_features=k.get("mf", "sqrt"), random_state=seed, n_jobs=-1),
        "HistGradBoost": lambda **k: HistGradientBoostingClassifier(max_iter=k.get("it", 150), learning_rate=k.get("lr", 0.1), max_depth=k.get("md", None), random_state=seed),
        "LogisticReg": lambda **k: LogisticRegression(C=k.get("C", 1.0), max_iter=300),
        "SVM_RBF": lambda **k: SVC(C=k.get("C", 10.0), gamma=k.get("g", "scale")),
    }


def prep(batches, train_ids, test_id, mode):
    """Return Xtr, ytr, Xte, yte after preprocessing. No test labels are ever used."""
    def z(X):  # per-batch standardisation
        s = X.std(axis=0); s[s == 0] = 1.0
        return (X - X.mean(axis=0)) / s
    Xtr = np.vstack([batches[b][0] if mode == "global" else z(batches[b][0]) for b in train_ids])
    ytr = np.concatenate([batches[b][1] for b in train_ids])
    Xte, yte = batches[test_id]
    if mode == "per_batch":
        Xte = z(Xte)
    else:
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    return Xtr, ytr, Xte, yte


def fit_eval(make, batches, test_id, mode):
    Xtr, ytr, Xte, yte = prep(batches, [b for b in sorted(batches) if b < test_id], test_id, mode)
    m = make(); t = time.time(); m.fit(Xtr, ytr); p = m.predict(Xte)
    return {"test_batch": test_id, "accuracy": float(accuracy_score(yte, p)), "macro_f1": float(f1_score(yte, p, average="macro")),
            "fit_s": round(time.time() - t, 1)}, yte, p


def mean_over(batches, make, mode, tests):
    rs = [fit_eval(make, batches, t, mode)[0] for t in tests]
    return {"folds": rs, "mean_accuracy": float(np.mean([r["accuracy"] for r in rs])), "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rs]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/gas"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="models")
    a = ap.parse_args(argv)
    t0 = time.time()
    batches = {k: load_batch(os.path.join(a.data, f"batch{k}.dat")) for k in range(1, 11)}
    log = lambda *x: print(*x, flush=True)
    summary = {"batches": {k: {"samples": int(len(v[1])), "per_class": {GASES[c]: int((v[1] == c).sum()) for c in GASES}} for k, v in batches.items()},
               "protocol": {"selection_test_batches": SEL_TEST, "final_test_batches": FINAL_TEST, "seed": a.seed}}
    fam = families(a.seed)

    # ---- 1. model family x preprocessing on the selection folds
    log("[1/4] model family x preprocessing on selection folds", SEL_TEST)
    sel = {}
    for name, mode in itertools.product(fam, ["global", "per_batch"]):
        r = mean_over(batches, fam[name], mode, SEL_TEST); sel[f"{name}|{mode}"] = r
        log(f"   {name:14s} {mode:9s} acc={r['mean_accuracy']:.4f} f1={r['mean_macro_f1']:.4f}  ({time.time()-t0:.0f}s)")
    summary["selection_stage1"] = {k: {"mean_accuracy": round(v["mean_accuracy"], 4), "mean_macro_f1": round(v["mean_macro_f1"], 4)} for k, v in sel.items()}
    best_key = max(sel, key=lambda k: sel[k]["mean_macro_f1"]); best_name, best_mode = best_key.split("|")
    log("   best family/preprocessing:", best_key)

    # ---- 2. small hyper-parameter search for the winning family (selection folds only)
    grids = {"RandomForest": [dict(n=n, mf=mf) for n in (200, 400) for mf in ("sqrt", 0.3)],
             "ExtraTrees": [dict(n=n, mf=mf) for n in (200, 400) for mf in ("sqrt", 0.3)],
             "HistGradBoost": [dict(it=it, lr=lr, md=md) for it in (150,) for lr in (0.05, 0.1) for md in (None, 6)],
             "LogisticReg": [dict(C=c) for c in (0.1, 1.0, 10.0)],
             "SVM_RBF": [dict(C=c, g=g) for c in (3.0, 10.0, 30.0) for g in ("scale", 0.0005)]}
    log("[2/4] hyper-parameter search for", best_name)
    hp = []
    for kw in grids[best_name]:
        r = mean_over(batches, lambda kw=kw: fam[best_name](**kw), best_mode, SEL_TEST)
        hp.append({"params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in kw.items()}, "mean_accuracy": round(r["mean_accuracy"], 4), "mean_macro_f1": round(r["mean_macro_f1"], 4)})
        log("   ", hp[-1], f"({time.time()-t0:.0f}s)")
    best_hp = max(hp, key=lambda h: h["mean_macro_f1"]); summary["selection_stage2"] = {"family": best_name, "preprocessing": best_mode, "grid": hp, "best": best_hp}
    make_best = lambda: fam[best_name](**best_hp["params"])

    # ---- 3. final held-out evaluation (batches 8, 9, 10), touched once
    log("[3/4] final held-out evaluation on batches", FINAL_TEST)
    def final(make, mode):
        ys, ps, rows = [], [], []
        for t in FINAL_TEST:
            r, y, p = fit_eval(make, batches, t, mode); rows.append(r); ys.append(y); ps.append(p)
        y, p = np.concatenate(ys), np.concatenate(ps)
        rep = classification_report(y, p, labels=list(GASES), target_names=list(GASES.values()), output_dict=True, zero_division=0)
        return {"per_batch": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
                "mean_accuracy": round(float(np.mean([r["accuracy"] for r in rows])), 4), "mean_macro_f1": round(float(np.mean([r["macro_f1"] for r in rows])), 4),
                "pooled_accuracy": round(float(accuracy_score(y, p)), 4), "pooled_macro_f1": round(float(f1_score(y, p, average="macro")), 4),
                "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])} for c in GASES.values()},
                "confusion_matrix(rows=true,cols=pred; Ethanol..Toluene)": confusion_matrix(y, p, labels=list(GASES)).tolist()}
    base = final(fam["RandomForest"], "global")      # the project's original baseline family (global scaler)
    tuned = final(make_best, best_mode)
    tuned_global = final(make_best, "global") if best_mode != "global" else tuned
    summary["final_test"] = {"baseline_RandomForest_global": base, f"selected_{best_name}_{best_mode}": tuned, f"selected_{best_name}_global": tuned_global}
    log("   baseline RF/global   acc", base["mean_accuracy"], "f1", base["mean_macro_f1"])
    log(f"   selected {best_name}/{best_mode}  acc", tuned["mean_accuracy"], "f1", tuned["mean_macro_f1"])
    log(f"   selected {best_name}/global      acc", tuned_global["mean_accuracy"], "f1", tuned_global["mean_macro_f1"])

    # ---- 4. deployment model: all 10 batches
    log("[4/4] fit deployment model on all 10 batches")
    mode = best_mode
    X = np.vstack([batches[b][0] if mode == "global" else (lambda M: (M - M.mean(0)) / np.where(M.std(0) == 0, 1, M.std(0)))(batches[b][0]) for b in sorted(batches)])
    y = np.concatenate([batches[b][1] for b in sorted(batches)])
    sc = StandardScaler().fit(X) if mode == "global" else None
    model = make_best(); model.fit(sc.transform(X) if sc is not None else X, y)
    os.makedirs(a.out, exist_ok=True)
    joblib.dump({"model": model, "scaler": sc, "preprocessing": mode, "classes": GASES, "family": best_name, "params": best_hp["params"]}, os.path.join(a.out, "gas_uci_model.joblib"))
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    summary["total_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(a.out, "metrics", "gas_uci_training.json"), "w"), indent=2)
    log("done in", summary["total_seconds"], "s -> models/gas_uci_model.joblib, models/metrics/gas_uci_training.json")


if __name__ == "__main__":
    main()
