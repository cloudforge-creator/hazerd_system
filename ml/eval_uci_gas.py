"""Research validation of the gas-classification method on the REAL UCI
"Gas Sensor Array Drift" dataset (16 sensors x 8 features, 6 gases, 10 batches).

    python -m ml.eval_uci_gas --data data/raw/gas

This validates the METHOD on real sensor data. It is not the runtime model: the MQ-2
gives one ADC value, the UCI array gives 128 features. Report both honestly.

Protocols
  A  random stratified split (optimistic: samples within a batch are correlated)
  B  train on batches 1..k, test on batch k+1   (realistic: sensor drift over time)
  C  train on batches 1-5, test on batches 6-10 (hardest drift case)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_batch(path: str):
    """Lines look like:  1;10.000000 1:15596.1621 2:1.868245 ... 128:-0.5"""
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, *feats = line.split()
            label = int(head.split(";")[0])
            row = {}
            for tok in feats:
                i, v = tok.split(":")
                row[int(i)] = float(v)
            X.append(row)
            y.append(label)
    n = max(max(r) for r in X)
    M = np.zeros((len(X), n))
    for r, row in enumerate(X):
        for i, v in row.items():
            M[r, i - 1] = v
    return M, np.array(y)


def make_model(seed):
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))


def score(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {"accuracy": round(float(accuracy_score(yte, p)), 4),
            "macro_f1": round(float(f1_score(yte, p, average="macro")), 4)}


def run(batches: dict, seed: int = 42) -> dict:
    ks = sorted(batches)
    Xall = np.vstack([batches[k][0] for k in ks])
    yall = np.concatenate([batches[k][1] for k in ks])
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, stratify=yall, random_state=seed)
    out = {"A_random_split": score(make_model(seed), Xtr, ytr, Xte, yte)}

    rolling = []
    for k in ks[:-1]:
        tr = [b for b in ks if b <= k]
        Xa = np.vstack([batches[b][0] for b in tr]); ya = np.concatenate([batches[b][1] for b in tr])
        rolling.append({"train_batches": f"1-{k}", "test_batch": k + 1,
                        **score(make_model(seed), Xa, ya, *batches[k + 1])})
    out["B_next_batch"] = {"folds": rolling,
                           "mean_accuracy": round(float(np.mean([r["accuracy"] for r in rolling])), 4),
                           "mean_macro_f1": round(float(np.mean([r["macro_f1"] for r in rolling])), 4)}
    if len(ks) >= 10:
        tr = [b for b in ks if b <= 5]; te = [b for b in ks if b > 5]
        out["C_train1-5_test6-10"] = score(
            make_model(seed),
            np.vstack([batches[b][0] for b in tr]), np.concatenate([batches[b][1] for b in tr]),
            np.vstack([batches[b][0] for b in te]), np.concatenate([batches[b][1] for b in te]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/gas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/metrics/uci_gas_eval.json")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "batch*.dat")))
    if not files:
        raise SystemExit(f"No batch*.dat files in {a.data}. See docs/DATASETS.md.")
    batches = {int("".join(c for c in os.path.basename(f) if c.isdigit())): load_batch(f) for f in files}
    result = run(batches, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
