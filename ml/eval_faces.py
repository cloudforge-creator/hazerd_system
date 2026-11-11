"""Measure face-recognition accuracy on YOUR people and camera, and pick a safe threshold.

Put photos (or frames saved from your camera) in folders named after each person:
    data/faces/Ali Khan/*.jpg      data/faces/Sara Noor/*.jpg      ... (at least 3 people, 5+ photos each)

    python -m ml.eval_faces --data data/faces --enroll 3

Reports, per threshold: correct-name rate, wrong-name rate, rejected-as-unknown rate (for enrolled people) and the
FALSE ACCEPT rate (a person who is NOT enrolled being named as someone). Choose the lowest threshold whose false
accept rate is acceptable for your site (the roadmap target is <= 1%). Quote these numbers, not the library's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from src.personnel.registry import EMBED_DIM, PersonStore
from src.storage import Storage

MIN_USABLE_CORRECT = 0.80     # a threshold that rarely recognises enrolled people is not 'safe', just useless
THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.71, 0.05)]


def _store(emb_by_person: dict, enrolled: dict) -> PersonStore:
    st = PersonStore(Storage())
    for name, vecs in enrolled.items():
        pid = st.add_person(name, "eval", consent=True)
        for v in vecs:
            st.add_embedding(pid, v)
    return st


def evaluate(emb_by_person: dict, k_enroll: int = 3, margin: float = 0.05, seed: int = 42, thresholds=THRESHOLDS) -> dict:
    rng = np.random.default_rng(seed)
    usable = {n: list(v) for n, v in emb_by_person.items() if len(v) >= k_enroll + 1}
    if len(usable) < 3:
        raise ValueError(f"need at least 3 people with {k_enroll + 1}+ usable photos each (have {len(usable)}); add more photos")
    split = {}
    for n, v in usable.items():
        idx = rng.permutation(len(v))
        split[n] = ([v[i] for i in idx[:k_enroll]], [v[i] for i in idx[k_enroll:]])
    known = _store(usable, {n: tr for n, (tr, _) in split.items()})
    rows = []
    for thr in thresholds:
        correct = wrong = rejected = n_known = 0
        for n, (_, test) in split.items():
            for e in test:
                n_known += 1
                p, _ = known.match(e, threshold=thr, margin=margin)
                if p is None:
                    rejected += 1
                elif p["name"] == n:
                    correct += 1
                else:
                    wrong += 1
        fa = n_imp = 0
        for held_out, (tr, test) in split.items():          # impostor: the person is NOT in the gallery
            gal = _store(usable, {n: t for n, (t, _) in split.items() if n != held_out})
            for e in tr + test:
                n_imp += 1
                fa += gal.match(e, threshold=thr, margin=margin)[0] is not None
        rows.append({"threshold": thr, "correct_name_rate": round(correct / n_known, 4), "wrong_name_rate": round(wrong / n_known, 4),
                     "rejected_unknown_rate": round(rejected / n_known, 4), "false_accept_rate": round(fa / n_imp, 4)})
    ok = [r for r in rows if r["false_accept_rate"] <= 0.01]
    best = max(ok, key=lambda r: r["correct_name_rate"]) if ok else None
    warning = None
    if best is None:
        warning = "no threshold reached a false-accept rate of 1% or less: add photos, improve lighting, or review the camera"
    elif best["correct_name_rate"] < MIN_USABLE_CORRECT:
        warning = (f"no threshold is both safe and useful: the false-accept rate is low only because enrolled people are "
                   f"rarely recognised either (best correct-name rate {best['correct_name_rate']:.0%}). Add photos, improve "
                   f"lighting, or review the camera")
        best = None
    return {"people": len(usable), "enroll_per_person": k_enroll, "test_images": n_known, "impostor_images": n_imp,
            "margin": margin, "sweep": rows, "suggested_threshold_for_FAR<=1%": best["threshold"] if best else None,
            "warning": warning}


def load_embeddings(root: str) -> tuple:
    from src.personnel.face import FaceEngine
    import cv2
    engine = FaceEngine()
    out, skipped = {}, {}
    for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        name = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            img = cv2.imread(f)
            emb, msg = (None, "unreadable") if img is None else engine.embed_single_face(img)
            if emb is None:
                skipped.setdefault(name, []).append(f"{os.path.basename(f)}: {msg}")
            else:
                out.setdefault(name, []).append(emb)
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/faces")
    ap.add_argument("--enroll", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default="models/metrics/face_eval.json")
    a = ap.parse_args()
    embs, skipped = load_embeddings(a.data)
    for n, s in skipped.items():
        print(f"skipped {len(s)} photo(s) of {n}: " + "; ".join(s[:3]))
    res = evaluate(embs, a.enroll, a.margin)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({**res, "skipped": skipped}, open(a.out, "w"), indent=2)
    print(f"{res['people']} people, {res['test_images']} test photos, {res['impostor_images']} impostor tests")
    print("threshold  correct  wrong-name  rejected  FALSE-ACCEPT")
    for r in res["sweep"]:
        print(f"  {r['threshold']:.2f}     {r['correct_name_rate']:.3f}    {r['wrong_name_rate']:.3f}      {r['rejected_unknown_rate']:.3f}     {r['false_accept_rate']:.3f}")
    print("suggested threshold (false accept <= 1%):", res["suggested_threshold_for_FAR<=1%"] or "NONE", res["warning"] or "")


if __name__ == "__main__":
    main()
