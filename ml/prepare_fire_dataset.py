"""Build a leak-safe manifest for the fire/smoke image classifier (D-Fire, YOLO format).

    python -m ml.prepare_fire_dataset --root data/raw/fire --out data/fire_manifest.csv

Classes: none / smoke / fire  (an image containing any fire box is 'fire', else any smoke box
is 'smoke', else 'none'). Near-duplicate images (dHash Hamming <= 3) are grouped and each
group goes to ONE split, so the same scene can never appear in both train and test.
Corrupt images are skipped and counted.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

CLASSES = ["none", "smoke", "fire"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def yolo_label(label_path: str, fire_id: int, smoke_id: int) -> str:
    if not os.path.exists(label_path):
        return "none"
    ids = set()
    for line in open(label_path):
        parts = line.split()
        if parts:
            try:
                ids.add(int(float(parts[0])))
            except ValueError:
                pass
    return "fire" if fire_id in ids else ("smoke" if smoke_id in ids else "none")


def dhash(img) -> np.uint64:
    from PIL import Image
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.frombuffer(np.packbits(bits).tobytes(), dtype=">u8")[0].astype(np.uint64)


def group_duplicates(hashes: np.ndarray, max_dist: int = 3) -> np.ndarray:
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        x = hashes[i + 1:] ^ hashes[i]
        d = _POP[x.view(np.uint8)].reshape(-1, 8).sum(1)
        for j in np.flatnonzero(d <= max_dist):
            ri, rj = find(i), find(i + 1 + j)
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def assign_splits(labels, groups, seed, ratios=(0.70, 0.15, 0.15)):
    """Group-level, class-stratified split balanced by image count."""
    rng = np.random.default_rng(seed)
    names = ("train", "val", "test")
    split = np.empty(len(labels), dtype=object)
    sev = {"none": 0, "smoke": 1, "fire": 2}
    ugroups = np.unique(groups)
    gl = {g: max((labels[i] for i in np.flatnonzero(groups == g)), key=lambda l: sev[l]) for g in ugroups}
    for cls in CLASSES:
        gs = [g for g in ugroups if gl[g] == cls]
        rng.shuffle(gs)
        total = sum(int((groups == g).sum()) for g in gs)
        target = [r * total for r in ratios]
        cur = [0, 0, 0]
        for g in gs:
            k = int(np.argmax([(target[i] - cur[i]) / max(target[i], 1e-9) for i in range(3)]))
            split[groups == g] = names[k]
            cur[k] += int((groups == g).sum())
    return split


def build_manifest(root: str, fire_id=1, smoke_id=0, seed=42, max_dist=3):
    import pandas as pd
    from PIL import Image
    rows, skipped = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != "images":
            continue
        label_dir = os.path.join(os.path.dirname(dirpath), "labels")
        for f in sorted(filenames):
            if not f.lower().endswith(EXTS):
                continue
            p = os.path.join(dirpath, f)
            try:
                with Image.open(p) as im:
                    im.load()
                    h = dhash(im)
            except Exception:
                skipped += 1
                continue
            lab = yolo_label(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"), fire_id, smoke_id)
            rows.append((p, lab, h))
    if not rows:
        raise SystemExit(f"No images found under any 'images' folder in {root}")
    df = pd.DataFrame(rows, columns=["path", "label", "hash"])
    hashes = df["hash"].to_numpy(dtype=np.uint64)
    df["group"] = group_duplicates(hashes, max_dist)
    df["split"] = assign_splits(df["label"].to_numpy(), df["group"].to_numpy(), seed)
    df["hash"] = df["hash"].astype(str)
    df.attrs["skipped_corrupt"] = skipped
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/fire")
    ap.add_argument("--out", default="data/fire_manifest.csv")
    ap.add_argument("--fire-id", type=int, default=1)
    ap.add_argument("--smoke-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = build_manifest(a.root, a.fire_id, a.smoke_id, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"images={len(df)} skipped_corrupt={df.attrs['skipped_corrupt']} "
          f"duplicate_groups={int((df.groupby('group').size() > 1).sum())}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
