"""Download the public datasets used by the training scripts.

    python scripts/get_datasets.py --links          # print every link and where files go
    python scripts/get_datasets.py --uci            # UCI gas drift  -> data/raw/gas/
    python scripts/get_datasets.py --nab            # NAB temperature -> data/raw/temperature/
    python scripts/get_datasets.py --face-models    # YuNet + SFace ONNX models -> models/face/
    python scripts/get_datasets.py --all            # the three above

D-Fire (fire/smoke images, ~21,000) is large and hosted on OneDrive/Kaggle, so it is NOT downloaded here:
see docs/DATASETS_AND_TRAINING.md (train it on Colab). Raw data stays out of Git.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import urllib.request
import zipfile

UCI_PAGE = "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset"   # Vergara 2012, DOI 10.24432/C5RP6W (ID 270 is a different dataset)
UCI_ZIP = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
# Backup copy of the same 10 batch files in the project's own GitHub repository
REPO_RAW = "https://raw.githubusercontent.com/cloudforge-creator/fyp-hazard-prevention-system/main/data/gas_dataset/"
LINKS = {
    "UCI Gas Sensor Array Drift (gas)": UCI_PAGE,
    "NAB machine temperature (temperature)": "https://github.com/numenta/NAB (data/realKnownCause/machine_temperature_system_failure.csv, labels/combined_windows.json)",
    "D-Fire GitHub page (fire)": "https://github.com/gaiasd/DFireDataset",
    "D-Fire images + labels (OneDrive)": "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw",
    "D-Fire Kaggle copy, ready to use": "https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo",
    "YuNet + SFace face models (OpenCV Zoo)": "https://github.com/opencv/opencv_zoo/tree/main/models",
}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hazard-system-dataset-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def uci(dest: str = "data/raw/gas") -> int:
    os.makedirs(dest, exist_ok=True)
    try:
        print("downloading UCI zip ...")
        z = zipfile.ZipFile(io.BytesIO(_get(UCI_ZIP, 300)))
        n = 0
        for name in z.namelist():
            base = os.path.basename(name)
            if base.startswith("batch") and base.endswith(".dat"):
                open(os.path.join(dest, base), "wb").write(z.read(name)); n += 1
        if n == 10:
            return n
        print(f"zip had {n} batch files, expected 10; trying the backup copy")
    except Exception as exc:
        print("UCI download failed:", exc.__class__.__name__, exc, "- trying the backup copy in the project repository")
    n = 0
    for k in range(1, 11):
        data = _get(f"{REPO_RAW}batch{k}.dat", 300)
        open(os.path.join(dest, f"batch{k}.dat"), "wb").write(data); n += 1
        print(f"  batch{k}.dat ({len(data) / 1e6:.1f} MB)")
    return n


def verify_uci(dest: str = "data/raw/gas") -> int:
    """The original drift dataset has 13,910 samples with 128 features in 10 batches and 6 classes."""
    sys.path.insert(0, os.getcwd())
    from ml.eval_uci_gas import load_batch
    total, classes = 0, set()
    for k in range(1, 11):
        X, y = load_batch(os.path.join(dest, f"batch{k}.dat"))
        if X.shape[1] != 128:
            raise SystemExit(f"batch{k}.dat has {X.shape[1]} features, expected 128: wrong dataset?")
        total += len(y); classes |= set(y.tolist())
    ok = total == 13910 and classes == {1, 2, 3, 4, 5, 6}
    print(f"verified: {total} samples, classes {sorted(classes)} -> {'OK' if ok else 'UNEXPECTED (expected 13910 samples and 6 classes)'}")
    if not ok:
        raise SystemExit("UCI data does not match the Gas Sensor Array Drift Dataset (ID 224)")
    return total


def nab() -> None:
    sys.path.insert(0, os.getcwd())
    from ml.train_temp_lstm import download_nab
    download_nab()


def face_models() -> None:
    subprocess.check_call([sys.executable, os.path.join("scripts", "download_face_models.py")])


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in ("links", "uci", "nab", "face-models", "all"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args(argv)
    if a.links or not any((a.uci, a.nab, a.face_models, a.all)):
        for k, v in LINKS.items():
            print(f"{k}\n    {v}")
        print("\nD-Fire is not downloaded here: see docs/DATASETS_AND_TRAINING.md")
    if a.uci or a.all:
        print("UCI batches present:", uci())
        verify_uci()
    if a.nab or a.all:
        nab()
    if a.face_models or a.all:
        face_models()


if __name__ == "__main__":
    main()
