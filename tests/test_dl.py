import json
import os
import shutil

import numpy as np
import pytest

from ml.prepare_fire_dataset import assign_splits, build_manifest, group_duplicates
from src.decision_engine import Signals
from src.signals import CombinedSignals
from src.temp_anomaly import TempAnomaly
from tests.helpers import reading

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# ---------------- synthetic fire dataset (YOLO layout) ----------------
def make_dataset(root, per_class=40, size=64, seed=0):
    rng = np.random.default_rng(seed)
    img_dir, lab_dir = os.path.join(root, "train", "images"), os.path.join(root, "train", "labels")
    os.makedirs(img_dir)
    os.makedirs(lab_dir)
    tint = {"fire": (200, 80, 20), "smoke": (130, 130, 130), "none": (30, 120, 40)}
    for cls, rgb in tint.items():
        for i in range(per_class):
            arr = np.clip(rng.normal(rgb, 60, (size, size, 3)), 0, 255).astype("uint8")
            name = f"{cls}_{i}"
            Image.fromarray(arr).save(os.path.join(img_dir, name + ".png"))
            if cls != "none":
                cid = 1 if cls == "fire" else 0
                open(os.path.join(lab_dir, name + ".txt"), "w").write(f"{cid} 0.5 0.5 0.2 0.2\n")
    for i in range(5):      # exact duplicates of fire images
        shutil.copy(os.path.join(img_dir, f"fire_{i}.png"), os.path.join(img_dir, f"dup_fire_{i}.png"))
        shutil.copy(os.path.join(lab_dir, f"fire_{i}.txt"), os.path.join(lab_dir, f"dup_fire_{i}.txt"))
    open(os.path.join(img_dir, "corrupt.jpg"), "wb").write(b"not an image")


def test_group_duplicates_and_splits():
    h = np.array([0, 1, 0xFFFFFFFF00000000, 0xFFFFFFFF00000001, 0x00000000FFFFFFFF], dtype=np.uint64)  # two near pairs + a loner
    g = group_duplicates(h, 3)
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2] and g[4] not in (g[0], g[2])
    labels = np.array(["fire"] * 60 + ["none"] * 60 + ["smoke"] * 60, dtype=object)
    groups = np.arange(180) // 2          # pairs share a group
    sp = assign_splits(labels, groups, 1)
    for gid in np.unique(groups):
        assert len(set(sp[groups == gid])) == 1
    assert 0.6 < (sp == "train").mean() < 0.8


def test_manifest_labels_dedup_and_corrupt(tmp_path):
    root = str(tmp_path / "fire")
    make_dataset(root)
    df = build_manifest(root, seed=1)
    assert df.attrs["skipped_corrupt"] == 1
    assert set(df["label"]) == {"none", "smoke", "fire"}
    assert (df["label"] == "fire").sum() == 45
    for i in range(5):                    # duplicate pairs must share a group and a split
        names = df["path"].map(os.path.basename)
        a = df[names == f"fire_{i}.png"].iloc[0]
        b = df[names == f"dup_fire_{i}.png"].iloc[0]
        assert a["group"] == b["group"] and a["split"] == b["split"]
    assert set(df["split"]) == {"train", "val", "test"}


# ---------------- signal merging ----------------
def test_combined_signals_takes_max_and_ors_alarms():
    a = lambda r: Signals(fire_prob=0.2, ml_hazard=0.5, ml_early_alarm=False)
    b = lambda r: Signals(fire_prob=0.9, temp_anomaly=0.3, ml_early_alarm=True)
    s = CombinedSignals(a, b, None)(reading(0))
    assert s.fire_prob == 0.9 and s.ml_hazard == 0.5 and s.temp_anomaly == 0.3 and s.ml_early_alarm is True
    assert CombinedSignals()(reading(0)) == Signals()


# ---------------- temperature anomaly runtime (stub model) ----------------
def test_temp_anomaly_flags_a_jump_and_resets_on_fault():
    meta = {"window": 5, "smooth": 3, "mean": 0.0, "std": 1.0, "threshold": 0.5}
    persistence = lambda x: np.array([[float(np.asarray(x).reshape(-1)[-1])]])
    ta = TempAnomaly(model=persistence, meta=meta)
    assert ta.enabled
    out = [ta(reading(i, temp=25.0)).temp_anomaly for i in range(12)]
    assert all(v in (None, 0.0) for v in out) and out[-1] == 0.0
    jumped = [ta(reading(20 + i, temp=40.0)).temp_anomaly for i in range(3)]
    assert jumped[-1] > 0.5
    assert ta(reading(99, faults=["temperature:missing"], temp=None)) == Signals()
    assert len(ta.temps) == 0


def test_temp_anomaly_refuses_non_device_model(tmp_path):
    (tmp_path / "temp_lstm.keras").write_bytes(b"x")
    json.dump({"source": "NAB", "interval_s": 300}, open(tmp_path / "temp_lstm_meta.json", "w"))
    ta = TempAnomaly(str(tmp_path))
    assert not ta.enabled and "not used at runtime" in ta.status


# ---------------- fire CNN runtime (stub model) ----------------
def test_fire_signals_average_and_camera_fault():
    pytest.importorskip("cv2")
    from src.fire_cnn import FireCNN, FireSignals
    meta = {"class_names": ["none", "smoke", "fire"], "img_size": 32}
    seq = iter([0.9, 0.9, 0.0, 0.9])
    cnn = FireCNN(model=lambda x, training=None: np.array([[0.0, 0.0, next(seq)]]), meta=meta)

    class Src:
        frames = [np.zeros((48, 48, 3), np.uint8)] * 3 + [None, np.zeros((48, 48, 3), np.uint8)]

        def read(self):
            return self.frames.pop(0)

    fs = FireSignals(Src(), cnn, smooth=3)
    vals = [fs(reading(i)).fire_prob for i in range(3)]
    assert vals[0] == pytest.approx(0.9) and vals[2] == pytest.approx(0.6)
    assert fs(reading(3)) == Signals() and fs.camera_fault_frames == 1
    assert fs(reading(4)).fire_prob == pytest.approx(0.9)      # history restarted after the fault


# ---------------- end-to-end with real TensorFlow (tiny) ----------------
tf = pytest.importorskip("tensorflow")


def test_cnn_pipeline_end_to_end(tmp_path):
    pytest.importorskip("cv2")
    from ml.prepare_fire_dataset import main as _unused  # noqa: F401
    from ml.train_cnn import main as train_cnn
    from src.fire_cnn import FireCNN
    root = str(tmp_path / "fire")
    make_dataset(root, per_class=40)
    df = build_manifest(root, seed=1)
    manifest = str(tmp_path / "m.csv")
    df.to_csv(manifest, index=False)
    out = str(tmp_path / "models")
    m = train_cnn(["--manifest", manifest, "--out", out, "--img", "64", "--batch", "16", "--weights", "none",
                   "--epochs-head", "1", "--epochs-ft", "1", "--unfreeze", "10"], quiet=True)
    assert os.path.exists(os.path.join(out, "fire_cnn.keras"))
    assert "fire_vs_rest_test" in m and "three_class_test" in m
    cnn = FireCNN(out)
    assert cnn.enabled
    probs = cnn.predict(np.zeros((90, 120, 3), np.uint8))
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)


def test_lstm_device_mode_end_to_end(tmp_path):
    import pandas as pd
    from ml.train_temp_lstm import main as train_lstm
    rng = np.random.default_rng(0)
    t = np.arange(1500)
    temp = 27 + np.sin(t / 40.0) + rng.normal(0, 0.05, len(t))
    csv = str(tmp_path / "log.csv")
    pd.DataFrame({"timestamp": t * 0.5, "temperature": temp, "faults": [""] * len(t)}).to_csv(csv, index=False)
    out = str(tmp_path / "models")
    train_lstm(["--csv", csv, "--value-col", "temperature", "--interval-s", "0.5", "--source", "device",
                "--window", "20", "--epochs", "2", "--out", out], quiet=True)
    meta = json.load(open(os.path.join(out, "temp_lstm_meta.json")))
    assert meta["source"] == "device" and meta["threshold"] > 0
    ta = TempAnomaly(out)
    assert ta.enabled
    vals = [ta(reading(i, temp=float(temp[i]))).temp_anomaly for i in range(80)]
    assert vals[-1] is not None and 0.0 <= vals[-1] <= 1.0


def test_fire_cnn_predict_is_deterministic_with_augmentation_layers():
    """Regression: a saved model that contains random augmentation layers must run in inference
    mode at runtime (Keras 3 applies the augmentation when the model is called without training=False)."""
    tf = pytest.importorskip("tensorflow")
    from src.fire_cnn import FireCNN
    L = tf.keras.layers
    model = tf.keras.Sequential([L.Input((32, 32, 3)), L.RandomFlip("horizontal"), L.RandomRotation(0.2),
                                 L.RandomZoom(0.3), L.RandomContrast(0.5), L.Rescaling(1 / 127.5, offset=-1.0),
                                 L.Flatten(), L.Dense(3, activation="softmax")])
    meta = {"class_names": ["none", "smoke", "fire"], "img_size": 32}
    cnn = FireCNN(model=model, meta=meta)
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    first = cnn.predict(frame)
    for _ in range(5):
        assert np.array_equal(cnn.predict(frame), first)


def test_fire_cnn_portable_weights_fallback_matches_original(tmp_path, monkeypatch):
    """If the .keras file cannot be loaded (Keras/TensorFlow version mismatch), the runtime must fall back to the
    portable .npz weights and give the same probabilities; stale weights must be refused."""
    tf = pytest.importorskip("tensorflow")
    from src import fire_cnn
    from src.fire_cnn import FireCNN, export_weights
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights=None, pooling="avg")
    model = tf.keras.Sequential([L.Input((32, 32, 3)), L.RandomFlip("horizontal"), L.RandomRotation(0.05), L.RandomZoom(0.1),
                                 L.RandomContrast(0.2), L.Rescaling(1 / 127.5, offset=-1.0), base, L.Dropout(0.3),
                                 L.Dense(3, activation="softmax")])
    model.save(str(tmp_path / "fire_cnn.keras"))
    (tmp_path / "fire_cnn_meta.json").write_text(json.dumps({"class_names": ["none", "smoke", "fire"], "img_size": 32}))
    export_weights(str(tmp_path / "fire_cnn.keras"), str(tmp_path / fire_cnn.WEIGHTS_NPZ))
    frame = np.random.default_rng(1).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    expected = FireCNN(str(tmp_path)).predict(frame)
    monkeypatch.setattr(tf.keras.models, "load_model", lambda *a, **k: (_ for _ in ()).throw(TypeError("version mismatch")))
    cnn = FireCNN(str(tmp_path))
    assert cnn.enabled and "portable" in cnn.status
    assert np.allclose(cnn.predict(frame), expected, atol=1e-5)
    with open(tmp_path / "fire_cnn.keras", "ab") as fh:        # .keras changed after the export -> stale weights
        fh.write(b"x")
    stale = FireCNN(str(tmp_path))
    assert not stale.enabled and "failed to load" in stale.status
