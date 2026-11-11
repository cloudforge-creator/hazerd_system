"""Runtime fire detection from camera frames (WBS 5.2.3, 9.3).

FrameSource  : webcam index, video file (loops), or a folder of images (for demos).
FireCNN      : loads models/fire_cnn.keras (+ meta); input is raw RGB 0..255 (the model
               rescales internally), output is class probabilities.
FireSignals  : turns frames into Signals(fire_prob=...), averaged over the last frames.
Missing model/camera never silently becomes 'no fire': status says what is active.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque

import numpy as np

from src import config
from src.decision_engine import Signals


class FrameSource:
    def __init__(self, source):
        import cv2
        self.cv2 = cv2
        self._images, self._i, self.cap, self._is_file = None, 0, None, False
        if isinstance(source, str) and os.path.isdir(source):
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            self._images = sorted(os.path.join(source, f) for f in os.listdir(source)
                                  if f.lower().endswith(exts))
            if not self._images:
                raise RuntimeError(f"No images in {source}")
        else:
            self._is_file = not str(source).isdigit()
            self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera/video: {source}")

    def read(self):
        if self._images is not None:
            img = self.cv2.imread(self._images[self._i % len(self._images)])
            self._i += 1
            return img
        ok, frame = self.cap.read()
        if not ok and self._is_file:             # video ended -> loop for demos
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None             # camera failure -> None (caller counts it)

    def close(self):
        if self.cap is not None:
            self.cap.release()


WEIGHTS_NPZ = "fire_cnn_weights.npz"


def build_inference_model(img_size: int, n_classes: int = 3):
    """Inference-only twin of the training model: Rescaling -> MobileNetV2(avg pool) -> Dense(softmax).
    The training model also has random augmentation layers and a Dropout; neither has weights and both are
    inactive at inference, so they are left out on purpose (this also makes the augmentation defect impossible)."""
    import tensorflow as tf
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                             weights=None, pooling="avg")
    return tf.keras.Sequential([L.Input((img_size, img_size, 3)), L.Rescaling(1 / 127.5, offset=-1.0),
                                base, L.Dense(n_classes, activation="softmax")])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_weights(model_path: str, out_path: str) -> str:
    """Save the weights of a trained fire_cnn.keras as a plain NumPy .npz (independent of Keras/TensorFlow
    versions). Run this where the .keras loads (Colab right after training, or any matching environment)."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    base = next(l for l in model.layers if hasattr(l, "layers"))            # the nested MobileNetV2 model
    dense = [l for l in model.layers if l.__class__.__name__ == "Dense"][-1]
    arrays = {f"base_{i:04d}": w for i, w in enumerate(base.get_weights())}
    k, b = dense.get_weights()
    np.savez(out_path, dense_kernel=k, dense_bias=b, img_size=np.array(model.input_shape[1]),
             n_classes=np.array(k.shape[1]), n_base=np.array(len(arrays)), source_sha256=np.array(_sha256(model_path)), **arrays)
    return out_path


class FireCNN:
    def __init__(self, model_dir: str = config.MODEL_DIR, model=None, meta=None):
        self.model, self.meta = model, meta
        self.status = "fire CNN not loaded (no model found)"
        if model is None:
            self._load(model_dir)
        elif meta is not None:
            self.status = "fire CNN loaded"

    def _load(self, model_dir: str) -> None:
        mp = os.path.join(model_dir, "fire_cnn.keras")
        jp = os.path.join(model_dir, "fire_cnn_meta.json")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            return
        err = None
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(mp, compile=False)
            self.meta = json.load(open(jp))
            self.status = "fire CNN loaded"
            return
        except Exception as exc:                       # typically a Keras/TensorFlow version mismatch
            err = exc
        # fallback: portable weights exported from the same .keras file
        wp = os.path.join(model_dir, WEIGHTS_NPZ)
        self.model = None
        if os.path.exists(wp):
            try:
                z = np.load(wp, allow_pickle=False)
                if str(z["source_sha256"]) != _sha256(mp):
                    raise ValueError("fire_cnn_weights.npz does not match fire_cnn.keras (re-export it)")
                model = build_inference_model(int(z["img_size"]), int(z["n_classes"]))
                base = next(l for l in model.layers if hasattr(l, "layers"))
                base.set_weights([z[f"base_{i:04d}"] for i in range(int(z["n_base"]))])
                model.layers[-1].set_weights([z["dense_kernel"], z["dense_bias"]])
                self.model, self.meta = model, json.load(open(jp))
                self.status = "fire CNN loaded (portable weights; the .keras file does not load in this TensorFlow)"
                return
            except Exception as exc2:
                err = f"{err}; fallback also failed: {exc2.__class__.__name__}: {exc2}"
        msg = " ".join(str(err).split())[:220]
        self.status = (f"fire CNN failed to load ({err.__class__.__name__ if not isinstance(err, str) else 'error'}: {msg}) - camera ignored. "
                       f"Fix: run 'python -m ml.export_fire_weights' where the model loads, or use the TensorFlow version it was trained with.")

    @property
    def enabled(self) -> bool:
        return self.model is not None and self.meta is not None

    @property
    def class_names(self) -> list:
        return self.meta["class_names"]

    def predict(self, frame_bgr) -> np.ndarray:
        import cv2
        s = int(self.meta["img_size"])
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (s, s)).astype("float32")[None]
        # training=False is essential: the saved model contains random augmentation layers
        # (flip, rotation, zoom, contrast) that are ACTIVE by default in Keras 3 when the model
        # is called directly. model.predict() disables them; a bare model(x) does not.
        return np.asarray(self.model(x, training=False)).reshape(-1)


class FireSignals:
    def __init__(self, source, cnn: FireCNN, smooth: int = 3):
        self.source, self.cnn = source, cnn
        self.hist = deque(maxlen=smooth)
        self.camera_fault_frames = 0

    def __call__(self, reading) -> Signals:
        frame = self.source.read()
        if frame is None:
            self.camera_fault_frames += 1
            self.hist.clear()
            return Signals()
        probs = self.cnn.predict(frame)
        self.hist.append(float(probs[self.cnn.class_names.index("fire")]))
        return Signals(fire_prob=float(np.mean(self.hist)))
