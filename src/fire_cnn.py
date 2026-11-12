"""Runtime fire detection from camera frames (WBS 5.2.3, 9.3).

FrameSource  : webcam index, video file (loops), or a folder of images (for demos).
FireCNN      : loads models/fire_cnn.keras or portable .npz weights (+ meta).
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
        if not ok and self._is_file:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


WEIGHTS_NPZ = "fire_cnn_weights.npz"


def build_inference_model(img_size: int, n_classes: int = 3):
    """Inference-only twin of the training model."""
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
    """Export trained Keras weights to a portable NumPy archive."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    base = next(l for l in model.layers if hasattr(l, "layers"))
    dense = [l for l in model.layers if l.__class__.__name__ == "Dense"][-1]
    arrays = {f"base_{i:04d}": w for i, w in enumerate(base.get_weights())}
    k, b = dense.get_weights()
    np.savez(out_path, dense_kernel=k, dense_bias=b, img_size=np.array(model.input_shape[1]),
             n_classes=np.array(k.shape[1]), n_base=np.array(len(arrays)),
             source_sha256=np.array(_sha256(model_path)), **arrays)
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
        wp = os.path.join(model_dir, WEIGHTS_NPZ)
        if not os.path.exists(jp):
            self.status = "fire CNN not loaded (metadata missing)"
            return
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta.get("class_names"), list) or len(meta["class_names"]) < 2:
                raise ValueError("metadata class_names must contain at least two classes")
            img_size = int(meta["img_size"])
            if img_size <= 0:
                raise ValueError("metadata img_size must be positive")
        except Exception as exc:
            self.status = f"fire CNN metadata invalid ({exc.__class__.__name__}: {exc})"
            return

        keras_error = None
        if os.path.exists(mp):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(mp, compile=False)
                self.meta = meta
                self.status = "fire CNN loaded"
                return
            except Exception as exc:
                keras_error = exc

        # Portable weights are independently loadable; the .keras artifact is optional.
        if os.path.exists(wp):
            try:
                with np.load(wp, allow_pickle=False) as z:
                    n_classes = int(z["n_classes"])
                    if n_classes != len(meta["class_names"]) or int(z["img_size"]) != img_size:
                        raise ValueError("portable weights do not match metadata dimensions/classes")
                    if os.path.exists(mp) and "source_sha256" in z.files:
                        if str(z["source_sha256"]) != _sha256(mp):
                            raise ValueError("portable weights do not match fire_cnn.keras; re-export them")
                    model = build_inference_model(img_size, n_classes)
                    base = next(l for l in model.layers if hasattr(l, "layers"))
                    base.set_weights([z[f"base_{i:04d}"] for i in range(int(z["n_base"]))])
                    model.layers[-1].set_weights([z["dense_kernel"], z["dense_bias"]])
                self.model, self.meta = model, meta
                reason = "keras artifact missing" if not os.path.exists(mp) else "Keras load failed"
                self.status = f"fire CNN loaded (portable .npz weights; {reason})"
                return
            except Exception as exc:
                keras_error = exc

        detail = f"; Keras: {keras_error}" if keras_error else ""
        self.status = f"fire CNN failed to load (no usable model/portable weights{detail}) - camera ignored"

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
