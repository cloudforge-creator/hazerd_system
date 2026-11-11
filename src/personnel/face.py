"""Face detection + recognition with OpenCV's built-in YuNet (detector) and SFace (128-d embeddings).

Needs two ONNX model files (download with: python scripts/download_face_models.py).
Recognition accuracy depends on your camera, lighting and enrolment photos: MEASURE it with
python -m ml.eval_faces before trusting it for anything that matters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src import config
from src.personnel.registry import normalize


@dataclass
class Detection:
    box: tuple                       # x, y, w, h in pixels
    score: float
    embedding: Optional[np.ndarray] = None


class FaceEngine:
    def __init__(self, detect_model: str = None, recog_model: str = None, score: float = None):
        import cv2
        detect_model = detect_model or config.FACE_DETECT_MODEL
        recog_model = recog_model or config.FACE_RECOG_MODEL
        for p in (detect_model, recog_model):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing face model: {p} (run: python scripts/download_face_models.py)")
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(detect_model, "", (320, 320),
                                             score if score is not None else config.FACE_DETECT_SCORE, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(recog_model, "")

    @staticmethod
    def models_present() -> tuple:
        missing = [p for p in (config.FACE_DETECT_MODEL, config.FACE_RECOG_MODEL) if not os.path.exists(p)]
        return (not missing), ("face models found" if not missing else "missing: " + ", ".join(missing))

    def detect_and_embed(self, frame_bgr, embed: bool = True) -> list:
        h, w = frame_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame_bgr)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = (int(round(float(v))) for v in f[:4])
            x, y = max(x, 0), max(y, 0)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 0 or bh <= 0:
                continue
            emb = None
            if embed:
                aligned = self.rec.alignCrop(frame_bgr, f)
                emb = normalize(self.rec.feature(aligned))
            out.append(Detection((x, y, bw, bh), float(f[14]), emb))
        return out

    def embed_single_face(self, frame_bgr) -> tuple:
        """For enrolment. Returns (embedding or None, message). Requires exactly one clear face."""
        dets = self.detect_and_embed(frame_bgr)
        if not dets:
            return None, "no face found"
        if len(dets) > 1:
            return None, "more than one face in the picture; use a photo of one person"
        d = dets[0]
        if d.box[2] < config.FACE_MIN_SIZE_PX:
            return None, f"face too small ({d.box[2]} px); move closer (need at least {config.FACE_MIN_SIZE_PX} px)"
        return d.embedding, "ok"
