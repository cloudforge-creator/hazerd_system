"""Check that face detection + recognition work on YOUR camera, before you rely on them.

    python scripts/check_face_setup.py              # grabs two frames from webcam 0
    python scripts/check_face_setup.py photo.jpg    # or use a photo

Stand in front of the camera alone. Expect: 1 face, embedding length 128, and a similarity of two frames of the
same person well above 0.363 (usually 0.6-0.95). Low values mean poor lighting, a blurry or too-small face.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import cv2
    import numpy as np
    from src import config
    from src.personnel.face import FaceEngine

    ok, msg = FaceEngine.models_present()
    print(msg)
    if not ok:
        print("-> run: python scripts/download_face_models.py")
        return 1
    engine = FaceEngine()
    frames = []
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print("cannot read", sys.argv[1])
            return 1
        frames = [img]
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("cannot open webcam 0")
            return 1
        for _ in range(10):                      # let the exposure settle
            cap.read()
        for _ in range(2):
            good, f = cap.read()
            if good:
                frames.append(f)
            time.sleep(0.7)
        cap.release()
    embs = []
    for i, f in enumerate(frames, 1):
        t0 = time.time()
        dets = engine.detect_and_embed(f)
        ms = (time.time() - t0) * 1000
        print(f"frame {i}: {f.shape[1]}x{f.shape[0]}, faces found: {len(dets)}, {ms:.0f} ms")
        for d in dets:
            print(f"   box={d.box} score={d.score:.2f} width={d.box[2]}px "
                  f"({'OK' if d.box[2] >= config.FACE_MIN_SIZE_PX else 'TOO SMALL for recognition'}) "
                  f"embedding={d.embedding.shape} norm={float(np.linalg.norm(d.embedding)):.2f}")
        if len(dets) == 1:
            embs.append(dets[0].embedding)
    if len(embs) == 2:
        sim = float(embs[0] @ embs[1])
        print(f"similarity of the two frames (same person): {sim:.3f}   threshold in use: {config.FACE_MATCH_THRESHOLD}")
        print("GOOD" if sim > config.FACE_MATCH_THRESHOLD + 0.15 else "WEAK: improve lighting / distance, or re-check the camera")
    elif frames and not embs:
        print("No single clear face found. Face the camera, add light, move closer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
