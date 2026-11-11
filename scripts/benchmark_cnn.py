"""Time the fire CNN on THIS machine, including camera capture.

    python scripts/benchmark_cnn.py                    # laptop camera (index 0), 100 frames
    python scripts/benchmark_cnn.py --source 1 --n 200
    python scripts/benchmark_cnn.py --source path\\to\\video.mp4

The system runs one monitoring cycle every 0.5 s. This prints how long capture + CNN take per frame
(mean, p50, p95, p99, max), compares them with the 0.5 s budget and says what to do if it does not fit.
Add the 10-40 ms of the rest of the cycle (see evaluation/results/REPORT.md, C3) for a full-cycle figure.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pct(a, q):
    return float(np.percentile(a, q))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--model-dir", default="models")
    a = ap.parse_args(argv)
    import cv2
    from src.fire_cnn import FireCNN
    cnn = FireCNN(a.model_dir)
    if not cnn.enabled:
        raise SystemExit(cnn.status)
    print("[model]", cnn.status)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (isinstance(src, int) and os.name == "nt") else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open '{a.source}'. Close other apps using the camera or try --source 1.")
    for _ in range(5):                       # warm-up: camera auto-exposure and the first TensorFlow call
        ok, img = cap.read()
        if ok:
            cnn.predict(img)
    cap_ms, cnn_ms = [], []
    for _ in range(a.n):
        t0 = time.perf_counter(); ok, img = cap.read(); t1 = time.perf_counter()
        if not ok:
            break
        cnn.predict(img); t2 = time.perf_counter()
        cap_ms.append((t1 - t0) * 1000); cnn_ms.append((t2 - t1) * 1000)
    cap.release()
    if not cnn_ms:
        raise SystemExit("no frames were read")
    tot = np.array(cap_ms) + np.array(cnn_ms)
    print(f"\nframes timed: {len(tot)}")
    for name, v in (("camera read", np.array(cap_ms)), ("CNN predict", np.array(cnn_ms)), ("capture + CNN", tot)):
        print(f"  {name:14s} mean {v.mean():7.1f} ms | p50 {pct(v, 50):7.1f} | p95 {pct(v, 95):7.1f} | p99 {pct(v, 99):7.1f} | max {v.max():7.1f}")
    p99 = pct(tot, 99)
    print("\nBUDGET: one monitoring cycle = 500 ms (the rest of the cycle needs about 10-40 ms)")
    if p99 <= 300:
        print(f"  OK: capture + CNN p99 = {p99:.0f} ms fits comfortably.")
    elif p99 <= 450:
        print(f"  TIGHT: p99 = {p99:.0f} ms. It fits, but a busy PC may overrun. Consider running the CNN every 2nd cycle.")
    else:
        print(f"  TOO SLOW: p99 = {p99:.0f} ms exceeds the budget. Run the CNN every 2nd to 4th cycle or in a separate thread.")


if __name__ == "__main__":
    main()
