"""Use and validate the fire CNN on YOUR laptop camera.

    python scripts/webcam_validate.py live                          # see P(fire) on the live camera (press q to quit)
    python scripts/webcam_validate.py record --label normal --minutes 30
    python scripts/webcam_validate.py record --label fire --minutes 4
    python scripts/webcam_validate.py report                         # false alarms per hour + detection, with a verdict

--source 0 is the laptop camera (try 1 if 0 is not it). --source can also be a video file or a folder of images.
Frames are taken at 2 per second, which is how often the monitoring system runs the CNN, and the fire score is the
mean of the last 3 frames, exactly as in the system. Thresholds 0.40 / 0.70 / 0.80 are the decision-engine levels
(WARNING / CRITICAL / forced CRITICAL).

What to record (safe, no real fire needed):
  normal : 30-60 minutes of ordinary use of the room: you moving, lamps on and off, daylight changes, screens, orange
           or red objects, steam from a cup - everything that is NOT fire. Do not show fire.
  fire   : 3-5 minutes of a fire video playing on a phone or second screen held in front of the camera. This is a
           screen recapture, which is easier or harder than real flame in unknown ways, so it checks that the
           model reacts, not how it performs on real fire. A supervised candle flame is a better test if you can.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THRESH = (0.40, 0.70, 0.80)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def open_source(src):
    import cv2
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        return cap
    return cv2.VideoCapture(src)


def frames(src, fps):
    """Yield (seconds, BGR frame) at `fps` frames per second from a camera, a video file or a folder of images."""
    import cv2
    if isinstance(src, str) and os.path.isdir(src):
        for i, f in enumerate(sorted(x for x in os.listdir(src) if x.lower().endswith(IMG_EXT))):
            img = cv2.imread(os.path.join(src, f))
            if img is not None:
                yield i / fps, img
        return
    cap = open_source(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera/video '{src}'. Close other apps using the camera, or try --source 1.")
    is_file = isinstance(src, str) and not src.isdigit()
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(vfps / fps))) if is_file else 1
    t0, nxt, n = time.monotonic(), 0.0, 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            if is_file:
                if (n - 1) % step == 0:
                    yield (n - 1) / vfps, img
            else:
                now = time.monotonic() - t0
                if now >= nxt:
                    nxt = now + 1.0 / fps
                    yield now, img
    finally:
        cap.release()


def load_cnn(model_dir):
    from src.fire_cnn import FireCNN
    cnn = FireCNN(model_dir)
    if not cnn.enabled:
        raise SystemExit(cnn.status)
    print("[model]", cnn.status)
    return cnn


def fire_p(cnn, img):
    p = cnn.predict(img)
    return p, float(p[cnn.class_names.index("fire")])


def level(p):
    # This is a CNN score band only. The runtime decision engine requires an
    # independent physical sensor before allowing the CNN to cause CRITICAL.
    return "HIGH - NEEDS SENSOR CONFIRMATION" if p >= 0.80 else "ELEVATED - NEEDS SENSOR CONFIRMATION" if p >= 0.70 else "WARNING" if p >= 0.40 else "LOW"


def episodes(mask):
    return int(((mask[1:] & ~mask[:-1]).sum()) + (1 if len(mask) and mask[0] else 0))


def smooth(p, k=3):
    return np.array([p[max(0, i - k + 1): i + 1].mean() for i in range(len(p))])


def cmd_live(a):
    import cv2
    cnn = load_cnn(a.model_dir); hist = []; log = None
    if a.log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
        log = csv.writer(open(a.log, "w", newline="")); log.writerow(["t_s", "p_fire", "p_fire_smoothed", "level"])
    print("press q in the window to quit" if not a.no_window else "running (Ctrl+C to stop)")
    t_end = time.monotonic() + a.minutes * 60 if a.minutes else None
    try:
        for t, img in frames(a.source, a.fps):
            p, pf = fire_p(cnn, img); hist.append(pf); sm = float(np.mean(hist[-3:])); lv = level(sm)
            if log: log.writerow([round(t, 2), round(pf, 4), round(sm, 4), lv])
            print(f"{t:7.1f}s  P(fire)={pf:.3f}  smoothed={sm:.3f}  {lv}")
            if not a.no_window:
                col = (0, 160, 0) if lv == "LOW" else (0, 200, 255) if lv == "WARNING" else (0, 0, 255)
                out = img.copy(); cv2.rectangle(out, (0, 0), (out.shape[1], 70), (0, 0, 0), -1)
                cv2.putText(out, f"fire {sm:.2f}  {lv}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                names = " ".join(f"{n}:{v:.2f}" for n, v in zip(cnn.class_names, p))
                cv2.putText(out, names, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("fire CNN (q = quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if t_end and time.monotonic() > t_end:
                break
    except KeyboardInterrupt:
        pass


def cmd_record(a):
    import cv2
    out = os.path.join(a.out, a.label); os.makedirs(out, exist_ok=True)
    start = len(os.listdir(out)); n = 0
    t_end = time.monotonic() + a.minutes * 60 if a.minutes else None
    print(f"recording '{a.label}' to {out} at {a.fps} frames/s for {a.minutes} min (Ctrl+C or q to stop early)")
    try:
        for t, img in frames(a.source, a.fps):
            cv2.imwrite(os.path.join(out, f"{start + n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
            if a.preview:
                cv2.imshow("recording (q = stop)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 120 == 0:
                print(f"  {n} frames ({n / a.fps / 60:.1f} min)")
            if t_end and time.monotonic() > t_end:
                break
    except KeyboardInterrupt:
        pass
    print(f"saved {n} frames ({n / a.fps / 60:.1f} min)")


def probs_for(cnn, folder):
    import cv2
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    if not files:
        raise SystemExit(f"no images in {folder}")
    ps = []
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(folder, f))
        ps.append(fire_p(cnn, img)[1] if img is not None else 0.0)
        if (i + 1) % 300 == 0:
            print(f"  scored {i + 1}/{len(files)}")
    return files, np.array(ps)


def cmd_report(a):
    cnn = load_cnn(a.model_dir); rep = {"fps": a.fps, "thresholds": list(THRESH), "criteria": {}}
    neg_dir, pos_dir = os.path.join(a.out, "normal"), os.path.join(a.out, "fire")
    verdict_warn = verdict_crit = None
    if os.path.isdir(neg_dir):
        files, p = probs_for(cnn, neg_dir); sm = smooth(p); hours = len(p) / a.fps / 3600.0
        rows = [{"threshold": t, "frames_above": int((sm >= t).sum()), "episodes": episodes(sm >= t), "episodes_per_hour": round(episodes(sm >= t) / hours, 2)} for t in THRESH]
        rep["normal"] = {"frames": len(p), "minutes": round(hours * 60, 1), "max_p_fire_smoothed": round(float(sm.max()), 4), "by_threshold": rows}
        enough = hours >= 0.4
        verdict_warn = (rows[0]["episodes_per_hour"] <= a.max_warn_per_hour) if enough else None
        verdict_crit = (rows[1]["episodes"] == 0) if enough else None
        worst = np.argsort(-sm)[:12]; fa = os.path.join(a.out, "report", "highest_scoring_normal_frames"); os.makedirs(fa, exist_ok=True)
        for r, i in enumerate(worst):
            shutil.copy(os.path.join(neg_dir, files[i]), os.path.join(fa, f"{r:02d}_p{sm[i]:.2f}_{files[i]}"))
        print(f"\nNORMAL footage: {len(p)} frames = {hours * 60:.1f} min; highest smoothed P(fire) = {sm.max():.3f}")
        for r in rows:
            print(f"  smoothed >= {r['threshold']:.2f}: {r['frames_above']:5d} frames, {r['episodes']:3d} alert episodes = {r['episodes_per_hour']} per hour")
        print(f"  the 12 highest-scoring normal frames were copied to {fa} - LOOK at them: they are what the model confuses with fire")
        if hours < 0.4:
            print("  NOTE: under 24 minutes of normal footage: too short for a reliable false-alarm rate")
    else:
        print(f"no normal footage in {neg_dir} (record it first)")
    detected = None
    if os.path.isdir(pos_dir):
        files, p = probs_for(cnn, pos_dir); sm = smooth(p)
        rows = [{"threshold": t, "frame_recall": round(float((sm >= t).mean()), 4), "detected": bool((sm >= t).any()),
                 "first_detection_s": round(float(np.argmax(sm >= t) / a.fps), 1) if (sm >= t).any() else None} for t in THRESH]
        rep["fire"] = {"frames": len(p), "max_p_fire_smoothed": round(float(sm.max()), 4), "by_threshold": rows}
        detected = rows[0]["detected"] and rows[0]["frame_recall"] >= a.min_fire_recall
        print(f"\nFIRE footage: {len(p)} frames; highest smoothed P(fire) = {sm.max():.3f}")
        for r in rows:
            print(f"  smoothed >= {r['threshold']:.2f}: {r['frame_recall'] * 100:5.1f}% of frames, first at {r['first_detection_s']} s")
    else:
        print(f"no fire footage in {pos_dir} (record it first)")
    rep["criteria"] = {"max_warning_episodes_per_hour": a.max_warn_per_hour, "min_fire_frame_recall_at_0.40": a.min_fire_recall,
                       "engine_policy": "CNN alone is capped below CRITICAL; independent gas, temperature or flame evidence is required for a CNN critical override",
                       "cnn_high_score_threshold": 0.70}
    print("\nVERDICT (suggested criteria, change them with the options if your supervisor requires other values)")
    if "normal" in rep and verdict_warn is None:
        print("  false alarms: NO VERDICT - record at least 24 minutes of normal footage first")
    if verdict_warn is not None:
        print(f"  false alarms at WARNING (0.40): {'OK' if verdict_warn else 'TOO MANY'} (limit {a.max_warn_per_hour} episodes per hour)")
        print(f"  CNN-only high-score episodes (0.70): {'NONE - OK' if verdict_crit else 'PRESENT - engine is guarded; keep independent sensor corroboration required'}")
    if detected is not None:
        print(f"  reacts to fire at WARNING (0.40) on at least {a.min_fire_recall * 100:.0f}% of frames: {'YES' if detected else 'NO'}")
    rep["verdict"] = {"warning_false_alarms_ok": verdict_warn, "no_critical_false_alarm": verdict_crit, "reacts_to_fire": detected}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "webcam_validation.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(a.out, 'webcam_validation.json')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--model-dir", default="models"); p.add_argument("--source", default="0"); p.add_argument("--fps", type=float, default=2.0)
        p.add_argument("--out", default="data/webcam")
    l = sub.add_parser("live"); common(l); l.add_argument("--no-window", action="store_true"); l.add_argument("--minutes", type=float, default=0); l.add_argument("--log", default=None)
    r = sub.add_parser("record"); common(r); r.add_argument("--label", required=True, choices=["normal", "fire"]); r.add_argument("--minutes", type=float, default=30); r.add_argument("--preview", action="store_true")
    p = sub.add_parser("report"); common(p); p.add_argument("--max-warn-per-hour", type=float, default=2.0); p.add_argument("--min-fire-recall", type=float, default=0.5)
    a = ap.parse_args(argv)
    {"live": cmd_live, "record": cmd_record, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
