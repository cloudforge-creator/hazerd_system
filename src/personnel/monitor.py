"""Personnel monitor: face detect -> identify -> track -> zone rules -> events -> labelled video.

Runs in its own thread at PERSON_PROCESS_FPS. Safety rules:
  * a person inside a hazard zone while the system is WARNING/CRITICAL is reported to the decision engine
    (-> EMERGENCY notification naming them); this never lowers or raises the hazard level itself.
  * an exception in one frame never kills the thread (counted in `errors`).
  * position = centre of the FACE: a person facing away or hidden is not seen. This is not a replacement
    for physical lock-out / evacuation procedures.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from src import config
from src.decision_engine import Level, Signals
from src.personnel.registry import PersonStore
from src.personnel.tracker import FaceTracker
from src.personnel.zones import point_in_polygon


def ascii_label(text: str) -> str:
    """cv2.putText cannot draw non-Latin characters; the dashboard table shows the full name."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in text)


class PersonnelMonitor(threading.Thread):
    def __init__(self, hub, engine, store: PersonStore, zones_provider: Callable[[], list],
                 level_getter: Callable[[], Level], clock=time.time):
        super().__init__(daemon=True)
        self.hub, self.engine, self.store = hub, engine, store
        self.zones_provider, self.level_getter, self.clock = zones_provider, level_getter, clock
        self.tracker = FaceTracker()
        self._lock = threading.Lock()
        self._snapshot, self._in_hazard = [], []
        self._stop_evt = threading.Event()
        self._last_ts = 0.0
        self._last_frame_at = None
        self._camera_stale = False
        self.frames = self.errors = 0
        self.last_error: Optional[str] = None
        self.process_ms = 0.0

    # ------------------------------------------------------------------ public state
    def snapshot(self) -> list:
        with self._lock:
            return [dict(p) for p in self._snapshot]

    def people_in_hazard(self) -> list:
        with self._lock:
            return list(self._in_hazard)

    def __call__(self, reading) -> Signals:         # Signals provider for the decision engine
        people = self.people_in_hazard()
        return Signals(people_in_hazard=people) if people else Signals()

    def status(self) -> dict:
        return {"running": self.is_alive(), "frames": self.frames, "errors": self.errors,
                "last_error": self.last_error, "process_ms": round(self.process_ms, 1),
                "tracks": len(self.snapshot()), "camera_state": "STALE" if self._camera_stale else "LIVE"}

    def _expire_stale_state(self) -> None:
        """Expire tracks and occupancy when the camera stops producing fresh frames."""
        closed = list(self.tracker.tracks)
        self.tracker.tracks.clear()
        for track in closed:
            if track.person:
                try:
                    self.store.log_event("PERSON_LEFT", track.id, track.person, detail="camera frame became stale")
                except Exception as exc:
                    self.errors += 1
                    self.last_error = f"{exc.__class__.__name__}: {exc}"
        with self._lock:
            self._snapshot, self._in_hazard = [], []
        self._camera_stale = True

    # ------------------------------------------------------------------ thread
    def run(self) -> None:
        period = 1.0 / config.PERSON_PROCESS_FPS
        while not self._stop_evt.is_set():
            t0 = time.time()
            frame, ts = self.hub.latest()
            if frame is not None and ts != self._last_ts:
                self._last_ts = ts
                self._last_frame_at = self.clock()
                self._camera_stale = False
                try:
                    self.process(frame, ts)
                except Exception as exc:
                    self.errors += 1
                    self.last_error = f"{exc.__class__.__name__}: {exc}"
            elif frame is None and self._last_frame_at is not None:
                stale_after = getattr(self.hub, "stale_s", 2.0)
                if self.clock() - self._last_frame_at > stale_after and not self._camera_stale:
                    self._expire_stale_state()
            self._stop_evt.wait(max(0.01, period - (time.time() - t0)))

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=3)

    # ------------------------------------------------------------------ one frame
    def process(self, frame, ts: float) -> list:
        t0 = time.time()
        self._last_frame_at = self.clock()
        self._camera_stale = False
        h, w = frame.shape[:2}
        dets = []
        for d in self.engine.detect_and_embed(frame):
            small = d.box[2] < config.FACE_MIN_SIZE_PX or d.embedding is None
            person, score = (None, 0.0) if small else self.store.match(d.embedding)
            dets.append({"box": d.box, "person": person, "score": score, "small": small})
        tracks, closed = self.tracker.update(dets, ts)

        zones, level = self.zones_provider(), self.level_getter()
        alert = level != Level.SAFE
        hazard_people, snap = [], []
        for t in tracks:
            cx, cy = (t.box[0] + t.box[2] / 2) / w, (t.box[1] + t.box[3] / 2) / h
            now_in = {z.name for z in zones if point_in_polygon(cx, cy, z.points)}
            for name in now_in - t.zones:
                self.store.log_event("ZONE_ENTER", t.id, t.person, name)
            for name in t.zones - now_in:
                self.store.log_event("ZONE_EXIT", t.id, t.person, name)
            t.zones = now_in
            if t.person and "seen" not in t.flags:
                t.flags.add("seen")
                self.store.log_event("PERSON_SEEN", t.id, t.person)
            if t.unknown and "unknown" not in t.flags:
                t.flags.add("unknown")
                self.store.log_event("UNKNOWN_PERSON", t.id, None, detail=f"track {t.id}")
            decided = bool(t.person) or t.unknown
            for z in zones:
                if z.name not in now_in:
                    continue
                if z.allowed_designations and decided and (t.person is None or t.person["designation"] not in z.allowed_designations):
                    if ("unauth", z.name) not in t.flags:
                        t.flags.add(("unauth", z.name))
                        self.store.log_event("UNAUTHORIZED_ZONE", t.id, t.person, z.name,
                                             "unknown person" if t.person is None else "designation not allowed")
                if z.hazard_when_alert and alert:
                    hazard_people.append(t.label())
                    if ("hazard", z.name) not in t.flags:
                        t.flags.add(("hazard", z.name))
                        self.store.log_event("PERSON_IN_HAZARD", t.id, t.person, z.name, f"level={level.value}")
                elif not alert:
                    t.flags.discard(("hazard", z.name))
            snap.append({"track_id": t.id, "name": t.person["name"] if t.person else None,
                         "designation": t.person["designation"] if t.person else None, "status": t.status,
                         "zones": sorted(now_in), "seen_s": round(ts - t.first_ts, 1),
                         "box": [round(t.box[0] / w, 3), round(t.box[1] / h, 3), round(t.box[2] / w, 3), round(t.box[3] / h, 3)]})
        for t in closed:
            if t.person:
                self.store.log_event("PERSON_LEFT", t.id, t.person, detail=f"seen {round(t.last_ts - t.first_ts, 1)} s")
        with self._lock:
            self._snapshot, self._in_hazard = snap, sorted(set(hazard_people))
        self.frames += 1
        self.process_ms = (time.time() - t0) * 1000
        self.hub.set_annotated(self.annotate(frame, tracks, zones, level, alert))
        return snap

    # ------------------------------------------------------------------ drawing
    def annotate(self, frame, tracks, zones, level, alert):
        import cv2
        import numpy as np
        out = frame.copy()
        h, w = out.shape[:2]
        overlay = out.copy()
        for z in zones:
            pts = np.array([[int(x * w), int(y * h)] for x, y in z.points], dtype=np.int32)
            color = (0, 0, 220) if (z.hazard_when_alert and alert) else (0, 160, 0)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], True, color, 2)
            cv2.putText(out, ascii_label(z.name), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
        colors = {"identified": (0, 170, 0), "identifying": (0, 165, 255), "unknown": (0, 0, 220)}
        for t in tracks:
            x, y, bw, bh = t.box
            c = colors[t.status]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)
            label = ascii_label(t.label())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y - th - 8, 0)
            cv2.rectangle(out, (x, top), (x + tw + 6, top + th + 8), c, -1)
            cv2.putText(out, label, (x + 3, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        banner = (0, 0, 200) if level == Level.CRITICAL else ((0, 140, 255) if level == Level.WARNING else (0, 130, 0))
        cv2.rectangle(out, (0, 0), (w, 22), banner, -1)
        cv2.putText(out, f"SYSTEM: {level.value}   people: {len(tracks)}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out
