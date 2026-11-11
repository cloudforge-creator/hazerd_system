"""One thread owns the camera; everything else (fire CNN, personnel, live view) reads the latest frame.

Two consumers must never read one webcam directly (they would steal frames from each other).
`hub.read()` has the same shape as FrameSource.read(), so FireSignals can use the hub unchanged.
A frame older than `stale_s` is treated as no frame, so a frozen camera is never mistaken for a live one.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

MAX_STREAM_CLIENTS = 3


class CameraHub:
    def __init__(self, source, fps: float = 10.0, stale_s: float = 2.0, clock=time.time):
        self.source, self.fps, self.stale_s, self.clock = source, fps, stale_s, clock
        self._lock = threading.Lock()
        self._frame, self._ts = None, 0.0
        self._annotated: Optional[bytes] = None
        self._annotated_ts = 0.0
        self._stop = threading.Event()
        self.frames = self.failures = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.streams = 0
        self._streams_lock = threading.Lock()

    def start(self) -> "CameraHub":
        self._thread.start()
        return self

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self.source.read()
            except Exception:
                frame = None
            if frame is None:
                self.failures += 1
            else:
                with self._lock:
                    self._frame, self._ts = frame, self.clock()
                self.frames += 1
            self._stop.wait(max(0.0, period - (time.time() - t0)))

    def latest(self):
        """(frame copy, timestamp) or (None, 0.0) if there is no fresh frame."""
        with self._lock:
            if self._frame is None or self.clock() - self._ts > self.stale_s:
                return None, 0.0
            return self._frame.copy(), self._ts

    def read(self):
        return self.latest()[0]

    def set_annotated(self, frame_bgr) -> None:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._annotated, self._annotated_ts = buf.tobytes(), self.clock()

    def jpeg(self, prefer_annotated: bool = True) -> Optional[bytes]:
        """Annotated frame if the personnel monitor is feeding one recently, else the raw frame as JPEG."""
        import cv2
        with self._lock:
            if prefer_annotated and self._annotated is not None and self.clock() - self._annotated_ts < self.stale_s:
                return self._annotated
        frame, _ = self.latest()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None

    def acquire_stream(self) -> bool:
        with self._streams_lock:
            if self.streams >= MAX_STREAM_CLIENTS:
                return False
            self.streams += 1
            return True

    def release_stream(self) -> None:
        with self._streams_lock:
            self.streams = max(0, self.streams - 1)

    def status(self) -> dict:
        _, ts = self.latest()
        return {"running": self._thread.is_alive() and not self._stop.is_set(), "frames": self.frames,
                "failures": self.failures, "frame_age_s": round(self.clock() - ts, 2) if ts else None,
                "stream_clients": self.streams}

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        close = getattr(self.source, "close", None)
        if close:
            close()
