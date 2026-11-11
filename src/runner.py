"""Runs the monitoring loop in a background thread for the dashboard."""
from __future__ import annotations

import threading
import time

from src import config

MAX_CONSECUTIVE_ERRORS = 20


class LiveState:
    def __init__(self, mode: str = "unknown", ai: list | None = None):
        self.lock = threading.Lock()
        self.mode, self.ai = mode, list(ai or [])
        self.reading = self.decision = None
        self.executed = []
        self.started = time.time()
        self.steps = self.errors = 0
        self.last_error = None
        self.crashed = False

    def update(self, reading, decision, executed) -> None:
        with self.lock:
            self.reading, self.decision, self.executed = reading, decision, list(executed)
            self.steps += 1

    def error(self, exc: Exception) -> None:
        with self.lock:
            self.errors += 1
            self.last_error = f"{exc.__class__.__name__}: {exc}"


class Runner(threading.Thread):
    def __init__(self, system, live: LiveState, cycle: float, pace: bool, on_step=None,
                 prune_every: float = config.PRUNE_EVERY_S, keep_rows: int = config.READINGS_KEEP_ROWS):
        super().__init__(daemon=True)
        self.system, self.live, self.cycle, self.pace, self.on_step = system, live, cycle, pace, on_step
        self.prune_every, self.keep_rows = prune_every, keep_rows
        self._stop_evt = threading.Event()

    def run(self) -> None:
        self.system.start()
        consecutive, last_prune = 0, time.time()
        while not self._stop_evt.is_set():
            t0 = time.time()
            if t0 - last_prune >= self.prune_every:
                last_prune = t0
                try:
                    n = self.system.storage.prune_readings(self.keep_rows)
                    if n:
                        self.system.storage.log_system("prune", f"deleted {n} old sensor readings")
                except Exception as exc:
                    self.live.error(exc)
            try:
                r, d, ex = self.system.step()
                self.live.update(r, d, ex)
                consecutive = 0
                if self.on_step:
                    self.on_step(r, d, ex)
            except Exception as exc:
                consecutive += 1
                self.live.error(exc)
                self.system.storage.log_system("runner_error", f"{exc.__class__.__name__}: {exc}")
                if consecutive >= MAX_CONSECUTIVE_ERRORS:
                    self._fail_safe()
                    return
            if self.pace:
                self._stop_evt.wait(max(0.0, self.cycle - (time.time() - t0)))

    def _fail_safe(self) -> None:
        self.live.crashed = True
        try:
            self.system.responder.link.send("RELAY:0")
            self.system.storage.log_system(
                "relay_shutdown_sent",
                "RELAY:0 serial write completed; firmware acknowledgement not available",
            )
        except Exception as exc:
            self.live.error(exc)
            try:
                self.system.storage.log_system(
                    "relay_shutdown_failed",
                    f"RELAY:0 command could not be sent: {exc.__class__.__name__}: {exc}",
                )
            except Exception:
                pass
        try:
            self.system.storage.log_system("runner_crashed", "monitoring stopped; relay shutdown command attempted")
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=5)
