"""HTTP sensor input (WBS 2.4.6): an ESP32 (or anything) POSTs readings to /api/ingest.

QueueSensorReader looks like SerialSensorReader to the pipeline. Commands for the device (RELAY:0, BUZZER:1 ...)
are returned in the HTTP reply, so the device needs no inbound connection. If no reading arrives within
`timeout` seconds the reader reports a link:no_data FAULT (never a stale value), which the decision engine
turns into a persistent-fault WARNING and the runner into a fail-safe shutdown command.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Optional

from src.sensor_reader import Reading, build_reading


class QueueSensorReader:
    def __init__(self, timeout: float = 2.0, clock=time.time):
        self.q = queue.Queue(maxsize=50)
        self.timeout, self.clock = timeout, clock
        self.commands = deque(maxlen=20)
        self._lock = threading.Lock()
        self.last_ingest: Optional[float] = None
        self.port = "http-ingest"

    def put(self, fields: dict) -> None:
        reading = build_reading({k.upper(): v for k, v in fields.items()}, "http", self.clock())
        try:
            self.q.put_nowait(reading)
        except queue.Full:                      # drop the oldest, keep the freshest
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(reading)
        self.last_ingest = self.clock()

    def read(self) -> Reading:
        try:
            return self.q.get(timeout=self.timeout)
        except queue.Empty:
            return Reading(timestamp=self.clock(), faults=["link:no_data"], source="http")

    def send(self, command: str) -> None:
        with self._lock:
            self.commands.append(command)

    def take_commands(self) -> list:
        with self._lock:
            out = list(self.commands)
            self.commands.clear()
            return out

    def close(self) -> None:
        pass
