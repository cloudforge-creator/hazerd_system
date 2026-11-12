"""M4 alert and control (WBS 7.x): buzzer, notifications, latched relay shutdown.

Relay convention: relay ENERGIZED = machine powered.  arm() powers the machine at
start-up, so if the PC software dies the Arduino boots with the relay off (fail-safe).
RELAY_SHUTDOWN is LATCHED: it stays off until reset_latch() is called while SAFE.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Optional

from src import config
from src.decision_engine import RANK, Decision, Level

log = logging.getLogger("response")


class LogNotifier:
    """Placeholder channel; replace/add Telegram, e-mail or Firebase later."""

    def __init__(self):
        self.messages = deque(maxlen=1000)   # bounded: runs for weeks

    def send(self, subject: str, body: str) -> None:
        self.messages.append((subject, body))
        log.warning("NOTIFY %s | %s", subject, body)


class ResponseManager:
    def __init__(self, link, notifiers=None, clock: Callable[[], float] = time.time,
                 notify_cooldown: float = config.NOTIFY_COOLDOWN_S):
        self.link = link
        self.notifiers = list(notifiers) if notifiers is not None else [LogNotifier()]
        self.clock = clock
        self.cooldown = notify_cooldown
        self.buzzer_on = False
        self.relay_latched = False
        self.executed = deque(maxlen=1000)   # response confirmation log (WBS 7.4.3), bounded
        self._notified_rank = 0
        self._last_notify = float("-inf")
        self._last_emergency = float("-inf")

    def arm(self) -> None:
        """Power the machine (relay on). Called once at start-up."""
        heartbeat = getattr(self.link, "heartbeat", None)
        if callable(heartbeat):
            heartbeat()
        self.link.send("RELAY:1")
        self.relay_latched = False

    def reset_latch(self, current_level: Level) -> bool:
        """Manual reset after a shutdown. Refused unless the system is SAFE."""
        if current_level != Level.SAFE:
            return False
        self.arm()
        return True

    def _record(self, action: str, ts: float) -> None:
        self.executed.append({"time": ts, "action": action, "sent": True})

    def execute(self, d: Decision) -> list:
        done = []
        now = self.clock()

        if "BUZZER_ON" in d.actions and not self.buzzer_on:
            self.link.send("BUZZER:1")
            self.buzzer_on = True
            done.append("BUZZER_ON")
        if d.level == Level.SAFE and not d.early_warning and self.buzzer_on:
            self.link.send("BUZZER:0")
            self.buzzer_on = False
            done.append("BUZZER_OFF")

        if "RELAY_SHUTDOWN" in d.actions and not self.relay_latched:
            self.link.send("RELAY:0")
            self.relay_latched = True
            done.append("RELAY_SHUTDOWN")

        if "NOTIFY_ADMIN" in d.actions:
            rank = max(RANK[d.level], 1 if d.early_warning else 0)
            if rank > self._notified_rank or now - self._last_notify >= self.cooldown:
                subject = f"[{d.level.value}] hazard system"
                body = f"score={d.score:.2f} predicted={d.predicted_score:.2f} " + "; ".join(d.reasons)
                for n in self.notifiers:
                    n.send(subject, body)
                self._notified_rank = max(self._notified_rank, rank)
                self._last_notify = now
                done.append("NOTIFY_ADMIN")
        elif d.level == Level.SAFE and not d.early_warning:
            self._notified_rank = 0

        if "NOTIFY_EMERGENCY" in d.actions and now - self._last_emergency >= config.EMERGENCY_COOLDOWN_S:
            who = next((r for r in d.reasons if r.startswith("people_in_hazard_zone=")), "people in hazard zone")
            for n in self.notifiers:
                n.send("[EMERGENCY] person in hazard zone", f"system={d.level.value} score={d.score:.2f} " + who)
            self._last_emergency = now
            done.append("NOTIFY_EMERGENCY")

        for a in done:
            self._record(a, now)
        return done
