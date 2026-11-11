"""Wires M1 -> M2 (optional AI) -> M3 -> M4 -> storage (WBS 9.7 end-to-end flow)."""
from __future__ import annotations

from typing import Callable, Optional

from src.decision_engine import RANK, DecisionEngine, Level, Signals
from src.response import ResponseManager
from src.storage import Storage


class HazardSystem:
    def __init__(self, reader, engine: Optional[DecisionEngine] = None,
                 responder: Optional[ResponseManager] = None,
                 storage: Optional[Storage] = None,
                 signals_provider: Optional[Callable] = None):
        self.reader = reader
        self.engine = engine or DecisionEngine()
        self.responder = responder or ResponseManager(reader)
        self.storage = storage or Storage()
        self.signals_provider = signals_provider   # reading -> Signals (AI models plug in here)
        self._prev_level = Level.SAFE
        self._prev_early = False
        self.last_decision = None

    def start(self) -> None:
        self.responder.arm()
        self.storage.log_system("start")

    def step(self):
        reading = self.reader.read()
        signals = self.signals_provider(reading) if self.signals_provider else Signals()
        decision = self.engine.decide(reading, signals)
        executed = self.responder.execute(decision)
        self.storage.log_cycle(reading, decision)

        level_changed = decision.level != self._prev_level
        early_rise = decision.early_warning and not self._prev_early
        if level_changed or early_rise:
            self.storage.log_event(decision)
        escalated = RANK[decision.level] > RANK[self._prev_level] and decision.level != Level.SAFE
        if escalated or early_rise:
            kind = decision.level.value if escalated else "EARLY_WARNING"
            self.storage.add_alert(decision.timestamp, kind, "; ".join(decision.reasons))

        self._prev_level = decision.level
        self._prev_early = decision.early_warning
        self.last_decision = decision
        return reading, decision, executed

    def stop(self) -> None:
        self.storage.log_system("stop")
        self.reader.close()
