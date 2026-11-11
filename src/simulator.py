"""Simulated sensors so all software can be built and tested before hardware arrives."""
from __future__ import annotations

import random
import time

from src.sensor_reader import Reading, build_reading

SCENARIOS = ("normal", "gas_leak", "overheat", "fire")


class SimulatedSensorReader:
    def __init__(self, scenario: str = "normal", dt: float = 0.5, seed: int = 42):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scenario = scenario
        self.dt = dt
        self.rng = random.Random(seed)
        self.t = 0.0
        self.t0 = time.time()  # virtual clock: timestamp = t0 + t
        self.commands = []

    def _profile(self, t: float):
        gas, temp, hum, flame = 180.0, 27.0, 55.0, 0
        if t >= 10:
            e = t - 10
            if self.scenario == "gas_leak":
                gas += min(520, e * 18)
            elif self.scenario == "overheat":
                temp += min(45, e * 0.8)
                hum -= min(20, e * 0.3)
            elif self.scenario == "fire":
                temp += min(35, e * 1.0)
                gas += min(250, e * 8)
                flame = 1 if t >= 20 else 0
        return gas, temp, hum, flame

    def read(self) -> Reading:
        gas, temp, hum, flame = self._profile(self.t)
        fields = {
            "GAS": int(max(0, min(1023, gas + self.rng.gauss(0, 6)))),
            "TEMP": round(temp + self.rng.gauss(0, 0.2), 2),
            "HUM": round(hum + self.rng.gauss(0, 0.8), 1),
            "FLAME": flame,
        }
        ts = self.t0 + self.t
        self.t += self.dt
        return build_reading(fields, "simulation", ts)

    def send(self, command: str) -> None:
        self.commands.append(command)

    def close(self) -> None:
        pass
