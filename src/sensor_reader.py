"""M1 data acquisition: parsing, validation and the serial source."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from src import config


@dataclass
class Reading:
    timestamp: float
    gas_raw: Optional[int] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    flame: Optional[int] = None
    faults: list = field(default_factory=list)
    source: str = "unknown"

    @property
    def valid(self) -> bool:
        return not self.faults


def parse_line(line: str) -> dict:
    out = {}
    for part in line.strip().split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        out[key.strip().upper()] = value.strip()
    return out


def _to_float(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def validate(reading: Reading) -> list:
    faults = []
    for name, (lo, hi) in config.VALID_RANGES.items():
        value = getattr(reading, name)
        if value is None:
            faults.append(f"{name}:missing")
        elif not lo <= value <= hi:
            faults.append(f"{name}:out_of_range")
    return faults


def _to_integer(value: Optional[float]) -> Optional[int]:
    if value is None or not value.is_integer():
        return None
    return int(value)


def build_reading(fields: dict, source: str, timestamp: Optional[float] = None) -> Reading:
    gas = _to_float(fields.get("GAS"))
    temp = _to_float(fields.get("TEMP"))
    hum = _to_float(fields.get("HUM"))
    flame = _to_float(fields.get("FLAME"))
    gas_int, flame_int = _to_integer(gas), _to_integer(flame)
    reading = Reading(
        timestamp=time.time() if timestamp is None else timestamp,
        gas_raw=gas_int,
        temperature=temp,
        humidity=hum,
        flame=flame_int,
        source=source,
    )
    reading.faults = validate(reading)
    if gas is not None and gas_int is None:
        reading.faults.append("gas_raw:not_integer")
    if flame is not None and flame_int is None:
        reading.faults.append("flame:not_integer")
    return reading


class SerialSensorReader:
    """Reads the Arduino; serial mode never silently falls back to simulation."""

    def __init__(self, port: Optional[str] = None, baud: int = config.SERIAL_BAUD,
                 timeout: float = 2.0):
        import serial
        from serial.tools import list_ports

        if port is None:
            ports = list(list_ports.comports())
            if not ports:
                raise RuntimeError("No serial port found. Connect the board or use --sim.")
            port = ports[0].device
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def read(self) -> Reading:
        raw = self.ser.readline().decode("ascii", errors="ignore")
        now = time.time()
        if not raw.strip():
            return Reading(timestamp=now, faults=["link:no_data"], source="serial")
        reading = build_reading(parse_line(raw), "serial", now)
        if reading.valid:
            self.send("HEARTBEAT")
        return reading

    def send(self, command: str) -> None:
        self.ser.write((command + "\n").encode("ascii"))

    def close(self) -> None:
        self.ser.close()
