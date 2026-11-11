"""M1 data acquisition: parsing, validation and the serial source.

Serial line format sent by the Arduino (one line per cycle):
    GAS:312,TEMP:27.4,HUM:55.0,FLAME:0
"""
from __future__ import annotations

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
    """'GAS:312,TEMP:27.4' -> {'GAS': '312', 'TEMP': '27.4'}"""
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
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN -> None


def validate(reading: Reading) -> list:
    """Return a list of fault strings; empty list means the reading is valid."""
    faults = []
    for name, (lo, hi) in config.VALID_RANGES.items():
        value = getattr(reading, name)
        if value is None:
            faults.append(f"{name}:missing")
        elif not lo <= value <= hi:
            faults.append(f"{name}:out_of_range")
    return faults


def build_reading(fields: dict, source: str, timestamp: Optional[float] = None) -> Reading:
    gas = _to_float(fields.get("GAS"))
    temp = _to_float(fields.get("TEMP"))
    hum = _to_float(fields.get("HUM"))
    flame = _to_float(fields.get("FLAME"))
    reading = Reading(
        timestamp=time.time() if timestamp is None else timestamp,
        gas_raw=int(gas) if gas is not None else None,
        temperature=temp,
        humidity=hum,
        flame=int(flame) if flame is not None else None,
        source=source,
    )
    reading.faults = validate(reading)
    return reading


class SerialSensorReader:
    """Reads the Arduino. If the board is missing it RAISES - it never silently
    falls back to simulation, so you always know which mode you are in."""

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
        time.sleep(2.0)  # Arduino resets when the port opens
        self.ser.reset_input_buffer()

    def read(self) -> Reading:
        raw = self.ser.readline().decode("ascii", errors="ignore")
        now = time.time()
        if not raw.strip():
            return Reading(timestamp=now, faults=["link:no_data"], source="serial")
        return build_reading(parse_line(raw), "serial", now)

    def send(self, command: str) -> None:
        """e.g. 'RELAY:1', 'RELAY:0', 'BUZZER:1', 'BUZZER:0'"""
        self.ser.write((command + "\n").encode("ascii"))

    def close(self) -> None:
        self.ser.close()
