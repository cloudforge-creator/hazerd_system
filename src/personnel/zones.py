"""Camera zones in NORMALISED coordinates (0..1), so they survive a change of camera resolution.

A person's position is the centre of their detected FACE, so a zone must cover where heads appear
(not the floor). Zone fields:
  hazard_when_alert     people inside are flagged while the system is WARNING/CRITICAL
  allowed_designations  if non-empty, anyone else (including Unknown) inside is an unauthorised entry
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

MAX_ZONES = 20


@dataclass
class Zone:
    name: str
    points: list
    hazard_when_alert: bool = True
    allowed_designations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "points": [list(p) for p in self.points],
                "hazard_when_alert": self.hazard_when_alert, "allowed_designations": list(self.allowed_designations)}


def point_in_polygon(x: float, y: float, pts: list) -> bool:
    inside, n, j = False, len(pts), len(pts) - 1
    for i in range(n):
        xi, yi, xj, yj = pts[i][0], pts[i][1], pts[j][0], pts[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def validate_zones(raw) -> tuple:
    """raw: list of dicts (e.g. parsed JSON). Returns (zones, errors); nothing is accepted if there are errors."""
    errors, zones, seen = [], [], set()
    if not isinstance(raw, list):
        return [], ["zones must be a JSON list"]
    if len(raw) > MAX_ZONES:
        errors.append(f"at most {MAX_ZONES} zones")
    for i, z in enumerate(raw[:MAX_ZONES]):
        tag = f"zone {i + 1}"
        if not isinstance(z, dict):
            errors.append(f"{tag}: must be an object")
            continue
        name = z.get("name")
        if not isinstance(name, str) or not (1 <= len(name.strip()) <= 40) or any(ord(c) < 32 for c in name):
            errors.append(f"{tag}: name must be 1-40 characters")
            continue
        name = name.strip()
        tag = f"zone '{name}'"
        if name.lower() in seen:
            errors.append(f"{tag}: duplicate name")
        seen.add(name.lower())
        pts = z.get("points")
        ok = isinstance(pts, list) and len(pts) >= 3 and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 for v in p)
            for p in pts)
        if not ok:
            errors.append(f"{tag}: needs at least 3 points, each [x, y] between 0 and 1")
            continue
        des = z.get("allowed_designations", [])
        if not isinstance(des, list) or not all(isinstance(d, str) and 1 <= len(d.strip()) <= 60 for d in des):
            errors.append(f"{tag}: allowed_designations must be a list of short strings")
            continue
        hz = z.get("hazard_when_alert", True)
        if not isinstance(hz, bool):
            errors.append(f"{tag}: hazard_when_alert must be true or false")
            continue
        zones.append(Zone(name, [(float(p[0]), float(p[1])) for p in pts], hz, [d.strip() for d in des]))
    return ([], errors) if errors else (zones, [])


def load_zones(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        zones, errors = validate_zones(json.load(open(path)))
    except (OSError, ValueError):
        return []
    return zones if not errors else []


def save_zones(path: str, zones: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump([z.to_dict() for z in zones], open(path, "w"), indent=2)


class ZoneHolder:
    """Shared, live-editable zone list: the dashboard saves into it, the monitor reads from it."""

    def __init__(self, path: str):
        self.path = path
        self.zones = load_zones(path)

    def __call__(self) -> list:
        return self.zones

    def set(self, zones: list) -> None:
        save_zones(self.path, zones)
        self.zones = list(zones)
