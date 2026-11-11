"""Admin-editable thresholds (WBS 6.2.3, 8.5). Validated, persisted, audited."""
from __future__ import annotations

import json
import os

from src import config

# name: (type, min, max, description)
TUNABLES = {
    "WARNING_THRESHOLD": (float, 0.05, 0.95, "Fused risk score that raises WARNING"),
    "CRITICAL_THRESHOLD": (float, 0.10, 1.00, "Fused risk score that raises CRITICAL (relay shutdown)"),
    "GAS_SAFE_MAX": (int, 0, 1000, "Gas ADC value at/below which the gas score is 0"),
    "GAS_CRITICAL": (int, 1, 1023, "Gas ADC value at/above which the gas score is 1"),
    "TEMP_SAFE_MAX": (float, 0.0, 100.0, "Temperature (C) at/below which the temperature score is 0"),
    "TEMP_CRITICAL": (float, 1.0, 120.0, "Temperature (C) at/above which the temperature score is 1"),
    "TEMP_RATE_CRITICAL": (float, 0.1, 20.0, "Rise rate (C/s) that gives rate score 1"),
    "FIRE_PROB_OVERRIDE": (float, 0.5, 1.0, "CNN fire probability threshold; CRITICAL override also requires independent gas, temperature or flame evidence"),
    "CLEAR_CYCLES": (int, 1, 120, "Calm cycles needed before the alert level drops"),
    "NOTIFY_COOLDOWN_S": (float, 0.0, 3600.0, "Minimum seconds between repeated notifications"),
    "EMERGENCY_COOLDOWN_S": (float, 0.0, 600.0, "Minimum seconds between repeated EMERGENCY (person in hazard zone) notifications"),
    "CORROBORATION_BONUS": (float, 0.0, 0.5, "Risk added for each extra independent sensor that agrees"),
    "CORROBORATION_MIN": (float, 0.05, 0.9, "A sensor score above this counts as 'agreeing'"),
    "FAULT_CYCLES_LIMIT": (int, 2, 120, "Consecutive faulty cycles before a sensor-fault WARNING"),
    "FACE_MATCH_THRESHOLD": (float, 0.2, 0.9, "Face similarity needed to name a person (higher = fewer wrong names, more 'Unknown')"),
    "FACE_MATCH_MARGIN": (float, 0.0, 0.3, "Best face match must beat the runner-up by this much"),
    "FACE_MIN_SIZE_PX": (int, 30, 300, "Faces narrower than this are tracked but not identified"),
    "PERSON_TRACK_MAX_MISSED": (int, 1, 60, "Frames a person's track survives without a detection"),
}
_ORIGINAL = {k: getattr(config, k) for k in TUNABLES}


def current() -> dict:
    return {k: getattr(config, k) for k in TUNABLES}


def validate(values: dict):
    """Return (clean_values, errors). Only known keys are accepted."""
    clean, errors = {}, []
    for k, raw in values.items():
        if k not in TUNABLES:
            errors.append(f"{k}: unknown setting")
            continue
        typ, lo, hi, _ = TUNABLES[k]
        try:
            v = typ(float(raw)) if typ is int and float(raw) == int(float(raw)) else typ(raw)
        except (TypeError, ValueError):
            errors.append(f"{k}: must be a {typ.__name__}")
            continue
        if not lo <= v <= hi:
            errors.append(f"{k}: must be between {lo} and {hi}")
            continue
        clean[k] = v
    merged = {**current(), **clean}
    if merged["WARNING_THRESHOLD"] >= merged["CRITICAL_THRESHOLD"]:
        errors.append("WARNING_THRESHOLD must be lower than CRITICAL_THRESHOLD")
    if merged["GAS_SAFE_MAX"] >= merged["GAS_CRITICAL"]:
        errors.append("GAS_SAFE_MAX must be lower than GAS_CRITICAL")
    if merged["TEMP_SAFE_MAX"] >= merged["TEMP_CRITICAL"]:
        errors.append("TEMP_SAFE_MAX must be lower than TEMP_CRITICAL")
    return clean, errors


def apply(clean: dict, responder=None) -> dict:
    """Apply validated values live. Returns {name: (old, new)} for what actually changed."""
    changed = {}
    for k, v in clean.items():
        old = getattr(config, k)
        if old != v:
            setattr(config, k, v)
            changed[k] = (old, v)
    if responder is not None and "NOTIFY_COOLDOWN_S" in clean:
        responder.cooldown = clean["NOTIFY_COOLDOWN_S"]
    return changed


def save(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(current(), open(path, "w"), indent=2)


def load_overrides(path: str, responder=None) -> list:
    """Load persisted settings at start-up; invalid files are ignored loudly, never half-applied."""
    if not os.path.exists(path):
        return []
    try:
        clean, errors = validate(json.load(open(path)))
    except (OSError, ValueError) as exc:
        return [f"could not read {path}: {exc}"]
    if errors:
        return [f"ignored {path}: " + "; ".join(errors)]
    apply(clean, responder)
    return []


def reset_to_defaults(responder=None) -> None:
    apply(dict(_ORIGINAL), responder)
