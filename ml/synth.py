"""Randomized, labeled scenario generator (bootstrap training data).

IMPORTANT: this is SYNTHETIC. Metrics measured on it prove the pipeline works and
quantify robustness to the effects modelled here; they are NOT field accuracy.
Retrain on logged hardware data (data/logs/sensor_log.csv) once the sensors arrive.

Ground-truth state comes from the TRUE hazard (noise- and humidity-free gas level,
true temperature, true flame), never from the observed sensor values.
"""
from __future__ import annotations

import numpy as np

from src import config

DT = config.CYCLE_SECONDS
STEPS = 300                       # 150 s per run
SCENARIOS = ("normal", "humid", "spray", "gas_leak", "slow_leak", "overheat", "fire")
PROBS = (0.25, 0.12, 0.13, 0.15, 0.08, 0.14, 0.13)

HUM_COEF = 2.0     # raw ADC units per %RH above 55 (assumed MQ-2 cross-sensitivity)
TEMP_COEF = 1.5    # raw ADC units per deg C above 27 (assumed)


def true_state(gas_hazard, temp_true, flame_true) -> np.ndarray:
    g = np.clip((gas_hazard - config.GAS_SAFE_MAX) / (config.GAS_CRITICAL - config.GAS_SAFE_MAX), 0, 1)
    t = np.clip((temp_true - config.TEMP_SAFE_MAX) / (config.TEMP_CRITICAL - config.TEMP_SAFE_MAX), 0, 1)
    s = np.maximum(g, t)
    state = np.where(s >= config.CRITICAL_THRESHOLD, 2, np.where(s >= config.WARNING_THRESHOLD, 1, 0))
    return np.where(flame_true == 1, 2, state).astype(np.int8)


def generate_run(rng: np.random.Generator, scenario: str, steps: int = STEPS) -> dict:
    t = np.arange(steps) * DT
    base_gas = rng.uniform(120, 260)
    base_temp = rng.uniform(20, 36)
    base_hum = rng.uniform(30, 80)
    onset = rng.uniform(15, 80)
    ramp = np.maximum(t - onset, 0.0)

    temp_true = base_temp + rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * t / rng.uniform(60, 200))
    hum = base_hum + rng.uniform(1, 4) * np.sin(2 * np.pi * t / rng.uniform(60, 200))
    gas_hazard = np.full(steps, base_gas)
    spike = np.zeros(steps)
    flame_true = np.zeros(steps, dtype=int)

    if scenario == "humid":
        hum = hum + np.minimum(rng.uniform(0.3, 1.0) * ramp, rng.uniform(15, 35))
    elif scenario == "spray":      # benign transient: must NOT count as a hazard
        amp, tau = rng.uniform(60, 160), rng.uniform(2, 6)
        spike = np.where(t >= onset, amp * np.exp(-(t - onset) / tau), 0.0)
    elif scenario == "gas_leak":
        gas_hazard = base_gas + np.minimum(rng.uniform(6, 30) * ramp, rng.uniform(280, 700))
    elif scenario == "slow_leak":
        gas_hazard = base_gas + np.minimum(rng.uniform(1, 4) * ramp, rng.uniform(120, 260))
    elif scenario == "overheat":
        temp_true = temp_true + np.minimum(rng.uniform(0.1, 0.9) * ramp, rng.uniform(15, 50))
        hum = hum - np.minimum(0.2 * ramp, 15)
    elif scenario == "fire":
        temp_true = temp_true + np.minimum(rng.uniform(0.3, 1.2) * ramp, rng.uniform(20, 45))
        gas_hazard = gas_hazard + np.minimum(rng.uniform(3, 12) * ramp, rng.uniform(100, 260))
        flame_true = (t >= onset + rng.uniform(5, 25)).astype(int)
    elif scenario != "normal":
        raise ValueError(scenario)

    temp_true = np.minimum(temp_true, 79.0)
    hum = np.clip(hum, 5, 98)
    state = true_state(gas_hazard, temp_true, flame_true)

    drift = np.cumsum(rng.normal(0, 0.3, steps))
    gas_obs = (gas_hazard + spike + HUM_COEF * (hum - 55) + TEMP_COEF * (temp_true - 27)
               + drift + rng.normal(0, rng.uniform(4, 10), steps))
    temp_obs = temp_true + rng.normal(0, rng.uniform(0.1, 0.3), steps)
    hum_obs = hum + rng.normal(0, rng.uniform(0.5, 1.2), steps)
    flame_obs = (flame_true & (rng.random(steps) > 0.03)).astype(int)   # 3% drop-outs

    return {
        "scenario": scenario,
        "gas": np.clip(np.rint(gas_obs), 0, 1023),
        "temp": np.clip(np.round(temp_obs, 2), -10, 80),
        "hum": np.clip(np.round(hum_obs, 1), 0, 100),
        "flame": flame_obs,
        "state": state,
    }


def generate_runs(n_runs: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(SCENARIOS), size=n_runs, p=PROBS)
    return [generate_run(np.random.default_rng([seed, i]), SCENARIOS[s])
            for i, s in enumerate(picks)]
