"""C1-C5 evaluation harness (WBS 10). Every number comes from running the real pipeline.

C1 multi-source hazard detection | C2 predictive early warning | C3 decision latency
C4 automated safety response     | C5 scalability, soak, load and stress

HONESTY NOTES (also written into the report):
* C1/C2/C4 use SYNTHETIC held-out scenarios (different seed from training). They validate the
  software and its robustness to the modelled effects; they are not field accuracy.
* C4 counts relay commands ISSUED. There is no hardware feedback line yet, so physical actuation is
  unconfirmed until the hardware loopback test.
* C3/C5 timings depend on the machine they run on. Quote the numbers from YOUR machine.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import tracemalloc
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ml.synth import DT, generate_run, generate_runs
from ml.train_ml import event_report, first_alarm
from src import config
from src.ai_runtime import MLSignals
from src.decision_engine import RANK, DecisionEngine
from src.pipeline import HazardSystem
from src.response import LogNotifier, ResponseManager
from src.sensor_reader import Reading, build_reading
from src.storage import Storage

logging.getLogger("response").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

TARGETS = {
    "C1_macro_f1": 0.90, "C1_critical_recall": 0.95, "C1_false_critical_clean_runs": 0.05,
    "C2_detection_rate": 0.90, "C2_false_alarm_run_rate": 0.10,
    "C3_p99_step_ms": 500.0,
    "C4_shutdown_within_5s": 0.95, "C4_false_shutdown_clean_runs": 0.05,
    "C5_soak_heap_mb_per_day": 10.0, "C5_http_p95_ms": 250.0,
}


# --------------------------------------------------------------------------- infrastructure
class ArrayReader:
    """Replays a run as a sensor source; records every command with the step it was sent on."""

    def __init__(self, run: dict):
        self.run, self.i, self.sent = run, 0, []

    def read(self) -> Reading:
        i, r = self.i, self.run
        self.i += 1
        return build_reading({"GAS": int(r["gas"][i]), "TEMP": float(r["temp"][i]),
                              "HUM": float(r["hum"][i]), "FLAME": int(r["flame"][i])}, "synthetic", i * DT)

    def send(self, cmd: str) -> None:
        self.sent.append((self.i - 1, cmd))

    def close(self) -> None:
        pass


class FaultyReader:
    """Injects sensor faults: dropped link, NaN, out-of-range value."""

    def __init__(self, inner, seed, p_drop=0.05, p_nan=0.03, p_range=0.02, forced=()):
        self.inner, self.rng, self.forced = inner, np.random.default_rng(seed), set(forced)
        self.p = (p_drop, p_nan, p_range)

    @property
    def i(self):
        return self.inner.i

    @property
    def sent(self):
        return self.inner.sent

    def read(self) -> Reading:
        r = self.inner.read()
        step, u = self.inner.i - 1, self.rng.random()
        if step in self.forced or u < self.p[0]:
            return Reading(timestamp=r.timestamp, faults=["link:no_data"], source="synthetic")
        if u < self.p[0] + self.p[1]:
            return build_reading({"GAS": r.gas_raw, "TEMP": "NAN", "HUM": r.humidity, "FLAME": r.flame},
                                 "synthetic", r.timestamp)
        if u < sum(self.p):
            return build_reading({"GAS": 5000, "TEMP": r.temperature, "HUM": r.humidity, "FLAME": r.flame},
                                 "synthetic", r.timestamp)
        return r

    def send(self, cmd):
        self.inner.send(cmd)

    def close(self):
        pass


def make_system(reader, ml, db_path: str = ":memory:"):
    notifier = LogNotifier()
    responder = ResponseManager(reader, [notifier], clock=lambda: max(reader.i, 0) * DT)
    return HazardSystem(reader, responder=responder, storage=Storage(db_path), signals_provider=ml), notifier


def reset_ml(ml):
    if ml is not None:
        ml.buffer.clear()
        ml._consec = 0


def concat_runs(runs: list) -> dict:
    return {k: np.concatenate([r[k] for r in runs]) for k in ("gas", "temp", "hum", "flame", "state")}


def run_trace(run: dict, ml, reader_wrap=None) -> dict:
    reset_ml(ml)
    inner = ArrayReader(run)
    reader = reader_wrap(inner) if reader_wrap else inner
    system, notifier = make_system(reader, ml)
    system.start()
    n = len(run["state"])
    levels, early, executed = np.zeros(n, int), np.zeros(n, bool), []
    for i in range(n):
        _, d, ex = system.step()
        levels[i], early[i] = RANK[d.level], d.early_warning
        executed.append(ex)
    return {"levels": levels, "early": early, "executed": executed, "sent": inner.sent,
            "scenario": run["scenario"], "state": run["state"], "notifications": len(notifier.messages)}


def lat_stats(seconds) -> dict:
    ms = np.asarray(seconds) * 1000.0
    return {"n": int(len(ms)), "mean_ms": round(float(ms.mean()), 3), "p50_ms": round(float(np.percentile(ms, 50)), 3),
            "p95_ms": round(float(np.percentile(ms, 95)), 3), "p99_ms": round(float(np.percentile(ms, 99)), 3),
            "max_ms": round(float(ms.max()), 3)}


# --------------------------------------------------------------------------- C1
def c1_detection(traces: list) -> dict:
    y = np.concatenate([t["state"] for t in traces]).astype(int)
    p = np.concatenate([t["levels"] for t in traces]).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    names = ["SAFE", "WARNING", "CRITICAL"]
    per = {n: {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4)}
           for n, a, b, c in zip(names, pr, rc, f1)}
    delays, detected, crit_runs, clean_runs, false_crit = [], 0, 0, 0, 0
    for t in traces:
        tc, sc = first_alarm(t["state"] == 2), first_alarm(t["levels"] == 2)
        if tc is None:
            clean_runs += 1
            false_crit += sc is not None
        else:
            crit_runs += 1
            if sc is not None and sc <= tc + int(10 / DT):
                detected += 1
                delays.append((sc - tc) * DT)
    by_sc = {}
    for t in traces:
        d = by_sc.setdefault(t["scenario"], [0, 0])
        d[0] += int((t["state"] == t["levels"]).sum())
        d[1] += len(t["state"])
    return {"accuracy": round(float((y == p).mean()), 4), "macro_f1": round(float(np.mean(f1)), 4),
            "per_class": per, "confusion_matrix(rows=true,cols=pred; SAFE,WARNING,CRITICAL)":
                confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "critical_event_detection_rate": round(detected / max(crit_runs, 1), 4),
            "median_detection_delay_s(negative=earlier than truth)": round(float(np.median(delays)), 2) if delays else None,
            "false_critical_rate_on_clean_runs": round(false_crit / max(clean_runs, 1), 4),
            "runs": {"with_critical": crit_runs, "clean": clean_runs},
            "step_accuracy_by_scenario": {k: round(v[0] / v[1], 4) for k, v in by_sc.items()}}


# --------------------------------------------------------------------------- C2
def c2_early_warning(traces: list) -> dict:
    return event_report(traces, [first_alarm(t["early"]) for t in traces])


# --------------------------------------------------------------------------- C4
def c4_response(traces: list) -> dict:
    out, tol = {}, int(5 / DT)
    groups = {}
    for t in traces:
        groups.setdefault(t["scenario"], []).append(t)
    clean_n = clean_shut = 0
    false_by_sc = {}
    for sc, ts in groups.items():
        crit = [t for t in ts if first_alarm(t["state"] == 2) is not None]
        for t in ts:
            if first_alarm(t["state"] == 2) is None:
                clean_n += 1
                bad = any(c == "RELAY:0" for _, c in t["sent"])
                clean_shut += bad
                f = false_by_sc.setdefault(sc, [0, 0])
                f[0] += bad
                f[1] += 1
        if not crit:
            continue
        shut, within, buz, note, delays = 0, 0, 0, 0, []
        for t in crit:
            tc = first_alarm(t["state"] == 2)
            steps = {c: next((s for s, cc in t["sent"] if cc == c), None) for c in ("RELAY:0", "BUZZER:1")}
            notify = next((i for i, e in enumerate(t["executed"]) if "NOTIFY_ADMIN" in e), None)
            if steps["RELAY:0"] is not None:
                shut += 1
                delays.append((steps["RELAY:0"] - tc) * DT)
                within += steps["RELAY:0"] <= tc + tol
            buz += steps["BUZZER:1"] is not None
            note += notify is not None
        n = len(crit)
        out[sc] = {"hazard_runs": n, "shutdown_issued_rate": round(shut / n, 4),
                   "shutdown_within_5s_of_true_critical": round(within / n, 4),
                   "median_shutdown_delay_s(negative=earlier)": round(float(np.median(delays)), 2) if delays else None,
                   "alarm_rate": round(buz / n, 4), "notification_rate": round(note / n, 4)}
    n_all = sum(v["hazard_runs"] for v in out.values())
    out["ALL_HAZARDS"] = {
        "hazard_runs": n_all,
        "shutdown_within_5s_of_true_critical": round(sum(v["shutdown_within_5s_of_true_critical"] * v["hazard_runs"]
                                                         for k, v in out.items() if k != "ALL_HAZARDS") / max(n_all, 1), 4),
        "false_shutdown_rate_on_clean_runs": round(clean_shut / max(clean_n, 1), 4), "clean_runs": clean_n,
        "false_shutdowns_by_scenario(count/clean_runs)": {k: f"{v[0]}/{v[1]}" for k, v in false_by_sc.items()}}
    out["note"] = "Counts relay commands ISSUED; physical actuation is unconfirmed until the hardware loopback test."
    return out


# --------------------------------------------------------------------------- C3
def c3_latency(ml, steps: int, seed: int) -> dict:
    runs, total, i = [], 0, 0
    while total < steps:
        r = generate_run(np.random.default_rng([seed, 9000 + i]), ("normal", "gas_leak", "fire", "overheat")[i % 4])
        runs.append(r)
        total += len(r["state"])
        i += 1
    big = concat_runs(runs)
    out = {}
    for label, provider in (("rule_only", None), ("rule_plus_ml", ml)):
        if label == "rule_plus_ml" and provider is None:
            continue
        reset_ml(provider)
        with tempfile.TemporaryDirectory() as td:
            reader = ArrayReader(big)
            system, _ = make_system(reader, provider, os.path.join(td, "lat.db"))   # real disk writes
            system.start()
            full = []
            for _ in range(steps):
                t0 = time.perf_counter()
                system.step()
                full.append(time.perf_counter() - t0)
            system.storage.close()
        eng, rd, dec, ai = DecisionEngine(), ArrayReader(big), [], []
        reset_ml(provider)
        for _ in range(steps):
            r = rd.read()
            t0 = time.perf_counter()
            sig = provider(r) if provider else None
            t1 = time.perf_counter()
            eng.decide(r, sig)
            dec.append(time.perf_counter() - t1)
            ai.append(t1 - t0)
        out[label] = {"full_cycle(read+decide+respond+DB)": lat_stats(full), "decision_only": lat_stats(dec),
                      "ai_inference_only": lat_stats(ai)}
    out["cycle_budget_ms"] = config.CYCLE_SECONDS * 1000
    return out


# --------------------------------------------------------------------------- C5
def c5_soak(ml, cycles: int, seed: int) -> dict:
    """Accelerated continuous run. Leak test = SUSTAINED growth of the Python heap in the second half
    of the run, projected to MB per simulated day (a % of a tiny start-up heap is meaningless).
    Latency drift compares ordinary SAFE cycles only, so hazard events with DB writes do not skew it."""
    runs, total = [], 0
    for r in generate_runs(max(cycles // 300 + 2, 4), seed + 1):
        runs.append(r)
        total += len(r["state"])
        if total >= cycles:
            break
    big = concat_runs(runs)
    cycles = min(cycles, len(big["state"]))
    reset_ml(ml)
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "soak.db")
        reader = ArrayReader(big)
        system, _ = make_system(reader, ml, path)
        system.start()
        tracemalloc.start()
        rss0 = proc.memory_info().rss / 1e6 if proc else None
        heap, errors = {}, 0
        times, safe = np.zeros(cycles), np.zeros(cycles, dtype=bool)   # preallocated: harness must not grow the heap
        mark = {max(1, cycles * k // 10): k for k in range(1, 11)}
        t_start = time.perf_counter()
        for c in range(1, cycles + 1):
            t0 = time.perf_counter()
            is_safe = False
            try:
                _, d, ex = system.step()
                is_safe = RANK[d.level] == 0 and not ex
            except Exception:
                errors += 1
            times[c - 1] = time.perf_counter() - t0
            safe[c - 1] = is_safe
            if c in mark:
                heap[mark[c]] = tracemalloc.get_traced_memory()[0]
        elapsed = time.perf_counter() - t_start
        tracemalloc.stop()
        rss1 = proc.memory_info().rss / 1e6 if proc else None
        size = os.path.getsize(path)
        rows = system.storage.count("sensor_readings")
        system.storage.close()
    t, ok = times, safe
    d = max(len(t) // 10, 1)

    from src.features import WINDOW
    first, last = slice(WINDOW, WINDOW + d), slice(len(t) - d, len(t))   # skip the ML warm-up window
    growth_kb = (heap[10] - heap[5]) / 1024
    per_cycle_kb = growth_kb / max(cycles / 2, 1)
    return {"simulated_hours": round(cycles * DT / 3600, 3), "cycles": cycles, "errors": errors,
            "wall_seconds": round(elapsed, 2), "cycles_per_second": round(cycles / elapsed, 1),
            "speedup_vs_real_time": round(cycles * DT / elapsed, 1),
            "safe_cycle_p95_ms_first_10pct_after_warmup": (round(float(np.percentile(t[first][ok[first]], 95)) * 1000, 3) if ok[first].any() else None),
            "safe_cycle_p95_ms_last_10pct": (round(float(np.percentile(t[last][ok[last]], 95)) * 1000, 3) if ok[last].any() else None),
            "python_heap_kb_by_decile": [round(heap[k] / 1024, 1) for k in range(1, 11)],
            "heap_growth_kb_second_half": round(growth_kb, 1),
            "projected_heap_growth_mb_per_day_at_2hz": round(per_cycle_kb * 172800 / 1024, 2),
            "process_rss_mb_start_end": [round(rss0, 1), round(rss1, 1)] if proc else "psutil not installed",
            "db_rows": rows, "db_bytes_per_cycle": round(size / max(cycles, 1), 1),
            "projected_db_mb_per_day_at_2hz": round(size / max(cycles, 1) * 172800 / 1e6, 1),
            "note": "heap = Python allocations only (tracemalloc); sqlite's C memory is not included."}


def c5_sensor_load(counts: list, cycles: int, seed: int) -> dict:
    out, budget = {}, config.CYCLE_SECONDS * 1000
    for k in counts:
        systems = []
        for j in range(k):
            run = generate_run(np.random.default_rng([seed, 5000 + j]), ("normal", "gas_leak")[j % 2])
            s, _ = make_system(ArrayReader(run), None)
            s.start()
            systems.append(s)
        times = []
        for _ in range(cycles):
            t0 = time.perf_counter()
            for s in systems:
                s.step()
            times.append(time.perf_counter() - t0)
        st = lat_stats(times)
        out[str(k)] = {"p95_ms_for_whole_cycle": st["p95_ms"], "max_ms": st["max_ms"],
                       "within_500ms_budget": st["p95_ms"] < budget}
    ok = [int(k) for k, v in out.items() if v["within_500ms_budget"]]
    out["max_nodes_tested_within_budget"] = max(ok) if ok else 0
    out["note"] = "Rule-based pipeline, in-memory DB, one Python process: a software scaling indicator, not a hardware limit."
    return out


def c5_http_load(threads: int, per_thread: int, seed: int) -> dict:
    from werkzeug.serving import make_server

    from src.dashboard import create_app
    from src.runner import LiveState, Runner
    from src.users import UserStore

    sim = generate_run(np.random.default_rng([seed, 1]), "gas_leak", steps=20000)
    reader = ArrayReader(sim)
    system, _ = make_system(reader, None)
    live = LiveState(mode="LOAD TEST")
    UserStore(system.storage).create("loadtest", "load-test-pass-1", "viewer")
    runner = Runner(system, live, cycle=0.01, pace=True)
    app = create_app(system, live, UserStore(system.storage), "load-secret",
                     runtime_config_path=os.path.join(tempfile.gettempdir(), "load_rc.json"))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runner.start()
    time.sleep(0.3)
    base = f"http://127.0.0.1:{port}"
    lat, errors, lock = [], [0], threading.Lock()

    def worker():
        jar = CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        html = op.open(base + "/login").read().decode()
        tok = re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)
        op.open(base + "/login", urllib.parse.urlencode(
            {"username": "loadtest", "password": "load-test-pass-1", "csrf_token": tok}).encode())
        for j in range(per_thread):
            url = base + ("/api/status" if j % 2 == 0 else "/api/history?n=120")
            t0 = time.perf_counter()
            try:
                ok = op.open(url).status == 200
            except Exception:
                ok = False
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt)
                errors[0] += (not ok)

    steps0, t0 = live.steps, time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    steps1 = live.steps
    runner.stop()
    server.shutdown()
    st = lat_stats(lat)
    return {"concurrent_clients": threads, "requests": len(lat), "errors": errors[0],
            "requests_per_second": round(len(lat) / wall, 1), **{f"latency_{k}": v for k, v in st.items() if k != "n"},
            "monitoring_cycles_completed_during_load": steps1 - steps0,
            "monitoring_cycles_expected_at_100hz": int(wall * 100),
            "monitoring_crashed": live.crashed}


def c5_stress(ml, runs_per_scenario: int, seed: int) -> dict:
    out = {}
    # 1) fault storm on normal operation: ~10% of cycles faulty
    normal = generate_run(np.random.default_rng([seed, 7]), "normal", steps=7200)
    reset_ml(ml)
    inner = ArrayReader(normal)
    system, _ = make_system(FaultyReader(inner, seed), ml)
    system.start()
    exceptions = crit = degraded = 0
    for _ in range(7200):
        try:
            _, d, _ = system.step()
            crit += RANK[d.level] == 2
            degraded += d.degraded
        except Exception:
            exceptions += 1
    out["fault_storm_normal_operation"] = {"cycles": 7200, "exceptions": exceptions,
                                           "degraded_cycle_pct": round(degraded / 72, 2), "false_critical_cycles": int(crit)}
    # 2) forced sensor dropout must raise a persistent-fault WARNING, then recover
    forced = range(1000, 1010)
    run = generate_run(np.random.default_rng([seed, 8]), "normal", steps=1400)
    reset_ml(ml)
    inner = ArrayReader(run)
    system, _ = make_system(FaultyReader(inner, seed, 0, 0, 0, forced=forced), ml)
    system.start()
    first_warn = recovered = None
    for i in range(1400):
        _, d, _ = system.step()
        if first_warn is None and i >= 1000 and RANK[d.level] >= 1:
            first_warn = i - 1000
        if first_warn is not None and i > 1010 and recovered is None and RANK[d.level] == 0:
            recovered = i - 1010
    out["forced_10_cycle_dropout"] = {"warning_raised_after_cycles": first_warn,
                                      "fault_limit": config.FAULT_CYCLES_LIMIT,
                                      "cleared_after_cycles": recovered}
    # 3) real hazards with 10% faulty cycles: is the shutdown still issued?
    res = {}
    for sc in ("gas_leak", "overheat", "fire"):
        shut_f = shut_c = truth = 0
        for j in range(runs_per_scenario):
            r = generate_run(np.random.default_rng([seed, 3000 + j]), sc)
            truth += first_alarm(r["state"] == 2) is not None
            for faulty in (True, False):
                reset_ml(ml)
                inner = ArrayReader(r)
                s, _ = make_system(FaultyReader(inner, seed + j) if faulty else inner, ml)
                s.start()
                for _ in range(len(r["state"])):
                    s.step()
                hit = any(c == "RELAY:0" for _, c in inner.sent)
                if faulty:
                    shut_f += hit
                else:
                    shut_c += hit
        n = runs_per_scenario
        res[sc] = {"runs": n, "runs_that_truly_reach_critical": truth,
                   "shutdown_rate_no_faults": round(shut_c / n, 4), "shutdown_rate_10pct_faults": round(shut_f / n, 4)}
    out["shutdown_under_faults_vs_clean"] = res
    return out


# --------------------------------------------------------------------------- orchestration
def verdict(ok) -> str:
    return "PASS" if ok else "FAIL"


def build_report(R: dict) -> str:
    L = ["# C1-C5 Evaluation Report", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | seed {R['seed']} | models: "
         f"{'rule + ML' if R['ml_enabled'] else 'RULE-BASED ONLY (no ML models found)'} | mode: {'quick' if R['quick'] else 'full'}", "",
         "> **Read first.** C1/C2/C4 use synthetic held-out scenarios, not field data. C4 counts relay commands issued, "
         "not physical actuation. C3/C5 timings are specific to the machine that ran this. See limitations below.", ""]
    rows = []
    for cfg in R["configs"]:
        c1, c2, c4 = R[cfg]["C1"], R[cfg]["C2"], R[cfg]["C4"]
        crit_rec = c1["per_class"]["CRITICAL"]["recall"]
        rows += [
            (f"C1 macro-F1 ({cfg})", c1["macro_f1"], f">= {TARGETS['C1_macro_f1']}", c1["macro_f1"] >= TARGETS["C1_macro_f1"]),
            (f"C1 CRITICAL recall ({cfg})", crit_rec, f">= {TARGETS['C1_critical_recall']}", crit_rec >= TARGETS["C1_critical_recall"]),
            (f"C1 false CRITICAL, clean runs ({cfg})", c1["false_critical_rate_on_clean_runs"],
             f"<= {TARGETS['C1_false_critical_clean_runs']}", c1["false_critical_rate_on_clean_runs"] <= TARGETS["C1_false_critical_clean_runs"]),
            (f"C2 early-warning detection ({cfg})", c2["detection_rate"], f">= {TARGETS['C2_detection_rate']}",
             c2["detection_rate"] >= TARGETS["C2_detection_rate"]),
            (f"C2 false-alarm runs ({cfg})", c2["false_alarm_run_rate"], f"<= {TARGETS['C2_false_alarm_run_rate']}",
             c2["false_alarm_run_rate"] <= TARGETS["C2_false_alarm_run_rate"]),
            (f"C4 shutdown within 5 s ({cfg})", c4["ALL_HAZARDS"]["shutdown_within_5s_of_true_critical"],
             f">= {TARGETS['C4_shutdown_within_5s']}", c4["ALL_HAZARDS"]["shutdown_within_5s_of_true_critical"] >= TARGETS["C4_shutdown_within_5s"]),
            (f"C4 false shutdown, clean runs ({cfg})", c4["ALL_HAZARDS"]["false_shutdown_rate_on_clean_runs"],
             f"<= {TARGETS['C4_false_shutdown_clean_runs']}", c4["ALL_HAZARDS"]["false_shutdown_rate_on_clean_runs"] <= TARGETS["C4_false_shutdown_clean_runs"])]
    for cfg, v in R["C3"].items():
        if cfg == "cycle_budget_ms":
            continue
        p99 = v["full_cycle(read+decide+respond+DB)"]["p99_ms"]
        rows.append((f"C3 full-cycle p99 ms ({cfg})", p99, f"<= {TARGETS['C3_p99_step_ms']}", p99 <= TARGETS["C3_p99_step_ms"]))
    g = R["C5"]["soak"]["projected_heap_growth_mb_per_day_at_2hz"]
    rows += [("C5 soak heap growth, MB/day projected", g, f"<= {TARGETS['C5_soak_heap_mb_per_day']}", g <= TARGETS["C5_soak_heap_mb_per_day"]),
             ("C5 soak errors", R["C5"]["soak"]["errors"], "== 0", R["C5"]["soak"]["errors"] == 0),
             ("C5 dashboard p95 ms under load", R["C5"]["http_load"]["latency_p95_ms"], f"<= {TARGETS['C5_http_p95_ms']}",
              R["C5"]["http_load"]["latency_p95_ms"] <= TARGETS["C5_http_p95_ms"]),
             ("C5 dashboard errors under load", R["C5"]["http_load"]["errors"], "== 0", R["C5"]["http_load"]["errors"] == 0),
             ("C5 stress exceptions", R["C5"]["stress"]["fault_storm_normal_operation"]["exceptions"], "== 0",
              R["C5"]["stress"]["fault_storm_normal_operation"]["exceptions"] == 0)]
    L += ["## Summary against targets", "", "| Measure | Result | Target | Verdict |", "|---|---|---|---|"]
    L += [f"| {n} | {v} | {t} | **{verdict(ok)}** |" for n, v, t, ok in rows]
    L += ["", "Targets are the ones set in the project roadmap; a FAIL is reported as measured, not tuned away.", ""]
    L += ["## Full results", "", "```json", json.dumps({k: R[k] for k in R["configs"] + ["C3", "C5"]}, indent=2), "```", "",
          "## What this does and does not prove", "",
          "- Synthetic held-out scenarios (seed different from training) measure software behaviour and robustness to the modelled effects (humidity/temperature cross-sensitivity, benign spikes, noise, dropouts). They do **not** measure field accuracy.",
          "- **Circularity (C1/C4):** ground-truth labels use the same threshold values as the rule engine, applied to noise-free latent signals. High rule-based scores therefore show robustness to noise and lag, NOT that the thresholds are right for real hazards. Calibrate on hardware.",
          "- **C1 is carried by the SAFE and CRITICAL classes.** WARNING-class F1 is much lower (see per_class); the macro-F1 pass is marginal.",
          "- **C2 is a trade-off, not a free win.** Adding the ML early-alarm raises detection but also false-alarm runs (see C2 by_scenario: mostly overheat and slow-leak runs that warm up but never reach CRITICAL). Pick the operating point deliberately.",
          "- False shutdowns on clean runs come from near-critical plateaus (see `false_shutdowns_by_scenario`), not from normal operation.",
          "- The early-warning model is limited by fires: the flame sensor can trip within about 5-10 s of the first temperature rise, leaving little warning for any predictor.",
          "- C4 counts commands issued; the hardware loopback test (relay + load) must confirm physical shutdown.",
          "- The soak leak criterion measures sustained heap growth in the second half of the run, projected per day. An earlier draft used percent growth from the 10% mark, which was ill-defined for a ~13 KB start-up heap, so it was replaced before the full run.",
          "- C5 projected database growth shows every cycle is stored; the dashboard runner now prunes sensor readings to the newest 14 days (config READINGS_KEEP_ROWS).",
          "- Dashboard load: the monitoring loop was run at an artificial 100 Hz; it completed fewer cycles while 10 clients hammered the dashboard. At the real 2 Hz it needs about 1-4 ms per cycle, so headroom is large, but this contention is real.",
          "- Re-run on real logged data after hardware arrives and quote those numbers."]
    return "\n".join(L)


def run_all(quick: bool = False, out_dir: str = "evaluation/results", seed: int = 2026, model_dir: str = None,
            progress=print) -> dict:
    model_dir = model_dir or config.MODEL_DIR
    ml = MLSignals(model_dir)
    ml_on = ml.enabled
    n_runs = 60 if quick else 400
    runs = generate_runs(n_runs, seed)
    R = {"seed": seed, "quick": quick, "ml_enabled": ml_on, "configs": ["rule_only"] + (["rule_plus_ml"] if ml_on else [])}
    progress(f"[C1/C2/C4] {n_runs} held-out runs x {len(R['configs'])} configuration(s) ...")
    for cfg in R["configs"]:
        traces = [run_trace(r, ml if cfg == "rule_plus_ml" else None) for r in runs]
        R[cfg] = {"C1": c1_detection(traces), "C2": c2_early_warning(traces), "C4": c4_response(traces)}
    progress("[C3] latency ...")
    R["C3"] = c3_latency(ml if ml_on else None, 400 if quick else 6000, seed)
    progress("[C5] soak / sensor load / dashboard load / stress ...")
    R["C5"] = {"soak": c5_soak(ml if ml_on else None, 600 if quick else 7200, seed),
               "sensor_load": c5_sensor_load([1, 5] if quick else [1, 10, 50, 100, 200], 5 if quick else 20, seed),
               "http_load": c5_http_load(3 if quick else 10, 20 if quick else 100, seed),
               "stress": c5_stress(ml if ml_on else None, 4 if quick else 25, seed)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(R, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8").write(build_report(R))
    return R
