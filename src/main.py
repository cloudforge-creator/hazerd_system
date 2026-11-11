"""Entry point.

    python -m src.main --sim --scenario gas_leak --duration 40
    python -m src.main --port COM3
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from types import SimpleNamespace

from src import config
from src.ai_runtime import MLSignals
from src.camera_hub import CameraHub
from src.fire_cnn import FireCNN, FireSignals, FrameSource
from src.ingest import QueueSensorReader
from src.notifiers import AsyncNotifier, load_notifiers
from src.response import LogNotifier, ResponseManager
from src.signals import CombinedSignals
from src.temp_anomaly import TempAnomaly
from src.pipeline import HazardSystem
from src.sensor_reader import SerialSensorReader
from src.simulator import SCENARIOS, SimulatedSensorReader
from src.storage import Storage

FIELDS = ["timestamp", "source", "gas_raw", "temperature", "humidity", "flame", "faults"]


def append_row(path: str, reading) -> None:
    """CSV log - also the raw material for building ML datasets later."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow([
            f"{reading.timestamp:.3f}", reading.source, reading.gas_raw,
            reading.temperature, reading.humidity, reading.flame,
            ";".join(reading.faults),
        ])


def cleanup(parts) -> None:
    for obj in (parts.monitor, parts.hub, parts.notifier):
        stop = getattr(obj, "stop", None)
        if stop:
            try:
                stop()
            except Exception:
                pass


def run_dashboard(args, system, reader, ai_lines, parts) -> None:
    from src import runtime_config
    from src.dashboard import create_app, load_secret_key, serve
    from src.runner import LiveState, Runner
    from src.users import UserStore

    for msg in runtime_config.load_overrides("data/runtime_config.json", system.responder):
        print("[config]", msg)
    users = UserStore(system.storage)
    if not users.list():
        print("[auth] No dashboard users yet. Create one first:  python -m src.users add admin --role admin")
    mode = f"SIMULATION ({args.scenario})" if args.sim else f"HARDWARE ({reader.port})"
    live = LiveState(mode=mode, ai=ai_lines)
    runner = Runner(system, live, config.CYCLE_SECONDS, pace=args.sim,
                    on_step=lambda r, d, ex: append_row(args.log, r))
    app = create_app(system, live, users, load_secret_key(), runner=runner, hub=parts.hub, monitor=parts.monitor,
                     people=parts.people, face_engine=parts.face_engine, zones=parts.zones, notifier=parts.notifier,
                     notify_notes=parts.notify_notes, ingest=parts.ingest, device_key=parts.device_key)
    if parts.monitor is not None:
        parts.monitor.start()
    runner.start()
    print(f"[web] dashboard on http://{args.host}:{args.web_port}  (Ctrl+C to stop)")
    try:
        serve(app, args.host, args.web_port)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        cleanup(parts)
        system.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use the simulator")
    ap.add_argument("--scenario", default="normal", choices=SCENARIOS)
    ap.add_argument("--port", default=None, help="serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--http-ingest", action="store_true", help="sensors POST to /api/ingest (needs --dashboard and env HAZARD_DEVICE_KEY)")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = forever)")
    ap.add_argument("--log", default=config.LOG_PATH)
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--no-ml", action="store_true", help="rule-based only")
    ap.add_argument("--camera", default=None, help="webcam index (0), video file, or folder of images")
    ap.add_argument("--people", action="store_true", help="enable personnel recognition/tracking (needs --camera and the face models)")
    ap.add_argument("--zones", default=config.ZONES_PATH)
    ap.add_argument("--notify-config", default=config.NOTIFY_CONFIG_PATH)
    ap.add_argument("--no-notify", action="store_true", help="do not send Telegram/e-mail/webhook notifications")
    ap.add_argument("--dashboard", action="store_true", help="run the web dashboard (needs users: python -m src.users add)")
    ap.add_argument("--host", default="127.0.0.1", help="dashboard bind address (default: this machine only)")
    ap.add_argument("--web-port", type=int, default=5000)
    args = ap.parse_args()

    parts = SimpleNamespace(hub=None, monitor=None, people=None, face_engine=None, zones=None, notifier=None,
                            notify_notes=[], ingest=None, device_key=None)
    if args.http_ingest:
        if not args.dashboard:
            raise SystemExit("--http-ingest needs --dashboard (the sensors post to the dashboard's /api/ingest)")
        key = os.environ.get("HAZARD_DEVICE_KEY", "")
        if len(key) < 16:
            raise SystemExit("set the environment variable HAZARD_DEVICE_KEY to a random secret of at least 16 characters")
        reader = parts.ingest = QueueSensorReader()
        parts.device_key = key
        print("[mode] HTTP INGEST (waiting for the device to POST /api/ingest)")
    elif args.sim:
        reader = SimulatedSensorReader(args.scenario, dt=config.CYCLE_SECONDS)
        print(f"[mode] SIMULATION ({args.scenario})")
    else:
        reader = SerialSensorReader(args.port)
        print(f"[mode] HARDWARE on {reader.port}")

    storage = Storage(args.db)
    providers, ai_lines = [], []
    if not args.no_ml:
        ml = MLSignals()
        print(f"[ai] state/early-warning: {ml.status}")
        ai_lines.append(ml.status)
        providers.append(ml if ml.enabled else None)
        temp = TempAnomaly()
        print(f"[ai] temperature LSTM: {temp.status}")
        ai_lines.append(temp.status)
        providers.append(temp if temp.enabled else None)
    else:
        print("[ai] ML disabled by --no-ml")
        ai_lines.append("ML disabled (--no-ml)")

    holder = {}
    if args.camera is not None:
        try:
            parts.hub = CameraHub(FrameSource(args.camera)).start()
        except RuntimeError as exc:
            raise SystemExit(f"camera error: {exc}")
        cnn = FireCNN()
        print(f"[ai] {cnn.status}")
        ai_lines.append(cnn.status)
        if cnn.enabled:
            providers.append(FireSignals(parts.hub, cnn))
        if args.people:
            from src.personnel.face import FaceEngine
            from src.personnel.monitor import PersonnelMonitor
            from src.personnel.registry import PersonStore
            from src.personnel.zones import ZoneHolder
            from src.decision_engine import Level
            parts.people, parts.zones = PersonStore(storage), ZoneHolder(args.zones)
            ok, msg = FaceEngine.models_present()
            if ok:
                parts.face_engine = FaceEngine()
                parts.monitor = PersonnelMonitor(parts.hub, parts.face_engine, parts.people, parts.zones,
                                                 lambda: holder["sys"].engine.level if "sys" in holder else Level.SAFE)
                providers.append(parts.monitor)
                print("[people] recognition on")
                ai_lines.append("person recognition on")
            else:
                print(f"[people] OFF: {msg}  (run: python scripts/download_face_models.py)")
                ai_lines.append("person recognition off (face models missing)")
        else:
            parts.zones = None
    else:
        print("[ai] camera off (use --camera 0 | video | image folder)")
        ai_lines.append("camera off")
        if args.people:
            print("[people] --people needs --camera; ignored")

    notifiers = [LogNotifier()]
    if not args.no_notify:
        router, notes = load_notifiers(args.notify_config)
        parts.notify_notes = notes
        for n in notes:
            print("[notify]", n)
        if router is not None:
            parts.notifier = AsyncNotifier(router, on_result=storage.log_system)
            notifiers.append(parts.notifier)

    active = [p for p in providers if p is not None]
    system = HazardSystem(reader, responder=ResponseManager(reader, notifiers), storage=storage,
                          signals_provider=CombinedSignals(*active) if active else None)
    holder["sys"] = system
    if args.dashboard:
        run_dashboard(args, system, reader, ai_lines, parts)
        return
    if parts.monitor is not None:
        parts.monitor.start()
    system.start()
    start = time.time()
    try:
        while True:
            t0 = time.time()
            r, d, executed = system.step()
            append_row(args.log, r)
            extra = f" -> {','.join(executed)}" if executed else ""
            early = " EARLY-WARNING" if d.early_warning else ""
            ml_txt = f" ml={d.ml_score:.2f}" if d.ml_score else ""
            print(f"{d.level.value:<8} score={d.score:.2f} pred={d.predicted_score:.2f} "
                  f"gas={r.gas_raw} temp={r.temperature} flame={r.flame}{ml_txt}{early}{extra}")
            if args.duration and time.time() - start >= args.duration:
                break
            if args.sim:
                time.sleep(max(0.0, config.CYCLE_SECONDS - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(parts)
        system.stop()


if __name__ == "__main__":
    main()
