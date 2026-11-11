import re
import threading
import time

import pytest

from src import config, runtime_config
from src.decision_engine import Level
from src.pipeline import HazardSystem
from src.runner import LiveState, Runner
from src.simulator import SimulatedSensorReader
from src.storage import Storage
from src.users import UserStore

flask = pytest.importorskip("flask")
from src.dashboard import create_app  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def restore_config():
    yield
    runtime_config.reset_to_defaults()


@pytest.fixture
def env(tmp_path):
    sim = SimulatedSensorReader("gas_leak", dt=0.5)
    sent = []
    sim.send = sent.append
    system = HazardSystem(sim, storage=Storage())
    system.start()
    live = LiveState(mode="SIMULATION (test)", ai=["rule-based only"])
    for _ in range(80):                       # run into CRITICAL so there are alerts
        live.update(*system.step())
    users = UserStore(system.storage)
    users.create("boss", "correct-horse-1", "admin")
    users.create("viewer1", "viewer-pass-1", "viewer")
    clock = Clock()
    app = create_app(system, live, users, "test-secret", runtime_config_path=str(tmp_path / "rc.json"), clock=clock)
    return {"app": app, "system": system, "live": live, "users": users, "sent": sent, "clock": clock,
            "rc": tmp_path / "rc.json"}


def token(client, url="/login"):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)


def login(client, user, pw):
    return client.post("/login", data={"username": user, "password": pw, "csrf_token": token(client)})


# ---------------- authentication ----------------
def test_pages_and_api_require_login(env):
    c = env["app"].test_client()
    assert c.get("/").status_code == 302 and "/login" in c.get("/").headers["Location"]
    r = c.get("/api/status")
    assert r.status_code == 401 and r.get_json()["error"]


def test_login_success_failure_and_logging(env):
    c = env["app"].test_client()
    assert login(c, "boss", "wrong-password").status_code == 401
    assert login(c, "nobody", "whatever-1").status_code == 401
    assert login(c, "boss", "correct-horse-1").status_code == 302
    assert c.get("/").status_code == 200
    kinds = [e["kind"] for e in env["system"].storage.recent_system(20)]
    assert "login" in kinds and "login_failed" in kinds


def test_lockout_after_repeated_failures_then_recovers(env):
    c = env["app"].test_client()
    for _ in range(5):
        assert login(c, "boss", "bad-password-x").status_code == 401
    assert login(c, "boss", "correct-horse-1").status_code == 429       # locked even with the right password
    env["clock"].t += 61
    assert login(c, "boss", "correct-horse-1").status_code == 302


def test_open_redirect_is_blocked(env):
    c = env["app"].test_client()
    r = c.post("/login?next=//evil.example", data={"username": "boss", "password": "correct-horse-1",
                                                   "csrf_token": token(c)})
    assert r.status_code == 302 and r.headers["Location"] == "/"


def test_post_without_csrf_is_rejected(env):
    c = env["app"].test_client()
    assert c.post("/login", data={"username": "boss", "password": "correct-horse-1"}).status_code == 400
    login(c, "boss", "correct-horse-1")
    assert c.post("/control/reset", data={}).status_code == 400


def test_disabled_user_cannot_login(env):
    env["users"].disable("viewer1")
    assert login(env["app"].test_client(), "viewer1", "viewer-pass-1").status_code == 401


# ---------------- roles ----------------
def test_viewer_can_read_but_not_act(env):
    c = env["app"].test_client()
    login(c, "viewer1", "viewer-pass-1")
    for url in ("/", "/alerts", "/logs?kind=events", "/config", "/api/history", "/api/alerts"):
        assert c.get(url).status_code == 200
    t = token(c, "/")
    first = env["system"].storage.recent_alerts(1)[0]["id"]
    assert c.post(f"/alerts/{first}/ack", data={"csrf_token": t}).status_code == 403
    assert c.post("/control/reset", data={"csrf_token": t}).status_code == 403
    assert c.post("/config", data={"csrf_token": t, "WARNING_THRESHOLD": "0.3"}).status_code == 403
    assert config.WARNING_THRESHOLD == 0.40


def test_admin_can_acknowledge_alert(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    aid = env["system"].storage.recent_alerts(1, active_only=True)[0]["id"]
    r = c.post(f"/alerts/{aid}/ack", data={"csrf_token": token(c, "/alerts")})
    assert r.status_code == 302
    assert aid not in [a["id"] for a in env["system"].storage.recent_alerts(50, active_only=True)]


# ---------------- configuration ----------------
def test_config_rejects_invalid_and_cross_inconsistent_values(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    t = token(c, "/config")
    c.post("/config", data={"csrf_token": t, "WARNING_THRESHOLD": "0.9", "CRITICAL_THRESHOLD": "0.5"})
    c.post("/config", data={"csrf_token": t, "GAS_CRITICAL": "99999"})
    c.post("/config", data={"csrf_token": t, "CLEAR_CYCLES": "abc"})
    assert config.WARNING_THRESHOLD == 0.40 and config.CRITICAL_THRESHOLD == 0.70
    assert config.GAS_CRITICAL == 600 and not env["rc"].exists()


def test_config_valid_change_applies_persists_and_is_audited(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    c.post("/config", data={"csrf_token": token(c, "/config"), "WARNING_THRESHOLD": "0.35",
                            "NOTIFY_COOLDOWN_S": "10"})
    assert config.WARNING_THRESHOLD == 0.35 and env["system"].responder.cooldown == 10.0
    assert env["rc"].exists()
    audit = [e for e in env["system"].storage.recent_system(10) if e["kind"] == "config_change"]
    assert audit and "boss" in audit[0]["detail"] and "WARNING_THRESHOLD" in audit[0]["detail"]


def test_unknown_config_keys_are_ignored(env):
    clean, errors = runtime_config.validate({"EVIL": "1", "WARNING_THRESHOLD": "0.3"})
    assert "EVIL" not in clean and any("unknown" in e for e in errors)


def test_persisted_overrides_load_and_bad_file_is_refused(tmp_path):
    p = tmp_path / "rc.json"
    p.write_text('{"WARNING_THRESHOLD": 0.3}')
    assert runtime_config.load_overrides(str(p)) == [] and config.WARNING_THRESHOLD == 0.3
    runtime_config.reset_to_defaults()
    p.write_text('{"WARNING_THRESHOLD": 0.9, "CRITICAL_THRESHOLD": 0.5}')
    assert runtime_config.load_overrides(str(p)) and config.WARNING_THRESHOLD == 0.40


# ---------------- control, API, headers ----------------
def test_reset_latch_refused_unless_safe(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    assert env["system"].responder.relay_latched and env["system"].engine.level == Level.CRITICAL
    c.post("/control/reset", data={"csrf_token": token(c, "/")})
    assert env["system"].responder.relay_latched                       # refused
    env["system"].engine.level = Level.SAFE
    c.post("/control/reset", data={"csrf_token": token(c, "/")})
    assert not env["system"].responder.relay_latched and env["sent"][-1] == "RELAY:1"


def test_status_api_contents(env):
    c = env["app"].test_client()
    login(c, "viewer1", "viewer-pass-1")
    s = c.get("/api/status").get_json()
    assert s["level"] == "CRITICAL" and s["relay"].startswith("SHUT DOWN") and s["buzzer"] == "ON"
    assert s["reading"]["gas_raw"] > 500 and s["steps"] == 80 and s["mode"].startswith("SIMULATION")
    assert len(c.get("/api/history?n=30").get_json()) == 30


def test_security_headers_and_no_inline_scripts(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    for url in ("/login", "/", "/alerts", "/config", "/logs"):
        r = c.get(url)
        html = r.get_data(as_text=True)
        assert "script-src 'self'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Frame-Options"] == "DENY" and r.headers["Cache-Control"] == "no-store"
        assert not re.search(r"<script(?![^>]*\ssrc=)", html) and " style=" not in html


def test_logs_rejects_unknown_kind_and_escapes_html(env):
    c = env["app"].test_client()
    login(c, "boss", "correct-horse-1")
    assert c.get("/logs?kind=users").status_code == 400
    env["system"].storage.log_system("note", "<script>alert(1)</script>")
    html = c.get("/logs?kind=system").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_healthz_reflects_crash(env):
    c = env["app"].test_client()
    assert c.get("/healthz").status_code == 200
    env["live"].crashed = True
    assert c.get("/healthz").status_code == 503


# ---------------- users ----------------
def test_user_store_rules_and_hashing():
    store = UserStore(Storage())
    store.create("alice", "long-enough-pw", "viewer")
    with pytest.raises(ValueError):
        store.create("alice", "another-long-pw", "viewer")          # duplicate (case-insensitive too)
    with pytest.raises(ValueError):
        store.create("ALICE", "another-long-pw", "viewer")
    with pytest.raises(ValueError):
        store.create("bob", "short", "viewer")
    with pytest.raises(ValueError):
        store.create("bob", "long-enough-pw", "root")
    with pytest.raises(ValueError):
        store.create("b o b", "long-enough-pw", "viewer")
    stored = store.s.query("SELECT password_hash FROM users")[0]["password_hash"]
    assert "long-enough-pw" not in stored and stored != ""
    assert store.verify("alice", "long-enough-pw")["role"] == "viewer"
    assert store.verify("alice", "wrong") is None and store.verify("ghost", "x") is None


# ---------------- background runner ----------------
def test_runner_runs_and_stops_cleanly():
    sim = SimulatedSensorReader("normal", dt=0.5)
    sim.send = lambda c: None
    system = HazardSystem(sim, storage=Storage())
    live = LiveState()
    runner = Runner(system, live, cycle=0.002, pace=True)
    runner.start()
    deadline = time.time() + 5
    while live.steps < 30 and time.time() < deadline:
        time.sleep(0.01)
    runner.stop()
    assert live.steps >= 30 and not live.crashed and not runner.is_alive()
    assert system.storage.count("sensor_readings") >= 30


def test_runner_failsafe_shuts_down_relay_when_reader_keeps_failing():
    class Broken:
        sent = []

        def read(self):
            raise OSError("board unplugged")

        def send(self, c):
            self.sent.append(c)

        def close(self):
            pass

    reader = Broken()
    system = HazardSystem(reader, storage=Storage())
    live = LiveState()
    runner = Runner(system, live, cycle=0.001, pace=False)
    runner.start()
    runner.join(timeout=5)
    assert live.crashed and "RELAY:0" in reader.sent and system.responder.relay_latched
    assert any(e["kind"] == "runner_crashed" for e in system.storage.recent_system(50))


def test_storage_is_safe_under_concurrent_writes():
    st = Storage()

    def work():
        for i in range(200):
            st.log_system("t", str(i))
    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert st.count("system_events") == 1200


# ---------------- retention + bounded memory ----------------
def test_prune_keeps_newest_rows_only():
    from src.decision_engine import DecisionEngine
    from tests.helpers import reading
    st, eng = Storage(), DecisionEngine()
    for i in range(50):
        r = reading(i)
        st.log_cycle(r, eng.decide(r))
    assert st.prune_readings(10) == 40 and st.count("sensor_readings") == 10
    assert min(x["ts"] for x in st.recent_readings(100)) == 40
    assert st.prune_readings(10) == 0
    assert Storage().prune_readings(10) == 0                       # empty table is fine



def test_runner_prunes_in_background():
    sim = SimulatedSensorReader("normal", dt=0.5)
    sim.send = lambda c: None

    system = HazardSystem(sim, storage=Storage())
    live = LiveState()

    runner = Runner(
        system,
        live,
        cycle=0.002,
        pace=True,
        prune_every=0.02,
        keep_rows=25
    )

    runner.start()

    try:
        deadline = time.monotonic() + 5.0

        while live.steps <= 100 and time.monotonic() < deadline:
            time.sleep(0.01)

    finally:
        runner.stop()

    assert live.steps > 100, (
        f"Runner completed only {live.steps} steps within 5 seconds"
    )

    stored_rows = system.storage.count("sensor_readings")

    assert stored_rows < live.steps, (
        f"Pruning failed: stored={stored_rows}, steps={live.steps}"
    )



def test_response_records_are_bounded():
    from src.response import LogNotifier, ResponseManager
    assert LogNotifier().messages.maxlen == 1000
    assert ResponseManager(object()).executed.maxlen == 1000
