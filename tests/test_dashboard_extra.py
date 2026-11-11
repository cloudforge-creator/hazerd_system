import re
import time

import numpy as np
import pytest

from src.camera_hub import MAX_STREAM_CLIENTS, CameraHub
from src.dashboard import create_app
from src.dashboard.extra import MjpegStream
from src.ingest import QueueSensorReader
from src.personnel.registry import EMBED_DIM, PersonStore, normalize
from src.personnel.zones import ZoneHolder
from src.pipeline import HazardSystem
from src.runner import LiveState
from src.simulator import SimulatedSensorReader
from src.storage import Storage
from src.users import UserStore

cv2 = pytest.importorskip("cv2")


def png_bytes(color=(10, 20, 30), size=(48, 64)):
    img = np.full((size[0], size[1], 3), color, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


class StaticSource:
    def read(self):
        return np.full((120, 160, 3), 40, np.uint8)

    def close(self):
        pass


class StubEngine:
    def __init__(self):
        self.result = (normalize(np.random.default_rng(1).normal(size=EMBED_DIM)), "ok")
        self.calls = 0

    def embed_single_face(self, frame):
        self.calls += 1
        return self.result


class StubMonitor:
    def snapshot(self):
        return [{"track_id": 1, "name": "Ali Khan", "designation": "Operator", "status": "identified", "zones": ["Press"], "seen_s": 3.0}]

    def status(self):
        return {"running": True, "frames": 5, "errors": 0, "last_error": None, "process_ms": 12.0, "tracks": 1}


class StubNotifier:
    def __init__(self):
        self.sent, self.stats = [], {"queued": 0, "delivered": 0, "failed": 0, "dropped": 0, "last_error": None}

    def send(self, s, b):
        self.sent.append((s, b))


@pytest.fixture
def env(tmp_path):
    sim = SimulatedSensorReader("normal", dt=0.5)
    sim.send = lambda c: None
    system = HazardSystem(sim, storage=Storage())
    live = LiveState(mode="SIMULATION (test)")
    users = UserStore(system.storage)
    users.create("boss", "correct-horse-1", "admin")
    users.create("viewer1", "viewer-pass-1", "viewer")
    hub = CameraHub(StaticSource(), fps=50).start()
    t0 = time.time()
    while hub.frames < 2 and time.time() - t0 < 5:
        time.sleep(0.01)
    people, engine, notifier = PersonStore(system.storage), StubEngine(), StubNotifier()
    zones, ingest = ZoneHolder(str(tmp_path / "zones.json")), QueueSensorReader(timeout=0.05)
    app = create_app(system, live, users, "test-secret", runtime_config_path=str(tmp_path / "rc.json"), hub=hub,
                     monitor=StubMonitor(), people=people, face_engine=engine, zones=zones, notifier=notifier,
                     notify_notes=["telegram enabled"], ingest=ingest, device_key="dev-key-123")
    yield {"app": app, "system": system, "hub": hub, "people": people, "engine": engine, "zones": zones, "ingest": ingest,
           "notifier": notifier, "users": users, "storage": system.storage}
    hub.stop()


def token(client, url="/login"):
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', client.get(url).get_data(as_text=True)).group(1)


def login(client, user="boss", pw="correct-horse-1"):
    assert client.post("/login", data={"username": user, "password": pw, "csrf_token": token(client)}).status_code == 302
    return client


def admin(env):
    return login(env["app"].test_client())


def viewer(env):
    return login(env["app"].test_client(), "viewer1", "viewer-pass-1")


def flashes(client, url):
    return client.get(url).get_data(as_text=True)


# ---------------- camera ----------------
def test_camera_pages_need_login_and_work_for_viewers(env):
    c = env["app"].test_client()
    for url in ("/camera", "/camera/stream", "/camera/snapshot.jpg", "/personnel"):
        assert c.get(url).status_code == 302                      # pages redirect to the login form
    assert c.get("/api/people").status_code == 401                # APIs answer 401, like every other /api route
    v = viewer(env)
    assert v.get("/camera").status_code == 200 and v.get("/personnel").status_code == 200
    r = v.get("/camera/snapshot.jpg")
    assert r.status_code == 200 and r.mimetype == "image/jpeg" and r.data[:2] == b"\xff\xd8"
    people = v.get("/api/people").get_json()
    assert people["people"][0]["name"] == "Ali Khan" and people["monitor"]["running"] and people["camera"]["frames"] >= 2


def test_mjpeg_stream_serves_frames_and_releases_slots(env):
    c, hub = viewer(env), env["hub"]
    r = c.get("/camera/stream", buffered=False)
    assert r.status_code == 200 and "multipart/x-mixed-replace" in r.content_type
    chunk = next(iter(r.response))
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in chunk and hub.streams == 1
    r.close()
    assert hub.streams == 0
    held = [c.get("/camera/stream", buffered=False) for _ in range(MAX_STREAM_CLIENTS)]
    assert hub.streams == MAX_STREAM_CLIENTS
    assert c.get("/camera/stream").status_code == 503
    for h in held:
        h.close()
    assert hub.streams == 0


def test_mjpeg_stream_releases_once_even_if_never_read():
    class FakeHub:
        released = 0
        def jpeg(self): return b"\xff\xd8x"
        def release_stream(self): FakeHub.released += 1
    s = MjpegStream(FakeHub())
    s.close(); s.close()
    assert FakeHub.released == 1
    dead = MjpegStream(type("H", (), {"jpeg": lambda self: None, "release_stream": lambda self: None})(), wait=0.2)
    assert b"image/jpeg" in next(dead)                      # dead camera -> placeholder image, not a hang


def test_no_camera_configured(env):
    sim = SimulatedSensorReader("normal", dt=0.5)
    system = HazardSystem(sim, storage=Storage())
    UserStore(system.storage).create("boss", "correct-horse-1", "admin")
    app = create_app(system, LiveState(), UserStore(system.storage), "k")
    c = login(app.test_client())
    assert "No camera is connected" in c.get("/camera").get_data(as_text=True)
    assert c.get("/camera/stream").status_code == 404 and c.get("/camera/snapshot.jpg").status_code == 503
    assert "Person recognition is off" in c.get("/personnel").get_data(as_text=True)
    assert c.post("/personnel/add", data={"csrf_token": token(c, "/camera"), "name": "A"}).status_code == 404
    assert c.get("/api/people").get_json()["people"] == []


# ---------------- zones ----------------
GOOD_ZONES = '[{"name": "Press", "points": [[0.1,0.1],[0.6,0.1],[0.6,0.9]], "hazard_when_alert": true, "allowed_designations": ["Operator"]}]'


def test_zones_admin_saves_valid_rejects_invalid_viewer_forbidden(env):
    a = admin(env)
    t = token(a, "/camera")
    a.post("/zones/save", data={"csrf_token": t, "zones_json": GOOD_ZONES})
    assert [z.name for z in env["zones"]()] == ["Press"] and (env["zones"].path and open(env["zones"].path).read())
    assert any(e["kind"] == "zones_changed" for e in env["storage"].recent_system(10))
    a.post("/zones/save", data={"csrf_token": t, "zones_json": "{oops"})
    a.post("/zones/save", data={"csrf_token": t, "zones_json": '[{"name": "X", "points": [[0,0],[2,2],[1,1]]}]'})
    assert [z.name for z in env["zones"]()] == ["Press"]                      # unchanged by the bad ones
    v = viewer(env)
    assert v.post("/zones/save", data={"csrf_token": token(v, "/camera"), "zones_json": "[]"}).status_code == 403
    assert "readonly" in v.get("/camera").get_data(as_text=True)


# ---------------- personnel ----------------
def add_person(a, name="Sara Noor", consent="yes"):
    return a.post("/personnel/add", data={"csrf_token": token(a, "/personnel"), "name": name, "designation": "Supervisor",
                                          "employee_id": "E7", "consent": consent})


def test_adding_a_person_requires_consent_and_admin(env):
    a = admin(env)
    add_person(a, consent="")
    assert env["people"].list_people() == []
    add_person(a)
    assert env["people"].list_people()[0]["name"] == "Sara Noor"
    v = viewer(env)
    assert v.post("/personnel/add", data={"csrf_token": token(v, "/personnel"), "name": "X", "designation": "Y", "consent": "yes"}).status_code == 403
    html = v.get("/personnel").get_data(as_text=True)
    assert "Sara Noor" in html and "Enrol photos" not in html and "Add a person" not in html
    assert a.post("/personnel/add", data={"name": "Z", "designation": "Y", "consent": "yes"}).status_code == 400    # no CSRF


def test_enrol_photos_upload_validation(env):
    a = admin(env)
    add_person(a)
    pid = env["people"].list_people()[0]["id"]
    t = token(a, "/personnel")
    from io import BytesIO
    r = a.post(f"/personnel/{pid}/enroll", data={"csrf_token": t, "photos": [(BytesIO(png_bytes()), "a.png"), (BytesIO(png_bytes((9, 9, 9))), "b.png")]},
               content_type="multipart/form-data")
    assert r.status_code == 302 and env["people"].list_people()[0]["samples"] == 2 and env["engine"].calls == 2
    a.post(f"/personnel/{pid}/enroll", data={"csrf_token": t, "photos": (BytesIO(b"not an image"), "x.png")}, content_type="multipart/form-data")
    assert env["engine"].calls == 2 and "not a readable image" in flashes(a, "/personnel")
    big = b"\x00" * (4 * 1024 * 1024 + 10)
    a.post(f"/personnel/{pid}/enroll", data={"csrf_token": t, "photos": (BytesIO(big), "huge.png")}, content_type="multipart/form-data")
    assert env["engine"].calls == 2 and "larger than 4 MB" in flashes(a, "/personnel")
    env["engine"].result = (None, "no face found")
    a.post(f"/personnel/{pid}/enroll", data={"csrf_token": t, "photos": (BytesIO(png_bytes()), "n.png")}, content_type="multipart/form-data")
    assert env["people"].list_people()[0]["samples"] == 2 and "no face found" in flashes(a, "/personnel")
    a.post(f"/personnel/{pid}/enroll", data={"csrf_token": t}, content_type="multipart/form-data")
    assert "Choose at least one photo" in flashes(a, "/personnel")
    assert a.post("/personnel/999/enroll", data={"csrf_token": t}, content_type="multipart/form-data").status_code == 404


def test_enrol_without_models_and_live_capture(env):
    a = admin(env)
    add_person(a)
    pid = env["people"].list_people()[0]["id"]
    a.post(f"/personnel/{pid}/enroll-live", data={"csrf_token": token(a, "/personnel")})
    assert env["people"].list_people()[0]["samples"] == 1
    assert any(e["kind"] == "face_enrolled" for e in env["storage"].recent_system(10))


def test_person_actions_toggle_clear_delete_purge(env):
    a = admin(env)
    add_person(a)
    p = env["people"].list_people()[0]
    env["people"].add_embedding(p["id"], np.random.default_rng(3).normal(size=EMBED_DIM))
    env["people"].log_event("PERSON_SEEN", 1, env["people"].get(p["id"]))
    t = token(a, "/personnel")
    a.post(f"/personnel/{p['id']}/toggle", data={"csrf_token": t})
    assert env["people"].list_people()[0]["disabled"] == 1
    a.post(f"/personnel/{p['id']}/toggle", data={"csrf_token": t})
    a.post(f"/personnel/{p['id']}/clear", data={"csrf_token": t})
    assert env["people"].list_people()[0]["samples"] == 0
    a.post(f"/personnel/{p['id']}/delete", data={"csrf_token": t})
    assert env["people"].list_people() == [] and env["people"].recent_events() == []
    assert a.post(f"/personnel/{p['id']}/delete", data={"csrf_token": t}).status_code == 404
    assert a.post("/personnel/1/explode", data={"csrf_token": t}).status_code == 404


# ---------------- users ----------------
def test_user_management_rules(env):
    a, v = admin(env), viewer(env)
    assert v.get("/users").status_code == 403 and a.get("/users").status_code == 200
    t = token(a, "/users")
    a.post("/users/add", data={"csrf_token": t, "username": "newop", "password": "short", "role": "viewer"})
    assert "at least 8" in flashes(a, "/users") and "newop" not in [u["username"] for u in env["users"].list()]
    a.post("/users/add", data={"csrf_token": t, "username": "newop", "password": "long-enough-1", "role": "viewer"})
    assert env["users"].role_of("newop") == "viewer"
    a.post("/users/boss/disable", data={"csrf_token": t})
    assert "cannot disable your own" in flashes(a, "/users")
    env["users"].create("boss2", "other-admin-pw1", "admin")
    a.post("/users/boss2/disable", data={"csrf_token": t})
    assert env["users"].verify("boss2", "other-admin-pw1") is None
    a.post("/users/boss2/enable", data={"csrf_token": t})
    assert env["users"].verify("boss2", "other-admin-pw1")
    a.post("/users/newop/password", data={"csrf_token": t, "password": "brand-new-pw-1"})
    assert env["users"].verify("newop", "brand-new-pw-1") and env["users"].verify("newop", "long-enough-1") is None
    b2 = login(env["app"].test_client(), "boss2", "other-admin-pw1")
    env["users"].disable("boss")
    b2.post("/users/boss2/disable", data={"csrf_token": token(b2, "/users")})
    assert "cannot disable" in flashes(b2, "/users") and env["users"].verify("boss2", "other-admin-pw1")


# ---------------- notifications + logs + config ----------------
def test_notification_test_button_and_status_block(env):
    a, v = admin(env), viewer(env)
    html = v.get("/config").get_data(as_text=True)
    assert "telegram enabled" in html and "Send test notification" not in html
    a.post("/notifications/test", data={"csrf_token": token(a, "/config")})
    assert env["notifier"].sent[0][0].startswith("[TEST]")
    assert v.post("/notifications/test", data={"csrf_token": token(v, "/config")}).status_code == 403


def test_people_events_log_page(env):
    env["people"].log_event("PERSON_SEEN", 1, {"id": 1, "name": "Ali Khan", "designation": "Operator"}, "Press")
    v = viewer(env)
    html = v.get("/logs?kind=people").get_data(as_text=True)
    assert "PERSON_SEEN" in html and "Ali Khan" in html


# ---------------- device ingest ----------------
def post_ingest(c, body, key="dev-key-123", **kw):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    return c.post("/api/ingest", json=body, headers=h, **kw)


def test_ingest_authentication_validation_and_commands(env):
    c = env["app"].test_client()
    assert post_ingest(c, {"gas": 1}, key="nope").status_code == 401
    assert post_ingest(c, {"gas": 1}, key=None).status_code == 401
    assert post_ingest(c, {"gas": 1}, key="nope").status_code == 401
    assert len([e for e in env["storage"].recent_system(20) if e["kind"] == "ingest_denied"]) == 1      # rate limited log
    assert post_ingest(c, {"gas": 1, "evil": 2}).status_code == 400
    assert post_ingest(c, {"gas": "300"}).status_code == 400 and post_ingest(c, {"gas": True}).status_code == 400
    assert post_ingest(c, [1, 2]).status_code == 400
    assert c.post("/api/ingest", data=b"x" * 2000, headers={"Authorization": "Bearer dev-key-123", "Content-Type": "application/json"}).status_code == 413
    env["ingest"].send("RELAY:1")
    r = post_ingest(c, {"gas": 312, "temp": 27.4, "hum": 55.0, "flame": 0})
    assert r.status_code == 200 and r.get_json() == {"ok": True, "commands": ["RELAY:1"]}
    assert post_ingest(c, {"gas": 300}).get_json()["commands"] == []                          # delivered only once
    rd = env["ingest"].read()
    assert rd.source == "http" and rd.valid and rd.gas_raw == 312 and rd.temperature == 27.4


def test_ingest_disabled_without_key_and_missing_fields_become_faults(env):
    sim = SimulatedSensorReader("normal", dt=0.5)
    system = HazardSystem(sim, storage=Storage())
    app = create_app(system, LiveState(), UserStore(system.storage), "k", ingest=QueueSensorReader(), device_key=None)
    assert app.test_client().post("/api/ingest", json={"gas": 1}, headers={"Authorization": "Bearer x"}).status_code == 404
    c = env["app"].test_client()
    post_ingest(c, {"gas": 300})                                                              # temp, hum, flame missing
    rd = env["ingest"].read()
    assert not rd.valid and "temperature:missing" in rd.faults


def test_queue_reader_reports_a_dead_link_as_a_fault_not_a_stale_value():
    r = QueueSensorReader(timeout=0.05)
    r.put({"gas": 300, "temp": 25, "hum": 50, "flame": 0})
    assert r.read().valid
    dead = r.read()
    assert not dead.valid and dead.faults == ["link:no_data"]
    for i in range(80):
        r.put({"gas": i, "temp": 25, "hum": 50, "flame": 0})                                  # bounded: oldest dropped
    assert r.q.qsize() <= 50 and r.read().gas_raw >= 30


def test_role_change_and_last_admin_protection(env):
    a = admin(env)
    t = token(a, "/users")
    a.post("/users/viewer1/role", data={"csrf_token": t, "role": "admin"})
    assert env["users"].role_of("viewer1") == "admin"
    a.post("/users/viewer1/role", data={"csrf_token": t, "role": "viewer"})
    assert env["users"].role_of("viewer1") == "viewer"
    a.post("/users/boss/role", data={"csrf_token": t, "role": "viewer"})            # boss is the only admin
    assert env["users"].role_of("boss") == "admin" and "last active admin" in flashes(a, "/users")
    a.post("/users/viewer1/role", data={"csrf_token": t, "role": "superuser"})
    assert env["users"].role_of("viewer1") == "viewer"
    v = viewer(env)
    assert v.post("/users/viewer1/role", data={"csrf_token": token(v, "/camera"), "role": "admin"}).status_code == 403
