import json
import time

import numpy as np
import pytest

from src import config
from src.decision_engine import DecisionEngine, Level, Signals
from src.personnel.registry import EMBED_DIM, MAX_SAMPLES, PersonStore, normalize
from src.personnel.tracker import FaceTracker, iou
from src.personnel.zones import Zone, load_zones, point_in_polygon, save_zones, validate_zones
from src.response import LogNotifier, ResponseManager
from src.signals import CombinedSignals
from src.storage import Storage
from tests.helpers import FakeLink, reading


def vec(seed, noise=0.0, noise_seed=0):
    v = np.random.default_rng(seed).normal(size=EMBED_DIM)
    if noise:
        v = v + np.random.default_rng(1000 + noise_seed).normal(size=EMBED_DIM) * noise
    return normalize(v)


@pytest.fixture
def store():
    return PersonStore(Storage())


def person(store, name="Ali Khan", des="Operator", seed=1, samples=2):
    pid = store.add_person(name, des, "E1", consent=True)
    for i in range(samples):
        store.add_embedding(pid, vec(seed, 0.15, i))
    return pid


# ---------------- registry ----------------
def test_enrolment_requires_consent_and_clean_text(store):
    with pytest.raises(ValueError, match="consent"):
        store.add_person("A B", "Operator", consent=False)
    with pytest.raises(ValueError):
        store.add_person("", "Operator", consent=True)
    with pytest.raises(ValueError):
        store.add_person("Ali\x00", "Operator", consent=True)
    with pytest.raises(ValueError):
        store.add_person("x" * 81, "Operator", consent=True)
    assert store.list_people() == []


def test_embedding_validation_and_limit(store):
    pid = store.add_person("Sara", "Engineer", consent=True)
    for bad in (np.zeros(EMBED_DIM), np.ones(5), np.full(EMBED_DIM, np.nan)):
        with pytest.raises(ValueError):
            store.add_embedding(pid, bad)
    for i in range(MAX_SAMPLES):
        store.add_embedding(pid, vec(i + 5))
    with pytest.raises(ValueError, match="at most"):
        store.add_embedding(pid, vec(99))
    with pytest.raises(ValueError):
        store.add_embedding(12345, vec(1))


def test_match_identifies_right_person_and_rejects_strangers(store):
    a, b = person(store, "Ali Khan", "Operator", 1), person(store, "Sara Noor", "Supervisor", 2)
    p, s = store.match(vec(1, 0.2, 77))
    assert p["id"] == a and p["designation"] == "Operator" and s > config.FACE_MATCH_THRESHOLD
    assert store.match(vec(2, 0.2, 78))[0]["id"] == b
    stranger, score = store.match(vec(500))
    assert stranger is None and score < config.FACE_MATCH_THRESHOLD
    assert PersonStore(Storage()).match(vec(1)) == (None, 0.0)           # empty gallery


def test_margin_blocks_lookalikes(store):
    person(store, "Twin One", "Operator", 7, samples=1)
    pid2 = store.add_person("Twin Two", "Operator", consent=True)
    store.add_embedding(pid2, vec(7, 0.02, 5))                            # almost identical template
    who, best = store.match(vec(7, 0.02, 9))
    assert best > config.FACE_MATCH_THRESHOLD and who is None            # ambiguous -> not guessed


def test_disabled_and_deleted_people_are_not_matched_and_data_is_purged(store):
    a = person(store, seed=3)
    assert store.match(vec(3))[0]["id"] == a
    store.set_disabled(a, True)
    assert store.match(vec(3))[0] is None
    store.set_disabled(a, False)
    store.log_event("PERSON_SEEN", 1, store.get(a))
    store.delete_person(a)
    assert store.match(vec(3))[0] is None and store.list_people() == []
    assert store.s.count("face_embeddings") == 0 and store.recent_events() == []
    with pytest.raises(ValueError):
        store.delete_person(a)


def test_gallery_cache_sees_new_enrolments(store):
    assert store.match(vec(4))[0] is None
    person(store, seed=4)
    assert store.match(vec(4))[0] is not None


# ---------------- zones ----------------
def test_point_in_polygon():
    sq = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    assert point_in_polygon(0.5, 0.5, sq) and not point_in_polygon(0.1, 0.5, sq) and not point_in_polygon(0.5, 0.9, sq)
    tri = [(0, 0), (1, 0), (0, 1)]
    assert point_in_polygon(0.2, 0.2, tri) and not point_in_polygon(0.8, 0.8, tri)


def test_zone_validation_is_strict_and_all_or_nothing():
    good = [{"name": "Press", "points": [[0, 0], [1, 0], [1, 1]], "allowed_designations": ["Operator"]}]
    zs, errs = validate_zones(good)
    assert not errs and zs[0].hazard_when_alert is True and zs[0].allowed_designations == ["Operator"]
    bad_cases = [
        "nope", [{"name": "", "points": [[0, 0], [1, 0], [1, 1]]}], [{"name": "A", "points": [[0, 0], [1, 0]]}],
        [{"name": "A", "points": [[0, 0], [1, 0], [2, 1]]}], [{"name": "A", "points": [[0, 0], [1, 0], [True, 1]]}],
        [{"name": "A", "points": [[0, 0], [1, 0], [1, 1]], "hazard_when_alert": "yes"}],
        [{"name": "A", "points": [[0, 0], [1, 0], [1, 1]]}, {"name": "a", "points": [[0, 0], [1, 0], [1, 1]]}],
        [{"name": "A", "points": [[0, 0], [1, 0], [1, 1]], "allowed_designations": [5]}]]
    for raw in bad_cases:
        zs, errs = validate_zones(raw)
        assert errs and zs == []


def test_zones_roundtrip_and_corrupt_file(tmp_path):
    p = str(tmp_path / "z.json")
    assert load_zones(p) == []
    save_zones(p, [Zone("A", [(0, 0), (1, 0), (1, 1)], False, ["Operator"])])
    z = load_zones(p)[0]
    assert z.name == "A" and z.hazard_when_alert is False and z.allowed_designations == ["Operator"]
    open(p, "w").write("{not json")
    assert load_zones(p) == []


# ---------------- tracker ----------------
def det(box, p=None, score=0.8, small=False):
    return {"box": box, "person": p, "score": score, "small": small}


A = {"id": 1, "name": "Ali Khan", "designation": "Operator"}
B = {"id": 2, "name": "Sara Noor", "designation": "Supervisor"}


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_identity_needs_votes_and_follows_the_person():
    tr, box = FaceTracker(), [100, 100, 80, 80]
    for i in range(config.PERSON_MIN_VOTES - 1):
        t = tr.update([det(tuple(box), A)], i)[0][0]
        assert t.person is None and t.status == "identifying"
        box[0] += 6
    t = tr.update([det(tuple(box), A)], 10)[0][0]
    assert t.person == A and t.id == 1 and t.label() == "Ali Khan (Operator)"
    for i in range(5):                                   # moves a bit, same track id, same identity
        box[0] += 8
        t = tr.update([det(tuple(box), None)], 20 + i)[0][0]     # unmatched frames do not erase identity
    assert t.id == 1 and t.person == A


def test_unmatched_face_becomes_unknown_then_can_still_be_recognised():
    tr = FaceTracker()
    for i in range(config.PERSON_UNKNOWN_AFTER):
        t = tr.update([det((50, 50, 80, 80), None)], i)[0][0]
    assert t.unknown and t.label() == "Unknown person #1"
    for i in range(config.PERSON_MIN_VOTES):
        t = tr.update([det((50, 50, 80, 80), A)], 10 + i)[0][0]
    assert t.person == A and not t.unknown


def test_small_faces_are_tracked_but_never_identified():
    tr = FaceTracker()
    for i in range(10):
        t = tr.update([det((10, 10, 30, 30), A, small=True)], i)[0][0]
    assert t.person is None and not t.unknown and t.hits == 10


def test_two_people_do_not_swap_and_tracks_expire():
    tr = FaceTracker()
    for i in range(5):
        tracks, _ = tr.update([det((20 + i * 3, 50, 70, 70), A), det((300 - i * 3, 60, 70, 70), B)], i)
    names = {t.person["name"] for t in tracks}
    assert names == {"Ali Khan", "Sara Noor"} and len({t.id for t in tracks}) == 2
    closed_all = []
    for i in range(config.PERSON_TRACK_MAX_MISSED + 2):
        _, closed = tr.update([], 100 + i)
        closed_all += closed
    assert len(closed_all) == 2 and tr.tracks == []


def test_identity_switch_needs_consistent_contrary_evidence():
    tr = FaceTracker()
    for i in range(4):
        t = tr.update([det((50, 50, 80, 80), A)], i)[0][0]
    for i in range(3):                                          # three stray B matches are not enough
        t = tr.update([det((50, 50, 80, 80), B)], 10 + i)[0][0]
    assert t.person == A
    for i in range(8):
        t = tr.update([det((50, 50, 80, 80), B)], 20 + i)[0][0]
    assert t.person == B


# ---------------- decision engine / response / signals ----------------
def test_people_in_hazard_triggers_emergency_only_when_system_is_not_safe():
    d = DecisionEngine().decide(reading(0, gas=700), Signals(people_in_hazard=["Ali Khan (Operator)"]))
    assert "NOTIFY_EMERGENCY" in d.actions and "people_in_hazard_zone=Ali Khan (Operator)" in d.reasons
    assert d.level == Level.CRITICAL
    safe = DecisionEngine().decide(reading(0), Signals(people_in_hazard=["Ali Khan (Operator)"]))
    assert safe.level == Level.SAFE and "NOTIFY_EMERGENCY" not in safe.actions
    plain = DecisionEngine().decide(reading(0, gas=700))
    assert "NOTIFY_EMERGENCY" not in plain.actions


def test_emergency_notification_names_people_and_respects_cooldown():
    link, note, t = FakeLink(), LogNotifier(), [0.0]
    r = ResponseManager(link, [note], clock=lambda: t[0], notify_cooldown=30)
    eng = DecisionEngine()
    sig = Signals(people_in_hazard=["Ali Khan (Operator)", "Unknown person #2"])
    r.execute(eng.decide(reading(0, gas=700), sig))
    subjects = [s for s, _ in note.messages]
    assert "[EMERGENCY] person in hazard zone" in subjects
    body = dict(note.messages)["[EMERGENCY] person in hazard zone"]
    assert "Ali Khan (Operator)" in body and "Unknown person #2" in body
    n = len(note.messages)
    t[0] = 5
    r.execute(eng.decide(reading(1, gas=700), sig))
    assert len([s for s, _ in note.messages if s.startswith("[EMERGENCY]")]) == 1       # cooled down
    t[0] = 100
    r.execute(eng.decide(reading(2, gas=700), sig))
    assert len([s for s, _ in note.messages if s.startswith("[EMERGENCY]")]) == 2


def test_combined_signals_merges_people_lists():
    a = lambda r: Signals(people_in_hazard=["B (x)", "A (y)"])
    b = lambda r: Signals(people_in_hazard=["A (y)"])
    assert CombinedSignals(a, b)(reading(0)).people_in_hazard == ["A (y)", "B (x)"]
    assert CombinedSignals(lambda r: Signals())(reading(0)).people_in_hazard is None


# ---------------- monitor + hub (need OpenCV for drawing/encoding) ----------------
cv2 = pytest.importorskip("cv2")
from src.camera_hub import MAX_STREAM_CLIENTS, CameraHub  # noqa: E402
from src.personnel.face import Detection, FaceEngine  # noqa: E402
from src.personnel.monitor import PersonnelMonitor, ascii_label  # noqa: E402


class StubEngine:
    """Returns scripted detections per frame; embeddings come from the registry seeds."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def detect_and_embed(self, frame, embed=True):
        self.calls += 1
        if isinstance(self.script[0], Exception):
            raise self.script[0]
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


class StaticSource:
    def __init__(self):
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.closed = False

    def read(self):
        return self.frame

    def close(self):
        self.closed = True


def make_monitor(store, script, zones, level=Level.SAFE):
    hub = CameraHub(StaticSource())
    lv = {"v": level}
    mon = PersonnelMonitor(hub, StubEngine(script), store, lambda: zones, lambda: lv["v"])
    return mon, hub, lv


FRAME = np.zeros((240, 320, 3), np.uint8)
ZONE = Zone("Press line", [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)], True, ["Operator"])    # left half


def feed(mon, n, start=0.0):
    for i in range(n):
        mon.process(FRAME, start + i)


def test_known_person_is_named_logged_and_zone_aware(store):
    pid = person(store, "Ali Khan", "Operator", 1)
    d = Detection((20, 80, 70, 70), 0.9, vec(1, 0.2, 3))                    # left half, big enough
    mon, hub, lv = make_monitor(store, [[d]], [ZONE])
    feed(mon, 5)
    snap = mon.snapshot()[0]
    assert snap["name"] == "Ali Khan" and snap["designation"] == "Operator" and snap["zones"] == ["Press line"]
    kinds = [e["kind"] for e in store.recent_events(50)]
    assert "PERSON_SEEN" in kinds and "ZONE_ENTER" in kinds and "UNAUTHORIZED_ZONE" not in kinds    # Operator is allowed
    assert mon.people_in_hazard() == [] and mon(reading(0)) == Signals()


def test_person_in_hazard_zone_during_alert_is_reported_once_and_cleared(store):
    person(store, "Ali Khan", "Operator", 1)
    d = Detection((20, 80, 70, 70), 0.9, vec(1, 0.2, 3))
    mon, hub, lv = make_monitor(store, [[d]], [ZONE], level=Level.CRITICAL)
    feed(mon, 6)
    assert mon.people_in_hazard() == ["Ali Khan (Operator)"]
    assert mon(reading(0)).people_in_hazard == ["Ali Khan (Operator)"]
    feed(mon, 4, start=10)
    assert [e["kind"] for e in store.recent_events(100)].count("PERSON_IN_HAZARD") == 1
    lv["v"] = Level.SAFE
    feed(mon, 2, start=20)
    assert mon.people_in_hazard() == []
    lv["v"] = Level.WARNING                                                   # a new episode is reported again
    feed(mon, 2, start=30)
    assert [e["kind"] for e in store.recent_events(100)].count("PERSON_IN_HAZARD") == 2


def test_unknown_person_in_restricted_zone_is_flagged(store):
    person(store, "Ali Khan", "Operator", 1)
    d = Detection((20, 80, 70, 70), 0.9, vec(999))                            # not enrolled
    mon, hub, lv = make_monitor(store, [[d]], [ZONE])
    feed(mon, 8)
    ev = {e["kind"]: e for e in store.recent_events(100)}
    assert "UNKNOWN_PERSON" in ev and ev["UNAUTHORIZED_ZONE"]["detail"] == "unknown person"
    assert mon.snapshot()[0]["status"] == "unknown"


def test_wrong_designation_is_unauthorised_but_other_zone_is_fine(store):
    person(store, "Sara Noor", "Visitor", 2)
    d = Detection((20, 80, 70, 70), 0.9, vec(2, 0.2, 4))
    mon, hub, lv = make_monitor(store, [[d]], [ZONE])
    feed(mon, 6)
    assert any(e["kind"] == "UNAUTHORIZED_ZONE" and e["name"] == "Sara Noor" for e in store.recent_events(100))
    d_right = Detection((230, 80, 70, 70), 0.9, vec(2, 0.2, 4))               # right half: outside the zone
    mon2, _, _ = make_monitor(PersonStore(Storage()), [[d_right]], [ZONE])
    assert mon2.process(FRAME, 0)[0]["zones"] == []


def test_person_leaving_is_logged_and_small_faces_stay_unidentified(store):
    person(store, "Ali Khan", "Operator", 1)
    d = Detection((20, 80, 70, 70), 0.9, vec(1, 0.2, 3))
    mon, hub, lv = make_monitor(store, [[d], [d], [d], [d], []], [])
    feed(mon, 4)
    mon.engine.script = [[]]
    feed(mon, config.PERSON_TRACK_MAX_MISSED + 3, start=10)
    assert any(e["kind"] == "PERSON_LEFT" for e in store.recent_events(100)) and mon.snapshot() == []
    tiny = Detection((10, 10, 20, 20), 0.9, vec(1, 0.1, 1))
    mon2, _, _ = make_monitor(store, [[tiny]], [])
    feed(mon2, 6)
    assert mon2.snapshot()[0]["status"] == "identifying" and mon2.snapshot()[0]["name"] is None


def test_annotated_frame_is_published_to_the_hub(store):
    person(store, "Ali Khan", "Operator", 1)
    d = Detection((20, 80, 70, 70), 0.9, vec(1, 0.2, 3))
    mon, hub, lv = make_monitor(store, [[d]], [ZONE], level=Level.WARNING)
    feed(mon, 5)
    jpg = hub.jpeg()
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == FRAME.shape and img.sum() > 0                          # something was drawn
    assert ascii_label("Ahmad عمر") == "Ahmad ???"


def test_monitor_thread_survives_errors_and_stops(store):
    hub = CameraHub(StaticSource(), fps=50).start()
    mon = PersonnelMonitor(hub, StubEngine([RuntimeError("boom")]), store, lambda: [], lambda: Level.SAFE)
    mon.start()
    t0 = time.time()
    while mon.errors < 2 and time.time() - t0 < 5:
        time.sleep(0.05)
    assert mon.errors >= 2 and mon.is_alive() and "boom" in mon.last_error
    mon.stop()
    hub.stop()
    assert not mon.is_alive()


def test_hub_latest_staleness_failures_and_stream_cap():
    now = [100.0]
    src = StaticSource()
    hub = CameraHub(src, fps=100, stale_s=2.0, clock=lambda: now[0]).start()
    t0 = time.time()
    while hub.frames < 3 and time.time() - t0 < 5:
        time.sleep(0.01)
    f1, _ = hub.latest()
    f1[:] = 255                                                                # a copy: must not alter the hub's frame
    assert hub.latest()[0].sum() == 0 and hub.read() is not None
    assert cv2.imdecode(np.frombuffer(hub.jpeg(), np.uint8), 1).shape == FRAME.shape
    now[0] += 10                                                               # frozen camera -> no frame
    hub._stop.set(); hub._thread.join(timeout=2)
    assert hub.latest() == (None, 0.0) and hub.read() is None and hub.jpeg() is None
    assert all(hub.acquire_stream() for _ in range(MAX_STREAM_CLIENTS)) and not hub.acquire_stream()
    hub.release_stream()
    assert hub.acquire_stream()

    class Broken:
        def read(self):
            raise OSError("camera unplugged")
    b = CameraHub(Broken(), fps=100).start()
    time.sleep(0.2)
    assert b.failures > 0 and b.latest()[0] is None
    b.stop()


def test_face_engine_reports_missing_models_clearly():
    ok, msg = FaceEngine.models_present()
    if not ok:
        assert "missing" in msg
        with pytest.raises(FileNotFoundError, match="download_face_models"):
            FaceEngine()
