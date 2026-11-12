from src.decision_engine import Decision, Level
from src.response import LogNotifier, ResponseManager
from tests.helpers import FakeLink


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(level, early=False, score=0.8, actions=None):
    default = {"CRITICAL": ["LOG_ALERT", "BUZZER_ON", "NOTIFY_ADMIN", "RELAY_SHUTDOWN"],
               "WARNING": ["LOG_ALERT", "NOTIFY_ADMIN"], "SAFE": []}
    return Decision(0, level, score, 0, 0, 0, score, level, early, False, ["x"],
                    actions if actions is not None else default[level.value])


def setup():
    link, clock, note = FakeLink(), Clock(), LogNotifier()
    return link, clock, note, ResponseManager(link, [note], clock, notify_cooldown=30)


def test_arm_powers_machine():
    link, _, _, r = setup()
    r.arm()
    assert link.sent == ["RELAY:1"]


def test_critical_sounds_buzzer_notifies_and_shuts_down_once():
    link, _, note, r = setup()
    done = r.execute(make(Level.CRITICAL))
    assert set(done) == {"BUZZER_ON", "RELAY_SHUTDOWN", "NOTIFY_ADMIN"}
    assert "BUZZER:1" in link.sent and "RELAY:0" in link.sent
    r.execute(make(Level.CRITICAL))
    assert link.sent.count("RELAY:0") == 1 and link.sent.count("BUZZER:1") == 1
    assert len(note.messages) == 1


def test_notification_cooldown_and_escalation_bypass():
    link, clock, note, r = setup()
    r.execute(make(Level.WARNING))
    clock.t = 5
    r.execute(make(Level.WARNING))
    assert len(note.messages) == 1            # cooled down
    r.execute(make(Level.CRITICAL))
    assert len(note.messages) == 2            # escalation notifies immediately
    clock.t = 100
    r.execute(make(Level.CRITICAL))
    assert len(note.messages) == 3            # cooldown elapsed


def test_buzzer_turns_off_when_safe_but_relay_stays_latched():
    link, _, _, r = setup()
    r.execute(make(Level.CRITICAL))
    done = r.execute(make(Level.SAFE))
    assert "BUZZER_OFF" in done and r.relay_latched


def test_reset_latch_refused_unless_safe():
    link, _, _, r = setup()
    r.execute(make(Level.CRITICAL))
    assert r.reset_latch(Level.WARNING) is False and r.relay_latched
    assert r.reset_latch(Level.SAFE) is True and not r.relay_latched
    assert link.sent[-1] == "RELAY:1"


def test_early_warning_notifies_without_shutdown():
    link, _, note, r = setup()
    r.execute(make(Level.SAFE, early=True, actions=["LOG_ALERT", "NOTIFY_ADMIN"]))
    assert len(note.messages) == 1 and "RELAY:0" not in link.sent


def test_arm_refreshes_watchdog_before_enabling_serial_relay():
    class HeartbeatLink:
        def __init__(self):
            self.events = []

        def heartbeat(self):
            self.events.append("HEARTBEAT")

        def send(self, command):
            self.events.append(command)

    link = HeartbeatLink()
    response = ResponseManager(link)
    response.arm()
    assert link.events == ["HEARTBEAT", "RELAY:1"]
