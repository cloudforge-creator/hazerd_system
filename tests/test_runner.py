from src.runner import LiveState, Runner


class FakeStorage:
    def __init__(self):
        self.events = []

    def log_system(self, kind, detail):
        self.events.append((kind, detail))


class FakeLink:
    def __init__(self, error=None):
        self.error = error
        self.commands = []

    def send(self, command):
        if self.error:
            raise self.error
        self.commands.append(command)


class FakeSystem:
    def __init__(self, link):
        self.storage = FakeStorage()
        self.responder = type("Responder", (), {"link": link})()


def test_fail_safe_logs_serial_write_without_claiming_physical_shutdown():
    link = FakeLink()
    system = FakeSystem(link)
    live = LiveState()
    runner = Runner(system, live, cycle=0.5, pace=False)
    runner._fail_safe()
    assert link.commands == ["RELAY:0"]
    assert any(kind == "relay_shutdown_sent" and "acknowledgement not available" in detail
               for kind, detail in system.storage.events)
    assert live.crashed


def test_fail_safe_reports_send_failure_truthfully():
    system = FakeSystem(FakeLink(OSError("serial disconnected")))
    live = LiveState()
    runner = Runner(system, live, cycle=0.5, pace=False)
    runner._fail_safe()
    assert any(kind == "relay_shutdown_failed" and "serial disconnected" in detail
               for kind, detail in system.storage.events)
    assert not any(kind == "relay_shutdown_sent" for kind, _ in system.storage.events)
    assert live.errors == 1
