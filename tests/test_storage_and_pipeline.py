from src.decision_engine import Level
from src.pipeline import HazardSystem
from src.simulator import SimulatedSensorReader
from src.storage import Storage


def run(scenario, seconds=60):
    sim = SimulatedSensorReader(scenario, dt=0.5)
    sent = []
    sim.send = sent.append
    system = HazardSystem(sim, storage=Storage())
    system.start()
    for _ in range(int(seconds / 0.5)):
        system.step()
    return system, sent


def test_normal_run_has_no_alerts_and_no_shutdown():
    system, sent = run("normal", 60)
    assert system.storage.count("alerts") == 0
    assert "RELAY:0" not in sent and sent[0] == "RELAY:1"
    assert system.storage.count("sensor_readings") == 120


def test_gas_leak_end_to_end():
    system, sent = run("gas_leak", 60)
    kinds = [a["level"] for a in reversed(system.storage.recent_alerts())]
    assert kinds[0] == "EARLY_WARNING" and kinds[-1] == "CRITICAL"
    assert "WARNING" in kinds
    assert "RELAY:0" in sent and "BUZZER:1" in sent
    assert system.last_decision.level == Level.CRITICAL
    assert system.storage.count("hazard_events") >= 3


def test_fire_end_to_end_shuts_down():
    system, sent = run("fire", 40)
    assert "RELAY:0" in sent


def test_acknowledge_alert():
    system, _ = run("gas_leak", 60)
    first = system.storage.recent_alerts(active_only=True)[0]
    system.storage.acknowledge_alert(first["id"])
    ids = [a["id"] for a in system.storage.recent_alerts(active_only=True)]
    assert first["id"] not in ids
