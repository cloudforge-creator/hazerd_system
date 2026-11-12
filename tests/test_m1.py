from src.sensor_reader import build_reading, parse_line
from src.simulator import SCENARIOS, SimulatedSensorReader


def run(scenario, n=60):
    sim = SimulatedSensorReader(scenario, dt=1.0)
    return [sim.read() for _ in range(n)]


def test_parse_line():
    assert parse_line("GAS:312,TEMP:27.4,HUM:55.0,FLAME:0\n") == {
        "GAS": "312", "TEMP": "27.4", "HUM": "55.0", "FLAME": "0"}


def test_valid_reading():
    r = build_reading(parse_line("GAS:312,TEMP:27.4,HUM:55.0,FLAME:0"), "serial")
    assert r.valid and r.gas_raw == 312 and r.flame == 0


def test_nan_temperature_is_a_fault_not_a_zero():
    r = build_reading(parse_line("GAS:312,TEMP:NAN,HUM:55.0,FLAME:0"), "serial")
    assert not r.valid and "temperature:missing" in r.faults


def test_out_of_range_is_a_fault():
    r = build_reading(parse_line("GAS:2000,TEMP:27,HUM:55,FLAME:0"), "serial")
    assert "gas_raw:out_of_range" in r.faults


def test_missing_field_is_a_fault():
    r = build_reading(parse_line("GAS:300,TEMP:27"), "serial")
    assert "humidity:missing" in r.faults and "flame:missing" in r.faults


def test_all_scenarios_produce_valid_readings():
    for name in SCENARIOS:
        assert all(r.valid for r in run(name))


def test_gas_leak_escalates():
    gas = [r.gas_raw for r in run("gas_leak")]
    assert gas[0] < 300 and max(gas) > 600


def test_fire_triggers_flame():
    flame = [r.flame for r in run("fire")]
    assert flame[0] == 0 and flame[-1] == 1


def test_simulator_is_deterministic():
    a = SimulatedSensorReader("normal", seed=1).read().gas_raw
    b = SimulatedSensorReader("normal", seed=1).read().gas_raw
    assert a == b


def test_non_finite_sensor_values_become_faults_not_exceptions():
    for value in ("Infinity", "-Infinity", "NaN"):
        r = build_reading(parse_line(f"GAS:{value},TEMP:25,HUM:50,FLAME:0"), "serial")
        assert not r.valid and r.gas_raw is None and "gas_raw:missing" in r.faults
    r = build_reading(parse_line("GAS:300,TEMP:25,HUM:50,FLAME:Infinity"), "serial")
    assert not r.valid and r.flame is None and "flame:missing" in r.faults


def test_integer_sensor_fields_reject_fractional_values():
    r = build_reading(parse_line("GAS:12.5,TEMP:25,HUM:50,FLAME:0.5"), "serial")
    assert not r.valid and r.gas_raw is None and r.flame is None
    assert "gas_raw:not_integer" in r.faults and "flame:not_integer" in r.faults
