from src.sensor_reader import Reading


class FakeLink:
    def __init__(self):
        self.sent = []

    def send(self, cmd):
        self.sent.append(cmd)

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


def reading(ts, gas=180, temp=27.0, hum=55.0, flame=0, faults=None):
    return Reading(timestamp=ts, gas_raw=gas, temperature=temp, humidity=hum,
                   flame=flame, faults=list(faults or []), source="test")
