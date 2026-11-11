# Hardware shopping list and wiring

## Must buy (core prototype)
| # | Item | Qty | WBS | Notes |
|---|------|-----|-----|-------|
| 1 | Arduino Uno R3 + USB A-to-B cable | 1 | 2.3.1 | Clone is fine if it has a CH340/ATmega16U2 USB chip |
| 2 | MQ-2 gas sensor module (AO + DO pins) | 1 | 2.3.2 | Needs 24-48 h powered burn-in before first calibration |
| 3 | DHT22 / AM2302 temperature + humidity module | 1 | 2.3.3 | Prefer the 3-pin module (has pull-up resistor) |
| 4 | IR flame sensor module (LM393, 760-1100 nm) | 1 | 2.3.4 | DO output is active LOW |
| 5 | USB webcam, 720p or better | 1 | 2.3.5 | Any UVC webcam works with OpenCV |
| 6 | Active buzzer module, 5 V | 1 | 2.3.6 | Active type: sounds when pin is HIGH |
| 7 | 1-channel 5 V relay module with optocoupler | 1 | 2.3.7 | Buy 2-channel if you want a spare |
| 8 | 5 V 2 A USB adapter | 1 | 2.3.8 | Powers the Arduino and relay coil |
| 9 | Breadboard (830 points) | 1 | 2.3.9 | |
| 10 | Jumper wires: M-M, M-F, F-F (40 each) | 1 set | 2.3.9 | |
| 11 | Demo load: 12 V DC fan or 12 V lamp + 12 V adapter | 1 | 2.3.7 | Use LOW VOLTAGE only. Do NOT switch mains 220 V for the demo |

## Strongly recommended
- Red/green/yellow LEDs + 220 ohm resistors (status lights)
- Project box or acrylic base to mount everything
- Multimeter
- Camera tripod/stand
- Spare MQ-2 and flame sensor (cheap, and they fail)

## Optional upgrades
- ESP32 DevKit (Wi-Fi/MQTT instead of USB serial)
- 16x2 I2C LCD
- MQ-135 (air quality) for a second gas channel
- Raspberry Pi 4 for a standalone deployment demo

## Test materials (safe, small quantities, ventilated room)
- Butane lighter (release unlit gas near MQ-2), candle (flame sensor), hair dryer (temperature)

## Wiring (Arduino Uno)
| Part | Part pin | Arduino pin |
|------|----------|-------------|
| MQ-2 | VCC / GND / AO | 5V / GND / A0 |
| DHT22 | VCC / GND / DATA | 5V / GND / D2 |
| Flame sensor | VCC / GND / DO | 5V / GND / D3 |
| Relay | VCC / GND / IN | 5V / GND / D7 |
| Buzzer | + / - | D8 / GND |

Relay load side: COM and NO in series with the 12 V load. Set RELAY_ACTIVE_LOW in the .ino if your relay behaves inverted.
