// Predictive AI Hazard System - sensor node (Arduino Uno)
// Output (every 500 ms):  GAS:312,TEMP:27.4,HUM:55.0,FLAME:0
// Commands from PC:       RELAY:1 | RELAY:0 | BUZZER:1 | BUZZER:0
// Library needed: "DHT sensor library" by Adafruit (+ Adafruit Unified Sensor)

#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT22

const int GAS_PIN = A0;     // MQ-2 AO
const int FLAME_PIN = 3;    // flame sensor DO
const int RELAY_PIN = 7;    // relay IN
const int BUZZER_PIN = 8;   // buzzer +
const bool RELAY_ACTIVE_LOW = true;  // most relay modules energize on LOW

DHT dht(DHTPIN, DHTTYPE);
unsigned long lastSend = 0;
String cmd = "";

void setRelay(bool on) {
  digitalWrite(RELAY_PIN, (on ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

void handleCommand(const String &c) {
  if (c == "RELAY:1") setRelay(true);
  else if (c == "RELAY:0") setRelay(false);
  else if (c == "BUZZER:1") digitalWrite(BUZZER_PIN, HIGH);
  else if (c == "BUZZER:0") digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(9600);
  pinMode(FLAME_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  setRelay(false);                 // safe state at boot
  digitalWrite(BUZZER_PIN, LOW);
  dht.begin();
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { cmd.trim(); handleCommand(cmd); cmd = ""; }
    else cmd += ch;
  }

  if (millis() - lastSend >= 500) {
    lastSend = millis();
    int gas = analogRead(GAS_PIN);
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    int flame = (digitalRead(FLAME_PIN) == LOW) ? 1 : 0;  // module is active LOW

    Serial.print("GAS:");   Serial.print(gas);
    Serial.print(",TEMP:"); Serial.print(isnan(t) ? "NAN" : String(t, 1));
    Serial.print(",HUM:");  Serial.print(isnan(h) ? "NAN" : String(h, 1));
    Serial.print(",FLAME:"); Serial.println(flame);
  }
}
