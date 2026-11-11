// ESP32 sensor node that POSTs readings over Wi-Fi to the hazard dashboard (HTTP ingest).
// NOT TESTED ON A BOARD (written without hardware): treat as a starting point and test carefully.
// Libraries: WiFi, HTTPClient (bundled with the ESP32 core), "DHT sensor library" (Adafruit), ArduinoJson.
// Server side: start the dashboard with  --http-ingest  and set the env var HAZARD_DEVICE_KEY to the same key.
// Use https:// behind a TLS proxy for anything beyond a lab bench: the key travels in a header.

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.50:5000/api/ingest";   // PC running the dashboard
const char* DEVICE_KEY = "CHANGE_ME_SAME_AS_HAZARD_DEVICE_KEY";

#define DHTPIN 4
#define DHTTYPE DHT22
const int GAS_PIN = 34;      // MQ-2 AO via a voltage divider (ESP32 ADC max is 3.3 V!)
const int FLAME_PIN = 27;    // flame sensor DO (active LOW)
const int RELAY_PIN = 26;    // relay IN
const int BUZZER_PIN = 25;
const bool RELAY_ACTIVE_LOW = true;
DHT dht(DHTPIN, DHTTYPE);
unsigned long lastSend = 0;

void setRelay(bool on) { digitalWrite(RELAY_PIN, (on ^ RELAY_ACTIVE_LOW) ? HIGH : LOW); }

void apply(const String& c) {
  if (c == "RELAY:1") setRelay(true);
  else if (c == "RELAY:0") setRelay(false);
  else if (c == "BUZZER:1") digitalWrite(BUZZER_PIN, HIGH);
  else if (c == "BUZZER:0") digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(FLAME_PIN, INPUT); pinMode(RELAY_PIN, OUTPUT); pinMode(BUZZER_PIN, OUTPUT);
  setRelay(false);                                   // safe at boot: the server arms the relay
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); }
}

void loop() {
  if (millis() - lastSend < 500) return;
  lastSend = millis();
  if (WiFi.status() != WL_CONNECTED) { setRelay(false); WiFi.reconnect(); return; }   // lost link -> fail safe

  StaticJsonDocument<192> doc;
  doc["gas"] = analogRead(GAS_PIN) / 4;              // 12-bit ADC -> 10-bit scale used by the server
  float t = dht.readTemperature(), h = dht.readHumidity();
  if (!isnan(t)) doc["temp"] = t;                    // missing fields become faults on the server, never zeros
  if (!isnan(h)) doc["hum"] = h;
  doc["flame"] = digitalRead(FLAME_PIN) == LOW ? 1 : 0;
  String body; serializeJson(doc, body);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", String("Bearer ") + DEVICE_KEY);
  int code = http.POST(body);
  if (code == 200) {
    StaticJsonDocument<256> reply;
    if (!deserializeJson(reply, http.getString())) {
      for (JsonVariant c : reply["commands"].as<JsonArray>()) apply(String((const char*)c));
    }
  } else {
    setRelay(false);                                 // server unreachable / rejected -> fail safe
  }
  http.end();
}
