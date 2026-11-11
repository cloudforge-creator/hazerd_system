"""SQLite storage (WBS 3.4, 9.6). Thread-safe: the monitoring loop and the web dashboard share it."""
from __future__ import annotations

import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, source TEXT, gas_raw INTEGER,
  temperature REAL, humidity REAL, flame INTEGER, faults TEXT,
  level TEXT, score REAL, predicted_score REAL);
CREATE TABLE IF NOT EXISTS hazard_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, score REAL,
  reasons TEXT, actions TEXT);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, message TEXT,
  status TEXT DEFAULT 'ACTIVE');
CREATE TABLE IF NOT EXISTS system_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS personnel(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, designation TEXT NOT NULL,
  employee_id TEXT, disabled INTEGER NOT NULL DEFAULT 0, consent_ts REAL, created_ts REAL);
CREATE TABLE IF NOT EXISTS face_embeddings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL, embedding BLOB NOT NULL, created_ts REAL);
CREATE TABLE IF NOT EXISTS personnel_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, track_id INTEGER, person_id INTEGER,
  name TEXT, designation TEXT, zone TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE COLLATE NOCASE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('admin','viewer')),
  disabled INTEGER NOT NULL DEFAULT 0,
  created_ts REAL);
"""
TABLES = {"sensor_readings", "hazard_events", "alerts", "system_events", "users", "personnel", "face_embeddings",
          "personnel_events"}


class Storage:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.executescript(SCHEMA)

    # -- generic helpers (used by UserStore too) --
    def execute(self, sql: str, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def query(self, sql: str, args=()) -> list:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # -- domain methods --
    def log_cycle(self, reading, decision) -> None:
        self.execute(
            "INSERT INTO sensor_readings(ts,source,gas_raw,temperature,humidity,flame,faults,"
            "level,score,predicted_score) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (reading.timestamp, reading.source, reading.gas_raw, reading.temperature,
             reading.humidity, reading.flame, ";".join(reading.faults),
             decision.level.value, decision.score, decision.predicted_score))

    def log_event(self, decision) -> None:
        self.execute("INSERT INTO hazard_events(ts,level,score,reasons,actions) VALUES(?,?,?,?,?)",
                     (decision.timestamp, decision.level.value, decision.score,
                      ";".join(decision.reasons), ";".join(decision.actions)))

    def add_alert(self, ts: float, level: str, message: str) -> int:
        return self.execute("INSERT INTO alerts(ts,level,message) VALUES(?,?,?)",
                            (ts, level, message)).lastrowid

    def acknowledge_alert(self, alert_id: int) -> bool:
        return self.execute("UPDATE alerts SET status='ACKNOWLEDGED' WHERE id=? AND status='ACTIVE'",
                            (alert_id,)).rowcount > 0

    def log_system(self, kind: str, detail: str = "") -> None:
        self.execute("INSERT INTO system_events(ts,kind,detail) VALUES(?,?,?)", (time.time(), kind, detail))

    def recent_alerts(self, n: int = 50, active_only: bool = False) -> list:
        where = "WHERE status='ACTIVE' " if active_only else ""
        return self.query(f"SELECT * FROM alerts {where}ORDER BY id DESC LIMIT ?", (n,))

    def recent_events(self, n: int = 50) -> list:
        return self.query("SELECT * FROM hazard_events ORDER BY id DESC LIMIT ?", (n,))

    def recent_readings(self, n: int = 100) -> list:
        return self.query("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?", (n,))

    def recent_people_events(self, n: int = 50) -> list:
        return self.query("SELECT * FROM personnel_events ORDER BY id DESC LIMIT ?", (n,))

    def recent_system(self, n: int = 50) -> list:
        return self.query("SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (n,))

    def prune_readings(self, keep_rows: int) -> int:
        """Keep only the newest `keep_rows` sensor readings (by id, so it is clock-independent).
        Alerts, hazard events and system events are small and are kept."""
        return self.execute("DELETE FROM sensor_readings WHERE id <= (SELECT MAX(id) FROM sensor_readings) - ?",
                            (int(keep_rows),)).rowcount

    def count(self, table: str) -> int:
        assert table in TABLES
        return self.query(f"SELECT COUNT(*) AS c FROM {table}")[0]["c"]

    def close(self) -> None:
        with self.lock:
            self.db.close()
