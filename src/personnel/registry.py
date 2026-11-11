"""Personnel registry: who people are (name, designation) and their face templates.

PRIVACY: face templates are biometric data. Only numeric embeddings are stored (never photos), enrolment
requires a recorded consent flag, templates can be deleted (with the person's events), and nothing leaves
this machine. Use it only with the informed consent of the people enrolled and follow your site's rules.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from src import config
from src.storage import Storage

EMBED_DIM = 128          # SFace embedding size
MAX_SAMPLES = 10


def normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if v.shape[0] != EMBED_DIM or not np.isfinite(n) or n == 0.0:
        raise ValueError(f"embedding must be a finite non-zero vector of length {EMBED_DIM}")
    return v / n


def _clean(text: Optional[str], label: str, maxlen: int, required: bool = True) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        if required:
            raise ValueError(f"{label} is required")
        return None
    if len(t) > maxlen:
        raise ValueError(f"{label} must be at most {maxlen} characters")
    if any(ord(c) < 32 for c in t):
        raise ValueError(f"{label} contains control characters")
    return t


class PersonStore:
    def __init__(self, storage: Storage):
        self.s = storage
        self._lock = threading.Lock()
        self._cache = None

    # ---- people ----
    def add_person(self, name, designation, employee_id=None, consent: bool = False) -> int:
        name = _clean(name, "name", 80)
        designation = _clean(designation, "designation", 80)
        employee_id = _clean(employee_id, "employee id", 40, required=False)
        if not consent:
            raise ValueError("consent is required: the person must agree to face enrolment")
        now = time.time()
        pid = self.s.execute("INSERT INTO personnel(name,designation,employee_id,consent_ts,created_ts) VALUES(?,?,?,?,?)",
                             (name, designation, employee_id, now, now)).lastrowid
        self._invalidate()
        return pid

    def get(self, person_id: int) -> Optional[dict]:
        rows = self.s.query("SELECT id,name,designation,employee_id,disabled FROM personnel WHERE id=?", (int(person_id),))
        return rows[0] if rows else None

    def list_people(self) -> list:
        return self.s.query(
            "SELECT p.id,p.name,p.designation,p.employee_id,p.disabled,p.created_ts,"
            "(SELECT COUNT(*) FROM face_embeddings e WHERE e.person_id=p.id) AS samples "
            "FROM personnel p ORDER BY p.name")

    def set_disabled(self, person_id: int, disabled: bool) -> None:
        if not self.s.execute("UPDATE personnel SET disabled=? WHERE id=?", (int(disabled), int(person_id))).rowcount:
            raise ValueError("no such person")
        self._invalidate()

    def delete_person(self, person_id: int, purge_events: bool = True) -> None:
        if not self.get(person_id):
            raise ValueError("no such person")
        self.s.execute("DELETE FROM face_embeddings WHERE person_id=?", (int(person_id),))
        if purge_events:
            self.s.execute("DELETE FROM personnel_events WHERE person_id=?", (int(person_id),))
        self.s.execute("DELETE FROM personnel WHERE id=?", (int(person_id),))
        self._invalidate()

    # ---- templates ----
    def add_embedding(self, person_id: int, embedding) -> int:
        if not self.get(person_id):
            raise ValueError("no such person")
        if self.s.query("SELECT COUNT(*) AS c FROM face_embeddings WHERE person_id=?", (int(person_id),))[0]["c"] >= MAX_SAMPLES:
            raise ValueError(f"at most {MAX_SAMPLES} face samples per person")
        vec = normalize(embedding)
        n = self.s.execute("INSERT INTO face_embeddings(person_id,embedding,created_ts) VALUES(?,?,?)",
                           (int(person_id), vec.tobytes(), time.time())).lastrowid
        self._invalidate()
        return n

    def clear_embeddings(self, person_id: int) -> None:
        self.s.execute("DELETE FROM face_embeddings WHERE person_id=?", (int(person_id),))
        self._invalidate()

    def _invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def _gallery(self) -> list:
        with self._lock:
            if self._cache is None:
                rows = self.s.query(
                    "SELECT p.id,p.name,p.designation,e.embedding FROM face_embeddings e "
                    "JOIN personnel p ON p.id=e.person_id WHERE p.disabled=0 ORDER BY p.id")
                by = {}
                for r in rows:
                    d = by.setdefault(r["id"], [{"id": r["id"], "name": r["name"], "designation": r["designation"]}, []])
                    d[1].append(np.frombuffer(r["embedding"], dtype=np.float32))
                self._cache = [(p, np.vstack(v)) for p, v in by.values()]
            return self._cache

    def match(self, embedding, threshold: Optional[float] = None, margin: Optional[float] = None):
        """Return (person dict or None, best cosine score). A match needs the threshold AND a margin over the
        runner-up, so two similar-looking people are not confused."""
        thr = config.FACE_MATCH_THRESHOLD if threshold is None else threshold
        mrg = config.FACE_MATCH_MARGIN if margin is None else margin
        gallery = self._gallery()
        if not gallery:
            return None, 0.0
        q = normalize(embedding)
        scored = sorted(((float((M @ q).max()), p) for p, M in gallery), key=lambda t: t[0], reverse=True)
        best, person = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        if best >= thr and (best - second) >= mrg:
            return person, best
        return None, best

    # ---- events ----
    def log_event(self, kind: str, track_id=None, person: Optional[dict] = None, zone=None, detail="") -> None:
        self.s.execute(
            "INSERT INTO personnel_events(ts,kind,track_id,person_id,name,designation,zone,detail) VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), kind, track_id, person["id"] if person else None, person["name"] if person else None,
             person["designation"] if person else None, zone, detail))

    def recent_events(self, n: int = 50) -> list:
        return self.s.recent_people_events(n)
