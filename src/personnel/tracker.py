"""Tracks faces across frames so a person keeps ONE identity (and one set of events) as they move.

Association is greedy IoU with a centre-distance fallback. Identity is decided by VOTING over recent
matches, not one frame: it needs PERSON_MIN_VOTES consistent matches to confirm, and 4 consecutive
contrary matches to switch. A face that never matches becomes 'Unknown' after PERSON_UNKNOWN_AFTER tries.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from src import config


@dataclass
class Track:
    id: int
    box: tuple
    first_ts: float
    last_ts: float
    hits: int = 1
    missed: int = 0
    votes: deque = field(default_factory=lambda: deque(maxlen=7))
    person: Optional[dict] = None
    unknown: bool = False
    contrary: int = 0
    zones: set = field(default_factory=set)
    flags: set = field(default_factory=set)       # events already emitted for this track
    last_score: float = 0.0
    attempts: int = 0

    @property
    def status(self) -> str:
        return "identified" if self.person else ("unknown" if self.unknown else "identifying")

    def label(self) -> str:
        if self.person:
            return f"{self.person['name']} ({self.person['designation']})"
        return f"Unknown person #{self.id}" if self.unknown else "Identifying..."


def iou(a: tuple, b: tuple) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def _centre_close(a: tuple, b: tuple) -> bool:
    ax, ay, bx, by = a[0] + a[2] / 2, a[1] + a[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < 0.75 * max(a[2], a[3], b[2], b[3])


class FaceTracker:
    def __init__(self):
        self.tracks: list = []
        self._next = 1

    def update(self, detections: list, ts: float):
        """detections: dicts {box, person (dict|None), score, small (bool)}. Returns (active, closed)."""
        pairs = sorted(((iou(t.box, d["box"]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)),
                       key=lambda p: p[0], reverse=True)
        used_t, used_d, assign = set(), set(), {}
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            if score >= 0.15 or _centre_close(self.tracks[ti].box, detections[di]["box"]):
                assign[di] = ti
                used_t.add(ti)
                used_d.add(di)
        for di, d in enumerate(detections):
            if di in assign:
                t = self.tracks[assign[di]]
                t.box, t.last_ts, t.missed, t.hits = d["box"], ts, 0, t.hits + 1
            else:
                t = Track(id=self._next, box=d["box"], first_ts=ts, last_ts=ts)
                self._next += 1
                self.tracks.append(t)
            if not d.get("small"):
                self._vote(t, d.get("person"), d.get("score", 0.0))
        closed = []
        for ti, t in list(enumerate(self.tracks)):
            if ti not in used_t and t.last_ts != ts:
                t.missed += 1
        for t in list(self.tracks):
            if t.missed > config.PERSON_TRACK_MAX_MISSED:
                closed.append(t)
                self.tracks.remove(t)
        return list(self.tracks), closed

    def _vote(self, t: Track, person: Optional[dict], score: float) -> None:
        t.attempts += 1
        t.last_score = score
        t.votes.append(person["id"] if person else None)
        counts = Counter(v for v in t.votes if v is not None)
        top, n = (counts.most_common(1)[0] if counts else (None, 0))
        confirmed_candidate = top is not None and n >= config.PERSON_MIN_VOTES
        if t.person is None:
            if confirmed_candidate and person is not None and person["id"] == top:
                t.person, t.unknown, t.contrary = person, False, 0
            elif t.attempts >= config.PERSON_UNKNOWN_AFTER and not confirmed_candidate:
                t.unknown = True
        else:
            if person is not None and person["id"] != t.person["id"]:
                t.contrary += 1
                if t.contrary >= 4 and top is not None and n >= config.PERSON_MIN_VOTES and person["id"] == top:
                    t.person, t.contrary = person, 0
            else:
                t.contrary = 0
