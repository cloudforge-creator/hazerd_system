"""Camera, personnel, zones, users, notifications and device-ingest routes (kept apart from the core pages)."""
from __future__ import annotations

import hmac
import json
import time
from types import SimpleNamespace

from flask import Response, abort, flash, jsonify, redirect, render_template, request, session, url_for

from src.personnel.zones import validate_zones
from src.users import ROLES

MAX_PHOTOS, MAX_PHOTO_BYTES = 5, 4 * 1024 * 1024
BOUNDARY = b"frame"


def _no_signal_jpeg() -> bytes:
    import cv2
    import numpy as np
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(img, "NO CAMERA SIGNAL", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class MjpegStream:
    """Iterator with an explicit close(): the viewer slot is released exactly once, even if the client disconnects
    before the first frame (a plain generator would leak the slot in that case)."""

    def __init__(self, hub, period: float = 0.12, wait: float = 3.0):
        self.hub, self.period, self.wait, self.released = hub, period, wait, False
        self._next_t = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        delay = self._next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        t0, jpg = time.time(), None
        while jpg is None and time.time() - t0 < self.wait:
            jpg = self.hub.jpeg()
            if jpg is None:
                time.sleep(0.1)
        jpg = jpg or _no_signal_jpeg()             # periodically hand control back even with a dead camera
        self._next_t = time.time() + self.period
        return b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"

    def close(self) -> None:
        if not self.released:
            self.released = True
            self.hub.release_stream()


def register(app, ctx: SimpleNamespace) -> None:
    login_required, admin_required, storage = ctx.login_required, ctx.admin_required, ctx.storage
    last_denied = {"t": 0.0}

    def log(kind: str, detail: str) -> None:
        storage.log_system(kind, detail)

    # ------------------------------------------------------------------ camera
    @app.route("/camera")
    @login_required
    def camera_page():
        zones = ctx.zones() if ctx.zones else []
        return render_template("camera.html", has_camera=ctx.hub is not None, has_people=ctx.monitor is not None,
                               zones_json=json.dumps([z.to_dict() for z in zones], indent=2),
                               is_admin=session.get("role") == "admin")

    @app.route("/camera/stream")
    @login_required
    def camera_stream():
        if ctx.hub is None:
            abort(404)
        if not ctx.hub.acquire_stream():
            return Response("Too many viewers on the live camera (limit reached).", status=503)
        return Response(MjpegStream(ctx.hub), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/camera/snapshot.jpg")
    @login_required
    def camera_snapshot():
        jpg = ctx.hub.jpeg() if ctx.hub is not None else None
        if jpg is None:
            return Response("no camera frame", status=503)
        return Response(jpg, mimetype="image/jpeg")

    @app.route("/api/people")
    @login_required
    def api_people():
        return jsonify(people=ctx.monitor.snapshot() if ctx.monitor else [],
                       monitor=ctx.monitor.status() if ctx.monitor else None,
                       camera=ctx.hub.status() if ctx.hub else None)

    @app.route("/zones/save", methods=["POST"])
    @admin_required
    def zones_save():
        if ctx.zones is None:
            abort(404)
        try:
            raw = json.loads(request.form.get("zones_json", ""))
        except ValueError:
            flash("Zones must be valid JSON.", "error")
            return redirect(url_for("camera_page"))
        zones, errors = validate_zones(raw)
        if errors:
            for e in errors:
                flash(e, "error")
        else:
            ctx.zones.set(zones)
            log("zones_changed", f"by={session['user']} zones={[z.name for z in zones]}")
            flash(f"Saved {len(zones)} zone(s).", "ok")
        return redirect(url_for("camera_page"))

    # ------------------------------------------------------------------ personnel
    @app.route("/personnel")
    @login_required
    def personnel_page():
        ok = ctx.face_engine is not None
        return render_template("personnel.html", people=ctx.people.list_people() if ctx.people else [],
                               is_admin=session.get("role") == "admin", engine_ok=ok, has_hub=ctx.hub is not None,
                               enabled=ctx.people is not None)

    def need_people():
        if ctx.people is None:
            abort(404)

    @app.route("/personnel/add", methods=["POST"])
    @admin_required
    def personnel_add():
        need_people()
        try:
            pid = ctx.people.add_person(request.form.get("name"), request.form.get("designation"),
                                        request.form.get("employee_id"), consent=request.form.get("consent") == "yes")
            log("personnel_added", f"id={pid} by={session['user']}")
            flash("Person added. Now enrol their face (upload photos or capture from the live camera).", "ok")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("personnel_page"))

    def enrol(person_id: int, frames: list) -> None:
        if ctx.face_engine is None:
            flash("Face models are not installed (run: python scripts/download_face_models.py).", "error")
            return
        done = 0
        for label, frame in frames:
            emb, msg = ctx.face_engine.embed_single_face(frame)
            if emb is None:
                flash(f"{label}: {msg}", "error")
                continue
            try:
                ctx.people.add_embedding(person_id, emb)
                done += 1
            except ValueError as exc:
                flash(f"{label}: {exc}", "error")
                break
        if done:
            log("face_enrolled", f"person={person_id} samples={done} by={session['user']}")
            flash(f"Enrolled {done} face sample(s). Use several: different angles and lighting.", "ok")

    @app.route("/personnel/<int:person_id>/enroll", methods=["POST"])
    @admin_required
    def personnel_enroll(person_id):
        need_people()
        if not ctx.people.get(person_id):
            abort(404)
        import cv2
        import numpy as np
        files = request.files.getlist("photos")[:MAX_PHOTOS]
        if not files or not any(f.filename for f in files):
            flash("Choose at least one photo.", "error")
            return redirect(url_for("personnel_page"))
        frames = []
        for f in files:
            data = f.read(MAX_PHOTO_BYTES + 1)
            if len(data) > MAX_PHOTO_BYTES:
                flash(f"{f.filename}: larger than 4 MB.", "error")
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                flash(f"{f.filename}: not a readable image.", "error")
                continue
            frames.append((f.filename or "photo", img))
        enrol(person_id, frames)
        return redirect(url_for("personnel_page"))

    @app.route("/personnel/<int:person_id>/enroll-live", methods=["POST"])
    @admin_required
    def personnel_enroll_live(person_id):
        need_people()
        if not ctx.people.get(person_id):
            abort(404)
        frame = ctx.hub.latest()[0] if ctx.hub is not None else None
        if frame is None:
            flash("No live camera frame available.", "error")
        else:
            enrol(person_id, [("live camera", frame)])
        return redirect(url_for("personnel_page"))

    @app.route("/personnel/<int:person_id>/<action>", methods=["POST"])
    @admin_required
    def personnel_action(person_id, action):
        need_people()
        person = ctx.people.get(person_id)
        if not person or action not in ("toggle", "delete", "clear"):
            abort(404)
        if action == "toggle":
            ctx.people.set_disabled(person_id, not person["disabled"])
            flash(("Re-enabled " if person["disabled"] else "Disabled ") + person["name"], "ok")
        elif action == "clear":
            ctx.people.clear_embeddings(person_id)
            flash(f"Removed all face samples of {person['name']}.", "ok")
        else:
            ctx.people.delete_person(person_id, purge_events=True)
            flash(f"Deleted {person['name']}, their face samples and their events.", "ok")
        log(f"personnel_{action}", f"id={person_id} by={session['user']}")
        return redirect(url_for("personnel_page"))

    # ------------------------------------------------------------------ users
    @app.route("/users")
    @admin_required
    def users_page():
        return render_template("users.html", users=ctx.users.list(), me=session["user"], roles=ROLES)

    @app.route("/users/add", methods=["POST"])
    @admin_required
    def users_add():
        try:
            ctx.users.create(request.form.get("username", ""), request.form.get("password", ""), request.form.get("role", "viewer"))
            log("user_created", f"{request.form.get('username')} role={request.form.get('role')} by={session['user']}")
            flash("User created.", "ok")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("users_page"))

    @app.route("/users/<username>/<action>", methods=["POST"])
    @admin_required
    def users_action(username, action):
        try:
            if action == "disable":
                if username.lower() == session["user"].lower():
                    raise ValueError("you cannot disable your own account")
                if ctx.users.role_of(username) == "admin" and ctx.users.active_admins() <= 1:
                    raise ValueError("cannot disable the last active admin")
                ctx.users.disable(username)
            elif action == "enable":
                ctx.users.enable(username)
            elif action == "role":
                new_role = request.form.get("role", "")
                if new_role != "admin" and ctx.users.role_of(username) == "admin" and ctx.users.active_admins() <= 1:
                    raise ValueError("cannot remove the last active admin's admin role")
                ctx.users.set_role(username, new_role)
            elif action == "password":
                ctx.users.set_password(username, request.form.get("password", ""))
            else:
                abort(404)
            log(f"user_{action}", f"{username} by={session['user']}")
            flash(f"{action} ok for {username}.", "ok")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("users_page"))

    # ------------------------------------------------------------------ notifications
    @app.route("/notifications/test", methods=["POST"])
    @admin_required
    def notifications_test():
        if ctx.notifier is None:
            flash("No notification channel is configured (see data/notifications.example.json).", "error")
        else:
            ctx.notifier.send("[TEST] Hazard monitor notification test", f"Sent by {session['user']} from the dashboard.")
            flash("Test notification queued. Check the system log for the delivery result.", "ok")
        return redirect(url_for("config_page"))

    # ------------------------------------------------------------------ device ingest (HTTP sensor input)
    @app.route("/api/ingest", methods=["POST"])
    def api_ingest():
        if ctx.ingest is None or not ctx.device_key:
            abort(404)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], ctx.device_key):
            if time.time() - last_denied["t"] > 60:
                last_denied["t"] = time.time()
                log("ingest_denied", f"bad device key from {request.remote_addr}")
            return jsonify(error="unauthorized"), 401
        if (request.content_length or 0) > 1024:
            return jsonify(error="payload too large"), 413
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not set(body) <= {"gas", "temp", "hum", "flame"}:
            return jsonify(error="expected a JSON object with gas, temp, hum, flame"), 400
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in body.values()):
            return jsonify(error="values must be numbers"), 400
        ctx.ingest.put(body)
        return jsonify(ok=True, commands=ctx.ingest.take_commands())
