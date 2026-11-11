"""Flask admin dashboard (WBS 8.x): monitoring, alerts, logs, configuration, access control.

Security: hashed passwords, session cookies (HttpOnly, SameSite=Lax), CSRF token on every POST,
login lockout, role checks (viewer / admin), strict Content-Security-Policy (no inline scripts),
no default account. It speaks plain HTTP: keep it on localhost, or put it behind a TLS reverse
proxy / VPN before exposing it on a network.
"""
from __future__ import annotations

import hmac
import os
import secrets
import time
from datetime import timedelta
from functools import wraps

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, session, url_for)

from src import runtime_config
from src.decision_engine import Level

LOCK_AFTER, LOCK_SECONDS = 5, 60


def load_secret_key(path: str = "data/secret_key") -> str:
    if os.environ.get("HAZARD_SECRET"):
        return os.environ["HAZARD_SECRET"]
    if os.path.exists(path):
        return open(path).read().strip()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def create_app(system, live, users, secret_key: str, runner=None,
               runtime_config_path: str = "data/runtime_config.json", clock=time.time,
               hub=None, monitor=None, people=None, face_engine=None, zones=None, notifier=None,
               notify_notes=None, ingest=None, device_key=None) -> Flask:
    app = Flask(__name__)
    app.secret_key = secret_key
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      PERMANENT_SESSION_LIFETIME=timedelta(hours=8), MAX_CONTENT_LENGTH=20 * 1024 * 1024)
    attempts = {}                                  # ip -> [fail_count, locked_until]
    storage = system.storage

    # ------------------------------------------------------------ helpers
    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    def fmt_time(ts):
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
        except (TypeError, ValueError, OverflowError, OSError):
            return str(ts)

    def fmt_cell(value, column=""):
        if column == "ts":
            return fmt_time(value)
        if isinstance(value, float):
            return f"{value:.3f}"
        return "" if value is None else value

    app.jinja_env.filters["dt"] = fmt_time
    app.jinja_env.filters["cell"] = fmt_cell

    def is_api() -> bool:
        return request.path.startswith("/api/")

    def login_required(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            if "user" not in session:
                if is_api():
                    return jsonify(error="authentication required"), 401
                return redirect(url_for("login", next=request.path))
            return fn(*a, **k)
        return wrapper

    def admin_required(fn):
        @wraps(fn)
        @login_required
        def wrapper(*a, **k):
            if session.get("role") != "admin":
                if is_api():
                    return jsonify(error="admin role required"), 403
                abort(403)
            return fn(*a, **k)
        return wrapper

    @app.before_request
    def csrf_protect():
        if request.method == "POST" and request.path != "/api/ingest":      # device endpoint uses a bearer key instead
            sent = request.form.get("csrf_token", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "CSRF token missing or invalid")

    @app.after_request
    def headers(resp):
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                                                   "img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if not request.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403, message="Your role does not allow this action."), 403

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, message=getattr(e, "description", "Bad request")), 400

    def notify_status() -> dict:
        return {"notes": list(notify_notes or []), "stats": dict(notifier.stats) if hasattr(notifier, "stats") else None,
                "enabled": notifier is not None}

    def snapshot() -> dict:
        with live.lock:
            r, d, ex = live.reading, live.decision, list(live.executed)
            steps, errors, last_error, crashed = live.steps, live.errors, live.last_error, live.crashed
        resp = system.responder
        out = {"level": d.level.value if d else None, "score": d.score if d else None,
               "predicted_score": d.predicted_score if d else None,
               "early_warning": bool(d.early_warning) if d else False,
               "degraded": bool(d.degraded) if d else False,
               "reasons": list(d.reasons) if d else [], "actions": list(d.actions) if d else [],
               "executed": ex, "mode": live.mode, "ai": live.ai,
               "relay": "SHUT DOWN (latched)" if resp.relay_latched else "POWERED",
               "buzzer": "ON" if resp.buzzer_on else "OFF",
               "uptime_s": round(time.time() - live.started, 1), "steps": steps, "errors": errors,
               "last_error": last_error, "crashed": crashed,
               "runner_alive": bool(runner.is_alive()) if runner else None,
               "reading": None, "reading_age_s": None}
        if r:
            out["reading"] = {"gas_raw": r.gas_raw, "temperature": r.temperature, "humidity": r.humidity,
                              "flame": r.flame, "faults": list(r.faults), "source": r.source}
            out["reading_age_s"] = round(max(0.0, time.time() - r.timestamp), 1) if r.source != "simulation" else 0.0
        return out

    # ------------------------------------------------------------ auth
    @app.route("/login", methods=["GET", "POST"])
    def login():
        no_users = not users.list()
        if request.method == "POST":
            ip = request.remote_addr or "?"
            fails, locked_until = attempts.get(ip, [0, 0.0])
            if locked_until > clock():
                return render_template("login.html", no_users=no_users,
                                       error="Too many failed attempts. Try again in a minute."), 429
            user = users.verify(request.form.get("username", ""), request.form.get("password", ""))
            if user is None:
                fails += 1
                attempts[ip] = [fails, clock() + LOCK_SECONDS if fails >= LOCK_AFTER else 0.0]
                if fails >= LOCK_AFTER:
                    attempts[ip][0] = 0
                storage.log_system("login_failed", f"user={request.form.get('username', '')[:32]} ip={ip}")
                return render_template("login.html", no_users=no_users, error="Invalid username or password."), 401
            attempts.pop(ip, None)
            session.clear()
            session.permanent = True
            session.update(user=user["username"], role=user["role"], csrf=secrets.token_hex(16))
            storage.log_system("login", f"user={user['username']} role={user['role']}")
            nxt = request.args.get("next", "/")
            if not nxt.startswith("/") or nxt.startswith("//") or "\\" in nxt:
                nxt = "/"
            return redirect(nxt)
        return render_template("login.html", no_users=no_users, error=None)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ------------------------------------------------------------ pages
    @app.route("/")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/alerts")
    @login_required
    def alerts():
        return render_template("alerts.html", alerts=storage.recent_alerts(200),
                               active=len(storage.recent_alerts(500, active_only=True)))

    @app.route("/alerts/<int:alert_id>/ack", methods=["POST"])
    @admin_required
    def ack(alert_id):
        if storage.acknowledge_alert(alert_id):
            storage.log_system("alert_ack", f"id={alert_id} by={session['user']}")
            flash(f"Alert {alert_id} acknowledged.", "ok")
        else:
            flash("Alert not found or already acknowledged.", "error")
        return redirect(url_for("alerts"))

    @app.route("/logs")
    @login_required
    def logs():
        kind = request.args.get("kind", "events")
        n = min(max(request.args.get("n", 100, type=int), 1), 500)
        fetch = {"readings": storage.recent_readings, "events": storage.recent_events,
                 "system": storage.recent_system, "people": storage.recent_people_events}
        if kind not in fetch:
            abort(400, "unknown log kind")
        rows = fetch[kind](n)
        return render_template("logs.html", kind=kind, n=n, rows=rows,
                               columns=list(rows[0].keys()) if rows else [])

    @app.route("/config", methods=["GET", "POST"])
    @login_required
    def config_page():
        if request.method == "POST":
            if session.get("role") != "admin":
                abort(403)
            values = {k: request.form[k] for k in runtime_config.TUNABLES if k in request.form}
            clean, errors = runtime_config.validate(values)
            if errors:
                for e in errors:
                    flash(e, "error")
            else:
                changed = runtime_config.apply(clean, system.responder)
                if changed:
                    runtime_config.save(runtime_config_path)
                    detail = "; ".join(f"{k}: {o} -> {n}" for k, (o, n) in changed.items())
                    storage.log_system("config_change", f"by={session['user']} {detail}")
                    flash("Saved: " + detail, "ok")
                else:
                    flash("No changes.", "ok")
            return redirect(url_for("config_page"))
        return render_template("config.html", notify=notify_status(), tunables=runtime_config.TUNABLES,
                               values=runtime_config.current(), is_admin=session.get("role") == "admin")

    @app.route("/control/reset", methods=["POST"])
    @admin_required
    def reset_latch():
        if system.responder.reset_latch(system.engine.level):
            storage.log_system("relay_reset", f"by={session['user']}")
            flash("Machine re-armed (relay powered).", "ok")
        else:
            flash(f"Refused: system is {system.engine.level.value}. Reset is only allowed when SAFE.", "error")
        return redirect(url_for("dashboard"))

    # ------------------------------------------------------------ JSON API (read-only)
    @app.route("/api/status")
    @login_required
    def api_status():
        return jsonify(snapshot())

    @app.route("/api/history")
    @login_required
    def api_history():
        n = min(max(request.args.get("n", 120, type=int), 1), 1000)
        return jsonify(list(reversed(storage.recent_readings(n))))

    @app.route("/api/alerts")
    @login_required
    def api_alerts():
        return jsonify(storage.recent_alerts(100, active_only=request.args.get("active") == "1"))

    @app.route("/healthz")
    def healthz():
        ok = not live.crashed and (runner is None or runner.is_alive())
        return jsonify(status="ok" if ok else "degraded"), (200 if ok else 503)

    from types import SimpleNamespace
    from src.dashboard.extra import register
    register(app, SimpleNamespace(login_required=login_required, admin_required=admin_required, storage=storage,
                                  users=users, hub=hub, monitor=monitor, people=people, face_engine=face_engine,
                                  zones=zones, notifier=notifier, ingest=ingest, device_key=device_key))
    return app


def serve(app, host: str, port: int) -> None:
    if host not in ("127.0.0.1", "localhost"):
        print("WARNING: dashboard is plain HTTP. Put it behind a TLS reverse proxy or VPN before exposing it.")
    try:
        from waitress import serve as wserve
        wserve(app, host=host, port=port, threads=16)
    except ImportError:
        app.run(host=host, port=port, threaded=True, debug=False)
