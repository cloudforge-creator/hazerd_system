"""Notification channels (WBS 7.2): Telegram, e-mail, webhook, with audience routing.

* Secrets (bot token, SMTP password, webhook secret) come from ENVIRONMENT VARIABLES named in the config
  file, never from the file itself, so the config can be shared safely.
* Sending happens in a background worker with a bounded queue: a dead network can slow notifications
  but can never stall the monitoring/shutdown loop. Delivery results are recorded in `stats`.
* Audience routing by subject:  [EMERGENCY] -> admin + user lists, [CRITICAL] -> admin + user lists,
  anything else (warnings, early warnings, test) -> admin list only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import smtplib
import threading
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Callable, Optional


def audiences_for(subject: str) -> tuple:
    s = subject.upper()
    return ("admin", "user") if s.startswith("[EMERGENCY]") or s.startswith("[CRITICAL]") else ("admin",)


class TelegramChannel:
    name = "telegram"

    def __init__(self, token: str, admin_chat_ids=(), user_chat_ids=(), api_base: str = "https://api.telegram.org",
                 opener: Callable = urllib.request.urlopen, timeout: float = 10.0):
        self.token, self.api_base, self.opener, self.timeout = token, api_base.rstrip("/"), opener, timeout
        self.recipients = {"admin": [str(c) for c in admin_chat_ids], "user": [str(c) for c in user_chat_ids]}

    def deliver(self, audience: str, subject: str, body: str) -> int:
        n = 0
        for chat in self.recipients.get(audience, []):
            data = urllib.parse.urlencode({"chat_id": chat, "text": f"{subject}\n{body}"[:4000]}).encode()
            req = urllib.request.Request(f"{self.api_base}/bot{self.token}/sendMessage", data=data)
            with self.opener(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"telegram HTTP {resp.status}")
            n += 1
        return n


class EmailChannel:
    name = "email"

    def __init__(self, host, port, sender, username=None, password=None, admin_to=(), user_to=(), use_tls=True,
                 smtp_factory: Callable = smtplib.SMTP, timeout: float = 15.0):
        self.host, self.port, self.sender, self.username, self.password = host, int(port), sender, username, password
        self.use_tls, self.smtp_factory, self.timeout = use_tls, smtp_factory, timeout
        self.recipients = {"admin": list(admin_to), "user": list(user_to)}

    def deliver(self, audience: str, subject: str, body: str) -> int:
        to = self.recipients.get(audience, [])
        if not to:
            return 0
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject[:200], self.sender, ", ".join(to)
        msg.set_content(body)
        with self.smtp_factory(self.host, self.port, timeout=self.timeout) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password or "")
            smtp.send_message(msg)
        return len(to)


class WebhookChannel:
    name = "webhook"

    def __init__(self, url: str, secret: Optional[str] = None, opener: Callable = urllib.request.urlopen, timeout: float = 10.0):
        self.url, self.secret, self.opener, self.timeout = url, secret, opener, timeout
        self.recipients = {"admin": [url], "user": [url]}

    def deliver(self, audience: str, subject: str, body: str) -> int:
        if audience != "admin":                       # one POST per event, not one per audience
            return 0
        payload = json.dumps({"subject": subject, "body": body, "ts": time.time()}).encode()
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Hazard-Signature"] = hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()
        with self.opener(urllib.request.Request(self.url, data=payload, headers=headers), timeout=self.timeout) as resp:
            if not 200 <= resp.status < 300:
                raise RuntimeError(f"webhook HTTP {resp.status}")
        return 1


class NotificationRouter:
    """Looks like a notifier (send(subject, body)); fans out to channels by audience."""

    def __init__(self, channels: list):
        self.channels = channels

    def send(self, subject: str, body: str) -> int:
        sent = errors = 0
        last = None
        for aud in audiences_for(subject):
            for ch in self.channels:
                try:
                    sent += ch.deliver(aud, subject, body)
                except Exception as exc:                      # one broken channel must not block the others
                    errors += 1
                    last = f"{ch.name}: {exc.__class__.__name__}: {exc}"
        if errors and not sent:
            raise RuntimeError(last)
        return sent


class AsyncNotifier:
    """Bounded queue + worker thread around any notifier. send() never blocks and never raises."""

    def __init__(self, inner, on_result: Optional[Callable] = None, retries: int = 2, backoff: float = 1.0, maxsize: int = 100):
        self.inner, self.on_result, self.retries, self.backoff = inner, on_result, retries, backoff
        self.q = queue.Queue(maxsize=maxsize)
        self.stats = {"queued": 0, "delivered": 0, "failed": 0, "dropped": 0, "last_error": None}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def send(self, subject: str, body: str) -> None:
        try:
            self.q.put_nowait((subject, body))
            self.stats["queued"] += 1
        except queue.Full:
            self.stats["dropped"] += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                subject, body = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            for attempt in range(self.retries + 1):
                try:
                    n = self.inner.send(subject, body)
                    self.stats["delivered"] += 1
                    if self.on_result:
                        self.on_result("notify_ok", f"{subject} -> {n} recipient(s)")
                    break
                except Exception as exc:
                    self.stats["last_error"] = f"{exc.__class__.__name__}: {exc}"
                    if attempt == self.retries:
                        self.stats["failed"] += 1
                        if self.on_result:
                            self.on_result("notify_failed", f"{subject}: {self.stats['last_error']}")
                    else:
                        self._stop.wait(self.backoff * (attempt + 1))
            self.q.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while self.q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.q.unfinished_tasks

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)


def load_notifiers(path: str, env=os.environ) -> tuple:
    """Build channels from the JSON config. Returns (router or None, status lines). Secrets come from env vars."""
    if not os.path.exists(path):
        return None, [f"no notification config ({path}); copy data/notifications.example.json to enable Telegram/email/webhook"]
    try:
        cfg = json.load(open(path))
    except (OSError, ValueError) as exc:
        return None, [f"could not read {path}: {exc}"]
    channels, notes = [], []
    t = cfg.get("telegram", {})
    if t.get("enabled"):
        token = env.get(t.get("bot_token_env", ""), "")
        if token:
            channels.append(TelegramChannel(token, t.get("admin_chat_ids", []), t.get("user_chat_ids", [])))
            notes.append("telegram enabled")
        else:
            notes.append(f"telegram skipped: env var {t.get('bot_token_env')} is not set")
    e = cfg.get("email", {})
    if e.get("enabled"):
        if e.get("smtp_host") and e.get("from"):
            channels.append(EmailChannel(e["smtp_host"], e.get("smtp_port", 587), e["from"], env.get(e.get("username_env", ""), None),
                                         env.get(e.get("password_env", ""), None), e.get("admin_to", []), e.get("user_to", []),
                                         e.get("use_tls", True)))
            notes.append("email enabled")
        else:
            notes.append("email skipped: smtp_host and from are required")
    w = cfg.get("webhook", {})
    if w.get("enabled"):
        if str(w.get("url", "")).startswith(("https://", "http://")):
            channels.append(WebhookChannel(w["url"], env.get(w.get("secret_env", ""), None)))
            notes.append("webhook enabled")
        else:
            notes.append("webhook skipped: url must start with http(s)://")
    return (NotificationRouter(channels) if channels else None), notes or ["notification config has no enabled channels"]
