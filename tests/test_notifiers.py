import hashlib
import hmac
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.notifiers import (AsyncNotifier, EmailChannel, NotificationRouter, TelegramChannel, WebhookChannel,
                           audiences_for, load_notifiers)


class Handler(BaseHTTPRequestHandler):
    received, status = [], 200

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        Handler.received.append((self.path, dict(self.headers), self.rfile.read(n)))
        self.send_response(Handler.status)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.received, Handler.status = [], 200
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_audience_routing():
    assert audiences_for("[EMERGENCY] person in hazard zone") == ("admin", "user")
    assert audiences_for("[CRITICAL] hazard system") == ("admin", "user")
    assert audiences_for("[WARNING] hazard system") == ("admin",)
    assert audiences_for("[SAFE] hazard system") == ("admin",)


def test_telegram_sends_to_the_right_chats(server):
    tg = TelegramChannel("TOKEN123", ["1", "2"], ["9"], api_base=server)
    assert tg.deliver("admin", "[WARNING] x", "body text") == 2
    assert tg.deliver("user", "[CRITICAL] x", "body") == 1
    paths = [p for p, _, _ in Handler.received]
    assert paths == ["/botTOKEN123/sendMessage"] * 3
    sent = [urllib.parse.parse_qs(b.decode()) for _, _, b in Handler.received]
    assert [s["chat_id"][0] for s in sent] == ["1", "2", "9"] and "[WARNING] x" in sent[0]["text"][0]


def test_telegram_http_error_is_raised(server):
    Handler.status = 500
    with pytest.raises(Exception):
        TelegramChannel("T", ["1"], api_base=server).deliver("admin", "s", "b")


def test_webhook_signs_payload(server):
    wh = WebhookChannel(server + "/hook", secret="s3cret")
    assert wh.deliver("admin", "[CRITICAL] x", "details") == 1 and wh.deliver("user", "s", "b") == 0
    path, headers, raw = Handler.received[0]
    assert path == "/hook" and json.loads(raw)["subject"] == "[CRITICAL] x"
    assert headers["X-Hazard-Signature"] == hmac.new(b"s3cret", raw, hashlib.sha256).hexdigest()


def test_email_uses_tls_login_and_audience(monkeypatch):
    log = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None): log.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): log.append("tls")
        def login(self, u, p): log.append(("login", u, p))
        def send_message(self, msg): log.append(("send", msg["To"], msg["Subject"]))

    em = EmailChannel("smtp.x", 587, "from@x", "u", "p", ["a@x", "b@x"], ["c@x"], smtp_factory=FakeSMTP)
    assert em.deliver("admin", "[WARNING] t", "b") == 2 and em.deliver("user", "[CRITICAL] t", "b") == 1
    assert log[:4] == [("connect", "smtp.x", 587), "tls", ("login", "u", "p"), ("send", "a@x, b@x", "[WARNING] t")]
    assert EmailChannel("h", 25, "f", admin_to=["a"], smtp_factory=FakeSMTP).deliver("user", "s", "b") == 0


def test_router_isolates_a_broken_channel():
    class Good:
        name = "good"; calls = []
        def deliver(self, a, s, b): Good.calls.append(a); return 1
    class Bad:
        name = "bad"
        def deliver(self, a, s, b): raise OSError("down")
    assert NotificationRouter([Bad(), Good()]).send("[CRITICAL] x", "b") == 2 and Good.calls == ["admin", "user"]
    with pytest.raises(RuntimeError, match="bad"):
        NotificationRouter([Bad()]).send("[WARNING] x", "b")


def test_async_notifier_never_blocks_and_retries_then_reports():
    results, attempts = [], []

    class Flaky:
        def send(self, s, b):
            attempts.append(s)
            if len(attempts) < 3:
                raise OSError("net")
            return 2
    an = AsyncNotifier(Flaky(), on_result=lambda k, d: results.append((k, d)), retries=2, backoff=0.01)
    t0 = time.time()
    an.send("[WARNING] a", "b")
    assert time.time() - t0 < 0.05 and an.flush(5)
    assert an.stats["delivered"] == 1 and results == [("notify_ok", "[WARNING] a -> 2 recipient(s)")]

    class Dead:
        def send(self, s, b): raise OSError("dead")
    dead = AsyncNotifier(Dead(), on_result=lambda k, d: results.append((k, d)), retries=1, backoff=0.01)
    dead.send("[CRITICAL] z", "b")
    assert dead.flush(5) and dead.stats["failed"] == 1 and "OSError" in dead.stats["last_error"]
    assert results[-1][0] == "notify_failed"
    an.stop(); dead.stop()


def test_async_queue_is_bounded():
    class Slow:
        def send(self, s, b): time.sleep(0.3)
    an = AsyncNotifier(Slow(), maxsize=2, retries=0)
    for i in range(10):
        an.send(f"s{i}", "b")
    assert an.stats["dropped"] >= 5
    an.stop()


def test_config_loading_uses_env_secrets_and_reports_problems(tmp_path):
    p = tmp_path / "n.json"
    p.write_text(json.dumps({"telegram": {"enabled": True, "bot_token_env": "TG", "admin_chat_ids": ["1"]},
                             "email": {"enabled": True, "smtp_host": "h"},
                             "webhook": {"enabled": True, "url": "ftp://bad"}}))
    router, notes = load_notifiers(str(p), env={})
    assert router is None and any("TG is not set" in n for n in notes) and any("email skipped" in n for n in notes)
    assert any("webhook skipped" in n for n in notes)
    router, notes = load_notifiers(str(p), env={"TG": "tok"})
    assert router and [c.name for c in router.channels] == ["telegram"] and "tok" not in json.dumps(notes)
    assert load_notifiers(str(tmp_path / "missing.json"))[0] is None
    (tmp_path / "bad.json").write_text("{nope")
    assert load_notifiers(str(tmp_path / "bad.json"))[0] is None
