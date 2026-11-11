"""Dashboard users (WBS 8.6). Passwords are hashed; there is NO default account.

    python -m src.users add admin --role admin
    python -m src.users add operator1 --role viewer
    python -m src.users list | passwd NAME | disable NAME
"""
from __future__ import annotations

import argparse
import getpass
import re
import time
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

from src import config
from src.storage import Storage

ROLES = ("admin", "viewer")
_NAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _hash(password: str) -> str:
    try:
        return generate_password_hash(password)                 # scrypt where available
    except (AttributeError, ValueError):
        return generate_password_hash(password, method="pbkdf2:sha256:600000")


_DUMMY = _hash("dummy-password-for-timing")


class UserStore:
    def __init__(self, storage: Storage):
        self.s = storage

    def create(self, username: str, password: str, role: str = "viewer") -> None:
        if not _NAME.match(username or ""):
            raise ValueError("username must be 3-32 chars: letters, digits, _ . -")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if len(password or "") < 8 or password.lower() == username.lower():
            raise ValueError("password must be at least 8 characters and not equal the username")
        if self.s.query("SELECT 1 FROM users WHERE username=?", (username,)):
            raise ValueError("username already exists")
        self.s.execute("INSERT INTO users(username,password_hash,role,created_ts) VALUES(?,?,?,?)",
                       (username, _hash(password), role, time.time()))

    def verify(self, username: str, password: str) -> Optional[dict]:
        rows = self.s.query("SELECT * FROM users WHERE username=?", (username or "",))
        if not rows or rows[0]["disabled"]:
            check_password_hash(_DUMMY, password or "")          # equalize timing for unknown users
            return None
        u = rows[0]
        return {"username": u["username"], "role": u["role"]} if check_password_hash(u["password_hash"], password or "") else None

    def list(self) -> list:
        return self.s.query("SELECT username, role, disabled FROM users ORDER BY username")

    def set_password(self, username: str, password: str) -> None:
        if len(password or "") < 8:
            raise ValueError("password must be at least 8 characters")
        if not self.s.execute("UPDATE users SET password_hash=? WHERE username=?", (_hash(password), username)).rowcount:
            raise ValueError("no such user")

    def enable(self, username: str) -> None:
        if not self.s.execute("UPDATE users SET disabled=0 WHERE username=?", (username,)).rowcount:
            raise ValueError("no such user")

    def set_role(self, username: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not self.s.execute("UPDATE users SET role=? WHERE username=?", (role, username)).rowcount:
            raise ValueError("no such user")

    def active_admins(self) -> int:
        return self.s.query("SELECT COUNT(*) AS c FROM users WHERE role='admin' AND disabled=0")[0]["c"]

    def role_of(self, username: str):
        rows = self.s.query("SELECT role FROM users WHERE username=?", (username,))
        return rows[0]["role"] if rows else None

    def disable(self, username: str) -> None:
        if not self.s.execute("UPDATE users SET disabled=1 WHERE username=?", (username,)).rowcount:
            raise ValueError("no such user")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=config.DB_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("username"); a.add_argument("--role", choices=ROLES, default="viewer")
    sub.add_parser("list")
    for name in ("passwd", "disable"):
        sub.add_parser(name).add_argument("username")
    args = ap.parse_args(argv)
    store = UserStore(Storage(args.db))
    try:
        if args.cmd == "add":
            pw = getpass.getpass("Password: ")
            if pw != getpass.getpass("Repeat password: "):
                raise ValueError("passwords do not match")
            store.create(args.username, pw, args.role)
            print(f"created {args.role} '{args.username}'")
        elif args.cmd == "list":
            for u in store.list():
                print(f"{u['username']:<20} {u['role']:<7} {'DISABLED' if u['disabled'] else ''}")
        elif args.cmd == "passwd":
            pw = getpass.getpass("New password: ")
            if pw != getpass.getpass("Repeat password: "):
                raise ValueError("passwords do not match")
            store.set_password(args.username, pw)
            print("password updated")
        else:
            store.disable(args.username)
            print("disabled")
    except ValueError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
