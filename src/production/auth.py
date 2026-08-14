"""Local Argon2id users and revocable signed sessions."""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from src.production.database import ProductionDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    role: str


class AuthService:
    def __init__(
        self,
        database: ProductionDatabase,
        secret: str | None = None,
        access_minutes: int = 15,
        refresh_days: int = 7,
    ) -> None:
        try:
            from argon2 import PasswordHasher
        except ImportError as exc:
            raise ImportError("argon2-cffi is required for production authentication") from exc
        self.database = database
        self.password_hasher = PasswordHasher()
        self.secret = secret or os.environ.get("RAG_AUTH_SECRET") or secrets.token_urlsafe(48)
        self.access_minutes = access_minutes
        self.refresh_days = refresh_days

    def has_users(self) -> bool:
        with self.database.connect() as connection:
            return bool(connection.execute("SELECT 1 FROM users LIMIT 1").fetchone())

    def bootstrap_admin(self, username: str, password: str) -> Principal:
        if self.has_users():
            raise PermissionError("Bootstrap is disabled after the first user is created")
        return self.create_user(username, password, "admin", actor=None)

    def create_user(self, username: str, password: str, role: str, actor: Principal | None) -> Principal:
        if actor is not None and actor.role != "admin":
            raise PermissionError("Administrator role required")
        username = self._validate_username(username)
        self._validate_password(password)
        if role not in {"admin", "user"}:
            raise ValueError("role must be admin or user")
        user_id = str(uuid.uuid4())
        timestamp = _now().isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO users(id,username,password_hash,role,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (user_id, username, self.password_hasher.hash(password), role, timestamp, timestamp),
            )
        return Principal(user_id, username, role)

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE AND active=1", (username,)
            ).fetchone()
        if row is None:
            raise PermissionError("Invalid credentials")
        try:
            self.password_hasher.verify(row["password_hash"], password)
        except Exception as exc:
            raise PermissionError("Invalid credentials") from exc
        principal = Principal(row["id"], row["username"], row["role"])
        return self._new_session(principal)

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        digest = hashlib.sha256(refresh_token.encode()).hexdigest()
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                """SELECT s.*,u.username,u.role,u.active FROM sessions s JOIN users u ON u.id=s.user_id
                   WHERE s.refresh_hash=? AND s.revoked_at IS NULL""", (digest,)
            ).fetchone()
            if row is None or not row["active"] or datetime.fromisoformat(row["expires_at"]) <= _now():
                raise PermissionError("Invalid or expired refresh token")
            connection.execute(
                "UPDATE sessions SET revoked_at=?,last_used_at=? WHERE id=?",
                (_now().isoformat(), _now().isoformat(), row["id"]),
            )
        return self._new_session(Principal(row["user_id"], row["username"], row["role"]))

    def logout(self, refresh_token: str) -> None:
        digest = hashlib.sha256(refresh_token.encode()).hexdigest()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE sessions SET revoked_at=? WHERE refresh_hash=? AND revoked_at IS NULL",
                (_now().isoformat(), digest),
            )

    def verify_access(self, token: str) -> Principal:
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"], audience="modular-rag")
        except Exception as exc:
            raise PermissionError("Invalid or expired access token") from exc
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT id,username,role,active FROM users WHERE id=?", (payload["sub"],)
            ).fetchone()
        if row is None or not row["active"]:
            raise PermissionError("User is inactive")
        return Principal(row["id"], row["username"], row["role"])

    def list_users(self, actor: Principal) -> list[dict[str, Any]]:
        if actor.role != "admin":
            raise PermissionError("Administrator role required")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id,username,role,active,created_at,updated_at FROM users ORDER BY username"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_user(self, user_id: str, changes: dict[str, Any], actor: Principal) -> dict[str, Any]:
        if actor.role != "admin":
            raise PermissionError("Administrator role required")
        allowed = {"role", "active", "password"}
        if not changes or not set(changes) <= allowed:
            raise ValueError("Unsupported user update")
        if changes.get("role") not in {None, "admin", "user"}:
            raise ValueError("role must be admin or user")
        if "password" in changes:
            self._validate_password(str(changes["password"]))
        with self.database.transaction(immediate=True) as connection:
            target = connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if target is None:
                raise KeyError(user_id)
            if user_id == actor.user_id and changes.get("active") is False:
                raise ValueError("Administrator cannot deactivate the current account")
            role = changes.get("role", target["role"])
            active = int(changes.get("active", bool(target["active"])))
            password_hash = (
                self.password_hasher.hash(str(changes["password"]))
                if "password" in changes else target["password_hash"]
            )
            if target["role"] == "admin" and (role != "admin" or not active):
                count = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1"
                ).fetchone()[0]
                if count <= 1:
                    raise ValueError("At least one active administrator is required")
            connection.execute(
                "UPDATE users SET role=?,active=?,password_hash=?,updated_at=? WHERE id=?",
                (role, active, password_hash, _now().isoformat(), user_id),
            )
            if not active or "password" in changes:
                connection.execute(
                    "UPDATE sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                    (_now().isoformat(), user_id),
                )
            updated = connection.execute(
                "SELECT id,username,role,active,created_at,updated_at FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(updated)

    def _new_session(self, principal: Principal) -> dict[str, Any]:
        now = _now()
        refresh = secrets.token_urlsafe(48)
        session_id = str(uuid.uuid4())
        expires = now + timedelta(days=self.refresh_days)
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO sessions(id,user_id,refresh_hash,expires_at,created_at) VALUES (?,?,?,?,?)",
                (session_id, principal.user_id, hashlib.sha256(refresh.encode()).hexdigest(), expires.isoformat(), now.isoformat()),
            )
        access = jwt.encode(
            {"sub": principal.user_id, "role": principal.role, "sid": session_id,
             "aud": "modular-rag", "iat": now, "exp": now + timedelta(minutes=self.access_minutes)},
            self.secret, algorithm="HS256",
        )
        return {"access_token": access, "refresh_token": refresh, "token_type": "bearer",
                "expires_in": self.access_minutes * 60,
                "user": {"id": principal.user_id, "username": principal.username, "role": principal.role}}

    @staticmethod
    def _validate_username(username: str) -> str:
        value = username.strip()
        if not 3 <= len(value) <= 64 or not all(c.isalnum() or c in "._-" for c in value):
            raise ValueError("Username must be 3-64 letters, digits, dot, underscore or hyphen")
        return value

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < 12 or len(password) > 256:
            raise ValueError("Password must be 12-256 characters")
