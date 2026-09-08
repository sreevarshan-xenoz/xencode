#!/usr/bin/env python3
"""
SQLite User Store

Persistent user storage using SQLite for production-ready auth.
Replaces in-memory user store from AuthManager.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from xencode.models.user import Permission, User, UserRole, UserSession


class SQLiteUserStore:
    """Thread-safe SQLite-backed user store."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".xencode" / "users.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = self._local.conn
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        with self._tx() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login TEXT,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    is_verified INTEGER NOT NULL DEFAULT 0,
                    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    require_password_change INTEGER NOT NULL DEFAULT 0,
                    two_factor_enabled INTEGER NOT NULL DEFAULT 0,
                    two_factor_secret TEXT,
                    permissions TEXT DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS email_verification (
                    id TEXT PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_email_token ON email_verification(token);
            """)

    def _row_to_user(self, row: sqlite3.Row) -> User:
        data = dict(row)
        data["role"] = UserRole(data["role"])
        data["is_active"] = bool(data["is_active"])
        data["is_verified"] = bool(data["is_verified"])
        data["failed_login_attempts"] = int(data["failed_login_attempts"])
        data["require_password_change"] = bool(data["require_password_change"])
        data["two_factor_enabled"] = bool(data["two_factor_enabled"])
        data["permissions"] = [Permission.from_dict(p) for p in json.loads(data["permissions"] or "[]")]
        if data.get("locked_until"):
            data["locked_until"] = datetime.fromisoformat(data["locked_until"])
        return User.from_dict(data)

    # --- User CRUD ---

    def add_user(self, user: User) -> bool:
        try:
            with self._tx() as conn:
                conn.execute(
                    """INSERT INTO users
                       (id, username, email, password_hash, salt, full_name,
                        created_at, updated_at, last_login, role, is_active,
                        is_verified, failed_login_attempts, locked_until,
                        require_password_change, two_factor_enabled,
                        two_factor_secret, permissions)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user.id, user.username, user.email, user.password_hash,
                        user.salt, user.full_name,
                        user.created_at.isoformat(), user.updated_at.isoformat(),
                        user.last_login.isoformat() if user.last_login else None,
                        user.role.value, int(user.is_active), int(user.is_verified),
                        user.failed_login_attempts,
                        user.locked_until.isoformat() if user.locked_until else None,
                        int(user.require_password_change), int(user.two_factor_enabled),
                        user.two_factor_secret,
                        json.dumps([p.to_dict() for p in user.permissions]),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, user_id: str) -> Optional[User]:
        row = self._get_conn().execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[User]:
        row = self._get_conn().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        row = self._get_conn().execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def update_user(self, user: User) -> bool:
        with self._tx() as conn:
            n = conn.execute(
                """UPDATE users SET
                   username=?, email=?, password_hash=?, salt=?,
                   full_name=?, updated_at=?, last_login=?, role=?,
                   is_active=?, is_verified=?, failed_login_attempts=?,
                   locked_until=?, require_password_change=?, two_factor_enabled=?,
                   two_factor_secret=?, permissions=?
                   WHERE id=?""",
                (
                    user.username, user.email, user.password_hash, user.salt,
                    user.full_name, datetime.now().isoformat(),
                    user.last_login.isoformat() if user.last_login else None,
                    user.role.value, int(user.is_active), int(user.is_verified),
                    user.failed_login_attempts,
                    user.locked_until.isoformat() if user.locked_until else None,
                    int(user.require_password_change), int(user.two_factor_enabled),
                    user.two_factor_secret,
                    json.dumps([p.to_dict() for p in user.permissions]),
                    user.id,
                ),
            ).rowcount
        return n > 0

    def delete_user(self, user_id: str) -> bool:
        with self._tx() as conn:
            n = conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount
        return n > 0

    def get_all_users(self, include_inactive: bool = False) -> List[User]:
        query = "SELECT * FROM users" if include_inactive else "SELECT * FROM users WHERE is_active = 1"
        rows = self._get_conn().execute(query).fetchall()
        return [self._row_to_user(r) for r in rows]

    # --- Sessions ---

    def add_session(self, session: UserSession) -> bool:
        try:
            with self._tx() as conn:
                conn.execute(
                    """INSERT INTO sessions
                       (id, user_id, token, created_at, expires_at,
                        last_activity, ip_address, user_agent, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session.id, session.user_id, session.token,
                        session.created_at.isoformat(),
                        session.expires_at.isoformat(),
                        session.last_activity.isoformat(),
                        session.ip_address, session.user_agent, int(session.is_active),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_session(self, session_id: str) -> Optional[UserSession]:
        row = self._get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        rows = self._get_conn().execute(
            "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1", (user_id,)
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def update_session(self, session: UserSession) -> bool:
        with self._tx() as conn:
            n = conn.execute(
                """UPDATE sessions SET
                   token=?, last_activity=?, is_active=? WHERE id=?""",
                (
                    session.token,
                    session.last_activity.isoformat(),
                    int(session.is_active),
                    session.id,
                ),
            ).rowcount
        return n > 0

    def revoke_session(self, session_id: str) -> bool:
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE id = ?", (session_id,)
            ).rowcount
        return n > 0

    def revoke_all_user_sessions(self, user_id: str) -> int:
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE user_id = ?", (user_id,)
            ).rowcount
        return n

    def cleanup_expired_sessions(self) -> int:
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            ).rowcount
        return n

    def _row_to_session(self, row: sqlite3.Row) -> UserSession:
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        return UserSession(
            id=data["id"], user_id=data["user_id"], token=data["token"],
            created_at=data["created_at"], expires_at=data["expires_at"],
            last_activity=data["last_activity"], ip_address=data["ip_address"],
            user_agent=data["user_agent"], is_active=data["is_active"],
        )

    # --- Email verification ---

    def create_email_verification(self, user_id: str, token: str, expires_hours: int = 24) -> bool:
        import secrets
        try:
            with self._tx() as conn:
                conn.execute(
                    """INSERT INTO email_verification
                       (id, user_id, token, expires_at, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        secrets.token_hex(16),
                        user_id,
                        token,
                        (datetime.now().replace(microsecond=0) +
                         __import__("datetime").timedelta(hours=expires_hours)).isoformat(),
                        datetime.now().isoformat(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_email_token(self, token: str) -> Optional[str]:
        """Verify token and return user_id if valid. Marks token as used."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM email_verification WHERE token = ? AND used = 0", (token,)
        ).fetchone()
        if not row:
            return None
        expires = datetime.fromisoformat(row["expires_at"])
        if expires < datetime.now():
            return None
        conn.execute(
            "UPDATE email_verification SET used = 1 WHERE id = ?", (row["id"],)
        )
        conn.commit()
        return row["user_id"]
