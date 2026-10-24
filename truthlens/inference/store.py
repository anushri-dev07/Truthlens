from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path("artifacts/truthlens.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    filename TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    created_at TEXT,
    verdict TEXT,
    ai_probability REAL,
    status TEXT,
    result_json TEXT,
    owner_id TEXT REFERENCES users(id)
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
    if "owner_id" not in columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN owner_id TEXT REFERENCES users(id)")
    return conn


def create_user(email: str, password_hash: str) -> str | None:
    user_id = uuid.uuid4().hex
    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, password_hash, datetime.now(UTC).isoformat()),
            )
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email: str) -> dict | None:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def get_user(user_id: str) -> dict | None:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def save_analysis(result: dict, owner_id: str | None = None) -> str:
    record_id = uuid.uuid4().hex[:12]
    file_meta = result.get("file", {}) or {}
    created = datetime.now(UTC).isoformat()
    result["id"] = record_id
    result["created_at"] = created
    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT INTO analyses (id, filename, size_bytes, sha256, created_at, "
                "verdict, ai_probability, status, result_json, owner_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    file_meta.get("name"),
                    file_meta.get("size_bytes"),
                    file_meta.get("sha256"),
                    created,
                    result.get("verdict"),
                    result.get("ai_probability"),
                    result.get("status"),
                    json.dumps(result),
                    owner_id,
                ),
            )
        conn.close()
    except Exception:
        result.pop("id", None)
        result.pop("created_at", None)
        return ""
    return record_id


def list_analyses(limit: int = 50, owner_id: str | None = None) -> list[dict]:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, size_bytes, sha256, created_at, verdict, "
            "ai_probability, status FROM analyses WHERE owner_id IS ? "
            "ORDER BY created_at DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_analysis(record_id: str, owner_id: str | None = None) -> dict | None:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE id = ? AND owner_id IS ?",
            (record_id, owner_id),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return json.loads(row["result_json"])
    except Exception:
        return None


def delete_analysis(record_id: str, owner_id: str) -> bool:
    try:
        conn = _connect()
        with conn:
            cursor = conn.execute(
                "DELETE FROM analyses WHERE id = ? AND owner_id = ?",
                (record_id, owner_id),
            )
        conn.close()
        return cursor.rowcount == 1
    except Exception:
        return False
