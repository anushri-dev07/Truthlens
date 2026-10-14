from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path("artifacts/truthlens.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    filename TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    created_at TEXT,
    verdict TEXT,
    ai_probability REAL,
    status TEXT,
    result_json TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(_SCHEMA)
    return conn


def save_analysis(result: dict) -> str:
    record_id = uuid.uuid4().hex[:12]
    file_meta = result.get("file", {}) or {}
    created = datetime.now(UTC).isoformat()
    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT INTO analyses (id, filename, size_bytes, sha256, created_at, "
                "verdict, ai_probability, status, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
        conn.close()
    except Exception:
        return ""
    result["id"] = record_id
    result["created_at"] = created
    return record_id


def list_analyses(limit: int = 50) -> list[dict]:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, filename, size_bytes, sha256, created_at, verdict, "
            "ai_probability, status FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_analysis(record_id: str) -> dict | None:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE id = ?", (record_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return json.loads(row["result_json"])
    except Exception:
        return None
