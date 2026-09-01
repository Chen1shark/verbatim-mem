"""SQLite + FTS5。"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from app.schemas import AddRequest, Message

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(request_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_request_id ON messages(request_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    user_id UNINDEXED,
    id UNINDEXED,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, user_id, id)
    VALUES (new.rowid, new.content, new.user_id, new.id);
END;
"""


class ConflictError(Exception):
    """相同 request_id、不同 fingerprint。"""


def fingerprint(user_id: str, session_id: str, messages: list[Message]) -> str:
    """user_id + session_id + messages 的 SHA-256。不含 request_id。"""
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "messages": [
            {
                "content": item.content,
                "role": item.role,
                "timestamp": item.timestamp,
            }
            for item in messages
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MemoryStore:
    """单连接；Uvicorn 仅 1 worker。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        self._require_conn().executescript(SCHEMA_SQL)

    def ping(self) -> None:
        self._require_conn().execute("SELECT 1").fetchone()

    def add(self, body: AddRequest) -> str:
        """写入 messages 与 FTS5。返回 created 或 duplicate。冲突抛 ConflictError。"""
        content_hash = fingerprint(body.user_id, body.session_id, body.messages)
        now = int(time.time() * 1000)
        conn = self._require_conn()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT content_hash FROM requests WHERE request_id = ?",
                    (body.request_id,),
                ).fetchone()
                if existing is not None:
                    if existing["content_hash"] == content_hash:
                        conn.execute("COMMIT")
                        return "duplicate"
                    conn.execute("ROLLBACK")
                    raise ConflictError()
                conn.execute(
                    """
                    INSERT INTO requests (
                        request_id, user_id, session_id, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        body.request_id,
                        body.user_id,
                        body.session_id,
                        content_hash,
                        now,
                    ),
                )
                for index, message in enumerate(body.messages):
                    conn.execute(
                        """
                        INSERT INTO messages (
                            id, request_id, user_id, session_id,
                            role, timestamp, content, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"{body.request_id}:{index}",
                            body.request_id,
                            body.user_id,
                            body.session_id,
                            message.role,
                            message.timestamp,
                            message.content,
                            now,
                        ),
                    )
                conn.execute("COMMIT")
                return "created"
            except ConflictError:
                raise
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                existing = conn.execute(
                    "SELECT content_hash FROM requests WHERE request_id = ?",
                    (body.request_id,),
                ).fetchone()
                if existing is not None and existing["content_hash"] == content_hash:
                    return "duplicate"
                raise ConflictError() from None
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn
