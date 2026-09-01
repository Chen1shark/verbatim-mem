"""SQLite + FTS5。"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.embeddings import Embedder
from app.schemas import AddRequest, Message
from app.vector_index import UserFaissIndex

logger = logging.getLogger("verbatim_mem")

_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")
_FTS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "my",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "not",
        "but",
        "if",
        "then",
        "so",
        "can",
        "could",
        "would",
        "should",
        "will",
        "just",
        "please",
        "me",
        "i",
        "we",
        "you",
        "they",
        "it",
    }
)


def fts_match_query(raw: str) -> str | None:
    """把自然语言问句收成 FTS5 MATCH。问句里的 ? / 引号不能原样丢给 MATCH。"""
    found = _FTS_TOKEN.findall(raw)
    if not found:
        return None
    tokens: list[str] = []
    seen: set[str] = set()
    for token in found:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(key)
    keep = [token for token in tokens if token not in _FTS_STOPWORDS]
    use = keep or tokens
    parts: list[str] = []
    for token in use:
        parts.append(f'"{token}"')
        if len(token) >= 2:
            parts.append(f"{token}*")
    return " OR ".join(parts)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rrf_merge(
    fts_hits: list[dict[str, object]],
    dense_hits: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, object]] = {}
    for rank, hit in enumerate(fts_hits, start=1):
        item_id = str(hit["id"])
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
        by_id[item_id] = hit
    for rank, hit in enumerate(dense_hits, start=1):
        item_id = str(hit["id"])
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(item_id, hit)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k]
    merged: list[dict[str, object]] = []
    for item_id in ordered:
        item = dict(by_id[item_id])
        item["score"] = scores[item_id]
        merged.append(item)
    return merged

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

CREATE TABLE IF NOT EXISTS message_vectors (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (id) REFERENCES messages(id)
);

CREATE INDEX IF NOT EXISTS idx_message_vectors_user_id ON message_vectors(user_id);

CREATE TABLE IF NOT EXISTS vector_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

RRF_K = 60
VECTOR_META_MODEL = "embedding_identity"


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
    """单连接；Uvicorn 仅 1 worker。原文在 SQLite，向量按 user_id 进 FAISS。"""

    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        retrieval_mode: str = "hybrid",
    ) -> None:
        self.db_path = db_path
        self._embedder = embedder
        self.retrieval_mode = retrieval_mode.strip().lower() or "hybrid"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._faiss: dict[str, UserFaissIndex] = {}

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
        self._faiss.clear()

    def init_schema(self) -> None:
        conn = self._require_conn()
        conn.executescript(SCHEMA_SQL)
        self._ensure_vector_meta()
        self._backfill_vectors()
        self._rebuild_faiss()

    def warmup(self) -> None:
        self._embedder.warmup()

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
                ids = [
                    f"{body.request_id}:{index}" for index in range(len(body.messages))
                ]
                texts = [message.content for message in body.messages]
                vectors = self._embedder.encode_docs(texts)
                self._write_vectors(conn, body.user_id, ids, vectors)
                conn.execute("COMMIT")
                self._index_vectors(body.user_id, ids, vectors)
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

    def search(self, user_id: str, query: str, top_k: int) -> list[dict[str, object]]:
        """该 user_id 下 FTS5 ∪ FAISS，RRF 合并。没有命中返回空列表。"""
        fts_hits: list[dict[str, object]] = []
        dense_hits: list[dict[str, object]] = []
        with self._lock:
            if self.retrieval_mode in {"fts", "hybrid"}:
                fts_hits = self._search_fts(user_id, query, top_k)
            if self.retrieval_mode in {"dense", "hybrid"}:
                dense_hits = self._search_dense(user_id, query, top_k)
        if self.retrieval_mode == "fts":
            return fts_hits[:top_k]
        if self.retrieval_mode == "dense":
            return dense_hits[:top_k]
        return _rrf_merge(fts_hits, dense_hits, top_k)

    def _search_fts(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, object]]:
        match = fts_match_query(query)
        if match is None:
            return []
        conn = self._require_conn()
        sql = """
            SELECT
                m.id,
                m.content,
                m.timestamp,
                bm25(messages_fts) AS rank
            FROM messages_fts
            INNER JOIN messages AS m ON m.rowid = messages_fts.rowid
            WHERE messages_fts MATCH ?
              AND messages_fts.user_id = ?
              AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, (match, user_id, user_id, top_k)).fetchall()
        except sqlite3.OperationalError:
            logger.info(
                "search fts_error user_id=%s top_k=%s hits=0",
                user_id,
                top_k,
            )
            return []
        hits: list[dict[str, object]] = []
        for row in rows:
            rank = row["rank"]
            score = 0.0 if rank is None else -float(rank)
            hits.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "score": score,
                    "created_at": ms_to_iso(row["timestamp"]),
                }
            )
        return hits

    def _search_dense(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, object]]:
        bucket = self._faiss.get(user_id)
        if bucket is None or bucket.index.ntotal == 0:
            return []
        take = min(max(top_k, 20), bucket.index.ntotal)
        query_vec = self._embedder.encode_query(query)
        pairs = bucket.search(query_vec, take)
        return self._hits_for_ids(user_id, pairs)

    def _hits_for_ids(
        self, user_id: str, pairs: list[tuple[str, float]]
    ) -> list[dict[str, object]]:
        if not pairs:
            return []
        conn = self._require_conn()
        ids = [item_id for item_id, _score in pairs]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT id, content, timestamp
            FROM messages
            WHERE user_id = ? AND id IN ({placeholders})
            """,
            (user_id, *ids),
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        hits: list[dict[str, object]] = []
        for item_id, score in pairs:
            row = by_id.get(item_id)
            if row is None:
                continue
            hits.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "score": score,
                    "created_at": ms_to_iso(row["timestamp"]),
                }
            )
        return hits

    def _ensure_vector_meta(self) -> None:
        conn = self._require_conn()
        expected = self._embedder.identity
        row = conn.execute(
            "SELECT value FROM vector_meta WHERE key = ?",
            (VECTOR_META_MODEL,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO vector_meta (key, value) VALUES (?, ?)",
                (VECTOR_META_MODEL, expected),
            )
            return
        if row["value"] != expected:
            raise RuntimeError(
                "embedding model mismatch; use a new MEMORY_DB_PATH "
                f"(have {row['value']}, want {expected})"
            )

    def _backfill_vectors(self) -> None:
        conn = self._require_conn()
        missing = conn.execute(
            """
            SELECT m.id, m.user_id, m.content
            FROM messages AS m
            LEFT JOIN message_vectors AS v ON v.id = m.id
            WHERE v.id IS NULL
            ORDER BY m.created_at, m.id
            """
        ).fetchall()
        if not missing:
            return
        ids = [row["id"] for row in missing]
        user_ids = [row["user_id"] for row in missing]
        texts = [row["content"] for row in missing]
        vectors = self._embedder.encode_docs(texts)
        conn.execute("BEGIN IMMEDIATE")
        try:
            grouped: dict[str, tuple[list[str], list[int]]] = {}
            for offset, (item_id, user_id) in enumerate(zip(ids, user_ids, strict=True)):
                bucket = grouped.setdefault(user_id, ([], []))
                bucket[0].append(item_id)
                bucket[1].append(offset)
            for user_id, (item_ids, offsets) in grouped.items():
                subset = vectors[offsets]
                self._write_vectors(conn, user_id, item_ids, subset)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _rebuild_faiss(self) -> None:
        self._faiss.clear()
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT id, user_id, dim, embedding FROM message_vectors ORDER BY rowid"
        ).fetchall()
        grouped: dict[str, tuple[list[str], list[np.ndarray]]] = {}
        for row in rows:
            if int(row["dim"]) != self._embedder.dim:
                raise RuntimeError("stored vector dim does not match embedder")
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            bucket = grouped.setdefault(row["user_id"], ([], []))
            bucket[0].append(row["id"])
            bucket[1].append(vector)
        for user_id, (ids, parts) in grouped.items():
            matrix = np.vstack(parts)
            self._index_vectors(user_id, ids, matrix)

    def _write_vectors(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        for item_id, vector in zip(ids, matrix, strict=True):
            conn.execute(
                """
                INSERT INTO message_vectors (id, user_id, dim, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (item_id, user_id, int(self._embedder.dim), vector.tobytes()),
            )

    def _index_vectors(self, user_id: str, ids: list[str], vectors: np.ndarray) -> None:
        bucket = self._faiss.get(user_id)
        if bucket is None:
            bucket = UserFaissIndex(self._embedder.dim)
            self._faiss[user_id] = bucket
        bucket.add(ids, vectors)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn
