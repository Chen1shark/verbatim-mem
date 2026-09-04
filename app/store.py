"""SQLite + FTS5。"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import RetrievalConfig
from app.embeddings import Embedder
from app.intent import (
    CURRENT_STATE,
    HISTORICAL_STATE,
    PREFERENCE,
    _TEMPORAL_NEW_EN,
    _TEMPORAL_NEW_PHRASES,
    _TEMPORAL_OLD_EN,
    _TEMPORAL_OLD_PHRASES,
    classify_intent,
)
from app.schemas import AddRequest, Message
from app.vector_index import UserFaissIndex

logger = logging.getLogger("verbatim_mem")

_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")
_NUM_RE = re.compile(r"\d+")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_OPTION_PREFIX = re.compile(r"^[A-Ha-h][.)\:]\s*")
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
def _dedupe(items: list[str]) -> list[str]:
    """保序去重。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _english_tokens(raw: str) -> list[str]:
    """抽 [A-Za-z0-9]+ 小写保序去重。"""
    return _dedupe(_FTS_TOKEN.findall(raw.lower()))


def _light_stems(token: str) -> list[str]:
    """保守英文后缀 ies/ing/ed/es/s，茎长至少 3。"""
    stems: list[str] = []
    if token.endswith("ies") and len(token) > 5:
        stems.append(token[:-3] + "y")
    elif token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 4:
            stems.append(stem)
    elif token.endswith("ed") and len(token) > 5:
        stem = token[:-2]
        if len(stem) >= 4:
            stems.append(stem)
    elif token.endswith("es") and len(token) > 5:
        stems.append(token[:-2])
    elif token.endswith("s") and not token.endswith("ss") and len(token) >= 4:
        stems.append(token[:-1])
    return [item for item in stems if item != token and len(item) >= 3]


def index_clues(text: str) -> str:
    """messages.clues：对实词做 _light_stems，写入 messages_fts.clues。"""
    english = [token for token in _english_tokens(text) if token not in _FTS_STOPWORDS]
    extra: list[str] = []
    for token in english:
        extra.extend(_light_stems(token))
    return " ".join(_dedupe(extra))


def fts_match_query(raw: str) -> str | None:
    """问句收成 FTS5 MATCH：英文词+_light_stems，"token" OR token*。"""
    english = _english_tokens(raw)
    keep = [token for token in english if token not in _FTS_STOPWORDS]
    use = keep or english
    pieces: list[str] = []
    seen: set[str] = set()

    def _add(token: str, prefix: bool) -> bool:
        if not token or token in seen:
            return len(pieces) >= FTS_CLAUSE_CAP
        seen.add(token)
        pieces.append(f'"{token}"')
        if prefix and len(token) >= 2:
            pieces.append(f"{token}*")
        return len(pieces) >= FTS_CLAUSE_CAP

    stemmed: list[str] = []
    for token in use:
        stemmed.extend(_light_stems(token))
    for token in _dedupe(list(use) + stemmed):
        if _add(token, prefix=True):
            return " OR ".join(pieces)
    if not pieces:
        return None
    return " OR ".join(pieces)


def ms_to_iso(ms: int) -> str:
    """messages.timestamp 毫秒 → SearchItem.created_at 的 UTC ISO-8601。"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


RRF_K = 60
RRF_W_FTS = 1.2
RRF_W_DENSE = 1.0
RRF_W_BLOCKS = 1.1
TIME_WEIGHT = 0.0
TIME_WEIGHT_TEMPORAL = 0.08
LEXICAL_WEIGHT = 0.25
NUMERIC_WEIGHT = 0.2
ENTITY_WEIGHT = 0.22
OPTION_WEIGHT = 0.18
PREF_WEIGHT = 0.12
UPDATE_WEIGHT = 0.08
NEIGHBOR_WINDOW = 1
NEIGHBOR_SCORE_DELTA = 0.001
_DEFAULT_RETRIEVAL = RetrievalConfig(
    rrf_k=RRF_K,
    rrf_w_fts=RRF_W_FTS,
    rrf_w_dense=RRF_W_DENSE,
    rrf_w_blocks=RRF_W_BLOCKS,
    lexical_weight=LEXICAL_WEIGHT,
    numeric_weight=NUMERIC_WEIGHT,
    entity_weight=ENTITY_WEIGHT,
    option_weight=OPTION_WEIGHT,
    time_weight_temporal=TIME_WEIGHT_TEMPORAL,
    preference_weight=PREF_WEIGHT,
    update_weight=UPDATE_WEIGHT,
    neighbor_window=NEIGHBOR_WINDOW,
    neighbor_score_delta=NEIGHBOR_SCORE_DELTA,
)
_SESSION_ORDER = "timestamp IS NULL, timestamp, source_order, created_at, id"
FTS_CLAUSE_CAP = 48
BLOCK_WIDTHS = (2, 3)
VECTOR_META_MODEL = "embedding_identity"
_PREF_CONTENT = re.compile(
    r"\b(i\s+(like|love|prefer|enjoy|hate|dislike)|my\s+favorite|i\s+don't\s+like|"
    r"i\s+do\s+not\s+like)\b",
    re.IGNORECASE,
)
_UPDATE_CUES = (
    "actually",
    "i moved",
    "i've moved",
    "i have moved",
    "not anymore",
    "no longer",
    "i now live",
    "i now work",
    "correction",
    "i was wrong",
)


def _span_if_contiguous(positions: list[int]) -> tuple[int, int] | None:
    """已排序的 positions 若是连续整数，返回闭区间 [lo, hi]。"""
    if not positions:
        return None
    lo = positions[0]
    hi = positions[-1]
    if hi - lo + 1 != len(positions):
        return None
    return lo, hi


def _search_texts(query: str, options: list[str] | None) -> list[str]:
    """query 一路，每个 SearchRequest.options 去前缀后再与 query 拼一路。"""
    texts = [query]
    if not options:
        return texts
    for opt in options:
        cleaned = _OPTION_PREFIX.sub("", opt).strip()
        if cleaned:
            texts.append(f"{query} {cleaned}")
    return texts


def _meaningful_tokens(raw: str) -> list[str]:
    """英文词去 _FTS_STOPWORDS；若全是停用词则退回原 token。"""
    tokens = _english_tokens(raw)
    keep = [token for token in tokens if token not in _FTS_STOPWORDS]
    return keep or tokens


def _lexical_token_set(raw: str) -> set[str]:
    """_meaningful_tokens + _light_stems。"""
    english = _meaningful_tokens(raw)
    tokens = set(english)
    extra: list[str] = []
    for token in tokens:
        extra.extend(_light_stems(token))
    tokens.update(extra)
    return tokens


def _query_core(query: str, options: list[str] | None = None) -> list[str]:
    """问句实词保序，作覆盖目标；不含 SearchRequest.options。"""
    del options
    return _meaningful_tokens(query)


def _term_variants(term: str) -> set[str]:
    """原词 + _light_stems。"""
    variants = {term}
    variants.update(_light_stems(term))
    extra: list[str] = []
    for item in variants:
        extra.extend(_light_stems(item))
    variants.update(extra)
    return variants


def _covers_term(doc_lex: set[str], term: str) -> bool:
    """doc_lex 是否覆盖 term 的 _term_variants。"""
    return bool(_term_variants(term) & doc_lex)


def _doc_query_cover(content: str, core: set[str]) -> set[str]:
    """content 覆盖到的问句实词。"""
    if not core:
        return set()
    doc_lex = _lexical_token_set(content)
    return {term for term in core if _covers_term(doc_lex, term)}


def _proper_nouns(*texts: str) -> set[str]:
    """_PROPER 命中且不在 _FTS_STOPWORDS 的小写专名。"""
    names: set[str] = set()
    for text in texts:
        for match in _PROPER.findall(text):
            key = match.lower()
            if key in _FTS_STOPWORDS:
                continue
            names.add(key)
    return names


def _option_needles(options: list[str] | None) -> list[str]:
    """SearchRequest.options 去掉 _OPTION_PREFIX 后的小写 needle。"""
    if not options:
        return []
    needles: list[str] = []
    for opt in options:
        text = _OPTION_PREFIX.sub("", opt).strip()
        if len(text) >= 2:
            needles.append(text.lower())
    return needles


def _temporal_alpha(
    query: str,
    temporal_weight: float = TIME_WEIGHT_TEMPORAL,
    *,
    intent_temporal: bool = True,
) -> float:
    """classify_intent 为 current_state 时 +temporal_weight，historical_state 时为负；intent_temporal=False 时只认 _TEMPORAL_* 词。"""
    if not intent_temporal:
        lowered = query.lower()
        tokens = {token.lower() for token in _FTS_TOKEN.findall(query)}
        old = bool(tokens & _TEMPORAL_OLD_EN) or any(
            phrase in lowered for phrase in _TEMPORAL_OLD_PHRASES
        )
        new = bool(tokens & _TEMPORAL_NEW_EN) or any(
            phrase in lowered for phrase in _TEMPORAL_NEW_PHRASES
        )
        if old and not new:
            return -temporal_weight
        if new:
            return temporal_weight
        return TIME_WEIGHT
    intent = classify_intent(query)
    if intent == CURRENT_STATE:
        return temporal_weight
    if intent == HISTORICAL_STATE:
        return -temporal_weight
    return TIME_WEIGHT


def _hit_from_row(row: sqlite3.Row, score: float) -> dict[str, object]:
    """messages 行转内部 hit：id / content / score / created_at / timestamp / session_id / role / source_order。"""
    raw_ts = row["timestamp"]
    ts = int(raw_ts) if raw_ts is not None else None
    keys = set(row.keys())
    source_order = int(row["source_order"]) if "source_order" in keys else 0
    role = str(row["role"]) if "role" in keys else None
    return {
        "id": row["id"],
        "content": row["content"],
        "score": score,
        "created_at": ms_to_iso(ts) if ts is not None else None,
        "timestamp": ts,
        "session_id": row["session_id"],
        "role": role,
        "source_order": source_order,
    }


def _public_hit(hit: dict[str, object]) -> dict[str, object]:
    """去掉内部 timestamp、session_id、source_order、rrf_score、_origin，只留 SearchItem。"""
    return {
        "id": hit["id"],
        "content": hit["content"],
        "score": hit["score"],
        "created_at": hit["created_at"],
        "role": hit.get("role"),
    }


def _apply_recency(
    hits: list[dict[str, object]],
    query: str,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """有时间意图时，timestamp 在候选集内 min-max 为 rec，score 加 α * rec；缺 timestamp 的 hit 不加。"""
    if not hits:
        return []
    weights = cfg or _DEFAULT_RETRIEVAL
    alpha = _temporal_alpha(
        query,
        weights.time_weight_temporal,
        intent_temporal=weights.intent_temporal,
    )
    dated = [
        int(hit["timestamp"])
        for hit in hits
        if hit.get("timestamp") is not None
    ]
    min_ts = min(dated) if dated else 0
    max_ts = max(dated) if dated else 0
    ts_span = max_ts - min_ts
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        ts = hit.get("timestamp")
        if alpha != 0.0 and ts is not None and ts_span > 0:
            rec = (int(ts) - min_ts) / ts_span
            item["score"] = float(hit["score"]) + alpha * rec
        ranked.append(item)

    def _tie(item: dict[str, object]) -> int:
        ts = item.get("timestamp")
        if ts is None or alpha == 0.0:
            return 0
        if alpha < 0:
            return int(ts)
        return -int(ts)

    ranked.sort(key=lambda item: (-float(item["score"]), _tie(item), str(item["id"])))
    return ranked


def _apply_lexical(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """按问句实词覆盖率加 lexical_weight，content 侧用 _term_variants 对齐。"""
    if not hits:
        return []
    core = _query_core(query, options)
    if not core:
        return hits
    weight = (cfg or _DEFAULT_RETRIEVAL).lexical_weight
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        covered = _doc_query_cover(str(hit["content"]), set(core))
        item["score"] = float(hit["score"]) + weight * (len(covered) / len(core))
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_numeric(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """query 与 content 的 _NUM_RE 数字有交集则 score 加 numeric_weight。"""
    del options
    if not hits:
        return []
    qnums = set(_NUM_RE.findall(query))
    if not qnums:
        return hits
    weight = (cfg or _DEFAULT_RETRIEVAL).numeric_weight
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        cnums = set(_NUM_RE.findall(str(hit["content"])))
        if qnums & cnums:
            item["score"] = float(hit["score"]) + weight
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_entity(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """query 的 _proper_nouns 与 content 的 _english_tokens 有交集则加 entity_weight。"""
    del options
    if not hits:
        return []
    names = _proper_nouns(query)
    if not names:
        return hits
    weight = (cfg or _DEFAULT_RETRIEVAL).entity_weight
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        content_tokens = set(_english_tokens(str(hit["content"])))
        if names & content_tokens:
            item["score"] = float(hit["score"]) + weight
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_options(
    hits: list[dict[str, object]],
    options: list[str] | None,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """_option_needles 作为子串出现在 content 则加 option_weight。"""
    if not hits:
        return []
    needles = _option_needles(options)
    if not needles:
        return hits
    weight = (cfg or _DEFAULT_RETRIEVAL).option_weight
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        content = str(hit["content"]).lower()
        if any(needle in content for needle in needles):
            item["score"] = float(hit["score"]) + weight
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_update(
    hits: list[dict[str, object]],
    query: str,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """content 含 _UPDATE_CUES：current_state 加 update_weight，historical_state 减 update_weight。"""
    if not hits:
        return []
    intent = classify_intent(query)
    weight = (cfg or _DEFAULT_RETRIEVAL).update_weight
    if weight == 0.0:
        return hits
    if intent == HISTORICAL_STATE:
        signed = -weight
    elif intent == CURRENT_STATE:
        signed = weight
    else:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        content = str(hit["content"]).lower()
        if any(cue in content for cue in _UPDATE_CUES):
            item["score"] = float(hit["score"]) + signed
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_preference(
    hits: list[dict[str, object]],
    query: str,
    cfg: RetrievalConfig | None = None,
) -> list[dict[str, object]]:
    """classify_intent 为 preference 且 content 匹配 _PREF_CONTENT 时加 preference_weight。"""
    if not hits:
        return []
    if classify_intent(query) != PREFERENCE:
        return hits
    weight = (cfg or _DEFAULT_RETRIEVAL).preference_weight
    if weight == 0.0:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        if _PREF_CONTENT.search(str(hit["content"])):
            item["score"] = float(hit["score"]) + weight
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _cover_reorder(
    hits: list[dict[str, object]],
    query: str,
) -> list[dict[str, object]]:
    """先取最高分句，再优先补问句里尚未覆盖、且在候选池中较稀有的实词。"""
    if len(hits) <= 1:
        return hits
    core = set(_query_core(query))
    if not core:
        return hits
    covers = [_doc_query_cover(str(hit["content"]), core) for hit in hits]
    df: dict[str, int] = {}
    for covered in covers:
        for term in covered:
            df[term] = df.get(term, 0) + 1
    n_docs = len(hits)
    idf = {
        term: math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0 for term in core
    }
    scores = [float(hit["score"]) for hit in hits]
    selected = [0]
    covered_terms = set(covers[0])
    remaining = list(range(1, len(hits)))
    while remaining:
        def _key(index: int) -> tuple[float, float]:
            novel = covers[index] - covered_terms
            gain = sum(idf[term] for term in novel)
            return (gain, scores[index])

        best_i = max(remaining, key=_key)
        remaining.remove(best_i)
        selected.append(best_i)
        covered_terms |= covers[best_i]
    return [hits[index] for index in selected]


def _rrf_merge(
    hit_lists: list[list[dict[str, object]]],
    top_k: int,
    weights: tuple[float, ...] | None = None,
    rrf_k: int = RRF_K,
) -> list[dict[str, object]]:
    """多路按 weight/(rrf_k+rank) 合并去重，截断为 top_k。"""
    if weights is None:
        weights = tuple(1.0 for _ in hit_lists)
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, object]] = {}
    for hits, weight in zip(hit_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            item_id = str(hit["id"])
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rrf_k + rank)
            by_id.setdefault(item_id, hit)
    if not scores:
        return []
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_k]
    merged: list[dict[str, object]] = []
    for item_id in ordered:
        item = dict(by_id[item_id])
        item["score"] = scores[item_id]
        item["rrf_score"] = scores[item_id]
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
    timestamp INTEGER,
    content TEXT NOT NULL,
    clues TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    source_order INTEGER NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(request_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_request_id ON messages(request_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id, timestamp, source_order);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    clues,
    user_id UNINDEXED,
    id UNINDEXED,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, clues, user_id, id)
    VALUES (new.rowid, new.content, new.clues, new.user_id, new.id);
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

CREATE VIRTUAL TABLE IF NOT EXISTS message_blocks_fts USING fts5(
    content,
    user_id UNINDEXED,
    session_id UNINDEXED,
    left_id UNINDEXED,
    mid_id UNINDEXED,
    right_id UNINDEXED,
    tokenize = 'unicode61'
);
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
    """单连接；Uvicorn 仅 1 worker。原文在 SQLite，向量按 user_id 进 FAISS。"""

    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        retrieval_mode: str = "hybrid",
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        """绑定 MEMORY_DB_PATH、Embedder、MEMORY_RETRIEVAL_MODE、RetrievalConfig。"""
        self.db_path = db_path
        self._embedder = embedder
        self.retrieval_mode = retrieval_mode.strip().lower() or "hybrid"
        self.retrieval = retrieval or _DEFAULT_RETRIEVAL
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._faiss: dict[str, UserFaissIndex] = {}

    def open(self) -> None:
        """打开 sqlite3：WAL、synchronous=NORMAL、busy_timeout=30000、foreign_keys=ON、temp_store=MEMORY、cache_size=-65536，写入 _conn。"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")
        self._conn = conn

    def close(self) -> None:
        """关闭 _conn，清空 _faiss，调用 Embedder.close。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._faiss.clear()
        self._embedder.close()

    def init_schema(self) -> None:
        """SCHEMA_SQL 后 _ensure_message_time_schema、_ensure_index_schema、_ensure_vector_meta、_backfill_vectors、_backfill_blocks、_rebuild_faiss。"""
        conn = self._require_conn()
        conn.executescript(SCHEMA_SQL)
        rebuilt = self._ensure_message_time_schema()
        self._ensure_index_schema(force_fts=rebuilt)
        self._ensure_vector_meta()
        self._backfill_vectors()
        self._backfill_blocks()
        self._rebuild_faiss()

    def warmup(self) -> None:
        """调用 Embedder.warmup。"""
        self._embedder.warmup()

    def ping(self) -> None:
        """对 _conn 执行 SELECT 1，给 GET /health 探活。"""
        self._require_conn().execute("SELECT 1").fetchone()

    def add(self, body: AddRequest) -> str:
        """查重后在锁外 Embedder.encode_docs；事务 executemany 写 requests / messages（含 index_clues）/ message_vectors，_write_session_blocks(new_ids) 维护 message_blocks_fts，再 _index_vectors。返回 created 或 duplicate。冲突抛 ConflictError。"""
        content_hash = fingerprint(body.user_id, body.session_id, body.messages)
        now = int(time.time() * 1000)
        conn = self._require_conn()
        ids = [f"{body.request_id}:{index}" for index in range(len(body.messages))]
        texts = [message.content for message in body.messages]

        with self._lock:
            existing = conn.execute(
                "SELECT content_hash FROM requests WHERE request_id = ?",
                (body.request_id,),
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return "duplicate"
                raise ConflictError()

        vectors = self._embedder.encode_docs(texts)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape != (len(ids), self._embedder.dim):
            raise RuntimeError("embedding shape does not match messages")

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
                source_base_row = conn.execute(
                    """
                    SELECT COALESCE(MAX(source_order), -1) AS max_order
                    FROM messages
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (body.user_id, body.session_id),
                ).fetchone()
                source_base = int(source_base_row["max_order"]) if source_base_row else -1
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
                conn.executemany(
                    """
                    INSERT INTO messages (
                        id, request_id, user_id, session_id,
                        role, timestamp, content, clues, created_at, source_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            ids[index],
                            body.request_id,
                            body.user_id,
                            body.session_id,
                            message.role,
                            message.timestamp,
                            message.content,
                            index_clues(message.content),
                            now,
                            source_base + 1 + index,
                        )
                        for index, message in enumerate(body.messages)
                    ],
                )
                self._write_vectors(conn, body.user_id, ids, matrix)
                self._write_session_blocks(conn, body.user_id, body.session_id, ids)
                conn.execute("COMMIT")
                self._index_vectors(body.user_id, ids, matrix)
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

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        options: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """该 user_id 下 FTS5 ∪ message_blocks_fts ∪ FAISS 分路召回，加权 _rrf_merge 后字面/数字/专名/选项/_apply_update/_apply_preference/_apply_recency、_cover_reorder，再 _compose_results。"""
        cfg = self.retrieval
        channels = _search_texts(query, options)
        query_vecs: np.ndarray | None = None
        if self.retrieval_mode in {"dense", "hybrid"}:
            query_vecs = self._embedder.encode_docs(channels)
        with self._lock:
            channel_rankings: list[list[dict[str, object]]] = []
            for index, search_text in enumerate(channels):
                qvec = None if query_vecs is None else query_vecs[index : index + 1]
                channel_rankings.append(
                    self._search_channel(user_id, search_text, qvec, cfg)
                )
            if len(channel_rankings) == 1:
                ranked = channel_rankings[0]
            else:
                ranked = _rrf_merge(
                    channel_rankings, cfg.candidate_pool, rrf_k=cfg.rrf_k
                )
            ranked = _apply_lexical(ranked, query, options, cfg)
            ranked = _apply_numeric(ranked, query, options, cfg)
            ranked = _apply_entity(ranked, query, options, cfg)
            ranked = _apply_options(ranked, options, cfg)
            ranked = _apply_update(ranked, query, cfg)
            ranked = _apply_preference(ranked, query, cfg)
            ranked = _apply_recency(ranked, query, cfg)
            ranked = _cover_reorder(ranked, query)
            return self._compose_results(user_id, ranked, top_k)

    def _search_channel(
        self,
        user_id: str,
        search_text: str,
        query_vec: np.ndarray | None,
        cfg: RetrievalConfig,
    ) -> list[dict[str, object]]:
        """单路 search_text：FTS / blocks / dense 再 _rrf_merge 到 cfg.candidate_pool。"""
        fts_hits: list[dict[str, object]] = []
        block_hits: list[dict[str, object]] = []
        dense_hits: list[dict[str, object]] = []
        if self.retrieval_mode in {"fts", "hybrid"}:
            fts_hits = self._search_fts(user_id, search_text, cfg.fts_pool)
            block_hits = self._search_blocks(user_id, search_text, cfg.block_pool)
        if self.retrieval_mode in {"dense", "hybrid"}:
            dense_hits = self._search_dense(user_id, query_vec, cfg.dense_pool)
        if self.retrieval_mode == "fts":
            return _rrf_merge(
                [fts_hits, block_hits],
                cfg.candidate_pool,
                (cfg.rrf_w_fts, cfg.rrf_w_blocks),
                rrf_k=cfg.rrf_k,
            )
        if self.retrieval_mode == "dense":
            return dense_hits[: cfg.candidate_pool]
        return _rrf_merge(
            [fts_hits, dense_hits, block_hits],
            cfg.candidate_pool,
            (cfg.rrf_w_fts, cfg.rrf_w_dense, cfg.rrf_w_blocks),
            rrf_k=cfg.rrf_k,
        )

    def _search_fts(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, object]]:
        """messages_fts MATCH 且 user_id 双条件过滤，按 bm25 取 top_k，score=-bm25。"""
        match = fts_match_query(query)
        if match is None:
            return []
        conn = self._require_conn()
        sql = """
            SELECT
                m.id,
                m.content,
                m.timestamp,
                m.session_id,
                m.role,
                m.source_order,
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
            hits.append(_hit_from_row(row, score))
        return hits

    def _search_dense(
        self, user_id: str, query_vec: np.ndarray | None, top_k: int
    ) -> list[dict[str, object]]:
        """该 user_id 的 UserFaissIndex 近邻，再 _hits_for_ids 回 messages。"""
        if query_vec is None:
            return []
        bucket = self._faiss.get(user_id)
        if bucket is None or bucket.index.ntotal == 0:
            return []
        take = min(max(top_k, 20), bucket.index.ntotal)
        pairs = bucket.search(query_vec, take)
        return self._hits_for_ids(user_id, pairs)

    def _search_blocks(
        self, user_id: str, query: str, top_k: int
    ) -> list[dict[str, object]]:
        """message_blocks_fts MATCH，把 left_id / mid_id / right_id 回 messages，score=-bm25。"""
        match = fts_match_query(query)
        if match is None:
            return []
        conn = self._require_conn()
        sql = """
            SELECT left_id, mid_id, right_id, bm25(message_blocks_fts) AS rank
            FROM message_blocks_fts
            WHERE message_blocks_fts MATCH ?
              AND user_id = ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, (match, user_id, top_k)).fetchall()
        except sqlite3.OperationalError:
            logger.info(
                "search block_fts_error user_id=%s top_k=%s hits=0",
                user_id,
                top_k,
            )
            return []
        best: dict[str, float] = {}
        order: list[str] = []
        for row in rows:
            rank = row["rank"]
            score = 0.0 if rank is None else -float(rank)
            for item_id in (str(row["left_id"]), str(row["mid_id"] or ""), str(row["right_id"])):
                if not item_id:
                    continue
                if item_id not in best:
                    order.append(item_id)
                    best[item_id] = score
                elif score > best[item_id]:
                    best[item_id] = score
        pairs = [(item_id, best[item_id]) for item_id in order]
        return self._hits_for_ids(user_id, pairs)

    def _hits_for_ids(
        self, user_id: str, pairs: list[tuple[str, float]]
    ) -> list[dict[str, object]]:
        """FAISS id 回表：WHERE user_id = ? AND id IN (...)，拼 _hit_from_row。"""
        if not pairs:
            return []
        conn = self._require_conn()
        ids = [item_id for item_id, _score in pairs]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT id, content, timestamp, session_id, role, source_order
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
            hits.append(_hit_from_row(row, score))
        return hits

    def _compose_results(
        self,
        user_id: str,
        hits: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """先取 seed_k 条直接命中，再按 neighbor_budget / neighbor_window 补同 session 邻句，剩余槽位填其它命中。"""
        if not hits:
            return []
        cfg = self.retrieval
        seed_n = min(cfg.seed_k, top_k, len(hits))
        seeds = hits[:seed_n]
        seed_ids = {str(hit["id"]) for hit in seeds}
        session_ids: list[str] = []
        seen_sessions: set[str] = set()
        for hit in seeds:
            session_id = str(hit.get("session_id") or "")
            if not session_id or session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            session_ids.append(session_id)
        by_id: dict[str, dict[str, object]] = {str(hit["id"]): hit for hit in hits}
        order_by_session: dict[str, list[str]] = {}
        if session_ids:
            conn = self._require_conn()
            placeholders = ",".join("?" * len(session_ids))
            rows = conn.execute(
                f"""
                SELECT id, content, timestamp, session_id, role, source_order
                FROM messages
                WHERE user_id = ? AND session_id IN ({placeholders})
                ORDER BY session_id, {_SESSION_ORDER}
                """,
                (user_id, *session_ids),
            ).fetchall()
            for row in rows:
                item_id = str(row["id"])
                session_id = str(row["session_id"])
                order_by_session.setdefault(session_id, []).append(item_id)
                if item_id not in by_id:
                    by_id[item_id] = _hit_from_row(row, 0.0)
        index_of: dict[str, int] = {}
        for ids in order_by_session.values():
            for index, item_id in enumerate(ids):
                index_of[item_id] = index

        budget = min(cfg.neighbor_budget, max(0, top_k - seed_n))
        neighbor_queue: list[dict[str, object]] = []
        seen_neighbors: set[str] = set()
        window = max(0, cfg.neighbor_window)
        for seed in seeds:
            hit_id = str(seed["id"])
            session_id = str(seed.get("session_id") or "")
            ids = order_by_session.get(session_id, [])
            idx = index_of.get(hit_id)
            if idx is None:
                continue
            for distance in range(1, window + 1):
                candidates: list[str] = []
                if idx - distance >= 0:
                    candidates.append(ids[idx - distance])
                if idx + distance < len(ids):
                    candidates.append(ids[idx + distance])
                for item_id in candidates:
                    if item_id in seed_ids or item_id in seen_neighbors:
                        continue
                    seen_neighbors.add(item_id)
                    neighbor = dict(by_id[item_id])
                    neighbor["score"] = float(seed["score"]) - (
                        cfg.neighbor_score_delta * distance
                    )
                    neighbor["_origin"] = "neighbor"
                    neighbor_queue.append(neighbor)

        chosen_neighbors = neighbor_queue[:budget]
        neighbor_ids = {str(hit["id"]) for hit in chosen_neighbors}
        used = set(seed_ids) | neighbor_ids
        leftover = top_k - seed_n - len(chosen_neighbors)
        fillers: list[dict[str, object]] = []
        for hit in hits[seed_n:]:
            if leftover <= 0:
                break
            item_id = str(hit["id"])
            if item_id in used:
                continue
            fillers.append(dict(hit))
            fillers[-1]["_origin"] = "filler"
            used.add(item_id)
            leftover -= 1

        assembled: list[dict[str, object]] = []
        seen: set[str] = set()
        neighbor_by_id = {str(hit["id"]): hit for hit in chosen_neighbors}
        for seed in seeds:
            hit_id = str(seed["id"])
            if hit_id not in seen:
                seen.add(hit_id)
                item = dict(seed)
                item["_origin"] = "seed"
                assembled.append(item)
            session_id = str(seed.get("session_id") or "")
            ids = order_by_session.get(session_id, [])
            idx = index_of.get(hit_id)
            if idx is None:
                continue
            for distance in range(1, window + 1):
                candidates = []
                if idx - distance >= 0:
                    candidates.append(ids[idx - distance])
                if idx + distance < len(ids):
                    candidates.append(ids[idx + distance])
                for item_id in candidates:
                    if item_id in neighbor_by_id and item_id not in seen:
                        seen.add(item_id)
                        assembled.append(neighbor_by_id[item_id])
        for hit in fillers:
            item_id = str(hit["id"])
            if item_id in seen:
                continue
            seen.add(item_id)
            assembled.append(hit)
        return [_public_hit(item) for item in assembled[:top_k]]

    def _fts_has_column(self, table: str, column: str) -> bool:
        """SELECT column FROM table LIMIT 0 能执行则为 True。"""
        conn = self._require_conn()
        try:
            conn.execute(f"SELECT {column} FROM {table} LIMIT 0")
        except sqlite3.OperationalError:
            return False
        return True

    def _ensure_index_schema(self, force_fts: bool = False) -> None:
        """补 messages.clues、messages_fts.clues、message_blocks_fts.mid_id；clues 变了或 force_fts 则重建 messages_fts。"""
        conn = self._require_conn()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "clues" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN clues TEXT NOT NULL DEFAULT ''")
        clues_changed = self._refresh_message_clues()
        if clues_changed or force_fts or not self._fts_has_column("messages_fts", "clues"):
            conn.execute("DROP TRIGGER IF EXISTS messages_ai")
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.execute(
                """
                CREATE VIRTUAL TABLE messages_fts USING fts5(
                    content,
                    clues,
                    user_id UNINDEXED,
                    id UNINDEXED,
                    tokenize = 'unicode61'
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, clues, user_id, id)
                    VALUES (new.rowid, new.content, new.clues, new.user_id, new.id);
                END
                """
            )
            conn.execute(
                """
                INSERT INTO messages_fts(rowid, content, clues, user_id, id)
                SELECT rowid, content, clues, user_id, id FROM messages
                """
            )
        if not self._fts_has_column("message_blocks_fts", "mid_id"):
            conn.execute("DROP TABLE IF EXISTS message_blocks_fts")
            conn.execute(
                """
                CREATE VIRTUAL TABLE message_blocks_fts USING fts5(
                    content,
                    user_id UNINDEXED,
                    session_id UNINDEXED,
                    left_id UNINDEXED,
                    mid_id UNINDEXED,
                    right_id UNINDEXED,
                    tokenize = 'unicode61'
                )
                """
            )

    def _ensure_message_time_schema(self) -> bool:
        """messages.timestamp 可空，并保证 source_order；旧表重建后返回 True。"""
        conn = self._require_conn()
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(messages)")}
        if not cols:
            return False
        if "clues" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN clues TEXT NOT NULL DEFAULT ''")
            cols = {row[1]: row for row in conn.execute("PRAGMA table_info(messages)")}
        timestamp_notnull = int(cols["timestamp"][3]) if "timestamp" in cols else 1
        has_source_order = "source_order" in cols
        if timestamp_notnull == 0 and has_source_order:
            return False
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TRIGGER IF EXISTS messages_ai")
        conn.execute("DROP TABLE IF EXISTS messages_migrate")
        conn.execute(
            """
            CREATE TABLE messages_migrate (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp INTEGER,
                content TEXT NOT NULL,
                clues TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                source_order INTEGER NOT NULL,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO messages_migrate (
                id, request_id, user_id, session_id, role,
                timestamp, content, clues, created_at, source_order
            )
            SELECT
                id, request_id, user_id, session_id, role,
                CASE WHEN timestamp = 0 THEN NULL ELSE timestamp END,
                content,
                COALESCE(clues, ''),
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, session_id
                    ORDER BY timestamp, id
                ) - 1
            FROM messages
            """
        )
        conn.execute("DROP TABLE messages")
        conn.execute("ALTER TABLE messages_migrate RENAME TO messages")
        conn.execute("DROP INDEX IF EXISTS idx_messages_user_id")
        conn.execute("DROP INDEX IF EXISTS idx_messages_request_id")
        conn.execute("DROP INDEX IF EXISTS idx_messages_user_session")
        conn.execute("CREATE INDEX idx_messages_user_id ON messages(user_id)")
        conn.execute("CREATE INDEX idx_messages_request_id ON messages(request_id)")
        conn.execute(
            """
            CREATE INDEX idx_messages_user_session
            ON messages(user_id, session_id, timestamp, source_order)
            """
        )
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.execute("PRAGMA foreign_keys=ON")
        return True

    def _refresh_message_clues(self) -> bool:
        """按 messages.content 重算 index_clues 写回 clues；有改动返回 True。"""
        conn = self._require_conn()
        rows = conn.execute("SELECT id, content, clues FROM messages").fetchall()
        changed = False
        for row in rows:
            wanted = index_clues(row["content"])
            if (row["clues"] or "") != wanted:
                conn.execute(
                    "UPDATE messages SET clues = ? WHERE id = ?",
                    (wanted, row["id"]),
                )
                changed = True
        return changed

    def _ensure_vector_meta(self) -> None:
        """读写 vector_meta.embedding_identity；与 Embedder.identity 不一致则拒绝启动。"""
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
        """messages 缺 message_vectors 的行走 Embedder.encode_docs，再 _write_vectors。"""
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
        if vectors.shape[0] != len(ids):
            raise RuntimeError("embedding shape does not match messages")
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

    def _backfill_blocks(self) -> None:
        """按 messages 的 user_id、session_id 调用 _write_session_blocks。"""
        conn = self._require_conn()
        sessions = conn.execute(
            "SELECT DISTINCT user_id, session_id FROM messages"
        ).fetchall()
        for row in sessions:
            self._write_session_blocks(conn, row["user_id"], row["session_id"])

    def _write_session_blocks(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        new_ids: list[str] | None = None,
    ) -> None:
        """维护 message_blocks_fts 的 2/3 句窗：new_ids 连续则只改插入点，否则整 session 重建。"""
        rows = conn.execute(
            """
            SELECT id, content
            FROM messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY timestamp IS NULL, timestamp, source_order, created_at, id
            """,
            (user_id, session_id),
        ).fetchall()
        if new_ids is None:
            self._rebuild_session_blocks(conn, user_id, session_id, rows)
            return
        index_of = {str(row["id"]): index for index, row in enumerate(rows)}
        positions: list[int] = []
        for item_id in new_ids:
            pos = index_of.get(str(item_id))
            if pos is None:
                self._rebuild_session_blocks(conn, user_id, session_id, rows)
                return
            positions.append(pos)
        positions.sort()
        span = _span_if_contiguous(positions)
        if span is None:
            self._rebuild_session_blocks(conn, user_id, session_id, rows)
            return
        lo, hi = span
        n = len(rows)
        if lo > 0 and hi + 1 < n:
            stale = [
                (rows[lo - 1]["id"], rows[hi + 1]["id"]),
            ]
            if lo > 1:
                stale.append((rows[lo - 2]["id"], rows[hi + 1]["id"]))
            if hi + 2 < n:
                stale.append((rows[lo - 1]["id"], rows[hi + 2]["id"]))
            for left_id, right_id in stale:
                conn.execute(
                    """
                    DELETE FROM message_blocks_fts
                    WHERE user_id = ? AND session_id = ? AND left_id = ? AND right_id = ?
                    """,
                    (user_id, session_id, left_id, right_id),
                )
        self._insert_block_windows(conn, user_id, session_id, rows, lo, hi)

    def _rebuild_session_blocks(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        rows: list[sqlite3.Row],
    ) -> None:
        """DELETE 该 session 的 message_blocks_fts，再按 BLOCK_WIDTHS 写入。"""
        conn.execute(
            "DELETE FROM message_blocks_fts WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        if rows:
            self._insert_block_windows(conn, user_id, session_id, rows, 0, len(rows) - 1)

    def _insert_block_windows(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        rows: list[sqlite3.Row],
        lo: int,
        hi: int,
    ) -> None:
        """写入与 [lo, hi] 相交的 2 句窗和 3 句窗；content 拼 index_clues。"""
        n = len(rows)
        payload: list[tuple[str, str, str, str, str, str]] = []
        for width in BLOCK_WIDTHS:
            last_start = n - width
            if last_start < 0:
                continue
            start = max(0, lo - (width - 1))
            stop = min(hi, last_start)
            for index in range(start, stop + 1):
                parts = [str(rows[index + offset]["content"]) for offset in range(width)]
                joined = " ".join(parts)
                clues = index_clues(joined)
                indexed = f"{joined} {clues}".strip() if clues else joined
                left_id = str(rows[index]["id"])
                right_id = str(rows[index + width - 1]["id"])
                mid_id = str(rows[index + 1]["id"]) if width == 3 else ""
                payload.append((indexed, user_id, session_id, left_id, mid_id, right_id))
        if not payload:
            return
        conn.executemany(
            """
            INSERT INTO message_blocks_fts (
                content, user_id, session_id, left_id, mid_id, right_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    def _rebuild_faiss(self) -> None:
        """从 message_vectors 按 user_id 重建 _faiss 里的 UserFaissIndex。"""
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
        """executemany INSERT message_vectors（id, user_id, dim, embedding BLOB）。"""
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        conn.executemany(
            """
            INSERT INTO message_vectors (id, user_id, dim, embedding)
            VALUES (?, ?, ?, ?)
            """,
            [
                (item_id, user_id, int(self._embedder.dim), vector.tobytes())
                for item_id, vector in zip(ids, matrix, strict=True)
            ],
        )

    def _index_vectors(self, user_id: str, ids: list[str], vectors: np.ndarray) -> None:
        """把向量写入该 user_id 的 UserFaissIndex。"""
        bucket = self._faiss.get(user_id)
        if bucket is None:
            bucket = UserFaissIndex(self._embedder.dim)
            self._faiss[user_id] = bucket
        bucket.add(ids, vectors)

    def _require_conn(self) -> sqlite3.Connection:
        """返回已 open 的 sqlite3.Connection。"""
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn
