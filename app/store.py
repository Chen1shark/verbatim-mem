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

from app.embeddings import Embedder
from app.rerank import Reranker
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
_SYN_GROUPS = (
    ("name", "named", "called", "nickname", "names"),
    ("live", "lives", "lived", "living", "home", "house", "apartment"),
    ("move", "moved", "moving", "relocate", "relocated"),
    ("work", "works", "worked", "working", "job", "jobs"),
    ("like", "likes", "liked", "love", "loves", "prefer", "preferred", "favorite", "favourite"),
    ("friend", "friends"),
    ("wife", "husband", "spouse", "married"),
    ("child", "children", "kid", "kids", "son", "daughter"),
    ("pet", "pets"),
    ("school", "college", "university"),
    ("born", "birthday", "birth"),
    ("eat", "ate", "eats", "eating", "food", "meal", "meals"),
    ("brother", "sister", "sibling", "siblings"),
    ("car", "cars", "drive", "drove", "driving"),
    ("play", "plays", "played", "playing", "game", "games"),
    ("travel", "trip", "trips", "flew", "visit", "visited"),
    ("hometown", "hometowns"),
)
_SYN: dict[str, tuple[str, ...]] = {}
for _group in _SYN_GROUPS:
    for _token in _group:
        _SYN[_token] = tuple(item for item in _group if item != _token)


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


def _expand_synonyms(tokens: list[str]) -> list[str]:
    """按 _SYN 展开，不含原词。"""
    extra: list[str] = []
    seen = set(tokens)
    for token in tokens:
        for other in _SYN.get(token, ()):
            if other in seen:
                continue
            seen.add(other)
            extra.append(other)
    return extra


def index_clues(text: str) -> str:
    """messages.clues：_SYN 同义词 + _light_stems，写入 messages_fts.clues。"""
    english = [token for token in _english_tokens(text) if token not in _FTS_STOPWORDS]
    bag = _dedupe(english + _expand_synonyms(english))
    extra: list[str] = list(_expand_synonyms(english))
    for token in bag:
        extra.extend(_light_stems(token))
    return " ".join(_dedupe(extra))


def fts_match_query(raw: str) -> str | None:
    """问句收成 FTS5 MATCH：英文词+同义词+_light_stems，"token" OR token*。"""
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

    query_terms = _dedupe(use + _expand_synonyms(use))
    stemmed: list[str] = []
    for token in query_terms:
        stemmed.extend(_light_stems(token))
    for token in query_terms + _dedupe(stemmed):
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
RECALL_POOL_MIN = 30
TIME_WEIGHT = 0.04
TIME_WEIGHT_TEMPORAL = 0.08
LEXICAL_WEIGHT = 0.25
NUMERIC_WEIGHT = 0.2
ENTITY_WEIGHT = 0.22
OPTION_WEIGHT = 0.18
PREF_WEIGHT = 0.12
UPDATE_WEIGHT = 0.08
NEIGHBOR_WINDOW = 2
NEIGHBOR_SCORE_DELTA = 0.001
FTS_CLAUSE_CAP = 48
BLOCK_WIDTHS = (2, 3)
VECTOR_META_MODEL = "embedding_identity"
_TEMPORAL_NEW_EN = frozenset(
    {
        "now",
        "currently",
        "already",
        "latest",
        "recently",
        "anymore",
        "nowadays",
        "lately",
        "still",
    }
)
_TEMPORAL_OLD_EN = frozenset(
    {
        "previous",
        "previously",
        "before",
        "earlier",
        "formerly",
    }
)
_TEMPORAL_NEW_PHRASES = (
    "right now",
    "these days",
    "as of now",
)
_TEMPORAL_OLD_PHRASES = (
    "used to",
    "no longer",
    "back then",
    "at the time",
)
_UPDATE_CUES = (
    "actually",
    "instead",
    "i mean",
    "turns out",
    "never mind",
    "wait no",
    "i was wrong",
    "to be clear",
)
_PREF_Q = frozenset(
    {
        "like",
        "likes",
        "liked",
        "love",
        "loves",
        "prefer",
        "preferred",
        "favorite",
        "favourite",
        "enjoy",
        "enjoys",
        "enjoyed",
        "hobby",
        "hobbies",
        "hate",
        "hates",
    }
)
_PREF_CUES = (
    "i like",
    "i love",
    "i prefer",
    "i enjoy",
    "i hate",
    "i always",
    "my favorite",
    "my favourite",
    "i don't like",
    "i do not like",
    "i'm a fan",
    "i am a fan",
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


def _pool_k(top_k: int) -> int:
    """Search 召回池：min(100, max(top_k * 2, RECALL_POOL_MIN))，供 FTS LIMIT / FAISS take / RRF。"""
    return min(100, max(top_k * 2, RECALL_POOL_MIN))


def _query_text(query: str, options: list[str] | None) -> str:
    """query 拼上 SearchRequest.options，供 fts_match_query / encode_query。"""
    if not options:
        return query
    extra = " ".join(item for item in options if item)
    if not extra.strip():
        return query
    return f"{query} {extra}"


def _meaningful_tokens(raw: str) -> list[str]:
    """英文词去 _FTS_STOPWORDS；若全是停用词则退回原 token。"""
    tokens = _english_tokens(raw)
    keep = [token for token in tokens if token not in _FTS_STOPWORDS]
    return keep or tokens


def _lexical_token_set(raw: str) -> set[str]:
    """_meaningful_tokens + _expand_synonyms + _light_stems。"""
    english = _meaningful_tokens(raw)
    tokens = set(english)
    tokens.update(_expand_synonyms(english))
    extra: list[str] = []
    for token in tokens:
        extra.extend(_light_stems(token))
    tokens.update(extra)
    return tokens


def _query_core(query: str, options: list[str] | None = None) -> list[str]:
    """问句实词保序，作覆盖目标；不含同义词膨胀。"""
    return _meaningful_tokens(_query_text(query, options))


def _term_variants(term: str) -> set[str]:
    """原词 + _light_stems + _SYN。"""
    variants = {term}
    variants.update(_light_stems(term))
    variants.update(_expand_synonyms([term]))
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


def _temporal_alpha(query: str) -> float:
    """now / _TEMPORAL_NEW_PHRASES → +TIME_WEIGHT_TEMPORAL；previously / _TEMPORAL_OLD_PHRASES → 负值；同时出现偏新；否则 TIME_WEIGHT。"""
    lowered = query.lower()
    tokens = {token.lower() for token in _FTS_TOKEN.findall(query)}
    old = bool(tokens & _TEMPORAL_OLD_EN) or any(
        phrase in lowered for phrase in _TEMPORAL_OLD_PHRASES
    )
    new = bool(tokens & _TEMPORAL_NEW_EN) or any(
        phrase in lowered for phrase in _TEMPORAL_NEW_PHRASES
    )
    if old and not new:
        return -TIME_WEIGHT_TEMPORAL
    if new:
        return TIME_WEIGHT_TEMPORAL
    return TIME_WEIGHT


def _hit_from_row(row: sqlite3.Row, score: float) -> dict[str, object]:
    """messages 行转内部 hit：id / content / score / created_at / timestamp / session_id。"""
    ts = int(row["timestamp"])
    return {
        "id": row["id"],
        "content": row["content"],
        "score": score,
        "created_at": ms_to_iso(ts),
        "timestamp": ts,
        "session_id": row["session_id"],
    }


def _public_hit(hit: dict[str, object]) -> dict[str, object]:
    """去掉内部 timestamp、session_id，只留 SearchItem 的 id / content / score / created_at。"""
    return {
        "id": hit["id"],
        "content": hit["content"],
        "score": hit["score"],
        "created_at": hit["created_at"],
    }


def _apply_recency(
    hits: list[dict[str, object]], query: str
) -> list[dict[str, object]]:
    """timestamp 在候选集内 min-max 为 rec；score 加 α * rec，不改写相关度。"""
    if not hits:
        return []
    alpha = _temporal_alpha(query)
    stamps = [int(hit["timestamp"]) for hit in hits]
    min_ts = min(stamps)
    max_ts = max(stamps)
    ts_span = max_ts - min_ts
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        rec = 0.0 if ts_span == 0 else (int(hit["timestamp"]) - min_ts) / ts_span
        item["score"] = float(hit["score"]) + alpha * rec
        ranked.append(item)
    if alpha < 0:
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                int(item["timestamp"]),
                str(item["id"]),
            )
        )
    else:
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["timestamp"]),
                str(item["id"]),
            )
        )
    return ranked


def _apply_lexical(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
) -> list[dict[str, object]]:
    """按问句实词覆盖率加 LEXICAL_WEIGHT，content 侧用 _term_variants 对齐。"""
    if not hits:
        return []
    core = _query_core(query, options)
    if not core:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        covered = _doc_query_cover(str(hit["content"]), set(core))
        item["score"] = float(hit["score"]) + LEXICAL_WEIGHT * (len(covered) / len(core))
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_numeric(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
) -> list[dict[str, object]]:
    """query+options 与 content 的 _NUM_RE 数字有交集则 score 加 NUMERIC_WEIGHT。"""
    if not hits:
        return []
    qnums = set(_NUM_RE.findall(_query_text(query, options)))
    if not qnums:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        cnums = set(_NUM_RE.findall(str(hit["content"])))
        if qnums & cnums:
            item["score"] = float(hit["score"]) + NUMERIC_WEIGHT
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_entity(
    hits: list[dict[str, object]],
    query: str,
    options: list[str] | None,
) -> list[dict[str, object]]:
    """query/options 的 _proper_nouns 与 content 的 _english_tokens 有交集则加 ENTITY_WEIGHT。"""
    if not hits:
        return []
    names = _proper_nouns(query, *(options or []))
    if not names:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        content_tokens = set(_english_tokens(str(hit["content"])))
        if names & content_tokens:
            item["score"] = float(hit["score"]) + ENTITY_WEIGHT
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_options(
    hits: list[dict[str, object]],
    options: list[str] | None,
) -> list[dict[str, object]]:
    """_option_needles 作为子串出现在 content 则加 OPTION_WEIGHT。"""
    if not hits:
        return []
    needles = _option_needles(options)
    if not needles:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        content = str(hit["content"]).lower()
        if any(needle in content for needle in needles):
            item["score"] = float(hit["score"]) + OPTION_WEIGHT
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_pref(
    hits: list[dict[str, object]],
    query: str,
) -> list[dict[str, object]]:
    """query 命中 _PREF_Q 且 content 含 _PREF_CUES 则加 PREF_WEIGHT。"""
    if not hits:
        return []
    qtokens = set(_english_tokens(query))
    if not (qtokens & _PREF_Q):
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        lowered = str(hit["content"]).lower()
        if any(cue in lowered for cue in _PREF_CUES):
            item["score"] = float(hit["score"]) + PREF_WEIGHT
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return ranked


def _apply_update(
    hits: list[dict[str, object]],
    query: str,
) -> list[dict[str, object]]:
    """问句不是偏旧时，content 含 _UPDATE_CUES 则加 UPDATE_WEIGHT。"""
    if not hits or _temporal_alpha(query) < 0:
        return hits
    ranked: list[dict[str, object]] = []
    for hit in hits:
        item = dict(hit)
        lowered = str(hit["content"]).lower()
        if any(cue in lowered for cue in _UPDATE_CUES):
            item["score"] = float(hit["score"]) + UPDATE_WEIGHT
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
) -> list[dict[str, object]]:
    """多路按 weight/(RRF_K+rank) 合并去重，截断为 top_k。"""
    if weights is None:
        weights = tuple(1.0 for _ in hit_lists)
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, object]] = {}
    for hits, weight in zip(hit_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            item_id = str(hit["id"])
            scores[item_id] = scores.get(item_id, 0.0) + weight / (RRF_K + rank)
            by_id.setdefault(item_id, hit)
    if not scores:
        return []
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
    clues TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(request_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_request_id ON messages(request_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id, timestamp);

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
        reranker: Reranker | None = None,
    ) -> None:
        """绑定 MEMORY_DB_PATH、Embedder、MEMORY_RETRIEVAL_MODE、可选 Reranker。"""
        self.db_path = db_path
        self._embedder = embedder
        self.retrieval_mode = retrieval_mode.strip().lower() or "hybrid"
        self._reranker = reranker
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
        """SCHEMA_SQL 后 _ensure_index_schema、_ensure_vector_meta、_backfill_vectors、_backfill_blocks、_rebuild_faiss。"""
        conn = self._require_conn()
        conn.executescript(SCHEMA_SQL)
        self._ensure_index_schema()
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
                        role, timestamp, content, clues, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            ids[index],
                            body.request_id,
                            body.user_id,
                            body.session_id,
                            message.role,
                            message.timestamp if message.timestamp is not None else 0,
                            message.content,
                            index_clues(message.content),
                            now,
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
        """该 user_id 下 FTS5 ∪ message_blocks_fts ∪ FAISS，加权 _rrf_merge 后字面/数字/专名/选项/人设/纠错加分、可选 Reranker、_apply_recency、_cover_reorder，再 _expand_neighbors(ranked[:top_k])。"""
        pool_k = _pool_k(top_k)
        search_text = _query_text(query, options)
        query_vec: np.ndarray | None = None
        if self.retrieval_mode in {"dense", "hybrid"}:
            query_vec = self._embedder.encode_query(search_text)
        with self._lock:
            fts_hits: list[dict[str, object]] = []
            block_hits: list[dict[str, object]] = []
            dense_hits: list[dict[str, object]] = []
            if self.retrieval_mode in {"fts", "hybrid"}:
                fts_hits = self._search_fts(user_id, search_text, pool_k)
                block_hits = self._search_blocks(user_id, search_text, pool_k)
            if self.retrieval_mode in {"dense", "hybrid"}:
                dense_hits = self._search_dense(user_id, query_vec, pool_k)
            if self.retrieval_mode == "fts":
                ranked = _rrf_merge(
                    [fts_hits, block_hits], pool_k, (RRF_W_FTS, RRF_W_BLOCKS)
                )
            elif self.retrieval_mode == "dense":
                ranked = dense_hits
            else:
                ranked = _rrf_merge(
                    [fts_hits, dense_hits, block_hits],
                    pool_k,
                    (RRF_W_FTS, RRF_W_DENSE, RRF_W_BLOCKS),
                )
            ranked = _apply_lexical(ranked, query, options)
            ranked = _apply_numeric(ranked, query, options)
            ranked = _apply_entity(ranked, query, options)
            ranked = _apply_options(ranked, options)
            ranked = _apply_pref(ranked, query)
            ranked = _apply_update(ranked, query)
            if self._reranker is not None:
                ranked = self._reranker.rerank(search_text, ranked)
            ranked = _apply_recency(ranked, query)
            ranked = _cover_reorder(ranked, query)
            return self._expand_neighbors(user_id, ranked[:top_k], top_k)

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
            SELECT id, content, timestamp, session_id
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

    def _expand_neighbors(
        self,
        user_id: str,
        hits: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """同一 user_id、session_id，ORDER BY timestamp, id 取命中 ±NEIGHBOR_WINDOW；超 top_k 先丢邻句，再 _public_hit。"""
        if not hits:
            return []
        hit_ids = {str(hit["id"]) for hit in hits}
        session_ids: list[str] = []
        seen_sessions: set[str] = set()
        for hit in hits:
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
                SELECT id, content, timestamp, session_id
                FROM messages
                WHERE user_id = ? AND session_id IN ({placeholders})
                ORDER BY session_id, timestamp, id
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

        assembled: list[dict[str, object]] = []
        seen: set[str] = set()
        for hit in hits:
            hit_id = str(hit["id"])
            session_id = str(hit.get("session_id") or "")
            ids = order_by_session.get(session_id, [])
            idx = index_of.get(hit_id)
            neighbor_ids: list[str] = []
            if idx is not None:
                for distance in range(1, NEIGHBOR_WINDOW + 1):
                    if idx - distance >= 0:
                        neighbor_ids.append(ids[idx - distance])
                    if idx + distance < len(ids):
                        neighbor_ids.append(ids[idx + distance])
            if hit_id not in seen:
                seen.add(hit_id)
                assembled.append(dict(hit))
            for item_id in neighbor_ids:
                if item_id in seen:
                    continue
                neighbor = dict(by_id[item_id])
                seen.add(item_id)
                neighbor["score"] = float(hit["score"]) - NEIGHBOR_SCORE_DELTA
                assembled.append(neighbor)

        while len(assembled) > top_k:
            drop_at = None
            for i in range(len(assembled) - 1, -1, -1):
                if str(assembled[i]["id"]) not in hit_ids:
                    drop_at = i
                    break
            if drop_at is None:
                assembled = assembled[:top_k]
                break
            assembled.pop(drop_at)
        return [_public_hit(item) for item in assembled]

    def _fts_has_column(self, table: str, column: str) -> bool:
        """SELECT column FROM table LIMIT 0 能执行则为 True。"""
        conn = self._require_conn()
        try:
            conn.execute(f"SELECT {column} FROM {table} LIMIT 0")
        except sqlite3.OperationalError:
            return False
        return True

    def _ensure_index_schema(self) -> None:
        """补 messages.clues、messages_fts.clues、message_blocks_fts.mid_id；clues 变了则重建 messages_fts。"""
        conn = self._require_conn()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "clues" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN clues TEXT NOT NULL DEFAULT ''")
        clues_changed = self._refresh_message_clues()
        if clues_changed or not self._fts_has_column("messages_fts", "clues"):
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
            ORDER BY timestamp, id
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
