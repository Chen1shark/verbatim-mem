"""HashEmbedder、OpenAICompatibleEmbedder；build_embedder 按 model_name 选择。"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import zlib
from collections import OrderedDict
from typing import Protocol

import httpx
import numpy as np

logger = logging.getLogger("verbatim_mem")

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_HASH_DIM = 384
_MAX_BATCH = 10
_MAX_ATTEMPTS = 3
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_CACHE_CAP = 4096


def _supports_dimensions(model: str) -> bool:
    """model 名含 text-embedding 时 embeddings 请求带 dimensions。"""
    name = model.lower()
    return "text-embedding" in name


class Embedder(Protocol):
    dim: int
    identity: str

    def encode_docs(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class HashEmbedder:
    """token crc32 词袋，dim=_HASH_DIM。build_embedder("hash"|"test"|"off") 返回此类。"""

    dim = _HASH_DIM
    identity = "hash:384"

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """多句走 _hash_encode，dim=_HASH_DIM。"""
        return _hash_encode(texts, self.dim)

    def encode_query(self, text: str) -> np.ndarray:
        """单句走 _hash_encode。"""
        return _hash_encode([text], self.dim)

    def warmup(self) -> None:
        """HashEmbedder 无外部调用。"""
        return None

    def close(self) -> None:
        """HashEmbedder 无连接。"""
        return None


class OpenAICompatibleEmbedder:
    """POST {base_url}/embeddings，body 含 model / input / dimensions。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        dim: int,
    ) -> None:
        """绑定 Settings.embedding_model / embedding_api_key / embedding_base_url / embedding_dim。"""
        self.model = model
        self._api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.identity = f"{self.base_url}|{self.model}|{self.dim}"
        self._client = httpx.Client(timeout=60.0)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """多句走 _encode；命中 _cache 的句子不再 POST。"""
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """单句走 _encode。"""
        return self._encode([text])

    def warmup(self) -> None:
        """encode_docs(["ok"]) 探活；缺 EMBEDDING_API_KEY 则失败。"""
        if not self._api_key:
            raise RuntimeError("EMBEDDING_API_KEY is not set")
        self.encode_docs(["ok"])

    def close(self) -> None:
        """关闭 httpx.Client。"""
        self._client.close()

    def _encode(self, texts: list[str]) -> np.ndarray:
        """按 _text_key 查 _cache；未命中走 _encode_uncached，再 vstack。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [_text_key(text) for text in texts]
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        queued: set[str] = set()
        with self._cache_lock:
            for key, text in zip(keys, texts, strict=True):
                if key in self._cache or key in queued:
                    continue
                queued.add(key)
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            fresh = self._encode_uncached(missing_texts)
            with self._cache_lock:
                for key, vector in zip(missing_keys, fresh, strict=True):
                    if key in self._cache:
                        continue
                    self._cache_put(key, np.ascontiguousarray(vector, dtype=np.float32))
        with self._cache_lock:
            rows = [self._cache_touch(key) for key in keys]
            return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)

    def _cache_touch(self, key: str) -> np.ndarray:
        """返回 _cache[key] 并 move_to_end。须持有 _cache_lock。"""
        vector = self._cache[key]
        self._cache.move_to_end(key)
        return vector

    def _cache_put(self, key: str, vector: np.ndarray) -> None:
        """写入 _cache，超出 _CACHE_CAP 则 popitem(last=False)。须持有 _cache_lock。"""
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_CAP:
            self._cache.popitem(last=False)

    def _encode_uncached(self, texts: list[str]) -> np.ndarray:
        """按 _MAX_BATCH 切块调用 _encode_batch 再 vstack。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [
            self._encode_batch(texts[offset : offset + _MAX_BATCH])
            for offset in range(0, len(texts), _MAX_BATCH)
        ]
        return np.vstack(batches)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """POST {base_url}/embeddings；_RETRY_STATUSES 与传输错误最多 _MAX_ATTEMPTS 次，校验 dim 后 L2 归一化。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self._api_key:
            raise RuntimeError("EMBEDDING_API_KEY is not set")
        url = f"{self.base_url}/embeddings"
        payload: dict[str, object] = {"model": self.model, "input": texts}
        if _supports_dimensions(self.model):
            payload["dimensions"] = self.dim
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            response: httpx.Response | None = None
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                logger.info("embedding_http_error attempt=%s status=failed", attempt + 1)
                last_error = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay_s(attempt, None))
                    continue
                break
            if response.status_code in _RETRY_STATUSES:
                logger.info(
                    "embedding_retry attempt=%s status=%s",
                    attempt + 1,
                    response.status_code,
                )
                last_error = RuntimeError("embedding request failed")
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay_s(attempt, response))
                continue
            try:
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPError as exc:
                logger.info("embedding_http_error attempt=%s status=failed", attempt + 1)
                raise RuntimeError("embedding request failed") from exc
            try:
                items = sorted(body["data"], key=lambda item: int(item["index"]))
                matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise RuntimeError("embedding request failed") from exc
            if matrix.ndim != 2 or matrix.shape[0] != len(texts):
                raise RuntimeError("embedding batch size mismatch")
            if matrix.shape[1] != self.dim:
                raise RuntimeError(
                    f"embedding dim {matrix.shape[1]} != EMBEDDING_DIM {self.dim}"
                )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            matrix /= norms
            return np.ascontiguousarray(matrix, dtype=np.float32)
        raise RuntimeError("embedding request failed") from last_error


def build_embedder(
    model_name: str,
    api_key: str = "",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    dim: int = 2560,
) -> Embedder:
    """model_name 为 hash/test/off 返回 HashEmbedder，否则 OpenAICompatibleEmbedder。"""
    name = (model_name or "hash").strip()
    if name.lower() in {"hash", "test", "off"}:
        return HashEmbedder()
    return OpenAICompatibleEmbedder(
        model=name,
        api_key=api_key,
        base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dim=dim,
    )


def _text_key(text: str) -> str:
    """SHA-256(utf-8)，作 OpenAICompatibleEmbedder._cache 键。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _retry_delay_s(attempt: int, response: httpx.Response | None) -> float:
    """attempt 从 0 计；优先 Retry-After（封顶 5s），否则 0.5 * 2^attempt（封顶 2s）。"""
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), 5.0)
            except ValueError:
                pass
    return min(0.5 * (2**attempt), 2.0)


def _hash_encode(texts: list[str], dim: int) -> np.ndarray:
    """按 token crc32 累加词袋并 L2 归一化，给 HashEmbedder 用。"""
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN.findall(text.lower()):
            index = zlib.crc32(token.encode("utf-8")) % dim
            matrix[row, index] += 1.0
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix
