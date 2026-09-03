"""HashEmbedder、OpenAICompatibleEmbedder；build_embedder 按 model_name 选择。"""

from __future__ import annotations

import logging
import re
import zlib
from typing import Protocol

import httpx
import numpy as np

logger = logging.getLogger("verbatim_mem")

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_HASH_DIM = 384
_MAX_BATCH = 10


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

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """多句走 _encode。"""
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """单句走 _encode。"""
        return self._encode([text])

    def warmup(self) -> None:
        """encode_docs(["ok"]) 探活；缺 EMBEDDING_API_KEY 则失败。"""
        if not self._api_key:
            raise RuntimeError("EMBEDDING_API_KEY is not set")
        self.encode_docs(["ok"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        """按 _MAX_BATCH 切块调用 _encode_batch 再 vstack。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [
            self._encode_batch(texts[offset : offset + _MAX_BATCH])
            for offset in range(0, len(texts), _MAX_BATCH)
        ]
        return np.vstack(batches)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """POST {base_url}/embeddings，校验 dim 后 L2 归一化。"""
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
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            logger.info("embedding_http_error status=failed")
            raise RuntimeError("embedding request failed") from exc
        items = sorted(body["data"], key=lambda item: int(item["index"]))
        matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if matrix.shape[1] != self.dim:
            raise RuntimeError(
                f"embedding dim {matrix.shape[1]} != EMBEDDING_DIM {self.dim}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        matrix /= norms
        return np.ascontiguousarray(matrix, dtype=np.float32)


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
