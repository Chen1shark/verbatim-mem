"""可选 CrossEncoder；Settings.memory_rerank_model 为空则 build_reranker 返回 None。"""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(
        self, query: str, hits: list[dict[str, object]]
    ) -> list[dict[str, object]]: ...


class CrossEncoderReranker:
    """sentence_transformers.CrossEncoder；predict(query, content) 写回 hit['score']。"""

    def __init__(self, model_name: str) -> None:
        """加载 MEMORY_RERANK_MODEL；缺 sentence-transformers 则失败。"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for MEMORY_RERANK_MODEL"
            ) from exc
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, hits: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """对 hits 的 content 与 query 打分，按 score 降序。"""
        if not hits:
            return []
        pairs = [(query, str(hit["content"])) for hit in hits]
        scores = self._model.predict(pairs)
        ranked: list[dict[str, object]] = []
        for hit, score in zip(hits, scores, strict=True):
            item = dict(hit)
            item["score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return ranked


def build_reranker(model_name: str) -> Reranker | None:
    """model_name 非空则 CrossEncoderReranker，否则 None。"""
    name = (model_name or "").strip()
    if not name:
        return None
    return CrossEncoderReranker(name)
