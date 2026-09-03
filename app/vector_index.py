"""按 user_id 分桶的 FAISS IndexFlatIP。隔离发生在选桶，不在召回后再筛。"""

from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required") from exc


class UserFaissIndex:
    """一个 user_id 桶：IndexFlatIP + ids 列表。"""

    def __init__(self, dim: int) -> None:
        """创建 IndexFlatIP(dim)。"""
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """L2 归一化后写入 IndexFlatIP，ids 与向量按下标对齐。"""
        vecs = _as_matrix(vectors, self.dim)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """IndexFlatIP 内积近邻，返回 (messages.id, score)。"""
        total = self.index.ntotal
        if total == 0 or k <= 0:
            return []
        take = min(k, total)
        q = _as_matrix(query, self.dim)
        faiss.normalize_L2(q)
        scores, indexes = self.index.search(q, take)
        hits: list[tuple[str, float]] = []
        for score, pos in zip(scores[0], indexes[0], strict=True):
            if pos < 0:
                continue
            hits.append((self.ids[int(pos)], float(score)))
        return hits


def _as_matrix(vectors: np.ndarray, dim: int) -> np.ndarray:
    """收成 float32 二维 (n, dim)，供 IndexFlatIP.add / search。"""
    vecs = np.ascontiguousarray(vectors, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    if vecs.shape[1] != dim:
        raise ValueError(f"vector dim {vecs.shape[1]} != {dim}")
    return vecs
