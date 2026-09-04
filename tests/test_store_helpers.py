"""_temporal_alpha、_span_if_contiguous、fts_match_query、index_clues、_cover_reorder、_rrf_merge、build_reranker。"""

from app.rerank import build_reranker
from app.store import (
    RRF_W_FTS,
    TIME_WEIGHT,
    TIME_WEIGHT_TEMPORAL,
    _cover_reorder,
    _light_stems,
    _rrf_merge,
    _span_if_contiguous,
    _temporal_alpha,
    fts_match_query,
    index_clues,
)


def test_temporal_alpha_previously_is_negative() -> None:
    """previously → -TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("What was my cat named previously?") == -TIME_WEIGHT_TEMPORAL


def test_temporal_alpha_now_is_positive() -> None:
    """now → TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("Where do I live now?") == TIME_WEIGHT_TEMPORAL


def test_temporal_alpha_default() -> None:
    """无时序线索 → TIME_WEIGHT。"""
    assert _temporal_alpha("What is the name of my cat?") == TIME_WEIGHT


def test_temporal_alpha_now_beats_previously() -> None:
    """同时有 previously 与 now 时偏新。"""
    assert (
        _temporal_alpha("I previously lived in A, where do I live now?")
        == TIME_WEIGHT_TEMPORAL
    )


def test_temporal_alpha_move_year_is_default() -> None:
    """问过去哪年搬家不是当前态线索。"""
    assert _temporal_alpha("Where did I move in 2021?") == TIME_WEIGHT


def test_temporal_alpha_used_to_is_negative() -> None:
    """used to → -TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("Where did I used to live?") == -TIME_WEIGHT_TEMPORAL


def test_build_reranker_blank_is_none() -> None:
    """MEMORY_RERANK_MODEL 空串 → None。"""
    assert build_reranker("") is None
    assert build_reranker("  ") is None


def test_span_if_contiguous() -> None:
    """连续下标返回 [lo, hi]；空或不连续返回 None。"""
    assert _span_if_contiguous([]) is None
    assert _span_if_contiguous([3]) == (3, 3)
    assert _span_if_contiguous([1, 2, 3]) == (1, 3)
    assert _span_if_contiguous([1, 3]) is None


def test_index_clues_expands_called() -> None:
    """called 写入 name / named 等到 index_clues。"""
    clues = set(index_clues("I have a cat called Luna.").split())
    assert "name" in clues
    assert "named" in clues


def test_fts_match_query_includes_synonym() -> None:
    """name 问句 MATCH 含 called。"""
    en = fts_match_query("What is the name of my cat?")
    assert en is not None
    assert '"called"' in en
    assert '"name"' in en


def test_light_stems_skips_named() -> None:
    """_light_stems：cats→cat；named 不切成 nam。"""
    assert _light_stems("cats") == ["cat"]
    assert _light_stems("named") == []
    assert _light_stems("working") == ["work"]


def test_fts_match_query_stems_cats() -> None:
    """cats 问句 MATCH 含 cat。"""
    match = fts_match_query("How many cats do I have?")
    assert match is not None
    assert '"cat"' in match


def test_index_clues_stems_cats() -> None:
    """cats 写入 cat 到 index_clues。"""
    clues = set(index_clues("I have two cats.").split())
    assert "cat" in clues


def test_rrf_merge_weights_prefer_fts() -> None:
    """_rrf_merge：同 rank 时 RRF_W_FTS 路排前。"""
    fts = [{"id": "fts", "content": "a", "score": 1.0}]
    dense = [{"id": "dense", "content": "b", "score": 1.0}]
    merged = _rrf_merge([fts, dense], 2, (RRF_W_FTS, 1.0))
    assert [item["id"] for item in merged] == ["fts", "dense"]


def test_cover_reorder_fills_uncovered_query_term() -> None:
    """_cover_reorder：问句第二实词只出现在低分句时，该句仍进入前三。"""
    hits = [
        {"id": "a", "content": "The river flooded the valley.", "score": 1.0},
        {"id": "b", "content": "The river rose overnight.", "score": 0.95},
        {"id": "c", "content": "The river broke the levee.", "score": 0.9},
        {"id": "d", "content": "The concert started at dusk.", "score": 0.4},
    ]
    ordered = [str(item["id"]) for item in _cover_reorder(hits, "river concert")]
    assert ordered[0] == "a"
    assert "d" in ordered[:3]
