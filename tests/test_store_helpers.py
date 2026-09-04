"""_temporal_alpha、_span_if_contiguous、fts_match_query、index_clues、_cover_reorder、_rrf_merge。"""

from app.store import (
    RRF_W_FTS,
    TIME_WEIGHT,
    TIME_WEIGHT_TEMPORAL,
    _cover_reorder,
    _light_stems,
    _rrf_merge,
    _search_texts,
    _span_if_contiguous,
    _temporal_alpha,
    fts_match_query,
    index_clues,
)


def test_temporal_alpha_previously_is_negative() -> None:
    """previously → -TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("Which cabinet was used previously?") == -TIME_WEIGHT_TEMPORAL


def test_temporal_alpha_now_is_positive() -> None:
    """now → TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("Which cabinet is used now?") == TIME_WEIGHT_TEMPORAL


def test_temporal_alpha_default() -> None:
    """无时序线索 → TIME_WEIGHT。"""
    assert _temporal_alpha("Which gasket SKU is stocked?") == TIME_WEIGHT
    assert TIME_WEIGHT == 0.0


def test_temporal_alpha_now_beats_previously() -> None:
    """同时有 previously 与 now 时偏新。"""
    assert (
        _temporal_alpha("Cabinet K7 was used previously, which cabinet is used now?")
        == TIME_WEIGHT_TEMPORAL
    )


def test_temporal_alpha_move_year_is_default() -> None:
    """问句含年份不是当前态线索。"""
    assert _temporal_alpha("Which lot passed inspection in 2021?") == TIME_WEIGHT


def test_temporal_alpha_used_to_is_negative() -> None:
    """used to → -TIME_WEIGHT_TEMPORAL。"""
    assert _temporal_alpha("Which cabinet used to hold the wrench?") == -TIME_WEIGHT_TEMPORAL


def test_span_if_contiguous() -> None:
    """连续下标返回 [lo, hi]；空或不连续返回 None。"""
    assert _span_if_contiguous([]) is None
    assert _span_if_contiguous([3]) == (3, 3)
    assert _span_if_contiguous([1, 2, 3]) == (1, 3)
    assert _span_if_contiguous([1, 3]) is None


def test_index_clues_stems_working() -> None:
    """working 写入 work 到 index_clues。"""
    clues = set(index_clues("The widgets were working.").split())
    assert "work" in clues


def test_fts_match_query_stems_working() -> None:
    """working 问句 MATCH 含 work。"""
    en = fts_match_query("Are the widgets working?")
    assert en is not None
    assert '"working"' in en
    assert '"work"' in en


def test_light_stems_skips_named() -> None:
    """_light_stems：cats→cat；named 不切成 nam。"""
    assert _light_stems("cats") == ["cat"]
    assert _light_stems("named") == []
    assert _light_stems("working") == ["work"]
    assert _light_stems("widgets") == ["widget"]


def test_fts_match_query_stems_widgets() -> None:
    """widgets 问句 MATCH 含 widget。"""
    match = fts_match_query("How many widgets are stocked?")
    assert match is not None
    assert '"widget"' in match


def test_index_clues_stems_widgets() -> None:
    """widgets 写入 widget 到 index_clues。"""
    clues = set(index_clues("Two widgets failed inspection.").split())
    assert "widget" in clues


def test_rrf_merge_weights_prefer_fts() -> None:
    """_rrf_merge：同 rank 时 RRF_W_FTS 路排前。"""
    fts = [{"id": "fts", "content": "a", "score": 1.0}]
    dense = [{"id": "dense", "content": "b", "score": 1.0}]
    merged = _rrf_merge([fts, dense], 2, (RRF_W_FTS, 1.0))
    assert [item["id"] for item in merged] == ["fts", "dense"]


def test_search_texts_splits_options() -> None:
    """_search_texts：无 options 只返回 query；有 options 则每项去前缀后单路。"""
    query = "Which gasket fits the manifold?"
    assert _search_texts(query, None) == [query]
    assert _search_texts(query, ["A. Viton", "B. Nitrile"]) == [
        query,
        f"{query} Viton",
        f"{query} Nitrile",
    ]


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
