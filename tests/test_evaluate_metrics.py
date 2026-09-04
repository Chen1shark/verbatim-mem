"""neighbor_delta：window>0 相对 window=0 的 gold 差。"""

from bench.evaluate_retrieval import neighbor_delta


def test_neighbor_delta_counts_gained_and_lost_gold() -> None:
    """with 多出 gold 记 neighbor_gold_gained；without 独有 gold 记 neighbor_gold_lost。"""
    without = [
        {
            "evidence": ["gold-a", "gold-b"],
            "hits": ["gold-a", "noise"],
        }
    ]
    with_rows = [
        {
            "evidence": ["gold-a", "gold-b"],
            "hits": ["gold-a", "gold-b", "neighbor"],
        }
    ]
    delta = neighbor_delta(with_rows, without)
    assert delta["neighbor_gold_gained"] == 1
    assert delta["neighbor_gold_lost"] == 0
    assert delta["neighbor_extra_hits_mean"] == 2.0
