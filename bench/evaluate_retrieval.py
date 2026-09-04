"""根据 replay 结果算 Evidence Recall@100、零证据、首条排名、条数、重复率、Search 延迟。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.replay_add_search import load_eval, make_client, replay, _remove_sqlite

SYNTHETIC = ROOT / "bench" / "fixtures" / "synthetic.json"


def _percentile(values: list[float], p: float) -> float:
    """values 的 p 分位；空列表为 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def evaluate_rows(rows: list[dict]) -> dict[str, float | int]:
    """对 replay 的 rows 汇总 Recall@100 与延迟。"""
    if not rows:
        return {
            "questions": 0,
            "recall_at_100": 0.0,
            "zero_hit_questions": 0,
            "mean_first_rank": 0.0,
            "mean_returned": 0.0,
            "duplicate_rate": 0.0,
            "search_mean_ms": 0.0,
            "search_p95_ms": 0.0,
            "neighbor_extra_hits_mean": 0.0,
            "neighbor_gold_gained": 0,
            "neighbor_gold_lost": 0,
        }
    recalls: list[float] = []
    first_ranks: list[float] = []
    zero = 0
    returned = 0
    dup_num = 0
    dup_den = 0
    latencies = [float(row["search_ms"]) for row in rows]
    for row in rows:
        gold = [str(item) for item in row["evidence"]]
        hits = [str(item) for item in row["hits"]]
        returned += len(hits)
        dup_den += len(hits)
        dup_num += len(hits) - len(set(hits))
        if not gold:
            recalls.append(1.0)
            continue
        found = sum(1 for item in gold if item in hits)
        recalls.append(found / len(gold))
        if found == 0:
            zero += 1
            continue
        ranks = [hits.index(item) + 1 for item in gold if item in hits]
        first_ranks.append(float(min(ranks)))
    return {
        "questions": len(rows),
        "recall_at_100": 100.0 * (sum(recalls) / len(recalls)),
        "zero_hit_questions": zero,
        "mean_first_rank": (
            statistics.fmean(first_ranks) if first_ranks else 0.0
        ),
        "mean_returned": returned / len(rows),
        "duplicate_rate": (dup_num / dup_den) if dup_den else 0.0,
        "search_mean_ms": statistics.fmean(latencies),
        "search_p95_ms": _percentile(latencies, 0.95),
        "neighbor_extra_hits_mean": 0.0,
        "neighbor_gold_gained": 0,
        "neighbor_gold_lost": 0,
    }


def format_report(version: str, metrics: dict[str, float | int]) -> str:
    """Version / Recall@100 / Zero-hit / Search P95 文本。"""
    return (
        f"Version: {version}\n"
        f"Recall@100: {metrics['recall_at_100']:.2f}%\n"
        f"Zero-hit questions: {metrics['zero_hit_questions']}\n"
        f"Mean first rank: {metrics['mean_first_rank']:.2f}\n"
        f"Mean returned: {metrics['mean_returned']:.2f}\n"
        f"Duplicate rate: {metrics['duplicate_rate']:.4f}\n"
        f"Search mean: {metrics['search_mean_ms']:.2f} ms\n"
        f"Search P95: {metrics['search_p95_ms']:.2f} ms\n"
        f"Neighbor extra hits mean: {metrics.get('neighbor_extra_hits_mean', 0.0):.2f}\n"
        f"Neighbor gold gained: {metrics.get('neighbor_gold_gained', 0)}\n"
        f"Neighbor gold lost: {metrics.get('neighbor_gold_lost', 0)}\n"
    )


def neighbor_delta(
    rows_with: list[dict], rows_without: list[dict]
) -> dict[str, float | int]:
    """window>0 相对 window=0：邻句多出的 hits、新增 gold、挤掉的 gold。"""
    if not rows_with or len(rows_with) != len(rows_without):
        return {
            "neighbor_extra_hits_mean": 0.0,
            "neighbor_gold_gained": 0,
            "neighbor_gold_lost": 0,
        }
    extra = 0
    gained = 0
    lost = 0
    for with_row, without_row in zip(rows_with, rows_without, strict=True):
        gold = {str(item) for item in with_row["evidence"]}
        with_hits = {str(item) for item in with_row["hits"]}
        without_hits = {str(item) for item in without_row["hits"]}
        extra += len(with_hits - without_hits)
        gained += len((gold & with_hits) - without_hits)
        lost += len((gold & without_hits) - with_hits)
    n = len(rows_with)
    return {
        "neighbor_extra_hits_mean": extra / n,
        "neighbor_gold_gained": gained,
        "neighbor_gold_lost": lost,
    }


def merge_neighbor_metrics(
    metrics: dict[str, float | int], delta: dict[str, float | int]
) -> dict[str, float | int]:
    """把 neighbor_delta 写进 evaluate_rows 的 metrics。"""
    merged = dict(metrics)
    merged.update(delta)
    return merged


def main(argv: list[str] | None = None) -> int:
    """load_eval → replay → evaluate_rows，写 --output。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SYNTHETIC)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "reports" / "latest.json",
    )
    parser.add_argument("--version", default="dev")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = load_eval(args.input)
    db_path = args.db or (args.output.parent / f"{args.output.stem}.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite(db_path)
    with make_client(db_path) as client:
        result = replay(client, payload)
    metrics = evaluate_rows(result["rows"])
    report = {
        "version": args.version,
        "input": str(args.input),
        "metrics": metrics,
        "text": format_report(args.version, metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report["text"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
