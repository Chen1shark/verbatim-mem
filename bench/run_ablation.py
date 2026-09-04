"""对 candidate_pool、neighbor_window、intent_temporal 跑同一份 eval JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluate_retrieval import (
    evaluate_rows,
    format_report,
    merge_neighbor_metrics,
    neighbor_delta,
)
from bench.replay_add_search import _remove_sqlite, load_eval, make_client, replay

SYNTHETIC = ROOT / "bench" / "fixtures" / "synthetic.json"

POOLS = (100, 300, 500)
WINDOWS = (0, 1, 2)


def _run(
    payload: dict[str, Any],
    db_path: Path,
    **overrides: Any,
) -> dict[str, Any]:
    """make_client(**overrides) 后 replay。"""
    if db_path.exists():
        _remove_sqlite(db_path)
    with make_client(db_path, **overrides) as client:
        return replay(client, payload)


def _pool_overrides(pool: int) -> dict[str, int]:
    """MEMORY_CANDIDATE_POOL 与分路池上限。"""
    return {
        "memory_candidate_pool": pool,
        "memory_fts_pool": min(300, pool),
        "memory_dense_pool": min(300, pool),
        "memory_block_pool": min(200, pool),
    }


def main(argv: list[str] | None = None) -> int:
    """pool / window / intent_temporal 消融，window 相对 0 写邻句 gold 差。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SYNTHETIC)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "reports" / "ablation-pool.json",
    )
    args = parser.parse_args(argv)
    payload = load_eval(args.input)
    runs: list[dict[str, object]] = []
    window_rows: dict[int, list[dict]] = {}

    for pool in POOLS:
        db_path = args.output.parent / f"ablation-pool-{pool}.db"
        result = _run(payload, db_path, **_pool_overrides(pool))
        metrics = evaluate_rows(result["rows"])
        runs.append(
            {
                "name": f"pool={pool}",
                "candidate_pool": pool,
                "metrics": metrics,
                "text": format_report(f"pool={pool}", metrics),
            }
        )
        print(runs[-1]["text"], end="")

    for window in WINDOWS:
        db_path = args.output.parent / f"ablation-window-{window}.db"
        result = _run(payload, db_path, memory_neighbor_window=window)
        window_rows[window] = result["rows"]
        metrics = evaluate_rows(result["rows"])
        if window > 0 and 0 in window_rows:
            metrics = merge_neighbor_metrics(
                metrics, neighbor_delta(result["rows"], window_rows[0])
            )
        runs.append(
            {
                "name": f"window={window}",
                "memory_neighbor_window": window,
                "metrics": metrics,
                "text": format_report(f"window={window}", metrics),
            }
        )
        print(runs[-1]["text"], end="")

    for enabled in (False, True):
        label = "on" if enabled else "off"
        db_path = args.output.parent / f"ablation-intent-{label}.db"
        result = _run(payload, db_path, memory_intent_temporal=enabled)
        metrics = evaluate_rows(result["rows"])
        runs.append(
            {
                "name": f"intent_temporal={label}",
                "memory_intent_temporal": enabled,
                "metrics": metrics,
                "text": format_report(f"intent_temporal={label}", metrics),
            }
        )
        print(runs[-1]["text"], end="")

    for pref, update in ((0.0, 0.0), (0.12, 0.08)):
        name = f"pref={pref}-update={update}"
        db_path = args.output.parent / f"ablation-{name}.db"
        result = _run(
            payload,
            db_path,
            memory_preference_weight=pref,
            memory_update_weight=update,
        )
        metrics = evaluate_rows(result["rows"])
        runs.append(
            {
                "name": name,
                "memory_preference_weight": pref,
                "memory_update_weight": update,
                "metrics": metrics,
                "text": format_report(name, metrics),
            }
        )
        print(runs[-1]["text"], end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"input": str(args.input), "runs": runs}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
