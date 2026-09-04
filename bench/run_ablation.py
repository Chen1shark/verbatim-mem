"""对 candidate_pool=100/300/500 跑同一份 eval JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluate_retrieval import evaluate_rows, format_report
from bench.replay_add_search import load_eval, make_client, replay

SYNTHETIC = ROOT / "bench" / "fixtures" / "synthetic.json"
POOLS = (100, 300, 500)


def main(argv: list[str] | None = None) -> int:
    """每个 MEMORY_CANDIDATE_POOL 写一份 metrics。"""
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
    for pool in POOLS:
        db_path = args.output.parent / f"ablation-pool-{pool}.db"
        if db_path.exists():
            db_path.unlink()
        fts_pool = min(300, pool)
        dense_pool = min(300, pool)
        block_pool = min(200, pool)
        with make_client(
            db_path,
            memory_candidate_pool=pool,
            memory_fts_pool=fts_pool,
            memory_dense_pool=dense_pool,
            memory_block_pool=block_pool,
        ) as client:
            result = replay(client, payload)
        metrics = evaluate_rows(result["rows"])
        runs.append(
            {
                "candidate_pool": pool,
                "metrics": metrics,
                "text": format_report(f"pool={pool}", metrics),
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
