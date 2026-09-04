"""把 AML_PUBLIC_DATA 或 --input 写成 bench/data/eval.json；没有公开数据则提示用 synthetic。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "bench" / "data" / "eval.json"
SYNTHETIC = ROOT / "bench" / "fixtures" / "synthetic.json"


def _looks_like_eval(payload: object) -> bool:
    """要求 conversations 与 questions 两个列表。"""
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("conversations"), list) and isinstance(
        payload.get("questions"), list
    )


def main(argv: list[str] | None = None) -> int:
    """读取 --input / AML_PUBLIC_DATA，写入 --output。"""
    parser = argparse.ArgumentParser(description="Prepare retrieval eval JSON.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="copy bench/fixtures/synthetic.json",
    )
    args = parser.parse_args(argv)
    src = args.input
    if src is None and not args.synthetic:
        env_path = os.environ.get("AML_PUBLIC_DATA", "").strip()
        if env_path:
            src = Path(env_path)
    if args.synthetic or src is None:
        src = SYNTHETIC
        if not args.synthetic and not os.environ.get("AML_PUBLIC_DATA"):
            print(
                "no public dataset; copying bench/fixtures/synthetic.json. "
                "set AML_PUBLIC_DATA or --input for the official split.",
                file=sys.stderr,
            )
    if not src.is_file():
        print(f"missing input: {src}", file=sys.stderr)
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not _looks_like_eval(payload):
        print(
            "input must be JSON with conversations[] and questions[]",
            file=sys.stderr,
        )
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
