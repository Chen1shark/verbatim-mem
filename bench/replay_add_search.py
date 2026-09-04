"""对 eval JSON 执行 Add 后 Search，供 evaluate_retrieval 统计 Recall@100。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API_KEY = "bench-key"


def _remove_sqlite(path: Path) -> None:
    """删除 path 及同名 -wal / -shm。"""
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix) if suffix else path
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass


def load_eval(path: Path) -> dict[str, Any]:
    """读取 conversations / questions JSON。"""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def make_client(db_path: Path, **overrides: Any) -> TestClient:
    """TestClient(create_app(Settings))，embedding_model=hash。"""
    settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=str(db_path),
        embedding_model="hash",
        memory_retrieval_mode="hybrid",
        **overrides,
    )
    return TestClient(create_app(settings))


def replay(
    client: TestClient,
    payload: dict[str, Any],
    top_k: int = 100,
) -> dict[str, Any]:
    """先 Add conversations，再对 questions Search，记录 hits 与 search_ms。"""
    headers = {"X-Api-Key": API_KEY}
    for index, conv in enumerate(payload.get("conversations") or []):
        body = {
            "request_id": conv.get("request_id") or f"bench:conv:{index}",
            "user_id": conv["user_id"],
            "session_id": conv.get("session_id") or f"bench:sess:{index}",
            "messages": conv["messages"],
        }
        response = client.post("/add", json=body, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"add failed status={response.status_code}")
    rows: list[dict[str, Any]] = []
    for q_index, question in enumerate(payload.get("questions") or []):
        started = time.perf_counter()
        search_body = {
            "query": question["query"],
            "user_id": question["user_id"],
            "top_k": top_k,
        }
        if question.get("options"):
            search_body["options"] = question["options"]
        response = client.post("/search", json=search_body, headers=headers)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(f"search failed status={response.status_code}")
        data = response.json()["data"]
        rows.append(
            {
                "index": q_index,
                "query": question["query"],
                "evidence": list(question.get("evidence") or []),
                "hits": [item["content"] for item in data],
                "ids": [item["id"] for item in data],
                "search_ms": elapsed_ms,
            }
        )
    return {"rows": rows}
