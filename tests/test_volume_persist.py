"""同一 MEMORY_DB_PATH 在 create_app 重建后仍可 Search。"""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API_KEY = "test-key"
SAMPLE = "I adopted a cat named Luna."


def test_add_survives_create_app_reopen(db_path: str) -> None:
    """create_app 关闭后再开同一 memory_db_path，Search 仍命中 Add 原话。"""
    settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="hybrid",
    )
    payload = {
        "request_id": "eval:run:dataset:conv-0:chunk-0",
        "user_id": "eval:run:dataset:conv-0",
        "session_id": "eval:run:sample:0",
        "messages": [
            {"role": "user", "timestamp": 1704067200000, "content": SAMPLE}
        ],
    }
    headers = {"X-Api-Key": API_KEY}
    with TestClient(create_app(settings)) as first:
        assert first.post("/add", json=payload, headers=headers).status_code == 200
    with TestClient(create_app(settings)) as second:
        response = second.post(
            "/search",
            json={
                "query": "What is the name of my cat?",
                "user_id": "eval:run:dataset:conv-0",
                "top_k": 100,
            },
            headers=headers,
        )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"] == SAMPLE
