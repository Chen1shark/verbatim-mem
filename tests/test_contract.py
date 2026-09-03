"""GET /health、POST /add、POST /search；库表 messages / messages_fts / message_vectors。"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

SAMPLE_CONTENT = "I adopted a cat named Luna."
API_KEY = "test-key"


def _payload(**overrides):
    """AddRequest 字段。"""
    body = {
        "request_id": "eval:run:dataset:conv-0:chunk-0",
        "user_id": "eval:run:dataset:conv-0",
        "session_id": "eval:run:sample:0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": SAMPLE_CONTENT,
            }
        ],
    }
    body.update(overrides)
    return body


def _auth(header: str = "X-Api-Key") -> dict[str, str]:
    """require_api_key 认的 X-Api-Key / Bearer / Token，值为 API_KEY。"""
    if header == "X-Api-Key":
        return {"X-Api-Key": API_KEY}
    if header == "Bearer":
        return {"Authorization": f"Bearer {API_KEY}"}
    return {"Authorization": f"Token {API_KEY}"}


def _search_body(**overrides):
    body = {
        "query": "What is the name of my cat?",
        "user_id": "eval:run:dataset:conv-0",
        "top_k": 100,
    }
    body.update(overrides)
    return body


# --- health / 鉴权 ---


def test_health_does_not_need_key(client: TestClient) -> None:
    """GET /health 不经过 require_api_key。"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_add_without_key_is_401(client: TestClient) -> None:
    """POST /add 无头 → 401。"""
    response = client.post("/add", json=_payload())
    assert response.status_code == 401


def test_add_wrong_key_is_401(client: TestClient) -> None:
    """X-Api-Key 对不上 memory_api_key → 401。"""
    response = client.post(
        "/add",
        json=_payload(),
        headers={"X-Api-Key": "nope"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("header", ["X-Api-Key", "Bearer", "Token"])
def test_add_accepts_three_auth_headers(client: TestClient, header: str) -> None:
    """_extract_token：X-Api-Key、Bearer、Token。"""
    response = client.post("/add", json=_payload(), headers=_auth(header))
    assert response.status_code == 200


# --- Add：原话、幂等、409 ---


def test_add_persists_verbatim(client: TestClient, db_path: str) -> None:
    """AddResponse 回 ID；messages.content、messages_fts MATCH 'Luna' 均为 SAMPLE_CONTENT。"""
    response = client.post("/add", json=_payload(), headers=_auth())
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "eval:run:dataset:conv-0:chunk-0",
        "user_id": "eval:run:dataset:conv-0",
        "session_id": "eval:run:sample:0",
    }
    assert SAMPLE_CONTENT not in response.text

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT content FROM messages").fetchone()
        assert row is not None
        assert row[0] == SAMPLE_CONTENT
        fts = conn.execute(
            "SELECT content FROM messages_fts WHERE messages_fts MATCH 'Luna'"
        ).fetchone()
        assert fts is not None
        assert fts[0] == SAMPLE_CONTENT


def test_add_is_idempotent(client: TestClient, db_path: str) -> None:
    """同 request_id + 同 fingerprint：两次 200，messages / requests 各一行。"""
    body = _payload()
    first = client.post("/add", json=body, headers=_auth())
    second = client.post("/add", json=body, headers=_auth())
    assert first.status_code == 200
    assert second.status_code == 200
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE request_id = ?",
            (body["request_id"],),
        ).fetchone()[0]
        requests = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    assert count == 1
    assert requests == 1


def test_add_conflict_on_different_content(client: TestClient, db_path: str) -> None:
    """同 request_id、不同 fingerprint → 409；detail 不含 content。"""
    client.post("/add", json=_payload(), headers=_auth())
    conflict = client.post(
        "/add",
        json=_payload(
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067200000,
                    "content": "I adopted a dog named Mars.",
                }
            ]
        ),
        headers=_auth(),
    )
    assert conflict.status_code == 409
    assert SAMPLE_CONTENT not in conflict.text
    assert "Mars" not in conflict.text
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_add_conflict_on_different_user_id(client: TestClient) -> None:
    """fingerprint 含 user_id，改 user_id 仍是 409。"""
    client.post("/add", json=_payload(), headers=_auth())
    conflict = client.post(
        "/add",
        json=_payload(user_id="someone-else"),
        headers=_auth(),
    )
    assert conflict.status_code == 409


def test_add_multiple_messages(client: TestClient, db_path: str) -> None:
    """一条 requests，多行 messages；id = request_id:下标。"""
    body = _payload(
        messages=[
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": SAMPLE_CONTENT,
            },
            {
                "role": "assistant",
                "timestamp": 1704067201000,
                "content": "Luna is a lovely name.",
            },
        ]
    )
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 200
    with sqlite3.connect(db_path) as conn:
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        requests = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        ids = [
            row[0]
            for row in conn.execute("SELECT id FROM messages ORDER BY timestamp")
        ]
    assert messages == 2
    assert requests == 1
    assert ids == [
        "eval:run:dataset:conv-0:chunk-0:0",
        "eval:run:dataset:conv-0:chunk-0:1",
    ]


def test_add_missing_field_is_422(client: TestClient) -> None:
    """缺 request_id → 422；body 不含 content。"""
    body = _payload()
    del body["request_id"]
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 422
    assert SAMPLE_CONTENT not in response.text


def test_add_empty_messages_is_422(client: TestClient) -> None:
    """AddRequest.messages min_length=1。"""
    response = client.post(
        "/add",
        json=_payload(messages=[]),
        headers=_auth(),
    )
    assert response.status_code == 422


def test_add_ignores_unknown_fields(client: TestClient) -> None:
    """AddRequest extra=ignore。"""
    body = _payload()
    body["unexpected"] = "field"
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 200


def test_add_alias_path(client: TestClient) -> None:
    """同一 handler 挂 /v1/memory/add。"""
    response = client.post("/v1/memory/add", json=_payload(), headers=_auth())
    assert response.status_code == 200


def test_logs_omit_secret_and_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """caplog 不含 SAMPLE_CONTENT、API_KEY。"""
    caplog.set_level("INFO")
    client.post("/add", json=_payload(), headers=_auth())
    text = caplog.text
    assert SAMPLE_CONTENT not in text
    assert API_KEY not in text


def test_openapi_declares_api_key(client: TestClient) -> None:
    """securitySchemes.ApiKey = X-Api-Key；/add /search 带 security，/health 不带。"""
    spec = client.get("/openapi.json").json()
    schemes = spec["components"]["securitySchemes"]
    assert schemes["ApiKey"]["name"] == "X-Api-Key"
    assert schemes["ApiKey"]["type"] == "apiKey"
    add_security = spec["paths"]["/add"]["post"].get("security", [])
    search_security = spec["paths"]["/search"]["post"].get("security", [])
    assert {"ApiKey": []} in add_security
    assert {"ApiKey": []} in search_security
    assert "security" not in spec["paths"]["/health"]["get"]


# --- Search：原话、隔离、top_k ---


def test_search_without_key_is_401(client: TestClient) -> None:
    """POST /search 无头 → 401。"""
    response = client.post("/search", json=_search_body())
    assert response.status_code == 401


def test_search_returns_verbatim_immediately(client: TestClient) -> None:
    """SearchItem.content 等于入库原话，不是模型生成的答案。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post("/search", json=_search_body(), headers=_auth())
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) >= 1
    assert data[0]["content"] == SAMPLE_CONTENT
    assert data[0]["id"] == "eval:run:dataset:conv-0:chunk-0:0"
    assert data[0]["created_at"] == "2024-01-01T00:00:00Z"
    assert isinstance(data[0]["score"], float)
    assert "Luna" in data[0]["content"]
    assert data[0]["content"] != "Luna"


def test_search_keyword_query_hits_named(client: TestClient) -> None:
    """query='cat name' MATCH 到 messages 里的 named。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(query="cat name"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert SAMPLE_CONTENT in contents


def test_search_isolates_users(client: TestClient) -> None:
    """Search SQL 带 user_id；两人各只能拿到自己的 messages。"""
    client.post("/add", json=_payload(), headers=_auth())
    client.post(
        "/add",
        json=_payload(
            request_id="other:chunk-0",
            user_id="someone-else",
            session_id="other-session",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067200000,
                    "content": "I adopted a cat named Mars.",
                }
            ],
        ),
        headers=_auth(),
    )
    mine = client.post("/search", json=_search_body(query="cat"), headers=_auth())
    theirs = client.post(
        "/search",
        json=_search_body(query="cat", user_id="someone-else"),
        headers=_auth(),
    )
    assert [item["content"] for item in mine.json()["data"]] == [SAMPLE_CONTENT]
    assert [item["content"] for item in theirs.json()["data"]] == [
        "I adopted a cat named Mars."
    ]


def test_search_unknown_user_is_empty(client: TestClient) -> None:
    """库中无该 user_id → 200 {"data": []}。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(user_id="no-such-user"),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_search_respects_top_k(client: TestClient) -> None:
    """SearchRequest.top_k 截断 data 长度。"""
    body = _payload(
        messages=[
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": SAMPLE_CONTENT,
            },
            {
                "role": "assistant",
                "timestamp": 1704067201000,
                "content": "Luna is a lovely name.",
            },
            {
                "role": "user",
                "timestamp": 1704067202000,
                "content": "Luna likes tuna.",
            },
        ]
    )
    client.post("/add", json=body, headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(query="Luna", top_k=2),
        headers=_auth(),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 2


def test_search_missing_field_is_422(client: TestClient) -> None:
    """缺 SearchRequest.user_id → 422。"""
    response = client.post(
        "/search",
        json={"query": "cat", "top_k": 10},
        headers=_auth(),
    )
    assert response.status_code == 422


def test_search_ignores_unknown_fields(client: TestClient) -> None:
    """SearchRequest extra=ignore；options 当前不参与召回。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(options=["A. Luna", "B. Mars"], unexpected="field"),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"] == SAMPLE_CONTENT


def test_search_punctuation_query_does_not_500(client: TestClient) -> None:
    """query 含引号、OR 时 MATCH 失败仍 200 + data。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(query='" OR Luna --'),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert "data" in response.json()


def test_search_alias_path(client: TestClient) -> None:
    """同一 handler 挂 /v1/memory/search。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/v1/memory/search",
        json=_search_body(),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"] == SAMPLE_CONTENT


def test_search_logs_omit_query_and_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """caplog 不含 query、SAMPLE_CONTENT、API_KEY。"""
    caplog.set_level("INFO")
    client.post("/add", json=_payload(), headers=_auth())
    client.post("/search", json=_search_body(), headers=_auth())
    text = caplog.text
    assert SAMPLE_CONTENT not in text
    assert "What is the name of my cat?" not in text
    assert API_KEY not in text


# --- Search：_apply_recency / _expand_neighbors ---

OLD_CAT = "My cat is named Luna."
NEW_CAT = "My cat is named Mars."
NEIGHBOR_PREV = "I adopted a cat."
NEIGHBOR_HIT = "Her name is Luna."
NEIGHBOR_NEXT = "She likes tuna."
UNRELATED_NEW = "Today the weather is sunny and warm."
OTHER_SESSION_DECOY = "The weather is sunny."


def test_search_prefers_newer_statement(client: TestClient, db_path: str) -> None:
    """_apply_recency：新旧两句都在 data 里时新句更前；messages 仍两行。"""
    client.post("/add", json=_payload(messages=[
        {"role": "user", "timestamp": 1704067200000, "content": OLD_CAT},
    ]), headers=_auth())
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[
                {"role": "user", "timestamp": 1704153600000, "content": NEW_CAT},
            ],
        ),
        headers=_auth(),
    )
    response = client.post("/search", json=_search_body(), headers=_auth())
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert NEW_CAT in contents
    assert OLD_CAT in contents
    assert contents.index(NEW_CAT) < contents.index(OLD_CAT)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_search_recency_does_not_outrank_unrelated(client: TestClient) -> None:
    """_apply_recency 相关度为主：SAMPLE_CONTENT 排在无关新句前。"""
    client.post("/add", json=_payload(), headers=_auth())
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            session_id="other-session",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704153600000,
                    "content": UNRELATED_NEW,
                }
            ],
        ),
        headers=_auth(),
    )
    response = client.post("/search", json=_search_body(), headers=_auth())
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert SAMPLE_CONTENT in contents
    assert contents[0] == SAMPLE_CONTENT
    if UNRELATED_NEW in contents:
        assert contents.index(SAMPLE_CONTENT) < contents.index(UNRELATED_NEW)


def test_search_includes_session_neighbors(client: TestClient) -> None:
    """_expand_neighbors：命中 ±1 且同 session。"""
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": NEIGHBOR_PREV},
                {"role": "assistant", "timestamp": 1704067201000, "content": NEIGHBOR_HIT},
                {"role": "user", "timestamp": 1704067202000, "content": NEIGHBOR_NEXT},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Luna"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert NEIGHBOR_HIT in contents
    assert NEIGHBOR_PREV in contents
    assert NEIGHBOR_NEXT in contents


def test_search_neighbors_skip_other_session(fts_client: TestClient) -> None:
    """_expand_neighbors 的 SQL 带 session_id，不跨会话拼接。"""
    fts_client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": NEIGHBOR_PREV},
                {"role": "assistant", "timestamp": 1704067201000, "content": NEIGHBOR_HIT},
                {"role": "user", "timestamp": 1704067202000, "content": NEIGHBOR_NEXT},
            ]
        ),
        headers=_auth(),
    )
    fts_client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:other",
            session_id="other-session",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067201500,
                    "content": OTHER_SESSION_DECOY,
                }
            ],
        ),
        headers=_auth(),
    )
    response = fts_client.post(
        "/search",
        json=_search_body(query="Luna"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert NEIGHBOR_HIT in contents
    assert NEIGHBOR_PREV in contents
    assert NEIGHBOR_NEXT in contents
    assert OTHER_SESSION_DECOY not in contents


def test_search_top_k_one_keeps_hit_not_neighbor(client: TestClient) -> None:
    """top_k=1 时邻句让位，只留命中句。"""
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": NEIGHBOR_PREV},
                {"role": "assistant", "timestamp": 1704067201000, "content": NEIGHBOR_HIT},
                {"role": "user", "timestamp": 1704067202000, "content": NEIGHBOR_NEXT},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Luna", top_k=1),
        headers=_auth(),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["content"] == NEIGHBOR_HIT


# --- message_vectors / FAISS（HashEmbedder.dim=384）---


def test_add_writes_vectors(client: TestClient, db_path: str) -> None:
    """message_vectors.dim / embedding 长度对应 HashEmbedder；vector_meta.embedding_identity=hash:384。"""
    response = client.post("/add", json=_payload(), headers=_auth())
    assert response.status_code == 200
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT dim, length(embedding) FROM message_vectors"
        ).fetchone()
        identity = conn.execute(
            "SELECT value FROM vector_meta WHERE key = 'embedding_identity'"
        ).fetchone()
    assert row is not None
    assert row[0] == 384
    assert row[1] == 384 * 4
    assert identity[0] == "hash:384"


def test_dense_search_hits_verbatim(dense_client: TestClient) -> None:
    """retrieval_mode=dense：SearchItem.content 仍为 SAMPLE_CONTENT。"""
    dense_client.post("/add", json=_payload(), headers=_auth())
    response = dense_client.post(
        "/search",
        json=_search_body(query="Luna"),
        headers=_auth(),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data[0]["content"] == SAMPLE_CONTENT


def test_dense_isolates_users(dense_client: TestClient) -> None:
    """VectorIndex 按 user_id 分桶；dense 搜索结果仍隔离。"""
    dense_client.post("/add", json=_payload(), headers=_auth())
    dense_client.post(
        "/add",
        json=_payload(
            request_id="other:chunk-0",
            user_id="someone-else",
            session_id="other-session",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067200000,
                    "content": "I adopted a cat named Mars.",
                }
            ],
        ),
        headers=_auth(),
    )
    mine = dense_client.post(
        "/search",
        json=_search_body(query="cat"),
        headers=_auth(),
    )
    theirs = dense_client.post(
        "/search",
        json=_search_body(query="cat", user_id="someone-else"),
        headers=_auth(),
    )
    assert [item["content"] for item in mine.json()["data"]] == [SAMPLE_CONTENT]
    assert [item["content"] for item in theirs.json()["data"]] == [
        "I adopted a cat named Mars."
    ]


def test_dense_unknown_user_is_empty(dense_client: TestClient) -> None:
    """无该 user_id 的 FAISS 桶 → {"data": []}。"""
    dense_client.post("/add", json=_payload(), headers=_auth())
    response = dense_client.post(
        "/search",
        json=_search_body(user_id="no-such-user", query="Luna"),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_vectors_survive_reopen(db_path: str) -> None:
    """第二个 create_app 从 message_vectors 重建 IndexFlatIP，Search 仍命中。"""
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="dense",
    )
    with TestClient(create_app(settings)) as first:
        assert first.post("/add", json=_payload(), headers=_auth()).status_code == 200
    with TestClient(create_app(settings)) as second:
        response = second.post(
            "/search",
            json=_search_body(query="Luna"),
            headers=_auth(),
        )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"] == SAMPLE_CONTENT
