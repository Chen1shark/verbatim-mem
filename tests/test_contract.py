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
    assert data[0]["role"] == "user"
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
    """SearchRequest extra=ignore；options 参与召回但仍 200。"""
    client.post("/add", json=_payload(), headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(options=["A. Viton", "B. Nitrile"], unexpected="field"),
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


# --- Search：_apply_recency / _compose_results ---

OLD_CAT = "My cat is named Luna."
NEW_CAT = "My cat is named Mars."
NEIGHBOR_PREV = "I adopted a cat."
NEIGHBOR_HIT = "Her name is Luna."
NEIGHBOR_NEXT = "She likes tuna."
UNRELATED_NEW = "Today the weather is sunny and warm."
OTHER_SESSION_DECOY = "The weather is sunny."


def test_search_prefers_newer_statement(client: TestClient, db_path: str) -> None:
    """_apply_recency：query 含 now 时新旧两句都在 data 里且新句更前。"""
    older = "The torque wrench lives in cabinet K7."
    newer = "The torque wrench lives in cabinet P2."
    client.post("/add", json=_payload(messages=[
        {"role": "user", "timestamp": 1704067200000, "content": older},
    ]), headers=_auth())
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[
                {"role": "user", "timestamp": 1704153600000, "content": newer},
            ],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Where does the torque wrench live now?"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert newer in contents
    assert older in contents
    assert contents.index(newer) < contents.index(older)
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
    """_compose_results：命中 ±1 且同 session。"""
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
    """_compose_results 的 SQL 带 session_id，不跨会话拼接。"""
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


NEIGHBOR_FAR_PREV = "I went to the shelter."
NEIGHBOR_FAR_NEXT = "We bought a scratching post."


def test_search_includes_plus_minus_two_neighbors(client: TestClient) -> None:
    """_compose_results：neighbor_window=1，±1 必在；±2 可由 message_blocks_fts 进入。"""
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": NEIGHBOR_FAR_PREV},
                {"role": "user", "timestamp": 1704067201000, "content": NEIGHBOR_PREV},
                {"role": "assistant", "timestamp": 1704067202000, "content": NEIGHBOR_HIT},
                {"role": "user", "timestamp": 1704067203000, "content": NEIGHBOR_NEXT},
                {"role": "user", "timestamp": 1704067204000, "content": NEIGHBOR_FAR_NEXT},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Luna", top_k=5),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert NEIGHBOR_HIT in contents
    assert NEIGHBOR_PREV in contents
    assert NEIGHBOR_NEXT in contents
    assert NEIGHBOR_FAR_PREV in contents
    assert NEIGHBOR_FAR_NEXT in contents


def test_search_top_k_three_keeps_hit_first(client: TestClient) -> None:
    """top_k=3 时精排命中句在前，条数为 3。"""
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": NEIGHBOR_FAR_PREV},
                {"role": "user", "timestamp": 1704067201000, "content": NEIGHBOR_PREV},
                {"role": "assistant", "timestamp": 1704067202000, "content": NEIGHBOR_HIT},
                {"role": "user", "timestamp": 1704067203000, "content": NEIGHBOR_NEXT},
                {"role": "user", "timestamp": 1704067204000, "content": NEIGHBOR_FAR_NEXT},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Luna", top_k=3),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert len(contents) == 3
    assert contents[0] == NEIGHBOR_HIT


def test_search_prefers_older_when_previously(client: TestClient) -> None:
    """_temporal_alpha 负值：query 含 previously 时较旧陈述排在较新陈述前。"""
    older = "The torque wrench lives in cabinet K7."
    newer = "The torque wrench lives in cabinet P2."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": older}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704153600000, "content": newer}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Which cabinet was used previously?"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert older in contents
    assert newer in contents
    assert contents.index(older) < contents.index(newer)


def test_search_options_boost_matching_content(client: TestClient) -> None:
    """SearchRequest.options 含 Viton 时 Viton 句排在 Nitrile 句前。"""
    viton = "The manifold uses a Viton gasket."
    nitrile = "The pump uses a Nitrile gasket."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": viton}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067201000,
                    "content": nitrile,
                }
            ],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(
            query="Which gasket material is stocked?",
            options=["A. Viton", "B. EPDM"],
        ),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert viton in contents
    assert nitrile in contents
    assert contents.index(viton) < contents.index(nitrile)


def test_add_missing_timestamp_is_200(client: TestClient, db_path: str) -> None:
    """Message.timestamp 缺省仍写入；messages.timestamp 为 NULL。"""
    body = _payload()
    del body["messages"][0]["timestamp"]
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 200
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT timestamp, content FROM messages").fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] == SAMPLE_CONTENT


def test_add_writes_session_blocks(client: TestClient, db_path: str) -> None:
    """两条 messages 写入一条 message_blocks_fts。"""
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
        count = conn.execute("SELECT COUNT(*) FROM message_blocks_fts").fetchone()[0]
    assert count == 1


def test_add_session_blocks_appended_chunk(client: TestClient, db_path: str) -> None:
    """同 session 第二 chunk 后 message_blocks_fts 含 2 句窗与 3 句窗。"""
    first = _payload(
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
    second = _payload(
        request_id="eval:run:dataset:conv-0:chunk-1",
        messages=[
            {
                "role": "user",
                "timestamp": 1704067202000,
                "content": "Luna likes tuna.",
            }
        ],
    )
    assert client.post("/add", json=first, headers=_auth()).status_code == 200
    assert client.post("/add", json=second, headers=_auth()).status_code == 200
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM message_blocks_fts").fetchone()[0]
        pairs = {
            (row[0], row[1])
            for row in conn.execute("SELECT left_id, right_id FROM message_blocks_fts")
        }
    assert count == 3
    assert pairs == {
        (
            "eval:run:dataset:conv-0:chunk-0:0",
            "eval:run:dataset:conv-0:chunk-0:1",
        ),
        (
            "eval:run:dataset:conv-0:chunk-0:1",
            "eval:run:dataset:conv-0:chunk-1:0",
        ),
        (
            "eval:run:dataset:conv-0:chunk-0:0",
            "eval:run:dataset:conv-0:chunk-1:0",
        ),
    }


def test_add_session_blocks_middle_insert(client: TestClient, db_path: str) -> None:
    """后写入中间 timestamp 时删掉旧邻接对，左右与新句成 2 句窗，两端成 3 句窗。"""
    first = _payload(
        messages=[
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": SAMPLE_CONTENT,
            },
            {
                "role": "assistant",
                "timestamp": 1704067202000,
                "content": "Luna likes tuna.",
            },
        ]
    )
    middle = _payload(
        request_id="eval:run:dataset:conv-0:chunk-1",
        messages=[
            {
                "role": "assistant",
                "timestamp": 1704067201000,
                "content": "Luna is a lovely name.",
            }
        ],
    )
    assert client.post("/add", json=first, headers=_auth()).status_code == 200
    assert client.post("/add", json=middle, headers=_auth()).status_code == 200
    with sqlite3.connect(db_path) as conn:
        pairs = {
            (row[0], row[1])
            for row in conn.execute("SELECT left_id, right_id FROM message_blocks_fts")
        }
    assert pairs == {
        (
            "eval:run:dataset:conv-0:chunk-0:0",
            "eval:run:dataset:conv-0:chunk-1:0",
        ),
        (
            "eval:run:dataset:conv-0:chunk-1:0",
            "eval:run:dataset:conv-0:chunk-0:1",
        ),
        (
            "eval:run:dataset:conv-0:chunk-0:0",
            "eval:run:dataset:conv-0:chunk-0:1",
        ),
    }


def test_add_too_many_messages_is_422(client: TestClient) -> None:
    """AddRequest.messages max_length=500。"""
    messages = [
        {"role": "user", "timestamp": 1704067200000, "content": "x"} for _ in range(501)
    ]
    response = client.post(
        "/add",
        json=_payload(messages=messages),
        headers=_auth(),
    )
    assert response.status_code == 422


def test_add_overlong_content_is_422(client: TestClient) -> None:
    """Message.content max_length=100_000。"""
    response = client.post(
        "/add",
        json=_payload(
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067200000,
                    "content": "x" * 100_001,
                }
            ]
        ),
        headers=_auth(),
    )
    assert response.status_code == 422


def test_add_embedding_failure_does_not_persist(
    client: TestClient, db_path: str
) -> None:
    """encode_docs 抛 RuntimeError → 500；requests / messages 仍为空。"""

    def boom(_texts: list[str]):
        raise RuntimeError("embedding request failed")

    client.app.state.store._embedder.encode_docs = boom  # type: ignore[method-assign]
    response = client.post("/add", json=_payload(), headers=_auth())
    assert response.status_code == 500
    assert SAMPLE_CONTENT not in response.text
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


def test_add_embedding_shape_mismatch_does_not_persist(
    client: TestClient, db_path: str
) -> None:
    """encode_docs 行数对不上 messages → 500；requests / messages 仍为空。"""
    import numpy as np

    def bad(_texts: list[str]):
        return np.zeros((2, 384), dtype=np.float32)

    client.app.state.store._embedder.encode_docs = bad  # type: ignore[method-assign]
    response = client.post("/add", json=_payload(), headers=_auth())
    assert response.status_code == 500
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


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


def test_add_writes_stem_clues(client: TestClient, db_path: str) -> None:
    """index_clues：working → work 写入 messages.clues；Search content 仍是原话。"""
    content = "The widgets were working."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": content}]
        ),
        headers=_auth(),
    )
    with sqlite3.connect(db_path) as conn:
        clues = conn.execute("SELECT clues FROM messages").fetchone()[0]
    assert "work" in clues.split()
    response = client.post(
        "/search",
        json=_search_body(query="widgets"),
        headers=_auth(),
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["content"] == content
    assert "work" not in response.json()["data"][0]["content"].split()


def test_search_lexical_prefers_overlapping_token(client: TestClient) -> None:
    """问句与其中一句共享稀有词时，该句排在另一句前。"""
    viton = "The manifold uses a Viton gasket."
    nitrile = "The pump uses a Nitrile gasket."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": viton}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704067201000, "content": nitrile}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Which gasket is Viton?"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert viton in contents
    assert nitrile in contents
    assert contents.index(viton) < contents.index(nitrile)


def test_search_year_prefers_matching_number(client: TestClient) -> None:
    """_apply_numeric：问 2021 时含 2021 的句排在含 2019 的句前。"""
    lot_2021 = "Batch lot 2021 passed inspection."
    lot_2019 = "Batch lot 2019 passed inspection."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": lot_2021}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704153600000, "content": lot_2019}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Which lot number is 2021?"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert lot_2021 in contents
    assert lot_2019 in contents
    assert contents.index(lot_2021) < contents.index(lot_2019)


def test_search_multihop_triple_window(client: TestClient) -> None:
    """三句窗：专名与地点分在首尾句时，两句都能进 data。"""
    first = "Foreman Helix logged the shift."
    middle = "They inspected the kiln together."
    last = "The kiln sits in bay 4."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": first},
                {"role": "assistant", "timestamp": 1704067201000, "content": middle},
                {"role": "user", "timestamp": 1704067202000, "content": last},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Where is the kiln that Helix inspected?", top_k=5),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert first in contents
    assert last in contents


def test_search_entity_prefers_query_proper_noun(client: TestClient) -> None:
    """_apply_entity：问 Helix 时含 Helix 的句排在含 Quorum 的句前。"""
    helix = "Foreman Helix logged bay 4."
    quorum = "Foreman Quorum logged bay 9."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": helix},
                {"role": "assistant", "timestamp": 1704067201000, "content": quorum},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Which bay did Helix log?"),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert helix in contents
    assert quorum in contents
    assert contents.index(helix) < contents.index(quorum)


def test_search_two_aspect_query_keeps_second_fact(client: TestClient) -> None:
    """_cover_reorder：问句含两类事实时，第二类原话仍进 top_k=3。"""
    gasket = "The gasket SKU is NBR-4407."
    shelf = "NBR-4407 sits on shelf D."
    night = "Shelf D is locked at night."
    kiln = "The kiln sits in bay 4."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": gasket},
                {"role": "assistant", "timestamp": 1704067201000, "content": shelf},
                {"role": "user", "timestamp": 1704067202000, "content": night},
            ]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:conf",
            session_id="eval:run:sample:conf",
            messages=[
                {"role": "user", "timestamp": 1704067203000, "content": kiln}
            ],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(
            query="What is the gasket SKU and where is the kiln?",
            top_k=3,
        ),
        headers=_auth(),
    )
    assert response.status_code == 200
    contents = [item["content"] for item in response.json()["data"]]
    assert kiln in contents


def test_search_missing_timestamp_omits_epoch(client: TestClient) -> None:
    """messages.timestamp 为空时 SearchItem.created_at 为 null，正文不含 1970。"""
    content = "The spare gasket SKU is NBR-4407."
    body = _payload(
        messages=[{"role": "user", "content": content}]
    )
    client.post("/add", json=body, headers=_auth())
    response = client.post(
        "/search",
        json=_search_body(query="NBR-4407"),
        headers=_auth(),
    )
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert item["content"] == content
    assert item["created_at"] is None
    assert "1970" not in response.text


def test_search_same_timestamp_keeps_add_order(client: TestClient, db_path: str) -> None:
    """同一 timestamp 时 messages.source_order 与 Add 顺序一致。"""
    first = "The inlet valve closed."
    second = "The outlet valve opened."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": first},
                {"role": "assistant", "timestamp": 1704067200000, "content": second},
            ]
        ),
        headers=_auth(),
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT content, source_order FROM messages ORDER BY source_order, id"
        ).fetchall()
    assert [row[0] for row in rows] == [first, second]
    assert [row[1] for row in rows] == [0, 1]


def test_search_middle_insert_keeps_timestamp_order(client: TestClient, db_path: str) -> None:
    """后写入中间 timestamp 后，同 session 邻句仍按 timestamp、source_order。"""
    prev_msg = "The manifold was purged."
    hit_msg = "The spare gasket SKU is NBR-4407."
    next_msg = "Shelf D was relabeled."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": prev_msg},
                {"role": "user", "timestamp": 1704067202000, "content": next_msg},
            ]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[
                {
                    "role": "assistant",
                    "timestamp": 1704067201000,
                    "content": hit_msg,
                }
            ],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="NBR-4407"),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert hit_msg in contents
    assert prev_msg in contents
    assert next_msg in contents
    with sqlite3.connect(db_path) as conn:
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT content FROM messages ORDER BY timestamp IS NULL, timestamp, source_order, id"
            )
        ]
    assert ids == [prev_msg, hit_msg, next_msg]


def test_search_order_survives_restart(db_path: str) -> None:
    """重启 create_app 后 source_order 仍决定同 timestamp 顺序。"""
    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="fts",
    )
    prev_msg = "The manifold was purged."
    hit_msg = "The spare gasket SKU is NBR-4407."
    payload = _payload(
        messages=[
            {"role": "user", "timestamp": 1704067200000, "content": prev_msg},
            {"role": "assistant", "timestamp": 1704067200000, "content": hit_msg},
        ]
    )
    with TestClient(create_app(settings)) as first:
        assert first.post("/add", json=payload, headers=_auth()).status_code == 200
    with TestClient(create_app(settings)) as second:
        response = second.post(
            "/search",
            json=_search_body(query="NBR-4407"),
            headers=_auth(),
        )
    contents = [item["content"] for item in response.json()["data"]]
    assert hit_msg in contents
    assert prev_msg in contents


def test_search_no_default_recency(client: TestClient) -> None:
    """无时间意图时较新陈述不因 recency 排到较旧陈述前。"""
    older = "The torque wrench lives in cabinet K7."
    newer = "The torque wrench lives in cabinet P2."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": older}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704153600000, "content": newer}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Where does the torque wrench live?"),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert older in contents
    assert newer in contents
    assert contents.index(older) < contents.index(newer)


def test_search_specific_older_fact_not_overtaken_by_newer(client: TestClient) -> None:
    """问句只覆盖旧事实时，较新的无关更新不得凭时间排到第一。"""
    older = "The autoclave was calibrated in 2017."
    newer = "The inlet gasket was replaced yesterday."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": older}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704153600000, "content": newer}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="When was the autoclave calibrated?"),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert older in contents
    assert contents[0] == older


def test_search_before_prefers_older_statement(client: TestClient) -> None:
    """query 含 before 时较旧陈述排在较新陈述前。"""
    older = "The torque wrench lives in cabinet K7."
    newer = "The torque wrench lives in cabinet P2."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": older}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[{"role": "user", "timestamp": 1704153600000, "content": newer}],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="Where did the torque wrench live before?"),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert older in contents
    assert newer in contents
    assert contents.index(older) < contents.index(newer)


def test_search_options_none_matches_plain_query(client: TestClient) -> None:
    """options=None 时行为与无 options 字段相同，仍召回原话。"""
    content = "The centrifuge rotor is rated 15000 rpm."
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": content}]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(query="centrifuge rotor", options=None),
        headers=_auth(),
    )
    assert response.json()["data"][0]["content"] == content


def test_search_wrong_option_does_not_drown_evidence(client: TestClient) -> None:
    """错误选项里的长关键词不把正确原话挤出 data。"""
    evidence = "The spare gasket SKU is NBR-4407."
    distractor = (
        "The dual-stage rotary vane pump oil change schedule is posted on bay 3."
    )
    client.post(
        "/add",
        json=_payload(
            messages=[{"role": "user", "timestamp": 1704067200000, "content": evidence}]
        ),
        headers=_auth(),
    )
    client.post(
        "/add",
        json=_payload(
            request_id="eval:run:dataset:conv-0:chunk-1",
            messages=[
                {
                    "role": "user",
                    "timestamp": 1704067201000,
                    "content": distractor,
                }
            ],
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(
            query="What is the spare gasket SKU?",
            options=[
                "A. NBR-4407",
                "B. dual-stage rotary vane pump oil change schedule",
            ],
        ),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert evidence in contents
    assert contents.index(evidence) < contents.index(distractor)


def test_search_options_keep_both_choice_evidence(client: TestClient) -> None:
    """多选项各自通道能同时召回两条独立原话。"""
    cold = "The reagent must stay at minus eighty."
    rotor = "The centrifuge rotor is rated 15000 rpm."
    client.post(
        "/add",
        json=_payload(
            messages=[
                {"role": "user", "timestamp": 1704067200000, "content": cold},
                {"role": "user", "timestamp": 1704067201000, "content": rotor},
            ]
        ),
        headers=_auth(),
    )
    response = client.post(
        "/search",
        json=_search_body(
            query="Which equipment ratings are recorded?",
            options=["A. minus eighty", "B. 15000 rpm"],
        ),
        headers=_auth(),
    )
    contents = [item["content"] for item in response.json()["data"]]
    assert cold in contents
    assert rotor in contents


def test_old_messages_schema_migrates_null_timestamp(db_path: str) -> None:
    """旧 messages.timestamp NOT NULL 且为 0 时，_ensure_message_time_schema 写成 NULL。"""
    from app.config import Settings
    from app.main import create_app

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                content TEXT NOT NULL,
                clues TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            );
            CREATE INDEX idx_messages_user_session
                ON messages(user_id, session_id, timestamp);
            INSERT INTO requests VALUES ('r0', 'u0', 's0', 'hash', 1);
            INSERT INTO messages VALUES (
                'r0:0', 'r0', 'u0', 's0', 'user', 0,
                'The spare gasket SKU is NBR-4407.', '', 1
            );
            """
        )
    settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="hybrid",
    )
    with TestClient(create_app(settings)) as client:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT timestamp, source_order FROM messages"
            ).fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] == 0
        response = client.post(
            "/search",
            json=_search_body(user_id="u0", query="NBR-4407"),
            headers=_auth(),
        )
    assert response.status_code == 200
    assert "1970" not in response.text
    assert response.json()["data"][0]["created_at"] is None
    assert response.json()["data"][0]["content"] == "The spare gasket SKU is NBR-4407."
