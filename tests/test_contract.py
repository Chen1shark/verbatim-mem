import sqlite3

import pytest
from fastapi.testclient import TestClient

SAMPLE_CONTENT = "I adopted a cat named Luna."
API_KEY = "test-key"


def _payload(**overrides):
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
    if header == "X-Api-Key":
        return {"X-Api-Key": API_KEY}
    if header == "Bearer":
        return {"Authorization": f"Bearer {API_KEY}"}
    return {"Authorization": f"Token {API_KEY}"}


def test_health_does_not_need_key(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_add_without_key_is_401(client: TestClient) -> None:
    response = client.post("/add", json=_payload())
    assert response.status_code == 401


def test_add_wrong_key_is_401(client: TestClient) -> None:
    response = client.post(
        "/add",
        json=_payload(),
        headers={"X-Api-Key": "nope"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("header", ["X-Api-Key", "Bearer", "Token"])
def test_add_accepts_three_auth_headers(client: TestClient, header: str) -> None:
    response = client.post("/add", json=_payload(), headers=_auth(header))
    assert response.status_code == 200


def test_add_persists_verbatim(client: TestClient, db_path: str) -> None:
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
    client.post("/add", json=_payload(), headers=_auth())
    conflict = client.post(
        "/add",
        json=_payload(user_id="someone-else"),
        headers=_auth(),
    )
    assert conflict.status_code == 409


def test_add_multiple_messages(client: TestClient, db_path: str) -> None:
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
    body = _payload()
    del body["request_id"]
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 422
    assert SAMPLE_CONTENT not in response.text


def test_add_empty_messages_is_422(client: TestClient) -> None:
    response = client.post(
        "/add",
        json=_payload(messages=[]),
        headers=_auth(),
    )
    assert response.status_code == 422


def test_add_ignores_unknown_fields(client: TestClient) -> None:
    body = _payload()
    body["unexpected"] = "field"
    response = client.post("/add", json=body, headers=_auth())
    assert response.status_code == 200


def test_add_alias_path(client: TestClient) -> None:
    response = client.post("/v1/memory/add", json=_payload(), headers=_auth())
    assert response.status_code == 200


def test_logs_omit_secret_and_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    client.post("/add", json=_payload(), headers=_auth())
    text = caplog.text
    assert SAMPLE_CONTENT not in text
    assert API_KEY not in text


def test_openapi_declares_api_key(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    schemes = spec["components"]["securitySchemes"]
    assert schemes["ApiKey"]["name"] == "X-Api-Key"
    assert schemes["ApiKey"]["type"] == "apiKey"
    add_security = spec["paths"]["/add"]["post"].get("security", [])
    assert {"ApiKey": []} in add_security
    assert "security" not in spec["paths"]["/health"]["get"]
