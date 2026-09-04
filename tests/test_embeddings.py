"""build_embedder、OpenAICompatibleEmbedder。"""

import httpx
import numpy as np
import pytest

from app.embeddings import OpenAICompatibleEmbedder, _MAX_ATTEMPTS, build_embedder

PROD_MODEL = "qwen3.7-text-embedding"
PROD_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_build_embedder_hash_skips_api() -> None:
    """build_embedder("hash") → HashEmbedder，shape (1, 384)。"""
    embedder = build_embedder("hash", api_key="should-not-use")
    assert embedder.identity == "hash:384"
    vector = embedder.encode_query("Luna")
    assert vector.shape == (1, 384)


def test_qwen_embedder_sends_model_and_dimensions(monkeypatch) -> None:
    """httpx POST json["model"]、json["dimensions"]；Authorization Bearer。"""
    seen: dict[str, object] = {}

    class FakeResp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [1.0] * 2560}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, json: dict, headers: dict):
            seen["url"] = url
            seen["json"] = json
            seen["authorization"] = headers["Authorization"]
            return FakeResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=2560,
    )
    vector = embedder.encode_query("hi")
    embedder.close()
    assert seen["url"].endswith("/embeddings")
    assert seen["json"]["model"] == PROD_MODEL
    assert seen["json"]["dimensions"] == 2560
    assert seen["authorization"] == "Bearer sk-test"
    assert vector.shape == (1, 2560)


def test_qwen_embedder_normalizes(monkeypatch) -> None:
    """encode_docs 对假响应 [3,4,0] L2 归一化。"""

    class FakeResp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [3.0, 4.0, 0.0]}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, json: dict, headers: dict):
            assert json["model"] == PROD_MODEL
            return FakeResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=3,
    )
    vector = embedder.encode_docs(["hi"])
    embedder.close()
    assert vector.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(vector[0]), 1.0, atol=1e-5)


def test_qwen_embedder_retries_then_succeeds(monkeypatch) -> None:
    """_RETRY_STATUSES 第一次 503，第二次 200。"""
    monkeypatch.setattr("app.embeddings.time.sleep", lambda _seconds: None)
    posts = {"n": 0}

    class FailResp:
        status_code = 503
        headers: dict = {}

    class OkResp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [1.0] * 2560}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def post(self, url: str, json: dict, headers: dict):
            posts["n"] += 1
            if posts["n"] == 1:
                return FailResp()
            return OkResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=2560,
    )
    vector = embedder.encode_query("hi")
    embedder.close()
    assert posts["n"] == 2
    assert vector.shape == (1, 2560)


def test_qwen_embedder_retry_exhausted_raises(monkeypatch) -> None:
    """连续 _MAX_ATTEMPTS 次 503 → RuntimeError。"""
    monkeypatch.setattr("app.embeddings.time.sleep", lambda _seconds: None)
    posts = {"n": 0}

    class FailResp:
        status_code = 503
        headers: dict = {}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def post(self, url: str, json: dict, headers: dict):
            posts["n"] += 1
            return FailResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=2560,
    )
    with pytest.raises(RuntimeError, match="embedding request failed"):
        embedder.encode_query("hi")
    embedder.close()
    assert posts["n"] == _MAX_ATTEMPTS


def test_qwen_embedder_does_not_retry_400(monkeypatch) -> None:
    """非 _RETRY_STATUSES 的 400 不重试。"""
    posts = {"n": 0}

    class BadResp:
        status_code = 400
        headers: dict = {}

        def raise_for_status(self) -> None:
            raise httpx.HTTPError("bad request")

        def json(self) -> dict:
            raise AssertionError("should not parse")

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def post(self, url: str, json: dict, headers: dict):
            posts["n"] += 1
            return BadResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=2560,
    )
    with pytest.raises(RuntimeError, match="embedding request failed"):
        embedder.encode_query("hi")
    embedder.close()
    assert posts["n"] == 1


def test_qwen_embedder_caches_identical_text(monkeypatch) -> None:
    """同一句子第二次 encode_query 不再 POST；同批重复只请求一次。"""
    posts = {"n": 0}

    class OkResp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [1.0] * 3}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def post(self, url: str, json: dict, headers: dict):
            posts["n"] += 1
            assert len(json["input"]) == 1
            return OkResp()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=3,
    )
    first = embedder.encode_docs(["hi", "hi"])
    second = embedder.encode_query("hi")
    embedder.close()
    assert posts["n"] == 1
    assert first.shape == (2, 3)
    assert second.shape == (1, 3)
    np.testing.assert_allclose(first[0], first[1])
    np.testing.assert_allclose(first[0], second[0])


def test_retry_delay_prefers_retry_after() -> None:
    """Retry-After 封顶 5s；无头则 0.5 * 2^attempt 封顶 2s。"""
    from app.embeddings import _retry_delay_s

    assert _retry_delay_s(0, None) == 0.5
    assert _retry_delay_s(2, None) == 2.0
    response = httpx.Response(503, headers={"Retry-After": "3"})
    assert _retry_delay_s(0, response) == 3.0
    capped = httpx.Response(429, headers={"Retry-After": "9"})
    assert _retry_delay_s(0, capped) == 5.0
