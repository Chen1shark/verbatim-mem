"""build_embedder、OpenAICompatibleEmbedder。"""

import numpy as np

from app.embeddings import OpenAICompatibleEmbedder, build_embedder

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

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=2560,
    )
    vector = embedder.encode_query("hi")
    assert seen["url"].endswith("/embeddings")
    assert seen["json"]["model"] == PROD_MODEL
    assert seen["json"]["dimensions"] == 2560
    assert seen["authorization"] == "Bearer sk-test"
    assert vector.shape == (1, 2560)


def test_qwen_embedder_normalizes(monkeypatch) -> None:
    """encode_docs 对假响应 [3,4,0] L2 归一化。"""

    class FakeResp:
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

    monkeypatch.setattr("app.embeddings.httpx.Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(
        model=PROD_MODEL,
        api_key="sk-test",
        base_url=PROD_BASE,
        dim=3,
    )
    vector = embedder.encode_docs(["hi"])
    assert vector.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(vector[0]), 1.0, atol=1e-5)
