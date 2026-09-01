"""create_app(Settings) 的 pytest 夹具。"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API_KEY = "test-key"


@pytest.fixture
def db_path(tmp_path) -> str:
    """Settings.memory_db_path。"""
    return str(tmp_path / "memory.db")


@pytest.fixture
def settings(db_path: str) -> Settings:
    """直接构造 Settings。embedding_model=hash 走 HashEmbedder；memory_retrieval_mode=hybrid。"""
    return Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="hybrid",
    )


@pytest.fixture
def client(settings: Settings):
    """TestClient(create_app(settings))。"""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def dense_client(db_path: str):
    """与 client 相同，memory_retrieval_mode=dense。"""
    dense_settings = Settings(
        memory_api_key=API_KEY,
        memory_db_path=db_path,
        embedding_model="hash",
        memory_retrieval_mode="dense",
    )
    with TestClient(create_app(dense_settings)) as test_client:
        yield test_client
