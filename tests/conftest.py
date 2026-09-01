import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API_KEY = "test-key"


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "memory.db")


@pytest.fixture
def settings(db_path: str) -> Settings:
    return Settings(memory_api_key=API_KEY, memory_db_path=db_path)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
