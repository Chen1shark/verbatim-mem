"""Settings：MEMORY_API_KEY、MEMORY_DB_PATH、embedding、MEMORY_RETRIEVAL_MODE。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """密钥、库路径、付费 embedding、召回模式、可选 MEMORY_RERANK_MODEL。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memory_api_key: str = ""
    memory_db_path: str = "data/memory.db"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dim: int = 2560
    memory_retrieval_mode: str = "hybrid"
    memory_rerank_model: str = ""
