from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """`MEMORY_API_KEY`、`MEMORY_DB_PATH`。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memory_api_key: str = ""
    memory_db_path: str = "data/memory.db"
