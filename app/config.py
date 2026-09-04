"""Settings：MEMORY_API_KEY、MEMORY_DB_PATH、embedding、MEMORY_RETRIEVAL_MODE、召回池与排序权重。"""

from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """密钥、库路径、付费 embedding、召回模式、召回池与 ranking 权重。"""

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
    memory_candidate_pool: int = 500
    memory_fts_pool: int = 300
    memory_dense_pool: int = 300
    memory_block_pool: int = 200
    memory_seed_k: int = 25
    memory_neighbor_window: int = 1
    memory_neighbor_budget: int = 40
    memory_rrf_k: int = 60
    memory_rrf_w_fts: float = 1.2
    memory_rrf_w_dense: float = 1.0
    memory_rrf_w_blocks: float = 1.1
    memory_lexical_weight: float = 0.25
    memory_numeric_weight: float = 0.2
    memory_entity_weight: float = 0.22
    memory_option_weight: float = 0.18
    memory_time_weight_temporal: float = 0.08


@dataclass(frozen=True)
class RetrievalConfig:
    """MemoryStore.search 的召回池、邻句预算、RRF 与 ranking 权重。"""

    candidate_pool: int = 500
    fts_pool: int = 300
    dense_pool: int = 300
    block_pool: int = 200
    seed_k: int = 25
    neighbor_window: int = 1
    neighbor_budget: int = 40
    rrf_k: int = 60
    rrf_w_fts: float = 1.2
    rrf_w_dense: float = 1.0
    rrf_w_blocks: float = 1.1
    lexical_weight: float = 0.25
    numeric_weight: float = 0.2
    entity_weight: float = 0.22
    option_weight: float = 0.18
    time_weight_temporal: float = 0.08
    neighbor_score_delta: float = 0.001

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalConfig":
        """从 Settings.memory_* 填 RetrievalConfig。"""
        return cls(
            candidate_pool=settings.memory_candidate_pool,
            fts_pool=settings.memory_fts_pool,
            dense_pool=settings.memory_dense_pool,
            block_pool=settings.memory_block_pool,
            seed_k=settings.memory_seed_k,
            neighbor_window=settings.memory_neighbor_window,
            neighbor_budget=settings.memory_neighbor_budget,
            rrf_k=settings.memory_rrf_k,
            rrf_w_fts=settings.memory_rrf_w_fts,
            rrf_w_dense=settings.memory_rrf_w_dense,
            rrf_w_blocks=settings.memory_rrf_w_blocks,
            lexical_weight=settings.memory_lexical_weight,
            numeric_weight=settings.memory_numeric_weight,
            entity_weight=settings.memory_entity_weight,
            option_weight=settings.memory_option_weight,
            time_weight_temporal=settings.memory_time_weight_temporal,
        )
