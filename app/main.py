"""uvicorn app.main:app --workers 1"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import RetrievalConfig, Settings
from app.embeddings import build_embedder
from app.logging_cfg import setup_logging
from app.routers import health, memory
from app.store import MemoryStore

logger = logging.getLogger("verbatim_mem")

OPENAPI_TAGS = [
    {"name": "health", "description": "探活，不鉴权"},
    {"name": "memory", "description": "Add 同步写入原文与向量；Search 为 FTS5 ∪ 邻句块 ∪ FAISS"},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """组装 Settings、build_embedder、MemoryStore，挂 /health /add /search。"""
    setup_logging()
    resolved = settings or Settings()
    embedder = build_embedder(
        resolved.embedding_model,
        api_key=resolved.embedding_api_key,
        base_url=resolved.embedding_base_url,
        dim=resolved.embedding_dim,
    )
    store = MemoryStore(
        resolved.memory_db_path,
        embedder=embedder,
        retrieval_mode=resolved.memory_retrieval_mode,
        retrieval=RetrievalConfig.from_settings(resolved),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """MemoryStore.open → init_schema → warmup；退出 close。"""
        store.open()
        store.init_schema()
        store.warmup()
        yield
        store.close()

    application = FastAPI(
        title="verbatim-mem",
        description="AML Add/Search。SQLite 原文 + FTS5 ∪ FAISS，只交原话。",
        version="0.2.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        swagger_ui_parameters={"persistAuthorization": True},
    )
    application.state.settings = resolved
    application.state.store = store

    @application.middleware("http")
    async def log_http(request: Request, call_next):  # type: ignore[no-untyped-def]
        """打 method / path / status / duration_ms，不含 query 与 content。"""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "http method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        """HTTPException → {"detail": ...}。"""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        """RequestValidationError → 422 {"detail": "invalid request"}。"""
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @application.exception_handler(RuntimeError)
    async def runtime_exception_handler(
        _request: Request, _exc: RuntimeError
    ) -> JSONResponse:
        """RuntimeError（如 embedding 失败）→ 500 {"detail": "internal error"}。"""
        logger.info("runtime_error status=500")
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    application.include_router(health.router)
    application.include_router(memory.router)
    return application


app = create_app()
