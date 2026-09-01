"""POST /add。"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies import require_api_key
from app.schemas import AddRequest, AddResponse
from app.store import ConflictError, MemoryStore

logger = logging.getLogger("verbatim_mem")

router = APIRouter(tags=["memory"])

_ADD_RESPONSES = {
    401: {"description": "未授权"},
    409: {"description": "request_id 冲突"},
    422: {"description": "请求体不合法"},
}


@router.post(
    "/add",
    response_model=AddResponse,
    summary="写入原话",
    responses=_ADD_RESPONSES,
)
@router.post(
    "/v1/memory/add",
    response_model=AddResponse,
    include_in_schema=False,
    responses=_ADD_RESPONSES,
)
def add_memory(
    request: Request,
    body: AddRequest,
    _: None = Depends(require_api_key),
) -> AddResponse:
    store: MemoryStore = request.app.state.store
    started = time.perf_counter()
    try:
        outcome = store.add(body)
    except ConflictError:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "add outcome=conflict request_id=%s user_id=%s messages=%s duration_ms=%s status=409",
            body.request_id,
            body.user_id,
            len(body.messages),
            elapsed_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="request_id conflict",
        ) from None
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "add outcome=%s request_id=%s user_id=%s messages=%s duration_ms=%s status=200",
        outcome,
        body.request_id,
        body.user_id,
        len(body.messages),
        elapsed_ms,
    )
    return AddResponse(
        success=True,
        request_id=body.request_id,
        user_id=body.user_id,
        session_id=body.session_id,
    )
