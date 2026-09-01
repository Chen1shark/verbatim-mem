from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="探活")
def health(request: Request) -> dict[str, str]:
    request.app.state.store.ping()
    return {"status": "ok"}
