"""X-Api-Key / Bearer / Token 对 Settings.memory_api_key。"""

from typing import Annotated

from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-Api-Key",
    auto_error=False,
    scheme_name="ApiKey",
    description="MEMORY_API_KEY；亦支持 Authorization Bearer / Token",
)


def _extract_token(x_api_key: str | None, authorization: str | None) -> str | None:
    """从 X-Api-Key 或 Authorization Bearer/Token 取出密钥。"""
    if x_api_key:
        return x_api_key
    if not authorization:
        return None
    for prefix in ("Bearer ", "Token "):
        if authorization.startswith(prefix):
            return authorization[len(prefix) :]
    return None


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Security(api_key_header)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """token 对不上 Settings.memory_api_key 则 401。"""
    expected = request.app.state.settings.memory_api_key
    token = _extract_token(x_api_key, authorization)
    if not expected or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
