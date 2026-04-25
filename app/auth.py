from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


@dataclass(frozen=True)
class RequestContext:
    user_id: str


def get_request_context(
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> RequestContext:
    api_key = x_api_key if isinstance(x_api_key, str) else None
    user_id = x_user_id if isinstance(x_user_id, str) else None

    if not settings.auth_enabled:
        return RequestContext(user_id=user_id or settings.default_user_id)

    if not settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="APP_API_KEY is required when AUTH_ENABLED=true",
        )
    if api_key != settings.app_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-User-Id is required")
    return RequestContext(user_id=user_id)
