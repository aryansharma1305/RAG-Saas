from fastapi import HTTPException

from app.auth import get_request_context
from app.config import Settings


def test_auth_disabled_uses_default_user() -> None:
    settings = Settings(AUTH_ENABLED=False, DEFAULT_USER_ID="local")

    context = get_request_context(settings=settings)

    assert context.user_id == "local"


def test_auth_enabled_requires_valid_api_key() -> None:
    settings = Settings(AUTH_ENABLED=True, APP_API_KEY="secret")

    try:
        get_request_context(x_api_key="wrong", x_user_id="user_1", settings=settings)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected invalid API key to fail")


def test_auth_enabled_returns_user_context() -> None:
    settings = Settings(AUTH_ENABLED=True, APP_API_KEY="secret")

    context = get_request_context(x_api_key="secret", x_user_id="user_1", settings=settings)

    assert context.user_id == "user_1"
