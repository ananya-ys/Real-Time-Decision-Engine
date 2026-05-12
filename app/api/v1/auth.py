"""
Auth API — login, token generation, API key management.

ENDPOINTS:
  POST /api/v1/auth/token          → exchange credentials for JWT
  POST /api/v1/auth/api-keys       → generate API key (ADMIN only)
  GET  /api/v1/auth/me             → return current user info
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import CurrentUser, Role, create_access_token, create_api_key
from app.core.config import get_settings
from app.dependencies.auth import get_current_user, require_role

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = structlog.get_logger(__name__)


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"  # requested role


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class ApiKeyResponse(BaseModel):
    api_key: str
    role: str
    environment: str


@router.post("/token", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    """
    Exchange credentials for a JWT access token.

    In development: any credentials are accepted.
    In production: validate against user store (DB, LDAP, OAuth provider).
    """
    settings = get_settings()

    if settings.app_env == "development":
        # Dev mode: accept any credentials, use requested role
        try:
            role = Role(body.role)
        except ValueError:
            role = Role.VIEWER

        token = create_access_token(user_id=body.username, role=role)
        logger.info("dev_login", username=body.username, role=role.value)

        return TokenResponse(
            access_token=token,
            role=role.value,
            expires_in_minutes=480,
        )

    # Production: implement real credential validation here
    # Example: check DB, LDAP, or delegate to OAuth
    raise HTTPException(
        status_code=501,
        detail=(
            "Real credential validation not implemented. "
            "In production, replace this with your identity provider."
        ),
    )


@router.post("/api-keys", response_model=ApiKeyResponse)
async def generate_api_key(
    environment: str = "prod",
    role: str = "service",
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> ApiKeyResponse:
    """
    Generate an API key for machine-to-machine authentication.
    Requires ADMIN role.

    In production: store the key hash in DB, not the raw key.
    Return the raw key once — it cannot be retrieved again.
    """
    key = create_api_key(environment=environment)
    logger.warning(
        "api_key_generated",
        generated_by=admin.user_id,
        environment=environment,
        role=role,
        key_prefix=key[:20],
    )

    return ApiKeyResponse(api_key=key, role=role, environment=environment)


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return current authenticated user's identity."""
    return {
        "user_id": user.user_id,
        "role": user.role.value,
        "token_id": user.token_id,
    }
