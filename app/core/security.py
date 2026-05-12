"""
Security module — placeholder for auth infrastructure.

WHY THIS EXISTS:
- Establish the security module before any feature code.
- Adding auth 'later' means it gets skipped. Establish the pattern now.
- API key validation for internal service-to-service calls.

WHAT BREAKS IF WRONG:
- No security module = auth bolted on as afterthought = gaps.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify API key for protected endpoints.

    In production, this would validate against a key store.
    For the RTDE portfolio project, we use the secret_key as a simple gate.
    """
    if settings.app_env == "development":
        return "dev-user"
    if not api_key or api_key != settings.secret_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
