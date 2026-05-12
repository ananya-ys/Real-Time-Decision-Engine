"""
Authentication — JWT + API key dual-mode auth.

WHY THIS EXISTS:
Phase 10 placeholder used get_current_actor() which returned a hardcoded dict.
Production requires real identity: signed JWT tokens for human operators,
API keys for Celery workers and the simulation service.

JWT PAYLOAD:
  {
    "sub":  "user@company.com",       # subject (user ID)
    "role": "operator",                # role from Role enum
    "exp":  1735689600,                # expiry unix timestamp
    "iat":  1735603200,                # issued at
    "jti":  "uuid4"                    # token ID for blacklisting
  }

API KEY FORMAT:
  rtde_<environment>_<random_hex_32>
  e.g. rtde_prod_a3f8c2d1...

FLOW:
  Request → extract_token() → verify JWT or API key
  → get role → inject into Depends(get_current_user)
  → route uses user.role for RBAC check

WHAT BREAKS IF WRONG:
- No expiry: stolen token valid forever.
- No jti: can't revoke individual tokens (need to rotate SECRET_KEY).
- Role in DB instead of JWT: DB round-trip on every request → latency.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.rbac import Role

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours for operator tokens
_API_KEY_PREFIX = "rtde_"


class TokenPayload(BaseModel):
    """Decoded JWT payload."""

    sub: str  # user identifier
    role: str  # role name
    exp: int  # expiry timestamp
    iat: int  # issued at
    jti: str  # token ID


class CurrentUser(BaseModel):
    """Resolved user identity after token verification."""

    user_id: str
    role: Role
    token_id: str


def create_access_token(
    user_id: str,
    role: Role,
    expire_minutes: int = _ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        user_id: Unique identifier (email, UUID, username).
        role: User's RBAC role.
        expire_minutes: Token validity period.

    Returns:
        Signed JWT string.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expire_minutes)

    payload = {
        "sub": user_id,
        "role": role.value,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def verify_access_token(token: str) -> CurrentUser:
    """
    Verify and decode a JWT access token.

    Args:
        token: JWT string from Authorization header.

    Returns:
        CurrentUser with decoded identity.

    Raises:
        JWTError: If token is invalid, expired, or tampered.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise JWTError(f"Invalid token: {exc}") from exc

    role_str = payload.get("role", "")
    try:
        role = Role(role_str)
    except ValueError as exc:
        raise JWTError(f"Unknown role in token: {role_str}") from exc

    return CurrentUser(
        user_id=payload["sub"],
        role=role,
        token_id=payload.get("jti", ""),
    )


def create_api_key(environment: str = "prod") -> str:
    """
    Generate an API key for machine-to-machine auth (Celery workers, simulators).

    Format: rtde_{env}_{random_hex_32}
    """
    import secrets

    return f"{_API_KEY_PREFIX}{environment}_{secrets.token_hex(32)}"


def is_api_key(token: str) -> bool:
    """Check if the provided token is an API key (not a JWT)."""
    return token.startswith(_API_KEY_PREFIX)
