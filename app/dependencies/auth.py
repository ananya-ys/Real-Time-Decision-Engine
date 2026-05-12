"""
Auth dependencies — get_current_user, require_role, optional_auth.

WHY THIS EXISTS:
Every protected endpoint uses Depends(require_role(Role.OPERATOR)).
The dependency:
  1. Extracts token from Authorization header or X-API-Key header.
  2. Verifies it (JWT decode or API key lookup).
  3. Returns CurrentUser with verified identity.
  4. Raises 401 if missing, 403 if wrong role.

USAGE:
  @router.post("/operator/kill-switch/activate")
  async def activate(
      user: CurrentUser = Depends(require_role(Role.OPERATOR)),
  ) -> dict:
      ...

In development (APP_ENV=development):
  Auth is bypassed. All requests get ADMIN role.
  This preserves existing behavior during local dev.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.core.auth import CurrentUser, verify_access_token
from app.core.config import get_settings
from app.core.rbac import Permission, Role, has_permission

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Hardcoded API keys for services (in production: stored in DB/vault)
# Format: api_key_value → role
_KNOWN_API_KEYS: dict[str, Role] = {
    # Add real keys via environment or DB in production
}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    api_key: str | None = Security(_api_key_header),
) -> CurrentUser:
    """
    Extract and verify user identity from request.

    Priority:
    1. X-API-Key header → SERVICE role (machine-to-machine)
    2. Authorization: Bearer <jwt> → role from JWT payload
    3. Development mode → bypass auth, ADMIN role

    Raises:
        HTTPException(401): Missing or invalid credentials.
        HTTPException(403): Valid credentials, insufficient role.
    """
    settings = get_settings()

    # Development bypass
    if settings.app_env == "development":
        return CurrentUser(user_id="dev-user", role=Role.ADMIN, token_id="dev")

    # API key auth (for Celery workers, simulators)
    if api_key:
        role = _KNOWN_API_KEYS.get(api_key)
        if role:
            return CurrentUser(
                user_id=f"service:{api_key[:8]}",
                role=role,
                token_id="api_key",
            )
        raise HTTPException(status_code=401, detail="Invalid API key")

    # JWT auth (for human operators)
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Provide Bearer token or X-API-Key header.",
        )

    try:
        user = verify_access_token(credentials.credentials)
        return user
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def require_role(minimum_role: Role):  # type: ignore[no-untyped-def]
    """
    Dependency factory: enforce minimum role requirement.

    Usage: Depends(require_role(Role.OPERATOR))
    Returns CurrentUser if authorized.
    Raises 403 if role is insufficient.
    """

    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        role_hierarchy = {
            Role.VIEWER: 0,
            Role.SERVICE: 1,
            Role.OPERATOR: 2,
            Role.ADMIN: 3,
        }
        user_level = role_hierarchy.get(user.role, -1)
        required_level = role_hierarchy.get(minimum_role, 999)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Insufficient role. Required: {minimum_role.value}. Yours: {user.role.value}."
                ),
            )
        return user

    return _check


def require_permission(permission: Permission):  # type: ignore[no-untyped-def]
    """
    Fine-grained permission check.

    Usage: Depends(require_permission(Permission.ADMIN_PROMOTE_POLICY))
    """

    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not has_permission(user.role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Missing permission: {permission.value}",
            )
        return user

    return _check
