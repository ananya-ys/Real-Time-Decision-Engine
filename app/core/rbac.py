"""
RBAC — Role-Based Access Control for RTDE.

WHY THIS EXISTS:
The review found zero operator permission model. Every sensitive endpoint
was accessible to anyone. In production: operators need controls that
data scientists don't. Admins need controls that operators don't.

ROLE HIERARCHY:
  VIEWER   → read-only: decisions, rewards, dashboard
  OPERATOR → + kill switch, manual override, freeze exploration
  ADMIN    → + promote policy, RBAC management, delete versions
  SERVICE  → machine-to-machine: make decisions, submit rewards

IMPLEMENTATION:
- JWT tokens with role claim in payload
- API key alternative for SERVICE role (Celery workers, simulators)
- Every protected route uses Depends(require_role(Role.OPERATOR))
- Violation → 403 Forbidden with audit log entry

WHAT BREAKS IF WRONG:
- No RBAC: any user can trigger rollback, kill exploration, promote untested models
- Role check in router instead of dependency: skipped on new routes
- No audit log of RBAC denials: invisible attack surface
"""

from __future__ import annotations

import enum


class Role(str, enum.Enum):
    """User roles in ascending privilege order."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    SERVICE = "service"


# Role hierarchy: each role inherits all permissions of lower roles
_ROLE_HIERARCHY: dict[Role, set[Role]] = {
    Role.VIEWER: {Role.VIEWER},
    Role.OPERATOR: {Role.VIEWER, Role.OPERATOR},
    Role.ADMIN: {Role.VIEWER, Role.OPERATOR, Role.ADMIN},
    Role.SERVICE: {Role.SERVICE},  # service is separate, not hierarchical
}


class Permission(str, enum.Enum):
    """Fine-grained permissions mapped to roles."""

    # Read permissions
    READ_DECISIONS = "read:decisions"
    READ_REWARDS = "read:rewards"
    READ_DASHBOARD = "read:dashboard"
    READ_POLICIES = "read:policies"
    READ_AUDIT = "read:audit"

    # Decision making (SERVICE or OPERATOR+)
    MAKE_DECISIONS = "make:decisions"
    SUBMIT_REWARDS = "submit:rewards"

    # Operator controls (OPERATOR+)
    OPERATOR_KILL_SWITCH = "operator:kill_switch"
    OPERATOR_MANUAL_OVERRIDE = "operator:manual_override"
    OPERATOR_FREEZE_EXPLORATION = "operator:freeze_exploration"
    OPERATOR_CANARY_MANAGE = "operator:canary_manage"
    OPERATOR_ACKNOWLEDGE_ALERT = "operator:acknowledge_alert"

    # Policy lifecycle (ADMIN only)
    ADMIN_PROMOTE_POLICY = "admin:promote_policy"
    ADMIN_APPROVE_PROMOTION = "admin:approve_promotion"
    ADMIN_RETIRE_POLICY = "admin:retire_policy"
    ADMIN_MANAGE_RBAC = "admin:manage_rbac"
    ADMIN_TRIGGER_BACKTEST = "admin:trigger_backtest"
    ADMIN_DELETE_VERSION = "admin:delete_version"


# Mapping from role to granted permissions
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_DECISIONS,
        Permission.READ_REWARDS,
        Permission.READ_DASHBOARD,
        Permission.READ_POLICIES,
        Permission.READ_AUDIT,
    },
    Role.OPERATOR: {
        Permission.READ_DECISIONS,
        Permission.READ_REWARDS,
        Permission.READ_DASHBOARD,
        Permission.READ_POLICIES,
        Permission.READ_AUDIT,
        Permission.MAKE_DECISIONS,
        Permission.SUBMIT_REWARDS,
        Permission.OPERATOR_KILL_SWITCH,
        Permission.OPERATOR_MANUAL_OVERRIDE,
        Permission.OPERATOR_FREEZE_EXPLORATION,
        Permission.OPERATOR_CANARY_MANAGE,
        Permission.OPERATOR_ACKNOWLEDGE_ALERT,
    },
    Role.ADMIN: set(),  # populated below
    Role.SERVICE: {
        Permission.MAKE_DECISIONS,
        Permission.SUBMIT_REWARDS,
    },
}

# Fix forward reference in ADMIN permissions
ROLE_PERMISSIONS[Role.ADMIN] = {
    Permission.READ_DECISIONS,
    Permission.READ_REWARDS,
    Permission.READ_DASHBOARD,
    Permission.READ_POLICIES,
    Permission.READ_AUDIT,
    Permission.MAKE_DECISIONS,
    Permission.SUBMIT_REWARDS,
    Permission.OPERATOR_KILL_SWITCH,
    Permission.OPERATOR_MANUAL_OVERRIDE,
    Permission.OPERATOR_FREEZE_EXPLORATION,
    Permission.OPERATOR_CANARY_MANAGE,
    Permission.OPERATOR_ACKNOWLEDGE_ALERT,
    Permission.ADMIN_PROMOTE_POLICY,
    Permission.ADMIN_APPROVE_PROMOTION,
    Permission.ADMIN_RETIRE_POLICY,
    Permission.ADMIN_MANAGE_RBAC,
    Permission.ADMIN_TRIGGER_BACKTEST,
    Permission.ADMIN_DELETE_VERSION,
}


def has_permission(role: Role, permission: Permission) -> bool:
    """Check if a role grants a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


def role_can(role: Role, *permissions: Permission) -> bool:
    """Check if a role grants ALL specified permissions."""
    granted = ROLE_PERMISSIONS.get(role, set())
    return all(p in granted for p in permissions)
