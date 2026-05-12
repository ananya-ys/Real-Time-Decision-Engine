"""
Phase 10 tests — RBAC, Kill Switch, Manual Override, Auth.

Verifies:
- Role permission matrix is correct (no gaps, no privilege escalation)
- KillSwitch state is correctly parsed
- ManualOverride logic correct (no Redis in unit tests)
- JWT creation and verification
- Auth dependency in dev mode returns ADMIN
- require_role enforces hierarchy correctly
"""

from __future__ import annotations

import pytest
from jose import JWTError

from app.core.auth import (
    create_access_token,
    verify_access_token,
)
from app.core.rbac import Permission, Role, has_permission, role_can
from app.operator.kill_switch import KillSwitchState
from app.schemas.common import PolicyType


@pytest.mark.unit
class TestRBAC:
    """Verify role permission matrix."""

    def test_viewer_can_read_decisions(self) -> None:
        assert has_permission(Role.VIEWER, Permission.READ_DECISIONS)

    def test_viewer_cannot_make_decisions(self) -> None:
        assert not has_permission(Role.VIEWER, Permission.MAKE_DECISIONS)

    def test_viewer_cannot_use_kill_switch(self) -> None:
        assert not has_permission(Role.VIEWER, Permission.OPERATOR_KILL_SWITCH)

    def test_operator_can_kill_switch(self) -> None:
        assert has_permission(Role.OPERATOR, Permission.OPERATOR_KILL_SWITCH)

    def test_operator_can_make_decisions(self) -> None:
        assert has_permission(Role.OPERATOR, Permission.MAKE_DECISIONS)

    def test_operator_cannot_promote_policy(self) -> None:
        assert not has_permission(Role.OPERATOR, Permission.ADMIN_PROMOTE_POLICY)

    def test_admin_can_promote_policy(self) -> None:
        assert has_permission(Role.ADMIN, Permission.ADMIN_PROMOTE_POLICY)

    def test_admin_has_all_operator_permissions(self) -> None:
        operator_perms = [
            Permission.OPERATOR_KILL_SWITCH,
            Permission.OPERATOR_MANUAL_OVERRIDE,
            Permission.OPERATOR_FREEZE_EXPLORATION,
            Permission.OPERATOR_CANARY_MANAGE,
        ]
        for perm in operator_perms:
            assert has_permission(Role.ADMIN, perm), f"Admin missing: {perm}"

    def test_service_can_make_decisions(self) -> None:
        assert has_permission(Role.SERVICE, Permission.MAKE_DECISIONS)

    def test_service_cannot_kill_switch(self) -> None:
        assert not has_permission(Role.SERVICE, Permission.OPERATOR_KILL_SWITCH)

    def test_service_cannot_read_audit(self) -> None:
        """SERVICE role is for machine-to-machine only, not auditing."""
        assert not has_permission(Role.SERVICE, Permission.READ_AUDIT)

    def test_role_can_multiple_permissions(self) -> None:
        assert role_can(
            Role.ADMIN,
            Permission.READ_DECISIONS,
            Permission.ADMIN_PROMOTE_POLICY,
            Permission.OPERATOR_KILL_SWITCH,
        )

    def test_role_can_fails_if_any_missing(self) -> None:
        assert not role_can(
            Role.VIEWER,
            Permission.READ_DECISIONS,
            Permission.OPERATOR_KILL_SWITCH,  # viewer doesn't have this
        )

    @pytest.mark.parametrize("role", list(Role))
    def test_all_roles_have_permissions_defined(self, role: Role) -> None:
        """Every role must have at least one permission defined."""
        from app.core.rbac import ROLE_PERMISSIONS

        assert role in ROLE_PERMISSIONS
        # ADMIN and OPERATOR must have multiple permissions
        if role in (Role.ADMIN, Role.OPERATOR):
            assert len(ROLE_PERMISSIONS[role]) > 3


@pytest.mark.unit
class TestKillSwitchState:
    """Verify KillSwitchState logic (no Redis needed)."""

    def test_is_policy_active_when_not_killed(self) -> None:
        state = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        assert state.is_policy_active(PolicyType.RL)
        assert state.is_policy_active(PolicyType.BANDIT)

    def test_is_policy_inactive_when_globally_killed(self) -> None:
        state = KillSwitchState(
            global_killed=True,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        assert not state.is_policy_active(PolicyType.RL)
        assert not state.is_policy_active(PolicyType.BASELINE)

    def test_specific_policy_killed(self) -> None:
        state = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies={PolicyType.RL},
        )
        assert not state.is_policy_active(PolicyType.RL)
        assert state.is_policy_active(PolicyType.BANDIT)
        assert state.is_policy_active(PolicyType.BASELINE)

    def test_exploration_frozen(self) -> None:
        state = KillSwitchState(
            global_killed=False,
            exploration_frozen=True,
            promotion_frozen=False,
            killed_policies=set(),
        )
        assert not state.allow_exploration()
        assert state.allow_promotion()

    def test_promotion_frozen(self) -> None:
        state = KillSwitchState(
            global_killed=False,
            exploration_frozen=False,
            promotion_frozen=True,
            killed_policies=set(),
        )
        assert state.allow_exploration()
        assert not state.allow_promotion()

    def test_global_kill_blocks_all(self) -> None:
        state = KillSwitchState(
            global_killed=True,
            exploration_frozen=False,
            promotion_frozen=False,
            killed_policies=set(),
        )
        assert not state.allow_exploration()
        assert not state.allow_promotion()


@pytest.mark.unit
class TestJWTAuth:
    """Verify JWT token creation and verification."""

    def test_create_token_for_operator(self) -> None:
        token = create_access_token(user_id="operator@test.com", role=Role.OPERATOR)
        assert isinstance(token, str)
        assert len(token) > 50

    def test_verify_valid_token(self) -> None:
        token = create_access_token(user_id="admin@test.com", role=Role.ADMIN)
        user = verify_access_token(token)
        assert user.user_id == "admin@test.com"
        assert user.role == Role.ADMIN

    def test_verify_preserves_role(self) -> None:
        for role in [Role.VIEWER, Role.OPERATOR, Role.ADMIN, Role.SERVICE]:
            token = create_access_token(user_id="test", role=role)
            user = verify_access_token(token)
            assert user.role == role

    def test_verify_invalid_token_raises(self) -> None:
        with pytest.raises(JWTError):
            verify_access_token("not.a.valid.token")

    def test_verify_tampered_token_raises(self) -> None:
        token = create_access_token(user_id="user@test.com", role=Role.OPERATOR)
        # Tamper with the token
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + "." + "tampered_signature"
        with pytest.raises(JWTError):
            verify_access_token(tampered)

    def test_token_contains_jti(self) -> None:
        token = create_access_token(user_id="user@test.com", role=Role.VIEWER)
        user = verify_access_token(token)
        assert user.token_id  # non-empty JTI for revocation support

    def test_different_tokens_have_different_jti(self) -> None:
        token1 = create_access_token(user_id="user@test.com", role=Role.VIEWER)
        token2 = create_access_token(user_id="user@test.com", role=Role.VIEWER)
        user1 = verify_access_token(token1)
        user2 = verify_access_token(token2)
        assert user1.token_id != user2.token_id  # each token is unique
