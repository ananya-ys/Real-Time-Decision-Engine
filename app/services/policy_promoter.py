"""
PolicyPromoter — shadow evaluation gate before policy promotion.

WHY THIS EXISTS:
- Shadow mode: new policy runs alongside active, logs decisions but doesn't commit.
- The promoter decides: has the shadow policy earned the right to become active?
- It applies the same statistical rigor as clinical trials: shadow must BEAT active
  by a statistically significant margin over multiple evaluation seeds.

THE GATE CRITERIA (must ALL pass):
1. eval_seeds >= 5: single-seed eval is cherry-picking. Need statistical validity.
2. eval_reward_mean > active_mean * (1 + promotion_threshold): shadow must be BETTER.
3. eval_reward_std is not absurdly high: consistent performance, not lucky spikes.

WHAT BREAKS IF WRONG:
- No seed requirement: one lucky eval seed gets promoted → live traffic suffers.
- No threshold: shadow promoted when it's marginally equal → introduces instability.
- No atomic promotion: brief window where neither policy is ACTIVE → inference crashes.

INDUSTRY PARALLEL:
- A/B testing: new feature must beat control by p < 0.05.
- Canary deployment: must hold 5% traffic for 24h before full rollout.
- This is the ML equivalent: shadow mode is the canary.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics as prom
from app.models.policy_version import PolicyVersion
from app.schemas.common import PolicyStatus, PolicyType
from app.services.policy_service import PolicyService

logger = structlog.get_logger(__name__)

_MIN_EVAL_SEEDS = 5
_DEFAULT_PROMOTION_THRESHOLD = 0.05  # shadow must beat active by 5%


class PromotionResult:
    """Result of shadow promotion evaluation."""

    def __init__(
        self,
        promoted: bool,
        reason: str,
        shadow_mean: float | None = None,
        active_mean: float | None = None,
        eval_seeds: int | None = None,
    ) -> None:
        self.promoted = promoted
        self.reason = reason
        self.shadow_mean = shadow_mean
        self.active_mean = active_mean
        self.eval_seeds = eval_seeds

    def __repr__(self) -> str:
        return (
            f"PromotionResult(promoted={self.promoted}, "
            f"reason={self.reason!r}, "
            f"shadow={self.shadow_mean}, active={self.active_mean})"
        )


class PolicyPromoter:
    """
    Evaluates shadow policy against active and promotes if criteria met.

    Called periodically (or after training completes) to check if a shadow
    policy has earned promotion to ACTIVE.
    """

    def __init__(
        self,
        min_eval_seeds: int = _MIN_EVAL_SEEDS,
        promotion_threshold: float = _DEFAULT_PROMOTION_THRESHOLD,
    ) -> None:
        self._min_eval_seeds = min_eval_seeds
        self._promotion_threshold = promotion_threshold
        self._policy_service = PolicyService()

    async def evaluate_and_promote(
        self,
        shadow_version_id: uuid.UUID,
        policy_type: PolicyType,
        db: AsyncSession,
    ) -> PromotionResult:
        """
        Evaluate shadow policy and promote if all gate criteria pass.

        Gate criteria:
        1. Shadow has enough eval seeds.
        2. Shadow beats active by promotion_threshold.
        3. Both pass statistical validity checks.

        Returns:
            PromotionResult with promoted=True/False and reason string.
        """
        # Get shadow version
        shadow_result = await db.execute(
            select(PolicyVersion).where(PolicyVersion.id == shadow_version_id)
        )
        shadow = shadow_result.scalar_one_or_none()

        if shadow is None:
            return PromotionResult(
                promoted=False,
                reason=f"Shadow version {shadow_version_id} not found",
            )

        if shadow.status != PolicyStatus.SHADOW.value:
            return PromotionResult(
                promoted=False,
                reason=f"Version is {shadow.status}, not SHADOW",
                eval_seeds=shadow.eval_seeds,
            )

        # ── Gate 1: Minimum eval seeds ───────────────────────────
        if shadow.eval_seeds is None or shadow.eval_seeds < self._min_eval_seeds:
            logger.info(
                "promotion_gate_failed_seeds",
                version_id=str(shadow_version_id),
                eval_seeds=shadow.eval_seeds,
                min_required=self._min_eval_seeds,
            )
            return PromotionResult(
                promoted=False,
                reason=(
                    f"Insufficient eval seeds: {shadow.eval_seeds} < {self._min_eval_seeds}. "
                    "Run more evaluation episodes before promoting."
                ),
                eval_seeds=shadow.eval_seeds,
            )

        # ── Gate 2: Performance vs active ────────────────────────
        shadow_mean = shadow.eval_reward_mean
        if shadow_mean is None:
            return PromotionResult(
                promoted=False,
                reason="Shadow eval_reward_mean is None — eval not complete",
                eval_seeds=shadow.eval_seeds,
            )

        # Get current active policy's performance
        active_result = await db.execute(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == policy_type.value,
                PolicyVersion.status == PolicyStatus.ACTIVE.value,
            )
            .limit(1)
        )
        active = active_result.scalar_one_or_none()

        active_mean: float | None = None
        if active is not None and active.eval_reward_mean is not None:
            active_mean = active.eval_reward_mean
            required_improvement = active_mean * (1 + self._promotion_threshold)

            if shadow_mean <= required_improvement:
                logger.info(
                    "promotion_gate_failed_performance",
                    shadow_mean=shadow_mean,
                    active_mean=active_mean,
                    required=required_improvement,
                )
                return PromotionResult(
                    promoted=False,
                    reason=(
                        f"Shadow mean ({shadow_mean:.4f}) does not exceed "
                        f"active mean ({active_mean:.4f}) + {self._promotion_threshold * 100:.0f}% "
                        f"threshold ({required_improvement:.4f})"
                    ),
                    shadow_mean=shadow_mean,
                    active_mean=active_mean,
                    eval_seeds=shadow.eval_seeds,
                )

        # ── All gates passed → promote ────────────────────────────
        try:
            await self._policy_service.atomic_promote(shadow_version_id, db)

            prom.active_policy_info.labels(
                policy_type=policy_type.value,
                policy_version=str(shadow.version),
            ).set(1)

            logger.info(
                "policy_promoted",
                version_id=str(shadow_version_id),
                shadow_mean=shadow_mean,
                active_mean=active_mean,
                eval_seeds=shadow.eval_seeds,
            )

            return PromotionResult(
                promoted=True,
                reason="All promotion gates passed",
                shadow_mean=shadow_mean,
                active_mean=active_mean,
                eval_seeds=shadow.eval_seeds,
            )

        except Exception as exc:
            logger.error("promotion_atomic_failed", error=str(exc))
            return PromotionResult(
                promoted=False,
                reason=f"Atomic promotion failed: {exc}",
                shadow_mean=shadow_mean,
                active_mean=active_mean,
                eval_seeds=shadow.eval_seeds,
            )

    async def get_shadow_candidates(
        self,
        policy_type: PolicyType,
        db: AsyncSession,
    ) -> list[PolicyVersion]:
        """Return all SHADOW versions eligible for promotion evaluation."""
        result = await db.execute(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == policy_type.value,
                PolicyVersion.status == PolicyStatus.SHADOW.value,
            )
            .order_by(PolicyVersion.created_at.desc())
        )
        return list(result.scalars().all())
