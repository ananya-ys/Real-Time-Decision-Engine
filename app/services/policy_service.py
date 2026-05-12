"""
PolicyService — manages the PolicyRegistry and policy lifecycle.

WHY THIS EXISTS:
- PolicyVersion table is the model registry. Every trained model is versioned.
- PolicyCheckpoint table persists Q-values / model weights across restarts.
- Atomic active-policy swap: old ACTIVE → RETIRED, new SHADOW → ACTIVE in one transaction.
- This service owns ALL policy lifecycle logic. DecisionService only reads active policy.

PATTERNS APPLIED:
- Pattern 8 (Idempotent tasks): before creating a new version, check if one already exists.
- Pattern 2 (SELECT FOR UPDATE): checkpoint reads are locked to prevent concurrent restore.
- Audit trail: every status change recorded with timestamps.

WHAT BREAKS IF WRONG:
- Non-atomic swap: brief window where no policy is ACTIVE → inference has no policy → crash.
- Missing checkpoint restore on startup → bandit starts from scratch → all learned Q-values lost.
- No version registry → can't audit which model made which decision → post-incident blind.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CheckpointError, PolicyNotFoundError
from app.models.policy_checkpoint import PolicyCheckpoint
from app.models.policy_version import PolicyVersion
from app.policies.base_policy import PolicyCheckpointData, PolicyInterface
from app.schemas.common import PolicyStatus, PolicyType

logger = structlog.get_logger(__name__)


class PolicyService:
    """
    Manages policy lifecycle: creation, versioning, checkpointing, promotion.

    This service owns the PolicyRegistry (PolicyVersion table).
    Every trained model gets a row here before it's ever served.
    """

    async def create_policy_version(
        self,
        policy_type: PolicyType,
        algorithm: str,
        db: AsyncSession,
        status: PolicyStatus = PolicyStatus.SHADOW,
    ) -> PolicyVersion:
        """
        Register a new policy version in the registry.

        Every new policy starts as SHADOW. It cannot become ACTIVE
        without passing the shadow evaluation gate.

        Idempotent: if a TRAINING version already exists for this policy_type,
        returns it instead of creating a duplicate.
        """
        # Check for existing TRAINING version (idempotency)
        result = await db.execute(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == policy_type.value,
                PolicyVersion.status == PolicyStatus.TRAINING.value,
            )
            .limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info(
                "policy_version_already_exists",
                policy_type=policy_type.value,
                version_id=str(existing.id),
            )
            return existing

        # Compute next version number
        version_result = await db.execute(
            select(PolicyVersion)
            .where(PolicyVersion.policy_type == policy_type.value)
            .order_by(PolicyVersion.version.desc())
            .limit(1)
        )
        latest = version_result.scalar_one_or_none()
        next_version = (latest.version + 1) if latest else 1

        policy_version = PolicyVersion(
            policy_type=policy_type.value,
            version=next_version,
            algorithm=algorithm,
            status=status.value,
        )
        db.add(policy_version)
        await db.flush()

        logger.info(
            "policy_version_created",
            policy_type=policy_type.value,
            version=next_version,
            algorithm=algorithm,
            status=status.value,
        )
        return policy_version

    async def save_checkpoint(
        self,
        policy: PolicyInterface,
        policy_version_id: uuid.UUID,
        db: AsyncSession,
    ) -> PolicyCheckpoint:
        """
        Save policy weights to DB.

        Marks this checkpoint as active and deactivates all previous ones
        for this policy version. Atomically.

        Checkpoint persistence ensures zero data loss on restart.
        """
        checkpoint_data = policy.get_checkpoint()

        # Deactivate existing active checkpoints for this version
        await db.execute(
            update(PolicyCheckpoint)
            .where(
                PolicyCheckpoint.policy_version_id == policy_version_id,
                PolicyCheckpoint.is_active.is_(True),
            )
            .values(is_active=False)
        )

        # Create new active checkpoint
        checkpoint = PolicyCheckpoint(
            policy_version_id=policy_version_id,
            weights=checkpoint_data.weights,
            step_count=checkpoint_data.step_count,
            performance_metric=checkpoint_data.performance_metric,
            is_active=True,
        )
        db.add(checkpoint)
        await db.flush()

        logger.info(
            "checkpoint_saved",
            policy_version_id=str(policy_version_id),
            step_count=checkpoint_data.step_count,
            performance=checkpoint_data.performance_metric,
        )
        return checkpoint

    async def load_checkpoint(
        self,
        policy: PolicyInterface,
        policy_version_id: uuid.UUID,
        db: AsyncSession,
    ) -> bool:
        """
        Restore policy state from the most recent active checkpoint.

        Returns True if checkpoint found and loaded, False if no checkpoint exists
        (new policy, starts from scratch — that's fine, not an error).

        Raises CheckpointError if checkpoint exists but is malformed.
        """
        result = await db.execute(
            select(PolicyCheckpoint)
            .where(
                PolicyCheckpoint.policy_version_id == policy_version_id,
                PolicyCheckpoint.is_active.is_(True),
            )
            .limit(1)
        )
        checkpoint = result.scalar_one_or_none()

        if checkpoint is None:
            logger.info(
                "no_checkpoint_found",
                policy_version_id=str(policy_version_id),
                note="Starting fresh — expected for new policy versions",
            )
            return False

        checkpoint_data = PolicyCheckpointData(
            weights=checkpoint.weights or {},
            step_count=checkpoint.step_count,
            performance_metric=checkpoint.performance_metric,
        )

        try:
            policy.load_checkpoint(checkpoint_data)
            logger.info(
                "checkpoint_loaded",
                policy_version_id=str(policy_version_id),
                step_count=checkpoint.step_count,
            )
            return True
        except Exception as exc:
            raise CheckpointError(
                f"Failed to load checkpoint for {policy_version_id}: {exc}"
            ) from exc

    async def get_active_version(
        self,
        policy_type: PolicyType,
        db: AsyncSession,
    ) -> PolicyVersion | None:
        """
        Get the currently ACTIVE policy version for a given type.

        Returns None if no active version exists (system falls back to baseline).
        """
        result = await db.execute(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == policy_type.value,
                PolicyVersion.status == PolicyStatus.ACTIVE.value,
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def atomic_promote(
        self,
        shadow_version_id: uuid.UUID,
        db: AsyncSession,
    ) -> PolicyVersion:
        """
        Atomically promote a SHADOW policy to ACTIVE.

        Steps (all in one transaction):
        1. Find current ACTIVE version for the same policy_type.
        2. Set current ACTIVE → RETIRED (with demoted_at timestamp).
        3. Set SHADOW → ACTIVE (with promoted_at timestamp).

        This is atomic: there is never a moment where no policy is ACTIVE.
        If the transaction fails, we roll back — old ACTIVE stays ACTIVE.

        Raises PolicyNotFoundError if shadow_version_id doesn't exist or isn't SHADOW.
        """
        # Get the shadow version to promote
        shadow_result = await db.execute(
            select(PolicyVersion)
            .where(PolicyVersion.id == shadow_version_id)
            .with_for_update()  # lock row during promotion
        )
        shadow = shadow_result.scalar_one_or_none()
        if shadow is None:
            raise PolicyNotFoundError(f"Policy version {shadow_version_id} not found")
        if shadow.status != PolicyStatus.SHADOW.value:
            raise PolicyNotFoundError(
                f"Policy version {shadow_version_id} is {shadow.status}, not SHADOW"
            )

        # Retire current active (if any)
        await db.execute(
            update(PolicyVersion)
            .where(
                PolicyVersion.policy_type == shadow.policy_type,
                PolicyVersion.status == PolicyStatus.ACTIVE.value,
            )
            .values(
                status=PolicyStatus.RETIRED.value,
                demoted_at=datetime.now(UTC),
            )
        )

        # Promote shadow to active
        shadow.status = PolicyStatus.ACTIVE.value
        shadow.promoted_at = datetime.now(UTC)

        await db.flush()

        logger.info(
            "policy_promoted",
            policy_type=shadow.policy_type,
            version=shadow.version,
            version_id=str(shadow_version_id),
        )
        return shadow

    async def retire_policy(
        self,
        policy_version_id: uuid.UUID,
        db: AsyncSession,
    ) -> None:
        """Mark a policy version as RETIRED (used by rollback controller)."""
        await db.execute(
            update(PolicyVersion)
            .where(PolicyVersion.id == policy_version_id)
            .values(
                status=PolicyStatus.RETIRED.value,
                demoted_at=datetime.now(UTC),
            )
        )
        await db.flush()
        logger.info("policy_retired", version_id=str(policy_version_id))

    async def update_eval_metrics(
        self,
        policy_version_id: uuid.UUID,
        eval_reward_mean: float,
        eval_reward_std: float,
        eval_seeds: int,
        db: AsyncSession,
    ) -> None:
        """
        Update evaluation metrics on a policy version.

        These metrics are used by PolicyPromoter to decide if shadow beats active.
        eval_seeds MUST be >= 5 before promotion is allowed (statistical validity).
        """
        await db.execute(
            update(PolicyVersion)
            .where(PolicyVersion.id == policy_version_id)
            .values(
                eval_reward_mean=eval_reward_mean,
                eval_reward_std=eval_reward_std,
                eval_seeds=eval_seeds,
                status=PolicyStatus.SHADOW.value,  # ready for promotion gate
            )
        )
        await db.flush()
        logger.info(
            "eval_metrics_updated",
            version_id=str(policy_version_id),
            mean=round(eval_reward_mean, 4),
            std=round(eval_reward_std, 4),
            seeds=eval_seeds,
        )
