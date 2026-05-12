"""
Model registry — ALL models imported here so Base.metadata is complete.
Alembic autogenerate reads Base.metadata — missing imports = missing tables.
"""

from app.models.approval_request import ApprovalRequest
from app.models.decision_log import DecisionLog
from app.models.drift_event import DriftEvent
from app.models.environment_state import EnvironmentState
from app.models.exploration_guard_log import ExplorationGuardLog
from app.models.incident import Incident
from app.models.operator_event import OperatorEvent
from app.models.policy_checkpoint import PolicyCheckpoint
from app.models.policy_version import PolicyVersion
from app.models.reward_log import RewardLog
from app.models.scaling_action import ScalingAction

__all__ = [
    "ApprovalRequest",
    "DecisionLog",
    "DriftEvent",
    "EnvironmentState",
    "ExplorationGuardLog",
    "Incident",
    "OperatorEvent",
    "PolicyCheckpoint",
    "PolicyVersion",
    "RewardLog",
    "ScalingAction",
]
