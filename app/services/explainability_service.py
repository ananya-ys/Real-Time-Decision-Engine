"""
ExplainabilityService — deep decision explainability.

WHY THIS EXISTS:
The review: "decision lab shows actions and Q-values, but not enough to answer:
why this action, why now, what changed, what was suppressed by guards,
and what would have happened under a nearby alternative."

WHAT WE EXPLAIN:
  1. Why this action: Q-value ordering + confidence spread analysis
  2. Why now: state features that drove the decision (feature attribution)
  3. What ExplorationGuard did: allowed or suppressed, why
  4. What alternatives existed: counterfactual for all 5 actions
  5. How baseline would have decided (reference comparison)
  6. Risk assessment: was this action near a safety boundary?

OUTPUT: structured explanation usable in UI "explain this decision" panel.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.policies.baseline_policy import BaselinePolicy
from app.schemas.common import ActionType
from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)

_ACTIONS = [
    ActionType.SCALE_UP_3,
    ActionType.SCALE_UP_1,
    ActionType.HOLD,
    ActionType.SCALE_DOWN_1,
    ActionType.SCALE_DOWN_3,
]

_ACTION_DELTAS = {
    ActionType.SCALE_UP_3: 3,
    ActionType.SCALE_UP_1: 1,
    ActionType.HOLD: 0,
    ActionType.SCALE_DOWN_1: -1,
    ActionType.SCALE_DOWN_3: -3,
}


class DecisionExplanation:
    """Complete explanation for a single decision."""

    def __init__(
        self,
        chosen_action: str,
        why_chosen: str,
        feature_attribution: dict[str, str],
        guard_explanation: str,
        alternatives: list[dict[str, Any]],
        baseline_comparison: dict[str, Any],
        risk_assessment: dict[str, Any],
        confidence_analysis: dict[str, Any],
    ) -> None:
        self.chosen_action = chosen_action
        self.why_chosen = why_chosen
        self.feature_attribution = feature_attribution
        self.guard_explanation = guard_explanation
        self.alternatives = alternatives
        self.baseline_comparison = baseline_comparison
        self.risk_assessment = risk_assessment
        self.confidence_analysis = confidence_analysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_action": self.chosen_action,
            "why_chosen": self.why_chosen,
            "feature_attribution": self.feature_attribution,
            "guard_explanation": self.guard_explanation,
            "alternatives": self.alternatives,
            "baseline_comparison": self.baseline_comparison,
            "risk_assessment": self.risk_assessment,
            "confidence_analysis": self.confidence_analysis,
        }


class ExplainabilityService:
    """
    Generates human-readable explanations for scaling decisions.
    """

    def __init__(self) -> None:
        self._baseline = BaselinePolicy()

    async def explain(
        self,
        state: SystemState,
        chosen_action: str,
        q_values: dict[str, float] | None = None,
        explore_allowed: bool = True,
        suppression_reason: str | None = None,
        policy_type: str = "BASELINE",
    ) -> DecisionExplanation:
        """
        Generate a full explanation for a decision.

        Args:
            state: The system state at decision time.
            chosen_action: The action that was taken.
            q_values: Q-values for all actions (if available from ML policy).
            explore_allowed: Whether ExplorationGuard allowed exploration.
            suppression_reason: Why guard suppressed, if it did.
            policy_type: BASELINE, BANDIT, or RL.

        Returns:
            DecisionExplanation with all analysis components.
        """
        # 1. Why chosen
        why = self._explain_choice(state, chosen_action, q_values, policy_type)

        # 2. Feature attribution
        attribution = self._attribute_features(state, chosen_action)

        # 3. Guard explanation
        guard_exp = self._explain_guard(explore_allowed, suppression_reason)

        # 4. Alternatives (reward-based counterfactuals)
        alternatives = self._compute_alternatives(state, chosen_action, q_values)

        # 5. Baseline comparison
        baseline_comp = await self._compare_to_baseline(state, chosen_action)

        # 6. Risk assessment
        risk = self._assess_risk(state, chosen_action)

        # 7. Confidence analysis
        confidence = self._analyze_confidence(q_values, chosen_action)

        return DecisionExplanation(
            chosen_action=chosen_action,
            why_chosen=why,
            feature_attribution=attribution,
            guard_explanation=guard_exp,
            alternatives=alternatives,
            baseline_comparison=baseline_comp,
            risk_assessment=risk,
            confidence_analysis=confidence,
        )

    def _explain_choice(
        self,
        state: SystemState,
        action: str,
        q_values: dict[str, float] | None,
        policy_type: str,
    ) -> str:
        if policy_type == "BASELINE":
            return self._baseline_narrative(state, action)

        if q_values:
            sorted_q = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            if sorted_q[0][0] == action:
                runner_up = sorted_q[1][0] if len(sorted_q) > 1 else "HOLD"
                spread = sorted_q[0][1] - sorted_q[1][1] if len(sorted_q) > 1 else 0
                return (
                    f"The {policy_type} policy selected {action} because it had the "
                    f"highest Q-value ({q_values.get(action, 0):.4f}), with a "
                    f"confidence spread of {spread:.4f} over the second-best option "
                    f"({runner_up}). {'High confidence' if spread > 0.5 else 'Low confidence — policies were close'}."
                )

        return f"Action {action} was selected by the {policy_type} policy."

    def _baseline_narrative(self, state: SystemState, action: str) -> str:
        cpu = state.cpu_utilization
        latency = state.p99_latency_ms

        narratives = {
            "SCALE_UP_3": f"CPU at {cpu:.0%} (above 90% critical threshold) or latency at {latency:.0f}ms (above 800ms critical). Aggressive scale-up required.",
            "SCALE_UP_1": f"CPU at {cpu:.0%} (above 80% warning threshold) or latency at {latency:.0f}ms (above 500ms). Moderate scale-up triggered.",
            "HOLD": f"CPU at {cpu:.0%} and latency at {latency:.0f}ms are within acceptable ranges. No scaling action required.",
            "SCALE_DOWN_1": f"CPU at {cpu:.0%} (below 30% low threshold) and latency at {latency:.0f}ms (below 100ms). Resources underutilized, scale down to save cost.",
            "SCALE_DOWN_3": f"CPU at {cpu:.0%} and latency at {latency:.0f}ms very low. Aggressive scale-down for significant cost savings.",
        }
        return narratives.get(action, f"Baseline threshold rule selected {action}.")

    def _attribute_features(self, state: SystemState, action: str) -> dict[str, str]:
        """Which features most influenced this decision."""
        attribution = {}

        # CPU influence
        if state.cpu_utilization > 0.9:
            attribution["cpu_utilization"] = (
                f"CRITICAL ({state.cpu_utilization:.0%}) — primary scale-up driver"
            )
        elif state.cpu_utilization > 0.8:
            attribution["cpu_utilization"] = (
                f"HIGH ({state.cpu_utilization:.0%}) — secondary scale-up driver"
            )
        elif state.cpu_utilization < 0.3:
            attribution["cpu_utilization"] = (
                f"LOW ({state.cpu_utilization:.0%}) — primary scale-down driver"
            )
        else:
            attribution["cpu_utilization"] = f"NORMAL ({state.cpu_utilization:.0%}) — neutral"

        # Latency influence
        if state.p99_latency_ms > 800:
            attribution["p99_latency_ms"] = (
                f"CRITICAL ({state.p99_latency_ms:.0f}ms) — SLA breached"
            )
        elif state.p99_latency_ms > 500:
            attribution["p99_latency_ms"] = (
                f"ELEVATED ({state.p99_latency_ms:.0f}ms) — approaching SLA limit"
            )
        elif state.p99_latency_ms < 100:
            attribution["p99_latency_ms"] = (
                f"LOW ({state.p99_latency_ms:.0f}ms) — system underloaded"
            )
        else:
            attribution["p99_latency_ms"] = (
                f"ACCEPTABLE ({state.p99_latency_ms:.0f}ms) — within target"
            )

        # Request rate
        attribution["request_rate"] = f"{state.request_rate:.0f} RPS — " + (
            "HIGH LOAD"
            if state.request_rate > 5000
            else "MODERATE"
            if state.request_rate > 1000
            else "LOW"
        )

        # Instance count context
        pct_capacity = state.instance_count / 20.0  # assuming max 20
        attribution["instance_count"] = (
            f"{state.instance_count} instances ({pct_capacity:.0%} of max capacity)"
        )

        # Time context
        hour = state.hour_of_day
        if 9 <= hour <= 17:
            attribution["hour_of_day"] = f"Business hours ({hour}:00) — peak traffic window"
        elif 0 <= hour <= 6:
            attribution["hour_of_day"] = f"Off-peak hours ({hour}:00) — low traffic expected"
        else:
            attribution["hour_of_day"] = f"Shoulder hours ({hour}:00)"

        return attribution

    def _explain_guard(self, explore_allowed: bool, suppression_reason: str | None) -> str:
        if explore_allowed:
            return (
                "ExplorationGuard allowed exploration. System was in a stable state: "
                "latency within normal range, SLA violation rate acceptable, "
                "request rate below high-load threshold."
            )
        reason_map = {
            "HIGH_LATENCY": "latency exceeded the warning threshold — exploiting best-known action to avoid risk",
            "HIGH_LOAD": "request rate exceeded high-load threshold — protecting SLA during traffic spike",
            "SLA_VIOLATION_STREAK": "consecutive SLA violations detected — suppressing exploration to stabilize",
        }
        reason_text = reason_map.get(
            suppression_reason or "",
            f"system stress detected ({suppression_reason})",
        )
        return (
            f"ExplorationGuard SUPPRESSED exploration: {reason_text}. "
            "The policy is exploiting its best-known action rather than exploring alternatives. "
            "This is a safety protection — exploration during stress can worsen SLA violations."
        )

    def _compute_alternatives(
        self,
        state: SystemState,
        chosen_action: str,
        q_values: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """Compare all 5 possible actions."""
        from app.services.reward_service import RewardService

        svc = RewardService()
        alternatives = []

        for action in _ACTIONS:
            delta = _ACTION_DELTAS[action]
            resulting_instances = max(1, min(20, state.instance_count + delta))
            reward = svc.compute_reward(
                p99_latency_ms=state.p99_latency_ms,
                instance_count=resulting_instances,
                last_action_delta=delta,
            )

            alt: dict[str, Any] = {
                "action": action.value,
                "resulting_instances": resulting_instances,
                "estimated_reward": round(reward.total_reward, 4),
                "sla_would_be_violated": reward.sla_violated,
                "is_chosen": action.value == chosen_action,
            }

            if q_values and action.value in q_values:
                alt["q_value"] = round(q_values[action.value], 4)

            alternatives.append(alt)

        # Sort by estimated reward descending
        alternatives.sort(key=lambda x: x["estimated_reward"], reverse=True)
        return alternatives

    async def _compare_to_baseline(self, state: SystemState, chosen_action: str) -> dict[str, Any]:
        """What would the baseline deterministic policy have chosen?"""
        baseline_decision = await self._baseline.decide(state, explore=False)
        same = baseline_decision.action.value == chosen_action

        return {
            "baseline_would_choose": baseline_decision.action.value,
            "baseline_would_set_instances": baseline_decision.instances_after,
            "agrees_with_baseline": same,
            "interpretation": (
                "ML policy agrees with deterministic baseline."
                if same
                else (
                    f"ML policy chose {chosen_action} vs baseline's "
                    f"{baseline_decision.action.value}. "
                    "This divergence may reflect learned patterns beyond threshold rules."
                )
            ),
        }

    def _assess_risk(self, state: SystemState, action: str) -> dict[str, Any]:
        """Is this action near safety boundaries?"""
        risks = []
        delta = _ACTION_DELTAS.get(ActionType(action), 0)
        new_instances = state.instance_count + delta

        if new_instances <= 1:
            risks.append("CRITICAL: Would reduce to minimum instances (1). No redundancy.")
        elif new_instances <= 2:
            risks.append("HIGH: Very few instances — single failure point risk.")

        if new_instances >= 20:
            risks.append("INFO: At maximum instance count ceiling.")

        if state.p99_latency_ms > 500 and delta < 0:
            risks.append("WARNING: Scaling down during high latency — may worsen SLA breach.")

        if state.cpu_utilization > 0.85 and delta <= 0:
            risks.append("WARNING: Not scaling up despite high CPU — monitor closely.")

        return {
            "risk_level": "CRITICAL"
            if any("CRITICAL" in r for r in risks)
            else "HIGH"
            if any("HIGH" in r or "WARNING" in r for r in risks)
            else "LOW",
            "risk_factors": risks or ["No significant risks identified."],
            "resulting_instances": new_instances,
            "instances_delta": delta,
        }

    def _analyze_confidence(
        self, q_values: dict[str, float] | None, chosen_action: str
    ) -> dict[str, Any]:
        if not q_values:
            return {
                "confidence_level": "unknown",
                "spread": None,
                "interpretation": "No Q-values available (baseline or early-stage policy).",
            }

        sorted_q = sorted(q_values.values(), reverse=True)
        spread = sorted_q[0] - sorted_q[1] if len(sorted_q) > 1 else 0
        max_q = sorted_q[0]

        if spread > 1.0:
            level = "HIGH"
            interpretation = "Policy is highly confident in this action."
        elif spread > 0.3:
            level = "MEDIUM"
            interpretation = "Policy has moderate confidence. Second-best action was considered."
        else:
            level = "LOW"
            interpretation = "Policy has low confidence. Multiple actions had similar Q-values. Outcome is uncertain."

        return {
            "confidence_level": level,
            "spread": round(spread, 4),
            "best_q_value": round(max_q, 4),
            "chosen_q_value": round(q_values.get(chosen_action, 0), 4),
            "interpretation": interpretation,
        }
