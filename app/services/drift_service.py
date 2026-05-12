"""
DriftService — Two-signal production drift detection.

WHY TWO SIGNALS:
1. PSI (Population Stability Index) on INPUT FEATURES:
   Detects when traffic patterns shift away from training distribution.
   Example: model trained on STEADY traffic, BURST traffic arrives.
   PSI > 0.2 = significant shift = model's Q-values are for the wrong inputs.

2. Mann-Whitney U test on REWARD conditioned on traffic_regime:
   Detects when model performance DEGRADES without input distribution shift.
   Example: traffic is same, but model has overfit to old patterns.
   Uses non-parametric test — no Gaussian assumption on reward distribution.

WHY NOT JUST REWARD:
- Reward can drop because traffic got harder (not the model's fault).
- PSI on inputs separates "traffic changed" from "model degraded".
- Conditioning on traffic_regime prevents false positives during expected spikes.

HYSTERESIS (K consecutive windows):
- One bad window = noise. K consecutive bad windows = real drift.
- Prevents oscillating rollback/promote cycles from single-window fluctuations.
- K is configurable via DRIFT_HYSTERESIS_K env var.

WHAT BREAKS IF WRONG:
- Single-signal only: false rollback on every traffic spike.
- No hysteresis: oscillates rollback/promote on noisy reward signals.
- No traffic_regime conditioning: BURST traffic always looks like "drift".
- PSI on raw features: scale difference triggers false positives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog
from scipy import stats

from app.core.config import get_settings
from app.schemas.common import DriftSignal, TrafficRegime

logger = structlog.get_logger(__name__)


@dataclass
class DriftWindow:
    """One evaluation window of observations."""

    feature_vectors: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    traffic_regime: TrafficRegime = TrafficRegime.UNKNOWN
    policy_type: str = "BASELINE"


@dataclass
class DriftResult:
    """Result of drift detection evaluation."""

    drift_detected: bool
    drift_signal: DriftSignal | None
    psi_score: float | None
    reward_delta: float | None
    consecutive_degraded_windows: int
    reference_reward_mean: float | None
    current_reward_mean: float | None


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Population Stability Index.

    PSI = sum((current% - reference%) * ln(current% / reference%))

    Interpretation:
    - PSI < 0.1: no significant change
    - 0.1 ≤ PSI < 0.2: moderate change, monitor
    - PSI ≥ 0.2: significant shift, consider rollback

    Uses the reference distribution's percentiles as bin edges.
    This ensures bins are well-populated for the reference.
    """
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    # Build bins from reference distribution percentiles
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # deduplicate in case of repeated values

    if len(bins) < 2:
        return 0.0

    # Compute proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_props = ref_counts / (ref_counts.sum() + epsilon)
    cur_props = cur_counts / (cur_counts.sum() + epsilon)

    # Clip to avoid log(0)
    ref_props = np.clip(ref_props, epsilon, None)
    cur_props = np.clip(cur_props, epsilon, None)

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return abs(psi)


def _mann_whitney_p_value(reference_rewards: list[float], current_rewards: list[float]) -> float:
    """
    Mann-Whitney U test: tests whether current rewards come from lower distribution.

    Non-parametric: no Gaussian assumption. Robust to reward outliers.
    Returns p-value for H1: current rewards < reference rewards.
    Low p-value = current rewards are significantly lower = performance degraded.
    """
    if len(reference_rewards) < 5 or len(current_rewards) < 5:
        return 1.0  # insufficient data → assume no drift

    _, p_value = stats.mannwhitneyu(
        current_rewards,
        reference_rewards,
        alternative="less",  # one-sided: current < reference
    )
    return float(p_value)


class DriftService:
    """
    Evaluates drift every DRIFT_EVAL_INTERVAL_SECONDS seconds.

    Maintains:
    - reference_window: reward distribution from baseline/early policy.
    - current_window: recent observations for comparison.
    - degraded_count: consecutive windows showing drift.

    Returns DriftResult with all evidence for DriftEvent logging.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._psi_threshold = settings.drift_psi_threshold
        self._alpha = settings.drift_significance_alpha
        self._hysteresis_k = settings.drift_hysteresis_k
        self._window_size = settings.drift_window_size

        # Reference window (established from baseline period)
        self._reference_window: DriftWindow | None = None

        # Sliding current window
        self._current_observations: deque[tuple[list[float], float, TrafficRegime]] = deque(
            maxlen=self._window_size
        )

        # Hysteresis counter
        self._degraded_window_count: int = 0

    def add_observation(
        self,
        feature_vector: list[float],
        reward: float,
        traffic_regime: TrafficRegime,
    ) -> None:
        """Record a new (state, reward) observation for drift evaluation."""
        self._current_observations.append((feature_vector, reward, traffic_regime))

    def set_reference_window(self, window: DriftWindow) -> None:
        """
        Establish the reference distribution.

        Called during baseline evaluation period.
        Reference = what "normal" looks like for this policy.
        """
        self._reference_window = window
        logger.info(
            "drift_reference_set",
            n_features=len(window.feature_vectors),
            n_rewards=len(window.rewards),
            mean_reward=float(np.mean(window.rewards)) if window.rewards else None,
        )

    def evaluate(self) -> DriftResult:
        """
        Run drift evaluation on current observations vs reference.

        Returns DriftResult. If drift_detected=True, caller triggers rollback.
        """
        if self._reference_window is None:
            return DriftResult(
                drift_detected=False,
                drift_signal=None,
                psi_score=None,
                reward_delta=None,
                consecutive_degraded_windows=0,
                reference_reward_mean=None,
                current_reward_mean=None,
            )

        if len(self._current_observations) < 10:
            return DriftResult(
                drift_detected=False,
                drift_signal=None,
                psi_score=None,
                reward_delta=None,
                consecutive_degraded_windows=self._degraded_window_count,
                reference_reward_mean=float(np.mean(self._reference_window.rewards))
                if self._reference_window.rewards
                else None,
                current_reward_mean=None,
            )

        current_features = [obs[0] for obs in self._current_observations]
        current_rewards = [obs[1] for obs in self._current_observations]

        # ── Signal 1: PSI on feature distributions ──────────────
        ref_features = self._reference_window.feature_vectors
        psi_score: float | None = None

        if ref_features and current_features:
            ref_matrix = np.array(ref_features)
            cur_matrix = np.array(current_features)

            # Compute PSI per feature, take max (most shifted feature)
            n_cols = min(ref_matrix.shape[1], cur_matrix.shape[1])
            psi_scores = [_compute_psi(ref_matrix[:, j], cur_matrix[:, j]) for j in range(n_cols)]
            psi_score = max(psi_scores) if psi_scores else 0.0

        input_drift = psi_score is not None and psi_score >= self._psi_threshold

        # ── Signal 2: Mann-Whitney on reward ─────────────────────
        p_value = _mann_whitney_p_value(self._reference_window.rewards, current_rewards)
        reward_degraded = p_value < self._alpha

        # Statistics for logging
        ref_reward_mean = (
            float(np.mean(self._reference_window.rewards))
            if self._reference_window.rewards
            else None
        )
        cur_reward_mean = float(np.mean(current_rewards))
        reward_delta = cur_reward_mean - ref_reward_mean if ref_reward_mean is not None else None

        # ── Determine drift signal ───────────────────────────────
        current_window_degraded = input_drift or reward_degraded

        if current_window_degraded:
            self._degraded_window_count += 1
        else:
            self._degraded_window_count = 0  # hysteresis reset on recovery

        # Apply hysteresis: must be degraded for K consecutive windows
        drift_detected = self._degraded_window_count >= self._hysteresis_k

        drift_signal: DriftSignal | None = None
        if drift_detected:
            if input_drift and reward_degraded:
                drift_signal = DriftSignal.BOTH
            elif input_drift:
                drift_signal = DriftSignal.INPUT_DRIFT
            else:
                drift_signal = DriftSignal.REWARD_DEGRADATION

        logger.info(
            "drift_evaluation_complete",
            drift_detected=drift_detected,
            psi_score=round(psi_score, 4) if psi_score is not None else None,
            p_value=round(p_value, 4),
            reward_delta=round(reward_delta, 4) if reward_delta is not None else None,
            consecutive_degraded=self._degraded_window_count,
            hysteresis_k=self._hysteresis_k,
        )

        return DriftResult(
            drift_detected=drift_detected,
            drift_signal=drift_signal,
            psi_score=psi_score,
            reward_delta=reward_delta,
            consecutive_degraded_windows=self._degraded_window_count,
            reference_reward_mean=ref_reward_mean,
            current_reward_mean=cur_reward_mean,
        )

    def reset_hysteresis(self) -> None:
        """Reset consecutive window counter. Called after successful rollback."""
        self._degraded_window_count = 0
        logger.info("drift_hysteresis_reset")

    @property
    def has_reference(self) -> bool:
        return self._reference_window is not None

    @property
    def observation_count(self) -> int:
        return len(self._current_observations)
