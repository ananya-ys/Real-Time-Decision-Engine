"""
StateNormalizer — versioned feature normalization for neural network input.

WHY THIS EXISTS:
- Neural networks are NOT scale-invariant.
  cpu_utilization lives in [0.0, 1.0].
  request_rate lives in [0, 50000].
  p99_latency_ms lives in [0, 10000].
  Without normalization, the network effectively ignores low-magnitude features.

- VERSION LOCK: The normalizer MUST be trained on the same distribution as the
  model weights. If you load new weights with old normalization statistics,
  the model silently receives inputs on a different scale than training.
  This produces wrong Q-values with no error, no warning, no alert.

THE PRODUCTION PATTERN:
  1. Fit normalizer on N ticks of simulation data.
  2. Save normalizer (with version ID) alongside model weights.
  3. On load: if normalizer_path != weights normalizer_path → loud failure.
  This is the normalizer-model version lock that prevents silent wrong inference.

NORMALIZATION METHOD:
  Z-score normalization: (x - mean) / (std + ε)
  - Centers features at 0, scales to unit variance.
  - Robust to outliers (unlike min-max which clips at observed extremes).
  - ε = 1e-8 prevents division by zero for constant features.

WHAT BREAKS IF WRONG:
- No normalization: Q-network converges slowly or diverges.
- Wrong normalization stats (version mismatch): silent wrong predictions.
- Fit on wrong data distribution: normalizer tuned for STEADY traffic
  but deployed on BURST traffic → all features look like outliers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from app.schemas.state import SystemState

logger = structlog.get_logger(__name__)

# Feature extraction order — MUST be stable across versions
_FEATURE_ORDER = [
    "cpu_utilization",
    "request_rate",
    "p99_latency_ms",
    "instance_count",
    "hour_of_day",
    "day_of_week",
]
_N_FEATURES = len(_FEATURE_ORDER)


class StateNormalizer:
    """
    Min-max + Z-score normalizer fitted on simulation data.

    Fitted once on N simulation ticks before any policy training.
    Saved alongside model weights. Version-locked.

    fit() → save() → (training) → (load with same version) → normalize()
    """

    def __init__(self, version_id: str = "v1") -> None:
        self._version_id = version_id
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def version_id(self) -> str:
        return self._version_id

    @property
    def n_features(self) -> int:
        return _N_FEATURES

    def fit(self, states: list[SystemState]) -> StateNormalizer:
        """
        Fit normalization statistics on a sample of states.

        Args:
            states: List of SystemState observations from simulation.
                    Should cover all traffic patterns for robust statistics.

        Returns:
            self (for chaining: normalizer.fit(states).save(path))

        Raises:
            ValueError: If fewer than 10 states provided (insufficient sample).
        """
        if len(states) < 10:
            raise ValueError(f"Need at least 10 states to fit normalizer, got {len(states)}")

        # Build feature matrix [n_samples, n_features]
        matrix = np.array([self._extract_features(s) for s in states], dtype=np.float32)

        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)

        # Clip std to prevent division by zero for constant features
        self._std = np.clip(self._std, 1e-8, None)

        self._fitted = True

        logger.info(
            "normalizer_fitted",
            n_samples=len(states),
            version_id=self._version_id,
            feature_means=self._mean.tolist(),
            feature_stds=self._std.tolist(),
        )
        return self

    def normalize(self, state: SystemState) -> np.ndarray:
        """
        Normalize a single state to zero-mean, unit-variance.

        Args:
            state: Raw system state from environment.

        Returns:
            np.ndarray of shape (n_features,) with normalized values.

        Raises:
            RuntimeError: If normalizer has not been fitted.
                          Loud failure — never silently wrong.
        """
        if not self._fitted or self._mean is None or self._std is None:
            raise RuntimeError(
                "StateNormalizer.normalize() called before fit(). "
                "This means the policy loaded weights without loading the normalizer. "
                "Check PolicyVersion.normalizer_path."
            )

        features = self._extract_features(state)
        normalized = (np.array(features, dtype=np.float32) - self._mean) / self._std
        return normalized

    def normalize_batch(self, states: list[SystemState]) -> np.ndarray:
        """Normalize a batch of states. Returns (n_states, n_features) array."""
        return np.array([self.normalize(s) for s in states], dtype=np.float32)

    def save(self, path: Path) -> None:
        """
        Save normalizer statistics to disk.

        Saved as JSON (human-readable + version-trackable).
        Includes version_id so mismatches are detected on load.
        Uses atomic write (temp → rename) to prevent partial saves.
        """
        if not self._fitted or self._mean is None or self._std is None:
            raise RuntimeError("Cannot save unfitted normalizer")

        data = {
            "version_id": self._version_id,
            "n_features": _N_FEATURES,
            "feature_order": _FEATURE_ORDER,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }

        # Atomic write: write to temp file, rename on success
        # Prevents partial saves if process is killed during write
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")

        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.rename(path)
            logger.info("normalizer_saved", path=str(path), version_id=self._version_id)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save normalizer to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path, expected_version_id: str | None = None) -> StateNormalizer:
        """
        Load normalizer from disk.

        Args:
            path: Path to the saved normalizer JSON.
            expected_version_id: If provided, validates that loaded normalizer
                                  matches the expected version. Mismatch → loud error.

        Raises:
            FileNotFoundError: If normalizer file does not exist.
            ValueError: If version mismatch detected (the v3 version lock).
            RuntimeError: If file is malformed.
        """
        from app.core.exceptions import CheckpointError

        if not path.exists():
            raise FileNotFoundError(
                f"Normalizer not found at {path}. "
                "Every model must have a paired normalizer. "
                "Check PolicyVersion.normalizer_path."
            )

        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Malformed normalizer at {path}: {exc}") from exc

        loaded_version = data.get("version_id")

        # Version lock check — catches normalizer/model mismatch
        if expected_version_id is not None and loaded_version != expected_version_id:
            raise CheckpointError(
                f"Normalizer version mismatch: expected '{expected_version_id}', "
                f"got '{loaded_version}'. "
                "The model and normalizer must be trained together. "
                "This is silent wrong inference if you ignore it."
            )

        normalizer = cls(version_id=str(loaded_version))
        normalizer._mean = np.array(data["mean"], dtype=np.float32)
        normalizer._std = np.array(data["std"], dtype=np.float32)
        normalizer._fitted = True

        logger.info(
            "normalizer_loaded",
            path=str(path),
            version_id=loaded_version,
        )
        return normalizer

    @staticmethod
    def _extract_features(state: SystemState) -> list[float]:
        """
        Extract feature vector from SystemState in stable order.

        Order MUST match _FEATURE_ORDER exactly.
        Any change to feature order requires a new normalizer version.
        """
        return [
            state.cpu_utilization,
            state.request_rate,
            state.p99_latency_ms,
            float(state.instance_count),
            float(state.hour_of_day),
            float(state.day_of_week),
        ]
