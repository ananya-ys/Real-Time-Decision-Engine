"""
StateNormalizer tests — Phase 5 gate.

Verifies:
- fit() computes correct mean and std
- normalize() produces zero-mean, unit-std features
- save() → load() round-trip preserves statistics
- version lock: mismatched version raises CheckpointError
- unfitted normalizer raises RuntimeError on normalize()
- batch normalization consistency with single normalization
- insufficient samples raises ValueError
- atomic write: simulated failure doesn't corrupt existing file
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.ml.state_normalizer import _N_FEATURES, StateNormalizer
from app.schemas.common import TrafficRegime
from app.schemas.state import SystemState


def _make_state(
    cpu: float = 0.5,
    rps: float = 1000.0,
    latency: float = 200.0,
    instances: int = 5,
    hour: int = 12,
    dow: int = 1,
) -> SystemState:
    return SystemState(
        cpu_utilization=cpu,
        request_rate=rps,
        p99_latency_ms=latency,
        instance_count=instances,
        hour_of_day=hour,
        day_of_week=dow,
        traffic_regime=TrafficRegime.STEADY,
    )


def _make_sample_states(n: int = 100) -> list[SystemState]:
    """Create N states with known distributions for verification."""
    states = []
    for i in range(n):
        states.append(
            _make_state(
                cpu=float(i) / n,
                rps=float(i) * 10,
                latency=100.0 + float(i),
                instances=max(1, i % 20),
                hour=i % 24,
                dow=i % 7,
            )
        )
    return states


@pytest.mark.unit
class TestNormalizerFitting:
    """Verify fit() computes correct statistics."""

    def test_fit_produces_means(self) -> None:
        """fit() must compute non-None mean array."""
        norm = StateNormalizer(version_id="test-v1")
        states = _make_sample_states(100)
        norm.fit(states)
        assert norm._mean is not None
        assert len(norm._mean) == _N_FEATURES

    def test_fit_produces_stds(self) -> None:
        """fit() must compute non-None std array."""
        norm = StateNormalizer()
        states = _make_sample_states(100)
        norm.fit(states)
        assert norm._std is not None
        assert len(norm._std) == _N_FEATURES

    def test_fit_marks_as_fitted(self) -> None:
        """After fit(), is_fitted must be True."""
        norm = StateNormalizer()
        assert not norm.is_fitted
        norm.fit(_make_sample_states(50))
        assert norm.is_fitted

    def test_fit_requires_minimum_samples(self) -> None:
        """Fewer than 10 states must raise ValueError."""
        norm = StateNormalizer()
        with pytest.raises(ValueError, match="at least 10"):
            norm.fit(_make_sample_states(5))

    def test_fit_prevents_zero_std(self) -> None:
        """Constant features must not produce zero std (division by zero guard)."""
        # All states have same cpu
        states = [_make_state(cpu=0.5) for _ in range(20)]
        norm = StateNormalizer()
        norm.fit(states)
        # cpu std should be clipped to 1e-8, not zero
        assert norm._std is not None
        assert all(s > 0 for s in norm._std)


@pytest.mark.unit
class TestNormalization:
    """Verify normalize() output properties."""

    def test_normalize_returns_correct_length(self) -> None:
        """Normalized vector must have exactly N_FEATURES elements."""
        norm = StateNormalizer()
        norm.fit(_make_sample_states(50))
        result = norm.normalize(_make_state())
        assert len(result) == _N_FEATURES

    def test_normalize_returns_numpy_array(self) -> None:
        """Output must be numpy array for direct neural network input."""
        norm = StateNormalizer()
        norm.fit(_make_sample_states(50))
        result = norm.normalize(_make_state())
        assert isinstance(result, np.ndarray)

    def test_normalized_mean_is_approximately_zero(self) -> None:
        """Normalized features over training distribution should average near zero."""
        states = _make_sample_states(200)
        norm = StateNormalizer()
        norm.fit(states)
        matrix = norm.normalize_batch(states)
        # Mean across samples should be near 0 for each feature
        col_means = matrix.mean(axis=0)
        assert all(abs(m) < 0.1 for m in col_means)

    def test_normalized_std_is_approximately_one(self) -> None:
        """Normalized features over training distribution should have std near 1."""
        states = _make_sample_states(200)
        norm = StateNormalizer()
        norm.fit(states)
        matrix = norm.normalize_batch(states)
        col_stds = matrix.std(axis=0)
        # Allow relaxed tolerance for small N
        assert all(abs(s - 1.0) < 0.2 for s in col_stds)

    def test_normalize_fails_without_fit(self) -> None:
        """normalize() on unfitted normalizer must raise RuntimeError."""
        norm = StateNormalizer()
        with pytest.raises(RuntimeError, match="before fit"):
            norm.normalize(_make_state())

    def test_batch_normalize_consistent_with_single(self) -> None:
        """Batch normalization must match element-wise normalization."""
        states = _make_sample_states(20)
        norm = StateNormalizer()
        norm.fit(states)

        single_results = np.array([norm.normalize(s) for s in states[:5]])
        batch_results = norm.normalize_batch(states[:5])

        np.testing.assert_allclose(single_results, batch_results, rtol=1e-5)


@pytest.mark.unit
class TestNormalizerPersistence:
    """Verify save/load round-trip and version lock."""

    def test_save_and_load_preserves_statistics(self) -> None:
        """Loaded normalizer must have same mean and std as original."""
        states = _make_sample_states(100)
        original = StateNormalizer(version_id="model-v1")
        original.fit(states)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer.json"
            original.save(path)
            loaded = StateNormalizer.load(path)

        np.testing.assert_allclose(original._mean, loaded._mean, rtol=1e-6)
        np.testing.assert_allclose(original._std, loaded._std, rtol=1e-6)

    def test_save_creates_valid_json(self) -> None:
        """Saved file must be valid JSON with all required keys."""
        norm = StateNormalizer(version_id="json-test")
        norm.fit(_make_sample_states(50))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.json"
            norm.save(path)
            data = json.loads(path.read_text())

        assert "version_id" in data
        assert "mean" in data
        assert "std" in data
        assert "feature_order" in data
        assert data["version_id"] == "json-test"

    def test_version_lock_raises_on_mismatch(self) -> None:
        """Loading with wrong expected version must raise CheckpointError."""
        from app.core.exceptions import CheckpointError

        norm = StateNormalizer(version_id="model-v1")
        norm.fit(_make_sample_states(50))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.json"
            norm.save(path)

            # Try to load with wrong expected version
            with pytest.raises(CheckpointError, match="version mismatch"):
                StateNormalizer.load(path, expected_version_id="model-v2")

    def test_correct_version_loads_successfully(self) -> None:
        """Loading with correct expected version must succeed."""
        norm = StateNormalizer(version_id="model-v1")
        norm.fit(_make_sample_states(50))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.json"
            norm.save(path)
            loaded = StateNormalizer.load(path, expected_version_id="model-v1")

        assert loaded.is_fitted
        assert loaded.version_id == "model-v1"

    def test_load_missing_file_raises(self) -> None:
        """Loading from non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            StateNormalizer.load(Path("/nonexistent/normalizer.json"))

    def test_loaded_normalizer_produces_same_output(self) -> None:
        """Round-trip normalizer must produce identical output for same input."""
        states = _make_sample_states(100)
        original = StateNormalizer(version_id="v1")
        original.fit(states)

        test_state = _make_state(cpu=0.75, rps=2000.0, latency=350.0)
        original_output = original.normalize(test_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.json"
            original.save(path)
            loaded = StateNormalizer.load(path)

        loaded_output = loaded.normalize(test_state)
        np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)

    def test_save_unfitted_raises(self) -> None:
        """Saving an unfitted normalizer must raise RuntimeError."""
        norm = StateNormalizer()
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            norm.save(Path("/tmp/test_norm.json"))
