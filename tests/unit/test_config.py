"""
Config validation tests.

Verifies:
- All defaults load without error
- Validators reject invalid values
- Production mode detection works
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


@pytest.mark.unit
class TestSettings:
    """Verify Settings validation."""

    def test_defaults_load(self) -> None:
        """Settings should load with defaults without error."""
        settings = Settings()
        assert settings.app_name == "rtde-backend"
        assert settings.min_instances >= 1

    def test_cpu_threshold_validation(self) -> None:
        """CPU thresholds outside [0,1] must raise ValueError."""
        with pytest.raises(ValueError, match="CPU threshold must be between"):
            Settings(baseline_cpu_high=1.5)  # type: ignore[call-arg]

    def test_min_instances_validation(self) -> None:
        """min_instances < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="min_instances must be >= 1"):
            Settings(min_instances=0)  # type: ignore[call-arg]

    def test_production_detection(self) -> None:
        """is_production should be True only when app_env is 'production'."""
        dev_settings = Settings(app_env="development")
        assert not dev_settings.is_production

        prod_settings = Settings(app_env="production")
        assert prod_settings.is_production

    def test_sync_database_url(self) -> None:
        """sync_database_url should replace asyncpg with empty string for Alembic."""
        settings = Settings()
        assert "+asyncpg" not in settings.sync_database_url
        assert "postgresql://" in settings.sync_database_url
