"""
Root test configuration — shared fixtures.

Unit tests (marked @pytest.mark.unit): run without DB/Redis.
Integration tests (marked @pytest.mark.integration): require DB + Redis.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.dependencies.db import get_db
from app.main import create_app

settings = get_settings()


def _port_open(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# ── Detect available services ─────────────────────────────────────
DB_AVAILABLE = _port_open("localhost", 5432)
REDIS_AVAILABLE = _port_open("localhost", 6379)

# ── Marks for conditional skipping ───────────────────────────────
skip_if_no_db = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason="PostgreSQL not available on localhost:5432 — run with docker compose up",
)
skip_if_no_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="Redis not available on localhost:6379 — run with docker compose up",
)


# ── Pytest hooks: auto-skip integration tests without infra ───────
def pytest_collection_modifyitems(items: list) -> None:
    """Auto-apply skip markers to integration tests when services are down."""
    for item in items:
        if "integration" in item.keywords and not DB_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="DB unavailable — skipping integration test"))
        if "concurrency" in item.keywords and not DB_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="DB unavailable — skipping concurrency test"))
        if "chaos" in item.keywords and not DB_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="DB unavailable — skipping chaos test"))


# ── Shared app instance ──────────────────────────────────────────
_app = create_app()


# ── Mock DB session for unit tests ───────────────────────────────
class MockAsyncSession:
    """
    Minimal mock of AsyncSession that satisfies unit test dependencies.
    All write operations are no-ops. Queries return empty results.
    """

    def __init__(self) -> None:
        # db.add() and db.delete() are synchronous in SQLAlchemy
        self.add = MagicMock(return_value=None)
        self.delete = MagicMock(return_value=None)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def execute(self, stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        result.fetchall.return_value = []
        return result

    async def scalar(self, stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    async def get(self, model, pk):  # type: ignore[no-untyped-def]
        return None

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aexit__(self, *args):  # type: ignore[no-untyped-def]
        pass

    def __await__(self):  # type: ignore[no-untyped-def]
        return self._noop().__await__()

    async def _noop(self):  # type: ignore[no-untyped-def]
        return self


async def _mock_get_db() -> AsyncGenerator[MockAsyncSession, None]:
    """Unit-test DB stub — no real connection made."""
    yield MockAsyncSession()


# Override dependency for all tests by default (unit test mode)
_app.dependency_overrides[get_db] = _mock_get_db


# ── Integration DB setup (only when DB is available) ─────────────
# Sentinel: always defined so db_session fixture never hits NameError
_real_session_factory = None  # replaced below if DB_AVAILABLE

if DB_AVAILABLE:
    _real_engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        echo=False,
    )
    _real_session_factory = async_sessionmaker(
        bind=_real_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def _real_get_db() -> AsyncGenerator[AsyncSession, None]:
        session = _real_session_factory()
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture
def use_real_db():  # type: ignore[no-untyped-def]
    """
    Fixture: switch to real DB for this test.
    Skipped automatically if DB unavailable.
    """
    if not DB_AVAILABLE:
        pytest.skip("DB not available")
    _app.dependency_overrides[get_db] = _real_get_db
    yield
    _app.dependency_overrides[get_db] = _mock_get_db


# ── Shared HTTP client ────────────────────────────────────────────
@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client (uses mock DB by default)."""
    transport = ASGITransport(app=_app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session() -> AsyncGenerator:
    """
    DB session fixture.
    Returns real session if DB available, otherwise mock session.
    Guard: _real_session_factory is only defined when DB_AVAILABLE=True.
    Without this guard, tests crash with NameError when DB is down.
    """
    if DB_AVAILABLE and _real_session_factory is not None:
        session = _real_session_factory()
        try:
            yield session
        finally:
            await session.close()
    else:
        yield MockAsyncSession()
