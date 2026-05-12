"""
Async database engine and session factory.

WHY THIS EXISTS:
- Async engine handles concurrent requests. Sync engine blocks under load.
- Connection pool prevents exhaustion under burst traffic.
- Base model class provides shared metadata for all models.

WHAT BREAKS IF WRONG:
- Missing pool config = connections leak = DB crashes under load.
- Sync engine in async app = every DB call blocks the event loop.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models. Provides shared metadata."""

    pass


def create_engine() -> AsyncEngine:
    """Create async engine with production-grade pool settings."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,  # verify connections before use — catches stale connections
        echo=settings.db_echo,
    )


engine = create_engine()

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # prevent lazy-load in async context
)
