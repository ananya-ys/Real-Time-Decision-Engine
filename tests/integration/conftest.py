"""
Integration test configuration — real PostgreSQL with transactional isolation.
Each test rolls back its writes. Requires docker-compose services running.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.core.database import Base

settings = get_settings()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def integration_engine():
    engine = create_async_engine(
        settings.database_url, pool_size=5, max_overflow=10, pool_pre_ping=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db(integration_engine) -> AsyncGenerator[AsyncSession, None]:
    async with integration_engine.begin() as conn:
        session = AsyncSession(bind=conn, expire_on_commit=False)
        await conn.execute(text("SAVEPOINT test_start"))
        try:
            yield session
        finally:
            await conn.execute(text("ROLLBACK TO SAVEPOINT test_start"))
            await session.close()
