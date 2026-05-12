"""
Database session dependency for FastAPI.

WHY THIS EXISTS:
- Session lifecycle managed in one place. Leak = connection pool exhaustion.
- Yields session, ensures close on both success and exception.
- Every route gets a fresh session via Depends(get_db).

WHAT BREAKS IF WRONG:
- No finally close = sessions leak = pool exhausted = app hangs.
- Session shared across requests = data corruption under concurrency.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session, close after request completes."""
    session = async_session_factory()
    try:
        yield session
    finally:
        await session.close()
