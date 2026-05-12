#!/usr/bin/env python3
"""
Wait for PostgreSQL to be ready before running migrations.

WHY THIS EXISTS:
On Render / Railway / cold-start deploys, the database service starts
at the same time as the app. `alembic upgrade head` called immediately
will fail with `connection refused`, causing the deploy to error and
trigger a restart loop.

This script retries the DB connection for up to 60 seconds before
proceeding. It exits 0 if DB is ready, exits 1 if it times out.

USAGE:
  python scripts/wait_for_db.py  # called from startCommand in render.yaml
"""

import os
import sys
import time

import psycopg2


def wait_for_db(max_wait: int = 60, interval: int = 2) -> None:
    # Strip asyncpg driver prefix for psycopg2 sync connection
    db_url = os.environ.get("DATABASE_URL", "")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    db_url = db_url.replace("postgres://", "postgresql://")

    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        elapsed = time.time() - start

        try:
            conn = psycopg2.connect(db_url, connect_timeout=3)
            conn.close()
            print(f"Database ready after {elapsed:.1f}s ({attempt} attempts)")
            return
        except Exception as exc:
            if elapsed >= max_wait:
                print(
                    f"ERROR: Database not ready after {max_wait}s. Last error: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            remaining = max_wait - elapsed
            print(
                f"[{elapsed:.0f}s] DB not ready (attempt {attempt}): {exc}. "
                f"Retrying in {interval}s... ({remaining:.0f}s remaining)"
            )
            time.sleep(interval)


if __name__ == "__main__":
    wait_for_db()
