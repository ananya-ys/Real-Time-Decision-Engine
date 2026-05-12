"""
WebSocket endpoint — real-time decision stream for the dashboard.

WHY THIS EXISTS:
The frontend needs live updates without polling every second.
WebSocket pushes:
  - New decisions as they happen
  - Kill switch state changes
  - Drift detection results
  - Trust score updates
  - Cost tracking

CHANNELS:
  ws://host/ws/decisions  → new decision feed
  ws://host/ws/system     → system health feed (every 5s)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.core.database import async_session_factory
from app.models.decision_log import DecisionLog

router = APIRouter(tags=["websocket"])
logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("websocket_connected", total=len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("websocket_disconnected", total=len(self._connections))

    async def broadcast(self, message: dict) -> None:  # type: ignore[type-arg]
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


decision_manager = ConnectionManager()
system_manager = ConnectionManager()


@router.websocket("/ws/decisions")
async def decisions_feed(websocket: WebSocket) -> None:
    """
    Real-time feed of scaling decisions.
    Sends last 5 decisions immediately on connect,
    then new decisions as they arrive (polled every 1s).
    """
    await decision_manager.connect(websocket)
    # Track latest seen decision timestamp for proper deduplication.
    # Using created_at (not id) because UUIDs have no ordering guarantee.
    last_seen_at: datetime | None = None

    def _decision_payload(d: DecisionLog) -> dict:  # type: ignore[name-defined]
        # Ensure created_at is timezone-aware before isoformat()
        # asyncpg returns tz-aware datetimes, but tests may use naive ones
        created = d.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return {
            "type": "decision",
            "id": str(d.id),
            "trace_id": str(d.trace_id),
            "policy_type": d.policy_type,
            "action": d.action,
            "latency_ms": d.latency_ms,
            "fallback_flag": d.fallback_flag,
            "shadow_flag": d.shadow_flag,
            "created_at": created.isoformat() if created else None,
        }

    try:
        # Send last 5 decisions immediately on connect
        async with async_session_factory() as session:
            result = await session.execute(
                select(DecisionLog).order_by(DecisionLog.created_at.desc()).limit(5)
            )
            recent = list(reversed(result.scalars().all()))
            for d in recent:
                await websocket.send_text(json.dumps(_decision_payload(d)))
            if recent:
                last_seen_at = recent[-1].created_at

        # Poll for genuinely new decisions every 0.8s
        while True:
            await asyncio.sleep(0.8)
            async with async_session_factory() as session:
                q = select(DecisionLog).order_by(DecisionLog.created_at.desc()).limit(10)
                result = await session.execute(q)
                decisions = list(reversed(result.scalars().all()))

                for d in decisions:
                    # Only send decisions strictly newer than the last one we sent
                    if last_seen_at is None or d.created_at > last_seen_at:
                        await websocket.send_text(json.dumps(_decision_payload(d)))
                        last_seen_at = d.created_at

    except WebSocketDisconnect:
        decision_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("websocket_error", error=str(exc))
        decision_manager.disconnect(websocket)


@router.websocket("/ws/system")
async def system_feed(websocket: WebSocket) -> None:
    """
    Real-time system health broadcast every 5 seconds.
    Sends: active policy, kill switch state, SLO status.
    """
    await system_manager.connect(websocket)

    try:
        while True:
            # Build system snapshot
            payload: dict = {
                "type": "system_health",
                "timestamp": datetime.now(UTC).isoformat(),
                "active_policy": "BASELINE",
                "kill_switch_active": False,
                "exploration_frozen": False,
                "maintenance_mode": False,
            }

            try:
                from app.operator.kill_switch import KillSwitch

                ks = KillSwitch()
                state = await ks.get_state()
                payload["kill_switch_active"] = state.global_killed
                payload["exploration_frozen"] = state.exploration_frozen
            except Exception:
                pass

            try:
                from app.operator.manual_override import ManualOverride

                mo = ManualOverride()
                payload["maintenance_mode"] = await mo.is_maintenance_mode()
                payload["baseline_forced"] = await mo.is_baseline_forced()
            except Exception:
                pass

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        system_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("system_ws_error", error=str(exc))
        system_manager.disconnect(websocket)
