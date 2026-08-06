#!/usr/bin/env python3
"""
Vault API Router

FastAPI router for JsonFileCredentialVault operations including
health checks, vault initialization, and credential migration.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field

# Import vault with graceful fallback
try:
    from ...auth.json_file_vault import JsonFileCredentialVault
    VAULT_AVAILABLE = True
except ImportError:
    JsonFileCredentialVault = None  # type: ignore
    VAULT_AVAILABLE = False

# Import authentication
from xencode.api.auth import verify_jwt_token, resolve_jwt_secret

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VaultHealthResponse(BaseModel):
    """Vault health check response"""
    available: bool
    vault_exists: bool
    vault_path: str
    is_readable: bool = False
    is_valid_json: bool = False
    credential_count: int = 0
    encryption_available: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class VaultInitRequest(BaseModel):
    """Request to initialize a vault"""
    vault_path: Optional[str] = None
    master_key: Optional[str] = None


class VaultInitResponse(BaseModel):
    """Vault initialization response"""
    created: bool
    vault_path: str = ""
    already_exists: bool = False
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class VaultMigrateRequest(BaseModel):
    """Request to migrate credentials from config to vault"""
    config_path: Optional[str] = None
    vault_path: Optional[str] = None
    master_key: Optional[str] = None
    delete_after: bool = False


class VaultMigrateResponse(BaseModel):
    """Vault migration response"""
    migrated: int = 0
    skipped: int = 0
    vault_path: str = ""
    config_path: str = ""
    errors: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_vault_or_404() -> JsonFileCredentialVault:
    """Get vault instance or raise 404 if unavailable."""
    if not VAULT_AVAILABLE or JsonFileCredentialVault is None:
        raise HTTPException(
            status_code=503,
            detail="JsonFileCredentialVault module is not available"
        )
    return JsonFileCredentialVault()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=VaultHealthResponse)
async def vault_health(
    _auth: Dict[str, Any] = Depends(verify_jwt_token),
):
    """Check vault health and status

    Returns vault file existence, readability, credential count,
    and encryption availability without instantiating a full vault.
    """
    if not VAULT_AVAILABLE or JsonFileCredentialVault is None:
        return VaultHealthResponse(
            available=False,
            vault_exists=False,
            vault_path=str(Path.home() / ".xencode" / "vault.json"),
        )

    try:
        health = JsonFileCredentialVault.health_check()
        return VaultHealthResponse(
            available=True,
            vault_exists=health.get("vault_exists", False),
            vault_path=health.get("vault_path", ""),
            is_readable=health.get("is_readable", False),
            is_valid_json=health.get("is_valid_json", False),
            credential_count=health.get("credential_count", 0),
            encryption_available=health.get("encryption_available", False),
        )
    except Exception as e:
        logger.error("Vault health check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@router.post("/init", response_model=VaultInitResponse)
async def vault_init(
    request: VaultInitRequest,
    _auth: Dict[str, Any] = Depends(verify_jwt_token),
):
    """Initialize (create) a new credential vault

    Creates the vault directory and an empty vault file.
    Safe to call multiple times — won't overwrite an existing vault.
    """
    vault = _get_vault_or_404()

    try:
        vault_path = Path(request.vault_path) if request.vault_path else None
        result = JsonFileCredentialVault.init_vault(
            vault_path=vault_path,
            master_key=request.master_key,
        )
        return VaultInitResponse(
            created=result.get("created", False),
            vault_path=result.get("vault_path", ""),
            already_exists=result.get("already_exists", False),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error("Vault init failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Vault initialization failed: {e}")


@router.post("/migrate", response_model=VaultMigrateResponse)
async def vault_migrate(
    request: VaultMigrateRequest,
    _auth: Dict[str, Any] = Depends(verify_jwt_token),
):
    """Migrate plaintext API keys from config file into the vault

    Scans the config file for known API key fields and stores
    any plaintext values into the encrypted vault.
    """
    if not VAULT_AVAILABLE or JsonFileCredentialVault is None:
        raise HTTPException(
            status_code=503,
            detail="JsonFileCredentialVault module is not available"
        )

    try:
        config_path = Path(request.config_path) if request.config_path else None
        vault_path = Path(request.vault_path) if request.vault_path else None

        result = JsonFileCredentialVault.migrate_from_config(
            config_path=config_path,
            vault_path=vault_path,
            master_key=request.master_key,
            delete_after=request.delete_after,
        )

        return VaultMigrateResponse(
            migrated=result.get("migrated", 0),
            skipped=result.get("skipped", 0),
            vault_path=result.get("vault_path", ""),
            config_path=result.get("config_path", ""),
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.error("Vault migration failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Vault migration failed: {e}")


# ---------------------------------------------------------------------------
# WebSocket health monitor
# ---------------------------------------------------------------------------


class VaultHealthMonitor:
    """Manages WebSocket connections for real-time vault health monitoring."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self._tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("Vault health WS client connected (%d active)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        self.active_connections.discard(websocket)
        task = self._tasks.pop(websocket, None)
        if task and not task.done():
            task.cancel()
        logger.info("Vault health WS client disconnected (%d active)", len(self.active_connections))

    def register_monitor_task(self, websocket: WebSocket, task: asyncio.Task) -> None:
        """Register the background health-monitoring task for a connection.

        Any previously registered task for this websocket is cancelled.
        """
        old = self._tasks.pop(websocket, None)
        if old and not old.done():
            old.cancel()
        self._tasks[websocket] = task


# Singleton monitor instance
vault_health_monitor = VaultHealthMonitor()


def _build_health_payload(vault_path: Optional[Path] = None) -> dict:
    """Build a health JSON payload suitable for WebSocket broadcast.

    Gracefully handles the vault module being unavailable.
    """
    if not VAULT_AVAILABLE or JsonFileCredentialVault is None:
        return {
            "type": "health_update",
            "available": False,
            "vault_exists": False,
            "vault_path": str(vault_path or Path.home() / ".xencode" / "vault.json"),
            "is_readable": False,
            "is_valid_json": False,
            "credential_count": 0,
            "encryption_available": False,
            "timestamp": datetime.now().isoformat(),
        }

    try:
        health = JsonFileCredentialVault.health_check(vault_path=vault_path)
        return {
            "type": "health_update",
            "available": True,
            "vault_exists": health.get("vault_exists", False),
            "vault_path": health.get("vault_path", str(
                vault_path or Path.home() / ".xencode" / "vault.json"
            )),
            "is_readable": health.get("is_readable", False),
            "is_valid_json": health.get("is_valid_json", False),
            "credential_count": health.get("credential_count", 0),
            "encryption_available": health.get("encryption_available", False),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.error("WebSocket health check failed: %s", exc)
        return {
            "type": "error",
            "message": f"Health check failed: {exc}",
            "timestamp": datetime.now().isoformat(),
        }


async def _health_monitor_loop(
    websocket: WebSocket,
    interval: float,
    vault_path: Optional[Path],
) -> None:
    """Background task: periodically send vault health updates to a single client.

    Sleeps first, then sends — so the first update arrives after *interval*
    seconds rather than immediately. The initial snapshot is sent directly
    from the endpoint handler on connect.

    This coroutine is wrapped in an asyncio.Task and cancelled when the
    client disconnects or changes the interval.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            payload = _build_health_payload(vault_path=vault_path)
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                # Connection likely dropped; stop the loop
                break
    except asyncio.CancelledError:
        pass


@router.websocket("/health/ws")
async def vault_health_ws(
    websocket: WebSocket,
    interval: float = Query(5.0, ge=1.0, le=300.0, description="Polling interval in seconds"),
    vault_path: Optional[str] = Query(None, description="Custom vault file path"),
    token: Optional[str] = Query(None, description="JWT access token for authentication"),
) -> None:
    """WebSocket endpoint for real-time vault health monitoring

    Sends periodic health updates (by default every 5 seconds).

    **Query parameters:**
    - `interval` — polling interval in seconds (1–300, default 5)
    - `vault_path` — optional custom path to the vault file

    **Client messages:**
    - `{"type": "ping"}` → server responds with `{"type": "pong"}`
    - `{"type": "set_interval", "interval": 10}` → change polling frequency
    - `{"type": "check_now"}` → triggers an immediate health update
    """
    resolved_vault_path = Path(vault_path) if vault_path else None

    # Authenticate WebSocket via token query parameter
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    try:
        from xencode.auth.jwt_handler import JWTHandler
        _ws_secret = resolve_jwt_secret()
        _ws_jwt = JWTHandler(secret_key=_ws_secret)
        _ws_payload = _ws_jwt.verify_token(token, token_type="access")
        if _ws_payload is None:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
    except Exception:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await vault_health_monitor.connect(websocket)

    # Send an immediate health snapshot on connect
    try:
        initial = _build_health_payload(vault_path=resolved_vault_path)
        await websocket.send_text(json.dumps(initial))
    except Exception:
        vault_health_monitor.disconnect(websocket)
        return

    # Start the periodic monitoring task
    monitor_task = asyncio.create_task(
        _health_monitor_loop(websocket, interval, resolved_vault_path)
    )
    vault_health_monitor.register_monitor_task(websocket, monitor_task)

    current_interval = interval

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                }))

            elif msg_type == "set_interval":
                new_interval = float(msg.get("interval", current_interval))
                new_interval = max(1.0, min(300.0, new_interval))
                current_interval = new_interval

                # Restart the monitor loop with the new interval
                monitor_task.cancel()
                monitor_task = asyncio.create_task(
                    _health_monitor_loop(websocket, current_interval, resolved_vault_path)
                )
                vault_health_monitor.register_monitor_task(websocket, monitor_task)

                await websocket.send_text(json.dumps({
                    "type": "interval_updated",
                    "interval": current_interval,
                    "timestamp": datetime.now().isoformat(),
                }))

            elif msg_type == "check_now":
                payload = _build_health_payload(vault_path=resolved_vault_path)
                await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError:
        # Invalid JSON from client — send error but keep connection alive
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid JSON message",
                "timestamp": datetime.now().isoformat(),
            }))
        except Exception:
            pass
    except Exception as e:
        logger.warning("Vault health WS error: %s", e)
    finally:
        vault_health_monitor.disconnect(websocket)


router.tags = ["Vault"]
