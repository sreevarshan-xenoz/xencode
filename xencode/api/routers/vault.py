#!/usr/bin/env python3
"""
Vault API Router

FastAPI router for JsonFileCredentialVault operations including
health checks, vault initialization, and credential migration.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Import vault with graceful fallback
try:
    from ...auth.json_file_vault import JsonFileCredentialVault
    VAULT_AVAILABLE = True
except ImportError:
    JsonFileCredentialVault = None  # type: ignore
    VAULT_AVAILABLE = False

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
async def vault_health():
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
async def vault_init(request: VaultInitRequest):
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
async def vault_migrate(request: VaultMigrateRequest):
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


router.tags = ["Vault"]
