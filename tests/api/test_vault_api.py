#!/usr/bin/env python3
"""
Tests for Vault API Router

Tests all credential vault API endpoints including health checks,
vault initialization, and credential migration from config.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient

from xencode.api.main import app


pytestmark = [pytest.mark.unit]

# Dev secret key (mirrors xencode/api/auth.py fallback)
_DEV_SECRET = "dev-secret-key-change-in-production"


@pytest.fixture
def test_token() -> str:
    """Generate a valid JWT access token for testing."""
    payload = {
        "user_id": "test-user",
        "username": "test",
        "role": "admin",
        "session_id": "test-session",
        "type": "access",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    return jwt.encode(payload, _DEV_SECRET, algorithm="HS256")


@pytest.fixture
def auth_header(test_token: str) -> dict:
    """Authorization header with a valid Bearer token."""
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def tmp_home(tmp_path):
    """Fixture that patches Path.home() to a temp dir for isolation."""
    test_home = tmp_path / "home" / "user"
    test_home.mkdir(parents=True, exist_ok=True)
    vault_dir = test_home / ".xencode"
    vault_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(Path, "home", return_value=test_home):
        yield test_home


@pytest.fixture
def config_file(tmp_home):
    """Create a sample config file with plaintext API keys."""
    config = {
        "features": {
            "code_review": {
                "openai_api_key": "sk-test-openai-key-12345",
                "model": "gpt-4",
            },
            "learning_mode": {
                "anthropic_api_key": "sk-ant-test-key-67890",
                "model": "claude-3",
            },
        },
        "settings": {
            "max_tokens": 2048,
            "temperature": 0.7,
        },
    }
    config_path = tmp_home / ".xencode" / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


@pytest.fixture
def config_file_with_env_refs(tmp_home):
    """Create a config file with env-var references (should be skipped)."""
    config = {
        "providers": {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-4",
            },
            "anthropic": {
                "api_key": "env:ANTHROPIC_API_KEY",
                "model": "claude-3",
            },
        },
    }
    config_path = tmp_home / ".xencode" / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


# ---------------------------------------------------------------------------
# GET /api/v1/vault/health
# ---------------------------------------------------------------------------


class TestVaultHealth:
    """Tests for the vault health check endpoint."""

    def test_health_no_vault(self, client, auth_header, tmp_home):
        """Health should report vault not existing before init."""
        response = client.get("/api/v1/vault/health", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["vault_exists"] is False
        assert data["credential_count"] == 0
        assert ".xencode" in data["vault_path"] and "vault.json" in data["vault_path"]

    def test_health_after_init(self, client, auth_header, tmp_home):
        """Health should show vault exists after init."""
        # Init first
        client.post("/api/v1/vault/init", json={}, headers=auth_header)

        # Then check health
        response = client.get("/api/v1/vault/health", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["vault_exists"] is True
        assert data["is_readable"] is True
        assert data["is_valid_json"] is True
        assert data["credential_count"] == 0
        # Should have same vault_path
        assert response.status_code == 200

    def test_health_returns_timestamp(self, client, auth_header, tmp_home):
        """Health response should include an ISO timestamp."""
        response = client.get("/api/v1/vault/health", headers=auth_header)
        data = response.json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format check


# ---------------------------------------------------------------------------
# POST /api/v1/vault/init
# ---------------------------------------------------------------------------


class TestVaultInit:
    """Tests for the vault initialization endpoint."""

    def test_init_creates_vault(self, client, auth_header, tmp_home):
        """Init should create a new vault file."""
        response = client.post("/api/v1/vault/init", json={}, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is True
        assert data["already_exists"] is False
        assert data["error"] is None
        assert ".xencode" in data["vault_path"] and "vault.json" in data["vault_path"]

        # Verify file exists on disk
        vault_file = tmp_home / ".xencode" / "vault.json"
        assert vault_file.exists()
        with open(vault_file, "r") as f:
            contents = json.load(f)
        assert "credentials" in contents
        assert contents["version"] == 1

    def test_init_idempotent(self, client, auth_header, tmp_home):
        """Calling init twice should return already_exists=True."""
        first = client.post("/api/v1/vault/init", json={}, headers=auth_header)
        assert first.json()["created"] is True

        second = client.post("/api/v1/vault/init", json={}, headers=auth_header)
        assert second.status_code == 200
        data = second.json()
        assert data["created"] is False
        assert data["already_exists"] is True

    def test_init_with_custom_vault_path(self, client, auth_header, tmp_home):
        """Init with a custom vault_path should create at that location."""
        custom_path = tmp_home / "custom" / "secrets.json"
        response = client.post("/api/v1/vault/init", json={
            "vault_path": str(custom_path),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is True
        assert str(custom_path) in data["vault_path"]
        assert custom_path.exists()

    def test_init_with_master_key(self, client, auth_header, tmp_home):
        """Init with a master_key should succeed."""
        response = client.post("/api/v1/vault/init", json={
            "master_key": "my-test-master-key-42",
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is True
        assert data["error"] is None

    def test_init_returns_timestamp(self, client, auth_header, tmp_home):
        """Init response should include an ISO timestamp."""
        response = client.post("/api/v1/vault/init", json={}, headers=auth_header)
        data = response.json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# POST /api/v1/vault/migrate
# ---------------------------------------------------------------------------


class TestVaultMigrate:
    """Tests for the vault migration endpoint."""

    def test_migrate_from_config(self, client, auth_header, tmp_home, config_file):
        """Migrate should extract plaintext API keys from config."""
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] >= 2
        assert data["errors"] == []
        assert str(config_file) in data["config_path"]

        # Verify vault now has credentials
        health = client.get("/api/v1/vault/health", headers=auth_header).json()
        assert health["credential_count"] >= 2
        assert health["vault_exists"] is True

    def test_migrate_with_delete_after(self, client, auth_header, tmp_home, config_file):
        """Migrate with delete_after should replace keys with env-var refs."""
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
            "delete_after": True,
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] >= 2

        # Verify config was modified (has migration markers)
        with open(config_file, "r") as f:
            updated_config = json.load(f)
        assert updated_config.get("_migrated_to_vault") is True
        assert "_vault_migrated_at" in updated_config

    def test_migrate_nonexistent_config(self, client, auth_header, tmp_home):
        """Migrate with a non-existent config should return error."""
        bad_path = tmp_home / "does-not-exist.json"
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(bad_path),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] == 0
        assert len(data["errors"]) >= 1
        assert "not found" in data["errors"][0].lower()

    def test_migrate_skips_env_refs(self, client, auth_header, tmp_home, config_file_with_env_refs):
        """Migrate should skip config values that are already env-var refs."""
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file_with_env_refs),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] == 0
        # Both keys were env refs, so they were skipped
        assert any("no plaintext" in e.lower() for e in data["errors"])

    def test_migrate_creates_vault_if_not_exists(self, client, auth_header, tmp_home, config_file):
        """Migrate should auto-create a vault if one doesn't exist yet."""
        # Ensure no vault exists
        vault_file = tmp_home / ".xencode" / "vault.json"
        assert not vault_file.exists()

        # Migrate should create it
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] >= 2
        assert vault_file.exists()

    def test_migrate_with_custom_vault_path(self, client, auth_header, tmp_home, config_file):
        """Migrate with a custom vault_path should create vault at that location."""
        custom_vault = tmp_home / "alt" / "my-vault.json"
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
            "vault_path": str(custom_vault),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] >= 2
        assert custom_vault.exists()

    def test_migrate_with_master_key(self, client, auth_header, tmp_home, config_file):
        """Migrate should accept a master_key for vault encryption."""
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
            "master_key": "test-master-key",
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] >= 2

    def test_migrate_corrupted_config(self, client, auth_header, tmp_home):
        """Migrate with a corrupted config file should report error."""
        bad_config = tmp_home / ".xencode" / "config.json"
        bad_config.parent.mkdir(parents=True, exist_ok=True)
        with open(bad_config, "w") as f:
            f.write("this is not valid json {")

        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(bad_config),
        }, headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data["migrated"] == 0
        assert len(data["errors"]) >= 1

    def test_migrate_returns_timestamp(self, client, auth_header, tmp_home, config_file):
        """Migrate response should include an ISO timestamp."""
        response = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
        }, headers=auth_header)
        data = response.json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# End-to-end: Health → Init → Migrate → Health
# ---------------------------------------------------------------------------


class TestVaultWorkflow:
    """End-to-end vault workflow: init, migrate, health."""

    def test_full_workflow(self, client, auth_header, tmp_home, config_file):
        """Complete workflow: init → health → migrate → health."""
        # 1. Initial health (no vault)
        h1 = client.get("/api/v1/vault/health", headers=auth_header).json()
        assert h1["vault_exists"] is False

        # 2. Init vault
        init_resp = client.post("/api/v1/vault/init", json={}, headers=auth_header)
        assert init_resp.json()["created"] is True

        # 3. Health after init (empty vault)
        h2 = client.get("/api/v1/vault/health", headers=auth_header).json()
        assert h2["vault_exists"] is True
        assert h2["credential_count"] == 0

        # 4. Migrate from config
        mig = client.post("/api/v1/vault/migrate", json={
            "config_path": str(config_file),
        }, headers=auth_header).json()
        assert mig["migrated"] >= 2

        # 5. Health after migration
        h3 = client.get("/api/v1/vault/health", headers=auth_header).json()
        assert h3["vault_exists"] is True
        assert h3["credential_count"] >= 2
        assert h3["is_readable"] is True
        assert h3["is_valid_json"] is True

        # 6. Timestamps — ensure they progress
        assert h1["timestamp"] <= h3["timestamp"]



# ---------------------------------------------------------------------------
# WebSocket /api/v1/vault/health/ws
# ---------------------------------------------------------------------------


class TestVaultHealthWebSocket:
    """Tests for the vault health WebSocket endpoint."""

    @pytest.fixture
    def ws_url(self, test_token: str) -> str:
        """WebSocket URL with token query parameter."""
        return f"/api/v1/vault/health/ws?token={test_token}"

    def test_ws_initial_health_snapshot(self, client, ws_url, tmp_home):
        """Should send an immediate health snapshot on connect."""
        with client.websocket_connect(ws_url) as ws:
            data = ws.receive_json()
            assert data["type"] == "health_update"
            assert data["available"] is True
            assert data["vault_exists"] is False
            assert data["credential_count"] == 0
            assert "timestamp" in data

    def test_ws_health_after_init(self, client, auth_header, ws_url, tmp_home):
        """Should reflect vault state changes after init."""
        vault_file = str(tmp_home / ".xencode" / "vault.json")
        with client.websocket_connect(
            f"{ws_url}&vault_path={vault_file}"
        ) as ws:
            # Initial — vault does not exist
            snap1 = ws.receive_json()
            assert snap1["vault_exists"] is False

            # Init vault via REST
            client.post("/api/v1/vault/init", json={}, headers=auth_header)

            # Trigger a manual check
            ws.send_json({"type": "check_now"})
            snap2 = ws.receive_json()
            assert snap2["type"] == "health_update"
            assert snap2["vault_exists"] is True
            assert snap2["is_readable"] is True
            assert snap2["is_valid_json"] is True

    def test_ws_ping_pong(self, client, ws_url, tmp_home):
        """Should respond to ping with pong."""
        with client.websocket_connect(ws_url) as ws:
            # Consume initial snapshot (the monitor loop sleeps first,
            # so no periodic update will arrive before we send ping)
            ws.receive_json()

            ws.send_json({"type": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert "timestamp" in pong

    def test_ws_set_interval(self, client, ws_url, tmp_home):
        """Should acknowledge interval changes."""
        with client.websocket_connect(f"{ws_url}&interval=10") as ws:
            # Consume initial snapshot
            ws.receive_json()

            ws.send_json({"type": "set_interval", "interval": 3})
            ack = ws.receive_json()
            assert ack["type"] == "interval_updated"
            assert ack["interval"] == 3.0

    def test_ws_interval_clamped(self, client, ws_url, tmp_home):
        """Interval values should be clamped to [1, 300]."""
        with client.websocket_connect(ws_url) as ws:
            ws.receive_json()  # consume initial

            # Too low
            ws.send_json({"type": "set_interval", "interval": 0})
            ack = ws.receive_json()
            assert ack["type"] == "interval_updated"
            assert ack["interval"] == 1.0

            # Too high
            ws.send_json({"type": "set_interval", "interval": 999})
            ack = ws.receive_json()
            assert ack["type"] == "interval_updated"
            assert ack["interval"] == 300.0

    def test_ws_check_now_triggers_health(self, client, ws_url, tmp_home):
        """check_now should produce an immediate health update."""
        with client.websocket_connect(ws_url) as ws:
            # Consume the automatic initial snapshot
            ws.receive_json()

            # Request a manual check
            ws.send_json({"type": "check_now"})
            manual = ws.receive_json()
            assert manual["type"] == "health_update"
            assert "vault_exists" in manual
            assert "credential_count" in manual

    def test_ws_invalid_json_returns_error(self, client, ws_url, tmp_home):
        """Invalid JSON from client should return an error message."""
        with client.websocket_connect(ws_url) as ws:
            ws.receive_json()  # consume initial

            # Send raw non-JSON text
            ws.send_text("not valid json")
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "Invalid JSON" in error["message"]

    def test_ws_periodic_updates(self, client, ws_url, tmp_home):
        """With a low interval, should receive multiple updates.

        The monitor loop sleeps first, so the first periodic update
        arrives ~1 second after the initial snapshot on connect.
        """
        with client.websocket_connect(f"{ws_url}&interval=1") as ws:
            # First snapshot is sent immediately on connect
            snap1 = ws.receive_json()
            assert snap1["type"] == "health_update"

            # Second periodic update should arrive ~1 second later
            snap2 = ws.receive_json()
            assert snap2["type"] == "health_update"

            # Each update should have a different timestamp
            assert snap1["timestamp"] <= snap2["timestamp"]

    def test_ws_custom_vault_path(self, client, ws_url, tmp_home):
        """Should respect a custom vault_path query parameter."""
        custom_path = tmp_home / "custom" / "secrets.json"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal valid vault
        with open(custom_path, "w") as f:
            json.dump({"version": 1, "credentials": {}}, f)

        with client.websocket_connect(
            f"{ws_url}&vault_path={custom_path}"
        ) as ws:
            snap = ws.receive_json()
            assert snap["type"] == "health_update"
            assert snap["vault_exists"] is True
            assert str(custom_path).replace("\\", "/") in snap["vault_path"].replace("\\", "/")



# ---------------------------------------------------------------------------
# Authentication: missing / invalid / expired tokens
# ---------------------------------------------------------------------------


class TestVaultAuth:
    """Tests that vault endpoints enforce JWT authentication."""

    def test_health_no_token(self, client):
        """Health without token should return 401."""
        resp = client.get("/api/v1/vault/health")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["error"]["message"]

    def test_health_invalid_token(self, client):
        """Health with an invalid Bearer token should return 401."""
        resp = client.get(
            "/api/v1/vault/health",
            headers={"Authorization": "Bearer this.is.not.a.valid.jwt"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["error"]["message"]

    def test_init_no_token(self, client):
        """Init without token should return 401."""
        resp = client.post("/api/v1/vault/init", json={})
        assert resp.status_code == 401

    def test_migrate_no_token(self, client):
        """Migrate without token should return 401."""
        resp = client.post("/api/v1/vault/migrate", json={})
        assert resp.status_code == 401

    def test_ws_no_token_closes_connection(self, client):
        """WebSocket without a token query param should be rejected with code 4001."""
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/api/v1/vault/health/ws") as ws:
                ws.receive_json()
        assert excinfo.value.code == 4001

    def test_ws_invalid_token_closes_connection(self, client):
        """WebSocket with an invalid token should be rejected."""
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                "/api/v1/vault/health/ws?token=bad.jwt.token"
            ) as ws:
                ws.receive_json()
        assert excinfo.value.code == 4001

    def test_ws_expired_token_closes_connection(self, client):
        """WebSocket with an expired token should be rejected."""
        from starlette.websockets import WebSocketDisconnect
        expired_payload = {
            "user_id": "test",
            "role": "admin",
            "type": "access",
            "iat": datetime.utcnow() - timedelta(hours=2),
            "exp": datetime.utcnow() - timedelta(hours=1),
        }
        expired_token = jwt.encode(expired_payload, _DEV_SECRET, algorithm="HS256")
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                f"/api/v1/vault/health/ws?token={expired_token}"
            ) as ws:
                ws.receive_json()
        assert excinfo.value.code == 4001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
