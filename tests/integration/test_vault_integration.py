#!/usr/bin/env python3
"""
Integration Tests for Vault API — Live Dev Server

Spins up a real uvicorn subprocess on a random port (with a dedicated temp
``HOME`` directory) and tests the vault endpoints (health, init, migrate)
over actual HTTP.

This is in contrast to the unit tests in ``tests/api/test_vault_api.py`` which
use FastAPI's in-process ``TestClient``.
"""

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import jwt
import pytest

pytestmark = [pytest.mark.integration]

# Dev secret key (mirrors xencode/api/auth.py fallback)
_DEV_SECRET = "dev-secret-key-change-in-production"


@pytest.fixture(scope="session")
def test_token() -> str:
    """Generate a valid JWT access token for testing."""
    payload = {
        "user_id": "integration-test",
        "username": "integration",
        "role": "admin",
        "session_id": "integration-session",
        "type": "access",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=2),
    }
    return jwt.encode(payload, _DEV_SECRET, algorithm="HS256")

# ---------------------------------------------------------------------------
# Module-scoped server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory) -> tuple[str, Path]:
    """Start a uvicorn dev server on a random port.

    The server runs with a temp ``HOME`` directory so tests can create vault
    files at the default location (``~/.xencode/vault.json``) that the
    server's health endpoint can discover.

    Yields ``(url, vault_home)`` where *vault_home* is the temp home
    directory the server sees as ``Path.home()``.
    """
    vault_home = tmp_path_factory.mktemp("vault_home")

    # Pick a random available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = os.environ.copy()
    env["HOME"] = str(vault_home)
    # Also set USERPROFILE on Windows so Path.home() resolves correctly
    env["USERPROFILE"] = str(vault_home)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "xencode.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
            "--lifespan",
            "off",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = f"http://127.0.0.1:{port}"

    # Wait for the server to be ready
    deadline = time.monotonic() + 15
    started = False
    while time.monotonic() < deadline:
        try:
            with httpx.Client() as client:
                r = client.get(f"{url}/health", timeout=1)
                if r.status_code == 200:
                    started = True
                    break
        except Exception:
            time.sleep(0.5)

    if not started:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        raise RuntimeError(
            f"Dev server failed to start on port {port} within 20s"
        )

    yield url, vault_home

    # Teardown
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)

    # Drain pipes to avoid ResourceWarning
    proc.stdout.close()
    proc.stderr.close()


@pytest.fixture
def server_url(server: tuple[str, Path]) -> str:
    """The live server base URL."""
    return server[0]


@pytest.fixture
def auth_header(test_token: str) -> dict:
    """Authorization header with a valid Bearer token."""
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture
def vault_home(server: tuple[str, Path]) -> Path:
    """The temp HOME directory that the server uses.

    Vault files placed at ``vault_home / ".xencode" / "vault.json"`` are
    visible to the server's ``GET /api/v1/vault/health`` endpoint.
    """
    return server[1]


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_vault(vault_home: Path) -> None:
    """Remove the default vault file after every test.

    Because the server process is module-scoped, tests that initialise a vault
    at the default location would otherwise leak state into subsequent tests.
    """
    yield
    vault_file = vault_home / ".xencode" / "vault.json"
    if vault_file.exists():
        vault_file.unlink()


@pytest.fixture
def client(server_url: str, auth_header: dict) -> httpx.Client:
    """Return an httpx Client pre-configured with the live server URL and auth."""
    with httpx.Client(base_url=server_url, headers=auth_header) as c:
        yield c


def _create_config(vault_home: Path, **overrides: str) -> Path:
    """Write a minimal config with plaintext API keys under *vault_home*."""
    config = {
        "features": {
            "code_review": {
                "openai_api_key": overrides.get(
                    "openai_api_key", "sk-live-test-openai-98765"
                ),
                "model": "gpt-4",
            },
            "learning_mode": {
                "anthropic_api_key": overrides.get(
                    "anthropic_api_key", "sk-live-test-anthropic-54321"
                ),
                "model": "claude-3",
            },
        },
        "settings": {"max_tokens": 2048, "temperature": 0.7},
    }
    config_path = vault_home / ".xencode" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


# ===================================================================
# Tests
# ===================================================================


class TestVaultHealthIntegration:
    """GET /api/v1/vault/health against the live server."""

    def test_health_returns_200(self, client: httpx.Client) -> None:
        """The endpoint responds with 200 and a valid JSON body."""
        resp = client.get("/api/v1/vault/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "vault_exists" in data
        assert "credential_count" in data
        assert "vault_path" in data
        assert "timestamp" in data

    def test_health_vault_not_found(self, client: httpx.Client) -> None:
        """No vault file exists yet — vault_exists is False."""
        resp = client.get("/api/v1/vault/health")
        data = resp.json()
        assert data["vault_exists"] is False
        assert data["credential_count"] == 0

    def test_health_after_init(self, client: httpx.Client, vault_home: Path) -> None:
        """Init vault at the default location, then health finds it."""
        resp = client.post("/api/v1/vault/init", json={})
        assert resp.status_code == 200
        assert resp.json()["created"] is True

        resp2 = client.get("/api/v1/vault/health")
        data = resp2.json()
        assert data["vault_exists"] is True
        assert data["is_readable"] is True
        assert data["is_valid_json"] is True
        assert data["credential_count"] == 0

    def test_health_encryption_field(self, client: httpx.Client) -> None:
        """encryption_available reflects whether Fernet is importable."""
        resp = client.get("/api/v1/vault/health")
        assert isinstance(resp.json()["encryption_available"], bool)

    def test_health_timestamp_format(self, client: httpx.Client) -> None:
        """Timestamp is ISO 8601 formatted."""
        ts = client.get("/api/v1/vault/health").json()["timestamp"]
        assert "T" in ts

    def test_health_no_auth(self, vault_home: Path, server_url: str) -> None:
        """Health without Authorization header returns 401."""
        with httpx.Client(base_url=server_url) as unauth:
            resp = unauth.get("/api/v1/vault/health")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["error"]["message"]


class TestVaultInitIntegration:
    """POST /api/v1/vault/init against the live server."""

    def test_init_creates_vault(self, client: httpx.Client, vault_home: Path) -> None:
        """Successful init creates the vault file on disk."""
        resp = client.post("/api/v1/vault/init", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] is True
        assert data["already_exists"] is False
        assert data["error"] is None
        default_vault = vault_home / ".xencode" / "vault.json"
        assert default_vault.exists()

    def test_init_idempotent(self, client: httpx.Client) -> None:
        """Second call returns already_exists."""
        client.post("/api/v1/vault/init", json={})
        resp = client.post("/api/v1/vault/init", json={})
        data = resp.json()
        assert data["created"] is False
        assert data["already_exists"] is True

    def test_init_with_master_key(self, client: httpx.Client) -> None:
        """Init with a master key succeeds."""
        resp = client.post(
            "/api/v1/vault/init",
            json={"master_key": "integration-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] is True
        assert data["error"] is None

    def test_init_with_custom_vault_path(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Init with a custom vault_path creates the file at that path."""
        custom_path = vault_home / "custom" / "secrets.json"
        resp = client.post(
            "/api/v1/vault/init",
            json={"vault_path": str(custom_path)},
        )
        assert resp.status_code == 200
        assert resp.json()["created"] is True
        assert custom_path.exists()

    def test_init_timestamp(self, client: httpx.Client) -> None:
        """Response includes ISO timestamp."""
        ts = client.post("/api/v1/vault/init", json={}).json()["timestamp"]
        assert "T" in ts


class TestVaultMigrateIntegration:
    """POST /api/v1/vault/migrate against the live server."""

    def test_migrate_basic(self, client: httpx.Client, vault_home: Path) -> None:
        """Migrate extracts API keys from config into vault."""
        config_p = _create_config(vault_home)

        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["migrated"] >= 2
        assert data["errors"] == []

    def test_migrate_delete_after(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """delete_after removes plaintext keys from config."""
        config_p = _create_config(vault_home)

        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p), "delete_after": True},
        )
        assert resp.status_code == 200
        assert resp.json()["migrated"] >= 2

        # Config should now have migration markers
        with open(config_p, "r") as f:
            updated = json.load(f)
        assert updated.get("_migrated_to_vault") is True
        assert "_vault_migrated_at" in updated

    def test_migrate_nonexistent_config(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Non-existent config returns error."""
        bad = vault_home / "nope.json"
        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(bad)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["migrated"] == 0
        assert len(data["errors"]) >= 1
        assert "not found" in data["errors"][0].lower()

    def test_migrate_skips_env_refs(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Config values that are env-var refs (${}, env:) are skipped."""
        config_p = vault_home / ".xencode" / "config.json"
        config_p.parent.mkdir(parents=True, exist_ok=True)
        with open(config_p, "w") as f:
            json.dump(
                {
                    "providers": {
                        "openai": {"api_key": "${OPENAI_API_KEY}"},
                        "anthropic": {"api_key": "env:ANTHROPIC_API_KEY"},
                    }
                },
                f,
            )

        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["migrated"] == 0
        assert any("no plaintext" in e.lower() for e in data["errors"])

    def test_migrate_auto_creates_vault(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Migrate creates a vault automatically if none exists."""
        config_p = _create_config(vault_home)
        default_vault = vault_home / ".xencode" / "vault.json"

        assert not default_vault.exists()
        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p)},
        )
        assert resp.status_code == 200
        assert resp.json()["migrated"] >= 2
        assert default_vault.exists()

    def test_migrate_corrupted_config(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Malformed JSON config is caught and reported."""
        config_p = vault_home / ".xencode" / "config.json"
        config_p.parent.mkdir(parents=True, exist_ok=True)
        with open(config_p, "w") as f:
            f.write("{ this is not valid json")

        resp = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["migrated"] == 0
        assert len(data["errors"]) >= 1


class TestVaultWorkflowIntegration:
    """End-to-end: health → init → migrate → health."""

    def test_full_workflow(self, client: httpx.Client, vault_home: Path) -> None:
        """Realistic workflow exercising all three REST endpoints."""
        config_p = _create_config(vault_home)

        # 1. Health — no vault
        h1 = client.get("/api/v1/vault/health").json()
        assert h1["vault_exists"] is False

        # 2. Init
        init_r = client.post("/api/v1/vault/init", json={}).json()
        assert init_r["created"] is True

        # 3. Health after init
        h2 = client.get("/api/v1/vault/health").json()
        assert h2["vault_exists"] is True
        assert h2["credential_count"] == 0

        # 4. Migrate
        mig = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p)},
        ).json()
        assert mig["migrated"] >= 2

        # 5. Health after migrate
        h3 = client.get("/api/v1/vault/health").json()
        assert h3["vault_exists"] is True
        assert h3["credential_count"] >= 2
        assert h3["is_readable"] is True
        assert h3["is_valid_json"] is True

        # 6. Timestamps progress
        assert h1["timestamp"] < h3["timestamp"]

    def test_full_workflow_with_master_key(
        self, client: httpx.Client, vault_home: Path
    ) -> None:
        """Full workflow using a master key for encryption."""
        config_p = _create_config(vault_home)

        # Init with master key
        init_r = client.post(
            "/api/v1/vault/init",
            json={"master_key": "my-master-key"},
        ).json()
        assert init_r["created"] is True

        # Migrate with same master key
        mig = client.post(
            "/api/v1/vault/migrate",
            json={"config_path": str(config_p), "master_key": "my-master-key"},
        ).json()
        assert mig["migrated"] >= 2

        # Verify
        h = client.get("/api/v1/vault/health").json()
        assert h["credential_count"] >= 2


class TestLiveServerHeaders:
    """Verify real HTTP behaviours only observable on a live server."""

    def test_has_server_date_headers(self, client: httpx.Client) -> None:
        """Live server includes standard HTTP headers."""
        resp = client.get("/api/v1/vault/health")
        assert "server" in resp.headers or "date" in resp.headers
        assert resp.headers.get("content-type") == "application/json"

    def test_404_for_unknown_route(self, client: httpx.Client) -> None:
        """A vault endpoint on the wrong prefix returns 404."""
        resp = client.get("/api/v1/vault/nonexistent")
        assert resp.status_code == 404

    def test_init_no_auth(self, vault_home: Path, server_url: str) -> None:
        """Init without Authorization header returns 401."""
        with httpx.Client(base_url=server_url) as unauth:
            resp = unauth.post("/api/v1/vault/init", json={})
        assert resp.status_code == 401

    def test_migrate_no_auth(self, vault_home: Path, server_url: str) -> None:
        """Migrate without Authorization header returns 401."""
        with httpx.Client(base_url=server_url) as unauth:
            resp = unauth.post("/api/v1/vault/migrate", json={})
        assert resp.status_code == 401

    def test_invalid_token(self, server_url: str) -> None:
        """Invalid Bearer token returns 401."""
        with httpx.Client(
            base_url=server_url,
            headers={"Authorization": "Bearer not-a-real-token"},
        ) as bad:
            resp = bad.get("/api/v1/vault/health")
        assert resp.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
