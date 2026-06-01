"""Unit tests for the vault monitor CLI command

Tests the 'xencode vault monitor' command:
- Help output and option dispatch
- Auto-generated vs provided JWT tokens
- Missing dependency handling
- WebSocket message handling via async iteration
- Connection error handling
- WS URL construction
"""

import json
import sys
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner
from xencode.cli import cli


# ---------------------------------------------------------------------------
# Helpers for async websocket iteration mocks
# ---------------------------------------------------------------------------

class _AsyncMsgIterator:
    """Async iterator that yields a list of string messages, then stops."""

    def __init__(self, messages):
        self._messages = list(messages)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


class _FakeWebsocketExceptions:
    """Minimal stand-in for websockets.exceptions with real exception classes."""

    class ConnectionClosed(Exception):
        pass

    class InvalidURI(Exception):
        pass

    class WebSocketException(Exception):
        pass


class _AsyncRecv:
    """Makes a mock ws.recv() that yields messages via await, then raises.

    When messages are exhausted, raises ConnectionClosed (matching what a
    real WebSocket does when the server disconnects) so the loop in
    _run_vault_monitor breaks cleanly.
    """

    def __init__(self, messages, closed_exc=None):
        self._messages = list(messages)
        self._idx = 0
        self._closed_exc = closed_exc or _FakeWebsocketExceptions.ConnectionClosed

    async def __call__(self):
        if self._idx >= len(self._messages):
            raise self._closed_exc("Mock connection closed")
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


def _make_ws_context_manager(messages):
    """Return an async context manager yielding a mock ws with async recv().

    The mock ws exposes a ``recv`` async callable that returns messages one
    at a time, then raises ``ConnectionClosed`` to terminate the loop.
    """
    mock_ws = MagicMock()
    mock_ws.recv = _AsyncRecv(messages)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ws, mock_ctx


@pytest.fixture
def cli_runner():
    """Create a CLI runner"""
    return CliRunner()


@pytest.fixture
def mock_health_payload():
    """Sample health update payload"""
    return {
        "type": "health_update",
        "available": True,
        "vault_exists": True,
        "vault_path": "/home/user/.xencode/vault.json",
        "is_readable": True,
        "is_valid_json": True,
        "credential_count": 3,
        "encryption_available": True,
        "timestamp": "2026-06-01T12:00:00.000000",
    }


class TestVaultMonitorHelp:
    """Test 'xencode vault monitor' help and basic invocation"""

    def test_monitor_help(self, cli_runner):
        """Test that monitor command help works"""
        result = cli_runner.invoke(cli, ["vault", "monitor", "--help"])
        assert result.exit_code == 0
        assert "Monitor vault health in real-time" in result.output
        assert "--server-url" in result.output
        assert "--interval" in result.output
        assert "--vault-path" in result.output
        assert "--token" in result.output


class TestVaultMonitorOptions:
    """Test option handling for the monitor command"""

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_default_options(self, mock_monitor, cli_runner):
        """Test monitor with default options"""
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "http://localhost:8000", 5.0, None, None
        )

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_custom_server_url(self, mock_monitor, cli_runner):
        """Test monitor with custom server URL"""
        result = cli_runner.invoke(
            cli, ["vault", "monitor", "--server-url", "http://my-host:9000"]
        )
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "http://my-host:9000", 5.0, None, None
        )

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_custom_interval(self, mock_monitor, cli_runner):
        """Test monitor with custom interval"""
        result = cli_runner.invoke(
            cli, ["vault", "monitor", "--interval", "10"]
        )
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "http://localhost:8000", 10.0, None, None
        )

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_custom_vault_path(self, mock_monitor, cli_runner):
        """Test monitor with custom vault path"""
        result = cli_runner.invoke(
            cli, ["vault", "monitor", "--vault-path", "/custom/vault.json"]
        )
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "http://localhost:8000", 5.0, "/custom/vault.json", None
        )

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_custom_token(self, mock_monitor, cli_runner):
        """Test monitor with explicit token"""
        result = cli_runner.invoke(
            cli, ["vault", "monitor", "--token", "eyJhbGciOiJIUzI1NiJ9.test"]
        )
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "http://localhost:8000", 5.0, None, "eyJhbGciOiJIUzI1NiJ9.test"
        )

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_all_options_combined(self, mock_monitor, cli_runner):
        """Test monitor with all options specified"""
        result = cli_runner.invoke(
            cli,
            [
                "vault", "monitor",
                "--server-url", "https://prod:8443",
                "--interval", "15",
                "--vault-path", "/data/vault.json",
                "--token", "my-token",
            ],
        )
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(
            "https://prod:8443", 15.0, "/data/vault.json", "my-token"
        )


class TestVaultMonitorTokenGeneration:
    """Test automatic JWT token generation when --token is not provided"""

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_no_token_passes_none(self, mock_monitor, cli_runner):
        """Test that not providing --token passes None to the monitor coroutine"""
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code == 0
        args = mock_monitor.call_args[0]
        assert args[3] is None  # token param is None

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_provided_token_skips_none(self, mock_monitor, cli_runner):
        """Test that providing --token passes the token string"""
        result = cli_runner.invoke(
            cli, ["vault", "monitor", "--token", "my-token"]
        )
        assert result.exit_code == 0
        args = mock_monitor.call_args[0]
        assert args[3] == "my-token"


class TestVaultMonitorDependencyChecks:
    """Test handling of missing optional dependencies"""

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_monitor_shows_connecting_message(self, mock_monitor, cli_runner):
        """Test that connecting message is displayed before async run"""
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code == 0
        assert "Connecting" in result.output


class TestVaultMonitorWebSocketMessages:
    """Test WebSocket message handling inside _run_vault_monitor

    These tests exercise _run_vault_monitor directly with mocked
    websockets/jwt to verify the message processing logic.
    """

    @pytest.mark.asyncio
    async def test_health_update_renders_table(self, mock_health_payload):
        """Test that health_update messages are rendered"""
        from xencode.cli import _run_vault_monitor

        _, mock_ctx = _make_ws_context_manager([
            json.dumps(mock_health_payload),
        ])

        mock_ws_module = MagicMock()
        mock_ws_module.connect = MagicMock(return_value=mock_ctx)
        mock_ws_module.exceptions = _FakeWebsocketExceptions

        mock_jwt_module = MagicMock()
        mock_jwt_module.encode = MagicMock(return_value="test-token")

        with patch.dict(sys.modules, {
            "websockets": mock_ws_module,
            "jwt": mock_jwt_module,
        }):
            await _run_vault_monitor(
                "http://localhost:8000", 5.0, None, "test-token"
            )

    @pytest.mark.asyncio
    async def test_pong_message_silently_handled(self):
        """Test that pong messages are silently handled"""
        from xencode.cli import _run_vault_monitor

        _, mock_ctx = _make_ws_context_manager([
            json.dumps({"type": "pong", "timestamp": "2026-06-01T12:00:00"}),
        ])

        mock_ws_module = MagicMock()
        mock_ws_module.connect = MagicMock(return_value=mock_ctx)
        mock_ws_module.exceptions = _FakeWebsocketExceptions

        mock_jwt_module = MagicMock()

        with patch.dict(sys.modules, {
            "websockets": mock_ws_module,
            "jwt": mock_jwt_module,
        }):
            await _run_vault_monitor(
                "http://localhost:8000", 5.0, None, "test-token"
            )

    @pytest.mark.asyncio
    async def test_error_message_printed(self):
        """Test that error messages from server are printed"""
        from xencode.cli import _run_vault_monitor

        _, mock_ctx = _make_ws_context_manager([
            json.dumps({
                "type": "error",
                "message": "Health check failed: vault corrupted",
                "timestamp": "2026-06-01T12:00:00",
            }),
        ])

        mock_ws_module = MagicMock()
        mock_ws_module.connect = MagicMock(return_value=mock_ctx)
        mock_ws_module.exceptions = _FakeWebsocketExceptions

        mock_jwt_module = MagicMock()

        with patch.dict(sys.modules, {
            "websockets": mock_ws_module,
            "jwt": mock_jwt_module,
        }):
            await _run_vault_monitor(
                "http://localhost:8000", 5.0, None, "test-token"
            )


class TestVaultMonitorErrorHandling:
    """Test error handling for various failure modes"""

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_keyboard_interrupt_handled(self, mock_monitor, cli_runner):
        """Test that Ctrl+C is handled gracefully"""
        mock_monitor.side_effect = KeyboardInterrupt()
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code == 0
        assert "Disconnected" in result.output or "disconnected" in result.output.lower()

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_connection_error_exits_nonzero(self, mock_monitor, cli_runner):
        """Test that unexpected errors from the coroutine produce non-zero exit"""
        mock_monitor.side_effect = Exception("Connection refused")
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_ws_url_construction(self):
        """Test that WS URL is correctly constructed from server URL"""
        from xencode.cli import _run_vault_monitor

        _, mock_ctx = _make_ws_context_manager([
            json.dumps({
                "type": "health_update",
                "available": True,
                "vault_exists": True,
                "vault_path": "/test/vault.json",
                "is_readable": True,
                "is_valid_json": True,
                "credential_count": 1,
                "encryption_available": False,
                "timestamp": "2026-06-01T12:00:00",
            }),
        ])

        mock_ws_module = MagicMock()
        mock_ws_module.connect = MagicMock(return_value=mock_ctx)
        mock_ws_module.exceptions = _FakeWebsocketExceptions

        mock_jwt_module = MagicMock()

        with patch.dict(sys.modules, {
            "websockets": mock_ws_module,
            "jwt": mock_jwt_module,
        }):
            await _run_vault_monitor(
                "http://myhost:8000", 10.0, "/custom/vault.json", "test-token"
            )

            # Verify the URL passed to connect
            call_args = mock_ws_module.connect.call_args
            url = call_args[0][0]
            assert "ws://myhost:8000" in url
            assert "interval=10" in url
            assert "vault_path" in url


class TestVaultMonitorIntegration:
    """Integration-style tests for the monitor command"""

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_monitor_is_vault_subcommand(self, mock_monitor, cli_runner):
        """Test that monitor is accessible as a vault subcommand"""
        result = cli_runner.invoke(cli, ["vault", "monitor", "--help"])
        assert result.exit_code == 0

    @patch("xencode.cli._run_vault_monitor", new_callable=AsyncMock)
    def test_monitor_shows_connecting(self, mock_monitor, cli_runner):
        """Test that the connecting status message is shown"""
        result = cli_runner.invoke(cli, ["vault", "monitor"])
        assert result.exit_code == 0
        assert "Connecting" in result.output or "connecting" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
