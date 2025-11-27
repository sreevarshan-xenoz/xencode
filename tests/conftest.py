"""Test fixtures and utilities for Xencode tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


# Mock fixtures removed in favor of real system tests.
# See real_ollama_client fixture below.


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / ".xencode"
        config_path.mkdir()
        yield config_path


# Remaining mock fixtures removed.


@pytest.fixture
def real_ollama_client():
    """Fixture for real Ollama client interaction."""
    # This assumes Ollama is running locally.
    # Tests using this should be skipped if Ollama is not available.
    import requests
    return requests

@pytest.fixture
def system_checker():
    """Fixture for SystemChecker."""
    from xencode.system_checker import SystemChecker
    return SystemChecker()

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring a running Ollama instance"
    )

@pytest.fixture(autouse=True)
def skip_if_no_ollama(request):
    """Skip tests marked with requires_ollama if Ollama is not running."""
    if request.node.get_closest_marker("requires_ollama"):
        import requests
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
        except requests.RequestException:
            pytest.fail("Test requires running Ollama instance, but it is not accessible.")

