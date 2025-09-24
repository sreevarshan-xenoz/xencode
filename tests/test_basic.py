"""Simple test to verify basic functionality."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that we can import the main modules."""
    try:
        import xencode_core

        assert hasattr(xencode_core, 'DEFAULT_MODEL')
        assert xencode_core.DEFAULT_MODEL == "qwen3:4b"
    except ImportError as e:
        pytest.fail(f"Failed to import xencode_core: {e}")


def test_constants():
    """Test that constants are properly defined."""
    import xencode_core

    assert isinstance(xencode_core.DEFAULT_MODEL, str)
    assert len(xencode_core.DEFAULT_MODEL) > 0


def test_console_initialization():
    """Test that Rich console is initialized."""
    import xencode_core

    assert hasattr(xencode_core, 'console')
    assert xencode_core.console is not None


if __name__ == "__main__":
    pytest.main([__file__])
