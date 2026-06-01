import shutil
import subprocess
from pathlib import Path

import pytest

def test_rust_memory_operations(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")

    workspace_root = Path(__file__).resolve().parents[2] / "rust"
    
    # Run memory list
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "memory",
            "list",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    # Should say "No conversation sessions found." or "Conversation Sessions:"
    assert "conversation" in completed.stdout.lower()

    # Create a dummy memory entry using query (with a dummy non-existing model to fail fast but still attempt)
    # Actually query requires Ollama, so just check the help command of query
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "query",
            "--help",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    assert "--model" in completed.stdout
    assert "--session" in completed.stdout
