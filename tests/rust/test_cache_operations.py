import shutil
import subprocess
from pathlib import Path

import pytest

def test_rust_cache_operations(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")

    workspace_root = Path(__file__).resolve().parents[2] / "rust"
    
    # Run cache stats
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "cache",
            "stats",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    assert "Cache Statistics:" in completed.stdout
    assert "entries:" in completed.stdout
    
    # Run cache clear
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "cache",
            "clear",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    assert "Cache cleared successfully" in completed.stdout or "No cache to clear" in completed.stdout
