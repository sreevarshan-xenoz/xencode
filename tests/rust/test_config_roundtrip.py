import json
import shutil
import subprocess
from pathlib import Path

import pytest

def test_rust_config_show(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")

    workspace_root = Path(__file__).resolve().parents[2] / "rust"
    
    # Run config show
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "config",
            "show",
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )
    
    config = json.loads(completed.stdout)
    assert "default_model" in config
    assert "ollama_url" in config
    assert "cache_enabled" in config
    assert "api_keys" in config
