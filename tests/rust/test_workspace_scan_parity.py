import os
import shutil
import subprocess
from pathlib import Path

import pytest

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
    "node_modules",
    "target",
    "venv",
}


def python_scan(root: Path) -> list[str]:
    entries: list[str] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in EXCLUDED_DIRS and not dirname.startswith(".")
        ]

        current_path = Path(current)
        for dirname in dirnames:
            entries.append((current_path / dirname).relative_to(root).as_posix())
        for filename in filenames:
            if filename.startswith("."):
                continue
            entries.append((current_path / filename).relative_to(root).as_posix())

    return sorted(entries)


def test_rust_workspace_scan_matches_python_baseline(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "artifact").write_text("ignored\n", encoding="utf-8")

    workspace_root = Path(__file__).resolve().parents[2] / "rust"
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "xencode-cli",
            "--",
            "scan",
            str(tmp_path),
        ],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )

    rust_paths = sorted(
        line.split("\t", 2)[2].replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip()
    )

    assert rust_paths == python_scan(tmp_path)
