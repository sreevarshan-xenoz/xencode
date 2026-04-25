#!/usr/bin/env python3
"""Baseline benchmarks for the Python-to-Rust migration.

The script intentionally uses only the Python standard library so it can run
before project dependencies are installed. Results are printed as JSON lines so
they can be appended to a file and compared over time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_EXCLUDED_DIRS = {
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


def measure(name: str, iterations: int, fn: Callable[[], object]) -> dict:
    durations = []
    last_result = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_result = fn()
        durations.append((time.perf_counter() - start) * 1000)

    return {
        "benchmark": name,
        "iterations": iterations,
        "min_ms": min(durations),
        "median_ms": statistics.median(durations),
        "max_ms": max(durations),
        "result": last_result,
    }


def scan_workspace(root: Path, include_hidden: bool = False) -> dict:
    files = 0
    directories = 0
    total_bytes = 0

    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if should_include(dirname, include_hidden)
        ]

        visible_filenames = [
            filename
            for filename in filenames
            if should_include(filename, include_hidden)
        ]

        directories += len(dirnames)
        files += len(visible_filenames)

        for filename in visible_filenames:
            path = Path(current) / filename
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    return {
        "files": files,
        "directories": directories,
        "bytes": total_bytes,
    }


def should_include(name: str, include_hidden: bool) -> bool:
    if name in DEFAULT_EXCLUDED_DIRS:
        return False
    if not include_hidden and name.startswith("."):
        return False
    return True


def command_probe(command: list[str], cwd: Path) -> dict:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout_bytes": len(completed.stdout.encode("utf-8", errors="replace")),
            "stderr_bytes": len(completed.stderr.encode("utf-8", errors="replace")),
        }
    except FileNotFoundError:
        return {"error": "command not found", "command": command[0]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "command": command}


def emit_json_lines(results: Iterable[dict]) -> None:
    for result in results:
        print(json.dumps(result, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--include-hidden", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    iterations = max(1, args.iterations)

    results = [
        measure(
            "python_workspace_scan",
            iterations,
            lambda: scan_workspace(root, include_hidden=args.include_hidden),
        ),
        measure(
            "python_import_xencode_core",
            iterations,
            lambda: command_probe(
                [sys.executable, "-c", "import xencode_core; print('ok')"],
                root,
            ),
        ),
        measure(
            "python_cli_help",
            iterations,
            lambda: command_probe([sys.executable, "xencode_cli.py", "--help"], root),
        ),
    ]

    emit_json_lines(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
