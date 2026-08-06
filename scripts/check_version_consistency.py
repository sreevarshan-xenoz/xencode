#!/usr/bin/env python3
"""Validate that version strings are consistent across release-critical files."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract_with_regex(path: Path, pattern: str, label: str) -> str:
    content = path.read_text(encoding="utf-8")
    match = re.search(pattern, content)
    if not match:
        raise RuntimeError(f"Could not find {label} in {path}")
    return match.group(1)


def main() -> int:
    setup_version = extract_with_regex(
        ROOT / "setup.py",
        r'version\s*=\s*"([^"]+)"',
        "setup.py version",
    )
    package_version = extract_with_regex(
        ROOT / "xencode" / "__init__.py",
        r'__version__\s*=\s*"([^"]+)"',
        "package __version__",
    )
    cli_version = extract_with_regex(
        ROOT / "xencode" / "cli.py",
        r'@click\.version_option\(version="([^"]+)"',
        "CLI version",
    )

    versions = {
        "setup.py": setup_version,
        "xencode/__init__.py": package_version,
        "xencode/cli.py": cli_version,
    }

    distinct_versions = set(versions.values())
    if len(distinct_versions) != 1:
        print("Version mismatch detected:")
        for file_name, version in versions.items():
            print(f"  - {file_name}: {version}")
        return 1

    unified_version = distinct_versions.pop()
    changelog_path = ROOT / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    if f"[{unified_version}]" not in changelog:
        print(
            f"Version {unified_version} is not present in CHANGELOG.md release headings."
        )
        return 1

    print(f"Version consistency check passed: {unified_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
