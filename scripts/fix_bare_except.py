#!/usr/bin/env python3
"""
Fix bare except clauses across the codebase

Replaces bare 'except:' with 'except Exception:' and adds logging where appropriate.
"""

import re
from pathlib import Path
from typing import Tuple

# Files to skip (test files, generated code, etc.)
SKIP_DIRS = {
    'tests', '__pycache__', '.venv', 'venv', 'node_modules',
    'build', 'dist', '.git', '.pytest_cache', '.ruff_cache'
}

SKIP_FILES = {
    'conftest.py',  # Test configuration
}

def should_skip(path: Path) -> bool:
    """Check if path should be skipped"""
    # Check directory
    for part in path.parts:
        if part in SKIP_DIRS:
            return True

    # Check filename
    if path.name in SKIP_FILES:
        return True

    return False


def fix_bare_except(content: str) -> Tuple[str, int]:
    """
    Fix bare except clauses in content

    Returns:
        Tuple of (fixed_content, count_of_fixes)
    """
    fixes = 0

    # Pattern 1: Simple bare except with pass
    # except:\n                pass
    pattern1 = r'(\s+)except:\s*\n(\s+)pass\s*(?=#|$|\n)'

    def replace_pass_with_logged_exception(match):
        nonlocal fixes
        fixes += 1
        indent = match.group(1)
        pass_indent = match.group(2)
        return f"{indent}except Exception:\n{pass_indent}    pass  # Silently ignore\n"

    content = re.sub(pattern1, replace_pass_with_logged_exception, content)

    # Pattern 2: Bare except with other content (not pass)
    # except:\n                something_else
    pattern2 = r'(\s+)except:\s*\n(?!\s+pass\s*$)'

    def replace_bare_except(match):
        nonlocal fixes
        fixes += 1
        indent = match.group(1)
        return f"{indent}except Exception:\n"

    content = re.sub(pattern2, replace_bare_except, content)

    return content, fixes


def fix_file(file_path: Path) -> int:
    """
    Fix bare except clauses in a file

    Returns:
        Number of fixes applied
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0

    fixed_content, fixes = fix_bare_except(content)

    if fixes > 0:
        file_path.write_text(fixed_content, encoding='utf-8')
        print(f"  Fixed {fixes} bare except(s) in {file_path}")

    return fixes


def main():
    """Main entry point"""
    root = Path(r'd:\xencode\xencode')

    if not root.exists():
        print(f"Error: {root} does not exist")
        return

    print("Fixing bare except clauses...")
    print("=" * 60)

    total_fixes = 0
    files_fixed = 0

    # Find all Python files
    py_files = [f for f in root.rglob('*.py') if f.is_file() and not should_skip(f)]

    for py_file in py_files:
        fixes = fix_file(py_file)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1

    print("=" * 60)
    print(f"Fixed {total_fixes} bare except clause(s) in {files_fixed} file(s)")


if __name__ == '__main__':
    main()
