#!/usr/bin/env python3
"""Temporary audit script — will be deleted after use."""
import ast

# Fix encoding for Windows console
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Check 1: Bare except clauses
print("=" * 70)
print("1. BARE EXCEPT CLAUSES")
print("=" * 70)
risks = []
for root, dirs, fns in os.walk('xencode'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    for f in fns:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, encoding='utf-8', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    if re.search(r'except\s*:', line) and 'noqa' not in line:
                        risks.append((path, i + 1, line.strip()))
        except Exception:
            pass
print(f"Total bare except clauses: {len(risks)}")
for path, line_no, text in risks[:20]:
    print(f"  {os.path.relpath(path)}:{line_no}: {text}")
if len(risks) > 20:
    print(f"  ... and {len(risks) - 20} more")

# Check 2: TODOs / FIXMEs / HACKs
print()
print("=" * 70)
print("2. TODOs / FIXMEs / HACKs / XXXs")
print("=" * 70)
todos = []
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'htmlcov', 'dist', 'build', '.ruff_cache'}
for root, dirs, fns in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in fns:
        ext = os.path.splitext(f)[1].lower()
        if ext not in ('.py', '.rs', '.ts', '.js', '.md', '.toml', '.yaml', '.yml', '.json'):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, encoding='utf-8', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    stripped = line.strip()
                    if any(k in stripped.upper() for k in ['TODO', 'FIXME', 'HACK', 'XXX']):
                        if 'noqa' not in stripped and stripped != '...':
                            todos.append((path, i + 1, stripped[:120]))
        except Exception:
            pass
print(f"Total TODOs/FIXMEs/HACKs/XXXs: {len(todos)}")
for path, line_no, text in todos[:40]:
    try:
        print(f"  {os.path.relpath(path)}:{line_no}: {text}")
    except Exception:
        print(f"  {os.path.relpath(path)}:{line_no}: [non-printable chars]")
if len(todos) > 40:
    print(f"  ... and {len(todos) - 40} more")

# Check 3: Syntax errors
print()
print("=" * 70)
print("3. SYNTAX ERRORS")
print("=" * 70)
errors = []
py_count = 0
for root, dirs, fns in os.walk('xencode'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'venv')]
    for f in fns:
        if not f.endswith('.py'):
            continue
        py_count += 1
        path = os.path.join(root, f)
        try:
            with open(path, 'rb') as fh:
                content = fh.read()
            ast.parse(content)
        except SyntaxError as e:
            errors.append((path, str(e)))
        except Exception:
            pass
print(f"Files checked: {py_count}")
print(f"Syntax errors: {len(errors)}")
for path, err in errors:
    print(f"  {os.path.relpath(path)}: {err}")

# Check 4: File counts and LOC
print()
print("=" * 70)
print("4. PROJECT SIZE")
print("=" * 70)
py_loc = 0
rs_loc = 0
js_loc = 0
for root, dirs, fns in os.walk('.'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
    for f in fns:
        ext = os.path.splitext(f)[1].lower()
        path = os.path.join(root, f)
        try:
            with open(path, 'rb') as fh:
                lines = fh.readlines()
            if ext == '.py':
                py_loc += len(lines)
            elif ext == '.rs':
                rs_loc += len(lines)
            elif ext in ('.js', '.ts', '.tsx', '.jsx'):
                js_loc += len(lines)
        except Exception:
            pass
print(f"Python (.py):  {py_loc} lines")
print(f"Rust (.rs):    {rs_loc} lines")
print(f"JS/TS:         {js_loc} lines")

# Check 5: Danger patterns - eval, exec, subprocess shell
print()
print("=" * 70)
print("5. DANGEROUS PATTERNS")
print("=" * 70)
danger = []
for root, dirs, fns in os.walk('xencode'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    for f in fns:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, encoding='utf-8', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    if re.search(r'\beval\s*\(', stripped) and 'noqa' not in stripped:
                        danger.append((path, i+1, stripped[:100], 'eval()'))
                    if re.search(r'\bexec\s*\(', stripped) and 'noqa' not in stripped:
                        danger.append((path, i+1, stripped[:100], 'exec()'))
                    if re.search(r'shell\s*=\s*True', stripped) and 'noqa' not in stripped:
                        danger.append((path, i+1, stripped[:100], 'shell=True'))
        except Exception:
            pass

if not danger:
    print("No dangerous patterns (eval/exec/shell=True) found.")
else:
    print(f"Found {len(danger)} dangerous patterns:")
    for path, line_no, text, pattern in danger:
        print(f"  [{pattern}] {os.path.relpath(path)}:{line_no}: {text}")

# Check 6: Missing __init__.py in packages
print()
print("=" * 70)
print("6. PACKAGE INIT FILES")
print("=" * 70)
for root, dirs, fns in os.walk('xencode'):
    if root == 'xencode':
        continue
    has_init = '__init__.py' in fns
    has_py = any(f.endswith('.py') for f in fns)
    parent_init = os.path.exists(os.path.join(root, '..', '__init__.py'))
    if has_py and not has_init:
        print(f"  WARNING: No __init__.py: {os.path.relpath(root)}")
    elif not has_py and has_init:
        print(f"  WARNING: Empty __init__.py: {os.path.relpath(root)}")

print("\nDone!")
