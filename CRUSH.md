# CRUSH.md

## Commands
- Build: `pip install -r requirements.txt`
- Lint: `ruff .`
- Typecheck: `mypy xencode_core.py`
- Test (all): `./test.sh`
- Test (single): `pytest xencode_core.py -k test_name`

## Code Style
- **Imports**: Standard library > third-party > local (isort order)
- **Formatting**: Black (88 chars), Ruff linting (E, F, I, W rules)
- **Types**: MyPy strict mode, all functions require type hints
- **Naming**: snake_case (vars), PascalCase (classes), _prefix (private)
- **Error Handling**: No bare except, context managers for resources
- **Testing**: Tests mirror structure, pytest fixtures, >90% coverage
- **Docs**: Google-style docstrings for public interfaces

## Conventions
- Pre-commit hooks run linter
- Tests must pass before commit
- No print() statements in main code
- Use logging instead of print()

## Special Notes
- test_claude_style.sh validates prompt handling
- xencode.sh is entrypoint, do not modify directly

## Codebase Context
- xencode_core.py: Core logic (file ops, model handling)
- Main entrypoints:
  • chat_mode(): Interactive CLI
  • file operations: create/read/write/delete
  • streaming Query API (Claude-style)
- State tracking: online_status, active_model