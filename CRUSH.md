# Development Guide

## Build Commands
- **Dependencies**: `pip install -r requirements.txt`
- **Lint**: `ruff check .`
- **Format**: `black .`
- **Type Check**: `mypy xencode_core.py`
- **Test All**: `./scripts/test.sh`
- **Test Unit**: `pytest tests/ -k test_name`

## Code Standards
- **Import Order**: Standard library → third-party → local (isort)
- **Formatting**: Black (88 char limit), Ruff linting
- **Types**: MyPy strict mode, type hints required
- **Naming**: snake_case (variables), PascalCase (classes), _prefix (private)
- **Error Handling**: Specific exceptions, context managers for resources
- **Testing**: pytest with >90% coverage
- **Documentation**: Google-style docstrings

## Project Architecture
- **xencode_core.py**: Core AI interaction logic
- **xencode.sh**: Shell wrapper and entry point
- **install.sh**: Multi-platform installation
- **Enhanced modules**: Optional feature extensions

## Development Workflow
1. Run tests before changes: `./scripts/test.sh`
2. Make changes with proper type hints
3. Run linting: `ruff check . && black .`
4. Verify tests pass: `pytest tests/`
5. Test integration: `./scripts/test_claude_style.sh`

## Special Notes
- No direct print() in core modules (use Rich console)
- Maintain backward compatibility
- Test on multiple distributions
- Update documentation for user-facing changes