# Contributing to Xencode

Thank you for your interest in contributing to Xencode! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Reporting](#issue-reporting)
- [Security](#security)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xencode.git
   cd xencode
   ```

3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Setup

### Required Dependencies

- Python 3.8+
- Ollama (for local model inference)
- Git

### Optional Dependencies

- Node.js (for Bytebot UI development)
- Docker (for containerized testing)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=xencode --cov-report=html

# Run specific test category
pytest -m "unit"
pytest -m "integration"
```

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Python version
   - OS version
   - Steps to reproduce
   - Expected vs actual behavior
   - Code snippets if applicable

### Suggesting Features

1. Open a GitHub issue with the "enhancement" label
2. Describe the use case
3. Explain why this feature would be valuable
4. Provide examples if possible

### Submitting Code

1. **Create a branch** from `dev`:
   ```bash
   git checkout dev
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Run tests** to ensure everything passes:
   ```bash
   pytest
   ```

5. **Run linting**:
   ```bash
   ruff check xencode/
   mypy xencode/
   ```

6. **Commit your changes** with clear messages (see below)

7. **Push and create a PR**

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 88)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Type hints are required for all public APIs

### Code Organization

- Keep functions small and focused (< 50 lines ideal)
- Use descriptive variable and function names
- Add docstrings for public APIs
- Group related functionality into modules

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new ensemble reasoning method
fix: resolve memory leak in cache system
docs: update API documentation
test: add tests for security auditor
refactor: extract CLI commands into submodules
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Security Best Practices

- **Never commit secrets** (API keys, passwords, tokens)
- Use environment variables or the credential vault
- Run `bandit` security scanner:
  ```bash
  bandit -r xencode/
  ```
- Validate all user inputs
- Use parameterized queries for database operations

## Testing

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests

```python
import pytest
from xencode.module import MyClass

class TestMyClass:
    def test_basic_functionality(self):
        """Test basic operation"""
        obj = MyClass()
        result = obj.do_something()
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case handling"""
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.do_something_invalid()
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=xencode --cov-report=term-missing

# Specific test file
pytest tests/features/test_feature.py

# Specific test function
pytest tests/features/test_feature.py::TestClass::test_method

# Slow tests only
pytest -m slow
```

## Pull Request Guidelines

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Security scan passes (`bandit`)
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with `dev`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No security issues introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly
4. **Squash commits** if requested

## Issue Reporting

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information needed

### Issue Templates

Use the provided GitHub issue templates for:
- Bug reports
- Feature requests
- Security issues

## Security

### Reporting Security Issues

**Do not open public issues for security vulnerabilities.**

Email security concerns to: security@xenoz.com

### Security Best Practices for Contributors

1. Never commit credentials or secrets
2. Use the credential vault for sensitive data
3. Validate all inputs
4. Follow secure coding guidelines
5. Run security scans before submitting PRs

## Questions?

- Check existing [documentation](docs/)
- Search [closed issues](https://github.com/sreevarshan-xenoz/xencode/issues?q=is%3Aissue+is%3Aclosed)
- Join our [Discord](https://discord.com/invite/d9ewZkWPTP)

Thank you for contributing to Xencode! 🎉
