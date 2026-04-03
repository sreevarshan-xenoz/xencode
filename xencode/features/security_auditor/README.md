# Security Auditor Feature

Proactive vulnerability scanning and security auditing for your codebase.

## Overview

The Security Auditor feature provides comprehensive security analysis including:

- **OWASP Top 10 vulnerability detection** - Identifies common security issues like injection, XSS, hardcoded secrets, and more
- **Dependency CVE scanning** - Analyzes dependencies for known security vulnerabilities
- **Security tool integration** - Integrates with Bandit (Python) and Snyk (dependencies)
- **Detailed reporting** - Generates markdown and HTML reports with fix suggestions
- **Risk classification** - Categorizes vulnerabilities by severity (Critical, High, Medium, Low)

## Features

### Vulnerability Scanner

Detects OWASP Top 10 vulnerabilities:

- **Injection** - SQL injection, command injection, code injection
- **Cross-Site Scripting (XSS)** - DOM-based XSS vulnerabilities
- **Hardcoded Secrets** - Passwords, API keys, tokens in code
- **Weak Cryptography** - MD5, SHA1, weak encryption algorithms
- **Path Traversal** - Directory traversal vulnerabilities
- **Insecure Deserialization** - Pickle, YAML unsafe loading
- **Broken Access Control** - Missing authentication decorators

### Dependency Analyzer

- Scans `requirements.txt`, `Pipfile`, `package.json`
- Checks against CVE database
- Identifies vulnerable package versions
- Suggests fixed versions

### Security Report Generator

- Comprehensive security reports
- Summary statistics by risk level
- Detailed vulnerability descriptions
- Fix suggestions with code examples
- Markdown and HTML output formats

### External Tool Integration

- **Bandit** - Python security linter
- **Snyk** - Dependency vulnerability scanner

## Installation

The Security Auditor is included with Xencode. Optional external tools:

```bash
# Install Bandit for Python security scanning
pip install bandit

# Install Snyk for dependency scanning
npm install -g snyk
```

## Usage

### CLI Commands

```bash
# Scan code for vulnerabilities
xencode security scan <path>

# Analyze dependencies
xencode security dependencies <path>

# Generate security report
xencode security report <path> --format markdown

# Run full security audit
xencode security audit <path>
```

### Python API

```python
import asyncio
from xencode.features.security_auditor import SecurityAuditor
from xencode.features.base import FeatureConfig

async def main():
    # Create configuration
    config = FeatureConfig(
        name="security_auditor",
        enabled=True,
        config={
            'checks': ['owasp_top_10', 'dependency_vulnerabilities'],
            'tools': ['bandit', 'snyk'],
            'exclude_patterns': ['*/tests/*', '*/.venv/*']
        }
    )
    
    # Initialize auditor
    auditor = SecurityAuditor(config)
    await auditor.initialize()
    
    # Scan for vulnerabilities
    result = await auditor.scan('path/to/code')
    print(f"Found {result['summary']['total_vulnerabilities']} vulnerabilities")
    
    # Generate report
    report = await auditor.report('path/to/code', format='markdown')
    print(report['content'])
    
    await auditor.shutdown()

asyncio.run(main())
```

## Configuration

```yaml
security_auditor:
  enabled: true
  checks:
    - owasp_top_10
    - dependency_vulnerabilities
    - code_patterns
  tools:
    - bandit
    - snyk
  exclude_patterns:
    - "*/tests/*"
    - "*/.venv/*"
    - "*/node_modules/*"
  max_severity: info  # Report all severities
```

## Vulnerability Types

### Critical

- SQL Injection
- Command Injection
- Hardcoded Secrets (passwords, API keys)
- Insecure Deserialization

### High

- Cross-Site Scripting (XSS)
- Path Traversal
- Broken Access Control
- Vulnerable Dependencies (CVSS > 7.0)

### Medium

- Weak Cryptography
- Security Misconfiguration
- Vulnerable Dependencies (CVSS 4.0-7.0)

### Low

- Information Disclosure
- Insufficient Logging
- Vulnerable Dependencies (CVSS < 4.0)

## Report Example

```markdown
# Security Audit Report

**Generated:** 2024-01-15T10:30:00

**Scan Path:** /path/to/project

## Summary

- **Total Vulnerabilities:** 5
- **Code Vulnerabilities:** 3
- **Dependency Vulnerabilities:** 2

### By Risk Level

- 🔴 **Critical:** 2
- 🟠 **High:** 1
- 🟡 **Medium:** 2
- 🟢 **Low:** 0

## Code Vulnerabilities

### Hardcoded password
- **Risk Level:** CRITICAL
- **Type:** hardcoded_secrets
- **Location:** app.py:15
- **Code:** `password = "admin123"`
- **Fix:** Store secrets in environment variables or a secure vault

### SQL injection via string concatenation
- **Risk Level:** CRITICAL
- **Type:** injection
- **Location:** database.py:42
- **Code:** `query = "SELECT * FROM users WHERE id = " + user_id`
- **Fix:** Use parameterized queries

## Recommendations

- ⚠️  URGENT: 2 critical vulnerabilities found
- 🔑 Found 1 hardcoded secret
- 📦 2 vulnerable dependencies found
- 📚 Review OWASP Top 10 guidelines
```

## Best Practices

1. **Run regularly** - Integrate into CI/CD pipeline
2. **Fix critical issues first** - Prioritize by risk level
3. **Keep dependencies updated** - Regularly update packages
4. **Use environment variables** - Never hardcode secrets
5. **Enable all checks** - Use both internal and external tools
6. **Review reports** - Don't just scan, act on findings

## Integration with CI/CD

### GitHub Actions

```yaml
name: Security Audit

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Xencode
        run: pip install xencode
      - name: Run Security Audit
        run: xencode security audit . --format json
```

### GitLab CI

```yaml
security_audit:
  script:
    - pip install xencode
    - xencode security audit . --format json
  artifacts:
    reports:
      security: security_report.json
```

## Limitations

- Pattern-based detection may have false positives
- CVE database is simplified (use Snyk for comprehensive scanning)
- External tools (Bandit, Snyk) must be installed separately
- Some vulnerabilities require manual review

## Contributing

To add new vulnerability patterns:

1. Edit `VulnerabilityScanner._init_patterns()`
2. Add pattern regex and description
3. Map to appropriate `VulnerabilityType`
4. Add fix suggestion in `_generate_fix_suggestion()`

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Snyk Documentation](https://docs.snyk.io/)
- [CVE Database](https://cve.mitre.org/)
- [CVSS Scoring](https://www.first.org/cvss/)

## License

Part of the Xencode project. See main LICENSE file.
