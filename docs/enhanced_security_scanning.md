# Enhanced Security Scanning with Bandit Integration

## Overview

The Enhanced Security Analyzer provides comprehensive vulnerability detection for multiple programming languages with integrated Bandit scanning for Python code. It combines pattern-based analysis, CVE database matching, OWASP Top 10 detection, and automated security reporting.

## Features

### 🛡️ Multi-Method Security Analysis

- **Bandit Integration**: Automatic Python security scanning with 50+ security rules
- **Vulnerability Database**: CVE-based pattern matching for known vulnerabilities
- **OWASP Top 10**: Detection of common web application security risks
- **Pattern-Based Analysis**: Custom security patterns for multiple languages
- **Dependency Analysis**: Detection of vulnerable package usage

### 📊 Comprehensive Reporting

- **Summary Reports**: Executive overview with risk assessment
- **Detailed Reports**: Technical analysis with remediation guidance
- **Executive Reports**: Business impact and investment recommendations
- **Security Metrics**: Compliance scoring and risk level determination

### 🔍 Vulnerability Coverage

#### Code Injection
- `eval()`, `exec()`, dynamic code execution
- Function constructors and script evaluation
- Template injection vulnerabilities

#### SQL Injection
- String concatenation in queries
- Unsafe parameter binding
- Dynamic query construction

#### Cross-Site Scripting (XSS)
- `innerHTML` assignments
- `document.write()` usage
- Unsafe DOM manipulation

#### Path Traversal
- Unsafe file path operations
- Directory traversal sequences
- User input in file paths

#### Cryptographic Issues
- Weak hash algorithms (MD5, SHA1)
- Insecure random number generation
- Deprecated cryptographic functions

#### Secrets Management
- Hardcoded passwords and API keys
- Embedded tokens and credentials
- Configuration secrets in code

#### Deserialization
- Unsafe pickle operations
- YAML loading vulnerabilities
- XML deserialization risks

#### Command Injection
- Shell command execution
- Process creation with user input
- System command vulnerabilities

## Installation

### Prerequisites

```bash
# Install Bandit for enhanced Python scanning (optional but recommended)
pip install bandit

# Install required dependencies
pip install -r requirements.txt
```

### Dependencies

The enhanced security analyzer requires:

- `bandit>=1.7.0` (optional, for Python scanning)
- `asyncio` (for async processing)
- `json` (for Bandit result parsing)
- `re` (for pattern matching)
- `subprocess` (for Bandit integration)

## Usage

### Basic Security Analysis

```python
from xencode.analyzers.security_analyzer import SecurityAnalyzer
from xencode.models.code_analysis import Language

# Initialize the analyzer
analyzer = SecurityAnalyzer()

# Analyze Python code
code = """
import os
password = "hardcoded123"
os.system(user_input)
eval(user_code)
"""

analysis_issues, security_issues = await analyzer.analyze_security(
    code, Language.PYTHON, "example.py"
)

print(f"Found {len(analysis_issues)} security issues")
```

### Comprehensive Security Scan

```python
# Run comprehensive scan with reporting
result = await analyzer.run_comprehensive_scan(
    code, 
    Language.PYTHON, 
    "vulnerable_app.py",
    generate_report=True
)

# Access results
print(f"Security Score: {result['metrics']['security_score']}/100")
print(f"Risk Level: {result['metrics']['risk_level']}")
print(f"Total Issues: {result['metrics']['total_issues']}")

# View detailed report
if result['report']:
    print(result['report'])
```

### Language Support

#### Python
- Full Bandit integration (50+ security rules)
- CVE pattern matching
- OWASP Top 10 detection
- Custom security patterns

#### JavaScript
- XSS vulnerability detection
- Prototype pollution risks
- Code injection patterns
- Weak cryptography usage

#### Java
- Deserialization vulnerabilities
- Command injection risks
- SQL injection patterns

## Configuration

### Bandit Integration

The analyzer automatically detects Bandit availability:

```python
# Check Bandit status
if analyzer.bandit_integration.bandit_available:
    print("Bandit integration active")
else:
    print("Bandit not available - install with: pip install bandit")
```

### Custom Security Patterns

Add custom security patterns for specific needs:

```python
# Extend security patterns
analyzer.security_patterns[Language.PYTHON]['custom_pattern'] = [
    (r'dangerous_function\s*\(', 'Custom dangerous function usage'),
    (r'unsafe_operation\s*\(', 'Unsafe operation detected')
]
```

## Security Reports

### Report Types

#### Summary Report
- Executive overview
- Risk assessment
- Top security concerns
- Actionable recommendations

#### Detailed Report
- Technical vulnerability details
- Remediation guidance
- Compliance analysis
- Security metrics

#### Executive Report
- Business impact assessment
- Investment recommendations
- Compliance status
- Risk metrics

### Generating Reports

```python
# Generate different report types
summary_report = await analyzer.generate_security_report(issues, 'summary')
detailed_report = await analyzer.generate_security_report(issues, 'detailed')
executive_report = await analyzer.generate_security_report(issues, 'executive')
```

## Security Metrics

### Risk Assessment

The analyzer calculates comprehensive security metrics:

- **Security Score**: 0-100 scale based on vulnerability severity
- **Risk Level**: CRITICAL, HIGH, MEDIUM, LOW
- **Compliance Percentage**: Adherence to security standards
- **Issue Categories**: Breakdown by vulnerability type

### Scoring Algorithm

```
Security Score = 100 - (Critical × 25 + High × 10 + Medium × 5)
Risk Level = CRITICAL if Critical > 0, HIGH if High > 3, etc.
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install bandit
          pip install -r requirements.txt
      - name: Run security scan
        run: python scripts/security_scan.py
```

### Pre-commit Hook

```python
#!/usr/bin/env python3
# .git/hooks/pre-commit
import asyncio
from xencode.analyzers.security_analyzer import SecurityAnalyzer

async def scan_staged_files():
    analyzer = SecurityAnalyzer()
    # Scan staged Python files
    # Fail commit if critical issues found
    pass

if __name__ == "__main__":
    asyncio.run(scan_staged_files())
```

## Best Practices

### Security Scanning Workflow

1. **Regular Scanning**: Run security scans on every commit
2. **Threshold Management**: Set acceptable risk thresholds
3. **Issue Tracking**: Track and remediate security issues
4. **Team Training**: Educate developers on secure coding

### Performance Optimization

- Use async processing for large codebases
- Cache analysis results for unchanged files
- Parallel scanning for multiple files
- Incremental scanning for large projects

### False Positive Management

- Review and validate security findings
- Customize patterns for specific contexts
- Use confidence scores for prioritization
- Maintain exception lists for known safe patterns

## Troubleshooting

### Common Issues

#### Bandit Not Found
```bash
# Install Bandit
pip install bandit

# Verify installation
bandit --version
```

#### Performance Issues
```python
# Use async processing
result = await analyzer.run_comprehensive_scan(
    code, language, file_path, generate_report=False
)

# Process files in batches
for batch in file_batches:
    results = await asyncio.gather(*[
        analyzer.analyze_security(code, lang, path) 
        for code, lang, path in batch
    ])
```

#### Memory Usage
- Process large files in chunks
- Use streaming for very large codebases
- Clear caches periodically
- Monitor memory usage during scans

## API Reference

### SecurityAnalyzer

Main class for security analysis with integrated scanning methods.

#### Methods

- `analyze_security(code, language, file_path)`: Core security analysis
- `run_comprehensive_scan(code, language, file_path, generate_report)`: Full scan with reporting
- `generate_security_report(issues, report_type)`: Generate security reports

### BanditIntegration

Handles Bandit security scanner integration for Python code.

#### Methods

- `scan_python_code(code, file_path)`: Scan Python code with Bandit
- `convert_bandit_to_analysis_issues(results, file_path)`: Convert Bandit results

### VulnerabilityDatabase

Database of known vulnerabilities and security patterns.

#### Methods

- `check_vulnerabilities(code, language)`: Check against vulnerability database

### SecurityReportGenerator

Generates comprehensive security reports in multiple formats.

#### Methods

- `generate_report(issues, report_type)`: Generate formatted security reports

## Contributing

### Adding New Security Patterns

1. Define patterns in appropriate language section
2. Add CWE mappings for new vulnerability types
3. Include mitigation advice
4. Add comprehensive tests

### Extending Language Support

1. Add language to security patterns dictionary
2. Implement language-specific vulnerability checks
3. Add appropriate test coverage
4. Update documentation

## License

This enhanced security scanning system is part of the Xencode project and follows the same licensing terms.