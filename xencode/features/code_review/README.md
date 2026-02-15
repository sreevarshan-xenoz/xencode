# AI Review Engine

The AI Review Engine is an intelligent code review system that uses ensemble reasoning to provide actionable suggestions for improving code quality and security.

## Features

### 1. Pattern Matching
The engine detects common code patterns that may indicate issues:
- **Complexity patterns**: Nested loops, deep conditionals
- **Naming patterns**: Single-letter variables, unclear names
- **Documentation patterns**: Missing docstrings

### 2. Semantic Analysis
Uses ensemble AI models to provide:
- Overall code quality assessment
- Architectural concerns
- Maintainability considerations
- Performance implications

### 3. Example Generation
Provides concrete code examples for fixing issues:
- Before/after comparisons
- Best practice implementations
- Security-focused solutions

### 4. Report Generation
Generates formatted review reports in multiple formats:
- **Text**: Plain text reports for terminal output
- **Markdown**: GitHub/GitLab-compatible markdown reports
- **HTML**: Rich, styled HTML reports for web viewing
- **JSON**: Structured data for programmatic access

## Usage

```python
from xencode.features.code_review import AIReviewEngine, CodeLinter, ReportGenerator

# Initialize the engine
engine = AIReviewEngine()
await engine.initialize()

# Analyze code
linter = CodeLinter()
code_analysis = await linter.analyze(files)

# Generate review
review = await engine.generate_review(
    title='PR Title',
    description='PR Description',
    files=files,
    code_analysis=code_analysis
)

# Generate formatted reports
generator = ReportGenerator()

# Text report
text_report = generator.generate_text_report(review, pr_data)
print(text_report)

# Markdown report
markdown_report = generator.generate_markdown_report(review, pr_data)

# HTML report
html_report = generator.generate_html_report(review, pr_data)

# JSON report
json_report = generator.generate_json_report(review, pr_data)
```

## Review Output

The engine generates a comprehensive review with:

```python
{
    'summary': {
        'title': 'PR title',
        'description': 'PR description',
        'files_analyzed': 3,
        'ai_summary': 'Overall assessment from AI'
    },
    'issues': [
        {
            'type': 'sqli',
            'severity': 'critical',
            'message': 'SQL injection detected',
            'file': 'auth.py',
            'line': 10
        }
    ],
    'suggestions': [
        {
            'title': 'SQL Injection Prevention',
            'description': 'Use parameterized queries',
            'example': '# Good\ncursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
            'severity': 'critical',
            'file': 'auth.py',
            'line': 10
        }
    ],
    'patterns_detected': [
        {
            'type': 'complexity',
            'pattern': 'nested_structure',
            'file': 'complex.py',
            'message': 'High complexity detected'
        }
    ],
    'semantic_analysis': {
        'analysis': 'AI-generated semantic insights',
        'confidence': 0.85,
        'consensus_score': 0.92
    },
    'positive_feedback': [
        {
            'title': 'Good Security Posture',
            'message': 'No critical security issues found',
            'score': 85
        }
    ]
}
```

## Report Formats

### Text Report
Plain text format suitable for terminal output:
- Clear section headers
- Severity-based grouping
- Easy to read in console

### Markdown Report
GitHub/GitLab-compatible markdown:
- Formatted headers and sections
- Code blocks for examples
- Severity badges with emojis
- Links and formatting

### HTML Report
Rich, styled HTML for web viewing:
- Color-coded severity levels
- Professional styling
- Responsive design
- Easy to share and archive

### JSON Report
Structured data format:
- Programmatic access
- Integration with other tools
- Complete metadata
- Quality score calculation

## Supported Issue Types

### Security Issues
- **SQL Injection (sqli)**: Detects unsafe database queries
- **XSS (xss)**: Identifies cross-site scripting vulnerabilities
- **CSRF (csrf)**: Finds missing CSRF protection
- **Hardcoded Secrets**: Detects passwords, API keys, tokens
- **Insecure Crypto**: Identifies weak cryptographic algorithms
- **Path Traversal**: Finds unsafe file path operations
- **Command Injection**: Detects unsafe command execution

### Code Quality Issues
- **Complexity**: Nested structures, deep conditionals
- **Naming**: Poor variable/function names
- **Documentation**: Missing docstrings
- **Language-specific**: Python bare except, TypeScript any, etc.

## Integration with Ensemble System

The AI Review Engine integrates with Xencode's ensemble reasoning system to provide:

1. **Multi-model consensus**: Uses multiple AI models for better accuracy
2. **Confidence scoring**: Provides confidence levels for suggestions
3. **Semantic understanding**: Goes beyond pattern matching to understand code intent
4. **Context-aware suggestions**: Considers the broader codebase context

## Configuration

The engine can be configured through the CodeReviewFeature:

```python
config = FeatureConfig(
    name="code_review",
    enabled=True,
    config={
        'supported_languages': ['python', 'javascript', 'typescript', 'go', 'rust'],
        'severity_levels': ['critical', 'high', 'medium', 'low'],
        'security_checks': ['owasp', 'sqli', 'xss', 'csrf']
    }
)
```

## Testing

Comprehensive test coverage includes:
- Unit tests for pattern detection
- Unit tests for suggestion generation
- Unit tests for report generation (all formats)
- Integration tests for complete review flow
- Tests for all security issue types

Run tests:
```bash
pytest tests/features/test_code_review.py::TestAIReviewEngine -v
pytest tests/features/test_ai_review_engine_integration.py -v
pytest tests/features/test_report_generator.py -v
```

## Performance

- Pattern matching: <100ms for typical files
- Semantic analysis: 1-3s depending on ensemble models
- Complete review: 2-5s for typical PRs

## Future Enhancements

- [ ] Custom pattern rules
- [ ] Language-specific semantic analysis
- [ ] Integration with external security tools
- [ ] Machine learning-based pattern detection
- [ ] Historical analysis for trend detection
