# Review Report Generation

## Overview

The Review Report Generation feature provides comprehensive, formatted reports for code reviews with proper severity categorization. Reports can be generated in multiple formats to suit different use cases.

## Implementation

### ReportGenerator Class

Located in `xencode/features/code_review.py`, the `ReportGenerator` class provides methods to generate reports in four different formats:

1. **Text Format** - Plain text reports for terminal output
2. **Markdown Format** - GitHub/GitLab-compatible markdown
3. **HTML Format** - Rich, styled HTML for web viewing
4. **JSON Format** - Structured data for programmatic access

### Key Features

#### Severity Categorization
- Issues and suggestions are grouped by severity level
- Severity levels: `critical`, `high`, `medium`, `low`
- Color-coded display (in HTML and terminal)
- Emoji indicators (in Markdown)

#### Quality Score Calculation
- Automatic quality score based on issue severity
- Formula: 100 - (critical×20 + high×10 + medium×5 + low×2)
- Minimum score: 0, Maximum score: 100

#### Report Sections
Each report includes:
- **PR Information**: Title, URL, author, branches
- **Summary**: Files analyzed, AI assessment
- **Issues**: Grouped by severity with details
- **Suggestions**: Actionable fixes with code examples
- **Patterns Detected**: Code patterns identified
- **Semantic Analysis**: AI-powered insights
- **Positive Feedback**: Recognition of good practices

## Usage

### Basic Usage

```python
from xencode.features.code_review import ReportGenerator

# Create generator
generator = ReportGenerator()

# Generate text report
text_report = generator.generate_text_report(review, pr_data)
print(text_report)

# Generate markdown report
markdown_report = generator.generate_markdown_report(review, pr_data)

# Generate HTML report
html_report = generator.generate_html_report(review, pr_data)

# Generate JSON report
json_report = generator.generate_json_report(review, pr_data)
```

### Using with CodeReviewFeature

```python
from xencode.features.code_review import CodeReviewFeature

# Initialize feature
feature = CodeReviewFeature(config)
await feature._initialize()

# Analyze PR
report = await feature.analyze_pr(pr_url, platform='github')

# Generate formatted report
text_report = feature.generate_formatted_report(
    report['review'], 
    report['pr'], 
    format='text'
)

# Or other formats
markdown_report = feature.generate_formatted_report(
    report['review'], 
    report['pr'], 
    format='markdown'
)

html_report = feature.generate_formatted_report(
    report['review'], 
    report['pr'], 
    format='html'
)

json_report = feature.generate_formatted_report(
    report['review'], 
    report['pr'], 
    format='json'
)
```

## Report Format Examples

### Text Report
```
================================================================================
CODE REVIEW REPORT
================================================================================

Pull Request Information:
  Title: Add authentication feature
  URL: https://github.com/owner/repo/pull/123
  Author: developer
  Branch: feature/auth → main

Summary:
  Files Analyzed: 3
  AI Assessment: Good implementation with minor security concerns

Issues Found:

  CRITICAL (1 issue(s)):
  ----------------------------------------------------------------------------
    [SQLI] SQL injection vulnerability detected
    Location: auth.py:42

...
```

### Markdown Report
```markdown
# Code Review Report

## Pull Request Information

- **Title:** Add authentication feature
- **URL:** https://github.com/owner/repo/pull/123
- **Author:** developer
- **Branch:** `feature/auth` → `main`

## Summary

- **Files Analyzed:** 3
- **AI Assessment:** Good implementation with minor security concerns

## Issues Found

### 🔴 CRITICAL (1 issue(s))

#### SQLI

**Message:** SQL injection vulnerability detected

**Location:** `auth.py:42`

...
```

### HTML Report
Rich, styled HTML with:
- Professional styling
- Color-coded severity levels
- Responsive design
- Code syntax highlighting
- Easy navigation

### JSON Report
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00Z",
    "report_version": "1.0"
  },
  "pr_info": {
    "title": "Add authentication feature",
    "url": "https://github.com/owner/repo/pull/123",
    "author": "developer"
  },
  "summary": {
    "files_analyzed": 3,
    "total_issues": 4,
    "severity_counts": {
      "critical": 1,
      "high": 1,
      "medium": 1,
      "low": 1
    },
    "quality_score": 63
  },
  "issues_by_severity": {
    "critical": [...],
    "high": [...],
    "medium": [...],
    "low": [...]
  },
  ...
}
```

## Testing

Comprehensive test suite in `tests/features/test_report_generator.py`:

- ✅ Text report generation
- ✅ Markdown report generation
- ✅ HTML report generation
- ✅ JSON report generation
- ✅ Report generation without PR data
- ✅ Report generation with empty sections
- ✅ HTML escaping
- ✅ Severity grouping
- ✅ Quality score calculation
- ✅ Integration with CodeReviewFeature

Run tests:
```bash
pytest tests/features/test_report_generator.py -v
```

## Examples

See `examples/report_generation_example.py` for a complete demonstration of all report formats.

Run the example:
```bash
python examples/report_generation_example.py
```

## Integration

The report generation functionality integrates seamlessly with:
- **PR Analyzers**: GitHub, GitLab, Bitbucket
- **Code Linter**: Security and quality checks
- **AI Review Engine**: Semantic analysis and suggestions
- **CLI Commands**: (to be implemented in task 2.1.8)
- **TUI Components**: (to be implemented in task 2.1.9)

## Future Enhancements

Potential improvements:
- PDF report generation
- Email-friendly report format
- Customizable report templates
- Report comparison (before/after)
- Historical trend analysis
- Integration with CI/CD pipelines
- Slack/Teams notification format

## Requirements Validation

This implementation satisfies:
- ✅ Requirement 1.5: Generate review reports with severity levels
- ✅ Multiple report formats (text, markdown, HTML, JSON)
- ✅ Severity categorization (critical, high, medium, low)
- ✅ Quality score calculation
- ✅ Comprehensive test coverage
- ✅ Documentation and examples
