# Code Review CLI Commands - Usage Examples

This document demonstrates how to use the new code review CLI commands.

## Available Commands

The `xencode review` command group provides three subcommands:
- `pr` - Review a pull request
- `file` - Review a specific file
- `directory` - Review an entire directory

## Command Examples

### 1. Review a Pull Request

```bash
# Basic PR review
xencode review pr https://github.com/owner/repo/pull/123

# Review with specific platform
xencode review pr https://gitlab.com/owner/repo/-/merge_requests/45 --platform gitlab

# Filter by severity
xencode review pr https://github.com/owner/repo/pull/123 --severity high

# Output to file in markdown format
xencode review pr https://github.com/owner/repo/pull/123 --format markdown --output report.md

# JSON format for CI/CD integration
xencode review pr https://github.com/owner/repo/pull/123 --format json --output report.json
```

### 2. Review a Single File

```bash
# Basic file review
xencode review file src/main.py

# Specify language explicitly
xencode review file app.js --language javascript

# Filter by severity and save to file
xencode review file code.rs --severity critical --format markdown --output review.md

# HTML report
xencode review file index.html --format html --output report.html
```

### 3. Review a Directory

```bash
# Review entire directory
xencode review directory src/

# Review current directory
xencode review directory .

# Filter by language
xencode review directory . --language python

# Use file patterns
xencode review directory app/ --patterns "*.js" --patterns "*.ts"

# Comprehensive review with filters
xencode review directory . --severity high --format markdown --output full_review.md

# Multiple patterns with severity filter
xencode review directory src/ --patterns "*.py" --patterns "*.pyx" --severity medium
```

## Output Formats

All commands support multiple output formats:

- `text` (default) - Plain text output for terminal viewing
- `markdown` - Markdown format for documentation
- `json` - JSON format for programmatic processing
- `html` - HTML format for web viewing

## Severity Levels

Filter issues by severity:

- `critical` - Only show critical issues
- `high` - Show high and critical issues
- `medium` - Show medium, high, and critical issues
- `low` - Show all issues (default)

## Integration Examples

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Code Review
  run: |
    xencode review pr ${{ github.event.pull_request.html_url }} \
      --format json \
      --output review.json
    
    # Process results
    python scripts/process_review.py review.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Review staged files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|js|ts)$'); do
    xencode review file "$file" --severity high
done
```

### Automated Reports

```bash
#!/bin/bash
# Generate weekly code review report

xencode review directory . \
  --format html \
  --output "reports/review_$(date +%Y%m%d).html"
```

## Tips

1. **Use severity filters** to focus on important issues first
2. **Save reports** for documentation and tracking improvements
3. **Integrate with CI/CD** for automated code quality checks
4. **Use JSON format** for programmatic processing
5. **Combine with other tools** like git hooks for automated reviews

## Getting Help

```bash
# General help
xencode review --help

# Command-specific help
xencode review pr --help
xencode review file --help
xencode review directory --help
```
