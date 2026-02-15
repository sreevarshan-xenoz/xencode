# Code Review TUI Components

Interactive Text User Interface components for AI-powered code review in Xencode.

## Overview

The Code Review TUI provides an interactive terminal interface for reviewing pull requests, analyzing code, and viewing AI-generated suggestions with severity-based color coding.

## Components

### 1. CodeReviewPanel

Main interface for code review with tabbed navigation.

**Features:**
- PR URL input with platform selection (GitHub/GitLab/Bitbucket)
- Tabbed interface for Issues, Suggestions, and Summary
- Real-time review analysis
- Severity-based color coding (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low)

**Usage:**
```python
from xencode.tui.widgets.code_review_panel import CodeReviewPanel

# Create panel
panel = CodeReviewPanel()

# Update with review data
panel.update_review(review_data, pr_data)

# Listen for events
@app.on(CodeReviewPanel.ReviewStarted)
async def handle_review(event):
    pr_url = event.pr_url
    platform = event.platform
    # Analyze PR...
```

**Keyboard Shortcuts:**
- `1` - Show Issues tab
- `2` - Show Suggestions tab
- `3` - Show Summary tab
- `r` - Refresh review

### 2. ReviewDiffViewer

Enhanced diff viewer with inline AI suggestions.

**Features:**
- Syntax-highlighted diff display
- Inline suggestion panels
- Line-by-line suggestions
- Severity-based highlighting
- File-level and line-level suggestions

**Usage:**
```python
from xencode.tui.widgets.review_diff_viewer import ReviewDiffViewer

# Create viewer with diff and suggestions
viewer = ReviewDiffViewer(
    diff=diff_content,
    suggestions=ai_suggestions
)

# Update suggestions
viewer.update_suggestions(new_suggestions)
```

### 3. ReviewDiffPanel

Complete diff panel with controls and statistics.

**Features:**
- Diff statistics (additions/deletions/files)
- Suggestion counts by severity
- Filter controls (show all, critical only)
- Export functionality
- Integrated diff viewer with suggestions

**Usage:**
```python
from xencode.tui.widgets.review_diff_viewer import ReviewDiffPanel

# Create panel
panel = ReviewDiffPanel(
    diff=diff_content,
    suggestions=suggestions,
    title="Security Review"
)

# Update diff
panel.update_diff(new_diff, new_suggestions, "Updated Review")
```

### 4. ReviewHistoryPanel

Panel showing review history with timestamps.

**Features:**
- Chronological review history
- PR titles and issue counts
- Timestamp tracking
- Clear history functionality

**Usage:**
```python
from xencode.tui.widgets.code_review_panel import ReviewHistoryPanel

# Create history panel
history = ReviewHistoryPanel()

# Add review to history
history.add_review(review_data, pr_data)

# Clear history
history.clear_history()
```

### 5. ReviewSummaryPanel

Summary panel with review statistics and quality score.

**Features:**
- Files analyzed count
- Issue counts by severity
- Quality score calculation
- AI assessment summary
- Rich table formatting

**Usage:**
```python
from xencode.tui.widgets.code_review_panel import ReviewSummaryPanel

# Create summary panel
summary = ReviewSummaryPanel(review_data)
```

## Data Structures

### Review Data Format

```python
review = {
    'summary': {
        'files_analyzed': 5,
        'ai_summary': 'Code quality is good with minor issues'
    },
    'issues': [
        {
            'type': 'sqli',
            'severity': 'critical',
            'message': 'Potential SQL injection detected',
            'file': 'app.py',
            'line': 42
        }
    ],
    'suggestions': [
        {
            'title': 'SQL Injection Prevention',
            'description': 'Use parameterized queries',
            'severity': 'critical',
            'file': 'app.py',
            'line': 42,
            'example': 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
        }
    ],
    'patterns_detected': [
        {
            'type': 'complexity',
            'pattern': 'nested_structure',
            'file': 'app.py',
            'message': 'High complexity detected',
            'severity': 'medium'
        }
    ],
    'semantic_analysis': {
        'analysis': 'The code follows good practices overall',
        'confidence': 0.85,
        'consensus_score': 0.92
    },
    'positive_feedback': [
        {
            'title': 'Good Security Posture',
            'message': 'No critical security issues',
            'score': 85
        }
    ]
}
```

### PR Data Format

```python
pr_data = {
    'url': 'https://github.com/user/repo/pull/123',
    'title': 'Add new feature',
    'description': 'This PR adds a new feature',
    'platform': 'github',
    'author': 'testuser',
    'base_branch': 'main',
    'head_branch': 'feature/new-feature',
    'files': [
        {
            'filename': 'app.py',
            'status': 'modified',
            'additions': 10,
            'deletions': 5,
            'patch': '...'
        }
    ]
}
```

## Severity Levels

The TUI uses color-coded severity levels:

- **🔴 Critical** - Security vulnerabilities, data loss risks
- **🟠 High** - Significant bugs, performance issues
- **🟡 Medium** - Code quality issues, maintainability concerns
- **🟢 Low** - Style issues, minor improvements

## Quality Score Calculation

Quality score is calculated as:
```
score = 100
score -= critical_issues * 20
score -= high_issues * 10
score -= medium_issues * 5
score -= low_issues * 2
score = max(0, score)
```

## Integration Example

Complete example integrating all components:

```python
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Container
from textual.widgets import Header, Footer

from xencode.tui.widgets.code_review_panel import CodeReviewPanel, ReviewHistoryPanel
from xencode.features.code_review import CodeReviewFeature
from xencode.features.base import FeatureConfig


class CodeReviewApp(App):
    """Code Review TUI Application"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize code review feature
        config = FeatureConfig(name="code_review", enabled=True)
        self.code_review = CodeReviewFeature(config)
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container():
            with Horizontal():
                # Main review panel
                self.review_panel = CodeReviewPanel()
                yield self.review_panel
                
                # History panel
                self.history_panel = ReviewHistoryPanel()
                yield self.history_panel
        
        yield Footer()
    
    async def on_mount(self) -> None:
        await self.code_review.initialize()
    
    async def on_code_review_panel_review_started(self, event):
        # Analyze PR
        result = await self.code_review.analyze_pr(
            event.pr_url, 
            event.platform
        )
        
        # Update UI
        self.review_panel.update_review(
            result['review'], 
            result['pr']
        )
        
        # Add to history
        self.history_panel.add_review(
            result['review'], 
            result['pr']
        )


if __name__ == "__main__":
    app = CodeReviewApp()
    app.run()
```

## Styling

The components use Textual's CSS system for styling. Key classes:

- `.critical` - Critical severity styling (red)
- `.high` - High severity styling (orange)
- `.medium` - Medium severity styling (blue)
- `.low` - Low severity styling (green)
- `.dim` - Dimmed text for secondary information
- `.bold` - Bold text for emphasis
- `.success` - Success/positive feedback styling

## Testing

Run tests with:
```bash
pytest tests/tui/test_code_review_panel.py -v
pytest tests/tui/test_review_diff_viewer.py -v
```

## Examples

See `examples/tui_code_review_example.py` for complete working examples:

```bash
# Run full code review example
python examples/tui_code_review_example.py

# Run diff viewer example
python examples/tui_code_review_example.py diff
```

## Performance

The TUI components are optimized for:
- Fast rendering with Textual's reactive system
- Efficient diff parsing and display
- Minimal memory usage for large diffs
- Smooth scrolling with ScrollableContainer

## Future Enhancements

Planned improvements:
- [ ] Interactive suggestion application
- [ ] Diff navigation shortcuts
- [ ] Search within diff
- [ ] Export to multiple formats (PDF, HTML)
- [ ] Collaborative review features
- [ ] Integration with git commands
- [ ] Custom severity thresholds
- [ ] Filtering by file type

## Contributing

When adding new TUI components:
1. Follow Textual best practices
2. Use reactive properties for dynamic updates
3. Implement proper message passing
4. Add comprehensive tests
5. Update this documentation

## License

Part of the Xencode project. See main LICENSE file.
