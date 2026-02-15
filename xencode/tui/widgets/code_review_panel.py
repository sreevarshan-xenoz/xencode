#!/usr/bin/env python3
"""
Code Review Panel Widget for Xencode TUI

Interactive code review interface with PR analysis, diff viewing, and AI suggestions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, Select, ListView, ListItem
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


class ReviewIssueItem(ListItem):
    """A single review issue in the list"""
    
    DEFAULT_CSS = """
    ReviewIssueItem {
        height: auto;
        padding: 1;
        margin: 0 1;
    }
    
    ReviewIssueItem.critical {
        background: $error-darken-2;
        border-left: thick $error;
    }
    
    ReviewIssueItem.high {
        background: $warning-darken-2;
        border-left: thick $warning;
    }
    
    ReviewIssueItem.medium {
        background: $accent-darken-2;
        border-left: thick $accent;
    }
    
    ReviewIssueItem.low {
        background: $success-darken-2;
        border-left: thick $success;
    }
    
    ReviewIssueItem:hover {
        background: $boost;
    }
    """
    
    def __init__(self, issue: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.issue = issue
        self.severity = issue.get('severity', 'medium')
        self.add_class(self.severity)
    
    def compose(self) -> ComposeResult:
        """Compose the issue item"""
        issue_type = self.issue.get('type', 'unknown').upper()
        message = self.issue.get('message', 'No message')
        file_path = self.issue.get('file', 'N/A')
        line = self.issue.get('line', 0)
        
        # Severity badge
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        emoji = severity_emoji.get(self.severity, '⚪')
        
        yield Label(f"{emoji} [{issue_type}] {message}")
        yield Label(f"   📄 {file_path}:{line}", classes="dim")


class SuggestionItem(ListItem):
    """A single fix suggestion in the list"""
    
    DEFAULT_CSS = """
    SuggestionItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $primary;
    }
    
    SuggestionItem:hover {
        background: $boost;
    }
    """
    
    def __init__(self, suggestion: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.suggestion = suggestion
    
    def compose(self) -> ComposeResult:
        """Compose the suggestion item"""
        title = self.suggestion.get('title', 'Suggestion')
        description = self.suggestion.get('description', '')
        file_path = self.suggestion.get('file', '')
        
        yield Label(f"💡 {title}", classes="bold")
        yield Label(f"   {description}")
        if file_path:
            yield Label(f"   📄 {file_path}", classes="dim")


class ReviewSummaryPanel(Static):
    """Summary panel showing review statistics"""
    
    DEFAULT_CSS = """
    ReviewSummaryPanel {
        height: auto;
        padding: 1 2;
        background: $panel;
        border: solid $primary;
        margin: 1;
    }
    """
    
    def __init__(self, review: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.review = review
    
    def render(self) -> Panel:
        """Render the summary panel"""
        summary = self.review.get('summary', {})
        issues = self.review.get('issues', [])
        
        # Count issues by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            severity = issue.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate quality score
        quality_score = 100
        quality_score -= severity_counts['critical'] * 20
        quality_score -= severity_counts['high'] * 10
        quality_score -= severity_counts['medium'] * 5
        quality_score -= severity_counts['low'] * 2
        quality_score = max(0, quality_score)
        
        # Create table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold")
        table.add_column("Value")
        
        table.add_row("Files Analyzed", str(summary.get('files_analyzed', 0)))
        table.add_row("Total Issues", str(len(issues)))
        table.add_row("🔴 Critical", str(severity_counts['critical']))
        table.add_row("🟠 High", str(severity_counts['high']))
        table.add_row("🟡 Medium", str(severity_counts['medium']))
        table.add_row("🟢 Low", str(severity_counts['low']))
        table.add_row("Quality Score", f"{quality_score}/100")
        
        if 'ai_summary' in summary:
            table.add_row("AI Assessment", summary['ai_summary'])
        
        return Panel(table, title="📊 Review Summary", border_style="green")


class CodeReviewPanel(Container):
    """Main code review panel with tabs for issues, suggestions, and diff"""
    
    DEFAULT_CSS = """
    CodeReviewPanel {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    CodeReviewPanel > #review-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    CodeReviewPanel > #review-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }
    
    CodeReviewPanel > #review-content {
        height: 1fr;
    }
    
    #pr-url-input {
        width: 60%;
    }
    
    #platform-select {
        width: 20%;
    }
    
    #analyze-button {
        width: 20%;
    }
    
    #tab-buttons {
        height: 3;
        background: $panel;
    }
    
    .tab-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button.active {
        background: $primary;
    }
    
    #issues-list {
        height: 1fr;
    }
    
    #suggestions-list {
        height: 1fr;
    }
    
    #diff-viewer {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("1", "show_issues", "Issues"),
        Binding("2", "show_suggestions", "Suggestions"),
        Binding("3", "show_diff", "Diff"),
        Binding("r", "refresh", "Refresh"),
    ]
    
    class ReviewStarted(Message):
        """Message sent when review is started"""
        def __init__(self, pr_url: str, platform: str):
            super().__init__()
            self.pr_url = pr_url
            self.platform = platform
    
    class IssueSelected(Message):
        """Message sent when an issue is selected"""
        def __init__(self, issue: Dict[str, Any]):
            super().__init__()
            self.issue = issue
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_review: Optional[Dict[str, Any]] = None
        self.current_pr_data: Optional[Dict[str, Any]] = None
        self.current_tab = "issues"
    
    def compose(self) -> ComposeResult:
        """Compose the code review panel"""
        # Header
        yield Label("🔍 AI Code Reviewer", id="review-header")
        
        # Controls
        with Horizontal(id="review-controls"):
            yield Input(placeholder="Enter PR URL (GitHub/GitLab/Bitbucket)", id="pr-url-input")
            yield Select(
                options=[
                    ("GitHub", "github"),
                    ("GitLab", "gitlab"),
                    ("Bitbucket", "bitbucket")
                ],
                value="github",
                id="platform-select"
            )
            yield Button("Analyze", variant="primary", id="analyze-button")
        
        # Content area
        with Vertical(id="review-content"):
            # Tab buttons
            with Horizontal(id="tab-buttons"):
                yield Button("Issues", classes="tab-button active", id="tab-issues")
                yield Button("Suggestions", classes="tab-button", id="tab-suggestions")
                yield Button("Summary", classes="tab-button", id="tab-summary")
            
            # Tab content (initially show placeholder)
            with ScrollableContainer(id="issues-list"):
                yield Label("Enter a PR URL and click Analyze to start", classes="dim")
            
            with ScrollableContainer(id="suggestions-list", classes="hidden"):
                yield Label("No suggestions yet", classes="dim")
            
            with ScrollableContainer(id="summary-view", classes="hidden"):
                yield Label("No review data yet", classes="dim")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "analyze-button":
            self._start_review()
        elif button_id == "tab-issues":
            self._show_tab("issues")
        elif button_id == "tab-suggestions":
            self._show_tab("suggestions")
        elif button_id == "tab-summary":
            self._show_tab("summary")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)"""
        if event.input.id == "pr-url-input":
            self._start_review()
    
    def _start_review(self) -> None:
        """Start the review process"""
        pr_url_input = self.query_one("#pr-url-input", Input)
        platform_select = self.query_one("#platform-select", Select)
        
        pr_url = pr_url_input.value.strip()
        platform = platform_select.value
        
        if not pr_url:
            self.notify("Please enter a PR URL", severity="warning")
            return
        
        # Post message to parent app
        self.post_message(self.ReviewStarted(pr_url, platform))
    
    def _show_tab(self, tab_name: str) -> None:
        """Show a specific tab"""
        self.current_tab = tab_name
        
        # Update button styles
        for button in self.query(".tab-button"):
            if button.id == f"tab-{tab_name}":
                button.add_class("active")
            else:
                button.remove_class("active")
        
        # Show/hide content
        issues_list = self.query_one("#issues-list")
        suggestions_list = self.query_one("#suggestions-list")
        summary_view = self.query_one("#summary-view")
        
        if tab_name == "issues":
            issues_list.remove_class("hidden")
            suggestions_list.add_class("hidden")
            summary_view.add_class("hidden")
        elif tab_name == "suggestions":
            issues_list.add_class("hidden")
            suggestions_list.remove_class("hidden")
            summary_view.add_class("hidden")
        elif tab_name == "summary":
            issues_list.add_class("hidden")
            suggestions_list.add_class("hidden")
            summary_view.remove_class("hidden")
    
    def update_review(self, review: Dict[str, Any], pr_data: Dict[str, Any] = None) -> None:
        """Update the panel with new review data"""
        self.current_review = review
        self.current_pr_data = pr_data
        
        # Update issues list
        issues_list = self.query_one("#issues-list", ScrollableContainer)
        issues_list.remove_children()
        
        issues = review.get('issues', [])
        if issues:
            for issue in issues:
                issues_list.mount(ReviewIssueItem(issue))
        else:
            issues_list.mount(Label("✅ No issues found!", classes="success"))
        
        # Update suggestions list
        suggestions_list = self.query_one("#suggestions-list", ScrollableContainer)
        suggestions_list.remove_children()
        
        suggestions = review.get('suggestions', [])
        if suggestions:
            for suggestion in suggestions:
                suggestions_list.mount(SuggestionItem(suggestion))
        else:
            suggestions_list.mount(Label("No suggestions available", classes="dim"))
        
        # Update summary view
        summary_view = self.query_one("#summary-view", ScrollableContainer)
        summary_view.remove_children()
        summary_view.mount(ReviewSummaryPanel(review))
        
        # Show positive feedback if any
        positive_feedback = review.get('positive_feedback', [])
        if positive_feedback:
            for feedback in positive_feedback:
                summary_view.mount(Label(f"✨ {feedback.get('title', 'Good work!')}", classes="success"))
                summary_view.mount(Label(f"   {feedback.get('message', '')}", classes="dim"))
        
        # Show patterns detected
        patterns = review.get('patterns_detected', [])
        if patterns:
            summary_view.mount(Label("\n🔍 Patterns Detected:", classes="bold"))
            for pattern in patterns:
                summary_view.mount(Label(f"  • {pattern.get('message', 'Pattern detected')}"))
        
        # Show semantic analysis
        semantic = review.get('semantic_analysis', {})
        if semantic and 'analysis' in semantic:
            summary_view.mount(Label("\n🧠 Semantic Analysis:", classes="bold"))
            summary_view.mount(Label(f"  {semantic['analysis']}"))
            if 'confidence' in semantic:
                summary_view.mount(Label(f"  Confidence: {semantic['confidence']:.2%}", classes="dim"))
        
        self.notify(f"Review complete: {len(issues)} issues found", severity="information")
    
    def action_show_issues(self) -> None:
        """Show issues tab"""
        self._show_tab("issues")
    
    def action_show_suggestions(self) -> None:
        """Show suggestions tab"""
        self._show_tab("suggestions")
    
    def action_show_diff(self) -> None:
        """Show diff tab"""
        self._show_tab("summary")
    
    def action_refresh(self) -> None:
        """Refresh the current review"""
        if self.current_review:
            self.update_review(self.current_review, self.current_pr_data)
        else:
            self.notify("No review to refresh", severity="warning")


class ReviewHistoryPanel(ScrollableContainer):
    """Panel showing review history"""
    
    DEFAULT_CSS = """
    ReviewHistoryPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the history panel"""
        yield Label("📜 Review History", classes="bold")
        
        if not self.history:
            yield Label("No reviews yet", classes="dim")
    
    def add_review(self, review: Dict[str, Any], pr_data: Dict[str, Any] = None) -> None:
        """Add a review to the history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history_item = {
            'timestamp': timestamp,
            'review': review,
            'pr_data': pr_data
        }
        
        self.history.append(history_item)
        
        # Add to UI
        pr_title = pr_data.get('title', 'Unknown PR') if pr_data else 'File Review'
        issues_count = len(review.get('issues', []))
        
        self.mount(Label(f"[{timestamp}] {pr_title} - {issues_count} issues"))
    
    def clear_history(self) -> None:
        """Clear the review history"""
        self.history.clear()
        self.remove_children()
        self.mount(Label("📜 Review History", classes="bold"))
        self.mount(Label("No reviews yet", classes="dim"))
