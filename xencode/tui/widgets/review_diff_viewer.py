#!/usr/bin/env python3
"""
Enhanced Diff Viewer for Code Review

Displays git diffs with inline AI suggestions and severity-based highlighting.
"""

from typing import Dict, List, Optional, Any
from textual.app import ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer, Horizontal
from textual.widgets import Static, Label, Button
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
from rich.syntax import Syntax
from rich.panel import Panel


class DiffLineWithSuggestion(Static):
    """A diff line with optional inline suggestion"""
    
    DEFAULT_CSS = """
    DiffLineWithSuggestion {
        height: auto;
        padding: 0 1;
    }
    
    DiffLineWithSuggestion.addition {
        background: #1a3d1a;
        color: #55ff55;
    }
    
    DiffLineWithSuggestion.deletion {
        background: #3d1a1a;
        color: #ff5555;
    }
    
    DiffLineWithSuggestion.hunk-header {
        background: #1a1a3d;
        color: #5555ff;
        text-style: bold;
    }
    
    DiffLineWithSuggestion.context {
        color: #888888;
    }
    
    DiffLineWithSuggestion.has-suggestion {
        border-left: thick $warning;
    }
    """
    
    def __init__(self, line: str, line_number: int = 0, 
                 suggestion: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.line = line
        self.line_number = line_number
        self.suggestion = suggestion
        self._apply_style()
    
    def _apply_style(self):
        """Apply CSS class based on line type"""
        if self.line.startswith('+') and not self.line.startswith('+++'):
            self.add_class("addition")
        elif self.line.startswith('-') and not self.line.startswith('---'):
            self.add_class("deletion")
        elif self.line.startswith('@@'):
            self.add_class("hunk-header")
        else:
            self.add_class("context")
        
        if self.suggestion:
            self.add_class("has-suggestion")
    
    def render(self) -> Text:
        """Render the line with line number and optional suggestion"""
        ln = f"{self.line_number:4d} │ " if self.line_number > 0 else "     │ "
        text = Text(f"{ln}{self.line}")
        
        # Add suggestion indicator
        if self.suggestion:
            text.append("\n     │ ")
            text.append("💡 ", style="yellow")
            text.append(self.suggestion.get('title', 'Suggestion'), style="yellow bold")
        
        return text


class InlineSuggestionPanel(Static):
    """Expandable panel showing detailed suggestion"""
    
    DEFAULT_CSS = """
    InlineSuggestionPanel {
        height: auto;
        padding: 1 2;
        margin: 0 2;
        background: $warning-darken-2;
        border-left: thick $warning;
    }
    
    InlineSuggestionPanel.critical {
        background: $error-darken-2;
        border-left: thick $error;
    }
    
    InlineSuggestionPanel.high {
        background: $warning-darken-2;
        border-left: thick $warning;
    }
    
    InlineSuggestionPanel.medium {
        background: $accent-darken-2;
        border-left: thick $accent;
    }
    
    InlineSuggestionPanel.low {
        background: $success-darken-2;
        border-left: thick $success;
    }
    """
    
    def __init__(self, suggestion: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.suggestion = suggestion
        severity = suggestion.get('severity', 'medium')
        self.add_class(severity)
    
    def render(self) -> Panel:
        """Render the suggestion panel"""
        title = self.suggestion.get('title', 'Suggestion')
        description = self.suggestion.get('description', '')
        example = self.suggestion.get('example', '')
        severity = self.suggestion.get('severity', 'medium')
        
        # Severity emoji
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        emoji = severity_emoji.get(severity, '⚪')
        
        content = Text()
        content.append(f"{emoji} {title}\n\n", style="bold")
        content.append(f"{description}\n", style="")
        
        if example:
            content.append("\nExample:\n", style="bold")
            content.append(example, style="dim")
        
        return Panel(content, border_style=severity)


class ReviewDiffViewer(ScrollableContainer):
    """Enhanced diff viewer with inline AI suggestions"""
    
    DEFAULT_CSS = """
    ReviewDiffViewer {
        height: 100%;
        border: solid $primary;
        background: $surface-darken-1;
    }
    
    ReviewDiffViewer > .diff-header {
        background: $primary-darken-2;
        padding: 0 1;
        text-style: bold;
    }
    
    ReviewDiffViewer > .file-stats {
        background: $panel;
        padding: 0 1;
        height: 1;
    }
    """
    
    diff_content = reactive("", init=False)
    
    def __init__(self, diff: str = "", suggestions: List[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.suggestions = suggestions or []
        self.suggestions_by_line: Dict[str, List[Dict[str, Any]]] = {}
        self._index_suggestions()
        # Set diff_content after initialization to avoid watcher issues
        self._diff_content = diff
    
    def _index_suggestions(self) -> None:
        """Index suggestions by file and line number"""
        self.suggestions_by_line.clear()
        
        for suggestion in self.suggestions:
            file_path = suggestion.get('file', '')
            line = suggestion.get('line', 0)
            
            if file_path and line:
                key = f"{file_path}:{line}"
                if key not in self.suggestions_by_line:
                    self.suggestions_by_line[key] = []
                self.suggestions_by_line[key].append(suggestion)
    
    def compose(self) -> ComposeResult:
        """Compose the diff viewer with suggestions"""
        diff_to_display = getattr(self, '_diff_content', self.diff_content)
        
        if not diff_to_display:
            yield Label("No diff to display", classes="dim")
            return
        
        current_file = ""
        line_num = 0
        
        for line in diff_to_display.splitlines():
            # Track file headers
            if line.startswith('diff --git'):
                # Extract filename
                parts = line.split(' b/')
                if len(parts) > 1:
                    current_file = parts[1]
                    
                    # File header with stats
                    yield Label(f"📄 {current_file}", classes="diff-header")
                    
                    # Show file-level suggestions if any
                    file_suggestions = [s for s in self.suggestions if s.get('file') == current_file and not s.get('line')]
                    for suggestion in file_suggestions:
                        yield InlineSuggestionPanel(suggestion)
                
                line_num = 0
            
            elif line.startswith('@@'):
                # Parse hunk header for line numbers
                try:
                    parts = line.split(' ')
                    new_range = parts[2]  # +start,count
                    line_num = int(new_range.split(',')[0].lstrip('+'))
                except (IndexError, ValueError):
                    pass
                
                yield DiffLineWithSuggestion(line, 0)
            
            elif line.startswith('+') and not line.startswith('+++'):
                # Addition line - check for suggestions
                key = f"{current_file}:{line_num}"
                suggestion = self.suggestions_by_line.get(key, [None])[0]
                
                yield DiffLineWithSuggestion(line, line_num, suggestion)
                
                # Show detailed suggestion if present
                if suggestion:
                    yield InlineSuggestionPanel(suggestion)
                
                line_num += 1
            
            elif line.startswith('-') and not line.startswith('---'):
                # Deletion line
                yield DiffLineWithSuggestion(line, 0)
            
            elif not line.startswith('---') and not line.startswith('+++'):
                # Context line
                yield DiffLineWithSuggestion(line, line_num)
                line_num += 1
    
    def watch_diff_content(self, new_diff: str) -> None:
        """React to diff content changes"""
        self.remove_children()
        self.call_later(self._recompose)
    
    def _recompose(self) -> None:
        """Recompose the widget with new diff"""
        for widget in self.compose():
            self.mount(widget)
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Update the suggestions and recompose"""
        self.suggestions = suggestions
        self._index_suggestions()
        self.remove_children()
        self.call_later(self._recompose)


class ReviewDiffPanel(Container):
    """Complete diff panel with controls and suggestions"""
    
    DEFAULT_CSS = """
    ReviewDiffPanel {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    ReviewDiffPanel > #diff-title {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
        text-style: bold;
    }
    
    ReviewDiffPanel > #diff-stats {
        dock: top;
        height: 3;
        background: $surface-darken-1;
        padding: 0 2;
    }
    
    ReviewDiffPanel > #diff-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }
    
    .control-button {
        width: 1fr;
        margin: 0 1;
    }
    """
    
    class ApplySuggestion(Message):
        """Message sent when user wants to apply a suggestion"""
        def __init__(self, suggestion: Dict[str, Any]):
            super().__init__()
            self.suggestion = suggestion
    
    def __init__(self, diff: str = "", suggestions: List[Dict[str, Any]] = None,
                 title: str = "Code Review Diff", **kwargs):
        super().__init__(**kwargs)
        self.diff = diff
        self.suggestions = suggestions or []
        self.title = title
        self._stats = self._calculate_stats(diff)
    
    def _calculate_stats(self, diff: str) -> Dict[str, int]:
        """Calculate diff statistics"""
        additions = sum(1 for line in diff.splitlines() 
                       if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff.splitlines() 
                       if line.startswith('-') and not line.startswith('---'))
        files = len([line for line in diff.splitlines() if line.startswith('diff --git')])
        
        # Count suggestions by severity
        critical = sum(1 for s in self.suggestions if s.get('severity') == 'critical')
        high = sum(1 for s in self.suggestions if s.get('severity') == 'high')
        medium = sum(1 for s in self.suggestions if s.get('severity') == 'medium')
        low = sum(1 for s in self.suggestions if s.get('severity') == 'low')
        
        return {
            "additions": additions,
            "deletions": deletions,
            "files": files,
            "suggestions": len(self.suggestions),
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low
        }
    
    def compose(self) -> ComposeResult:
        """Compose the diff panel"""
        yield Label(f"📝 {self.title}", id="diff-title")
        
        # Stats
        with Vertical(id="diff-stats"):
            stats_line1 = (f"+{self._stats['additions']} -{self._stats['deletions']} "
                          f"({self._stats['files']} files)")
            yield Label(stats_line1)
            
            stats_line2 = (f"💡 {self._stats['suggestions']} suggestions: "
                          f"🔴 {self._stats['critical']} "
                          f"🟠 {self._stats['high']} "
                          f"🟡 {self._stats['medium']} "
                          f"🟢 {self._stats['low']}")
            yield Label(stats_line2)
        
        # Controls
        with Horizontal(id="diff-controls"):
            yield Button("Show All", classes="control-button", id="show-all")
            yield Button("Critical Only", classes="control-button", id="show-critical")
            yield Button("Export", classes="control-button", id="export-diff")
        
        # Diff viewer
        yield ReviewDiffViewer(self.diff, self.suggestions)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "show-all":
            self._filter_suggestions(None)
        elif button_id == "show-critical":
            self._filter_suggestions("critical")
        elif button_id == "export-diff":
            self._export_diff()
    
    def _filter_suggestions(self, severity: Optional[str]) -> None:
        """Filter suggestions by severity"""
        viewer = self.query_one(ReviewDiffViewer)
        
        if severity:
            filtered = [s for s in self.suggestions if s.get('severity') == severity]
            viewer.update_suggestions(filtered)
            self.notify(f"Showing {len(filtered)} {severity} suggestions", severity="information")
        else:
            viewer.update_suggestions(self.suggestions)
            self.notify(f"Showing all {len(self.suggestions)} suggestions", severity="information")
    
    def _export_diff(self) -> None:
        """Export diff with suggestions to file"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"review_diff_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(f"Code Review Diff - {self.title}\n")
                f.write("=" * 80 + "\n\n")
                f.write(self.diff)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("AI Suggestions:\n\n")
                
                for i, suggestion in enumerate(self.suggestions, 1):
                    f.write(f"{i}. [{suggestion.get('severity', 'medium').upper()}] "
                           f"{suggestion.get('title', 'Suggestion')}\n")
                    f.write(f"   {suggestion.get('description', '')}\n")
                    f.write(f"   Location: {suggestion.get('file', 'N/A')}:"
                           f"{suggestion.get('line', 0)}\n\n")
            
            self.notify(f"Exported to {filename}", severity="information")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def update_diff(self, new_diff: str, suggestions: List[Dict[str, Any]] = None,
                   title: str = None) -> None:
        """Update the diff content and suggestions"""
        self.diff = new_diff
        self.suggestions = suggestions or []
        self._stats = self._calculate_stats(new_diff)
        
        if title:
            self.title = title
            self.query_one("#diff-title", Label).update(f"📝 {title}")
        
        # Update stats
        stats_container = self.query_one("#diff-stats", Vertical)
        stats_labels = list(stats_container.query(Label))
        
        if len(stats_labels) >= 2:
            stats_line1 = (f"+{self._stats['additions']} -{self._stats['deletions']} "
                          f"({self._stats['files']} files)")
            stats_labels[0].update(stats_line1)
            
            stats_line2 = (f"💡 {self._stats['suggestions']} suggestions: "
                          f"🔴 {self._stats['critical']} "
                          f"🟠 {self._stats['high']} "
                          f"🟡 {self._stats['medium']} "
                          f"🟢 {self._stats['low']}")
            stats_labels[1].update(stats_line2)
        
        # Update viewer
        viewer = self.query_one(ReviewDiffViewer)
        viewer.diff_content = new_diff
        viewer.update_suggestions(self.suggestions)
