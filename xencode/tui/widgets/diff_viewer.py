#!/usr/bin/env python3
"""
Interactive Diff Viewer Widget for Xencode TUI

Displays git diffs with syntax highlighting from Rich.
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, Label
from textual.reactive import reactive
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel


class DiffLine(Static):
    """A single line in the diff with appropriate styling"""
    
    DEFAULT_CSS = """
    DiffLine {
        height: auto;
        padding: 0 1;
    }
    
    DiffLine.addition {
        background: #1a3d1a;
        color: #55ff55;
    }
    
    DiffLine.deletion {
        background: #3d1a1a;
        color: #ff5555;
    }
    
    DiffLine.hunk-header {
        background: #1a1a3d;
        color: #5555ff;
        text-style: bold;
    }
    
    DiffLine.context {
        color: #888888;
    }
    """
    
    def __init__(self, line: str, line_number: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.line = line
        self.line_number = line_number
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
            
    def render(self) -> Text:
        """Render the line with line number"""
        ln = f"{self.line_number:4d} │ " if self.line_number > 0 else "     │ "
        return Text(f"{ln}{self.line}")


class DiffViewer(ScrollableContainer):
    """Interactive diff viewer with line-by-line display"""
    
    DEFAULT_CSS = """
    DiffViewer {
        height: 100%;
        border: solid $primary;
        background: $surface-darken-1;
    }
    
    DiffViewer > .diff-header {
        background: $primary-darken-2;
        padding: 0 1;
        text-style: bold;
    }
    """
    
    diff_content = reactive("")
    
    def __init__(self, diff: str = "", **kwargs):
        super().__init__(**kwargs)
        self.diff_content = diff
        
    def compose(self) -> ComposeResult:
        """Compose the diff viewer"""
        if not self.diff_content:
            yield Label("No diff to display", classes="dim")
            return
            
        current_file = ""
        line_num = 0
        
        for line in self.diff_content.splitlines():
            # Track file headers
            if line.startswith('diff --git'):
                # Extract filename
                parts = line.split(' b/')
                if len(parts) > 1:
                    current_file = parts[1]
                    yield Label(f"📄 {current_file}", classes="diff-header")
                line_num = 0
            elif line.startswith('@@'):
                # Parse hunk header for line numbers
                # Format: @@ -start,count +start,count @@
                try:
                    parts = line.split(' ')
                    new_range = parts[2]  # +start,count
                    line_num = int(new_range.split(',')[0].lstrip('+'))
                except (IndexError, ValueError):
                    pass
                yield DiffLine(line, 0)
            elif line.startswith('+') and not line.startswith('+++'):
                yield DiffLine(line, line_num)
                line_num += 1
            elif line.startswith('-') and not line.startswith('---'):
                yield DiffLine(line, 0)  # Deletions don't count toward line numbers
            elif not line.startswith('---') and not line.startswith('+++'):
                yield DiffLine(line, line_num)
                line_num += 1
                
    def watch_diff_content(self, new_diff: str) -> None:
        """React to diff content changes"""
        # Remove all children and recompose
        for child in list(self.children):
            child.remove()
        # Use call_later to avoid issues with reactive updates
        self.call_later(self._recompose)
        
    def _recompose(self) -> None:
        """Recompose the widget with new diff"""
        for widget in self.compose():
            self.mount(widget)


class DiffPanel(Container):
    """A complete diff panel with controls"""
    
    DEFAULT_CSS = """
    DiffPanel {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    DiffPanel > #diff-title {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
        text-style: bold;
    }
    
    DiffPanel > #diff-stats {
        dock: top;
        height: 1;
        background: $surface-darken-1;
        padding: 0 2;
    }
    """
    
    def __init__(self, diff: str = "", title: str = "Git Diff", **kwargs):
        super().__init__(**kwargs)
        self.diff = diff
        self.title = title
        self._stats = self._calculate_stats(diff)
        
    def _calculate_stats(self, diff: str) -> dict:
        """Calculate diff statistics"""
        additions = sum(1 for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff.splitlines() if line.startswith('-') and not line.startswith('---'))
        files = len([line for line in diff.splitlines() if line.startswith('diff --git')])
        return {"additions": additions, "deletions": deletions, "files": files}
        
    def compose(self) -> ComposeResult:
        yield Label(f"📝 {self.title}", id="diff-title")
        stats_text = f"+{self._stats['additions']} -{self._stats['deletions']} ({self._stats['files']} files)"
        yield Label(stats_text, id="diff-stats")
        yield DiffViewer(self.diff)
        
    def update_diff(self, new_diff: str, title: str = None) -> None:
        """Update the diff content"""
        self.diff = new_diff
        self._stats = self._calculate_stats(new_diff)
        
        if title:
            self.query_one("#diff-title", Label).update(f"📝 {title}")
            
        stats_text = f"+{self._stats['additions']} -{self._stats['deletions']} ({self._stats['files']} files)"
        self.query_one("#diff-stats", Label).update(stats_text)
        
        viewer = self.query_one(DiffViewer)
        viewer.diff_content = new_diff
