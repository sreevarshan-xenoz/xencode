#!/usr/bin/env python3
"""
Terminal Assistant Panel Widget for Xencode TUI

Interactive terminal assistance interface with command suggestions, explanations,
error fixes, learning progress, and history browsing.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, ListView, ListItem, ProgressBar
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


class CommandSuggestionItem(ListItem):
    """A single command suggestion in the list"""
    
    DEFAULT_CSS = """
    CommandSuggestionItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $primary;
    }
    
    CommandSuggestionItem:hover {
        background: $boost;
    }
    
    CommandSuggestionItem.high-score {
        border-left: thick $success;
    }
    """
    
    def __init__(self, suggestion: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.suggestion = suggestion
        score = suggestion.get('score', 0)
        if score > 10:
            self.add_class('high-score')
    
    def compose(self) -> ComposeResult:
        """Compose the suggestion item"""
        command = self.suggestion.get('command', '')
        score = self.suggestion.get('score', 0)
        source = self.suggestion.get('source', 'unknown')
        explanation = self.suggestion.get('explanation', '')
        reason = self.suggestion.get('reason', '')
        
        # Score indicator
        score_emoji = '⭐' * min(5, int(score / 5))
        
        yield Label(f"{score_emoji} {command}", classes="bold")
        if explanation:
            yield Label(f"   {explanation}", classes="dim")
        if reason:
            yield Label(f"   💡 {reason}", classes="dim")
        yield Label(f"   Source: {source} | Score: {score:.1f}", classes="dim")


class CommandSuggestionPanel(Container):
    """Panel showing context-aware command suggestions"""
    
    DEFAULT_CSS = """
    CommandSuggestionPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    CommandSuggestionPanel > #suggestion-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    CommandSuggestionPanel > #suggestion-input {
        dock: top;
        height: 3;
        padding: 0 2;
    }
    
    CommandSuggestionPanel > #suggestion-list {
        height: 1fr;
    }
    
    #partial-input {
        width: 100%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("enter", "select", "Select"),
    ]
    
    class SuggestionSelected(Message):
        """Message sent when a suggestion is selected"""
        def __init__(self, command: str):
            super().__init__()
            self.command = command
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suggestions: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the suggestion panel"""
        yield Label("💡 Command Suggestions", id="suggestion-header")
        
        with Horizontal(id="suggestion-input"):
            yield Input(placeholder="Type partial command...", id="partial-input")
        
        with ScrollableContainer(id="suggestion-list"):
            yield Label("Type a command or press Ctrl+R to get suggestions", classes="dim")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        if event.input.id == "partial-input":
            # Request suggestions from parent
            self.post_message(self.SuggestionSelected(event.value))
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Update the panel with new suggestions"""
        self.suggestions = suggestions
        
        suggestion_list = self.query_one("#suggestion-list", ScrollableContainer)
        suggestion_list.remove_children()
        
        if suggestions:
            for suggestion in suggestions:
                suggestion_list.mount(CommandSuggestionItem(suggestion))
        else:
            suggestion_list.mount(Label("No suggestions available", classes="dim"))
    
    def action_refresh(self) -> None:
        """Refresh suggestions"""
        partial_input = self.query_one("#partial-input", Input)
        self.post_message(self.SuggestionSelected(partial_input.value))
    
    def action_select(self) -> None:
        """Select current suggestion"""
        # Get focused item
        suggestion_list = self.query_one("#suggestion-list", ScrollableContainer)
        items = list(suggestion_list.query(CommandSuggestionItem))
        if items:
            self.post_message(self.SuggestionSelected(items[0].suggestion['command']))


class CommandExplanationViewer(ScrollableContainer):
    """Viewer for detailed command explanations"""
    
    DEFAULT_CSS = """
    CommandExplanationViewer {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_explanation: Optional[Dict[str, Any]] = None
    
    def compose(self) -> ComposeResult:
        """Compose the explanation viewer"""
        yield Label("📖 Command Explanation", classes="bold")
        yield Label("Select a command to see its explanation", classes="dim")
    
    def show_explanation(self, explanation: Dict[str, Any]) -> None:
        """Show explanation for a command"""
        self.current_explanation = explanation
        self.remove_children()
        
        command = explanation.get('command', '')
        description = explanation.get('description', '')
        arguments = explanation.get('arguments', [])
        examples = explanation.get('examples', [])
        warnings = explanation.get('warnings', [])
        
        # Title
        self.mount(Label(f"📖 {command}", classes="bold"))
        self.mount(Label(""))
        
        # Description
        if description:
            self.mount(Label("Description:", classes="bold"))
            self.mount(Label(f"  {description}"))
            self.mount(Label(""))
        
        # Arguments
        if arguments:
            self.mount(Label("Arguments:", classes="bold"))
            for arg in arguments:
                value = arg.get('value', '')
                desc = arg.get('description', '')
                self.mount(Label(f"  {value}: {desc}"))
            self.mount(Label(""))
        
        # Examples
        if examples:
            self.mount(Label("Examples:", classes="bold"))
            for example in examples:
                self.mount(Label(f"  $ {example}", classes="dim"))
            self.mount(Label(""))
        
        # Warnings
        if warnings:
            self.mount(Label("Warnings:", classes="bold"))
            for warning in warnings:
                self.mount(Label(f"  {warning}", classes="warning"))


class ErrorFixItem(ListItem):
    """A single error fix suggestion"""
    
    DEFAULT_CSS = """
    ErrorFixItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $warning;
    }
    
    ErrorFixItem:hover {
        background: $boost;
    }
    
    ErrorFixItem.high-confidence {
        border-left: thick $success;
    }
    """
    
    def __init__(self, fix: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.fix = fix
        confidence = fix.get('confidence', 0)
        if confidence > 0.8:
            self.add_class('high-confidence')
    
    def compose(self) -> ComposeResult:
        """Compose the fix item"""
        fix_cmd = self.fix.get('fix', '')
        explanation = self.fix.get('explanation', '')
        confidence = self.fix.get('confidence', 0)
        category = self.fix.get('category', 'unknown')
        requires_sudo = self.fix.get('requires_sudo', False)
        requires_install = self.fix.get('requires_install', False)
        
        # Confidence indicator
        conf_emoji = '🟢' if confidence > 0.8 else '🟡' if confidence > 0.5 else '🔴'
        
        yield Label(f"{conf_emoji} {fix_cmd}", classes="bold")
        yield Label(f"   {explanation}")
        yield Label(f"   Category: {category} | Confidence: {confidence:.0%}", classes="dim")
        
        if requires_sudo:
            yield Label("   ⚠️  Requires sudo privileges", classes="warning")
        if requires_install:
            install_cmd = self.fix.get('install_command', '')
            yield Label(f"   📦 Install first: {install_cmd}", classes="dim")


class ErrorFixPanel(Container):
    """Panel showing error fix suggestions"""
    
    DEFAULT_CSS = """
    ErrorFixPanel {
        height: 100%;
        border: solid $warning;
        background: $surface;
    }
    
    ErrorFixPanel > #fix-header {
        dock: top;
        height: 3;
        background: $warning;
        padding: 0 2;
    }
    
    ErrorFixPanel > #fix-context {
        dock: top;
        height: auto;
        background: $error-darken-2;
        padding: 1 2;
    }
    
    ErrorFixPanel > #fix-list {
        height: 1fr;
    }
    """
    
    class FixSelected(Message):
        """Message sent when a fix is selected"""
        def __init__(self, fix_command: str):
            super().__init__()
            self.fix_command = fix_command
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_error: Optional[str] = None
        self.current_command: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        """Compose the fix panel"""
        yield Label("🔧 Error Fix Suggestions", id="fix-header")
        
        with Vertical(id="fix-context"):
            yield Label("No error to fix", classes="dim")
        
        with ScrollableContainer(id="fix-list"):
            yield Label("Run a command that fails to see fix suggestions", classes="dim")
    
    def show_fixes(self, command: str, error: str, fixes: List[Dict[str, Any]]) -> None:
        """Show fix suggestions for an error"""
        self.current_command = command
        self.current_error = error
        
        # Update context
        context_container = self.query_one("#fix-context", Vertical)
        context_container.remove_children()
        context_container.mount(Label(f"Command: {command}", classes="bold"))
        context_container.mount(Label(f"Error: {error[:100]}...", classes="error"))
        
        # Update fixes
        fix_list = self.query_one("#fix-list", ScrollableContainer)
        fix_list.remove_children()
        
        if fixes:
            for fix in fixes:
                fix_list.mount(ErrorFixItem(fix))
        else:
            fix_list.mount(Label("No automatic fixes available", classes="dim"))


class LearningProgressTracker(Container):
    """Tracker showing learning progress and statistics"""
    
    DEFAULT_CSS = """
    LearningProgressTracker {
        height: 100%;
        border: solid $success;
        background: $surface;
        padding: 1;
    }
    
    #progress-header {
        height: 3;
        background: $success;
        padding: 0 2;
    }
    
    #progress-stats {
        height: auto;
        padding: 1;
    }
    
    #progress-skills {
        height: 1fr;
    }
    
    .skill-item {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
    }
    
    .progress-bar {
        width: 100%;
        height: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_stats: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the progress tracker"""
        yield Label("📊 Learning Progress", id="progress-header")
        
        with Vertical(id="progress-stats"):
            yield Label("No learning data yet", classes="dim")
        
        with ScrollableContainer(id="progress-skills"):
            yield Label("Use commands to track your progress", classes="dim")
    
    def update_progress(self, stats: Dict[str, Any]) -> None:
        """Update learning progress display"""
        self.learning_stats = stats
        
        # Update stats
        stats_container = self.query_one("#progress-stats", Vertical)
        stats_container.remove_children()
        
        total_commands = stats.get('total_commands_learned', 0)
        total_executions = stats.get('total_executions', 0)
        mastered = len(stats.get('mastered_commands', []))
        
        stats_container.mount(Label(f"Commands Learned: {total_commands}"))
        stats_container.mount(Label(f"Total Executions: {total_executions}"))
        stats_container.mount(Label(f"Mastered Commands: {mastered}"))
        
        # Update skill levels
        skills_container = self.query_one("#progress-skills", ScrollableContainer)
        skills_container.remove_children()
        
        skill_levels = stats.get('skill_levels', {})
        learning_progress = stats.get('learning_progress', {})
        
        if skill_levels:
            skills_container.mount(Label("Skill Levels:", classes="bold"))
            for cmd, level in sorted(skill_levels.items(), key=lambda x: x[1], reverse=True):
                progress_info = learning_progress.get(cmd, {})
                mastery = progress_info.get('mastery_level', 0)
                total_uses = progress_info.get('total_uses', 0)
                
                skill_widget = Vertical(classes="skill-item")
                skill_widget.mount(Label(f"{cmd}: {level:.0%} skill | {mastery:.0%} mastery ({total_uses} uses)"))
                skills_container.mount(skill_widget)
        else:
            skills_container.mount(Label("No skill data yet", classes="dim"))


class CommandHistoryBrowser(Container):
    """Browser for command history with search"""
    
    DEFAULT_CSS = """
    CommandHistoryBrowser {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    CommandHistoryBrowser > #history-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    CommandHistoryBrowser > #history-search {
        dock: top;
        height: 3;
        padding: 0 2;
    }
    
    CommandHistoryBrowser > #history-list {
        height: 1fr;
    }
    
    #search-input {
        width: 100%;
    }
    
    .history-item {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
    }
    
    .history-item.success {
        border-left: thick $success;
    }
    
    .history-item.failure {
        border-left: thick $error;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+f", "focus_search", "Search"),
    ]
    
    class CommandSelected(Message):
        """Message sent when a command is selected from history"""
        def __init__(self, command: str):
            super().__init__()
            self.command = command
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: List[Dict[str, Any]] = []
        self.filtered_history: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the history browser"""
        yield Label("📜 Command History", id="history-header")
        
        with Horizontal(id="history-search"):
            yield Input(placeholder="Search history...", id="search-input")
        
        with ScrollableContainer(id="history-list"):
            yield Label("No command history yet", classes="dim")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "search-input":
            self._filter_history(event.value)
    
    def update_history(self, history: List[Dict[str, Any]]) -> None:
        """Update the history display"""
        self.history = history
        self.filtered_history = history
        self._render_history()
    
    def _filter_history(self, pattern: str) -> None:
        """Filter history by pattern"""
        if not pattern:
            self.filtered_history = self.history
        else:
            self.filtered_history = [
                item for item in self.history
                if pattern.lower() in item.get('command', '').lower()
            ]
        self._render_history()
    
    def _render_history(self) -> None:
        """Render the filtered history"""
        history_list = self.query_one("#history-list", ScrollableContainer)
        history_list.remove_children()
        
        if self.filtered_history:
            for item in reversed(self.filtered_history[-50:]):  # Show last 50
                command = item.get('command', '')
                timestamp = item.get('timestamp', '')
                success = item.get('success', True)
                context = item.get('context', {})
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp
                
                # Create history item
                history_widget = Vertical(classes=f"history-item {'success' if success else 'failure'}")
                
                status_emoji = '✅' if success else '❌'
                history_widget.mount(Label(f"{status_emoji} {command}", classes="bold"))
                history_widget.mount(Label(f"   {time_str}", classes="dim"))
                
                project_type = context.get('project_type')
                if project_type:
                    history_widget.mount(Label(f"   Project: {project_type}", classes="dim"))
                
                history_list.mount(history_widget)
        else:
            history_list.mount(Label("No matching commands found", classes="dim"))
    
    def action_focus_search(self) -> None:
        """Focus the search input"""
        search_input = self.query_one("#search-input", Input)
        search_input.focus()


class TerminalAssistantPanel(Container):
    """Main terminal assistant panel with tabs"""
    
    DEFAULT_CSS = """
    TerminalAssistantPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    TerminalAssistantPanel > #ta-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    TerminalAssistantPanel > #ta-tabs {
        dock: top;
        height: 3;
        background: $panel;
    }
    
    TerminalAssistantPanel > #ta-content {
        height: 1fr;
    }
    
    .tab-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button.active {
        background: $primary;
    }
    """
    
    BINDINGS = [
        Binding("1", "show_suggestions", "Suggestions"),
        Binding("2", "show_explanation", "Explanation"),
        Binding("3", "show_fixes", "Fixes"),
        Binding("4", "show_progress", "Progress"),
        Binding("5", "show_history", "History"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_tab = "suggestions"
    
    def compose(self) -> ComposeResult:
        """Compose the terminal assistant panel"""
        yield Label("🖥️  Terminal Assistant", id="ta-header")
        
        with Horizontal(id="ta-tabs"):
            yield Button("Suggestions", classes="tab-button active", id="tab-suggestions")
            yield Button("Explanation", classes="tab-button", id="tab-explanation")
            yield Button("Fixes", classes="tab-button", id="tab-fixes")
            yield Button("Progress", classes="tab-button", id="tab-progress")
            yield Button("History", classes="tab-button", id="tab-history")
        
        with Container(id="ta-content"):
            yield CommandSuggestionPanel(id="suggestions-panel")
            yield CommandExplanationViewer(id="explanation-panel", classes="hidden")
            yield ErrorFixPanel(id="fixes-panel", classes="hidden")
            yield LearningProgressTracker(id="progress-panel", classes="hidden")
            yield CommandHistoryBrowser(id="history-panel", classes="hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tab button presses"""
        button_id = event.button.id
        
        if button_id == "tab-suggestions":
            self._show_tab("suggestions")
        elif button_id == "tab-explanation":
            self._show_tab("explanation")
        elif button_id == "tab-fixes":
            self._show_tab("fixes")
        elif button_id == "tab-progress":
            self._show_tab("progress")
        elif button_id == "tab-history":
            self._show_tab("history")
    
    def _show_tab(self, tab_name: str) -> None:
        """Show a specific tab"""
        self.current_tab = tab_name
        
        # Update button styles
        for button in self.query(".tab-button"):
            if button.id == f"tab-{tab_name}":
                button.add_class("active")
            else:
                button.remove_class("active")
        
        # Show/hide panels
        panels = {
            "suggestions": "suggestions-panel",
            "explanation": "explanation-panel",
            "fixes": "fixes-panel",
            "progress": "progress-panel",
            "history": "history-panel"
        }
        
        for panel_tab, panel_id in panels.items():
            panel = self.query_one(f"#{panel_id}")
            if panel_tab == tab_name:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")
    
    def action_show_suggestions(self) -> None:
        """Show suggestions tab"""
        self._show_tab("suggestions")
    
    def action_show_explanation(self) -> None:
        """Show explanation tab"""
        self._show_tab("explanation")
    
    def action_show_fixes(self) -> None:
        """Show fixes tab"""
        self._show_tab("fixes")
    
    def action_show_progress(self) -> None:
        """Show progress tab"""
        self._show_tab("progress")
    
    def action_show_history(self) -> None:
        """Show history tab"""
        self._show_tab("history")
    
    # Convenience methods for updating panels
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Update command suggestions"""
        panel = self.query_one("#suggestions-panel", CommandSuggestionPanel)
        panel.update_suggestions(suggestions)
    
    def show_explanation(self, explanation: Dict[str, Any]) -> None:
        """Show command explanation"""
        panel = self.query_one("#explanation-panel", CommandExplanationViewer)
        panel.show_explanation(explanation)
        self._show_tab("explanation")
    
    def show_error_fixes(self, command: str, error: str, fixes: List[Dict[str, Any]]) -> None:
        """Show error fix suggestions"""
        panel = self.query_one("#fixes-panel", ErrorFixPanel)
        panel.show_fixes(command, error, fixes)
        self._show_tab("fixes")
    
    def update_learning_progress(self, stats: Dict[str, Any]) -> None:
        """Update learning progress"""
        panel = self.query_one("#progress-panel", LearningProgressTracker)
        panel.update_progress(stats)
    
    def update_command_history(self, history: List[Dict[str, Any]]) -> None:
        """Update command history"""
        panel = self.query_one("#history-panel", CommandHistoryBrowser)
        panel.update_history(history)
