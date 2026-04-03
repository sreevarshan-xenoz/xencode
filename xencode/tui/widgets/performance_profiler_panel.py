#!/usr/bin/env python3
"""
Performance Profiler Panel Widget for Xencode TUI

Interactive performance profiler interface with profile viewer, bottleneck visualization,
metrics dashboard, before/after comparison, and optimization suggestions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, ListView, ListItem, ProgressBar, DataTable, Tree
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel


class MetricsDashboard(Container):
    """Dashboard showing performance metrics summary"""
    
    DEFAULT_CSS = """
    MetricsDashboard {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    
    #metrics-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    #metrics-content {
        height: 1fr;
        padding: 1;
    }
    
    .summary-row {
        height: auto;
        padding: 1;
        margin: 0 1;
    }
    
    .metric-card {
        height: 8;
        border: solid $primary;
        background: $panel;
        padding: 1;
        margin: 1;
    }
    
    .metric-card.slow {
        border: solid $error;
        background: $error-darken-3;
    }
    
    .metric-card.moderate {
        border: solid $warning;
        background: $warning-darken-3;
    }
    
    .metric-card.fast {
        border: solid $success;
    }
    
    .metric-title {
        height: 1;
        text-style: bold;
    }
    
    .metric-value {
        height: 3;
        content-align: center middle;
        text-style: bold;
    }
    
    .metric-label {
        height: 1;
        content-align: center middle;
        text-style: dim;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = {
            'execution_time': 0.0,
            'cpu_percent': 0.0,
            'memory_usage': 0.0,
            'bottlenecks': 0,
            'hotspots': 0
        }
    
    def compose(self) -> ComposeResult:
        """Compose the metrics dashboard"""
        yield Label("⚡ Performance Metrics", id="metrics-header")
        
        with ScrollableContainer(id="metrics-content"):
            # Execution time card
            exec_class = self._get_exec_time_class(self.metrics['execution_time'])
            with Container(classes=f"metric-card {exec_class}"):
                yield Label("⏱️ Execution Time", classes="metric-title")
                yield Label(f"{self.metrics['execution_time']:.4f}s", id="exec-time-value", classes="metric-value")
                yield Label("Total Runtime", classes="metric-label")
            
            # CPU usage card
            cpu_class = self._get_cpu_class(self.metrics['cpu_percent'])
            with Container(classes=f"metric-card {cpu_class}"):
                yield Label("🖥️ CPU Usage", classes="metric-title")
                yield Label(f"{self.metrics['cpu_percent']:.2f}%", id="cpu-value", classes="metric-value")
                yield Label("Processor Load", classes="metric-label")
            
            # Memory usage card
            mem_class = self._get_memory_class(self.metrics['memory_usage'])
            with Container(classes=f"metric-card {mem_class}"):
                yield Label("💾 Memory Usage", classes="metric-title")
                yield Label(f"{self.metrics['memory_usage']:.2f} MB", id="memory-value", classes="metric-value")
                yield Label("RAM Consumption", classes="metric-label")
            
            # Bottlenecks card
            with Container(classes="metric-card"):
                yield Label("🐌 Bottlenecks", classes="metric-title")
                yield Label(str(self.metrics['bottlenecks']), id="bottlenecks-value", classes="metric-value")
                yield Label("Slow Functions", classes="metric-label")
            
            # Hotspots card
            with Container(classes="metric-card"):
                yield Label("🔥 Hotspots", classes="metric-title")
                yield Label(str(self.metrics['hotspots']), id="hotspots-value", classes="metric-value")
                yield Label("Frequently Called", classes="metric-label")
    
    def _get_exec_time_class(self, time: float) -> str:
        """Get CSS class based on execution time"""
        if time > 5.0:
            return "slow"
        elif time > 1.0:
            return "moderate"
        return "fast"
    
    def _get_cpu_class(self, cpu: float) -> str:
        """Get CSS class based on CPU usage"""
        if cpu > 80:
            return "slow"
        elif cpu > 50:
            return "moderate"
        return "fast"
    
    def _get_memory_class(self, memory: float) -> str:
        """Get CSS class based on memory usage"""
        if memory > 500:
            return "slow"
        elif memory > 200:
            return "moderate"
        return "fast"
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics"""
        self.metrics.update(metrics)
        
        try:
            self.query_one("#exec-time-value", Label).update(f"{metrics.get('execution_time', 0):.4f}s")
            self.query_one("#cpu-value", Label).update(f"{metrics.get('cpu_percent', 0):.2f}%")
            self.query_one("#memory-value", Label).update(f"{metrics.get('memory_usage', 0):.2f} MB")
            self.query_one("#bottlenecks-value", Label).update(str(metrics.get('bottlenecks', 0)))
            self.query_one("#hotspots-value", Label).update(str(metrics.get('hotspots', 0)))
        except:
            pass


class BottleneckVisualization(Container):
    """Visualization of performance bottlenecks"""
    
    DEFAULT_CSS = """
    BottleneckVisualization {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }
    
    #bottleneck-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    #bottleneck-content {
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bottlenecks: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the bottleneck visualization"""
        yield Label("🐌 Performance Bottlenecks", id="bottleneck-header")
        
        with ScrollableContainer(id="bottleneck-content"):
            table = DataTable(id="bottleneck-table")
            table.add_columns("Function", "Time (s)", "% of Total", "Calls", "Time/Call")
            yield table
    
    def update_bottlenecks(self, bottlenecks: List[Dict[str, Any]], total_time: float = 1.0) -> None:
        """Update the bottleneck visualization"""
        self.bottlenecks = bottlenecks
        
        try:
            table = self.query_one("#bottleneck-table", DataTable)
            table.clear()
            
            if bottlenecks:
                for bottleneck in bottlenecks:
                    func_name = bottleneck.get('function', 'unknown')
                    time = bottleneck.get('time', 0.0)
                    percentage = (time / total_time * 100) if total_time > 0 else 0.0
                    calls = bottleneck.get('calls', 0)
                    time_per_call = time / calls if calls > 0 else 0.0
                    
                    # Truncate long function names
                    if len(func_name) > 40:
                        func_name = func_name[:37] + "..."
                    
                    table.add_row(
                        func_name,
                        f"{time:.4f}",
                        f"{percentage:.1f}%",
                        str(calls),
                        f"{time_per_call:.6f}"
                    )
        except:
            pass


class ProfileViewer(Container):
    """Viewer for detailed profile results"""
    
    DEFAULT_CSS = """
    ProfileViewer {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    #profile-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    #profile-content {
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function_stats: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the profile viewer"""
        yield Label("📊 Function Statistics", id="profile-header")
        
        with ScrollableContainer(id="profile-content"):
            table = DataTable(id="profile-table")
            table.add_columns("Function", "File", "Calls", "Total Time", "Cumulative", "Time/Call")
            yield table
    
    def update_profile(self, function_stats: List[Dict[str, Any]]) -> None:
        """Update the profile viewer"""
        self.function_stats = function_stats
        
        try:
            table = self.query_one("#profile-table", DataTable)
            table.clear()
            
            if function_stats:
                for stat in function_stats[:50]:  # Show top 50
                    func_name = stat.get('function', 'unknown')
                    file_name = stat.get('file', '')
                    
                    # Truncate long names
                    if len(func_name) > 30:
                        func_name = func_name[:27] + "..."
                    if len(file_name) > 30:
                        file_name = "..." + file_name[-27:]
                    
                    table.add_row(
                        func_name,
                        file_name,
                        str(stat.get('calls', 0)),
                        f"{stat.get('total_time', 0):.4f}",
                        f"{stat.get('cumulative_time', 0):.4f}",
                        f"{stat.get('time_per_call', 0):.6f}"
                    )
        except:
            pass


class OptimizationSuggestions(Container):
    """Panel showing optimization suggestions"""
    
    DEFAULT_CSS = """
    OptimizationSuggestions {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    
    #opt-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    #opt-content {
        height: 1fr;
        padding: 1;
    }
    
    .suggestion-item {
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $panel;
        border-left: thick $primary;
    }
    
    .suggestion-title {
        height: auto;
        text-style: bold;
        padding: 0 0 1 0;
    }
    
    .suggestion-section {
        height: auto;
        padding: 0 0 1 2;
    }
    
    .code-snippet {
        background: $surface;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suggestions: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the optimization suggestions"""
        yield Label("💡 Optimization Suggestions", id="opt-header")
        
        with ScrollableContainer(id="opt-content"):
            yield Label("Run analysis to see optimization suggestions", classes="dim", id="opt-placeholder")
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Update optimization suggestions"""
        self.suggestions = suggestions
        
        opt_content = self.query_one("#opt-content", ScrollableContainer)
        opt_content.remove_children()
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                with Container(classes="suggestion-item"):
                    # Title
                    func_name = suggestion.get('function_name', 'unknown')
                    opt_content.mount(Label(f"{i}. {func_name}", classes="suggestion-title"))
                    
                    # Issue
                    with Container(classes="suggestion-section"):
                        opt_content.mount(Label("Issue:", classes="bold"))
                        opt_content.mount(Label(suggestion.get('issue', 'N/A')))
                    
                    # Suggestion
                    with Container(classes="suggestion-section"):
                        opt_content.mount(Label("Suggestion:", classes="bold"))
                        opt_content.mount(Label(suggestion.get('suggestion', 'N/A')))
                    
                    # Before/After examples
                    if suggestion.get('example_before'):
                        with Container(classes="suggestion-section"):
                            opt_content.mount(Label("Before:", classes="bold"))
                            with Container(classes="code-snippet"):
                                opt_content.mount(Label(suggestion['example_before']))
                    
                    if suggestion.get('example_after'):
                        with Container(classes="suggestion-section"):
                            opt_content.mount(Label("After:", classes="bold"))
                            with Container(classes="code-snippet"):
                                opt_content.mount(Label(suggestion['example_after']))
                    
                    # Estimated improvement
                    with Container(classes="suggestion-section"):
                        opt_content.mount(Label("Estimated Improvement:", classes="bold"))
                        opt_content.mount(Label(suggestion.get('estimated_improvement', 'N/A'), classes="dim"))
        else:
            opt_content.mount(Label("No optimization suggestions available", classes="dim"))


class ComparisonView(Container):
    """View for before/after performance comparison"""
    
    DEFAULT_CSS = """
    ComparisonView {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    
    #comparison-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    #comparison-content {
        height: 1fr;
        padding: 1;
    }
    
    .comparison-section {
        height: auto;
        padding: 1;
        margin: 1 0;
    }
    
    .comparison-summary {
        background: $panel;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    
    .improved {
        color: $success;
        text-style: bold;
    }
    
    .regressed {
        color: $error;
        text-style: bold;
    }
    
    .unchanged {
        color: $text-muted;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comparison: Optional[Dict[str, Any]] = None
    
    def compose(self) -> ComposeResult:
        """Compose the comparison view"""
        yield Label("📈 Performance Comparison", id="comparison-header")
        
        with ScrollableContainer(id="comparison-content"):
            yield Label("Run comparison to see before/after results", classes="dim", id="comparison-placeholder")
    
    def update_comparison(self, comparison: Dict[str, Any]) -> None:
        """Update the comparison view"""
        self.comparison = comparison
        
        comp_content = self.query_one("#comparison-content", ScrollableContainer)
        comp_content.remove_children()
        
        if comparison:
            # Paths
            with Container(classes="comparison-section"):
                comp_content.mount(Label("Baseline:", classes="bold"))
                comp_content.mount(Label(comparison.get('baseline_path', 'N/A'), classes="dim"))
                comp_content.mount(Label("Comparison:", classes="bold"))
                comp_content.mount(Label(comparison.get('comparison_path', 'N/A'), classes="dim"))
            
            # Summary
            with Container(classes="comparison-section"):
                comp_content.mount(Label("Summary:", classes="bold"))
                with Container(classes="comparison-summary"):
                    summary_text = comparison.get('summary', 'No summary available')
                    for line in summary_text.split('\n'):
                        if '✓' in line:
                            comp_content.mount(Label(line, classes="improved"))
                        elif '✗' in line:
                            comp_content.mount(Label(line, classes="regressed"))
                        else:
                            comp_content.mount(Label(line))
            
            # Execution time
            exec_diff = comparison.get('execution_time_diff', 0.0)
            exec_percent = comparison.get('execution_time_percent', 0.0)
            with Container(classes="comparison-section"):
                comp_content.mount(Label("Execution Time Change:", classes="bold"))
                if exec_diff < 0:
                    comp_content.mount(Label(f"✓ {abs(exec_percent):.1f}% faster ({abs(exec_diff):.4f}s)", classes="improved"))
                elif exec_diff > 0:
                    comp_content.mount(Label(f"✗ {exec_percent:.1f}% slower (+{exec_diff:.4f}s)", classes="regressed"))
                else:
                    comp_content.mount(Label("= Unchanged", classes="unchanged"))
            
            # Improved functions
            improved = comparison.get('improved_functions', [])
            if improved:
                with Container(classes="comparison-section"):
                    comp_content.mount(Label(f"Improved Functions ({len(improved)}):", classes="bold"))
                    table = DataTable()
                    table.add_columns("Function", "Baseline", "New", "Change")
                    for func in improved[:10]:
                        table.add_row(
                            func.get('function', 'unknown')[:30],
                            f"{func.get('baseline_time', 0):.4f}s",
                            f"{func.get('comparison_time', 0):.4f}s",
                            f"{func.get('percent_change', 0):.1f}%"
                        )
                    comp_content.mount(table)
            
            # Regressed functions
            regressed = comparison.get('regressed_functions', [])
            if regressed:
                with Container(classes="comparison-section"):
                    comp_content.mount(Label(f"Regressed Functions ({len(regressed)}):", classes="bold"))
                    table = DataTable()
                    table.add_columns("Function", "Baseline", "New", "Change")
                    for func in regressed[:10]:
                        table.add_row(
                            func.get('function', 'unknown')[:30],
                            f"{func.get('baseline_time', 0):.4f}s",
                            f"{func.get('comparison_time', 0):.4f}s",
                            f"+{func.get('percent_change', 0):.1f}%"
                        )
                    comp_content.mount(table)
        else:
            comp_content.mount(Label("No comparison data available", classes="dim"))


class PerformanceProfilerPanel(Container):
    """Main performance profiler panel with all components"""
    
    DEFAULT_CSS = """
    PerformanceProfilerPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    PerformanceProfilerPanel > #pp-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    PerformanceProfilerPanel > #pp-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }
    
    PerformanceProfilerPanel > #pp-tabs {
        dock: top;
        height: 3;
        background: $panel;
    }
    
    PerformanceProfilerPanel > #pp-content {
        height: 1fr;
    }
    
    .control-button {
        width: 1fr;
        margin: 0 1;
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
        Binding("ctrl+p", "profile", "Profile"),
        Binding("ctrl+a", "analyze", "Analyze"),
        Binding("ctrl+o", "optimize", "Optimize"),
        Binding("ctrl+c", "compare", "Compare"),
        Binding("1", "show_dashboard", "Dashboard"),
        Binding("2", "show_bottlenecks", "Bottlenecks"),
        Binding("3", "show_profile", "Profile"),
        Binding("4", "show_suggestions", "Suggestions"),
        Binding("5", "show_comparison", "Comparison"),
    ]
    
    class ProfilerAction(Message):
        """Message sent when a profiler action is requested"""
        def __init__(self, action: str, path: str = None, **kwargs):
            super().__init__()
            self.action = action
            self.path = path
            self.kwargs = kwargs
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_tab = "dashboard"
        self.profile_path = "."
    
    def compose(self) -> ComposeResult:
        """Compose the performance profiler panel"""
        yield Label("⚡ Performance Profiler", id="pp-header")
        
        with Horizontal(id="pp-controls"):
            yield Button("▶️ Profile", classes="control-button", id="btn-profile")
            yield Button("🔍 Analyze", classes="control-button", id="btn-analyze")
            yield Button("💡 Optimize", classes="control-button", id="btn-optimize")
            yield Button("📊 Compare", classes="control-button", id="btn-compare")
        
        with Horizontal(id="pp-tabs"):
            yield Button("Dashboard", classes="tab-button active", id="tab-dashboard")
            yield Button("Bottlenecks", classes="tab-button", id="tab-bottlenecks")
            yield Button("Profile", classes="tab-button", id="tab-profile")
            yield Button("Suggestions", classes="tab-button", id="tab-suggestions")
            yield Button("Comparison", classes="tab-button", id="tab-comparison")
        
        with Container(id="pp-content"):
            yield MetricsDashboard(id="dashboard-panel")
            yield BottleneckVisualization(id="bottlenecks-panel", classes="hidden")
            yield ProfileViewer(id="profile-panel", classes="hidden")
            yield OptimizationSuggestions(id="suggestions-panel", classes="hidden")
            yield ComparisonView(id="comparison-panel", classes="hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn-profile":
            self.action_profile()
        elif button_id == "btn-analyze":
            self.action_analyze()
        elif button_id == "btn-optimize":
            self.action_optimize()
        elif button_id == "btn-compare":
            self.action_compare()
        elif button_id.startswith("tab-"):
            tab_name = button_id.replace("tab-", "")
            self._show_tab(tab_name)
    
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
            "dashboard": "dashboard-panel",
            "bottlenecks": "bottlenecks-panel",
            "profile": "profile-panel",
            "suggestions": "suggestions-panel",
            "comparison": "comparison-panel"
        }
        
        for name, panel_id in panels.items():
            panel = self.query_one(f"#{panel_id}")
            if name == tab_name:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")
    
    def action_profile(self) -> None:
        """Run performance profile"""
        self.post_message(self.ProfilerAction("profile", self.profile_path))
    
    def action_analyze(self) -> None:
        """Analyze performance"""
        self.post_message(self.ProfilerAction("analyze", self.profile_path))
    
    def action_optimize(self) -> None:
        """Get optimization suggestions"""
        self.post_message(self.ProfilerAction("optimize", self.profile_path))
    
    def action_compare(self) -> None:
        """Compare performance"""
        self.post_message(self.ProfilerAction("compare", self.profile_path))
    
    def action_show_dashboard(self) -> None:
        """Show dashboard tab"""
        self._show_tab("dashboard")
    
    def action_show_bottlenecks(self) -> None:
        """Show bottlenecks tab"""
        self._show_tab("bottlenecks")
    
    def action_show_profile(self) -> None:
        """Show profile tab"""
        self._show_tab("profile")
    
    def action_show_suggestions(self) -> None:
        """Show suggestions tab"""
        self._show_tab("suggestions")
    
    def action_show_comparison(self) -> None:
        """Show comparison tab"""
        self._show_tab("comparison")
    
    # Convenience methods for updating components
    
    def update_dashboard(self, metrics: Dict[str, Any]) -> None:
        """Update metrics dashboard"""
        dashboard = self.query_one("#dashboard-panel", MetricsDashboard)
        dashboard.update_metrics(metrics)
    
    def update_bottlenecks(self, bottlenecks: List[Dict[str, Any]], total_time: float = 1.0) -> None:
        """Update bottleneck visualization"""
        bottleneck_panel = self.query_one("#bottlenecks-panel", BottleneckVisualization)
        bottleneck_panel.update_bottlenecks(bottlenecks, total_time)
    
    def update_profile(self, function_stats: List[Dict[str, Any]]) -> None:
        """Update profile viewer"""
        profile_panel = self.query_one("#profile-panel", ProfileViewer)
        profile_panel.update_profile(function_stats)
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Update optimization suggestions"""
        suggestions_panel = self.query_one("#suggestions-panel", OptimizationSuggestions)
        suggestions_panel.update_suggestions(suggestions)
    
    def update_comparison(self, comparison: Dict[str, Any]) -> None:
        """Update comparison view"""
        comparison_panel = self.query_one("#comparison-panel", ComparisonView)
        comparison_panel.update_comparison(comparison)
    
    def set_profile_path(self, path: str) -> None:
        """Set the profile path"""
        self.profile_path = path
