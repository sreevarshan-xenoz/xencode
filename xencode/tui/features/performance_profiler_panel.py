"""Performance Profiler TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, DataTable
from textual.reactive import reactive
from typing import List, Dict, Any

from .base_feature_panel import BaseFeaturePanel


class BottleneckCard(Static):
    """Card for a performance bottleneck."""
    
    DEFAULT_CSS = """
    BottleneckCard {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $warning;
        background: $panel;
    }
    
    BottleneckCard:hover {
        background: $primary;
    }
    """
    
    def __init__(self, function: str, time_ms: float, calls: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = function
        self.time_ms = time_ms
        self.calls = calls
    
    def render(self) -> str:
        return (
            f"[bold]{self.function}[/bold]\n"
            f"Time: {self.time_ms:.2f}ms | Calls: {self.calls}"
        )


class PerformanceProfilerPanel(BaseFeaturePanel):
    """Panel for performance profiling and optimization."""
    
    DEFAULT_CSS = """
    PerformanceProfilerPanel {
        height: 100%;
    }
    
    .profiler-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .profiler-content {
        height: 1fr;
        padding: 1;
    }
    
    .profiler-summary {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $accent;
        background: $panel;
    }
    """
    
    profiling = reactive(False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="performance_profiler",
            title="⚡ Performance Profiler",
            *args,
            **kwargs
        )
        self.bottlenecks: List[Dict[str, Any]] = []
        self.profile_results: Dict[str, Any] = {}
    
    def compose(self):
        """Compose the performance profiler panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._build_content()
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Controls
            with Horizontal(classes="profiler-controls"):
                yield Button("Run Profile", id="btn-profile", variant="primary")
                yield Button("Analyze", id="btn-analyze")
                yield Button("Optimize", id="btn-optimize")
                yield Button("Compare", id="btn-compare")
            
            # Content area
            with ScrollableContainer(classes="profiler-content"):
                if self.bottlenecks:
                    self._render_results()
                else:
                    yield Label(
                        "Click 'Run Profile' to analyze code performance.",
                        classes="feature-empty"
                    )
    
    def _render_results(self) -> None:
        """Render profiling results."""
        # Summary
        total_time = sum(b["time_ms"] for b in self.bottlenecks)
        total_calls = sum(b["calls"] for b in self.bottlenecks)
        
        yield Static(
            f"Total Time: {total_time:.2f}ms | Total Calls: {total_calls}",
            classes="profiler-summary"
        )
        
        # Bottlenecks
        for bottleneck in self.bottlenecks:
            yield BottleneckCard(
                bottleneck["function"],
                bottleneck["time_ms"],
                bottleneck["calls"]
            )
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-profile":
            await self._run_profile()
        elif button_id == "btn-analyze":
            await self._analyze_results()
        elif button_id == "btn-optimize":
            await self._show_optimizations()
        elif button_id == "btn-compare":
            await self._compare_profiles()
    
    async def _run_profile(self) -> None:
        """Run performance profile."""
        self.set_status("loading")
        self.profiling = True
        
        try:
            # TODO: Integrate with actual performance profiler
            self.bottlenecks = [
                {"function": "process_data()", "time_ms": 245.3, "calls": 1000},
                {"function": "fetch_records()", "time_ms": 189.7, "calls": 500},
                {"function": "validate_input()", "time_ms": 52.1, "calls": 2000},
            ]
            self._build_content()
            self.set_status("enabled")
        except Exception as e:
            self.show_empty_state(f"Error profiling: {e}")
            self.set_status("disabled")
        finally:
            self.profiling = False
    
    async def _analyze_results(self) -> None:
        """Analyze profiling results."""
        # TODO: Implement analysis
        pass
    
    async def _show_optimizations(self) -> None:
        """Show optimization suggestions."""
        # TODO: Implement optimization suggestions
        pass
    
    async def _compare_profiles(self) -> None:
        """Compare profile results."""
        # TODO: Implement comparison
        pass
