"""Performance Profiler TUI panel."""

from typing import Any, Dict, List

from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

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
            *args,
            feature_name="performance_profiler",
            title="⚡ Performance Profiler",
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
            # Integrate with actual performance profiler
            import cProfile
            import io
            import pstats

            # Profile current project
            profiler = cProfile.Profile()
            profiler.enable()

            # Run a simple operation to profile
            from xencode.performance.optimizer import PerformanceOptimizer
            optimizer = PerformanceOptimizer()
            # Start monitoring for system-level metrics
            await optimizer.start_monitoring()

            profiler.disable()

            # Stop monitoring to prevent background task leak
            await optimizer.stop_monitoring()

            # Get stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

            # Parse stats for UI
            self.bottlenecks = []
            for func, (_cc, nc, tt, _ct, _callers) in stats.stats.items():
                if tt > 0.01:  # Only show functions taking > 10ms
                    self.bottlenecks.append({
                        "function": f"{func[2]}:{func[1]}",
                        "time_ms": tt * 1000,
                        "calls": nc
                    })
                if len(self.bottlenecks) >= 20:
                    break

            self._build_content()
            self.set_status("enabled")
        except ImportError:
            # Fallback if profiler not available
            self.bottlenecks = []
            self._build_content()
            self.set_status("disabled")
        except Exception as e:
            self.show_empty_state(f"Error profiling: {e}")
            self.set_status("disabled")
        finally:
            self.profiling = False

    async def _analyze_results(self) -> None:
        """Analyze profiling results."""
        if not self.bottlenecks:
            self.notify("No profiling results to analyze")
            return

        # Analyze top bottlenecks
        top_issues = sorted(self.bottlenecks, key=lambda x: x["time_ms"], reverse=True)[:5]
        analysis = "Top Performance Issues:\n\n"
        for i, bottleneck in enumerate(top_issues, 1):
            analysis += f"{i}. {bottleneck['function']} - {bottleneck['time_ms']:.1f}ms ({bottleneck['calls']} calls)\n"
        analysis += "\nRecommendation: Focus on optimizing the top 3 functions first."

        self.notify(analysis)

    async def _show_optimizations(self) -> None:
        """Show optimization suggestions."""
        if not self.bottlenecks:
            self.notify("No profiling results available")
            return

        # Generate optimization suggestions based on patterns
        suggestions = []
        for bottleneck in self.bottlenecks[:5]:
            func_name = bottleneck["function"]
            if "loop" in func_name.lower() or "process" in func_name.lower():
                suggestions.append(f"• {func_name}: Consider vectorization or parallelization")
            elif "fetch" in func_name.lower() or "query" in func_name.lower():
                suggestions.append(f"• {func_name}: Add caching or optimize database query")
            elif "validate" in func_name.lower():
                suggestions.append(f"• {func_name}: Use compiled validation library")

        if suggestions:
            self.notify("Optimization Suggestions:\n\n" + "\n".join(suggestions))
        else:
            self.notify("No specific optimizations suggested")

    async def _compare_profiles(self) -> None:
        """Compare profile results."""
        if not self.bottlenecks:
            self.notify("No current profile to compare")
            return

        # Compare with baseline or previous run
        self.notify("Profile comparison: Load a previous profile to compare with current results")
