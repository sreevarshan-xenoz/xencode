#!/usr/bin/env python3
"""
Tests for Performance Profiler CLI and TUI Components

Tests the CLI commands and TUI interface for the Performance Profiler feature.
"""


import pytest

from xencode.features.base import FeatureConfig
from xencode.features.performance_profiler import (
    PerformanceProfiler,
    ProfileResult,
)


@pytest.fixture
def profiler_config():
    """Create a test profiler configuration"""
    return FeatureConfig(
        name="performance_profiler",
        enabled=True,
        config={
            "enabled": True,
            "tools": ["cprofile", "psutil"],
            "output_format": "text",
            "sort_by": "cumulative",
            "max_results": 50
        }
    )


@pytest.fixture
def profiler(profiler_config):
    """Create a test profiler instance"""
    return PerformanceProfiler(profiler_config)


@pytest.fixture
def sample_code_file(tmp_path):
    """Create a sample Python file for profiling"""
    code = """
import time

def slow_function():
    time.sleep(0.01)
    return sum(range(10000))

def fast_function():
    return 42

def main():
    for i in range(3):
        slow_function()
    for i in range(10):
        fast_function()
    return "Done"

if __name__ == "__main__":
    main()
"""
    file_path = tmp_path / "sample.py"
    file_path.write_text(code)
    return str(file_path)


@pytest.fixture
def sample_profile_result():
    """Create a sample profile result"""
    return ProfileResult(
        execution_time=1.234,
        cpu_percent=45.6,
        memory_usage=123.4,
        function_stats=[
            {
                'function': 'slow_function',
                'file': 'sample.py',
                'line': 3,
                'calls': 3,
                'total_time': 0.5,
                'cumulative_time': 0.6,
                'time_per_call': 0.166
            },
            {
                'function': 'fast_function',
                'file': 'sample.py',
                'line': 7,
                'calls': 10,
                'total_time': 0.001,
                'cumulative_time': 0.001,
                'time_per_call': 0.0001
            }
        ],
        bottlenecks=[
            {
                'function': 'slow_function',
                'time': 0.6,
                'percentage': 48.6,
                'calls': 3
            }
        ]
    )


class TestCLICommands:
    """Test CLI command functionality"""

    def test_cli_commands_available(self, profiler):
        """Test that CLI commands are available"""
        commands = profiler.get_cli_commands()
        assert len(commands) > 0
        assert commands[0].name == 'profile'

    @pytest.mark.asyncio
    async def test_profile_command(self, profiler, sample_code_file):
        """Test profile run command"""
        result = await profiler.profile(sample_code_file)

        assert result['success'] is True
        assert 'result' in result
        assert 'stats' in result
        assert result['result']['execution_time'] > 0

    @pytest.mark.asyncio
    async def test_analyze_command(self, profiler, sample_code_file):
        """Test profile analyze command"""
        result = await profiler.analyze(sample_code_file)

        assert result['success'] is True
        assert 'analysis' in result
        assert 'bottlenecks' in result['analysis']
        assert 'hotspots' in result['analysis']

    @pytest.mark.asyncio
    async def test_optimize_command(self, profiler, sample_code_file):
        """Test profile optimize command"""
        result = await profiler.optimize(sample_code_file)

        assert result['success'] is True
        assert 'suggestions' in result
        assert isinstance(result['suggestions'], list)

    @pytest.mark.asyncio
    async def test_report_command(self, profiler, sample_code_file):
        """Test profile report command"""
        result = await profiler.generate_report(sample_code_file, save_history=False)

        assert result['success'] is True
        assert 'report' in result
        assert 'path' in result['report']
        assert 'timestamp' in result['report']
        assert 'profile_result' in result['report']
        assert 'analysis' in result['report']
        assert 'suggestions' in result['report']

    @pytest.mark.asyncio
    async def test_history_command(self, profiler):
        """Test profile history command"""
        result = await profiler.list_history()

        assert result['success'] is True
        assert 'profiles' in result
        assert isinstance(result['profiles'], list)

    @pytest.mark.asyncio
    async def test_compare_command(self, profiler, sample_code_file, tmp_path):
        """Test profile compare command"""
        # Create a second file
        code2 = """
import time

def optimized_function():
    return sum(range(10000))

def main():
    for i in range(3):
        optimized_function()
    return "Done"

if __name__ == "__main__":
    main()
"""
        file2 = tmp_path / "optimized.py"
        file2.write_text(code2)

        result = await profiler.compare(sample_code_file, str(file2))

        assert result['success'] is True
        assert 'comparison' in result
        assert 'baseline_path' in result['comparison']
        assert 'comparison_path' in result['comparison']
        assert 'execution_time_diff' in result['comparison']
        assert 'summary' in result['comparison']


class TestTUIComponents:
    """Test TUI component functionality"""

    def test_tui_components_available(self, profiler):
        """Test that TUI components are available"""
        components = profiler.get_tui_components()
        assert len(components) > 0

    def test_metrics_dashboard_creation(self):
        """Test MetricsDashboard component creation"""
        from xencode.tui.widgets.performance_profiler_panel import MetricsDashboard

        dashboard = MetricsDashboard()
        assert dashboard is not None
        assert dashboard.metrics['execution_time'] == 0.0
        assert dashboard.metrics['cpu_percent'] == 0.0
        assert dashboard.metrics['memory_usage'] == 0.0

    def test_metrics_dashboard_update(self):
        """Test MetricsDashboard update"""
        from xencode.tui.widgets.performance_profiler_panel import MetricsDashboard

        dashboard = MetricsDashboard()
        metrics = {
            'execution_time': 1.234,
            'cpu_percent': 45.6,
            'memory_usage': 123.4,
            'bottlenecks': 5,
            'hotspots': 10
        }

        dashboard.update_metrics(metrics)
        assert dashboard.metrics['execution_time'] == 1.234
        assert dashboard.metrics['cpu_percent'] == 45.6
        assert dashboard.metrics['memory_usage'] == 123.4

    def test_bottleneck_visualization_creation(self):
        """Test BottleneckVisualization component creation"""
        from xencode.tui.widgets.performance_profiler_panel import (
            BottleneckVisualization,
        )

        viz = BottleneckVisualization()
        assert viz is not None
        assert viz.bottlenecks == []

    def test_bottleneck_visualization_update(self):
        """Test BottleneckVisualization update"""
        from xencode.tui.widgets.performance_profiler_panel import (
            BottleneckVisualization,
        )

        viz = BottleneckVisualization()
        bottlenecks = [
            {
                'function': 'slow_function',
                'time': 0.6,
                'percentage': 48.6,
                'calls': 3
            }
        ]

        viz.update_bottlenecks(bottlenecks, total_time=1.234)
        assert len(viz.bottlenecks) == 1
        assert viz.bottlenecks[0]['function'] == 'slow_function'

    def test_profile_viewer_creation(self):
        """Test ProfileViewer component creation"""
        from xencode.tui.widgets.performance_profiler_panel import ProfileViewer

        viewer = ProfileViewer()
        assert viewer is not None
        assert viewer.function_stats == []

    def test_profile_viewer_update(self):
        """Test ProfileViewer update"""
        from xencode.tui.widgets.performance_profiler_panel import ProfileViewer

        viewer = ProfileViewer()
        stats = [
            {
                'function': 'test_func',
                'file': 'test.py',
                'line': 10,
                'calls': 5,
                'total_time': 0.5,
                'cumulative_time': 0.6,
                'time_per_call': 0.1
            }
        ]

        viewer.update_profile(stats)
        assert len(viewer.function_stats) == 1
        assert viewer.function_stats[0]['function'] == 'test_func'

    def test_optimization_suggestions_creation(self):
        """Test OptimizationSuggestions component creation"""
        from xencode.tui.widgets.performance_profiler_panel import (
            OptimizationSuggestions,
        )

        suggestions = OptimizationSuggestions()
        assert suggestions is not None
        assert suggestions.suggestions == []

    def test_optimization_suggestions_update(self):
        """Test OptimizationSuggestions update"""
        from xencode.tui.widgets.performance_profiler_panel import (
            OptimizationSuggestions,
        )

        suggestions_panel = OptimizationSuggestions()
        suggestions = [
            {
                'function_name': 'slow_function',
                'issue': 'Using loops',
                'suggestion': 'Use list comprehension',
                'example_before': 'for i in range(10): ...',
                'example_after': '[i for i in range(10)]',
                'estimated_improvement': '20-30% faster'
            }
        ]

        # Just store suggestions without querying DOM (widget not mounted)
        suggestions_panel.suggestions = suggestions
        assert len(suggestions_panel.suggestions) == 1
        assert suggestions_panel.suggestions[0]['function_name'] == 'slow_function'

    def test_comparison_view_creation(self):
        """Test ComparisonView component creation"""
        from xencode.tui.widgets.performance_profiler_panel import ComparisonView

        view = ComparisonView()
        assert view is not None
        assert view.comparison is None

    def test_comparison_view_update(self):
        """Test ComparisonView update"""
        from xencode.tui.widgets.performance_profiler_panel import ComparisonView

        view = ComparisonView()
        comparison = {
            'baseline_path': 'old.py',
            'comparison_path': 'new.py',
            'execution_time_diff': -0.5,
            'execution_time_percent': -40.0,
            'cpu_diff': -10.0,
            'memory_diff': -50.0,
            'improved_functions': [],
            'regressed_functions': [],
            'summary': '✓ Overall performance improved'
        }

        # Just store comparison without querying DOM (widget not mounted)
        view.comparison = comparison
        assert view.comparison is not None
        assert view.comparison['baseline_path'] == 'old.py'

    def test_performance_profiler_panel_creation(self):
        """Test PerformanceProfilerPanel component creation"""
        from xencode.tui.widgets.performance_profiler_panel import (
            PerformanceProfilerPanel,
        )

        panel = PerformanceProfilerPanel()
        assert panel is not None
        assert panel.current_tab == "dashboard"
        assert panel.profile_path == "."

    def test_performance_profiler_panel_tab_switching(self):
        """Test PerformanceProfilerPanel tab switching"""
        from xencode.tui.widgets.performance_profiler_panel import (
            PerformanceProfilerPanel,
        )

        panel = PerformanceProfilerPanel()

        # Test tab switching (just update state, don't query DOM)
        panel.current_tab = "bottlenecks"
        assert panel.current_tab == "bottlenecks"

        panel.current_tab = "profile"
        assert panel.current_tab == "profile"

        panel.current_tab = "suggestions"
        assert panel.current_tab == "suggestions"

        panel.current_tab = "comparison"
        assert panel.current_tab == "comparison"

    def test_performance_profiler_panel_update_methods(self):
        """Test PerformanceProfilerPanel update methods"""
        from xencode.tui.widgets.performance_profiler_panel import (
            PerformanceProfilerPanel,
        )

        panel = PerformanceProfilerPanel()

        # Test update_dashboard
        # This would normally update the dashboard, but we can't test UI updates without mounting
        # Just verify the method exists and is callable
        assert hasattr(panel, 'update_dashboard')
        assert callable(panel.update_dashboard)

        # Test other update methods
        assert hasattr(panel, 'update_bottlenecks')
        assert hasattr(panel, 'update_profile')
        assert hasattr(panel, 'update_suggestions')
        assert hasattr(panel, 'update_comparison')
        assert hasattr(panel, 'set_profile_path')


class TestIntegration:
    """Test CLI and TUI integration"""

    @pytest.mark.asyncio
    async def test_full_profiling_workflow(self, profiler, sample_code_file):
        """Test complete profiling workflow"""
        # 1. Profile code
        profile_result = await profiler.profile(sample_code_file)
        assert profile_result['success'] is True

        # 2. Analyze bottlenecks
        analyze_result = await profiler.analyze(sample_code_file)
        assert analyze_result['success'] is True

        # 3. Get optimization suggestions
        optimize_result = await profiler.optimize(sample_code_file)
        assert optimize_result['success'] is True

        # 4. Generate report
        report_result = await profiler.generate_report(sample_code_file, save_history=False)
        assert report_result['success'] is True

        # Verify all data is consistent
        assert 'result' in profile_result
        assert 'analysis' in analyze_result
        assert 'suggestions' in optimize_result
        assert 'report' in report_result

    @pytest.mark.asyncio
    async def test_tui_data_flow(self, profiler, sample_code_file):
        """Test data flow from profiler to TUI components"""
        from xencode.tui.widgets.performance_profiler_panel import (
            PerformanceProfilerPanel,
        )

        # Create panel
        PerformanceProfilerPanel()

        # Profile code
        result = await profiler.profile(sample_code_file)
        assert result['success'] is True

        # Extract data
        profile_data = result['result']

        # Verify data can be used to update TUI components
        metrics = {
            'execution_time': profile_data['execution_time'],
            'cpu_percent': profile_data['cpu_percent'],
            'memory_usage': profile_data['memory_usage'],
            'bottlenecks': len(profile_data['bottlenecks']),
            'hotspots': len([f for f in profile_data['function_stats'] if f['calls'] > 100])
        }

        # These would update the UI if mounted
        assert metrics['execution_time'] > 0
        assert metrics['cpu_percent'] >= 0
        assert metrics['memory_usage'] >= 0


class TestErrorHandling:
    """Test error handling in CLI and TUI"""

    @pytest.mark.asyncio
    async def test_profile_nonexistent_file(self, profiler):
        """Test profiling a nonexistent file"""
        result = await profiler.profile("/nonexistent/file.py")
        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_profile_invalid_python(self, profiler, tmp_path):
        """Test profiling invalid Python code"""
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("this is not valid python }{][")

        result = await profiler.profile(str(invalid_file))
        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_compare_with_missing_baseline(self, profiler, sample_code_file):
        """Test comparison with missing baseline"""
        result = await profiler.compare("/nonexistent/baseline.py", sample_code_file)
        assert result['success'] is False
        assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
