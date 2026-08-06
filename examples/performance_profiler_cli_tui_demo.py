#!/usr/bin/env python3
"""
Performance Profiler CLI and TUI Demo

This script demonstrates the CLI commands and TUI interface for the Performance Profiler feature.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xencode.features.base import FeatureConfig
from xencode.features.performance_profiler import PerformanceProfiler


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def demo_cli_commands():
    """Demonstrate CLI command functionality"""
    print_section("Performance Profiler CLI Demo")

    # Create profiler instance
    config = FeatureConfig(
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

    profiler = PerformanceProfiler(config)
    await profiler.initialize()

    # Create a sample Python file to profile
    sample_code = """
import time

def slow_function():
    '''A deliberately slow function'''
    time.sleep(0.1)
    result = sum(range(1000000))
    return result

def fast_function():
    '''A fast function'''
    return 42

def main():
    '''Main function'''
    for i in range(5):
        slow_function()

    for i in range(100):
        fast_function()

    return "Done"

if __name__ == "__main__":
    main()
"""

    # Write sample code to temp file
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    sample_file = temp_dir / "sample_profile.py"
    sample_file.write_text(sample_code)

    print("Sample code created at:", sample_file)
    print("\nSample code:")
    print("-" * 80)
    print(sample_code)
    print("-" * 80)

    # Demo 1: Profile code
    print_section("1. Profile Code Execution")
    print(f"Command: xencode profile run {sample_file}")
    print("\nRunning profile...\n")

    result = await profiler.profile(str(sample_file))
    if result['success']:
        print(result['stats'])
    else:
        print(f"Error: {result['error']}")

    # Demo 2: Analyze for bottlenecks
    print_section("2. Analyze Performance Bottlenecks")
    print(f"Command: xencode profile analyze {sample_file}")
    print("\nAnalyzing bottlenecks...\n")

    result = await profiler.analyze(str(sample_file))
    if result['success']:
        analysis = result['analysis']
        print(f"Total execution time: {analysis['total_execution_time']:.4f}s")
        print(f"\nBottlenecks found: {len(analysis['bottlenecks'])}")
        for b in analysis['bottlenecks']:
            print(f"  - {b['function']}: {b['time']:.4f}s ({b['percentage']:.1f}%)")

        print(f"\nHotspots found: {len(analysis['hotspots'])}")
        for h in analysis['hotspots']:
            print(f"  - {h['function']}: {h['calls']} calls")

        print(f"\nPatterns detected: {len(analysis['patterns'])}")
        for p in analysis['patterns']:
            print(f"  - {p['type']}: {p['description']}")
    else:
        print(f"Error: {result['error']}")

    # Demo 3: Get optimization suggestions
    print_section("3. Get Optimization Suggestions")
    print(f"Command: xencode profile optimize {sample_file}")
    print("\nGenerating suggestions...\n")

    result = await profiler.optimize(str(sample_file))
    if result['success']:
        suggestions = result['suggestions']
        print(f"Optimization suggestions: {len(suggestions)}")
        for i, s in enumerate(suggestions, 1):
            print(f"\n{i}. {s['function_name']}")
            print(f"   Issue: {s['issue']}")
            print(f"   Suggestion: {s['suggestion']}")
            print(f"   Estimated improvement: {s['estimated_improvement']}")
            print("\n   Before:")
            for line in s['example_before'].split('\n'):
                print(f"     {line}")
            print("\n   After:")
            for line in s['example_after'].split('\n'):
                print(f"     {line}")
    else:
        print(f"Error: {result['error']}")

    # Demo 4: Generate comprehensive report
    print_section("4. Generate Performance Report")
    print(f"Command: xencode profile report {sample_file}")
    print("\nGenerating report...\n")

    result = await profiler.generate_report(str(sample_file), save_history=True)
    if result['success']:
        report = result['report']
        print(f"Path: {report['path']}")
        print(f"Timestamp: {report['timestamp']}")

        profile = report['profile_result']
        print(f"\nExecution Time: {profile['execution_time']:.4f}s")
        print(f"CPU Usage: {profile['cpu_percent']:.2f}%")
        print(f"Memory Usage: {profile['memory_usage']:.2f} MB")

        analysis = report['analysis']
        print(f"\nBottlenecks: {len(analysis['bottlenecks'])}")
        print(f"Hotspots: {len(analysis['hotspots'])}")

        suggestions = report['suggestions']
        print(f"\nOptimization Suggestions: {len(suggestions)}")

        if result.get('history_path'):
            print(f"\nSaved to history: {result['history_path']}")
    else:
        print(f"Error: {result['error']}")

    # Demo 5: List profile history
    print_section("5. List Profile History")
    print("Command: xencode profile history")
    print("\nListing history...\n")

    result = await profiler.list_history()
    if result['success']:
        profiles = result['profiles']
        print(f"Saved profiles: {len(profiles)}")
        for profile in profiles[:5]:  # Show last 5
            print(f"  {profile['timestamp']}: {profile['path']}")
            print(f"    File: {profile['filepath']}")
    else:
        print(f"Error: {result['error']}")

    await profiler.shutdown()

    print_section("CLI Demo Complete")
    print("Available CLI commands:")
    print("  xencode profile run <path>           - Run performance profile")
    print("  xencode profile analyze <path>       - Analyze bottlenecks")
    print("  xencode profile optimize <path>      - Get optimization suggestions")
    print("  xencode profile compare <p1> <p2>    - Compare two profiles")
    print("  xencode profile report <path>        - Generate comprehensive report")
    print("  xencode profile trends <path>        - Visualize performance trends")
    print("  xencode profile history              - List saved profiles")


def demo_tui_components():
    """Demonstrate TUI component structure"""
    print_section("Performance Profiler TUI Demo")

    print("TUI Components Available:")
    print("\n1. MetricsDashboard")
    print("   - Displays execution time, CPU usage, memory usage")
    print("   - Shows bottleneck and hotspot counts")
    print("   - Color-coded performance indicators")

    print("\n2. BottleneckVisualization")
    print("   - Table view of performance bottlenecks")
    print("   - Shows function name, time, percentage, calls")
    print("   - Sortable columns for analysis")

    print("\n3. ProfileViewer")
    print("   - Detailed function statistics")
    print("   - Shows all profiled functions with timing data")
    print("   - Displays file locations and call counts")

    print("\n4. OptimizationSuggestions")
    print("   - Lists optimization recommendations")
    print("   - Shows before/after code examples")
    print("   - Displays estimated performance improvements")

    print("\n5. ComparisonView")
    print("   - Before/after performance comparison")
    print("   - Shows improved and regressed functions")
    print("   - Displays percentage changes and trends")

    print("\n6. PerformanceProfilerPanel (Main Panel)")
    print("   - Integrates all components")
    print("   - Tab-based navigation")
    print("   - Control buttons for actions")
    print("   - Keyboard shortcuts:")
    print("     - Ctrl+P: Profile code")
    print("     - Ctrl+A: Analyze performance")
    print("     - Ctrl+O: Get optimization suggestions")
    print("     - Ctrl+C: Compare profiles")
    print("     - 1-5: Switch between tabs")

    print("\nTUI Features:")
    print("  ✓ Interactive dashboard with real-time metrics")
    print("  ✓ Visual bottleneck identification")
    print("  ✓ Detailed function profiling data")
    print("  ✓ Actionable optimization suggestions")
    print("  ✓ Before/after comparison visualization")
    print("  ✓ Keyboard shortcuts for quick navigation")
    print("  ✓ Color-coded performance indicators")
    print("  ✓ Responsive layout with scrollable content")

    print("\nTo use the TUI:")
    print("  1. Import PerformanceProfilerPanel from xencode.tui.widgets.performance_profiler_panel")
    print("  2. Add it to your Textual app")
    print("  3. Handle ProfilerAction messages for user interactions")
    print("  4. Update components with profiling results")

    print("\nExample integration:")
    print("""
    from xencode.tui.widgets.performance_profiler_panel import PerformanceProfilerPanel

    class MyApp(App):
        def compose(self):
            yield PerformanceProfilerPanel()

        def on_performance_profiler_panel_profiler_action(self, message):
            if message.action == "profile":
                # Run profiling
                result = await profiler.profile(message.path)
                # Update dashboard
                self.query_one(PerformanceProfilerPanel).update_dashboard(result)
    """)


def main():
    """Main demo function"""
    print("\n" + "=" * 80)
    print("  PERFORMANCE PROFILER CLI AND TUI DEMO")
    print("=" * 80)

    # Run CLI demo
    asyncio.run(demo_cli_commands())

    # Show TUI demo
    demo_tui_components()

    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - xencode/features/performance_profiler.py")
    print("  - xencode/tui/widgets/performance_profiler_panel.py")
    print("  - xencode/features/performance_profiler/CLI_TUI_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
