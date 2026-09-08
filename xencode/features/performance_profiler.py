"""
Performance Profiler Feature

Provides code performance analysis, bottleneck detection, and optimization suggestions.
"""

import cProfile
import io
import json
import pstats
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from xencode.features.base import FeatureBase, FeatureConfig


class ProfilerTool(Enum):
    """Available profiling tools"""
    CPROFILE = "cprofile"
    PSUTIL = "psutil"
    MEMORY_PROFILER = "memory_profiler"


@dataclass
class PerformanceProfilerConfig:
    """Configuration for Performance Profiler"""
    enabled: bool = True
    tools: List[str] = field(default_factory=lambda: ["cprofile", "psutil"])
    output_format: str = "text"
    sort_by: str = "cumulative"
    max_results: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceProfilerConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            tools=data.get('tools', ["cprofile", "psutil"]),
            output_format=data.get('output_format', 'text'),
            sort_by=data.get('sort_by', 'cumulative'),
            max_results=data.get('max_results', 50)
        )


@dataclass
class ProfileResult:
    """Result from a profiling run"""
    execution_time: float
    cpu_percent: float
    memory_usage: float
    function_stats: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'execution_time': self.execution_time,
            'cpu_percent': self.cpu_percent,
            'memory_usage': self.memory_usage,
            'function_stats': self.function_stats,
            'bottlenecks': self.bottlenecks
        }


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for code"""
    function_name: str
    issue: str
    suggestion: str
    example_before: str
    example_after: str
    estimated_improvement: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'function_name': self.function_name,
            'issue': self.issue,
            'suggestion': self.suggestion,
            'example_before': self.example_before,
            'example_after': self.example_after,
            'estimated_improvement': self.estimated_improvement
        }


@dataclass
class PerformanceComparison:
    """Comparison between two performance profiles"""
    baseline_path: str
    comparison_path: str
    execution_time_diff: float
    execution_time_percent: float
    cpu_diff: float
    memory_diff: float
    improved_functions: List[Dict[str, Any]]
    regressed_functions: List[Dict[str, Any]]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'baseline_path': self.baseline_path,
            'comparison_path': self.comparison_path,
            'execution_time_diff': self.execution_time_diff,
            'execution_time_percent': self.execution_time_percent,
            'cpu_diff': self.cpu_diff,
            'memory_diff': self.memory_diff,
            'improved_functions': self.improved_functions,
            'regressed_functions': self.regressed_functions,
            'summary': self.summary
        }


@dataclass
class PerformanceReport:
    """Performance report with trends and analysis"""
    path: str
    timestamp: str
    profile_result: ProfileResult
    analysis: Dict[str, Any]
    suggestions: List[OptimizationSuggestion]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'path': self.path,
            'timestamp': self.timestamp,
            'profile_result': self.profile_result.to_dict(),
            'analysis': self.analysis,
            'suggestions': [s.to_dict() for s in self.suggestions]
        }

    @dataclass
    class PerformanceComparison:
        """Comparison between two performance profiles"""
        baseline_path: str
        comparison_path: str
        execution_time_diff: float
        execution_time_percent: float
        cpu_diff: float
        memory_diff: float
        improved_functions: List[Dict[str, Any]]
        regressed_functions: List[Dict[str, Any]]
        summary: str

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary"""
            return {
                'baseline_path': self.baseline_path,
                'comparison_path': self.comparison_path,
                'execution_time_diff': self.execution_time_diff,
                'execution_time_percent': self.execution_time_percent,
                'cpu_diff': self.cpu_diff,
                'memory_diff': self.memory_diff,
                'improved_functions': self.improved_functions,
                'regressed_functions': self.regressed_functions,
                'summary': self.summary
            }


    @dataclass
    class PerformanceReport:
        """Performance report with trends and analysis"""
        path: str
        timestamp: str
        profile_result: ProfileResult
        analysis: Dict[str, Any]
        suggestions: List[OptimizationSuggestion]

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary"""
            return {
                'path': self.path,
                'timestamp': self.timestamp,
                'profile_result': self.profile_result.to_dict(),
                'analysis': self.analysis,
                'suggestions': [s.to_dict() for s in self.suggestions]
            }


class ProfilingEngine:
    """Core profiling engine using cProfile and psutil"""

    def __init__(self, config: PerformanceProfilerConfig):
        self.config = config
        self.profiler = None
        self.process = psutil.Process()

    async def profile_code(self, code_path: str, function_name: Optional[str] = None) -> ProfileResult:
        """Profile code execution"""
        # Read the code file
        code_file = Path(code_path)
        if not code_file.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")

        # Start profiling
        profiler = cProfile.Profile()

        # Get initial resource usage
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Profile the code
        profiler.enable()

        try:
            # Execute the code
            with open(code_file, 'r') as f:
                code = f.read()
                exec(compile(code, code_path, 'exec'))
        except Exception as e:
            profiler.disable()
            raise RuntimeError(f"Error executing code: {str(e)}")  from e

        profiler.disable()

        # Get final resource usage
        end_time = time.time()
        end_cpu = self.process.cpu_percent()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Extract statistics
        stats = pstats.Stats(profiler)
        stats.sort_stats(self.config.sort_by)

        # Convert stats to list of dicts
        function_stats = []
        for func, (_cc, nc, tt, ct, _callers) in stats.stats.items():
            filename, line, func_name = func
            function_stats.append({
                'function': func_name,
                'file': filename,
                'line': line,
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            })

        # Limit results
        function_stats = function_stats[:self.config.max_results]

        # Identify bottlenecks (top 10 slowest functions)
        bottlenecks = sorted(
            function_stats,
            key=lambda x: x['cumulative_time'],
            reverse=True
        )[:10]

        return ProfileResult(
            execution_time=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_usage=end_memory - start_memory,
            function_stats=function_stats,
            bottlenecks=bottlenecks
        )

    def get_stats_string(self, profile_result: ProfileResult) -> str:
        """Get formatted statistics string"""
        output = io.StringIO()
        output.write(f"Execution Time: {profile_result.execution_time:.4f}s\n")
        output.write(f"CPU Usage: {profile_result.cpu_percent:.2f}%\n")
        output.write(f"Memory Usage: {profile_result.memory_usage:.2f} MB\n\n")
        output.write("Top Functions by Cumulative Time:\n")
        output.write("-" * 80 + "\n")

        for stat in profile_result.function_stats[:20]:
            output.write(
                f"{stat['function']:<40} "
                f"{stat['calls']:>8} calls  "
                f"{stat['cumulative_time']:>10.4f}s\n"
            )

        return output.getvalue()


class BottleneckAnalyzer:
    """Analyzes profiling results to identify bottlenecks"""

    def __init__(self):
        pass

    async def analyze(self, profile_result: ProfileResult) -> Dict[str, Any]:
        """Analyze profile results for bottlenecks"""
        analysis = {
            'total_execution_time': profile_result.execution_time,
            'bottlenecks': [],
            'hotspots': [],
            'patterns': []
        }

        # Identify bottlenecks (functions taking >10% of total time)
        time_threshold = profile_result.execution_time * 0.1
        for func_stat in profile_result.function_stats:
            if func_stat['cumulative_time'] > time_threshold:
                percentage = 0.0
                if profile_result.execution_time > 0:
                    percentage = (func_stat['cumulative_time'] / profile_result.execution_time) * 100

                analysis['bottlenecks'].append({
                    'function': func_stat['function'],
                    'time': func_stat['cumulative_time'],
                    'percentage': percentage,
                    'calls': func_stat['calls']
                })

        # Identify hotspots (frequently called functions)
        call_threshold = 1000
        for func_stat in profile_result.function_stats:
            if func_stat['calls'] > call_threshold:
                analysis['hotspots'].append({
                    'function': func_stat['function'],
                    'calls': func_stat['calls'],
                    'time_per_call': func_stat['time_per_call']
                })

        # Identify patterns
        if profile_result.memory_usage > 100:  # >100MB
            analysis['patterns'].append({
                'type': 'high_memory',
                'description': 'High memory usage detected',
                'value': profile_result.memory_usage
            })

        if profile_result.cpu_percent > 80:
            analysis['patterns'].append({
                'type': 'high_cpu',
                'description': 'High CPU usage detected',
                'value': profile_result.cpu_percent
            })

        return analysis


class OptimizationEngine:
    """Generates optimization suggestions based on profiling results"""

    def __init__(self):
        self.optimization_patterns = self._init_patterns()

    def _init_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize optimization patterns"""
        return {
            'list_comprehension': {
                'issue': 'Using loops instead of list comprehensions',
                'suggestion': 'Replace loops with list comprehensions for better performance',
                'example_before': 'result = []\nfor i in range(100):\n    result.append(i * 2)',
                'example_after': 'result = [i * 2 for i in range(100)]',
                'improvement': '20-30% faster'
            },
            'string_concatenation': {
                'issue': 'Using + operator for string concatenation in loops',
                'suggestion': 'Use join() or f-strings for string concatenation',
                'example_before': 's = ""\nfor i in items:\n    s += str(i)',
                'example_after': 's = "".join(str(i) for i in items)',
                'improvement': '50-70% faster for large strings'
            },
            'dict_lookup': {
                'issue': 'Repeated dictionary lookups',
                'suggestion': 'Cache dictionary values in local variables',
                'example_before': 'for i in range(1000):\n    x = config["key"]',
                'example_after': 'key_value = config["key"]\nfor i in range(1000):\n    x = key_value',
                'improvement': '10-15% faster'
            },
            'function_calls': {
                'issue': 'Excessive function calls in tight loops',
                'suggestion': 'Move invariant function calls outside loops',
                'example_before': 'for i in items:\n    result = expensive_func() + i',
                'example_after': 'cached = expensive_func()\nfor i in items:\n    result = cached + i',
                'improvement': 'Depends on function cost'
            }
        }

    async def generate_suggestions(self, analysis: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions"""
        suggestions = []

        # Suggest optimizations for bottlenecks
        for bottleneck in analysis.get('bottlenecks', []):
            func_name = bottleneck['function']

            # Check for common patterns
            if 'loop' in func_name.lower() or 'iter' in func_name.lower():
                pattern = self.optimization_patterns['list_comprehension']
                suggestions.append(OptimizationSuggestion(
                    function_name=func_name,
                    issue=pattern['issue'],
                    suggestion=pattern['suggestion'],
                    example_before=pattern['example_before'],
                    example_after=pattern['example_after'],
                    estimated_improvement=pattern['improvement']
                ))

            if 'str' in func_name.lower() or 'concat' in func_name.lower():
                pattern = self.optimization_patterns['string_concatenation']
                suggestions.append(OptimizationSuggestion(
                    function_name=func_name,
                    issue=pattern['issue'],
                    suggestion=pattern['suggestion'],
                    example_before=pattern['example_before'],
                    example_after=pattern['example_after'],
                    estimated_improvement=pattern['improvement']
                ))

        # Suggest optimizations for hotspots
        for hotspot in analysis.get('hotspots', []):
            func_name = hotspot['function']
            pattern = self.optimization_patterns['function_calls']
            suggestions.append(OptimizationSuggestion(
                function_name=func_name,
                issue=pattern['issue'],
                suggestion=pattern['suggestion'],
                example_before=pattern['example_before'],
                example_after=pattern['example_after'],
                estimated_improvement=pattern['improvement']
            ))

        # Suggest optimizations for patterns
        for pattern_info in analysis.get('patterns', []):
            if pattern_info['type'] == 'high_memory':
                suggestions.append(OptimizationSuggestion(
                    function_name='<general>',
                    issue='High memory usage detected',
                    suggestion='Consider using generators, iterators, or processing data in chunks',
                    example_before='data = [process(x) for x in large_list]',
                    example_after='data = (process(x) for x in large_list)  # Generator',
                    estimated_improvement='Significant memory reduction'
                ))

        return suggestions


class PerformanceComparator:
    """Compares performance across code versions"""

    def __init__(self):
        self.history_dir = Path.home() / '.xencode' / 'performance_history'
        self.history_dir.mkdir(parents=True, exist_ok=True)

    async def compare(self, baseline: ProfileResult, comparison: ProfileResult,
                     baseline_path: str, comparison_path: str) -> PerformanceComparison:
        """Compare two performance profiles"""
        # Calculate execution time difference
        exec_time_diff = comparison.execution_time - baseline.execution_time
        exec_time_percent = 0.0
        if baseline.execution_time > 0:
            exec_time_percent = (exec_time_diff / baseline.execution_time) * 100

        # Calculate CPU and memory differences
        cpu_diff = comparison.cpu_percent - baseline.cpu_percent
        memory_diff = comparison.memory_usage - baseline.memory_usage

        # Compare function-level performance
        improved_functions = []
        regressed_functions = []

        # Create lookup for baseline functions
        baseline_funcs = {f['function']: f for f in baseline.function_stats}

        for comp_func in comparison.function_stats:
            func_name = comp_func['function']
            if func_name in baseline_funcs:
                base_func = baseline_funcs[func_name]
                time_diff = comp_func['cumulative_time'] - base_func['cumulative_time']

                if time_diff < -0.001:  # Improved (faster)
                    percent_change = 0.0
                    if base_func['cumulative_time'] > 0:
                        percent_change = (time_diff / base_func['cumulative_time']) * 100
                    improved_functions.append({
                        'function': func_name,
                        'baseline_time': base_func['cumulative_time'],
                        'comparison_time': comp_func['cumulative_time'],
                        'time_diff': time_diff,
                        'percent_change': percent_change
                    })
                elif time_diff > 0.001:  # Regressed (slower)
                    percent_change = 0.0
                    if base_func['cumulative_time'] > 0:
                        percent_change = (time_diff / base_func['cumulative_time']) * 100
                    regressed_functions.append({
                        'function': func_name,
                        'baseline_time': base_func['cumulative_time'],
                        'comparison_time': comp_func['cumulative_time'],
                        'time_diff': time_diff,
                        'percent_change': percent_change
                    })

        # Sort by absolute time difference
        improved_functions.sort(key=lambda x: abs(x['time_diff']), reverse=True)
        regressed_functions.sort(key=lambda x: abs(x['time_diff']), reverse=True)

        # Generate summary
        summary = self._generate_summary(
            exec_time_diff, exec_time_percent, cpu_diff, memory_diff,
            len(improved_functions), len(regressed_functions)
        )

        return PerformanceComparison(
            baseline_path=baseline_path,
            comparison_path=comparison_path,
            execution_time_diff=exec_time_diff,
            execution_time_percent=exec_time_percent,
            cpu_diff=cpu_diff,
            memory_diff=memory_diff,
            improved_functions=improved_functions[:10],  # Top 10
            regressed_functions=regressed_functions[:10],  # Top 10
            summary=summary
        )

    def _generate_summary(self, exec_time_diff: float, exec_time_percent: float,
                         cpu_diff: float, memory_diff: float,
                         improved_count: int, regressed_count: int) -> str:
        """Generate comparison summary"""
        lines = []

        # Overall performance
        if exec_time_diff < 0:
            lines.append(f"✓ Overall performance improved by {abs(exec_time_percent):.1f}% ({abs(exec_time_diff):.4f}s faster)")
        elif exec_time_diff > 0:
            lines.append(f"✗ Overall performance regressed by {exec_time_percent:.1f}% ({exec_time_diff:.4f}s slower)")
        else:
            lines.append("= Overall performance unchanged")

        # CPU usage
        if cpu_diff < 0:
            lines.append(f"✓ CPU usage decreased by {abs(cpu_diff):.1f}%")
        elif cpu_diff > 0:
            lines.append(f"✗ CPU usage increased by {cpu_diff:.1f}%")

        # Memory usage
        if memory_diff < 0:
            lines.append(f"✓ Memory usage decreased by {abs(memory_diff):.1f} MB")
        elif memory_diff > 0:
            lines.append(f"✗ Memory usage increased by {memory_diff:.1f} MB")

        # Function-level changes
        lines.append("\nFunction-level changes:")
        lines.append(f"  {improved_count} functions improved")
        lines.append(f"  {regressed_count} functions regressed")

        return "\n".join(lines)

    async def save_profile(self, path: str, profile_result: ProfileResult) -> str:
        """Save profile result to history"""
        timestamp = datetime.now().isoformat()
        filename = f"{Path(path).stem}_{timestamp.replace(':', '-')}.json"
        filepath = self.history_dir / filename

        data = {
            'path': path,
            'timestamp': timestamp,
            'profile': profile_result.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    async def load_profile(self, filepath: str) -> tuple[str, ProfileResult]:
        """Load profile result from history"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        profile_data = data['profile']
        profile_result = ProfileResult(
            execution_time=profile_data['execution_time'],
            cpu_percent=profile_data['cpu_percent'],
            memory_usage=profile_data['memory_usage'],
            function_stats=profile_data['function_stats'],
            bottlenecks=profile_data['bottlenecks']
        )

        return data['path'], profile_result

    async def list_profiles(self, path: Optional[str] = None) -> List[Dict[str, str]]:
        """List saved profiles"""
        profiles = []

        for filepath in self.history_dir.glob('*.json'):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Filter by path if specified
                if path and data['path'] != path:
                    continue

                profiles.append({
                    'filepath': str(filepath),
                    'path': data['path'],
                    'timestamp': data['timestamp']
                })
            except Exception:
                continue

        # Sort by timestamp (newest first)
        profiles.sort(key=lambda x: x['timestamp'], reverse=True)

        return profiles

    async def visualize_trends(self, path: str) -> Dict[str, Any]:
        """Visualize performance trends over time"""
        profiles = await self.list_profiles(path)

        if len(profiles) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 profiles to visualize trends'
            }

        # Load all profiles
        trend_data = []
        for profile_info in profiles:
            _, profile_result = await self.load_profile(profile_info['filepath'])
            trend_data.append({
                'timestamp': profile_info['timestamp'],
                'execution_time': profile_result.execution_time,
                'cpu_percent': profile_result.cpu_percent,
                'memory_usage': profile_result.memory_usage
            })

        # Sort by timestamp
        trend_data.sort(key=lambda x: x['timestamp'])

        # Calculate trends
        first = trend_data[0]
        last = trend_data[-1]

        exec_time_trend = 0.0
        if first['execution_time'] > 0:
            exec_time_trend = ((last['execution_time'] - first['execution_time']) / first['execution_time']) * 100

        cpu_trend = last['cpu_percent'] - first['cpu_percent']
        memory_trend = last['memory_usage'] - first['memory_usage']

        return {
            'success': True,
            'path': path,
            'profile_count': len(trend_data),
            'data': trend_data,
            'trends': {
                'execution_time_percent': exec_time_trend,
                'cpu_percent_change': cpu_trend,
                'memory_mb_change': memory_trend
            }
        }


class PerformanceProfiler(FeatureBase):
    """Performance Profiler feature implementation"""

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.profiler_config = PerformanceProfilerConfig.from_dict(config.config)
        self.profiling_engine = ProfilingEngine(self.profiler_config)
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.optimization_engine = OptimizationEngine()
        self.comparator = PerformanceComparator()

    @property
    def name(self) -> str:
        """Feature name"""
        return "performance_profiler"

    @property
    def description(self) -> str:
        """Feature description"""
        return "Code performance analysis, bottleneck detection, and optimization suggestions"

    async def _initialize(self) -> None:
        """Initialize the performance profiler"""
        # Verify required tools are available
        import importlib.util

        missing_tools = [
            tool
            for tool in ("cProfile", "psutil")
            if importlib.util.find_spec(tool) is None
        ]
        if missing_tools:
            raise RuntimeError(
                f"Required profiling tools not available: {', '.join(missing_tools)}"
            )

    async def _shutdown(self) -> None:
        """Shutdown the performance profiler"""
        pass

    async def profile(self, path: str, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Profile code execution"""
        self.track_analytics('profile', {'path': path})

        try:
            # Run profiling
            profile_result = await self.profiling_engine.profile_code(path, function_name)

            # Get formatted stats
            stats_string = self.profiling_engine.get_stats_string(profile_result)

            return {
                'success': True,
                'path': path,
                'result': profile_result.to_dict(),
                'stats': stats_string
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def analyze(self, path: str) -> Dict[str, Any]:
        """Analyze profiling results for bottlenecks"""
        self.track_analytics('analyze', {'path': path})

        try:
            # Profile the code first
            profile_result = await self.profiling_engine.profile_code(path)

            # Analyze for bottlenecks
            analysis = await self.bottleneck_analyzer.analyze(profile_result)

            return {
                'success': True,
                'path': path,
                'analysis': analysis
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def optimize(self, path: str) -> Dict[str, Any]:
        """Generate optimization suggestions"""
        self.track_analytics('optimize', {'path': path})

        try:
            # Profile and analyze
            profile_result = await self.profiling_engine.profile_code(path)
            analysis = await self.bottleneck_analyzer.analyze(profile_result)

            # Generate suggestions
            suggestions = await self.optimization_engine.generate_suggestions(analysis)

            return {
                'success': True,
                'path': path,
                'suggestions': [s.to_dict() for s in suggestions]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def compare(self, baseline_path: str, comparison_path: str) -> Dict[str, Any]:
        """Compare performance between two code versions"""
        self.track_analytics('compare', {'baseline': baseline_path, 'comparison': comparison_path})

        try:
            # Profile both versions
            baseline_result = await self.profiling_engine.profile_code(baseline_path)
            comparison_result = await self.profiling_engine.profile_code(comparison_path)

            # Compare results
            comparison = await self.comparator.compare(
                baseline_result, comparison_result,
                baseline_path, comparison_path
            )

            return {
                'success': True,
                'comparison': comparison.to_dict()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_report(self, path: str, save_history: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        self.track_analytics('generate_report', {'path': path})

        try:
            # Profile the code
            profile_result = await self.profiling_engine.profile_code(path)

            # Analyze for bottlenecks
            analysis = await self.bottleneck_analyzer.analyze(profile_result)

            # Generate suggestions
            suggestions = await self.optimization_engine.generate_suggestions(analysis)

            # Create report
            timestamp = datetime.now().isoformat()
            report = PerformanceReport(
                path=path,
                timestamp=timestamp,
                profile_result=profile_result,
                analysis=analysis,
                suggestions=suggestions
            )

            # Save to history if requested
            history_path = None
            if save_history:
                history_path = await self.comparator.save_profile(path, profile_result)

            return {
                'success': True,
                'report': report.to_dict(),
                'history_path': history_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def visualize_trends(self, path: str) -> Dict[str, Any]:
        """Visualize performance trends over time"""
        self.track_analytics('visualize_trends', {'path': path})

        try:
            trends = await self.comparator.visualize_trends(path)
            return trends
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def list_history(self, path: Optional[str] = None) -> Dict[str, Any]:
        """List saved performance profiles"""
        self.track_analytics('list_history', {'path': path})

        try:
            profiles = await self.comparator.list_profiles(path)
            return {
                'success': True,
                'profiles': profiles
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for this feature"""
        import click

        @click.group(name='profile')
        def profile_group():
            """Performance profiling commands"""
            pass

        @profile_group.command(name='run')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--function', '-f', help='Specific function to profile')
        def profile_run(path: str, function: Optional[str]):
            """Run performance profile on code"""
            import asyncio
            result = asyncio.run(self.profile(path, function))

            if result['success']:
                click.echo(result['stats'])
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='analyze')
        @click.argument('path', type=click.Path(exists=True))
        def profile_analyze(path: str):
            """Analyze profile results for bottlenecks"""
            import asyncio
            result = asyncio.run(self.analyze(path))

            if result['success']:
                analysis = result['analysis']
                click.echo(f"\nBottlenecks found: {len(analysis['bottlenecks'])}")
                for b in analysis['bottlenecks']:
                    click.echo(f"  - {b['function']}: {b['time']:.4f}s ({b['percentage']:.1f}%)")

                click.echo(f"\nHotspots found: {len(analysis['hotspots'])}")
                for h in analysis['hotspots']:
                    click.echo(f"  - {h['function']}: {h['calls']} calls")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='optimize')
        @click.argument('path', type=click.Path(exists=True))
        def profile_optimize(path: str):
            """Get optimization suggestions"""
            import asyncio
            result = asyncio.run(self.optimize(path))

            if result['success']:
                suggestions = result['suggestions']
                click.echo(f"\nOptimization suggestions: {len(suggestions)}")
                for i, s in enumerate(suggestions, 1):
                    click.echo(f"\n{i}. {s['function_name']}")
                    click.echo(f"   Issue: {s['issue']}")
                    click.echo(f"   Suggestion: {s['suggestion']}")
                    click.echo(f"   Estimated improvement: {s['estimated_improvement']}")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='compare')
        @click.argument('baseline', type=click.Path(exists=True))
        @click.argument('comparison', type=click.Path(exists=True))
        def profile_compare(baseline: str, comparison: str):
            """Compare performance between two code versions"""
            import asyncio
            result = asyncio.run(self.compare(baseline, comparison))

            if result['success']:
                comp = result['comparison']
                click.echo("\n" + "=" * 80)
                click.echo("PERFORMANCE COMPARISON")
                click.echo("=" * 80)
                click.echo(f"\nBaseline: {comp['baseline_path']}")
                click.echo(f"Comparison: {comp['comparison_path']}")
                click.echo("\n" + comp['summary'])

                if comp['improved_functions']:
                    click.echo("\n" + "-" * 80)
                    click.echo("TOP IMPROVED FUNCTIONS:")
                    click.echo("-" * 80)
                    for func in comp['improved_functions'][:5]:
                        click.echo(f"  {func['function']}: {func['percent_change']:.1f}% faster ({abs(func['time_diff']):.4f}s)")

                if comp['regressed_functions']:
                    click.echo("\n" + "-" * 80)
                    click.echo("TOP REGRESSED FUNCTIONS:")
                    click.echo("-" * 80)
                    for func in comp['regressed_functions'][:5]:
                        click.echo(f"  {func['function']}: {func['percent_change']:.1f}% slower (+{func['time_diff']:.4f}s)")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='report')
        @click.argument('path', type=click.Path(exists=True))
        @click.option('--save-history/--no-save-history', default=True, help='Save to history')
        def profile_report(path: str, save_history: bool):
            """Generate comprehensive performance report"""
            import asyncio
            result = asyncio.run(self.generate_report(path, save_history))

            if result['success']:
                report = result['report']
                click.echo("\n" + "=" * 80)
                click.echo("PERFORMANCE REPORT")
                click.echo("=" * 80)
                click.echo(f"\nPath: {report['path']}")
                click.echo(f"Timestamp: {report['timestamp']}")

                profile = report['profile_result']
                click.echo(f"\nExecution Time: {profile['execution_time']:.4f}s")
                click.echo(f"CPU Usage: {profile['cpu_percent']:.2f}%")
                click.echo(f"Memory Usage: {profile['memory_usage']:.2f} MB")

                analysis = report['analysis']
                click.echo(f"\nBottlenecks: {len(analysis['bottlenecks'])}")
                click.echo(f"Hotspots: {len(analysis['hotspots'])}")

                suggestions = report['suggestions']
                click.echo(f"\nOptimization Suggestions: {len(suggestions)}")

                if save_history and result.get('history_path'):
                    click.echo(f"\nSaved to history: {result['history_path']}")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='trends')
        @click.argument('path', type=click.Path(exists=True))
        def profile_trends(path: str):
            """Visualize performance trends over time"""
            import asyncio
            result = asyncio.run(self.visualize_trends(path))

            if result['success']:
                click.echo("\n" + "=" * 80)
                click.echo("PERFORMANCE TRENDS")
                click.echo("=" * 80)
                click.echo(f"\nPath: {result['path']}")
                click.echo(f"Profiles analyzed: {result['profile_count']}")

                trends = result['trends']
                click.echo("\nOverall trends:")

                exec_trend = trends['execution_time_percent']
                if exec_trend < 0:
                    click.echo(f"  ✓ Execution time: {abs(exec_trend):.1f}% faster")
                elif exec_trend > 0:
                    click.echo(f"  ✗ Execution time: {exec_trend:.1f}% slower")
                else:
                    click.echo("  = Execution time: unchanged")

                cpu_trend = trends['cpu_percent_change']
                if cpu_trend < 0:
                    click.echo(f"  ✓ CPU usage: {abs(cpu_trend):.1f}% lower")
                elif cpu_trend > 0:
                    click.echo(f"  ✗ CPU usage: {cpu_trend:.1f}% higher")

                mem_trend = trends['memory_mb_change']
                if mem_trend < 0:
                    click.echo(f"  ✓ Memory usage: {abs(mem_trend):.1f} MB lower")
                elif mem_trend > 0:
                    click.echo(f"  ✗ Memory usage: {mem_trend:.1f} MB higher")

                click.echo("\nData points:")
                for i, data in enumerate(result['data'], 1):
                    click.echo(f"  {i}. {data['timestamp']}: {data['execution_time']:.4f}s")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        @profile_group.command(name='history')
        @click.option('--path', type=click.Path(exists=True), help='Filter by path')
        def profile_history(path: Optional[str]):
            """List saved performance profiles"""
            import asyncio
            result = asyncio.run(self.list_history(path))

            if result['success']:
                profiles = result['profiles']
                click.echo(f"\nSaved profiles: {len(profiles)}")
                for profile in profiles:
                    click.echo(f"  {profile['timestamp']}: {profile['path']}")
                    click.echo(f"    File: {profile['filepath']}")
            else:
                click.echo(f"Error: {result['error']}", err=True)

        return [profile_group]

    def get_tui_components(self) -> List[Any]:
        """Get TUI components for this feature"""
        from xencode.tui.widgets.performance_profiler_panel import (
            PerformanceProfilerPanel,
        )
        return [PerformanceProfilerPanel]

    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for this feature"""
        return [
            {
                'path': '/api/profile/run',
                'method': 'POST',
                'handler': self.profile,
            },
            {
                'path': '/api/profile/analyze',
                'method': 'POST',
                'handler': self.analyze,
            },
            {
                'path': '/api/profile/optimize',
                'method': 'POST',
                'handler': self.optimize,
            },
            {
                'path': '/api/profile/compare',
                'method': 'POST',
                'handler': self.compare,
            },
            {
                'path': '/api/profile/report',
                'method': 'POST',
                'handler': self.generate_report,
            },
            {
                'path': '/api/profile/trends',
                'method': 'GET',
                'handler': self.visualize_trends,
            },
            {
                'path': '/api/profile/history',
                'method': 'GET',
                'handler': self.list_history,
            },
        ]
