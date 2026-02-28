"""
Tests for Performance Profiler Feature
"""

import pytest
import asyncio
from pathlib import Path
from xencode.features.performance_profiler import (
    PerformanceProfiler,
    PerformanceProfilerConfig,
    ProfilingEngine,
    BottleneckAnalyzer,
    OptimizationEngine,
    ProfileResult,
    FeatureConfig
)


@pytest.fixture
def profiler_config():
    """Create a test profiler configuration"""
    return PerformanceProfilerConfig(
        enabled=True,
        tools=["cprofile", "psutil"],
        output_format="text",
        sort_by="cumulative",
        max_results=50
    )


@pytest.fixture
def feature_config():
    """Create a test feature configuration"""
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
def sample_profile_result():
    """Create a sample profile result for testing"""
    return ProfileResult(
        execution_time=1.0,
        cpu_percent=50.0,
        memory_usage=150.0,
        function_stats=[
            {
                'function': 'slow_function',
                'file': 'test.py',
                'line': 10,
                'calls': 100,
                'total_time': 0.5,
                'cumulative_time': 0.5,
                'time_per_call': 0.005
            },
            {
                'function': 'fast_function',
                'file': 'test.py',
                'line': 20,
                'calls': 2000,
                'total_time': 0.1,
                'cumulative_time': 0.1,
                'time_per_call': 0.00005
            },
            {
                'function': 'medium_function',
                'file': 'test.py',
                'line': 30,
                'calls': 50,
                'total_time': 0.2,
                'cumulative_time': 0.2,
                'time_per_call': 0.004
            }
        ],
        bottlenecks=[
            {
                'function': 'slow_function',
                'file': 'test.py',
                'line': 10,
                'calls': 100,
                'total_time': 0.5,
                'cumulative_time': 0.5,
                'time_per_call': 0.005
            }
        ]
    )


class TestBottleneckAnalyzer:
    """Tests for BottleneckAnalyzer class"""
    
    @pytest.mark.asyncio
    async def test_analyze_identifies_bottlenecks(self, sample_profile_result):
        """Test that analyzer identifies bottlenecks (functions taking >10% of time)"""
        analyzer = BottleneckAnalyzer()
        analysis = await analyzer.analyze(sample_profile_result)
        
        # Should identify slow_function as bottleneck (0.5s / 1.0s = 50%)
        assert 'bottlenecks' in analysis
        assert len(analysis['bottlenecks']) > 0
        
        bottleneck_functions = [b['function'] for b in analysis['bottlenecks']]
        assert 'slow_function' in bottleneck_functions
        
        # Verify bottleneck details
        slow_bottleneck = next(b for b in analysis['bottlenecks'] if b['function'] == 'slow_function')
        assert slow_bottleneck['time'] == 0.5
        assert slow_bottleneck['percentage'] == 50.0
        assert slow_bottleneck['calls'] == 100
    
    @pytest.mark.asyncio
    async def test_analyze_detects_hotspots(self, sample_profile_result):
        """Test that analyzer detects hotspots (frequently called functions)"""
        analyzer = BottleneckAnalyzer()
        analysis = await analyzer.analyze(sample_profile_result)
        
        # Should identify fast_function as hotspot (2000 calls > 1000 threshold)
        assert 'hotspots' in analysis
        assert len(analysis['hotspots']) > 0
        
        hotspot_functions = [h['function'] for h in analysis['hotspots']]
        assert 'fast_function' in hotspot_functions
        
        # Verify hotspot details
        fast_hotspot = next(h for h in analysis['hotspots'] if h['function'] == 'fast_function')
        assert fast_hotspot['calls'] == 2000
        assert fast_hotspot['time_per_call'] == 0.00005
    
    @pytest.mark.asyncio
    async def test_analyze_memory_patterns(self, sample_profile_result):
        """Test that analyzer detects high memory usage patterns"""
        analyzer = BottleneckAnalyzer()
        analysis = await analyzer.analyze(sample_profile_result)
        
        # Should detect high memory usage (150MB > 100MB threshold)
        assert 'patterns' in analysis
        memory_patterns = [p for p in analysis['patterns'] if p['type'] == 'high_memory']
        assert len(memory_patterns) > 0
        
        memory_pattern = memory_patterns[0]
        assert memory_pattern['description'] == 'High memory usage detected'
        assert memory_pattern['value'] == 150.0
    
    @pytest.mark.asyncio
    async def test_analyze_cpu_patterns(self):
        """Test that analyzer detects high CPU usage patterns"""
        analyzer = BottleneckAnalyzer()
        
        # Create profile result with high CPU usage
        high_cpu_result = ProfileResult(
            execution_time=1.0,
            cpu_percent=85.0,  # >80% threshold
            memory_usage=50.0,
            function_stats=[],
            bottlenecks=[]
        )
        
        analysis = await analyzer.analyze(high_cpu_result)
        
        # Should detect high CPU usage
        assert 'patterns' in analysis
        cpu_patterns = [p for p in analysis['patterns'] if p['type'] == 'high_cpu']
        assert len(cpu_patterns) > 0
        
        cpu_pattern = cpu_patterns[0]
        assert cpu_pattern['description'] == 'High CPU usage detected'
        assert cpu_pattern['value'] == 85.0
    
    @pytest.mark.asyncio
    async def test_analyze_no_bottlenecks(self):
        """Test analyzer when no bottlenecks are present"""
        analyzer = BottleneckAnalyzer()
        
        # Create profile result with all fast functions
        fast_result = ProfileResult(
            execution_time=1.0,
            cpu_percent=20.0,
            memory_usage=50.0,
            function_stats=[
                {
                    'function': 'fast1',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 0.05,
                    'cumulative_time': 0.05,
                    'time_per_call': 0.0005
                },
                {
                    'function': 'fast2',
                    'file': 'test.py',
                    'line': 20,
                    'calls': 100,
                    'total_time': 0.05,
                    'cumulative_time': 0.05,
                    'time_per_call': 0.0005
                }
            ],
            bottlenecks=[]
        )
        
        analysis = await analyzer.analyze(fast_result)
        
        # Should have no bottlenecks (all functions < 10% threshold)
        assert len(analysis['bottlenecks']) == 0
        assert len(analysis['hotspots']) == 0
        assert len(analysis['patterns']) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_total_execution_time(self, sample_profile_result):
        """Test that analyzer includes total execution time"""
        analyzer = BottleneckAnalyzer()
        analysis = await analyzer.analyze(sample_profile_result)
        
        assert 'total_execution_time' in analysis
        assert analysis['total_execution_time'] == 1.0


class TestOptimizationEngine:
    """Tests for OptimizationEngine class"""
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_for_bottlenecks(self):
        """Test that optimization engine generates suggestions for bottlenecks"""
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [
                {
                    'function': 'loop_function',
                    'time': 0.5,
                    'percentage': 50.0,
                    'calls': 100
                }
            ],
            'hotspots': [],
            'patterns': []
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        assert len(suggestions) > 0
        assert any('loop' in s.function_name.lower() for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_for_hotspots(self):
        """Test that optimization engine generates suggestions for hotspots"""
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [],
            'hotspots': [
                {
                    'function': 'frequently_called',
                    'calls': 5000,
                    'time_per_call': 0.001
                }
            ],
            'patterns': []
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        assert len(suggestions) > 0
        assert any('frequently_called' in s.function_name for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_suggestions_for_memory_patterns(self):
        """Test that optimization engine generates suggestions for memory patterns"""
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [],
            'hotspots': [],
            'patterns': [
                {
                    'type': 'high_memory',
                    'description': 'High memory usage detected',
                    'value': 200.0
                }
            ]
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        assert len(suggestions) > 0
        memory_suggestions = [s for s in suggestions if 'memory' in s.issue.lower()]
        assert len(memory_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_suggestions_include_before_after_examples(self):
        """
        **Validates: Requirements 9.3**
        Test that optimization suggestions include before/after code examples
        """
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [
                {
                    'function': 'loop_function',
                    'time': 0.5,
                    'percentage': 50.0,
                    'calls': 100
                }
            ],
            'hotspots': [],
            'patterns': []
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        # Verify all suggestions have before/after examples
        for suggestion in suggestions:
            assert hasattr(suggestion, 'example_before'), "Suggestion missing 'example_before'"
            assert hasattr(suggestion, 'example_after'), "Suggestion missing 'example_after'"
            assert suggestion.example_before, "example_before is empty"
            assert suggestion.example_after, "example_after is empty"
            assert suggestion.example_before != suggestion.example_after, "Before and after examples should be different"
    
    @pytest.mark.asyncio
    async def test_suggestions_include_performance_estimates(self):
        """
        **Validates: Requirements 9.3**
        Test that optimization suggestions include performance improvement estimates
        """
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [
                {
                    'function': 'string_concat_loop',
                    'time': 0.8,
                    'percentage': 80.0,
                    'calls': 50
                }
            ],
            'hotspots': [],
            'patterns': []
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        # Verify all suggestions have performance estimates
        for suggestion in suggestions:
            assert hasattr(suggestion, 'estimated_improvement'), "Suggestion missing 'estimated_improvement'"
            assert suggestion.estimated_improvement, "estimated_improvement is empty"
            # Verify it contains some indication of improvement (percentage, time, or description)
            assert any(keyword in suggestion.estimated_improvement.lower() 
                      for keyword in ['faster', '%', 'reduction', 'improvement', 'depends']), \
                   f"estimated_improvement should describe performance gain: {suggestion.estimated_improvement}"
    
    @pytest.mark.asyncio
    async def test_suggestions_provide_actionable_recommendations(self):
        """
        **Validates: Requirements 9.3**
        Test that optimization suggestions provide clear, actionable recommendations
        """
        engine = OptimizationEngine()
        
        analysis = {
            'bottlenecks': [
                {
                    'function': 'loop_function',
                    'time': 0.5,
                    'percentage': 50.0,
                    'calls': 100
                }
            ],
            'hotspots': [
                {
                    'function': 'frequently_called',
                    'calls': 5000,
                    'time_per_call': 0.001
                }
            ],
            'patterns': [
                {
                    'type': 'high_memory',
                    'description': 'High memory usage detected',
                    'value': 200.0
                }
            ]
        }
        
        suggestions = await engine.generate_suggestions(analysis)
        
        # Verify all suggestions have required fields
        for suggestion in suggestions:
            assert hasattr(suggestion, 'function_name'), "Suggestion missing 'function_name'"
            assert hasattr(suggestion, 'issue'), "Suggestion missing 'issue'"
            assert hasattr(suggestion, 'suggestion'), "Suggestion missing 'suggestion'"
            
            assert suggestion.function_name, "function_name is empty"
            assert suggestion.issue, "issue is empty"
            assert suggestion.suggestion, "suggestion is empty"
            
            # Verify suggestion is actionable (contains verbs or instructions)
            actionable_keywords = ['use', 'replace', 'move', 'cache', 'consider', 'avoid', 'implement']
            assert any(keyword in suggestion.suggestion.lower() for keyword in actionable_keywords), \
                   f"Suggestion should be actionable: {suggestion.suggestion}"
    
    @pytest.mark.asyncio
    async def test_optimization_patterns_coverage(self):
        """
        **Validates: Requirements 9.3**
        Test that optimization engine has patterns for common performance issues
        """
        engine = OptimizationEngine()
        
        # Verify optimization patterns are initialized
        assert hasattr(engine, 'optimization_patterns'), "Engine missing optimization_patterns"
        patterns = engine.optimization_patterns
        
        # Verify common optimization patterns exist
        expected_patterns = ['list_comprehension', 'string_concatenation', 'dict_lookup', 'function_calls']
        for pattern_name in expected_patterns:
            assert pattern_name in patterns, f"Missing optimization pattern: {pattern_name}"
            
            pattern = patterns[pattern_name]
            assert 'issue' in pattern, f"Pattern {pattern_name} missing 'issue'"
            assert 'suggestion' in pattern, f"Pattern {pattern_name} missing 'suggestion'"
            assert 'example_before' in pattern, f"Pattern {pattern_name} missing 'example_before'"
            assert 'example_after' in pattern, f"Pattern {pattern_name} missing 'example_after'"
            assert 'improvement' in pattern, f"Pattern {pattern_name} missing 'improvement'"


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class"""
    
    @pytest.mark.asyncio
    async def test_profiler_initialization(self, feature_config):
        """Test that profiler initializes correctly"""
        profiler = PerformanceProfiler(feature_config)
        
        assert profiler.name == "performance_profiler"
        assert "performance analysis" in profiler.description.lower()
        assert profiler.bottleneck_analyzer is not None
        assert profiler.optimization_engine is not None
    
    @pytest.mark.asyncio
    async def test_analyze_method(self, feature_config, tmp_path):
        """Test the analyze method"""
        profiler = PerformanceProfiler(feature_config)
        await profiler.initialize()
        
        # Create a simple test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

slow_function()
""")
        
        result = await profiler.analyze(str(test_file))
        
        assert result['success'] is True
        assert 'analysis' in result
        assert 'bottlenecks' in result['analysis']
        assert 'hotspots' in result['analysis']
        assert 'patterns' in result['analysis']
    
    @pytest.mark.asyncio
    async def test_optimize_method(self, feature_config, tmp_path):
        """Test the optimize method"""
        profiler = PerformanceProfiler(feature_config)
        await profiler.initialize()
        
        # Create a simple test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
def loop_function():
    result = []
    for i in range(1000):
        result.append(i * 2)
    return result

loop_function()
""")
        
        result = await profiler.optimize(str(test_file))
        
        assert result['success'] is True
        assert 'suggestions' in result
        assert isinstance(result['suggestions'], list)


class TestPerformanceComparator:
    """Tests for PerformanceComparator class"""
    
    @pytest.mark.asyncio
    async def test_compare_improved_performance(self):
        """
        **Validates: Requirements 9.5**
        Test comparing two profiles where performance improved
        """
        from xencode.features.performance_profiler import PerformanceComparator
        
        comparator = PerformanceComparator()
        
        # Create baseline profile (slower)
        baseline = ProfileResult(
            execution_time=2.0,
            cpu_percent=60.0,
            memory_usage=200.0,
            function_stats=[
                {
                    'function': 'slow_func',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 1.0,
                    'cumulative_time': 1.0,
                    'time_per_call': 0.01
                }
            ],
            bottlenecks=[]
        )
        
        # Create comparison profile (faster)
        comparison = ProfileResult(
            execution_time=1.0,
            cpu_percent=40.0,
            memory_usage=150.0,
            function_stats=[
                {
                    'function': 'slow_func',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 0.5,
                    'cumulative_time': 0.5,
                    'time_per_call': 0.005
                }
            ],
            bottlenecks=[]
        )
        
        result = await comparator.compare(baseline, comparison, 'baseline.py', 'optimized.py')
        
        # Verify comparison results
        assert result.execution_time_diff < 0, "Execution time should be faster"
        assert result.execution_time_percent < 0, "Execution time percent should be negative (improvement)"
        assert result.cpu_diff < 0, "CPU usage should be lower"
        assert result.memory_diff < 0, "Memory usage should be lower"
        assert len(result.improved_functions) > 0, "Should have improved functions"
        assert "improved" in result.summary.lower(), "Summary should mention improvement"
    
    @pytest.mark.asyncio
    async def test_compare_regressed_performance(self):
        """
        **Validates: Requirements 9.5**
        Test comparing two profiles where performance regressed
        """
        from xencode.features.performance_profiler import PerformanceComparator
        
        comparator = PerformanceComparator()
        
        # Create baseline profile (faster)
        baseline = ProfileResult(
            execution_time=1.0,
            cpu_percent=40.0,
            memory_usage=150.0,
            function_stats=[
                {
                    'function': 'func',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 0.5,
                    'cumulative_time': 0.5,
                    'time_per_call': 0.005
                }
            ],
            bottlenecks=[]
        )
        
        # Create comparison profile (slower)
        comparison = ProfileResult(
            execution_time=2.0,
            cpu_percent=60.0,
            memory_usage=200.0,
            function_stats=[
                {
                    'function': 'func',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 1.0,
                    'cumulative_time': 1.0,
                    'time_per_call': 0.01
                }
            ],
            bottlenecks=[]
        )
        
        result = await comparator.compare(baseline, comparison, 'baseline.py', 'regressed.py')
        
        # Verify comparison results
        assert result.execution_time_diff > 0, "Execution time should be slower"
        assert result.execution_time_percent > 0, "Execution time percent should be positive (regression)"
        assert result.cpu_diff > 0, "CPU usage should be higher"
        assert result.memory_diff > 0, "Memory usage should be higher"
        assert len(result.regressed_functions) > 0, "Should have regressed functions"
        assert "regressed" in result.summary.lower(), "Summary should mention regression"
    
    @pytest.mark.asyncio
    async def test_save_and_load_profile(self, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test saving and loading performance profiles
        """
        from xencode.features.performance_profiler import PerformanceComparator
        
        comparator = PerformanceComparator()
        # Override history directory for testing
        comparator.history_dir = tmp_path
        
        # Create a profile
        profile = ProfileResult(
            execution_time=1.5,
            cpu_percent=50.0,
            memory_usage=175.0,
            function_stats=[
                {
                    'function': 'test_func',
                    'file': 'test.py',
                    'line': 10,
                    'calls': 100,
                    'total_time': 0.8,
                    'cumulative_time': 0.8,
                    'time_per_call': 0.008
                }
            ],
            bottlenecks=[]
        )
        
        # Save profile
        saved_path = await comparator.save_profile('test.py', profile)
        assert Path(saved_path).exists(), "Profile file should be created"
        
        # Load profile
        loaded_path, loaded_profile = await comparator.load_profile(saved_path)
        
        # Verify loaded data
        assert loaded_path == 'test.py'
        assert loaded_profile.execution_time == profile.execution_time
        assert loaded_profile.cpu_percent == profile.cpu_percent
        assert loaded_profile.memory_usage == profile.memory_usage
        assert len(loaded_profile.function_stats) == len(profile.function_stats)
    
    @pytest.mark.asyncio
    async def test_list_profiles(self, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test listing saved performance profiles
        """
        from xencode.features.performance_profiler import PerformanceComparator
        
        comparator = PerformanceComparator()
        comparator.history_dir = tmp_path
        
        # Create multiple profiles
        profile1 = ProfileResult(
            execution_time=1.0,
            cpu_percent=40.0,
            memory_usage=150.0,
            function_stats=[],
            bottlenecks=[]
        )
        
        profile2 = ProfileResult(
            execution_time=1.5,
            cpu_percent=50.0,
            memory_usage=175.0,
            function_stats=[],
            bottlenecks=[]
        )
        
        await comparator.save_profile('test1.py', profile1)
        await comparator.save_profile('test2.py', profile2)
        
        # List all profiles
        profiles = await comparator.list_profiles()
        assert len(profiles) == 2, "Should have 2 saved profiles"
        
        # List profiles for specific path
        profiles_filtered = await comparator.list_profiles('test1.py')
        assert len(profiles_filtered) == 1, "Should have 1 profile for test1.py"
        assert profiles_filtered[0]['path'] == 'test1.py'
    
    @pytest.mark.asyncio
    async def test_visualize_trends(self, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test visualizing performance trends over time
        """
        from xencode.features.performance_profiler import PerformanceComparator
        import time
        
        comparator = PerformanceComparator()
        comparator.history_dir = tmp_path
        
        # Create profiles with improving performance
        for i in range(3):
            profile = ProfileResult(
                execution_time=2.0 - (i * 0.5),  # Getting faster
                cpu_percent=60.0 - (i * 10.0),   # Using less CPU
                memory_usage=200.0 - (i * 25.0), # Using less memory
                function_stats=[],
                bottlenecks=[]
            )
            await comparator.save_profile('test.py', profile)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Visualize trends
        result = await comparator.visualize_trends('test.py')
        
        assert result['success'] is True
        assert result['profile_count'] == 3
        assert 'data' in result
        assert 'trends' in result
        
        trends = result['trends']
        assert trends['execution_time_percent'] < 0, "Execution time should be trending down"
        assert trends['cpu_percent_change'] < 0, "CPU usage should be trending down"
        assert trends['memory_mb_change'] < 0, "Memory usage should be trending down"
    
    @pytest.mark.asyncio
    async def test_visualize_trends_insufficient_data(self, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test visualizing trends with insufficient data
        """
        from xencode.features.performance_profiler import PerformanceComparator
        
        comparator = PerformanceComparator()
        comparator.history_dir = tmp_path
        
        # Create only one profile
        profile = ProfileResult(
            execution_time=1.0,
            cpu_percent=40.0,
            memory_usage=150.0,
            function_stats=[],
            bottlenecks=[]
        )
        await comparator.save_profile('test.py', profile)
        
        # Try to visualize trends
        result = await comparator.visualize_trends('test.py')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'at least 2' in result['error'].lower()


class TestPerformanceReporting:
    """Tests for performance reporting functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_report(self, feature_config, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test generating comprehensive performance report
        """
        profiler = PerformanceProfiler(feature_config)
        await profiler.initialize()
        
        # Override history directory for testing
        profiler.comparator.history_dir = tmp_path
        
        # Create a test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
def test_function():
    total = 0
    for i in range(10000):
        total += i
    return total

test_function()
""")
        
        result = await profiler.generate_report(str(test_file), save_history=True)
        
        assert result['success'] is True
        assert 'report' in result
        
        report = result['report']
        assert 'path' in report
        assert 'timestamp' in report
        assert 'profile_result' in report
        assert 'analysis' in report
        assert 'suggestions' in report
        
        # Verify history was saved
        assert result['history_path'] is not None
        assert Path(result['history_path']).exists()
    
    @pytest.mark.asyncio
    async def test_generate_report_no_history(self, feature_config, tmp_path):
        """
        **Validates: Requirements 9.5**
        Test generating report without saving to history
        """
        profiler = PerformanceProfiler(feature_config)
        await profiler.initialize()
        
        # Create a test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
def test_function():
    return sum(range(1000))

test_function()
""")
        
        result = await profiler.generate_report(str(test_file), save_history=False)
        
        assert result['success'] is True
        assert 'report' in result
        assert result['history_path'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
