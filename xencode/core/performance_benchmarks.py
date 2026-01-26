"""
Performance benchmarks for Xencode
Provides tools for measuring and benchmarking performance
"""
import time
import statistics
from typing import Callable, Any, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import cProfile
import pstats
from io import StringIO
import memory_profiler
from functools import wraps


@dataclass
class BenchmarkResult:
    """Data class to represent benchmark results"""
    name: str
    execution_times: List[float]
    memory_usage: Optional[List[Dict[str, float]]] = None
    cpu_usage: Optional[List[float]] = None
    timestamp: datetime = datetime.now()
    
    @property
    def mean_time(self) -> float:
        """Mean execution time"""
        return statistics.mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def median_time(self) -> float:
        """Median execution time"""
        return statistics.median(self.execution_times) if self.execution_times else 0.0
    
    @property
    def min_time(self) -> float:
        """Minimum execution time"""
        return min(self.execution_times) if self.execution_times else 0.0
    
    @property
    def max_time(self) -> float:
        """Maximum execution time"""
        return max(self.execution_times) if self.execution_times else 0.0
    
    @property
    def std_deviation(self) -> float:
        """Standard deviation of execution times"""
        if len(self.execution_times) < 2:
            return 0.0
        return statistics.stdev(self.execution_times)


class BenchmarkSuite:
    """A suite for running performance benchmarks"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.benchmarks: Dict[str, Callable] = {}
    
    def add_benchmark(self, name: str, func: Callable, *args, **kwargs) -> None:
        """Add a benchmark function to the suite
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        self.benchmarks[name] = lambda: func(*args, **kwargs)
    
    def run_benchmark(self, name: str, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
        """Run a specific benchmark
        
        Args:
            name: Name of the benchmark to run
            iterations: Number of iterations to run
            warmup: Number of warmup iterations to run before measuring
            
        Returns:
            Benchmark result
        """
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        # Warmup iterations
        for _ in range(warmup):
            self.benchmarks[name]()
        
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.benchmarks[name]()
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
        
        result = BenchmarkResult(
            name=name,
            execution_times=execution_times
        )
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self, iterations: int = 10, warmup: int = 2) -> List[BenchmarkResult]:
        """Run all registered benchmarks
        
        Args:
            iterations: Number of iterations to run for each benchmark
            warmup: Number of warmup iterations to run before measuring
            
        Returns:
            List of benchmark results
        """
        results = []
        for name in self.benchmarks:
            result = self.run_benchmark(name, iterations, warmup)
            results.append(result)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmark results
        
        Returns:
            Summary of benchmark results
        """
        if not self.results:
            return {"message": "No benchmarks have been run"}
        
        summary = {
            "total_benchmarks": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
        
        for result in self.results:
            summary["benchmarks"][result.name] = {
                "mean_time": result.mean_time,
                "median_time": result.median_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "std_deviation": result.std_deviation,
                "iterations": len(result.execution_times)
            }
        
        return summary
    
    def compare_results(self, result1: BenchmarkResult, result2: BenchmarkResult) -> Dict[str, float]:
        """Compare two benchmark results
        
        Args:
            result1: First benchmark result
            result2: Second benchmark result
            
        Returns:
            Comparison results showing differences
        """
        return {
            "mean_difference": result1.mean_time - result2.mean_time,
            "median_difference": result1.median_time - result2.median_time,
            "speedup_factor": result2.mean_time / result1.mean_time if result1.mean_time > 0 else float('inf'),
            "percent_change": ((result1.mean_time - result2.mean_time) / result2.mean_time * 100) if result2.mean_time > 0 else 0
        }


def benchmark_function(iterations: int = 10, warmup: int = 2):
    """Decorator to benchmark a function
    
    Args:
        iterations: Number of iterations to run
        warmup: Number of warmup iterations to run before measuring
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Warmup iterations
            for _ in range(warmup):
                func(*args, **kwargs)
            
            execution_times = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            
            mean_time = statistics.mean(execution_times)
            print(f"Benchmark results for {func.__name__}:")
            print(f"  Mean execution time: {mean_time:.6f}s")
            print(f"  Min execution time: {min(execution_times):.6f}s")
            print(f"  Max execution time: {max(execution_times):.6f}s")
            print(f"  Std deviation: {statistics.stdev(execution_times) if len(execution_times) > 1 else 0:.6f}s")
            
            return result
        return wrapper
    return decorator


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function using cProfile
    
    Args:
        func: Function to profile
        
    Returns:
        Profiled function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Profiling results for {func.__name__}:")
        print(s.getvalue())
        
        return result
    return wrapper


def memory_profile_function(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function
    
    Args:
        func: Function to profile for memory usage
        
    Returns:
        Memory-profiled function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Note: This requires the memory-profiler package
        # In a real implementation, we'd check if the package is available
        try:
            mem_usage = memory_profiler.memory_usage((func, args, kwargs), retval=True)
            result, peak_usage = mem_usage[1], max(mem_usage[0])
            
            print(f"Memory profiling results for {func.__name__}:")
            print(f"  Peak memory usage: {peak_usage:.2f} MB")
            
            return result
        except ImportError:
            print("memory-profiler not available, skipping memory profiling")
            return func(*args, **kwargs)
    return wrapper


class PerformanceMonitor:
    """Monitors performance in real-time"""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'execution_time': 0,
            'function_calls': 0,
            'memory_before': None,
            'memory_after': None
        }
        self._enabled = True
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.metrics['start_time'] = time.perf_counter()
        self.metrics['function_calls'] = 0
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if self.metrics['start_time'] is not None:
            self.metrics['end_time'] = time.perf_counter()
            self.metrics['execution_time'] = self.metrics['end_time'] - self.metrics['start_time']
    
    def increment_calls(self):
        """Increment the function call counter"""
        self.metrics['function_calls'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()
    
    def report(self) -> str:
        """Generate a performance report
        
        Returns:
            Performance report as a string
        """
        if self.metrics['start_time'] is None:
            return "Monitoring not started"
        
        end_time = self.metrics['end_time'] or time.perf_counter()
        execution_time = end_time - self.metrics['start_time']
        
        report = f"Performance Report:\n"
        report += f"  Execution time: {execution_time:.6f}s\n"
        report += f"  Function calls: {self.metrics['function_calls']}\n"
        if self.metrics['function_calls'] > 0:
            report += f"  Avg time per call: {execution_time / self.metrics['function_calls']:.6f}s\n"
        
        return report


# Global benchmark suite instance
benchmark_suite = BenchmarkSuite()


def get_benchmark_suite() -> BenchmarkSuite:
    """Get the global benchmark suite instance"""
    return benchmark_suite


def run_simple_benchmark(func: Callable, *args, name: str = None, iterations: int = 10, warmup: int = 2, **kwargs) -> BenchmarkResult:
    """Run a simple benchmark on a function
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        name: Name for the benchmark (defaults to function name)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations to run before measuring
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Benchmark result
    """
    bench_name = name or func.__name__
    suite = get_benchmark_suite()
    suite.add_benchmark(bench_name, func, *args, **kwargs)
    return suite.run_benchmark(bench_name, iterations, warmup)


# Example benchmark functions for Xencode components
def benchmark_cache_operations():
    """Benchmark cache operations"""
    from xencode.core.cache import ResponseCache
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        # Benchmark set operation
        start_time = time.perf_counter()
        cache.set("test_prompt", "test_model", "test_response")
        set_time = time.perf_counter() - start_time
        
        # Benchmark get operation
        start_time = time.perf_counter()
        result = cache.get("test_prompt", "test_model")
        get_time = time.perf_counter() - start_time
        
        print(f"Cache set operation: {set_time:.6f}s")
        print(f"Cache get operation: {get_time:.6f}s")
        print(f"Cache result: {result}")


def benchmark_model_operations():
    """Benchmark model operations"""
    from xencode.core.models import ModelManager
    
    manager = ModelManager()
    
    # Benchmark model refresh
    start_time = time.perf_counter()
    manager.refresh_models()
    refresh_time = time.perf_counter() - start_time
    
    print(f"Model refresh operation: {refresh_time:.6f}s")
    print(f"Available models: {len(manager.available_models)}")


def benchmark_memory_operations():
    """Benchmark memory operations"""
    from xencode.core.memory import ConversationMemory
    
    memory = ConversationMemory()
    
    # Benchmark adding messages
    start_time = time.perf_counter()
    for i in range(10):
        memory.add_message("user", f"Test message {i}", "test_model")
    add_time = time.perf_counter() - start_time
    
    # Benchmark getting context
    start_time = time.perf_counter()
    context = memory.get_context()
    get_time = time.perf_counter() - start_time
    
    print(f"Adding 10 messages: {add_time:.6f}s")
    print(f"Getting context: {get_time:.6f}s")
    print(f"Context length: {len(context)}")


if __name__ == "__main__":
    print("Running Xencode performance benchmarks...\n")
    
    print("Benchmarking cache operations:")
    benchmark_cache_operations()
    print()
    
    print("Benchmarking model operations:")
    benchmark_model_operations()
    print()
    
    print("Benchmarking memory operations:")
    benchmark_memory_operations()
    print()
    
    # Run with the benchmark decorator
    @benchmark_function(iterations=5)
    def test_fast_function():
        return sum(range(1000))
    
    print("Testing decorated function:")
    test_fast_function()