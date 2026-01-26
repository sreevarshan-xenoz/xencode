"""
Performance benchmarking module for Xencode
Provides tools to measure and compare performance of different components
"""
import asyncio
import statistics
import time
from typing import Any, Callable, Dict, List

import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from xencode.core.cache import ResponseCache

console = Console()

class BenchmarkResult:
    """Class to store benchmark results"""
    def __init__(self, name: str, times: List[float], errors: List[Exception]):
        self.name = name
        self.times = times
        self.errors = errors
        self.mean_time = statistics.mean(times) if times else 0
        self.median_time = statistics.median(times) if times else 0
        self.min_time = min(times) if times else 0
        self.max_time = max(times) if times else 0
        self.std_dev = statistics.stdev(times) if len(times) > 1 else 0
        self.error_rate = len(errors) / (len(times) + len(errors)) if (times or errors) else 0


class PerformanceBenchmarkSuite:
    """Suite of performance benchmarks for Xencode components"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(self, func: Callable, name: str, iterations: int = 10, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a synchronous function"""
        times = []
        errors = []
        
        console.print(f"[cyan]Running benchmark: {name}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Benchmarking {name}...", total=iterations)
            
            for i in range(iterations):
                try:
                    start_time = time.perf_counter()
                    func(*args, **kwargs)
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                except Exception as e:
                    errors.append(e)
                
                progress.update(task, advance=1)
        
        result = BenchmarkResult(name, times, errors)
        self.results.append(result)
        return result
    
    async def benchmark_async_function(self, func, name: str, iterations: int = 10, *args, **kwargs) -> BenchmarkResult:
        """Benchmark an asynchronous function"""
        times = []
        errors = []
        
        console.print(f"[cyan]Running async benchmark: {name}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Async benchmarking {name}...", total=iterations)
            
            for i in range(iterations):
                try:
                    start_time = time.perf_counter()
                    await func(*args, **kwargs)
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                except Exception as e:
                    errors.append(e)
                
                progress.update(task, advance=1)
        
        result = BenchmarkResult(name, times, errors)
        self.results.append(result)
        return result
    
    def benchmark_cache_performance(self, cache: ResponseCache, iterations: int = 100) -> BenchmarkResult:
        """Benchmark cache performance"""
        def cache_ops():
            # Create unique prompts for each iteration to avoid cache hits skewing results
            import random
            import string
            prompt = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            model = "test-model"
            response = f"Response to {prompt}"
            
            # Set and get operations
            cache.set(prompt, model, response)
            retrieved = cache.get(prompt, model)
            assert retrieved == response
        
        return self.benchmark_function(
            cache_ops,
            f"Cache Operations (size={iterations})",
            iterations
        )
    
    def benchmark_api_response_time(self, api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], iterations: int = 10) -> BenchmarkResult:
        """Benchmark API response time"""
        def api_call():
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return self.benchmark_function(
            api_call,
            f"API Response Time ({api_endpoint})",
            iterations
        )
    
    async def benchmark_concurrent_api_calls(self, api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], concurrent_requests: int = 10) -> BenchmarkResult:
        """Benchmark concurrent API calls using aiohttp"""
        async def api_call(session):
            async with session.post(api_endpoint, headers=headers, json=payload, timeout=30) as response:
                return await response.json()
        
        async def concurrent_calls():
            connector = aiohttp.TCPConnector(limit=concurrent_requests)
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = [api_call(session) for _ in range(concurrent_requests)]
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return await self.benchmark_async_function(
            concurrent_calls,
            f"Concurrent API Calls ({concurrent_requests} reqs)",
            1  # Just run once for concurrent test
        )
    
    def print_results(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            console.print(Panel("No benchmark results to display", style="yellow"))
            return
        
        table = Table(title="Performance Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Test Name", style="cyan")
        table.add_column("Mean (s)", style="green")
        table.add_column("Median (s)", style="green")
        table.add_column("Min (s)", style="green")
        table.add_column("Max (s)", style="green")
        table.add_column("Std Dev", style="yellow")
        table.add_column("Error Rate", style="red")
        
        for result in self.results:
            table.add_row(
                result.name,
                f"{result.mean_time:.4f}",
                f"{result.median_time:.4f}",
                f"{result.min_time:.4f}",
                f"{result.max_time:.4f}",
                f"{result.std_dev:.4f}",
                f"{result.error_rate:.2%}"
            )
        
        console.print(table)
        
        # Print summary
        total_time = sum(result.mean_time for result in self.results)
        console.print(f"\n[cyan]Total benchmark time: {total_time:.4f}s[/cyan]")
        console.print(f"[cyan]Number of tests run: {len(self.results)}[/cyan]")


def run_performance_benchmarks():
    """Run a comprehensive set of performance benchmarks"""
    console.print(Panel("🚀 Xencode Performance Benchmark Suite", style="bold blue"))
    
    suite = PerformanceBenchmarkSuite()
    
    # Benchmark cache performance
    cache = ResponseCache(max_size=50)
    suite.benchmark_cache_performance(cache, iterations=50)
    
    # Example of how to benchmark API calls (would need actual endpoint)
    # suite.benchmark_api_response_time(
    #     "http://localhost:11434/api/generate",
    #     {"Content-Type": "application/json"},
    #     {"model": "test", "prompt": "test", "stream": False}
    # )
    
    # Print results
    suite.print_results()
    
    # Save results to file
    import json
    from datetime import datetime
    from pathlib import Path
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "name": result.name,
                "mean_time": result.mean_time,
                "median_time": result.median_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "std_dev": result.std_dev,
                "error_rate": result.error_rate,
                "sample_size": len(result.times)
            }
            for result in suite.results
        ]
    }
    
    # Create benchmarks directory if it doesn't exist
    bench_dir = Path("benchmarks")
    bench_dir.mkdir(exist_ok=True)
    
    # Write results to file
    results_file = bench_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n💾 Benchmark results saved to: {results_file}")
    
    return suite


if __name__ == "__main__":
    run_performance_benchmarks()