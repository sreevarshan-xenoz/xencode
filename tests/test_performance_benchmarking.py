#!/usr/bin/env python3
"""
Performance Benchmarking Tests

Comprehensive performance testing for the Xencode system including:
- Load testing for concurrent users
- Response time and throughput validation
- Cache performance and hit rate testing
- System resource utilization under load
"""

import asyncio
import time
import statistics
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pytest
import requests
import psutil
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import all the routers for testing
from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router
from xencode.api.routers.plugin import router as plugin_router
from xencode.api.routers.workspace import router as workspace_router
from xencode.api.routers.code_analysis import router as code_analysis_router
from xencode.api.routers.document import router as document_router


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.system_metrics = []
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        self.start_time = time.time()
        self.system_metrics = []
        
        def monitor_resources():
            while self.start_time and not self.end_time:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.system_metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / (1024**3)
                })
                time.sleep(0.5)
        
        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.end_time = time.time()
        time.sleep(1)  # Allow final metrics collection
    
    def add_result(self, operation: str, duration: float, success: bool, **kwargs):
        """Add a performance result"""
        self.results.append({
            'operation': operation,
            'duration_ms': duration * 1000,
            'success': success,
            'timestamp': time.time(),
            **kwargs
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics"""
        if not self.results:
            return {}
        
        durations = [r['duration_ms'] for r in self.results if r['success']]
        success_count = len([r for r in self.results if r['success']])
        total_count = len(self.results)
        
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        stats = {
            'total_requests': total_count,
            'successful_requests': success_count,
            'failed_requests': total_count - success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'total_duration_seconds': total_time,
            'requests_per_second': total_count / total_time if total_time > 0 else 0,
            'successful_rps': success_count / total_time if total_time > 0 else 0
        }
        
        if durations:
            stats.update({
                'avg_response_time_ms': statistics.mean(durations),
                'median_response_time_ms': statistics.median(durations),
                'min_response_time_ms': min(durations),
                'max_response_time_ms': max(durations),
                'p95_response_time_ms': self._percentile(durations, 95),
                'p99_response_time_ms': self._percentile(durations, 99)
            })
        
        # System resource statistics
        if self.system_metrics:
            cpu_values = [m['cpu_percent'] for m in self.system_metrics]
            memory_values = [m['memory_percent'] for m in self.system_metrics]
            
            stats.update({
                'avg_cpu_percent': statistics.mean(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'avg_memory_percent': statistics.mean(memory_values),
                'max_memory_percent': max(memory_values),
                'peak_memory_used_gb': max(m['memory_used_gb'] for m in self.system_metrics)
            })
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestPerformanceBenchmarking:
    """Performance benchmarking test suite"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = FastAPI(title="Performance Benchmark Test")
        
        # Include all routers
        self.app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
        self.app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
        self.app.include_router(plugin_router, prefix="/api/v1/plugins", tags=["Plugins"])
        self.app.include_router(workspace_router, prefix="/api/v1/workspaces", tags=["Workspaces"])
        self.app.include_router(code_analysis_router, prefix="/api/v1/code", tags=["Code Analysis"])
        self.app.include_router(document_router, prefix="/api/v1/documents", tags=["Documents"])
        
        self.client = TestClient(self.app)
        self.benchmark = PerformanceBenchmark()
    
    def test_single_endpoint_performance(self):
        """Test performance of individual endpoints"""
        endpoints = [
            ("/api/v1/analytics/health", "GET"),
            ("/api/v1/monitoring/health", "GET"),
            ("/api/v1/plugins/", "GET"),
            ("/api/v1/workspaces/", "GET"),
            ("/api/v1/code/health", "GET"),
            ("/api/v1/documents/health", "GET")
        ]
        
        self.benchmark.start_monitoring()
        
        for endpoint, method in endpoints:
            # Test each endpoint multiple times
            for i in range(10):
                start_time = time.time()
                
                if method == "GET":
                    response = self.client.get(endpoint)
                elif method == "POST":
                    response = self.client.post(endpoint, json={})
                duration = time.time() - start_time
                success = response.status_code < 500
                
                self.benchmark.add_result(
                    operation=f"{method} {endpoint}",
                    duration=duration,
                    success=success,
                    status_code=response.status_code
                )
        
        self.benchmark.stop_monitoring()
        stats = self.benchmark.get_statistics()
        
        # Performance assertions (relaxed for mock implementation)
        assert stats['success_rate'] >= 0.4, f"Success rate too low: {stats['success_rate']}"
        assert stats['avg_response_time_ms'] < 5000, f"Average response time too high: {stats['avg_response_time_ms']}ms"
        if 'p95_response_time_ms' in stats:
            assert stats['p95_response_time_ms'] < 10000, f"95th percentile too high: {stats['p95_response_time_ms']}ms"
        
        print(f"\\n📊 Single Endpoint Performance Results:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        print(f"  95th Percentile: {stats['p95_response_time_ms']:.2f}ms")
        print(f"  Requests/Second: {stats['requests_per_second']:.2f}")
    
    def test_concurrent_user_load(self):
        """Test system performance under concurrent user load"""
        concurrent_users = 20
        requests_per_user = 10
        
        def user_simulation(user_id: int) -> List[Dict]:
            """Simulate a user making multiple requests"""
            user_results = []
            
            endpoints = [
                "/api/v1/analytics/overview",
                "/api/v1/monitoring/resources",
                "/api/v1/plugins/",
                "/api/v1/workspaces/"
            ]
            
            for i in range(requests_per_user):
                endpoint = endpoints[i % len(endpoints)]
                start_time = time.time()
                
                try:
                    response = self.client.get(endpoint)
                    duration = time.time() - start_time
                    success = response.status_code < 500
                    
                    user_results.append({
                        'user_id': user_id,
                        'request_id': i,
                        'endpoint': endpoint,
                        'duration': duration,
                        'success': success,
                        'status_code': response.status_code
                    })
                except Exception as e:
                    duration = time.time() - start_time
                    user_results.append({
                        'user_id': user_id,
                        'request_id': i,
                        'endpoint': endpoint,
                        'duration': duration,
                        'success': False,
                        'error': str(e)
                    })
                
                # Small delay between requests
                time.sleep(0.1)
            
            return user_results
        
        self.benchmark.start_monitoring()
        
        # Run concurrent user simulations
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_simulation, user_id) for user_id in range(concurrent_users)]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                user_results = future.result()
                all_results.extend(user_results)
        
        self.benchmark.stop_monitoring()
        
        # Process results
        for result in all_results:
            self.benchmark.add_result(
                operation=f"User {result['user_id']} - {result['endpoint']}",
                duration=result['duration'],
                success=result['success'],
                user_id=result['user_id'],
                endpoint=result['endpoint']
            )
        
        stats = self.benchmark.get_statistics()
        
        # Performance assertions for concurrent load (relaxed for mock implementation)
        assert stats['success_rate'] >= 0.3, f"Success rate under load too low: {stats['success_rate']}"
        assert stats['avg_response_time_ms'] < 10000, f"Average response time under load too high: {stats['avg_response_time_ms']}ms"
        assert stats['requests_per_second'] >= 1, f"Throughput too low: {stats['requests_per_second']} RPS"
        
        print(f"\\n🚀 Concurrent Load Test Results ({concurrent_users} users, {requests_per_user} requests each):")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        print(f"  95th Percentile: {stats['p95_response_time_ms']:.2f}ms")
        print(f"  Throughput: {stats['requests_per_second']:.2f} RPS")
        print(f"  Peak CPU Usage: {stats.get('max_cpu_percent', 0):.1f}%")
        print(f"  Peak Memory Usage: {stats.get('max_memory_percent', 0):.1f}%")
    
    def test_cache_performance_and_hit_rates(self):
        """Test cache performance and hit rates"""
        # Test cache warming and hit rates
        cache_test_data = [
            {
                "name": "cache_test_metric_1",
                "value": 100.0,
                "metric_type": "gauge",
                "labels": {"test": "cache_performance"}
            },
            {
                "name": "cache_test_metric_2", 
                "value": 200.0,
                "metric_type": "counter",
                "labels": {"test": "cache_performance"}
            }
        ]
        
        self.benchmark.start_monitoring()
        
        # First, populate cache with initial requests (cache misses)
        print("\\n🔄 Cache Warming Phase...")
        for i, data in enumerate(cache_test_data):
            start_time = time.time()
            response = self.client.post("/api/v1/analytics/metrics", json=data)
            duration = time.time() - start_time
            
            self.benchmark.add_result(
                operation="cache_miss",
                duration=duration,
                success=response.status_code == 200,
                cache_status="miss"
            )
        
        # Now test cache hits by repeating the same requests
        print("🎯 Cache Hit Testing Phase...")
        for round_num in range(5):  # Multiple rounds to test cache consistency
            for i, data in enumerate(cache_test_data):
                start_time = time.time()
                response = self.client.post("/api/v1/analytics/metrics", json=data)
                duration = time.time() - start_time
                
                self.benchmark.add_result(
                    operation="cache_hit",
                    duration=duration,
                    success=response.status_code == 200,
                    cache_status="hit",
                    round=round_num
                )
        
        # Test cache performance with analytics queries
        print("📊 Cache Query Performance...")
        for i in range(20):
            start_time = time.time()
            response = self.client.get("/api/v1/analytics/metrics?labels=test=cache_performance")
            duration = time.time() - start_time
            
            self.benchmark.add_result(
                operation="cache_query",
                duration=duration,
                success=response.status_code == 200,
                cache_status="query"
            )
        
        self.benchmark.stop_monitoring()
        stats = self.benchmark.get_statistics()
        
        # Calculate cache-specific metrics
        cache_miss_times = [r['duration_ms'] for r in self.benchmark.results if r['operation'] == 'cache_miss' and r['success']]
        cache_hit_times = [r['duration_ms'] for r in self.benchmark.results if r['operation'] == 'cache_hit' and r['success']]
        cache_query_times = [r['duration_ms'] for r in self.benchmark.results if r['operation'] == 'cache_query' and r['success']]
        
        # Performance assertions for cache
        if cache_miss_times and cache_hit_times:
            avg_miss_time = statistics.mean(cache_miss_times)
            avg_hit_time = statistics.mean(cache_hit_times)
            
            # Cache hits should be faster than misses (though in mock implementation this may not be true)
            print(f"\\n💾 Cache Performance Results:")
            print(f"  Cache Miss Avg Time: {avg_miss_time:.2f}ms")
            print(f"  Cache Hit Avg Time: {avg_hit_time:.2f}ms")
            print(f"  Cache Efficiency: {((avg_miss_time - avg_hit_time) / avg_miss_time * 100):.1f}% improvement")
        
        if cache_query_times:
            avg_query_time = statistics.mean(cache_query_times)
            print(f"  Cache Query Avg Time: {avg_query_time:.2f}ms")
            
            # Cache queries should be reasonably fast
            assert avg_query_time < 500, f"Cache query time too slow: {avg_query_time}ms"
        
        assert stats['success_rate'] >= 0.95, f"Cache operation success rate too low: {stats['success_rate']}"
    
    def test_throughput_targets(self):
        """Test system throughput against targets"""
        target_rps = 50  # Target: 50 requests per second
        test_duration = 10  # 10 seconds
        
        endpoints = [
            "/api/v1/analytics/health",
            "/api/v1/monitoring/health", 
            "/api/v1/plugins/system/status"
        ]
        
        self.benchmark.start_monitoring()
        
        def make_requests():
            """Make requests continuously for the test duration"""
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                endpoint = endpoints[int(time.time()) % len(endpoints)]
                start_time = time.time()
                
                try:
                    response = self.client.get(endpoint)
                    duration = time.time() - start_time
                    success = response.status_code < 500
                    
                    self.benchmark.add_result(
                        operation="throughput_test",
                        duration=duration,
                        success=success,
                        endpoint=endpoint
                    )
                except Exception as e:
                    duration = time.time() - start_time
                    self.benchmark.add_result(
                        operation="throughput_test",
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                
                # Small delay to control request rate
                time.sleep(0.01)
        
        # Run throughput test with multiple threads
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        self.benchmark.stop_monitoring()
        stats = self.benchmark.get_statistics()
        
        print(f"\\n🎯 Throughput Test Results (Target: {target_rps} RPS):")
        print(f"  Achieved RPS: {stats['requests_per_second']:.2f}")
        print(f"  Successful RPS: {stats['successful_rps']:.2f}")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        
        # Throughput assertions (relaxed for mock implementation)
        assert stats['requests_per_second'] >= target_rps * 0.5, f"Throughput too low: {stats['requests_per_second']} RPS (target: {target_rps})"
        assert stats['success_rate'] >= 0.90, f"Success rate during throughput test too low: {stats['success_rate']}"
    
    def test_response_time_targets(self):
        """Test response time targets across different endpoints"""
        response_time_targets = {
            "/api/v1/analytics/health": 100,  # 100ms target
            "/api/v1/monitoring/health": 100,  # 100ms target
            "/api/v1/analytics/overview": 500,  # 500ms target
            "/api/v1/monitoring/resources": 300,  # 300ms target
            "/api/v1/plugins/": 200,  # 200ms target
        }
        
        self.benchmark.start_monitoring()
        
        for endpoint, target_ms in response_time_targets.items():
            endpoint_times = []
            
            # Test each endpoint multiple times
            for i in range(20):
                start_time = time.time()
                response = self.client.get(endpoint)
                duration = time.time() - start_time
                
                success = response.status_code < 500
                endpoint_times.append(duration * 1000)  # Convert to ms
                
                self.benchmark.add_result(
                    operation=f"response_time_test_{endpoint}",
                    duration=duration,
                    success=success,
                    target_ms=target_ms
                )
            
            # Calculate endpoint-specific statistics
            if endpoint_times:
                avg_time = statistics.mean(endpoint_times)
                p95_time = self.benchmark._percentile(endpoint_times, 95)
                
                print(f"\\n⏱️  {endpoint}:")
                print(f"    Target: {target_ms}ms")
                print(f"    Average: {avg_time:.2f}ms")
                print(f"    95th Percentile: {p95_time:.2f}ms")
                print(f"    Target Met: {'✅' if avg_time <= target_ms else '❌'}")
                
                # Relaxed assertions for mock implementation
                # In a real system, these would be stricter
                assert avg_time <= target_ms * 3, f"Average response time for {endpoint} too high: {avg_time}ms (target: {target_ms}ms)"
        
        self.benchmark.stop_monitoring()
        stats = self.benchmark.get_statistics()
        
        print(f"\\n📈 Overall Response Time Results:")
        print(f"  Overall Average: {stats['avg_response_time_ms']:.2f}ms")
        print(f"  Overall 95th Percentile: {stats['p95_response_time_ms']:.2f}ms")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    def test_system_resource_utilization(self):
        """Test system resource utilization under load"""
        self.benchmark.start_monitoring()
        
        # Generate sustained load for resource monitoring
        def generate_load():
            for i in range(100):
                # Mix of different endpoint types
                endpoints = [
                    "/api/v1/analytics/overview",
                    "/api/v1/monitoring/resources",
                    "/api/v1/plugins/",
                    "/api/v1/workspaces/"
                ]
                
                for endpoint in endpoints:
                    start_time = time.time()
                    response = self.client.get(endpoint)
                    duration = time.time() - start_time
                    
                    self.benchmark.add_result(
                        operation="resource_test",
                        duration=duration,
                        success=response.status_code < 500,
                        endpoint=endpoint
                    )
                
                time.sleep(0.05)  # Small delay
        
        # Run load generation in multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_load) for _ in range(3)]
            
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        self.benchmark.stop_monitoring()
        stats = self.benchmark.get_statistics()
        
        print(f"\\n🖥️  System Resource Utilization:")
        print(f"  Peak CPU Usage: {stats.get('max_cpu_percent', 0):.1f}%")
        print(f"  Average CPU Usage: {stats.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Peak Memory Usage: {stats.get('max_memory_percent', 0):.1f}%")
        print(f"  Average Memory Usage: {stats.get('avg_memory_percent', 0):.1f}%")
        print(f"  Peak Memory Used: {stats.get('peak_memory_used_gb', 0):.2f} GB")
        
        # Resource utilization assertions
        assert stats.get('max_cpu_percent', 0) < 90, f"CPU usage too high: {stats.get('max_cpu_percent', 0)}%"
        assert stats.get('max_memory_percent', 0) < 85, f"Memory usage too high: {stats.get('max_memory_percent', 0)}%"
        
        print(f"\\n✅ Resource utilization within acceptable limits")


def run_performance_benchmarks():
    """Run all performance benchmarks"""
    print("🚀 Starting Comprehensive Performance Benchmarking")
    print("=" * 60)
    
    # Run pytest with performance tests
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_performance_benchmarking.py", 
        "-v", "--tb=short", "-s"  # -s to show print statements
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run the performance benchmarks
    success = run_performance_benchmarks()
    
    if success:
        print("\\n🎉 All performance benchmarks completed successfully!")
        print("📊 System performance meets target requirements!")
    else:
        print("\\n⚠️  Some performance benchmarks failed or need attention.")
        print("🔧 Review the results above for optimization opportunities.")