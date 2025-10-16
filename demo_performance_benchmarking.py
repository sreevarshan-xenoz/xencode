#!/usr/bin/env python3
"""
Demo: Performance Benchmarking System

Demonstrates comprehensive performance testing and benchmarking capabilities
for the Xencode system including load testing, throughput analysis, and
resource monitoring.
"""

import asyncio
import time
import statistics
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psutil

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import all the routers
from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router
from xencode.api.routers.plugin import router as plugin_router
from xencode.api.routers.workspace import router as workspace_router
from xencode.api.routers.code_analysis import router as code_analysis_router
from xencode.api.routers.document import router as document_router


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self):
        self.metrics = []
        self.is_monitoring = False
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.metrics = []
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    # Collect network stats if available
                    try:
                        network = psutil.net_io_counters()
                        network_stats = {
                            'bytes_sent': network.bytes_sent,
                            'bytes_recv': network.bytes_recv,
                            'packets_sent': network.packets_sent,
                            'packets_recv': network.packets_recv
                        }
                    except:
                        network_stats = {}
                    
                    metric = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_used_gb': disk.used / (1024**3),
                        'disk_free_gb': disk.free / (1024**3),
                        **network_stats
                    }
                    
                    self.metrics.append(metric)
                    time.sleep(1)  # Collect metrics every second
                    
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                    time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        time.sleep(1.5)  # Allow final collection
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            'monitoring_duration_seconds': duration,
            'total_samples': len(self.metrics),
            'cpu_stats': {
                'average': statistics.mean(cpu_values),
                'maximum': max(cpu_values),
                'minimum': min(cpu_values),
                'p95': self._percentile(cpu_values, 95)
            },
            'memory_stats': {
                'average_percent': statistics.mean(memory_values),
                'maximum_percent': max(memory_values),
                'peak_used_gb': max(m['memory_used_gb'] for m in self.metrics),
                'minimum_available_gb': min(m['memory_available_gb'] for m in self.metrics)
            },
            'disk_stats': {
                'current_usage_percent': self.metrics[-1]['disk_percent'],
                'current_used_gb': self.metrics[-1]['disk_used_gb'],
                'current_free_gb': self.metrics[-1]['disk_free_gb']
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class LoadTester:
    """Load testing utilities"""
    
    def __init__(self):
        self.results = []
        self.active_tests = 0
    
    async def simulate_user_load(self, concurrent_users: int, requests_per_user: int, 
                                endpoints: List[str]) -> Dict[str, Any]:
        """Simulate concurrent user load"""
        import aiohttp
        import asyncio
        
        async def user_session(session: aiohttp.ClientSession, user_id: int) -> List[Dict]:
            """Simulate a single user session"""
            user_results = []
            
            for request_id in range(requests_per_user):
                endpoint = endpoints[request_id % len(endpoints)]
                start_time = time.time()
                
                try:
                    async with session.get(f"http://localhost:8000{endpoint}") as response:
                        await response.text()  # Read response
                        duration = time.time() - start_time
                        
                        user_results.append({
                            'user_id': user_id,
                            'request_id': request_id,
                            'endpoint': endpoint,
                            'duration_ms': duration * 1000,
                            'status_code': response.status,
                            'success': response.status < 500
                        })
                        
                except Exception as e:
                    duration = time.time() - start_time
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'endpoint': endpoint,
                        'duration_ms': duration * 1000,
                        'status_code': 0,
                        'success': False,
                        'error': str(e)
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            return user_results
        
        # Run concurrent user sessions
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [user_session(session, user_id) for user_id in range(concurrent_users)]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        combined_results = []
        for result in all_results:
            if isinstance(result, list):
                combined_results.extend(result)
        
        # Calculate statistics
        successful_requests = [r for r in combined_results if r['success']]
        failed_requests = [r for r in combined_results if not r['success']]
        
        if successful_requests:
            durations = [r['duration_ms'] for r in successful_requests]
            
            stats = {
                'total_requests': len(combined_results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(combined_results),
                'avg_response_time_ms': statistics.mean(durations),
                'median_response_time_ms': statistics.median(durations),
                'p95_response_time_ms': self._percentile(durations, 95),
                'p99_response_time_ms': self._percentile(durations, 99),
                'min_response_time_ms': min(durations),
                'max_response_time_ms': max(durations)
            }
        else:
            stats = {
                'total_requests': len(combined_results),
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0,
                'error': 'No successful requests'
            }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


def create_benchmarking_app() -> FastAPI:
    """Create FastAPI application with benchmarking endpoints"""
    
    app = FastAPI(
        title="Xencode Performance Benchmarking",
        description="Performance testing and benchmarking system for Xencode",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include all routers
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(plugin_router, prefix="/api/v1/plugins", tags=["Plugins"])
    app.include_router(workspace_router, prefix="/api/v1/workspaces", tags=["Workspaces"])
    app.include_router(code_analysis_router, prefix="/api/v1/code", tags=["Code Analysis"])
    app.include_router(document_router, prefix="/api/v1/documents", tags=["Documents"])
    
    # Global performance monitor
    performance_monitor = PerformanceMonitor()
    load_tester = LoadTester()
    
    @app.get("/")
    async def root():
        """Root endpoint with benchmarking overview"""
        return {
            "message": "Xencode Performance Benchmarking System",
            "version": "1.0.0",
            "description": "Comprehensive performance testing and monitoring",
            "features": [
                "Real-time performance monitoring",
                "Load testing with concurrent users",
                "Response time analysis",
                "Throughput benchmarking",
                "Resource utilization tracking",
                "Cache performance testing"
            ],
            "endpoints": {
                "start_monitoring": "/benchmark/monitor/start",
                "stop_monitoring": "/benchmark/monitor/stop",
                "get_metrics": "/benchmark/monitor/metrics",
                "load_test": "/benchmark/load-test",
                "throughput_test": "/benchmark/throughput",
                "response_time_test": "/benchmark/response-time"
            }
        }
    
    @app.post("/benchmark/monitor/start")
    async def start_monitoring():
        """Start performance monitoring"""
        if performance_monitor.is_monitoring:
            return {"message": "Monitoring already active", "status": "active"}
        
        performance_monitor.start_monitoring()
        return {
            "message": "Performance monitoring started",
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/benchmark/monitor/stop")
    async def stop_monitoring():
        """Stop performance monitoring"""
        if not performance_monitor.is_monitoring:
            return {"message": "Monitoring not active", "status": "inactive"}
        
        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()
        
        return {
            "message": "Performance monitoring stopped",
            "status": "stopped",
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
    
    @app.get("/benchmark/monitor/metrics")
    async def get_monitoring_metrics():
        """Get current monitoring metrics"""
        if not performance_monitor.is_monitoring:
            return {"error": "Monitoring not active"}
        
        return {
            "status": "monitoring",
            "current_metrics": performance_monitor.metrics[-10:] if performance_monitor.metrics else [],
            "total_samples": len(performance_monitor.metrics),
            "monitoring_duration": time.time() - performance_monitor.start_time if performance_monitor.start_time else 0
        }
    
    @app.get("/benchmark/monitor/stream")
    async def stream_metrics():
        """Stream real-time metrics"""
        def generate_metrics():
            while performance_monitor.is_monitoring:
                if performance_monitor.metrics:
                    latest_metric = performance_monitor.metrics[-1]
                    yield f"data: {json.dumps(latest_metric)}\\n\\n"
                time.sleep(1)
        
        return StreamingResponse(generate_metrics(), media_type="text/plain")
    
    @app.post("/benchmark/load-test")
    async def run_load_test(
        concurrent_users: int = 10,
        requests_per_user: int = 5,
        target_endpoints: List[str] = None
    ):
        """Run load testing with concurrent users"""
        if target_endpoints is None:
            target_endpoints = [
                "/api/v1/analytics/health",
                "/api/v1/monitoring/health",
                "/api/v1/plugins/",
                "/api/v1/workspaces/"
            ]
        
        print(f"🚀 Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Start monitoring during load test
        was_monitoring = performance_monitor.is_monitoring
        if not was_monitoring:
            performance_monitor.start_monitoring()
        
        try:
            # Run the load test
            results = await load_tester.simulate_user_load(
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
                endpoints=target_endpoints
            )
            
            # Add test metadata
            results.update({
                "test_config": {
                    "concurrent_users": concurrent_users,
                    "requests_per_user": requests_per_user,
                    "target_endpoints": target_endpoints,
                    "total_expected_requests": concurrent_users * requests_per_user
                },
                "timestamp": datetime.now().isoformat(),
                "test_type": "load_test"
            })
            
            return results
            
        finally:
            # Stop monitoring if we started it
            if not was_monitoring:
                performance_monitor.stop_monitoring()
    
    @app.post("/benchmark/throughput")
    async def run_throughput_test(
        target_rps: int = 50,
        duration_seconds: int = 10,
        endpoint: str = "/api/v1/analytics/health"
    ):
        """Run throughput benchmarking"""
        print(f"🎯 Starting throughput test: {target_rps} RPS for {duration_seconds}s on {endpoint}")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Start monitoring
        was_monitoring = performance_monitor.is_monitoring
        if not was_monitoring:
            performance_monitor.start_monitoring()
        
        try:
            import aiohttp
            
            async def make_request(session: aiohttp.ClientSession):
                request_start = time.time()
                try:
                    async with session.get(f"http://localhost:8000{endpoint}") as response:
                        await response.text()
                        duration = time.time() - request_start
                        return {
                            'timestamp': request_start,
                            'duration_ms': duration * 1000,
                            'status_code': response.status,
                            'success': response.status < 500
                        }
                except Exception as e:
                    duration = time.time() - request_start
                    return {
                        'timestamp': request_start,
                        'duration_ms': duration * 1000,
                        'status_code': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate delay between requests to achieve target RPS
            delay_between_requests = 1.0 / target_rps
            
            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                while time.time() < end_time:
                    result = await make_request(session)
                    results.append(result)
                    
                    # Control request rate
                    await asyncio.sleep(delay_between_requests)
            
            # Calculate statistics
            actual_duration = time.time() - start_time
            successful_requests = [r for r in results if r['success']]
            
            if successful_requests:
                durations = [r['duration_ms'] for r in successful_requests]
                
                stats = {
                    "test_config": {
                        "target_rps": target_rps,
                        "duration_seconds": duration_seconds,
                        "endpoint": endpoint
                    },
                    "results": {
                        "actual_duration": actual_duration,
                        "total_requests": len(results),
                        "successful_requests": len(successful_requests),
                        "failed_requests": len(results) - len(successful_requests),
                        "achieved_rps": len(results) / actual_duration,
                        "successful_rps": len(successful_requests) / actual_duration,
                        "success_rate": len(successful_requests) / len(results),
                        "avg_response_time_ms": statistics.mean(durations),
                        "p95_response_time_ms": load_tester._percentile(durations, 95),
                        "target_achieved": (len(results) / actual_duration) >= (target_rps * 0.9)
                    },
                    "timestamp": datetime.now().isoformat(),
                    "test_type": "throughput_test"
                }
            else:
                stats = {
                    "test_config": {
                        "target_rps": target_rps,
                        "duration_seconds": duration_seconds,
                        "endpoint": endpoint
                    },
                    "results": {
                        "error": "No successful requests",
                        "total_requests": len(results),
                        "successful_requests": 0
                    },
                    "timestamp": datetime.now().isoformat(),
                    "test_type": "throughput_test"
                }
            
            return stats
            
        finally:
            if not was_monitoring:
                performance_monitor.stop_monitoring()
    
    @app.post("/benchmark/response-time")
    async def run_response_time_test(
        endpoints: List[str] = None,
        samples_per_endpoint: int = 20
    ):
        """Run response time benchmarking"""
        if endpoints is None:
            endpoints = [
                "/api/v1/analytics/health",
                "/api/v1/monitoring/health",
                "/api/v1/analytics/overview",
                "/api/v1/monitoring/resources",
                "/api/v1/plugins/"
            ]
        
        print(f"⏱️  Starting response time test: {len(endpoints)} endpoints, {samples_per_endpoint} samples each")
        
        results = {}
        
        import aiohttp
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for endpoint in endpoints:
                endpoint_results = []
                
                for i in range(samples_per_endpoint):
                    start_time = time.time()
                    
                    try:
                        async with session.get(f"http://localhost:8000{endpoint}") as response:
                            await response.text()
                            duration = time.time() - start_time
                            
                            endpoint_results.append({
                                'sample': i + 1,
                                'duration_ms': duration * 1000,
                                'status_code': response.status,
                                'success': response.status < 500
                            })
                            
                    except Exception as e:
                        duration = time.time() - start_time
                        endpoint_results.append({
                            'sample': i + 1,
                            'duration_ms': duration * 1000,
                            'status_code': 0,
                            'success': False,
                            'error': str(e)
                        })
                    
                    # Small delay between samples
                    await asyncio.sleep(0.1)
                
                # Calculate endpoint statistics
                successful_samples = [r for r in endpoint_results if r['success']]
                
                if successful_samples:
                    durations = [r['duration_ms'] for r in successful_samples]
                    
                    results[endpoint] = {
                        'total_samples': len(endpoint_results),
                        'successful_samples': len(successful_samples),
                        'success_rate': len(successful_samples) / len(endpoint_results),
                        'avg_response_time_ms': statistics.mean(durations),
                        'median_response_time_ms': statistics.median(durations),
                        'min_response_time_ms': min(durations),
                        'max_response_time_ms': max(durations),
                        'p95_response_time_ms': load_tester._percentile(durations, 95),
                        'p99_response_time_ms': load_tester._percentile(durations, 99)
                    }
                else:
                    results[endpoint] = {
                        'total_samples': len(endpoint_results),
                        'successful_samples': 0,
                        'success_rate': 0,
                        'error': 'No successful requests'
                    }
        
        return {
            "test_config": {
                "endpoints": endpoints,
                "samples_per_endpoint": samples_per_endpoint
            },
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "test_type": "response_time_test"
        }
    
    @app.get("/benchmark/system-info")
    async def get_system_info():
        """Get current system information"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "system_info": {
                "cpu_cores": cpu_count,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_percent": disk.percent
            },
            "timestamp": datetime.now().isoformat()
        }
    
    return app


async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    
    print("🚀 Xencode Performance Benchmarking Demo")
    print("=" * 50)
    
    print("\\n📊 Performance Benchmarking Features:")
    print("  ✅ Real-time system monitoring")
    print("  ✅ Concurrent user load testing")
    print("  ✅ Throughput benchmarking")
    print("  ✅ Response time analysis")
    print("  ✅ Resource utilization tracking")
    print("  ✅ Cache performance testing")
    print("  ✅ Streaming metrics")
    
    print("\\n🎯 Available Benchmark Tests:")
    
    benchmarks = [
        {
            "name": "Load Testing",
            "endpoint": "POST /benchmark/load-test",
            "description": "Test system under concurrent user load",
            "parameters": "concurrent_users, requests_per_user, target_endpoints"
        },
        {
            "name": "Throughput Testing", 
            "endpoint": "POST /benchmark/throughput",
            "description": "Measure requests per second capacity",
            "parameters": "target_rps, duration_seconds, endpoint"
        },
        {
            "name": "Response Time Testing",
            "endpoint": "POST /benchmark/response-time", 
            "description": "Analyze response time distribution",
            "parameters": "endpoints, samples_per_endpoint"
        },
        {
            "name": "System Monitoring",
            "endpoint": "POST /benchmark/monitor/start",
            "description": "Real-time resource monitoring",
            "parameters": "None (background monitoring)"
        }
    ]
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\\n  {i}. {benchmark['name']}")
        print(f"     Endpoint: {benchmark['endpoint']}")
        print(f"     Description: {benchmark['description']}")
        print(f"     Parameters: {benchmark['parameters']}")
    
    print("\\n📈 Performance Metrics Collected:")
    print("  • Response time (avg, median, p95, p99)")
    print("  • Throughput (requests per second)")
    print("  • Success rate and error analysis")
    print("  • CPU and memory utilization")
    print("  • Disk usage and I/O")
    print("  • Network statistics")
    print("  • Cache hit rates and performance")
    
    print("\\n🔧 Example Usage:")
    print("  1. Start monitoring: POST /benchmark/monitor/start")
    print("  2. Run load test: POST /benchmark/load-test")
    print("     {")
    print('       "concurrent_users": 20,')
    print('       "requests_per_user": 10')
    print("     }")
    print("  3. Check throughput: POST /benchmark/throughput")
    print("     {")
    print('       "target_rps": 100,')
    print('       "duration_seconds": 30')
    print("     }")
    print("  4. Stop monitoring: POST /benchmark/monitor/stop")
    
    print("\\n📊 Performance Targets:")
    print("  • Response Time: < 500ms average")
    print("  • Throughput: > 50 RPS")
    print("  • Success Rate: > 95%")
    print("  • CPU Usage: < 80% under load")
    print("  • Memory Usage: < 85% under load")
    print("  • Cache Hit Rate: > 80%")


def main():
    """Main demo function"""
    
    # Run the demo
    asyncio.run(demo_performance_benchmarking())
    
    print("\\n🚀 Starting Performance Benchmarking Server...")
    print("📖 Interactive API Docs: http://localhost:8000/docs")
    print("🎯 Load Test: POST http://localhost:8000/benchmark/load-test")
    print("📊 Throughput Test: POST http://localhost:8000/benchmark/throughput")
    print("⏱️  Response Time Test: POST http://localhost:8000/benchmark/response-time")
    print("📈 Start Monitoring: POST http://localhost:8000/benchmark/monitor/start")
    print("🖥️  System Info: GET http://localhost:8000/benchmark/system-info")
    print("⏹️  Press Ctrl+C to stop")
    
    # Create and run the benchmarking app
    app = create_benchmarking_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )


if __name__ == "__main__":
    main()