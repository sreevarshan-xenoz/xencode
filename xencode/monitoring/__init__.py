#!/usr/bin/env python3
"""
Monitoring Package

Provides system monitoring, health checks, and observability features
for comprehensive system monitoring and alerting.
"""

from typing import Optional

# Import monitoring components with graceful fallback
try:
    from .metrics_collector import PrometheusMetricsCollector
    PROMETHEUS_METRICS_AVAILABLE = True
except ImportError:
    PrometheusMetricsCollector = None
    PROMETHEUS_METRICS_AVAILABLE = False

try:
    from .health_monitor import HealthMonitor
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HealthMonitor = None
    HEALTH_MONITOR_AVAILABLE = False

try:
    from .alert_manager import AlertManager
    ALERT_MANAGER_AVAILABLE = True
except ImportError:
    AlertManager = None
    ALERT_MANAGER_AVAILABLE = False

try:
    from .performance_optimizer import PerformanceOptimizer
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PerformanceOptimizer = None
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from .resource_manager import ResourceManager, get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    ResourceManager = None
    get_resource_manager = None
    RESOURCE_MANAGER_AVAILABLE = False

# Benchmark components
try:
    from .benchmark_engine import (
        BenchmarkEngine,
        BenchmarkTask,
        BenchmarkResult,
        BenchmarkMetrics,
        TaskType,
        create_benchmark_engine,
        run_benchmark,
    )
    BENCHMARK_ENGINE_AVAILABLE = True
except ImportError:
    BenchmarkEngine = None
    BenchmarkTask = None
    BenchmarkResult = None
    BenchmarkMetrics = None
    TaskType = None
    create_benchmark_engine = None
    run_benchmark = None
    BENCHMARK_ENGINE_AVAILABLE = False

try:
    from .benchmark_store import (
        BenchmarkStore,
        BenchmarkRecord,
        BenchmarkQuery,
        create_benchmark_store,
    )
    BENCHMARK_STORE_AVAILABLE = True
except ImportError:
    BenchmarkStore = None
    BenchmarkRecord = None
    BenchmarkQuery = None
    create_benchmark_store = None
    BENCHMARK_STORE_AVAILABLE = False

try:
    from .benchmark_suites import (
        BenchmarkSuites,
        BenchmarkDataset,
        create_benchmark_suites,
    )
    BENCHMARK_SUITES_AVAILABLE = True
except ImportError:
    BenchmarkSuites = None
    BenchmarkDataset = None
    create_benchmark_suites = None
    BENCHMARK_SUITES_AVAILABLE = False

try:
    from .benchmark_recommendations import (
        BenchmarkRecommendations,
        ModelRecommendation,
        generate_recommendations,
    )
    BENCHMARK_RECOMMENDATIONS_AVAILABLE = True
except ImportError:
    BenchmarkRecommendations = None
    ModelRecommendation = None
    generate_recommendations = None
    BENCHMARK_RECOMMENDATIONS_AVAILABLE = False


def get_monitoring_status() -> dict:
    """Get status of monitoring components"""
    return {
        "prometheus_metrics_available": PROMETHEUS_METRICS_AVAILABLE,
        "health_monitor_available": HEALTH_MONITOR_AVAILABLE,
        "alert_manager_available": ALERT_MANAGER_AVAILABLE,
        "performance_optimizer_available": PERFORMANCE_OPTIMIZER_AVAILABLE,
        "resource_manager_available": RESOURCE_MANAGER_AVAILABLE,
        "benchmark_engine_available": BENCHMARK_ENGINE_AVAILABLE,
        "benchmark_store_available": BENCHMARK_STORE_AVAILABLE,
        "benchmark_suites_available": BENCHMARK_SUITES_AVAILABLE,
        "benchmark_recommendations_available": BENCHMARK_RECOMMENDATIONS_AVAILABLE,
    }


__all__ = [
    # Core Monitoring
    'PrometheusMetricsCollector',
    'HealthMonitor',
    'AlertManager',
    'PerformanceOptimizer',
    'ResourceManager',
    'get_resource_manager',
    # Benchmark Engine
    'BenchmarkEngine',
    'BenchmarkTask',
    'BenchmarkResult',
    'BenchmarkMetrics',
    'TaskType',
    'create_benchmark_engine',
    'run_benchmark',
    # Benchmark Store
    'BenchmarkStore',
    'BenchmarkRecord',
    'BenchmarkQuery',
    'create_benchmark_store',
    # Benchmark Suites
    'BenchmarkSuites',
    'BenchmarkDataset',
    'create_benchmark_suites',
    # Benchmark Recommendations
    'BenchmarkRecommendations',
    'ModelRecommendation',
    'generate_recommendations',
    # Status
    'get_monitoring_status',
    # Availability Flags
    'PROMETHEUS_METRICS_AVAILABLE',
    'HEALTH_MONITOR_AVAILABLE',
    'ALERT_MANAGER_AVAILABLE',
    'PERFORMANCE_OPTIMIZER_AVAILABLE',
    'RESOURCE_MANAGER_AVAILABLE',
    'BENCHMARK_ENGINE_AVAILABLE',
    'BENCHMARK_STORE_AVAILABLE',
    'BENCHMARK_SUITES_AVAILABLE',
    'BENCHMARK_RECOMMENDATIONS_AVAILABLE',
]