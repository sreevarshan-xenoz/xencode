#!/usr/bin/env python3
"""
Monitoring Package

Provides system monitoring, health checks, and observability features
for comprehensive system monitoring and alerting.
"""

import logging

logger = logging.getLogger(__name__)
from typing import Optional

# Import monitoring components with graceful fallback
try:
    from .metrics_collector import PrometheusMetricsCollector
    PROMETHEUS_METRICS_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import PrometheusMetricsCollector: %s", e)
    PrometheusMetricsCollector = None
    PROMETHEUS_METRICS_AVAILABLE = False

try:
    from .performance_optimizer import PerformanceOptimizer
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import PerformanceOptimizer: %s", e)
    PerformanceOptimizer = None
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from .resource_manager import ResourceManager, get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import ResourceManager: %s", e)
    ResourceManager = None
    get_resource_manager = None
    RESOURCE_MANAGER_AVAILABLE = False
# Benchmark components
try:
    from .benchmark_engine import (
        BenchmarkEngine,
        BenchmarkMetrics,
        BenchmarkResult,
        BenchmarkTask,
        TaskType,
        create_benchmark_engine,
        run_benchmark,
    )
    BENCHMARK_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import BenchmarkEngine: %s", e)
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
        BenchmarkQuery,
        BenchmarkRecord,
        BenchmarkStore,
        create_benchmark_store,
    )
    BENCHMARK_STORE_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import BenchmarkStore: %s", e)
    BenchmarkStore = None
    BenchmarkRecord = None
    BenchmarkQuery = None
    create_benchmark_store = None
    BENCHMARK_STORE_AVAILABLE = False

try:
    from .benchmark_suites import (
        BenchmarkDataset,
        BenchmarkSuites,
        create_benchmark_suites,
    )
    BENCHMARK_SUITES_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import BenchmarkSuites: %s", e)
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
except ImportError as e:
    logger.warning("Failed to import BenchmarkRecommendations: %s", e)
    BenchmarkRecommendations = None
    ModelRecommendation = None
    generate_recommendations = None
    BENCHMARK_RECOMMENDATIONS_AVAILABLE = False

# Backward compatibility aliases
HealthMonitor = ResourceManager
AlertManager = None


def get_monitoring_status() -> dict:
    """Get status of monitoring components"""
    return {
        "prometheus_metrics_available": PROMETHEUS_METRICS_AVAILABLE,
        "health_monitor_available": RESOURCE_MANAGER_AVAILABLE,
        "alert_manager_available": False,
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
    'PerformanceOptimizer',
    'ResourceManager',
    'get_resource_manager',
    'HealthMonitor',
    'AlertManager',
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
    'PERFORMANCE_OPTIMIZER_AVAILABLE',
    'RESOURCE_MANAGER_AVAILABLE',
    'BENCHMARK_ENGINE_AVAILABLE',
    'BENCHMARK_STORE_AVAILABLE',
    'BENCHMARK_SUITES_AVAILABLE',
    'BENCHMARK_RECOMMENDATIONS_AVAILABLE',
]
