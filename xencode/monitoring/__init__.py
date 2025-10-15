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


def get_monitoring_status() -> dict:
    """Get status of monitoring components"""
    return {
        "prometheus_metrics_available": PROMETHEUS_METRICS_AVAILABLE,
        "health_monitor_available": HEALTH_MONITOR_AVAILABLE,
        "alert_manager_available": ALERT_MANAGER_AVAILABLE,
        "performance_optimizer_available": PERFORMANCE_OPTIMIZER_AVAILABLE,
        "resource_manager_available": RESOURCE_MANAGER_AVAILABLE
    }


__all__ = [
    'PrometheusMetricsCollector',
    'HealthMonitor',
    'AlertManager',
    'PerformanceOptimizer',
    'ResourceManager',
    'get_resource_manager',
    'get_monitoring_status',
    'PROMETHEUS_METRICS_AVAILABLE',
    'HEALTH_MONITOR_AVAILABLE',
    'ALERT_MANAGER_AVAILABLE',
    'PERFORMANCE_OPTIMIZER_AVAILABLE',
    'RESOURCE_MANAGER_AVAILABLE'
]