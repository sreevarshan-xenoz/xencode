#!/usr/bin/env python3
"""
Analytics Package

Provides comprehensive analytics and monitoring capabilities for Xencode,
including metrics collection, performance monitoring, and data visualization.
"""

from typing import Optional

# Import analytics components with graceful fallback
try:
    from .metrics_collector import MetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    MetricsCollector = None
    METRICS_COLLECTOR_AVAILABLE = False

try:
    from .event_tracker import EventTracker
    EVENT_TRACKER_AVAILABLE = True
except ImportError:
    EventTracker = None
    EVENT_TRACKER_AVAILABLE = False

try:
    from .analytics_engine import AnalyticsEngine
    ANALYTICS_ENGINE_AVAILABLE = True
except ImportError:
    AnalyticsEngine = None
    ANALYTICS_ENGINE_AVAILABLE = False


def get_analytics_status() -> dict:
    """Get status of analytics components"""
    return {
        "metrics_collector_available": METRICS_COLLECTOR_AVAILABLE,
        "event_tracker_available": EVENT_TRACKER_AVAILABLE,
        "analytics_engine_available": ANALYTICS_ENGINE_AVAILABLE
    }


__all__ = [
    'MetricsCollector',
    'EventTracker',
    'AnalyticsEngine',
    'get_analytics_status',
    'METRICS_COLLECTOR_AVAILABLE',
    'EVENT_TRACKER_AVAILABLE',
    'ANALYTICS_ENGINE_AVAILABLE'
]