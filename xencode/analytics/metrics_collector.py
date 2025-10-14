#!/usr/bin/env python3
"""
Metrics Collector

Implements comprehensive metrics collection with Prometheus integration
for real-time monitoring and observability.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
from pathlib import Path

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, start_http_server, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def labels(self, *args, **kwargs): return self


class MetricType(str, Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries


@dataclass
class AnalyticsEvent:
    """Represents an analytics event"""
    event_type: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collector with Prometheus integration
    
    Collects system metrics, user analytics, and performance data
    with real-time aggregation and export capabilities.
    """
    
    def __init__(self, 
                 registry: Optional[Any] = None,
                 enable_prometheus: bool = True,
                 metrics_port: int = 8000,
                 storage_path: Optional[Path] = None):
        
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_port = metrics_port
        self.storage_path = storage_path or Path.home() / ".xencode" / "analytics"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Prometheus registry
        if self.enable_prometheus:
            self.registry = registry or CollectorRegistry()
        else:
            self.registry = None
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Event storage
        self.events: List[AnalyticsEvent] = []
        self.event_callbacks: List[Callable[[AnalyticsEvent], None]] = []
        
        # Aggregation settings
        self.aggregation_interval = 60  # seconds
        self.max_events_in_memory = 10000
        self.max_metrics_age_days = 30
        
        # Background tasks
        self._aggregation_task = None
        self._cleanup_task = None
        self._running = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics"""
        
        # System performance metrics
        self.register_metric(MetricDefinition(
            name="xencode_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total number of requests processed",
            labels=["method", "endpoint", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_request_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Request duration in seconds",
            labels=["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_active_users",
            metric_type=MetricType.GAUGE,
            description="Number of currently active users"
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_memory_usage_bytes",
            metric_type=MetricType.GAUGE,
            description="Memory usage in bytes",
            labels=["component"]
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_cpu_usage_percent",
            metric_type=MetricType.GAUGE,
            description="CPU usage percentage",
            labels=["component"]
        ))
        
        # AI model metrics
        self.register_metric(MetricDefinition(
            name="xencode_ai_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total AI model requests",
            labels=["model", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_ai_response_time_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="AI model response time in seconds",
            labels=["model"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        ))
        
        # Plugin metrics
        self.register_metric(MetricDefinition(
            name="xencode_plugins_loaded",
            metric_type=MetricType.GAUGE,
            description="Number of loaded plugins"
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_plugin_errors_total",
            metric_type=MetricType.COUNTER,
            description="Total plugin errors",
            labels=["plugin", "error_type"]
        ))
        
        # Workspace metrics
        self.register_metric(MetricDefinition(
            name="xencode_workspaces_active",
            metric_type=MetricType.GAUGE,
            description="Number of active workspaces"
        ))
        
        self.register_metric(MetricDefinition(
            name="xencode_workspace_operations_total",
            metric_type=MetricType.COUNTER,
            description="Total workspace operations",
            labels=["operation", "status"]
        ))
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition"""
        with self._lock:
            self.metric_definitions[definition.name] = definition
            
            if self.enable_prometheus:
                if definition.metric_type == MetricType.COUNTER:
                    metric = Counter(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.GAUGE:
                    metric = Gauge(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.HISTOGRAM:
                    metric = Histogram(
                        definition.name,
                        definition.description,
                        definition.labels,
                        buckets=definition.buckets,
                        registry=self.registry
                    )
                elif definition.metric_type == MetricType.SUMMARY:
                    metric = Summary(
                        definition.name,
                        definition.description,
                        definition.labels,
                        registry=self.registry
                    )
                
                self.metrics[definition.name] = metric
    
    def increment_counter(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            
            # Store for non-Prometheus backends
            self._store_metric_value(metric_name, value, labels, "increment")
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        with self._lock:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            
            # Store for non-Prometheus backends
            self._store_metric_value(metric_name, value, labels, "set")
    
    def observe_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a histogram metric"""
        with self._lock:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            
            # Store for non-Prometheus backends
            self._store_metric_value(metric_name, value, labels, "observe")
    
    def time_histogram(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations with histogram"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                return metric.labels(**labels).time()
            else:
                return metric.time()
        else:
            # Fallback timer
            return self._FallbackTimer(self, metric_name, labels)
    
    class _FallbackTimer:
        """Fallback timer when Prometheus is not available"""
        
        def __init__(self, collector, metric_name: str, labels: Optional[Dict[str, str]]):
            self.collector = collector
            self.metric_name = metric_name
            self.labels = labels
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                self.collector._store_metric_value(self.metric_name, duration, self.labels, "observe")
    
    def _store_metric_value(self, metric_name: str, value: float, labels: Optional[Dict[str, str]], operation: str) -> None:
        """Store metric value for non-Prometheus backends"""
        timestamp = datetime.now()
        
        # Create metric entry
        metric_entry = {
            'timestamp': timestamp.isoformat(),
            'metric_name': metric_name,
            'value': value,
            'labels': labels or {},
            'operation': operation
        }
        
        # Store in memory (with rotation)
        if not hasattr(self, '_metric_history'):
            self._metric_history = []
        
        self._metric_history.append(metric_entry)
        
        # Rotate if too many entries
        if len(self._metric_history) > self.max_events_in_memory:
            self._metric_history = self._metric_history[-self.max_events_in_memory//2:]
    
    def track_event(self, event: AnalyticsEvent) -> None:
        """Track an analytics event"""
        with self._lock:
            self.events.append(event)
            
            # Rotate events if too many
            if len(self.events) > self.max_events_in_memory:
                self.events = self.events[-self.max_events_in_memory//2:]
            
            # Call event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
    
    def add_event_callback(self, callback: Callable[[AnalyticsEvent], None]) -> None:
        """Add callback for analytics events"""
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[AnalyticsEvent], None]) -> None:
        """Remove event callback"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    async def start(self) -> None:
        """Start the metrics collector"""
        if self._running:
            return
        
        self._running = True
        
        # Start Prometheus HTTP server
        if self.enable_prometheus:
            try:
                start_http_server(self.metrics_port, registry=self.registry)
                print(f"Prometheus metrics server started on port {self.metrics_port}")
            except Exception as e:
                print(f"Failed to start Prometheus server: {e}")
        
        # Start background tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        print("MetricsCollector started")
    
    async def stop(self) -> None:
        """Stop the metrics collector"""
        self._running = False
        
        # Cancel background tasks
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save metrics to disk
        await self._save_metrics_to_disk()
        
        print("MetricsCollector stopped")
    
    async def _aggregation_loop(self) -> None:
        """Background loop for metrics aggregation"""
        while self._running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_interval)
            except Exception as e:
                print(f"Error in aggregation loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup tasks"""
        while self._running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics for storage and analysis"""
        with self._lock:
            # Calculate aggregated metrics
            current_time = datetime.now()
            
            # Example aggregations
            if hasattr(self, '_metric_history'):
                recent_metrics = [
                    m for m in self._metric_history
                    if datetime.fromisoformat(m['timestamp']) > current_time - timedelta(minutes=5)
                ]
                
                # Store aggregated data
                aggregated_data = {
                    'timestamp': current_time.isoformat(),
                    'total_metrics': len(recent_metrics),
                    'unique_metric_names': len(set(m['metric_name'] for m in recent_metrics)),
                    'events_count': len(self.events)
                }
                
                # Save aggregated data
                await self._save_aggregated_data(aggregated_data)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and events"""
        cutoff_date = datetime.now() - timedelta(days=self.max_metrics_age_days)
        
        with self._lock:
            # Clean up old events
            self.events = [
                event for event in self.events
                if event.timestamp > cutoff_date
            ]
            
            # Clean up old metric history
            if hasattr(self, '_metric_history'):
                self._metric_history = [
                    metric for metric in self._metric_history
                    if datetime.fromisoformat(metric['timestamp']) > cutoff_date
                ]
    
    async def _save_metrics_to_disk(self) -> None:
        """Save current metrics to disk"""
        try:
            metrics_file = self.storage_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'metric_definitions': {
                    name: {
                        'name': defn.name,
                        'type': defn.metric_type.value,
                        'description': defn.description,
                        'labels': defn.labels
                    }
                    for name, defn in self.metric_definitions.items()
                },
                'events': [
                    {
                        'event_type': event.event_type,
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'session_id': event.session_id,
                        'properties': event.properties,
                        'metrics': event.metrics
                    }
                    for event in self.events[-1000:]  # Save last 1000 events
                ],
                'metric_history': getattr(self, '_metric_history', [])[-1000:]  # Save last 1000 metrics
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metrics to disk: {e}")
    
    async def _save_aggregated_data(self, data: Dict[str, Any]) -> None:
        """Save aggregated data"""
        try:
            aggregated_file = self.storage_path / "aggregated_metrics.jsonl"
            
            with open(aggregated_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
                
        except Exception as e:
            print(f"Error saving aggregated data: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        with self._lock:
            return {
                'total_metric_definitions': len(self.metric_definitions),
                'total_events': len(self.events),
                'prometheus_enabled': self.enable_prometheus,
                'metrics_port': self.metrics_port if self.enable_prometheus else None,
                'storage_path': str(self.storage_path),
                'running': self._running,
                'recent_events': len([
                    e for e in self.events
                    if e.timestamp > datetime.now() - timedelta(hours=1)
                ])
            }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if self.enable_prometheus and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[AnalyticsEvent]:
        """Get events by type"""
        with self._lock:
            matching_events = [e for e in self.events if e.event_type == event_type]
            return matching_events[-limit:]
    
    def get_metrics_by_name(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metric history by name"""
        with self._lock:
            if hasattr(self, '_metric_history'):
                matching_metrics = [
                    m for m in self._metric_history
                    if m['metric_name'] == metric_name
                ]
                return matching_metrics[-limit:]
            return []


# Global metrics collector instance
metrics_collector = MetricsCollector()