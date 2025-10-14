#!/usr/bin/env python3
"""
Prometheus Metrics Collector

Specialized metrics collector focused on Prometheus integration
for system monitoring and observability.
"""

import asyncio
import psutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, start_http_server,
        CONTENT_TYPE_LATEST, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    disk_usage_percent: float
    disk_used_bytes: int
    disk_total_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    uptime_seconds: float


class PrometheusMetricsCollector:
    """
    Prometheus-focused metrics collector for system monitoring
    
    Provides comprehensive system metrics collection with Prometheus
    integration for monitoring and alerting.
    """
    
    def __init__(self, 
                 registry: Optional[Any] = None,
                 metrics_port: int = 8000,
                 collect_system_metrics: bool = True):
        
        self.registry = registry or REGISTRY if PROMETHEUS_AVAILABLE else None
        self.metrics_port = metrics_port
        self.collect_system_metrics = collect_system_metrics
        
        # Metrics
        self.metrics = {}
        
        # System monitoring
        self._system_start_time = time.time()
        self._last_network_stats = None
        
        # Background tasks
        self._collection_task = None
        self._running = False
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_prometheus_metrics()
    
    def _initialize_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics"""
        
        # System metrics
        if self.collect_system_metrics:
            self.metrics['cpu_usage'] = Gauge(
                'xencode_cpu_usage_percent',
                'CPU usage percentage',
                ['component'],
                registry=self.registry
            )
            
            self.metrics['memory_usage'] = Gauge(
                'xencode_memory_usage_bytes',
                'Memory usage in bytes',
                ['type'],
                registry=self.registry
            )
            
            self.metrics['memory_percent'] = Gauge(
                'xencode_memory_usage_percent',
                'Memory usage percentage',
                registry=self.registry
            )
            
            self.metrics['disk_usage'] = Gauge(
                'xencode_disk_usage_bytes',
                'Disk usage in bytes',
                ['type'],
                registry=self.registry
            )
            
            self.metrics['disk_percent'] = Gauge(
                'xencode_disk_usage_percent',
                'Disk usage percentage',
                registry=self.registry
            )
            
            self.metrics['network_bytes'] = Counter(
                'xencode_network_bytes_total',
                'Network bytes transferred',
                ['direction'],
                registry=self.registry
            )
            
            self.metrics['load_average'] = Gauge(
                'xencode_load_average',
                'System load average',
                ['period'],
                registry=self.registry
            )
            
            self.metrics['uptime'] = Gauge(
                'xencode_uptime_seconds',
                'System uptime in seconds',
                registry=self.registry
            )
        
        # Application metrics
        self.metrics['requests_total'] = Counter(
            'xencode_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['request_duration'] = Histogram(
            'xencode_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.metrics['active_connections'] = Gauge(
            'xencode_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.metrics['plugin_status'] = Gauge(
            'xencode_plugin_status',
            'Plugin status (1=loaded, 0=unloaded)',
            ['plugin_name'],
            registry=self.registry
        )
        
        self.metrics['workspace_operations'] = Counter(
            'xencode_workspace_operations_total',
            'Total workspace operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.metrics['ai_model_requests'] = Counter(
            'xencode_ai_model_requests_total',
            'Total AI model requests',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.metrics['ai_model_duration'] = Histogram(
            'xencode_ai_model_duration_seconds',
            'AI model request duration',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            'xencode_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.metrics['cache_operations'] = Counter(
            'xencode_cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.metrics['cache_size'] = Gauge(
            'xencode_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
    
    async def start(self) -> None:
        """Start metrics collection"""
        if self._running or not PROMETHEUS_AVAILABLE:
            return
        
        self._running = True
        
        # Start HTTP server for metrics endpoint
        try:
            start_http_server(self.metrics_port, registry=self.registry)
            print(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            print(f"Failed to start Prometheus server: {e}")
        
        # Start collection task
        if self.collect_system_metrics:
            self._collection_task = asyncio.create_task(self._collection_loop())
        
        print("PrometheusMetricsCollector started")
    
    async def stop(self) -> None:
        """Stop metrics collection"""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        print("PrometheusMetricsCollector stopped")
    
    async def _collection_loop(self) -> None:
        """Background loop for collecting system metrics"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(15)  # Collect every 15 seconds
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics"""
        try:
            # Get system metrics
            system_metrics = self._get_system_metrics()
            
            # Update Prometheus metrics
            if 'cpu_usage' in self.metrics:
                self.metrics['cpu_usage'].labels(component='system').set(system_metrics.cpu_percent)
            
            if 'memory_usage' in self.metrics:
                self.metrics['memory_usage'].labels(type='used').set(system_metrics.memory_used_bytes)
                self.metrics['memory_usage'].labels(type='total').set(system_metrics.memory_total_bytes)
            
            if 'memory_percent' in self.metrics:
                self.metrics['memory_percent'].set(system_metrics.memory_percent)
            
            if 'disk_usage' in self.metrics:
                self.metrics['disk_usage'].labels(type='used').set(system_metrics.disk_used_bytes)
                self.metrics['disk_usage'].labels(type='total').set(system_metrics.disk_total_bytes)
            
            if 'disk_percent' in self.metrics:
                self.metrics['disk_percent'].set(system_metrics.disk_usage_percent)
            
            if 'network_bytes' in self.metrics and self._last_network_stats:
                # Calculate network delta
                sent_delta = system_metrics.network_bytes_sent - self._last_network_stats[0]
                recv_delta = system_metrics.network_bytes_recv - self._last_network_stats[1]
                
                if sent_delta >= 0:  # Handle counter resets
                    self.metrics['network_bytes'].labels(direction='sent').inc(sent_delta)
                if recv_delta >= 0:
                    self.metrics['network_bytes'].labels(direction='received').inc(recv_delta)
            
            self._last_network_stats = (system_metrics.network_bytes_sent, system_metrics.network_bytes_recv)
            
            if 'load_average' in self.metrics and system_metrics.load_average:
                if len(system_metrics.load_average) >= 3:
                    self.metrics['load_average'].labels(period='1m').set(system_metrics.load_average[0])
                    self.metrics['load_average'].labels(period='5m').set(system_metrics.load_average[1])
                    self.metrics['load_average'].labels(period='15m').set(system_metrics.load_average[2])
            
            if 'uptime' in self.metrics:
                self.metrics['uptime'].set(system_metrics.uptime_seconds)
                
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
        except (AttributeError, OSError):
            load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
        
        # Uptime
        uptime = time.time() - self._system_start_time
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_bytes=memory.used,
            memory_total_bytes=memory.total,
            disk_usage_percent=disk.percent,
            disk_used_bytes=disk.used,
            disk_total_bytes=disk.total,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            load_average=list(load_avg),
            uptime_seconds=uptime
        )
    
    # Public API methods for application metrics
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float) -> None:
        """Record HTTP request metrics"""
        if 'requests_total' in self.metrics:
            self.metrics['requests_total'].labels(method=method, endpoint=endpoint, status=status).inc()
        
        if 'request_duration' in self.metrics:
            self.metrics['request_duration'].labels(method=method, endpoint=endpoint).observe(duration)
    
    def set_active_connections(self, count: int) -> None:
        """Set number of active connections"""
        if 'active_connections' in self.metrics:
            self.metrics['active_connections'].set(count)
    
    def set_plugin_status(self, plugin_name: str, loaded: bool) -> None:
        """Set plugin status"""
        if 'plugin_status' in self.metrics:
            self.metrics['plugin_status'].labels(plugin_name=plugin_name).set(1 if loaded else 0)
    
    def record_workspace_operation(self, operation: str, status: str) -> None:
        """Record workspace operation"""
        if 'workspace_operations' in self.metrics:
            self.metrics['workspace_operations'].labels(operation=operation, status=status).inc()
    
    def record_ai_request(self, model: str, status: str, duration: float) -> None:
        """Record AI model request"""
        if 'ai_model_requests' in self.metrics:
            self.metrics['ai_model_requests'].labels(model=model, status=status).inc()
        
        if 'ai_model_duration' in self.metrics:
            self.metrics['ai_model_duration'].labels(model=model).observe(duration)
    
    def record_error(self, component: str, error_type: str) -> None:
        """Record error occurrence"""
        if 'errors_total' in self.metrics:
            self.metrics['errors_total'].labels(component=component, error_type=error_type).inc()
    
    def record_cache_operation(self, operation: str, result: str) -> None:
        """Record cache operation"""
        if 'cache_operations' in self.metrics:
            self.metrics['cache_operations'].labels(operation=operation, result=result).inc()
    
    def set_cache_size(self, cache_type: str, size_bytes: int) -> None:
        """Set cache size"""
        if 'cache_size' in self.metrics:
            self.metrics['cache_size'].labels(cache_type=cache_type).set(size_bytes)
    
    def get_metrics_endpoint(self) -> str:
        """Get metrics endpoint URL"""
        return f"http://localhost:{self.metrics_port}/metrics"
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"
    
    def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        return {
            'running': self._running,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'metrics_port': self.metrics_port,
            'collect_system_metrics': self.collect_system_metrics,
            'metrics_endpoint': self.get_metrics_endpoint() if self._running else None,
            'registered_metrics': list(self.metrics.keys()) if self.metrics else []
        }


# Global Prometheus metrics collector instance
prometheus_metrics_collector = PrometheusMetricsCollector()