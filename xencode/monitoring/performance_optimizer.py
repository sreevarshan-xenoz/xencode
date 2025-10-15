#!/usr/bin/env python3
"""
Performance Optimizer and Alert System

Automated performance monitoring with intelligent optimization and alerting
for all Xencode components including document processing, code analysis,
and workspace management.
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import deque, defaultdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(str, Enum):
    """Types of performance metrics"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    duration_seconds: int = 60  # How long threshold must be exceeded
    enabled: bool = True


@dataclass
class PerformanceAlert:
    """Performance alert with context"""
    alert_id: str
    metric_type: MetricType
    severity: AlertSeverity
    title: str
    description: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    component: str = "system"
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Automated optimization action"""
    action_id: str
    action_type: str
    description: str
    target_component: str
    parameters: Dict[str, Any]
    estimated_impact: str
    risk_level: str = "low"
    executed: bool = False
    executed_at: Optional[datetime] = None
    result: Optional[str] = None


class PerformanceMetricsCollector:
    """Collects performance metrics from all system components"""
    
    def __init__(self):
        self.metrics_history: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collection_interval = 5  # seconds
        self.running = False
    
    async def collect_system_metrics(self) -> Dict[MetricType, float]:
        """Collect current system performance metrics"""
        metrics = {}
        
        if PSUTIL_AVAILABLE:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics[MetricType.CPU_USAGE] = cpu_percent
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics[MetricType.MEMORY_USAGE] = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics[MetricType.DISK_USAGE] = (disk.used / disk.total) * 100
        
        return metrics
    
    async def collect_application_metrics(self) -> Dict[MetricType, float]:
        """Collect application-specific performance metrics"""
        metrics = {}
        
        try:
            # Cache performance
            from ..cache.multimodal_cache import get_multimodal_cache
            cache_system = await get_multimodal_cache()
            cache_stats = await cache_system.get_cache_statistics()
            
            base_stats = cache_stats.get('base_cache', {})
            metrics[MetricType.CACHE_HIT_RATE] = base_stats.get('hit_rate', 0)
            
        except Exception as e:
            logger.warning(f"Failed to collect cache metrics: {e}")
        
        return metrics
    
    async def start_collection(self):
        """Start continuous metrics collection"""
        self.running = True
        
        while self.running:
            try:
                # Collect system metrics
                system_metrics = await self.collect_system_metrics()
                application_metrics = await self.collect_application_metrics()
                
                # Combine metrics
                all_metrics = {**system_metrics, **application_metrics}
                
                # Store in history
                timestamp = time.time()
                for metric_type, value in all_metrics.items():
                    self.metrics_history[metric_type].append((timestamp, value))
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval * 2)
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
    
    def get_recent_metrics(self, metric_type: MetricType, 
                          duration_minutes: int = 5) -> List[Tuple[float, float]]:
        """Get recent metrics for a specific type"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        if metric_type not in self.metrics_history:
            return []
        
        return [(ts, value) for ts, value in self.metrics_history[metric_type] 
                if ts >= cutoff_time]
    
    def calculate_trend(self, metric_type: MetricType, 
                       duration_minutes: int = 10) -> Optional[str]:
        """Calculate trend for a metric (increasing, decreasing, stable)"""
        recent_data = self.get_recent_metrics(metric_type, duration_minutes)
        
        if len(recent_data) < 5:
            return None
        
        values = [value for _, value in recent_data]
        
        # Simple trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"


class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Default thresholds
        self.thresholds = {
            MetricType.CPU_USAGE: PerformanceThreshold(
                MetricType.CPU_USAGE, 70.0, 85.0, 95.0, 60
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                MetricType.MEMORY_USAGE, 75.0, 90.0, 95.0, 60
            ),
            MetricType.DISK_USAGE: PerformanceThreshold(
                MetricType.DISK_USAGE, 80.0, 90.0, 95.0, 300
            ),
            MetricType.RESPONSE_TIME: PerformanceThreshold(
                MetricType.RESPONSE_TIME, 1000.0, 5000.0, 10000.0, 30
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                MetricType.ERROR_RATE, 1.0, 5.0, 10.0, 60
            ),
            MetricType.CACHE_HIT_RATE: PerformanceThreshold(
                MetricType.CACHE_HIT_RATE, 80.0, 60.0, 40.0, 120  # Lower is worse for hit rate
            )
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def update_threshold(self, metric_type: MetricType, threshold: PerformanceThreshold):
        """Update performance threshold"""
        self.thresholds[metric_type] = threshold
    
    async def check_thresholds(self, metrics: Dict[MetricType, float]):
        """Check metrics against thresholds and generate alerts"""
        for metric_type, current_value in metrics.items():
            if metric_type not in self.thresholds:
                continue
            
            threshold = self.thresholds[metric_type]
            if not threshold.enabled:
                continue
            
            # Determine severity
            severity = self._determine_severity(current_value, threshold, metric_type)
            
            if severity:
                await self._create_or_update_alert(
                    metric_type, severity, current_value, threshold
                )
            else:
                # Check if we should resolve existing alert
                await self._resolve_alert_if_exists(metric_type)
    
    def _determine_severity(self, value: float, threshold: PerformanceThreshold, 
                          metric_type: MetricType) -> Optional[AlertSeverity]:
        """Determine alert severity based on value and thresholds"""
        
        # Special handling for cache hit rate (lower is worse)
        if metric_type == MetricType.CACHE_HIT_RATE:
            if value <= threshold.emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= threshold.warning_threshold:
                return AlertSeverity.WARNING
        else:
            # Normal metrics (higher is worse)
            if threshold.emergency_threshold and value >= threshold.emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= threshold.warning_threshold:
                return AlertSeverity.WARNING
        
        return None
    
    async def _create_or_update_alert(self, metric_type: MetricType, 
                                    severity: AlertSeverity, 
                                    current_value: float,
                                    threshold: PerformanceThreshold):
        """Create new alert or update existing one"""
        alert_key = f"{metric_type.value}_{severity.value}"
        
        if alert_key in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_key]
            alert.current_value = current_value
            alert.timestamp = datetime.now()
        else:
            # Create new alert
            alert = PerformanceAlert(
                alert_id=alert_key,
                metric_type=metric_type,
                severity=severity,
                title=self._generate_alert_title(metric_type, severity),
                description=self._generate_alert_description(
                    metric_type, current_value, threshold
                ),
                current_value=current_value,
                threshold_value=self._get_threshold_value(threshold, severity),
                timestamp=datetime.now()
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def _resolve_alert_if_exists(self, metric_type: MetricType):
        """Resolve alert if it exists and conditions are met"""