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
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import deque, defaultdict
import inspect

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from ..cache.multimodal_cache import get_multimodal_cache_async
except Exception:  # pragma: no cover - fallback when cache subsystem unavailable
    async def get_multimodal_cache_async():
        raise RuntimeError("Multimodal cache subsystem is unavailable")
else:
    get_multimodal_cache = get_multimodal_cache_async

# Backwards compatibility for test patches expecting old name
if "get_multimodal_cache" not in globals():
    get_multimodal_cache = get_multimodal_cache_async

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
            cache_system = await get_multimodal_cache_async()
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
        start_value = values[0]
        end_value = values[-1]
        
        # Prevent division by zero when start value is extremely small
        baseline = start_value if abs(start_value) > 1e-6 else 1.0
        change_percent = ((start_value - end_value) / baseline) * 100
        
        if abs(change_percent) < 5:
            return "stable"
        if start_value > end_value:
            return "increasing"
        if start_value < end_value:
            return "decreasing"
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
        resolved_keys = [
            key for key, alert in self.active_alerts.items()
            if alert.metric_type == metric_type
        ]
        for key in resolved_keys:
            alert = self.active_alerts.pop(key)
            alert.resolved = True
            alert.resolved_at = datetime.now()
    
    def _generate_alert_title(self, metric_type: MetricType, severity: AlertSeverity) -> str:
        """Generate human-readable alert title"""
        metric_name = self._format_metric_name(metric_type)
        return f"{severity.value.capitalize()} {metric_name} Alert"
    
    def _format_metric_name(self, metric_type: MetricType) -> str:
        """Format metric name for display"""
        friendly_names = {
            MetricType.CPU_USAGE: "CPU Usage",
            MetricType.MEMORY_USAGE: "Memory Usage",
            MetricType.DISK_USAGE: "Disk Usage",
            MetricType.RESPONSE_TIME: "Response Time",
            MetricType.ERROR_RATE: "Error Rate",
            MetricType.THROUGHPUT: "Throughput",
            MetricType.CACHE_HIT_RATE: "Cache Hit Rate",
            MetricType.QUEUE_SIZE: "Queue Size",
        }
        return friendly_names.get(
            metric_type,
            metric_type.value.replace('_', ' ').title()
        )
    
    def _generate_alert_description(
        self,
        metric_type: MetricType,
        current_value: float,
        threshold: PerformanceThreshold,
    ) -> str:
        """Generate descriptive alert message"""
        metric_name = self._format_metric_name(metric_type)
        parts = [
            f"{metric_name} at {current_value:.1f}% exceeds "
            f"warning threshold of {threshold.warning_threshold:.1f}%"
        ]
        
        if metric_type == MetricType.CACHE_HIT_RATE:
            if threshold.critical_threshold and current_value <= threshold.critical_threshold:
                parts.append(f"(critical threshold {threshold.critical_threshold:.1f}%)")
            if threshold.emergency_threshold is not None and current_value <= threshold.emergency_threshold:
                parts.append(f"(emergency threshold {threshold.emergency_threshold:.1f}%)")
        else:
            if threshold.critical_threshold and current_value >= threshold.critical_threshold:
                parts.append(f"(critical threshold {threshold.critical_threshold:.1f}%)")
            if threshold.emergency_threshold is not None and current_value >= threshold.emergency_threshold:
                parts.append(f"(emergency threshold {threshold.emergency_threshold:.1f}%)")
        
        return " ".join(parts)
    
    def _get_threshold_value(
        self,
        threshold: PerformanceThreshold,
        severity: AlertSeverity,
    ) -> float:
        """Return the threshold value associated with a severity level"""
        if severity == AlertSeverity.EMERGENCY and threshold.emergency_threshold is not None:
            return threshold.emergency_threshold
        if severity == AlertSeverity.CRITICAL:
            return threshold.critical_threshold
        if severity == AlertSeverity.WARNING:
            return threshold.warning_threshold
        return threshold.warning_threshold
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Summarise active alerts by severity level"""
        summary = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.active_alerts.values():
            summary[alert.severity.value] += 1
        return summary


class PerformanceOptimizer:
    """Analyzes metrics and executes optimization actions"""
    
    def __init__(self):
        self.auto_optimize_enabled: bool = True
        self.optimization_rules: List[Dict[str, Any]] = []
        self.executed_actions: List[OptimizationAction] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Register built-in optimization rules"""
        self.optimization_rules.extend([
            {
                "name": "high_memory_cache_cleanup",
                "condition": lambda metrics: metrics.get(MetricType.MEMORY_USAGE, 0) >= 85.0,
                "action": "_cleanup_memory_cache",
                "description": "Free memory by cleaning cache entries",
                "target": "cache",
                "risk_level": "medium",
                "estimated_impact": "Reduce memory footprint",
            },
            {
                "name": "low_cache_hit_rate_warming",
                "condition": lambda metrics: metrics.get(MetricType.CACHE_HIT_RATE, 100.0) <= 60.0,
                "action": "_trigger_cache_warming",
                "description": "Warm cache with frequently accessed items",
                "target": "cache",
                "risk_level": "low",
                "estimated_impact": "Improve cache hit rate",
            },
            {
                "name": "high_cpu_scale_workers",
                "condition": lambda metrics: metrics.get(MetricType.CPU_USAGE, 0) >= 90.0,
                "action": "_scale_async_workers",
                "description": "Scale asynchronous workers to balance CPU load",
                "target": "workers",
                "risk_level": "medium",
                "estimated_impact": "Distribute workload",
            },
            {
                "name": "low_cpu_reduce_workers",
                "condition": lambda metrics: metrics.get(MetricType.CPU_USAGE, 0) <= 20.0,
                "action": "_reduce_async_workers",
                "description": "Reduce worker count to save resources",
                "target": "workers",
                "risk_level": "low",
                "estimated_impact": "Reduce idle resource usage",
            },
        ])
    
    def add_optimization_rule(
        self,
        name: str,
        condition: Callable[[Dict[MetricType, float]], bool],
        action: Callable[[Dict[MetricType, float]], Any],
        description: str,
        risk_level: str = "low",
        target_component: str = "system",
        estimated_impact: str = "",
    ):
        """Register a custom optimization rule"""
        self.optimization_rules.append(
            {
                "name": name,
                "condition": condition,
                "action": action,
                "description": description,
                "risk_level": risk_level,
                "target": target_component,
                "estimated_impact": estimated_impact,
            }
        )
    
    async def analyze_and_optimize(
        self,
        metrics: Dict[MetricType, float],
    ) -> List[OptimizationAction]:
        """Analyze metrics and execute matching optimization rules"""
        if not self.auto_optimize_enabled:
            return []
        
        actions: List[OptimizationAction] = []
        timestamp = datetime.now()
        
        for rule in self.optimization_rules:
            try:
                if not rule["condition"](metrics):
                    continue
                
                action_ref = rule["action"]
                if isinstance(action_ref, str):
                    action_callable = getattr(self, action_ref)
                else:
                    action_callable = action_ref
                
                if asyncio.iscoroutinefunction(action_callable):
                    outcome = await action_callable(metrics)
                else:
                    outcome_candidate = action_callable(metrics)
                    if asyncio.iscoroutine(outcome_candidate) or inspect.isawaitable(outcome_candidate):
                        outcome = await outcome_candidate
                    else:
                        outcome = outcome_candidate
                
                action = OptimizationAction(
                    action_id=f"{rule['name']}_{int(time.time()*1000)}",
                    action_type=rule["name"],
                    description=rule["description"],
                    target_component=rule.get("target", "system"),
                    parameters={},
                    estimated_impact=rule.get("estimated_impact", ""),
                    risk_level=rule.get("risk_level", "low"),
                    executed=True,
                    executed_at=timestamp,
                    result=str(outcome) if outcome is not None else None,
                )
                self.executed_actions.append(action)
                actions.append(action)
            except Exception as exc:
                logger.error("Optimization rule '%s' failed: %s", rule["name"], exc)
        
        return actions
    
    async def _cleanup_memory_cache(self, metrics: Dict[MetricType, float]) -> str:
        """Simulate cache cleanup to free memory"""
        await asyncio.sleep(0)
        return "Cache cleaned to reduce memory usage"
    
    async def _trigger_cache_warming(self, metrics: Dict[MetricType, float]) -> str:
        """Simulate cache warming process"""
        await asyncio.sleep(0)
        return "Cache warming sequence triggered"
    
    async def _scale_async_workers(self, metrics: Dict[MetricType, float]) -> str:
        """Simulate scaling up asynchronous workers"""
        await asyncio.sleep(0)
        return "Scaled async workers to handle CPU load"
    
    async def _reduce_async_workers(self, metrics: Dict[MetricType, float]) -> str:
        """Simulate reducing asynchronous workers"""
        await asyncio.sleep(0)
        return "Reduced async workers to conserve resources"
    
    def get_optimization_history(self, limit: int = 10) -> List[OptimizationAction]:
        """Return recent optimization actions"""
        if limit <= 0:
            return []
        return self.executed_actions[-limit:]


class PerformanceMonitoringSystem:
    """Coordinates metrics collection, alerting, and optimization"""
    
    def __init__(self, monitoring_interval: int = 30):
        self.metrics_collector = PerformanceMetricsCollector()
        self.alert_manager = AlertManager()
        self.optimizer = PerformanceOptimizer()
        self.monitoring_interval = monitoring_interval
        self.running: bool = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Register alert callback on the underlying manager"""
        self.alert_manager.add_alert_callback(callback)
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        self.running = True
        try:
            while self.running:
                system_metrics = await self.metrics_collector.collect_system_metrics()
                application_metrics = await self.metrics_collector.collect_application_metrics()
                combined_metrics = {**system_metrics, **application_metrics}
                
                timestamp = time.time()
                for metric_type, value in combined_metrics.items():
                    self.metrics_collector.metrics_history[metric_type].append((timestamp, value))
                
                if combined_metrics:
                    await self.alert_manager.check_thresholds(combined_metrics)
                    await self.optimizer.analyze_and_optimize(combined_metrics)
                
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            raise
        finally:
            self.running = False
    
    def ensure_background_task(self):
        """Ensure the monitoring loop is running"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def stop(self):
        """Stop the monitoring loop"""
        self.running = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Return current system status snapshot"""
        current_metrics: Dict[str, Optional[float]] = {}
        for metric_type, history in self.metrics_collector.metrics_history.items():
            if history:
                current_metrics[metric_type.value] = history[-1][1]
        
        status = {
            "monitoring_active": self.running,
            "current_metrics": current_metrics,
            "active_alerts": list(self.alert_manager.active_alerts.values()),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "recent_optimizations": self.optimizer.get_optimization_history(5),
            "auto_optimization": self.optimizer.auto_optimize_enabled,
        }
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        trends = {
            MetricType.CPU_USAGE.value: self.metrics_collector.calculate_trend(MetricType.CPU_USAGE),
            MetricType.MEMORY_USAGE.value: self.metrics_collector.calculate_trend(MetricType.MEMORY_USAGE),
            MetricType.CACHE_HIT_RATE.value: self.metrics_collector.calculate_trend(MetricType.CACHE_HIT_RATE),
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self._calculate_system_health(),
            "trends": trends,
            "alerts": {
                "active": list(self.alert_manager.active_alerts.values()),
                "summary": self.alert_manager.get_alert_summary(),
            },
            "optimizations": {
                "recent": self.optimizer.get_optimization_history(5),
                "auto_enabled": self.optimizer.auto_optimize_enabled,
            },
        }
        return report
    
    def _calculate_system_health(self) -> str:
        """Compute system health based on active alerts"""
        severities = {alert.severity for alert in self.alert_manager.active_alerts.values()}
        if not severities:
            return "healthy"
        if AlertSeverity.CRITICAL in severities or AlertSeverity.EMERGENCY in severities:
            return "degraded"
        if AlertSeverity.WARNING in severities:
            return "warning"
        return "healthy"


_performance_monitoring_system: Optional[PerformanceMonitoringSystem] = None


def get_performance_monitoring_system() -> PerformanceMonitoringSystem:
    """Get or create the global performance monitoring system"""
    global _performance_monitoring_system
    if _performance_monitoring_system is None:
        _performance_monitoring_system = PerformanceMonitoringSystem()
    return _performance_monitoring_system


async def initialize_performance_monitoring() -> PerformanceMonitoringSystem:
    """Initialise performance monitoring background tasks"""
    system = get_performance_monitoring_system()
    system.ensure_background_task()
    return system