"""
Performance Monitoring and Optimization
Implements PerformanceOptimizer for system tuning, real-time performance metrics,
automatic optimization recommendations, and performance regression detection.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime, timedelta
import threading
import time
import psutil
import os
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"
    CONCURRENT_USERS = "concurrent_users"
    CACHE_HIT_RATE = "cache_hit_rate"
    GC_PRESSURE = "gc_pressure"


class OptimizationTarget(Enum):
    """Targets for performance optimization."""
    SPEED = "speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    CPU_EFFICIENCY = "cpu_efficiency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"


class PerformanceIssueSeverity(Enum):
    """Severity levels for performance issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    metric_id: str
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    source_component: str
    unit: str
    metadata: Dict[str, Any]


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison."""
    baseline_id: str
    metric_type: PerformanceMetricType
    baseline_value: float
    std_deviation: float
    time_window: timedelta
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceRecommendation:
    """A recommendation for performance optimization."""
    recommendation_id: str
    issue_description: str
    suggested_action: str
    target_component: str
    optimization_target: OptimizationTarget
    expected_impact: float  # Expected improvement percentage
    confidence_score: float  # 0.0 to 1.0
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceIssue:
    """A detected performance issue."""
    issue_id: str
    severity: PerformanceIssueSeverity
    metric_type: PerformanceMetricType
    current_value: float
    threshold_value: float
    component: str
    description: str
    detected_at: datetime
    status: str  # open, acknowledged, resolved
    metadata: Dict[str, Any]


class MetricsCollector:
    """Collects performance metrics from various sources."""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 metrics
        self.system_metrics_enabled = True
        self.application_metrics_enabled = True
        self.collection_interval = 1  # seconds
        self.last_collection_time = datetime.now()
        
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(PerformanceMetric(
            metric_id=f"sys_cpu_{secrets.token_hex(8)}",
            metric_type=PerformanceMetricType.CPU_USAGE,
            value=cpu_percent,
            timestamp=current_time,
            source_component="system",
            unit="percent",
            metadata={"core_count": psutil.cpu_count()}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            metric_id=f"sys_memory_{secrets.token_hex(8)}",
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            value=memory.percent,
            timestamp=current_time,
            source_component="system",
            unit="percent",
            metadata={"total_gb": memory.total / (1024**3)}
        ))
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.append(PerformanceMetric(
                metric_id=f"sys_disk_read_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.DISK_IO,
                value=disk_io.read_bytes,
                timestamp=current_time,
                source_component="system",
                unit="bytes",
                metadata={"operation": "read"}
            ))
            metrics.append(PerformanceMetric(
                metric_id=f"sys_disk_write_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.DISK_IO,
                value=disk_io.write_bytes,
                timestamp=current_time,
                source_component="system",
                unit="bytes",
                metadata={"operation": "write"}
            ))
        
        # Network I/O metrics
        net_io = psutil.net_io_counters()
        if net_io:
            metrics.append(PerformanceMetric(
                metric_id=f"sys_net_sent_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.NETWORK_IO,
                value=net_io.bytes_sent,
                timestamp=current_time,
                source_component="system",
                unit="bytes",
                metadata={"direction": "outbound"}
            ))
            metrics.append(PerformanceMetric(
                metric_id=f"sys_net_recv_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.NETWORK_IO,
                value=net_io.bytes_recv,
                timestamp=current_time,
                source_component="system",
                unit="bytes",
                metadata={"direction": "inbound"}
            ))
        
        # Process-specific metrics
        current_process = psutil.Process(os.getpid())
        with current_process.oneshot():
            metrics.append(PerformanceMetric(
                metric_id=f"proc_cpu_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.CPU_USAGE,
                value=current_process.cpu_percent(),
                timestamp=current_time,
                source_component="application",
                unit="percent",
                metadata={"pid": current_process.pid}
            ))
            
            memory_info = current_process.memory_info()
            metrics.append(PerformanceMetric(
                metric_id=f"proc_memory_{secrets.token_hex(8)}",
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                value=memory_info.rss / (1024 * 1024),  # MB
                timestamp=current_time,
                source_component="application",
                unit="MB",
                metadata={"pid": current_process.pid, "type": "rss"}
            ))
        
        return metrics
        
    def collect_application_metrics(self) -> List[PerformanceMetric]:
        """Collect application-level performance metrics."""
        # This would be extended based on specific application metrics
        # For now, we'll return an empty list
        return []
        
    def get_recent_metrics(self, metric_type: PerformanceMetricType, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics of a specific type."""
        all_metrics = []
        for metric_list in self.metrics_buffer.values():
            all_metrics.extend([
                m for m in metric_list 
                if m.metric_type == metric_type
            ])
        
        # Sort by timestamp and return the most recent
        all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return all_metrics[:limit]
        
    def store_metrics(self, metrics: List[PerformanceMetric]):
        """Store collected metrics in buffer."""
        for metric in metrics:
            self.metrics_buffer[metric.source_component].append(metric)


class BaselineCalculator:
    """Calculates performance baselines."""
    
    def __init__(self):
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.metric_history: Dict[PerformanceMetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        
    def update_baseline(self, metric_type: PerformanceMetricType, value: float):
        """Update baseline with a new metric value."""
        # Add to history
        self.metric_history[metric_type].append(value)
        
        # Recalculate baseline if we have enough data
        if len(self.metric_history[metric_type]) >= 100:  # Minimum 100 samples
            values = list(self.metric_history[metric_type])
            baseline_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            baseline_id = f"baseline_{metric_type.value}_{secrets.token_hex(8)}"
            
            baseline = PerformanceBaseline(
                baseline_id=baseline_id,
                metric_type=metric_type,
                baseline_value=baseline_value,
                std_deviation=std_dev,
                time_window=timedelta(hours=1),  # Hourly baseline
                created_at=datetime.now(),
                metadata={"sample_count": len(values)}
            )
            
            # Store the baseline
            self.baselines[f"{metric_type.value}_hourly"] = baseline
            
    def get_baseline(self, metric_type: PerformanceMetricType) -> Optional[PerformanceBaseline]:
        """Get the current baseline for a metric type."""
        key = f"{metric_type.value}_hourly"
        return self.baselines.get(key)
        
    def is_regression(self, metric_type: PerformanceMetricType, current_value: float, threshold_std: float = 2.0) -> bool:
        """Check if current value represents a regression compared to baseline."""
        baseline = self.get_baseline(metric_type)
        if not baseline:
            return False  # No baseline to compare to
            
        # Calculate z-score
        if baseline.std_deviation == 0:
            return abs(current_value - baseline.baseline_value) > threshold_std
            
        z_score = abs(current_value - baseline.baseline_value) / baseline.std_deviation
        return z_score > threshold_std


class PerformanceIssueDetector:
    """Detects performance issues."""
    
    def __init__(self):
        self.issues: List[PerformanceIssue] = []
        self.thresholds: Dict[PerformanceMetricType, float] = {
            PerformanceMetricType.CPU_USAGE: 80.0,  # percent
            PerformanceMetricType.MEMORY_USAGE: 85.0,  # percent
            PerformanceMetricType.ERROR_RATE: 5.0,  # percent
            PerformanceMetricType.LATENCY: 1000.0,  # milliseconds
            PerformanceMetricType.RESPONSE_TIME: 2000.0  # milliseconds
        }
        self.issue_callbacks: List[callable] = []
        
    def set_threshold(self, metric_type: PerformanceMetricType, threshold: float):
        """Set threshold for a metric type."""
        self.thresholds[metric_type] = threshold
        
    def detect_issues(self, metrics: List[PerformanceMetric]) -> List[PerformanceIssue]:
        """Detect performance issues from a list of metrics."""
        detected_issues = []
        
        for metric in metrics:
            threshold = self.thresholds.get(metric.metric_type)
            if threshold is not None and metric.value > threshold:
                # Determine severity based on how much the threshold is exceeded
                excess_ratio = (metric.value - threshold) / threshold
                if excess_ratio > 0.5:  # More than 50% over threshold
                    severity = PerformanceIssueSeverity.CRITICAL
                elif excess_ratio > 0.2:  # More than 20% over threshold
                    severity = PerformanceIssueSeverity.ERROR
                elif excess_ratio > 0.1:  # More than 10% over threshold
                    severity = PerformanceIssueSeverity.WARNING
                else:
                    severity = PerformanceIssueSeverity.INFO
                    
                issue = PerformanceIssue(
                    issue_id=f"issue_{secrets.token_hex(8)}",
                    severity=severity,
                    metric_type=metric.metric_type,
                    current_value=metric.value,
                    threshold_value=threshold,
                    component=metric.source_component,
                    description=f"{metric.metric_type.value} is {metric.value:.2f}{metric.unit}, exceeding threshold of {threshold}{metric.unit}",
                    detected_at=metric.timestamp,
                    status="open",
                    metadata={"metric_id": metric.metric_id}
                )
                
                detected_issues.append(issue)
                
                # Store the issue
                self.issues.append(issue)
                
                # Trigger callbacks
                for callback in self.issue_callbacks:
                    try:
                        callback(issue)
                    except Exception as e:
                        logger.error(f"Error in issue callback: {str(e)}")
        
        return detected_issues
        
    def add_issue_callback(self, callback: callable):
        """Add a callback to be called when issues are detected."""
        self.issue_callbacks.append(callback)


class OptimizationAdvisor:
    """Provides optimization recommendations."""
    
    def __init__(self):
        self.recommendations: List[PerformanceRecommendation] = []
        self.known_optimizations = {
            PerformanceMetricType.CPU_USAGE: [
                {
                    "action": "Implement caching for expensive computations",
                    "impact": 0.3,  # 30% reduction expected
                    "confidence": 0.8
                },
                {
                    "action": "Optimize algorithm complexity",
                    "impact": 0.4,  # 40% reduction expected
                    "confidence": 0.7
                },
                {
                    "action": "Use asynchronous processing",
                    "impact": 0.25,  # 25% reduction expected
                    "confidence": 0.75
                }
            ],
            PerformanceMetricType.MEMORY_USAGE: [
                {
                    "action": "Implement object pooling",
                    "impact": 0.35,  # 35% reduction expected
                    "confidence": 0.8
                },
                {
                    "action": "Use generators instead of lists",
                    "impact": 0.2,  # 20% reduction expected
                    "confidence": 0.9
                },
                {
                    "action": "Optimize data structures",
                    "impact": 0.3,  # 30% reduction expected
                    "confidence": 0.75
                }
            ],
            PerformanceMetricType.LATENCY: [
                {
                    "action": "Implement CDN for static assets",
                    "impact": 0.5,  # 50% reduction expected
                    "confidence": 0.85
                },
                {
                    "action": "Move computation closer to users",
                    "impact": 0.4,  # 40% reduction expected
                    "confidence": 0.8
                },
                {
                    "action": "Optimize database queries",
                    "impact": 0.3,  # 30% reduction expected
                    "confidence": 0.9
                }
            ]
        }
        
    def generate_recommendations(self, issues: List[PerformanceIssue]) -> List[PerformanceRecommendation]:
        """Generate optimization recommendations based on detected issues."""
        recommendations = []
        
        for issue in issues:
            if issue.metric_type in self.known_optimizations:
                optimizations = self.known_optimizations[issue.metric_type]
                
                for opt in optimizations:
                    # Calculate expected impact based on current issue severity
                    expected_impact = opt["impact"]
                    if issue.severity == PerformanceIssueSeverity.CRITICAL:
                        expected_impact *= 1.2  # Higher impact for critical issues
                    elif issue.severity == PerformanceIssueSeverity.ERROR:
                        expected_impact *= 1.1  # Slightly higher impact for errors
                    
                    # Cap the impact at 100%
                    expected_impact = min(expected_impact, 1.0)
                    
                    recommendation = PerformanceRecommendation(
                        recommendation_id=f"rec_{secrets.token_hex(8)}",
                        issue_description=issue.description,
                        suggested_action=opt["action"],
                        target_component=issue.component,
                        optimization_target=self._map_metric_to_target(issue.metric_type),
                        expected_impact=expected_impact * 100,  # Convert to percentage
                        confidence_score=opt["confidence"],
                        created_at=datetime.now(),
                        metadata={
                            "issue_id": issue.issue_id,
                            "original_value": issue.current_value,
                            "threshold_value": issue.threshold_value
                        }
                    )
                    
                    recommendations.append(recommendation)
                    self.recommendations.append(recommendation)
        
        return recommendations
        
    def _map_metric_to_target(self, metric_type: PerformanceMetricType) -> OptimizationTarget:
        """Map a metric type to an optimization target."""
        mapping = {
            PerformanceMetricType.CPU_USAGE: OptimizationTarget.CPU_EFFICIENCY,
            PerformanceMetricType.MEMORY_USAGE: OptimizationTarget.MEMORY_EFFICIENCY,
            PerformanceMetricType.LATENCY: OptimizationTarget.SPEED,
            PerformanceMetricType.RESPONSE_TIME: OptimizationTarget.SPEED,
            PerformanceMetricType.ERROR_RATE: OptimizationTarget.RELIABILITY
        }
        return mapping.get(metric_type, OptimizationTarget.SPEED)


class PerformanceOptimizer:
    """
    Performance optimizer for system tuning with real-time metrics,
    optimization recommendations, and regression detection.
    """
    
    def __init__(self, collection_interval: int = 5):
        self.metrics_collector = MetricsCollector()
        self.baseline_calculator = BaselineCalculator()
        self.issue_detector = PerformanceIssueDetector()
        self.optimization_advisor = OptimizationAdvisor()
        self.collection_interval = collection_interval
        self.monitoring_task = None
        self.regression_detection_enabled = True
        self.optimization_enabled = True
        self.performance_history: Dict[PerformanceMetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.stop_event = asyncio.Event()
        
    async def start_monitoring(self):
        """Start the performance monitoring loop."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop the performance monitoring loop."""
        self.stop_event.set()
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                app_metrics = self.metrics_collector.collect_application_metrics()
                all_metrics = system_metrics + app_metrics
                
                # Store metrics
                self.metrics_collector.store_metrics(all_metrics)
                
                # Update baselines
                for metric in all_metrics:
                    self.baseline_calculator.update_baseline(metric.metric_type, metric.value)
                    # Add to performance history
                    self.performance_history[metric.metric_type].append({
                        "timestamp": metric.timestamp,
                        "value": metric.value,
                        "source": metric.source_component
                    })
                
                # Detect issues
                detected_issues = self.issue_detector.detect_issues(all_metrics)
                
                # Generate recommendations if issues are found
                if detected_issues and self.optimization_enabled:
                    recommendations = self.optimization_advisor.generate_recommendations(detected_issues)
                    for rec in recommendations:
                        logger.info(f"Optimization recommendation: {rec.suggested_action} "
                                  f"(expected impact: {rec.expected_impact:.1f}%)")
                
                # Check for regressions
                if self.regression_detection_enabled:
                    await self._check_for_regressions(all_metrics)
                
                # Wait for next collection
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(), 
                        timeout=self.collection_interval
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue the loop
                    continue
                    
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _check_for_regressions(self, metrics: List[PerformanceMetric]):
        """Check for performance regressions."""
        for metric in metrics:
            is_reg = self.baseline_calculator.is_regression(metric.metric_type, metric.value)
            if is_reg:
                logger.warning(f"Performance regression detected: {metric.metric_type.value} = {metric.value}")
                
                # Could trigger additional actions here
                # For example, alerting, automatic rollbacks, etc.
                
    def get_current_metrics(self) -> Dict[PerformanceMetricType, float]:
        """Get the most recent values for all metric types."""
        current_metrics = {}
        
        for metric_type in PerformanceMetricType:
            recent_metrics = self.metrics_collector.get_recent_metrics(metric_type, limit=1)
            if recent_metrics:
                current_metrics[metric_type] = recent_metrics[0].value
            else:
                current_metrics[metric_type] = 0.0  # Default value
                
        return current_metrics
        
    def get_performance_trend(self, metric_type: PerformanceMetricType, hours: int = 1) -> Dict[str, Any]:
        """Get performance trend for a metric type over the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get historical data
        history = self.performance_history[metric_type]
        relevant_data = [
            item for item in history 
            if item["timestamp"] >= cutoff_time
        ]
        
        if not relevant_data:
            return {
                "metric_type": metric_type.value,
                "time_period_hours": hours,
                "data_points": 0,
                "average_value": 0.0,
                "min_value": 0.0,
                "max_value": 0.0,
                "trend": "unknown"
            }
        
        values = [item["value"] for item in relevant_data]
        timestamps = [item["timestamp"] for item in relevant_data]
        
        # Calculate statistics
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)
        
        # Calculate trend (simple linear regression slope)
        if len(values) > 1:
            # Convert timestamps to numeric values for regression
            time_nums = [(t - timestamps[0]).total_seconds() for t in timestamps]
            slope, _, _, _, _ = stats.linregress(time_nums, values)
            trend_direction = "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            "metric_type": metric_type.value,
            "time_period_hours": hours,
            "data_points": len(values),
            "average_value": avg_value,
            "min_value": min_value,
            "max_value": max_value,
            "trend": trend_direction,
            "slope": slope if 'slope' in locals() else 0
        }
        
    def get_optimization_recommendations(self) -> List[PerformanceRecommendation]:
        """Get all optimization recommendations."""
        return self.optimization_advisor.recommendations
        
    def get_detected_issues(self) -> List[PerformanceIssue]:
        """Get all detected performance issues."""
        return self.issue_detector.issues
        
    def get_performance_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Get all performance baselines."""
        return self.baseline_calculator.baselines
        
    def set_optimization_target(self, target: OptimizationTarget):
        """Set the primary optimization target."""
        # This could be used to prioritize certain types of optimizations
        logger.info(f"Set optimization target to {target.value}")
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics and status."""
        current_metrics = self.get_current_metrics()
        
        # Get recent issues
        recent_issues = [
            issue for issue in self.get_detected_issues()
            if datetime.now() - issue.detected_at < timedelta(hours=1)
        ]
        
        # Get recent recommendations
        recent_recommendations = self.get_optimization_recommendations()
        
        # Calculate health score based on metrics
        health_score = 100  # Start with perfect health
        for metric_type, value in current_metrics.items():
            threshold = self.issue_detector.thresholds.get(metric_type, float('inf'))
            if value > threshold:
                # Reduce health score based on how much threshold is exceeded
                excess_ratio = min((value - threshold) / threshold, 1.0)  # Cap at 100% excess
                health_score -= excess_ratio * 30  # Up to 30 points deduction per metric
        
        health_score = max(0, health_score)  # Ensure non-negative
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": round(health_score, 2),
            "current_metrics": {k.value: v for k, v in current_metrics.items()},
            "recent_issues_count": len(recent_issues),
            "critical_issues_count": len([i for i in recent_issues if i.severity == PerformanceIssueSeverity.CRITICAL]),
            "recommendations_count": len(recent_recommendations),
            "baselines_available": len(self.get_performance_baselines()),
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done()
        }
        
    def add_performance_issue_callback(self, callback: callable):
        """Add a callback to be called when performance issues are detected."""
        self.issue_detector.add_issue_callback(callback)
        
    def enable_regression_detection(self):
        """Enable performance regression detection."""
        self.regression_detection_enabled = True
        logger.info("Performance regression detection enabled")
        
    def disable_regression_detection(self):
        """Disable performance regression detection."""
        self.regression_detection_enabled = False
        logger.info("Performance regression detection disabled")
        
    def enable_optimization_advice(self):
        """Enable optimization recommendations."""
        self.optimization_enabled = True
        logger.info("Optimization advice enabled")
        
    def disable_optimization_advice(self):
        """Disable optimization recommendations."""
        self.optimization_enabled = False
        logger.info("Optimization advice disabled")


# Convenience function for easy use
async def create_performance_optimizer(
    collection_interval: int = 5
) -> PerformanceOptimizer:
    """
    Convenience function to create a performance optimizer.
    
    Args:
        collection_interval: Interval in seconds for collecting metrics
        
    Returns:
        PerformanceOptimizer instance
    """
    optimizer = PerformanceOptimizer(collection_interval)
    await optimizer.start_monitoring()
    return optimizer