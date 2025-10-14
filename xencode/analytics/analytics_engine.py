#!/usr/bin/env python3
"""
Analytics Engine

Main analytics engine that coordinates metrics collection, event tracking,
and data analysis for comprehensive system monitoring and insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from .metrics_collector import MetricsCollector, AnalyticsEvent, MetricType
from .event_tracker import EventTracker, TrackedEvent, EventCategory, EventPriority, EventContext


@dataclass
class AnalyticsConfig:
    """Configuration for analytics engine"""
    enable_metrics: bool = True
    enable_events: bool = True
    enable_prometheus: bool = True
    metrics_port: int = 8000
    storage_path: Optional[Path] = None
    aggregation_interval: int = 60
    max_events_in_memory: int = 10000
    max_metrics_age_days: int = 30


class AnalyticsEngine:
    """
    Main analytics engine that provides comprehensive monitoring and insights
    
    Coordinates metrics collection, event tracking, and data analysis
    to provide real-time monitoring and historical analytics.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        
        # Initialize components
        self.metrics_collector = None
        self.event_tracker = None
        
        if self.config.enable_metrics:
            self.metrics_collector = MetricsCollector(
                enable_prometheus=self.config.enable_prometheus,
                metrics_port=self.config.metrics_port,
                storage_path=self.config.storage_path
            )
        
        if self.config.enable_events:
            self.event_tracker = EventTracker(
                storage_path=self.config.storage_path
            )
        
        # Analytics state
        self._running = False
        self._analysis_task = None
        
        # Integration callbacks
        self._setup_integration()
    
    def _setup_integration(self) -> None:
        """Set up integration between metrics and events"""
        if self.event_tracker and self.metrics_collector:
            # Forward events to metrics collector
            self.event_tracker.add_event_callback("*", self._event_to_metrics)
    
    def _event_to_metrics(self, event: TrackedEvent) -> None:
        """Convert tracked events to metrics"""
        if not self.metrics_collector:
            return
        
        try:
            # Convert event to analytics event for metrics collector
            analytics_event = AnalyticsEvent(
                event_type=event.event_type,
                timestamp=event.timestamp,
                user_id=event.context.user_id,
                session_id=event.context.session_id,
                properties=event.properties,
                metrics=event.metrics
            )
            
            # Track the event in metrics collector
            self.metrics_collector.track_event(analytics_event)
            
            # Update relevant metrics based on event type
            if event.category == EventCategory.USER_ACTION:
                self.metrics_collector.increment_counter(
                    "xencode_user_actions_total",
                    labels={"action": event.event_type, "user_id": event.context.user_id or "unknown"}
                )
            
            elif event.category == EventCategory.ERROR:
                self.metrics_collector.increment_counter(
                    "xencode_errors_total",
                    labels={"error_type": event.properties.get("error_type", "unknown")}
                )
            
            elif event.category == EventCategory.PERFORMANCE:
                if "duration_ms" in event.metrics:
                    self.metrics_collector.observe_histogram(
                        "xencode_operation_duration_seconds",
                        event.metrics["duration_ms"] / 1000.0,
                        labels={"operation": event.properties.get("operation", "unknown")}
                    )
            
            elif event.category == EventCategory.AI_INTERACTION:
                self.metrics_collector.increment_counter(
                    "xencode_ai_requests_total",
                    labels={
                        "model": event.properties.get("model", "unknown"),
                        "status": "success" if event.metrics.get("success", 0) > 0 else "error"
                    }
                )
                
                if "duration_ms" in event.metrics:
                    self.metrics_collector.observe_histogram(
                        "xencode_ai_response_time_seconds",
                        event.metrics["duration_ms"] / 1000.0,
                        labels={"model": event.properties.get("model", "unknown")}
                    )
            
        except Exception as e:
            print(f"Error converting event to metrics: {e}")
    
    async def start(self) -> None:
        """Start the analytics engine"""
        if self._running:
            return
        
        self._running = True
        
        # Start components
        if self.metrics_collector:
            await self.metrics_collector.start()
        
        if self.event_tracker:
            await self.event_tracker.start()
        
        # Start analysis task
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        print("AnalyticsEngine started")
    
    async def stop(self) -> None:
        """Stop the analytics engine"""
        self._running = False
        
        # Cancel analysis task
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        if self.metrics_collector:
            await self.metrics_collector.stop()
        
        if self.event_tracker:
            await self.event_tracker.stop()
        
        print("AnalyticsEngine stopped")
    
    async def _analysis_loop(self) -> None:
        """Background loop for analytics analysis"""
        while self._running:
            try:
                await self._perform_analysis()
                await asyncio.sleep(self.config.aggregation_interval)
            except Exception as e:
                print(f"Error in analysis loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_analysis(self) -> None:
        """Perform analytics analysis and generate insights"""
        try:
            # Analyze recent events
            if self.event_tracker:
                recent_events = self.event_tracker.get_recent_events(hours=1)
                
                # Analyze error patterns
                error_events = [e for e in recent_events if e.category == EventCategory.ERROR]
                if len(error_events) > 10:  # Threshold for error spike
                    await self._handle_error_spike(error_events)
                
                # Analyze performance patterns
                perf_events = [e for e in recent_events if e.category == EventCategory.PERFORMANCE]
                await self._analyze_performance_trends(perf_events)
                
                # Analyze user activity
                user_events = [e for e in recent_events if e.category == EventCategory.USER_ACTION]
                await self._analyze_user_activity(user_events)
        
        except Exception as e:
            print(f"Error performing analysis: {e}")
    
    async def _handle_error_spike(self, error_events: List[TrackedEvent]) -> None:
        """Handle error spike detection"""
        error_types = {}
        for event in error_events:
            error_type = event.properties.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Log error spike
        print(f"Error spike detected: {len(error_events)} errors in last hour")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} occurrences")
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge("xencode_error_spike_detected", 1.0)
    
    async def _analyze_performance_trends(self, perf_events: List[TrackedEvent]) -> None:
        """Analyze performance trends"""
        if not perf_events:
            return
        
        # Calculate average response times by operation
        operation_times = {}
        for event in perf_events:
            operation = event.properties.get("operation", "unknown")
            duration = event.metrics.get("duration_ms", 0)
            
            if operation not in operation_times:
                operation_times[operation] = []
            operation_times[operation].append(duration)
        
        # Update performance metrics
        if self.metrics_collector:
            for operation, times in operation_times.items():
                avg_time = sum(times) / len(times)
                self.metrics_collector.set_gauge(
                    "xencode_avg_operation_time_ms",
                    avg_time,
                    labels={"operation": operation}
                )
    
    async def _analyze_user_activity(self, user_events: List[TrackedEvent]) -> None:
        """Analyze user activity patterns"""
        if not user_events:
            return
        
        # Count unique active users
        active_users = set()
        for event in user_events:
            if event.context.user_id:
                active_users.add(event.context.user_id)
        
        # Update user metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge("xencode_active_users", len(active_users))
    
    # Public API methods
    
    def track_user_action(self, action: str, user_id: str, **kwargs) -> Optional[str]:
        """Track user action"""
        if self.event_tracker:
            return self.event_tracker.track_user_action(action, user_id, **kwargs)
        return None
    
    def track_system_event(self, event: str, component: str, **kwargs) -> Optional[str]:
        """Track system event"""
        if self.event_tracker:
            return self.event_tracker.track_system_event(event, component, **kwargs)
        return None
    
    def track_performance(self, operation: str, duration_ms: float, component: str, **kwargs) -> Optional[str]:
        """Track performance event"""
        if self.event_tracker:
            return self.event_tracker.track_performance_event(operation, duration_ms, component, **kwargs)
        return None
    
    def track_error(self, error_type: str, error_message: str, component: str, **kwargs) -> Optional[str]:
        """Track error event"""
        if self.event_tracker:
            return self.event_tracker.track_error_event(error_type, error_message, component, **kwargs)
        return None
    
    def track_ai_interaction(self, model: str, operation: str, user_id: str, duration_ms: float, **kwargs) -> Optional[str]:
        """Track AI interaction"""
        if self.event_tracker:
            return self.event_tracker.track_ai_interaction(model, operation, user_id, duration_ms, **kwargs)
        return None
    
    def increment_counter(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric"""
        if self.metrics_collector:
            self.metrics_collector.increment_counter(metric_name, value, labels)
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric"""
        if self.metrics_collector:
            self.metrics_collector.set_gauge(metric_name, value, labels)
    
    def observe_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe histogram metric"""
        if self.metrics_collector:
            self.metrics_collector.observe_histogram(metric_name, value, labels)
    
    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Time operation with histogram"""
        if self.metrics_collector:
            return self.metrics_collector.time_histogram(metric_name, labels)
        else:
            # Return a no-op context manager
            return self._NoOpTimer()
    
    class _NoOpTimer:
        """No-op timer when metrics are disabled"""
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    def get_recent_events(self, hours: int = 24, limit: int = 1000) -> List[TrackedEvent]:
        """Get recent events"""
        if self.event_tracker:
            return self.event_tracker.get_recent_events(hours, limit)
        return []
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[TrackedEvent]:
        """Get events for user"""
        if self.event_tracker:
            return self.event_tracker.get_events_by_user(user_id, limit)
        return []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            'analytics_engine_running': self._running,
            'config': {
                'enable_metrics': self.config.enable_metrics,
                'enable_events': self.config.enable_events,
                'enable_prometheus': self.config.enable_prometheus,
                'metrics_port': self.config.metrics_port
            }
        }
        
        if self.metrics_collector:
            summary['metrics'] = self.metrics_collector.get_metrics_summary()
        
        if self.event_tracker:
            summary['events'] = self.event_tracker.get_statistics()
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics"""
        if self.metrics_collector:
            return self.metrics_collector.export_prometheus_metrics()
        return "# Metrics collector not available\n"
    
    def generate_analytics_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'time_period_hours': hours,
            'summary': self.get_metrics_summary()
        }
        
        if self.event_tracker:
            recent_events = self.event_tracker.get_recent_events(hours)
            
            # Event statistics
            events_by_category = {}
            events_by_type = {}
            unique_users = set()
            
            for event in recent_events:
                # By category
                category = event.category.value
                events_by_category[category] = events_by_category.get(category, 0) + 1
                
                # By type
                events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
                
                # Unique users
                if event.context.user_id:
                    unique_users.add(event.context.user_id)
            
            report['events'] = {
                'total_events': len(recent_events),
                'events_by_category': events_by_category,
                'events_by_type': events_by_type,
                'unique_users': len(unique_users)
            }
            
            # Error analysis
            error_events = [e for e in recent_events if e.category == EventCategory.ERROR]
            if error_events:
                error_types = {}
                for event in error_events:
                    error_type = event.properties.get("error_type", "unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                report['errors'] = {
                    'total_errors': len(error_events),
                    'error_types': error_types
                }
            
            # Performance analysis
            perf_events = [e for e in recent_events if e.category == EventCategory.PERFORMANCE]
            if perf_events:
                total_duration = sum(e.metrics.get("duration_ms", 0) for e in perf_events)
                avg_duration = total_duration / len(perf_events) if perf_events else 0
                
                report['performance'] = {
                    'total_operations': len(perf_events),
                    'average_duration_ms': avg_duration,
                    'total_duration_ms': total_duration
                }
        
        return report


# Global analytics engine instance
analytics_engine = AnalyticsEngine()