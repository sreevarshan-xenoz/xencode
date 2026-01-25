#!/usr/bin/env python3
"""
Advanced Analytics and Monitoring System for Xencode

Comprehensive analytics, monitoring, and performance tracking system
for the Xencode AI assistant platform.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import statistics
import psutil
import sqlite3
from pathlib import Path
import threading
from collections import defaultdict, deque
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
import aiofiles

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: float
    metric_type: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class SystemEvent:
    """System event for monitoring"""
    timestamp: float
    event_type: str
    severity: str  # info, warning, error, critical
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """Detected usage pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: float  # 0-1
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    optimization_id: str
    title: str
    description: str
    potential_savings: float  # in dollars or percentage
    implementation_effort: str  # low, medium, high
    impact_score: float  # 0-1
    recommended_actions: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and stores performance metrics"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".xencode" / "analytics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory buffers for recent metrics
        self.metric_buffer: List[PerformanceMetric] = []
        self.event_buffer: List[SystemEvent] = []
        
        # Locks for thread safety
        self.metric_lock = threading.Lock()
        self.event_lock = threading.Lock()
        
        # Background collection tasks
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None

    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_type TEXT,
                    value REAL,
                    tags TEXT,
                    unit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    severity TEXT,
                    message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)")

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self.metric_lock:
            self.metric_buffer.append(metric)
            
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (timestamp, metric_type, value, tags, unit)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.timestamp,
                    metric.metric_type,
                    metric.value,
                    json.dumps(metric.tags),
                    metric.unit
                ))

    def record_event(self, event: SystemEvent):
        """Record a system event"""
        with self.event_lock:
            self.event_buffer.append(event)
            
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events (timestamp, event_type, severity, message, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event.timestamp,
                    event.event_type,
                    event.severity,
                    event.message,
                    json.dumps(event.metadata)
                ))

    def get_metrics_by_type(self, metric_type: str, 
                           start_time: float = None, 
                           end_time: float = None) -> List[PerformanceMetric]:
        """Get metrics of a specific type within time range"""
        query = "SELECT timestamp, metric_type, value, tags, unit FROM metrics WHERE metric_type = ?"
        params = [metric_type]
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"  # Limit to prevent huge result sets
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                results.append(PerformanceMetric(
                    timestamp=row[0],
                    metric_type=row[1],
                    value=row[2],
                    tags=json.loads(row[3]) if row[3] else {},
                    unit=row[4]
                ))
        
        return results

    def get_events_by_severity(self, severity: str, 
                              start_time: float = None, 
                              end_time: float = None) -> List[SystemEvent]:
        """Get events of a specific severity within time range"""
        query = "SELECT timestamp, event_type, severity, message, metadata FROM events WHERE severity = ?"
        params = [severity]
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                results.append(SystemEvent(
                    timestamp=row[0],
                    event_type=row[1],
                    severity=row[2],
                    message=row[3],
                    metadata=json.loads(row[4]) if row[4] else {}
                ))
        
        return results

    def get_aggregated_metrics(self, metric_type: str, 
                              aggregation_window_minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated metrics for a time window"""
        end_time = time.time()
        start_time = end_time - (aggregation_window_minutes * 60)
        
        query = """
            SELECT 
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as count,
                TOTAL(value) as total_value
            FROM metrics 
            WHERE metric_type = ? AND timestamp >= ? AND timestamp <= ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (metric_type, start_time, end_time))
            row = cursor.fetchone()
            
            if row:
                return {
                    "average": row[0],
                    "minimum": row[1],
                    "maximum": row[2],
                    "count": row[3],
                    "total": row[4],
                    "window_minutes": aggregation_window_minutes
                }
        
        return {}

    async def start_background_collection(self):
        """Start background metrics collection"""
        self.running = True
        
        async def collection_loop():
            while self.running:
                try:
                    # Collect system metrics
                    await self._collect_system_metrics()
                    
                    # Collect application metrics
                    await self._collect_application_metrics()
                    
                    # Sleep before next collection
                    await asyncio.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
                    await asyncio.sleep(5)  # Brief pause before retrying
        
        self.collection_task = asyncio.create_task(collection_loop())

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric(PerformanceMetric(
            timestamp=time.time(),
            metric_type="system.cpu.usage_percent",
            value=cpu_percent,
            unit="%"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric(PerformanceMetric(
            timestamp=time.time(),
            metric_type="system.memory.usage_percent",
            value=memory.percent,
            unit="%"
        ))
        
        self.record_metric(PerformanceMetric(
            timestamp=time.time(),
            metric_type="system.memory.available_mb",
            value=memory.available / (1024 * 1024),
            unit="MB"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.record_metric(PerformanceMetric(
            timestamp=time.time(),
            metric_type="system.disk.usage_percent",
            value=(disk.used / disk.total) * 100,
            unit="%"
        ))
        
        # Process count
        process_count = len(psutil.pids())
        self.record_metric(PerformanceMetric(
            timestamp=time.time(),
            metric_type="system.process.count",
            value=process_count,
            unit="count"
        ))

    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        # For now, we'll add some placeholder metrics
        # In a real system, these would come from various application components
        pass

    def stop_background_collection(self):
        """Stop background metrics collection"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()


class UsagePatternAnalyzer:
    """Analyzes usage patterns and trends"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.patterns: List[UsagePattern] = []
        self.trend_analyzer = TrendAnalyzer()

    def detect_usage_patterns(self, days_back: int = 7) -> List[UsagePattern]:
        """Detect usage patterns over the specified time period"""
        end_time = time.time()
        start_time = end_time - (days_back * 24 * 60 * 60)  # Convert days to seconds
        
        # Get metrics for analysis
        response_time_metrics = self.metrics_collector.get_metrics_by_type(
            "response.time.ms", start_time, end_time
        )
        
        usage_frequency_metrics = self.metrics_collector.get_metrics_by_type(
            "usage.frequency", start_time, end_time
        )
        
        patterns = []
        
        # Detect response time patterns
        if response_time_metrics:
            avg_response_time = statistics.mean([m.value for m in response_time_metrics])
            patterns.append(UsagePattern(
                pattern_id=f"response_time_{uuid.uuid4()}",
                pattern_type="performance",
                description=f"Average response time: {avg_response_time:.2f}ms",
                frequency=len(response_time_metrics) / days_back,  # per day
                confidence=0.8,
                metadata={"average_response_time_ms": avg_response_time}
            ))
        
        # Detect usage frequency patterns
        if usage_frequency_metrics:
            avg_daily_usage = len(usage_frequency_metrics) / days_back
            patterns.append(UsagePattern(
                pattern_id=f"usage_freq_{uuid.uuid4()}",
                pattern_type="usage",
                description=f"Average daily usage: {avg_daily_usage:.2f} requests",
                frequency=avg_daily_usage,
                confidence=0.7,
                metadata={"average_daily_requests": avg_daily_usage}
            ))
        
        # Detect peak usage times
        hourly_usage = defaultdict(int)
        for metric in usage_frequency_metrics:
            hour = datetime.fromtimestamp(metric.timestamp).hour
            hourly_usage[hour] += 1
        
        if hourly_usage:
            peak_hour = max(hourly_usage, key=hourly_usage.get)
            patterns.append(UsagePattern(
                pattern_id=f"peak_usage_{uuid.uuid4()}",
                pattern_type="timing",
                description=f"Peak usage hour: {peak_hour}:00",
                frequency=hourly_usage[peak_hour] / days_back,
                confidence=0.9,
                metadata={"peak_hour": peak_hour, "peak_requests": hourly_usage[peak_hour]}
            ))
        
        self.patterns = patterns
        return patterns

    def get_insights(self) -> Dict[str, Any]:
        """Get analytical insights"""
        patterns = self.detect_usage_patterns(days_back=7)
        
        insights = {
            "total_patterns_detected": len(patterns),
            "performance_patterns": [p for p in patterns if p.pattern_type == "performance"],
            "usage_patterns": [p for p in patterns if p.pattern_type == "usage"],
            "timing_patterns": [p for p in patterns if p.pattern_type == "timing"],
            "recommendations": []
        }
        
        # Add recommendations based on patterns
        for pattern in patterns:
            if pattern.pattern_type == "performance" and pattern.confidence > 0.8:
                avg_response = pattern.metadata.get("average_response_time_ms", 0)
                if avg_response > 500:  # More than half a second
                    insights["recommendations"].append({
                        "type": "performance",
                        "message": f"High average response time ({avg_response:.2f}ms) detected. Consider optimization.",
                        "priority": "high"
                    })
        
        return insights


class TrendAnalyzer:
    """Analyzes trends in metrics data"""

    def __init__(self):
        self.trends = {}

    def analyze_trend(self, metric_type: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze trend for a specific metric"""
        end_time = time.time()
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        # Get metrics
        metrics = self._get_metrics_for_trend_analysis(metric_type, start_time, end_time)
        
        if len(metrics) < 2:
            return {
                "metric_type": metric_type,
                "trend_direction": "insufficient_data",
                "trend_strength": 0.0,
                "slope": 0.0,
                "r_squared": 0.0,
                "data_points": len(metrics)
            }
        
        # Calculate trend using linear regression
        timestamps = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        # Normalize timestamps for calculation
        norm_timestamps = [(t - timestamps[0]) for t in timestamps]
        
        # Calculate slope and R-squared
        n = len(values)
        sum_x = sum(norm_timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(norm_timestamps, values))
        sum_x2 = sum(x * x for x in norm_timestamps)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Calculate R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_reg = sum((slope * x + (mean_y - slope * sum_x / n) - mean_y) ** 2 for x in norm_timestamps)
        r_squared = ss_reg / ss_tot if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0.1:
            direction = "increasing"
        elif slope < -0.1:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calculate trend strength (absolute slope normalized)
        max_value = max(values) if values else 1
        strength = min(abs(slope) / (max_value / 10 if max_value > 0 else 1), 1.0)
        
        return {
            "metric_type": metric_type,
            "trend_direction": direction,
            "trend_strength": strength,
            "slope": slope,
            "r_squared": r_squared,
            "data_points": len(metrics),
            "start_value": values[0] if values else None,
            "end_value": values[-1] if values else None
        }

    def _get_metrics_for_trend_analysis(self, metric_type: str, start_time: float, end_time: float):
        """Get metrics for trend analysis (could be optimized for large datasets)"""
        # This is a simplified version - in production, you'd want to aggregate data
        # to avoid loading too much into memory
        all_metrics = self._get_all_metrics_of_type(metric_type)
        
        # Filter by time range
        filtered_metrics = [m for m in all_metrics if start_time <= m.timestamp <= end_time]
        
        # Sort by timestamp
        filtered_metrics.sort(key=lambda m: m.timestamp)
        
        return filtered_metrics

    def _get_all_metrics_of_type(self, metric_type: str):
        """Get all metrics of a specific type (placeholder - would connect to DB in real implementation)"""
        # This would normally query the database
        # For now, returning empty list as this is handled by MetricsCollector
        return []


class CostOptimizer:
    """Analyzes costs and provides optimization recommendations"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

    def analyze_costs(self) -> List[CostOptimization]:
        """Analyze costs and provide optimization recommendations"""
        optimizations = []
        
        # Analyze resource usage patterns
        cpu_metrics = self.metrics_collector.get_metrics_by_type("system.cpu.usage_percent")
        memory_metrics = self.metrics_collector.get_metrics_by_type("system.memory.usage_percent")
        
        # Check for consistently low resource usage (potential for downsizing)
        if cpu_metrics:
            avg_cpu = statistics.mean([m.value for m in cpu_metrics])
            if avg_cpu < 20:  # Less than 20% CPU usage
                optimizations.append(CostOptimization(
                    optimization_id=f"cpu_downsize_{uuid.uuid4()}",
                    title="CPU Resource Optimization",
                    description="Average CPU usage is consistently low (<20%)",
                    potential_savings=0.3,  # 30% cost reduction potential
                    implementation_effort="medium",
                    impact_score=0.8,
                    recommended_actions=[
                        "Consider using smaller instance types",
                        "Implement auto-scaling based on demand",
                        "Schedule downtime for non-critical periods"
                    ]
                ))
        
        if memory_metrics:
            avg_memory = statistics.mean([m.value for m in memory_metrics])
            if avg_memory < 30:  # Less than 30% memory usage
                optimizations.append(CostOptimization(
                    optimization_id=f"memory_opt_{uuid.uuid4()}",
                    title="Memory Resource Optimization",
                    description="Average memory usage is consistently low (<30%)",
                    potential_savings=0.25,  # 25% cost reduction potential
                    implementation_effort="low",
                    impact_score=0.7,
                    recommended_actions=[
                        "Downsize memory allocation",
                        "Implement memory pooling",
                        "Optimize application memory usage"
                    ]
                ))
        
        # Check for performance issues that might lead to over-provisioning
        response_time_metrics = self.metrics_collector.get_metrics_by_type("response.time.ms")
        if response_time_metrics:
            avg_response_time = statistics.mean([m.value for m in response_time_metrics])
            if avg_response_time > 1000:  # More than 1 second
                optimizations.append(CostOptimization(
                    optimization_id=f"perf_opt_{uuid.uuid4()}",
                    title="Performance Optimization",
                    description="High response times may indicate inefficient resource usage",
                    potential_savings=0.15,  # Indirect savings through efficiency
                    implementation_effort="high",
                    impact_score=0.9,
                    recommended_actions=[
                        "Profile and optimize slow operations",
                        "Implement caching strategies",
                        "Optimize database queries",
                        "Upgrade to faster storage"
                    ]
                ))
        
        return optimizations

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost analysis summary"""
        optimizations = self.analyze_costs()
        
        total_potential_savings = sum(opt.potential_savings for opt in optimizations)
        high_priority_optimizations = [opt for opt in optimizations if opt.impact_score > 0.7]
        
        return {
            "total_optimizations_identified": len(optimizations),
            "high_priority_optimizations": len(high_priority_optimizations),
            "total_potential_savings_fraction": total_potential_savings,
            "estimated_monthly_savings_usd": total_potential_savings * 1000,  # Placeholder calculation
            "optimization_categories": {
                "resource_optimization": len([opt for opt in optimizations if "Resource" in opt.title]),
                "performance_optimization": len([opt for opt in optimizations if "Performance" in opt.title]),
                "efficiency_optimization": len([opt for opt in optimizations if "Optimization" in opt.title])
            }
        }


class AnalyticsDashboard:
    """Real-time analytics dashboard"""

    def __init__(self, metrics_collector: MetricsCollector, 
                 pattern_analyzer: UsagePatternAnalyzer, 
                 cost_optimizer: CostOptimizer):
        self.metrics_collector = metrics_collector
        self.pattern_analyzer = pattern_analyzer
        self.cost_optimizer = cost_optimizer
        self.refresh_interval = 5  # seconds

    def create_system_health_table(self) -> Table:
        """Create system health monitoring table"""
        table = Table(title="System Health")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        # Get latest metrics
        cpu_metrics = self.metrics_collector.get_metrics_by_type("system.cpu.usage_percent", 
                                                               time.time() - 300, time.time())  # Last 5 minutes
        memory_metrics = self.metrics_collector.get_metrics_by_type("system.memory.usage_percent",
                                                                  time.time() - 300, time.time())
        
        if cpu_metrics:
            latest_cpu = cpu_metrics[0].value
            cpu_status = "🔴" if latest_cpu > 80 else "🟡" if latest_cpu > 60 else "🟢"
            table.add_row("CPU Usage", f"{latest_cpu:.1f}%", cpu_status)
        else:
            table.add_row("CPU Usage", "N/A", "❓")

        if memory_metrics:
            latest_memory = memory_metrics[0].value
            memory_status = "🔴" if latest_memory > 85 else "🟡" if latest_memory > 70 else "🟢"
            table.add_row("Memory Usage", f"{latest_memory:.1f}%", memory_status)
        else:
            table.add_row("Memory Usage", "N/A", "❓")

        # Add disk usage
        disk_metrics = self.metrics_collector.get_metrics_by_type("system.disk.usage_percent",
                                                                time.time() - 300, time.time())
        if disk_metrics:
            latest_disk = disk_metrics[0].value
            disk_status = "🔴" if latest_disk > 90 else "🟡" if latest_disk > 75 else "🟢"
            table.add_row("Disk Usage", f"{latest_disk:.1f}%", disk_status)
        else:
            table.add_row("Disk Usage", "N/A", "❓")

        # Add process count
        process_metrics = self.metrics_collector.get_metrics_by_type("system.process.count",
                                                                   time.time() - 300, time.time())
        if process_metrics:
            latest_processes = process_metrics[0].value
            process_status = "🔴" if latest_processes > 500 else "🟡" if latest_processes > 200 else "🟢"
            table.add_row("Processes", f"{int(latest_processes)}", process_status)
        else:
            table.add_row("Processes", "N/A", "❓")

        return table

    def create_performance_table(self) -> Table:
        """Create performance metrics table"""
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Average", style="magenta")
        table.add_column("Min", style="green")
        table.add_column("Max", style="red")
        table.add_column("Unit", style="blue")

        # Response time metrics
        response_agg = self.metrics_collector.get_aggregated_metrics("response.time.ms", 60)
        if response_agg["count"] > 0:
            table.add_row(
                "Response Time",
                f"{response_agg['average']:.2f}",
                f"{response_agg['minimum']:.2f}",
                f"{response_agg['maximum']:.2f}",
                "ms"
            )

        # Request rate metrics
        request_agg = self.metrics_collector.get_aggregated_metrics("request.rate", 60)
        if request_agg["count"] > 0:
            table.add_row(
                "Requests/Minute",
                f"{response_agg['average']:.2f}",
                f"{response_agg['minimum']:.2f}",
                f"{response_agg['maximum']:.2f}",
                "req/min"
            )

        return table

    def create_insights_panel(self) -> Panel:
        """Create insights and recommendations panel"""
        insights = self.pattern_analyzer.get_insights()
        cost_summary = self.cost_optimizer.get_cost_summary()

        content = f"""
[bold]Usage Insights:[/bold]
• Patterns Detected: {insights['total_patterns_detected']}
• Performance Issues: {len(insights['performance_patterns'])}
• Peak Usage Hours: {len(insights['timing_patterns'])}

[bold]Cost Optimization:[/bold]
• Potential Savings: {cost_summary['total_potential_savings_fraction']*100:.1f}%
• High Priority Items: {cost_summary['high_priority_optimizations']}
• Estimated Monthly Savings: ${cost_summary['estimated_monthly_savings_usd']:.2f}

[bold]Recommendations:[/bold]
"""
        for rec in insights.get('recommendations', []):
            content += f"• {rec['message']} [Priority: {rec['priority']}]\n"

        return Panel(content, title="Analytics Insights & Recommendations", border_style="yellow")

    async def display_dashboard(self):
        """Display the analytics dashboard"""
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                # Create dashboard components
                health_table = self.create_system_health_table()
                performance_table = self.create_performance_table()
                insights_panel = self.create_insights_panel()

                # Combine into a layout
                from rich.layout import Layout
                layout = Layout()

                layout.split_column(
                    Layout(health_table, size=7),
                    Layout(performance_table, ratio=1),
                    Layout(insights_panel, ratio=2)
                )

                live.update(layout)
                await asyncio.sleep(self.refresh_interval)


class AnalyticsReportingSystem:
    """Generates reports from analytics data"""

    def __init__(self, metrics_collector: MetricsCollector,
                 pattern_analyzer: UsagePatternAnalyzer,
                 cost_optimizer: CostOptimizer):
        self.metrics_collector = metrics_collector
        self.pattern_analyzer = pattern_analyzer
        self.cost_optimizer = cost_optimizer

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily analytics report"""
        yesterday = time.time() - (24 * 60 * 60)
        
        report = {
            "report_date": datetime.now().isoformat(),
            "period": "daily",
            "system_summary": self._get_system_summary(yesterday, time.time()),
            "usage_patterns": [p.__dict__ for p in self.pattern_analyzer.detect_usage_patterns(days_back=1)],
            "cost_analysis": self.cost_optimizer.get_cost_summary(),
            "events_summary": self._get_events_summary(yesterday, time.time()),
            "performance_summary": self._get_performance_summary(yesterday, time.time())
        }
        
        return report

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly analytics report"""
        week_ago = time.time() - (7 * 24 * 60 * 60)
        
        report = {
            "report_date": datetime.now().isoformat(),
            "period": "weekly",
            "system_summary": self._get_system_summary(week_ago, time.time()),
            "usage_patterns": [p.__dict__ for p in self.pattern_analyzer.detect_usage_patterns(days_back=7)],
            "cost_analysis": self.cost_optimizer.get_cost_summary(),
            "events_summary": self._get_events_summary(week_ago, time.time()),
            "performance_summary": self._get_performance_summary(week_ago, time.time()),
            "trends": self._analyze_trends(week_ago, time.time())
        }
        
        return report

    def _get_system_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get system summary for the time period"""
        cpu_metrics = self.metrics_collector.get_metrics_by_type("system.cpu.usage_percent", start_time, end_time)
        memory_metrics = self.metrics_collector.get_metrics_by_type("system.memory.usage_percent", start_time, end_time)
        
        summary = {
            "total_metrics_collected": len(cpu_metrics) + len(memory_metrics),
            "cpu": {},
            "memory": {}
        }
        
        if cpu_metrics:
            values = [m.value for m in cpu_metrics]
            summary["cpu"] = {
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        if memory_metrics:
            values = [m.value for m in memory_metrics]
            summary["memory"] = {
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        return summary

    def _get_events_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get events summary for the time period"""
        all_events = []
        for severity in ["info", "warning", "error", "critical"]:
            events = self.metrics_collector.get_events_by_severity(severity, start_time, end_time)
            all_events.extend(events)
        
        summary = {
            "total_events": len(all_events),
            "by_severity": {
                "info": len([e for e in all_events if e.severity == "info"]),
                "warning": len([e for e in all_events if e.severity == "warning"]),
                "error": len([e for e in all_events if e.severity == "error"]),
                "critical": len([e for e in all_events if e.severity == "critical"])
            },
            "recent_events": [e.__dict__ for e in all_events[:10]]  # Last 10 events
        }
        
        return summary

    def _get_performance_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get performance summary for the time period"""
        response_metrics = self.metrics_collector.get_metrics_by_type("response.time.ms", start_time, end_time)
        
        summary = {
            "response_time": {}
        }
        
        if response_metrics:
            values = [m.value for m in response_metrics]
            summary["response_time"] = {
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "percentile_95": self._calculate_percentile(values, 95),
                "percentile_99": self._calculate_percentile(values, 99)
            }
        
        return summary

    def _analyze_trends(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze trends for the time period"""
        analyzer = TrendAnalyzer()
        
        trends = {
            "cpu_usage": analyzer.analyze_trend("system.cpu.usage_percent", 7),
            "memory_usage": analyzer.analyze_trend("system.memory.usage_percent", 7),
            "response_time": analyzer.analyze_trend("response.time.ms", 7)
        }
        
        return trends

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to file"""
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(report, indent=2))


async def main():
    """Main function to demonstrate the analytics system"""
    console.print("[bold green]📊 Initializing Advanced Analytics System[/bold green]")
    
    # Initialize components
    metrics_collector = MetricsCollector()
    pattern_analyzer = UsagePatternAnalyzer(metrics_collector)
    cost_optimizer = CostOptimizer(metrics_collector)
    dashboard = AnalyticsDashboard(metrics_collector, pattern_analyzer, cost_optimizer)
    reporting_system = AnalyticsReportingSystem(metrics_collector, pattern_analyzer, cost_optimizer)
    
    # Start background collection
    await metrics_collector.start_background_collection()
    
    console.print("[blue]✅ Analytics system initialized[/blue]")
    console.print("[yellow]📊 Collecting metrics for 10 seconds...[/yellow]")
    
    # Let it collect some data
    await asyncio.sleep(10)
    
    # Generate reports
    console.print("[blue]📄 Generating reports...[/blue]")
    daily_report = reporting_system.generate_daily_report()
    weekly_report = reporting_system.generate_weekly_report()
    
    console.print(f"[green]✅ Daily report generated: {len(daily_report['system_summary'])} metrics[/green]")
    console.print(f"[green]✅ Weekly report generated: {len(weekly_report['trends'])} trends analyzed[/green]")
    
    # Show some insights
    insights = pattern_analyzer.get_insights()
    console.print(f"[blue]🔍 Detected {insights['total_patterns_detected']} usage patterns[/blue]")
    
    cost_summary = cost_optimizer.get_cost_summary()
    console.print(f"[blue]💰 Identified potential for {cost_summary['total_potential_savings_fraction']*100:.1f}% cost savings[/blue]")
    
    # Stop collection
    metrics_collector.stop_background_collection()
    
    console.print("[green]✅ Advanced Analytics System Demo Completed[/green]")


if __name__ == "__main__":
    # Don't run by default to avoid external dependencies
    # asyncio.run(main())
    pass