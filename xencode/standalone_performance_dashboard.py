#!/usr/bin/env python3
"""
Standalone Performance Monitoring Dashboard

A self-contained performance monitoring dashboard that doesn't depend on
other xencode modules, making it easier to test and deploy independently.
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque, defaultdict
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich import box

# Import system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(str, Enum):
    """Types of metrics being monitored"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    metric_type: MetricType
    severity: AlertSeverity
    title: str
    description: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    comparison: str = "greater_than"


class MetricBuffer:
    """Circular buffer for storing metric history"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
    
    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        self.data.append(value)
        self.timestamps.append(timestamp)
    
    def get_recent(self, seconds: int = 300) -> List[Tuple[float, float]]:
        """Get recent values within specified time window"""
        cutoff_time = time.time() - seconds
        recent_data = []
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp >= cutoff_time:
                recent_data.append((timestamp, self.data[i]))
        
        return recent_data
    
    def get_average(self, seconds: int = 300) -> float:
        """Get average value for time window"""
        recent_data = self.get_recent(seconds)
        if not recent_data:
            return 0.0
        
        return statistics.mean([value for _, value in recent_data])
    
    def get_trend(self, seconds: int = 300) -> str:
        """Analyze trend direction"""
        recent_data = self.get_recent(seconds)
        if len(recent_data) < 2:
            return "stable"
        
        # Simple trend analysis using first and last values
        first_half = recent_data[:len(recent_data)//2]
        second_half = recent_data[len(recent_data)//2:]
        
        if not first_half or not second_half:
            return "stable"
        
        first_avg = statistics.mean([value for _, value in first_half])
        second_avg = statistics.mean([value for _, value in second_half])
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"


class PerformanceMonitor:
    """Core performance monitoring engine"""
    
    def __init__(self):
        self.metric_buffers: Dict[str, MetricBuffer] = defaultdict(lambda: MetricBuffer(1000))
        self.thresholds: Dict[MetricType, MetricThreshold] = self._initialize_default_thresholds()
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance tracking
        self.last_collection_time = time.time()
        self.collection_interval = 5.0  # seconds
        
        # System monitoring
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
    
    def _initialize_default_thresholds(self) -> Dict[MetricType, MetricThreshold]:
        """Initialize default performance thresholds"""
        return {
            MetricType.CPU: MetricThreshold(
                MetricType.CPU, 
                warning_threshold=70.0, 
                critical_threshold=85.0, 
                emergency_threshold=95.0
            ),
            MetricType.MEMORY: MetricThreshold(
                MetricType.MEMORY, 
                warning_threshold=75.0, 
                critical_threshold=90.0, 
                emergency_threshold=98.0
            ),
            MetricType.DISK: MetricThreshold(
                MetricType.DISK, 
                warning_threshold=80.0, 
                critical_threshold=90.0, 
                emergency_threshold=95.0
            ),
            MetricType.RESPONSE_TIME: MetricThreshold(
                MetricType.RESPONSE_TIME, 
                warning_threshold=2.0, 
                critical_threshold=5.0, 
                emergency_threshold=10.0
            ),
            MetricType.ERROR_RATE: MetricThreshold(
                MetricType.ERROR_RATE, 
                warning_threshold=5.0, 
                critical_threshold=10.0, 
                emergency_threshold=25.0
            )
        }
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        if not PSUTIL_AVAILABLE:
            # Return mock data for testing
            return {
                'cpu_usage': 45.0,
                'memory_usage': 60.0,
                'disk_usage': 70.0,
                'process_cpu': 15.0,
                'process_memory_mb': 100.0
            }
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['cpu_usage'] = cpu_percent
            self.metric_buffers['cpu_usage'].add(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            self.metric_buffers['memory_usage'].add(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = disk.percent
            metrics['disk_used_gb'] = disk.used / (1024**3)
            metrics['disk_total_gb'] = disk.total / (1024**3)
            self.metric_buffers['disk_usage'].add(disk.percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['network_bytes_sent'] = network.bytes_sent
            metrics['network_bytes_recv'] = network.bytes_recv
            
            # Process-specific metrics
            metrics['process_cpu'] = self.process.cpu_percent()
            process_memory = self.process.memory_info()
            metrics['process_memory_mb'] = process_memory.rss / (1024**2)
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                metrics['load_1m'] = load_avg[0]
                metrics['load_5m'] = load_avg[1]
                metrics['load_15m'] = load_avg[2]
            except (AttributeError, OSError):
                pass  # Windows doesn't have load average
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def record_performance_metric(self, metric_type: MetricType, value: float) -> None:
        """Record a performance metric"""
        metric_name = metric_type.value
        self.metric_buffers[metric_name].add(value)
        
        # Check thresholds and generate alerts
        self._check_thresholds(metric_type, value)
    
    def _check_thresholds(self, metric_type: MetricType, value: float) -> None:
        """Check if metric value exceeds thresholds"""
        if metric_type not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_type]
        severity = None
        
        if threshold.emergency_threshold and value >= threshold.emergency_threshold:
            severity = AlertSeverity.EMERGENCY
        elif value >= threshold.critical_threshold:
            severity = AlertSeverity.CRITICAL
        elif value >= threshold.warning_threshold:
            severity = AlertSeverity.WARNING
        
        if severity:
            self._create_alert(metric_type, severity, value, threshold)
    
    def _create_alert(self, metric_type: MetricType, severity: AlertSeverity, 
                     current_value: float, threshold: MetricThreshold) -> None:
        """Create a performance alert"""
        alert_id = f"{metric_type.value}_{severity.value}_{int(time.time())}"
        
        # Check if similar alert already exists
        for alert in self.active_alerts:
            if (alert.metric_type == metric_type and 
                alert.severity == severity and 
                not alert.resolved):
                return  # Don't create duplicate alerts
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_type=metric_type,
            severity=severity,
            title=f"{metric_type.value.title()} {severity.value.title()}",
            description=f"{metric_type.value} is {current_value:.2f}, exceeding {severity.value} threshold",
            current_value=current_value,
            threshold_value=threshold.critical_threshold,
            timestamp=datetime.now()
        )
        
        self.active_alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_metric_summary(self, metric_type: MetricType, seconds: int = 300) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        metric_name = metric_type.value
        buffer = self.metric_buffers[metric_name]
        recent_data = buffer.get_recent(seconds)
        
        if not recent_data:
            return {
                'current': 0.0,
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'trend': 'stable',
                'data_points': 0
            }
        
        values = [value for _, value in recent_data]
        
        return {
            'current': values[-1] if values else 0.0,
            'average': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'trend': buffer.get_trend(seconds),
            'data_points': len(values)
        }
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False


class DashboardRenderer:
    """Renders the performance monitoring dashboard"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.monitor = performance_monitor
        self.console = Console()
        self.last_update = time.time()
    
    def create_system_metrics_panel(self) -> Panel:
        """Create system metrics panel"""
        metrics = self.monitor.collect_system_metrics()
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", style="green", width=12)
        table.add_column("Trend", style="yellow", width=12)
        table.add_column("Status", style="white", width=8)
        
        # CPU metrics
        cpu_usage = metrics.get('cpu_usage', 0)
        cpu_trend = self.monitor.metric_buffers['cpu_usage'].get_trend()
        cpu_status = self._get_status_indicator(cpu_usage, 70, 85)
        table.add_row("CPU Usage", f"{cpu_usage:.1f}%", cpu_trend, cpu_status)
        
        # Memory metrics
        memory_usage = metrics.get('memory_usage', 0)
        memory_trend = self.monitor.metric_buffers['memory_usage'].get_trend()
        memory_status = self._get_status_indicator(memory_usage, 75, 90)
        table.add_row("Memory Usage", f"{memory_usage:.1f}%", memory_trend, memory_status)
        
        # Disk metrics
        disk_usage = metrics.get('disk_usage', 0)
        disk_trend = self.monitor.metric_buffers['disk_usage'].get_trend()
        disk_status = self._get_status_indicator(disk_usage, 80, 90)
        table.add_row("Disk Usage", f"{disk_usage:.1f}%", disk_trend, disk_status)
        
        # Process metrics
        process_cpu = metrics.get('process_cpu', 0)
        process_memory = metrics.get('process_memory_mb', 0)
        table.add_row("Process CPU", f"{process_cpu:.1f}%", "stable", "🟢")
        table.add_row("Process Memory", f"{process_memory:.1f}MB", "stable", "🟢")
        
        return Panel(table, title="📊 System Metrics", border_style="green")
    
    def create_alerts_panel(self) -> Panel:
        """Create alerts panel"""
        active_alerts = self.monitor.get_active_alerts()
        
        if not active_alerts:
            return Panel(
                Align.center(Text("✅ No active alerts", style="green")),
                title="🚨 Active Alerts",
                border_style="green"
            )
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Severity", style="red", width=10)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Description", style="white", width=40)
        table.add_column("Time", style="dim", width=12)
        
        for alert in active_alerts[:10]:  # Show only recent 10 alerts
            severity_icon = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.CRITICAL: "🔴",
                AlertSeverity.EMERGENCY: "🚨"
            }.get(alert.severity, "❓")
            
            time_str = alert.timestamp.strftime("%H:%M:%S")
            
            table.add_row(
                f"{severity_icon} {alert.severity.value.upper()}",
                alert.metric_type.value,
                alert.description[:40] + "..." if len(alert.description) > 40 else alert.description,
                time_str
            )
        
        return Panel(table, title="🚨 Active Alerts", border_style="red")
    
    def create_resource_utilization_panel(self) -> Panel:
        """Create resource utilization panel"""
        metrics = self.monitor.collect_system_metrics()
        
        # Create progress bars for resource utilization
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=False
        )
        
        # Add resource utilization bars
        cpu_task = progress.add_task("CPU", total=100, completed=metrics.get('cpu_usage', 0))
        memory_task = progress.add_task("Memory", total=100, completed=metrics.get('memory_usage', 0))
        disk_task = progress.add_task("Disk", total=100, completed=metrics.get('disk_usage', 0))
        
        return Panel(progress, title="💻 Resource Utilization", border_style="yellow")
    
    def _get_status_indicator(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get status indicator based on thresholds"""
        if value >= critical_threshold:
            return "🔴"
        elif value >= warning_threshold:
            return "🟡"
        else:
            return "🟢"
    
    def render_dashboard(self) -> Layout:
        """Render the complete performance monitoring dashboard"""
        layout = Layout()
        
        # Main layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(Panel(
            Align.center(Text("🔍 Xencode Performance Monitoring Dashboard", style="bold cyan")),
            style="white on blue"
        ))
        
        # Main content area
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Left side - main metrics
        layout["left"].split_column(
            Layout(name="system_metrics"),
            Layout(name="resources")
        )
        
        # Right side - alerts
        layout["right"].update(self.create_alerts_panel())
        
        # Update panels
        layout["system_metrics"].update(self.create_system_metrics_panel())
        layout["resources"].update(self.create_resource_utilization_panel())
        
        # Footer
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_interval = time.time() - self.last_update
        layout["footer"].update(Panel(
            Align.center(Text(f"Last Updated: {current_time} | Update Interval: {update_interval:.1f}s", style="dim")),
            style="white on dark_blue"
        ))
        
        self.last_update = time.time()
        return layout


class StandalonePerformanceDashboard:
    """Standalone performance monitoring dashboard"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.renderer = DashboardRenderer(self.performance_monitor)
        self.console = Console()
        self.is_running = False
        
        # Background tasks
        self._monitoring_task = None
    
    async def start(self) -> None:
        """Start the performance monitoring dashboard"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        print("🚀 Standalone Performance Monitoring Dashboard started")
    
    async def stop(self) -> None:
        """Stop the performance monitoring dashboard"""
        self.is_running = False
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        print("🛑 Standalone Performance Monitoring Dashboard stopped")
    
    async def start_live_dashboard(self, refresh_interval: float = 2.0) -> None:
        """Start live dashboard with auto-refresh"""
        await self.start()
        
        with Live(self.renderer.render_dashboard(), refresh_per_second=1/refresh_interval, console=self.console) as live:
            try:
                while self.is_running:
                    live.update(self.renderer.render_dashboard())
                    await asyncio.sleep(refresh_interval)
            except KeyboardInterrupt:
                await self.stop()
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                metrics = self.performance_monitor.collect_system_metrics()
                
                # Record key performance metrics
                if 'cpu_usage' in metrics:
                    self.performance_monitor.record_performance_metric(MetricType.CPU, metrics['cpu_usage'])
                if 'memory_usage' in metrics:
                    self.performance_monitor.record_performance_metric(MetricType.MEMORY, metrics['memory_usage'])
                if 'disk_usage' in metrics:
                    self.performance_monitor.record_performance_metric(MetricType.DISK, metrics['disk_usage'])
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.performance_monitor.alert_callbacks.append(callback)
    
    def generate_sample_data(self) -> None:
        """Generate sample data for demonstration"""
        import random
        
        # Generate sample metrics
        for i in range(50):
            # Simulate varying system load
            cpu_usage = random.uniform(30, 95)
            memory_usage = random.uniform(40, 90)
            disk_usage = random.uniform(50, 85)
            
            self.performance_monitor.record_performance_metric(MetricType.CPU, cpu_usage)
            self.performance_monitor.record_performance_metric(MetricType.MEMORY, memory_usage)
            self.performance_monitor.record_performance_metric(MetricType.DISK, disk_usage)


# Demo function
async def run_standalone_dashboard_demo():
    """Run standalone performance monitoring dashboard demo"""
    console = Console()
    console.print("🚀 Starting Standalone Performance Monitoring Dashboard Demo...\n")
    
    dashboard = StandalonePerformanceDashboard()
    
    # Add alert callback for demonstration
    def alert_callback(alert: PerformanceAlert):
        console.print(f"🚨 ALERT: {alert.title} - {alert.description}")
    
    dashboard.add_alert_callback(alert_callback)
    
    # Generate some sample data
    console.print("📊 Generating sample metrics data...")
    dashboard.generate_sample_data()
    console.print("✅ Sample data generated\n")
    
    console.print("🔄 Starting live dashboard (Press Ctrl+C to stop)...")
    
    try:
        await dashboard.start_live_dashboard(refresh_interval=1.0)
    except KeyboardInterrupt:
        console.print("\n👋 Dashboard stopped")
    finally:
        await dashboard.stop()


if __name__ == "__main__":
    asyncio.run(run_standalone_dashboard_demo())