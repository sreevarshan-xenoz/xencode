"""
Performance Monitoring Dashboard for Xencode TUI

Real-time performance metrics and system monitoring.
"""

from typing import Dict, List, Optional, Any
import time
import asyncio
from datetime import datetime
from rich.text import Text
from rich.table import Table
from textual.widgets import Static, DataTable, Label, Sparkline
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.reactive import reactive
from textual.timer import Timer
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
from dataclasses import dataclass, field


@dataclass
class PerformanceMetric:
    """Data class for performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_sent: int
    network_recv: int
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    active_models: List[str] = field(default_factory=list)


class PerformanceChart(Sparkline):
    """Custom sparkline for performance metrics"""
    pass


class PerformanceDashboard(Container):
    """Performance monitoring dashboard with real-time metrics"""

    DEFAULT_CSS = """
    PerformanceDashboard {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    PerformanceDashboard .dashboard-header {
        text-style: bold;
        color: $accent;
        content-align: center middle;
        width: 100%;
        margin-bottom: 1;
    }

    PerformanceDashboard .metric-card {
        height: 10;
        border: solid $secondary;
        padding: 1;
        background: $panel;
        margin: 0.5 0;
    }

    PerformanceDashboard .metric-value {
        text-style: bold;
        font-size: large;
    }

    PerformanceDashboard .metric-label {
        color: $text-muted;
    }

    PerformanceDashboard .chart-container {
        height: 8;
        border: solid $secondary;
        padding: 1;
        background: $panel;
        margin: 0.5 0;
    }

    PerformanceDashboard DataTable {
        height: 20;
        margin: 1 0;
    }

    PerformanceDashboard .status-ok {
        color: $success;
    }

    PerformanceDashboard .status-warning {
        color: $warning;
    }

    PerformanceDashboard .status-critical {
        color: $error;
    }
    """

    # Reactive properties
    current_metrics = reactive(None)
    metric_history = reactive(lambda: [])
    update_interval = reactive(2.0)  # seconds

    def __init__(self, *args, **kwargs):
        """Initialize performance dashboard"""
        super().__init__(*args, **kwargs)
        self.border_title = "📊 Performance Dashboard"
        self.metric_history = []
        self.update_timer: Optional[Timer] = None
        self.is_monitoring = True

    def compose(self):
        """Compose the performance dashboard"""
        yield Label("Xencode Performance Dashboard", classes="dashboard-header")
        
        # System metrics grid
        with Grid(id="system-metrics-grid", classes="metrics-grid"):
            self.cpu_card = Static(id="cpu-card", classes="metric-card")
            self.memory_card = Static(id="memory-card", classes="metric-card")
            self.disk_card = Static(id="disk-card", classes="metric-card")
            self.gpu_card = Static(id="gpu-card", classes="metric-card")
            self.network_card = Static(id="network-card", classes="metric-card")
            self.response_card = Static(id="response-card", classes="metric-card")
            
            yield self.cpu_card
            yield self.memory_card
            yield self.disk_card
            yield self.gpu_card
            yield self.network_card
            yield self.response_card

        # Charts
        yield Label("Performance Charts", classes="section-title")
        
        with Horizontal():
            with Vertical():
                yield Label("CPU Usage (%)", classes="chart-title")
                self.cpu_chart = Sparkline(id="cpu-chart", data=[], summary="CPU Usage")
                yield self.cpu_chart
            with Vertical():
                yield Label("Memory Usage (%)", classes="chart-title")
                self.memory_chart = Sparkline(id="memory-chart", data=[], summary="Memory Usage")
                yield self.memory_chart

        with Horizontal():
            with Vertical():
                yield Label("Response Time (ms)", classes="chart-title")
                self.response_chart = Sparkline(id="response-chart", data=[], summary="Response Time")
                yield self.response_chart
            with Vertical():
                yield Label("Throughput (req/s)", classes="chart-title")
                self.throughput_chart = Sparkline(id="throughput-chart", data=[], summary="Throughput")
                yield self.throughput_chart

        # Detailed metrics table
        yield Label("Detailed Metrics", classes="section-title")
        self.metrics_table = DataTable(id="metrics-table", zebra_stripes=True)
        yield self.metrics_table

        # Controls
        with Horizontal():
            yield Label("Update Interval:", classes="control-label")
            # In a real implementation, this would be an input widget
            yield Label(f"{self.update_interval}s", classes="control-value")
            yield Label("Status:", classes="control-label")
            self.status_label = Label("Monitoring", id="status-label", classes="status-ok")
            yield self.status_label

    def on_mount(self) -> None:
        """Called when widget is mounted"""
        self.update_timer = self.set_interval(self.update_interval, self.collect_metrics)
        self.update_display()
        
    def collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_sent = net_io.bytes_sent
            network_recv = net_io.bytes_recv
            
            # GPU metrics (if available)
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            try:
                if GPUtil is not None:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_percent = gpu.load * 100
                        gpu_memory_percent = gpu.memoryUtil * 100
            except:
                pass
            
            # Simulate response time and throughput metrics
            # In a real implementation, these would come from the AI model requests
            response_time = round(0.5 + (hash(time.time()) % 1000) / 1000, 3)  # Simulated response time
            throughput = round(5 + (hash(time.time()) % 20), 2)  # Simulated throughput
            
            # Get active models (simulated)
            active_models = ["qwen3:4b", "llama3.1:8b"]  # Simulated active models
            
            # Create metric object
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_sent=network_sent,
                network_recv=network_recv,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                response_time=response_time,
                throughput=throughput,
                active_models=active_models
            )
            
            # Store in history (keep last 50 measurements)
            self.metric_history.append(metric)
            if len(self.metric_history) > 50:
                self.metric_history = self.metric_history[-50:]
            
            self.current_metrics = metric
            self.update_display()
            
        except Exception as e:
            # Log error but don't crash the dashboard
            print(f"Error collecting metrics: {e}")

    def update_display(self):
        """Update the display with current metrics"""
        if not self.current_metrics:
            return

        # Update metric cards
        self.update_cpu_card()
        self.update_memory_card()
        self.update_disk_card()
        self.update_gpu_card()
        self.update_network_card()
        self.update_response_card()

        # Update charts
        self.update_charts()

        # Update metrics table
        self.update_metrics_table()

    def update_cpu_card(self):
        """Update CPU metric card"""
        metric = self.current_metrics
        status_class = self._get_status_class(metric.cpu_percent, 50, 80)
        
        card_content = (
            f"[b]CPU[/b]\n"
            f"[{status_class}]{metric.cpu_percent:.1f}%[/]\n"
            f"Usage\n"
            f"Processes: {len(psutil.pids())}"
        )
        self.cpu_card.update(card_content)

    def update_memory_card(self):
        """Update Memory metric card"""
        metric = self.current_metrics
        status_class = self._get_status_class(metric.memory_percent, 60, 85)
        
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        total_gb = memory.total / (1024**3)
        
        card_content = (
            f"[b]Memory[/b]\n"
            f"[{status_class}]{metric.memory_percent:.1f}%[/]\n"
            f"{used_gb:.1f}GB / {total_gb:.1f}GB\n"
            f"Available: {(memory.available / (1024**3)):.1f}GB"
        )
        self.memory_card.update(card_content)

    def update_disk_card(self):
        """Update Disk metric card"""
        metric = self.current_metrics
        status_class = self._get_status_class(metric.disk_percent, 70, 90)
        
        disk = psutil.disk_usage('/')
        used_gb = disk.used / (1024**3)
        total_gb = disk.total / (1024**3)
        
        card_content = (
            f"[b]Disk[/b]\n"
            f"[{status_class}]{metric.disk_percent:.1f}%[/]\n"
            f"{used_gb:.1f}GB / {total_gb:.1f}GB\n"
            f"Free: {(disk.free / (1024**3)):.1f}GB"
        )
        self.disk_card.update(card_content)

    def update_gpu_card(self):
        """Update GPU metric card"""
        metric = self.current_metrics
        if metric.gpu_percent > 0:
            status_class = self._get_status_class(metric.gpu_percent, 50, 80)
            card_content = (
                f"[b]GPU[/b]\n"
                f"[{status_class}]{metric.gpu_percent:.1f}%[/]\n"
                f"Memory: {metric.gpu_memory_percent:.1f}%\n"
                f"Active: {'Yes' if metric.gpu_percent > 0 else 'No'}"
            )
        else:
            card_content = (
                f"[b]GPU[/b]\n"
                f"N/A\n"
                f"No GPU detected\n"
                f"Using CPU"
            )
        self.gpu_card.update(card_content)

    def update_network_card(self):
        """Update Network metric card"""
        metric = self.current_metrics
        
        # Convert bytes to MB for display
        sent_mb = metric.network_sent / (1024 * 1024)
        recv_mb = metric.network_recv / (1024 * 1024)
        
        card_content = (
            f"[b]Network[/b]\n"
            f"↑ {sent_mb:.1f} MB\n"
            f"↓ {recv_mb:.1f} MB\n"
            f"Active Models: {len(metric.active_models)}"
        )
        self.network_card.update(card_content)

    def update_response_card(self):
        """Update Response metric card"""
        metric = self.current_metrics
        status_class = self._get_status_class(metric.response_time * 1000, 1000, 3000)  # Convert to ms
        
        card_content = (
            f"[b]Response[/b]\n"
            f"[{status_class}]{metric.response_time:.3f}s[/]\n"
            f"Throughput: {metric.throughput:.1f}/s\n"
            f"Active Models: {len(metric.active_models)}"
        )
        self.response_card.update(card_content)

    def update_charts(self):
        """Update performance charts"""
        if not self.metric_history:
            return

        # Get last 20 metrics for charts
        recent_metrics = self.metric_history[-20:]

        # Update CPU chart
        cpu_data = [m.cpu_percent for m in recent_metrics]
        self.cpu_chart.data = cpu_data

        # Update Memory chart
        memory_data = [m.memory_percent for m in recent_metrics]
        self.memory_chart.data = memory_data

        # Update Response chart
        response_data = [m.response_time * 1000 for m in recent_metrics]  # Convert to ms
        self.response_chart.data = response_data

        # Update Throughput chart
        throughput_data = [m.throughput for m in recent_metrics]
        self.throughput_chart.data = throughput_data

    def update_metrics_table(self):
        """Update the detailed metrics table"""
        # Clear existing table
        self.metrics_table.clear()
        
        if not self.metric_history:
            return

        # Add headers
        headers = ["Time", "CPU %", "Mem %", "Disk %", "GPU %", "Resp. Time (s)", "Throughput", "Active Models"]
        self.metrics_table.add_columns(*headers)
        
        # Add recent metrics (last 10)
        recent_metrics = self.metric_history[-10:]
        for metric in reversed(recent_metrics):  # Show newest first
            row = [
                metric.timestamp.strftime("%H:%M:%S"),
                f"{metric.cpu_percent:.1f}",
                f"{metric.memory_percent:.1f}",
                f"{metric.disk_percent:.1f}",
                f"{metric.gpu_percent:.1f}" if metric.gpu_percent > 0 else "N/A",
                f"{metric.response_time:.3f}",
                f"{metric.throughput:.1f}",
                str(len(metric.active_models))
            ]
            self.metrics_table.add_row(*row)

    def _get_status_class(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get CSS class based on metric value"""
        if value >= critical_threshold:
            return "status-critical"
        elif value >= warning_threshold:
            return "status-warning"
        else:
            return "status-ok"

    def action_toggle_monitoring(self) -> None:
        """Toggle performance monitoring on/off"""
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            if self.update_timer:
                self.update_timer.stop()
            self.update_timer = self.set_interval(self.update_interval, self.collect_metrics)
            self.status_label.update("Monitoring")
            self.status_label.classes = "status-ok"
        else:
            if self.update_timer:
                self.update_timer.stop()
            self.status_label.update("Paused")
            self.status_label.classes = "status-warning"

    def action_adjust_interval(self, new_interval: float) -> None:
        """Adjust the update interval"""
        self.update_interval = new_interval
        if self.update_timer:
            self.update_timer.stop()
        self.update_timer = self.set_interval(self.update_interval, self.collect_metrics)
        # Update the display to show new interval
        for child in self.children:
            if hasattr(child, 'id') and child.id == "status-label":
                continue  # Skip status label, we update it separately
        # In a real implementation, we would update the interval display here