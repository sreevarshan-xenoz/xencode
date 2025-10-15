# Performance Monitoring Dashboard

## Overview

The Performance Monitoring Dashboard is a comprehensive real-time system monitoring solution for Xencode that provides:

- **Real-time system metrics visualization** with live charts and indicators
- **Performance trend analysis** and anomaly detection
- **Automated alerting** for performance degradation
- **Interactive dashboard** with Rich UI components
- **Integration with Prometheus** metrics collector
- **Historical data analysis** and performance baselines

## Features

### 🔍 Real-time Monitoring
- CPU, memory, and disk usage tracking
- Network I/O monitoring
- Process-specific metrics
- Load average monitoring (Unix-like systems)

### 🚨 Intelligent Alerting
- Configurable thresholds for warning, critical, and emergency levels
- Automatic alert generation and resolution
- Duplicate alert prevention
- Alert callbacks for custom notifications

### 📊 Performance Analytics
- Trend analysis (increasing, decreasing, stable)
- Performance baselines and deviation detection
- Historical data retention with circular buffers
- Metric summaries with statistics

### 🎛️ Interactive Dashboard
- Live updating dashboard with Rich UI
- System metrics panel with status indicators
- Resource utilization progress bars
- Active alerts panel with severity indicators
- Performance trends visualization

## Architecture

### Core Components

1. **MetricBuffer**: Circular buffer for storing metric history
2. **PerformanceMonitor**: Core monitoring engine with threshold checking
3. **DashboardRenderer**: Rich UI components for dashboard visualization
4. **PerformanceMonitoringDashboard**: Main orchestrator with async background tasks

### Data Flow

```
System Metrics → MetricBuffer → Threshold Check → Alert Generation
                      ↓
Performance Monitor → Dashboard Renderer → Live Dashboard
                      ↓
Analytics Integration → Prometheus Metrics → External Monitoring
```

## Installation and Setup

### Dependencies

```bash
pip install rich psutil prometheus-client
```

### Basic Usage

```python
from xencode.standalone_performance_dashboard import StandalonePerformanceDashboard
import asyncio

async def main():
    dashboard = StandalonePerformanceDashboard()
    
    # Generate sample data for demonstration
    dashboard.generate_sample_data()
    
    # Start live dashboard
    await dashboard.start_live_dashboard(refresh_interval=2.0)

asyncio.run(main())
```

### Advanced Usage with Prometheus Integration

```python
from xencode.performance_monitoring_dashboard import PerformanceMonitoringDashboard
from xencode.monitoring.metrics_collector import PrometheusMetricsCollector

# Create Prometheus collector
prometheus_collector = PrometheusMetricsCollector(
    metrics_port=8000,
    collect_system_metrics=True
)

# Create integrated dashboard
dashboard = PerformanceMonitoringDashboard(prometheus_collector)

# Add custom alert callback
def alert_callback(alert):
    print(f"ALERT: {alert.title} - {alert.description}")

dashboard.add_alert_callback(alert_callback)

# Start dashboard
await dashboard.start()
await dashboard.start_live_dashboard()
```

## Configuration

### Performance Thresholds

Default thresholds can be customized:

```python
from xencode.standalone_performance_dashboard import MetricType, MetricThreshold

# Custom CPU thresholds
cpu_threshold = MetricThreshold(
    MetricType.CPU,
    warning_threshold=60.0,    # Warning at 60%
    critical_threshold=80.0,   # Critical at 80%
    emergency_threshold=95.0   # Emergency at 95%
)

monitor.thresholds[MetricType.CPU] = cpu_threshold
```

### Metric Buffer Configuration

```python
from xencode.standalone_performance_dashboard import MetricBuffer

# Custom buffer size (default: 1000)
buffer = MetricBuffer(max_size=5000)
```

## API Reference

### MetricBuffer

```python
class MetricBuffer:
    def __init__(self, max_size: int = 1000)
    def add(self, value: float, timestamp: Optional[float] = None)
    def get_recent(self, seconds: int = 300) -> List[Tuple[float, float]]
    def get_average(self, seconds: int = 300) -> float
    def get_trend(self, seconds: int = 300) -> str
```

### PerformanceMonitor

```python
class PerformanceMonitor:
    def collect_system_metrics(self) -> Dict[str, float]
    def record_performance_metric(self, metric_type: MetricType, value: float)
    def get_metric_summary(self, metric_type: MetricType, seconds: int = 300) -> Dict[str, Any]
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[PerformanceAlert]
    def resolve_alert(self, alert_id: str) -> bool
```

### StandalonePerformanceDashboard

```python
class StandalonePerformanceDashboard:
    async def start(self)
    async def stop(self)
    async def start_live_dashboard(self, refresh_interval: float = 2.0)
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None])
    def generate_sample_data(self)
```

## Metrics Collected

### System Metrics
- **CPU Usage**: Overall system CPU utilization percentage
- **Memory Usage**: RAM utilization percentage and absolute values
- **Disk Usage**: Disk space utilization percentage and absolute values
- **Network I/O**: Bytes sent and received
- **Load Average**: 1m, 5m, 15m load averages (Unix-like systems)

### Process Metrics
- **Process CPU**: CPU usage by the current process
- **Process Memory**: Memory usage by the current process

### Performance Metrics
- **Response Time**: Application response times
- **Error Rate**: Error occurrence rate
- **Throughput**: Request processing throughput

## Alert System

### Alert Severities
- **INFO**: Informational alerts
- **WARNING**: Performance degradation warnings
- **CRITICAL**: Critical performance issues
- **EMERGENCY**: System emergency conditions

### Alert Lifecycle
1. **Generation**: Automatic creation when thresholds are exceeded
2. **Notification**: Callback execution for custom handling
3. **Resolution**: Manual or automatic alert resolution
4. **Tracking**: Historical alert tracking and analysis

## Integration

### Prometheus Integration

The dashboard integrates with Prometheus for enterprise monitoring:

```python
# Metrics exposed on /metrics endpoint
xencode_cpu_usage_percent
xencode_memory_usage_bytes
xencode_disk_usage_percent
xencode_requests_total
xencode_errors_total
```

### Analytics Integration

Integration with Xencode analytics infrastructure:

```python
# Automatic event tracking
analytics_infrastructure.track_system_event(
    event="performance_alert",
    component="performance_monitor",
    alert_type=alert.metric_type.value,
    severity=alert.severity.value
)
```

## Performance Considerations

### Optimization Features
- **Circular Buffers**: Efficient memory usage with fixed-size buffers
- **Async Processing**: Non-blocking background monitoring
- **Duplicate Prevention**: Intelligent alert deduplication
- **Configurable Intervals**: Adjustable collection and refresh rates

### Resource Usage
- **Memory**: ~10-50MB depending on buffer sizes and history retention
- **CPU**: <5% overhead for monitoring and dashboard rendering
- **Network**: Minimal (only for Prometheus metrics export)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install rich psutil prometheus-client
   ```

2. **Permission Errors**: Some system metrics may require elevated privileges
   ```bash
   sudo python dashboard_script.py
   ```

3. **Port Conflicts**: Change Prometheus port if 8000 is in use
   ```python
   prometheus_collector = PrometheusMetricsCollector(metrics_port=8001)
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Basic Monitoring Script

```python
#!/usr/bin/env python3
import asyncio
from xencode.standalone_performance_dashboard import StandalonePerformanceDashboard

async def main():
    dashboard = StandalonePerformanceDashboard()
    
    # Add custom alert handler
    def handle_alert(alert):
        if alert.severity.value == "critical":
            print(f"🚨 CRITICAL: {alert.description}")
    
    dashboard.add_alert_callback(handle_alert)
    
    try:
        await dashboard.start_live_dashboard()
    except KeyboardInterrupt:
        print("Dashboard stopped")
    finally:
        await dashboard.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Metrics Integration

```python
from xencode.standalone_performance_dashboard import PerformanceMonitor, MetricType

monitor = PerformanceMonitor()

# Record custom application metrics
monitor.record_performance_metric(MetricType.RESPONSE_TIME, 1.5)  # 1.5 seconds
monitor.record_performance_metric(MetricType.ERROR_RATE, 2.5)     # 2.5% error rate
monitor.record_performance_metric(MetricType.THROUGHPUT, 150.0)   # 150 requests/sec

# Check for alerts
alerts = monitor.get_active_alerts()
for alert in alerts:
    print(f"Alert: {alert.title} - {alert.description}")
```

## Testing

### Running Tests

```bash
# Basic functionality test
python test_performance_dashboard.py

# Live dashboard demo
python demo_performance_dashboard.py

# Integrated dashboard demo
python demo_integrated_performance_dashboard.py
```

### Test Coverage

The test suite covers:
- ✅ MetricBuffer functionality
- ✅ PerformanceMonitor with thresholds and alerts
- ✅ System metrics collection
- ✅ Alert generation and resolution
- ✅ Dashboard rendering components
- ✅ Sample data generation
- ✅ Performance and stress testing

## Future Enhancements

### Planned Features
- **Machine Learning**: Anomaly detection using ML models
- **Distributed Monitoring**: Multi-node system monitoring
- **Custom Dashboards**: User-configurable dashboard layouts
- **Export Capabilities**: Data export to various formats
- **Mobile Support**: Responsive design for mobile devices

### Integration Roadmap
- **Grafana Integration**: Native Grafana dashboard support
- **Slack/Teams Alerts**: Direct messaging integration
- **Database Storage**: Long-term metrics storage
- **API Endpoints**: RESTful API for external integrations

## Contributing

### Development Setup

```bash
git clone <repository>
cd xencode
pip install -r requirements.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

### Testing Guidelines

- Write tests for all new functionality
- Ensure >90% code coverage
- Test both success and error scenarios
- Include performance benchmarks

## License

This performance monitoring dashboard is part of the Xencode project and is licensed under the MIT License.