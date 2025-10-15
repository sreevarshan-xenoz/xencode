#!/usr/bin/env python3
"""
Tests for Performance Monitoring Dashboard

Comprehensive test suite for the performance monitoring dashboard,
including metric collection, alerting, and dashboard rendering.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the components to test
try:
    from xencode.performance_monitoring_dashboard import (
        PerformanceMonitor, MetricBuffer, PerformanceAlert, MetricThreshold,
        AlertSeverity, MetricType, DashboardRenderer, PerformanceMonitoringDashboard
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    pytest.skip("Performance monitoring dashboard not available", allow_module_level=True)


class TestMetricBuffer:
    """Test the MetricBuffer class"""
    
    def test_metric_buffer_initialization(self):
        """Test metric buffer initialization"""
        buffer = MetricBuffer(max_size=100)
        assert buffer.max_size == 100
        assert len(buffer.data) == 0
        assert len(buffer.timestamps) == 0
    
    def test_add_metric_value(self):
        """Test adding metric values"""
        buffer = MetricBuffer(max_size=10)
        
        # Add some values
        buffer.add(50.0)
        buffer.add(60.0)
        buffer.add(70.0)
        
        assert len(buffer.data) == 3
        assert len(buffer.timestamps) == 3
        assert buffer.data[-1] == 70.0  # Latest value
    
    def test_buffer_max_size_limit(self):
        """Test that buffer respects max size limit"""
        buffer = MetricBuffer(max_size=3)
        
        # Add more values than max size
        for i in range(5):
            buffer.add(float(i))
        
        assert len(buffer.data) == 3
        assert len(buffer.timestamps) == 3
        assert list(buffer.data) == [2.0, 3.0, 4.0]  # Only latest 3 values
    
    def test_get_recent_values(self):
        """Test getting recent values within time window"""
        buffer = MetricBuffer()
        current_time = time.time()
        
        # Add values with specific timestamps
        buffer.add(10.0, current_time - 600)  # 10 minutes ago
        buffer.add(20.0, current_time - 300)  # 5 minutes ago
        buffer.add(30.0, current_time - 60)   # 1 minute ago
        buffer.add(40.0, current_time)        # Now
        
        # Get recent values (last 2 minutes)
        recent = buffer.get_recent(120)
        assert len(recent) == 2  # Only last 2 values
        assert recent[0][1] == 30.0
        assert recent[1][1] == 40.0
    
    def test_get_average(self):
        """Test calculating average for time window"""
        buffer = MetricBuffer()
        current_time = time.time()
        
        # Add values
        buffer.add(10.0, current_time - 60)
        buffer.add(20.0, current_time - 30)
        buffer.add(30.0, current_time)
        
        # Get average for last 2 minutes
        avg = buffer.get_average(120)
        assert avg == 20.0  # (10 + 20 + 30) / 3
    
    def test_get_trend_analysis(self):
        """Test trend analysis"""
        buffer = MetricBuffer()
        current_time = time.time()
        
        # Add increasing trend
        for i in range(10):
            buffer.add(float(i * 10), current_time - (90 - i * 10))
        
        trend = buffer.get_trend(100)
        assert trend == "increasing"
        
        # Add decreasing trend
        buffer = MetricBuffer()
        for i in range(10):
            buffer.add(float(90 - i * 10), current_time - (90 - i * 10))
        
        trend = buffer.get_trend(100)
        assert trend == "decreasing"


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        monitor = PerformanceMonitor()
        
        assert len(monitor.thresholds) > 0
        assert MetricType.CPU in monitor.thresholds
        assert MetricType.MEMORY in monitor.thresholds
        assert len(monitor.active_alerts) == 0
    
    def test_default_thresholds(self):
        """Test default threshold configuration"""
        monitor = PerformanceMonitor()
        
        cpu_threshold = monitor.thresholds[MetricType.CPU]
        assert cpu_threshold.warning_threshold == 70.0
        assert cpu_threshold.critical_threshold == 85.0
        assert cpu_threshold.emergency_threshold == 95.0
    
    def test_record_performance_metric(self):
        """Test recording performance metrics"""
        monitor = PerformanceMonitor()
        
        # Record a normal metric
        monitor.record_performance_metric(MetricType.CPU, 50.0)
        
        # Check that metric was recorded
        buffer = monitor.metric_buffers['cpu']
        assert len(buffer.data) == 1
        assert buffer.data[0] == 50.0
    
    def test_threshold_alert_generation(self):
        """Test alert generation when thresholds are exceeded"""
        monitor = PerformanceMonitor()
        
        # Record metric that exceeds warning threshold
        monitor.record_performance_metric(MetricType.CPU, 75.0)
        
        # Check that warning alert was generated
        alerts = monitor.get_active_alerts(AlertSeverity.WARNING)
        assert len(alerts) >= 1
        assert alerts[0].metric_type == MetricType.CPU
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_critical_alert_generation(self):
        """Test critical alert generation"""
        monitor = PerformanceMonitor()
        
        # Record metric that exceeds critical threshold
        monitor.record_performance_metric(MetricType.MEMORY, 92.0)
        
        # Check that critical alert was generated
        alerts = monitor.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(alerts) >= 1
        assert alerts[0].metric_type == MetricType.MEMORY
        assert alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_duplicate_alert_prevention(self):
        """Test that duplicate alerts are not created"""
        monitor = PerformanceMonitor()
        
        # Record multiple high CPU values
        monitor.record_performance_metric(MetricType.CPU, 90.0)
        monitor.record_performance_metric(MetricType.CPU, 91.0)
        monitor.record_performance_metric(MetricType.CPU, 92.0)
        
        # Should only have one critical alert
        critical_alerts = monitor.get_active_alerts(AlertSeverity.CRITICAL)
        cpu_alerts = [a for a in critical_alerts if a.metric_type == MetricType.CPU]
        assert len(cpu_alerts) == 1
    
    def test_alert_resolution(self):
        """Test alert resolution"""
        monitor = PerformanceMonitor()
        
        # Generate an alert
        monitor.record_performance_metric(MetricType.CPU, 90.0)
        alerts = monitor.get_active_alerts()
        assert len(alerts) >= 1
        
        # Resolve the alert
        alert_id = alerts[0].alert_id
        result = monitor.resolve_alert(alert_id)
        assert result is True
        
        # Check that alert is resolved
        resolved_alert = next((a for a in monitor.active_alerts if a.alert_id == alert_id), None)
        assert resolved_alert is not None
        assert resolved_alert.resolved is True
        assert resolved_alert.resolved_at is not None
    
    def test_metric_summary_calculation(self):
        """Test metric summary calculation"""
        monitor = PerformanceMonitor()
        
        # Add some metric values
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            monitor.record_performance_metric(MetricType.CPU, value)
        
        # Get summary
        summary = monitor.get_metric_summary(MetricType.CPU, 300)
        
        assert summary['current'] == 50.0  # Latest value
        assert summary['average'] == 30.0  # Average of all values
        assert summary['min'] == 10.0
        assert summary['max'] == 50.0
        assert summary['data_points'] == 5
    
    @patch('xencode.performance_monitoring_dashboard.psutil')
    def test_system_metrics_collection(self, mock_psutil):
        """Test system metrics collection"""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, used=8000000000, total=16000000000)
        mock_psutil.disk_usage.return_value = Mock(percent=70.0, used=500000000000, total=1000000000000)
        mock_psutil.net_io_counters.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)
        
        # Mock process
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 15.0
        mock_process.memory_info.return_value = Mock(rss=100000000)
        mock_psutil.Process.return_value = mock_process
        
        monitor = PerformanceMonitor()
        metrics = monitor.collect_system_metrics()
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'disk_usage' in metrics
        assert metrics['cpu_usage'] == 45.0
        assert metrics['memory_usage'] == 60.0
        assert metrics['disk_usage'] == 70.0


class TestDashboardRenderer:
    """Test the DashboardRenderer class"""
    
    def test_dashboard_renderer_initialization(self):
        """Test dashboard renderer initialization"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        assert renderer.monitor == monitor
        assert renderer.console is not None
    
    @patch('xencode.performance_monitoring_dashboard.psutil')
    def test_system_metrics_panel_creation(self, mock_psutil):
        """Test system metrics panel creation"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, used=8000000000, total=16000000000)
        mock_psutil.disk_usage.return_value = Mock(percent=70.0, used=500000000000, total=1000000000000)
        mock_psutil.net_io_counters.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)
        mock_psutil.Process.return_value = Mock(
            cpu_percent=Mock(return_value=15.0),
            memory_info=Mock(return_value=Mock(rss=100000000))
        )
        
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        panel = renderer.create_system_metrics_panel()
        assert panel is not None
        assert "System Metrics" in panel.title
    
    def test_alerts_panel_no_alerts(self):
        """Test alerts panel when no alerts are active"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        panel = renderer.create_alerts_panel()
        assert panel is not None
        assert "No active alerts" in str(panel.renderable)
    
    def test_alerts_panel_with_alerts(self):
        """Test alerts panel with active alerts"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        # Generate some alerts
        monitor.record_performance_metric(MetricType.CPU, 90.0)
        monitor.record_performance_metric(MetricType.MEMORY, 95.0)
        
        panel = renderer.create_alerts_panel()
        assert panel is not None
        assert "Active Alerts" in panel.title
    
    def test_performance_trends_panel(self):
        """Test performance trends panel creation"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        # Add some metric data
        for i in range(10):
            monitor.record_performance_metric(MetricType.CPU, float(50 + i))
            monitor.record_performance_metric(MetricType.MEMORY, float(60 + i))
        
        panel = renderer.create_performance_trends_panel()
        assert panel is not None
        assert "Performance Trends" in panel.title
    
    def test_complete_dashboard_rendering(self):
        """Test complete dashboard rendering"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        layout = renderer.render_dashboard()
        assert layout is not None
        
        # Check that all sections are present
        assert "header" in layout._children
        assert "main" in layout._children
        assert "footer" in layout._children


class TestPerformanceMonitoringDashboard:
    """Test the main PerformanceMonitoringDashboard class"""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        dashboard = PerformanceMonitoringDashboard()
        
        assert dashboard.performance_monitor is not None
        assert dashboard.renderer is not None
        assert dashboard.is_running is False
    
    @pytest.mark.asyncio
    async def test_dashboard_start_stop(self):
        """Test dashboard start and stop"""
        dashboard = PerformanceMonitoringDashboard()
        
        # Start dashboard
        await dashboard.start()
        assert dashboard.is_running is True
        
        # Stop dashboard
        await dashboard.stop()
        assert dashboard.is_running is False
    
    def test_alert_callback_registration(self):
        """Test alert callback registration"""
        dashboard = PerformanceMonitoringDashboard()
        
        callback_called = False
        def test_callback(alert):
            nonlocal callback_called
            callback_called = True
        
        dashboard.add_alert_callback(test_callback)
        
        # Generate an alert
        dashboard.performance_monitor.record_performance_metric(MetricType.CPU, 90.0)
        
        # Check that callback was called
        assert callback_called is True
    
    def test_dashboard_status(self):
        """Test dashboard status reporting"""
        dashboard = PerformanceMonitoringDashboard()
        
        status = dashboard.get_dashboard_status()
        
        assert 'running' in status
        assert 'active_alerts' in status
        assert 'monitoring_available' in status
        assert status['running'] is False  # Not started yet
        assert status['active_alerts'] == 0  # No alerts initially


class TestIntegration:
    """Integration tests for the performance monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from metrics to alerts to dashboard"""
        dashboard = PerformanceMonitoringDashboard()
        
        # Start dashboard
        await dashboard.start()
        
        try:
            # Generate some metrics that should trigger alerts
            monitor = dashboard.performance_monitor
            
            # Normal metrics
            monitor.record_performance_metric(MetricType.CPU, 50.0)
            monitor.record_performance_metric(MetricType.MEMORY, 60.0)
            
            # High metrics that should trigger alerts
            monitor.record_performance_metric(MetricType.CPU, 90.0)
            monitor.record_performance_metric(MetricType.MEMORY, 95.0)
            
            # Check that alerts were generated
            alerts = monitor.get_active_alerts()
            assert len(alerts) >= 2
            
            # Check that dashboard can render with alerts
            layout = dashboard.renderer.render_dashboard()
            assert layout is not None
            
            # Test alert resolution
            for alert in alerts:
                result = monitor.resolve_alert(alert.alert_id)
                assert result is True
            
            # Check that alerts are resolved
            active_alerts = monitor.get_active_alerts()
            assert len(active_alerts) == 0
            
        finally:
            await dashboard.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_metric_collection(self):
        """Test concurrent metric collection and processing"""
        dashboard = PerformanceMonitoringDashboard()
        
        await dashboard.start()
        
        try:
            # Simulate concurrent metric collection
            tasks = []
            for i in range(100):
                task = asyncio.create_task(
                    self._record_metric_async(dashboard.performance_monitor, i)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            # Check that all metrics were recorded
            cpu_buffer = dashboard.performance_monitor.metric_buffers['cpu']
            assert len(cpu_buffer.data) == 100
            
        finally:
            await dashboard.stop()
    
    async def _record_metric_async(self, monitor, value):
        """Helper method to record metrics asynchronously"""
        await asyncio.sleep(0.01)  # Small delay to simulate real collection
        monitor.record_performance_metric(MetricType.CPU, float(value % 100))


# Performance and stress tests
class TestPerformance:
    """Performance tests for the monitoring system"""
    
    def test_metric_buffer_performance(self):
        """Test metric buffer performance with large datasets"""
        buffer = MetricBuffer(max_size=10000)
        
        start_time = time.time()
        
        # Add 10,000 metrics
        for i in range(10000):
            buffer.add(float(i))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second)
        assert duration < 1.0
        assert len(buffer.data) == 10000
    
    def test_alert_generation_performance(self):
        """Test alert generation performance"""
        monitor = PerformanceMonitor()
        
        start_time = time.time()
        
        # Generate many metrics that trigger alerts
        for i in range(1000):
            monitor.record_performance_metric(MetricType.CPU, 90.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 2.0
        
        # Should only have one alert due to duplicate prevention
        alerts = monitor.get_active_alerts()
        cpu_alerts = [a for a in alerts if a.metric_type == MetricType.CPU]
        assert len(cpu_alerts) == 1
    
    def test_dashboard_rendering_performance(self):
        """Test dashboard rendering performance"""
        monitor = PerformanceMonitor()
        renderer = DashboardRenderer(monitor)
        
        # Add lots of metric data
        for i in range(1000):
            monitor.record_performance_metric(MetricType.CPU, float(i % 100))
            monitor.record_performance_metric(MetricType.MEMORY, float((i + 50) % 100))
        
        start_time = time.time()
        
        # Render dashboard multiple times
        for _ in range(10):
            layout = renderer.render_dashboard()
            assert layout is not None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should render quickly (less than 1 second for 10 renders)
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])