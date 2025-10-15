#!/usr/bin/env python3
"""
Tests for Performance Monitoring and Optimization System

Comprehensive test suite for performance monitoring, alerting,
and automated optimization functionality.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from xencode.monitoring.performance_optimizer import (
    PerformanceMonitoringSystem,
    PerformanceMetricsCollector,
    AlertManager,
    PerformanceOptimizer,
    PerformanceAlert,
    PerformanceThreshold,
    OptimizationAction,
    AlertSeverity,
    MetricType,
    get_performance_monitoring_system,
    initialize_performance_monitoring
)


class TestPerformanceMetricsCollector:
    """Test PerformanceMetricsCollector functionality"""
    
    def test_initialization(self):
        """Test metrics collector initialization"""
        collector = PerformanceMetricsCollector()
        
        assert collector.collection_interval == 5
        assert not collector.running
        assert len(collector.metrics_history) == 0
    
    @pytest.mark.asyncio
    @patch('xencode.monitoring.performance_optimizer.PSUTIL_AVAILABLE', True)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock psutil responses
        mock_cpu.return_value = 45.5
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 67.8
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = Mock()
        mock_disk_obj.used = 500 * 1024**3  # 500GB
        mock_disk_obj.total = 1000 * 1024**3  # 1TB
        mock_disk.return_value = mock_disk_obj
        
        collector = PerformanceMetricsCollector()
        metrics = await collector.collect_system_metrics()
        
        assert metrics[MetricType.CPU_USAGE] == 45.5
        assert metrics[MetricType.MEMORY_USAGE] == 67.8
        assert metrics[MetricType.DISK_USAGE] == 50.0  # 500/1000 * 100
    
    @pytest.mark.asyncio
    @patch('xencode.monitoring.performance_optimizer.PSUTIL_AVAILABLE', False)
    async def test_collect_system_metrics_no_psutil(self):
        """Test system metrics collection when psutil is not available"""
        collector = PerformanceMetricsCollector()
        metrics = await collector.collect_system_metrics()
        
        assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_collect_application_metrics(self):
        """Test application metrics collection"""
        collector = PerformanceMetricsCollector()
        
        with patch('xencode.monitoring.performance_optimizer.get_multimodal_cache') as mock_get_cache:
            mock_cache = AsyncMock()
            mock_cache.get_cache_statistics.return_value = {
                'base_cache': {'hit_rate': 85.5}
            }
            mock_get_cache.return_value = mock_cache
            
            metrics = await collector.collect_application_metrics()
            
            assert metrics[MetricType.CACHE_HIT_RATE] == 85.5
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics"""
        collector = PerformanceMetricsCollector()
        
        # Add some test data
        current_time = time.time()
        collector.metrics_history[MetricType.CPU_USAGE].extend([
            (current_time - 300, 50.0),  # 5 minutes ago
            (current_time - 180, 60.0),  # 3 minutes ago
            (current_time - 60, 70.0),   # 1 minute ago
            (current_time, 80.0)         # now
        ])
        
        # Get recent 2 minutes
        recent = collector.get_recent_metrics(MetricType.CPU_USAGE, 2)
        
        assert len(recent) == 2
        assert recent[0][1] == 70.0
        assert recent[1][1] == 80.0
    
    def test_calculate_trend(self):
        """Test trend calculation"""
        collector = PerformanceMetricsCollector()
        
        # Add increasing trend data
        current_time = time.time()
        increasing_data = [(current_time - i*60, 50 + i*5) for i in range(10, 0, -1)]
        collector.metrics_history[MetricType.CPU_USAGE].extend(increasing_data)
        
        trend = collector.calculate_trend(MetricType.CPU_USAGE, 10)
        assert trend == "increasing"
        
        # Add decreasing trend data
        collector.metrics_history[MetricType.MEMORY_USAGE].clear()
        decreasing_data = [(current_time - i*60, 90 - i*5) for i in range(10, 0, -1)]
        collector.metrics_history[MetricType.MEMORY_USAGE].extend(decreasing_data)
        
        trend = collector.calculate_trend(MetricType.MEMORY_USAGE, 10)
        assert trend == "decreasing"


class TestAlertManager:
    """Test AlertManager functionality"""
    
    def test_initialization(self):
        """Test alert manager initialization"""
        manager = AlertManager()
        
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_history) == 0
        assert len(manager.alert_callbacks) == 0
        assert MetricType.CPU_USAGE in manager.thresholds
        assert MetricType.MEMORY_USAGE in manager.thresholds
    
    def test_add_alert_callback(self):
        """Test adding alert callback"""
        manager = AlertManager()
        callback = Mock()
        
        manager.add_alert_callback(callback)
        
        assert len(manager.alert_callbacks) == 1
        assert callback in manager.alert_callbacks
    
    def test_update_threshold(self):
        """Test updating performance threshold"""
        manager = AlertManager()
        
        new_threshold = PerformanceThreshold(
            MetricType.CPU_USAGE, 80.0, 90.0, 95.0, 120
        )
        
        manager.update_threshold(MetricType.CPU_USAGE, new_threshold)
        
        assert manager.thresholds[MetricType.CPU_USAGE].warning_threshold == 80.0
        assert manager.thresholds[MetricType.CPU_USAGE].duration_seconds == 120
    
    def test_determine_severity(self):
        """Test severity determination logic"""
        manager = AlertManager()
        threshold = PerformanceThreshold(
            MetricType.CPU_USAGE, 70.0, 85.0, 95.0, 60
        )
        
        # Test normal metrics (higher is worse)
        assert manager._determine_severity(60.0, threshold, MetricType.CPU_USAGE) is None
        assert manager._determine_severity(75.0, threshold, MetricType.CPU_USAGE) == AlertSeverity.WARNING
        assert manager._determine_severity(90.0, threshold, MetricType.CPU_USAGE) == AlertSeverity.CRITICAL
        assert manager._determine_severity(96.0, threshold, MetricType.CPU_USAGE) == AlertSeverity.EMERGENCY
        
        # Test cache hit rate (lower is worse)
        cache_threshold = PerformanceThreshold(
            MetricType.CACHE_HIT_RATE, 80.0, 60.0, 40.0, 120
        )
        
        assert manager._determine_severity(85.0, cache_threshold, MetricType.CACHE_HIT_RATE) is None
        assert manager._determine_severity(75.0, cache_threshold, MetricType.CACHE_HIT_RATE) == AlertSeverity.WARNING
        assert manager._determine_severity(55.0, cache_threshold, MetricType.CACHE_HIT_RATE) == AlertSeverity.CRITICAL
        assert manager._determine_severity(35.0, cache_threshold, MetricType.CACHE_HIT_RATE) == AlertSeverity.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_check_thresholds_create_alert(self):
        """Test threshold checking and alert creation"""
        manager = AlertManager()
        callback = Mock()
        manager.add_alert_callback(callback)
        
        # Metrics that exceed thresholds
        metrics = {
            MetricType.CPU_USAGE: 90.0,  # Exceeds critical threshold (85.0)
            MetricType.MEMORY_USAGE: 60.0  # Below warning threshold (75.0)
        }
        
        await manager.check_thresholds(metrics)
        
        # Should create one alert for CPU
        assert len(manager.active_alerts) == 1
        assert len(manager.alert_history) == 1
        
        # Check alert details
        alert = list(manager.active_alerts.values())[0]
        assert alert.metric_type == MetricType.CPU_USAGE
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.current_value == 90.0
        
        # Callback should be called
        callback.assert_called_once()
    
    def test_generate_alert_title(self):
        """Test alert title generation"""
        manager = AlertManager()
        
        title = manager._generate_alert_title(MetricType.CPU_USAGE, AlertSeverity.WARNING)
        assert title == "Warning CPU Usage Alert"
        
        title = manager._generate_alert_title(MetricType.MEMORY_USAGE, AlertSeverity.CRITICAL)
        assert title == "Critical Memory Usage Alert"
    
    def test_generate_alert_description(self):
        """Test alert description generation"""
        manager = AlertManager()
        threshold = PerformanceThreshold(MetricType.CPU_USAGE, 70.0, 85.0, 95.0, 60)
        
        description = manager._generate_alert_description(MetricType.CPU_USAGE, 90.0, threshold)
        assert "90.0%" in description
        assert "70.0%" in description
    
    def test_get_alert_summary(self):
        """Test alert summary generation"""
        manager = AlertManager()
        
        # Add some test alerts
        alert1 = PerformanceAlert(
            alert_id="test1",
            metric_type=MetricType.CPU_USAGE,
            severity=AlertSeverity.WARNING,
            title="Test Alert 1",
            description="Test",
            current_value=75.0,
            threshold_value=70.0,
            timestamp=datetime.now()
        )
        
        alert2 = PerformanceAlert(
            alert_id="test2",
            metric_type=MetricType.MEMORY_USAGE,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert 2",
            description="Test",
            current_value=95.0,
            threshold_value=85.0,
            timestamp=datetime.now()
        )
        
        manager.active_alerts["test1"] = alert1
        manager.active_alerts["test2"] = alert2
        
        summary = manager.get_alert_summary()
        
        assert summary["warning"] == 1
        assert summary["critical"] == 1
        assert summary["info"] == 0
        assert summary["emergency"] == 0


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality"""
    
    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = PerformanceOptimizer()
        
        assert len(optimizer.optimization_rules) > 0
        assert len(optimizer.executed_actions) == 0
        assert optimizer.auto_optimize_enabled
    
    @pytest.mark.asyncio
    async def test_analyze_and_optimize_high_memory(self):
        """Test optimization for high memory usage"""
        optimizer = PerformanceOptimizer()
        
        # Mock the cache cleanup method
        optimizer._cleanup_memory_cache = AsyncMock(return_value="Cache cleaned")
        
        # High memory usage metrics
        metrics = {
            MetricType.MEMORY_USAGE: 90.0,  # Exceeds 85% threshold
            MetricType.CPU_USAGE: 50.0
        }
        
        actions = await optimizer.analyze_and_optimize(metrics)
        
        assert len(actions) == 1
        assert actions[0].action_type == "high_memory_cache_cleanup"
        assert actions[0].executed
        assert "Cache cleaned" in actions[0].result
    
    @pytest.mark.asyncio
    async def test_analyze_and_optimize_low_cache_hit_rate(self):
        """Test optimization for low cache hit rate"""
        optimizer = PerformanceOptimizer()
        
        # Mock the cache warming method
        optimizer._trigger_cache_warming = AsyncMock(return_value="Warming triggered")
        
        # Low cache hit rate metrics
        metrics = {
            MetricType.CACHE_HIT_RATE: 50.0,  # Below 60% threshold
            MetricType.CPU_USAGE: 30.0
        }
        
        actions = await optimizer.analyze_and_optimize(metrics)
        
        assert len(actions) == 1
        assert actions[0].action_type == "low_cache_hit_rate_warming"
        assert actions[0].executed
    
    @pytest.mark.asyncio
    async def test_analyze_and_optimize_disabled(self):
        """Test that optimization is skipped when disabled"""
        optimizer = PerformanceOptimizer()
        optimizer.auto_optimize_enabled = False
        
        # High resource usage metrics
        metrics = {
            MetricType.MEMORY_USAGE: 95.0,
            MetricType.CPU_USAGE: 95.0
        }
        
        actions = await optimizer.analyze_and_optimize(metrics)
        
        assert len(actions) == 0
    
    def test_add_optimization_rule(self):
        """Test adding custom optimization rule"""
        optimizer = PerformanceOptimizer()
        initial_count = len(optimizer.optimization_rules)
        
        def custom_condition(metrics):
            return metrics.get(MetricType.ERROR_RATE, 0) > 5.0
        
        async def custom_action(metrics):
            return "Custom action executed"
        
        optimizer.add_optimization_rule(
            "custom_error_handling",
            custom_condition,
            custom_action,
            "Handle high error rate",
            "medium"
        )
        
        assert len(optimizer.optimization_rules) == initial_count + 1
        
        # Find the new rule
        custom_rule = next(rule for rule in optimizer.optimization_rules 
                          if rule["name"] == "custom_error_handling")
        
        assert custom_rule["description"] == "Handle high error rate"
        assert custom_rule["risk_level"] == "medium"
    
    def test_get_optimization_history(self):
        """Test getting optimization history"""
        optimizer = PerformanceOptimizer()
        
        # Add some test actions
        for i in range(10):
            action = OptimizationAction(
                action_id=f"test_{i}",
                action_type="test_action",
                description=f"Test action {i}",
                target_component="test",
                parameters={},
                estimated_impact="Test impact"
            )
            optimizer.executed_actions.append(action)
        
        # Get recent 5 actions
        recent = optimizer.get_optimization_history(5)
        
        assert len(recent) == 5
        assert recent[-1].action_id == "test_9"  # Most recent


class TestPerformanceMonitoringSystem:
    """Test PerformanceMonitoringSystem integration"""
    
    def test_initialization(self):
        """Test monitoring system initialization"""
        system = PerformanceMonitoringSystem()
        
        assert isinstance(system.metrics_collector, PerformanceMetricsCollector)
        assert isinstance(system.alert_manager, AlertManager)
        assert isinstance(system.optimizer, PerformanceOptimizer)
        assert not system.running
        assert system.monitoring_interval == 30
    
    def test_add_alert_callback(self):
        """Test adding alert callback to system"""
        system = PerformanceMonitoringSystem()
        callback = Mock()
        
        system.add_alert_callback(callback)
        
        assert callback in system.alert_manager.alert_callbacks
    
    def test_get_system_status(self):
        """Test getting system status"""
        system = PerformanceMonitoringSystem()
        
        # Add some test data
        current_time = time.time()
        system.metrics_collector.metrics_history[MetricType.CPU_USAGE].append((current_time, 65.0))
        system.metrics_collector.metrics_history[MetricType.MEMORY_USAGE].append((current_time, 70.0))
        
        status = system.get_system_status()
        
        assert "monitoring_active" in status
        assert "current_metrics" in status
        assert "active_alerts" in status
        assert "alert_summary" in status
        assert "recent_optimizations" in status
        assert "auto_optimization" in status
        
        assert status["current_metrics"]["cpu_usage"] == 65.0
        assert status["current_metrics"]["memory_usage"] == 70.0
    
    def test_get_performance_report(self):
        """Test generating performance report"""
        system = PerformanceMonitoringSystem()
        
        report = system.get_performance_report()
        
        assert "timestamp" in report
        assert "system_health" in report
        assert "trends" in report
        assert "alerts" in report
        assert "optimizations" in report
        
        # Check trends structure
        trends = report["trends"]
        assert "cpu_usage" in trends
        assert "memory_usage" in trends
        assert "cache_hit_rate" in trends
    
    def test_calculate_system_health(self):
        """Test system health calculation"""
        system = PerformanceMonitoringSystem()
        
        # Test healthy system (no alerts)
        health = system._calculate_system_health()
        assert health == "healthy"
        
        # Add warning alert
        warning_alert = PerformanceAlert(
            alert_id="warning_test",
            metric_type=MetricType.CPU_USAGE,
            severity=AlertSeverity.WARNING,
            title="Warning Alert",
            description="Test warning",
            current_value=75.0,
            threshold_value=70.0,
            timestamp=datetime.now()
        )
        system.alert_manager.active_alerts["warning_test"] = warning_alert
        
        health = system._calculate_system_health()
        assert health == "warning"
        
        # Add critical alert
        critical_alert = PerformanceAlert(
            alert_id="critical_test",
            metric_type=MetricType.MEMORY_USAGE,
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            description="Test critical",
            current_value=95.0,
            threshold_value=85.0,
            timestamp=datetime.now()
        )
        system.alert_manager.active_alerts["critical_test"] = critical_alert
        
        health = system._calculate_system_health()
        assert health == "degraded"


class TestGlobalFunctions:
    """Test global functions"""
    
    def test_get_performance_monitoring_system(self):
        """Test getting global monitoring system instance"""
        # Reset global instance
        import xencode.monitoring.performance_optimizer
        xencode.monitoring.performance_optimizer._performance_monitoring_system = None
        
        system1 = get_performance_monitoring_system()
        system2 = get_performance_monitoring_system()
        
        # Should return same instance
        assert system1 is system2
        assert isinstance(system1, PerformanceMonitoringSystem)
    
    @pytest.mark.asyncio
    async def test_initialize_performance_monitoring(self):
        """Test performance monitoring initialization"""
        with patch('asyncio.create_task') as mock_create_task:
            system = await initialize_performance_monitoring()
            
            assert isinstance(system, PerformanceMonitoringSystem)
            mock_create_task.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])