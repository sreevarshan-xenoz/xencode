#!/usr/bin/env python3
"""
Test Suite for Advanced Analytics Dashboard

Comprehensive tests for metrics collection, analytics engine, cost optimization,
and dashboard rendering components.
"""

import pytest
import asyncio
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch
import time

# Add the system to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "xencode"))

from advanced_analytics_dashboard import (
    MetricsCollector, AnalyticsEngine, CostOptimizer, DashboardRenderer,
    AnalyticsDashboard, MetricPoint, PerformanceMetrics, CostMetrics
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(temp_file.name)
    temp_file.close()
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def metrics_collector(temp_db):
    """Create metrics collector with temporary database"""
    return MetricsCollector(temp_db)


@pytest.fixture
def analytics_engine(metrics_collector):
    """Create analytics engine with metrics collector"""
    return AnalyticsEngine(metrics_collector)


@pytest.fixture
def cost_optimizer(metrics_collector):
    """Create cost optimizer with metrics collector"""
    return CostOptimizer(metrics_collector)


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_database_initialization(self, temp_db):
        """Test database tables are created properly"""
        collector = MetricsCollector(temp_db)
        
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            
            # Check metrics table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'")
            assert cursor.fetchone() is not None
            
            # Check usage_events table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_events'")
            assert cursor.fetchone() is not None
    
    def test_record_metric(self, metrics_collector):
        """Test metric recording"""
        metadata = {"source": "test", "category": "performance"}
        metrics_collector.record_metric("test_metric", 42.5, metadata)
        
        # Check buffer
        assert len(metrics_collector.metrics_buffer["test_metric"]) == 1
        point = metrics_collector.metrics_buffer["test_metric"][0]
        assert point.value == 42.5
        assert point.metadata == metadata
        
        # Check database
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metric_name, value, metadata FROM metrics WHERE metric_name = ?", ("test_metric",))
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "test_metric"
            assert row[1] == 42.5
            assert json.loads(row[2]) == metadata
    
    def test_record_usage_event(self, metrics_collector):
        """Test usage event recording"""
        metrics_collector.record_usage_event(
            "chat_completion", "user123", "gpt-4", 150, 0.03, True
        )
        
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM usage_events")
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "chat_completion"  # event_type
            assert row[2] == "user123"  # user_id
            assert row[3] == "gpt-4"  # model
            assert row[4] == 150  # tokens
            assert row[5] == 0.03  # cost
            assert row[6] == True  # success
    
    def test_get_recent_metrics(self, metrics_collector):
        """Test retrieving recent metrics"""
        # Record metrics with different timestamps
        current_time = time.time()
        
        # Old metric (should not be included)
        old_point = MetricPoint(current_time - 3700, 10.0)  # 1+ hour ago
        metrics_collector.metrics_buffer["test_metric"].append(old_point)
        
        # Recent metric (should be included)
        recent_point = MetricPoint(current_time - 1800, 20.0)  # 30 minutes ago
        metrics_collector.metrics_buffer["test_metric"].append(recent_point)
        
        recent_metrics = metrics_collector.get_recent_metrics("test_metric", 60)  # Last hour
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 20.0
    
    def test_performance_metrics_calculation(self, metrics_collector):
        """Test performance metrics calculation"""
        # Add sample data
        metrics_collector.record_metric("response_time", 1.5)
        metrics_collector.record_metric("response_time", 2.0)
        metrics_collector.record_metric("tokens_per_second", 15.0)
        metrics_collector.record_metric("memory_usage", 65.0)
        metrics_collector.record_metric("cpu_usage", 45.0)
        metrics_collector.record_metric("request", 1)
        metrics_collector.record_metric("cache_hit", 1)
        
        perf_metrics = metrics_collector.get_performance_metrics()
        
        assert isinstance(perf_metrics, PerformanceMetrics)
        assert perf_metrics.response_time == 1.75  # Average of 1.5 and 2.0
        assert perf_metrics.tokens_per_second == 15.0
        assert perf_metrics.memory_usage == 65.0
        assert perf_metrics.cpu_usage == 45.0
        assert perf_metrics.request_count == 1
    
    def test_cache_hit_rate_calculation(self, metrics_collector):
        """Test cache hit rate calculation"""
        # Add cache metrics
        for _ in range(8):  # 8 hits
            metrics_collector.record_metric("cache_hit", 1)
        for _ in range(2):  # 2 misses
            metrics_collector.record_metric("cache_miss", 1)
        
        hit_rate = metrics_collector._calculate_cache_hit_rate()
        assert hit_rate == 80.0  # 8/(8+2) * 100


class TestAnalyticsEngine:
    """Test analytics engine functionality"""
    
    @pytest.mark.asyncio
    async def test_usage_pattern_analysis(self, analytics_engine):
        """Test usage pattern analysis"""
        # Add sample usage events
        collector = analytics_engine.collector
        collector.record_usage_event("chat", "user1", "gpt-4", 100, 0.02, True)
        collector.record_usage_event("chat", "user2", "gpt-3.5-turbo", 150, 0.003, True)
        collector.record_usage_event("chat", "user1", "gpt-4", 200, 0.04, False)
        
        patterns = analytics_engine.analyze_usage_patterns()
        
        assert "model_statistics" in patterns
        assert "peak_usage_hour" in patterns
        assert "usage_trends" in patterns
        assert "user_behavior" in patterns
        assert "performance_insights" in patterns
    
    def test_performance_insights_generation(self, analytics_engine):
        """Test performance insights generation"""
        # Mock performance metrics to trigger insights
        with patch.object(analytics_engine.collector, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = PerformanceMetrics(
                response_time=3.0,  # High response time
                cache_hit_rate=50.0,  # Low cache hit rate
                error_count=10,
                request_count=100,
                memory_usage=85.0  # High memory usage
            )
            
            insights = analytics_engine._generate_performance_insights()
            
            assert len(insights) >= 3  # Should have multiple insights
            assert any("High response times" in insight for insight in insights)
            assert any("Low cache hit rate" in insight for insight in insights)
            assert any("High memory usage" in insight for insight in insights)
    
    def test_growth_rate_calculation(self, analytics_engine):
        """Test growth rate calculation"""
        # Sample daily usage data: [(day, count), ...]
        daily_usage = [
            ("2024-01-01", 100),
            ("2024-01-02", 110),
            ("2024-01-03", 120),
            ("2024-01-04", 150),
            ("2024-01-05", 160),
            ("2024-01-06", 180)
        ]
        
        growth_rate = analytics_engine._calculate_growth_rate(daily_usage)
        
        # First half average: (100+110+120)/3 = 110
        # Second half average: (150+160+180)/3 = 163.33
        # Growth rate: (163.33-110)/110 * 100 ≈ 48.5%
        assert 40 < growth_rate < 60


class TestCostOptimizer:
    """Test cost optimization functionality"""
    
    def test_cost_metrics_calculation(self, cost_optimizer):
        """Test cost metrics calculation"""
        # Add sample usage events with costs
        collector = cost_optimizer.collector
        collector.record_usage_event("chat", "user1", "gpt-4", 100, 0.03, True)
        collector.record_usage_event("chat", "user2", "gpt-3.5-turbo", 150, 0.003, True)
        collector.record_usage_event("chat", "user3", "claude-3", 200, 0.015, True)
        
        cost_metrics = cost_optimizer.calculate_cost_metrics()
        
        assert isinstance(cost_metrics, CostMetrics)
        assert cost_metrics.total_cost > 0
        assert cost_metrics.cost_per_request > 0
        assert cost_metrics.cost_per_token > 0
        assert len(cost_metrics.cost_by_model) > 0
    
    def test_potential_savings_calculation(self, cost_optimizer):
        """Test potential savings calculation"""
        # Sample model costs data: (cost, requests, tokens, model)
        model_costs = [
            (0.30, 10, 1000, "gpt-4"),  # Expensive model
            (0.02, 20, 1000, "gpt-3.5-turbo"),  # Cheaper model
            (0.15, 5, 1000, "claude-3")  # Medium cost model
        ]
        
        savings = cost_optimizer._calculate_potential_savings(model_costs)
        
        # Should calculate savings for expensive models
        assert savings >= 0


class TestDashboardRenderer:
    """Test dashboard rendering functionality"""
    
    def test_performance_panel_creation(self, analytics_engine, cost_optimizer):
        """Test performance panel creation"""
        renderer = DashboardRenderer(analytics_engine, cost_optimizer)
        
        with patch.object(analytics_engine.collector, 'get_performance_metrics') as mock_metrics:
            mock_metrics.return_value = PerformanceMetrics(
                response_time=1.5,
                tokens_per_second=20.0,
                cache_hit_rate=85.0,
                memory_usage=60.0,
                request_count=100,
                error_count=2
            )
            
            panel = renderer.create_performance_panel()
            
            assert panel.title == "📊 Performance Metrics"
            assert "1.5" in str(panel)  # Response time
            assert "85.0%" in str(panel)  # Cache hit rate
    
    def test_usage_panel_creation(self, analytics_engine, cost_optimizer):
        """Test usage panel creation"""
        renderer = DashboardRenderer(analytics_engine, cost_optimizer)
        
        # Mock usage data
        mock_usage_data = {
            "model_statistics": [
                (50, 150.0, 0.25, "gpt-4", 10, 0.95),
                (30, 100.0, 0.06, "gpt-3.5-turbo", 8, 0.98)
            ]
        }
        
        with patch.object(analytics_engine, 'analyze_usage_patterns', return_value=mock_usage_data):
            panel = renderer.create_usage_panel()
            
            assert panel.title == "📈 Usage Statistics (24h)"
            assert "gpt-4" in str(panel)
            assert "95.0%" in str(panel)  # Success rate
    
    def test_cost_panel_creation(self, analytics_engine, cost_optimizer):
        """Test cost panel creation"""
        renderer = DashboardRenderer(analytics_engine, cost_optimizer)
        
        mock_cost_metrics = CostMetrics(
            total_cost=0.45,
            cost_per_request=0.009,
            cost_per_token=0.00003,
            cost_by_model={"gpt-4": 0.30, "gpt-3.5-turbo": 0.15},
            optimization_savings=0.05
        )
        
        with patch.object(cost_optimizer, 'calculate_cost_metrics', return_value=mock_cost_metrics):
            panel = renderer.create_cost_panel()
            
            assert panel.title == "💰 Cost Analysis"
            assert "$0.4500" in str(panel)  # Total cost
            assert "$0.0500" in str(panel)  # Potential savings
    
    def test_dashboard_layout_creation(self, analytics_engine, cost_optimizer):
        """Test complete dashboard layout"""
        renderer = DashboardRenderer(analytics_engine, cost_optimizer)
        
        # Mock all data sources
        with patch.object(analytics_engine.collector, 'get_performance_metrics') as mock_perf, \
             patch.object(analytics_engine, 'analyze_usage_patterns') as mock_usage, \
             patch.object(cost_optimizer, 'calculate_cost_metrics') as mock_cost, \
             patch.object(analytics_engine, '_generate_performance_insights') as mock_insights:
            
            mock_perf.return_value = PerformanceMetrics()
            mock_usage.return_value = {"model_statistics": []}
            mock_cost.return_value = CostMetrics()
            mock_insights.return_value = ["✅ System performing optimally"]
            
            layout = renderer.render_dashboard()
            
            assert layout is not None
            assert hasattr(layout, 'renderable')


class TestAnalyticsDashboard:
    """Test main dashboard orchestrator"""
    
    def test_dashboard_initialization(self, temp_db):
        """Test dashboard initialization"""
        dashboard = AnalyticsDashboard(temp_db)
        
        assert dashboard.metrics_collector is not None
        assert dashboard.analytics_engine is not None
        assert dashboard.cost_optimizer is not None
        assert dashboard.renderer is not None
        assert not dashboard.is_running
    
    def test_sample_data_generation(self, temp_db):
        """Test sample data generation"""
        dashboard = AnalyticsDashboard(temp_db)
        dashboard.generate_sample_data()
        
        # Check that data was generated
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            
            # Check metrics
            cursor.execute("SELECT COUNT(*) FROM metrics")
            metrics_count = cursor.fetchone()[0]
            assert metrics_count > 0
            
            # Check usage events
            cursor.execute("SELECT COUNT(*) FROM usage_events")
            events_count = cursor.fetchone()[0]
            assert events_count > 0
    
    def test_report_export(self, temp_db):
        """Test analytics report export"""
        dashboard = AnalyticsDashboard(temp_db)
        dashboard.generate_sample_data()
        
        # Test JSON export
        json_report = dashboard.export_report("json")
        assert json_report is not None
        
        # Validate JSON structure
        report_data = json.loads(json_report)
        assert "timestamp" in report_data
        assert "performance" in report_data
        assert "usage" in report_data
        assert "costs" in report_data
        assert "insights" in report_data
        
        # Test YAML export
        yaml_report = dashboard.export_report("yaml")
        assert yaml_report is not None
        assert "timestamp:" in yaml_report
    
    @pytest.mark.asyncio
    async def test_dashboard_lifecycle(self, temp_db):
        """Test dashboard start and stop"""
        dashboard = AnalyticsDashboard(temp_db)
        
        # Test stopping (should not crash when not running)
        dashboard.stop_dashboard()
        assert not dashboard.is_running
        
        # Test starting and stopping
        dashboard.is_running = True
        dashboard.stop_dashboard()
        assert not dashboard.is_running


# Integration Tests
class TestIntegration:
    """Integration tests for the complete analytics system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, temp_db):
        """Test complete analytics workflow"""
        dashboard = AnalyticsDashboard(temp_db)
        
        # Generate sample data
        dashboard.generate_sample_data()
        
        # Analyze usage patterns
        usage_patterns = dashboard.analytics_engine.analyze_usage_patterns()
        assert "model_statistics" in usage_patterns
        
        # Calculate cost metrics
        cost_metrics = dashboard.cost_optimizer.calculate_cost_metrics()
        assert isinstance(cost_metrics, CostMetrics)
        
        # Get performance metrics
        perf_metrics = dashboard.metrics_collector.get_performance_metrics()
        assert isinstance(perf_metrics, PerformanceMetrics)
        
        # Render dashboard
        layout = dashboard.renderer.render_dashboard()
        assert layout is not None
        
        # Export report
        report = dashboard.export_report()
        assert report is not None
        
        report_data = json.loads(report)
        assert report_data["performance"]["response_time"] >= 0
        assert len(report_data["insights"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])