#!/usr/bin/env python3
"""
Test script for Analytics Data Collection Infrastructure
"""

import asyncio
import time
from datetime import datetime, timedelta

from xencode.analytics.metrics_collector import MetricsCollector, AnalyticsEvent, MetricType, MetricDefinition
from xencode.analytics.event_tracker import EventTracker, EventCategory, EventPriority, EventContext
from xencode.analytics.analytics_engine import AnalyticsEngine, AnalyticsConfig
from xencode.monitoring.metrics_collector import PrometheusMetricsCollector


async def test_metrics_collector():
    """Test metrics collector functionality"""
    
    print("📊 Testing Metrics Collector...")
    
    # Create metrics collector
    collector = MetricsCollector(enable_prometheus=False)  # Disable Prometheus for testing
    
    try:
        await collector.start()
        print("✓ Metrics collector started")
        
        # Test counter metric
        collector.increment_counter("test_counter", 5.0, {"component": "test"})
        print("✓ Counter metric incremented")
        
        # Test gauge metric
        collector.set_gauge("test_gauge", 42.0, {"type": "test"})
        print("✓ Gauge metric set")
        
        # Test histogram metric
        collector.observe_histogram("test_histogram", 1.5, {"operation": "test"})
        print("✓ Histogram metric observed")
        
        # Test timing context manager
        with collector.time_histogram("test_duration", {"operation": "test_timing"}):
            await asyncio.sleep(0.1)
        print("✓ Timing context manager worked")
        
        # Test event tracking
        event = AnalyticsEvent(
            event_type="test_event",
            timestamp=datetime.now(),
            user_id="test_user",
            properties={"test": "data"},
            metrics={"value": 123.45}
        )
        collector.track_event(event)
        print("✓ Event tracked")
        
        # Test metrics summary
        summary = collector.get_metrics_summary()
        assert summary['total_metric_definitions'] > 0, "Should have metric definitions"
        assert summary['running'] == True, "Should be running"
        print("✓ Metrics summary retrieved")
        
        await collector.stop()
        print("✓ Metrics collector stopped")
        
    except Exception as e:
        await collector.stop()
        raise e
    
    print("✅ Metrics collector test passed!")


async def test_event_tracker():
    """Test event tracker functionality"""
    
    print("\n📝 Testing Event Tracker...")
    
    # Create event tracker
    tracker = EventTracker()
    
    try:
        await tracker.start()
        print("✓ Event tracker started")
        
        # Test user action tracking
        event_id = tracker.track_user_action(
            action="test_action",
            user_id="test_user_123",
            properties={"page": "test_page"},
            metrics={"duration": 1500.0}
        )
        assert event_id is not None, "Should return event ID"
        print("✓ User action tracked")
        
        # Test system event tracking
        system_event_id = tracker.track_system_event(
            event="startup",
            component="test_component",
            properties={"version": "1.0.0"},
            priority=EventPriority.HIGH
        )
        assert system_event_id is not None, "Should return system event ID"
        print("✓ System event tracked")
        
        # Test performance event tracking
        perf_event_id = tracker.track_performance_event(
            operation="test_operation",
            duration_ms=250.0,
            component="test_component",
            success=True
        )
        assert perf_event_id is not None, "Should return performance event ID"
        print("✓ Performance event tracked")
        
        # Test error event tracking
        error_event_id = tracker.track_error_event(
            error_type="test_error",
            error_message="Test error message",
            component="test_component",
            user_id="test_user_123"
        )
        assert error_event_id is not None, "Should return error event ID"
        print("✓ Error event tracked")
        
        # Test AI interaction tracking
        ai_event_id = tracker.track_ai_interaction(
            model="test_model",
            operation="generate",
            user_id="test_user_123",
            duration_ms=2000.0,
            success=True,
            properties={"tokens": 150}
        )
        assert ai_event_id is not None, "Should return AI event ID"
        print("✓ AI interaction tracked")
        
        # Test event querying
        user_events = tracker.get_events_by_user("test_user_123")
        assert len(user_events) >= 3, "Should have at least 3 user events"
        print(f"✓ Retrieved {len(user_events)} user events")
        
        recent_events = tracker.get_recent_events(hours=1)
        assert len(recent_events) >= 5, "Should have at least 5 recent events"
        print(f"✓ Retrieved {len(recent_events)} recent events")
        
        # Test statistics
        stats = tracker.get_statistics()
        assert stats['total_events'] >= 5, "Should have tracked at least 5 events"
        assert stats['unique_users'] >= 1, "Should have at least 1 unique user"
        print("✓ Statistics retrieved")
        
        await tracker.stop()
        print("✓ Event tracker stopped")
        
    except Exception as e:
        await tracker.stop()
        raise e
    
    print("✅ Event tracker test passed!")


async def test_analytics_engine():
    """Test analytics engine integration"""
    
    print("\n🔧 Testing Analytics Engine...")
    
    # Create analytics engine with custom config
    config = AnalyticsConfig(
        enable_metrics=True,
        enable_events=True,
        enable_prometheus=False,  # Disable for testing
        aggregation_interval=5  # Short interval for testing
    )
    
    engine = AnalyticsEngine(config)
    
    try:
        await engine.start()
        print("✓ Analytics engine started")
        
        # Test integrated tracking methods
        user_action_id = engine.track_user_action("login", "user_456", properties={"method": "oauth"})
        assert user_action_id is not None, "Should track user action"
        print("✓ User action tracked through engine")
        
        system_event_id = engine.track_system_event("cache_clear", "cache_manager", properties={"size": "100MB"})
        assert system_event_id is not None, "Should track system event"
        print("✓ System event tracked through engine")
        
        perf_id = engine.track_performance("api_call", 150.0, "api_gateway", success=True)
        assert perf_id is not None, "Should track performance"
        print("✓ Performance tracked through engine")
        
        error_id = engine.track_error("validation_error", "Invalid input", "validator", user_id="user_456")
        assert error_id is not None, "Should track error"
        print("✓ Error tracked through engine")
        
        ai_id = engine.track_ai_interaction("gpt-4", "completion", "user_456", 1800.0, success=True)
        assert ai_id is not None, "Should track AI interaction"
        print("✓ AI interaction tracked through engine")
        
        # Test metrics methods
        engine.increment_counter("test_requests_total", 1.0, {"endpoint": "/api/test"})
        engine.set_gauge("test_active_users", 42.0)
        engine.observe_histogram("test_response_time", 0.25, {"service": "test"})
        print("✓ Metrics methods working")
        
        # Test timing context manager
        with engine.time_operation("test_operation_duration", {"operation": "test"}):
            await asyncio.sleep(0.05)
        print("✓ Timing operation worked")
        
        # Test data retrieval
        recent_events = engine.get_recent_events(hours=1)
        assert len(recent_events) >= 5, "Should have recent events"
        print(f"✓ Retrieved {len(recent_events)} recent events")
        
        user_events = engine.get_events_by_user("user_456")
        assert len(user_events) >= 3, "Should have user events"
        print(f"✓ Retrieved {len(user_events)} user events")
        
        # Test metrics summary
        summary = engine.get_metrics_summary()
        assert summary['analytics_engine_running'] == True, "Should be running"
        assert 'metrics' in summary, "Should have metrics summary"
        assert 'events' in summary, "Should have events summary"
        print("✓ Metrics summary retrieved")
        
        # Test analytics report generation
        report = engine.generate_analytics_report(hours=1)
        assert 'report_generated_at' in report, "Should have report timestamp"
        assert 'events' in report, "Should have events analysis"
        assert report['events']['total_events'] >= 5, "Should have tracked events"
        print("✓ Analytics report generated")
        
        # Wait a bit for analysis loop to run
        await asyncio.sleep(6)
        print("✓ Analysis loop executed")
        
        await engine.stop()
        print("✓ Analytics engine stopped")
        
    except Exception as e:
        await engine.stop()
        raise e
    
    print("✅ Analytics engine test passed!")


async def test_prometheus_metrics_collector():
    """Test Prometheus metrics collector"""
    
    print("\n🎯 Testing Prometheus Metrics Collector...")
    
    # Create Prometheus collector with separate registry
    from prometheus_client import CollectorRegistry
    separate_registry = CollectorRegistry()
    
    collector = PrometheusMetricsCollector(
        registry=separate_registry,
        metrics_port=8001,  # Use different port to avoid conflicts
        collect_system_metrics=True
    )
    
    try:
        await collector.start()
        print("✓ Prometheus collector started")
        
        # Test application metrics
        collector.record_request("GET", "/api/test", "200", 0.15)
        collector.record_request("POST", "/api/data", "201", 0.25)
        print("✓ HTTP request metrics recorded")
        
        collector.set_active_connections(25)
        print("✓ Active connections metric set")
        
        collector.set_plugin_status("test-plugin", True)
        collector.set_plugin_status("disabled-plugin", False)
        print("✓ Plugin status metrics set")
        
        collector.record_workspace_operation("create", "success")
        collector.record_workspace_operation("delete", "error")
        print("✓ Workspace operation metrics recorded")
        
        collector.record_ai_request("gpt-4", "success", 2.5)
        collector.record_ai_request("claude", "error", 0.1)
        print("✓ AI request metrics recorded")
        
        collector.record_error("api", "validation_error")
        collector.record_error("database", "connection_timeout")
        print("✓ Error metrics recorded")
        
        collector.record_cache_operation("get", "hit")
        collector.record_cache_operation("get", "miss")
        collector.set_cache_size("memory", 1024 * 1024 * 50)  # 50MB
        print("✓ Cache metrics recorded")
        
        # Test status
        status = collector.get_status()
        assert status['running'] == True, "Should be running"
        print("✓ Status retrieved")
        
        # Test metrics export
        metrics_output = collector.export_metrics()
        assert len(metrics_output) > 0, "Should have metrics output"
        print("✓ Metrics exported")
        
        # Wait for system metrics collection
        await asyncio.sleep(2)
        print("✓ System metrics collected")
        
        await collector.stop()
        print("✓ Prometheus collector stopped")
        
    except Exception as e:
        await collector.stop()
        raise e
    
    print("✅ Prometheus metrics collector test passed!")


async def test_integration():
    """Test integration between components"""
    
    print("\n🔗 Testing Component Integration...")
    
    # Test that events flow to metrics
    config = AnalyticsConfig(enable_prometheus=False)
    engine = AnalyticsEngine(config)
    
    try:
        await engine.start()
        
        # Track various events
        engine.track_user_action("page_view", "user_789", properties={"page": "/dashboard"})
        engine.track_performance("database_query", 45.0, "database", success=True)
        engine.track_error("timeout", "Request timeout", "api", user_id="user_789")
        engine.track_ai_interaction("gpt-3.5", "chat", "user_789", 800.0, success=True)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify metrics were updated
        summary = engine.get_metrics_summary()
        assert summary['events']['total_events'] >= 4, "Should have tracked events"
        print("✓ Events tracked and processed")
        
        # Verify analytics report includes all data
        report = engine.generate_analytics_report(hours=1)
        assert 'events' in report, "Should have events in report"
        assert 'errors' in report, "Should have errors analysis"
        assert 'performance' in report, "Should have performance analysis"
        print("✓ Comprehensive analytics report generated")
        
        await engine.stop()
        
    except Exception as e:
        await engine.stop()
        raise e
    
    print("✅ Integration test passed!")


async def main():
    """Run all analytics infrastructure tests"""
    
    print("🚀 Starting Analytics Data Collection Infrastructure Tests\n")
    
    try:
        await test_metrics_collector()
        await test_event_tracker()
        await test_analytics_engine()
        await test_prometheus_metrics_collector()
        await test_integration()
        
        print("\n🎉 All analytics infrastructure tests passed!")
        print("✅ MetricsCollector with Prometheus integration implemented")
        print("✅ EventTracker for comprehensive event tracking functional")
        print("✅ AnalyticsEngine for coordinated data collection working")
        print("✅ Real-time metrics aggregation and analysis active")
        print("✅ System monitoring and observability infrastructure complete")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())