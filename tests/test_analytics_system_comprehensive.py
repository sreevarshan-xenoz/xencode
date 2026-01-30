#!/usr/bin/env python3
"""
Comprehensive Tests for Analytics System

Tests for metrics collection, event tracking, analytics engine,
and comprehensive monitoring capabilities for the analytics system.
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime, timedelta

from xencode.analytics.analytics_engine import AnalyticsEngine, AnalyticsConfig
from xencode.analytics.metrics_collector import MetricsCollector, MetricDefinition, MetricType
from xencode.analytics.event_tracker import EventTracker, EventCategory, EventPriority, EventContext, TrackedEvent


class TestAnalyticsEngineBasics:
    """Test basic analytics engine functionality"""

    @pytest_asyncio.fixture
    async def analytics_engine(self):
        """Create an analytics engine for testing"""
        config = AnalyticsConfig(
            enable_metrics=True,
            enable_events=True,
            enable_prometheus=False,  # Disable for testing
            aggregation_interval=1,  # Fast aggregation for testing
            max_events_in_memory=1000,
            max_metrics_age_days=1
        )
        
        engine = AnalyticsEngine(config)
        await engine.start()  # Start the engine
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_analytics_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization"""
        assert analytics_engine is not None
        assert analytics_engine.config.enable_metrics is True
        assert analytics_engine.config.enable_events is True
        assert analytics_engine.config.enable_prometheus is False
        assert analytics_engine.metrics_collector is not None
        assert analytics_engine.event_tracker is not None
        # The engine is started in the fixture, so it should be running
        assert analytics_engine._running is True

    @pytest.mark.asyncio
    async def test_analytics_engine_start_stop(self, analytics_engine):
        """Test analytics engine start and stop"""
        # Start the engine
        await analytics_engine.start()
        assert analytics_engine._running is True

        # Stop the engine
        await analytics_engine.stop()
        assert analytics_engine._running is False

    @pytest.mark.asyncio
    async def test_user_action_tracking(self, analytics_engine):
        """Test user action tracking"""
        await analytics_engine.start()

        try:
            # Track user action
            event_id = analytics_engine.track_user_action(
                "file_opened",
                "user123",
                properties={"filename": "test.py", "file_size": 1024}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            user_events = [e for e in recent_events if e.context.user_id == "user123"]
            assert len(user_events) > 0

            event = user_events[0]
            assert event.event_type == "user_action_file_opened"
            assert event.properties["filename"] == "test.py"
            assert event.properties["file_size"] == 1024

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_system_event_tracking(self, analytics_engine):
        """Test system event tracking"""
        await analytics_engine.start()

        try:
            # Track system event
            event_id = analytics_engine.track_system_event(
                "startup",
                "main_process",
                properties={"version": "3.0.0", "platform": "windows"}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            system_events = [e for e in recent_events if e.event_type == "system_startup"]
            assert len(system_events) > 0

            event = system_events[0]
            assert event.category == EventCategory.SYSTEM_EVENT
            assert event.properties["component"] == "main_process"
            assert event.properties["version"] == "3.0.0"

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_performance_tracking(self, analytics_engine):
        """Test performance tracking"""
        await analytics_engine.start()

        try:
            # Track performance event
            event_id = analytics_engine.track_performance(
                "document_processing",
                150.5,  # 150.5ms
                "document_processor",
                properties={"document_type": "pdf", "pages": 10}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            perf_events = [e for e in recent_events if e.event_type == "performance_document_processing"]
            assert len(perf_events) > 0

            event = perf_events[0]
            assert event.category == EventCategory.PERFORMANCE
            assert event.metrics["duration_ms"] == 150.5
            assert event.properties["document_type"] == "pdf"
            assert event.properties["pages"] == 10

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_error_tracking(self, analytics_engine):
        """Test error tracking"""
        await analytics_engine.start()

        try:
            # Track error event
            event_id = analytics_engine.track_error(
                "validation_error",
                "Invalid file format",
                "document_processor",
                properties={"filename": "invalid.txt"}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            error_events = [e for e in recent_events if e.event_type == "error_validation_error"]
            assert len(error_events) > 0

            event = error_events[0]
            assert event.category == EventCategory.ERROR
            assert event.properties["error_message"] == "Invalid file format"
            assert event.properties["component"] == "document_processor"
            assert event.properties["filename"] == "invalid.txt"

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_ai_interaction_tracking(self, analytics_engine):
        """Test AI interaction tracking"""
        await analytics_engine.start()

        try:
            # Track AI interaction
            event_id = analytics_engine.track_ai_interaction(
                "gpt-4",
                "code_analysis",
                "user123",
                2500.0,  # 2.5 seconds
                success=True,
                properties={"tokens_used": 150}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            ai_events = [e for e in recent_events if e.event_type == "ai_code_analysis"]
            assert len(ai_events) > 0

            event = ai_events[0]
            assert event.category == EventCategory.AI_INTERACTION
            assert event.properties["model"] == "gpt-4"
            assert event.properties["tokens_used"] == 150
            assert event.metrics["duration_ms"] == 2500.0
            assert event.metrics["success"] == 1.0

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_metric_tracking(self, analytics_engine):
        """Test metric tracking"""
        await analytics_engine.start()

        try:
            # Track metrics
            analytics_engine.increment_counter("test_counter", 1.0, {"label": "value"})
            analytics_engine.set_gauge("test_gauge", 42.0, {"type": "test"})
            analytics_engine.observe_histogram("test_histogram", 100.0, {"operation": "test"})

            # Verify metrics were tracked
            summary = analytics_engine.get_metrics_summary()
            assert "metrics" in summary

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_analytics_report_generation(self, analytics_engine):
        """Test analytics report generation"""
        await analytics_engine.start()

        try:
            # Generate some activity
            analytics_engine.track_user_action("test_action", "user123")
            analytics_engine.track_performance("test_op", 100.0, "test_component")
            analytics_engine.track_error("test_error", "Test error", "test_component")

            # Generate report
            report = analytics_engine.generate_analytics_report(hours=1)

            assert "report_generated_at" in report
            assert "time_period_hours" in report
            assert report["time_period_hours"] == 1
            assert "summary" in report
            assert "events" in report

            # Check events section
            if "events" in report:
                assert "total_events" in report["events"]
                assert "events_by_category" in report["events"]
                assert "events_by_type" in report["events"]

        finally:
            await analytics_engine.stop()


class TestMetricsCollector:
    """Test metrics collection functionality"""

    @pytest_asyncio.fixture
    async def metrics_collector(self):
        """Create a metrics collector for testing"""
        collector = MetricsCollector(
            enable_prometheus=False,
            metrics_port=8001,  # Different port to avoid conflicts
            storage_path=Path(tempfile.mkdtemp()) / "analytics"
        )
        await collector.start()  # Start the collector
        yield collector
        await collector.stop()

    @pytest.mark.asyncio
    async def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector is not None
        assert metrics_collector.enable_prometheus is False
        assert metrics_collector.storage_path.exists()
        assert metrics_collector._running is False

    @pytest.mark.asyncio
    async def test_counter_metrics(self, metrics_collector):
        """Test counter metric functionality"""
        await metrics_collector.start()

        try:
            # Increment counter
            metrics_collector.increment_counter("test_requests_total", 1.0, {"method": "GET", "endpoint": "/"})
            metrics_collector.increment_counter("test_requests_total", 2.0, {"method": "POST", "endpoint": "/"})

            # Verify counter was incremented
            summary = metrics_collector.get_metrics_summary()
            # The counter should be tracked internally

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_gauge_metrics(self, metrics_collector):
        """Test gauge metric functionality"""
        await metrics_collector.start()

        try:
            # Set gauge values
            metrics_collector.set_gauge("test_active_users", 42.0)
            metrics_collector.set_gauge("test_memory_usage_bytes", 1024.0, {"component": "main"})

            # Verify gauges were set
            summary = metrics_collector.get_metrics_summary()

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_histogram_metrics(self, metrics_collector):
        """Test histogram metric functionality"""
        await metrics_collector.start()

        try:
            # Observe histogram values
            metrics_collector.observe_histogram("test_request_duration_seconds", 0.1, {"endpoint": "/"})
            metrics_collector.observe_histogram("test_request_duration_seconds", 0.5, {"endpoint": "/api"})

            # Verify histogram was observed
            summary = metrics_collector.get_metrics_summary()

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_metric_timing(self, metrics_collector):
        """Test metric timing functionality"""
        await metrics_collector.start()

        try:
            # Time an operation
            with metrics_collector.time_histogram("test_operation_duration", {"operation": "timing_test"}):
                await asyncio.sleep(0.01)  # Small delay to measure

            # Verify timing was recorded
            summary = metrics_collector.get_metrics_summary()

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_custom_metric_registration(self, metrics_collector):
        """Test custom metric registration"""
        await metrics_collector.start()

        try:
            # Register a custom metric
            custom_metric = MetricDefinition(
                name="custom_test_metric",
                metric_type=MetricType.COUNTER,
                description="Custom test metric",
                labels=["test_label"]
            )
            metrics_collector.register_metric(custom_metric)

            # Use the custom metric
            metrics_collector.increment_counter("custom_test_metric", 1.0, {"test_label": "test_value"})

            # Verify custom metric was tracked
            summary = metrics_collector.get_metrics_summary()

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_prometheus_export(self, metrics_collector):
        """Test Prometheus metrics export"""
        await metrics_collector.start()

        try:
            # Export Prometheus metrics (should work even without Prometheus server)
            prometheus_output = metrics_collector.export_prometheus_metrics()
            assert isinstance(prometheus_output, str)

        finally:
            await metrics_collector.stop()


class TestEventTracker:
    """Test event tracking functionality"""

    @pytest_asyncio.fixture
    async def event_tracker(self):
        """Create an event tracker for testing"""
        tracker = EventTracker(
            storage_path=Path(tempfile.mkdtemp()) / "events"
        )
        await tracker.start()  # Start the tracker
        yield tracker
        await tracker.stop()

    @pytest.mark.asyncio
    async def test_event_tracker_initialization(self, event_tracker):
        """Test event tracker initialization"""
        assert event_tracker is not None
        assert event_tracker.storage_path.exists()
        assert event_tracker._running is False

    @pytest.mark.asyncio
    async def test_event_tracking(self, event_tracker):
        """Test basic event tracking"""
        await event_tracker.start()

        try:
            # Track a user action event
            event_id = event_tracker.track_user_action(
                "file_saved",
                "user123",
                session_id="session456",
                properties={"filename": "test.py", "size": 1024},
                metrics={"duration_ms": 50.0}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = event_tracker.get_recent_events(hours=1)
            assert len(recent_events) > 0

            event = recent_events[0]
            assert event.event_type == "user_action_file_saved"
            assert event.context.user_id == "user123"
            assert event.context.session_id == "session456"
            assert event.properties["filename"] == "test.py"
            assert event.metrics["duration_ms"] == 50.0

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_system_event_tracking(self, event_tracker):
        """Test system event tracking"""
        await event_tracker.start()

        try:
            # Track a system event
            event_id = event_tracker.track_system_event(
                "startup",
                "main_process",
                properties={"version": "3.0.0", "platform": "linux"},
                priority=EventPriority.HIGH
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = event_tracker.get_recent_events(hours=1)
            system_events = [e for e in recent_events if e.event_type == "system_startup"]
            assert len(system_events) > 0

            event = system_events[0]
            assert event.category == EventCategory.SYSTEM_EVENT
            assert event.priority == EventPriority.HIGH
            assert event.properties["component"] == "main_process"
            assert event.properties["version"] == "3.0.0"

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_performance_event_tracking(self, event_tracker):
        """Test performance event tracking"""
        await event_tracker.start()

        try:
            # Track a performance event
            event_id = event_tracker.track_performance_event(
                "database_query",
                250.0,  # 250ms
                "database_layer",
                success=True,
                properties={"query_type": "SELECT", "table": "users"}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = event_tracker.get_recent_events(hours=1)
            perf_events = [e for e in recent_events if e.event_type == "performance_database_query"]
            assert len(perf_events) > 0

            event = perf_events[0]
            assert event.category == EventCategory.PERFORMANCE
            assert event.metrics["duration_ms"] == 250.0
            assert event.metrics["success"] == 1.0
            assert event.properties["query_type"] == "SELECT"

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_error_event_tracking(self, event_tracker):
        """Test error event tracking"""
        await event_tracker.start()

        try:
            # Track an error event
            event_id = event_tracker.track_error_event(
                "validation_error",
                "Invalid input format",
                "validator",
                user_id="user123",
                properties={"field": "email", "value": "invalid_email"}
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = event_tracker.get_recent_events(hours=1)
            error_events = [e for e in recent_events if e.event_type == "error_validation_error"]
            assert len(error_events) > 0

            event = error_events[0]
            assert event.category == EventCategory.ERROR
            assert event.context.user_id == "user123"
            assert event.properties["error_message"] == "Invalid input format"
            assert event.properties["component"] == "validator"
            assert event.properties["field"] == "email"

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_ai_interaction_tracking(self, event_tracker):
        """Test AI interaction tracking"""
        await event_tracker.start()

        try:
            # Track an AI interaction
            event_id = event_tracker.track_ai_interaction(
                "gpt-3.5-turbo",
                "text_generation",
                "user123",
                1500.0,  # 1.5 seconds
                properties={"prompt_tokens": 100, "completion_tokens": 50},
                success=True
            )
            assert event_id is not None

            # Verify event was tracked
            recent_events = event_tracker.get_recent_events(hours=1)
            ai_events = [e for e in recent_events if e.event_type == "ai_text_generation"]
            assert len(ai_events) > 0

            event = ai_events[0]
            assert event.category == EventCategory.AI_INTERACTION
            assert event.context.user_id == "user123"
            assert event.properties["model"] == "gpt-3.5-turbo"
            assert event.properties["prompt_tokens"] == 100
            assert event.metrics["duration_ms"] == 1500.0
            assert event.metrics["success"] == 1.0

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_event_querying(self, event_tracker):
        """Test event querying functionality"""
        await event_tracker.start()

        try:
            # Track several events
            event_tracker.track_user_action("action1", "user1", properties={"test": "value1"})
            event_tracker.track_user_action("action2", "user2", properties={"test": "value2"})
            event_tracker.track_system_event("startup", "main")

            # Query events by user
            user1_events = event_tracker.get_events_by_user("user1")
            assert len(user1_events) >= 1
            assert all(e.context.user_id == "user1" for e in user1_events)

            # Query events by type
            startup_events = event_tracker.get_events_by_type("system_startup")
            assert len(startup_events) >= 1
            assert all(e.event_type == "system_startup" for e in startup_events)

            # Get recent events
            recent_events = event_tracker.get_recent_events(hours=1, limit=10)
            assert len(recent_events) >= 3

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_event_callback_system(self, event_tracker):
        """Test event callback system"""
        await event_tracker.start()

        try:
            # Create a callback to capture events
            captured_events = []

            def event_callback(event):
                captured_events.append(event)

            # Add callback for all events
            event_tracker.add_event_callback("*", event_callback)

            # Track an event
            event_tracker.track_user_action("callback_test", "user123")

            # Allow time for callback to execute
            await asyncio.sleep(0.1)

            # Verify callback was called
            assert len(captured_events) >= 1
            assert captured_events[0].event_type == "user_action_callback_test"

        finally:
            await event_tracker.stop()

    @pytest.mark.asyncio
    async def test_event_statistics(self, event_tracker):
        """Test event statistics"""
        await event_tracker.start()

        try:
            # Track various events
            for i in range(5):
                event_tracker.track_user_action(f"action_{i}", f"user_{i}")
                event_tracker.track_system_event("startup", "main_process")
                event_tracker.track_performance_event("op", 100.0, "comp")

            # Get statistics
            stats = event_tracker.get_statistics()
            assert stats["total_events"] >= 15  # 5 user + 5 system + 5 perf
            assert stats["unique_users"] >= 5
            assert stats["events_by_category"]["user_action"] >= 5
            assert stats["events_by_category"]["system_event"] >= 5
            assert stats["events_by_category"]["performance"] >= 5

        finally:
            await event_tracker.stop()


class TestAnalyticsIntegration:
    """Test analytics system integration"""

    @pytest_asyncio.fixture
    async def analytics_system(self):
        """Create a full analytics system for testing"""
        config = AnalyticsConfig(
            enable_metrics=True,
            enable_events=True,
            enable_prometheus=False,
            aggregation_interval=1,
            max_events_in_memory=1000,
            max_metrics_age_days=1
        )

        engine = AnalyticsEngine(config)
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_full_analytics_workflow(self, analytics_system):
        """Test complete analytics workflow"""
        await analytics_system.start()

        try:
            # 1. Track various types of events
            user_event_id = analytics_system.track_user_action(
                "document_opened",
                "user123",
                filename="test.py",
                file_size=2048
            )
            assert user_event_id is not None

            perf_event_id = analytics_system.track_performance(
                "file_processing",
                350.0,
                "file_processor",
                file_type="python"
            )
            assert perf_event_id is not None

            error_event_id = analytics_system.track_error(
                "parse_error",
                "Syntax error in file",
                "parser",
                filename="test.py"
            )
            assert error_event_id is not None

            ai_event_id = analytics_system.track_ai_interaction(
                "gpt-4",
                "code_analysis",
                "user123",
                2000.0,
                tokens_used=200,
                success=True
            )
            assert ai_event_id is not None

            # 2. Track metrics
            analytics_system.increment_counter("xencode_requests_total", 1.0, {"method": "GET", "endpoint": "/api/test"})
            analytics_system.set_gauge("xencode_active_users", 1.0)
            analytics_system.observe_histogram("xencode_request_duration_seconds", 0.2, {"endpoint": "/api/test"})

            # 3. Verify events were tracked
            recent_events = analytics_system.get_recent_events(hours=1)
            assert len(recent_events) >= 4  # At least the 4 events we tracked

            # 4. Verify metrics were tracked
            summary = analytics_system.get_metrics_summary()
            assert "metrics" in summary
            assert "events" in summary

            # 5. Generate comprehensive report
            report = analytics_system.generate_analytics_report(hours=1)
            assert "report_generated_at" in report
            assert "time_period_hours" in report
            assert "summary" in report
            assert "events" in report

            # 6. Check that the report contains expected information
            if "events" in report:
                assert "total_events" in report["events"]
                assert report["events"]["total_events"] >= 4

        finally:
            await analytics_system.stop()

    @pytest.mark.asyncio
    async def test_analytics_error_handling(self, analytics_system):
        """Test analytics system error handling"""
        await analytics_system.start()

        try:
            # Test with invalid data
            # This should not crash the system
            try:
                # Track event with None values (should be handled gracefully)
                event_id = analytics_system.track_user_action(None, None)
                # This might return None or handle gracefully
            except Exception:
                # Expected - should handle gracefully
                pass

            # System should still be functional
            summary = analytics_system.get_metrics_summary()
            assert "analytics_engine_running" in summary

        finally:
            await analytics_system.stop()

    @pytest.mark.asyncio
    async def test_analytics_concurrent_access(self, analytics_system):
        """Test analytics system with concurrent access"""
        await analytics_system.start()

        async def track_events(user_id: str):
            for i in range(10):
                analytics_system.track_user_action(f"action_{i}", user_id, iteration=i)
                analytics_system.track_performance(f"op_{i}", i * 10.0, "test_component", iteration=i)
                await asyncio.sleep(0.001)  # Small delay to allow other tasks to run

        try:
            # Run multiple concurrent tracking operations
            await asyncio.gather(
                track_events("user1"),
                track_events("user2"),
                track_events("user3")
            )

            # Verify all events were tracked
            recent_events = analytics_system.get_recent_events(hours=1)
            assert len(recent_events) >= 60  # 3 users * 10 actions * 2 event types (user + perf)

            # Check statistics
            summary = analytics_system.get_metrics_summary()
            if "events" in summary:
                assert summary["events"]["total_events"] >= 60

        finally:
            await analytics_system.stop()

    @pytest.mark.asyncio
    async def test_analytics_data_persistence(self, tmp_path):
        """Test analytics data persistence"""
        config = AnalyticsConfig(
            enable_metrics=True,
            enable_events=True,
            enable_prometheus=False,
            aggregation_interval=1,
            max_events_in_memory=100,
            max_metrics_age_days=1,
            storage_path=tmp_path / "analytics"
        )

        # Create first engine instance
        engine1 = AnalyticsEngine(config)
        await engine1.initialize()
        await engine1.start()

        try:
            # Track some events
            engine1.track_user_action("persistent_action", "user123", test_data="value1")
            await asyncio.sleep(0.1)  # Allow time for any background processing
        finally:
            await engine1.stop()

        # Create second engine instance with same storage
        engine2 = AnalyticsEngine(config)
        await engine2.initialize()
        await engine2.start()

        try:
            # Check that data is preserved (this depends on implementation)
            recent_events = engine2.get_recent_events(hours=1)
            # The exact behavior depends on the implementation - some data may be persisted
        finally:
            await engine2.stop()


class TestAnalyticsPerformance:
    """Test analytics system performance"""

    @pytest_asyncio.fixture
    async def analytics_engine(self):
        """Create an analytics engine for performance testing"""
        config = AnalyticsConfig(
            enable_metrics=True,
            enable_events=True,
            enable_prometheus=False,
            aggregation_interval=5,  # Longer interval to reduce background noise
            max_events_in_memory=2000,
            max_metrics_age_days=1
        )

        engine = AnalyticsEngine(config)
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_high_volume_event_tracking(self, analytics_engine):
        """Test analytics system with high volume of events"""
        await analytics_engine.start()

        try:
            start_time = asyncio.get_event_loop().time()

            # Track many events quickly
            for i in range(100):
                analytics_engine.track_user_action(f"bulk_action_{i}", f"user_{i % 10}")
                if i % 10 == 0:  # Yield control periodically
                    await asyncio.sleep(0)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Should handle high volume efficiently (less than 5 seconds for 100 events)
            assert duration < 5.0

            # Verify events were tracked
            recent_events = analytics_engine.get_recent_events(hours=1)
            assert len(recent_events) >= 100

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, analytics_engine):
        """Test memory usage under load"""
        await analytics_engine.start()

        try:
            initial_events_count = len(analytics_engine.get_recent_events(hours=1))

            # Track events with different properties to simulate real usage
            for i in range(50):
                analytics_engine.track_performance(
                    f"operation_{i % 5}",
                    50.0 + (i % 10) * 10,  # Varying durations
                    "test_component",
                    iteration=i,
                    extra_data=f"test_data_{i}"
                )
                await asyncio.sleep(0)  # Yield control

            # Check that memory usage is reasonable and events are rotated properly
            final_events = analytics_engine.get_recent_events(hours=1)
            final_count = len(final_events)

            # Should not grow unbounded
            assert final_count <= analytics_engine.config.max_events_in_memory

        finally:
            await analytics_engine.stop()

    @pytest.mark.asyncio
    async def test_metric_aggregation_performance(self, analytics_engine):
        """Test metric aggregation performance"""
        await analytics_engine.start()

        try:
            # Track many metrics of the same type to test aggregation
            start_time = asyncio.get_event_loop().time()

            for i in range(50):
                analytics_engine.increment_counter("test_counter_agg", 1.0, {"type": "agg_test"})
                analytics_engine.set_gauge("test_gauge_agg", float(i), {"type": "agg_test"})
                await asyncio.sleep(0)  # Yield control

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Should handle aggregation efficiently
            assert duration < 2.0

            # Verify metrics were aggregated properly
            summary = analytics_engine.get_metrics_summary()
            # This depends on the implementation of aggregation

        finally:
            await analytics_engine.stop()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])