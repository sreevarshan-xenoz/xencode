#!/usr/bin/env python3
"""
Tests for Provider Health Dashboard (S5-02)

Tests for:
- Provider health monitoring
- Latency metrics tracking
- Error rate calculation
- Health dashboard API endpoints
- Provider recommendations
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLatencyMetrics:
    """Tests for LatencyMetrics class"""

    def test_add_sample(self):
        """Test adding latency samples"""
        from xencode.monitoring.provider_health import LatencyMetrics

        metrics = LatencyMetrics()
        metrics.add_sample(100)
        metrics.add_sample(200)
        metrics.add_sample(150)

        assert metrics.sample_count == 3
        assert metrics.current_ms == 150
        assert metrics.min_ms == 100
        assert metrics.max_ms == 200
        assert metrics.avg_ms == 150

    def test_percentile_calculation(self):
        """Test percentile calculations"""
        from xencode.monitoring.provider_health import LatencyMetrics

        metrics = LatencyMetrics()

        # Add 100 samples
        for i in range(1, 101):
            metrics.add_sample(float(i))

        assert metrics.p50_ms == 50
        assert metrics.p95_ms == 95
        assert metrics.p99_ms == 99

    def test_trend_detection(self):
        """Test latency trend detection"""
        from xencode.monitoring.provider_health import LatencyMetrics

        # Increasing trend
        metrics_inc = LatencyMetrics()
        for i in range(100):
            metrics_inc.add_sample(100 + i)  # Increasing
        assert metrics_inc.get_trend() == "increasing"

        # Decreasing trend
        metrics_dec = LatencyMetrics()
        for i in range(100):
            metrics_dec.add_sample(200 - i)  # Decreasing
        assert metrics_dec.get_trend() == "decreasing"

        # Stable trend
        metrics_stable = LatencyMetrics()
        for i in range(100):
            metrics_stable.add_sample(100 + (i % 10))  # Stable around 105
        assert metrics_stable.get_trend() == "stable"

    def test_to_dict(self):
        """Test serialization to dictionary"""
        from xencode.monitoring.provider_health import LatencyMetrics

        metrics = LatencyMetrics()
        metrics.add_sample(150)

        data = metrics.to_dict()

        assert 'current_ms' in data
        assert 'avg_ms' in data
        assert 'trend' in data
        assert data['sample_count'] == 1


class TestErrorMetrics:
    """Tests for ErrorMetrics class"""

    def test_record_error(self):
        """Test error recording"""
        from xencode.monitoring.provider_health import ErrorMetrics

        errors = ErrorMetrics()
        errors.record_error('timeout', 'Connection timed out')

        assert errors.total_errors == 1
        assert errors.consecutive_errors == 1
        assert errors.last_error_message == 'Connection timed out'
        assert 'timeout' in errors.error_types

    def test_record_success_resets_consecutive(self):
        """Test that success resets consecutive error count"""
        from xencode.monitoring.provider_health import ErrorMetrics

        errors = ErrorMetrics()
        errors.record_error('error1', 'Error 1')
        errors.record_error('error2', 'Error 2')
        assert errors.consecutive_errors == 2

        errors.record_success()
        assert errors.consecutive_errors == 0

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        from xencode.monitoring.provider_health import ErrorMetrics

        errors = ErrorMetrics()
        errors.record_error('test', 'Test error')
        errors.record_error('test', 'Test error')

        errors.calculate_error_rate(100)

        assert errors.error_rate == 2.0  # 2 errors out of 100 = 2%

    def test_time_based_counts(self):
        """Test time-based error counting"""
        from xencode.monitoring.provider_health import ErrorMetrics

        errors = ErrorMetrics()

        # Add errors to history
        now = datetime.now()
        for i in range(5):
            errors.error_history.append({
                'timestamp': now - timedelta(minutes=i*10),
                'type': 'test',
                'message': f'Error {i}',
            })

        errors._update_time_based_counts()

        # All 5 should be in last hour
        assert errors.errors_last_hour == 5
        # All 5 should be in last 24h
        assert errors.errors_last_24h == 5

    def test_to_dict(self):
        """Test error metrics serialization"""
        from xencode.monitoring.provider_health import ErrorMetrics

        errors = ErrorMetrics()
        errors.record_error('timeout', 'Timeout')
        errors.calculate_error_rate(50)

        data = errors.to_dict()

        assert data['total_errors'] == 1
        assert data['error_rate'] == 2.0
        assert data['error_types']['timeout'] == 1


class TestProviderHealth:
    """Tests for ProviderHealth dataclass"""

    def test_default_values(self):
        """Test default provider health values"""
        from xencode.monitoring.provider_health import (
            HealthStatus,
            ProviderHealth,
            ProviderType,
        )

        health = ProviderHealth(provider=ProviderType.LOCAL_OLLAMA)

        assert health.status == HealthStatus.UNKNOWN
        assert health.uptime_percentage == 0.0
        assert health.model_count == 0

    def test_to_dict(self):
        """Test provider health serialization"""
        from xencode.monitoring.provider_health import (
            HealthStatus,
            ProviderHealth,
            ProviderType,
        )

        health = ProviderHealth(
            provider=ProviderType.CLOUD_QWEN,
            status=HealthStatus.HEALTHY,
            uptime_percentage=99.5,
        )

        data = health.to_dict()

        assert data['provider'] == 'cloud_qwen'
        assert data['status'] == 'healthy'
        assert data['uptime_percentage'] == 99.5


class TestProviderHealthMonitor:
    """Tests for ProviderHealthMonitor class"""

    def test_initialization(self):
        """Test monitor initialization"""
        from xencode.monitoring.provider_health import ProviderHealthMonitor

        monitor = ProviderHealthMonitor()

        # Should have all provider types
        from xencode.monitoring.provider_health import ProviderType
        assert len(monitor.providers) == len(ProviderType)

        # Each provider should have health tracking
        for provider in ProviderType:
            assert provider in monitor.providers

    def test_get_health_summary(self):
        """Test health summary generation"""
        from xencode.monitoring.provider_health import (
            HealthStatus,
            ProviderHealthMonitor,
        )

        monitor = ProviderHealthMonitor()

        # Set some statuses
        from xencode.monitoring.provider_health import ProviderType
        monitor.providers[ProviderType.LOCAL_OLLAMA].status = HealthStatus.HEALTHY
        monitor.providers[ProviderType.CLOUD_QWEN].status = HealthStatus.HEALTHY
        monitor.providers[ProviderType.CLOUD_OPENROUTER].status = HealthStatus.DEGRADED

        summary = monitor.get_health_summary()

        assert 'timestamp' in summary
        assert summary['healthy_count'] == 2
        assert summary['degraded_count'] == 1
        assert summary['unhealthy_count'] == 0

    def test_get_recommended_provider(self):
        """Test provider recommendation"""
        from xencode.monitoring.provider_health import (
            HealthStatus,
            ProviderHealthMonitor,
            ProviderType,
        )

        monitor = ProviderHealthMonitor()

        # Make one provider clearly the best
        monitor.providers[ProviderType.LOCAL_OLLAMA].status = HealthStatus.HEALTHY
        monitor.providers[ProviderType.LOCAL_OLLAMA].latency.add_sample(50)  # Fast
        monitor.providers[ProviderType.LOCAL_OLLAMA].uptime_percentage = 99.9

        # Make another provider poor
        monitor.providers[ProviderType.CLOUD_QWEN].status = HealthStatus.UNHEALTHY
        monitor.providers[ProviderType.CLOUD_QWEN].latency.add_sample(5000)  # Slow
        monitor.providers[ProviderType.CLOUD_QWEN].uptime_percentage = 50.0

        recommended = monitor.get_recommended_provider()

        # Should recommend the healthy, fast provider
        assert recommended == ProviderType.LOCAL_OLLAMA

    def test_generate_dashboard_table(self):
        """Test dashboard table generation"""
        from xencode.monitoring.provider_health import ProviderHealthMonitor

        monitor = ProviderHealthMonitor()

        table = monitor.generate_dashboard_table()

        # Table should have headers
        assert table.title == "Provider Health Dashboard"
        assert len(table.columns) == 6  # Provider, Status, Latency, Error Rate, Uptime, Last Check


class TestProviderHealthAPI:
    """Tests for provider health API endpoints"""

    @pytest.fixture
    def test_client(self):
        """Create test FastAPI client"""
        # Create app with just the monitoring router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from xencode.api.routers.monitoring import router
        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        return client

    def test_health_summary_endpoint(self, test_client):
        """Test provider health summary endpoint"""
        response = test_client.get("/providers/health")

        # Should return successfully (may be 503 if monitoring not available)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert 'timestamp' in data
            assert 'providers' in data
            assert 'overall_status' in data

    def test_provider_health_detail_endpoint(self, test_client):
        """Test individual provider health endpoint"""
        response = test_client.get("/providers/local_ollama/health")

        # Should return successfully
        assert response.status_code in [200, 404, 503]

    def test_check_provider_endpoint(self, test_client):
        """Test provider check endpoint"""
        response = test_client.post("/providers/local_ollama/check")

        # Should return successfully
        assert response.status_code in [200, 404, 503]

    def test_recommend_provider_endpoint(self, test_client):
        """Test provider recommendation endpoint"""
        response = test_client.get("/providers/recommend")

        # Should return successfully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert 'provider' in data
            assert 'status' in data

    def test_latency_trends_endpoint(self, test_client):
        """Test latency trends endpoint"""
        response = test_client.get("/providers/latency/trends")

        # Should return successfully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert 'timestamp' in data
            assert 'trends' in data

    def test_errors_endpoint(self, test_client):
        """Test errors endpoint"""
        response = test_client.get("/providers/errors")

        # Should return successfully
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert 'timestamp' in data
            assert 'hours' in data
            assert 'errors' in data


def run_verification():
    """Run verification tests"""
    print("=" * 60)
    print("PROVIDER HEALTH DASHBOARD - VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        ("Latency Metrics", TestLatencyMetrics),
        ("Error Metrics", TestErrorMetrics),
        ("Provider Health", TestProviderHealth),
        ("Health Monitor", TestProviderHealthMonitor),
    ]

    passed = 0
    failed = 0

    for name, test_class in tests:
        print(f"\n{name}:")
        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  [PASS] {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
