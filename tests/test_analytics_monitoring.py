#!/usr/bin/env python3
"""
Tests for Analytics and Monitoring System

Tests the analytics and monitoring API endpoints, metrics collection,
system health monitoring, and reporting features.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient

# Import the components to test
from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router


class TestAnalyticsAPI:
    """Test analytics API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(analytics_router, prefix="/analytics")
        self.client = TestClient(self.app)
    
    def test_get_analytics_overview(self):
        """Test analytics overview endpoint"""
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.get("/analytics/overview?time_range=1d")
            
            # Should return 200 with overview data (mock implementation)
            assert response.status_code == 200
            data = response.json()
            assert "total_events" in data
            assert "total_metrics" in data
            assert "system_health_score" in data
    
    def test_record_metric(self):
        """Test metric recording"""
        metric_data = {
            "name": "response_time_ms",
            "value": 45.2,
            "metric_type": "gauge",
            "labels": {"endpoint": "/api/test"},
            "timestamp": datetime.now().isoformat()
        }
        
        with patch('xencode.api.routers.analytics.get_metrics_collector') as mock_collector:
            mock_collector.return_value = AsyncMock()
            
            response = self.client.post("/analytics/metrics", json=metric_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "response_time_ms"
            assert data["value"] == 45.2
    
    def test_get_metrics(self):
        """Test metrics retrieval"""
        with patch('xencode.api.routers.analytics.get_metrics_collector') as mock_collector:
            mock_collector.return_value = AsyncMock()
            
            response = self.client.get("/analytics/metrics?name=response_time_ms&time_range=1h")
            
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert "total_count" in data
    
    def test_record_event(self):
        """Test event recording"""
        event_data = {
            "event_type": "plugin_execution",
            "event_data": {"plugin_id": "test-plugin"},
            "user_id": "user123",
            "session_id": "session456"
        }
        
        with patch('xencode.api.routers.analytics.get_event_tracker') as mock_tracker:
            mock_tracker.return_value = AsyncMock()
            
            response = self.client.post("/analytics/events", json=event_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["event_type"] == "plugin_execution"
            assert "id" in data
    
    def test_get_events(self):
        """Test events retrieval"""
        with patch('xencode.api.routers.analytics.get_event_tracker') as mock_tracker:
            mock_tracker.return_value = AsyncMock()
            
            response = self.client.get("/analytics/events?event_type=plugin_execution&limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert "events" in data
            assert "total_count" in data
    
    def test_generate_report(self):
        """Test report generation"""
        report_data = {
            "report_type": "performance",
            "format": "json",
            "time_range": "1d",
            "filters": {}
        }
        
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.post("/analytics/reports", json=report_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
            assert data["status"] == "generating"
    
    def test_get_report_status(self):
        """Test report status retrieval"""
        report_id = "test-report-123"
        
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.get(f"/analytics/reports/{report_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["report_id"] == report_id
    
    def test_download_report(self):
        """Test report download"""
        report_id = "test-report-123"
        
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.get(f"/analytics/reports/{report_id}/download")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
    
    def test_get_dashboard_data(self):
        """Test dashboard data retrieval"""
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.get("/analytics/dashboard?dashboard_type=overview&time_range=1h")
            
            assert response.status_code == 200
            data = response.json()
            assert "dashboard_type" in data
            assert "data" in data
            assert "last_updated" in data
    
    def test_analytics_health_check(self):
        """Test analytics health check"""
        response = self.client.get("/analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_get_insights(self):
        """Test AI insights retrieval"""
        with patch('xencode.api.routers.analytics.get_analytics_system') as mock_system:
            mock_system.return_value = AsyncMock()
            
            response = self.client.get("/analytics/insights?insight_type=performance&time_range=1d")
            
            assert response.status_code == 200
            data = response.json()
            assert "insight_type" in data
            assert "recommendations" in data
            assert "trends" in data


class TestMonitoringAPI:
    """Test monitoring API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(monitoring_router, prefix="/monitoring")
        self.client = TestClient(self.app)
    
    def test_get_system_health(self):
        """Test system health endpoint"""
        response = self.client.get("/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "health_score" in data
        assert "uptime_seconds" in data
        assert "memory_usage_percent" in data
        assert "cpu_usage_percent" in data
    
    def test_get_resource_usage_memory(self):
        """Test memory resource usage"""
        response = self.client.get("/monitoring/resources/memory")
        
        assert response.status_code == 200
        data = response.json()
        assert data["resource_type"] == "memory"
        assert "current_usage" in data
        assert "utilization_percent" in data
        assert data["unit"] == "GB"
    
    def test_get_resource_usage_cpu(self):
        """Test CPU resource usage"""
        response = self.client.get("/monitoring/resources/cpu")
        
        assert response.status_code == 200
        data = response.json()
        assert data["resource_type"] == "cpu"
        assert "current_usage" in data
        assert "utilization_percent" in data
        assert data["unit"] == "percent"
    
    def test_get_resource_usage_disk(self):
        """Test disk resource usage"""
        response = self.client.get("/monitoring/resources/disk")
        
        assert response.status_code == 200
        data = response.json()
        assert data["resource_type"] == "disk"
        assert "current_usage" in data
        assert "utilization_percent" in data
        assert data["unit"] == "GB"
    
    def test_get_all_resources(self):
        """Test all resources endpoint"""
        response = self.client.get("/monitoring/resources")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3  # At least memory, CPU, disk
        
        resource_types = [resource["resource_type"] for resource in data]
        assert "memory" in resource_types
        assert "cpu" in resource_types
        assert "disk" in resource_types
    
    def test_get_performance_metrics(self):
        """Test performance metrics"""
        with patch('xencode.api.routers.monitoring.get_performance_optimizer') as mock_optimizer:
            mock_optimizer.return_value = AsyncMock()
            
            response = self.client.get("/monitoring/performance")
            
            assert response.status_code == 200
            data = response.json()
            assert "response_time_ms" in data
            assert "throughput_rps" in data
            assert "error_rate_percent" in data
            assert "cache_hit_rate_percent" in data
    
    def test_trigger_cleanup(self):
        """Test cleanup trigger"""
        cleanup_data = {
            "resource_types": ["memory", "cache"],
            "priority": "normal",
            "force": False,
            "dry_run": False
        }
        
        response = self.client.post("/monitoring/cleanup", json=cleanup_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "cleanup_id" in data
        assert "tasks_executed" in data
        assert "memory_freed_mb" in data
    
    def test_get_alerts(self):
        """Test alerts retrieval"""
        response = self.client.get("/monitoring/alerts?severity=high&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Alerts depend on current system state, so we just check structure
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment"""
        alert_id = "test-alert-123"
        
        response = self.client.post(f"/monitoring/alerts/{alert_id}/acknowledge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["alert_id"] == alert_id
        assert data["acknowledged"] == True
    
    def test_get_processes(self):
        """Test processes retrieval"""
        response = self.client.get("/monitoring/processes?limit=10&sort_by=memory")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are processes
            process = data[0]
            assert "pid" in process
            assert "name" in process
            assert "memory_mb" in process
            assert "cpu_percent" in process
    
    def test_get_network_stats(self):
        """Test network statistics"""
        response = self.client.get("/monitoring/network")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are network interfaces
            interface = data[0]
            assert "interface" in interface
            assert "bytes_sent" in interface
            assert "bytes_recv" in interface
    
    def test_get_disk_stats(self):
        """Test disk statistics"""
        response = self.client.get("/monitoring/disk")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are disk partitions
            disk = data[0]
            assert "device" in disk
            assert "total_gb" in disk
            assert "used_gb" in disk
            assert "usage_percent" in disk
    
    def test_get_monitoring_dashboard(self):
        """Test monitoring dashboard"""
        with patch('xencode.api.routers.monitoring.get_monitoring_dashboard') as mock_dashboard:
            mock_dashboard.return_value = AsyncMock()
            
            response = self.client.get("/monitoring/dashboard")
            
            assert response.status_code == 200
            data = response.json()
            assert "dashboard_data" in data
            assert "last_updated" in data
            assert "refresh_interval" in data
    
    def test_update_monitoring_config(self):
        """Test monitoring configuration update"""
        config_data = {
            "resource_type": "memory",
            "interval_seconds": 60,
            "alert_threshold": 85.0,
            "enabled": True,
            "retention_days": 30
        }
        
        response = self.client.post("/monitoring/config", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["resource_type"] == "memory"
        assert data["alert_threshold"] == 85.0


class TestIntegratedAnalyticsMonitoring:
    """Test integrated analytics and monitoring functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(analytics_router, prefix="/analytics")
        self.app.include_router(monitoring_router, prefix="/monitoring")
        self.client = TestClient(self.app)
    
    def test_health_checks_consistency(self):
        """Test that both health checks are consistent"""
        analytics_health = self.client.get("/analytics/health")
        monitoring_health = self.client.get("/monitoring/health")
        
        assert analytics_health.status_code == 200
        assert monitoring_health.status_code == 200
        
        analytics_data = analytics_health.json()
        monitoring_data = monitoring_health.json()
        
        # Both should report system status
        assert "status" in analytics_data
        assert "overall_status" in monitoring_data
    
    def test_metrics_and_monitoring_correlation(self):
        """Test correlation between metrics and monitoring data"""
        # Record a performance metric
        metric_data = {
            "name": "cpu_usage_percent",
            "value": 75.5,
            "metric_type": "gauge",
            "labels": {"source": "system"}
        }
        
        with patch('xencode.api.routers.analytics.get_metrics_collector') as mock_collector:
            mock_collector.return_value = AsyncMock()
            
            metrics_response = self.client.post("/analytics/metrics", json=metric_data)
            assert metrics_response.status_code == 200
        
        # Get monitoring data for CPU
        monitoring_response = self.client.get("/monitoring/resources/cpu")
        assert monitoring_response.status_code == 200
        
        monitoring_data = monitoring_response.json()
        assert "cpu_usage_percent" in monitoring_data or "current_usage" in monitoring_data
    
    def test_alert_and_event_integration(self):
        """Test integration between alerts and events"""
        # Get current alerts
        alerts_response = self.client.get("/monitoring/alerts")
        assert alerts_response.status_code == 200
        
        # Record an event for high resource usage
        event_data = {
            "event_type": "resource_alert",
            "event_data": {
                "resource_type": "memory",
                "usage_percent": 85.0,
                "threshold": 80.0
            }
        }
        
        with patch('xencode.api.routers.analytics.get_event_tracker') as mock_tracker:
            mock_tracker.return_value = AsyncMock()
            
            event_response = self.client.post("/analytics/events", json=event_data)
            assert event_response.status_code == 200


def run_tests():
    """Run all analytics and monitoring tests"""
    print("🧪 Running Analytics & Monitoring Tests")
    print("=" * 50)
    
    # Run pytest
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_analytics_monitoring.py", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    
    if success:
        print("\n✅ All analytics and monitoring tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")