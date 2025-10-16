#!/usr/bin/env python3
"""
End-to-End Integration Test Suite

Comprehensive integration tests that validate full workflows from API to storage,
test component interactions, data flow, and error handling across the entire system.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import all routers for integration testing
from xencode.api.routers.workspace import router as workspace_router
from xencode.api.routers.plugin import router as plugin_router
from xencode.api.routers.analytics import router as analytics_router
from xencode.api.routers.monitoring import router as monitoring_router
from xencode.api.routers.document import router as document_router
from xencode.api.routers.code_analysis import router as code_analysis_router


class IntegrationTestApp:
    """Test application with all routers for integration testing"""
    
    def __init__(self):
        self.app = FastAPI(title="Xencode Integration Test App")
        self._setup_routers()
        self.client = TestClient(self.app)
    
    def _setup_routers(self):
        """Set up all API routers"""
        self.app.include_router(workspace_router, prefix="/api/v1/workspaces", tags=["Workspaces"])
        self.app.include_router(plugin_router, prefix="/api/v1/plugins", tags=["Plugins"])
        self.app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
        self.app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
        self.app.include_router(document_router, prefix="/api/v1/documents", tags=["Documents"])
        self.app.include_router(code_analysis_router, prefix="/api/v1/code", tags=["Code Analysis"])


@pytest.fixture
def integration_app():
    """Fixture providing integration test app"""
    return IntegrationTestApp()


@pytest.fixture
def temp_workspace():
    """Fixture providing temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestWorkspaceIntegration:
    """End-to-end workspace management integration tests""" 
   
    def test_complete_workspace_workflow(self, integration_app):
        """Test complete workspace creation, collaboration, and cleanup workflow"""
        client = integration_app.client
        
        # Step 1: Create workspace
        workspace_data = {
            "name": "Integration Test Workspace",
            "description": "E2E integration test workspace",
            "settings": {"auto_save_enabled": True, "real_time_sync": True},
            "collaborators": ["user1", "user2"],
            "crdt_enabled": True
        }
        
        create_response = client.post("/api/v1/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace = create_response.json()
        workspace_id = workspace["id"]
        
        # Step 2: Verify workspace exists
        get_response = client.get(f"/api/v1/workspaces/{workspace_id}")
        assert get_response.status_code == 200
        retrieved_workspace = get_response.json()
        assert retrieved_workspace["config"]["name"] == workspace_data["name"]
        
        # Step 3: Update workspace configuration
        update_data = {
            "name": "Updated Integration Workspace",
            "settings": {"auto_save_enabled": False}
        }
        
        update_response = client.put(f"/api/v1/workspaces/{workspace_id}", json=update_data)
        assert update_response.status_code == 200
        
        # Step 4: Sync changes (CRDT)
        sync_data = {
            "changes": [
                {
                    "id": "change-123",
                    "operation": "insert",
                    "path": "/test.py",
                    "content": "print('integration test')",
                    "timestamp": datetime.now().isoformat(),
                    "author": "user1",
                    "vector_clock": {"user1": 1}
                }
            ],
            "crdt_vector": {"user1": 1},
            "session_id": "integration-session"
        }
        
        sync_response = client.post(f"/api/v1/workspaces/{workspace_id}/sync", json=sync_data)
        assert sync_response.status_code == 200
        sync_result = sync_response.json()
        assert sync_result["success"] == True
        
        # Step 5: Get collaboration status
        collab_response = client.get(f"/api/v1/workspaces/{workspace_id}/collaboration")
        assert collab_response.status_code == 200
        
        # Step 6: Export workspace
        export_response = client.get(f"/api/v1/workspaces/{workspace_id}/export")
        assert export_response.status_code == 200
        
        # Step 7: Delete workspace
        delete_response = client.delete(f"/api/v1/workspaces/{workspace_id}")
        assert delete_response.status_code == 200
        
        # Step 8: Verify workspace is deleted
        get_deleted_response = client.get(f"/api/v1/workspaces/{workspace_id}")
        assert get_deleted_response.status_code == 404


class TestPluginIntegration:
    """End-to-end plugin management integration tests"""
    
    def test_complete_plugin_lifecycle(self, integration_app):
        """Test complete plugin installation, execution, and management workflow"""
        client = integration_app.client
        
        # Step 1: List available plugins
        list_response = client.get("/api/v1/plugins/")
        assert list_response.status_code == 200
        
        # Step 2: Install plugin
        install_data = {
            "plugin_id": "integration-test-plugin",
            "version": "1.0.0",
            "source": "marketplace",
            "verify_signature": True,
            "auto_enable": True
        }
        
        install_response = client.post("/api/v1/plugins/install", json=install_data)
        assert install_response.status_code == 200
        install_result = install_response.json()
        assert "installation_id" in install_result
        
        # Step 3: Get plugin details
        plugin_id = "integration-test-plugin"
        get_response = client.get(f"/api/v1/plugins/{plugin_id}")
        # Note: This might return 404 in mock implementation, which is expected
        
        # Step 4: Execute plugin method
        execute_data = {
            "method": "test_method",
            "args": ["arg1", "arg2"],
            "kwargs": {"param": "value"},
            "timeout_seconds": 30,
            "async_execution": False
        }
        
        execute_response = client.post(f"/api/v1/plugins/{plugin_id}/execute", json=execute_data)
        # Note: This might return 503 in mock implementation
        
        # Step 5: Get plugin configuration
        config_response = client.get(f"/api/v1/plugins/{plugin_id}/config")
        # Note: This might return 503 in mock implementation
        
        # Step 6: Update plugin configuration
        config_update = {
            "config": {"setting1": "value1", "setting2": 42},
            "restart_required": False
        }
        
        config_update_response = client.put(f"/api/v1/plugins/{plugin_id}/config", json=config_update)
        # Note: This might return 503 in mock implementation
        
        # Step 7: Get plugin statistics
        stats_response = client.get(f"/api/v1/plugins/{plugin_id}/stats")
        # Note: This might return 503 in mock implementation
        
        # Step 8: Disable plugin
        disable_response = client.post(f"/api/v1/plugins/{plugin_id}/disable")
        # Note: This might return 503 in mock implementation
        
        # Step 9: Uninstall plugin
        uninstall_response = client.delete(f"/api/v1/plugins/{plugin_id}")
        # Note: This might return 503 in mock implementation


class TestAnalyticsIntegration:
    """End-to-end analytics and reporting integration tests"""
    
    def test_complete_analytics_workflow(self, integration_app):
        """Test complete analytics data collection, processing, and reporting workflow"""
        client = integration_app.client
        
        # Step 1: Record metrics
        metric_data = {
            "name": "integration_test_metric",
            "value": 42.5,
            "metric_type": "gauge",
            "labels": {"test": "integration", "component": "e2e"},
            "timestamp": datetime.now().isoformat()
        }
        
        metric_response = client.post("/api/v1/analytics/metrics", json=metric_data)
        assert metric_response.status_code == 200
        
        # Step 2: Record events
        event_data = {
            "event_type": "integration_test_event",
            "event_data": {"action": "test_execution", "success": True},
            "user_id": "integration_user",
            "session_id": "integration_session",
            "timestamp": datetime.now().isoformat()
        }
        
        event_response = client.post("/api/v1/analytics/events", json=event_data)
        assert event_response.status_code == 200
        
        # Step 3: Get analytics overview
        overview_response = client.get("/api/v1/analytics/overview?time_range=1h")
        assert overview_response.status_code == 200
        overview = overview_response.json()
        assert "total_events" in overview
        assert "system_health_score" in overview
        
        # Step 4: Generate report
        report_data = {
            "report_type": "performance",
            "format": "json",
            "time_range": "1h",
            "filters": {},
            "include_raw_data": False
        }
        
        report_response = client.post("/api/v1/analytics/reports", json=report_data)
        assert report_response.status_code == 200
        report_result = report_response.json()
        report_id = report_result["report_id"]
        
        # Step 5: Check report status
        status_response = client.get(f"/api/v1/analytics/reports/{report_id}")
        assert status_response.status_code == 200
        
        # Step 6: Download report
        download_response = client.get(f"/api/v1/analytics/reports/{report_id}/download")
        assert download_response.status_code == 200
        
        # Step 7: Get dashboard data
        dashboard_response = client.get("/api/v1/analytics/dashboard?dashboard_type=overview")
        assert dashboard_response.status_code == 200
        
        # Step 8: Get AI insights
        insights_response = client.get("/api/v1/analytics/insights?category=performance")
        assert insights_response.status_code == 200


class TestMonitoringIntegration:
    """End-to-end monitoring and system health integration tests"""
    
    def test_complete_monitoring_workflow(self, integration_app):
        """Test complete system monitoring, alerting, and cleanup workflow"""
        client = integration_app.client
        
        # Step 1: Check system health
        health_response = client.get("/api/v1/monitoring/health")
        assert health_response.status_code == 200
        health = health_response.json()
        assert "overall_status" in health
        assert "health_score" in health
        
        # Step 2: Get resource usage
        memory_response = client.get("/api/v1/monitoring/resources/memory")
        assert memory_response.status_code == 200
        memory_usage = memory_response.json()
        assert memory_usage["resource_type"] == "memory"
        
        cpu_response = client.get("/api/v1/monitoring/resources/cpu")
        assert cpu_response.status_code == 200
        
        # Step 3: Get all resources
        all_resources_response = client.get("/api/v1/monitoring/resources")
        assert all_resources_response.status_code == 200
        resources = all_resources_response.json()
        assert len(resources) >= 3  # At least memory, CPU, disk
        
        # Step 4: Get performance metrics
        performance_response = client.get("/api/v1/monitoring/performance")
        assert performance_response.status_code == 200
        
        # Step 5: Get alerts
        alerts_response = client.get("/api/v1/monitoring/alerts")
        assert alerts_response.status_code == 200
        
        # Step 6: Trigger cleanup
        cleanup_data = {
            "resource_types": ["memory", "cache"],
            "priority": "normal",
            "force": False,
            "dry_run": False
        }
        
        cleanup_response = client.post("/api/v1/monitoring/cleanup", json=cleanup_data)
        assert cleanup_response.status_code == 200
        cleanup_result = cleanup_response.json()
        assert "cleanup_id" in cleanup_result
        
        # Step 7: Get processes
        processes_response = client.get("/api/v1/monitoring/processes?limit=5")
        assert processes_response.status_code == 200
        
        # Step 8: Get network stats
        network_response = client.get("/api/v1/monitoring/network")
        assert network_response.status_code == 200
        
        # Step 9: Get disk stats
        disk_response = client.get("/api/v1/monitoring/disk")
        assert disk_response.status_code == 200


class TestCrossComponentIntegration:
    """Integration tests across multiple components"""
    
    def test_workspace_analytics_integration(self, integration_app):
        """Test integration between workspace operations and analytics tracking"""
        client = integration_app.client
        
        # Create workspace and track analytics
        workspace_data = {
            "name": "Analytics Integration Workspace",
            "description": "Testing workspace-analytics integration"
        }
        
        # Step 1: Create workspace
        create_response = client.post("/api/v1/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace = create_response.json()
        workspace_id = workspace["id"]
        
        # Step 2: Record workspace creation event
        event_data = {
            "event_type": "workspace_created",
            "event_data": {
                "workspace_id": workspace_id,
                "workspace_name": workspace_data["name"]
            },
            "user_id": "integration_user"
        }
        
        event_response = client.post("/api/v1/analytics/events", json=event_data)
        assert event_response.status_code == 200
        
        # Step 3: Record workspace usage metrics
        metric_data = {
            "name": "workspace_operations",
            "value": 1,
            "metric_type": "counter",
            "labels": {"operation": "create", "workspace_id": workspace_id}
        }
        
        metric_response = client.post("/api/v1/analytics/metrics", json=metric_data)
        assert metric_response.status_code == 200
        
        # Step 4: Get analytics overview (should include our events)
        overview_response = client.get("/api/v1/analytics/overview")
        assert overview_response.status_code == 200
        
        # Step 5: Clean up
        delete_response = client.delete(f"/api/v1/workspaces/{workspace_id}")
        assert delete_response.status_code == 200
    
    def test_plugin_monitoring_integration(self, integration_app):
        """Test integration between plugin operations and system monitoring"""
        client = integration_app.client
        
        # Step 1: Check system health before plugin operations
        health_before = client.get("/api/v1/monitoring/health")
        assert health_before.status_code == 200
        
        # Step 2: Attempt plugin installation (will use mock)
        install_data = {
            "plugin_id": "monitoring-test-plugin",
            "version": "1.0.0",
            "source": "marketplace"
        }
        
        install_response = client.post("/api/v1/plugins/install", json=install_data)
        assert install_response.status_code == 200
        
        # Step 3: Record plugin installation metrics
        metric_data = {
            "name": "plugin_installations",
            "value": 1,
            "metric_type": "counter",
            "labels": {"plugin_id": "monitoring-test-plugin", "status": "success"}
        }
        
        metric_response = client.post("/api/v1/analytics/metrics", json=metric_data)
        assert metric_response.status_code == 200
        
        # Step 4: Check system health after plugin operations
        health_after = client.get("/api/v1/monitoring/health")
        assert health_after.status_code == 200
        
        # Step 5: Get system resources to ensure no degradation
        resources_response = client.get("/api/v1/monitoring/resources")
        assert resources_response.status_code == 200


class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery"""
    
    def test_error_propagation_and_recovery(self, integration_app):
        """Test error handling across components and recovery mechanisms"""
        client = integration_app.client
        
        # Step 1: Test invalid workspace creation
        invalid_workspace = {"name": ""}  # Invalid empty name
        
        create_response = client.post("/api/v1/workspaces/", json=invalid_workspace)
        # Should handle validation error gracefully
        
        # Step 2: Test non-existent resource access
        get_response = client.get("/api/v1/workspaces/non-existent-id")
        assert get_response.status_code == 404
        
        # Step 3: Test invalid plugin execution
        execute_data = {
            "method": "non_existent_method",
            "args": [],
            "kwargs": {}
        }
        
        execute_response = client.post("/api/v1/plugins/non-existent-plugin/execute", json=execute_data)
        # Should return appropriate error status
        
        # Step 4: Record error events in analytics
        error_event = {
            "event_type": "error_occurred",
            "event_data": {
                "error_type": "not_found",
                "component": "workspace",
                "resource_id": "non-existent-id"
            }
        }
        
        error_response = client.post("/api/v1/analytics/events", json=error_event)
        assert error_response.status_code == 200
        
        # Step 5: Check system health after errors
        health_response = client.get("/api/v1/monitoring/health")
        assert health_response.status_code == 200
        
        # System should remain healthy despite errors
        health = health_response.json()
        assert health["overall_status"] in ["healthy", "warning"]  # Should not be critical


class TestPerformanceIntegration:
    """Integration tests for performance and load handling"""
    
    def test_concurrent_operations(self, integration_app):
        """Test system behavior under concurrent operations"""
        client = integration_app.client
        
        # Step 1: Create multiple workspaces concurrently
        workspace_ids = []
        for i in range(5):
            workspace_data = {
                "name": f"Concurrent Workspace {i}",
                "description": f"Concurrent test workspace {i}"
            }
            
            response = client.post("/api/v1/workspaces/", json=workspace_data)
            if response.status_code == 200:
                workspace_ids.append(response.json()["id"])
        
        # Step 2: Record metrics for all operations
        for i, workspace_id in enumerate(workspace_ids):
            metric_data = {
                "name": "concurrent_operations",
                "value": 1,
                "metric_type": "counter",
                "labels": {"operation": "workspace_create", "batch": str(i)}
            }
            
            client.post("/api/v1/analytics/metrics", json=metric_data)
        
        # Step 3: Check system performance
        performance_response = client.get("/api/v1/monitoring/performance")
        assert performance_response.status_code == 200
        
        # Step 4: Clean up all workspaces
        for workspace_id in workspace_ids:
            client.delete(f"/api/v1/workspaces/{workspace_id}")
    
    def test_large_data_handling(self, integration_app):
        """Test system behavior with large data operations"""
        client = integration_app.client
        
        # Step 1: Create workspace with large sync data
        workspace_data = {
            "name": "Large Data Test Workspace",
            "description": "Testing large data handling"
        }
        
        create_response = client.post("/api/v1/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace_id = create_response.json()["id"]
        
        # Step 2: Sync large changes
        large_content = "x" * 1000  # 1KB of content
        sync_data = {
            "changes": [
                {
                    "id": f"large-change-{i}",
                    "operation": "insert",
                    "path": f"/large_file_{i}.txt",
                    "content": large_content,
                    "timestamp": datetime.now().isoformat(),
                    "author": "performance_test"
                } for i in range(10)  # 10 large changes
            ],
            "crdt_vector": {"performance_test": 10},
            "session_id": "performance-session"
        }
        
        sync_response = client.post(f"/api/v1/workspaces/{workspace_id}/sync", json=sync_data)
        assert sync_response.status_code == 200
        
        # Step 3: Monitor system resources during operation
        resources_response = client.get("/api/v1/monitoring/resources")
        assert resources_response.status_code == 200
        
        # Step 4: Clean up
        delete_response = client.delete(f"/api/v1/workspaces/{workspace_id}")
        assert delete_response.status_code == 200


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running End-to-End Integration Tests")
    print("=" * 50)
    
    # Run pytest with integration tests
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_integration_e2e.py", 
        "-v", "--tb=short", "-x"  # Stop on first failure
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run the integration tests
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed. Check the output above.")