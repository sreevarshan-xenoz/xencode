#!/usr/bin/env python3
"""
Tests for Plugin Management System

Tests the plugin management API endpoints, marketplace integration,
plugin execution, and monitoring features.
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from io import BytesIO

from fastapi.testclient import TestClient
from fastapi import UploadFile

# Import the components to test
from xencode.api.routers.plugin import router


class TestPluginAPI:
    """Test plugin API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(router, prefix="/plugins")
        self.client = TestClient(self.app)
        
        # Mock plugin data
        self.mock_plugin = {
            "id": "test-plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A test plugin",
            "license": "MIT",
            "permissions": ["read", "write"],
            "dependencies": [],
            "status": "installed",
            "installed": True,
            "enabled": True,
            "installed_at": datetime.now().isoformat()
        }
    
    def test_list_plugins(self):
        """Test plugin listing"""
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.get("/plugins/")
            
            # Should return 200 with plugin list (mock implementation)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    def test_list_plugins_with_filters(self):
        """Test plugin listing with filters"""
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.get("/plugins/?status=enabled&category=utilities")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_plugin(self):
        """Test getting specific plugin"""
        plugin_id = "file-operations"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.get(f"/plugins/{plugin_id}")
            
            # Should return 200 with plugin data (mock implementation)
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == plugin_id
    
    def test_get_plugin_not_found(self):
        """Test getting non-existent plugin"""
        plugin_id = "non-existent"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.get(f"/plugins/{plugin_id}")
            
            assert response.status_code == 404
    
    def test_install_plugin(self):
        """Test plugin installation"""
        install_data = {
            "plugin_id": "new-plugin",
            "version": "1.0.0",
            "source": "marketplace",
            "verify_signature": True,
            "auto_enable": True
        }
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.post("/plugins/install", json=install_data)
            
            # Should return 200 with installation started message
            assert response.status_code == 200
            data = response.json()
            assert "installation_id" in data
            assert data["status"] == "installing"
    
    def test_upload_plugin(self):
        """Test plugin upload"""
        # Create mock file content
        file_content = b"mock plugin file content"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_result = Mock()
            mock_result.name = "uploaded-plugin"
            mock_result.version = "1.0.0"
            mock_manager.return_value.install_from_file.return_value = mock_result
            
            response = self.client.post(
                "/plugins/upload",
                files={"file": ("test-plugin.zip", file_content, "application/zip")},
                data={"auto_enable": "true", "verify_signature": "false"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["plugin_id"] == "uploaded-plugin"
    
    def test_upload_invalid_file(self):
        """Test uploading invalid file"""
        file_content = b"invalid file content"
        
        response = self.client.post(
            "/plugins/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
            data={"auto_enable": "true"}
        )
        
        assert response.status_code == 400
    
    def test_update_plugin(self):
        """Test plugin update"""
        plugin_id = "test-plugin"
        update_data = {
            "version": "2.0.0",
            "auto_restart": True
        }
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_plugin = Mock()
            mock_plugin.metadata.version = "1.0.0"
            mock_manager.return_value.get_plugin.return_value = mock_plugin
            
            response = self.client.put(f"/plugins/{plugin_id}", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "update_id" in data
            assert data["current_version"] == "1.0.0"
    
    def test_enable_plugin(self):
        """Test plugin enabling"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.enable_plugin.return_value = True
            
            response = self.client.post(f"/plugins/{plugin_id}/enable")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "enabled"
    
    def test_disable_plugin(self):
        """Test plugin disabling"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.disable_plugin.return_value = True
            
            response = self.client.post(f"/plugins/{plugin_id}/disable")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"
    
    def test_uninstall_plugin(self):
        """Test plugin uninstallation"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.uninstall_plugin.return_value = True
            
            response = self.client.delete(f"/plugins/{plugin_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "uninstalled"
    
    def test_execute_plugin(self):
        """Test plugin execution"""
        plugin_id = "test-plugin"
        execute_data = {
            "method": "test_method",
            "args": ["arg1", "arg2"],
            "kwargs": {"param": "value"},
            "timeout_seconds": 30,
            "async_execution": False
        }
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.execute_plugin.return_value = "test result"
            
            response = self.client.post(f"/plugins/{plugin_id}/execute", json=execute_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["result"] == "test result"
    
    def test_execute_plugin_error(self):
        """Test plugin execution with error"""
        plugin_id = "test-plugin"
        execute_data = {
            "method": "failing_method",
            "args": [],
            "kwargs": {},
            "timeout_seconds": 30
        }
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.execute_plugin.side_effect = Exception("Test error")
            
            response = self.client.post(f"/plugins/{plugin_id}/execute", json=execute_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == False
            assert "Test error" in data["error"]
    
    def test_get_plugin_config(self):
        """Test getting plugin configuration"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_config = {"setting1": "value1", "setting2": 42}
            mock_manager.return_value.get_plugin_config.return_value = mock_config
            
            response = self.client.get(f"/plugins/{plugin_id}/config")
            
            assert response.status_code == 200
            data = response.json()
            assert data["plugin_id"] == plugin_id
            assert data["config"] == mock_config
    
    def test_update_plugin_config(self):
        """Test updating plugin configuration"""
        plugin_id = "test-plugin"
        config_data = {
            "config": {"new_setting": "new_value"},
            "restart_required": False
        }
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.update_plugin_config.return_value = True
            
            response = self.client.put(f"/plugins/{plugin_id}/config", json=config_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["plugin_id"] == plugin_id
    
    def test_get_plugin_stats(self):
        """Test getting plugin statistics"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_stats = {
                "total_executions": 100,
                "successful_executions": 95,
                "failed_executions": 5,
                "avg_execution_time_ms": 50.0,
                "total_memory_mb": 128.0,
                "uptime_hours": 24.0,
                "last_24h_executions": 50,
                "error_rate_percent": 5.0
            }
            mock_manager.return_value.get_plugin_stats.return_value = mock_stats
            
            response = self.client.get(f"/plugins/{plugin_id}/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["plugin_id"] == plugin_id
            assert data["total_executions"] == 100
    
    def test_get_plugin_logs(self):
        """Test getting plugin logs"""
        plugin_id = "test-plugin"
        
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_logs = [
                "2023-01-01 10:00:00 INFO Plugin started",
                "2023-01-01 10:01:00 DEBUG Processing request",
                "2023-01-01 10:02:00 INFO Request completed"
            ]
            mock_manager.return_value.get_plugin_logs.return_value = mock_logs
            
            response = self.client.get(f"/plugins/{plugin_id}/logs?lines=100&level=INFO")
            
            assert response.status_code == 200
            data = response.json()
            assert data["plugin_id"] == plugin_id
            assert len(data["logs"]) == 3


class TestMarketplaceAPI:
    """Test marketplace API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(router, prefix="/plugins")
        self.client = TestClient(self.app)
    
    def test_get_marketplace_info(self):
        """Test getting marketplace information"""
        with patch('xencode.api.routers.plugin.get_marketplace_client') as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            
            mock_info = {
                "total_plugins": 1000,
                "categories": ["Development", "Utilities"],
                "featured_plugins": ["plugin1", "plugin2"],
                "recent_updates": [],
                "status": "online"
            }
            mock_client.return_value.__aenter__.return_value.get_marketplace_info.return_value = mock_info
            
            response = self.client.get("/plugins/marketplace/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_plugins"] == 1000
            assert data["marketplace_status"] == "online"
    
    def test_search_marketplace(self):
        """Test marketplace search"""
        search_data = {
            "query": "file operations",
            "category": "utilities",
            "tags": ["filesystem"],
            "sort_by": "downloads",
            "limit": 10,
            "offset": 0
        }
        
        with patch('xencode.api.routers.plugin.get_marketplace_client') as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            
            mock_results = [
                {
                    "id": "file-ops",
                    "name": "File Operations",
                    "version": "1.0.0",
                    "author": "Developer",
                    "description": "File operations plugin",
                    "category": "utilities",
                    "tags": ["filesystem"],
                    "downloads": 1000,
                    "rating": 4.5,
                    "reviews_count": 50,
                    "url": "https://marketplace.xencode.dev/plugins/file-ops"
                }
            ]
            mock_client.return_value.__aenter__.return_value.search_plugins.return_value = mock_results
            
            response = self.client.post("/plugins/marketplace/search", json=search_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["plugins"]) == 1
            assert data["plugins"][0]["name"] == "File Operations"
    
    def test_get_marketplace_categories(self):
        """Test getting marketplace categories"""
        with patch('xencode.api.routers.plugin.get_marketplace_client') as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock()
            mock_client.return_value.__aexit__ = AsyncMock()
            
            mock_categories = ["Development", "Utilities", "Security", "AI/ML"]
            mock_client.return_value.__aenter__.return_value.get_categories.return_value = mock_categories
            
            response = self.client.get("/plugins/marketplace/categories")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["categories"]) == 4
            assert "Development" in data["categories"]


class TestPluginSystemStatus:
    """Test plugin system status endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(router, prefix="/plugins")
        self.client = TestClient(self.app)
    
    def test_get_system_status(self):
        """Test getting plugin system status"""
        with patch('xencode.api.routers.plugin.get_plugin_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_status = {
                "total_plugins": 10,
                "enabled_plugins": 8,
                "disabled_plugins": 2,
                "failed_plugins": 0,
                "memory_usage_mb": 256.0,
                "uptime_hours": 48.0
            }
            mock_manager.return_value.get_system_status.return_value = mock_status
            
            response = self.client.get("/plugins/system/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["total_plugins"] == 10
            assert data["enabled_plugins"] == 8
    
    def test_system_status_unavailable(self):
        """Test system status when components unavailable"""
        with patch('xencode.api.routers.plugin.PLUGIN_COMPONENTS_AVAILABLE', False):
            response = self.client.get("/plugins/system/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unavailable"


def run_tests():
    """Run all plugin management tests"""
    print("🧪 Running Plugin Management Tests")
    print("=" * 50)
    
    # Run pytest
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_plugin_management.py", 
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
        print("\n✅ All plugin management tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")