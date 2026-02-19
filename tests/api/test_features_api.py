#!/usr/bin/env python3
"""
Tests for Features API Router

Tests all feature management endpoints including configuration, status, and control.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

# Import the FastAPI app and router
from xencode.api.main import app
from xencode.api.routers import features


# Mock feature manager for testing
class MockFeatureManager:
    """Mock feature manager for testing"""
    
    def __init__(self):
        # Mock feature
        self.mock_feature = Mock()
        self.mock_feature.name = "test_feature"
        self.mock_feature.description = "Test feature description"
        self.mock_feature.version = "1.0.0"
        self.mock_feature.is_enabled = True
        self.mock_feature.is_initialized = True
        self.mock_feature.get_status.return_value = Mock(value="enabled")
        self.mock_feature.get_config.return_value = Mock(
            config={"key": "value"},
            dependencies=[]
        )
        self.mock_feature.get_cli_commands.return_value = []
        self.mock_feature.get_api_endpoints.return_value = []
        self.mock_feature.update_config = Mock()
        self.mock_feature.initialize = AsyncMock(return_value=True)
        
        self.features = {"test_feature": self.mock_feature}
    
    def get_all_features(self):
        return self.features
    
    def get_enabled_features(self):
        return self.features
    
    def get_feature(self, name):
        return self.features.get(name)
    
    async def initialize_feature(self, name, config=None):
        return True
    
    async def shutdown_feature(self, name):
        return name in self.features


@pytest.fixture
def mock_manager():
    """Create mock feature manager"""
    return MockFeatureManager()


@pytest.fixture
def client(mock_manager):
    """Create test client with dependency override"""
    # Override the dependency
    app.dependency_overrides[features.get_feature_manager] = lambda: mock_manager
    
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides.clear()


class TestListFeatures:
    """Tests for listing features"""
    
    def test_list_all_features(self, client):
        """Test listing all features"""
        response = client.get("/api/v1/features/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "features" in data
        assert "total" in data
        assert "timestamp" in data
        assert data["total"] == 1
        assert len(data["features"]) == 1
        
        feature = data["features"][0]
        assert feature["name"] == "test_feature"
        assert feature["enabled"] is True
        assert feature["initialized"] is True
    
    def test_list_enabled_features_only(self, client):
        """Test listing only enabled features"""
        response = client.get("/api/v1/features/?enabled_only=true")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 1


class TestGetFeature:
    """Tests for getting feature details"""
    
    def test_get_existing_feature(self, client):
        """Test getting details of an existing feature"""
        response = client.get("/api/v1/features/test_feature")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test_feature"
        assert data["description"] == "Test feature description"
        assert data["version"] == "1.0.0"
        assert data["enabled"] is True
        assert data["initialized"] is True
        assert "config" in data
        assert "dependencies" in data
    
    def test_get_nonexistent_feature(self, client):
        """Test getting a feature that doesn't exist"""
        response = client.get("/api/v1/features/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data


class TestEnableFeature:
    """Tests for enabling features"""
    
    def test_enable_new_feature(self, client, mock_manager):
        """Test enabling a feature that isn't loaded"""
        # Remove feature from manager
        mock_manager.features.clear()
        
        response = client.post("/api/v1/features/test_feature/enable")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["feature_name"] == "test_feature"
        assert "enabled successfully" in data["message"]
    
    def test_enable_existing_feature(self, client):
        """Test enabling a feature that's already loaded"""
        response = client.post("/api/v1/features/test_feature/enable")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
    
    def test_enable_feature_with_config(self, client, mock_manager):
        """Test enabling a feature with custom configuration"""
        # Remove feature from manager
        mock_manager.features.clear()
        
        config = {
            "name": "test_feature",
            "enabled": True,
            "version": "2.0.0",
            "config": {"custom": "value"},
            "dependencies": ["dep1"]
        }
        
        response = client.post("/api/v1/features/test_feature/enable", json=config)
        
        assert response.status_code == 200


class TestDisableFeature:
    """Tests for disabling features"""
    
    def test_disable_existing_feature(self, client):
        """Test disabling an existing feature"""
        response = client.post("/api/v1/features/test_feature/disable")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["feature_name"] == "test_feature"
        assert "disabled successfully" in data["message"]
    
    def test_disable_nonexistent_feature(self, client, mock_manager):
        """Test disabling a feature that doesn't exist"""
        # Remove feature from manager
        mock_manager.features.clear()
        
        response = client.post("/api/v1/features/nonexistent/disable")
        
        assert response.status_code == 404


class TestUpdateFeatureConfig:
    """Tests for updating feature configuration"""
    
    def test_update_config(self, client, mock_manager):
        """Test updating feature configuration"""
        new_config = {"new_key": "new_value"}
        
        response = client.put(
            "/api/v1/features/test_feature/config",
            json=new_config
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "updated successfully" in data["message"]
        
        # Verify update_config was called
        mock_manager.mock_feature.update_config.assert_called_once_with(new_config)
    
    def test_update_config_nonexistent_feature(self, client, mock_manager):
        """Test updating config for a feature that doesn't exist"""
        # Remove feature from manager
        mock_manager.features.clear()
        
        response = client.put(
            "/api/v1/features/nonexistent/config",
            json={"key": "value"}
        )
        
        assert response.status_code == 404


class TestGetFeatureStatus:
    """Tests for getting feature status"""
    
    def test_get_status(self, client):
        """Test getting feature status"""
        response = client.get("/api/v1/features/test_feature/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test_feature"
        assert data["status"] == "enabled"
        assert data["enabled"] is True
        assert data["initialized"] is True


class TestCollaborativeFeatures:
    """Tests for collaborative features (authenticated endpoints)"""
    
    def test_start_collaboration_without_auth(self, client):
        """Test starting collaboration without authentication"""
        response = client.post(
            "/api/v1/features/collaborative_coding/collaborate/start",
            json={"room_id": "test-room"}
        )
        
        # Should fail without authentication
        assert response.status_code == 401
    
    def test_start_collaboration_with_auth(self, client, mock_manager):
        """Test starting collaboration with authentication"""
        # Mock collaborative coding feature
        collab_feature = Mock()
        collab_feature.name = "collaborative_coding"
        collab_feature.description = "Collaborative coding feature"
        collab_feature.version = "1.0.0"
        collab_feature.is_enabled = True
        collab_feature.is_initialized = True
        collab_feature.get_status.return_value = Mock(value="enabled")
        
        mock_manager.features["collaborative_coding"] = collab_feature
        
        response = client.post(
            "/api/v1/features/collaborative_coding/collaborate/start",
            json={"room_id": "test-room"},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "test-room" in data["message"]
    
    def test_start_collaboration_unsupported_feature(self, client):
        """Test starting collaboration on a feature that doesn't support it"""
        response = client.post(
            "/api/v1/features/test_feature/collaborate/start",
            json={"room_id": "test-room"},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data


class TestFeatureAnalytics:
    """Tests for feature analytics"""
    
    def test_get_analytics(self, client):
        """Test getting feature analytics"""
        response = client.get("/api/v1/features/test_feature/analytics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["feature_name"] == "test_feature"
        assert "usage_count" in data
        assert "error_count" in data
        assert "average_response_time_ms" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
