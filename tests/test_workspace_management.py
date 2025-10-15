#!/usr/bin/env python3
"""
Tests for Workspace Management System

Tests the workspace management API endpoints, CRDT synchronization,
and real-time collaboration features.
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# Import the components to test
from xencode.api.routers.workspace import router, ConnectionManager
from xencode.models.workspace import (
    Workspace, WorkspaceConfig, WorkspaceFile, WorkspaceCollaborator,
    Change, ChangeType, WorkspaceType, WorkspaceStatus, CollaborationMode
)


class TestWorkspaceAPI:
    """Test workspace API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.app.include_router(router, prefix="/workspaces")
        self.client = TestClient(self.app)
        
        # Mock workspace data
        self.mock_workspace = Workspace(
            id="test-workspace-123",
            name="Test Workspace",
            description="A test workspace",
            owner_id="user-123",
            workspace_type=WorkspaceType.PROJECT,
            status=WorkspaceStatus.ACTIVE,
            collaboration_mode=CollaborationMode.SHARED
        )
    
    def test_create_workspace(self):
        """Test workspace creation"""
        workspace_data = {
            "name": "New Workspace",
            "description": "A new test workspace",
            "settings": {"auto_save_enabled": True},
            "collaborators": ["user1", "user2"],
            "crdt_enabled": True
        }
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            
            response = self.client.post("/workspaces/", json=workspace_data)
            
            # Should return 200 with workspace data
            assert response.status_code == 200
            data = response.json()
            assert data["config"]["name"] == workspace_data["name"]
            assert data["config"]["crdt_enabled"] == True
    
    def test_list_workspaces(self):
        """Test workspace listing"""
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.list_workspaces.return_value = [self.mock_workspace]
            
            response = self.client.get("/workspaces/")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_workspace(self):
        """Test getting specific workspace"""
        workspace_id = "test-workspace-123"
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.get_workspace.return_value = self.mock_workspace
            
            response = self.client.get(f"/workspaces/{workspace_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == workspace_id
    
    def test_get_workspace_not_found(self):
        """Test getting non-existent workspace"""
        workspace_id = "non-existent"
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.get_workspace.return_value = None
            
            response = self.client.get(f"/workspaces/{workspace_id}")
            
            assert response.status_code == 404
    
    def test_update_workspace(self):
        """Test workspace update"""
        workspace_id = "test-workspace-123"
        update_data = {
            "name": "Updated Workspace",
            "description": "Updated description",
            "settings": {"auto_save_enabled": False}
        }
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.get_workspace.return_value = self.mock_workspace
            mock_manager.return_value.update_workspace.return_value = self.mock_workspace
            
            response = self.client.put(f"/workspaces/{workspace_id}", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == workspace_id
    
    def test_delete_workspace(self):
        """Test workspace deletion"""
        workspace_id = "test-workspace-123"
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.delete_workspace.return_value = True
            
            response = self.client.delete(f"/workspaces/{workspace_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
    
    def test_sync_workspace_changes(self):
        """Test CRDT synchronization"""
        workspace_id = "test-workspace-123"
        sync_data = {
            "changes": [
                {
                    "id": "change-123",
                    "operation": "insert",
                    "path": "/test.py",
                    "content": "print('hello')",
                    "timestamp": datetime.now().isoformat(),
                    "author": "user1",
                    "vector_clock": {"user1": 1}
                }
            ],
            "crdt_vector": {"user1": 1},
            "session_id": "session-456"
        }
        
        with patch('xencode.api.routers.workspace.get_workspace_manager') as mock_manager:
            mock_manager.return_value = AsyncMock()
            mock_manager.return_value.get_workspace.return_value = self.mock_workspace
            
            # Mock sync result
            sync_result = Mock()
            sync_result.conflicts_resolved = 0
            sync_result.new_vector = {"user1": 1}
            mock_manager.return_value.sync_changes.return_value = sync_result
            
            response = self.client.post(f"/workspaces/{workspace_id}/sync", json=sync_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["conflicts_resolved"] == 0
    
    def test_get_collaboration_status(self):
        """Test collaboration status endpoint"""
        workspace_id = "test-workspace-123"
        
        response = self.client.get(f"/workspaces/{workspace_id}/collaboration")
        
        assert response.status_code == 200
        data = response.json()
        assert "workspace_id" in data
        assert "active_sessions" in data
        assert "collaborators" in data
    
    def test_export_workspace(self):
        """Test workspace export"""
        workspace_id = "test-workspace-123"
        
        response = self.client.get(f"/workspaces/{workspace_id}/export")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestConnectionManager:
    """Test WebSocket connection manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.connection_manager = ConnectionManager()
        self.mock_websocket = Mock(spec=WebSocket)
        self.mock_websocket.accept = AsyncMock()
        self.mock_websocket.send_text = AsyncMock()
        self.mock_websocket.close = AsyncMock()
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test WebSocket connection"""
        workspace_id = "test-workspace"
        user_id = "user123"
        session_id = "session456"
        
        await self.connection_manager.connect(
            self.mock_websocket, workspace_id, user_id, session_id
        )
        
        # Check connection was established
        assert workspace_id in self.connection_manager.active_connections
        assert self.mock_websocket in self.connection_manager.active_connections[workspace_id]
        assert self.mock_websocket in self.connection_manager.session_info
        
        # Verify accept was called
        self.mock_websocket.accept.assert_called_once()
    
    def test_disconnect_websocket(self):
        """Test WebSocket disconnection"""
        workspace_id = "test-workspace"
        user_id = "user123"
        session_id = "session456"
        
        # Manually add connection
        self.connection_manager.active_connections[workspace_id] = {self.mock_websocket}
        self.connection_manager.session_info[self.mock_websocket] = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "session_id": session_id
        }
        
        # Disconnect
        self.connection_manager.disconnect(self.mock_websocket)
        
        # Check connection was removed
        assert workspace_id not in self.connection_manager.active_connections
        assert self.mock_websocket not in self.connection_manager.session_info
    
    @pytest.mark.asyncio
    async def test_broadcast_to_workspace(self):
        """Test broadcasting messages to workspace"""
        workspace_id = "test-workspace"
        user_id = "user123"
        session_id = "session456"
        
        # Add connection
        self.connection_manager.active_connections[workspace_id] = {self.mock_websocket}
        self.connection_manager.session_info[self.mock_websocket] = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "session_id": session_id
        }
        
        message = {
            "type": "test_message",
            "content": "Hello, workspace!"
        }
        
        await self.connection_manager.broadcast_to_workspace(workspace_id, message)
        
        # Verify message was sent
        self.mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(self.mock_websocket.send_text.call_args[0][0])
        assert sent_message["type"] == "test_message"
    
    def test_get_workspace_sessions(self):
        """Test getting workspace sessions"""
        workspace_id = "test-workspace"
        user_id = "user123"
        session_id = "session456"
        
        # Add connection
        self.connection_manager.active_connections[workspace_id] = {self.mock_websocket}
        self.connection_manager.session_info[self.mock_websocket] = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "session_id": session_id,
            "connected_at": datetime.now()
        }
        
        sessions = self.connection_manager.get_workspace_sessions(workspace_id)
        
        assert len(sessions) == 1
        assert sessions[0]["user_id"] == user_id
        assert sessions[0]["session_id"] == session_id


class TestWorkspaceModels:
    """Test workspace data models"""
    
    def test_workspace_creation(self):
        """Test workspace model creation"""
        workspace = Workspace(
            name="Test Workspace",
            description="A test workspace",
            owner_id="user123",
            workspace_type=WorkspaceType.PROJECT
        )
        
        assert workspace.name == "Test Workspace"
        assert workspace.owner_id == "user123"
        assert workspace.workspace_type == WorkspaceType.PROJECT
        assert workspace.status == WorkspaceStatus.ACTIVE
        assert workspace.file_count == 0
        assert workspace.total_size_bytes == 0
    
    def test_workspace_file_operations(self):
        """Test workspace file operations"""
        workspace = Workspace(name="Test", owner_id="user123")
        
        # Add file
        file = WorkspaceFile(
            name="test.py",
            path="/test.py",
            content="print('hello')",
            size_bytes=14
        )
        
        assert workspace.add_file(file) == True
        assert workspace.file_count == 1
        assert workspace.total_size_bytes == 14
        
        # Get file
        retrieved_file = workspace.get_file(file.id)
        assert retrieved_file is not None
        assert retrieved_file.name == "test.py"
        
        # Remove file
        assert workspace.remove_file(file.id) == True
        assert workspace.file_count == 0
        assert workspace.total_size_bytes == 0
    
    def test_workspace_collaborator_operations(self):
        """Test workspace collaborator operations"""
        workspace = Workspace(name="Test", owner_id="user123")
        
        # Add collaborator
        collaborator = WorkspaceCollaborator(
            user_id="user456",
            username="testuser",
            role="editor"
        )
        
        assert workspace.add_collaborator(collaborator) == True
        assert len(workspace.collaborators) == 1
        
        # Get collaborator
        retrieved_collab = workspace.get_collaborator("user456")
        assert retrieved_collab is not None
        assert retrieved_collab.username == "testuser"
        
        # Update activity
        assert workspace.update_collaborator_activity("user456", "session123") == True
        assert retrieved_collab.is_active == True
        assert retrieved_collab.session_id == "session123"
        
        # Remove collaborator
        assert workspace.remove_collaborator("user456") == True
        assert len(workspace.collaborators) == 0
    
    def test_workspace_change_tracking(self):
        """Test workspace change tracking"""
        workspace = Workspace(name="Test", owner_id="user123")
        
        # Add change
        change = Change(
            workspace_id=workspace.id,
            file_id="file123",
            change_type=ChangeType.INSERT,
            position=0,
            content="print('hello')",
            author_id="user123"
        )
        
        workspace.add_change(change)
        
        assert len(workspace.changes) == 1
        assert workspace.vector_clock["user123"] == 1
    
    def test_workspace_permissions(self):
        """Test workspace permission checks"""
        workspace = Workspace(name="Test", owner_id="user123")
        
        # Owner can access and edit
        assert workspace.can_user_access("user123") == True
        assert workspace.can_user_edit("user123") == True
        
        # Non-collaborator cannot access
        assert workspace.can_user_access("user456") == False
        assert workspace.can_user_edit("user456") == False
        
        # Add collaborator with editor role
        collaborator = WorkspaceCollaborator(
            user_id="user456",
            role="editor"
        )
        workspace.add_collaborator(collaborator)
        
        # Collaborator can access and edit
        assert workspace.can_user_access("user456") == True
        assert workspace.can_user_edit("user456") == True
        
        # Change to viewer role
        collaborator.role = "viewer"
        
        # Viewer can access but not edit
        assert workspace.can_user_access("user456") == True
        assert workspace.can_user_edit("user456") == False
    
    def test_workspace_serialization(self):
        """Test workspace serialization"""
        workspace = Workspace(
            name="Test Workspace",
            description="A test workspace",
            owner_id="user123"
        )
        
        # Add some content
        file = WorkspaceFile(name="test.py", content="print('hello')")
        workspace.add_file(file)
        
        collaborator = WorkspaceCollaborator(user_id="user456", username="testuser")
        workspace.add_collaborator(collaborator)
        
        # Test serialization
        data = workspace.to_dict(include_content=True)
        
        assert data["name"] == "Test Workspace"
        assert data["owner_id"] == "user123"
        assert "files" in data
        assert "collaborators" in data
        assert len(data["files"]) == 1
        assert len(data["collaborators"]) == 1
        
        # Test deserialization
        restored_workspace = Workspace.from_dict(data)
        
        assert restored_workspace.name == workspace.name
        assert restored_workspace.owner_id == workspace.owner_id
        assert len(restored_workspace.files) == 1
        assert len(restored_workspace.collaborators) == 1


def run_tests():
    """Run all workspace management tests"""
    print("🧪 Running Workspace Management Tests")
    print("=" * 50)
    
    # Run pytest
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_workspace_management.py", 
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
        print("\n✅ All workspace management tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")