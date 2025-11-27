"""Tests for workspace manager."""

import pytest
import tempfile
from pathlib import Path

from xencode.collaboration.workspace_manager import WorkspaceManager
from xencode.collaboration.database import CollaborationDatabase
from xencode.collaboration.models import User, Role


class TestWorkspaceManager:
    """Test cases for WorkspaceManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db = CollaborationDatabase(self.temp_db.name)
        self.manager = WorkspaceManager(self.db)
        
        # Create a test user
        user = User(id="user1", username="testuser")
        self.db.create_user(user)

    def teardown_method(self):
        """Clean up test environment."""
        del self.manager
        del self.db
        self.temp_db.close()
        try:
            Path(self.temp_db.name).unlink()
        except PermissionError:
            pass

    def test_create_workspace(self):
        """Test creating a workspace."""
        workspace = self.manager.create_workspace("Test Workspace", "user1")
        
        assert workspace.name == "Test Workspace"
        assert workspace.created_by == "user1"

    def test_creator_is_admin(self):
        """Test that workspace creator becomes admin."""
        workspace = self.manager.create_workspace("Test Workspace", "user1")
        
        role = self.manager.get_member_role(workspace.id, "user1")
        assert role == Role.ADMIN

    def test_add_member(self):
        """Test adding a member to a workspace."""
        workspace = self.manager.create_workspace("Test Workspace", "user1")
        
        user2 = User(id="user2", username="testuser2")
        self.db.create_user(user2)
        
        member = self.manager.add_member(workspace.id, "user2", Role.EDITOR)
        
        assert member.user_id == "user2"
        assert member.role == Role.EDITOR

    def test_is_member(self):
        """Test checking workspace membership."""
        workspace = self.manager.create_workspace("Test Workspace", "user1")
        
        assert self.manager.is_member(workspace.id, "user1") is True
        assert self.manager.is_member(workspace.id, "unknown_user") is False

    def test_is_admin(self):
        """Test checking admin status."""
        workspace = self.manager.create_workspace("Test Workspace", "user1")
        
        assert self.manager.is_admin(workspace.id, "user1") is True
        
        user2 = User(id="user2", username="testuser2")
        self.db.create_user(user2)
        self.manager.add_member(workspace.id, "user2", Role.VIEWER)
        
        assert self.manager.is_admin(workspace.id, "user2") is False

    def test_list_user_workspaces(self):
        """Test listing user's workspaces."""
        workspace1 = self.manager.create_workspace("Workspace 1", "user1")
        workspace2 = self.manager.create_workspace("Workspace 2", "user1")
        
        workspaces = self.manager.list_user_workspaces("user1")
        
        assert len(workspaces) == 2
        workspace_names = [w.name for w in workspaces]
        assert "Workspace 1" in workspace_names
        assert "Workspace 2" in workspace_names
