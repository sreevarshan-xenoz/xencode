"""Tests for RBAC."""

import pytest
import tempfile
from pathlib import Path

from xencode.collaboration.rbac import RBAC
from xencode.collaboration.workspace_manager import WorkspaceManager
from xencode.collaboration.database import CollaborationDatabase
from xencode.collaboration.models import User, Role, Permission


class TestRBAC:
    """Test cases for RBAC."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db = CollaborationDatabase(self.temp_db.name)
        self.rbac = RBAC(self.db)
        self.workspace_manager = WorkspaceManager(self.db)
        
        # Create test users
        user1 = User(id="admin_user", username="admin")
        user2 = User(id="editor_user", username="editor")
        user3 = User(id="viewer_user", username="viewer")
        
        self.db.create_user(user1)
        self.db.create_user(user2)
        self.db.create_user(user3)
        
        # Create workspace
        self.workspace = self.workspace_manager.create_workspace("Test Workspace", "admin_user")
        self.workspace_manager.add_member(self.workspace.id, "editor_user", Role.EDITOR)
        self.workspace_manager.add_member(self.workspace.id, "viewer_user", Role.VIEWER)

    def teardown_method(self):
        """Clean up test environment."""
        del self.rbac
        del self.workspace_manager
        del self.db
        self.temp_db.close()
        try:
            Path(self.temp_db.name).unlink()
        except PermissionError:
            pass

    def test_admin_permissions(self):
        """Test admin has all permissions."""
        assert self.rbac.can_read(self.workspace.id, "admin_user")
        assert self.rbac.can_write(self.workspace.id, "admin_user")
        assert self.rbac.can_delete(self.workspace.id, "admin_user")
        assert self.rbac.can_share(self.workspace.id, "admin_user")
        assert self.rbac.is_admin(self.workspace.id, "admin_user")

    def test_editor_permissions(self):
        """Test editor permissions."""
        assert self.rbac.can_read(self.workspace.id, "editor_user")
        assert self.rbac.can_write(self.workspace.id, "editor_user")
        assert not self.rbac.can_delete(self.workspace.id, "editor_user")
        assert self.rbac.can_share(self.workspace.id, "editor_user")
        assert not self.rbac.is_admin(self.workspace.id, "editor_user")

    def test_viewer_permissions(self):
        """Test viewer permissions."""
        assert self.rbac.can_read(self.workspace.id, "viewer_user")
        assert not self.rbac.can_write(self.workspace.id, "viewer_user")
        assert not self.rbac.can_delete(self.workspace.id, "viewer_user")
        assert not self.rbac.can_share(self.workspace.id, "viewer_user")
        assert not self.rbac.is_admin(self.workspace.id, "viewer_user")

    def test_non_member_permissions(self):
        """Test non-member has no permissions."""
        assert not self.rbac.can_read(self.workspace.id, "unknown_user")
        assert not self.rbac.can_write(self.workspace.id, "unknown_user")
        assert not self.rbac.can_delete(self.workspace.id, "unknown_user")

    def test_get_role(self):
        """Test getting user role."""
        assert self.rbac.get_role(self.workspace.id, "admin_user") == Role.ADMIN
        assert self.rbac.get_role(self.workspace.id, "editor_user") == Role.EDITOR
        assert self.rbac.get_role(self.workspace.id, "viewer_user") == Role.VIEWER
        assert self.rbac.get_role(self.workspace.id, "unknown_user") is None
