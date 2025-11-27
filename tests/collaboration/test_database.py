"""Tests for collaboration database."""

import pytest
import tempfile
from pathlib import Path

from xencode.collaboration.database import CollaborationDatabase
from xencode.collaboration.models import User, Workspace, WorkspaceMember, Session, KnowledgeItem, Role


class TestCollaborationDatabase:
    """Test cases for CollaborationDatabase."""

    def setup_method(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db = CollaborationDatabase(self.temp_db.name)

    def teardown_method(self):
        """Clean up test database."""
        del self.db  # Close database connection
        self.temp_db.close()
        try:
            Path(self.temp_db.name).unlink()
        except PermissionError:
            pass  # Windows file lock

    def test_create_user(self):
        """Test creating a user."""
        user = User(id="user1", username="testuser", email="test@example.com")
        created_user = self.db.create_user(user)
        
        assert created_user.id == "user1"
        assert created_user.username == "testuser"

    def test_get_user(self):
        """Test retrieving a user."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        retrieved_user = self.db.get_user("user1")
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"

    def test_create_workspace(self):
        """Test creating a workspace."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        workspace = Workspace(id="ws1", name="Test Workspace", created_by="user1")
        created_workspace = self.db.create_workspace(workspace)
        
        assert created_workspace.id == "ws1"
        assert created_workspace.name == "Test Workspace"

    def test_add_member(self):
        """Test adding a member to a workspace."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        workspace = Workspace(id="ws1", name="Test Workspace", created_by="user1")
        self.db.create_workspace(workspace)
        
        member = WorkspaceMember(workspace_id="ws1", user_id="user1", role=Role.ADMIN)
        added_member = self.db.add_member(member)
        
        assert added_member.workspace_id == "ws1"
        assert added_member.role == Role.ADMIN

    def test_get_member_role(self):
        """Test retrieving a member's role."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        workspace = Workspace(id="ws1", name="Test Workspace", created_by="user1")
        self.db.create_workspace(workspace)
        
        member = WorkspaceMember(workspace_id="ws1", user_id="user1", role=Role.EDITOR)
        self.db.add_member(member)
        
        role = self.db.get_member_role("ws1", "user1")
        assert role == Role.EDITOR

    def test_create_session(self):
        """Test creating a session."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        workspace = Workspace(id="ws1", name="Test Workspace", created_by="user1")
        self.db.create_workspace(workspace)
        
        session = Session(id="sess1", workspace_id="ws1", title="Test Session", created_by="user1")
        created_session = self.db.create_session(session)
        
        assert created_session.id == "sess1"
        assert created_session.title == "Test Session"

    def test_search_knowledge(self):
        """Test searching knowledge base."""
        user = User(id="user1", username="testuser")
        self.db.create_user(user)
        
        workspace = Workspace(id="ws1", name="Test Workspace", created_by="user1")
        self.db.create_workspace(workspace)
        
        item = KnowledgeItem(
            id="kb1",
            workspace_id="ws1",
            title="Python Tips",
            content="Some Python programming tips",
            created_by="user1"
        )
        self.db.create_knowledge_item(item)
        
        results = self.db.search_knowledge("ws1", "Python")
        assert len(results) == 1
        assert results[0].title == "Python Tips"
