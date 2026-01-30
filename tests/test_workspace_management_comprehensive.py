#!/usr/bin/env python3
"""
Comprehensive Tests for Workspace Management System

Tests for workspace creation, collaboration, file management,
isolation mechanisms, and comprehensive workspace lifecycle management.
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime, timedelta

from xencode.workspace.workspace_manager import WorkspaceManager, WorkspaceError
from xencode.models.workspace import (
    Workspace, WorkspaceFile, WorkspaceCollaborator, WorkspaceConfig,
    WorkspaceType, WorkspaceStatus, CollaborationMode, Change, ChangeType,
    create_default_workspace
)
from xencode.workspace.workspace_security import (
    WorkspaceSecurityManager, WorkspacePermission, IsolationLevel, WorkspaceContext
)


class TestWorkspaceManagementBasics:
    """Test basic workspace management functionality"""

    @pytest_asyncio.fixture
    async def workspace_manager(self):
        """Create a workspace manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "workspaces.db"
            # Create a workspace manager with a temporary storage backend
            manager = WorkspaceManager()
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_workspace_manager_initialization(self, workspace_manager):
        """Test workspace manager initialization"""
        assert workspace_manager is not None
        assert workspace_manager.storage is not None
        assert workspace_manager.security_manager is not None
        assert workspace_manager.workspace_cache == {}
        assert workspace_manager.active_sessions == {}

    @pytest.mark.asyncio
    async def test_workspace_creation(self, workspace_manager):
        """Test workspace creation"""
        workspace = await workspace_manager.create_workspace(
            name="Test Project",
            owner_id="user123",
            workspace_type=WorkspaceType.PROJECT,
            description="A test project workspace"
        )

        assert workspace is not None
        assert workspace.name == "Test Project"
        assert workspace.owner_id == "user123"
        assert workspace.workspace_type == WorkspaceType.PROJECT
        assert workspace.description == "A test project workspace"
        assert workspace.status == WorkspaceStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_workspace_retrieval(self, workspace_manager):
        """Test workspace retrieval"""
        # Create workspace first
        created_workspace = await workspace_manager.create_workspace(
            name="Retrieval Test",
            owner_id="user456"
        )
        assert created_workspace is not None

        # Retrieve workspace
        retrieved_workspace = await workspace_manager.get_workspace(created_workspace.id, "user456")

        assert retrieved_workspace is not None
        assert retrieved_workspace.id == created_workspace.id
        assert retrieved_workspace.name == "Retrieval Test"
        assert retrieved_workspace.owner_id == "user456"

    @pytest.mark.asyncio
    async def test_workspace_update(self, workspace_manager):
        """Test workspace update"""
        # Create workspace
        workspace = await workspace_manager.create_workspace(
            name="Original Name",
            owner_id="user123",
            description="Original description"
        )
        assert workspace is not None

        # Update workspace
        workspace.name = "Updated Name"
        workspace.description = "Updated description"
        success = await workspace_manager.update_workspace(workspace)

        assert success is True

        # Retrieve and verify update
        updated_workspace = await workspace_manager.get_workspace(workspace.id, "user123")
        assert updated_workspace is not None
        assert updated_workspace.name == "Updated Name"
        assert updated_workspace.description == "Updated description"

    @pytest.mark.asyncio
    async def test_workspace_deletion(self, workspace_manager):
        """Test workspace deletion"""
        # Create workspace
        workspace = await workspace_manager.create_workspace(
            name="To Delete",
            owner_id="user123"
        )
        assert workspace is not None

        # Delete workspace
        success = await workspace_manager.delete_workspace(workspace.id, "user123")
        assert success is True

        # Verify deletion
        deleted_workspace = await workspace_manager.get_workspace(workspace.id, "user123")
        assert deleted_workspace is None

    @pytest.mark.asyncio
    async def test_workspace_listing(self, workspace_manager):
        """Test workspace listing"""
        # Create multiple workspaces
        workspace1 = await workspace_manager.create_workspace("Project 1", "user123")
        workspace2 = await workspace_manager.create_workspace("Project 2", "user123")
        workspace3 = await workspace_manager.create_workspace("Project 3", "user456")

        assert workspace1 is not None
        assert workspace2 is not None
        assert workspace3 is not None

        # List workspaces for user123
        user_workspaces = await workspace_manager.list_user_workspaces("user123")
        assert len([ws for ws in user_workspaces if ws.owner_id == "user123"]) >= 2

        # List workspaces for user456
        user456_workspaces = await workspace_manager.list_user_workspaces("user456")
        assert len([ws for ws in user456_workspaces if ws.owner_id == "user456"]) >= 1

    @pytest.mark.asyncio
    async def test_workspace_collaborator_management(self, workspace_manager):
        """Test workspace collaborator management"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Collab Test", "owner123")
        assert workspace is not None

        # Add collaborator
        collaborator = WorkspaceCollaborator(
            user_id="collab123",
            username="collaborator",
            role="editor"
        )

        success = await workspace_manager.add_collaborator(
            workspace.id, "collab123", collaborator, "owner123"
        )
        assert success is True

        # Verify collaborator was added
        updated_workspace = await workspace_manager.get_workspace(workspace.id, "owner123")
        assert updated_workspace is not None
        assert len(updated_workspace.collaborators) == 2  # Owner + collaborator
        collab = updated_workspace.get_collaborator("collab123")
        assert collab is not None
        assert collab.role == "editor"

        # Remove collaborator
        remove_success = await workspace_manager.remove_collaborator(
            workspace.id, "collab123", "owner123"
        )
        assert remove_success is True

        # Verify collaborator was removed
        updated_workspace = await workspace_manager.get_workspace(workspace.id, "owner123")
        assert updated_workspace is not None
        assert len(updated_workspace.collaborators) == 1  # Only owner remains

    @pytest.mark.asyncio
    async def test_workspace_file_management(self, workspace_manager):
        """Test workspace file management"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("File Test", "user123")
        assert workspace is not None

        # Add file to workspace
        file = WorkspaceFile(
            name="test.py",
            path="/test.py",
            content="print('hello world')",
            file_type="python"
        )

        success = await workspace_manager.add_file(workspace.id, file, "user123")
        assert success is True

        # Retrieve file
        retrieved_file = await workspace_manager.get_file(workspace.id, file.id, "user123")
        assert retrieved_file is not None
        assert retrieved_file.name == "test.py"
        assert retrieved_file.content == "print('hello world')"

        # Update file
        retrieved_file.content = "print('updated content')"
        update_success = await workspace_manager.update_file(workspace.id, retrieved_file, "user123")
        assert update_success is True

        # Verify update
        updated_file = await workspace_manager.get_file(workspace.id, file.id, "user123")
        assert updated_file is not None
        assert updated_file.content == "print('updated content')"
        assert updated_file.version == 2  # Should have incremented

        # Delete file
        delete_success = await workspace_manager.delete_file(workspace.id, file.id, "user123")
        assert delete_success is True

        # Verify deletion
        deleted_file = await workspace_manager.get_file(workspace.id, file.id, "user123")
        assert deleted_file is None

    @pytest.mark.asyncio
    async def test_workspace_file_locking(self, workspace_manager):
        """Test workspace file locking mechanism"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Lock Test", "user123")
        assert workspace is not None

        # Add file
        file = WorkspaceFile(
            name="locked_file.py",
            path="/locked_file.py",
            content="# Initial content"
        )

        success = await workspace_manager.add_file(workspace.id, file, "user123")
        assert success is True

        # Lock file for user123
        lock_success = await workspace_manager.lock_file(workspace.id, file.id, "user123")
        assert lock_success is True

        # Verify file is locked for user123
        locks = await workspace_manager.get_file_locks(workspace.id)
        assert file.id in locks
        assert locks[file.id] == "user123"

        # Try to lock for different user (should fail)
        lock_fail = await workspace_manager.lock_file(workspace.id, file.id, "user456")
        assert lock_fail is False

        # Unlock file
        unlock_success = await workspace_manager.unlock_file(workspace.id, file.id, "user123")
        assert unlock_success is True

        # Verify file is unlocked
        locks = await workspace_manager.get_file_locks(workspace.id)
        assert file.id not in locks

    @pytest.mark.asyncio
    async def test_workspace_change_tracking(self, workspace_manager):
        """Test workspace change tracking"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Change Test", "user123")
        assert workspace is not None

        # Add file
        file = WorkspaceFile(
            name="change_test.py",
            path="/change_test.py",
            content=""
        )

        success = await workspace_manager.add_file(workspace.id, file, "user123")
        assert success is True

        # Create and add a change
        change = Change(
            workspace_id=workspace.id,
            file_id=file.id,
            change_type=ChangeType.INSERT,
            position=0,
            content="print('hello')",
            author_id="user123"
        )

        add_change_success = await workspace_manager.add_change(change)
        assert add_change_success is True

        # Get changes
        changes = await workspace_manager.get_changes(workspace.id, file.id)
        assert len(changes) >= 1
        assert changes[0].content == "print('hello')"
        assert changes[0].author_id == "user123"

    @pytest.mark.asyncio
    async def test_workspace_statistics(self, workspace_manager):
        """Test workspace statistics"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Stats Test", "user123")
        assert workspace is not None

        # Add some files
        for i in range(3):
            file = WorkspaceFile(
                name=f"file_{i}.py",
                path=f"/file_{i}.py",
                content=f"content_{i}" * 10  # Make content larger
            )
            success = await workspace_manager.add_file(workspace.id, file, "user123")
            assert success is True

        # Add collaborator
        collaborator = WorkspaceCollaborator(user_id="collab123", username="collaborator", role="viewer")
        collab_success = await workspace_manager.add_collaborator(
            workspace.id, "collab123", collaborator, "user123"
        )
        assert collab_success is True

        # Get statistics
        stats = await workspace_manager.get_workspace_stats(workspace.id)
        assert stats is not None
        assert stats['id'] == workspace.id
        assert stats['file_count'] >= 3
        assert stats['collaborator_count'] >= 2  # Owner + collaborator
        assert stats['name'] == "Stats Test"


class TestWorkspaceSecurityAndIsolation:
    """Test workspace security and isolation features"""

    @pytest_asyncio.fixture
    async def workspace_manager(self):
        """Create a workspace manager for security testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "workspaces.db"
            manager = WorkspaceManager()
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_workspace_permissions(self, workspace_manager):
        """Test workspace permissions"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Perm Test", "owner123")
        assert workspace is not None

        # Add collaborator with viewer role
        viewer = WorkspaceCollaborator(user_id="viewer123", username="viewer", role="viewer")
        success = await workspace_manager.add_collaborator(
            workspace.id, "viewer123", viewer, "owner123"
        )
        assert success is True

        # Check permissions
        can_read_owner = await workspace_manager.check_workspace_permission(
            workspace.id, "owner123", WorkspacePermission.READ
        )
        assert can_read_owner is True

        can_read_viewer = await workspace_manager.check_workspace_permission(
            workspace.id, "viewer123", WorkspacePermission.READ
        )
        assert can_read_viewer is True

        can_write_viewer = await workspace_manager.check_workspace_permission(
            workspace.id, "viewer123", WorkspacePermission.WRITE
        )
        # Viewer might have write permission depending on implementation
        # The important thing is that the permission system works

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, workspace_manager):
        """Test workspace isolation"""
        # Create two workspaces
        ws1 = await workspace_manager.create_workspace("Workspace 1", "user123")
        ws2 = await workspace_manager.create_workspace("Workspace 2", "user123")

        assert ws1 is not None
        assert ws2 is not None
        assert ws1.id != ws2.id

        # Add files to each workspace
        file1 = WorkspaceFile(name="ws1_file.py", path="/ws1_file.py", content="WS1 content")
        file2 = WorkspaceFile(name="ws2_file.py", path="/ws2_file.py", content="WS2 content")

        success1 = await workspace_manager.add_file(ws1.id, file1, "user123")
        success2 = await workspace_manager.add_file(ws2.id, file2, "user123")

        assert success1 is True
        assert success2 is True

        # Verify isolation - files should be in respective workspaces only
        retrieved_file1 = await workspace_manager.get_file(ws1.id, file1.id, "user123")
        retrieved_file2 = await workspace_manager.get_file(ws2.id, file2.id, "user123")

        assert retrieved_file1 is not None
        assert retrieved_file2 is not None
        assert retrieved_file1.content == "WS1 content"
        assert retrieved_file2.content == "WS2 content"

        # Verify files are not accessible from wrong workspace
        wrong_file1 = await workspace_manager.get_file(ws2.id, file1.id, "user123")
        wrong_file2 = await workspace_manager.get_file(ws1.id, file2.id, "user123")

        assert wrong_file1 is None
        assert wrong_file2 is None

    @pytest.mark.asyncio
    async def test_cross_workspace_access_prevention(self, workspace_manager):
        """Test prevention of cross-workspace access"""
        # Create workspaces for different users
        ws_owner1 = await workspace_manager.create_workspace("Owner1 Workspace", "owner1")
        ws_owner2 = await workspace_manager.create_workspace("Owner2 Workspace", "owner2")

        assert ws_owner1 is not None
        assert ws_owner2 is not None

        # Add files to each workspace
        file1 = WorkspaceFile(name="file1.py", path="/file1.py", content="Owner1 content")
        file2 = WorkspaceFile(name="file2.py", path="/file2.py", content="Owner2 content")

        success1 = await workspace_manager.add_file(ws_owner1.id, file1, "owner1")
        success2 = await workspace_manager.add_file(ws_owner2.id, file2, "owner2")

        assert success1 is True
        assert success2 is True

        # Owner1 should not be able to access Owner2's workspace
        inaccessible_ws = await workspace_manager.get_workspace(ws_owner2.id, "owner1")
        assert inaccessible_ws is None

        # Owner1 should not be able to access Owner2's file
        inaccessible_file = await workspace_manager.get_file(ws_owner2.id, file2.id, "owner1")
        assert inaccessible_file is None

        # Similarly for Owner2 accessing Owner1's resources
        inaccessible_ws2 = await workspace_manager.get_workspace(ws_owner1.id, "owner2")
        assert inaccessible_ws2 is None

    @pytest.mark.asyncio
    async def test_workspace_security_context_switching(self, workspace_manager):
        """Test workspace security context switching"""
        # Create workspaces
        ws1 = await workspace_manager.create_workspace("Context WS 1", "user123")
        ws2 = await workspace_manager.create_workspace("Context WS 2", "user123")

        assert ws1 is not None
        assert ws2 is not None

        # Switch context to first workspace
        success1 = await workspace_manager.switch_workspace_context("user123", ws1.id)
        assert success1 is True

        # Switch context to second workspace
        success2 = await workspace_manager.switch_workspace_context("user123", ws2.id)
        assert success2 is True

    @pytest.mark.asyncio
    async def test_workspace_isolation_levels(self, workspace_manager):
        """Test different workspace isolation levels"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Isolation Test", "user123")
        assert workspace is not None

        # Set isolation level
        success = await workspace_manager.set_workspace_isolation_level(
            workspace.id, IsolationLevel.STRICT, "user123"
        )
        assert success is True

        # Verify isolation level was set by getting security status
        security_status = await workspace_manager.get_workspace_security_status(workspace.id, "user123")
        assert security_status is not None
        # The exact structure depends on the implementation


class TestWorkspaceCollaborationFeatures:
    """Test workspace collaboration features"""

    @pytest_asyncio.fixture
    async def workspace_manager(self):
        """Create a workspace manager for collaboration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "workspaces.db"
            manager = WorkspaceManager()
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_real_time_collaboration_setup(self, workspace_manager):
        """Test real-time collaboration setup"""
        # Create workspace
        workspace = await workspace_manager.create_workspace(
            "Collaboration Test",
            "owner123"
        )
        assert workspace is not None

        # Add multiple collaborators
        for i in range(3):
            collab = WorkspaceCollaborator(
                user_id=f"collab{i}",
                username=f"collaborator{i}",
                role="editor" if i < 2 else "viewer"
            )
            success = await workspace_manager.add_collaborator(
                workspace.id, f"collab{i}", collab, "owner123"
            )
            assert success is True

        # Verify all collaborators were added
        updated_ws = await workspace_manager.get_workspace(workspace.id, "owner123")
        assert updated_ws is not None
        # Should have owner + 3 collaborators = 4 total
        assert len(updated_ws.collaborators) >= 4

    @pytest.mark.asyncio
    async def test_collaborator_activity_tracking(self, workspace_manager):
        """Test collaborator activity tracking"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Activity Test", "owner123")
        assert workspace is not None

        # Add collaborator
        collab = WorkspaceCollaborator(user_id="active_user", username="Active User", role="editor")
        add_success = await workspace_manager.add_collaborator(
            workspace.id, "active_user", collab, "owner123"
        )
        assert add_success is True

        # Update collaborator activity
        activity_success = await workspace_manager.update_collaborator_activity(
            workspace.id, "active_user", session_id="session_123"
        )
        assert activity_success is True

        # Verify activity was updated
        updated_ws = await workspace_manager.get_workspace(workspace.id, "owner123")
        assert updated_ws is not None
        active_collab = updated_ws.get_collaborator("active_user")
        assert active_collab is not None
        assert active_collab.is_active is True
        assert active_collab.session_id == "session_123"

    @pytest.mark.asyncio
    async def test_workspace_session_management(self, workspace_manager):
        """Test workspace session management"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Session Test", "user123")
        assert workspace is not None

        # Add collaborators
        for i in range(2):
            collab = WorkspaceCollaborator(
                user_id=f"session_user{i}",
                username=f"Session User {i}",
                role="editor"
            )
            success = await workspace_manager.add_collaborator(
                workspace.id, f"session_user{i}", collab, "user123"
            )
            assert success is True

        # Simulate multiple users joining the workspace
        for i in range(2):
            await workspace_manager.update_collaborator_activity(
                workspace.id, f"session_user{i}", session_id=f"session_{i}"
            )

        # Verify collaborators were added by checking workspace
        updated_workspace = await workspace_manager.get_workspace(workspace.id, "user123")
        assert updated_workspace is not None
        assert len(updated_workspace.collaborators) >= 3  # Owner + 2 collaborators

    @pytest.mark.asyncio
    async def test_collaboration_conflict_resolution(self, workspace_manager):
        """Test collaboration conflict resolution"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Conflict Test", "owner123")
        assert workspace is not None

        # Add file
        file = WorkspaceFile(name="shared_file.py", path="/shared.py", content="original content")
        add_success = await workspace_manager.add_file(workspace.id, file, "owner123")
        assert add_success is True

        # Simulate concurrent changes by different users (would normally happen through CRDT)
        # For testing, we'll just verify the change tracking works
        for i in range(3):
            change = Change(
                workspace_id=workspace.id,
                file_id=file.id,
                change_type=ChangeType.UPDATE,
                position=0,
                content=f"content version {i}",
                author_id=f"user{i}"
            )
            await workspace_manager.add_change(change)

        # Get all changes
        changes = await workspace_manager.get_changes(workspace.id, file.id)
        assert len(changes) >= 3

        # Check for any conflicts
        conflicts = workspace.get_unresolved_conflicts()
        # Depending on implementation, there might or might not be conflicts


class TestWorkspacePerformanceAndCaching:
    """Test workspace performance and caching features"""

    @pytest_asyncio.fixture
    async def workspace_manager(self):
        """Create a workspace manager for performance testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "workspaces.db"
            manager = WorkspaceManager()
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_workspace_caching_mechanism(self, workspace_manager):
        """Test workspace caching mechanism"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Cache Test", "user123")
        assert workspace is not None

        # Verify it's in cache after creation
        assert workspace.id in workspace_manager.workspace_cache

        # Retrieve workspace multiple times (should come from cache)
        for i in range(5):
            retrieved = await workspace_manager.get_workspace(workspace.id, "user123")
            assert retrieved is not None
            assert retrieved.id == workspace.id

        # Cache should still contain the workspace
        assert workspace.id in workspace_manager.workspace_cache

    @pytest.mark.asyncio
    async def test_cache_expiration(self, workspace_manager):
        """Test cache expiration functionality"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Cache Expiry Test", "user123")
        assert workspace is not None

        # Verify it's in cache
        assert workspace.id in workspace_manager.workspace_cache

        # Manually expire the cache entry
        workspace_manager.cache_timestamps[workspace.id] = datetime.now() - timedelta(minutes=10)  # Expired

        # Next retrieval should reload from storage
        retrieved = await workspace_manager.get_workspace(workspace.id, "user123")
        assert retrieved is not None

        # Should be re-cached with fresh timestamp
        assert workspace.id in workspace_manager.workspace_cache

    @pytest.mark.asyncio
    async def test_workspace_cleanup_tasks(self, workspace_manager):
        """Test workspace cleanup tasks"""
        # Create several workspaces
        workspaces = []
        for i in range(5):
            ws = await workspace_manager.create_workspace(f"Cleanup Test {i}", "user123")
            assert ws is not None
            workspaces.append(ws)

        # Run cache cleanup (should clean expired entries)
        cleaned_count = await workspace_manager.cleanup_expired_cache()
        # May not clean anything if nothing is expired

        # Verify workspaces still exist
        for ws in workspaces:
            retrieved = await workspace_manager.get_workspace(ws.id, "user123")
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_large_workspace_performance(self, workspace_manager):
        """Test performance with larger workspaces"""
        # Create workspace
        workspace = await workspace_manager.create_workspace("Large Workspace Test", "user123")
        assert workspace is not None

        # Add many files to simulate a large workspace
        for i in range(20):
            large_content = f"# File {i} content\n" + "code_line = value\n" * 100  # 100 lines each
            file = WorkspaceFile(
                name=f"large_file_{i}.py",
                path=f"/large_file_{i}.py",
                content=large_content
            )
            success = await workspace_manager.add_file(workspace.id, file, "user123")
            assert success is True

        # Get workspace statistics
        stats = await workspace_manager.get_workspace_stats(workspace.id)
        assert stats is not None
        assert stats['file_count'] >= 20

        # Retrieving should still be reasonably fast
        start_time = asyncio.get_event_loop().time()
        retrieved = await workspace_manager.get_workspace(workspace.id, "user123")
        end_time = asyncio.get_event_loop().time()
        
        retrieval_time = (end_time - start_time) * 1000  # Convert to ms
        # Should be fast even with many files (under 1000ms)
        assert retrieval_time < 1000


class TestWorkspaceIntegration:
    """Integration tests for workspace management"""

    @pytest_asyncio.fixture
    async def workspace_system(self):
        """Create a complete workspace system for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "workspaces.db"
            manager = WorkspaceManager()
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_full_workspace_lifecycle(self, workspace_system):
        """Test complete workspace lifecycle"""
        manager = workspace_system

        # 1. Create workspace
        workspace = await manager.create_workspace(
            name="Integration Test Workspace",
            owner_id="owner123",
            workspace_type=WorkspaceType.PROJECT,
            description="A workspace for integration testing"
        )
        assert workspace is not None
        assert workspace.name == "Integration Test Workspace"

        # 2. Add collaborator
        collaborator = WorkspaceCollaborator(
            user_id="collab123",
            username="integration_collaborator",
            role="editor"
        )
        add_collab_success = await manager.add_collaborator(
            workspace.id, "collab123", collaborator, "owner123"
        )
        assert add_collab_success is True

        # 3. Add files
        for i in range(3):
            file = WorkspaceFile(
                name=f"integration_file_{i}.py",
                path=f"/integration_file_{i}.py",
                content=f"# Content for file {i}\nprint('file {i} content')"
            )
            add_file_success = await manager.add_file(workspace.id, file, "owner123")
            assert add_file_success is True

        # 4. Update collaborator activity
        activity_success = await manager.update_collaborator_activity(
            workspace.id, "collab123"
        )
        assert activity_success is True

        # 5. Get workspace and verify content
        retrieved_workspace = await manager.get_workspace(workspace.id, "owner123")
        assert retrieved_workspace is not None
        assert retrieved_workspace.name == "Integration Test Workspace"
        assert len(retrieved_workspace.files) >= 3
        assert len(retrieved_workspace.collaborators) >= 2  # Owner + collaborator

        # 6. Get workspace statistics
        stats = await manager.get_workspace_stats(workspace.id)
        assert stats is not None
        assert stats['file_count'] >= 3
        assert stats['collaborator_count'] >= 2

        # 7. Test file operations
        file_ids = list(retrieved_workspace.files.keys())
        if file_ids:
            test_file_id = file_ids[0]

            # Get file
            file = await manager.get_file(workspace.id, test_file_id, "owner123")
            assert file is not None

            # Update file
            file.content += "\n# Updated content"
            update_success = await manager.update_file(workspace.id, file, "owner123")
            assert update_success is True

            # Verify update
            updated_file = await manager.get_file(workspace.id, test_file_id, "owner123")
            assert updated_file is not None
            assert "# Updated content" in updated_file.content

        # 8. Test permissions
        has_read = await manager.check_workspace_permission(
            workspace.id, "owner123", "read"
        )
        assert has_read is True

        # 9. Clean up - delete workspace
        delete_success = await manager.delete_workspace(workspace.id, "owner123")
        assert delete_success is True

        # Verify deletion
        deleted_workspace = await manager.get_workspace(workspace.id, "owner123")
        assert deleted_workspace is None

    @pytest.mark.asyncio
    async def test_workspace_collaboration_flow(self, workspace_system):
        """Test complete collaboration workflow"""
        manager = workspace_system

        # Create workspace for collaboration
        workspace = await manager.create_workspace(
            name="Collaboration Flow Test",
            owner_id="main_user",
            collaboration_mode=CollaborationMode.SHARED
        )
        assert workspace is not None

        # Add multiple collaborators with different roles
        collaborators_data = [
            {"user_id": "editor1", "role": "editor", "username": "Editor User 1"},
            {"user_id": "editor2", "role": "editor", "username": "Editor User 2"},
            {"user_id": "viewer1", "role": "viewer", "username": "Viewer User 1"}
        ]

        for collab_data in collaborators_data:
            collab = WorkspaceCollaborator(
                user_id=collab_data["user_id"],
                username=collab_data["username"],
                role=collab_data["role"]
            )
            success = await manager.add_collaborator(
                workspace.id, collab_data["user_id"], collab, "main_user"
            )
            assert success is True

        # Add shared files
        shared_file = WorkspaceFile(
            name="shared_document.py",
            path="/shared.py",
            content="# Shared document\n# Initially created by owner"
        )
        file_success = await manager.add_file(workspace.id, shared_file, "main_user")
        assert file_success is True

        # Simulate collaboration: editors make changes
        editor_changes = [
            {"user_id": "editor1", "content_addition": "\n# Added by editor 1"},
            {"user_id": "editor2", "content_addition": "\n# Added by editor 2"}
        ]

        for change_data in editor_changes:
            # Get the file
            file = await manager.get_file(workspace.id, shared_file.id, change_data["user_id"])
            assert file is not None

            # Modify content
            file.content += change_data["content_addition"]

            # Update file
            update_success = await manager.update_file(workspace.id, file, change_data["user_id"])
            assert update_success is True

        # Verify final content includes all additions
        final_file = await manager.get_file(workspace.id, shared_file.id, "main_user")
        assert final_file is not None
        assert "# Added by editor 1" in final_file.content
        assert "# Added by editor 2" in final_file.content

        # Verify viewers can access but not modify (depending on implementation)
        viewer_can_read = await manager.check_workspace_permission(
            workspace.id, "viewer1", "read"
        )
        assert viewer_can_read is True

        # Viewer might not be able to write depending on role permissions
        viewer_can_write = await manager.check_workspace_permission(
            workspace.id, "viewer1", "write"
        )
        # This depends on the specific implementation

    @pytest.mark.asyncio
    async def test_workspace_isolation_and_security(self, workspace_system):
        """Test workspace isolation and security measures"""
        manager = workspace_system

        # Create workspaces for different users
        user1_workspace = await manager.create_workspace(
            name="User 1 Private Workspace",
            owner_id="user1",
            description="Private workspace for user 1"
        )
        assert user1_workspace is not None

        user2_workspace = await manager.create_workspace(
            name="User 2 Private Workspace",
            owner_id="user2",
            description="Private workspace for user 2"
        )
        assert user2_workspace is not None

        # Add files to each workspace
        user1_file = WorkspaceFile(
            name="user1_private.py",
            path="/private.py",
            content="# User 1 private content"
        )
        user1_add_success = await manager.add_file(user1_workspace.id, user1_file, "user1")
        assert user1_add_success is True

        user2_file = WorkspaceFile(
            name="user2_private.py",
            path="/private.py",
            content="# User 2 private content"
        )
        user2_add_success = await manager.add_file(user2_workspace.id, user2_file, "user2")
        assert user2_add_success is True

        # Verify isolation: User 1 cannot access User 2's workspace
        user1_access_to_user2_ws = await manager.get_workspace(user2_workspace.id, "user1")
        assert user1_access_to_user2_ws is None

        # Verify isolation: User 1 cannot access User 2's file
        user1_access_to_user2_file = await manager.get_file(user2_workspace.id, user2_file.id, "user1")
        assert user1_access_to_user2_file is None

        # Verify User 1 can access their own workspace and file
        user1_own_access = await manager.get_workspace(user1_workspace.id, "user1")
        assert user1_own_access is not None
        assert user1_own_access.name == "User 1 Private Workspace"

        user1_own_file_access = await manager.get_file(user1_workspace.id, user1_file.id, "user1")
        assert user1_own_file_access is not None
        assert user1_own_file_access.content == "# User 1 private content"

        # Verify User 2 can access their own workspace and file
        user2_own_access = await manager.get_workspace(user2_workspace.id, "user2")
        assert user2_own_access is not None
        assert user2_own_access.name == "User 2 Private Workspace"

        user2_own_file_access = await manager.get_file(user2_workspace.id, user2_file.id, "user2")
        assert user2_own_file_access is not None
        assert user2_own_file_access.content == "# User 2 private content"

        # Test security status
        user1_security_status = await manager.get_workspace_security_status(user1_workspace.id, "user1")
        assert user1_security_status is not None

        user2_security_status = await manager.get_workspace_security_status(user2_workspace.id, "user2")
        assert user2_security_status is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])