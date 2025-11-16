"""Integration tests for crush services."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from xencode.crush.db.connection import DatabaseConnection
from xencode.crush.db.migrations import run_migrations
from xencode.crush.services import (
    SessionService,
    MessageService,
    PermissionService,
    HistoryService,
    MessageRole,
    CreateMessageParams,
    PermissionStatus,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Apply migrations
        run_migrations(str(db_path))
        
        db = DatabaseConnection(str(db_path))
        
        yield db
        
        db.close()


@pytest.fixture
def session_service(temp_db):
    """Create session service."""
    return SessionService(temp_db)


@pytest.fixture
def message_service(temp_db):
    """Create message service."""
    return MessageService(temp_db)


@pytest.fixture
def permission_service(temp_db):
    """Create permission service."""
    return PermissionService(temp_db, working_dir="/tmp", skip_requests=False)


@pytest.fixture
def history_service(temp_db):
    """Create history service."""
    return HistoryService(temp_db)


def test_session_crud(session_service):
    """Test session CRUD operations."""
    # Create session
    session = session_service.create("Test Session")
    assert session.id is not None
    assert session.title == "Test Session"
    assert session.prompt_tokens == 0
    assert session.busy is False
    
    # Get session
    retrieved = session_service.get(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id
    assert retrieved.title == session.title
    
    # Update session
    session.title = "Updated Session"
    session.prompt_tokens = 100
    session.completion_tokens = 50
    session.cost = 0.01
    updated = session_service.save(session)
    assert updated.title == "Updated Session"
    assert updated.prompt_tokens == 100
    
    # List sessions
    sessions = session_service.list()
    assert len(sessions) >= 1
    assert any(s.id == session.id for s in sessions)
    
    # Delete session
    session_service.delete(session.id)
    deleted = session_service.get(session.id)
    assert deleted is None


def test_message_crud(session_service, message_service):
    """Test message CRUD operations."""
    # Create session first
    session = session_service.create("Test Session")
    
    # Create message
    params = CreateMessageParams(
        session_id=session.id,
        role=MessageRole.USER,
        content="Hello, world!"
    )
    message = message_service.create(params)
    assert message.id is not None
    assert message.session_id == session.id
    assert message.role == MessageRole.USER
    assert message.get_text_content() == "Hello, world!"
    
    # Get message
    retrieved = message_service.get(message.id)
    assert retrieved is not None
    assert retrieved.id == message.id
    
    # Update message (streaming)
    message.append_content(" More text.")
    message_service.update(message)
    updated = message_service.get(message.id)
    assert "More text." in updated.get_text_content()
    
    # List messages
    messages = message_service.list(session.id)
    assert len(messages) == 1
    assert messages[0].id == message.id
    
    # Count messages
    count = message_service.count(session.id)
    assert count == 1
    
    # Delete message
    message_service.delete(message.id)
    deleted = message_service.get(message.id)
    assert deleted is None


def test_message_tool_calls(session_service, message_service):
    """Test message with tool calls."""
    session = session_service.create("Test Session")
    
    # Create assistant message with tool call
    params = CreateMessageParams(
        session_id=session.id,
        role=MessageRole.ASSISTANT,
        content=""
    )
    message = message_service.create(params)
    
    # Add tool call
    call_id = message.add_tool_call("bash", {"command": "ls -la"})
    assert call_id is not None
    message_service.update(message)
    
    # Verify tool call
    retrieved = message_service.get(message.id)
    # Should have text part (empty) and tool call part
    assert len(retrieved.parts) >= 1
    tool_call_parts = [p for p in retrieved.parts if p.type.value == 'tool_call']
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0].data['tool'] == "bash"
    assert tool_call_parts[0].data['call_id'] == call_id
    
    # Add tool result
    message.add_tool_result("bash", "file1.txt\nfile2.txt", call_id)
    message_service.update(message)
    
    # Verify tool result
    retrieved = message_service.get(message.id)
    tool_result_parts = [p for p in retrieved.parts if p.type.value == 'tool_result']
    assert len(tool_result_parts) == 1


@pytest.mark.asyncio
async def test_permission_request_flow(session_service, permission_service):
    """Test permission request and approval flow."""
    session = session_service.create("Test Session")
    
    # Create permission request
    request = permission_service.create_request(
        session_id=session.id,
        tool_call_id="test-call-1",
        tool_name="bash",
        action="execute",
        description="Run ls command",
        params={"command": "ls"}
    )
    assert request.id is not None
    assert request.status == PermissionStatus.PENDING
    
    # List pending requests
    pending = permission_service.list_pending(session.id)
    assert len(pending) == 1
    assert pending[0].id == request.id
    
    # Approve request
    permission_service.approve(request.id)
    
    # Verify approval
    approved = permission_service.get(request.id)
    assert approved.status == PermissionStatus.APPROVED
    assert approved.resolved_at is not None


@pytest.mark.asyncio
async def test_permission_auto_approve(session_service, permission_service):
    """Test auto-approval modes."""
    session = session_service.create("Test Session")
    
    # Test tool-level auto-approval
    permission_service.allowed_tools.add("view")
    request = permission_service.create_request(
        session_id=session.id,
        tool_call_id="test-call-2",
        tool_name="view",
        action="read",
        path="/tmp/test.txt"
    )
    assert request.status == PermissionStatus.AUTO_APPROVED
    
    # Test session-level auto-approval
    permission_service.auto_approve_session(session.id)
    request2 = permission_service.create_request(
        session_id=session.id,
        tool_call_id="test-call-3",
        tool_name="bash",
        action="execute",
        params={"command": "echo test"}
    )
    assert request2.status == PermissionStatus.AUTO_APPROVED
    
    # Test YOLO mode
    permission_service.set_skip_requests(True)
    session2 = session_service.create("Test Session 2")
    request3 = permission_service.create_request(
        session_id=session2.id,
        tool_call_id="test-call-4",
        tool_name="write",
        action="create",
        path="/tmp/new.txt"
    )
    assert request3.status == PermissionStatus.AUTO_APPROVED


def test_history_versioning(session_service, history_service):
    """Test file history versioning."""
    session = session_service.create("Test Session")
    file_path = "/tmp/test.txt"
    
    # Create initial version
    v1 = history_service.create(session.id, file_path, "Version 1")
    assert v1.version == 1
    assert v1.get_content_str() == "Version 1"
    
    # Create second version
    v2 = history_service.create_version(session.id, file_path, "Version 2")
    assert v2.version == 2
    assert v2.get_content_str() == "Version 2"
    
    # Create third version
    v3 = history_service.create_version(session.id, file_path, "Version 3")
    assert v3.version == 3
    
    # List versions
    versions = history_service.list_versions(file_path, session.id)
    assert len(versions) == 3
    assert versions[0].version == 1
    assert versions[2].version == 3
    
    # Get specific version
    retrieved = history_service.get_version(file_path, session.id, 2)
    assert retrieved is not None
    assert retrieved.version == 2
    assert retrieved.get_content_str() == "Version 2"
    
    # Get latest version
    latest = history_service.get_by_path_and_session(file_path, session.id)
    assert latest.version == 3


def test_history_diff(history_service, session_service):
    """Test diff generation."""
    session = session_service.create("Test Session")
    file_path = "/tmp/test.txt"
    
    # Create versions
    history_service.create(session.id, file_path, "Line 1\nLine 2\nLine 3")
    history_service.create_version(session.id, file_path, "Line 1\nLine 2 modified\nLine 3\nLine 4")
    
    # Generate diff
    diff = history_service.generate_diff_between_versions(file_path, session.id, 1, 2)
    assert diff is not None
    assert "Line 2" in diff
    assert "modified" in diff
    assert "Line 4" in diff


def test_history_stats(session_service, history_service):
    """Test file history statistics."""
    session = session_service.create("Test Session")
    
    # Create multiple file versions
    history_service.create(session.id, "/tmp/file1.txt", "Content 1")
    history_service.create_version(session.id, "/tmp/file1.txt", "Content 1 updated")
    history_service.create(session.id, "/tmp/file2.txt", "Content 2")
    
    # Get stats
    stats = history_service.get_file_stats(session.id)
    assert stats['file_count'] == 2
    assert stats['version_count'] == 3
    assert stats['total_size'] > 0
    
    # List files
    files = history_service.list_files_in_session(session.id)
    assert len(files) == 2
    assert "/tmp/file1.txt" in files
    assert "/tmp/file2.txt" in files


def test_session_busy_state(session_service):
    """Test session busy state management."""
    session = session_service.create("Test Session")
    assert session.busy is False
    
    # Set busy
    session_service.set_busy(session.id, True)
    updated = session_service.get(session.id)
    assert updated.busy is True
    
    # Clear busy
    session_service.set_busy(session.id, False)
    updated = session_service.get(session.id)
    assert updated.busy is False


def test_session_token_tracking(session_service):
    """Test token usage tracking."""
    session = session_service.create("Test Session")
    
    # Update tokens
    session_service.update_tokens(session.id, 100, 50, 0.01)
    updated = session_service.get(session.id)
    assert updated.prompt_tokens == 100
    assert updated.completion_tokens == 50
    assert updated.cost == 0.01
    
    # Add more tokens
    session_service.update_tokens(session.id, 50, 25, 0.005)
    updated = session_service.get(session.id)
    assert updated.prompt_tokens == 150
    assert updated.completion_tokens == 75
    assert abs(updated.cost - 0.015) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
