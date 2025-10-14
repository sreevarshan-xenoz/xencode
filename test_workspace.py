#!/usr/bin/env python3
"""Test script for workspace management system"""

import asyncio
from datetime import datetime
from xencode.workspace.workspace_manager import WorkspaceManager
from xencode.workspace.crdt_engine import CRDTEngine
from xencode.workspace.collaboration_manager import CollaborationManager
from xencode.models.workspace import WorkspaceType, ChangeType, WorkspaceFile

async def test_workspace_management():
    """Test workspace management functionality"""
    
    try:
        print("Testing workspace management system...")
        
        # Create workspace manager
        workspace_manager = WorkspaceManager()
        await workspace_manager.initialize()
        
        # Test workspace creation
        print("\n1. Testing workspace creation...")
        workspace = await workspace_manager.create_workspace(
            name="Test Project",
            owner_id="user123",
            workspace_type=WorkspaceType.PROJECT,
            description="A test project workspace"
        )
        
        if workspace:
            print(f"✓ Workspace created: {workspace.name} ({workspace.id})")
            print(f"  Owner: {workspace.owner_id}")
            print(f"  Type: {workspace.workspace_type}")
            print(f"  Collaborators: {len(workspace.collaborators)}")
        else:
            print("✗ Workspace creation failed")
            return
        
        workspace_id = workspace.id
        
        # Test file operations
        print("\n2. Testing file operations...")
        
        # Create a test file
        test_file = WorkspaceFile(
            name="test.py",
            path="src/test.py",
            content="print('Hello, World!')\n",
            file_type="python",
            language="python"
        )
        
        file_added = await workspace_manager.add_file(workspace_id, test_file, "user123")
        if file_added:
            print(f"✓ File added: {test_file.name}")
            print(f"  Path: {test_file.path}")
            print(f"  Size: {test_file.size_bytes} bytes")
        else:
            print("✗ File addition failed")
        
        # Retrieve the file
        retrieved_file = await workspace_manager.get_file_by_path(workspace_id, "src/test.py", "user123")
        if retrieved_file:
            print(f"✓ File retrieved: {retrieved_file.name}")
            print(f"  Content: {retrieved_file.content.strip()}")
        else:
            print("✗ File retrieval failed")
        
        # Test CRDT engine
        print("\n3. Testing CRDT collaboration engine...")
        
        crdt_engine = CRDTEngine("node1")
        
        # Register document
        file_id = test_file.id
        crdt_engine.register_document(file_id, test_file.content)
        
        # Create changes
        change1 = crdt_engine.create_change(
            workspace_id, file_id, ChangeType.INSERT, 0, "# Test Script\n", author_id="user123"
        )
        
        change2 = crdt_engine.create_change(
            workspace_id, file_id, ChangeType.INSERT, len("# Test Script\n"), "import os\n", author_id="user456"
        )
        
        # Apply changes
        success1, conflict1 = await crdt_engine.apply_change(change1)
        success2, conflict2 = await crdt_engine.apply_change(change2)
        
        print(f"✓ Change 1 applied: {success1}, Conflict: {conflict1 is not None}")
        print(f"✓ Change 2 applied: {success2}, Conflict: {conflict2 is not None}")
        
        # Get final content
        final_content = crdt_engine.get_document_content(file_id)
        print(f"✓ Final document content:")
        print(f"  {repr(final_content)}")
        
        # Test collaboration manager
        print("\n4. Testing collaboration manager...")
        
        collab_manager = CollaborationManager(workspace_manager, crdt_engine)
        
        # Start collaboration sessions
        session1_started = await collab_manager.start_collaboration_session(
            workspace_id, "user123", "Alice", "session1"
        )
        session2_started = await collab_manager.start_collaboration_session(
            workspace_id, "user456", "Bob", "session2"
        )
        
        print(f"✓ Collaboration sessions started: Alice={session1_started}, Bob={session2_started}")
        
        # Update presence
        await collab_manager.update_presence(workspace_id, "user123", file_id, 10, 5, 15)
        await collab_manager.update_presence(workspace_id, "user456", file_id, 20, 18, 25)
        
        # Get presence info
        presence = collab_manager.get_workspace_presence(workspace_id)
        print(f"✓ Workspace presence:")
        print(f"  Online collaborators: {presence['online_count']}")
        print(f"  Active files: {len(presence['active_files'])}")
        
        for file_id_key, collaborators in presence['active_files'].items():
            print(f"    File {file_id_key}: {len(collaborators)} collaborators")
            for collab in collaborators:
                print(f"      - {collab['username']} at position {collab['cursor_position']}")
        
        # Test collaborative change
        print("\n5. Testing collaborative changes...")
        
        collab_change = await collab_manager.create_text_change(
            workspace_id, file_id, ChangeType.INSERT, 0, "# Collaborative Edit\n", author_id="user123"
        )
        
        success, conflict = await collab_manager.apply_change(collab_change)
        print(f"✓ Collaborative change applied: {success}, Conflict: {conflict is not None}")
        
        # Get updated content
        updated_content = collab_manager.get_document_content(file_id)
        print(f"✓ Updated content after collaboration:")
        print(f"  {repr(updated_content)}")
        
        # Test workspace statistics
        print("\n6. Testing workspace statistics...")
        
        workspace_stats = await workspace_manager.get_workspace_stats(workspace_id)
        if workspace_stats:
            print(f"✓ Workspace statistics:")
            print(f"  Name: {workspace_stats['name']}")
            print(f"  Files: {workspace_stats['file_count']}")
            print(f"  Size: {workspace_stats['total_size_bytes']} bytes")
            print(f"  Collaborators: {workspace_stats['collaborator_count']}")
            print(f"  Active: {workspace_stats['active_collaborators']}")
        
        system_stats = await workspace_manager.get_system_stats()
        print(f"✓ System statistics:")
        print(f"  Cached workspaces: {system_stats['cached_workspaces']}")
        print(f"  Active sessions: {system_stats['active_sessions']}")
        print(f"  File locks: {system_stats['file_locks']}")
        
        collab_stats = collab_manager.get_stats()
        print(f"✓ Collaboration statistics:")
        print(f"  Active sessions: {collab_stats['active_sessions']}")
        print(f"  Online collaborators: {collab_stats['online_collaborators']}")
        print(f"  Queued changes: {collab_stats['queued_changes']}")
        print(f"  Pending conflicts: {collab_stats['pending_conflicts']}")
        
        crdt_stats = crdt_engine.get_stats()
        print(f"✓ CRDT engine statistics:")
        print(f"  Registered documents: {crdt_stats['registered_documents']}")
        print(f"  Total operations: {crdt_stats['total_operations']}")
        print(f"  Conflict resolution: {crdt_stats['conflict_resolution_strategy']}")
        
        # Test cleanup
        print("\n7. Testing cleanup...")
        
        # End collaboration sessions
        await collab_manager.end_collaboration_session(workspace_id, "user123")
        await collab_manager.end_collaboration_session(workspace_id, "user456")
        
        # Clean up cache
        cleaned_cache = await workspace_manager.cleanup_expired_cache()
        print(f"✓ Cleaned up {cleaned_cache} expired cache entries")
        
        # Clean up inactive sessions
        cleaned_sessions = await collab_manager.cleanup_inactive_sessions(0)  # Clean all
        print(f"✓ Cleaned up {cleaned_sessions} inactive sessions")
        
        print("\n✓ Workspace management system test completed successfully!")
        
    except Exception as e:
        print(f"✗ Workspace management test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workspace_management())