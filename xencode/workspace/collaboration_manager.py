#!/usr/bin/env python3
"""
Collaboration Manager

Manages real-time collaboration features including presence awareness,
conflict resolution, and synchronization between collaborators.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from xencode.models.workspace import (
    Workspace, WorkspaceCollaborator, Change, Conflict, ChangeType
)
from xencode.workspace.crdt_engine import CRDTEngine
from xencode.workspace.workspace_manager import WorkspaceManager


class CollaboratorPresence:
    """Tracks collaborator presence and activity"""
    
    def __init__(self, user_id: str, username: str):
        self.user_id = user_id
        self.username = username
        self.is_online = False
        self.last_seen = datetime.now()
        self.current_file = None
        self.cursor_position = 0
        self.selection_start = 0
        self.selection_end = 0
        self.session_id = None
    
    def update_activity(self, file_id: Optional[str] = None, cursor_pos: int = 0) -> None:
        """Update collaborator activity"""
        self.last_seen = datetime.now()
        self.is_online = True
        if file_id:
            self.current_file = file_id
        self.cursor_position = cursor_pos
    
    def set_selection(self, start: int, end: int) -> None:
        """Set text selection"""
        self.selection_start = start
        self.selection_end = end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'is_online': self.is_online,
            'last_seen': self.last_seen.isoformat(),
            'current_file': self.current_file,
            'cursor_position': self.cursor_position,
            'selection_start': self.selection_start,
            'selection_end': self.selection_end,
            'session_id': self.session_id
        }


class CollaborationSession:
    """Manages a collaboration session for a workspace"""
    
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.collaborators: Dict[str, CollaboratorPresence] = {}
        self.active_changes: List[Change] = []
        self.pending_conflicts: List[Conflict] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_collaborator(self, user_id: str, username: str, session_id: str) -> None:
        """Add collaborator to session"""
        presence = CollaboratorPresence(user_id, username)
        presence.session_id = session_id
        presence.is_online = True
        self.collaborators[user_id] = presence
        self.last_activity = datetime.now()
    
    def remove_collaborator(self, user_id: str) -> None:
        """Remove collaborator from session"""
        if user_id in self.collaborators:
            del self.collaborators[user_id]
            self.last_activity = datetime.now()
    
    def update_collaborator_presence(self, user_id: str, file_id: Optional[str] = None, cursor_pos: int = 0) -> bool:
        """Update collaborator presence"""
        if user_id in self.collaborators:
            self.collaborators[user_id].update_activity(file_id, cursor_pos)
            self.last_activity = datetime.now()
            return True
        return False
    
    def get_online_collaborators(self) -> List[CollaboratorPresence]:
        """Get online collaborators"""
        cutoff = datetime.now() - timedelta(minutes=5)  # 5 minute timeout
        online = []
        
        for presence in self.collaborators.values():
            if presence.is_online and presence.last_seen > cutoff:
                online.append(presence)
            elif presence.last_seen <= cutoff:
                presence.is_online = False
        
        return online
    
    def add_change(self, change: Change) -> None:
        """Add change to session"""
        self.active_changes.append(change)
        self.last_activity = datetime.now()
        
        # Keep only recent changes (last 100)
        if len(self.active_changes) > 100:
            self.active_changes = self.active_changes[-100:]
    
    def add_conflict(self, conflict: Conflict) -> None:
        """Add conflict to session"""
        self.pending_conflicts.append(conflict)
        self.last_activity = datetime.now()
    
    def resolve_conflict(self, conflict_id: str, resolution_change: Change) -> bool:
        """Resolve conflict"""
        for i, conflict in enumerate(self.pending_conflicts):
            if conflict.id == conflict_id:
                conflict.resolved = True
                conflict.resolved_at = datetime.now()
                conflict.resolution_change = resolution_change
                self.pending_conflicts.pop(i)
                self.last_activity = datetime.now()
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        online_count = len(self.get_online_collaborators())
        
        return {
            'workspace_id': self.workspace_id,
            'total_collaborators': len(self.collaborators),
            'online_collaborators': online_count,
            'active_changes': len(self.active_changes),
            'pending_conflicts': len(self.pending_conflicts),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }


class CollaborationManager:
    """Manages real-time collaboration across workspaces"""
    
    def __init__(self, 
                 workspace_manager: Optional[WorkspaceManager] = None,
                 crdt_engine: Optional[CRDTEngine] = None):
        self.workspace_manager = workspace_manager
        self.crdt_engine = crdt_engine or CRDTEngine()
        
        # Active collaboration sessions
        self.sessions: Dict[str, CollaborationSession] = {}  # workspace_id -> session
        
        # Change synchronization
        self.sync_queue: Dict[str, List[Change]] = {}  # workspace_id -> changes
        self.sync_callbacks: Dict[str, List[callable]] = {}  # workspace_id -> callbacks
        
        # Conflict resolution
        self.auto_resolve_conflicts = True
        self.conflict_callbacks: List[callable] = []
    
    async def start_collaboration_session(self, 
                                        workspace_id: str,
                                        user_id: str,
                                        username: str,
                                        session_id: str) -> bool:
        """Start collaboration session for user"""
        
        # Create session if it doesn't exist
        if workspace_id not in self.sessions:
            self.sessions[workspace_id] = CollaborationSession(workspace_id)
            self.sync_queue[workspace_id] = []
            self.sync_callbacks[workspace_id] = []
        
        # Add collaborator to session
        session = self.sessions[workspace_id]
        session.add_collaborator(user_id, username, session_id)
        
        # Initialize CRDT for workspace files if workspace manager available
        if self.workspace_manager:
            workspace = await self.workspace_manager.get_workspace(workspace_id, user_id)
            if workspace:
                for file_id, file in workspace.files.items():
                    self.crdt_engine.register_document(file_id, file.content)
        
        return True
    
    async def end_collaboration_session(self, workspace_id: str, user_id: str) -> bool:
        """End collaboration session for user"""
        
        if workspace_id not in self.sessions:
            return False
        
        session = self.sessions[workspace_id]
        session.remove_collaborator(user_id)
        
        # Remove session if no collaborators left
        if not session.get_online_collaborators():
            del self.sessions[workspace_id]
            if workspace_id in self.sync_queue:
                del self.sync_queue[workspace_id]
            if workspace_id in self.sync_callbacks:
                del self.sync_callbacks[workspace_id]
        
        return True
    
    async def update_presence(self, 
                            workspace_id: str,
                            user_id: str,
                            file_id: Optional[str] = None,
                            cursor_position: int = 0,
                            selection_start: int = 0,
                            selection_end: int = 0) -> bool:
        """Update collaborator presence"""
        
        if workspace_id not in self.sessions:
            return False
        
        session = self.sessions[workspace_id]
        success = session.update_collaborator_presence(user_id, file_id, cursor_position)
        
        if success and user_id in session.collaborators:
            presence = session.collaborators[user_id]
            presence.set_selection(selection_start, selection_end)
        
        return success
    
    async def apply_change(self, change: Change) -> Tuple[bool, Optional[Conflict]]:
        """Apply change with CRDT conflict resolution"""
        
        workspace_id = change.workspace_id
        
        # Apply change through CRDT engine
        success, conflict = await self.crdt_engine.apply_change(change)
        
        # Add to session if exists
        if workspace_id in self.sessions:
            session = self.sessions[workspace_id]
            session.add_change(change)
            
            if conflict:
                session.add_conflict(conflict)
                
                # Auto-resolve if enabled
                if self.auto_resolve_conflicts:
                    resolved_change = await self._auto_resolve_conflict(conflict)
                    if resolved_change:
                        session.resolve_conflict(conflict.id, resolved_change)
                        conflict = None
        
        # Add to sync queue
        if workspace_id not in self.sync_queue:
            self.sync_queue[workspace_id] = []
        self.sync_queue[workspace_id].append(change)
        
        # Notify sync callbacks
        await self._notify_sync_callbacks(workspace_id, [change])
        
        return success, conflict
    
    async def _auto_resolve_conflict(self, conflict: Conflict) -> Optional[Change]:
        """Automatically resolve conflict"""
        
        # Use CRDT engine's conflict resolver
        resolved_change = await self.crdt_engine.conflict_resolver.resolve_conflict(
            conflict.change_a, conflict.change_b
        )
        
        if resolved_change:
            # Apply resolved change
            await self.crdt_engine.apply_change(resolved_change)
        
        return resolved_change
    
    async def sync_changes(self, workspace_id: str, since: Optional[datetime] = None) -> List[Change]:
        """Get changes for synchronization"""
        
        if workspace_id not in self.sync_queue:
            return []
        
        changes = self.sync_queue[workspace_id]
        
        if since:
            changes = [change for change in changes if change.timestamp > since]
        
        return changes.copy()
    
    async def broadcast_change(self, change: Change) -> None:
        """Broadcast change to all collaborators"""
        
        workspace_id = change.workspace_id
        
        if workspace_id in self.sync_callbacks:
            for callback in self.sync_callbacks[workspace_id]:
                try:
                    await callback(change)
                except Exception as e:
                    print(f"Error in sync callback: {e}")
    
    def add_sync_callback(self, workspace_id: str, callback: callable) -> None:
        """Add sync callback for workspace"""
        
        if workspace_id not in self.sync_callbacks:
            self.sync_callbacks[workspace_id] = []
        
        self.sync_callbacks[workspace_id].append(callback)
    
    def remove_sync_callback(self, workspace_id: str, callback: callable) -> None:
        """Remove sync callback"""
        
        if workspace_id in self.sync_callbacks:
            callbacks = self.sync_callbacks[workspace_id]
            if callback in callbacks:
                callbacks.remove(callback)
    
    async def _notify_sync_callbacks(self, workspace_id: str, changes: List[Change]) -> None:
        """Notify sync callbacks"""
        
        if workspace_id in self.sync_callbacks:
            for callback in self.sync_callbacks[workspace_id]:
                try:
                    await callback(changes)
                except Exception as e:
                    print(f"Error in sync callback: {e}")
    
    def get_collaborators(self, workspace_id: str) -> List[CollaboratorPresence]:
        """Get collaborators for workspace"""
        
        if workspace_id not in self.sessions:
            return []
        
        return self.sessions[workspace_id].get_online_collaborators()
    
    def get_workspace_presence(self, workspace_id: str) -> Dict[str, Any]:
        """Get presence information for workspace"""
        
        if workspace_id not in self.sessions:
            return {
                'collaborators': [],
                'online_count': 0,
                'active_files': {}
            }
        
        session = self.sessions[workspace_id]
        collaborators = session.get_online_collaborators()
        
        # Group by active file
        active_files = {}
        for presence in collaborators:
            if presence.current_file:
                if presence.current_file not in active_files:
                    active_files[presence.current_file] = []
                active_files[presence.current_file].append({
                    'user_id': presence.user_id,
                    'username': presence.username,
                    'cursor_position': presence.cursor_position,
                    'selection_start': presence.selection_start,
                    'selection_end': presence.selection_end
                })
        
        return {
            'collaborators': [c.to_dict() for c in collaborators],
            'online_count': len(collaborators),
            'active_files': active_files
        }
    
    def get_pending_conflicts(self, workspace_id: str) -> List[Conflict]:
        """Get pending conflicts for workspace"""
        
        if workspace_id not in self.sessions:
            return []
        
        return self.sessions[workspace_id].pending_conflicts.copy()
    
    async def resolve_conflict_manually(self, 
                                      workspace_id: str,
                                      conflict_id: str,
                                      resolution_change: Change,
                                      user_id: str) -> bool:
        """Manually resolve conflict"""
        
        if workspace_id not in self.sessions:
            return False
        
        session = self.sessions[workspace_id]
        
        # Mark resolution change author
        resolution_change.author_id = user_id
        resolution_change.timestamp = datetime.now()
        
        # Apply resolution change
        success, _ = await self.crdt_engine.apply_change(resolution_change)
        
        if success:
            # Mark conflict as resolved
            return session.resolve_conflict(conflict_id, resolution_change)
        
        return False
    
    async def create_text_change(self, 
                               workspace_id: str,
                               file_id: str,
                               change_type: ChangeType,
                               position: int,
                               content: str = "",
                               length: int = 0,
                               author_id: str = "") -> Change:
        """Create text change for collaboration"""
        
        return self.crdt_engine.create_change(
            workspace_id, file_id, change_type, position, content, length, author_id
        )
    
    def get_document_content(self, file_id: str) -> str:
        """Get current document content from CRDT"""
        return self.crdt_engine.get_document_content(file_id)
    
    async def cleanup_inactive_sessions(self, timeout_minutes: int = 30) -> int:
        """Clean up inactive collaboration sessions"""
        
        cutoff = datetime.now() - timedelta(minutes=timeout_minutes)
        inactive_sessions = []
        
        for workspace_id, session in self.sessions.items():
            if session.last_activity < cutoff:
                inactive_sessions.append(workspace_id)
        
        for workspace_id in inactive_sessions:
            del self.sessions[workspace_id]
            if workspace_id in self.sync_queue:
                del self.sync_queue[workspace_id]
            if workspace_id in self.sync_callbacks:
                del self.sync_callbacks[workspace_id]
        
        return len(inactive_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collaboration manager statistics"""
        
        total_collaborators = sum(len(session.collaborators) for session in self.sessions.values())
        total_online = sum(len(session.get_online_collaborators()) for session in self.sessions.values())
        total_changes = sum(len(changes) for changes in self.sync_queue.values())
        total_conflicts = sum(len(session.pending_conflicts) for session in self.sessions.values())
        
        return {
            'active_sessions': len(self.sessions),
            'total_collaborators': total_collaborators,
            'online_collaborators': total_online,
            'queued_changes': total_changes,
            'pending_conflicts': total_conflicts,
            'auto_resolve_conflicts': self.auto_resolve_conflicts,
            'crdt_stats': self.crdt_engine.get_stats()
        }


# Global collaboration manager instance
collaboration_manager = CollaborationManager()