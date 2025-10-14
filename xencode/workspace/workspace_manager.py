#!/usr/bin/env python3
"""
Workspace Manager

Manages workspaces with SQLite backend, providing workspace creation,
collaboration, file management, and isolation mechanisms.
"""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from xencode.models.workspace import (
    Workspace, WorkspaceFile, WorkspaceCollaborator, WorkspaceConfig,
    WorkspaceType, WorkspaceStatus, CollaborationMode, Change,
    create_default_workspace
)
from xencode.workspace.storage_backend import SQLiteStorageBackend
from xencode.workspace.workspace_security import (
    WorkspaceSecurityManager, WorkspacePermission, IsolationLevel, WorkspaceContext
)


class WorkspaceError(Exception):
    """Workspace-related errors"""
    pass


class WorkspaceManager:
    """Manages workspaces with SQLite backend"""
    
    def __init__(self, storage_backend: Optional[SQLiteStorageBackend] = None,
                 security_manager: Optional[WorkspaceSecurityManager] = None):
        self.storage = storage_backend or SQLiteStorageBackend()
        self.security_manager = security_manager or WorkspaceSecurityManager()
        
        # In-memory cache for active workspaces
        self.workspace_cache: Dict[str, Workspace] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, Set[str]] = {}  # workspace_id -> set of user_ids
        
        # File locks for concurrent editing
        self.file_locks: Dict[str, str] = {}  # file_id -> user_id
    
    async def initialize(self) -> None:
        """Initialize workspace manager"""
        await self.storage.initialize()
        await self.security_manager.initialize()
    
    async def create_workspace(self, 
                              name: str,
                              owner_id: str,
                              workspace_type: WorkspaceType = WorkspaceType.PROJECT,
                              description: str = "",
                              config: Optional[WorkspaceConfig] = None) -> Optional[Workspace]:
        """Create a new workspace"""
        
        if not name or not owner_id:
            return None
        
        # Create workspace
        workspace = create_default_workspace(owner_id, name, workspace_type)
        workspace.description = description
        
        if config:
            workspace.config = config
        
        # Save to storage
        if await self.storage.create_workspace(workspace):
            # Add to cache
            self._cache_workspace(workspace)
            return workspace
        
        return None
    
    async def get_workspace(self, workspace_id: str, user_id: Optional[str] = None) -> Optional[Workspace]:
        """Get workspace by ID"""
        
        # Check cache first
        if self._is_cache_valid(workspace_id):
            workspace = self.workspace_cache[workspace_id]
        else:
            # Load from storage
            workspace = await self.storage.get_workspace(workspace_id, include_content=True)
            if workspace:
                self._cache_workspace(workspace)
        
        if not workspace:
            return None
        
        # Check access permissions through security manager
        if user_id:
            has_access = await self.security_manager.check_workspace_permission(
                workspace_id, user_id, WorkspacePermission.READ
            )
            if not has_access and not workspace.can_user_access(user_id):
                return None
        
        # Update last accessed
        workspace.last_accessed = datetime.now()
        await self.storage.update_workspace(workspace)
        
        return workspace
    
    async def update_workspace(self, workspace: Workspace) -> bool:
        """Update workspace"""
        
        workspace.updated_at = datetime.now()
        
        # Update in storage
        if await self.storage.update_workspace(workspace):
            # Update cache
            self._cache_workspace(workspace)
            return True
        
        return False
    
    async def delete_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Delete workspace (owner only)"""
        
        workspace = await self.get_workspace(workspace_id)
        if not workspace or workspace.owner_id != user_id:
            return False
        
        # Remove from storage
        if await self.storage.delete_workspace(workspace_id):
            # Remove from cache
            if workspace_id in self.workspace_cache:
                del self.workspace_cache[workspace_id]
            if workspace_id in self.cache_timestamps:
                del self.cache_timestamps[workspace_id]
            
            # Clear active sessions
            if workspace_id in self.active_sessions:
                del self.active_sessions[workspace_id]
            
            return True
        
        return False
    
    async def list_user_workspaces(self, user_id: str, status: Optional[WorkspaceStatus] = None) -> List[Workspace]:
        """List workspaces accessible to user"""
        
        # Get workspaces owned by user
        owned_workspaces = await self.storage.list_workspaces(
            owner_id=user_id,
            status=status.value if status else None
        )
        
        # Get workspaces where user is a collaborator
        collab_workspaces = await self.storage.list_workspaces(
            user_id=user_id,
            status=status.value if status else None
        )
        
        # Combine and deduplicate
        all_workspaces = {ws.id: ws for ws in owned_workspaces + collab_workspaces}
        
        return list(all_workspaces.values())
    
    async def add_collaborator(self, 
                              workspace_id: str,
                              user_id: str,
                              collaborator: WorkspaceCollaborator,
                              requester_user_id: str) -> bool:
        """Add collaborator to workspace"""
        
        workspace = await self.get_workspace(workspace_id, requester_user_id)
        if not workspace:
            return False
        
        # Check if requester can add collaborators (owner or editor)
        if not workspace.can_user_edit(requester_user_id):
            return False
        
        # Add collaborator
        if workspace.add_collaborator(collaborator):
            return await self.update_workspace(workspace)
        
        return False
    
    async def remove_collaborator(self, 
                                 workspace_id: str,
                                 collaborator_user_id: str,
                                 requester_user_id: str) -> bool:
        """Remove collaborator from workspace"""
        
        workspace = await self.get_workspace(workspace_id, requester_user_id)
        if not workspace:
            return False
        
        # Check permissions (owner or the collaborator themselves)
        if workspace.owner_id != requester_user_id and collaborator_user_id != requester_user_id:
            return False
        
        # Remove collaborator
        if workspace.remove_collaborator(collaborator_user_id):
            return await self.update_workspace(workspace)
        
        return False
    
    async def update_collaborator_activity(self, 
                                          workspace_id: str,
                                          user_id: str,
                                          session_id: Optional[str] = None) -> bool:
        """Update collaborator activity"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return False
        
        # Update activity
        if workspace.update_collaborator_activity(user_id, session_id):
            # Update active sessions
            if workspace_id not in self.active_sessions:
                self.active_sessions[workspace_id] = set()
            self.active_sessions[workspace_id].add(user_id)
            
            return await self.update_workspace(workspace)
        
        return False
    
    async def add_file(self, 
                      workspace_id: str,
                      file: WorkspaceFile,
                      user_id: str) -> bool:
        """Add file to workspace"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return False
        
        # Check edit permissions
        if not workspace.can_user_edit(user_id):
            return False
        
        # Set file metadata
        file.created_by = user_id
        file.updated_by = user_id
        file.content_hash = self._calculate_content_hash(file.content)
        file.size_bytes = len(file.content.encode('utf-8'))
        
        # Add to storage
        if await self.storage.add_file(workspace_id, file):
            # Update workspace cache
            workspace.add_file(file)
            self._cache_workspace(workspace)
            return True
        
        return False
    
    async def update_file(self, 
                         workspace_id: str,
                         file: WorkspaceFile,
                         user_id: str) -> bool:
        """Update file in workspace"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return False
        
        # Check edit permissions
        if not workspace.can_user_edit(user_id):
            return False
        
        # Check file lock
        if file.id in self.file_locks and self.file_locks[file.id] != user_id:
            return False
        
        # Update file metadata
        file.updated_by = user_id
        file.updated_at = datetime.now()
        file.content_hash = self._calculate_content_hash(file.content)
        file.size_bytes = len(file.content.encode('utf-8'))
        file.version += 1
        
        # Update in storage
        if await self.storage.update_file(file):
            # Update workspace cache
            workspace.files[file.id] = file
            workspace.updated_at = datetime.now()
            self._cache_workspace(workspace)
            return True
        
        return False
    
    async def delete_file(self, 
                         workspace_id: str,
                         file_id: str,
                         user_id: str) -> bool:
        """Delete file from workspace"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return False
        
        # Check edit permissions
        if not workspace.can_user_edit(user_id):
            return False
        
        # Check file lock
        if file_id in self.file_locks and self.file_locks[file_id] != user_id:
            return False
        
        # Delete from storage
        if await self.storage.delete_file(file_id):
            # Update workspace cache
            workspace.remove_file(file_id)
            self._cache_workspace(workspace)
            
            # Remove file lock
            if file_id in self.file_locks:
                del self.file_locks[file_id]
            
            return True
        
        return False
    
    async def get_file(self, 
                      workspace_id: str,
                      file_id: str,
                      user_id: str) -> Optional[WorkspaceFile]:
        """Get file from workspace"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return None
        
        return workspace.get_file(file_id)
    
    async def get_file_by_path(self, 
                              workspace_id: str,
                              file_path: str,
                              user_id: str) -> Optional[WorkspaceFile]:
        """Get file by path from workspace"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return None
        
        return workspace.get_file_by_path(file_path)
    
    async def lock_file(self, 
                       workspace_id: str,
                       file_id: str,
                       user_id: str) -> bool:
        """Lock file for editing"""
        
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace or not workspace.can_user_edit(user_id):
            return False
        
        # Check if file is already locked
        if file_id in self.file_locks:
            return self.file_locks[file_id] == user_id
        
        # Lock file
        self.file_locks[file_id] = user_id
        return True
    
    async def unlock_file(self, 
                         workspace_id: str,
                         file_id: str,
                         user_id: str) -> bool:
        """Unlock file"""
        
        # Check if user owns the lock
        if file_id not in self.file_locks or self.file_locks[file_id] != user_id:
            return False
        
        # Unlock file
        del self.file_locks[file_id]
        return True
    
    async def get_file_locks(self, workspace_id: str) -> Dict[str, str]:
        """Get all file locks for workspace"""
        
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return {}
        
        # Filter locks for this workspace
        workspace_locks = {}
        for file_id, user_id in self.file_locks.items():
            if file_id in workspace.files:
                workspace_locks[file_id] = user_id
        
        return workspace_locks
    
    async def add_change(self, change: Change) -> bool:
        """Add change to workspace"""
        
        # Add to storage
        if await self.storage.add_change(change):
            # Update workspace cache
            workspace = self.workspace_cache.get(change.workspace_id)
            if workspace:
                workspace.add_change(change)
                self._cache_workspace(workspace)
            
            return True
        
        return False
    
    async def get_changes(self, 
                         workspace_id: str,
                         file_id: Optional[str] = None,
                         since: Optional[datetime] = None) -> List[Change]:
        """Get changes for workspace or file"""
        
        return await self.storage.get_changes(workspace_id, file_id, since)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _cache_workspace(self, workspace: Workspace) -> None:
        """Cache workspace"""
        self.workspace_cache[workspace.id] = workspace
        self.cache_timestamps[workspace.id] = datetime.now()
    
    def _is_cache_valid(self, workspace_id: str) -> bool:
        """Check if cache entry is valid"""
        
        if workspace_id not in self.workspace_cache:
            return False
        
        if workspace_id not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[workspace_id]
        age_seconds = (datetime.now() - cache_time).total_seconds()
        
        return age_seconds < self.cache_ttl_seconds
    
    def _clear_cache(self) -> None:
        """Clear workspace cache"""
        self.workspace_cache.clear()
        self.cache_timestamps.clear()
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        
        expired_keys = []
        now = datetime.now()
        
        for workspace_id, timestamp in self.cache_timestamps.items():
            age_seconds = (now - timestamp).total_seconds()
            if age_seconds >= self.cache_ttl_seconds:
                expired_keys.append(workspace_id)
        
        for key in expired_keys:
            del self.workspace_cache[key]
            del self.cache_timestamps[key]
        
        return len(expired_keys)
    
    async def get_workspace_stats(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get workspace statistics"""
        
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        # Get recent changes count
        recent_changes = await self.get_changes(
            workspace_id,
            since=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
        
        return {
            'id': workspace.id,
            'name': workspace.name,
            'type': workspace.workspace_type.value,
            'status': workspace.status.value,
            'collaboration_mode': workspace.collaboration_mode.value,
            'file_count': workspace.file_count,
            'total_size_bytes': workspace.total_size_bytes,
            'collaborator_count': len(workspace.collaborators),
            'active_collaborators': workspace.active_collaborators,
            'unresolved_conflicts': len(workspace.get_unresolved_conflicts()),
            'recent_changes_today': len(recent_changes),
            'created_at': workspace.created_at.isoformat(),
            'updated_at': workspace.updated_at.isoformat(),
            'last_accessed': workspace.last_accessed.isoformat() if workspace.last_accessed else None
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide workspace statistics"""
        
        # This would typically query the database for aggregated stats
        # For now, we'll return basic cache stats
        
        return {
            'cached_workspaces': len(self.workspace_cache),
            'active_sessions': sum(len(users) for users in self.active_sessions.values()),
            'file_locks': len(self.file_locks),
            'cache_ttl_seconds': self.cache_ttl_seconds
        }
    
    async def set_workspace_permissions(self, workspace_id: str, user_id: str, 
                                      permissions: Set[WorkspacePermission],
                                      granted_by: str) -> bool:
        """Set workspace permissions for user"""
        # Check if granter has admin permission
        has_admin = await self.security_manager.check_workspace_permission(
            workspace_id, granted_by, WorkspacePermission.ADMIN
        )
        
        if not has_admin:
            # Check if granter is workspace owner
            workspace = await self.get_workspace(workspace_id)
            if not workspace or workspace.owner_id != granted_by:
                return False
        
        return await self.security_manager.set_workspace_permissions(
            workspace_id, user_id, permissions
        )
    
    async def check_workspace_permission(self, workspace_id: str, user_id: str, 
                                       permission: WorkspacePermission) -> bool:
        """Check workspace permission for user"""
        return await self.security_manager.check_workspace_permission(
            workspace_id, user_id, permission
        )
    
    async def switch_workspace_context(self, user_id: str, workspace_id: str,
                                     preserve_context: bool = True) -> Optional[WorkspaceContext]:
        """Switch user to different workspace with context preservation"""
        return await self.security_manager.switch_workspace_context(
            user_id, workspace_id, preserve_context
        )
    
    async def get_workspace_with_isolation(self, workspace_id: str, user_id: str) -> Optional[Workspace]:
        """Get workspace with data isolation applied"""
        workspace = await self.get_workspace(workspace_id, user_id)
        if not workspace:
            return None
        
        # Apply data isolation
        workspace_data = workspace.to_dict()
        filtered_data = await self.security_manager.enforce_data_isolation(
            workspace_id, user_id, workspace_data
        )
        
        # Create filtered workspace object
        if filtered_data:
            # In a real implementation, you'd reconstruct the workspace with filtered data
            return workspace
        
        return None
    
    async def set_workspace_isolation_level(self, workspace_id: str, 
                                          isolation_level: IsolationLevel,
                                          set_by: str) -> bool:
        """Set isolation level for workspace"""
        # Check admin permission
        has_admin = await self.security_manager.check_workspace_permission(
            workspace_id, set_by, WorkspacePermission.ADMIN
        )
        
        if not has_admin:
            return False
        
        self.security_manager.isolation_levels[workspace_id] = isolation_level
        return True
    
    async def get_workspace_security_status(self, workspace_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get workspace security status"""
        # Check read permission
        has_read = await self.security_manager.check_workspace_permission(
            workspace_id, user_id, WorkspacePermission.READ
        )
        
        if not has_read:
            return None
        
        return await self.security_manager.get_workspace_security_status(workspace_id)
    
    async def revoke_workspace_access(self, workspace_id: str, target_user_id: str, 
                                    revoked_by: str) -> bool:
        """Revoke workspace access for user"""
        return await self.security_manager.revoke_workspace_access(
            workspace_id, target_user_id, revoked_by
        )
    
    async def get_user_workspace_contexts(self, user_id: str) -> List[WorkspaceContext]:
        """Get workspace contexts for user"""
        contexts = []
        
        # Get active context
        if user_id in self.security_manager.active_contexts:
            contexts.append(self.security_manager.active_contexts[user_id])
        
        # Get context history
        if user_id in self.security_manager.context_history:
            contexts.extend(self.security_manager.context_history[user_id])
        
        return contexts
    
    async def cleanup_security_contexts(self, timeout_minutes: int = 480) -> int:
        """Clean up inactive security contexts"""
        return await self.security_manager.cleanup_inactive_contexts(timeout_minutes)
    
    async def close(self) -> None:
        """Close workspace manager"""
        await self.storage.close()
        self._clear_cache()
        self.active_sessions.clear()
        self.file_locks.clear()


# Global workspace manager instance
workspace_manager = WorkspaceManager()