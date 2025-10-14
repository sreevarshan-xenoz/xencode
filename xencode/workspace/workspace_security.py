#!/usr/bin/env python3
"""
Workspace Security Manager

Implements workspace-level permission controls, data isolation,
and secure workspace switching with context preservation.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from xencode.models.user import User, UserRole, Permission, ResourceType, PermissionType
from xencode.models.workspace import Workspace, WorkspaceCollaborator, WorkspaceStatus


class WorkspacePermission(str, Enum):
    """Workspace-specific permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    COLLABORATE = "collaborate"
    INVITE = "invite"
    EXPORT = "export"


class IsolationLevel(str, Enum):
    """Data isolation levels"""
    STRICT = "strict"      # Complete isolation, no data sharing
    CONTROLLED = "controlled"  # Limited sharing with explicit permissions
    SHARED = "shared"      # Shared resources with access controls


class WorkspaceContext:
    """Represents workspace context for switching"""
    
    def __init__(self, workspace_id: str, user_id: str):
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        
        # Context state
        self.active_files: Set[str] = set()
        self.open_editors: Dict[str, Dict[str, Any]] = {}
        self.cursor_positions: Dict[str, Tuple[int, int]] = {}
        self.selection_ranges: Dict[str, Tuple[int, int, int, int]] = {}
        self.view_state: Dict[str, Any] = {}
        
        # Session data
        self.session_variables: Dict[str, Any] = {}
        self.temporary_data: Dict[str, Any] = {}
        
        # Security context
        self.permissions_cache: Dict[str, bool] = {}
        self.access_token: Optional[str] = None
    
    def update_file_context(self, file_id: str, cursor_line: int, cursor_col: int,
                          selection_start_line: int = 0, selection_start_col: int = 0,
                          selection_end_line: int = 0, selection_end_col: int = 0) -> None:
        """Update file editing context"""
        self.active_files.add(file_id)
        self.cursor_positions[file_id] = (cursor_line, cursor_col)
        self.selection_ranges[file_id] = (selection_start_line, selection_start_col,
                                        selection_end_line, selection_end_col)
        self.last_accessed = datetime.now()
    
    def add_editor_state(self, file_id: str, editor_state: Dict[str, Any]) -> None:
        """Add editor state for file"""
        self.open_editors[file_id] = editor_state
        self.last_accessed = datetime.now()
    
    def set_session_variable(self, key: str, value: Any) -> None:
        """Set session variable"""
        self.session_variables[key] = value
    
    def get_session_variable(self, key: str, default: Any = None) -> Any:
        """Get session variable"""
        return self.session_variables.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        return {
            'workspace_id': self.workspace_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'active_files': list(self.active_files),
            'open_editors': self.open_editors,
            'cursor_positions': {k: list(v) for k, v in self.cursor_positions.items()},
            'selection_ranges': {k: list(v) for k, v in self.selection_ranges.items()},
            'view_state': self.view_state,
            'session_variables': self.session_variables,
            'temporary_data': self.temporary_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceContext':
        """Create context from dictionary"""
        context = cls(data['workspace_id'], data['user_id'])
        context.created_at = datetime.fromisoformat(data['created_at'])
        context.last_accessed = datetime.fromisoformat(data['last_accessed'])
        context.active_files = set(data.get('active_files', []))
        context.open_editors = data.get('open_editors', {})
        context.cursor_positions = {k: tuple(v) for k, v in data.get('cursor_positions', {}).items()}
        context.selection_ranges = {k: tuple(v) for k, v in data.get('selection_ranges', {}).items()}
        context.view_state = data.get('view_state', {})
        context.session_variables = data.get('session_variables', {})
        context.temporary_data = data.get('temporary_data', {})
        return context


class WorkspaceSecurityManager:
    """Manages workspace-level security and isolation"""
    
    def __init__(self, permission_engine=None, audit_logger=None):
        self.permission_engine = permission_engine
        self.audit_logger = audit_logger
        
        # Workspace permissions
        self.workspace_permissions: Dict[str, Dict[str, Set[WorkspacePermission]]] = {}
        
        # Data isolation settings
        self.isolation_levels: Dict[str, IsolationLevel] = {}
        
        # Context management
        self.active_contexts: Dict[str, WorkspaceContext] = {}  # user_id -> context
        self.context_history: Dict[str, List[WorkspaceContext]] = {}  # user_id -> contexts
        
        # Security policies
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        
        # Access control lists
        self.workspace_acls: Dict[str, Dict[str, Set[str]]] = {}  # workspace_id -> {permission -> user_ids}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_data
    
    async def initialize(self) -> None:
        """Initialize security manager"""
        # Load default security policies
        await self._load_default_policies()
        
        # Initialize workspace ACLs
        await self._initialize_workspace_acls()
        
        print("WorkspaceSecurityManager initialized")
    
    async def _load_default_policies(self) -> None:
        """Load default security policies"""
        self.security_policies = {
            'default': {
                'isolation_level': IsolationLevel.CONTROLLED,
                'max_collaborators': 50,
                'session_timeout_minutes': 480,  # 8 hours
                'context_retention_days': 30,
                'require_2fa': False,
                'audit_all_actions': True,
                'data_encryption': True,
                'cross_workspace_access': False
            },
            'enterprise': {
                'isolation_level': IsolationLevel.STRICT,
                'max_collaborators': 100,
                'session_timeout_minutes': 240,  # 4 hours
                'context_retention_days': 90,
                'require_2fa': True,
                'audit_all_actions': True,
                'data_encryption': True,
                'cross_workspace_access': False
            }
        }
    
    async def _initialize_workspace_acls(self) -> None:
        """Initialize workspace access control lists"""
        # This would typically load from database
        pass
    
    async def set_workspace_permissions(self, workspace_id: str, user_id: str, 
                                      permissions: Set[WorkspacePermission]) -> bool:
        """Set permissions for user in workspace"""
        try:
            if workspace_id not in self.workspace_permissions:
                self.workspace_permissions[workspace_id] = {}
            
            self.workspace_permissions[workspace_id][user_id] = permissions
            
            # Update ACL
            if workspace_id not in self.workspace_acls:
                self.workspace_acls[workspace_id] = {}
            
            for permission in WorkspacePermission:
                if workspace_id not in self.workspace_acls:
                    self.workspace_acls[workspace_id] = {}
                if permission.value not in self.workspace_acls[workspace_id]:
                    self.workspace_acls[workspace_id][permission.value] = set()
                
                if permission in permissions:
                    self.workspace_acls[workspace_id][permission.value].add(user_id)
                else:
                    self.workspace_acls[workspace_id][permission.value].discard(user_id)
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="workspace_permissions_updated",
                    user_id=user_id,
                    resource_id=workspace_id,
                    details={
                        'permissions': [p.value for p in permissions],
                        'action': 'set_permissions'
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"Error setting workspace permissions: {e}")
            return False
    
    async def check_workspace_permission(self, workspace_id: str, user_id: str, 
                                       permission: WorkspacePermission) -> bool:
        """Check if user has specific permission in workspace"""
        try:
            # Check direct permissions
            if (workspace_id in self.workspace_permissions and 
                user_id in self.workspace_permissions[workspace_id]):
                user_permissions = self.workspace_permissions[workspace_id][user_id]
                
                # Admin permission grants all permissions
                if WorkspacePermission.ADMIN in user_permissions:
                    return True
                
                # Check specific permission
                if permission in user_permissions:
                    return True
            
            # Check through permission engine if available
            if self.permission_engine:
                return await self.permission_engine.check_permission(
                    user_id, ResourceType.WORKSPACE, workspace_id, permission.value
                )
            
            return False
            
        except Exception as e:
            print(f"Error checking workspace permission: {e}")
            return False
    
    async def enforce_data_isolation(self, workspace_id: str, user_id: str, 
                                   requested_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce data isolation based on workspace settings"""
        try:
            isolation_level = self.isolation_levels.get(workspace_id, IsolationLevel.CONTROLLED)
            
            if isolation_level == IsolationLevel.STRICT:
                # Strict isolation - only return data user has explicit access to
                return await self._filter_strict_isolation(workspace_id, user_id, requested_data)
            
            elif isolation_level == IsolationLevel.CONTROLLED:
                # Controlled isolation - apply access controls
                return await self._filter_controlled_isolation(workspace_id, user_id, requested_data)
            
            else:  # SHARED
                # Shared isolation - minimal filtering
                return await self._filter_shared_isolation(workspace_id, user_id, requested_data)
            
        except Exception as e:
            print(f"Error enforcing data isolation: {e}")
            return {}
    
    async def _filter_strict_isolation(self, workspace_id: str, user_id: str, 
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict isolation filtering"""
        filtered_data = {}
        
        for key, value in data.items():
            # Only include data if user has explicit read permission
            if await self.check_workspace_permission(workspace_id, user_id, WorkspacePermission.READ):
                # Additional checks for sensitive data
                if not self._is_sensitive_data(key, value):
                    filtered_data[key] = value
        
        return filtered_data
    
    async def _filter_controlled_isolation(self, workspace_id: str, user_id: str, 
                                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply controlled isolation filtering"""
        filtered_data = {}
        
        for key, value in data.items():
            # Check if user can access this data
            if await self._can_access_data(workspace_id, user_id, key):
                filtered_data[key] = self._sanitize_data(value)
        
        return filtered_data
    
    async def _filter_shared_isolation(self, workspace_id: str, user_id: str, 
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply shared isolation filtering"""
        # Minimal filtering for shared workspaces
        return {k: self._sanitize_data(v) for k, v in data.items()}
    
    def _is_sensitive_data(self, key: str, value: Any) -> bool:
        """Check if data is considered sensitive"""
        sensitive_keys = {
            'password', 'token', 'secret', 'key', 'credential',
            'private', 'confidential', 'internal'
        }
        
        return any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
    
    async def _can_access_data(self, workspace_id: str, user_id: str, data_key: str) -> bool:
        """Check if user can access specific data"""
        # Implement data-level access control logic
        return await self.check_workspace_permission(workspace_id, user_id, WorkspacePermission.READ)
    
    def _sanitize_data(self, value: Any) -> Any:
        """Sanitize data for security"""
        if isinstance(value, str):
            # Remove potential sensitive information
            if any(keyword in value.lower() for keyword in ['password', 'token', 'secret']):
                return '[REDACTED]'
        
        return value
    
    async def switch_workspace_context(self, user_id: str, new_workspace_id: str, 
                                     preserve_context: bool = True) -> Optional[WorkspaceContext]:
        """Switch user to different workspace with context preservation"""
        try:
            # Check permission to access new workspace
            if not await self.check_workspace_permission(new_workspace_id, user_id, WorkspacePermission.READ):
                raise PermissionError(f"User {user_id} does not have access to workspace {new_workspace_id}")
            
            # Save current context if preserving
            if preserve_context and user_id in self.active_contexts:
                current_context = self.active_contexts[user_id]
                await self._save_context(current_context)
            
            # Load or create new context
            new_context = await self._load_or_create_context(user_id, new_workspace_id)
            
            # Set as active context
            self.active_contexts[user_id] = new_context
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="workspace_context_switch",
                    user_id=user_id,
                    resource_id=new_workspace_id,
                    details={
                        'previous_workspace': getattr(self.active_contexts.get(user_id), 'workspace_id', None),
                        'preserve_context': preserve_context
                    }
                )
            
            return new_context
            
        except Exception as e:
            print(f"Error switching workspace context: {e}")
            return None
    
    async def _save_context(self, context: WorkspaceContext) -> bool:
        """Save workspace context"""
        try:
            user_id = context.user_id
            
            # Add to context history
            if user_id not in self.context_history:
                self.context_history[user_id] = []
            
            self.context_history[user_id].append(context)
            
            # Keep only recent contexts (based on retention policy)
            policy = self.security_policies.get('default', {})
            retention_days = policy.get('context_retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            self.context_history[user_id] = [
                ctx for ctx in self.context_history[user_id]
                if ctx.last_accessed > cutoff_date
            ]
            
            # In a real implementation, this would save to persistent storage
            return True
            
        except Exception as e:
            print(f"Error saving context: {e}")
            return False
    
    async def _load_or_create_context(self, user_id: str, workspace_id: str) -> WorkspaceContext:
        """Load existing context or create new one"""
        try:
            # Try to find existing context in history
            if user_id in self.context_history:
                for context in reversed(self.context_history[user_id]):
                    if context.workspace_id == workspace_id:
                        # Update last accessed and return
                        context.last_accessed = datetime.now()
                        return context
            
            # Create new context
            return WorkspaceContext(workspace_id, user_id)
            
        except Exception as e:
            print(f"Error loading/creating context: {e}")
            return WorkspaceContext(workspace_id, user_id)
    
    async def get_workspace_security_status(self, workspace_id: str) -> Dict[str, Any]:
        """Get security status for workspace"""
        try:
            # Count active users
            active_users = set()
            for context in self.active_contexts.values():
                if context.workspace_id == workspace_id:
                    active_users.add(context.user_id)
            
            # Get permissions summary
            permissions_summary = {}
            if workspace_id in self.workspace_permissions:
                for user_id, perms in self.workspace_permissions[workspace_id].items():
                    permissions_summary[user_id] = [p.value for p in perms]
            
            return {
                'workspace_id': workspace_id,
                'isolation_level': self.isolation_levels.get(workspace_id, IsolationLevel.CONTROLLED).value,
                'active_users': len(active_users),
                'total_users_with_access': len(permissions_summary),
                'permissions_summary': permissions_summary,
                'security_policy': self.security_policies.get('default', {}),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting workspace security status: {e}")
            return {}
    
    async def revoke_workspace_access(self, workspace_id: str, user_id: str, 
                                    revoked_by: str) -> bool:
        """Revoke all access to workspace for user"""
        try:
            # Check if revoker has admin permission
            if not await self.check_workspace_permission(workspace_id, revoked_by, WorkspacePermission.ADMIN):
                return False
            
            # Remove all permissions
            if (workspace_id in self.workspace_permissions and 
                user_id in self.workspace_permissions[workspace_id]):
                del self.workspace_permissions[workspace_id][user_id]
            
            # Update ACLs
            if workspace_id in self.workspace_acls:
                for permission_set in self.workspace_acls[workspace_id].values():
                    permission_set.discard(user_id)
            
            # Terminate active sessions
            if user_id in self.active_contexts:
                context = self.active_contexts[user_id]
                if context.workspace_id == workspace_id:
                    await self._save_context(context)
                    del self.active_contexts[user_id]
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="workspace_access_revoked",
                    user_id=revoked_by,
                    resource_id=workspace_id,
                    details={
                        'revoked_user': user_id,
                        'action': 'revoke_access'
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"Error revoking workspace access: {e}")
            return False
    
    async def cleanup_inactive_contexts(self, timeout_minutes: int = 480) -> int:
        """Clean up inactive workspace contexts"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
            inactive_users = []
            
            for user_id, context in self.active_contexts.items():
                if context.last_accessed < cutoff_time:
                    inactive_users.append(user_id)
            
            # Save and remove inactive contexts
            for user_id in inactive_users:
                context = self.active_contexts[user_id]
                await self._save_context(context)
                del self.active_contexts[user_id]
            
            return len(inactive_users)
            
        except Exception as e:
            print(f"Error cleaning up inactive contexts: {e}")
            return 0
    
    def get_active_workspace_contexts(self) -> Dict[str, List[str]]:
        """Get active contexts grouped by workspace"""
        workspace_contexts = {}
        
        for user_id, context in self.active_contexts.items():
            workspace_id = context.workspace_id
            if workspace_id not in workspace_contexts:
                workspace_contexts[workspace_id] = []
            workspace_contexts[workspace_id].append(user_id)
        
        return workspace_contexts


# Global workspace security manager instance
workspace_security_manager = WorkspaceSecurityManager()