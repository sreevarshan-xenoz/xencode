#!/usr/bin/env python3
"""
Permission Engine

Implements role-based access control with resource-level permissions,
permission inheritance, and fine-grained authorization checks.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from xencode.models.user import (
    User, UserRole, Permission, ResourceType, PermissionType,
    AuditLogEntry
)


class PermissionDeniedError(Exception):
    """Raised when permission is denied"""
    pass


class ResourceNotFoundError(Exception):
    """Raised when resource is not found"""
    pass


class PermissionEngine:
    """Manages role-based permissions and authorization"""
    
    def __init__(self):
        # Resource registry - maps resource IDs to metadata
        self.resources: Dict[str, Dict[str, Any]] = {}
        
        # Permission cache for performance
        self.permission_cache: Dict[str, Dict[str, bool]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Permission inheritance rules
        self.inheritance_rules = self._initialize_inheritance_rules()
        
        # Audit log
        self.audit_log: List[AuditLogEntry] = []
    
    def _initialize_inheritance_rules(self) -> Dict[ResourceType, List[ResourceType]]:
        """Initialize permission inheritance rules"""
        return {
            # Project permissions inherit to files within the project
            ResourceType.PROJECT: [ResourceType.FILE],
            
            # System permissions inherit to all other resources
            ResourceType.SYSTEM: [
                ResourceType.PROJECT, ResourceType.FILE, 
                ResourceType.ANALYSIS, ResourceType.CONFIGURATION
            ],
            
            # User permissions inherit to user-specific resources
            ResourceType.USER: []
        }
    
    def register_resource(self, 
                         resource_id: str,
                         resource_type: ResourceType,
                         metadata: Optional[Dict[str, Any]] = None,
                         parent_resource_id: Optional[str] = None) -> bool:
        """Register a resource in the system"""
        
        if resource_id in self.resources:
            return False  # Resource already exists
        
        self.resources[resource_id] = {
            'type': resource_type,
            'metadata': metadata or {},
            'parent_resource_id': parent_resource_id,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Clear permission cache
        self._clear_permission_cache()
        
        return True
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource from the system"""
        
        if resource_id not in self.resources:
            return False
        
        del self.resources[resource_id]
        
        # Clear permission cache
        self._clear_permission_cache()
        
        return True
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get resource information"""
        return self.resources.get(resource_id)
    
    def list_resources(self, 
                      resource_type: Optional[ResourceType] = None,
                      parent_resource_id: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """List resources, optionally filtered by type or parent"""
        
        resources = []
        for resource_id, resource_info in self.resources.items():
            # Filter by type
            if resource_type and resource_info['type'] != resource_type:
                continue
            
            # Filter by parent
            if parent_resource_id and resource_info.get('parent_resource_id') != parent_resource_id:
                continue
            
            resources.append((resource_id, resource_info))
        
        return resources
    
    async def check_permission(self, 
                              user: User,
                              resource_type: ResourceType,
                              permission_type: PermissionType,
                              resource_id: Optional[str] = None,
                              audit: bool = True) -> bool:
        """Check if user has permission for resource"""
        
        # Check cache first
        cache_key = f"{user.id}:{resource_type.value}:{permission_type.value}:{resource_id or 'all'}"
        
        if self._is_cache_valid(cache_key):
            result = self.permission_cache[cache_key]
            if audit:
                await self._audit_permission_check(user, resource_type, permission_type, resource_id, result)
            return result
        
        # Perform permission check
        result = await self._perform_permission_check(user, resource_type, permission_type, resource_id)
        
        # Cache result
        self.permission_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Audit the check
        if audit:
            await self._audit_permission_check(user, resource_type, permission_type, resource_id, result)
        
        return result
    
    async def _perform_permission_check(self,
                                       user: User,
                                       resource_type: ResourceType,
                                       permission_type: PermissionType,
                                       resource_id: Optional[str] = None) -> bool:
        """Perform the actual permission check"""
        
        # Check if user is active
        if not user.is_active:
            return False
        
        # Check if account is locked
        if user.is_locked():
            return False
        
        # Admin users have all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Check explicit user permissions first
        if user.has_permission(resource_type, permission_type, resource_id):
            return True
        
        # Check inherited permissions
        if await self._check_inherited_permissions(user, resource_type, permission_type, resource_id):
            return True
        
        # Check resource-specific permissions
        if resource_id and await self._check_resource_specific_permissions(user, resource_id, permission_type):
            return True
        
        return False
    
    async def _check_inherited_permissions(self,
                                          user: User,
                                          resource_type: ResourceType,
                                          permission_type: PermissionType,
                                          resource_id: Optional[str] = None) -> bool:
        """Check inherited permissions from parent resources"""
        
        # Check if resource has a parent
        if resource_id:
            resource_info = self.resources.get(resource_id)
            if resource_info and resource_info.get('parent_resource_id'):
                parent_id = resource_info['parent_resource_id']
                parent_info = self.resources.get(parent_id)
                
                if parent_info:
                    parent_type = parent_info['type']
                    
                    # Check if user has permission on parent resource
                    if user.has_permission(parent_type, permission_type, parent_id):
                        return True
        
        # Check inheritance rules
        for parent_type, child_types in self.inheritance_rules.items():
            if resource_type in child_types:
                # Check if user has permission on parent type
                if user.has_permission(parent_type, permission_type):
                    return True
        
        return False
    
    async def _check_resource_specific_permissions(self,
                                                  user: User,
                                                  resource_id: str,
                                                  permission_type: PermissionType) -> bool:
        """Check resource-specific permissions (e.g., ownership)"""
        
        resource_info = self.resources.get(resource_id)
        if not resource_info:
            return False
        
        metadata = resource_info.get('metadata', {})
        
        # Check ownership
        if metadata.get('owner_id') == user.id:
            # Owners have read/write/execute permissions
            if permission_type in [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE]:
                return True
        
        # Check if user is in allowed users list
        allowed_users = metadata.get('allowed_users', [])
        if user.id in allowed_users:
            allowed_permissions = metadata.get('allowed_permissions', [PermissionType.READ])
            if permission_type in allowed_permissions:
                return True
        
        # Check if user's role is in allowed roles
        allowed_roles = metadata.get('allowed_roles', [])
        if user.role in allowed_roles:
            role_permissions = metadata.get('role_permissions', {})
            user_role_permissions = role_permissions.get(user.role.value, [PermissionType.READ])
            if permission_type in user_role_permissions:
                return True
        
        return False
    
    async def require_permission(self,
                                user: User,
                                resource_type: ResourceType,
                                permission_type: PermissionType,
                                resource_id: Optional[str] = None) -> None:
        """Require permission or raise PermissionDeniedError"""
        
        if not await self.check_permission(user, resource_type, permission_type, resource_id):
            raise PermissionDeniedError(
                f"User {user.username} lacks {permission_type.value} permission "
                f"for {resource_type.value}" + (f" {resource_id}" if resource_id else "")
            )
    
    async def grant_permission(self,
                              user: User,
                              target_user: User,
                              resource_type: ResourceType,
                              permission_type: PermissionType,
                              resource_id: Optional[str] = None) -> bool:
        """Grant permission to a user (requires admin or ownership)"""
        
        # Check if granting user has admin permission
        if not await self.check_permission(user, ResourceType.USER, PermissionType.ADMIN):
            # Check if granting user owns the resource
            if resource_id:
                resource_info = self.resources.get(resource_id)
                if not resource_info or resource_info.get('metadata', {}).get('owner_id') != user.id:
                    return False
            else:
                return False
        
        # Grant permission
        permission = Permission(
            resource_type=resource_type,
            resource_id=resource_id,
            permission_type=permission_type,
            granted=True
        )
        
        target_user.add_permission(permission)
        
        # Clear permission cache
        self._clear_permission_cache()
        
        # Audit the grant
        await self._audit_permission_grant(user, target_user, resource_type, permission_type, resource_id)
        
        return True
    
    async def revoke_permission(self,
                               user: User,
                               target_user: User,
                               resource_type: ResourceType,
                               permission_type: PermissionType,
                               resource_id: Optional[str] = None) -> bool:
        """Revoke permission from a user (requires admin or ownership)"""
        
        # Check if revoking user has admin permission
        if not await self.check_permission(user, ResourceType.USER, PermissionType.ADMIN):
            # Check if revoking user owns the resource
            if resource_id:
                resource_info = self.resources.get(resource_id)
                if not resource_info or resource_info.get('metadata', {}).get('owner_id') != user.id:
                    return False
            else:
                return False
        
        # Revoke permission
        target_user.remove_permission(resource_type, permission_type, resource_id)
        
        # Clear permission cache
        self._clear_permission_cache()
        
        # Audit the revocation
        await self._audit_permission_revoke(user, target_user, resource_type, permission_type, resource_id)
        
        return True
    
    def get_user_permissions(self, user: User, resource_id: Optional[str] = None) -> List[Permission]:
        """Get all effective permissions for a user"""
        
        permissions = user.get_effective_permissions()
        
        # Filter by resource if specified
        if resource_id:
            permissions = [
                p for p in permissions 
                if p.resource_id is None or p.resource_id == resource_id
            ]
        
        return permissions
    
    def get_resource_permissions(self, resource_id: str) -> List[Tuple[str, Permission]]:
        """Get all users with permissions on a specific resource"""
        
        # This would typically query a database
        # For now, we'll return an empty list as this requires user enumeration
        return []
    
    async def check_bulk_permissions(self,
                                    user: User,
                                    permission_checks: List[Tuple[ResourceType, PermissionType, Optional[str]]]) -> Dict[str, bool]:
        """Check multiple permissions at once"""
        
        results = {}
        
        for resource_type, permission_type, resource_id in permission_checks:
            key = f"{resource_type.value}:{permission_type.value}:{resource_id or 'all'}"
            results[key] = await self.check_permission(user, resource_type, permission_type, resource_id, audit=False)
        
        return results
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        
        if cache_key not in self.permission_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        age_seconds = (datetime.now() - cache_time).total_seconds()
        
        return age_seconds < self.cache_ttl_seconds
    
    def _clear_permission_cache(self) -> None:
        """Clear permission cache"""
        self.permission_cache.clear()
        self.cache_timestamps.clear()
    
    async def _audit_permission_check(self,
                                     user: User,
                                     resource_type: ResourceType,
                                     permission_type: PermissionType,
                                     resource_id: Optional[str],
                                     result: bool) -> None:
        """Audit permission check"""
        
        entry = AuditLogEntry(
            user_id=user.id,
            username=user.username,
            action="permission_check",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                'permission_type': permission_type.value,
                'result': result
            },
            success=True
        )
        
        self.audit_log.append(entry)
    
    async def _audit_permission_grant(self,
                                     granting_user: User,
                                     target_user: User,
                                     resource_type: ResourceType,
                                     permission_type: PermissionType,
                                     resource_id: Optional[str]) -> None:
        """Audit permission grant"""
        
        entry = AuditLogEntry(
            user_id=granting_user.id,
            username=granting_user.username,
            action="permission_grant",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                'target_user_id': target_user.id,
                'target_username': target_user.username,
                'permission_type': permission_type.value
            },
            success=True
        )
        
        self.audit_log.append(entry)
    
    async def _audit_permission_revoke(self,
                                      revoking_user: User,
                                      target_user: User,
                                      resource_type: ResourceType,
                                      permission_type: PermissionType,
                                      resource_id: Optional[str]) -> None:
        """Audit permission revocation"""
        
        entry = AuditLogEntry(
            user_id=revoking_user.id,
            username=revoking_user.username,
            action="permission_revoke",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                'target_user_id': target_user.id,
                'target_username': target_user.username,
                'permission_type': permission_type.value
            },
            success=True
        )
        
        self.audit_log.append(entry)
    
    def get_audit_log(self, 
                     user_id: Optional[str] = None,
                     action: Optional[str] = None,
                     limit: int = 100) -> List[AuditLogEntry]:
        """Get audit log entries"""
        
        entries = self.audit_log
        
        # Filter by user
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        
        # Filter by action
        if action:
            entries = [e for e in entries if e.action == action]
        
        # Sort by timestamp (newest first) and limit
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get permission engine statistics"""
        
        return {
            'registered_resources': len(self.resources),
            'cached_permissions': len(self.permission_cache),
            'audit_log_entries': len(self.audit_log),
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'inheritance_rules': {
                parent.value: [child.value for child in children]
                for parent, children in self.inheritance_rules.items()
            }
        }
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        
        expired_keys = []
        now = datetime.now()
        
        for cache_key, timestamp in self.cache_timestamps.items():
            age_seconds = (now - timestamp).total_seconds()
            if age_seconds >= self.cache_ttl_seconds:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.permission_cache[key]
            del self.cache_timestamps[key]
        
        return len(expired_keys)


# Global permission engine instance
permission_engine = PermissionEngine()