#!/usr/bin/env python3
"""
User and Authentication Models

Defines data models for user management, authentication, and role-based access control.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import hashlib
import secrets


class UserRole(str, Enum):
    """User roles for access control"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class PermissionType(str, Enum):
    """Types of permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Types of resources that can be protected"""
    FILE = "file"
    PROJECT = "project"
    ANALYSIS = "analysis"
    SYSTEM = "system"
    USER = "user"
    CONFIGURATION = "configuration"


@dataclass
class Permission:
    """Represents a permission for a specific resource"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.FILE
    resource_id: Optional[str] = None  # Specific resource ID, None for all resources of type
    permission_type: PermissionType = PermissionType.READ
    granted: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'resource_type': self.resource_type.value,
            'resource_id': self.resource_id,
            'permission_type': self.permission_type.value,
            'granted': self.granted
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Create Permission from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            resource_type=ResourceType(data.get('resource_type', ResourceType.FILE)),
            resource_id=data.get('resource_id'),
            permission_type=PermissionType(data.get('permission_type', PermissionType.READ)),
            granted=data.get('granted', True)
        )


@dataclass
class UserSession:
    """Represents an active user session"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    token: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    last_activity: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid"""
        return self.is_active and not self.is_expired()
    
    def refresh(self, extend_hours: int = 24) -> None:
        """Refresh session expiration"""
        self.expires_at = datetime.now() + timedelta(hours=extend_hours)
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'token': self.token,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active
        }


@dataclass
class User:
    """User model with authentication and authorization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    password_hash: str = ""
    salt: str = field(default_factory=lambda: secrets.token_hex(32))
    
    # Profile information
    full_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Role and permissions
    role: UserRole = UserRole.VIEWER
    permissions: List[Permission] = field(default_factory=list)
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    # Security settings
    require_password_change: bool = False
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    
    def set_password(self, password: str) -> None:
        """Set user password with secure hashing"""
        self.salt = secrets.token_hex(32)
        self.password_hash = self._hash_password(password, self.salt)
        self.updated_at = datetime.now()
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return self.password_hash == self._hash_password(password, self.salt)
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2"""
        # Using PBKDF2 with SHA-256
        return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until is None:
            return False
        return datetime.now() < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock account for specified duration"""
        self.locked_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.updated_at = datetime.now()
    
    def unlock_account(self) -> None:
        """Unlock account"""
        self.locked_until = None
        self.failed_login_attempts = 0
        self.updated_at = datetime.now()
    
    def record_failed_login(self, max_attempts: int = 5) -> None:
        """Record failed login attempt and lock if necessary"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.lock_account()
        self.updated_at = datetime.now()
    
    def record_successful_login(self) -> None:
        """Record successful login"""
        self.last_login = datetime.now()
        self.failed_login_attempts = 0
        self.locked_until = None
        self.updated_at = datetime.now()
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to user"""
        # Remove existing permission for same resource/type combination
        self.permissions = [
            p for p in self.permissions 
            if not (p.resource_type == permission.resource_type and 
                   p.resource_id == permission.resource_id and
                   p.permission_type == permission.permission_type)
        ]
        self.permissions.append(permission)
        self.updated_at = datetime.now()
    
    def remove_permission(self, resource_type: ResourceType, 
                         permission_type: PermissionType,
                         resource_id: Optional[str] = None) -> None:
        """Remove permission from user"""
        self.permissions = [
            p for p in self.permissions 
            if not (p.resource_type == resource_type and 
                   p.resource_id == resource_id and
                   p.permission_type == permission_type)
        ]
        self.updated_at = datetime.now()
    
    def has_permission(self, resource_type: ResourceType, 
                      permission_type: PermissionType,
                      resource_id: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        
        # Admin role has all permissions
        if self.role == UserRole.ADMIN:
            return True
        
        # Check explicit permissions
        for permission in self.permissions:
            if (permission.resource_type == resource_type and
                permission.permission_type == permission_type and
                permission.granted):
                
                # If permission is for all resources of type (resource_id is None)
                # or for specific resource
                if permission.resource_id is None or permission.resource_id == resource_id:
                    return True
        
        # Check role-based permissions
        return self._has_role_permission(resource_type, permission_type)
    
    def _has_role_permission(self, resource_type: ResourceType, 
                           permission_type: PermissionType) -> bool:
        """Check role-based permissions"""
        
        # Define role-based permissions
        role_permissions = {
            UserRole.ADMIN: {
                # Admin has all permissions
                ResourceType.FILE: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN],
                ResourceType.PROJECT: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN],
                ResourceType.ANALYSIS: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN],
                ResourceType.SYSTEM: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN],
                ResourceType.USER: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN],
                ResourceType.CONFIGURATION: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.DELETE, PermissionType.ADMIN]
            },
            UserRole.DEVELOPER: {
                ResourceType.FILE: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE],
                ResourceType.PROJECT: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE],
                ResourceType.ANALYSIS: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE],
                ResourceType.SYSTEM: [PermissionType.READ],
                ResourceType.USER: [PermissionType.READ],
                ResourceType.CONFIGURATION: [PermissionType.READ, PermissionType.WRITE]
            },
            UserRole.ANALYST: {
                ResourceType.FILE: [PermissionType.READ, PermissionType.EXECUTE],
                ResourceType.PROJECT: [PermissionType.READ, PermissionType.EXECUTE],
                ResourceType.ANALYSIS: [PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE],
                ResourceType.SYSTEM: [PermissionType.READ],
                ResourceType.USER: [PermissionType.READ],
                ResourceType.CONFIGURATION: [PermissionType.READ]
            },
            UserRole.VIEWER: {
                ResourceType.FILE: [PermissionType.READ],
                ResourceType.PROJECT: [PermissionType.READ],
                ResourceType.ANALYSIS: [PermissionType.READ],
                ResourceType.SYSTEM: [],
                ResourceType.USER: [],
                ResourceType.CONFIGURATION: []
            },
            UserRole.GUEST: {
                ResourceType.FILE: [],
                ResourceType.PROJECT: [],
                ResourceType.ANALYSIS: [],
                ResourceType.SYSTEM: [],
                ResourceType.USER: [],
                ResourceType.CONFIGURATION: []
            }
        }
        
        role_perms = role_permissions.get(self.role, {})
        resource_perms = role_perms.get(resource_type, [])
        return permission_type in resource_perms
    
    def get_effective_permissions(self) -> List[Permission]:
        """Get all effective permissions (role-based + explicit)"""
        effective_permissions = []
        
        # Add role-based permissions
        for resource_type in ResourceType:
            for permission_type in PermissionType:
                if self._has_role_permission(resource_type, permission_type):
                    effective_permissions.append(Permission(
                        resource_type=resource_type,
                        permission_type=permission_type,
                        granted=True
                    ))
        
        # Add explicit permissions (these can override role-based)
        for permission in self.permissions:
            # Remove any role-based permission that matches
            effective_permissions = [
                p for p in effective_permissions
                if not (p.resource_type == permission.resource_type and
                       p.permission_type == permission.permission_type and
                       p.resource_id == permission.resource_id)
            ]
            effective_permissions.append(permission)
        
        return effective_permissions
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'role': self.role.value,
            'permissions': [p.to_dict() for p in self.permissions],
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'two_factor_enabled': self.two_factor_enabled
        }
        
        if include_sensitive:
            data.update({
                'password_hash': self.password_hash,
                'salt': self.salt,
                'failed_login_attempts': self.failed_login_attempts,
                'locked_until': self.locked_until.isoformat() if self.locked_until else None,
                'require_password_change': self.require_password_change,
                'two_factor_secret': self.two_factor_secret
            })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create User from dictionary"""
        
        # Parse permissions
        permissions = []
        for perm_data in data.get('permissions', []):
            permissions.append(Permission.from_dict(perm_data))
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        last_login = None
        if data.get('last_login'):
            last_login = datetime.fromisoformat(data['last_login'])
        
        locked_until = None
        if data.get('locked_until'):
            locked_until = datetime.fromisoformat(data['locked_until'])
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            username=data.get('username', ''),
            email=data.get('email', ''),
            password_hash=data.get('password_hash', ''),
            salt=data.get('salt', secrets.token_hex(32)),
            full_name=data.get('full_name', ''),
            created_at=created_at,
            updated_at=updated_at,
            last_login=last_login,
            role=UserRole(data.get('role', UserRole.VIEWER)),
            permissions=permissions,
            is_active=data.get('is_active', True),
            is_verified=data.get('is_verified', False),
            failed_login_attempts=data.get('failed_login_attempts', 0),
            locked_until=locked_until,
            require_password_change=data.get('require_password_change', False),
            two_factor_enabled=data.get('two_factor_enabled', False),
            two_factor_secret=data.get('two_factor_secret')
        )


@dataclass
class AuditLogEntry:
    """Audit log entry for security and compliance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    username: Optional[str] = None
    action: str = ""
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'username': self.username,
            'action': self.action,
            'resource_type': self.resource_type.value if self.resource_type else None,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'success': self.success,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Create AuditLogEntry from dictionary"""
        
        resource_type = None
        if data.get('resource_type'):
            resource_type = ResourceType(data['resource_type'])
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            user_id=data.get('user_id'),
            username=data.get('username'),
            action=data.get('action', ''),
            resource_type=resource_type,
            resource_id=data.get('resource_id'),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )


# Utility functions
def create_default_admin_user(username: str = "admin", 
                             password: str = "admin123",
                             email: str = "admin@xencode.local") -> User:
    """Create default admin user"""
    user = User(
        username=username,
        email=email,
        full_name="System Administrator",
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True
    )
    user.set_password(password)
    return user


def create_guest_user() -> User:
    """Create guest user for anonymous access"""
    return User(
        username="guest",
        email="guest@xencode.local",
        full_name="Guest User",
        role=UserRole.GUEST,
        is_active=True,
        is_verified=True
    )