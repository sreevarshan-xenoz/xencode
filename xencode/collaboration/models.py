"""Data models for collaboration features."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class Role(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(str, Enum):
    """Permissions for resources."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Workspace:
    """Workspace model."""
    id: str
    name: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "settings": self.settings
        }


@dataclass
class WorkspaceMember:
    """Workspace membership model."""
    workspace_id: str
    user_id: str
    role: Role
    joined_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None
        }


@dataclass
class Session:
    """Conversation session model."""
    id: str
    workspace_id: str
    title: Optional[str]
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    shared: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "shared": self.shared,
            "messages": self.messages
        }


@dataclass
class KnowledgeItem:
    """Knowledge base item model."""
    id: str
    workspace_id: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# RBAC Permission mappings
ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.SHARE, Permission.ADMIN],
    Role.EDITOR: [Permission.READ, Permission.WRITE, Permission.SHARE],
    Role.VIEWER: [Permission.READ],
    Role.GUEST: [Permission.READ]
}


def has_permission(role: Role, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])
