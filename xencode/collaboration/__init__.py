"""
Collaboration package for Xencode.

Provides workspace collaboration features including session management,
RBAC, knowledge bases, and database utilities.
"""

from .database import CollaborationDatabase
from .knowledge_base import KnowledgeBase
from .models import KnowledgeItem, Permission, Role, Session, Workspace, WorkspaceMember
from .rbac import RBAC
from .session_manager import SessionManager
from .workspace_manager import WorkspaceManager

__all__ = [
    "Session",
    "Workspace",
    "WorkspaceMember",
    "Role",
    "Permission",
    "KnowledgeItem",
    "RBAC",
    "SessionManager",
    "KnowledgeBase",
    "WorkspaceManager",
    "CollaborationDatabase",
]
