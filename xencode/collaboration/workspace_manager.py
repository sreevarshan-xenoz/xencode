"""Workspace management for teams."""

import uuid
from typing import Optional, List

from .database import CollaborationDatabase
from .models import Workspace, WorkspaceMember, Role, User


class WorkspaceManager:
    """Manage team workspaces."""

    def __init__(self, db: Optional[CollaborationDatabase] = None):
        self.db = db or CollaborationDatabase()

    def create_workspace(
        self,
        name: str,
        created_by: str
    ) -> Workspace:
        """Create a new workspace."""
        workspace = Workspace(
            id=str(uuid.uuid4()),
            name=name,
            created_by=created_by
        )
        workspace = self.db.create_workspace(workspace)
        
        # Add creator as admin
        self.add_member(workspace.id, created_by, Role.ADMIN)
        
        return workspace

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        return self.db.get_workspace(workspace_id)

    def list_user_workspaces(self, user_id: str) -> List[Workspace]:
        """List all workspaces a user is a member of."""
        return self.db.list_workspaces(user_id)

    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: Role = Role.VIEWER
    ) -> WorkspaceMember:
        """Add a member to a workspace."""
        member = WorkspaceMember(
            workspace_id=workspace_id,
            user_id=user_id,
            role=role
        )
        return self.db.add_member(member)

    def get_member_role(
        self,
        workspace_id: str,
        user_id: str
    ) -> Optional[Role]:
        """Get a user's role in a workspace."""
        return self.db.get_member_role(workspace_id, user_id)

    def is_member(self, workspace_id: str, user_id: str) -> bool:
        """Check if a user is a member of a workspace."""
        return self.get_member_role(workspace_id, user_id) is not None

    def is_admin(self, workspace_id: str, user_id: str) -> bool:
        """Check if a user is an admin of a workspace."""
        role = self.get_member_role(workspace_id, user_id)
        return role == Role.ADMIN if role else False
