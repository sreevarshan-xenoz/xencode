"""Role-Based Access Control."""

from typing import Optional

from .database import CollaborationDatabase
from .models import Role, Permission, has_permission as check_permission


class RBAC:
    """Role-Based Access Control manager."""

    def __init__(self, db: Optional[CollaborationDatabase] = None):
        self.db = db or CollaborationDatabase()

    def has_permission(
        self,
        workspace_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Check if a user has a specific permission in a workspace."""
        role = self.db.get_member_role(workspace_id, user_id)
        if not role:
            return False
        return check_permission(role, permission)

    def can_read(self, workspace_id: str, user_id: str) -> bool:
        """Check if user can read resources."""
        return self.has_permission(workspace_id, user_id, Permission.READ)

    def can_write(self, workspace_id: str, user_id: str) -> bool:
        """Check if user can write resources."""
        return self.has_permission(workspace_id, user_id, Permission.WRITE)

    def can_delete(self, workspace_id: str, user_id: str) -> bool:
        """Check if user can delete resources."""
        return self.has_permission(workspace_id, user_id, Permission.DELETE)

    def can_share(self, workspace_id: str, user_id: str) -> bool:
        """Check if user can share resources."""
        return self.has_permission(workspace_id, user_id, Permission.SHARE)

    def is_admin(self, workspace_id: str, user_id: str) -> bool:
        """Check if user has admin permissions."""
        return self.has_permission(workspace_id, user_id, Permission.ADMIN)

    def get_role(self, workspace_id: str, user_id: str) -> Optional[Role]:
        """Get user's role in a workspace."""
        return self.db.get_member_role(workspace_id, user_id)
