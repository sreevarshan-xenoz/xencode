"""Shared knowledge base for teams."""

import uuid
from typing import Optional, List

from .database import CollaborationDatabase
from .models import KnowledgeItem, Role, Permission, has_permission


class KnowledgeBase:
    """Manage shared knowledge repository."""

    def __init__(self, db: Optional[CollaborationDatabase] = None):
        self.db = db or CollaborationDatabase()

    def create_item(
        self,
        workspace_id: str,
        title: str,
        content: str,
        created_by: str,
        tags: Optional[List[str]] = None
    ) -> KnowledgeItem:
        """Create a new knowledge item."""
        item = KnowledgeItem(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            title=title,
            content=content,
            tags=tags or [],
            created_by=created_by
        )
        return self.db.create_knowledge_item(item)

    def search(
        self,
        workspace_id: str,
        query: str
    ) -> List[KnowledgeItem]:
        """Search knowledge items by title or content."""
        return self.db.search_knowledge(workspace_id, query)

    def search_by_tags(
        self,
        workspace_id: str,
        tags: List[str]
    ) -> List[KnowledgeItem]:
        """Search knowledge items by tags."""
        # This would need a more sophisticated query in the database
        # For now, we'll search each tag and combine results
        all_items = self.db.search_knowledge(workspace_id, "")
        
        matching_items = []
        for item in all_items:
            if any(tag in item.tags for tag in tags):
                matching_items.append(item)
        
        return matching_items

    def can_edit(
        self,
        workspace_id: str,
        user_id: str
    ) -> bool:
        """Check if a user can edit knowledge items."""
        role = self.db.get_member_role(workspace_id, user_id)
        if not role:
            return False
        return has_permission(role, Permission.WRITE)

    def can_delete(
        self,
        workspace_id: str,
        user_id: str
    ) -> bool:
        """Check if a user can delete knowledge items."""
        role = self.db.get_member_role(workspace_id, user_id)
        if not role:
            return False
        return has_permission(role, Permission.DELETE)
