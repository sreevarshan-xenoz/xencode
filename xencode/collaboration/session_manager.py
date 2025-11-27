"""Session management and sharing."""

import uuid
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from .database import CollaborationDatabase
from .models import Session, Role, Permission, has_permission


class SessionManager:
    """Manage conversation sessions and sharing."""

    def __init__(self, db: Optional[CollaborationDatabase] = None):
        self.db = db or CollaborationDatabase()

    def create_session(
        self,
        workspace_id: str,
        created_by: str,
        title: Optional[str] = None
    ) -> Session:
        """Create a new session."""
        session = Session(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            title=title or "Untitled Session",
            created_by=created_by
        )
        return self.db.create_session(session)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.db.get_session(session_id)

    def share_session(self, session_id: str, user_id: str) -> bool:
        """Share a session (mark as shared)."""
        session = self.db.get_session(session_id)
        if not session:
            return False

        # Check if user has permission to share
        role = self.db.get_member_role(session.workspace_id, user_id)
        if not role or not has_permission(role, Permission.SHARE):
            return False

        # Mark session as shared
        session.shared = True
        # Update in DB would go here
        return True

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a session to JSON format."""
        session = self.db.get_session(session_id)
        if not session:
            return None

        return session.to_dict()

    def import_session(
        self,
        session_data: Dict[str, Any],
        workspace_id: str,
        user_id: str
    ) -> Optional[Session]:
        """Import a session from JSON format."""
        session = Session(
            id=session_data.get("id", str(uuid.uuid4())),
            workspace_id=workspace_id,
            title=session_data.get("title"),
            created_by=user_id,
            shared=session_data.get("shared", False),
            messages=session_data.get("messages", [])
        )
        return self.db.create_session(session)

    def export_to_file(self, session_id: str, file_path: str) -> bool:
        """Export session to a JSON file."""
        session_data = self.export_session(session_id)
        if not session_data:
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception:
            return False

    def import_from_file(
        self,
        file_path: str,
        workspace_id: str,
        user_id: str
    ) -> Optional[Session]:
        """Import session from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            return self.import_session(session_data, workspace_id, user_id)
        except Exception:
            return None
