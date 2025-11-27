"""Database management for collaboration features."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import User, Workspace, WorkspaceMember, Session, KnowledgeItem, Role


class CollaborationDatabase:
    """SQLite database manager for collaboration features."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path.home() / ".xencode" / "collaboration.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Workspaces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_by TEXT REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settings TEXT DEFAULT '{}'
                )
            """)
            
            # Workspace members table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
                    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace_id, user_id)
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
                    title TEXT,
                    created_by TEXT REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    shared BOOLEAN DEFAULT FALSE,
                    messages TEXT DEFAULT '[]'
                )
            """)
            
            # Knowledge base table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    created_by TEXT REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_workspace ON knowledge_items(workspace_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge_items(tags)")
            
            conn.commit()

    # User operations
    def create_user(self, user: User) -> User:
        """Create a new user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (id, username, email, created_at) VALUES (?, ?, ?, ?)",
                (user.id, user.username, user.email, user.created_at)
            )
            conn.commit()
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now()
                )
        return None

    # Workspace operations
    def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create a new workspace."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO workspaces (id, name, created_by, created_at, settings) VALUES (?, ?, ?, ?, ?)",
                (workspace.id, workspace.name, workspace.created_by, workspace.created_at, json.dumps(workspace.settings))
            )
            conn.commit()
        return workspace

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, created_by, created_at, settings FROM workspaces WHERE id = ?", (workspace_id,))
            row = cursor.fetchone()
            
            if row:
                return Workspace(
                    id=row[0],
                    name=row[1],
                    created_by=row[2],
                    created_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                    settings=json.loads(row[4]) if row[4] else {}
                )
        return None

    def list_workspaces(self, user_id: str) -> List[Workspace]:
        """List all workspaces a user is a member of."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT w.id, w.name, w.created_by, w.created_at, w.settings
                FROM workspaces w
                JOIN workspace_members wm ON w.id = wm.workspace_id
                WHERE wm.user_id = ?
            """, (user_id,))
            
            workspaces = []
            for row in cursor.fetchall():
                workspaces.append(Workspace(
                    id=row[0],
                    name=row[1],
                    created_by=row[2],
                    created_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                    settings=json.loads(row[4]) if row[4] else {}
                ))
            return workspaces

    # Workspace member operations
    def add_member(self, member: WorkspaceMember) -> WorkspaceMember:
        """Add a member to a workspace."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            role_value = member.role.value if isinstance(member.role, Role) else member.role
            cursor.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role, joined_at) VALUES (?, ?, ?, ?)",
                (member.workspace_id, member.user_id, role_value, member.joined_at)
            )
            conn.commit()
        return member

    def get_member_role(self, workspace_id: str, user_id: str) -> Optional[Role]:
        """Get a user's role in a workspace."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id)
            )
            row = cursor.fetchone()
            return Role(row[0]) if row else None

    # Session operations
    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, workspace_id, title, created_by, created_at, shared, messages) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.id, session.workspace_id, session.title, session.created_by, session.created_at, session.shared, json.dumps(session.messages))
            )
            conn.commit()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, workspace_id, title, created_by, created_at, shared, messages FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                return Session(
                    id=row[0],
                    workspace_id=row[1],
                    title=row[2],
                    created_by=row[3],
                    created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                    shared=bool(row[5]),
                    messages=json.loads(row[6]) if row[6] else []
                )
        return None

    # Knowledge base operations
    def create_knowledge_item(self, item: KnowledgeItem) -> KnowledgeItem:
        """Create a new knowledge item."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge_items (id, workspace_id, title, content, tags, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item.id, item.workspace_id, item.title, item.content, json.dumps(item.tags), item.created_by, item.created_at, item.updated_at)
            )
            conn.commit()
        return item

    def search_knowledge(self, workspace_id: str, query: str) -> List[KnowledgeItem]:
        """Search knowledge items by title or content."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, workspace_id, title, content, tags, created_by, created_at, updated_at
                FROM knowledge_items
                WHERE workspace_id = ? AND (title LIKE ? OR content LIKE ?)
            """, (workspace_id, f"%{query}%", f"%{query}%"))
            
            items = []
            for row in cursor.fetchall():
                items.append(KnowledgeItem(
                    id=row[0],
                    workspace_id=row[1],
                    title=row[2],
                    content=row[3],
                    tags=json.loads(row[4]) if row[4] else [],
                    created_by=row[5],
                    created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                    updated_at=datetime.fromisoformat(row[7]) if row[7] else None
                ))
            return items
