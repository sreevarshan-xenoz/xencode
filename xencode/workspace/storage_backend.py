#!/usr/bin/env python3
"""
SQLite Storage Backend

Provides SQLite-based storage for workspaces with isolation,
transactions, and efficient querying capabilities.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from xencode.models.workspace import Workspace, WorkspaceFile, Change, Conflict


class SQLiteStorageBackend:
    """SQLite storage backend for workspaces"""
    
    def __init__(self, db_path: str = "workspaces.db"):
        if not AIOSQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite is required for SQLite storage. "
                "Install with: pip install aiosqlite"
            )
        
        self.db_path = Path(db_path)
        self.connection_pool: Dict[str, aiosqlite.Connection] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database schema"""
        if self._initialized:
            return
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Enable foreign keys
            await db.execute("PRAGMA foreign_keys = ON")
            
            # Create workspaces table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    workspace_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    collaboration_mode TEXT NOT NULL,
                    root_path TEXT,
                    config TEXT,  -- JSON
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed TEXT,
                    vector_clock TEXT,  -- JSON
                    total_size_bytes INTEGER DEFAULT 0,
                    file_count INTEGER DEFAULT 0,
                    active_collaborators INTEGER DEFAULT 0
                )
            """)
            
            # Create collaborators table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS collaborators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT,  -- JSON array
                    is_active BOOLEAN DEFAULT FALSE,
                    last_seen TEXT,
                    session_id TEXT,
                    joined_at TEXT NOT NULL,
                    invited_by TEXT,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE,
                    UNIQUE (workspace_id, user_id)
                )
            """)
            
            # Create files table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content TEXT,
                    content_hash TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT,
                    updated_by TEXT,
                    file_type TEXT,
                    language TEXT,
                    encoding TEXT DEFAULT 'utf-8',
                    version INTEGER DEFAULT 1,
                    vector_clock TEXT,  -- JSON
                    FOREIGN KEY (workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE,
                    UNIQUE (workspace_id, path)
                )
            """)
            
            # Create changes table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS changes (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    position INTEGER DEFAULT 0,
                    length INTEGER DEFAULT 0,
                    content TEXT,
                    author_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    vector_clock TEXT,  -- JSON
                    parent_changes TEXT,  -- JSON array
                    is_conflicted BOOLEAN DEFAULT FALSE,
                    conflict_resolution TEXT,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
                )
            """)
            
            # Create conflicts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    change_a_id TEXT NOT NULL,
                    change_b_id TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolution_strategy TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    resolution_change_id TEXT,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE,
                    FOREIGN KEY (change_a_id) REFERENCES changes (id) ON DELETE CASCADE,
                    FOREIGN KEY (change_b_id) REFERENCES changes (id) ON DELETE CASCADE,
                    FOREIGN KEY (resolution_change_id) REFERENCES changes (id) ON DELETE SET NULL
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_owner ON workspaces (owner_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_status ON workspaces (status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_collaborators_workspace ON collaborators (workspace_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_collaborators_user ON collaborators (user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_files_workspace ON files (workspace_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files (workspace_id, path)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_changes_workspace ON changes (workspace_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_changes_file ON changes (file_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_changes_timestamp ON changes (timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_workspace ON conflicts (workspace_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts (resolved)")
            
            await db.commit()
        
        self._initialized = True
    
    async def create_workspace(self, workspace: Workspace) -> bool:
        """Create a new workspace"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("BEGIN TRANSACTION")
                
                # Insert workspace
                await db.execute("""
                    INSERT INTO workspaces (
                        id, name, description, workspace_type, status, owner_id,
                        collaboration_mode, root_path, config, created_at, updated_at,
                        last_accessed, vector_clock, total_size_bytes, file_count,
                        active_collaborators
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workspace.id, workspace.name, workspace.description,
                    workspace.workspace_type.value, workspace.status.value,
                    workspace.owner_id, workspace.collaboration_mode.value,
                    workspace.root_path, json.dumps(workspace.config.to_dict()),
                    workspace.created_at.isoformat(), workspace.updated_at.isoformat(),
                    workspace.last_accessed.isoformat() if workspace.last_accessed else None,
                    json.dumps(workspace.vector_clock), workspace.total_size_bytes,
                    workspace.file_count, workspace.active_collaborators
                ))
                
                # Insert collaborators
                for collaborator in workspace.collaborators:
                    await db.execute("""
                        INSERT INTO collaborators (
                            workspace_id, user_id, username, role, permissions,
                            is_active, last_seen, session_id, joined_at, invited_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        workspace.id, collaborator.user_id, collaborator.username,
                        collaborator.role, json.dumps(collaborator.permissions),
                        collaborator.is_active,
                        collaborator.last_seen.isoformat() if collaborator.last_seen else None,
                        collaborator.session_id, collaborator.joined_at.isoformat(),
                        collaborator.invited_by
                    ))
                
                # Insert files
                for file in workspace.files.values():
                    await db.execute("""
                        INSERT INTO files (
                            id, workspace_id, name, path, content, content_hash,
                            size_bytes, created_at, updated_at, created_by, updated_by,
                            file_type, language, encoding, version, vector_clock
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file.id, workspace.id, file.name, file.path, file.content,
                        file.content_hash, file.size_bytes, file.created_at.isoformat(),
                        file.updated_at.isoformat(), file.created_by, file.updated_by,
                        file.file_type, file.language, file.encoding, file.version,
                        json.dumps(file.vector_clock)
                    ))
                
                await db.commit()
                return True
                
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error creating workspace: {e}")
            return False
    
    async def get_workspace(self, workspace_id: str, include_content: bool = True) -> Optional[Workspace]:
        """Get workspace by ID"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get workspace
                async with db.execute("""
                    SELECT * FROM workspaces WHERE id = ?
                """, (workspace_id,)) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return None
                
                # Convert row to dict
                columns = [desc[0] for desc in cursor.description]
                workspace_data = dict(zip(columns, row))
                
                # Parse JSON fields
                workspace_data['config'] = json.loads(workspace_data['config'])
                workspace_data['vector_clock'] = json.loads(workspace_data['vector_clock'])
                
                # Get collaborators
                collaborators = []
                async with db.execute("""
                    SELECT * FROM collaborators WHERE workspace_id = ?
                """, (workspace_id,)) as cursor:
                    async for row in cursor:
                        collab_columns = [desc[0] for desc in cursor.description]
                        collab_data = dict(zip(collab_columns, row))
                        collab_data['permissions'] = json.loads(collab_data['permissions'])
                        collaborators.append(collab_data)
                
                workspace_data['collaborators'] = collaborators
                
                # Get files if requested
                if include_content:
                    files = {}
                    async with db.execute("""
                        SELECT * FROM files WHERE workspace_id = ?
                    """, (workspace_id,)) as cursor:
                        async for row in cursor:
                            file_columns = [desc[0] for desc in cursor.description]
                            file_data = dict(zip(file_columns, row))
                            file_data['vector_clock'] = json.loads(file_data['vector_clock'])
                            files[file_data['id']] = file_data
                    
                    workspace_data['files'] = files
                    
                    # Get changes
                    changes = []
                    async with db.execute("""
                        SELECT * FROM changes WHERE workspace_id = ? ORDER BY timestamp
                    """, (workspace_id,)) as cursor:
                        async for row in cursor:
                            change_columns = [desc[0] for desc in cursor.description]
                            change_data = dict(zip(change_columns, row))
                            change_data['vector_clock'] = json.loads(change_data['vector_clock'])
                            change_data['parent_changes'] = json.loads(change_data['parent_changes'])
                            changes.append(change_data)
                    
                    workspace_data['changes'] = changes
                    
                    # Get conflicts
                    conflicts = []
                    async with db.execute("""
                        SELECT c.*, ca.*, cb.*
                        FROM conflicts c
                        JOIN changes ca ON c.change_a_id = ca.id
                        JOIN changes cb ON c.change_b_id = cb.id
                        WHERE c.workspace_id = ?
                    """, (workspace_id,)) as cursor:
                        async for row in cursor:
                            # This would need more complex parsing for conflicts
                            # For now, we'll skip detailed conflict loading
                            pass
                    
                    workspace_data['conflicts'] = []
                
                # Create workspace object
                return Workspace.from_dict(workspace_data)
                
        except Exception as e:
            print(f"Error getting workspace: {e}")
            return None
    
    async def update_workspace(self, workspace: Workspace) -> bool:
        """Update workspace"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE workspaces SET
                        name = ?, description = ?, workspace_type = ?, status = ?,
                        collaboration_mode = ?, root_path = ?, config = ?,
                        updated_at = ?, last_accessed = ?, vector_clock = ?,
                        total_size_bytes = ?, file_count = ?, active_collaborators = ?
                    WHERE id = ?
                """, (
                    workspace.name, workspace.description, workspace.workspace_type.value,
                    workspace.status.value, workspace.collaboration_mode.value,
                    workspace.root_path, json.dumps(workspace.config.to_dict()),
                    workspace.updated_at.isoformat(),
                    workspace.last_accessed.isoformat() if workspace.last_accessed else None,
                    json.dumps(workspace.vector_clock), workspace.total_size_bytes,
                    workspace.file_count, workspace.active_collaborators, workspace.id
                ))
                
                await db.commit()
                return True
                
        except Exception as e:
            print(f"Error updating workspace: {e}")
            return False
    
    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete workspace"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
                await db.commit()
                return True
                
        except Exception as e:
            print(f"Error deleting workspace: {e}")
            return False
    
    async def list_workspaces(self, 
                             owner_id: Optional[str] = None,
                             user_id: Optional[str] = None,
                             status: Optional[str] = None) -> List[Workspace]:
        """List workspaces with optional filters"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = "SELECT * FROM workspaces WHERE 1=1"
                params = []
                
                if owner_id:
                    query += " AND owner_id = ?"
                    params.append(owner_id)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if user_id and not owner_id:
                    # Find workspaces where user is a collaborator
                    query += " AND id IN (SELECT workspace_id FROM collaborators WHERE user_id = ?)"
                    params.append(user_id)
                
                query += " ORDER BY updated_at DESC"
                
                workspaces = []
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        columns = [desc[0] for desc in cursor.description]
                        workspace_data = dict(zip(columns, row))
                        workspace_data['config'] = json.loads(workspace_data['config'])
                        workspace_data['vector_clock'] = json.loads(workspace_data['vector_clock'])
                        workspace_data['collaborators'] = []
                        workspace_data['files'] = {}
                        workspace_data['changes'] = []
                        workspace_data['conflicts'] = []
                        
                        workspaces.append(Workspace.from_dict(workspace_data))
                
                return workspaces
                
        except Exception as e:
            print(f"Error listing workspaces: {e}")
            return []
    
    async def add_file(self, workspace_id: str, file: WorkspaceFile) -> bool:
        """Add file to workspace"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO files (
                        id, workspace_id, name, path, content, content_hash,
                        size_bytes, created_at, updated_at, created_by, updated_by,
                        file_type, language, encoding, version, vector_clock
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file.id, workspace_id, file.name, file.path, file.content,
                    file.content_hash, file.size_bytes, file.created_at.isoformat(),
                    file.updated_at.isoformat(), file.created_by, file.updated_by,
                    file.file_type, file.language, file.encoding, file.version,
                    json.dumps(file.vector_clock)
                ))
                
                # Update workspace file count and size
                await db.execute("""
                    UPDATE workspaces SET
                        file_count = file_count + 1,
                        total_size_bytes = total_size_bytes + ?,
                        updated_at = ?
                    WHERE id = ?
                """, (file.size_bytes, datetime.now().isoformat(), workspace_id))
                
                await db.commit()
                return True
                
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error adding file: {e}")
            return False
    
    async def update_file(self, file: WorkspaceFile) -> bool:
        """Update file"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get old file size
                async with db.execute("SELECT size_bytes FROM files WHERE id = ?", (file.id,)) as cursor:
                    row = await cursor.fetchone()
                    old_size = row[0] if row else 0
                
                # Update file
                await db.execute("""
                    UPDATE files SET
                        name = ?, path = ?, content = ?, content_hash = ?,
                        size_bytes = ?, updated_at = ?, updated_by = ?,
                        file_type = ?, language = ?, encoding = ?, version = ?,
                        vector_clock = ?
                    WHERE id = ?
                """, (
                    file.name, file.path, file.content, file.content_hash,
                    file.size_bytes, file.updated_at.isoformat(), file.updated_by,
                    file.file_type, file.language, file.encoding, file.version,
                    json.dumps(file.vector_clock), file.id
                ))
                
                # Update workspace size
                size_diff = file.size_bytes - old_size
                await db.execute("""
                    UPDATE workspaces SET
                        total_size_bytes = total_size_bytes + ?,
                        updated_at = ?
                    WHERE id = (SELECT workspace_id FROM files WHERE id = ?)
                """, (size_diff, datetime.now().isoformat(), file.id))
                
                await db.commit()
                return True
                
        except Exception as e:
            print(f"Error updating file: {e}")
            return False
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get file info before deletion
                async with db.execute("""
                    SELECT workspace_id, size_bytes FROM files WHERE id = ?
                """, (file_id,)) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return False
                    
                    workspace_id, size_bytes = row
                
                # Delete file
                await db.execute("DELETE FROM files WHERE id = ?", (file_id,))
                
                # Update workspace stats
                await db.execute("""
                    UPDATE workspaces SET
                        file_count = file_count - 1,
                        total_size_bytes = total_size_bytes - ?,
                        updated_at = ?
                    WHERE id = ?
                """, (size_bytes, datetime.now().isoformat(), workspace_id))
                
                await db.commit()
                return True
                
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    async def add_change(self, change: Change) -> bool:
        """Add change to workspace"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO changes (
                        id, workspace_id, file_id, change_type, position, length,
                        content, author_id, timestamp, vector_clock, parent_changes,
                        is_conflicted, conflict_resolution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    change.id, change.workspace_id, change.file_id, change.change_type.value,
                    change.position, change.length, change.content, change.author_id,
                    change.timestamp.isoformat(), json.dumps(change.vector_clock),
                    json.dumps(change.parent_changes), change.is_conflicted,
                    change.conflict_resolution
                ))
                
                await db.commit()
                return True
                
        except Exception as e:
            print(f"Error adding change: {e}")
            return False
    
    async def get_changes(self, 
                         workspace_id: str,
                         file_id: Optional[str] = None,
                         since: Optional[datetime] = None) -> List[Change]:
        """Get changes for workspace or file"""
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = "SELECT * FROM changes WHERE workspace_id = ?"
                params = [workspace_id]
                
                if file_id:
                    query += " AND file_id = ?"
                    params.append(file_id)
                
                if since:
                    query += " AND timestamp > ?"
                    params.append(since.isoformat())
                
                query += " ORDER BY timestamp"
                
                changes = []
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        columns = [desc[0] for desc in cursor.description]
                        change_data = dict(zip(columns, row))
                        change_data['vector_clock'] = json.loads(change_data['vector_clock'])
                        change_data['parent_changes'] = json.loads(change_data['parent_changes'])
                        changes.append(Change.from_dict(change_data))
                
                return changes
                
        except Exception as e:
            print(f"Error getting changes: {e}")
            return []
    
    async def close(self) -> None:
        """Close database connections"""
        for connection in self.connection_pool.values():
            await connection.close()
        self.connection_pool.clear()


# Global storage backend instance
storage_backend = SQLiteStorageBackend()