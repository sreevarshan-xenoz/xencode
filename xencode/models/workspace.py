#!/usr/bin/env python3
"""
Workspace Management Models

Defines data models for workspace management, collaboration,
and CRDT-based conflict resolution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class WorkspaceType(str, Enum):
    """Types of workspaces"""
    PROJECT = "project"
    ANALYSIS = "analysis"
    COLLABORATION = "collaboration"
    SANDBOX = "sandbox"
    TEMPLATE = "template"


class WorkspaceStatus(str, Enum):
    """Workspace status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"
    LOCKED = "locked"


class CollaborationMode(str, Enum):
    """Collaboration modes"""
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"
    READ_ONLY = "read_only"


class ChangeType(str, Enum):
    """Types of changes in CRDT operations"""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    MOVE = "move"
    METADATA = "metadata"


@dataclass
class WorkspaceConfig:
    """Workspace configuration settings"""
    # General settings
    auto_save_enabled: bool = True
    auto_save_interval_seconds: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backup_count: int = 10
    
    # Collaboration settings
    real_time_sync: bool = True
    conflict_resolution_strategy: str = "last_writer_wins"  # or "merge", "manual"
    max_collaborators: int = 10
    session_timeout_minutes: int = 60
    
    # Storage settings
    max_file_size_mb: int = 100
    max_workspace_size_mb: int = 1000
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    # Analysis settings
    auto_analysis_enabled: bool = True
    analysis_on_save: bool = True
    security_scanning_enabled: bool = True
    
    # Performance settings
    cache_enabled: bool = True
    cache_size_mb: int = 100
    index_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'auto_save_enabled': self.auto_save_enabled,
            'auto_save_interval_seconds': self.auto_save_interval_seconds,
            'backup_enabled': self.backup_enabled,
            'backup_interval_hours': self.backup_interval_hours,
            'max_backup_count': self.max_backup_count,
            'real_time_sync': self.real_time_sync,
            'conflict_resolution_strategy': self.conflict_resolution_strategy,
            'max_collaborators': self.max_collaborators,
            'session_timeout_minutes': self.session_timeout_minutes,
            'max_file_size_mb': self.max_file_size_mb,
            'max_workspace_size_mb': self.max_workspace_size_mb,
            'compression_enabled': self.compression_enabled,
            'encryption_enabled': self.encryption_enabled,
            'auto_analysis_enabled': self.auto_analysis_enabled,
            'analysis_on_save': self.analysis_on_save,
            'security_scanning_enabled': self.security_scanning_enabled,
            'cache_enabled': self.cache_enabled,
            'cache_size_mb': self.cache_size_mb,
            'index_enabled': self.index_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceConfig':
        """Create WorkspaceConfig from dictionary"""
        return cls(
            auto_save_enabled=data.get('auto_save_enabled', True),
            auto_save_interval_seconds=data.get('auto_save_interval_seconds', 30),
            backup_enabled=data.get('backup_enabled', True),
            backup_interval_hours=data.get('backup_interval_hours', 24),
            max_backup_count=data.get('max_backup_count', 10),
            real_time_sync=data.get('real_time_sync', True),
            conflict_resolution_strategy=data.get('conflict_resolution_strategy', 'last_writer_wins'),
            max_collaborators=data.get('max_collaborators', 10),
            session_timeout_minutes=data.get('session_timeout_minutes', 60),
            max_file_size_mb=data.get('max_file_size_mb', 100),
            max_workspace_size_mb=data.get('max_workspace_size_mb', 1000),
            compression_enabled=data.get('compression_enabled', True),
            encryption_enabled=data.get('encryption_enabled', False),
            auto_analysis_enabled=data.get('auto_analysis_enabled', True),
            analysis_on_save=data.get('analysis_on_save', True),
            security_scanning_enabled=data.get('security_scanning_enabled', True),
            cache_enabled=data.get('cache_enabled', True),
            cache_size_mb=data.get('cache_size_mb', 100),
            index_enabled=data.get('index_enabled', True)
        )


@dataclass
class WorkspaceFile:
    """Represents a file within a workspace"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: str = ""  # Relative path within workspace
    content: str = ""
    content_hash: str = ""
    size_bytes: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None  # User ID
    updated_by: Optional[str] = None  # User ID
    
    # File type and analysis
    file_type: str = ""
    language: Optional[str] = None
    encoding: str = "utf-8"
    
    # CRDT metadata
    version: int = 1
    vector_clock: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'content': self.content,
            'content_hash': self.content_hash,
            'size_bytes': self.size_bytes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by,
            'updated_by': self.updated_by,
            'file_type': self.file_type,
            'language': self.language,
            'encoding': self.encoding,
            'version': self.version,
            'vector_clock': self.vector_clock
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceFile':
        """Create WorkspaceFile from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            path=data.get('path', ''),
            content=data.get('content', ''),
            content_hash=data.get('content_hash', ''),
            size_bytes=data.get('size_bytes', 0),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            created_by=data.get('created_by'),
            updated_by=data.get('updated_by'),
            file_type=data.get('file_type', ''),
            language=data.get('language'),
            encoding=data.get('encoding', 'utf-8'),
            version=data.get('version', 1),
            vector_clock=data.get('vector_clock', {})
        )


@dataclass
class WorkspaceCollaborator:
    """Represents a collaborator in a workspace"""
    user_id: str = ""
    username: str = ""
    role: str = "viewer"  # owner, editor, viewer
    permissions: List[str] = field(default_factory=list)
    
    # Session info
    is_active: bool = False
    last_seen: Optional[datetime] = None
    session_id: Optional[str] = None
    
    # Collaboration metadata
    joined_at: datetime = field(default_factory=datetime.now)
    invited_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'session_id': self.session_id,
            'joined_at': self.joined_at.isoformat(),
            'invited_by': self.invited_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceCollaborator':
        """Create WorkspaceCollaborator from dictionary"""
        last_seen = None
        if data.get('last_seen'):
            last_seen = datetime.fromisoformat(data['last_seen'])
        
        return cls(
            user_id=data.get('user_id', ''),
            username=data.get('username', ''),
            role=data.get('role', 'viewer'),
            permissions=data.get('permissions', []),
            is_active=data.get('is_active', False),
            last_seen=last_seen,
            session_id=data.get('session_id'),
            joined_at=datetime.fromisoformat(data.get('joined_at', datetime.now().isoformat())),
            invited_by=data.get('invited_by')
        )


@dataclass
class Change:
    """Represents a change in the CRDT system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str = ""
    file_id: str = ""
    
    # Change details
    change_type: ChangeType = ChangeType.UPDATE
    position: int = 0  # Position in document for insert/delete
    length: int = 0    # Length for delete operations
    content: str = ""  # Content for insert/update operations
    
    # CRDT metadata
    author_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    vector_clock: Dict[str, int] = field(default_factory=dict)
    parent_changes: List[str] = field(default_factory=list)  # Parent change IDs
    
    # Conflict resolution
    is_conflicted: bool = False
    conflict_resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'workspace_id': self.workspace_id,
            'file_id': self.file_id,
            'change_type': self.change_type.value,
            'position': self.position,
            'length': self.length,
            'content': self.content,
            'author_id': self.author_id,
            'timestamp': self.timestamp.isoformat(),
            'vector_clock': self.vector_clock,
            'parent_changes': self.parent_changes,
            'is_conflicted': self.is_conflicted,
            'conflict_resolution': self.conflict_resolution
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Change':
        """Create Change from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            workspace_id=data.get('workspace_id', ''),
            file_id=data.get('file_id', ''),
            change_type=ChangeType(data.get('change_type', ChangeType.UPDATE)),
            position=data.get('position', 0),
            length=data.get('length', 0),
            content=data.get('content', ''),
            author_id=data.get('author_id', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            vector_clock=data.get('vector_clock', {}),
            parent_changes=data.get('parent_changes', []),
            is_conflicted=data.get('is_conflicted', False),
            conflict_resolution=data.get('conflict_resolution')
        )


@dataclass
class Conflict:
    """Represents a conflict between changes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str = ""
    file_id: str = ""
    
    # Conflicting changes
    change_a: Change = field(default_factory=lambda: Change())
    change_b: Change = field(default_factory=lambda: Change())
    
    # Conflict metadata
    detected_at: datetime = field(default_factory=datetime.now)
    resolution_strategy: str = "manual"  # manual, last_writer_wins, merge
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_change: Optional[Change] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'workspace_id': self.workspace_id,
            'file_id': self.file_id,
            'change_a': self.change_a.to_dict(),
            'change_b': self.change_b.to_dict(),
            'detected_at': self.detected_at.isoformat(),
            'resolution_strategy': self.resolution_strategy,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'resolution_change': self.resolution_change.to_dict() if self.resolution_change else None
        }


@dataclass
class Workspace:
    """Main workspace model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    workspace_type: WorkspaceType = WorkspaceType.PROJECT
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    
    # Ownership and collaboration
    owner_id: str = ""
    collaboration_mode: CollaborationMode = CollaborationMode.PRIVATE
    collaborators: List[WorkspaceCollaborator] = field(default_factory=list)
    
    # Files and content
    files: Dict[str, WorkspaceFile] = field(default_factory=dict)  # file_id -> WorkspaceFile
    root_path: str = ""
    
    # Configuration
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    
    # CRDT and collaboration
    changes: List[Change] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    vector_clock: Dict[str, int] = field(default_factory=dict)
    
    # Statistics
    total_size_bytes: int = 0
    file_count: int = 0
    active_collaborators: int = 0
    
    def add_file(self, file: WorkspaceFile) -> bool:
        """Add file to workspace"""
        if file.id in self.files:
            return False
        
        self.files[file.id] = file
        self.file_count = len(self.files)
        self.total_size_bytes += file.size_bytes
        self.updated_at = datetime.now()
        
        return True
    
    def remove_file(self, file_id: str) -> bool:
        """Remove file from workspace"""
        if file_id not in self.files:
            return False
        
        file = self.files[file_id]
        self.total_size_bytes -= file.size_bytes
        del self.files[file_id]
        self.file_count = len(self.files)
        self.updated_at = datetime.now()
        
        return True
    
    def get_file(self, file_id: str) -> Optional[WorkspaceFile]:
        """Get file by ID"""
        return self.files.get(file_id)
    
    def get_file_by_path(self, path: str) -> Optional[WorkspaceFile]:
        """Get file by path"""
        for file in self.files.values():
            if file.path == path:
                return file
        return None
    
    def add_collaborator(self, collaborator: WorkspaceCollaborator) -> bool:
        """Add collaborator to workspace"""
        # Check if user is already a collaborator
        for existing in self.collaborators:
            if existing.user_id == collaborator.user_id:
                return False
        
        # Check max collaborators limit
        if len(self.collaborators) >= self.config.max_collaborators:
            return False
        
        self.collaborators.append(collaborator)
        self.updated_at = datetime.now()
        
        return True
    
    def remove_collaborator(self, user_id: str) -> bool:
        """Remove collaborator from workspace"""
        for i, collaborator in enumerate(self.collaborators):
            if collaborator.user_id == user_id:
                del self.collaborators[i]
                self.updated_at = datetime.now()
                return True
        return False
    
    def get_collaborator(self, user_id: str) -> Optional[WorkspaceCollaborator]:
        """Get collaborator by user ID"""
        for collaborator in self.collaborators:
            if collaborator.user_id == user_id:
                return collaborator
        return None
    
    def update_collaborator_activity(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """Update collaborator activity"""
        collaborator = self.get_collaborator(user_id)
        if not collaborator:
            return False
        
        collaborator.is_active = True
        collaborator.last_seen = datetime.now()
        collaborator.session_id = session_id
        
        # Update active collaborators count
        self.active_collaborators = sum(1 for c in self.collaborators if c.is_active)
        
        return True
    
    def add_change(self, change: Change) -> None:
        """Add change to workspace"""
        self.changes.append(change)
        self.updated_at = datetime.now()
        
        # Update vector clock
        author_id = change.author_id
        if author_id not in self.vector_clock:
            self.vector_clock[author_id] = 0
        self.vector_clock[author_id] += 1
    
    def add_conflict(self, conflict: Conflict) -> None:
        """Add conflict to workspace"""
        self.conflicts.append(conflict)
        self.updated_at = datetime.now()
    
    def get_unresolved_conflicts(self) -> List[Conflict]:
        """Get unresolved conflicts"""
        return [conflict for conflict in self.conflicts if not conflict.resolved]
    
    def can_user_access(self, user_id: str) -> bool:
        """Check if user can access workspace"""
        # Owner can always access
        if self.owner_id == user_id:
            return True
        
        # Check if user is a collaborator
        collaborator = self.get_collaborator(user_id)
        if collaborator:
            return True
        
        # Check collaboration mode
        if self.collaboration_mode == CollaborationMode.PUBLIC:
            return True
        
        return False
    
    def can_user_edit(self, user_id: str) -> bool:
        """Check if user can edit workspace"""
        # Owner can always edit
        if self.owner_id == user_id:
            return True
        
        # Check collaborator role
        collaborator = self.get_collaborator(user_id)
        if collaborator and collaborator.role in ['owner', 'editor']:
            return True
        
        return False
    
    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'workspace_type': self.workspace_type.value,
            'status': self.status.value,
            'owner_id': self.owner_id,
            'collaboration_mode': self.collaboration_mode.value,
            'collaborators': [c.to_dict() for c in self.collaborators],
            'root_path': self.root_path,
            'config': self.config.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'vector_clock': self.vector_clock,
            'total_size_bytes': self.total_size_bytes,
            'file_count': self.file_count,
            'active_collaborators': self.active_collaborators
        }
        
        if include_content:
            data.update({
                'files': {file_id: file.to_dict() for file_id, file in self.files.items()},
                'changes': [change.to_dict() for change in self.changes],
                'conflicts': [conflict.to_dict() for conflict in self.conflicts]
            })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workspace':
        """Create Workspace from dictionary"""
        
        # Parse collaborators
        collaborators = []
        for collab_data in data.get('collaborators', []):
            collaborators.append(WorkspaceCollaborator.from_dict(collab_data))
        
        # Parse files
        files = {}
        for file_id, file_data in data.get('files', {}).items():
            files[file_id] = WorkspaceFile.from_dict(file_data)
        
        # Parse changes
        changes = []
        for change_data in data.get('changes', []):
            changes.append(Change.from_dict(change_data))
        
        # Parse conflicts
        conflicts = []
        for conflict_data in data.get('conflicts', []):
            conflict = Conflict()
            conflict.id = conflict_data.get('id', str(uuid.uuid4()))
            conflict.workspace_id = conflict_data.get('workspace_id', '')
            conflict.file_id = conflict_data.get('file_id', '')
            conflict.change_a = Change.from_dict(conflict_data.get('change_a', {}))
            conflict.change_b = Change.from_dict(conflict_data.get('change_b', {}))
            conflict.detected_at = datetime.fromisoformat(conflict_data.get('detected_at', datetime.now().isoformat()))
            conflict.resolution_strategy = conflict_data.get('resolution_strategy', 'manual')
            conflict.resolved = conflict_data.get('resolved', False)
            if conflict_data.get('resolved_at'):
                conflict.resolved_at = datetime.fromisoformat(conflict_data['resolved_at'])
            conflict.resolved_by = conflict_data.get('resolved_by')
            if conflict_data.get('resolution_change'):
                conflict.resolution_change = Change.from_dict(conflict_data['resolution_change'])
            conflicts.append(conflict)
        
        # Parse config
        config = WorkspaceConfig.from_dict(data.get('config', {}))
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        last_accessed = None
        if data.get('last_accessed'):
            last_accessed = datetime.fromisoformat(data['last_accessed'])
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            workspace_type=WorkspaceType(data.get('workspace_type', WorkspaceType.PROJECT)),
            status=WorkspaceStatus(data.get('status', WorkspaceStatus.ACTIVE)),
            owner_id=data.get('owner_id', ''),
            collaboration_mode=CollaborationMode(data.get('collaboration_mode', CollaborationMode.PRIVATE)),
            collaborators=collaborators,
            files=files,
            root_path=data.get('root_path', ''),
            config=config,
            created_at=created_at,
            updated_at=updated_at,
            last_accessed=last_accessed,
            changes=changes,
            conflicts=conflicts,
            vector_clock=data.get('vector_clock', {}),
            total_size_bytes=data.get('total_size_bytes', 0),
            file_count=data.get('file_count', 0),
            active_collaborators=data.get('active_collaborators', 0)
        )


# Utility functions
def create_default_workspace(owner_id: str, name: str, workspace_type: WorkspaceType = WorkspaceType.PROJECT) -> Workspace:
    """Create a default workspace"""
    workspace = Workspace(
        name=name,
        workspace_type=workspace_type,
        owner_id=owner_id,
        collaboration_mode=CollaborationMode.PRIVATE
    )
    
    # Add owner as collaborator
    owner_collaborator = WorkspaceCollaborator(
        user_id=owner_id,
        role="owner",
        permissions=["read", "write", "admin"],
        is_active=True
    )
    workspace.add_collaborator(owner_collaborator)
    
    return workspace