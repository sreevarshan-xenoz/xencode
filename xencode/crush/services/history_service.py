"""History service for tracking file modifications."""

import asyncio
import difflib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

from xencode.crush.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


def _safe_publish_event(coro):
    """Safely publish event, handling cases where no event loop is running."""
    try:
        asyncio.create_task(coro)
    except RuntimeError:
        # No event loop running, skip event publishing
        pass


@dataclass
class FileHistory:
    """Represents a file version in history."""
    id: str
    session_id: str
    file_path: str
    content: bytes
    version: int
    created_at: int = field(default_factory=lambda: int(time.time()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'file_path': self.file_path,
            'version': self.version,
            'created_at': self.created_at,
            'size': len(self.content),
        }
    
    @classmethod
    def from_row(cls, row) -> 'FileHistory':
        """Create from database row."""
        return cls(
            id=row['id'],
            session_id=row['session_id'],
            file_path=row['file_path'],
            content=row['content'],
            version=row['version'],
            created_at=row['created_at'],
        )
    
    def get_content_str(self, encoding: str = 'utf-8') -> str:
        """Get content as string.
        
        Args:
            encoding: Text encoding (default utf-8)
            
        Returns:
            Decoded content
        """
        try:
            return self.content.decode(encoding)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode content with {encoding}, trying latin-1")
            return self.content.decode('latin-1')


class HistoryEventType(Enum):
    """Types of history events."""
    VERSION_CREATED = "version_created"
    FILE_REVERTED = "file_reverted"


@dataclass
class HistoryEvent:
    """Event for file history updates."""
    type: HistoryEventType
    history: FileHistory


class HistoryEventStream:
    """Async event stream for file history updates."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to events."""
        async with self._lock:
            queue = asyncio.Queue(maxsize=100)
            self._queues[subscriber_id] = queue
            logger.debug(f"History subscriber {subscriber_id} subscribed")
            return queue
    
    async def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from events."""
        async with self._lock:
            self._queues.pop(subscriber_id, None)
            logger.debug(f"History subscriber {subscriber_id} unsubscribed")
    
    async def publish(self, event: HistoryEvent):
        """Publish event to all subscribers."""
        async with self._lock:
            dead_subscribers = []
            
            for subscriber_id, queue in self._queues.items():
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        f"History queue full for subscriber {subscriber_id}, dropping event"
                    )
                except Exception as e:
                    logger.error(
                        f"Error publishing history event to subscriber {subscriber_id}: {e}"
                    )
                    dead_subscribers.append(subscriber_id)
            
            # Clean up dead subscribers
            for subscriber_id in dead_subscribers:
                self._queues.pop(subscriber_id, None)


class HistoryService:
    """Tracks file modifications."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize history service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.event_stream = HistoryEventStream()
    
    def create(self, session_id: str, file_path: str, content: str) -> FileHistory:
        """Create initial file history entry.
        
        Args:
            session_id: Session identifier
            file_path: Path to file
            content: File content
            
        Returns:
            Created file history entry
        """
        return self.create_version(session_id, file_path, content)
    
    def create_version(self, session_id: str, file_path: str, content: str) -> FileHistory:
        """Create a new version of a file.
        
        Args:
            session_id: Session identifier
            file_path: Path to file
            content: File content
            
        Returns:
            Created file history entry
        """
        history_id = str(uuid.uuid4())
        
        # Get next version number
        version = self._get_next_version(session_id, file_path)
        
        # Convert content to bytes
        content_bytes = content.encode('utf-8')
        
        history = FileHistory(
            id=history_id,
            session_id=session_id,
            file_path=file_path,
            content=content_bytes,
            version=version,
        )
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO file_history (
                    id, session_id, file_path, content, version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    history.id,
                    history.session_id,
                    history.file_path,
                    history.content,
                    history.version,
                    history.created_at,
                )
            )
        
        logger.info(
            f"Created file history: {history_id} for {file_path} "
            f"(session: {session_id}, version: {version})"
        )
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                HistoryEvent(type=HistoryEventType.VERSION_CREATED, history=history)
            )
        )
        
        return history
    
    def _get_next_version(self, session_id: str, file_path: str) -> int:
        """Get next version number for a file in a session.
        
        Args:
            session_id: Session identifier
            file_path: File path
            
        Returns:
            Next version number
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT MAX(version) as max_version
            FROM file_history
            WHERE session_id = ? AND file_path = ?
            """,
            (session_id, file_path)
        )
        row = cursor.fetchone()
        
        max_version = row['max_version'] if row and row['max_version'] is not None else 0
        return max_version + 1
    
    def get_by_path_and_session(
        self,
        file_path: str,
        session_id: str
    ) -> Optional[FileHistory]:
        """Get latest file history for a specific session.
        
        Args:
            file_path: File path
            session_id: Session identifier
            
        Returns:
            Latest file history entry if found, None otherwise
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT * FROM file_history
            WHERE file_path = ? AND session_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (file_path, session_id)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return FileHistory.from_row(row)
    
    def list_versions(
        self,
        file_path: str,
        session_id: str
    ) -> List[FileHistory]:
        """List all versions of a file in a session.
        
        Args:
            file_path: File path
            session_id: Session identifier
            
        Returns:
            List of file history entries in chronological order
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT * FROM file_history
            WHERE file_path = ? AND session_id = ?
            ORDER BY version ASC
            """,
            (file_path, session_id)
        )
        
        versions = []
        for row in cursor.fetchall():
            versions.append(FileHistory.from_row(row))
        
        return versions
    
    def get_version(
        self,
        file_path: str,
        session_id: str,
        version: int
    ) -> Optional[FileHistory]:
        """Get a specific version of a file.
        
        Args:
            file_path: File path
            session_id: Session identifier
            version: Version number
            
        Returns:
            File history entry if found, None otherwise
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT * FROM file_history
            WHERE file_path = ? AND session_id = ? AND version = ?
            """,
            (file_path, session_id, version)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return FileHistory.from_row(row)
    
    def list_files_in_session(self, session_id: str) -> List[str]:
        """List all files tracked in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of unique file paths
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT DISTINCT file_path
            FROM file_history
            WHERE session_id = ?
            ORDER BY file_path
            """,
            (session_id,)
        )
        
        files = []
        for row in cursor.fetchall():
            files.append(row['file_path'])
        
        return files
    
    def generate_diff(
        self,
        old_content: str,
        new_content: str,
        file_path: str = "file"
    ) -> str:
        """Generate unified diff between two versions.
        
        Args:
            old_content: Old file content
            new_content: New file content
            file_path: File path for diff header
            
        Returns:
            Unified diff string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def generate_diff_between_versions(
        self,
        file_path: str,
        session_id: str,
        from_version: int,
        to_version: int
    ) -> Optional[str]:
        """Generate diff between two versions of a file.
        
        Args:
            file_path: File path
            session_id: Session identifier
            from_version: Starting version
            to_version: Ending version
            
        Returns:
            Unified diff string if both versions exist, None otherwise
        """
        old_history = self.get_version(file_path, session_id, from_version)
        new_history = self.get_version(file_path, session_id, to_version)
        
        if old_history is None or new_history is None:
            return None
        
        old_content = old_history.get_content_str()
        new_content = new_history.get_content_str()
        
        return self.generate_diff(old_content, new_content, file_path)
    
    def revert(
        self,
        file_path: str,
        session_id: str,
        version: int,
        write_to_disk: bool = True
    ) -> Optional[FileHistory]:
        """Revert a file to a previous version.
        
        Args:
            file_path: File path
            session_id: Session identifier
            version: Version to revert to
            write_to_disk: Whether to write the reverted content to disk
            
        Returns:
            File history entry for the reverted version, None if not found
        """
        history = self.get_version(file_path, session_id, version)
        
        if history is None:
            logger.warning(
                f"Cannot revert {file_path} to version {version}: version not found"
            )
            return None
        
        if write_to_disk:
            try:
                # Write content to disk
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(history.content)
                
                logger.info(
                    f"Reverted {file_path} to version {version} "
                    f"(session: {session_id})"
                )
                
                # Create a new version entry for the revert
                content_str = history.get_content_str()
                new_history = self.create_version(session_id, file_path, content_str)
                
                # Publish event asynchronously
                _safe_publish_event(
                    self.event_stream.publish(
                        HistoryEvent(
                            type=HistoryEventType.FILE_REVERTED,
                            history=new_history
                        )
                    )
                )
                
                return new_history
            except Exception as e:
                logger.error(f"Failed to revert {file_path} to version {version}: {e}")
                return None
        
        return history
    
    def get_file_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about files in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with file statistics
        """
        conn = self.db.connect()
        
        # Count unique files
        cursor = conn.execute(
            """
            SELECT COUNT(DISTINCT file_path) as file_count
            FROM file_history
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        file_count = row['file_count'] if row else 0
        
        # Count total versions
        cursor = conn.execute(
            """
            SELECT COUNT(*) as version_count
            FROM file_history
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        version_count = row['version_count'] if row else 0
        
        # Get total size
        cursor = conn.execute(
            """
            SELECT SUM(LENGTH(content)) as total_size
            FROM file_history
            WHERE session_id = ?
            """,
            (session_id,)
        )
        row = cursor.fetchone()
        total_size = row['total_size'] if row and row['total_size'] else 0
        
        return {
            'file_count': file_count,
            'version_count': version_count,
            'total_size': total_size,
        }
    
    def subscribe(self) -> HistoryEventStream:
        """Subscribe to file history events.
        
        Returns:
            Event stream for file history updates
        """
        return self.event_stream
    
    def cleanup_old_versions(
        self,
        session_id: str,
        keep_versions: int = 10
    ) -> int:
        """Clean up old versions, keeping only the most recent ones.
        
        Args:
            session_id: Session identifier
            keep_versions: Number of versions to keep per file
            
        Returns:
            Number of versions deleted
        """
        files = self.list_files_in_session(session_id)
        total_deleted = 0
        
        for file_path in files:
            versions = self.list_versions(file_path, session_id)
            
            if len(versions) > keep_versions:
                # Delete oldest versions
                versions_to_delete = versions[:-keep_versions]
                
                with self.db.transaction() as conn:
                    for version in versions_to_delete:
                        conn.execute(
                            "DELETE FROM file_history WHERE id = ?",
                            (version.id,)
                        )
                        total_deleted += 1
        
        if total_deleted > 0:
            logger.info(
                f"Cleaned up {total_deleted} old file versions in session {session_id}"
            )
        
        return total_deleted
