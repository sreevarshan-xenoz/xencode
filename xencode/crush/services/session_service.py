"""Session service for managing conversation sessions."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from xencode.crush.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


def _safe_publish_event(coro):
    """Safely publish event, handling cases where no event loop is running."""
    try:
        asyncio.create_task(coro)
    except RuntimeError:
        # No event loop running, skip event publishing
        pass


class SessionEventType(Enum):
    """Types of session events."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    BUSY_CHANGED = "busy_changed"


@dataclass
class Session:
    """Represents a conversation session."""
    id: str
    title: str
    created_at: int
    updated_at: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    summary_message_id: Optional[str] = None
    busy: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'cost': self.cost,
            'summary_message_id': self.summary_message_id,
            'busy': self.busy,
        }
    
    @classmethod
    def from_row(cls, row) -> 'Session':
        """Create session from database row."""
        return cls(
            id=row['id'],
            title=row['title'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            prompt_tokens=row['prompt_tokens'],
            completion_tokens=row['completion_tokens'],
            cost=row['cost'],
            summary_message_id=row['summary_message_id'],
            busy=bool(row['busy']),
        )


@dataclass
class SessionEvent:
    """Event for session updates."""
    type: SessionEventType
    session: Optional[Session] = None
    session_id: Optional[str] = None


class EventStream:
    """Async event stream for real-time updates."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to events.
        
        Args:
            subscriber_id: Unique identifier for subscriber
            
        Returns:
            Queue for receiving events
        """
        async with self._lock:
            queue = asyncio.Queue(maxsize=100)
            self._queues[subscriber_id] = queue
            logger.debug(f"Subscriber {subscriber_id} subscribed")
            return queue
    
    async def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from events.
        
        Args:
            subscriber_id: Subscriber to remove
        """
        async with self._lock:
            self._queues.pop(subscriber_id, None)
            logger.debug(f"Subscriber {subscriber_id} unsubscribed")
    
    async def publish(self, event: SessionEvent):
        """Publish event to all subscribers.
        
        Args:
            event: Event to publish
        """
        async with self._lock:
            dead_subscribers = []
            
            for subscriber_id, queue in self._queues.items():
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Queue full for subscriber {subscriber_id}, dropping event"
                    )
                except Exception as e:
                    logger.error(
                        f"Error publishing to subscriber {subscriber_id}: {e}"
                    )
                    dead_subscribers.append(subscriber_id)
            
            # Clean up dead subscribers
            for subscriber_id in dead_subscribers:
                self._queues.pop(subscriber_id, None)


class SessionService:
    """Manages conversation sessions."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize session service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.event_stream = EventStream()
    
    def create(self, title: str) -> Session:
        """Create a new session.
        
        Args:
            title: Session title
            
        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        now = int(time.time())
        
        session = Session(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, title, created_at, updated_at,
                    prompt_tokens, completion_tokens, cost, busy
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    session.prompt_tokens,
                    session.completion_tokens,
                    session.cost,
                    int(session.busy),
                )
            )
        
        logger.info(f"Created session: {session_id}")
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                SessionEvent(type=SessionEventType.CREATED, session=session)
            )
        )
        
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found, None otherwise
        """
        conn = self.db.connect()
        cursor = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return Session.from_row(row)
    
    def list(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """List all sessions sorted by last updated time.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of sessions
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            SELECT * FROM sessions
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(Session.from_row(row))
        
        return sessions
    
    def save(self, session: Session) -> Session:
        """Update session metadata.
        
        Args:
            session: Session to update
            
        Returns:
            Updated session
        """
        session.updated_at = int(time.time())
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET title = ?,
                    updated_at = ?,
                    prompt_tokens = ?,
                    completion_tokens = ?,
                    cost = ?,
                    summary_message_id = ?,
                    busy = ?
                WHERE id = ?
                """,
                (
                    session.title,
                    session.updated_at,
                    session.prompt_tokens,
                    session.completion_tokens,
                    session.cost,
                    session.summary_message_id,
                    int(session.busy),
                    session.id,
                )
            )
        
        logger.debug(f"Updated session: {session.id}")
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                SessionEvent(type=SessionEventType.UPDATED, session=session)
            )
        )
        
        return session
    
    def delete(self, session_id: str) -> None:
        """Delete a session and its associated messages.
        
        Args:
            session_id: Session to delete
        """
        with self.db.transaction() as conn:
            # Foreign key cascade will delete messages and other related data
            conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
        
        logger.info(f"Deleted session: {session_id}")
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                SessionEvent(
                    type=SessionEventType.DELETED,
                    session_id=session_id
                )
            )
        )
    
    def set_busy(self, session_id: str, busy: bool) -> None:
        """Set busy state for a session.
        
        Args:
            session_id: Session identifier
            busy: Busy state
        """
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE sessions SET busy = ?, updated_at = ? WHERE id = ?",
                (int(busy), int(time.time()), session_id)
            )
        
        logger.debug(f"Set session {session_id} busy={busy}")
        
        # Publish event asynchronously
        session = self.get(session_id)
        if session:
            _safe_publish_event(
                self.event_stream.publish(
                    SessionEvent(
                        type=SessionEventType.BUSY_CHANGED,
                        session=session
                    )
                )
            )
    
    def update_tokens(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float
    ) -> None:
        """Update token usage and cost for a session.
        
        Args:
            session_id: Session identifier
            prompt_tokens: Number of prompt tokens to add
            completion_tokens: Number of completion tokens to add
            cost: Cost to add
        """
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET prompt_tokens = prompt_tokens + ?,
                    completion_tokens = completion_tokens + ?,
                    cost = cost + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    int(time.time()),
                    session_id
                )
            )
        
        logger.debug(
            f"Updated tokens for session {session_id}: "
            f"+{prompt_tokens} prompt, +{completion_tokens} completion, +${cost:.4f}"
        )
    
    def subscribe(self) -> EventStream:
        """Subscribe to session events.
        
        Returns:
            Event stream for session updates
        """
        return self.event_stream
