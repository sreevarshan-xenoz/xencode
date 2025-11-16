"""Message service for storing and retrieving conversation messages."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
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


class MessageRole(Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPartType(Enum):
    """Content part types."""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    BINARY = "binary"


@dataclass
class ContentPart:
    """Represents a part of message content."""
    type: ContentPartType
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type.value,
            **self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentPart':
        """Create from dictionary."""
        part_type = ContentPartType(data['type'])
        part_data = {k: v for k, v in data.items() if k != 'type'}
        return cls(type=part_type, data=part_data)
    
    @classmethod
    def text(cls, text: str) -> 'ContentPart':
        """Create text content part."""
        return cls(type=ContentPartType.TEXT, data={'text': text})
    
    @classmethod
    def tool_call(cls, tool: str, input_data: Dict[str, Any], call_id: Optional[str] = None) -> 'ContentPart':
        """Create tool call content part."""
        return cls(
            type=ContentPartType.TOOL_CALL,
            data={
                'tool': tool,
                'input': input_data,
                'call_id': call_id or str(uuid.uuid4())
            }
        )
    
    @classmethod
    def tool_result(cls, tool: str, result: str, call_id: str, metadata: Optional[Dict] = None) -> 'ContentPart':
        """Create tool result content part."""
        return cls(
            type=ContentPartType.TOOL_RESULT,
            data={
                'tool': tool,
                'result': result,
                'call_id': call_id,
                'metadata': metadata or {}
            }
        )
    
    @classmethod
    def binary(cls, filename: str, encoding: str, data: str) -> 'ContentPart':
        """Create binary content part."""
        return cls(
            type=ContentPartType.BINARY,
            data={
                'filename': filename,
                'encoding': encoding,
                'data': data
            }
        )


@dataclass
class Message:
    """Represents a conversation message."""
    id: str
    session_id: str
    role: MessageRole
    parts: List[ContentPart] = field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    is_summary: bool = False
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role.value,
            'parts': [part.to_dict() for part in self.parts],
            'model': self.model,
            'provider': self.provider,
            'created_at': self.created_at,
            'is_summary': self.is_summary,
            'token_count': self.token_count,
        }
    
    @classmethod
    def from_row(cls, row) -> 'Message':
        """Create message from database row."""
        content_data = json.loads(row['content'])
        parts = [ContentPart.from_dict(part) for part in content_data.get('parts', [])]
        
        return cls(
            id=row['id'],
            session_id=row['session_id'],
            role=MessageRole(row['role']),
            parts=parts,
            model=row['model'],
            provider=row['provider'],
            created_at=row['created_at'],
            is_summary=bool(row['is_summary']),
            token_count=row['token_count'],
        )
    
    def append_content(self, text: str) -> None:
        """Append text to message (for streaming).
        
        Args:
            text: Text to append
        """
        # Find or create text part
        text_part = None
        for part in self.parts:
            if part.type == ContentPartType.TEXT:
                text_part = part
                break
        
        if text_part is None:
            text_part = ContentPart.text("")
            self.parts.append(text_part)
        
        text_part.data['text'] = text_part.data.get('text', '') + text
    
    def add_tool_call(self, tool: str, input_data: Dict[str, Any], call_id: Optional[str] = None) -> str:
        """Add a tool call to the message.
        
        Args:
            tool: Tool name
            input_data: Tool input parameters
            call_id: Optional call ID
            
        Returns:
            Tool call ID
        """
        tool_call = ContentPart.tool_call(tool, input_data, call_id)
        self.parts.append(tool_call)
        return tool_call.data['call_id']
    
    def add_tool_result(self, tool: str, result: str, call_id: str, metadata: Optional[Dict] = None) -> None:
        """Add a tool result to the message.
        
        Args:
            tool: Tool name
            result: Tool result
            call_id: Tool call ID
            metadata: Optional metadata
        """
        tool_result = ContentPart.tool_result(tool, result, call_id, metadata)
        self.parts.append(tool_result)
    
    def get_text_content(self) -> str:
        """Get all text content from message.
        
        Returns:
            Concatenated text content
        """
        texts = []
        for part in self.parts:
            if part.type == ContentPartType.TEXT:
                texts.append(part.data.get('text', ''))
        return ''.join(texts)
    
    def to_ai_message(self) -> Dict[str, Any]:
        """Convert to format expected by AI providers.
        
        Returns:
            Message in AI provider format
        """
        # This is a simplified version - actual implementation would vary by provider
        content = []
        
        for part in self.parts:
            if part.type == ContentPartType.TEXT:
                content.append({
                    'type': 'text',
                    'text': part.data['text']
                })
            elif part.type == ContentPartType.TOOL_CALL:
                content.append({
                    'type': 'tool_use',
                    'id': part.data['call_id'],
                    'name': part.data['tool'],
                    'input': part.data['input']
                })
            elif part.type == ContentPartType.TOOL_RESULT:
                content.append({
                    'type': 'tool_result',
                    'tool_use_id': part.data['call_id'],
                    'content': part.data['result']
                })
        
        return {
            'role': self.role.value,
            'content': content
        }


class MessageEventType(Enum):
    """Types of message events."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    STREAMING = "streaming"


@dataclass
class MessageEvent:
    """Event for message updates."""
    type: MessageEventType
    message: Optional[Message] = None
    message_id: Optional[str] = None
    session_id: Optional[str] = None


class MessageEventStream:
    """Async event stream for message updates."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to events."""
        async with self._lock:
            queue = asyncio.Queue(maxsize=100)
            self._queues[subscriber_id] = queue
            logger.debug(f"Message subscriber {subscriber_id} subscribed")
            return queue
    
    async def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from events."""
        async with self._lock:
            self._queues.pop(subscriber_id, None)
            logger.debug(f"Message subscriber {subscriber_id} unsubscribed")
    
    async def publish(self, event: MessageEvent):
        """Publish event to all subscribers."""
        async with self._lock:
            dead_subscribers = []
            
            for subscriber_id, queue in self._queues.items():
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Message queue full for subscriber {subscriber_id}, dropping event"
                    )
                except Exception as e:
                    logger.error(
                        f"Error publishing message event to subscriber {subscriber_id}: {e}"
                    )
                    dead_subscribers.append(subscriber_id)
            
            # Clean up dead subscribers
            for subscriber_id in dead_subscribers:
                self._queues.pop(subscriber_id, None)


@dataclass
class CreateMessageParams:
    """Parameters for creating a message."""
    session_id: str
    role: MessageRole
    content: Union[str, List[ContentPart]]
    model: Optional[str] = None
    provider: Optional[str] = None
    is_summary: bool = False


class MessageService:
    """Manages conversation messages."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize message service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.event_stream = MessageEventStream()
    
    def create(self, params: CreateMessageParams) -> Message:
        """Create a new message.
        
        Args:
            params: Message creation parameters
            
        Returns:
            Created message
        """
        message_id = str(uuid.uuid4())
        now = int(time.time())
        
        # Convert content to parts
        if isinstance(params.content, str):
            parts = [ContentPart.text(params.content)]
        else:
            parts = params.content
        
        message = Message(
            id=message_id,
            session_id=params.session_id,
            role=params.role,
            parts=parts,
            model=params.model,
            provider=params.provider,
            created_at=now,
            is_summary=params.is_summary,
        )
        
        # Serialize content
        content_json = json.dumps({'parts': [part.to_dict() for part in parts]})
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, role, content, model, provider,
                    created_at, is_summary, token_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role.value,
                    content_json,
                    message.model,
                    message.provider,
                    message.created_at,
                    int(message.is_summary),
                    message.token_count,
                )
            )
        
        logger.info(f"Created message: {message_id} in session {params.session_id}")
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                MessageEvent(type=MessageEventType.CREATED, message=message)
            )
        )
        
        return message
    
    def update(self, message: Message) -> None:
        """Update an existing message (for streaming).
        
        Args:
            message: Message to update
        """
        # Serialize content
        content_json = json.dumps({'parts': [part.to_dict() for part in message.parts]})
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE messages
                SET content = ?,
                    token_count = ?,
                    model = ?,
                    provider = ?
                WHERE id = ?
                """,
                (
                    content_json,
                    message.token_count,
                    message.model,
                    message.provider,
                    message.id,
                )
            )
        
        logger.debug(f"Updated message: {message.id}")
        
        # Publish event asynchronously
        _safe_publish_event(
            self.event_stream.publish(
                MessageEvent(type=MessageEventType.UPDATED, message=message)
            )
        )
    
    def list(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """List all messages in a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages in chronological order
        """
        conn = self.db.connect()
        
        if limit:
            cursor = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, limit)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,)
            )
        
        messages = []
        for row in cursor.fetchall():
            messages.append(Message.from_row(row))
        
        return messages
    
    def get(self, message_id: str) -> Optional[Message]:
        """Get a message by ID.
        
        Args:
            message_id: Message identifier
            
        Returns:
            Message if found, None otherwise
        """
        conn = self.db.connect()
        cursor = conn.execute(
            "SELECT * FROM messages WHERE id = ?",
            (message_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return Message.from_row(row)
    
    def delete(self, message_id: str) -> None:
        """Delete a message.
        
        Args:
            message_id: Message to delete
        """
        # Get message before deleting for event
        message = self.get(message_id)
        
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
        
        logger.info(f"Deleted message: {message_id}")
        
        # Publish event asynchronously
        if message:
            _safe_publish_event(
                self.event_stream.publish(
                    MessageEvent(
                        type=MessageEventType.DELETED,
                        message_id=message_id,
                        session_id=message.session_id
                    )
                )
            )
    
    def delete_by_session(self, session_id: str) -> None:
        """Delete all messages in a session.
        
        Args:
            session_id: Session identifier
        """
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
        
        logger.info(f"Deleted all messages in session: {session_id}")
    
    def count(self, session_id: str) -> int:
        """Count messages in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of messages
        """
        conn = self.db.connect()
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM messages WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return row['count'] if row else 0
    
    def subscribe(self) -> MessageEventStream:
        """Subscribe to message events.
        
        Returns:
            Event stream for message updates
        """
        return self.event_stream
