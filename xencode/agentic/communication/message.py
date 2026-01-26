"""
Message structure for inter-agent communication in Xencode
"""
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import uuid


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"


class MessageStatus(Enum):
    """Status of a message in the communication system."""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """Represents a message in the inter-agent communication system."""
    
    # Core message fields
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    sender_id: str = ""
    receiver_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Content fields
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None
    
    # Status and tracking
    status: MessageStatus = MessageStatus.PENDING
    correlation_id: Optional[str] = None  # For tracking related messages
    reply_to: Optional[str] = None  # For request-response patterns
    priority: int = 0  # Higher number means higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'metadata': self.metadata,
            'payload': self.payload,
            'status': self.status.value,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from dictionary representation."""
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=MessageType(data.get('message_type', 'request')),
            sender_id=data.get('sender_id', ''),
            receiver_id=data.get('receiver_id', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            content=data.get('content', ''),
            metadata=data.get('metadata', {}),
            payload=data.get('payload'),
            status=MessageStatus(data.get('status', 'pending')),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            priority=data.get('priority', 0)
        )
    
    def is_request_reply_pair(self, other: 'Message') -> bool:
        """Check if this message and another form a request-reply pair."""
        return (self.message_id == other.reply_to or 
                self.correlation_id == other.correlation_id)
    
    def clone(self) -> 'Message':
        """Create a copy of this message."""
        return Message(
            message_id=str(uuid.uuid4()),  # New ID for the clone
            message_type=self.message_type,
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            timestamp=datetime.now(),
            content=self.content,
            metadata=self.metadata.copy(),
            payload=self.payload.copy() if self.payload else None,
            status=MessageStatus.PENDING,
            correlation_id=self.correlation_id or self.message_id,
            reply_to=self.reply_to,
            priority=self.priority
        )


# Predefined message templates for common interactions
class MessageTemplates:
    """Common message templates for standard agent interactions."""
    
    @staticmethod
    def create_task_request(sender_id: str, receiver_id: str, task: str, 
                          task_id: Optional[str] = None) -> Message:
        """Create a task assignment message."""
        return Message(
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=task,
            payload={'task_id': task_id or str(uuid.uuid4())},
            priority=1
        )
    
    @staticmethod
    def create_task_result(sender_id: str, receiver_id: str, 
                          task_id: str, result: Any) -> Message:
        """Create a task result message."""
        return Message(
            message_type=MessageType.TASK_RESULT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=f"Result for task {task_id}",
            payload={'task_id': task_id, 'result': result},
            priority=1
        )
    
    @staticmethod
    def create_heartbeat(sender_id: str) -> Message:
        """Create a heartbeat message to signal agent availability."""
        return Message(
            message_type=MessageType.HEARTBEAT,
            sender_id=sender_id,
            receiver_id="",  # Broadcast message
            content="Heartbeat",
            payload={'timestamp': datetime.now().isoformat()},
            priority=-1  # Low priority
        )
    
    @staticmethod
    def create_error(sender_id: str, receiver_id: str, 
                    error_msg: str, original_message_id: Optional[str] = None) -> Message:
        """Create an error message."""
        return Message(
            message_type=MessageType.ERROR,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=error_msg,
            payload={'original_message_id': original_message_id},
            priority=10,  # High priority
            status=MessageStatus.FAILED
        )