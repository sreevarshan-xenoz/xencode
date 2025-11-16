"""Permission service for managing tool execution permissions."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
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


class PermissionStatus(Enum):
    """Permission request status."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    AUTO_APPROVED = "auto-approved"
    TIMEOUT = "timeout"


@dataclass
class PermissionRequest:
    """Represents a permission request for a tool operation."""
    id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    action: str
    path: Optional[str] = None
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    status: PermissionStatus = PermissionStatus.PENDING
    created_at: int = field(default_factory=lambda: int(time.time()))
    resolved_at: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'tool_call_id': self.tool_call_id,
            'tool_name': self.tool_name,
            'action': self.action,
            'path': self.path,
            'description': self.description,
            'params': self.params,
            'status': self.status.value,
            'created_at': self.created_at,
            'resolved_at': self.resolved_at,
        }
    
    @classmethod
    def from_row(cls, row) -> 'PermissionRequest':
        """Create from database row."""
        params = json.loads(row['params']) if row['params'] else None
        
        return cls(
            id=row['id'],
            session_id=row['session_id'],
            tool_call_id=row['tool_call_id'],
            tool_name=row['tool_name'],
            action=row['action'],
            path=row['path'],
            description=row['description'],
            params=params,
            status=PermissionStatus(row['status']),
            created_at=row['created_at'],
            resolved_at=row['resolved_at'],
        )


class PermissionEventType(Enum):
    """Types of permission events."""
    REQUESTED = "requested"
    APPROVED = "approved"
    DENIED = "denied"
    AUTO_APPROVED = "auto_approved"
    TIMEOUT = "timeout"


@dataclass
class PermissionEvent:
    """Event for permission updates."""
    type: PermissionEventType
    request: PermissionRequest


class PermissionEventStream:
    """Async event stream for permission updates."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._pending_requests: Dict[str, asyncio.Future] = {}
    
    async def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to events."""
        async with self._lock:
            queue = asyncio.Queue(maxsize=100)
            self._queues[subscriber_id] = queue
            logger.debug(f"Permission subscriber {subscriber_id} subscribed")
            return queue
    
    async def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from events."""
        async with self._lock:
            self._queues.pop(subscriber_id, None)
            logger.debug(f"Permission subscriber {subscriber_id} unsubscribed")
    
    async def publish(self, event: PermissionEvent):
        """Publish event to all subscribers."""
        async with self._lock:
            dead_subscribers = []
            
            for subscriber_id, queue in self._queues.items():
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Permission queue full for subscriber {subscriber_id}, dropping event"
                    )
                except Exception as e:
                    logger.error(
                        f"Error publishing permission event to subscriber {subscriber_id}: {e}"
                    )
                    dead_subscribers.append(subscriber_id)
            
            # Clean up dead subscribers
            for subscriber_id in dead_subscribers:
                self._queues.pop(subscriber_id, None)
            
            # Resolve pending futures
            request_id = event.request.id
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.done():
                    future.set_result(event.request.status)
    
    async def wait_for_approval(self, request_id: str, timeout: float = 300.0) -> PermissionStatus:
        """Wait for a permission request to be approved or denied.
        
        Args:
            request_id: Permission request ID
            timeout: Timeout in seconds (default 5 minutes)
            
        Returns:
            Final permission status
        """
        async with self._lock:
            future = asyncio.Future()
            self._pending_requests[request_id] = future
        
        try:
            status = await asyncio.wait_for(future, timeout=timeout)
            return status
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending_requests.pop(request_id, None)
            return PermissionStatus.TIMEOUT


class PermissionService:
    """Manages tool execution permissions."""
    
    def __init__(
        self,
        db: DatabaseConnection,
        working_dir: str,
        skip_requests: bool = False,
        allowed_tools: Optional[List[str]] = None
    ):
        """Initialize permission service.
        
        Args:
            db: Database connection
            working_dir: Working directory for path validation
            skip_requests: If True, auto-approve all requests (YOLO mode)
            allowed_tools: List of tools that bypass permission checks
        """
        self.db = db
        self.working_dir = working_dir
        self._skip_requests = skip_requests
        self.allowed_tools: Set[str] = set(allowed_tools or [])
        self.event_stream = PermissionEventStream()
        self._auto_approve_sessions: Set[str] = set()
    
    def create_request(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        action: str,
        path: Optional[str] = None,
        description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> PermissionRequest:
        """Create a permission request.
        
        Args:
            session_id: Session identifier
            tool_call_id: Tool call identifier
            tool_name: Name of the tool
            action: Action being performed
            path: Optional file path
            description: Human-readable description
            params: Tool parameters
            
        Returns:
            Created permission request
        """
        request_id = str(uuid.uuid4())
        
        request = PermissionRequest(
            id=request_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            action=action,
            path=path,
            description=description,
            params=params,
        )
        
        # Check if should auto-approve
        if self._should_auto_approve(session_id, tool_name):
            request.status = PermissionStatus.AUTO_APPROVED
            request.resolved_at = int(time.time())
        
        # Serialize params
        params_json = json.dumps(params) if params else None
        
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO permission_requests (
                    id, session_id, tool_call_id, tool_name, action,
                    path, description, params, status, created_at, resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.id,
                    request.session_id,
                    request.tool_call_id,
                    request.tool_name,
                    request.action,
                    request.path,
                    request.description,
                    params_json,
                    request.status.value,
                    request.created_at,
                    request.resolved_at,
                )
            )
        
        logger.info(
            f"Created permission request: {request_id} for {tool_name} "
            f"in session {session_id} (status: {request.status.value})"
        )
        
        # Publish event asynchronously
        event_type = (
            PermissionEventType.AUTO_APPROVED
            if request.status == PermissionStatus.AUTO_APPROVED
            else PermissionEventType.REQUESTED
        )
        _safe_publish_event(
            self.event_stream.publish(
                PermissionEvent(type=event_type, request=request)
            )
        )
        
        return request
    
    async def request(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        action: str,
        path: Optional[str] = None,
        description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0
    ) -> bool:
        """Request permission for an operation.
        
        Args:
            session_id: Session identifier
            tool_call_id: Tool call identifier
            tool_name: Name of the tool
            action: Action being performed
            path: Optional file path
            description: Human-readable description
            params: Tool parameters
            timeout: Timeout in seconds
            
        Returns:
            True if approved, False if denied
        """
        request = self.create_request(
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            action=action,
            path=path,
            description=description,
            params=params
        )
        
        # If auto-approved, return immediately
        if request.status == PermissionStatus.AUTO_APPROVED:
            return True
        
        # Wait for approval
        status = await self.event_stream.wait_for_approval(request.id, timeout)
        
        if status == PermissionStatus.TIMEOUT:
            # Update request status
            self._update_status(request.id, PermissionStatus.TIMEOUT)
            logger.warning(f"Permission request {request.id} timed out")
            return False
        
        return status == PermissionStatus.APPROVED
    
    def approve(self, request_id: str) -> None:
        """Approve a permission request.
        
        Args:
            request_id: Request to approve
        """
        self._update_status(request_id, PermissionStatus.APPROVED)
        
        request = self.get(request_id)
        if request:
            logger.info(f"Approved permission request: {request_id}")
            _safe_publish_event(
                self.event_stream.publish(
                    PermissionEvent(type=PermissionEventType.APPROVED, request=request)
                )
            )
    
    def deny(self, request_id: str) -> None:
        """Deny a permission request.
        
        Args:
            request_id: Request to deny
        """
        self._update_status(request_id, PermissionStatus.DENIED)
        
        request = self.get(request_id)
        if request:
            logger.info(f"Denied permission request: {request_id}")
            _safe_publish_event(
                self.event_stream.publish(
                    PermissionEvent(type=PermissionEventType.DENIED, request=request)
                )
            )
    
    def _update_status(self, request_id: str, status: PermissionStatus) -> None:
        """Update permission request status.
        
        Args:
            request_id: Request to update
            status: New status
        """
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE permission_requests
                SET status = ?, resolved_at = ?
                WHERE id = ?
                """,
                (status.value, int(time.time()), request_id)
            )
    
    def get(self, request_id: str) -> Optional[PermissionRequest]:
        """Get a permission request by ID.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Permission request if found, None otherwise
        """
        conn = self.db.connect()
        cursor = conn.execute(
            "SELECT * FROM permission_requests WHERE id = ?",
            (request_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return PermissionRequest.from_row(row)
    
    def list_pending(self, session_id: Optional[str] = None) -> List[PermissionRequest]:
        """List pending permission requests.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            List of pending requests
        """
        conn = self.db.connect()
        
        if session_id:
            cursor = conn.execute(
                """
                SELECT * FROM permission_requests
                WHERE session_id = ? AND status = ?
                ORDER BY created_at ASC
                """,
                (session_id, PermissionStatus.PENDING.value)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM permission_requests
                WHERE status = ?
                ORDER BY created_at ASC
                """,
                (PermissionStatus.PENDING.value,)
            )
        
        requests = []
        for row in cursor.fetchall():
            requests.append(PermissionRequest.from_row(row))
        
        return requests
    
    def auto_approve_session(self, session_id: str) -> None:
        """Auto-approve all requests for a session.
        
        Args:
            session_id: Session to auto-approve
        """
        self._auto_approve_sessions.add(session_id)
        logger.info(f"Auto-approval enabled for session: {session_id}")
    
    def disable_auto_approve_session(self, session_id: str) -> None:
        """Disable auto-approval for a session.
        
        Args:
            session_id: Session to disable auto-approval
        """
        self._auto_approve_sessions.discard(session_id)
        logger.info(f"Auto-approval disabled for session: {session_id}")
    
    def skip_requests(self) -> bool:
        """Check if permission requests should be skipped.
        
        Returns:
            True if in YOLO mode
        """
        return self._skip_requests
    
    def set_skip_requests(self, skip: bool) -> None:
        """Set skip requests mode (YOLO mode).
        
        Args:
            skip: Whether to skip permission requests
        """
        self._skip_requests = skip
        logger.info(f"Skip requests (YOLO mode) set to: {skip}")
    
    def _should_auto_approve(self, session_id: str, tool_name: str) -> bool:
        """Check if a request should be auto-approved.
        
        Args:
            session_id: Session identifier
            tool_name: Tool name
            
        Returns:
            True if should auto-approve
        """
        # Check YOLO mode
        if self._skip_requests:
            return True
        
        # Check session-level auto-approval
        if session_id in self._auto_approve_sessions:
            return True
        
        # Check tool-level auto-approval
        if tool_name in self.allowed_tools:
            return True
        
        return False
    
    def subscribe(self) -> PermissionEventStream:
        """Subscribe to permission events.
        
        Returns:
            Event stream for permission updates
        """
        return self.event_stream
    
    def cleanup_old_requests(self, max_age_seconds: int = 86400) -> int:
        """Clean up old permission requests.
        
        Args:
            max_age_seconds: Maximum age in seconds (default 24 hours)
            
        Returns:
            Number of requests deleted
        """
        cutoff_time = int(time.time()) - max_age_seconds
        
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM permission_requests WHERE created_at < ?",
                (cutoff_time,)
            )
            deleted = cursor.rowcount
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old permission requests")
        
        return deleted
