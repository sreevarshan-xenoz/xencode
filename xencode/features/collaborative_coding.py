"""
Collaborative Coding Feature

Provides real-time collaborative editing with WebSocket support,
presence tracking, and intelligent conflict resolution.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .base import FeatureBase, FeatureConfig, FeatureError


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ChangeType(Enum):
    """Types of document changes"""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    CURSOR_MOVE = "cursor_move"


@dataclass
class CollaborativeConfig:
    """Configuration for collaborative coding"""
    enabled: bool = True
    websocket_port: int = 8765
    websocket_host: str = "localhost"
    max_connections: int = 10
    sync_debounce_ms: int = 100
    batch_size: int = 50
    enable_voice_chat: bool = False
    enable_video_chat: bool = False
    session_timeout_minutes: int = 60
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollaborativeConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            websocket_port=data.get('websocket_port', 8765),
            websocket_host=data.get('websocket_host', 'localhost'),
            max_connections=data.get('max_connections', 10),
            sync_debounce_ms=data.get('sync_debounce_ms', 100),
            batch_size=data.get('batch_size', 50),
            enable_voice_chat=data.get('enable_voice_chat', False),
            enable_video_chat=data.get('enable_video_chat', False),
            session_timeout_minutes=data.get('session_timeout_minutes', 60)
        )


@dataclass
class User:
    """Represents a user in a collaboration session"""
    user_id: str
    username: str
    color: str
    cursor_position: int = 0
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    is_active: bool = True
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())

    # Presence tracking fields
    status: str = "online"  # online, offline, idle, away
    is_typing: bool = False
    current_file: Optional[str] = None
    current_location: Optional[str] = None
    idle_since: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'color': self.color,
            'cursor_position': self.cursor_position,
            'selection_start': self.selection_start,
            'selection_end': self.selection_end,
            'is_active': self.is_active,
            'last_activity': self.last_activity,
            'status': self.status,
            'is_typing': self.is_typing,
            'current_file': self.current_file,
            'current_location': self.current_location,
            'idle_since': self.idle_since
        }



@dataclass
class Change:
    """Represents a document change"""
    change_id: str
    user_id: str
    change_type: ChangeType
    position: int
    content: str
    timestamp: str
    version: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'change_id': self.change_id,
            'user_id': self.user_id,
            'change_type': self.change_type.value,
            'position': self.position,
            'content': self.content,
            'timestamp': self.timestamp,
            'version': self.version
        }


@dataclass
class Session:
    """Represents a collaboration session"""
    session_id: str
    name: str
    owner_id: str
    created_at: str
    document_content: str = ""
    document_version: int = 0
    participants: Dict[str, User] = field(default_factory=dict)
    change_history: List[Change] = field(default_factory=list)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'name': self.name,
            'owner_id': self.owner_id,
            'created_at': self.created_at,
            'document_version': self.document_version,
            'participants': {uid: user.to_dict() for uid, user in self.participants.items()},
            'participant_count': len(self.participants),
            'is_active': self.is_active
        }


class CollaborationManager:
    """Manages collaboration sessions and participants"""
    
    def __init__(self, config: CollaborativeConfig):
        self.config = config
        self.sessions: Dict[str, Session] = {}
        self.user_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
        ]
        self.color_index = 0
        self.presence_manager = PresenceManager(idle_timeout_seconds=300)
    
    def _get_next_color(self) -> str:
        """Get next available color for a user"""
        color = self.user_colors[self.color_index % len(self.user_colors)]
        self.color_index += 1
        return color
    
    async def start_session(self, name: str, owner_id: str, username: str) -> Dict[str, Any]:
        """Start a new collaboration session"""
        session_id = str(uuid.uuid4())
        
        # Create owner user
        owner = User(
            user_id=owner_id,
            username=username,
            color=self._get_next_color()
        )
        
        # Create session
        session = Session(
            session_id=session_id,
            name=name,
            owner_id=owner_id,
            created_at=datetime.now().isoformat(),
            participants={owner_id: owner}
        )
        
        self.sessions[session_id] = session
        
        return {
            'success': True,
            'session_id': session_id,
            'session': session.to_dict(),
            'user': owner.to_dict()
        }
    
    async def join_session(self, session_id: str, user_id: str, username: str) -> Dict[str, Any]:
        """Join an existing collaboration session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        # Check if session is full
        if len(session.participants) >= self.config.max_connections:
            return {
                'success': False,
                'error': 'Session is full'
            }
        
        # Check if user already in session
        if user_id in session.participants:
            user = session.participants[user_id]
            user.is_active = True
            user.last_activity = datetime.now().isoformat()
        else:
            # Create new user
            user = User(
                user_id=user_id,
                username=username,
                color=self._get_next_color()
            )
            session.participants[user_id] = user
        
        return {
            'success': True,
            'session_id': session_id,
            'session': session.to_dict(),
            'user': user.to_dict(),
            'document_content': session.document_content,
            'document_version': session.document_version
        }
    
    async def leave_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Leave a collaboration session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id in session.participants:
            session.participants[user_id].is_active = False
            
            # If owner leaves, close session
            if user_id == session.owner_id:
                session.is_active = False
                return {
                    'success': True,
                    'session_closed': True,
                    'message': 'Session closed by owner'
                }
        
        return {
            'success': True,
            'session_closed': False
        }
    
    async def sync_change(self, session_id: str, user_id: str, change_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize a change across all participants"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return {
                'success': False,
                'error': 'User not in session'
            }
        
        # Create change object
        change = Change(
            change_id=str(uuid.uuid4()),
            user_id=user_id,
            change_type=ChangeType(change_data.get('change_type', 'insert')),
            position=change_data.get('position', 0),
            content=change_data.get('content', ''),
            timestamp=datetime.now().isoformat(),
            version=session.document_version + 1
        )
        
        # Apply change to document
        await self._apply_change(session, change)
        
        # Add to history
        session.change_history.append(change)
        
        # Update user activity
        session.participants[user_id].last_activity = datetime.now().isoformat()
        
        return {
            'success': True,
            'change': change.to_dict(),
            'document_version': session.document_version,
            'broadcast_to': [uid for uid in session.participants.keys() if uid != user_id]
        }
    
    async def _apply_change(self, session: Session, change: Change) -> None:
        """Apply a change to the session document"""
        if change.change_type == ChangeType.INSERT:
            # Insert content at position
            session.document_content = (
                session.document_content[:change.position] +
                change.content +
                session.document_content[change.position:]
            )
        elif change.change_type == ChangeType.DELETE:
            # Delete content from position
            end_pos = change.position + len(change.content)
            session.document_content = (
                session.document_content[:change.position] +
                session.document_content[end_pos:]
            )
        elif change.change_type == ChangeType.REPLACE:
            # Replace content at position
            end_pos = change.position + len(change.content)
            session.document_content = (
                session.document_content[:change.position] +
                change.content +
                session.document_content[end_pos:]
            )
        
        # Increment version
        session.document_version += 1
    
    async def update_cursor(self, session_id: str, user_id: str, 
                          cursor_position: int, 
                          selection_start: Optional[int] = None,
                          selection_end: Optional[int] = None) -> Dict[str, Any]:
        """Update user cursor position"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return {
                'success': False,
                'error': 'User not in session'
            }
        
        user = session.participants[user_id]
        user.cursor_position = cursor_position
        user.selection_start = selection_start
        user.selection_end = selection_end
        user.last_activity = datetime.now().isoformat()
        
        return {
            'success': True,
            'user': user.to_dict(),
            'broadcast_to': [uid for uid in session.participants.keys() if uid != user_id]
        }
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        return {
            'success': True,
            'session': session.to_dict()
        }
    
    async def list_sessions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """List all active sessions"""
        sessions = []
        
        for session in self.sessions.values():
            if not session.is_active:
                continue
            
            # Filter by user if specified
            if user_id and user_id not in session.participants:
                continue
            
            sessions.append(session.to_dict())
        
        return {
            'success': True,
            'sessions': sessions,
            'count': len(sessions)
        }
    
    async def close_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Close a collaboration session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        # Only owner can close session
        if user_id != session.owner_id:
            return {
                'success': False,
                'error': 'Only session owner can close the session'
            }
        
        session.is_active = False
        
        return {
            'success': True,
            'message': 'Session closed successfully'
        }
    
    async def update_user_presence(self, session_id: str, user_id: str, status: str) -> Dict[str, Any]:
        """Update user presence status"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return {
                'success': False,
                'error': 'User not in session'
            }
        
        user = session.participants[user_id]
        result = await self.presence_manager.update_presence(user, status)
        
        return result
    
    async def update_typing_status(self, session_id: str, user_id: str, is_typing: bool) -> Dict[str, Any]:
        """Update user typing status"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return {
                'success': False,
                'error': 'User not in session'
            }
        
        user = session.participants[user_id]
        result = await self.presence_manager.update_typing_status(user, is_typing)
        
        return {
            **result,
            'broadcast_to': [uid for uid in session.participants.keys() if uid != user_id]
        }
    
    async def update_file_location(self, session_id: str, user_id: str, 
                                   file_path: Optional[str], location: Optional[str] = None) -> Dict[str, Any]:
        """Update user's current file and location"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return {
                'success': False,
                'error': 'User not in session'
            }
        
        user = session.participants[user_id]
        result = await self.presence_manager.update_file_location(user, file_path, location)
        
        return {
            **result,
            'broadcast_to': [uid for uid in session.participants.keys() if uid != user_id]
        }
    
    async def get_session_presence(self, session_id: str) -> Dict[str, Any]:
        """Get presence information for all users in a session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        return await self.presence_manager.get_all_presence(session.participants)
    
    async def check_idle_users(self, session_id: str) -> Dict[str, Any]:
        """Check and update idle status for all users in a session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': f'Session not found: {session_id}'
            }
        
        session = self.sessions[session_id]
        idle_users = []
        
        for user_id, user in session.participants.items():
            idle_check = await self.presence_manager.check_idle_status(user)
            if idle_check['is_idle']:
                idle_users.append({
                    'user_id': user_id,
                    'username': user.username,
                    'seconds_idle': idle_check['seconds_since_activity']
                })
        
        return {
            'success': True,
            'idle_users': idle_users,
            'idle_count': len(idle_users)
        }


class WebSocketEditor:
    """WebSocket-based real-time editor"""
    
    def __init__(self, config: CollaborativeConfig, manager: CollaborationManager):
        self.config = config
        self.manager = manager
        self.connections: Dict[str, Any] = {}  # user_id -> websocket
        self.connection_states: Dict[str, ConnectionState] = {}
        self.pending_changes: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> changes
        self.server = None
    
    async def start_server(self) -> Dict[str, Any]:
        """Start WebSocket server"""
        try:
            # Note: Actual WebSocket server implementation would use websockets library
            # This is a placeholder for the interface
            return {
                'success': True,
                'host': self.config.websocket_host,
                'port': self.config.websocket_port,
                'message': f'WebSocket server started on {self.config.websocket_host}:{self.config.websocket_port}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to start WebSocket server: {str(e)}'
            }
    
    async def stop_server(self) -> Dict[str, Any]:
        """Stop WebSocket server"""
        try:
            # Close all connections
            for user_id in list(self.connections.keys()):
                await self.disconnect(user_id)
            
            return {
                'success': True,
                'message': 'WebSocket server stopped'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to stop WebSocket server: {str(e)}'
            }
    
    async def connect(self, user_id: str, websocket: Any) -> Dict[str, Any]:
        """Handle new WebSocket connection"""
        self.connections[user_id] = websocket
        self.connection_states[user_id] = ConnectionState.CONNECTED
        
        return {
            'success': True,
            'user_id': user_id,
            'state': ConnectionState.CONNECTED.value
        }
    
    async def disconnect(self, user_id: str) -> Dict[str, Any]:
        """Handle WebSocket disconnection"""
        if user_id in self.connections:
            del self.connections[user_id]
        
        if user_id in self.connection_states:
            self.connection_states[user_id] = ConnectionState.DISCONNECTED
        
        return {
            'success': True,
            'user_id': user_id,
            'state': ConnectionState.DISCONNECTED.value
        }
    
    async def broadcast_change(self, session_id: str, change: Dict[str, Any], 
                             exclude_user: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast a change to all participants in a session"""
        session_info = await self.manager.get_session_info(session_id)
        
        if not session_info['success']:
            return session_info
        
        session = session_info['session']
        broadcast_count = 0
        
        for user_id in session['participants'].keys():
            if user_id == exclude_user:
                continue
            
            if user_id in self.connections:
                # In real implementation, would send via WebSocket
                # await self.connections[user_id].send(json.dumps(change))
                broadcast_count += 1
        
        return {
            'success': True,
            'broadcast_count': broadcast_count
        }
    
    async def broadcast_cursor(self, session_id: str, user_data: Dict[str, Any],
                             exclude_user: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast cursor position to all participants"""
        return await self.broadcast_change(session_id, {
            'type': 'cursor_update',
            'data': user_data
        }, exclude_user)
    
    async def get_connection_state(self, user_id: str) -> ConnectionState:
        """Get connection state for a user"""
        return self.connection_states.get(user_id, ConnectionState.DISCONNECTED)
    
    async def batch_changes(self, session_id: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch multiple changes for efficient synchronization"""
        if session_id not in self.pending_changes:
            self.pending_changes[session_id] = []
        
        self.pending_changes[session_id].extend(changes)
        
        # If batch size reached, flush changes
        if len(self.pending_changes[session_id]) >= self.config.batch_size:
            return await self.flush_changes(session_id)
        
        return {
            'success': True,
            'pending_count': len(self.pending_changes[session_id])
        }
    
    async def flush_changes(self, session_id: str) -> Dict[str, Any]:
        """Flush pending changes"""
        if session_id not in self.pending_changes:
            return {
                'success': True,
                'flushed_count': 0
            }
        
        changes = self.pending_changes[session_id]
        self.pending_changes[session_id] = []
        
        # Broadcast all changes
        for change in changes:
            await self.broadcast_change(session_id, change)
        
        return {
            'success': True,
            'flushed_count': len(changes)
        }
    
    async def broadcast_presence(self, session_id: str, user_data: Dict[str, Any],
                                exclude_user: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast presence update to all participants"""
        return await self.broadcast_change(session_id, {
            'type': 'presence_update',
            'data': user_data
        }, exclude_user)
    
    async def broadcast_typing(self, session_id: str, user_data: Dict[str, Any],
                              exclude_user: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast typing status to all participants"""
        return await self.broadcast_change(session_id, {
            'type': 'typing_update',
            'data': user_data
        }, exclude_user)
    
    async def broadcast_file_location(self, session_id: str, user_data: Dict[str, Any],
                                     exclude_user: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast file location change to all participants"""
        return await self.broadcast_change(session_id, {
            'type': 'file_location_update',
            'data': user_data
        }, exclude_user)


class ConflictResolver:
    """Handles merge conflicts in collaborative editing"""
    
    def __init__(self):
        self.resolution_strategies = {
            'last_write_wins': self._last_write_wins,
            'operational_transform': self._operational_transform,
            'crdt': self._crdt_merge,
            'smart_merge': self._smart_merge
        }
        # Edit history for rollback support
        self.edit_history = {}  # session_id -> history
        self.max_history_size = 1000
    
    async def resolve_conflict(self, changes: List[Change],
                             strategy: str = 'operational_transform',
                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """Resolve conflicts between concurrent changes"""
        if strategy not in self.resolution_strategies:
            return {
                'success': False,
                'error': f'Unknown resolution strategy: {strategy}'
            }
        
        resolver = self.resolution_strategies[strategy]
        result = await resolver(changes)
        
        # Record in history if session_id provided
        if result.get('success') and session_id:
            self._record_resolution(session_id, changes, result)
        
        return result
    

    def _record_resolution(self, session_id: str, original_changes: List[Change],
                          resolution: Dict[str, Any]) -> None:
        """Record conflict resolution in history for rollback"""
        if session_id not in self.edit_history:
            self.edit_history[session_id] = []
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'original_changes': [c.to_dict() for c in original_changes],
            'resolved_changes': resolution.get('resolved_changes', []),
            'strategy': resolution.get('strategy', 'unknown'),
            'conflict_type': self._detect_conflict_type(original_changes)
        }
        
        self.edit_history[session_id].append(history_entry)
        
        # Limit history size
        if len(self.edit_history[session_id]) > self.max_history_size:
            self.edit_history[session_id] = self.edit_history[session_id][-self.max_history_size:]


    async def get_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get edit history for a session"""
        if session_id not in self.edit_history:
            return {
                'success': True,
                'history': [],
                'count': 0
            }
        
        history = self.edit_history[session_id][-limit:]
        
        return {
            'success': True,
            'history': history,
            'count': len(history)
        }

    async def rollback(self, session_id: str, steps: int = 1) -> Dict[str, Any]:
        """
        Rollback edit history by specified steps
        
        Args:
            session_id: Session to rollback
            steps: Number of resolutions to rollback
        
        Returns:
            Dict with rollback information
        """
        if session_id not in self.edit_history:
            return {
                'success': False,
                'error': 'No history found for session'
            }
        
        history = self.edit_history[session_id]
        
        if steps > len(history):
            return {
                'success': False,
                'error': f'Cannot rollback {steps} steps, only {len(history)} available'
            }
        
        # Get entries to rollback
        rollback_entries = history[-steps:]
        
        # Remove from history
        self.edit_history[session_id] = history[:-steps]
        
        return {
            'success': True,
            'rolled_back': rollback_entries,
            'steps': steps,
            'remaining_history': len(self.edit_history[session_id])
        }

    async def _last_write_wins(self, changes: List[Change]) -> Dict[str, Any]:
        """Simple last-write-wins strategy"""
        if not changes:
            return {
                'success': True,
                'resolved_changes': []
            }
        
        # Sort by timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)
        
        return {
            'success': True,
            'resolved_changes': [sorted_changes[-1].to_dict()],
            'strategy': 'last_write_wins'
        }
    
    async def _operational_transform(self, changes: List[Change]) -> Dict[str, Any]:
        """Operational transformation for conflict resolution"""
        if not changes:
            return {
                'success': True,
                'resolved_changes': []
            }
        
        # Sort by timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)
        
        # Transform changes to be compatible
        transformed = []
        for i, change in enumerate(sorted_changes):
            transformed_change = change
            
            # Transform against all previous changes
            for prev_change in sorted_changes[:i]:
                transformed_change = self._transform_change(transformed_change, prev_change)
            
            transformed.append(transformed_change)
        
        return {
            'success': True,
            'resolved_changes': [c.to_dict() for c in transformed],
            'strategy': 'operational_transform'
        }
    
    def _transform_change(self, change: Change, against: Change) -> Change:
        """Transform a change against another change"""
        # Simple position adjustment based on previous change
        if against.position <= change.position:
            if against.change_type == ChangeType.INSERT:
                # Adjust position forward
                change.position += len(against.content)
            elif against.change_type == ChangeType.DELETE:
                # Adjust position backward
                change.position -= len(against.content)
        
        return change
    
    async def _crdt_merge(self, changes: List[Change]) -> Dict[str, Any]:
        """CRDT-based conflict-free merge"""
        # Simplified CRDT implementation
        # In production, would use a proper CRDT library
        
        if not changes:
            return {
                'success': True,
                'resolved_changes': []
            }
        
        # Sort by timestamp and user_id for deterministic ordering
        sorted_changes = sorted(changes, key=lambda c: (c.timestamp, c.user_id))
        
        return {
            'success': True,
            'resolved_changes': [c.to_dict() for c in sorted_changes],
            'strategy': 'crdt'
        }

    async def _smart_merge(self, changes: List[Change]) -> Dict[str, Any]:
        """
        Smart merge algorithm that intelligently handles concurrent edits
        
        This strategy:
        1. Analyzes the semantic context of changes
        2. Detects conflict types (overlapping, adjacent, independent)
        3. Applies appropriate resolution based on conflict type
        4. Preserves user intent where possible
        """
        if not changes:
            return {
                'success': True,
                'resolved_changes': []
            }
        
        # Sort by timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)
        
        # Detect conflict type
        conflict_type = self._detect_conflict_type(sorted_changes)
        
        # Apply strategy based on conflict type
        if conflict_type == 'no_conflict' or conflict_type == 'sequential_edits':
            # No real conflict, apply OT
            result = await self._operational_transform(sorted_changes)
            result['conflict_type'] = conflict_type
            return result
        
        elif conflict_type == 'same_position':
            # Multiple edits at same position
            resolved = await self._resolve_same_position(sorted_changes)
            return {
                'success': True,
                'resolved_changes': resolved,
                'strategy': 'smart_merge',
                'conflict_type': conflict_type,
                'resolution_method': 'merge_at_position'
            }
        
        elif conflict_type == 'overlapping_edits':
            # Overlapping edits require careful handling
            resolved = await self._resolve_overlapping(sorted_changes)
            return {
                'success': True,
                'resolved_changes': resolved,
                'strategy': 'smart_merge',
                'conflict_type': conflict_type,
                'resolution_method': 'intelligent_overlap'
            }
        
        # Fallback to OT
        result = await self._operational_transform(sorted_changes)
        result['conflict_type'] = conflict_type
        return result
    
    async def _resolve_same_position(self, changes: List[Change]) -> List[Dict[str, Any]]:
        """Resolve multiple changes at the same position"""
        if not changes:
            return []
        
        position = changes[0].position
        resolved = []
        current_offset = 0
        
        for change in changes:
            resolved_change = change.to_dict()
            resolved_change['position'] = position + current_offset
            
            if change.change_type == ChangeType.INSERT:
                current_offset += len(change.content)
            
            resolved.append(resolved_change)
        
        return resolved
    
    async def _resolve_overlapping(self, changes: List[Change]) -> List[Dict[str, Any]]:
        """Resolve overlapping edits intelligently"""
        if not changes:
            return []
        
        # Group changes by type
        inserts = [c for c in changes if c.change_type == ChangeType.INSERT]
        deletes = [c for c in changes if c.change_type == ChangeType.DELETE]
        
        resolved = []
        
        # Process deletes first (they remove content)
        for delete in deletes:
            resolved.append(delete.to_dict())
        
        # Then process inserts with adjusted positions
        offset = sum(len(d.content) for d in deletes)
        for insert in inserts:
            resolved_insert = insert.to_dict()
            resolved_insert['position'] = max(0, insert.position - offset)
            resolved.append(resolved_insert)
        
        return resolved



    def _detect_conflict_type(self, changes: List[Change]) -> str:
        """Detect the type of conflict between changes"""
        if len(changes) < 2:
            return 'no_conflict'
        
        # Check if same position first (before overlap check)
        if len(set(c.position for c in changes)) == 1:
            return 'same_position'
        
        # Check if changes overlap in position
        positions = [(c.position, c.position + len(c.content)) for c in changes]
        
        # Check for overlapping ranges
        for i, (start1, end1) in enumerate(positions):
            for start2, end2 in positions[i+1:]:
                if not (end1 <= start2 or end2 <= start1):
                    return 'overlapping_edits'
        
        return 'sequential_edits'


class PresenceManager:
    """Manages user presence tracking and activity monitoring"""
    
    def __init__(self, idle_timeout_seconds: int = 300):
        """
        Initialize presence manager
        
        Args:
            idle_timeout_seconds: Seconds of inactivity before marking user as idle (default: 5 minutes)
        """
        self.idle_timeout_seconds = idle_timeout_seconds
        self.activity_log: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> activity events
    
    async def update_presence(self, user: User, status: str) -> Dict[str, Any]:
        """
        Update user presence status
        
        Args:
            user: User object to update
            status: New status (online, offline, idle, away)
        
        Returns:
            Dict with success status and updated user data
        """
        if status not in ['online', 'offline', 'idle', 'away']:
            return {
                'success': False,
                'error': f'Invalid status: {status}'
            }
        
        old_status = user.status
        user.status = status
        user.last_activity = datetime.now().isoformat()
        
        if status == 'idle':
            user.idle_since = datetime.now().isoformat()
        elif status == 'online':
            user.idle_since = None
        
        # Log activity
        self._log_activity(user.user_id, 'presence_change', {
            'old_status': old_status,
            'new_status': status
        })
        
        return {
            'success': True,
            'user': user.to_dict(),
            'status_changed': old_status != status
        }
    
    async def update_typing_status(self, user: User, is_typing: bool) -> Dict[str, Any]:
        """
        Update user typing indicator
        
        Args:
            user: User object to update
            is_typing: Whether user is currently typing
        
        Returns:
            Dict with success status and updated user data
        """
        user.is_typing = is_typing
        user.last_activity = datetime.now().isoformat()
        
        # Reset idle status if typing
        if is_typing and user.status == 'idle':
            user.status = 'online'
            user.idle_since = None
        
        # Log activity
        self._log_activity(user.user_id, 'typing', {
            'is_typing': is_typing
        })
        
        return {
            'success': True,
            'user': user.to_dict()
        }
    
    async def update_file_location(self, user: User, file_path: Optional[str], 
                                   location: Optional[str] = None) -> Dict[str, Any]:
        """
        Update user's current file and location
        
        Args:
            user: User object to update
            file_path: Path to the file user is viewing/editing
            location: Optional location description (e.g., "line 42", "function main")
        
        Returns:
            Dict with success status and updated user data
        """
        user.current_file = file_path
        user.current_location = location
        user.last_activity = datetime.now().isoformat()
        
        # Reset idle status
        if user.status == 'idle':
            user.status = 'online'
            user.idle_since = None
        
        # Log activity
        self._log_activity(user.user_id, 'file_change', {
            'file': file_path,
            'location': location
        })
        
        return {
            'success': True,
            'user': user.to_dict()
        }
    
    async def check_idle_status(self, user: User) -> Dict[str, Any]:
        """
        Check if user should be marked as idle based on last activity
        
        Args:
            user: User object to check
        
        Returns:
            Dict with idle status and time since last activity
        """
        if user.status == 'offline':
            return {
                'is_idle': False,
                'reason': 'User is offline'
            }
        
        last_activity = datetime.fromisoformat(user.last_activity)
        now = datetime.now()
        seconds_since_activity = (now - last_activity).total_seconds()
        
        should_be_idle = seconds_since_activity >= self.idle_timeout_seconds
        
        # Auto-update to idle if threshold exceeded
        if should_be_idle and user.status == 'online':
            await self.update_presence(user, 'idle')
        
        return {
            'is_idle': should_be_idle,
            'seconds_since_activity': seconds_since_activity,
            'idle_threshold': self.idle_timeout_seconds,
            'current_status': user.status
        }
    
    async def record_activity(self, user: User, activity_type: str, 
                             details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record user activity
        
        Args:
            user: User object
            activity_type: Type of activity (typing, viewing, editing, cursor_move)
            details: Optional activity details
        
        Returns:
            Dict with success status
        """
        user.last_activity = datetime.now().isoformat()
        
        # Reset idle status on activity
        if user.status == 'idle':
            user.status = 'online'
            user.idle_since = None
        
        # Log activity
        self._log_activity(user.user_id, activity_type, details or {})
        
        return {
            'success': True,
            'user': user.to_dict()
        }
    
    def _log_activity(self, user_id: str, activity_type: str, details: Dict[str, Any]) -> None:
        """
        Internal method to log activity
        
        Args:
            user_id: User ID
            activity_type: Type of activity
            details: Activity details
        """
        if user_id not in self.activity_log:
            self.activity_log[user_id] = []
        
        self.activity_log[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'type': activity_type,
            'details': details
        })
        
        # Keep only last 100 activities per user
        if len(self.activity_log[user_id]) > 100:
            self.activity_log[user_id] = self.activity_log[user_id][-100:]
    
    async def get_activity_log(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get activity log for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of activities to return
        
        Returns:
            Dict with activity log
        """
        if user_id not in self.activity_log:
            return {
                'success': True,
                'activities': [],
                'count': 0
            }
        
        activities = self.activity_log[user_id][-limit:]
        
        return {
            'success': True,
            'activities': activities,
            'count': len(activities)
        }
    
    async def get_all_presence(self, users: Dict[str, User]) -> Dict[str, Any]:
        """
        Get presence information for all users
        
        Args:
            users: Dictionary of user_id -> User objects
        
        Returns:
            Dict with presence information for all users
        """
        presence_data = {}
        
        for user_id, user in users.items():
            # Check idle status
            idle_check = await self.check_idle_status(user)
            
            presence_data[user_id] = {
                'user': user.to_dict(),
                'idle_info': idle_check
            }
        
        return {
            'success': True,
            'presence': presence_data,
            'user_count': len(presence_data)
        }


class CollaborativeCodingFeature(FeatureBase):
    """Collaborative Coding feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.collab_config = CollaborativeConfig.from_dict(config.config)
        self.manager: Optional[CollaborationManager] = None
        self.editor: Optional[WebSocketEditor] = None
        self.resolver: Optional[ConflictResolver] = None
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "collaborative_coding"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Real-time collaborative coding with WebSocket support and intelligent conflict resolution"
    
    async def _initialize(self) -> None:
        """Initialize collaborative coding components"""
        # Initialize collaboration manager
        self.manager = CollaborationManager(self.collab_config)
        
        # Initialize WebSocket editor
        self.editor = WebSocketEditor(self.collab_config, self.manager)
        
        # Initialize conflict resolver
        self.resolver = ConflictResolver()
        
        # Start WebSocket server
        result = await self.editor.start_server()
        if not result['success']:
            raise FeatureError(f"Failed to start WebSocket server: {result.get('error')}")
    
    async def _shutdown(self) -> None:
        """Shutdown collaborative coding"""
        if self.editor:
            await self.editor.stop_server()
        
        self.manager = None
        self.editor = None
        self.resolver = None
    
    async def start(self, name: str, owner_id: str, username: str) -> Dict[str, Any]:
        """Start a collaboration session"""
        self.track_analytics('start_session', {'name': name})
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.start_session(name, owner_id, username)
    
    async def join(self, session_id: str, user_id: str, username: str) -> Dict[str, Any]:
        """Join a collaboration session"""
        self.track_analytics('join_session', {'session_id': session_id})
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.join_session(session_id, user_id, username)
    
    async def leave(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Leave a collaboration session"""
        self.track_analytics('leave_session', {'session_id': session_id})
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.leave_session(session_id, user_id)
    
    async def sync(self, session_id: str, user_id: str, change_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize changes across clients"""
        self.track_analytics('sync_change', {'session_id': session_id})
        
        if not self.manager or not self.editor:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        # Sync change through manager
        result = await self.manager.sync_change(session_id, user_id, change_data)
        
        if result['success']:
            # Broadcast to other participants
            await self.editor.broadcast_change(
                session_id,
                result['change'],
                exclude_user=user_id
            )
        
        return result
    
    async def update_cursor(self, session_id: str, user_id: str,
                          cursor_position: int,
                          selection_start: Optional[int] = None,
                          selection_end: Optional[int] = None) -> Dict[str, Any]:
        """Update user cursor position"""
        if not self.manager or not self.editor:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        result = await self.manager.update_cursor(
            session_id, user_id, cursor_position,
            selection_start, selection_end
        )
        
        if result['success']:
            # Broadcast cursor update
            await self.editor.broadcast_cursor(
                session_id,
                result['user'],
                exclude_user=user_id
            )
        
        return result
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.get_session_info(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """List active sessions"""
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.list_sessions(user_id)
    
    async def close_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Close a collaboration session"""
        self.track_analytics('close_session', {'session_id': session_id})
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.close_session(session_id, user_id)
    
    async def update_presence(self, session_id: str, user_id: str, status: str) -> Dict[str, Any]:
        """Update user presence status"""
        if not self.manager or not self.editor:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        result = await self.manager.update_user_presence(session_id, user_id, status)
        
        if result['success']:
            # Broadcast presence update
            await self.editor.broadcast_presence(
                session_id,
                result['user'],
                exclude_user=user_id
            )
        
        return result
    
    async def update_typing(self, session_id: str, user_id: str, is_typing: bool) -> Dict[str, Any]:
        """Update user typing status"""
        if not self.manager or not self.editor:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        result = await self.manager.update_typing_status(session_id, user_id, is_typing)
        
        if result['success']:
            # Broadcast typing update
            await self.editor.broadcast_typing(
                session_id,
                result['user'],
                exclude_user=user_id
            )
        
        return result
    
    async def update_file_location(self, session_id: str, user_id: str,
                                   file_path: Optional[str], location: Optional[str] = None) -> Dict[str, Any]:
        """Update user's current file and location"""
        if not self.manager or not self.editor:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        result = await self.manager.update_file_location(session_id, user_id, file_path, location)
        
        if result['success']:
            # Broadcast file location update
            await self.editor.broadcast_file_location(
                session_id,
                result['user'],
                exclude_user=user_id
            )
        
        return result
    
    async def get_presence(self, session_id: str) -> Dict[str, Any]:
        """Get presence information for all users in a session"""
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.get_session_presence(session_id)
    
    async def check_idle_users(self, session_id: str) -> Dict[str, Any]:
        """Check and update idle status for all users in a session"""
        if not self.manager:
            return {
                'success': False,
                'error': 'Collaboration manager not initialized'
            }
        
        return await self.manager.check_idle_users(session_id)
    
    async def resolve_conflicts(self, session_id: str, 
                               strategy: str = 'operational_transform') -> Dict[str, Any]:
        """Resolve conflicts in a session"""
        if not self.manager or not self.resolver:
            return {
                'success': False,
                'error': 'Collaboration components not initialized'
            }
        
        session_info = await self.manager.get_session_info(session_id)
        if not session_info['success']:
            return session_info
        
        # Get recent changes that might conflict
        session = self.manager.sessions[session_id]
        recent_changes = session.change_history[-10:]  # Last 10 changes
        
        return await self.resolver.resolve_conflict(recent_changes, strategy)
    

    async def get_edit_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get edit history for a session"""
        if not self.resolver:
            return {
                'success': False,
                'error': 'Conflict resolver not initialized'
            }
        
        return await self.resolver.get_history(session_id, limit)

    async def rollback_edits(self, session_id: str, steps: int = 1) -> Dict[str, Any]:
        """Rollback edit history by specified steps"""
        if not self.resolver:
            return {
                'success': False,
                'error': 'Conflict resolver not initialized'
            }
        
        return await self.resolver.rollback(session_id, steps)

    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for collaborative coding"""
        import click
        from concurrent.futures import ThreadPoolExecutor

        def run_sync(coro):
            """Run coroutine from both sync and async contexts safely."""
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(lambda: asyncio.run(coro)).result()

        @click.group(name='collab')
        def collab_group():
            """Collaborative coding commands"""
            pass

        @collab_group.command(name='start')
        @click.argument('name')
        @click.option('--owner-id', default='owner', help='Owner user ID')
        @click.option('--username', default='Owner', help='Display username')
        def start_cmd(name: str, owner_id: str, username: str):
            """Start a new collaboration session."""
            click.echo("Starting Collaboration Session...")
            result = run_sync(self.start(name=name, owner_id=owner_id, username=username))
            if result.get('success'):
                click.echo(f"Session started: {result.get('session_id', 'unknown')}")
            else:
                click.echo(f"Failed: {result.get('error', 'unknown error')}")

        @collab_group.command(name='join')
        @click.argument('session_id')
        @click.option('--user-id', default='user', help='User ID')
        @click.option('--username', default='User', help='Display username')
        def join_cmd(session_id: str, user_id: str, username: str):
            """Join an existing collaboration session."""
            click.echo("Joining Collaboration Session...")
            result = run_sync(self.join(session_id=session_id, user_id=user_id, username=username))
            if result.get('success'):
                click.echo(f"Joined session: {session_id}")
            else:
                click.echo(f"Failed: {result.get('error', 'unknown error')}")

        @collab_group.command(name='leave')
        @click.argument('session_id')
        @click.option('--user-id', default='user', help='User ID')
        def leave_cmd(session_id: str, user_id: str):
            """Leave a collaboration session."""
            result = run_sync(self.leave(session_id=session_id, user_id=user_id))
            if result.get('success'):
                click.echo(f"Left session: {session_id}")
            else:
                click.echo(f"Failed: {result.get('error', 'unknown error')}")

        @collab_group.command(name='list')
        @click.option('--user-id', default=None, help='Optional filter by user ID')
        def list_cmd(user_id: Optional[str]):
            """List active sessions."""
            result = run_sync(self.list_sessions(user_id=user_id))
            if not result.get('success'):
                click.echo(f"Failed: {result.get('error', 'unknown error')}")
                return
            click.echo(f"Active sessions: {result.get('count', 0)}")
            for session in result.get('sessions', []):
                click.echo(f"- {session.get('session_id')} ({session.get('name')})")

        @collab_group.command(name='users')
        @click.argument('session_id')
        def users_cmd(session_id: str):
            """List users in a collaboration session."""
            result = run_sync(self.get_session_info(session_id=session_id))
            if not result.get('success'):
                click.echo(f"Failed: {result.get('error', 'unknown error')}")
                return
            participants = result.get('session', {}).get('participants', {})
            click.echo(f"Users in {session_id}: {len(participants)}")
            for user_id, user in participants.items():
                click.echo(f"- {user_id}: {user.get('username', 'unknown')}")

        return [collab_group]
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for collaborative coding"""
        from xencode.tui.widgets.collaborative_coding_panel import CollaborativeCodingPanel
        return [CollaborativeCodingPanel]
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for collaborative coding"""
        return [
            {
                'path': '/api/collab/start',
                'method': 'POST',
                'handler': self.start
            },
            {
                'path': '/api/collab/join',
                'method': 'POST',
                'handler': self.join
            },
            {
                'path': '/api/collab/leave',
                'method': 'POST',
                'handler': self.leave
            },
            {
                'path': '/api/collab/sync',
                'method': 'POST',
                'handler': self.sync
            },
            {
                'path': '/api/collab/cursor',
                'method': 'POST',
                'handler': self.update_cursor
            },
            {
                'path': '/api/collab/session/{session_id}',
                'method': 'GET',
                'handler': self.get_session_info
            },
            {
                'path': '/api/collab/sessions',
                'method': 'GET',
                'handler': self.list_sessions
            },
            {
                'path': '/api/collab/close',
                'method': 'POST',
                'handler': self.close_session
            },
            {
                'path': '/api/collab/resolve',
                'method': 'POST',
                'handler': self.resolve_conflicts
            },
            {
                'path': '/api/collab/history/{session_id}',
                'method': 'GET',
                'handler': self.get_edit_history
            },
            {
                'path': '/api/collab/rollback',
                'method': 'POST',
                'handler': self.rollback_edits
            },
            {
                'path': '/api/collab/presence',
                'method': 'POST',
                'handler': self.update_presence
            },
            {
                'path': '/api/collab/typing',
                'method': 'POST',
                'handler': self.update_typing
            },
            {
                'path': '/api/collab/file-location',
                'method': 'POST',
                'handler': self.update_file_location
            },
            {
                'path': '/api/collab/presence/{session_id}',
                'method': 'GET',
                'handler': self.get_presence
            },
            {
                'path': '/api/collab/idle/{session_id}',
                'method': 'GET',
                'handler': self.check_idle_users
            }
        ]
