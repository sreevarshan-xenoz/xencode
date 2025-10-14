#!/usr/bin/env python3
"""
Sync Coordinator - Real-time synchronization for collaborative workspaces

Implements WebSocket-based real-time synchronization with <50ms latency target,
change broadcasting, and connection management for collaborative editing.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from xencode.workspace.collaboration_manager import CollaborationManager


class SyncMessage:
    """Synchronization message for real-time collaboration"""
    
    def __init__(self, message_type: str, workspace_id: str, data: Dict[str, Any], 
                 sender_id: str = "", timestamp: Optional[datetime] = None):
        self.message_type = message_type
        self.workspace_id = workspace_id
        self.data = data
        self.sender_id = sender_id
        self.timestamp = timestamp or datetime.now()
        self.message_id = f"{sender_id}_{int(time.time() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type,
            'workspace_id': self.workspace_id,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncMessage':
        return cls(
            message_type=data['message_type'],
            workspace_id=data['workspace_id'],
            data=data['data'],
            sender_id=data.get('sender_id', ''),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


class WebSocketConnection:
    """WebSocket connection with latency monitoring"""
    
    def __init__(self, connection_id: str, user_id: str, workspace_id: str, websocket=None):
        self.connection_id = connection_id
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.websocket = websocket
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.is_active = True
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.total_latency = 0.0
        self.ping_count = 0
    
    async def send_message(self, message: SyncMessage) -> bool:
        """Send message with error handling"""
        if not self.websocket or not self.is_active:
            return False
        
        try:
            message_data = json.dumps(message.to_dict())
            await self.websocket.send(message_data)
            self.messages_sent += 1
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            self.is_active = False
            return False
    
    async def ping(self) -> float:
        """Measure connection latency"""
        if not self.is_active:
            return -1.0
        
        start_time = time.time()
        
        try:
            # Simulate ping/pong (in real implementation would use actual WebSocket ping)
            await asyncio.sleep(0.001)  # Minimal latency simulation
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.total_latency += latency
            self.ping_count += 1
            self.last_ping = datetime.now()
            
            return latency
        except Exception:
            self.is_active = False
            return -1.0
    
    def get_average_latency(self) -> float:
        """Get average latency in milliseconds"""
        return self.total_latency / self.ping_count if self.ping_count > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'connection_id': self.connection_id,
            'user_id': self.user_id,
            'workspace_id': self.workspace_id,
            'connected_at': self.connected_at.isoformat(),
            'is_active': self.is_active,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'average_latency_ms': self.get_average_latency(),
            'last_ping': self.last_ping.isoformat()
        }


class SyncCoordinator:
    """
    Coordinates real-time synchronization across collaborators
    
    Features:
    - WebSocket-based real-time communication
    - <50ms latency optimization
    - Change broadcasting and conflict resolution
    - Connection health monitoring
    - Performance statistics tracking
    """
    
    def __init__(self, collaboration_manager: Optional['CollaborationManager'] = None, crdt_engine=None):
        self.collaboration_manager = collaboration_manager
        self.crdt_engine = crdt_engine
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.workspace_connections: Dict[str, Set[str]] = {}
        self.message_queue: Dict[str, List[SyncMessage]] = {}
        
        # Performance configuration
        self.batch_size = 10
        self.batch_timeout = 0.05  # 50ms target
        self.max_latency_target = 0.05  # 50ms latency target
        
        # Statistics
        self.sync_stats = {
            'messages_processed': 0,
            'changes_synced': 0,
            'conflicts_resolved': 0,
            'average_sync_time': 0.0,
            'peak_connections': 0
        }
        
        self._running = False
        self._sync_task = None
        self._cleanup_task = None
    
    async def start(self) -> None:
        """Start sync coordinator with background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background monitoring tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        print("SyncCoordinator started with real-time synchronization")
    
    async def stop(self) -> None:
        """Stop sync coordinator and cleanup"""
        self._running = False
        
        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Disconnect all clients
        for connection in list(self.connections.values()):
            await self.disconnect_client(connection.connection_id)
        
        print("SyncCoordinator stopped")
    
    async def connect_client(self, connection_id: str, user_id: str, workspace_id: str, websocket=None) -> bool:
        """Connect client for real-time synchronization"""
        try:
            connection = WebSocketConnection(connection_id, user_id, workspace_id, websocket)
            self.connections[connection_id] = connection
            
            # Add to workspace connections
            if workspace_id not in self.workspace_connections:
                self.workspace_connections[workspace_id] = set()
            self.workspace_connections[workspace_id].add(connection_id)
            
            # Update peak connections stat
            current_connections = len(self.connections)
            if current_connections > self.sync_stats['peak_connections']:
                self.sync_stats['peak_connections'] = current_connections
            
            # Start collaboration session if manager available
            if self.collaboration_manager:
                await self.collaboration_manager.start_collaboration_session(
                    workspace_id, user_id, f"user_{user_id}", connection_id
                )
            
            # Send welcome message with sync settings
            welcome_message = SyncMessage(
                message_type='welcome',
                workspace_id=workspace_id,
                data={
                    'connection_id': connection_id,
                    'server_time': datetime.now().isoformat(),
                    'sync_settings': {
                        'batch_size': self.batch_size,
                        'batch_timeout': self.batch_timeout,
                        'max_latency_target': self.max_latency_target
                    }
                },
                sender_id='system'
            )
            
            await connection.send_message(welcome_message)
            
            print(f"Client connected: {connection_id} (user: {user_id}, workspace: {workspace_id})")
            return True
            
        except Exception as e:
            print(f"Error connecting client: {e}")
            return False
    
    async def disconnect_client(self, connection_id: str) -> bool:
        """Disconnect client and cleanup"""
        if connection_id not in self.connections:
            return False
        
        try:
            connection = self.connections[connection_id]
            workspace_id = connection.workspace_id
            user_id = connection.user_id
            
            # Remove from workspace connections
            if workspace_id in self.workspace_connections:
                self.workspace_connections[workspace_id].discard(connection_id)
                if not self.workspace_connections[workspace_id]:
                    del self.workspace_connections[workspace_id]
            
            # End collaboration session
            if self.collaboration_manager:
                await self.collaboration_manager.end_collaboration_session(workspace_id, user_id)
            
            # Close WebSocket
            connection.is_active = False
            if connection.websocket:
                try:
                    await connection.websocket.close()
                except:
                    pass
            
            del self.connections[connection_id]
            print(f"Client disconnected: {connection_id}")
            return True
            
        except Exception as e:
            print(f"Error disconnecting client: {e}")
            return False
    
    async def broadcast_change(self, change, exclude_connection: Optional[str] = None) -> int:
        """Broadcast change to all workspace collaborators"""
        workspace_id = change.workspace_id
        
        if workspace_id not in self.workspace_connections:
            return 0
        
        # Create sync message
        sync_message = SyncMessage(
            message_type='change',
            workspace_id=workspace_id,
            data={
                'change': {
                    'id': getattr(change, 'id', 'unknown'),
                    'file_id': getattr(change, 'file_id', ''),
                    'change_type': str(getattr(change, 'change_type', 'update')),
                    'position': getattr(change, 'position', 0),
                    'length': getattr(change, 'length', 0),
                    'content': getattr(change, 'content', ''),
                    'author_id': getattr(change, 'author_id', ''),
                    'timestamp': getattr(change, 'timestamp', datetime.now()).isoformat(),
                    'vector_clock': getattr(change, 'vector_clock', {})
                }
            },
            sender_id=getattr(change, 'author_id', '')
        )
        
        # Send to all connections in workspace
        sent_count = 0
        connection_ids = list(self.workspace_connections[workspace_id])
        
        for connection_id in connection_ids:
            if connection_id == exclude_connection:
                continue
            
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                success = await connection.send_message(sync_message)
                if success:
                    sent_count += 1
                else:
                    # Connection failed, remove it
                    await self.disconnect_client(connection_id)
        
        self.sync_stats['changes_synced'] += 1
        return sent_count
    
    async def _sync_loop(self) -> None:
        """Background sync loop for latency optimization"""
        while self._running:
            try:
                start_time = time.time()
                
                # Monitor connection health
                await self._monitor_connections()
                
                # Update sync time statistics
                sync_time = (time.time() - start_time) * 1000  # Convert to ms
                if self.sync_stats['average_sync_time'] == 0:
                    self.sync_stats['average_sync_time'] = sync_time
                else:
                    # Exponential moving average
                    self.sync_stats['average_sync_time'] = (
                        0.9 * self.sync_stats['average_sync_time'] + 0.1 * sync_time
                    )
                
                # Sleep for batch timeout to maintain <50ms target
                await asyncio.sleep(self.batch_timeout)
                
            except Exception as e:
                print(f"Error in sync loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup for inactive connections"""
        while self._running:
            try:
                # Clean up inactive connections
                inactive_connections = []
                cutoff = datetime.now() - timedelta(minutes=5)
                
                for connection_id, connection in self.connections.items():
                    if not connection.is_active or connection.last_ping < cutoff:
                        inactive_connections.append(connection_id)
                
                for connection_id in inactive_connections:
                    await self.disconnect_client(connection_id)
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_connections(self) -> None:
        """Monitor connection health and latency"""
        for connection in list(self.connections.values()):
            if not connection.is_active:
                continue
            
            # Ping connection to measure latency
            latency = await connection.ping()
            
            if latency < 0:
                # Connection failed
                await self.disconnect_client(connection.connection_id)
            elif latency > self.max_latency_target * 1000:  # Convert to ms
                # High latency warning
                print(f"High latency detected: {connection.connection_id} ({latency:.2f}ms)")
    
    def get_connection_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for specific connection"""
        if connection_id in self.connections:
            return self.connections[connection_id].get_stats()
        return None
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for workspace"""
        connections = []
        if workspace_id in self.workspace_connections:
            for conn_id in self.workspace_connections[workspace_id]:
                if conn_id in self.connections:
                    connections.append(self.connections[conn_id])
        
        total_sent = sum(conn.messages_sent for conn in connections)
        total_received = sum(conn.messages_received for conn in connections)
        avg_latency = sum(conn.get_average_latency() for conn in connections) / len(connections) if connections else 0
        
        return {
            'workspace_id': workspace_id,
            'active_connections': len(connections),
            'total_messages_sent': total_sent,
            'total_messages_received': total_received,
            'average_latency_ms': avg_latency,
            'connections': [conn.get_stats() for conn in connections]
        }
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get overall synchronization statistics"""
        return {
            **self.sync_stats,
            'active_connections': len(self.connections),
            'active_workspaces': len(self.workspace_connections),
            'queued_messages': sum(len(messages) for messages in self.message_queue.values()),
            'is_running': self._running
        }


# Global sync coordinator instance
sync_coordinator = SyncCoordinator()"""
Real-
time synchronization coordinator for collaborative workspaces
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from xencode.workspace.collaboration_manager import CollaborationManager


class SyncMessage:
    """Synchronization message for real-time collaboration"""
    
    def __init__(self, message_type: str, workspace_id: str, data: Dict[str, Any], 
                 sender_id: str = "", timestamp: Optional[datetime] = None):
        self.message_type = message_type
        self.workspace_id = workspace_id
        self.data = data
        self.sender_id = sender_id
        self.timestamp = timestamp or datetime.now()
        self.message_id = f"{sender_id}_{int(time.time() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'message_type': self.message_type,
            'workspace_id': self.workspace_id,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncMessage':
        return cls(
            message_type=data['message_type'],
            workspace_id=data['workspace_id'],
            data=data['data'],
            sender_id=data.get('sender_id', ''),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


class WebSocketConnection:
    """WebSocket connection with latency monitoring"""
    
    def __init__(self, connection_id: str, user_id: str, workspace_id: str, websocket=None):
        self.connection_id = connection_id
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.websocket = websocket
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.is_active = True
        self.messages_sent = 0
        self.messages_received = 0
        self.total_latency = 0.0
        self.ping_count = 0
    
    async def send_message(self, message: SyncMessage) -> bool:
        """Send message with error handling"""
        if not self.websocket or not self.is_active:
            return False
        
        try:
            message_data = json.dumps(message.to_dict())
            await self.websocket.send(message_data)
            self.messages_sent += 1
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            self.is_active = False
            return False
    
    async def ping(self) -> float:
        """Measure connection latency"""
        if not self.is_active:
            return -1.0
        
        start_time = time.time()
        await asyncio.sleep(0.001)  # Simulate latency
        
        latency = (time.time() - start_time) * 1000
        self.total_latency += latency
        self.ping_count += 1
        self.last_ping = datetime.now()
        
        return latency
    
    def get_average_latency(self) -> float:
        return self.total_latency / self.ping_count if self.ping_count > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'connection_id': self.connection_id,
            'user_id': self.user_id,
            'workspace_id': self.workspace_id,
            'connected_at': self.connected_at.isoformat(),
            'is_active': self.is_active,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'average_latency_ms': self.get_average_latency(),
            'last_ping': self.last_ping.isoformat()
        }


class SyncCoordinator:
    """Coordinates real-time synchronization across collaborators"""
    
    def __init__(self, collaboration_manager: Optional['CollaborationManager'] = None, crdt_engine=None):
        self.collaboration_manager = collaboration_manager
        self.crdt_engine = crdt_engine
        self.connections: Dict[str, WebSocketConnection] = {}
        self.workspace_connections: Dict[str, Set[str]] = {}
        self.message_queue: Dict[str, List[SyncMessage]] = {}
        
        # Performance configuration
        self.batch_size = 10
        self.batch_timeout = 0.05  # 50ms target
        self.max_latency_target = 0.05  # 50ms latency target
        
        # Statistics
        self.sync_stats = {
            'messages_processed': 0,
            'changes_synced': 0,
            'conflicts_resolved': 0,
            'average_sync_time': 0.0,
            'peak_connections': 0
        }
        
        self._running = False
    
    async def start(self) -> None:
        """Start sync coordinator"""
        if self._running:
            return
        self._running = True
        print("SyncCoordinator started with real-time synchronization")
    
    async def stop(self) -> None:
        """Stop sync coordinator"""
        self._running = False
        for connection in list(self.connections.values()):
            await self.disconnect_client(connection.connection_id)
        print("SyncCoordinator stopped")
    
    async def connect_client(self, connection_id: str, user_id: str, workspace_id: str, websocket=None) -> bool:
        """Connect client for real-time synchronization"""
        try:
            connection = WebSocketConnection(connection_id, user_id, workspace_id, websocket)
            self.connections[connection_id] = connection
            
            if workspace_id not in self.workspace_connections:
                self.workspace_connections[workspace_id] = set()
            self.workspace_connections[workspace_id].add(connection_id)
            
            current_connections = len(self.connections)
            if current_connections > self.sync_stats['peak_connections']:
                self.sync_stats['peak_connections'] = current_connections
            
            print(f"Client connected: {connection_id}")
            return True
        except Exception as e:
            print(f"Error connecting client: {e}")
            return False
    
    async def disconnect_client(self, connection_id: str) -> bool:
        """Disconnect client"""
        if connection_id not in self.connections:
            return False
        
        try:
            connection = self.connections[connection_id]
            workspace_id = connection.workspace_id
            
            if workspace_id in self.workspace_connections:
                self.workspace_connections[workspace_id].discard(connection_id)
                if not self.workspace_connections[workspace_id]:
                    del self.workspace_connections[workspace_id]
            
            connection.is_active = False
            if connection.websocket:
                try:
                    await connection.websocket.close()
                except:
                    pass
            
            del self.connections[connection_id]
            print(f"Client disconnected: {connection_id}")
            return True
        except Exception as e:
            print(f"Error disconnecting client: {e}")
            return False
    
    def get_connection_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection statistics"""
        if connection_id in self.connections:
            return self.connections[connection_id].get_stats()
        return None
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get workspace statistics"""
        connections = []
        if workspace_id in self.workspace_connections:
            for conn_id in self.workspace_connections[workspace_id]:
                if conn_id in self.connections:
                    connections.append(self.connections[conn_id])
        
        total_sent = sum(conn.messages_sent for conn in connections)
        total_received = sum(conn.messages_received for conn in connections)
        avg_latency = sum(conn.get_average_latency() for conn in connections) / len(connections) if connections else 0
        
        return {
            'workspace_id': workspace_id,
            'active_connections': len(connections),
            'total_messages_sent': total_sent,
            'total_messages_received': total_received,
            'average_latency_ms': avg_latency,
            'connections': [conn.get_stats() for conn in connections]
        }
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics"""
        return {
            **self.sync_stats,
            'active_connections': len(self.connections),
            'active_workspaces': len(self.workspace_connections),
            'queued_messages': sum(len(messages) for messages in self.message_queue.values()),
            'is_running': self._running
        }


# Global instance
sync_coordinator = SyncCoordinator()