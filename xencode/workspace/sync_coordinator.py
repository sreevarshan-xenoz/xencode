#!/usr/bin/env python3
"""Real-time synchronization coordinator"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from xencode.workspace.collaboration_manager import CollaborationManager

class SyncMessage:
    def __init__(self, message_type: str, workspace_id: str, data: Dict[str, Any], sender_id: str = ""):
        self.message_type = message_type
        self.workspace_id = workspace_id
        self.data = data
        self.sender_id = sender_id
        self.timestamp = datetime.now()
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
            sender_id=data.get('sender_id', '')
        )

class WebSocketConnection:
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
        if not self.websocket or not self.is_active:
            return False
        try:
            message_data = json.dumps(message.to_dict())
            await self.websocket.send(message_data)
            self.messages_sent += 1
            return True
        except Exception:
            self.is_active = False
            return False
    
    async def ping(self) -> float:
        if not self.is_active:
            return -1.0
        start_time = time.time()
        await asyncio.sleep(0.001)
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
    def __init__(self, collaboration_manager: Optional['CollaborationManager'] = None, crdt_engine=None):
        self.collaboration_manager = collaboration_manager
        self.crdt_engine = crdt_engine
        self.connections: Dict[str, WebSocketConnection] = {}
        self.workspace_connections: Dict[str, Set[str]] = {}
        self.message_queue: Dict[str, List[SyncMessage]] = {}
        self.batch_size = 10
        self.batch_timeout = 0.05
        self.max_latency_target = 0.05
        self.sync_stats = {
            'messages_processed': 0,
            'changes_synced': 0,
            'conflicts_resolved': 0,
            'average_sync_time': 0.0,
            'peak_connections': 0
        }
        self._running = False
    
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        print("SyncCoordinator started with real-time synchronization")
    
    async def stop(self) -> None:
        self._running = False
        for connection in list(self.connections.values()):
            await self.disconnect_client(connection.connection_id)
        print("SyncCoordinator stopped")
    
    async def connect_client(self, connection_id: str, user_id: str, workspace_id: str, websocket=None) -> bool:
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
        if connection_id in self.connections:
            return self.connections[connection_id].get_stats()
        return None
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
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
        return {
            **self.sync_stats,
            'active_connections': len(self.connections),
            'active_workspaces': len(self.workspace_connections),
            'queued_messages': sum(len(messages) for messages in self.message_queue.values()),
            'is_running': self._running
        }

sync_coordinator = SyncCoordinator()
