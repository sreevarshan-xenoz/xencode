from typing import List, Dict, Any
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for collaboration sessions"""
    
    def __init__(self):
        # Map session_id -> List of WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept connection and add to session"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"Client connected to session {session_id}")
        
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove connection from session"""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"Client disconnected from session {session_id}")
                
    async def broadcast(self, message: Dict[str, Any], session_id: str, exclude: WebSocket = None):
        """Broadcast message to all clients in session"""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                if connection != exclude:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting to client: {e}")
                        # Could remove dead connection here
                        
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

manager = ConnectionManager()
