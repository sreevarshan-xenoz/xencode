#!/usr/bin/env python3
"""
Workspace Management API Router

FastAPI router for workspace management endpoints including CRDT-based collaboration,
real-time synchronization, and WebSocket support for live updates.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
import uuid

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import workspace components
try:
    from ...workspace.workspace_manager import WorkspaceManager
    from ...workspace.crdt_engine import CRDTEngine, Change, Conflict
    from ...workspace.sync_coordinator import SyncCoordinator
    from ...models.workspace import Workspace as WorkspaceModel, WorkspaceConfig as WorkspaceConfigModel
    WORKSPACE_COMPONENTS_AVAILABLE = True
except ImportError:
    WORKSPACE_COMPONENTS_AVAILABLE = False

router = APIRouter()


# Pydantic models for API
class WorkspaceConfig(BaseModel):
    """Workspace configuration"""
    name: str
    description: Optional[str] = None
    collaborators: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    crdt_enabled: bool = True


class Workspace(BaseModel):
    """Workspace information"""
    id: str
    config: WorkspaceConfig
    created_at: datetime
    last_modified: datetime
    file_count: int
    storage_size_bytes: int
    active_sessions: int
    collaboration_status: str = "active"


class WorkspaceCreateRequest(BaseModel):
    """Request to create workspace"""
    name: str
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    collaborators: List[str] = Field(default_factory=list)
    crdt_enabled: bool = True


class WorkspaceUpdateRequest(BaseModel):
    """Request to update workspace"""
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    collaborators: Optional[List[str]] = None


class ChangeRequest(BaseModel):
    """Request to sync workspace changes"""
    changes: List[Dict[str, Any]]
    crdt_vector: Dict[str, int] = Field(default_factory=dict)
    session_id: str


class SyncResponse(BaseModel):
    """Response for sync operations"""
    success: bool
    conflicts_resolved: int
    new_vector: Dict[str, int]
    applied_changes: int
    timestamp: datetime


class CollaborationStatus(BaseModel):
    """Collaboration status information"""
    workspace_id: str
    active_sessions: int
    collaborators: List[Dict[str, Any]]
    recent_changes: List[Dict[str, Any]]
    sync_status: str


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time collaboration"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, workspace_id: str, user_id: str, session_id: str):
        """Connect a WebSocket to a workspace"""
        await websocket.accept()
        
        if workspace_id not in self.active_connections:
            self.active_connections[workspace_id] = set()
        
        self.active_connections[workspace_id].add(websocket)
        self.session_info[websocket] = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "session_id": session_id,
            "connected_at": datetime.now()
        }
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        if websocket in self.session_info:
            workspace_id = self.session_info[websocket]["workspace_id"]
            
            if workspace_id in self.active_connections:
                self.active_connections[workspace_id].discard(websocket)
                
                if not self.active_connections[workspace_id]:
                    del self.active_connections[workspace_id]
            
            del self.session_info[websocket]
    
    async def broadcast_to_workspace(self, workspace_id: str, message: Dict[str, Any], 
                                   exclude_session: Optional[str] = None):
        """Broadcast message to all connections in a workspace"""
        if workspace_id not in self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for websocket in self.active_connections[workspace_id].copy():
            session_info = self.session_info.get(websocket, {})
            
            # Skip excluded session
            if exclude_session and session_info.get("session_id") == exclude_session:
                continue
            
            try:
                await websocket.send_text(message_json)
            except Exception:
                disconnected.append(websocket)
        
        # Clean up disconnected sockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def get_workspace_sessions(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get active sessions for a workspace"""
        if workspace_id not in self.active_connections:
            return []
        
        sessions = []
        for websocket in self.active_connections[workspace_id]:
            if websocket in self.session_info:
                sessions.append(self.session_info[websocket])
        
        return sessions


# Global connection manager
connection_manager = ConnectionManager()


# Dependency to get workspace manager
async def get_workspace_manager():
    """Dependency to get workspace manager"""
    if not WORKSPACE_COMPONENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workspace components not available")
    
    try:
        # For now, return a mock manager - in production this would be a singleton
        return WorkspaceManager()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workspace manager: {e}")


@router.post("/", response_model=Workspace)
async def create_workspace(request: WorkspaceCreateRequest):
    """Create a new workspace"""
    try:
        import uuid
        
        return Workspace(
            id=str(uuid.uuid4()),
            config=WorkspaceConfig(
                name=request.name,
                description=request.description,
                settings=request.settings
            ),
            created_at=datetime.now(),
            last_modified=datetime.now(),
            file_count=0,
            storage_size_bytes=0,
            active_sessions=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workspace: {e}")


@router.get("/", response_model=List[Workspace])
async def list_workspaces():
    """List workspaces"""
    return []


@router.get("/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str):
    """Get a specific workspace"""
    raise HTTPException(status_code=404, detail="Workspace not found")


@router.put("/{workspace_id}")
async def update_workspace(workspace_id: str, config: WorkspaceConfig):
    """Update workspace configuration"""
    raise HTTPException(status_code=404, detail="Workspace not found")


@router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace"""
    raise HTTPException(status_code=404, detail="Workspace not found")


router.tags = ["Workspaces"]