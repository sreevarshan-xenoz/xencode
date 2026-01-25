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
    from ...models.workspace import (
        Workspace as WorkspaceModel,
        WorkspaceConfig as WorkspaceConfigModel,
        WorkspaceType,
        WorkspaceStatus,
        CollaborationMode
    )
    WORKSPACE_COMPONENTS_AVAILABLE = True
except ImportError:
    # Define stub types for when workspace components are not available
    from typing import Any
    Change = Any
    Conflict = Any
    CRDTEngine = Any
    SyncCoordinator = Any
    WorkspaceManager = Any
    WorkspaceModel = Any
    WorkspaceConfigModel = Any
    WorkspaceType = Any
    WorkspaceStatus = Any
    CollaborationMode = Any
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
async def create_workspace(
    request: WorkspaceCreateRequest,
    workspace_manager = Depends(get_workspace_manager)
):
    """Create a new workspace with CRDT support"""
    try:
        workspace_id = str(uuid.uuid4())
        
        # Create workspace configuration using the actual model structure
        config = WorkspaceConfigModel()  # Use defaults
        
        # Override with request settings if provided
        if request.settings:
            for key, value in request.settings.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create workspace through manager
        if WORKSPACE_COMPONENTS_AVAILABLE:
            try:
                workspace = await workspace_manager.create_workspace(config)
            except Exception:
                # Fallback to mock implementation
                workspace = WorkspaceModel(
                    id=workspace_id,
                    name=request.name,
                    description=request.description or "",
                    owner_id="system",  # TODO: Get from auth context
                    config=config,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    file_count=0,
                    total_size_bytes=0,
                    active_collaborators=0
                )
        else:
            # Mock implementation for testing
            workspace = WorkspaceModel(
                id=workspace_id,
                name=request.name,
                description=request.description or "",
                owner_id="system",  # TODO: Get from auth context
                config=config,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_count=0,
                total_size_bytes=0,
                active_collaborators=0
            )
        
        return Workspace(
            id=workspace.id,
            config=WorkspaceConfig(
                name=workspace.config.name,
                description=workspace.config.description,
                collaborators=workspace.config.collaborators,
                settings=workspace.config.settings,
                crdt_enabled=workspace.config.crdt_enabled
            ),
            created_at=workspace.created_at,
            last_modified=workspace.last_modified,
            file_count=workspace.file_count,
            storage_size_bytes=workspace.storage_size_bytes,
            active_sessions=len(connection_manager.get_workspace_sessions(workspace.id))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workspace: {e}")


@router.get("/", response_model=List[Workspace])
async def list_workspaces(
    workspace_manager = Depends(get_workspace_manager)
):
    """List all workspaces"""
    try:
        if WORKSPACE_COMPONENTS_AVAILABLE:
            workspaces = await workspace_manager.list_workspaces()
        else:
            # Mock implementation
            workspaces = []
        
        result = []
        for ws in workspaces:
            result.append(Workspace(
                id=ws.id,
                config=WorkspaceConfig(
                    name=ws.config.name,
                    description=ws.config.description,
                    collaborators=ws.config.collaborators,
                    settings=ws.config.settings,
                    crdt_enabled=ws.config.crdt_enabled
                ),
                created_at=ws.created_at,
                last_modified=ws.last_modified,
                file_count=ws.file_count,
                storage_size_bytes=ws.storage_size_bytes,
                active_sessions=len(connection_manager.get_workspace_sessions(ws.id))
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workspaces: {e}")


@router.get("/{workspace_id}", response_model=Workspace)
async def get_workspace(
    workspace_id: str,
    workspace_manager = Depends(get_workspace_manager)
):
    """Get a specific workspace"""
    try:
        if WORKSPACE_COMPONENTS_AVAILABLE:
            workspace = await workspace_manager.get_workspace(workspace_id)
        else:
            # Mock implementation
            workspace = None
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return Workspace(
            id=workspace.id,
            config=WorkspaceConfig(
                name=workspace.config.name,
                description=workspace.config.description,
                collaborators=workspace.config.collaborators,
                settings=workspace.config.settings,
                crdt_enabled=workspace.config.crdt_enabled
            ),
            created_at=workspace.created_at,
            last_modified=workspace.last_modified,
            file_count=workspace.file_count,
            storage_size_bytes=workspace.storage_size_bytes,
            active_sessions=len(connection_manager.get_workspace_sessions(workspace.id))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workspace: {e}")


@router.put("/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: str, 
    request: WorkspaceUpdateRequest,
    workspace_manager = Depends(get_workspace_manager)
):
    """Update workspace configuration"""
    try:
        if WORKSPACE_COMPONENTS_AVAILABLE:
            workspace = await workspace_manager.get_workspace(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            # Update configuration
            if request.name is not None:
                workspace.config.name = request.name
            if request.description is not None:
                workspace.config.description = request.description
            if request.settings is not None:
                workspace.config.settings.update(request.settings)
            if request.collaborators is not None:
                workspace.config.collaborators = request.collaborators
            
            workspace = await workspace_manager.update_workspace(workspace)
        else:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Broadcast update to connected clients
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "workspace_updated",
                "workspace_id": workspace_id,
                "config": {
                    "name": workspace.config.name,
                    "description": workspace.config.description,
                    "settings": workspace.config.settings,
                    "collaborators": workspace.config.collaborators
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return Workspace(
            id=workspace.id,
            config=WorkspaceConfig(
                name=workspace.config.name,
                description=workspace.config.description,
                collaborators=workspace.config.collaborators,
                settings=workspace.config.settings,
                crdt_enabled=workspace.config.crdt_enabled
            ),
            created_at=workspace.created_at,
            last_modified=workspace.last_modified,
            file_count=workspace.file_count,
            storage_size_bytes=workspace.storage_size_bytes,
            active_sessions=len(connection_manager.get_workspace_sessions(workspace.id))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update workspace: {e}")


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    workspace_manager = Depends(get_workspace_manager)
):
    """Delete a workspace"""
    try:
        if WORKSPACE_COMPONENTS_AVAILABLE:
            success = await workspace_manager.delete_workspace(workspace_id)
            if not success:
                raise HTTPException(status_code=404, detail="Workspace not found")
        else:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Notify connected clients
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "workspace_deleted",
                "workspace_id": workspace_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Disconnect all clients from this workspace
        if workspace_id in connection_manager.active_connections:
            for websocket in connection_manager.active_connections[workspace_id].copy():
                try:
                    await websocket.close(code=1000, reason="Workspace deleted")
                except Exception:
                    pass
            del connection_manager.active_connections[workspace_id]
        
        return {"message": "Workspace deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {e}")


@router.post("/{workspace_id}/sync", response_model=SyncResponse)
async def sync_workspace_changes(
    workspace_id: str,
    request: ChangeRequest,
    background_tasks: BackgroundTasks,
    workspace_manager = Depends(get_workspace_manager)
):
    """Synchronize workspace changes using CRDT"""
    try:
        if not WORKSPACE_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="CRDT synchronization not available")
        
        # Verify workspace exists
        workspace = await workspace_manager.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Convert changes to CRDT format
        changes = []
        for change_data in request.changes:
            change = Change(
                id=change_data.get("id", str(uuid.uuid4())),
                operation=change_data.get("operation", "update"),
                path=change_data.get("path", ""),
                content=change_data.get("content", ""),
                timestamp=datetime.fromisoformat(change_data.get("timestamp", datetime.now().isoformat())),
                author=change_data.get("author", "unknown"),
                vector_clock=change_data.get("vector_clock", {})
            )
            changes.append(change)
        
        # Apply changes through CRDT engine
        sync_result = await workspace_manager.sync_changes(workspace_id, changes, request.crdt_vector)
        
        # Broadcast changes to other connected clients
        background_tasks.add_task(
            broadcast_changes_to_workspace,
            workspace_id,
            changes,
            request.session_id,
            sync_result.conflicts_resolved
        )
        
        return SyncResponse(
            success=True,
            conflicts_resolved=sync_result.conflicts_resolved,
            new_vector=sync_result.new_vector,
            applied_changes=len(changes),
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync changes: {e}")


@router.get("/{workspace_id}/collaboration", response_model=CollaborationStatus)
async def get_collaboration_status(workspace_id: str):
    """Get real-time collaboration status"""
    try:
        sessions = connection_manager.get_workspace_sessions(workspace_id)
        
        collaborators = []
        for session in sessions:
            collaborators.append({
                "user_id": session.get("user_id", "unknown"),
                "session_id": session.get("session_id", ""),
                "connected_at": session.get("connected_at", datetime.now()).isoformat(),
                "status": "active"
            })
        
        return CollaborationStatus(
            workspace_id=workspace_id,
            active_sessions=len(sessions),
            collaborators=collaborators,
            recent_changes=[],  # TODO: Implement recent changes tracking
            sync_status="active" if sessions else "idle"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collaboration status: {e}")


@router.websocket("/{workspace_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    workspace_id: str,
    user_id: str = "anonymous",
    session_id: str = None
):
    """WebSocket endpoint for real-time collaboration"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    await connection_manager.connect(websocket, workspace_id, user_id, session_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "workspace_id": workspace_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Notify other clients about new connection
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            },
            exclude_session=session_id
        )
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "change":
                # Process and broadcast changes
                await handle_realtime_change(workspace_id, message, session_id)
            elif message.get("type") == "cursor_position":
                # Broadcast cursor position to other clients
                await connection_manager.broadcast_to_workspace(
                    workspace_id,
                    {
                        "type": "cursor_update",
                        "user_id": user_id,
                        "session_id": session_id,
                        "position": message.get("position", {}),
                        "timestamp": datetime.now().isoformat()
                    },
                    exclude_session=session_id
                )
            elif message.get("type") == "ping":
                # Respond to ping with pong
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)
        
        # Notify other clients about disconnection
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/{workspace_id}/export")
async def export_workspace(workspace_id: str):
    """Export workspace data as streaming response"""
    try:
        async def generate_export():
            yield '{"workspace_id": "' + workspace_id + '", '
            yield '"exported_at": "' + datetime.now().isoformat() + '", '
            yield '"data": {'
            
            # TODO: Stream actual workspace data
            yield '"files": [], '
            yield '"changes": [], '
            yield '"metadata": {}'
            
            yield '}}'
        
        return StreamingResponse(
            generate_export(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=workspace_{workspace_id}.json"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export workspace: {e}")


# Helper functions
async def broadcast_changes_to_workspace(
    workspace_id: str, 
    changes: List[Change], 
    exclude_session: str,
    conflicts_resolved: int
):
    """Background task to broadcast changes to workspace clients"""
    try:
        changes_data = []
        for change in changes:
            changes_data.append({
                "id": change.id,
                "operation": change.operation,
                "path": change.path,
                "content": change.content,
                "timestamp": change.timestamp.isoformat(),
                "author": change.author,
                "vector_clock": change.vector_clock
            })
        
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "changes_applied",
                "changes": changes_data,
                "conflicts_resolved": conflicts_resolved,
                "timestamp": datetime.now().isoformat()
            },
            exclude_session=exclude_session
        )
    except Exception as e:
        print(f"Failed to broadcast changes: {e}")


async def handle_realtime_change(workspace_id: str, message: Dict[str, Any], session_id: str):
    """Handle real-time change from WebSocket client"""
    try:
        # Extract change data
        change_data = message.get("change", {})
        
        # Create change object
        change = Change(
            id=change_data.get("id", str(uuid.uuid4())),
            operation=change_data.get("operation", "update"),
            path=change_data.get("path", ""),
            content=change_data.get("content", ""),
            timestamp=datetime.now(),
            author=change_data.get("author", "unknown"),
            vector_clock=change_data.get("vector_clock", {})
        )
        
        # Apply change if workspace components are available
        if WORKSPACE_COMPONENTS_AVAILABLE:
            # TODO: Apply change through workspace manager
            pass
        
        # Broadcast to other clients
        await connection_manager.broadcast_to_workspace(
            workspace_id,
            {
                "type": "realtime_change",
                "change": {
                    "id": change.id,
                    "operation": change.operation,
                    "path": change.path,
                    "content": change.content,
                    "timestamp": change.timestamp.isoformat(),
                    "author": change.author
                },
                "timestamp": datetime.now().isoformat()
            },
            exclude_session=session_id
        )
        
    except Exception as e:
        print(f"Failed to handle realtime change: {e}")


router.tags = ["Workspaces"]