#!/usr/bin/env python3
"""
Workspace Management Package

Provides workspace management, CRDT-based collaboration,
and SQLite storage for the Xencode system.
"""

from typing import Optional

# Import main components with graceful fallback
try:
    from .workspace_manager import WorkspaceManager
    WORKSPACE_MANAGER_AVAILABLE = True
except ImportError:
    WorkspaceManager = None
    WORKSPACE_MANAGER_AVAILABLE = False

try:
    from .crdt_engine import CRDTEngine
    CRDT_ENGINE_AVAILABLE = True
except ImportError:
    CRDTEngine = None
    CRDT_ENGINE_AVAILABLE = False

try:
    from .storage_backend import SQLiteStorageBackend
    STORAGE_BACKEND_AVAILABLE = True
except ImportError:
    SQLiteStorageBackend = None
    STORAGE_BACKEND_AVAILABLE = False

try:
    from .collaboration_manager import CollaborationManager
    COLLABORATION_MANAGER_AVAILABLE = True
except ImportError:
    CollaborationManager = None
    COLLABORATION_MANAGER_AVAILABLE = False

try:
    from .sync_coordinator import SyncCoordinator, SyncMessage, WebSocketConnection
    SYNC_COORDINATOR_AVAILABLE = True
except ImportError:
    SyncCoordinator = None
    SyncMessage = None
    WebSocketConnection = None
    SYNC_COORDINATOR_AVAILABLE = False

try:
    from .workspace_security import (
        WorkspaceSecurityManager, WorkspacePermission, IsolationLevel, WorkspaceContext
    )
    WORKSPACE_SECURITY_AVAILABLE = True
except ImportError:
    WorkspaceSecurityManager = None
    WorkspacePermission = None
    IsolationLevel = None
    WorkspaceContext = None
    WORKSPACE_SECURITY_AVAILABLE = False


def get_workspace_status() -> dict:
    """Get status of workspace components"""
    return {
        "workspace_manager_available": WORKSPACE_MANAGER_AVAILABLE,
        "crdt_engine_available": CRDT_ENGINE_AVAILABLE,
        "storage_backend_available": STORAGE_BACKEND_AVAILABLE,
        "collaboration_manager_available": COLLABORATION_MANAGER_AVAILABLE,
        "sync_coordinator_available": SYNC_COORDINATOR_AVAILABLE,
        "workspace_security_available": WORKSPACE_SECURITY_AVAILABLE
    }


__all__ = [
    'WorkspaceManager',
    'CRDTEngine',
    'SQLiteStorageBackend',
    'CollaborationManager',
    'SyncCoordinator',
    'SyncMessage',
    'WebSocketConnection',
    'WorkspaceSecurityManager',
    'WorkspacePermission',
    'IsolationLevel',
    'WorkspaceContext',
    'get_workspace_status',
    'WORKSPACE_MANAGER_AVAILABLE',
    'CRDT_ENGINE_AVAILABLE',
    'STORAGE_BACKEND_AVAILABLE',
    'COLLABORATION_MANAGER_AVAILABLE',
    'SYNC_COORDINATOR_AVAILABLE',
    'WORKSPACE_SECURITY_AVAILABLE'
]