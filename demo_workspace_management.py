#!/usr/bin/env python3
"""
Demo: Workspace Management API

Demonstrates the workspace management endpoints including CRDT-based collaboration,
real-time synchronization, and WebSocket support.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the workspace router
from xencode.api.routers.workspace import router as workspace_router


def create_demo_app() -> FastAPI:
    """Create demo FastAPI application with workspace management"""
    
    app = FastAPI(
        title="Xencode Workspace Management Demo",
        description="Demo of workspace management with CRDT collaboration",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include workspace router
    app.include_router(workspace_router, prefix="/api/v1/workspaces", tags=["Workspaces"])
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Xencode Workspace Management Demo",
            "version": "1.0.0",
            "endpoints": {
                "workspaces": "/api/v1/workspaces",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "workspace_manager": "available",
                "crdt_engine": "available",
                "websocket_support": "available"
            }
        }
    
    return app


async def demo_workspace_operations():
    """Demo workspace operations"""
    
    print("🚀 Xencode Workspace Management Demo")
    print("=" * 50)
    
    # This would normally use an HTTP client to test the API
    # For now, we'll just show the available endpoints
    
    print("\n📋 Available Workspace Endpoints:")
    print("  POST   /api/v1/workspaces                    - Create workspace")
    print("  GET    /api/v1/workspaces                    - List workspaces")
    print("  GET    /api/v1/workspaces/{id}               - Get workspace")
    print("  PUT    /api/v1/workspaces/{id}               - Update workspace")
    print("  DELETE /api/v1/workspaces/{id}               - Delete workspace")
    print("  POST   /api/v1/workspaces/{id}/sync          - Sync changes (CRDT)")
    print("  GET    /api/v1/workspaces/{id}/collaboration - Get collaboration status")
    print("  WS     /api/v1/workspaces/{id}/ws            - WebSocket for real-time sync")
    print("  GET    /api/v1/workspaces/{id}/export        - Export workspace")
    
    print("\n🔧 Key Features:")
    print("  ✅ CRDT-based conflict resolution")
    print("  ✅ Real-time collaboration via WebSocket")
    print("  ✅ Workspace isolation and security")
    print("  ✅ File management with locking")
    print("  ✅ Change tracking and history")
    print("  ✅ Streaming export functionality")
    
    print("\n📊 Example Workspace Configuration:")
    example_config = {
        "name": "My Project",
        "description": "A collaborative coding project",
        "settings": {
            "auto_save_enabled": True,
            "real_time_sync": True,
            "max_collaborators": 10
        },
        "collaborators": ["user1", "user2"],
        "crdt_enabled": True
    }
    print(json.dumps(example_config, indent=2))
    
    print("\n🔄 Example CRDT Change:")
    example_change = {
        "changes": [
            {
                "id": "change-123",
                "operation": "insert",
                "path": "/src/main.py",
                "content": "print('Hello, World!')",
                "timestamp": datetime.now().isoformat(),
                "author": "user1",
                "vector_clock": {"user1": 1, "user2": 0}
            }
        ],
        "crdt_vector": {"user1": 1, "user2": 0},
        "session_id": "session-456"
    }
    print(json.dumps(example_change, indent=2))
    
    print("\n🌐 WebSocket Message Example:")
    ws_message = {
        "type": "realtime_change",
        "change": {
            "id": "change-789",
            "operation": "update",
            "path": "/src/utils.py",
            "content": "def helper(): pass",
            "timestamp": datetime.now().isoformat(),
            "author": "user2"
        }
    }
    print(json.dumps(ws_message, indent=2))
    
    print("\n🎯 To test the API:")
    print("  1. Run: python demo_workspace_management.py")
    print("  2. Open: http://localhost:8000/docs")
    print("  3. Try the workspace endpoints")
    print("  4. Use WebSocket client to test real-time features")


def main():
    """Main demo function"""
    
    # Run the demo
    asyncio.run(demo_workspace_operations())
    
    print("\n🚀 Starting FastAPI server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 WebSocket Test: ws://localhost:8000/api/v1/workspaces/{workspace_id}/ws")
    print("⏹️  Press Ctrl+C to stop")
    
    # Create and run the app
    app = create_demo_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )


if __name__ == "__main__":
    main()