#!/usr/bin/env python3
"""
Demo: Plugin Management API

Demonstrates the plugin management endpoints including marketplace integration,
plugin installation, execution, and monitoring.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the plugin router
from xencode.api.routers.plugin import router as plugin_router


def create_demo_app() -> FastAPI:
    """Create demo FastAPI application with plugin management"""
    
    app = FastAPI(
        title="Xencode Plugin Management Demo",
        description="Demo of plugin management with marketplace integration",
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
    
    # Include plugin router
    app.include_router(plugin_router, prefix="/api/v1/plugins", tags=["Plugins"])
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Xencode Plugin Management Demo",
            "version": "1.0.0",
            "endpoints": {
                "plugins": "/api/v1/plugins",
                "marketplace": "/api/v1/plugins/marketplace",
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
                "plugin_manager": "available",
                "marketplace_client": "available",
                "plugin_execution": "available"
            }
        }
    
    return app


async def demo_plugin_operations():
    """Demo plugin operations"""
    
    print("🔌 Xencode Plugin Management Demo")
    print("=" * 50)
    
    print("\n📋 Available Plugin Endpoints:")
    print("  GET    /api/v1/plugins                           - List plugins")
    print("  GET    /api/v1/plugins/{id}                      - Get plugin details")
    print("  POST   /api/v1/plugins/install                   - Install plugin")
    print("  POST   /api/v1/plugins/upload                    - Upload plugin file")
    print("  PUT    /api/v1/plugins/{id}                      - Update plugin")
    print("  POST   /api/v1/plugins/{id}/enable               - Enable plugin")
    print("  POST   /api/v1/plugins/{id}/disable              - Disable plugin")
    print("  DELETE /api/v1/plugins/{id}                      - Uninstall plugin")
    print("  POST   /api/v1/plugins/{id}/execute              - Execute plugin")
    print("  GET    /api/v1/plugins/{id}/config               - Get plugin config")
    print("  PUT    /api/v1/plugins/{id}/config               - Update plugin config")
    print("  GET    /api/v1/plugins/{id}/stats                - Get plugin stats")
    print("  GET    /api/v1/plugins/{id}/logs                 - Get plugin logs")
    
    print("\n🏪 Marketplace Endpoints:")
    print("  GET    /api/v1/plugins/marketplace/info          - Marketplace info")
    print("  POST   /api/v1/plugins/marketplace/search        - Search plugins")
    print("  GET    /api/v1/plugins/marketplace/categories    - Get categories")
    print("  GET    /api/v1/plugins/system/status             - System status")
    
    print("\n🔧 Key Features:")
    print("  ✅ Plugin lifecycle management (install/enable/disable/uninstall)")
    print("  ✅ Marketplace integration with search and discovery")
    print("  ✅ Plugin execution with timeout and monitoring")
    print("  ✅ Configuration management and updates")
    print("  ✅ Statistics and performance monitoring")
    print("  ✅ Log aggregation and debugging")
    print("  ✅ File upload for custom plugins")
    print("  ✅ Background installation and updates")
    
    print("\n📦 Example Plugin Installation:")
    install_request = {
        "plugin_id": "file-operations",
        "version": "1.0.0",
        "source": "marketplace",
        "verify_signature": True,
        "auto_enable": True
    }
    print(json.dumps(install_request, indent=2))
    
    print("\n⚡ Example Plugin Execution:")
    execute_request = {
        "method": "ls_dir",
        "args": ["/home/user/projects"],
        "kwargs": {"show_hidden": False},
        "timeout_seconds": 30,
        "async_execution": False
    }
    print(json.dumps(execute_request, indent=2))
    
    print("\n🔍 Example Marketplace Search:")
    search_request = {
        "query": "file operations",
        "category": "utilities",
        "tags": ["filesystem", "productivity"],
        "sort_by": "downloads",
        "limit": 10,
        "offset": 0
    }
    print(json.dumps(search_request, indent=2))
    
    print("\n⚙️ Example Plugin Configuration:")
    config_update = {
        "config": {
            "max_file_size_mb": 100,
            "allowed_extensions": [".py", ".js", ".md"],
            "auto_backup": True,
            "backup_interval_hours": 24
        },
        "restart_required": False
    }
    print(json.dumps(config_update, indent=2))
    
    print("\n📊 Example Plugin Statistics:")
    stats_example = {
        "plugin_id": "file-operations",
        "total_executions": 1250,
        "successful_executions": 1200,
        "failed_executions": 50,
        "average_execution_time_ms": 45.2,
        "total_memory_used_mb": 128.5,
        "uptime_hours": 168.5,
        "last_24h_executions": 85,
        "error_rate_percent": 4.0
    }
    print(json.dumps(stats_example, indent=2))
    
    print("\n🏪 Example Marketplace Info:")
    marketplace_info = {
        "total_plugins": 1247,
        "categories": [
            "Development Tools",
            "File Operations", 
            "Data Processing",
            "AI/ML",
            "Security",
            "Productivity"
        ],
        "featured_plugins": [
            "advanced-git-tools",
            "ai-code-reviewer", 
            "security-scanner",
            "performance-profiler"
        ],
        "marketplace_status": "online"
    }
    print(json.dumps(marketplace_info, indent=2))
    
    print("\n🎯 To test the API:")
    print("  1. Run: python demo_plugin_management.py")
    print("  2. Open: http://localhost:8000/docs")
    print("  3. Try the plugin management endpoints")
    print("  4. Upload a plugin file or search the marketplace")
    print("  5. Execute plugin methods and monitor performance")


def main():
    """Main demo function"""
    
    # Run the demo
    asyncio.run(demo_plugin_operations())
    
    print("\n🚀 Starting FastAPI server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔌 Plugin Management: http://localhost:8000/api/v1/plugins")
    print("🏪 Marketplace Search: http://localhost:8000/api/v1/plugins/marketplace/search")
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