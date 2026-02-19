#!/usr/bin/env python3
"""
Xencode API Package

FastAPI-based REST API for exposing all Xencode functionality including:
- Document processing endpoints
- Code analysis services
- Workspace management
- Analytics and monitoring
- Plugin system integration
- Resource management
"""

from typing import Optional

# Import API components with graceful fallback
try:
    from .main import app, get_app_status
    FASTAPI_AVAILABLE = True
except ImportError:
    app = None
    get_app_status = None
    FASTAPI_AVAILABLE = False

try:
    from .routers import (
        document_router,
        code_analysis_router,
        workspace_router,
        analytics_router,
        monitoring_router,
        plugin_router,
        features_router
    )
    ROUTERS_AVAILABLE = True
except ImportError:
    document_router = None
    code_analysis_router = None
    workspace_router = None
    analytics_router = None
    monitoring_router = None
    plugin_router = None
    features_router = None
    ROUTERS_AVAILABLE = False

try:
    from .middleware import setup_middleware
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    setup_middleware = None
    MIDDLEWARE_AVAILABLE = False


def get_api_status() -> dict:
    """Get status of API components"""
    return {
        "fastapi_available": FASTAPI_AVAILABLE,
        "routers_available": ROUTERS_AVAILABLE,
        "middleware_available": MIDDLEWARE_AVAILABLE
    }


__all__ = [
    'app',
    'get_app_status',
    'get_api_status',
    'document_router',
    'code_analysis_router',
    'workspace_router',
    'analytics_router',
    'monitoring_router',
    'plugin_router',
    'features_router',
    'setup_middleware',
    'FASTAPI_AVAILABLE',
    'ROUTERS_AVAILABLE',
    'MIDDLEWARE_AVAILABLE'
]