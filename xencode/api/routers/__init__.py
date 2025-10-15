#!/usr/bin/env python3
"""
API Routers Package

Contains all FastAPI routers for different functional areas of the Xencode API.
"""

# Import routers with graceful fallback
try:
    from .document import router as document_router
    DOCUMENT_ROUTER_AVAILABLE = True
except ImportError:
    document_router = None
    DOCUMENT_ROUTER_AVAILABLE = False

try:
    from .code_analysis import router as code_analysis_router
    CODE_ANALYSIS_ROUTER_AVAILABLE = True
except ImportError:
    code_analysis_router = None
    CODE_ANALYSIS_ROUTER_AVAILABLE = False

try:
    from .workspace import router as workspace_router
    WORKSPACE_ROUTER_AVAILABLE = True
except ImportError:
    workspace_router = None
    WORKSPACE_ROUTER_AVAILABLE = False

try:
    from .analytics import router as analytics_router
    ANALYTICS_ROUTER_AVAILABLE = True
except ImportError:
    analytics_router = None
    ANALYTICS_ROUTER_AVAILABLE = False

try:
    from .monitoring import router as monitoring_router
    MONITORING_ROUTER_AVAILABLE = True
except ImportError:
    monitoring_router = None
    MONITORING_ROUTER_AVAILABLE = False

try:
    from .plugin import router as plugin_router
    PLUGIN_ROUTER_AVAILABLE = True
except ImportError:
    plugin_router = None
    PLUGIN_ROUTER_AVAILABLE = False


def get_router_status() -> dict:
    """Get status of all routers"""
    return {
        "document_router": DOCUMENT_ROUTER_AVAILABLE,
        "code_analysis_router": CODE_ANALYSIS_ROUTER_AVAILABLE,
        "workspace_router": WORKSPACE_ROUTER_AVAILABLE,
        "analytics_router": ANALYTICS_ROUTER_AVAILABLE,
        "monitoring_router": MONITORING_ROUTER_AVAILABLE,
        "plugin_router": PLUGIN_ROUTER_AVAILABLE
    }


__all__ = [
    'document_router',
    'code_analysis_router',
    'workspace_router',
    'analytics_router',
    'monitoring_router',
    'plugin_router',
    'get_router_status'
]