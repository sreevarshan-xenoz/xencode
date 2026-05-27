#!/usr/bin/env python3
"""
API Routers Package

Contains all FastAPI routers for different functional areas of the Xencode API.
"""

import logging

logger = logging.getLogger(__name__)
# Import routers with graceful fallback
try:
    from .document import router as document_router
    DOCUMENT_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import document_router: %s", e)
    document_router = None
    DOCUMENT_ROUTER_AVAILABLE = False

try:
    from .code_analysis import router as code_analysis_router
    CODE_ANALYSIS_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import code_analysis_router: %s", e)
    code_analysis_router = None
    CODE_ANALYSIS_ROUTER_AVAILABLE = False

try:
    from .workspace import router as workspace_router
    WORKSPACE_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import workspace_router: %s", e)
    workspace_router = None
    WORKSPACE_ROUTER_AVAILABLE = False

try:
    from .analytics import router as analytics_router
    ANALYTICS_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import analytics_router: %s", e)
    analytics_router = None
    ANALYTICS_ROUTER_AVAILABLE = False

try:
    from .monitoring import router as monitoring_router
    MONITORING_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import monitoring_router: %s", e)
    monitoring_router = None
    MONITORING_ROUTER_AVAILABLE = False

try:
    from .plugin import router as plugin_router
    PLUGIN_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import plugin_router: %s", e)
    plugin_router = None
    PLUGIN_ROUTER_AVAILABLE = False

try:
    from .features import router as features_router
    FEATURES_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import features_router: %s", e)
    features_router = None
    FEATURES_ROUTER_AVAILABLE = False

try:
    from .fallback import router as fallback_router
    FALLBACK_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import fallback_router: %s", e)
    fallback_router = None
    FALLBACK_ROUTER_AVAILABLE = False

try:
    from .testing import router as testing_router
    TESTING_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import testing_router: %s", e)
    testing_router = None
    TESTING_ROUTER_AVAILABLE = False


def get_router_status() -> dict:
    """Get status of all routers"""
    return {
        "document_router": DOCUMENT_ROUTER_AVAILABLE,
        "code_analysis_router": CODE_ANALYSIS_ROUTER_AVAILABLE,
        "workspace_router": WORKSPACE_ROUTER_AVAILABLE,
        "analytics_router": ANALYTICS_ROUTER_AVAILABLE,
        "monitoring_router": MONITORING_ROUTER_AVAILABLE,
        "plugin_router": PLUGIN_ROUTER_AVAILABLE,
        "features_router": FEATURES_ROUTER_AVAILABLE,
        "fallback_router": FALLBACK_ROUTER_AVAILABLE,
        "testing_router": TESTING_ROUTER_AVAILABLE,
    }


__all__ = [
    'document_router',
    'code_analysis_router',
    'workspace_router',
    'analytics_router',
    'monitoring_router',
    'plugin_router',
    'features_router',
    'fallback_router',
    'testing_router',
    'get_router_status'
]