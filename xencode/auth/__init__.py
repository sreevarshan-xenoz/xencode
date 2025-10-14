#!/usr/bin/env python3
"""
Authentication and Authorization Package

Provides JWT-based authentication, role-based access control,
and security features for the Xencode system.
"""

from typing import Optional

# Import main components with graceful fallback
try:
    from .jwt_handler import JWTHandler
    JWT_HANDLER_AVAILABLE = True
except ImportError:
    JWTHandler = None
    JWT_HANDLER_AVAILABLE = False

try:
    from .auth_manager import AuthManager
    AUTH_MANAGER_AVAILABLE = True
except ImportError:
    AuthManager = None
    AUTH_MANAGER_AVAILABLE = False

try:
    from .permission_engine import PermissionEngine
    PERMISSION_ENGINE_AVAILABLE = True
except ImportError:
    PermissionEngine = None
    PERMISSION_ENGINE_AVAILABLE = False

try:
    from .audit_logger import AuditLogger
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AuditLogger = None
    AUDIT_LOGGER_AVAILABLE = False


def get_auth_status() -> dict:
    """Get status of authentication components"""
    return {
        "jwt_handler_available": JWT_HANDLER_AVAILABLE,
        "auth_manager_available": AUTH_MANAGER_AVAILABLE,
        "permission_engine_available": PERMISSION_ENGINE_AVAILABLE,
        "audit_logger_available": AUDIT_LOGGER_AVAILABLE
    }


__all__ = [
    'JWTHandler',
    'AuthManager',
    'PermissionEngine',
    'AuditLogger',
    'get_auth_status',
    'JWT_HANDLER_AVAILABLE',
    'AUTH_MANAGER_AVAILABLE',
    'PERMISSION_ENGINE_AVAILABLE',
    'AUDIT_LOGGER_AVAILABLE'
]