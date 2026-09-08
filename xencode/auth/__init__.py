#!/usr/bin/env python3
"""
Authentication and Authorization Package

Provides JWT-based authentication, role-based access control,
and security features for the Xencode system.
"""

import logging

logger = logging.getLogger(__name__)
from typing import Optional

# Import main components with graceful fallback
try:
    from .jwt_handler import JWTHandler
    JWT_HANDLER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import JWTHandler: %s", e)
    JWTHandler = None
    JWT_HANDLER_AVAILABLE = False

try:
    from .auth_manager import AuthManager
    AUTH_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import AuthManager: %s", e)
    AuthManager = None
    AUTH_MANAGER_AVAILABLE = False

try:
    from .permission_engine import PermissionEngine
    PERMISSION_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import PermissionEngine: %s", e)
    PermissionEngine = None
    PERMISSION_ENGINE_AVAILABLE = False

try:
    from .audit_logger import AuditLogger
    AUDIT_LOGGER_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import AuditLogger: %s", e)
    AuditLogger = None
    AUDIT_LOGGER_AVAILABLE = False

try:
    from .credential_vault import CredentialVault, FileBasedCredentialBackend
    FILE_BACKEND_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import FileBasedCredentialBackend: %s", e)
    FileBasedCredentialBackend = None
    CredentialVault = None
    FILE_BACKEND_AVAILABLE = False

try:
    from .json_file_vault import JsonFileCredentialVault
    JSON_FILE_VAULT_AVAILABLE = True
except ImportError as e:
    logger.warning("Failed to import JsonFileCredentialVault: %s", e)
    JsonFileCredentialVault = None
    JSON_FILE_VAULT_AVAILABLE = False


def get_auth_status() -> dict:
    """Get status of authentication components"""
    return {
        "jwt_handler_available": JWT_HANDLER_AVAILABLE,
        "auth_manager_available": AUTH_MANAGER_AVAILABLE,
        "permission_engine_available": PERMISSION_ENGINE_AVAILABLE,
        "audit_logger_available": AUDIT_LOGGER_AVAILABLE,
        "file_backend_available": FILE_BACKEND_AVAILABLE,
        "json_file_vault_available": JSON_FILE_VAULT_AVAILABLE,
    }


__all__ = [
    'JWTHandler',
    'AuthManager',
    'PermissionEngine',
    'AuditLogger',
    'FileBasedCredentialBackend',
    'CredentialVault',
    'JsonFileCredentialVault',
    'get_auth_status',
    'JWT_HANDLER_AVAILABLE',
    'AUTH_MANAGER_AVAILABLE',
    'PERMISSION_ENGINE_AVAILABLE',
    'AUDIT_LOGGER_AVAILABLE',
    'FILE_BACKEND_AVAILABLE',
    'JSON_FILE_VAULT_AVAILABLE',
]
