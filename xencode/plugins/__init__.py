#!/usr/bin/env python3
"""
Plugins Package

Contains all Xencode plugins including file operations, marketplace integration,
and plugin management utilities.
"""

# Import plugin components with graceful fallback
try:
    from .file_operations import FileOperationsPlugin, PluginContext
    FILE_OPERATIONS_AVAILABLE = True
except ImportError:
    FileOperationsPlugin = None
    PluginContext = None
    FILE_OPERATIONS_AVAILABLE = False

try:
    from .marketplace_client import MarketplaceClient
    MARKETPLACE_CLIENT_AVAILABLE = True
except ImportError:
    MarketplaceClient = None
    MARKETPLACE_CLIENT_AVAILABLE = False


def get_plugin_status() -> dict:
    """Get status of plugin components"""
    return {
        "file_operations_available": FILE_OPERATIONS_AVAILABLE,
        "marketplace_client_available": MARKETPLACE_CLIENT_AVAILABLE
    }


__all__ = [
    'FileOperationsPlugin',
    'PluginContext', 
    'MarketplaceClient',
    'get_plugin_status',
    'FILE_OPERATIONS_AVAILABLE',
    'MARKETPLACE_CLIENT_AVAILABLE'
]