#!/usr/bin/env python3
"""
Xencode Plugins Package

Contains plugin-related utilities and marketplace integration.
"""

from typing import Optional

# Import marketplace client with graceful fallback
try:
    from .marketplace_client import MarketplaceClient
    MARKETPLACE_CLIENT_AVAILABLE = True
except ImportError:
    MarketplaceClient = None
    MARKETPLACE_CLIENT_AVAILABLE = False


def get_plugins_status() -> dict:
    """Get status of plugin components"""
    return {
        "marketplace_client_available": MARKETPLACE_CLIENT_AVAILABLE
    }


__all__ = [
    'MarketplaceClient',
    'get_plugins_status',
    'MARKETPLACE_CLIENT_AVAILABLE'
]