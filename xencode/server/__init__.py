"""
Server package for Xencode.

Provides collaboration server functionality including FastAPI application,
WebSocket support, and database management.
"""

from .app import create_app, get_server_app
from .database import get_server_database, init_server_database
from .socket_manager import SocketManager

__all__ = [
    "create_app",
    "get_server_app",
    "SocketManager",
    "get_server_database",
    "init_server_database",
]
