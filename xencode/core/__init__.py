"""
Core module for Xencode - Main Application Logic
"""
from .files import create_file, read_file, write_file, delete_file
from .models import ModelManager, get_smart_default_model, get_available_models, list_models, update_model
from .memory import ConversationMemory
from .cache import ResponseCache

__all__ = [
    'create_file',
    'read_file',
    'write_file',
    'delete_file',
    'ModelManager',
    'get_smart_default_model',
    'get_available_models',
    'list_models',
    'update_model',
    'ConversationMemory',
    'ResponseCache',
    'APIClient',
    'get_api_client',
    'close_api_client',
]

# Lazy-load connection pool (requires optional aiohttp)
try:
    from .connection_pool import APIClient, get_api_client, close_api_client
except ImportError:
    APIClient = None
    get_api_client = None
    close_api_client = None