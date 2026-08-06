"""
Core module for Xencode - Main Application Logic
"""
from .cache import ResponseCache
from .connection_pool import APIClient, close_api_client, get_api_client
from .files import create_file, delete_file, read_file, write_file
from .memory import ConversationMemory
from .models import (
    ModelManager,
    get_available_models,
    get_smart_default_model,
    list_models,
    update_model,
)

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
    'close_api_client'
]
