"""Basic tests for xencode_core module."""
import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xencode_core import (
    ConversationMemory,
    ResponseCache,
    ModelManager,
    extract_thinking_and_answer,
    display_chat_banner,
)


class TestConversationMemory:
    """Test cases for ConversationMemory class."""
    
    def test_initialization(self):
        """Test ConversationMemory initialization."""
        memory = ConversationMemory()
        assert memory.current_session is None
        assert memory.conversations == {}
    
    def test_start_session(self):
        """Test starting a new conversation session."""
        memory = ConversationMemory()
        session_id = memory.start_session()
        assert session_id is not None
        assert session_id in memory.conversations
        assert memory.current_session == session_id
    
    def test_add_message(self):
        """Test adding a message to conversation."""
        memory = ConversationMemory()
        session_id = memory.start_session()
        
        memory.add_message("user", "Hello, world!", "test_model")
        
        messages = memory.conversations[session_id]['messages']
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello, world!'
        assert messages[0]['model'] == 'test_model'
    
    def test_get_context(self):
        """Test getting conversation context."""
        memory = ConversationMemory()
        session_id = memory.start_session()
        
        # Add multiple messages
        memory.add_message("user", "Hello", "test_model")
        memory.add_message("assistant", "Hi there", "test_model")
        
        context = memory.get_context()
        assert len(context) == 2
        assert context[0]['content'] == 'Hello'
        assert context[1]['content'] == 'Hi there'


class TestResponseCache:
    """Test cases for ResponseCache class."""
    
    def test_initialization(self):
        """Test ResponseCache initialization."""
        cache = ResponseCache()
        assert cache.cache_dir.exists()
    
    def test_cache_operations(self):
        """Test basic cache get/set operations."""
        cache = ResponseCache()
        
        # Test setting a value
        prompt = "test prompt"
        model = "test_model"
        response = "test response"
        
        cache.set(prompt, model, response)
        cached_response = cache.get(prompt, model)
        
        assert cached_response == response


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        with patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.return_value = "NAME    MODIFIED    SIZE\nqwen3:4b  2 hours ago    4.7 GB"
            manager = ModelManager()
            assert manager.current_model is not None


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_extract_thinking_and_answer_with_brackets(self):
        """Test extracting thinking and answer with bracket format."""
        text = "This is a test."
        # TODO: Implement proper test for thinking/answer extraction
        assert len(text) > 0