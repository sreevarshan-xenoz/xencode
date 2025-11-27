"""Basic tests for xencode_core module."""

import sys
from pathlib import Path
import pytest
# from unittest.mock import patch - Removed for real data testing

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xencode_core import (
    ConversationMemory,
    ModelManager,
    ResponseCache,
)


class TestConversationMemory:
    """Test cases for ConversationMemory class."""

    def test_initialization(self):
        """Test ConversationMemory initialization."""
        memory = ConversationMemory()
        # Reset to ensure clean state for testing
        memory.conversations = {}
        memory.current_session = None
        assert memory.current_session is None
        assert memory.conversations == {}

    def test_start_session(self):
        """Test starting a new conversation session."""
        memory = ConversationMemory()
        # Reset to ensure clean state for testing
        memory.conversations = {}
        memory.current_session = None
        session_id = memory.start_session()
        assert session_id is not None
        assert session_id in memory.conversations
        assert memory.current_session == session_id

    def test_add_message(self):
        """Test adding a message to conversation."""
        memory = ConversationMemory()
        # Reset to ensure clean state for testing
        memory.conversations = {}
        memory.current_session = None
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
        # Reset to ensure clean state for testing
        memory.conversations = {}
        memory.current_session = None
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

    @pytest.mark.requires_ollama
    def test_initialization(self):
        """Test ModelManager initialization with real system."""
        try:
            manager = ModelManager()
            # With real system, this should populate if Ollama is running and has models
            # We can't guarantee models are pulled, but we can check the list structure
            assert isinstance(manager.available_models, list)
        except Exception as e:
            pytest.fail(f"ModelManager initialization failed with real system: {e}")

    @pytest.mark.requires_ollama
    def test_refresh_models(self):
        """Test refreshing models from real Ollama instance."""
        manager = ModelManager()
        manager.refresh_models()
        assert isinstance(manager.available_models, list)



class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_extract_thinking_and_answer_with_brackets(self):
        """Test extracting thinking and answer with bracket format."""
        text = "This is a test."
        # TODO: Implement proper test for thinking/answer extraction
        assert len(text) > 0
