"""
Unit tests for xencode core modules
"""
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add the parent directory to the path so we can import xencode modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from xencode.core.files import create_file, read_file, write_file, delete_file
from xencode.core.models import ModelManager, get_available_models, get_smart_default_model, list_models, update_model
from xencode.core.memory import ConversationMemory
from xencode.core.cache import ResponseCache, CacheStats, CacheLevel, LRUCacheItem, InMemoryCache
from xencode.core.connection_pool import ConnectionPool, AsyncConnectionPool, APIClient


class TestFilesModule:
    """Test cases for the files module"""
    
    def test_create_file(self, tmp_path):
        """Test creating a file"""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!"
        
        # Since create_file prints to console, we'll just check if file is created
        create_file(str(test_file), content)
        
        assert test_file.exists()
        assert test_file.read_text() == content
    
    def test_read_file(self, tmp_path):
        """Test reading a file"""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content)
        
        # Since read_file prints to console, we'll just check if it executes without error
        result = read_file(str(test_file))
        assert result == content
    
    def test_write_file(self, tmp_path):
        """Test writing to a file"""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!"
        
        write_file(str(test_file), content)
        
        assert test_file.exists()
        assert test_file.read_text() == content
    
    def test_delete_file(self, tmp_path):
        """Test deleting a file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        assert test_file.exists()
        
        result = delete_file(str(test_file))
        
        assert not test_file.exists()
        assert result is True


class TestModelsModule:
    """Test cases for the models module"""
    
    @patch('subprocess.check_output')
    def test_get_available_models(self, mock_subprocess):
        """Test getting available models"""
        mock_subprocess.return_value = "NAME\nllama2:7b\nmistral:7b"
        
        models = get_available_models()
        
        assert "llama2:7b" in models
        assert "mistral:7b" in models
    
    def test_model_manager_initialization(self):
        """Test initializing ModelManager"""
        manager = ModelManager()
        
        assert isinstance(manager.available_models, list)
        assert manager.current_model is None
    
    def test_model_manager_refresh_models(self):
        """Test refreshing models in ModelManager"""
        manager = ModelManager()
        
        # Initially empty
        assert len(manager.available_models) == 0
        
        # Mock the subprocess call
        with patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.return_value = "NAME\nllama2:7b\nmistral:7b"
            manager.refresh_models()
            
            assert len(manager.available_models) == 2
            assert "llama2:7b" in manager.available_models
            assert "mistral:7b" in manager.available_models


class TestMemoryModule:
    """Test cases for the memory module"""
    
    def test_conversation_memory_initialization(self):
        """Test initializing ConversationMemory"""
        memory = ConversationMemory()
        
        assert memory.max_items == 50  # Default value
        assert isinstance(memory.conversations, dict)
        assert memory.current_session is None
    
    def test_conversation_memory_custom_max_items(self):
        """Test initializing ConversationMemory with custom max items"""
        memory = ConversationMemory(max_items=10)
        
        assert memory.max_items == 10
    
    def test_start_session(self):
        """Test starting a new session"""
        memory = ConversationMemory()
        
        session_id = memory.start_session()
        
        assert session_id is not None
        assert memory.current_session == session_id
        assert session_id in memory.conversations
    
    def test_add_message(self):
        """Test adding a message to a session"""
        memory = ConversationMemory()
        
        # Add a message
        memory.add_message("user", "Hello", "test-model")
        
        # Check that a session was created
        assert memory.current_session is not None
        assert memory.current_session in memory.conversations
        
        # Check that the message was added
        messages = memory.conversations[memory.current_session]['messages']
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello'
        assert messages[0]['model'] == 'test-model'
    
    def test_get_context(self):
        """Test getting context from a session"""
        memory = ConversationMemory()
        
        # Add some messages
        memory.add_message("user", "Hello", "test-model")
        memory.add_message("assistant", "Hi there!", "test-model")
        
        # Get context
        context = memory.get_context()
        
        assert len(context) == 2
        assert context[0]['role'] == 'user'
        assert context[1]['role'] == 'assistant'
    
    def test_list_sessions(self):
        """Test listing sessions"""
        memory = ConversationMemory()
        
        # Start a few sessions
        session1 = memory.start_session("session1")
        session2 = memory.start_session("session2")
        
        sessions = memory.list_sessions()
        
        assert session1 in sessions
        assert session2 in sessions
        assert len(sessions) == 2
    
    def test_switch_session(self):
        """Test switching between sessions"""
        memory = ConversationMemory()
        
        # Start sessions
        session1 = memory.start_session("session1")
        session2 = memory.start_session("session2")
        
        # Add a message to session1
        memory.add_message("user", "Message in session 1", "test-model")
        
        # Switch to session2
        result = memory.switch_session("session2")
        
        assert result is True
        assert memory.current_session == "session2"
        
        # Switch back to session1
        result = memory.switch_session("session1")
        
        assert result is True
        assert memory.current_session == "session1"
        
        # Verify message is still in session1
        context = memory.get_context()
        assert len(context) == 1
        assert context[0]['content'] == 'Message in session 1'


class TestCacheModule:
    """Test cases for the cache module"""
    
    def test_response_cache_initialization(self):
        """Test initializing ResponseCache"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            assert cache.cache_dir == Path(temp_dir)
            assert cache.max_size == 100  # Default value
            assert cache.ttl_seconds == 86400  # Default value
            assert isinstance(cache.memory_cache, InMemoryCache)
    
    def test_cache_get_set(self):
        """Test getting and setting values in cache"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir), max_size=10, ttl_seconds=3600)
            
            # Initially should return None
            result = cache.get("test_prompt", "test_model")
            assert result is None
            
            # Set a value
            cache.set("test_prompt", "test_model", "test_response")
            
            # Get the value back
            result = cache.get("test_prompt", "test_model")
            assert result == "test_response"
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a very short TTL for testing
            cache = ResponseCache(cache_dir=Path(temp_dir), ttl_seconds=0.001)  # 1ms TTL
            
            # Set a value
            cache.set("test_prompt", "test_model", "test_response")
            
            # Wait for expiration
            import time
            time.sleep(0.01)  # Sleep for 10ms
            
            # Should return None due to expiration
            result = cache.get("test_prompt", "test_model")
            assert result is None
    
    def test_in_memory_cache(self):
        """Test the in-memory cache"""
        cache = InMemoryCache(capacity=2)
        
        # Add items
        cache.put("key1", "value1", "model1")
        cache.put("key2", "value2", "model2")
        
        # Get items
        item1 = cache.get("key1")
        assert item1 is not None
        assert item1.value == "value1"
        assert item1.model == "model1"
        
        item2 = cache.get("key2")
        assert item2 is not None
        assert item2.value == "value2"
        assert item2.model == "model2"
        
        # Add another item to trigger eviction
        cache.put("key3", "value3", "model3")
        
        # key1 should still exist (was accessed recently), key2 might be evicted
        item1_after = cache.get("key1")
        assert item1_after is not None
        
        # Get stats
        stats = cache.get_stats()
        assert isinstance(stats, CacheStats)
    
    def test_lru_cache_item(self):
        """Test LRUCacheItem"""
        import time
        timestamp = time.time()
        item = LRUCacheItem("test_key", "test_value", "test_model", timestamp)
        
        assert item.key == "test_key"
        assert item.value == "test_value"
        assert item.model == "test_model"
        assert item.timestamp == timestamp
        assert item.access_count == 1
        assert item.last_access == timestamp


class TestConnectionPoolModule:
    """Test cases for the connection pool module"""
    
    def test_connection_pool_initialization(self):
        """Test initializing ConnectionPool"""
        pool = ConnectionPool(max_connections=5, max_retries=2)
        
        assert pool.max_connections == 5
        assert pool.session is not None
    
    def test_get_connection(self):
        """Test getting a connection from the pool"""
        pool = ConnectionPool()
        
        connection = pool.get_connection()
        
        # Should return a requests.Session object
        assert hasattr(connection, 'get')
        assert hasattr(connection, 'post')
    
    def test_async_connection_pool_initialization(self):
        """Test initializing AsyncConnectionPool"""
        pool = AsyncConnectionPool(max_connections=10, max_keepalive_connections=5)
        
        assert pool.max_connections == 10
        assert pool.max_keepalive_connections == 5
    
    @pytest.mark.asyncio
    async def test_async_connection_pool_get_session(self):
        """Test getting a session from async connection pool"""
        pool = AsyncConnectionPool()
        
        session = await pool.get_session()
        
        # Should return an aiohttp.ClientSession object
        assert hasattr(session, 'get')
        assert hasattr(session, 'post')
    
    def test_api_client_initialization(self):
        """Test initializing APIClient"""
        client = APIClient(base_url="http://test.com", max_connections=5)
        
        assert client.base_url == "http://test.com"
        assert client.sync_pool is not None
        assert client.async_pool is not None


if __name__ == "__main__":
    pytest.main([__file__])