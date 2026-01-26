"""
Unit tests for Xencode core functionality
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from xencode.core.files import create_file, read_file, write_file, delete_file
from xencode.core.memory import ConversationMemory
from xencode.core.cache import ResponseCache
from xencode.core.models import ModelManager, get_available_models
from xencode.security.validation import InputValidator, sanitize_user_input
from xencode.security.api_validation import APIResponseValidator


class TestFilesModule(unittest.TestCase):
    """Test cases for the files module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = Path(self.test_dir) / "test.txt"
        self.test_content = "This is a test file."
    
    def test_create_file(self):
        """Test file creation"""
        create_file(self.test_file, self.test_content)
        self.assertTrue(self.test_file.exists())
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, self.test_content)
    
    def test_read_file(self):
        """Test file reading"""
        # Create a test file first
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        # Read the file
        content = read_file(self.test_file)
        self.assertEqual(content, self.test_content)
    
    def test_write_file(self):
        """Test file writing"""
        write_file(self.test_file, self.test_content)
        self.assertTrue(self.test_file.exists())
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, self.test_content)
    
    def test_delete_file(self):
        """Test file deletion"""
        # Create a test file first
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        # Delete the file
        result = delete_file(self.test_file)
        self.assertTrue(result)
        self.assertFalse(self.test_file.exists())


class TestMemoryModule(unittest.TestCase):
    """Test cases for the memory module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory = ConversationMemory(max_items=5)
        # Start a fresh session for each test
        self.memory.start_session(f"test_session_{id(self)}")
    
    def test_add_message(self):
        """Test adding a message to memory"""
        self.memory.add_message("user", "Hello, world!", "test-model")
        context = self.memory.get_context()
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["role"], "user")
        self.assertEqual(context[0]["content"], "Hello, world!")
        self.assertEqual(context[0]["model"], "test-model")
    
    def test_get_context(self):
        """Test getting conversation context"""
        # Add multiple messages
        for i in range(10):
            self.memory.add_message("user", f"Message {i}", "test-model")
        
        # Get context with limited messages
        context = self.memory.get_context(max_messages=5)
        self.assertEqual(len(context), 5)
        
        # Get context with unlimited messages (within memory limits)
        context = self.memory.get_context(max_messages=20)
        self.assertEqual(len(context), 5)  # Should be limited by memory size
    
    def test_session_management(self):
        """Test session creation and switching"""
        session1 = self.memory.start_session("session1")
        self.assertEqual(session1, "session1")
        
        session2 = self.memory.start_session()
        self.assertIsNotNone(session2)
        self.assertNotEqual(session1, session2)
        
        # Add messages to different sessions
        self.memory.add_message("user", "Session 1 message", "model1")
        self.memory.switch_session("session2")
        self.memory.add_message("user", "Session 2 message", "model2")
        
        # Switch back and verify contexts
        self.memory.switch_session("session1")
        context1 = self.memory.get_context()
        self.assertEqual(len(context1), 1)
        self.assertEqual(context1[0]["content"], "Session 1 message")
        
        self.memory.switch_session("session2")
        context2 = self.memory.get_context()
        self.assertEqual(len(context2), 1)
        self.assertEqual(context2[0]["content"], "Session 2 message")


class TestCacheModule(unittest.TestCase):
    """Test cases for the cache module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache_dir = Path(tempfile.mkdtemp())
        self.cache = ResponseCache(cache_dir=self.cache_dir, max_size=10)
    
    def test_cache_set_get(self):
        """Test setting and getting cached responses"""
        prompt = "What is the capital of France?"
        model = "test-model"
        response = "The capital of France is Paris."
        
        # Set cache
        self.cache.set(prompt, model, response)
        
        # Get cache
        cached_response = self.cache.get(prompt, model)
        self.assertEqual(cached_response, response)
    
    def test_cache_miss(self):
        """Test cache miss scenario"""
        cached_response = self.cache.get("non-existent prompt", "test-model")
        self.assertIsNone(cached_response)
    
    def test_cache_expiration(self):
        """Test cache expiration (simulated by modifying timestamp)"""
        prompt = "Test prompt"
        model = "test-model"
        response = "Test response"
        
        # Set cache
        self.cache.set(prompt, model, response)
        
        # Modify the cache file's timestamp to simulate expiration
        cache_key = self.cache._get_cache_key(prompt, model)
        cache_file = self.cache.cache_dir / f"{cache_key}.json"
        
        # Read the cache file, modify timestamp to expired, and write back
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Set timestamp to 2 days ago (expired)
        data['timestamp'] = data['timestamp'] - (2 * 24 * 3600)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        # Try to get the expired cache
        cached_response = self.cache.get(prompt, model)
        self.assertIsNone(cached_response)


class TestModelsModule(unittest.TestCase):
    """Test cases for the models module"""
    
    @patch('subprocess.check_output')
    def test_get_available_models(self, mock_subprocess):
        """Test getting available models"""
        # Mock the subprocess output
        mock_subprocess.return_value = "NAME                           ID       SIZE  \nllama3.1:8b                    abc123   4.7 GB\nmistral:7b                      def456   5.3 GB\n"
        
        models = get_available_models()
        self.assertIn("llama3.1:8b", models)
        self.assertIn("mistral:7b", models)
    
    @patch('subprocess.check_output')
    @patch('requests.post')
    def test_model_manager(self, mock_requests_post, mock_subprocess):
        """Test model manager functionality"""
        # Mock subprocess output
        mock_subprocess.return_value = "NAME                           ID       SIZE  \nmistral:7b                      def456   5.3 GB\n"
        
        # Mock requests response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "hi"}
        mock_requests_post.return_value = mock_response
        
        manager = ModelManager()
        self.assertIn("mistral:7b", manager.available_models)
        
        # Test model health check
        is_healthy = manager.check_model_health("mistral:7b")
        self.assertTrue(is_healthy)
        
        # Test getting best model
        best_model = manager.get_best_model()
        self.assertEqual(best_model, "mistral:7b")


class TestSecurityValidation(unittest.TestCase):
    """Test cases for security validation"""
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        # Test normal input
        normal_input = "This is a normal input."
        sanitized = sanitize_user_input(normal_input)
        self.assertEqual(normal_input, sanitized)
        
        # Test dangerous input
        dangerous_input = "This contains rm -rf / dangerous command"
        sanitized = sanitize_user_input(dangerous_input)
        self.assertNotEqual(dangerous_input, sanitized)
        self.assertIn("[FILTERED]", sanitized)
    
    def test_file_path_validation(self):
        """Test file path validation"""
        validator = InputValidator()
        
        # Valid paths
        self.assertTrue(validator.validate_file_path("./valid_file.txt"))
        self.assertTrue(validator.validate_file_path("../valid_file.txt"))
        
        # Invalid paths (attempting directory traversal to system directories)
        self.assertFalse(validator.validate_file_path("../../../../etc/passwd"))
    
    def test_url_validation(self):
        """Test URL validation"""
        validator = InputValidator()
        
        # Valid URLs
        self.assertTrue(validator.validate_url("https://example.com"))
        self.assertTrue(validator.validate_url("http://example.org/path"))
        
        # Invalid URLs
        self.assertFalse(validator.validate_url("ftp://example.com"))  # Wrong scheme
        self.assertFalse(validator.validate_url("http://192.168.1.1"))  # Private IP
        self.assertFalse(validator.validate_url("http://localhost"))  # Localhost


class TestAPIValidation(unittest.TestCase):
    """Test cases for API response validation"""
    
    def test_ollama_response_validation(self):
        """Test Ollama response validation"""
        validator = APIResponseValidator()
        
        # Valid streaming response
        valid_streaming = {"response": "test response", "done": False}
        self.assertTrue(validator.validate_ollama_response(valid_streaming))
        
        # Valid non-streaming response
        valid_non_streaming = {"response": "test response"}
        self.assertTrue(validator.validate_ollama_response(valid_non_streaming))
        
        # Invalid response
        invalid_response = {"other": "data"}
        self.assertFalse(validator.validate_ollama_response(invalid_response))
    
    def test_model_list_validation(self):
        """Test model list response validation"""
        validator = APIResponseValidator()
        
        # Valid model list response
        valid_response = {"models": [{"name": "model1"}, {"name": "model2"}]}
        self.assertTrue(validator.validate_model_list_response(valid_response))
        
        # Invalid model list response
        invalid_response = {"models": [{"id": "model1"}]}  # Missing name
        self.assertFalse(validator.validate_model_list_response(invalid_response))
        
        # Another invalid response
        invalid_response2 = {"other": "data"}
        self.assertFalse(validator.validate_model_list_response(invalid_response2))


if __name__ == '__main__':
    unittest.main()