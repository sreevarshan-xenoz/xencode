"""
Integration tests for Xencode
Tests the integration between different modules and components
"""
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from xencode.core.files import create_file, read_file, write_file, delete_file
from xencode.core.models import ModelManager, get_available_models, get_smart_default_model
from xencode.core.memory import ConversationMemory
from xencode.core.cache import ResponseCache
from xencode.testing.mock_services import get_mock_ollama_service, get_mock_filesystem


class TestFileModelIntegration:
    """Test integration between file operations and model management"""
    
    def test_file_operations_with_model_context(self, tmp_path):
        """Test file operations in the context of model interactions"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        content = "This is a test file for model processing."
        create_file(str(test_file), content)
        
        # Verify file was created
        assert test_file.exists()
        assert test_file.read_text() == content
        
        # Simulate a model processing the file content
        with patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.return_value = "NAME\nllama2:7b\nmistral:7b"
            
            # Get available models
            models = get_available_models()
            assert len(models) > 0
            
            # Process the file content with a mock model
            # In a real scenario, this would involve sending the content to a model
            processed_content = f"Processed by model: {content}"
            
            # Write processed content to a new file
            processed_file = tmp_path / "processed.txt"
            write_file(str(processed_file), processed_content)
            
            # Verify processed file was created
            assert processed_file.exists()
            assert processed_file.read_text() == processed_content


class TestModelMemoryIntegration:
    """Test integration between model management and conversation memory"""
    
    def test_model_memory_interaction(self):
        """Test how models and memory work together"""
        # Initialize components
        model_manager = ModelManager()
        memory = ConversationMemory()
        
        # Start a session
        session_id = memory.start_session()
        
        # Add a conversation to memory
        memory.add_message("user", "Hello, model!", "test-model")
        memory.add_message("assistant", "Hello, user!", "test-model")
        
        # Get context from memory
        context = memory.get_context()
        assert len(context) == 2
        assert context[0]['role'] == 'user'
        assert context[1]['role'] == 'assistant'
        
        # Simulate model processing with context
        # In a real scenario, this would involve using the context in model requests
        context_summary = f"Context has {len(context)} messages"
        
        # Add the summary to memory as well
        memory.add_message("system", context_summary, "test-model")
        
        # Verify all messages are in memory
        updated_context = memory.get_context()
        assert len(updated_context) == 3
        assert updated_context[2]['content'] == context_summary


class TestCacheModelIntegration:
    """Test integration between caching and model operations"""
    
    def test_cached_model_responses(self, tmp_path):
        """Test caching of model responses"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Simulate a model query
            prompt = "What is the capital of France?"
            model = "test-model"
            response = "The capital of France is Paris."
            
            # Check that response is not in cache initially
            cached_response = cache.get(prompt, model)
            assert cached_response is None
            
            # Add response to cache (simulating model response)
            cache.set(prompt, model, response)
            
            # Retrieve from cache
            cached_response = cache.get(prompt, model)
            assert cached_response == response
            
            # Test with different prompt (should not be cached)
            different_prompt = "What is the capital of Germany?"
            different_cached_response = cache.get(different_prompt, model)
            assert different_cached_response is None


class TestFullWorkflowIntegration:
    """Test a full workflow integrating all components"""
    
    def test_complete_workflow(self, tmp_path):
        """Test a complete workflow: file -> model -> memory -> cache -> file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all components
            cache = ResponseCache(cache_dir=Path(temp_dir))
            memory = ConversationMemory()
            model_manager = ModelManager()
            
            # Step 1: Create an input file
            input_file = tmp_path / "input.txt"
            input_content = "Summarize this document: Machine learning is a subset of artificial intelligence."
            create_file(str(input_file), input_content)
            
            # Verify file creation
            assert input_file.exists()
            assert read_file(str(input_file)) == input_content
            
            # Step 2: Simulate model processing
            # In a real scenario, this would involve actual model inference
            model_response = "Summary: The document explains that machine learning is part of AI."
            model_name = "test-model"
            
            # Step 3: Add interaction to memory
            memory.start_session()
            memory.add_message("user", input_content, model_name)
            memory.add_message("assistant", model_response, model_name)
            
            # Verify memory contains the conversation
            context = memory.get_context()
            assert len(context) == 2
            
            # Step 4: Cache the response
            cache.set(input_content, model_name, model_response)
            
            # Verify caching worked
            cached = cache.get(input_content, model_name)
            assert cached == model_response
            
            # Step 5: Write result to output file
            output_file = tmp_path / "output.txt"
            write_file(str(output_file), model_response)
            
            # Verify output file
            assert output_file.exists()
            assert read_file(str(output_file)) == model_response
            
            # Step 6: Verify all components worked together
            # Check that memory, cache, and file operations all functioned correctly
            assert len(memory.get_context()) == 2
            assert cache.get(input_content, model_name) == model_response
            assert output_file.read_text() == model_response


class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""
    
    def test_error_propagation(self, tmp_path):
        """Test how errors propagate across integrated components"""
        # Test file operations with invalid paths
        invalid_path = "/invalid/path/file.txt"
        
        # This should handle the error gracefully
        result = delete_file(invalid_path)
        assert result is False  # Should return False on failure
        
        # Test cache with problematic inputs
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # These should handle errors gracefully
            result = cache.get(None, "model")  # None prompt
            assert result is None
            
            result = cache.get("prompt", None)  # None model
            assert result is None
            
            # Valid inputs should work normally
            cache.set("valid_prompt", "valid_model", "valid_response")
            result = cache.get("valid_prompt", "valid_model")
            assert result == "valid_response"


class TestPerformanceIntegration:
    """Test performance of integrated components"""
    
    def test_multiple_operations_performance(self, tmp_path):
        """Test performance when multiple components operate together"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            cache = ResponseCache(cache_dir=Path(temp_dir), max_size=50)
            memory = ConversationMemory(max_items=20)
            model_manager = ModelManager()
            
            # Perform multiple operations
            num_operations = 10
            
            for i in range(num_operations):
                # Create file
                file_path = tmp_path / f"test_{i}.txt"
                content = f"This is test file number {i}"
                create_file(str(file_path), content)
                
                # Add to memory
                memory.add_message("user", f"Question {i}", f"model_{i}")
                memory.add_message("assistant", f"Answer {i}", f"model_{i}")
                
                # Add to cache
                cache.set(f"prompt_{i}", f"model_{i}", f"response_{i}")
                
                # Verify operations worked
                assert file_path.exists()
                assert len(memory.get_context()) > 0
                assert cache.get(f"prompt_{i}", f"model_{i}") == f"response_{i}"
            
            # Verify final state
            assert len(list(tmp_path.glob("*.txt"))) == num_operations
            assert len(memory.get_context()) <= 20  # Limited by max_items
            # Cache size should be limited by max_size, but with only 10 items it should be fine


class TestMockIntegration:
    """Test integration using mock services"""
    
    def test_with_mock_ollama(self):
        """Test integration using mock Ollama service"""
        mock_service = get_mock_ollama_service()
        
        # Set up mock response
        mock_service.set_mock_response("test-model", "test prompt", "mock response")
        
        # Simulate using the mock service
        response = mock_service.generate("test-model", "test prompt")
        
        # Verify the response
        assert "mock response" in response['response']
        
        # Verify API call was logged
        assert len(mock_service.api_call_log) > 0
        assert mock_service.api_call_log[0]['params']['prompt'] == "test prompt"
    
    def test_with_mock_filesystem(self, tmp_path):
        """Test integration using mock filesystem"""
        mock_fs = get_mock_filesystem()
        
        # Set up mock file
        file_path = str(tmp_path / "mock_test.txt")
        content = "Mock file content"
        mock_fs.set_file_content(file_path, content)
        
        # Verify file exists in mock
        assert mock_fs.exists(file_path)
        
        # Read content from mock
        read_content = mock_fs.read_file(file_path)
        assert read_content == content


if __name__ == "__main__":
    # Run the integration tests
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    pytest.main([__file__, "-v"])