"""
Property-based tests for Xencode
Uses Hypothesis for property-based testing to test invariants
"""
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
import tempfile
from pathlib import Path
import time
from xencode.core.files import create_file, read_file, write_file, delete_file
from xencode.core.memory import ConversationMemory
from xencode.core.cache import ResponseCache
from xencode.core.models import ModelManager
from xencode.security.validation import InputValidator, validate_prompt, sanitize_prompt
import string


class TestFileOperationsProperties:
    """Property-based tests for file operations"""
    
    @given(
        file_content=st.text(min_size=1, max_size=1000),
        file_name=st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=50)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_file_create_read_consistency(self, tmp_path, file_content, file_name):
        """Test that content written to a file can be read back consistently"""
        file_path = tmp_path / f"{file_name}.txt"
        
        # Create file with content
        create_file(str(file_path), file_content)
        
        # Verify file exists
        assert file_path.exists()
        
        # Read the content back
        read_content = read_file(str(file_path))
        
        # The read content should match what was written
        assert read_content == file_content
    
    @given(
        original_content=st.text(min_size=1, max_size=500),
        new_content=st.text(min_size=1, max_size=500)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_file_write_overwrites_content(self, tmp_path, original_content, new_content):
        """Test that writing to a file overwrites the original content"""
        file_path = tmp_path / "test.txt"
        
        # Create file with original content
        create_file(str(file_path), original_content)
        assert read_file(str(file_path)) == original_content
        
        # Write new content
        write_file(str(file_path), new_content)
        
        # Verify the content has been overwritten
        assert read_file(str(file_path)) == new_content
    
    @given(
        file_content=st.text(min_size=1, max_size=100),
        file_name=st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_file_delete_removes_file(self, tmp_path, file_content, file_name):
        """Test that deleting a file removes it from the filesystem"""
        file_path = tmp_path / f"{file_name}.txt"
        
        # Create file
        create_file(str(file_path), file_content)
        assert file_path.exists()
        
        # Delete file
        result = delete_file(str(file_path))
        
        # Verify deletion
        assert result is True
        assert not file_path.exists()


class TestMemoryProperties:
    """Property-based tests for conversation memory"""
    
    @given(
        messages=st.lists(
            st.tuples(
                st.sampled_from(['user', 'assistant', 'system']),
                st.text(min_size=1, max_size=200),
                st.text(alphabet=string.ascii_letters + string.digits + '-_', min_size=1, max_size=20)
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_memory_preserves_added_messages(self, messages):
        """Test that messages added to memory are preserved when retrieved"""
        memory = ConversationMemory(max_items=50)  # High limit to accommodate all messages
        
        # Add all messages
        for role, content, model in messages:
            memory.add_message(role, content, model)
        
        # Get the context
        context = memory.get_context()
        
        # Verify all messages are present (might be limited by max_messages in get_context)
        # At least the last messages should be present
        expected_messages = messages[-10:]  # Last 10 messages (or all if fewer)
        
        # Check that the context contains the expected messages
        for i, (expected_role, expected_content, expected_model) in enumerate(expected_messages):
            if i < len(context):
                actual_msg = context[i]
                assert actual_msg['role'] == expected_role
                assert actual_msg['content'] == expected_content
                assert actual_msg['model'] == expected_model
    
    def test_memory_session_isolation(self):
        """Test that different sessions are isolated from each other"""
        memory = ConversationMemory()
        
        # Start session 1 and add messages
        session1_id = memory.start_session("session1")
        memory.add_message("user", "Hello from session 1", "model1")
        memory.add_message("assistant", "Response to session 1", "model1")
        
        # Start session 2 and add messages
        session2_id = memory.start_session("session2")
        memory.add_message("user", "Hello from session 2", "model2")
        memory.add_message("assistant", "Response to session 2", "model2")
        
        # Switch back to session 1 and verify it has its own messages
        memory.switch_session("session1")
        session1_context = memory.get_context()
        assert len(session1_context) == 2
        assert session1_context[0]['content'] == "Hello from session 1"
        
        # Switch to session 2 and verify it has its own messages
        memory.switch_session("session2")
        session2_context = memory.get_context()
        assert len(session2_context) == 2
        assert session2_context[0]['content'] == "Hello from session 2"


class TestCacheProperties:
    """Property-based tests for caching"""
    
    @given(
        prompt=st.text(min_size=1, max_size=500),
        model=st.text(alphabet=string.ascii_letters + string.digits + '-_', min_size=1, max_size=50),
        response=st.text(min_size=1, max_size=1000)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_store_retrieve_consistency(self, tmp_path, prompt, model, response):
        """Test that storing and retrieving from cache is consistent"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Store in cache
            cache.set(prompt, model, response)
            
            # Retrieve from cache
            retrieved = cache.get(prompt, model)
            
            # Should match
            assert retrieved == response
    
    @given(
        prompt=st.text(min_size=1, max_size=100),
        model=st.text(alphabet=string.ascii_letters + string.digits + '-_', min_size=1, max_size=20),
        response1=st.text(min_size=1, max_size=200),
        response2=st.text(min_size=1, max_size=200)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_overwrites_previous_value(self, tmp_path, prompt, model, response1, response2):
        """Test that caching a new value for the same key overwrites the previous value"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Store first response
            cache.set(prompt, model, response1)
            first_retrieved = cache.get(prompt, model)
            assert first_retrieved == response1
            
            # Store second response
            cache.set(prompt, model, response2)
            second_retrieved = cache.get(prompt, model)
            
            # Should get the second response
            assert second_retrieved == response2
            assert second_retrieved != response1


class TestValidationProperties:
    """Property-based tests for input validation and sanitization"""
    
    @given(
        prompt=st.text(min_size=1, max_size=1000)
    )
    def test_validate_prompt_never_crashes(self, prompt):
        """Test that validate_prompt never crashes regardless of input"""
        try:
            result = validate_prompt(prompt)
            # Result should be boolean
            assert isinstance(result, bool)
        except Exception:
            # If it raises an exception, that's a bug
            assert False, "validate_prompt should not raise exceptions"
    
    @given(
        prompt=st.text(min_size=1, max_size=1000)
    )
    def test_sanitize_prompt_always_returns_string(self, prompt):
        """Test that sanitize_prompt always returns a string"""
        result = sanitize_prompt(prompt)
        assert isinstance(result, str)
        # Result should be at least as valid as input (doesn't crash)
        assert isinstance(result, str)


class TestModelManagerProperties:
    """Property-based tests for ModelManager"""
    
    def test_model_manager_initialization_consistency(self):
        """Test that ModelManager initializes consistently"""
        # This test verifies that ModelManager can be initialized without crashing
        # In a real scenario with Ollama running, we'd test more properties
        manager = ModelManager()
        
        # Basic properties should be initialized
        assert isinstance(manager.available_models, list)
        assert isinstance(manager.model_health, dict)
        # Current model might be None if no models are available
        assert manager.current_model is None or isinstance(manager.current_model, str)
    
    @given(
        model_name=st.text(alphabet=string.ascii_letters + string.digits + '_-:.', min_size=1, max_size=50)
    )
    def test_model_name_validation_properties(self, model_name):
        """Test properties of model name validation"""
        # Import the validation function
        from xencode.security.validation import validate_model_name
        
        result = validate_model_name(model_name)
        # Result should always be boolean
        assert isinstance(result, bool)


class StatefulCacheMachine(RuleBasedStateMachine):
    """Stateful test for cache operations"""
    
    cache_storage = Bundle("cache")
    
    @initialize(target=cache_storage)
    def setup_cache(self):
        """Initialize the cache"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        self.temp_dir = temp_dir
        return ResponseCache(cache_dir=Path(temp_dir))
    
    @rule(
        target=cache_storage,
        cache=cache_storage,
        prompt=st.text(min_size=1, max_size=100),
        model=st.text(alphabet=string.ascii_letters + string.digits + '_-', min_size=1, max_size=20),
        response=st.text(min_size=1, max_size=200)
    )
    def store_in_cache(self, cache, prompt, model, response):
        """Store a value in the cache"""
        cache.set(prompt, model, response)
        # Store the value for later verification
        if not hasattr(self, 'stored_values'):
            self.stored_values = {}
        self.stored_values[(prompt, model)] = response
        return cache
    
    @rule(
        cache=cache_storage,
        prompt=st.text(min_size=1, max_size=100),
        model=st.text(alphabet=string.ascii_letters + string.digits + '_-', min_size=1, max_size=20)
    )
    def verify_cache_retrieval(self, cache, prompt, model):
        """Verify that cached values can be retrieved"""
        retrieved = cache.get(prompt, model)
        if hasattr(self, 'stored_values') and (prompt, model) in self.stored_values:
            expected = self.stored_values[(prompt, model)]
            assert retrieved == expected
        # If not stored, should return None
        else:
            # We can't assert it's None because it might be an expired entry
            # Just verify it doesn't crash
            assert retrieved is None or isinstance(retrieved, str)
    
    def teardown(self):
        """Clean up temporary directory"""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Create the test method from the stateful test
TestStatefulCache = StatefulCacheMachine.TestCase


class StatefulMemoryMachine(RuleBasedStateMachine):
    """Stateful test for memory operations"""
    
    memories = Bundle("memories")
    
    @initialize(target=memories)
    def setup_memory(self):
        """Initialize conversation memory"""
        return ConversationMemory(max_items=100)  # High limit for testing
    
    @rule(
        target=memories,
        memory=memories,
        role=st.sampled_from(['user', 'assistant', 'system']),
        content=st.text(min_size=1, max_size=100),
        model=st.text(alphabet=string.ascii_letters + string.digits + '_-', min_size=1, max_size=20)
    )
    def add_message(self, memory, role, content, model):
        """Add a message to memory"""
        memory.add_message(role, content, model)
        return memory
    
    @rule(
        memory=memories,
        max_messages=st.integers(min_value=1, max_value=10)
    )
    def verify_context_size(self, memory, max_messages):
        """Verify that context size respects the limit"""
        context = memory.get_context(max_messages=max_messages)
        # Context should not exceed the requested max_messages
        assert len(context) <= max_messages


# Create the test method from the stateful test
TestStatefulMemory = StatefulMemoryMachine.TestCase


def test_basic_properties_directly():
    """Direct property tests that don't fit the hypothesis pattern well"""
    
    # Test that memory limits work
    memory = ConversationMemory(max_items=3)
    
    # Add more messages than the limit
    for i in range(5):
        memory.add_message("user", f"Message {i}", "test-model")
    
    # Context should be limited
    context = memory.get_context()
    assert len(context) <= 3  # Respects the memory limit
    
    # Test cache size limiting
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir), max_size=2)
        
        # Add more items than the limit
        for i in range(5):
            cache.set(f"prompt_{i}", "model", f"response_{i}")
        
        # The cache size should be limited
        size = cache.get_cache_size()
        assert size <= 2


if __name__ == "__main__":
    # Run property-based tests
    import pytest
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run specific tests
    test_instance = TestFileOperationsProperties()
    
    # Example of running one test manually
    from hypothesis import example
    
    # This would normally be run via pytest
    print("Property-based tests defined. Run with pytest to execute.")