#!/usr/bin/env python3
"""
Tests for Phase 3 Intelligence Layer

Tests for:
- Context Indexer v2 (S4-01)
- Prompt Router (S4-02)
- Session Summarizer (S4-03)
- Project Profiles (S4-04)

Note: Some tests are skipped if heavy ML dependencies (torch, transformers) are not available
to avoid Windows compatibility issues.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add xencode to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all needed classes for tests
from xencode.core.project_profiles import (  # noqa: E402
    ModelProfile,
    ProjectProfileManager,
)
from xencode.memory.session_summarizer import (  # noqa: E402
    ImportanceLevel,
    MemorySummarizer,
)
from xencode.rag.context_indexer_v2 import (  # noqa: E402
    ContextIndexerV2,
    FileMetadata,
    IndexManifest,
    IndexStatus,
)
from xencode.routing.prompt_router import (  # noqa: E402
    PromptRouter,
    ProviderType,
    RoutingPolicy,
    TaskClassifier,
    TaskType,
)


class TestContextIndexerV2:
    """Tests for Context Indexer v2"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        # Import here to avoid top-level import issues
        try:
            from xencode.rag.context_indexer_v2 import (
                ContextIndexerV2,
                FileMetadata,
                IndexManifest,
                IndexStatus,
                Symbol,
            )
            self.ContextIndexerV2 = ContextIndexerV2
            self.IndexStatus = IndexStatus
            self.FileMetadata = FileMetadata
            self.Symbol = Symbol
            self.IndexManifest = IndexManifest
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    @pytest.mark.skipif(False, reason="Basic tests always run")
    def test_content_hash_calculation(self):
        """Test content hash calculation"""
        if not self.has_deps:
            pytest.skip("Dependencies not available")

        indexer = self.ContextIndexerV2()
        content = "test content"
        hash1 = indexer._calculate_content_hash(content)
        hash2 = indexer._calculate_content_hash(content)
        hash3 = indexer._calculate_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 produces 64 char hex

    def test_token_estimation(self):
        """Test token estimation"""
        indexer = ContextIndexerV2()
        text = "This is a test sentence with some words."
        estimated = indexer._estimate_tokens(text)

        # Rough approximation: 1 token ≈ 4 chars
        assert estimated > 0
        assert estimated <= len(text)

    def test_file_metadata_serialization(self):
        """Test FileMetadata serialization/deserialization"""
        metadata = FileMetadata(
            path="test.py",
            size=1024,
            modified_time=datetime.now().timestamp(),
            content_hash="abc123",
            indexed_time=datetime.now(),
            symbol_count=5,
            line_count=50,
            token_estimate=200,
            status=IndexStatus.COMPLETED,
        )

        # Serialize
        data = metadata.to_dict()
        assert data['path'] == "test.py"
        assert data['symbol_count'] == 5

        # Deserialize
        restored = FileMetadata.from_dict(data)
        assert restored.path == metadata.path
        assert restored.symbol_count == metadata.symbol_count
        assert restored.status == metadata.status

    def test_symbol_extraction_basic(self):
        """Test basic symbol extraction from Python code"""
        indexer = ContextIndexerV2()

        test_code = """
class TestClass:
    '''Test class docstring'''

    def test_method(self):
        pass

def test_function():
    '''Test function'''
    return 42

import os
from typing import List
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()

            symbols = indexer._extract_symbols(Path(f.name), test_code)

        # Clean up after closing the file handle (Windows keeps open files locked)
        os.unlink(f.name)

        # Check symbols extracted
        assert len(symbols) > 0

        symbol_types = [s.type for s in symbols]
        assert 'class' in symbol_types
        assert 'function' in symbol_types
        assert 'import' in symbol_types

    def test_index_manifest_serialization(self):
        """Test IndexManifest serialization"""

        manifest = IndexManifest(
            project_root="/test/project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_tokens=1000,
        )

        # Add a file
        manifest.files["test.py"] = FileMetadata(
            path="test.py",
            size=100,
            modified_time=datetime.now().timestamp(),
            content_hash="xyz789",
            indexed_time=datetime.now(),
        )

        # Serialize
        data = manifest.to_dict()
        assert data['project_root'] == "/test/project"
        assert 'test.py' in data['files']

        # Deserialize
        restored = IndexManifest.from_dict(data)
        assert restored.project_root == manifest.project_root
        assert 'test.py' in restored.files


class TestPromptRouter:
    """Tests for Prompt Router"""

    def test_task_classification_code_generation(self):
        """Test task classification for code generation"""
        classifier = TaskClassifier()

        prompt = "Write a Python function to sort a list"
        classification = classifier.classify(prompt)

        assert classification.task_type == TaskType.CODE_GENERATION
        assert classification.confidence > 0
        assert classification.requires_coding is True

    def test_task_classification_debugging(self):
        """Test task classification for debugging"""
        classifier = TaskClassifier()

        prompt = "Why is my code throwing a TypeError exception?"
        classification = classifier.classify(prompt)

        assert classification.task_type == TaskType.DEBUGGING
        assert classification.requires_reasoning is True

    def test_task_classification_code_review(self):
        """Test task classification for code review"""
        classifier = TaskClassifier()

        prompt = "Review this code for security vulnerabilities"
        classification = classifier.classify(prompt)

        assert classification.task_type == TaskType.CODE_REVIEW

    def test_task_classification_chat(self):
        """Test task classification for chat"""
        classifier = TaskClassifier()

        prompt = "Hello, how are you today?"
        classification = classifier.classify(prompt)

        assert classification.task_type == TaskType.CHAT

    def test_language_detection(self):
        """Test programming language detection"""
        classifier = TaskClassifier()

        # Python
        prompt_py = "Write a Python function with def keyword"
        lang_py = classifier._detect_language(prompt_py)
        assert lang_py == 'python'

        # JavaScript
        prompt_js = "Create a JavaScript function with => arrow syntax"
        lang_js = classifier._detect_language(prompt_js)
        assert lang_js == 'javascript'

    def test_complexity_detection(self):
        """Test complexity level detection"""
        classifier = TaskClassifier()

        # High complexity
        prompt_high = "Design a complex microservice architecture with multiple layers"
        complexity_high = classifier._detect_complexity(prompt_high)
        assert complexity_high == 'high'

        # Low complexity
        prompt_low = "Write a simple hello world example"
        complexity_low = classifier._detect_complexity(prompt_low)
        assert complexity_low == 'low'

    def test_routing_decision_basic(self):
        """Test basic routing decision"""
        router = PromptRouter()

        prompt = "Write a Python function"
        decision = router.route(prompt)

        assert decision.provider is not None
        assert decision.model is not None
        assert decision.reason is not None

    def test_routing_decision_coding_task(self):
        """Test routing for coding task"""
        router = PromptRouter()

        prompt = "Implement a REST API endpoint in Flask"
        decision = router.route(prompt)

        # Should route to a code-capable model
        assert decision.provider in [ProviderType.CLOUD_QWEN, ProviderType.LOCAL_OLLAMA]

    def test_routing_policy_application(self):
        """Test routing policy application"""
        router = PromptRouter()

        # Add custom policy
        custom_policy = RoutingPolicy(
            name="test_policy",
            priority=1,
            task_types=[TaskType.CODE_GENERATION],
            preferred_provider=ProviderType.LOCAL_OLLAMA,
            preferred_model="qwen2.5:7b",
        )
        router.add_policy(custom_policy)

        # Verify policy was added
        retrieved = router.get_policy("test_policy")
        assert retrieved is not None
        assert retrieved.preferred_model == "qwen2.5:7b"

    def test_provider_health_tracking(self):
        """Test provider health status updates"""
        router = PromptRouter()

        # Update health status
        router.update_provider_health(ProviderType.CLOUD_QWEN, 'unhealthy', 'Connection timeout')

        health = router.provider_health[ProviderType.CLOUD_QWEN]
        assert health['status'] == 'unhealthy'
        assert health['last_error'] == 'Connection timeout'


class TestSessionSummarizer:
    """Tests for Session Summarizer"""

    def test_add_turn_basic(self):
        """Test adding basic turn"""
        summarizer = MemorySummarizer(session_id="test_session")

        turn_id = summarizer.add_turn(
            user_message="Write a function",
            assistant_message="Here's a function...",
            model_used="qwen2.5:7b",
            tokens_used=100,
        )

        assert turn_id == "turn_1"
        assert len(summarizer.turns) == 1
        assert summarizer.total_tokens == 100

    def test_add_turn_importance_assessment(self):
        """Test importance assessment for turns"""
        summarizer = MemorySummarizer(session_id="test_session")

        # Regular turn
        summarizer.add_turn(
            user_message="What's 2+2?",
            assistant_message="4",
        )
        turn1 = summarizer.turns[0]
        assert turn1.importance == ImportanceLevel.MEDIUM

        # Important turn
        summarizer.add_turn(
            user_message="This is critical for the architecture decision",
            assistant_message="Important decision...",
        )
        turn2 = summarizer.turns[1]
        assert turn2.importance.value >= ImportanceLevel.HIGH.value

    def test_pin_memory(self):
        """Test pinning important memory"""
        summarizer = MemorySummarizer(session_id="test_session")

        pin_id = summarizer.pin_memory(
            content="User prefers Python 3.10+",
            category="user_preference",
        )

        assert pin_id == "pin_1"
        assert len(summarizer.pinned_memories) == 1
        assert pin_id in summarizer.pinned_memories

        pinned = summarizer.pinned_memories[pin_id]
        assert pinned.category == "user_preference"
        assert "Python" in pinned.content

    def test_topic_detection(self):
        """Test topic detection from messages"""
        summarizer = MemorySummarizer(session_id="test_session")

        # Code generation topic
        topic = summarizer._detect_topic("Write a Python function to sort")
        assert topic == 'code_generation'

        # Debugging topic
        topic = summarizer._detect_topic("Fix this error exception bug")
        assert topic == 'debugging'

    def test_get_context(self):
        """Test context retrieval"""
        summarizer = MemorySummarizer(
            session_id="test_session",
            summary_threshold=5,
        )

        # Add some turns
        for i in range(3):
            summarizer.add_turn(
                user_message=f"Question {i}",
                assistant_message=f"Answer {i}",
                tokens_used=50,
            )

        context = summarizer.get_context()

        assert 'pinned_memories' in context
        assert 'recent_turns' in context
        assert 'section_summaries' in context
        assert len(context['recent_turns']) == 3

    def test_session_stats(self):
        """Test session statistics"""
        summarizer = MemorySummarizer(session_id="test_session")

        # Add turns
        for i in range(5):
            summarizer.add_turn(
                user_message=f"Question {i}",
                assistant_message=f"Answer {i}",
                tokens_used=100,
            )

        stats = summarizer.get_stats()

        assert stats['total_turns'] == 5
        assert stats['total_tokens'] == 500
        assert 'context_window_usage' in stats


class TestProjectProfiles:
    """Tests for Project Profiles"""

    def test_model_profile_serialization(self):
        """Test ModelProfile serialization"""
        profile = ModelProfile(
            name="test_profile",
            default_model="qwen2.5:7b",
            code_model="qwen3-coder-next-instruct",
            cost_budget=20.0,
            local_first=True,
        )

        # Serialize
        data = profile.to_dict()
        assert data['name'] == "test_profile"
        assert data['code_model'] == "qwen3-coder-next-instruct"

        # Deserialize
        restored = ModelProfile.from_dict(data)
        assert restored.name == profile.name
        assert restored.code_model == profile.code_model

    def test_get_model_for_task(self):
        """Test getting model for task type"""
        profile = ModelProfile(name="test")

        # Code tasks
        assert profile.get_model_for_task('code') == profile.code_model
        assert profile.get_model_for_task('code_generation') == profile.code_model

        # Chat tasks
        assert profile.get_model_for_task('chat') == profile.chat_model

        # Reasoning tasks
        assert profile.get_model_for_task('reasoning') == profile.reasoning_model

        # Default
        assert profile.get_model_for_task('unknown') == profile.default_model

    def test_profile_manager_default_profiles(self):
        """Test default profiles availability"""
        manager = ProjectProfileManager()

        # Check default profiles exist
        assert 'default' in manager.DEFAULT_PROFILES
        assert 'cloud_optimized' in manager.DEFAULT_PROFILES
        assert 'local_only' in manager.DEFAULT_PROFILES
        assert 'high_performance' in manager.DEFAULT_PROFILES

    def test_profile_manager_get_profile(self):
        """Test getting profiles"""
        manager = ProjectProfileManager()

        # Get default profile
        profile = manager.get_profile('default')
        assert profile.name == 'default'

        # Get non-existent profile (should fall back to default)
        profile = manager.get_profile('nonexistent')
        assert profile.name == 'default'

    def test_profile_manager_config_summary(self):
        """Test configuration summary"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProjectProfileManager(project_root=Path(tmpdir))

            manager.create_default_config(profile_name='test')
            manager.load_config()

            summary = manager.get_config_summary()

            assert summary['status'] == 'loaded'
            assert 'profile_name' in summary
            assert 'default_model' in summary

    def test_environment_variable_substitution(self):
        """Test environment variable substitution"""
        manager = ProjectProfileManager()

        # Set test env var
        os.environ['TEST_API_KEY'] = 'test_key_123'

        # Test substitution
        test_value = "${TEST_API_KEY}"
        result = manager._substitute_env_vars(test_value)
        assert result == 'test_key_123'

        # Test in dict
        test_dict = {'api_key': '${TEST_API_KEY}', 'other': 'value'}
        result_dict = manager._substitute_env_vars(test_dict)
        assert result_dict['api_key'] == 'test_key_123'
        assert result_dict['other'] == 'value'

    def test_create_default_config(self):
        """Test creating default configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manager = ProjectProfileManager(project_root=tmp_path)

            # Create config
            config = manager.create_default_config(profile_name='default')

            assert config is not None
            assert config.profile.name == 'default'
            assert manager.config_path.exists()

            # Load it back
            loaded = manager.load_config()
            assert loaded is not None
            assert loaded.profile.name == 'default'


class TestIntegration:
    """Integration tests for Phase 3 components"""

    def test_context_indexing_and_search(self):
        """Test context indexing and search workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test file
            test_file = tmp_path / "test.py"
            test_file.write_text("""
def hello():
    '''Say hello'''
    print("Hello, World!")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
""")

            # Create indexer
            indexer = ContextIndexerV2(
                persist_directory=str(tmp_path / ".index")
            )

            # Index directory
            stats = indexer.index_directory(str(tmp_path), incremental=False, verbose=False)

            assert stats['indexed'] > 0
            assert stats['total_files'] > 0

            # Get stats
            index_stats = indexer.get_stats()
            assert index_stats['total_files'] > 0

            # Close the vector store client so Windows can release the DB file
            client = getattr(indexer.vector_store, 'client', None)
            if client is not None:
                client.close()

    def test_routing_with_context(self):
        """Test routing with context information"""
        router = PromptRouter()

        # Route with context
        context = {'file_type': '.py', 'has_error': False}
        decision = router.route(
            prompt="Write a sorting function",
            context=context,
        )

        assert decision.model is not None
        assert 'code' in decision.reason.lower() or decision.policy_applied

    def test_session_with_summarization(self):
        """Test session with automatic summarization"""
        summarizer = MemorySummarizer(
            session_id="integration_test",
            summary_threshold=3,  # Low threshold for testing
        )

        # Add turns to trigger summarization
        for i in range(5):
            summarizer.add_turn(
                user_message=f"Task {i}",
                assistant_message=f"Solution {i}",
                tokens_used=100,
            )

        # Check that consolidation happened
        stats = summarizer.get_stats()
        assert stats['total_turns'] <= 3  # Should have consolidated


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
