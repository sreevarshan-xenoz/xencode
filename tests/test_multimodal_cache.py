#!/usr/bin/env python3
"""
Tests for Multi-Modal Cache System

Comprehensive test suite for the enhanced caching system including
document processing cache, code analysis cache, and workspace cache.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from xencode.cache.multimodal_cache import (
    MultiModalCacheSystem,
    DocumentProcessingCache,
    CodeAnalysisCache,
    WorkspaceCache,
    CacheWarmingManager,
    CacheType,
    MultiModalCacheEntry,
    get_multimodal_cache,
    initialize_multimodal_cache
)
from xencode.models.document import ProcessedDocument, DocumentType
from xencode.models.code_analysis import CodeAnalysisResult
from xencode.models.workspace import Change
from xencode.advanced_cache_system import HybridCacheManager


class TestMultiModalCacheEntry:
    """Test MultiModalCacheEntry functionality"""
    
    def test_cache_entry_creation(self):
        """Test creating a multimodal cache entry"""
        entry = MultiModalCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=100,
            compressed=False,
            cache_type=CacheType.DOCUMENT_PROCESSING,
            content_hash="abc123",
            dependencies={"file1.txt", "file2.txt"},
            invalidation_rules={"files": ["file1.txt"]},
            warming_priority=5
        )
        
        assert entry.cache_type == CacheType.DOCUMENT_PROCESSING
        assert entry.content_hash == "abc123"
        assert "file1.txt" in entry.dependencies
        assert entry.warming_priority == 5
    
    def test_should_invalidate_file_based(self):
        """Test file-based invalidation logic"""
        entry = MultiModalCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=100,
            compressed=False,
            cache_type=CacheType.DOCUMENT_PROCESSING,
            content_hash="abc123",
            invalidation_rules={"files": ["file1.txt", "file2.txt"]}
        )
        
        # Should invalidate when watched file changes
        change_event = {
            'type': 'file_changed',
            'files': ['file1.txt']
        }
        assert entry.should_invalidate(change_event)
        
        # Should not invalidate for unrelated files
        change_event = {
            'type': 'file_changed',
            'files': ['file3.txt']
        }
        assert not entry.should_invalidate(change_event)
    
    def test_should_invalidate_pattern_based(self):
        """Test pattern-based invalidation logic"""
        entry = MultiModalCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=100,
            compressed=False,
            cache_type=CacheType.CODE_ANALYSIS,
            content_hash="abc123",
            invalidation_rules={"patterns": [".py", ".js"]}
        )
        
        # Should invalidate for matching patterns
        change_event = {
            'type': 'file_changed',
            'files': ['script.py', 'other.txt']
        }
        assert entry.should_invalidate(change_event)
        
        # Should not invalidate for non-matching patterns
        change_event = {
            'type': 'file_changed',
            'files': ['document.txt', 'readme.md']
        }
        assert not entry.should_invalidate(change_event)
    
    def test_should_invalidate_dependency_based(self):
        """Test dependency-based invalidation logic"""
        entry = MultiModalCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=100,
            compressed=False,
            cache_type=CacheType.WORKSPACE_DATA,
            content_hash="abc123",
            dependencies={"config.json", "settings.yaml"}
        )
        
        # Should invalidate when dependency changes
        change_event = {
            'type': 'file_changed',
            'files': ['config.json']
        }
        assert entry.should_invalidate(change_event)
        
        # Should not invalidate for non-dependencies
        change_event = {
            'type': 'file_changed',
            'files': ['unrelated.txt']
        }
        assert not entry.should_invalidate(change_event)


class TestDocumentProcessingCache:
    """Test DocumentProcessingCache functionality"""
    
    @pytest.fixture
    def mock_base_cache(self):
        """Create mock base cache"""
        mock_cache = Mock(spec=HybridCacheManager)
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.put_enhanced = AsyncMock(return_value=True)
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.get_entry = AsyncMock(return_value=None)
        return mock_cache
    
    @pytest.fixture
    def document_cache(self, mock_base_cache):
        """Create DocumentProcessingCache instance"""
        return DocumentProcessingCache(mock_base_cache)
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_generate_document_key(self, document_cache, temp_file):
        """Test document cache key generation"""
        key1 = document_cache._generate_document_key(
            temp_file, DocumentType.PDF, {"option1": "value1"}
        )
        key2 = document_cache._generate_document_key(
            temp_file, DocumentType.PDF, {"option1": "value1"}
        )
        key3 = document_cache._generate_document_key(
            temp_file, DocumentType.DOCX, {"option1": "value1"}
        )
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different document type should generate different key
        assert key1 != key3
        
        # Key should be reasonable length
        assert len(key1) == 24
    
    def test_calculate_content_hash(self, document_cache, temp_file):
        """Test content hash calculation"""
        hash1 = document_cache._calculate_content_hash(temp_file)
        hash2 = document_cache._calculate_content_hash(temp_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16
        
        # Non-existent file should return "missing"
        hash3 = document_cache._calculate_content_hash("/nonexistent/file.txt")
        assert hash3 == "missing"
    
    @pytest.mark.asyncio
    async def test_get_processed_document_cache_miss(self, document_cache, temp_file):
        """Test getting document when not cached"""
        result = await document_cache.get_processed_document(
            temp_file, DocumentType.PDF
        )
        
        assert result is None
        document_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_processed_document_cache_hit(self, document_cache, temp_file):
        """Test getting document when cached"""
        # Mock cached document
        cached_doc = ProcessedDocument(
            id="test_id",
            original_filename="test.pdf",
            document_type=DocumentType.PDF,
            extracted_text="Test content",
            metadata={},
            processing_time_ms=100,
            confidence_score=0.95
        )
        
        document_cache.base_cache.get.return_value = cached_doc
        
        result = await document_cache.get_processed_document(
            temp_file, DocumentType.PDF
        )
        
        assert result == cached_doc
        document_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_processed_document(self, document_cache, temp_file):
        """Test caching processed document"""
        processed_doc = ProcessedDocument(
            id="test_id",
            original_filename="test.pdf",
            document_type=DocumentType.PDF,
            extracted_text="Test content",
            metadata={},
            processing_time_ms=100,
            confidence_score=0.95
        )
        
        success = await document_cache.cache_processed_document(
            temp_file, DocumentType.PDF, processed_doc
        )
        
        assert success
        document_cache.base_cache.put_enhanced.assert_called_once()
        
        # Check that content hash was stored
        cache_key = document_cache._generate_document_key(temp_file, DocumentType.PDF)
        assert cache_key in document_cache.document_hashes
    
    @pytest.mark.asyncio
    async def test_invalidate_document_cache(self, document_cache, temp_file):
        """Test document cache invalidation"""
        # Setup some cached entries
        document_cache.document_hashes["key1"] = "hash1"
        document_cache.document_hashes["key2"] = "hash2"
        
        # Mock entries with dependencies
        mock_entry1 = Mock()
        mock_entry1.dependencies = {temp_file}
        mock_entry2 = Mock()
        mock_entry2.dependencies = {"other_file.txt"}
        
        document_cache.base_cache.get_entry.side_effect = [mock_entry1, mock_entry2]
        
        invalidated = await document_cache.invalidate_document_cache(temp_file)
        
        assert invalidated == 1  # Only one entry should be invalidated
        document_cache.base_cache.delete.assert_called_once()


class TestCodeAnalysisCache:
    """Test CodeAnalysisCache functionality"""
    
    @pytest.fixture
    def mock_base_cache(self):
        """Create mock base cache"""
        mock_cache = Mock(spec=HybridCacheManager)
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.put_enhanced = AsyncMock(return_value=True)
        return mock_cache
    
    @pytest.fixture
    def code_cache(self, mock_base_cache):
        """Create CodeAnalysisCache instance"""
        return CodeAnalysisCache(mock_base_cache)
    
    def test_generate_analysis_key(self, code_cache):
        """Test analysis cache key generation"""
        code = "print('hello world')"
        
        key1 = code_cache._generate_analysis_key(code, "python", {"option1": "value1"})
        key2 = code_cache._generate_analysis_key(code, "python", {"option1": "value1"})
        key3 = code_cache._generate_analysis_key(code, "javascript", {"option1": "value1"})
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different language should generate different key
        assert key1 != key3
        
        # Key should be reasonable length
        assert len(key1) == 24
    
    @pytest.mark.asyncio
    async def test_get_analysis_result_cache_miss(self, code_cache):
        """Test getting analysis result when not cached"""
        result = await code_cache.get_analysis_result("print('hello')", "python")
        
        assert result is None
        code_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_analysis_result_cache_hit(self, code_cache):
        """Test getting analysis result when cached"""
        # Mock cached result
        cached_result = Mock(spec=CodeAnalysisResult)
        code_cache.base_cache.get.return_value = cached_result
        
        result = await code_cache.get_analysis_result("print('hello')", "python")
        
        assert result == cached_result
        code_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_analysis_result(self, code_cache):
        """Test caching analysis result"""
        code = "print('hello world')"
        analysis_result = Mock(spec=CodeAnalysisResult)
        
        success = await code_cache.cache_analysis_result(code, "python", analysis_result)
        
        assert success
        code_cache.base_cache.put_enhanced.assert_called_once()
        
        # Check that code hash was stored
        cache_key = code_cache._generate_analysis_key(code, "python")
        assert cache_key in code_cache.code_hashes


class TestWorkspaceCache:
    """Test WorkspaceCache functionality"""
    
    @pytest.fixture
    def mock_base_cache(self):
        """Create mock base cache"""
        mock_cache = Mock(spec=HybridCacheManager)
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.put_enhanced = AsyncMock(return_value=True)
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.delete_pattern = AsyncMock(return_value=2)
        mock_cache.get_all_entries = AsyncMock(return_value=[])
        return mock_cache
    
    @pytest.fixture
    def workspace_cache(self, mock_base_cache):
        """Create WorkspaceCache instance"""
        return WorkspaceCache(mock_base_cache)
    
    def test_generate_workspace_key(self, workspace_cache):
        """Test workspace cache key generation"""
        key = workspace_cache._generate_workspace_key("workspace123", "files")
        
        assert key == "workspace_workspace123_files"
    
    @pytest.mark.asyncio
    async def test_get_workspace_data_cache_miss(self, workspace_cache):
        """Test getting workspace data when not cached"""
        result = await workspace_cache.get_workspace_data("workspace123", "files")
        
        assert result is None
        workspace_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_workspace_data_cache_hit(self, workspace_cache):
        """Test getting workspace data when cached"""
        cached_data = {"files": ["file1.txt", "file2.txt"]}
        workspace_cache.base_cache.get.return_value = cached_data
        
        result = await workspace_cache.get_workspace_data("workspace123", "files")
        
        assert result == cached_data
        workspace_cache.base_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_workspace_data(self, workspace_cache):
        """Test caching workspace data"""
        data = {"files": ["file1.txt", "file2.txt"]}
        
        success = await workspace_cache.cache_workspace_data(
            "workspace123", "files", data, version=1
        )
        
        assert success
        workspace_cache.base_cache.put_enhanced.assert_called_once()
        
        # Check that version was stored
        cache_key = workspace_cache._generate_workspace_key("workspace123", "files")
        assert workspace_cache.workspace_versions[cache_key] == 1
    
    @pytest.mark.asyncio
    async def test_invalidate_workspace_cache_no_change(self, workspace_cache):
        """Test workspace cache invalidation without specific change"""
        invalidated = await workspace_cache.invalidate_workspace_cache("workspace123")
        
        assert invalidated == 2  # Mock returns 2
        workspace_cache.base_cache.delete_pattern.assert_called_once_with("workspace_workspace123_")
    
    @pytest.mark.asyncio
    async def test_invalidate_workspace_cache_with_change(self, workspace_cache):
        """Test workspace cache invalidation with specific change"""
        # Mock change object
        change = Mock()
        change.path = "file1.txt"
        change.operation = "update"
        
        # Mock cache entry that should be invalidated
        mock_entry = MultiModalCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=100,
            compressed=False,
            cache_type=CacheType.WORKSPACE_DATA,
            content_hash="abc123",
            invalidation_rules={"workspace_changes": ["workspace123"]}
        )
        
        workspace_cache.base_cache.get_all_entries.return_value = [mock_entry]
        
        invalidated = await workspace_cache.invalidate_workspace_cache("workspace123", change)
        
        assert invalidated == 1
        workspace_cache.base_cache.delete.assert_called_once()


class TestCacheWarmingManager:
    """Test CacheWarmingManager functionality"""
    
    @pytest.fixture
    def mock_caches(self):
        """Create mock cache instances"""
        doc_cache = Mock(spec=DocumentProcessingCache)
        doc_cache.get_processed_document = AsyncMock(return_value=None)
        
        code_cache = Mock(spec=CodeAnalysisCache)
        code_cache.get_analysis_result = AsyncMock(return_value=None)
        
        workspace_cache = Mock(spec=WorkspaceCache)
        workspace_cache.get_workspace_data = AsyncMock(return_value=None)
        
        return doc_cache, code_cache, workspace_cache
    
    @pytest.fixture
    def warming_manager(self, mock_caches):
        """Create CacheWarmingManager instance"""
        doc_cache, code_cache, workspace_cache = mock_caches
        return CacheWarmingManager(doc_cache, code_cache, workspace_cache)
    
    @pytest.mark.asyncio
    async def test_add_warming_task(self, warming_manager):
        """Test adding warming tasks"""
        await warming_manager.add_warming_task("document", 5, file_path="test.pdf")
        await warming_manager.add_warming_task("workspace", 8, workspace_id="ws123")
        
        assert len(warming_manager.warming_queue) == 2
        
        # Higher priority should be first
        assert warming_manager.warming_queue[0][0] == 8  # workspace task
        assert warming_manager.warming_queue[1][0] == 5  # document task
    
    @pytest.mark.asyncio
    async def test_warm_frequently_accessed_documents(self, warming_manager, tmp_path):
        """Test warming frequently accessed documents"""
        # Create test files
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("PDF content")
        
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("DOCX content")
        
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("Unsupported content")
        
        file_paths = [str(pdf_file), str(docx_file), str(unsupported_file)]
        
        await warming_manager.warm_frequently_accessed_documents(file_paths)
        
        # Should add tasks for supported file types only
        assert len(warming_manager.warming_queue) == 2
        
        # Check task parameters
        tasks = {task[2]['file_path']: task[2] for _, _, task in warming_manager.warming_queue}
        assert str(pdf_file) in tasks
        assert str(docx_file) in tasks
        assert str(unsupported_file) not in tasks
    
    @pytest.mark.asyncio
    async def test_warm_workspace_data(self, warming_manager):
        """Test warming workspace data"""
        await warming_manager.warm_workspace_data("workspace123", ["files", "settings"])
        
        assert len(warming_manager.warming_queue) == 2
        
        # Check task parameters
        for priority, cache_type, params in warming_manager.warming_queue:
            assert cache_type == "workspace"
            assert params["workspace_id"] == "workspace123"
            assert params["data_type"] in ["files", "settings"]
    
    @pytest.mark.asyncio
    async def test_process_warming_queue(self, warming_manager):
        """Test processing warming queue"""
        # Add some tasks
        await warming_manager.add_warming_task("document", 5, file_path="test.pdf", document_type=DocumentType.PDF)
        await warming_manager.add_warming_task("workspace", 8, workspace_id="ws123", data_type="files")
        
        # Mock the warming methods
        warming_manager._warm_document = AsyncMock()
        warming_manager._warm_workspace = AsyncMock()
        
        await warming_manager.process_warming_queue(max_tasks=2)
        
        # Both tasks should be processed
        assert len(warming_manager.warming_queue) == 0
        warming_manager._warm_document.assert_called_once()
        warming_manager._warm_workspace.assert_called_once()


class TestMultiModalCacheSystem:
    """Test MultiModalCacheSystem integration"""
    
    @pytest.fixture
    def mock_base_cache(self):
        """Create mock base cache"""
        mock_cache = Mock(spec=HybridCacheManager)
        mock_cache.get_stats = AsyncMock(return_value={})
        mock_cache.get_entries_by_tag = AsyncMock(return_value=[])
        mock_cache.cleanup_expired = AsyncMock()
        mock_cache.optimize_memory = AsyncMock()
        return mock_cache
    
    @pytest.fixture
    def multimodal_cache(self, mock_base_cache):
        """Create MultiModalCacheSystem instance"""
        return MultiModalCacheSystem(mock_base_cache)
    
    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, multimodal_cache):
        """Test getting comprehensive cache statistics"""
        stats = await multimodal_cache.get_cache_statistics()
        
        assert 'base_cache' in stats
        assert 'multimodal' in stats
        assert 'cache_types' in stats
        
        # Check multimodal stats structure
        multimodal_stats = stats['multimodal']
        expected_keys = [
            'document_hits', 'document_misses', 'code_hits', 'code_misses',
            'workspace_hits', 'workspace_misses', 'invalidations', 'warming_tasks'
        ]
        for key in expected_keys:
            assert key in multimodal_stats
    
    @pytest.mark.asyncio
    async def test_handle_change_event_file_changed(self, multimodal_cache):
        """Test handling file change events"""
        # Mock document cache invalidation
        multimodal_cache.document_cache.invalidate_document_cache = AsyncMock(return_value=2)
        
        event = {
            'type': 'file_changed',
            'file_path': '/path/to/file.txt'
        }
        
        await multimodal_cache.handle_change_event(event)
        
        multimodal_cache.document_cache.invalidate_document_cache.assert_called_once_with('/path/to/file.txt')
        assert multimodal_cache.multimodal_stats['invalidations'] == 2
    
    @pytest.mark.asyncio
    async def test_handle_change_event_workspace_changed(self, multimodal_cache):
        """Test handling workspace change events"""
        # Mock workspace cache invalidation
        multimodal_cache.workspace_cache.invalidate_workspace_cache = AsyncMock(return_value=3)
        
        change = Mock()
        event = {
            'type': 'workspace_changed',
            'workspace_id': 'workspace123',
            'change': change
        }
        
        await multimodal_cache.handle_change_event(event)
        
        multimodal_cache.workspace_cache.invalidate_workspace_cache.assert_called_once_with('workspace123', change)
        assert multimodal_cache.multimodal_stats['invalidations'] == 3
    
    @pytest.mark.asyncio
    async def test_optimize_cache(self, multimodal_cache):
        """Test cache optimization"""
        await multimodal_cache.optimize_cache()
        
        multimodal_cache.base_cache.cleanup_expired.assert_called_once()
        multimodal_cache.base_cache.optimize_memory.assert_called_once()


class TestGlobalFunctions:
    """Test global functions and initialization"""
    
    def test_get_multimodal_cache(self):
        """Test getting global multimodal cache instance"""
        # Reset global instance
        import xencode.cache.multimodal_cache
        xencode.cache.multimodal_cache._multimodal_cache_system = None
        
        with patch('xencode.cache.multimodal_cache.get_cache_system') as mock_get_cache:
            mock_base_cache = Mock()
            mock_get_cache.return_value = mock_base_cache
            
            cache1 = get_multimodal_cache()
            cache2 = get_multimodal_cache()
            
            # Should return same instance
            assert cache1 is cache2
            assert isinstance(cache1, MultiModalCacheSystem)
    
    @pytest.mark.asyncio
    async def test_initialize_multimodal_cache(self):
        """Test multimodal cache initialization"""
        with patch('xencode.cache.multimodal_cache.get_multimodal_cache') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.start_background_warming = AsyncMock()
            mock_get_cache.return_value = mock_cache
            
            with patch('asyncio.create_task') as mock_create_task:
                result = await initialize_multimodal_cache()
                
                assert result == mock_cache
                mock_create_task.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])