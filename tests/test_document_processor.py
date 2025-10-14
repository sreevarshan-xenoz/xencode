#!/usr/bin/env python3
"""
Tests for Document Processor Base Architecture

Tests the document processor, data models, and processing pipeline.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from xencode.document_processor import DocumentProcessor, ProcessorInterface, FallbackHandler
from xencode.models.document import (
    DocumentType,
    ProcessedDocument,
    ProcessingOptions,
    ProcessingResult,
    ProcessingStatus,
    ContentType,
    StructuredContent,
    DocumentMetadata,
    detect_document_type,
    is_supported_document_type,
    estimate_processing_time
)


class MockProcessor(ProcessorInterface):
    """Mock processor for testing"""
    
    def __init__(self, supported_types: list = None):
        self.supported_types = supported_types or [DocumentType.TEXT]
        self.should_fail = False
        self.processing_delay = 0.0
    
    async def can_process(self, file_path: Path, document_type: DocumentType) -> bool:
        return document_type in self.supported_types
    
    async def process(self, file_path: Path, options: ProcessingOptions) -> ProcessedDocument:
        if self.processing_delay > 0:
            await asyncio.sleep(self.processing_delay)
        
        if self.should_fail:
            raise Exception("Mock processor failure")
        
        document = ProcessedDocument(
            original_filename=file_path.name,
            file_path=str(file_path),
            document_type=DocumentType.TEXT,
            extracted_text="Mock extracted text",
            processing_status=ProcessingStatus.COMPLETED,
            confidence_score=0.9
        )
        
        document.add_structured_content(
            ContentType.TEXT,
            "Mock structured content"
        )
        
        return document
    
    def get_supported_types(self) -> list:
        return self.supported_types


class TestDocumentModels:
    """Test document data models"""
    
    def test_document_type_detection(self):
        """Test document type detection from file extensions"""
        test_cases = [
            ('document.pdf', DocumentType.PDF),
            ('file.docx', DocumentType.DOCX),
            ('page.html', DocumentType.HTML),
            ('readme.md', DocumentType.MARKDOWN),
            ('script.py', DocumentType.CODE),
            ('data.txt', DocumentType.TEXT),
            ('unknown.xyz', DocumentType.UNKNOWN)
        ]
        
        for filename, expected_type in test_cases:
            assert detect_document_type(filename) == expected_type
    
    def test_supported_document_types(self):
        """Test supported document type checking"""
        supported = [
            DocumentType.PDF,
            DocumentType.DOCX,
            DocumentType.HTML,
            DocumentType.MARKDOWN,
            DocumentType.TEXT,
            DocumentType.CODE
        ]
        
        for doc_type in supported:
            assert is_supported_document_type(doc_type)
        
        assert not is_supported_document_type(DocumentType.UNKNOWN)
    
    def test_processing_time_estimation(self):
        """Test processing time estimation"""
        # 1MB file
        file_size = 1024 * 1024
        
        # PDF should take longer than text
        pdf_time = estimate_processing_time(file_size, DocumentType.PDF)
        text_time = estimate_processing_time(file_size, DocumentType.TEXT)
        
        assert pdf_time > text_time
        assert pdf_time > 0
        assert text_time > 0
    
    def test_processed_document_creation(self):
        """Test ProcessedDocument creation and methods"""
        doc = ProcessedDocument(
            original_filename="test.txt",
            document_type=DocumentType.TEXT,
            extracted_text="Test content"
        )
        
        # Test adding structured content
        doc.add_structured_content(
            ContentType.HEADING,
            "Test Heading",
            position={'page': 1, 'line': 1}
        )
        
        assert len(doc.structured_content) == 1
        assert doc.structured_content[0].content_type == ContentType.HEADING
        
        # Test getting content by type
        headings = doc.get_content_by_type(ContentType.HEADING)
        assert len(headings) == 1
        assert headings[0].text == "Test Heading"
        
        # Test text content retrieval
        text_content = doc.get_text_content()
        assert text_content == "Test content"
    
    def test_document_serialization(self):
        """Test document to_dict and from_dict methods"""
        original_doc = ProcessedDocument(
            original_filename="test.txt",
            document_type=DocumentType.TEXT,
            extracted_text="Test content",
            confidence_score=0.8
        )
        
        original_doc.add_structured_content(
            ContentType.TEXT,
            "Structured text",
            confidence=0.9
        )
        
        # Convert to dict
        doc_dict = original_doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert doc_dict['original_filename'] == "test.txt"
        assert doc_dict['document_type'] == DocumentType.TEXT.value
        
        # Convert back from dict
        restored_doc = ProcessedDocument.from_dict(doc_dict)
        assert restored_doc.original_filename == original_doc.original_filename
        assert restored_doc.document_type == original_doc.document_type
        assert restored_doc.extracted_text == original_doc.extracted_text
        assert len(restored_doc.structured_content) == len(original_doc.structured_content)
    
    def test_processing_options(self):
        """Test ProcessingOptions"""
        options = ProcessingOptions(
            timeout_seconds=60,
            max_file_size_mb=50,
            extract_metadata=True
        )
        
        assert options.timeout_seconds == 60
        assert options.max_file_size_mb == 50
        assert options.extract_metadata is True
        
        # Test serialization
        options_dict = options.to_dict()
        assert isinstance(options_dict, dict)
        assert options_dict['timeout_seconds'] == 60


class TestFallbackHandler:
    """Test fallback processing handler"""
    
    @pytest.fixture
    def fallback_handler(self):
        return FallbackHandler()
    
    @pytest.mark.asyncio
    async def test_text_fallback(self, fallback_handler):
        """Test text fallback processing"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for fallback")
            temp_path = Path(f.name)
        
        try:
            document = await fallback_handler._text_fallback(temp_path)
            
            assert document.extracted_text == "Test content for fallback"
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert document.confidence_score == 0.5
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_handle_processing_error(self, fallback_handler):
        """Test error handling with fallback"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Fallback content")
            temp_path = Path(f.name)
        
        try:
            error = Exception("Processing failed")
            document = await fallback_handler.handle_processing_error(
                error, temp_path, DocumentType.TEXT
            )
            
            assert len(document.errors) > 0
            assert "Processing failed" in document.errors[0]
            assert document.extracted_text == "Fallback content"
            assert "Used fallback processing method" in document.warnings
            
        finally:
            temp_path.unlink()


class TestDocumentProcessor:
    """Test main document processor"""
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
    
    @pytest.fixture
    def mock_processor(self):
        return MockProcessor([DocumentType.TEXT])
    
    def test_processor_registration(self, processor, mock_processor):
        """Test processor registration"""
        processor.register_processor(mock_processor)
        
        registered = processor.get_registered_processors()
        assert DocumentType.TEXT in registered
        assert registered[DocumentType.TEXT] == mock_processor
    
    @pytest.mark.asyncio
    async def test_document_validation(self, processor):
        """Test document validation"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            validation = await processor.validate_document(temp_path)
            
            assert validation['document_type'] == DocumentType.TEXT
            assert validation['file_size'] > 0
            assert validation['supported'] is True
            assert validation['estimated_processing_time_ms'] > 0
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_document_processing_success(self, processor, mock_processor):
        """Test successful document processing"""
        processor.register_processor(mock_processor)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            result = await processor.process_document(temp_path)
            
            assert result.success is True
            assert result.document is not None
            assert result.document.extracted_text == "Mock extracted text"
            assert result.document.processing_status == ProcessingStatus.COMPLETED
            assert result.processing_time_ms > 0
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_document_processing_failure(self, processor, mock_processor):
        """Test document processing with failure"""
        mock_processor.should_fail = True
        processor.register_processor(mock_processor)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            result = await processor.process_document(temp_path)
            
            # Should succeed with fallback
            assert result.success is True
            assert result.document is not None
            assert len(result.document.warnings) > 0
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_document_processing_timeout(self, processor, mock_processor):
        """Test document processing timeout"""
        mock_processor.processing_delay = 2.0  # 2 second delay
        processor.register_processor(mock_processor)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            options = ProcessingOptions(timeout_seconds=1)  # 1 second timeout
            result = await processor.process_document(temp_path, options=options)
            
            assert result.success is False
            assert "timeout" in result.error_message.lower()
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_file_size_limit(self, processor, mock_processor):
        """Test file size limit enforcement"""
        processor.register_processor(mock_processor)
        
        # Create temporary file with content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            options = ProcessingOptions(max_file_size_mb=0.000001)  # Very small limit
            result = await processor.process_document(temp_path, options=options)
            
            assert result.success is False
            assert "too large" in result.error_message.lower()
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_multiple_document_processing(self, processor, mock_processor):
        """Test processing multiple documents"""
        processor.register_processor(mock_processor)
        
        # Create multiple temporary files
        temp_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Test content {i}")
                temp_paths.append(Path(f.name))
        
        try:
            results = await processor.process_multiple_documents(temp_paths)
            
            assert len(results) == 3
            for result in results:
                assert result.success is True
                assert result.document is not None
            
        finally:
            for path in temp_paths:
                path.unlink()
    
    def test_processing_stats(self, processor):
        """Test processing statistics"""
        initial_stats = processor.get_processing_stats()
        assert initial_stats['total_processed'] == 0
        assert initial_stats['success_rate'] == 0.0
        
        # Simulate some processing
        processor.processing_stats['total_processed'] = 10
        processor.processing_stats['successful'] = 8
        processor.processing_stats['failed'] = 2
        
        stats = processor.get_processing_stats()
        assert stats['total_processed'] == 10
        assert stats['success_rate'] == 0.8
        assert stats['failure_rate'] == 0.2
        
        # Test reset
        processor.reset_stats()
        reset_stats = processor.get_processing_stats()
        assert reset_stats['total_processed'] == 0
    
    @pytest.mark.asyncio
    async def test_unsupported_document_type(self, processor):
        """Test handling of unsupported document types"""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = await processor.process_document(temp_path)
            
            assert result.success is False
            assert "unsupported" in result.error_message.lower()
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, processor):
        """Test handling of nonexistent files"""
        nonexistent_path = Path("/nonexistent/file.txt")
        
        result = await processor.process_document(nonexistent_path)
        
        assert result.success is False
        assert "not found" in result.error_message.lower()


# Integration test
@pytest.mark.asyncio
async def test_document_processor_integration():
    """Test complete document processor workflow"""
    processor = DocumentProcessor()
    mock_processor = MockProcessor([DocumentType.TEXT, DocumentType.MARKDOWN])
    processor.register_processor(mock_processor)
    
    # Create test files
    test_files = []
    
    # Text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a text file")
        test_files.append(Path(f.name))
    
    # Markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# This is a markdown file")
        test_files.append(Path(f.name))
    
    try:
        # Process files
        results = await processor.process_multiple_documents(test_files)
        
        assert len(results) == 2
        for result in results:
            assert result.success is True
            assert result.document is not None
            assert result.document.confidence_score > 0
        
        # Check stats
        stats = processor.get_processing_stats()
        assert stats['total_processed'] == 2
        assert stats['successful'] == 2
        assert stats['success_rate'] == 1.0
        
    finally:
        for path in test_files:
            path.unlink()


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])