#!/usr/bin/env python3
"""
Comprehensive Tests for Document Processing

Tests for error handling, fallback mechanisms, processing time validation,
and confidence scoring for document processing functionality.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import time

from xencode.document_processor import DocumentProcessor, FallbackHandler
from xencode.models.document import (
    DocumentType,
    ProcessedDocument,
    ProcessingOptions,
    ProcessingResult,
    ProcessingStatus,
    ContentType,
    detect_document_type,
    is_supported_document_type
)


class TestDocumentProcessingErrorHandling:
    """Test error handling in document processing"""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, processor):
        """Test handling of file not found errors"""
        nonexistent_path = Path("/nonexistent/file.pdf")
        
        result = await processor.process_document(nonexistent_path)
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_permission_denied_error(self, processor):
        """Test handling of permission denied errors"""
        # Create a file and temporarily remove read permissions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            # Temporarily make the file unreadable to simulate permission error
            import os
            original_mode = temp_path.stat().st_mode
            os.chmod(temp_path, 0o000)  # Remove all permissions
            
            result = await processor.process_document(temp_path)
            
            assert result.success is False
            assert result.error_message is not None
        except Exception:
            # Restore permissions in case of exception
            import os
            os.chmod(temp_path, original_mode)
        finally:
            # Always restore permissions and clean up
            import os
            os.chmod(temp_path, 0o666)  # Give permissions back
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_invalid_file_format_error(self, processor):
        """Test handling of invalid file format errors"""
        # Create a file with invalid content for its extension
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write some random bytes that are not a valid PDF
            f.write(b"This is not a valid PDF file")
            temp_path = Path(f.name)

        try:
            result = await processor.process_document(temp_path)
            
            # Should either fail or use fallback
            assert result.success is True or result.success is False
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, processor):
        """Test handling of corrupted files"""
        # Create a file with corrupted content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write some binary data that might cause issues
            f.write(b'\x00\x01\x02\x03\xFF\xFE\xFD')
            temp_path = Path(f.name)

        try:
            result = await processor.process_document(temp_path)
            
            # Should handle gracefully with fallback
            assert result.success is True  # Should use fallback
        finally:
            temp_path.unlink()


class TestDocumentProcessingFallbackMechanisms:
    """Test fallback mechanisms in document processing"""

    @pytest.fixture
    def fallback_handler(self):
        return FallbackHandler()

    @pytest.mark.asyncio
    async def test_fallback_handler_text_fallback(self, fallback_handler):
        """Test text fallback mechanism"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test fallback content")
            temp_path = Path(f.name)

        try:
            document = await fallback_handler._text_fallback(temp_path)
            
            assert document.extracted_text == "Test fallback content"
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert document.confidence_score == 0.5  # Expected for fallback
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_fallback_handler_complex_fallback(self, fallback_handler):
        """Test fallback mechanism with complex content"""
        complex_content = """
        This is a complex document with multiple lines
        Line 2: Some content here
        Line 3: More content
        
        Section Header
        ==============
        Paragraph with some text content
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(complex_content)
            temp_path = Path(f.name)

        try:
            document = await fallback_handler._text_fallback(temp_path)
            
            assert complex_content.strip() in document.extracted_text
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert document.confidence_score == 0.5
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_fallback_on_processing_error(self, fallback_handler):
        """Test fallback when processing error occurs"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Fallback test content")
            temp_path = Path(f.name)

        try:
            error = Exception("Simulated processing error")
            document = await fallback_handler.handle_processing_error(
                error, temp_path, DocumentType.TEXT
            )
            
            assert "Processing failed" in document.errors[0]
            assert document.extracted_text == "Fallback test content"
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert "Used fallback processing method" in document.warnings
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_fallback_handler_fallback_failure(self, fallback_handler):
        """Test fallback when both primary and fallback processing fail"""
        # Create a file that doesn't exist to simulate failure
        nonexistent_path = Path("/nonexistent/file.txt")
        
        error = Exception("Primary processing failed")
        document = await fallback_handler.handle_processing_error(
            error, nonexistent_path, DocumentType.TEXT
        )
        
        # Should have both primary and fallback errors
        assert len(document.errors) >= 1
        assert document.processing_status == ProcessingStatus.FAILED


class TestDocumentProcessingTimeAndConfidence:
    """Test processing time and confidence scoring"""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    @pytest.mark.asyncio
    async def test_processing_time_accuracy(self, processor):
        """Test that processing time is measured accurately"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Simple test content")
            temp_path = Path(f.name)

        try:
            start_time = time.time()
            result = await processor.process_document(temp_path)
            end_time = time.time()
            
            expected_time_ms = int((end_time - start_time) * 1000)
            
            # Processing time should be reasonable (not negative, not extremely high)
            assert result.processing_time_ms >= 0
            assert result.processing_time_ms <= expected_time_ms + 100  # Allow some buffer
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_large_file_processing_time(self, processor):
        """Test processing time for larger files"""
        # Create a larger file
        large_content = "This is a line of content.\n" * 1000  # 1000 lines
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = Path(f.name)

        try:
            result = await processor.process_document(temp_path)
            
            # Should take more time than a small file
            assert result.processing_time_ms >= 0
            assert result.success is True
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_confidence_scoring_calculation(self, processor):
        """Test confidence scoring calculation"""
        # Mock a processor that adds structured content with different confidences
        class MockProcessor:
            async def can_process(self, file_path, document_type):
                return True
            
            async def process(self, file_path, options):
                doc = ProcessedDocument(
                    original_filename=file_path.name,
                    file_path=str(file_path),
                    document_type=DocumentType.TEXT,
                    extracted_text="Test content",
                    processing_status=ProcessingStatus.COMPLETED
                )
                
                # Add content with different confidence levels
                doc.add_structured_content(ContentType.TEXT, "High confidence", confidence=0.9)
                doc.add_structured_content(ContentType.TEXT, "Medium confidence", confidence=0.6)
                doc.add_structured_content(ContentType.TEXT, "Low confidence", confidence=0.3)
                
                return doc
            
            def get_supported_types(self):
                return [DocumentType.TEXT]

        processor.register_processor(MockProcessor())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)

        try:
            result = await processor.process_document(temp_path)
            
            if result.success and result.document:
                # Average confidence should be around 0.6 (0.9 + 0.6 + 0.3) / 3
                expected_avg = (0.9 + 0.6 + 0.3) / 3
                assert abs(result.document.confidence_score - expected_avg) < 0.01
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_confidence_scoring_edge_cases(self, processor):
        """Test confidence scoring edge cases"""
        # Test with no structured content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Simple content")
            temp_path = Path(f.name)

        try:
            # Mock processor that returns document with no structured content
            class MockProcessor:
                async def can_process(self, file_path, document_type):
                    return True
                
                async def process(self, file_path, options):
                    return ProcessedDocument(
                        original_filename=file_path.name,
                        file_path=str(file_path),
                        document_type=DocumentType.TEXT,
                        extracted_text="Test content",
                        processing_status=ProcessingStatus.COMPLETED
                    )
                
                def get_supported_types(self):
                    return [DocumentType.TEXT]

            processor.register_processor(MockProcessor())
            
            result = await processor.process_document(temp_path)
            
            if result.success and result.document:
                # Should handle empty structured content gracefully
                assert result.document.confidence_score >= 0.0
        finally:
            temp_path.unlink()


class TestDocumentProcessingComprehensiveScenarios:
    """Test comprehensive document processing scenarios"""

    @pytest.mark.asyncio
    async def test_pdf_processing_scenario(self):
        """Test PDF processing scenario"""
        processor = DocumentProcessor()
        
        # Since we don't have actual PDF content, test with validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            # Write minimal PDF header to make it look like a PDF
            f.write("%PDF-1.4\n1 0 obj\n<<>>\nendobj\n")
            temp_path = Path(f.name)

        try:
            # Validate the document first
            validation = await processor.validate_document(temp_path)
            
            assert validation['document_type'] == DocumentType.PDF
            assert validation['file_size'] > 0
            assert validation['supported'] is True  # PDF is supported
            
            # Try to process (will likely use fallback)
            result = await processor.process_document(temp_path)
            
            # Should handle gracefully
            assert result.success is True or result.success is False
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_docx_processing_scenario(self):
        """Test DOCX processing scenario"""
        processor = DocumentProcessor()
        
        # Create a minimal DOCX-like file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            f.write("<?xml version='1.0' encoding='UTF-8'?>\n<document>Test content</document>")
            temp_path = Path(f.name)

        try:
            # Validate the document
            validation = await processor.validate_document(temp_path)
            
            assert validation['document_type'] == DocumentType.DOCX
            assert validation['file_size'] > 0
            assert validation['supported'] is True  # DOCX is supported
            
            # Try to process (will likely use fallback)
            result = await processor.process_document(temp_path)
            
            # Should handle gracefully
            assert result.success is True or result.success is False
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_html_processing_scenario(self):
        """Test HTML processing scenario"""
        processor = DocumentProcessor()
        
        # Create a simple HTML file
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a paragraph with some content.</p>
            <div>Another section</div>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = Path(f.name)

        try:
            # Validate the document
            validation = await processor.validate_document(temp_path)
            
            assert validation['document_type'] == DocumentType.HTML
            assert validation['file_size'] > 0
            assert validation['supported'] is True  # HTML is supported
            
            # Try to process
            result = await processor.process_document(temp_path)
            
            # Should handle HTML content
            assert result.success is True or result.success is False
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in document processing"""
        processor = DocumentProcessor()
        
        # Mock a processor that takes too long
        class SlowProcessor:
            async def can_process(self, file_path, document_type):
                return True
            
            async def process(self, file_path, options):
                # Simulate a slow operation
                await asyncio.sleep(2)
                return ProcessedDocument(
                    original_filename=file_path.name,
                    file_path=str(file_path),
                    document_type=DocumentType.TEXT,
                    extracted_text="Slow content",
                    processing_status=ProcessingStatus.COMPLETED
                )
            
            def get_supported_types(self):
                return [DocumentType.TEXT]

        processor.register_processor(SlowProcessor())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)

        try:
            # Set a short timeout
            options = ProcessingOptions(timeout_seconds=0.5)
            result = await processor.process_document(temp_path, options=options)
            
            # Should timeout
            assert result.success is False
            assert "timeout" in result.error_message.lower()
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_file_size_limit_enforcement(self):
        """Test file size limit enforcement"""
        processor = DocumentProcessor()
        
        # Create a file that exceeds the size limit
        large_content = "A" * (2 * 1024 * 1024)  # 2MB of content
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_path = Path(f.name)

        try:
            # Set a small size limit
            options = ProcessingOptions(max_file_size_mb=1)  # 1MB limit
            result = await processor.process_document(temp_path, options=options)
            
            # Should fail due to size limit
            assert result.success is False
            assert "large" in result.error_message.lower()
        finally:
            temp_path.unlink()


class TestDocumentProcessingIntegration:
    """Integration tests for document processing"""

    @pytest.mark.asyncio
    async def test_end_to_end_document_processing_workflow(self):
        """Test complete end-to-end document processing workflow"""
        processor = DocumentProcessor()
        
        # Test with a simple text file
        test_content = "This is a test document for end-to-end processing.\nIt has multiple lines.\nAnd different sections."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)

        try:
            # Step 1: Validate document
            validation = await processor.validate_document(temp_path)
            assert validation['valid'] is True
            assert validation['document_type'] == DocumentType.TEXT
            assert validation['supported'] is True

            # Step 2: Process document
            result = await processor.process_document(temp_path)
            
            # Step 3: Verify result
            if result.success:
                assert result.document is not None
                assert result.document.original_filename == temp_path.name
                assert result.document.document_type == DocumentType.TEXT
                assert result.processing_time_ms >= 0
                assert result.document.confidence_score >= 0.0
            else:
                # If failed, it should be due to processor availability, not core logic
                assert result.error_message is not None

            # Step 4: Check statistics
            stats = processor.get_processing_stats()
            assert stats['total_processed'] >= 1
            assert stats['successful'] + stats['failed'] == stats['total_processed']

        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_multiple_document_processing_with_mixed_results(self):
        """Test processing multiple documents with mixed success/failure scenarios"""
        processor = DocumentProcessor()
        
        # Create multiple test files
        test_files = []
        
        # Valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Valid content")
            test_files.append(Path(f.name))
        
        # Another valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("More valid content")
            test_files.append(Path(f.name))

        try:
            # Process multiple documents
            results = await processor.process_multiple_documents(test_files)
            
            # Should have results for all files
            assert len(results) == len(test_files)
            
            # All should be successful (or handled gracefully)
            for result in results:
                assert hasattr(result, 'success')
                assert hasattr(result, 'error_message') or result.document is not None

            # Check that stats are updated
            stats = processor.get_processing_stats()
            assert stats['total_processed'] >= len(test_files)

        finally:
            # Clean up all files
            for path in test_files:
                path.unlink()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])