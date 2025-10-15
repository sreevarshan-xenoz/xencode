#!/usr/bin/env python3
"""
Document Processor Base Architecture

Provides the main DocumentProcessor class with type detection,
unified processing pipeline, and fallback mechanisms for multi-modal
document processing.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from xencode.models.document import (
    DocumentType,
    ProcessedDocument,
    ProcessingOptions,
    ProcessingResult,
    ProcessingStatus,
    detect_document_type,
    is_supported_document_type,
    estimate_processing_time
)
from xencode.processors.base import ProcessorInterface

# Import processors
try:
    from xencode.processors.pdf_processor import PDFProcessor
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from xencode.processors.docx_processor import DOCXProcessor
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from xencode.processors.web_extractor import WebContentExtractor
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

try:
    from xencode.processors.text_processor import TextProcessor
    TEXT_AVAILABLE = True
except ImportError:
    TEXT_AVAILABLE = False



class FallbackHandler:
    """Handles fallback processing for unsupported formats"""
    
    def __init__(self):
        self.fallback_strategies = {
            DocumentType.UNKNOWN: self._text_fallback,
            DocumentType.PDF: self._text_extraction_fallback,
            DocumentType.DOCX: self._text_extraction_fallback,
            DocumentType.HTML: self._text_extraction_fallback,
        }
    
    async def handle_processing_error(self, 
                                    error: Exception, 
                                    file_path: Path,
                                    document_type: DocumentType) -> ProcessedDocument:
        """Handle processing error with appropriate fallback"""
        
        # Create basic document with error information
        document = ProcessedDocument(
            original_filename=file_path.name,
            file_path=str(file_path),
            document_type=document_type,
            processing_status=ProcessingStatus.FAILED
        )
        
        document.errors.append(f"Processing failed: {str(error)}")
        
        # Try fallback strategy - use TEXT fallback for all types
        try:
            fallback_document = await self._text_fallback(file_path)
            # Merge fallback content with error document
            document.extracted_text = fallback_document.extracted_text
            document.structured_content = fallback_document.structured_content
            document.processing_status = ProcessingStatus.COMPLETED
            document.warnings.append("Used fallback processing method")
        except Exception as fallback_error:
            document.errors.append(f"Fallback processing failed: {str(fallback_error)}")
        
        return document
    
    async def _text_fallback(self, file_path: Path) -> ProcessedDocument:
        """Basic text extraction fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            document = ProcessedDocument(
                original_filename=file_path.name,
                file_path=str(file_path),
                document_type=DocumentType.TEXT,
                extracted_text=content,
                processing_status=ProcessingStatus.COMPLETED,
                confidence_score=0.5  # Lower confidence for fallback
            )
            
            return document
            
        except Exception as e:
            raise Exception(f"Text fallback failed: {e}")
    
    async def _text_extraction_fallback(self, file_path: Path) -> ProcessedDocument:
        """Advanced text extraction fallback using multiple methods"""
        # This would implement more sophisticated text extraction
        # For now, use basic text fallback
        return await self._text_fallback(file_path)


class DocumentProcessor:
    """Main document processor with unified processing pipeline"""
    
    def __init__(self):
        self.processors: Dict[DocumentType, ProcessorInterface] = {}
        self.fallback_handler = FallbackHandler()
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'fallback_used': 0
        }
        
        # Auto-register available processors
        self._register_available_processors()
    
    def _register_available_processors(self) -> None:
        """Register all available processors"""
        try:
            if PDF_AVAILABLE:
                pdf_processor = PDFProcessor()
                self.register_processor(pdf_processor)
        except Exception as e:
            print(f"Warning: Could not register PDF processor: {e}")
        
        try:
            if DOCX_AVAILABLE:
                docx_processor = DOCXProcessor()
                self.register_processor(docx_processor)
        except Exception as e:
            print(f"Warning: Could not register DOCX processor: {e}")
        
        try:
            if WEB_AVAILABLE:
                web_processor = WebContentExtractor()
                self.register_processor(web_processor)
        except Exception as e:
            print(f"Warning: Could not register Web processor: {e}")
        
        try:
            if TEXT_AVAILABLE:
                text_processor = TextProcessor()
                self.register_processor(text_processor)
        except Exception as e:
            print(f"Warning: Could not register Text processor: {e}")
    
    def register_processor(self, 
                          processor: ProcessorInterface,
                          document_types: Optional[List[DocumentType]] = None) -> None:
        """Register a processor for specific document types"""
        if document_types is None:
            document_types = processor.get_supported_types()
        
        for doc_type in document_types:
            self.processors[doc_type] = processor
    
    def get_registered_processors(self) -> Dict[DocumentType, ProcessorInterface]:
        """Get all registered processors"""
        return self.processors.copy()
    
    async def process_document(self, 
                             file_path: Union[str, Path],
                             document_type: Optional[DocumentType] = None,
                             options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """
        Process document with appropriate handler and fallback mechanisms
        
        Args:
            file_path: Path to the document file
            document_type: Optional document type (will be detected if not provided)
            options: Processing options
            
        Returns:
            ProcessingResult with processed document or error information
        """
        start_time = time.time()
        
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if options is None:
            options = ProcessingOptions()
        
        # Validate file exists
        if not file_path.exists():
            return ProcessingResult(
                success=False,
                error_message=f"File not found: {file_path}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Detect document type if not provided
        if document_type is None:
            document_type = detect_document_type(file_path)
        
        # Check file size limits
        file_size = file_path.stat().st_size
        max_size_bytes = options.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return ProcessingResult(
                success=False,
                error_message=f"File too large: {file_size} bytes (max: {max_size_bytes})",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Check if document type is supported
        if not is_supported_document_type(document_type):
            return ProcessingResult(
                success=False,
                error_message=f"Unsupported document type: {document_type}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        self.processing_stats['total_processed'] += 1
        
        try:
            # Process with timeout
            document = await asyncio.wait_for(
                self._process_with_routing(file_path, document_type, options),
                timeout=options.timeout_seconds
            )
            
            # Calculate final metrics
            processing_time = int((time.time() - start_time) * 1000)
            document.processing_time_ms = processing_time
            document.confidence_score = document.calculate_confidence_score()
            
            # Determine extraction quality
            if document.confidence_score >= 0.8:
                document.extraction_quality = 'high'
            elif document.confidence_score >= 0.5:
                document.extraction_quality = 'medium'
            else:
                document.extraction_quality = 'low'
            
            self.processing_stats['successful'] += 1
            
            return ProcessingResult(
                success=True,
                document=document,
                processing_time_ms=processing_time
            )
            
        except asyncio.TimeoutError:
            self.processing_stats['failed'] += 1
            return ProcessingResult(
                success=False,
                error_message=f"Processing timeout after {options.timeout_seconds} seconds",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            # Try fallback processing
            try:
                document = await self.fallback_handler.handle_processing_error(
                    e, file_path, document_type
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                document.processing_time_ms = processing_time
                
                if document.processing_status == ProcessingStatus.COMPLETED:
                    self.processing_stats['fallback_used'] += 1
                    return ProcessingResult(
                        success=True,
                        document=document,
                        processing_time_ms=processing_time
                    )
                else:
                    self.processing_stats['failed'] += 1
                    return ProcessingResult(
                        success=False,
                        document=document,
                        error_message=f"Processing failed: {str(e)}",
                        processing_time_ms=processing_time
                    )
                    
            except Exception as fallback_error:
                self.processing_stats['failed'] += 1
                return ProcessingResult(
                    success=False,
                    error_message=f"Processing and fallback failed: {str(e)}, {str(fallback_error)}",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
    
    async def _process_with_routing(self, 
                                  file_path: Path,
                                  document_type: DocumentType,
                                  options: ProcessingOptions) -> ProcessedDocument:
        """Route processing to appropriate processor"""
        
        # Get processor for document type
        processor = self.processors.get(document_type)
        
        if processor is None:
            raise ValueError(f"No processor registered for document type: {document_type}")
        
        # Check if processor can handle this specific file
        if not await processor.can_process(file_path, document_type):
            raise ValueError(f"Processor cannot handle file: {file_path}")
        
        # Process the document
        document = await processor.process(file_path, options)
        
        # Validate processing result
        if document.processing_status == ProcessingStatus.FAILED:
            raise Exception("Processor returned failed status")
        
        return document
    
    async def process_multiple_documents(self,
                                       file_paths: List[Union[str, Path]],
                                       options: Optional[ProcessingOptions] = None,
                                       parallel: bool = True) -> List[ProcessingResult]:
        """Process multiple documents concurrently or sequentially"""
        
        if options is None:
            options = ProcessingOptions()
        
        if parallel and options.parallel_processing:
            # Process documents concurrently
            tasks = [
                self.process_document(file_path, options=options)
                for file_path in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ProcessingResult(
                        success=False,
                        error_message=f"Processing exception: {str(result)}"
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Process documents sequentially
            results = []
            for file_path in file_paths:
                result = await self.process_document(file_path, options=options)
                results.append(result)
            
            return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_processed']
            stats['failure_rate'] = stats['failed'] / stats['total_processed']
            stats['fallback_rate'] = stats['fallback_used'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['fallback_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'fallback_used': 0
        }
    
    async def extract_structured_content(self, 
                                       content: bytes,
                                       document_type: DocumentType,
                                       options: Optional[ProcessingOptions] = None) -> ProcessedDocument:
        """Extract structured content from raw bytes"""
        
        # Create temporary file for processing
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{document_type.value}') as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        
        try:
            result = await self.process_document(temp_path, document_type, options)
            if result.success and result.document:
                return result.document
            else:
                raise Exception(result.error_message or "Processing failed")
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def get_supported_document_types(self) -> List[DocumentType]:
        """Get list of all supported document types"""
        return list(self.processors.keys())
    
    async def process_document_with_ethics(self, 
                                         file_path: Union[str, Path],
                                         document_type: Optional[DocumentType] = None,
                                         options: Optional[ProcessingOptions] = None) -> Dict[str, Any]:
        """Process document with ethics compliance checking"""
        try:
            from .ethics_document_integration import document_ethics_processor
            
            # First process the document normally
            result = await self.process_document(file_path, document_type, options)
            
            if not result.success or not result.document:
                return {
                    'processing_result': result,
                    'ethics_report': None,
                    'error': result.error_message
                }
            
            # Extract text content for ethics analysis
            document_content = result.document.extracted_text or ""
            document_metadata = {
                'id': str(file_path),
                'name': result.document.original_filename,
                'type': result.document.document_type.value,
                'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                'processing_time_ms': result.document.processing_time_ms
            }
            
            # Run ethics compliance check
            ethics_report = await document_ethics_processor.process_document_with_ethics(
                document_content, document_metadata
            )
            
            return {
                'processing_result': result,
                'ethics_report': ethics_report,
                'compliance_score': ethics_report.compliance_score,
                'issues_found': len(ethics_report.issues_found),
                'recommendations': ethics_report.recommendations
            }
            
        except ImportError:
            return {
                'processing_result': result if 'result' in locals() else None,
                'ethics_report': None,
                'error': 'Ethics framework not available'
            }
        except Exception as e:
            return {
                'processing_result': result if 'result' in locals() else None,
                'ethics_report': None,
                'error': f'Ethics processing failed: {str(e)}'
            }
    
    async def validate_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate document without full processing"""
        
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        validation_result = {
            'valid': False,
            'document_type': DocumentType.UNKNOWN,
            'file_size': 0,
            'estimated_processing_time_ms': 0,
            'supported': False,
            'errors': []
        }
        
        try:
            # Check file exists
            if not file_path.exists():
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Get file info
            file_size = file_path.stat().st_size
            validation_result['file_size'] = file_size
            
            # Detect document type
            document_type = detect_document_type(file_path)
            validation_result['document_type'] = document_type
            
            # Check if supported
            validation_result['supported'] = is_supported_document_type(document_type)
            
            if not validation_result['supported']:
                validation_result['errors'].append(f"Document type {document_type} not supported")
                return validation_result
            
            # Estimate processing time
            validation_result['estimated_processing_time_ms'] = estimate_processing_time(
                file_size, document_type
            )
            
            # Check if processor is available
            if document_type not in self.processors:
                validation_result['errors'].append(f"No processor available for {document_type}")
                return validation_result
            
            validation_result['valid'] = True
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result


# Global document processor instance
document_processor = DocumentProcessor()