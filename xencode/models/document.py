#!/usr/bin/env python3
"""
Document Processing Data Models

Defines data models for document processing, including document types,
processed documents, and structured content representations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentType(str, Enum):
    """Supported document types for processing"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    TEXT = "text"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ContentType(str, Enum):
    """Types of structured content"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    HEADING = "heading"
    LIST = "list"
    LINK = "link"
    METADATA = "metadata"


@dataclass
class DocumentMetadata:
    """Document metadata information"""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    
    # PDF-specific metadata
    pdf_version: Optional[str] = None
    is_encrypted: bool = False
    has_forms: bool = False
    
    # DOCX-specific metadata
    docx_version: Optional[str] = None
    has_macros: bool = False
    
    # HTML-specific metadata
    html_title: Optional[str] = None
    meta_description: Optional[str] = None
    meta_keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result


@dataclass
class StructuredContent:
    """Structured content extracted from document"""
    content_type: ContentType
    text: str
    position: Optional[Dict[str, Any]] = None  # Page, coordinates, etc.
    formatting: Optional[Dict[str, Any]] = None  # Font, size, style, etc.
    attributes: Optional[Dict[str, Any]] = None  # Additional attributes
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content_type': self.content_type.value,
            'text': self.text,
            'position': self.position,
            'formatting': self.formatting,
            'attributes': self.attributes,
            'confidence': self.confidence
        }


@dataclass
class ProcessedDocument:
    """Processed document with extracted content and metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str = ""
    file_path: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    
    # Processing information
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time_ms: int = 0
    processed_at: datetime = field(default_factory=datetime.now)
    
    # Extracted content
    extracted_text: str = ""
    structured_content: List[StructuredContent] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_quality: Optional[str] = None  # 'high', 'medium', 'low'
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_structured_content(self, 
                             content_type: ContentType,
                             text: str,
                             **kwargs) -> None:
        """Add structured content to the document"""
        content = StructuredContent(
            content_type=content_type,
            text=text,
            **kwargs
        )
        self.structured_content.append(content)
    
    def get_content_by_type(self, content_type: ContentType) -> List[StructuredContent]:
        """Get all content of a specific type"""
        return [
            content for content in self.structured_content 
            if content.content_type == content_type
        ]
    
    def get_text_content(self) -> str:
        """Get all text content concatenated"""
        if self.extracted_text:
            return self.extracted_text
        
        # Fallback to structured content
        text_parts = []
        for content in self.structured_content:
            if content.content_type in [ContentType.TEXT, ContentType.HEADING]:
                text_parts.append(content.text)
        
        return '\n'.join(text_parts)
    
    def calculate_confidence_score(self) -> float:
        """Calculate overall confidence score"""
        if not self.structured_content:
            return 0.0
        
        total_confidence = sum(content.confidence for content in self.structured_content)
        return total_confidence / len(self.structured_content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'document_type': self.document_type.value,
            'processing_status': self.processing_status.value,
            'processing_time_ms': self.processing_time_ms,
            'processed_at': self.processed_at.isoformat(),
            'extracted_text': self.extracted_text,
            'structured_content': [content.to_dict() for content in self.structured_content],
            'metadata': self.metadata.to_dict(),
            'confidence_score': self.confidence_score,
            'extraction_quality': self.extraction_quality,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create ProcessedDocument from dictionary"""
        # Parse structured content
        structured_content = []
        for content_data in data.get('structured_content', []):
            content = StructuredContent(
                content_type=ContentType(content_data['content_type']),
                text=content_data['text'],
                position=content_data.get('position'),
                formatting=content_data.get('formatting'),
                attributes=content_data.get('attributes'),
                confidence=content_data.get('confidence', 1.0)
            )
            structured_content.append(content)
        
        # Parse metadata
        metadata_data = data.get('metadata', {})
        metadata = DocumentMetadata(**metadata_data)
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            original_filename=data.get('original_filename', ''),
            file_path=data.get('file_path'),
            document_type=DocumentType(data.get('document_type', DocumentType.UNKNOWN)),
            processing_status=ProcessingStatus(data.get('processing_status', ProcessingStatus.PENDING)),
            processing_time_ms=data.get('processing_time_ms', 0),
            processed_at=datetime.fromisoformat(data.get('processed_at', datetime.now().isoformat())),
            extracted_text=data.get('extracted_text', ''),
            structured_content=structured_content,
            metadata=metadata,
            confidence_score=data.get('confidence_score', 0.0),
            extraction_quality=data.get('extraction_quality'),
            errors=data.get('errors', []),
            warnings=data.get('warnings', [])
        )


@dataclass
class ProcessingOptions:
    """Options for document processing"""
    # General options
    extract_text: bool = True
    extract_metadata: bool = True
    extract_structured_content: bool = True
    
    # Quality options
    ocr_enabled: bool = False  # For scanned documents
    language_detection: bool = True
    content_filtering: bool = True
    
    # Performance options
    timeout_seconds: int = 30
    max_file_size_mb: int = 100
    parallel_processing: bool = False
    
    # Output options
    preserve_formatting: bool = True
    include_images: bool = False
    include_tables: bool = True
    
    # Security options
    validate_content: bool = True
    sanitize_output: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__.copy()


@dataclass
class ProcessingResult:
    """Result of document processing operation"""
    success: bool
    document: Optional[ProcessedDocument] = None
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'success': self.success,
            'document': self.document.to_dict() if self.document else None,
            'error_message': self.error_message,
            'processing_time_ms': self.processing_time_ms
        }


# Utility functions for document type detection
def detect_document_type(file_path: Union[str, Path]) -> DocumentType:
    """Detect document type from file extension"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    extension = file_path.suffix.lower()
    
    type_mapping = {
        '.pdf': DocumentType.PDF,
        '.docx': DocumentType.DOCX,
        '.doc': DocumentType.DOCX,  # Treat as DOCX for processing
        '.html': DocumentType.HTML,
        '.htm': DocumentType.HTML,
        '.md': DocumentType.MARKDOWN,
        '.markdown': DocumentType.MARKDOWN,
        '.txt': DocumentType.TEXT,
        '.py': DocumentType.CODE,
        '.js': DocumentType.CODE,
        '.ts': DocumentType.CODE,
        '.java': DocumentType.CODE,
        '.cpp': DocumentType.CODE,
        '.c': DocumentType.CODE,
        '.cs': DocumentType.CODE,
        '.php': DocumentType.CODE,
        '.rb': DocumentType.CODE,
        '.go': DocumentType.CODE,
        '.rs': DocumentType.CODE,
    }
    
    return type_mapping.get(extension, DocumentType.UNKNOWN)


def is_supported_document_type(document_type: DocumentType) -> bool:
    """Check if document type is supported for processing"""
    supported_types = {
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.HTML,
        DocumentType.MARKDOWN,
        DocumentType.TEXT,
        DocumentType.CODE
    }
    return document_type in supported_types


def estimate_processing_time(file_size_bytes: int, document_type: DocumentType) -> int:
    """Estimate processing time in milliseconds based on file size and type"""
    # Base processing time per MB for different document types
    base_times = {
        DocumentType.PDF: 2000,      # 2 seconds per MB
        DocumentType.DOCX: 1000,     # 1 second per MB
        DocumentType.HTML: 500,      # 0.5 seconds per MB
        DocumentType.MARKDOWN: 200,  # 0.2 seconds per MB
        DocumentType.TEXT: 100,      # 0.1 seconds per MB
        DocumentType.CODE: 150,      # 0.15 seconds per MB
    }
    
    file_size_mb = max(file_size_bytes / (1024 * 1024), 0.001)  # Minimum 0.001 MB
    base_time = base_times.get(document_type, 1000)
    
    return max(int(file_size_mb * base_time), 1)  # Minimum 1ms