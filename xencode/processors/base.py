#!/usr/bin/env python3
"""
Base Processor Interface

Defines the abstract interface that all document processors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from xencode.models.document import DocumentType, ProcessedDocument, ProcessingOptions


class ProcessorInterface(ABC):
    """Abstract interface for document processors"""
    
    @abstractmethod
    async def can_process(self, file_path: Path, document_type: DocumentType) -> bool:
        """Check if this processor can handle the document type"""
        pass
    
    @abstractmethod
    async def process(self, 
                     file_path: Path, 
                     options: ProcessingOptions) -> ProcessedDocument:
        """Process the document and return structured content"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[DocumentType]:
        """Get list of supported document types"""
        pass