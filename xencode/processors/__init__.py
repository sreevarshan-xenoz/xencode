#!/usr/bin/env python3
"""
Document Processors Package

Contains specialized processors for different document types:
- PDFProcessor: PDF document processing with PyMuPDF
- DOCXProcessor: DOCX document processing with python-docx  
- WebContentExtractor: HTML/web content processing with BeautifulSoup4
"""

from typing import List, Optional

# Import processors with graceful fallback
try:
    from .pdf_processor import PDFProcessor
    PDF_PROCESSOR_AVAILABLE = True
except ImportError:
    PDFProcessor = None
    PDF_PROCESSOR_AVAILABLE = False

try:
    from .docx_processor import DOCXProcessor
    DOCX_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCXProcessor = None
    DOCX_PROCESSOR_AVAILABLE = False

try:
    from .web_extractor import WebContentExtractor
    WEB_PROCESSOR_AVAILABLE = True
except ImportError:
    WebContentExtractor = None
    WEB_PROCESSOR_AVAILABLE = False

try:
    from .text_processor import TextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    TextProcessor = None
    TEXT_PROCESSOR_AVAILABLE = False


def get_available_processors() -> List[str]:
    """Get list of available processor names"""
    available = []
    
    if PDF_PROCESSOR_AVAILABLE:
        available.append("PDFProcessor")
    
    if DOCX_PROCESSOR_AVAILABLE:
        available.append("DOCXProcessor")
    
    if WEB_PROCESSOR_AVAILABLE:
        available.append("WebContentExtractor")
    
    if TEXT_PROCESSOR_AVAILABLE:
        available.append("TextProcessor")
    
    return available


def create_processor(processor_type: str, **kwargs):
    """Factory function to create processors"""
    
    if processor_type.lower() == "pdf" and PDF_PROCESSOR_AVAILABLE:
        return PDFProcessor(**kwargs)
    
    elif processor_type.lower() == "docx" and DOCX_PROCESSOR_AVAILABLE:
        return DOCXProcessor(**kwargs)
    
    elif processor_type.lower() in ["html", "web"] and WEB_PROCESSOR_AVAILABLE:
        return WebContentExtractor(**kwargs)
    
    elif processor_type.lower() in ["text", "markdown", "code"] and TEXT_PROCESSOR_AVAILABLE:
        return TextProcessor(**kwargs)
    
    else:
        raise ValueError(f"Processor type '{processor_type}' not available or not supported")


def get_processor_status() -> dict:
    """Get status of all processors"""
    return {
        "pdf_available": PDF_PROCESSOR_AVAILABLE,
        "docx_available": DOCX_PROCESSOR_AVAILABLE,
        "web_available": WEB_PROCESSOR_AVAILABLE,
        "text_available": TEXT_PROCESSOR_AVAILABLE,
        "total_available": len(get_available_processors())
    }


__all__ = [
    'PDFProcessor',
    'DOCXProcessor', 
    'WebContentExtractor',
    'TextProcessor',
    'get_available_processors',
    'create_processor',
    'get_processor_status',
    'PDF_PROCESSOR_AVAILABLE',
    'DOCX_PROCESSOR_AVAILABLE',
    'WEB_PROCESSOR_AVAILABLE',
    'TEXT_PROCESSOR_AVAILABLE'
]