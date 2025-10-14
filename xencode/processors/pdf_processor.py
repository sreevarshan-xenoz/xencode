#!/usr/bin/env python3
"""
PDF Document Processor

Implements PDF processing using PyMuPDF (fitz) for text extraction,
table detection, metadata extraction, and structured content analysis.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from xencode.processors.base import ProcessorInterface
from xencode.models.document import (
    ContentType,
    DocumentMetadata,
    DocumentType,
    ProcessedDocument,
    ProcessingOptions,
    ProcessingStatus,
    StructuredContent
)


class PDFProcessor(ProcessorInterface):
    """PDF document processor using PyMuPDF"""
    
    def __init__(self):
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )
        
        self.supported_types = [DocumentType.PDF]
        
        # Text extraction settings
        self.text_flags = (
            fitz.TEXT_PRESERVE_WHITESPACE |
            fitz.TEXT_PRESERVE_LIGATURES |
            fitz.TEXT_PRESERVE_SPANS
        )
    
    async def can_process(self, file_path: Path, document_type: DocumentType) -> bool:
        """Check if this processor can handle the document"""
        if document_type != DocumentType.PDF:
            return False
        
        if not PYMUPDF_AVAILABLE:
            return False
        
        try:
            # Try to open the PDF to verify it's valid
            doc = fitz.open(str(file_path))
            doc.close()
            return True
        except Exception:
            return False
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return self.supported_types.copy()
    
    async def process(self, 
                     file_path: Path, 
                     options: ProcessingOptions) -> ProcessedDocument:
        """Process PDF document and extract structured content"""
        
        document = ProcessedDocument(
            original_filename=file_path.name,
            file_path=str(file_path),
            document_type=DocumentType.PDF,
            processing_status=ProcessingStatus.PROCESSING
        )
        
        try:
            # Open PDF document
            pdf_doc = fitz.open(str(file_path))
            
            # Extract metadata
            if options.extract_metadata:
                document.metadata = await self._extract_metadata(pdf_doc, file_path)
            
            # Extract text and structured content
            if options.extract_text or options.extract_structured_content:
                await self._extract_content(pdf_doc, document, options)
            
            # Calculate confidence score
            document.confidence_score = self._calculate_confidence_score(document)
            
            # Set processing status
            document.processing_status = ProcessingStatus.COMPLETED
            
            pdf_doc.close()
            
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            document.errors.append(f"PDF processing error: {str(e)}")
            document.confidence_score = 0.0
        
        return document
    
    async def _extract_metadata(self, 
                               pdf_doc: 'fitz.Document', 
                               file_path: Path) -> DocumentMetadata:
        """Extract PDF metadata"""
        metadata = DocumentMetadata()
        
        try:
            # Get PDF metadata
            pdf_metadata = pdf_doc.metadata
            
            metadata.title = pdf_metadata.get('title')
            metadata.author = pdf_metadata.get('author')
            metadata.page_count = pdf_doc.page_count
            metadata.file_size = file_path.stat().st_size
            metadata.mime_type = 'application/pdf'
            
            # PDF-specific metadata
            metadata.pdf_version = f"PDF-{pdf_doc.pdf_version()}"
            metadata.is_encrypted = pdf_doc.needs_pass
            metadata.has_forms = pdf_doc.has_annots()
            
            # Calculate word count from first few pages for estimation
            word_count = 0
            sample_pages = min(3, pdf_doc.page_count)
            
            for page_num in range(sample_pages):
                page = pdf_doc[page_num]
                text = page.get_text()
                word_count += len(text.split())
            
            # Estimate total word count
            if sample_pages > 0:
                avg_words_per_page = word_count / sample_pages
                metadata.word_count = int(avg_words_per_page * pdf_doc.page_count)
            
        except Exception as e:
            # Continue processing even if metadata extraction fails
            pass
        
        return metadata
    
    async def _extract_content(self, 
                              pdf_doc: 'fitz.Document',
                              document: ProcessedDocument,
                              options: ProcessingOptions) -> None:
        """Extract text and structured content from PDF"""
        
        all_text_parts = []
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            
            # Extract plain text
            if options.extract_text:
                page_text = page.get_text()
                all_text_parts.append(page_text)
            
            # Extract structured content
            if options.extract_structured_content:
                await self._extract_structured_content_from_page(
                    page, page_num, document, options
                )
        
        # Set extracted text
        if options.extract_text:
            document.extracted_text = '\n\n'.join(all_text_parts)
    
    async def _extract_structured_content_from_page(self,
                                                   page: 'fitz.Page',
                                                   page_num: int,
                                                   document: ProcessedDocument,
                                                   options: ProcessingOptions) -> None:
        """Extract structured content from a single page"""
        
        try:
            # Get text with formatting information
            text_dict = page.get_text("dict")
            
            # Process blocks
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                # Process lines in block
                for line in block["lines"]:
                    line_text = ""
                    line_formatting = {}
                    
                    # Process spans (text runs with consistent formatting)
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        line_text += span_text
                        
                        # Collect formatting information
                        if span_text.strip():
                            font_info = {
                                'font': span.get('font', ''),
                                'size': span.get('size', 0),
                                'flags': span.get('flags', 0),
                                'color': span.get('color', 0)
                            }
                            line_formatting.update(font_info)
                    
                    if line_text.strip():
                        # Determine content type based on formatting and content
                        content_type = self._determine_content_type(
                            line_text, line_formatting
                        )
                        
                        # Create structured content
                        structured_content = StructuredContent(
                            content_type=content_type,
                            text=line_text.strip(),
                            position={
                                'page': page_num + 1,
                                'bbox': line.get('bbox', [])
                            },
                            formatting=line_formatting,
                            confidence=0.9
                        )
                        
                        document.structured_content.append(structured_content)
            
            # Extract tables if enabled
            if options.include_tables:
                await self._extract_tables_from_page(page, page_num, document)
            
        except Exception as e:
            document.warnings.append(f"Error extracting structured content from page {page_num + 1}: {str(e)}")
    
    def _determine_content_type(self, 
                               text: str, 
                               formatting: Dict[str, Any]) -> ContentType:
        """Determine content type based on text and formatting"""
        
        # Check for headings based on font size and formatting
        font_size = formatting.get('size', 0)
        font_flags = formatting.get('flags', 0)
        
        # Font flags: 16=bold, 2=italic, 4=superscript, 8=subscript
        is_bold = bool(font_flags & 16)
        
        # Heading detection
        if font_size > 14 or (is_bold and font_size > 12):
            # Additional checks for heading patterns
            if (len(text) < 100 and 
                (text.isupper() or 
                 re.match(r'^\d+\.?\s+[A-Z]', text) or
                 re.match(r'^[A-Z][^.!?]*$', text))):
                return ContentType.HEADING
        
        # Code block detection
        if self._is_code_like(text):
            return ContentType.CODE_BLOCK
        
        # List detection
        if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.?\s+', text):
            return ContentType.LIST
        
        # Link detection
        if 'http://' in text or 'https://' in text or '@' in text:
            return ContentType.LINK
        
        # Default to text
        return ContentType.TEXT
    
    def _is_code_like(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = [
            '{', '}', '()', '[]', ';', '//', '/*', '*/', 
            'function', 'class', 'def ', 'var ', 'let ', 'const ',
            'import ', 'from ', 'include', '#include'
        ]
        
        # Check for multiple code indicators
        indicator_count = sum(1 for indicator in code_indicators if indicator in text)
        
        # Also check for consistent indentation
        lines = text.split('\n')
        if len(lines) > 1:
            indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
            if indented_lines > len(lines) * 0.5:
                indicator_count += 1
        
        return indicator_count >= 2
    
    async def _extract_tables_from_page(self,
                                       page: 'fitz.Page',
                                       page_num: int,
                                       document: ProcessedDocument) -> None:
        """Extract tables from page using PyMuPDF table detection"""
        
        try:
            # Find tables on the page
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                # Extract table data
                table_data = table.extract()
                
                if table_data:
                    # Convert table to text representation
                    table_text = self._format_table_as_text(table_data)
                    
                    # Create structured content for table
                    table_content = StructuredContent(
                        content_type=ContentType.TABLE,
                        text=table_text,
                        position={
                            'page': page_num + 1,
                            'table_index': table_index,
                            'bbox': table.bbox
                        },
                        attributes={
                            'rows': len(table_data),
                            'columns': len(table_data[0]) if table_data else 0,
                            'raw_data': table_data
                        },
                        confidence=0.8
                    )
                    
                    document.structured_content.append(table_content)
        
        except Exception as e:
            document.warnings.append(f"Error extracting tables from page {page_num + 1}: {str(e)}")
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text"""
        if not table_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        for col_index in range(len(table_data[0])):
            max_width = max(
                len(str(row[col_index]) if col_index < len(row) else "")
                for row in table_data
            )
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        # Format table
        formatted_rows = []
        for row in table_data:
            formatted_cells = []
            for col_index, cell in enumerate(row):
                if col_index < len(col_widths):
                    cell_str = str(cell or "").ljust(col_widths[col_index])
                    formatted_cells.append(cell_str)
            formatted_rows.append(" | ".join(formatted_cells))
        
        return "\n".join(formatted_rows)
    
    def _calculate_confidence_score(self, document: ProcessedDocument) -> float:
        """Calculate confidence score for PDF processing"""
        
        if document.processing_status == ProcessingStatus.FAILED:
            return 0.0
        
        confidence_factors = []
        
        # Text extraction confidence
        if document.extracted_text:
            text_length = len(document.extracted_text.strip())
            if text_length > 100:
                confidence_factors.append(0.9)
            elif text_length > 10:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
        
        # Structured content confidence
        if document.structured_content:
            avg_content_confidence = sum(
                content.confidence for content in document.structured_content
            ) / len(document.structured_content)
            confidence_factors.append(avg_content_confidence)
        
        # Metadata confidence
        if document.metadata and document.metadata.page_count:
            confidence_factors.append(0.8)
        
        # Error penalty
        error_penalty = min(len(document.errors) * 0.1, 0.5)
        
        # Calculate final confidence
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, base_confidence - error_penalty)
        else:
            return 0.1  # Minimal confidence if no content extracted
    
    async def extract_text_from_region(self,
                                      file_path: Path,
                                      page_num: int,
                                      bbox: Tuple[float, float, float, float]) -> str:
        """Extract text from specific region of a PDF page"""
        
        try:
            pdf_doc = fitz.open(str(file_path))
            
            if page_num >= pdf_doc.page_count:
                raise ValueError(f"Page {page_num} does not exist")
            
            page = pdf_doc[page_num]
            
            # Create rectangle for the region
            rect = fitz.Rect(bbox)
            
            # Extract text from the region
            text = page.get_text(clip=rect)
            
            pdf_doc.close()
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from region: {str(e)}")
    
    async def get_page_images(self,
                             file_path: Path,
                             page_num: int,
                             min_width: int = 100,
                             min_height: int = 100) -> List[Dict[str, Any]]:
        """Extract images from a specific PDF page"""
        
        images = []
        
        try:
            pdf_doc = fitz.open(str(file_path))
            
            if page_num >= pdf_doc.page_count:
                raise ValueError(f"Page {page_num} does not exist")
            
            page = pdf_doc[page_num]
            
            # Get list of images on the page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get image data
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                
                # Check image dimensions
                if (base_image["width"] >= min_width and 
                    base_image["height"] >= min_height):
                    
                    image_info = {
                        'index': img_index,
                        'xref': xref,
                        'width': base_image["width"],
                        'height': base_image["height"],
                        'colorspace': base_image["colorspace"],
                        'bpc': base_image["bpc"],
                        'ext': base_image["ext"],
                        'size': len(base_image["image"])
                    }
                    
                    images.append(image_info)
            
            pdf_doc.close()
            
        except Exception as e:
            raise Exception(f"Error extracting images: {str(e)}")
        
        return images