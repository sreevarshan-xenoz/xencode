#!/usr/bin/env python3
"""
Web Content Extractor

Implements HTML/web content processing using BeautifulSoup4 for text extraction,
content sanitization, main content detection, and structured content analysis.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup, Comment
    from bs4.element import Tag, NavigableString
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None

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


class WebContentExtractor(ProcessorInterface):
    """Web content extractor using BeautifulSoup4"""
    
    def __init__(self):
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTML processing. "
                "Install with: pip install beautifulsoup4"
            )
        
        self.supported_types = [DocumentType.HTML]
        
        # Elements to remove (navigation, ads, etc.)
        self.noise_selectors = [
            'nav', 'header', 'footer', 'aside', 'sidebar',
            '.nav', '.navigation', '.menu', '.header', '.footer',
            '.sidebar', '.aside', '.advertisement', '.ad', '.ads',
            '.social', '.share', '.comment', '.comments',
            '.related', '.recommended', '.popup', '.modal',
            '.cookie', '.gdpr', '.newsletter', '.subscription',
            'script', 'style', 'noscript', 'iframe'
        ]
        
        # Main content selectors (in order of preference)
        self.main_content_selectors = [
            'main', 'article', '[role="main"]',
            '.main', '.content', '.post', '.article',
            '.entry', '.story', '.text', '.body',
            '#main', '#content', '#post', '#article',
            '#entry', '#story', '#text', '#body'
        ]
        
        # Heading tags
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    async def can_process(self, file_path: Path, document_type: DocumentType) -> bool:
        """Check if this processor can handle the document"""
        if document_type != DocumentType.HTML:
            return False
        
        if not BEAUTIFULSOUP_AVAILABLE:
            return False
        
        try:
            # Try to parse the HTML to verify it's valid
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            return soup.find() is not None  # Has at least one tag
        except Exception:
            return False
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return self.supported_types.copy()
    
    async def process(self, 
                     file_path: Path, 
                     options: ProcessingOptions) -> ProcessedDocument:
        """Process HTML document and extract structured content"""
        
        document = ProcessedDocument(
            original_filename=file_path.name,
            file_path=str(file_path),
            document_type=DocumentType.HTML,
            processing_status=ProcessingStatus.PROCESSING
        )
        
        try:
            # Read HTML content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            if options.extract_metadata:
                document.metadata = await self._extract_metadata(soup, file_path)
            
            # Clean and sanitize content
            if options.sanitize_output:
                soup = await self._sanitize_content(soup)
            
            # Extract main content
            main_content = await self._extract_main_content(soup)
            
            # Extract text and structured content
            if options.extract_text or options.extract_structured_content:
                await self._extract_content(main_content, document, options)
            
            # Extract tables
            if options.include_tables:
                await self._extract_tables(main_content, document)
            
            # Calculate confidence score
            document.confidence_score = self._calculate_confidence_score(document, soup)
            
            # Set processing status
            document.processing_status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            document.errors.append(f"HTML processing error: {str(e)}")
            document.confidence_score = 0.0
        
        return document
    
    async def _extract_metadata(self, 
                               soup: BeautifulSoup, 
                               file_path: Path) -> DocumentMetadata:
        """Extract HTML metadata"""
        metadata = DocumentMetadata()
        
        try:
            # Basic file info
            metadata.file_size = file_path.stat().st_size
            metadata.mime_type = 'text/html'
            
            # HTML title
            title_tag = soup.find('title')
            if title_tag:
                metadata.title = title_tag.get_text().strip()
                metadata.html_title = metadata.title
            
            # Meta tags
            meta_tags = soup.find_all('meta')
            
            for meta in meta_tags:
                name = meta.get('name', '').lower()
                property_attr = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description' or property_attr == 'og:description':
                    metadata.meta_description = content
                elif name == 'keywords':
                    metadata.meta_keywords = [k.strip() for k in content.split(',')]
                elif name == 'author' or property_attr == 'og:author':
                    metadata.author = content
                elif property_attr == 'og:title':
                    if not metadata.title:
                        metadata.title = content
            
            # Language detection
            html_tag = soup.find('html')
            if html_tag:
                lang = html_tag.get('lang')
                if lang:
                    metadata.language = lang
            
            # Word count estimation
            text_content = soup.get_text()
            if text_content:
                metadata.word_count = len(text_content.split())
            
        except Exception as e:
            # Continue processing even if metadata extraction fails
            pass
        
        return metadata
    
    async def _sanitize_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove noise elements and sanitize content"""
        
        try:
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove noise elements
            for selector in self.noise_selectors:
                elements = soup.select(selector)
                for element in elements:
                    element.extract()
            
            # Remove elements with noise-indicating attributes
            noise_patterns = [
                'advertisement', 'ad-', 'ads-', 'social', 'share',
                'comment', 'nav', 'menu', 'sidebar', 'footer'
            ]
            
            for element in soup.find_all():
                if element.name:
                    # Check class and id attributes
                    classes = element.get('class', [])
                    element_id = element.get('id', '')
                    
                    # Convert to strings for checking
                    class_str = ' '.join(classes).lower()
                    id_str = element_id.lower()
                    
                    # Check if element should be removed
                    should_remove = False
                    for pattern in noise_patterns:
                        if pattern in class_str or pattern in id_str:
                            should_remove = True
                            break
                    
                    if should_remove:
                        element.extract()
            
            return soup
            
        except Exception as e:
            # Return original soup if sanitization fails
            return soup
    
    async def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content area from HTML"""
        
        # Try main content selectors in order of preference
        for selector in self.main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element and self._has_substantial_content(main_element):
                return main_element
        
        # Fallback: use body or entire document
        body = soup.find('body')
        if body:
            return body
        
        return soup
    
    def _has_substantial_content(self, element) -> bool:
        """Check if element has substantial text content"""
        text = element.get_text().strip()
        return len(text) > 100  # At least 100 characters
    
    async def _extract_content(self, 
                              main_content: BeautifulSoup,
                              document: ProcessedDocument,
                              options: ProcessingOptions) -> None:
        """Extract text and structured content from HTML"""
        
        all_text_parts = []
        
        # Process all elements in main content
        for element in main_content.find_all():
            if not element.name:
                continue
            
            element_text = element.get_text().strip()
            
            if not element_text:
                continue
            
            # Skip if this text is already captured by a parent element
            if self._is_nested_content(element, main_content):
                continue
            
            # Add to full text
            if options.extract_text:
                all_text_parts.append(element_text)
            
            # Extract structured content
            if options.extract_structured_content:
                await self._extract_structured_content_from_element(
                    element, document, options
                )
        
        # Set extracted text
        if options.extract_text:
            # Remove duplicates while preserving order
            unique_text_parts = []
            seen = set()
            for text in all_text_parts:
                if text not in seen:
                    unique_text_parts.append(text)
                    seen.add(text)
            
            document.extracted_text = '\n\n'.join(unique_text_parts)
    
    def _is_nested_content(self, element, main_content) -> bool:
        """Check if element's content is already captured by parent"""
        # This is a simplified check - in practice, you might want more sophisticated logic
        parent = element.parent
        while parent and parent != main_content:
            if parent.name in self.heading_tags or parent.name in ['p', 'div', 'section']:
                # If parent is a content element, this might be nested
                parent_text = parent.get_text().strip()
                element_text = element.get_text().strip()
                
                # If element text is substantial part of parent text, consider it nested
                if len(element_text) > 50 and element_text in parent_text:
                    return True
            
            parent = parent.parent
        
        return False
    
    async def _extract_structured_content_from_element(self,
                                                      element,
                                                      document: ProcessedDocument,
                                                      options: ProcessingOptions) -> None:
        """Extract structured content from HTML element"""
        
        try:
            element_text = element.get_text().strip()
            if not element_text:
                return
            
            # Get element attributes and styling
            attributes = dict(element.attrs) if hasattr(element, 'attrs') else {}
            
            # Determine content type
            content_type = self._determine_content_type(element, element_text)
            
            # Create structured content
            structured_content = StructuredContent(
                content_type=content_type,
                text=element_text,
                position={
                    'tag': element.name,
                    'xpath': self._get_xpath(element)
                },
                attributes=attributes,
                confidence=0.8
            )
            
            document.structured_content.append(structured_content)
            
        except Exception as e:
            document.warnings.append(f"Error extracting structured content from element: {str(e)}")
    
    def _determine_content_type(self, element, text: str) -> ContentType:
        """Determine content type based on HTML element and text"""
        
        tag_name = element.name.lower()
        
        # Heading detection
        if tag_name in self.heading_tags:
            return ContentType.HEADING
        
        # Code detection
        if tag_name in ['code', 'pre', 'kbd', 'samp']:
            return ContentType.CODE_BLOCK
        
        # List detection
        if tag_name in ['li', 'ul', 'ol', 'dl']:
            return ContentType.LIST
        
        # Link detection
        if tag_name == 'a' or 'http://' in text or 'https://' in text:
            return ContentType.LINK
        
        # Table detection (handled separately, but just in case)
        if tag_name in ['table', 'tr', 'td', 'th']:
            return ContentType.TABLE
        
        # Check for code-like content in regular elements
        if self._is_code_like(text):
            return ContentType.CODE_BLOCK
        
        # Default to text
        return ContentType.TEXT
    
    def _is_code_like(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = [
            '{', '}', '()', '[]', ';', '//', '/*', '*/', 
            'function', 'class', 'def ', 'var ', 'let ', 'const ',
            'import ', 'from ', 'include', '#include', '<?', '?>'
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
    
    def _get_xpath(self, element) -> str:
        """Generate simple XPath for element"""
        try:
            path_parts = []
            current = element
            
            while current and current.name:
                tag = current.name
                # Add position if there are siblings with same tag
                siblings = [s for s in current.parent.children if hasattr(s, 'name') and s.name == tag] if current.parent else [current]
                if len(siblings) > 1:
                    position = siblings.index(current) + 1
                    tag += f'[{position}]'
                
                path_parts.append(tag)
                current = current.parent
            
            path_parts.reverse()
            return '/' + '/'.join(path_parts)
            
        except Exception:
            return f"/{element.name}" if element.name else "/unknown"
    
    async def _extract_tables(self,
                             main_content: BeautifulSoup,
                             document: ProcessedDocument) -> None:
        """Extract tables from HTML content"""
        
        try:
            tables = main_content.find_all('table')
            
            for table_index, table in enumerate(tables):
                # Extract table data
                table_data = []
                
                rows = table.find_all('tr')
                for row in rows:
                    row_data = []
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        cell_text = cell.get_text().strip()
                        row_data.append(cell_text)
                    if row_data:  # Only add non-empty rows
                        table_data.append(row_data)
                
                if table_data:
                    # Convert table to text representation
                    table_text = self._format_table_as_text(table_data)
                    
                    # Create structured content for table
                    table_content = StructuredContent(
                        content_type=ContentType.TABLE,
                        text=table_text,
                        position={
                            'table_index': table_index,
                            'xpath': self._get_xpath(table)
                        },
                        attributes={
                            'rows': len(table_data),
                            'columns': len(table_data[0]) if table_data else 0,
                            'raw_data': table_data,
                            'html_attributes': dict(table.attrs) if hasattr(table, 'attrs') else {}
                        },
                        confidence=0.9
                    )
                    
                    document.structured_content.append(table_content)
        
        except Exception as e:
            document.warnings.append(f"Error extracting tables: {str(e)}")
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text"""
        if not table_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        max_cols = max(len(row) for row in table_data)
        
        for col_index in range(max_cols):
            max_width = max(
                len(str(row[col_index]) if col_index < len(row) else "")
                for row in table_data
            )
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        # Format table
        formatted_rows = []
        for row in table_data:
            formatted_cells = []
            for col_index in range(max_cols):
                cell_value = row[col_index] if col_index < len(row) else ""
                cell_str = str(cell_value).ljust(col_widths[col_index])
                formatted_cells.append(cell_str)
            formatted_rows.append(" | ".join(formatted_cells))
        
        return "\n".join(formatted_rows)
    
    def _calculate_confidence_score(self, 
                                   document: ProcessedDocument, 
                                   soup: BeautifulSoup) -> float:
        """Calculate confidence score for HTML processing"""
        
        if document.processing_status == ProcessingStatus.FAILED:
            return 0.0
        
        confidence_factors = []
        
        # Text extraction confidence
        if document.extracted_text:
            text_length = len(document.extracted_text.strip())
            if text_length > 500:
                confidence_factors.append(0.9)
            elif text_length > 100:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
        
        # Structured content confidence
        if document.structured_content:
            avg_content_confidence = sum(
                content.confidence for content in document.structured_content
            ) / len(document.structured_content)
            confidence_factors.append(avg_content_confidence)
        
        # HTML structure quality
        structure_score = self._assess_html_structure(soup)
        confidence_factors.append(structure_score)
        
        # Metadata confidence
        if document.metadata and document.metadata.title:
            confidence_factors.append(0.8)
        
        # Error penalty
        error_penalty = min(len(document.errors) * 0.1, 0.5)
        
        # Calculate final confidence
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, base_confidence - error_penalty)
        else:
            return 0.2  # Minimal confidence if no content extracted
    
    def _assess_html_structure(self, soup: BeautifulSoup) -> float:
        """Assess the quality of HTML structure"""
        
        score = 0.5  # Base score
        
        try:
            # Check for semantic HTML elements
            semantic_elements = soup.find_all(['main', 'article', 'section', 'header', 'footer', 'nav', 'aside'])
            if semantic_elements:
                score += 0.2
            
            # Check for proper heading hierarchy
            headings = soup.find_all(self.heading_tags)
            if headings:
                score += 0.1
            
            # Check for meta tags
            meta_tags = soup.find_all('meta')
            if len(meta_tags) > 3:  # Has reasonable meta information
                score += 0.1
            
            # Check for title
            if soup.find('title'):
                score += 0.1
            
        except Exception:
            pass
        
        return min(score, 1.0)
    
    async def extract_links(self, file_path: Path, base_url: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract all links from HTML document"""
        
        links = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all anchor tags with href
            anchor_tags = soup.find_all('a', href=True)
            
            for anchor in anchor_tags:
                href = anchor['href']
                text = anchor.get_text().strip()
                
                # Resolve relative URLs if base_url provided
                if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                    href = urljoin(base_url, href)
                
                link_info = {
                    'url': href,
                    'text': text,
                    'title': anchor.get('title', ''),
                    'target': anchor.get('target', '')
                }
                
                links.append(link_info)
            
        except Exception as e:
            raise Exception(f"Error extracting links: {str(e)}")
        
        return links
    
    async def extract_images_info(self, file_path: Path, base_url: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract information about images in HTML document"""
        
        images = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all img tags
            img_tags = soup.find_all('img')
            
            for img in img_tags:
                src = img.get('src', '')
                
                # Resolve relative URLs if base_url provided
                if base_url and src and not src.startswith(('http://', 'https://', 'data:')):
                    src = urljoin(base_url, src)
                
                image_info = {
                    'src': src,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                }
                
                images.append(image_info)
            
        except Exception as e:
            raise Exception(f"Error extracting image information: {str(e)}")
        
        return images