#!/usr/bin/env python3
"""
Text Document Processor

Implements processing for plain text files, markdown, and code files.
Provides basic text extraction and simple structured content detection.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class TextProcessor(ProcessorInterface):
    """Text document processor for plain text, markdown, and code files"""
    
    def __init__(self):
        self.supported_types = [DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.CODE]
        
        # Markdown patterns
        self.markdown_patterns = {
            'heading': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`]+`', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE),
            'link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'bold': re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__'),
            'italic': re.compile(r'\*([^*]+)\*|_([^_]+)_')
        }
        
        # Code file extensions and their languages
        self.code_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.sh': 'bash',
            '.ps1': 'powershell'
        }
    
    async def can_process(self, file_path: Path, document_type: DocumentType) -> bool:
        """Check if this processor can handle the document"""
        if document_type not in self.supported_types:
            return False
        
        try:
            # Try to read the file as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to verify it's readable text
                f.read(1000)
            return True
        except Exception:
            return False
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return self.supported_types.copy()
    
    async def process(self, 
                     file_path: Path, 
                     options: ProcessingOptions) -> ProcessedDocument:
        """Process text document and extract structured content"""
        
        document = ProcessedDocument(
            original_filename=file_path.name,
            file_path=str(file_path),
            document_type=self._detect_specific_type(file_path),
            processing_status=ProcessingStatus.PROCESSING
        )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract metadata
            if options.extract_metadata:
                document.metadata = await self._extract_metadata(content, file_path)
            
            # Set extracted text
            if options.extract_text:
                document.extracted_text = content
            
            # Extract structured content
            if options.extract_structured_content:
                await self._extract_structured_content(content, document, options)
            
            # Calculate confidence score
            document.confidence_score = self._calculate_confidence_score(document)
            
            # Set processing status
            document.processing_status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            document.errors.append(f"Text processing error: {str(e)}")
            document.confidence_score = 0.0
        
        return document
    
    def _detect_specific_type(self, file_path: Path) -> DocumentType:
        """Detect specific document type based on file extension"""
        extension = file_path.suffix.lower()
        
        if extension in ['.md', '.markdown']:
            return DocumentType.MARKDOWN
        elif extension in self.code_languages:
            return DocumentType.CODE
        else:
            return DocumentType.TEXT
    
    async def _extract_metadata(self, 
                               content: str, 
                               file_path: Path) -> DocumentMetadata:
        """Extract metadata from text content"""
        metadata = DocumentMetadata()
        
        try:
            # Basic file info
            metadata.file_size = file_path.stat().st_size
            metadata.mime_type = 'text/plain'
            
            # Word and line count
            lines = content.split('\n')
            metadata.word_count = len(content.split())
            
            # For markdown files, try to extract title from first heading
            if file_path.suffix.lower() in ['.md', '.markdown']:
                metadata.mime_type = 'text/markdown'
                
                # Look for first heading as title
                heading_match = self.markdown_patterns['heading'].search(content)
                if heading_match:
                    metadata.title = heading_match.group(2).strip()
            
            # For code files, detect language
            elif file_path.suffix.lower() in self.code_languages:
                metadata.language = self.code_languages[file_path.suffix.lower()]
                metadata.mime_type = f'text/x-{metadata.language}'
            
            # Try to detect encoding
            metadata.encoding = 'utf-8'  # Assumed since we read successfully
            
        except Exception as e:
            # Continue processing even if metadata extraction fails
            pass
        
        return metadata
    
    async def _extract_structured_content(self, 
                                         content: str,
                                         document: ProcessedDocument,
                                         options: ProcessingOptions) -> None:
        """Extract structured content from text"""
        
        lines = content.split('\n')
        document_type = document.document_type
        
        if document_type == DocumentType.MARKDOWN:
            await self._extract_markdown_content(content, document)
        elif document_type == DocumentType.CODE:
            await self._extract_code_content(content, lines, document)
        else:
            await self._extract_plain_text_content(lines, document)
    
    async def _extract_markdown_content(self, 
                                       content: str,
                                       document: ProcessedDocument) -> None:
        """Extract structured content from markdown"""
        
        try:
            # Extract headings
            for match in self.markdown_patterns['heading'].finditer(content):
                heading_level = len(match.group(1))  # Number of # symbols
                heading_text = match.group(2).strip()
                
                structured_content = StructuredContent(
                    content_type=ContentType.HEADING,
                    text=heading_text,
                    attributes={
                        'level': heading_level,
                        'markdown_syntax': match.group(1)
                    },
                    confidence=0.95
                )
                document.structured_content.append(structured_content)
            
            # Extract code blocks
            for match in self.markdown_patterns['code_block'].finditer(content):
                code_text = match.group(0)
                
                # Determine if it's inline code or code block
                if code_text.startswith('```'):
                    # Code block
                    code_content = code_text.strip('`').strip()
                    # Try to extract language from first line
                    lines = code_content.split('\n')
                    language = lines[0].strip() if lines and not lines[0].strip().startswith(' ') else None
                    if language and len(language.split()) == 1 and len(language) < 20:
                        code_content = '\n'.join(lines[1:])
                    else:
                        language = None
                    
                    structured_content = StructuredContent(
                        content_type=ContentType.CODE_BLOCK,
                        text=code_content,
                        attributes={
                            'language': language,
                            'block_type': 'fenced'
                        },
                        confidence=0.9
                    )
                else:
                    # Inline code
                    code_content = code_text.strip('`')
                    structured_content = StructuredContent(
                        content_type=ContentType.CODE_BLOCK,
                        text=code_content,
                        attributes={
                            'block_type': 'inline'
                        },
                        confidence=0.8
                    )
                
                document.structured_content.append(structured_content)
            
            # Extract list items
            for match in self.markdown_patterns['list_item'].finditer(content):
                list_text = match.group(1).strip()
                
                structured_content = StructuredContent(
                    content_type=ContentType.LIST,
                    text=list_text,
                    attributes={
                        'list_type': 'unordered'
                    },
                    confidence=0.9
                )
                document.structured_content.append(structured_content)
            
            # Extract numbered list items
            for match in self.markdown_patterns['numbered_list'].finditer(content):
                list_text = match.group(1).strip()
                
                structured_content = StructuredContent(
                    content_type=ContentType.LIST,
                    text=list_text,
                    attributes={
                        'list_type': 'ordered'
                    },
                    confidence=0.9
                )
                document.structured_content.append(structured_content)
            
            # Extract links
            for match in self.markdown_patterns['link'].finditer(content):
                link_text = match.group(1)
                link_url = match.group(2)
                
                structured_content = StructuredContent(
                    content_type=ContentType.LINK,
                    text=link_text,
                    attributes={
                        'url': link_url,
                        'type': 'markdown_link'
                    },
                    confidence=0.95
                )
                document.structured_content.append(structured_content)
                
        except Exception as e:
            document.warnings.append(f"Error extracting markdown content: {str(e)}")
    
    async def _extract_code_content(self, 
                                   content: str,
                                   lines: List[str],
                                   document: ProcessedDocument) -> None:
        """Extract structured content from code files"""
        
        try:
            # Treat entire content as code block
            language = document.metadata.language if document.metadata else None
            
            structured_content = StructuredContent(
                content_type=ContentType.CODE_BLOCK,
                text=content,
                attributes={
                    'language': language,
                    'file_type': 'source_code',
                    'line_count': len(lines)
                },
                confidence=0.95
            )
            document.structured_content.append(structured_content)
            
            # Try to extract comments as separate content
            await self._extract_code_comments(lines, document, language)
            
        except Exception as e:
            document.warnings.append(f"Error extracting code content: {str(e)}")
    
    async def _extract_code_comments(self, 
                                    lines: List[str],
                                    document: ProcessedDocument,
                                    language: Optional[str]) -> None:
        """Extract comments from code"""
        
        try:
            comment_patterns = {
                'python': [r'#(.+)$', r'"""([\s\S]*?)"""', r"'''([\s\S]*?)'''"],
                'javascript': [r'//(.+)$', r'/\*([\s\S]*?)\*/'],
                'java': [r'//(.+)$', r'/\*([\s\S]*?)\*/'],
                'cpp': [r'//(.+)$', r'/\*([\s\S]*?)\*/'],
                'c': [r'//(.+)$', r'/\*([\s\S]*?)\*/'],
                'csharp': [r'//(.+)$', r'/\*([\s\S]*?)\*/'],
                'css': [r'/\*([\s\S]*?)\*/'],
                'html': [r'<!--([\s\S]*?)-->'],
                'sql': [r'--(.+)$', r'/\*([\s\S]*?)\*/'],
                'bash': [r'#(.+)$'],
                'powershell': [r'#(.+)$', r'<#([\s\S]*?)#>']
            }
            
            if not language or language not in comment_patterns:
                return
            
            content = '\n'.join(lines)
            patterns = comment_patterns[language]
            
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    comment_text = match.group(1).strip()
                    
                    if len(comment_text) > 10:  # Only capture substantial comments
                        structured_content = StructuredContent(
                            content_type=ContentType.TEXT,  # Comments are text content
                            text=comment_text,
                            attributes={
                                'content_subtype': 'comment',
                                'language': language
                            },
                            confidence=0.8
                        )
                        document.structured_content.append(structured_content)
                        
        except Exception as e:
            document.warnings.append(f"Error extracting code comments: {str(e)}")
    
    async def _extract_plain_text_content(self, 
                                         lines: List[str],
                                         document: ProcessedDocument) -> None:
        """Extract structured content from plain text"""
        
        try:
            # Group lines into paragraphs (separated by empty lines)
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_paragraph.append(line)
                else:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
            
            # Add final paragraph if exists
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            # Create structured content for each paragraph
            for para_index, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 10:  # Only substantial paragraphs
                    
                    # Try to detect if it's a heading (short, title-case, etc.)
                    content_type = ContentType.TEXT
                    if (len(paragraph) < 100 and 
                        para_index == 0 and 
                        paragraph.istitle()):
                        content_type = ContentType.HEADING
                    
                    structured_content = StructuredContent(
                        content_type=content_type,
                        text=paragraph,
                        position={
                            'paragraph_index': para_index
                        },
                        confidence=0.7
                    )
                    document.structured_content.append(structured_content)
                    
        except Exception as e:
            document.warnings.append(f"Error extracting plain text content: {str(e)}")
    
    def _calculate_confidence_score(self, document: ProcessedDocument) -> float:
        """Calculate confidence score for text processing"""
        
        if document.processing_status == ProcessingStatus.FAILED:
            return 0.0
        
        confidence_factors = []
        
        # Text extraction confidence (always high for text files)
        if document.extracted_text:
            confidence_factors.append(0.95)
        
        # Structured content confidence
        if document.structured_content:
            avg_content_confidence = sum(
                content.confidence for content in document.structured_content
            ) / len(document.structured_content)
            confidence_factors.append(avg_content_confidence)
        
        # Document type specific bonuses
        if document.document_type == DocumentType.MARKDOWN:
            # Markdown has clear structure, higher confidence
            confidence_factors.append(0.9)
        elif document.document_type == DocumentType.CODE:
            # Code files are well-structured
            confidence_factors.append(0.85)
        
        # Error penalty
        error_penalty = min(len(document.errors) * 0.1, 0.5)
        
        # Calculate final confidence
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            return max(0.0, base_confidence - error_penalty)
        else:
            return 0.8  # Good baseline for text files