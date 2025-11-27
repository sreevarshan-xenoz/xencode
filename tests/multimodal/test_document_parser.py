"""Tests for Document Parser."""

import pytest
from pathlib import Path
import tempfile

from xencode.multimodal.document_parser import DocumentParser


class TestDocumentParser:
    """Test cases for DocumentParser."""

    def setup_method(self):
        """Set up test documents."""
        self.temp_dir = tempfile.mkdtemp()
        self.parser = DocumentParser()

    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/document.pdf")

    def test_unsupported_file_type(self):
        """Test parsing an unsupported file type."""
        temp_file = Path(self.temp_dir) / "test.txt"
        temp_file.write_text("Hello world")
        
        result = self.parser.parse(str(temp_file))
        assert result["error"] is not None
        assert "Unsupported file type" in result["error"]

    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "Hello\n\n\n\nWorld    with  spaces"
        clean = self.parser.clean_text(dirty_text)
        
        assert "\n\n\n" not in clean
        assert "  " not in clean
