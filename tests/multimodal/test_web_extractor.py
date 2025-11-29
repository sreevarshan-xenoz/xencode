"""Tests for Web Extractor."""

import pytest
from unittest.mock import Mock, patch
from xencode.multimodal.web_extractor import WebExtractor

class TestWebExtractor:
    """Test cases for WebExtractor."""

    def setup_method(self):
        """Set up extractor."""
        self.extractor = WebExtractor()

    @patch('xencode.multimodal.web_extractor.requests.get')
    def test_extract_success(self, mock_get):
        """Test successful content extraction."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = b"""
            <html>
                <head>
                    <title>Test Page</title>
                    <meta name="description" content="Test description">
                </head>
                <body>
                    <main>
                        <h1>Header</h1>
                        <p>Main content paragraph.</p>
                    </main>
                </body>
            </html>
        """
        mock_get.return_value = mock_response

        result = self.extractor.extract("http://example.com")

        assert result["url"] == "http://example.com"
        assert result["title"] == "Test Page"
        assert result["metadata"]["description"] == "Test description"
        assert "Main content paragraph" in result["text"]
        assert result["error"] is None

    @patch('xencode.multimodal.web_extractor.requests.get')
    def test_extract_error(self, mock_get):
        """Test extraction with network error."""
        mock_get.side_effect = Exception("Network error")

        result = self.extractor.extract("http://example.com")

        assert result["error"] is not None
        assert "Network error" in result["error"]

    @patch('xencode.multimodal.web_extractor.BS4_AVAILABLE', False)
    def test_missing_dependency(self):
        """Test behavior when BeautifulSoup is missing."""
        result = self.extractor.extract("http://example.com")
        
        assert result["error"] is not None
        assert "beautifulsoup4 is not installed" in result["error"]

    @patch('xencode.multimodal.web_extractor.requests.get')
    def test_extract_no_main_content(self, mock_get):
        """Test extraction when no main tag exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>Just body content</p></body></html>"
        mock_get.return_value = mock_response

        result = self.extractor.extract("http://example.com")
        
        assert "Just body content" in result["text"]
