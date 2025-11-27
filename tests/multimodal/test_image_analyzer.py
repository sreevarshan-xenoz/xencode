"""Tests for Image Analyzer."""

import pytest
from pathlib import Path
import tempfile
from PIL import Image

from xencode.multimodal.image_analyzer import ImageAnalyzer


class TestImageAnalyzer:
    """Test cases for ImageAnalyzer."""

    def setup_method(self):
        """Set up test image."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.png"
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.test_image_path)
        
        self.analyzer = ImageAnalyzer()

    def test_analyze_existing_image(self):
        """Test analyzing an existing image."""
        result = self.analyzer.analyze(str(self.test_image_path))
        
        assert result["filename"] == "test_image.png"
        assert result["error"] is None
        assert result["metadata"]["format"] == "PNG"
        assert result["metadata"]["width"] == 100
        assert result["metadata"]["height"] == 100

    def test_analyze_nonexistent_image(self):
        """Test analyzing a non-existent image."""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze("/nonexistent/image.png")

    def test_metadata_extraction(self):
        """Test metadata extraction."""
        result = self.analyzer.analyze(str(self.test_image_path))
        
        metadata = result["metadata"]
        assert "format" in metadata
        assert "mode" in metadata
        assert "size" in metadata
        assert metadata["size"] == (100, 100)
