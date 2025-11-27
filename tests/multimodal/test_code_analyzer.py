"""Tests for Code Analyzer."""

import pytest
from pathlib import Path

from xencode.multimodal.code_analyzer import CodeAnalyzer


class TestCodeAnalyzer:
    """Test cases for CodeAnalyzer."""

    def setup_method(self):
        """Set up analyzer."""
        self.analyzer = CodeAnalyzer()

    def test_analyze_directory_not_exists(self):
        """Test analyzing a non-existent directory."""
        with pytest.raises(ValueError):
            self.analyzer.analyze_directory("/nonexistent/directory")

    def test_analyze_xencode_codebase(self):
        """Test analyzing the xencode codebase itself."""
        # Analyze the xencode directory
        result = self.analyzer.analyze_directory("d:/xencode/xencode", max_depth=2)
        
        assert result["total_files"] > 0
        assert "Python" in result["languages"]
        assert result["total_lines"] > 0

    def test_analyze_python_file(self):
        """Test analyzing a Python file."""
        # Analyze the code_analyzer.py file itself
        result = self.analyzer.analyze_python_file("d:/xencode/xencode/multimodal/code_analyzer.py")
        
        assert result["filename"] == "code_analyzer.py"
        assert result["error"] is None
        assert "CodeAnalyzer" in result["classes"]
        assert len(result["functions"]) > 0
        assert result["lines_of_code"] > 0

    def test_analyze_nonexistent_python_file(self):
        """Test analyzing a non-existent Python file."""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze_python_file("/nonexistent/file.py")
