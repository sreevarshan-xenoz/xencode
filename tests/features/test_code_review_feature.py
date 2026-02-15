"""
Unit tests for CodeReviewFeature class

Tests the main CodeReviewFeature class methods and integration points.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from xencode.features.code_review import CodeReviewFeature, CodeReviewConfig
from xencode.features.base import FeatureConfig


class TestCodeReviewFeature:
    """Tests for CodeReviewFeature class"""
    
    @pytest.fixture
    def feature_config(self):
        """Create feature configuration"""
        return FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig(
                supported_languages=['python', 'javascript', 'typescript', 'rust', 'go'],
                severity_levels=['critical', 'high', 'medium', 'low'],
                integrations=['github', 'gitlab', 'bitbucket']
            )
        )
    
    def test_feature_name(self, feature_config):
        """Test feature name property"""
        feature = CodeReviewFeature(feature_config)
        assert feature.name == "code_review"
    
    def test_feature_description(self, feature_config):
        """Test feature description property"""
        feature = CodeReviewFeature(feature_config)
        assert "AI-powered code review" in feature.description
    
    @pytest.mark.asyncio
    async def test_initialize(self, feature_config):
        """Test feature initialization"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Check that analyzers are initialized
        assert 'github' in feature._pr_analyzers
        assert 'gitlab' in feature._pr_analyzers
        assert 'bitbucket' in feature._pr_analyzers
        
        # Check that linter is initialized
        assert feature._linter is not None
        
        # Check that AI engine is initialized
        assert feature._ai_engine is not None
        
        # Check that report generator is initialized
        assert feature._report_generator is not None
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, feature_config):
        """Test feature shutdown"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Shutdown
        await feature._shutdown()
        
        # Verify sessions are closed
        for analyzer in feature._pr_analyzers.values():
            if hasattr(analyzer, '_session') and analyzer._session:
                assert analyzer._session.closed
    
    @pytest.mark.asyncio
    async def test_analyze_pr_github(self, feature_config):
        """Test analyzing GitHub PR"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Mock the GitHub analyzer
        mock_pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Test PR',
            'description': 'Test description',
            'files': [],
            'commits': [],
            'comments': [],
            'state': 'open',
            'author': 'test-user',
            'base_branch': 'main',
            'head_branch': 'feature',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        feature._pr_analyzers['github'].fetch_pr = AsyncMock(return_value=mock_pr_data)
        
        result = await feature.analyze_pr('https://github.com/owner/repo/pull/123', 'github')
        
        assert result is not None
        assert 'summary' in result
        assert 'issues' in result
        assert 'suggestions' in result
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_pr_invalid_platform(self, feature_config):
        """Test analyzing PR with invalid platform"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        with pytest.raises(ValueError, match="Unsupported platform"):
            await feature.analyze_pr('https://example.com/pr/123', 'invalid')
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_pr_fetch_failure(self, feature_config):
        """Test handling PR fetch failure"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        feature._pr_analyzers['github'].fetch_pr = AsyncMock(side_effect=Exception("API Error"))
        
        with pytest.raises(Exception, match="API Error"):
            await feature.analyze_pr('https://github.com/owner/repo/pull/123', 'github')
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_file_python(self, feature_config, tmp_path):
        """Test analyzing a Python file"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def add(a, b):
    return a + b
""")
        
        result = await feature.analyze_file(str(test_file), 'python')
        
        assert result is not None
        assert 'issues' in result
        assert 'suggestions' in result
        assert 'file' in result
        assert result['file'] == str(test_file)
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_file_auto_detect_language(self, feature_config, tmp_path):
        """Test analyzing file with auto-detected language"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        test_file = tmp_path / "test.js"
        test_file.write_text("console.log('hello');")
        
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert result['language'] == 'javascript'
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_file_nonexistent(self, feature_config):
        """Test analyzing non-existent file"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        with pytest.raises(FileNotFoundError):
            await feature.analyze_file('/nonexistent/file.py')
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_directory(self, feature_config, tmp_path):
        """Test analyzing a directory"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Create test files
        (tmp_path / "file1.py").write_text("def test1(): pass")
        (tmp_path / "file2.py").write_text("def test2(): pass")
        (tmp_path / "file3.js").write_text("function test3() {}")
        
        result = await feature.analyze_directory(str(tmp_path))
        
        assert result is not None
        assert 'files' in result
        assert 'issues' in result
        assert 'suggestions' in result
        assert len(result['files']) == 3
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_directory_with_patterns(self, feature_config, tmp_path):
        """Test analyzing directory with file patterns"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        # Create test files
        (tmp_path / "file1.py").write_text("def test1(): pass")
        (tmp_path / "file2.py").write_text("def test2(): pass")
        (tmp_path / "file3.js").write_text("function test3() {}")
        
        result = await feature.analyze_directory(str(tmp_path), patterns=['*.py'])
        
        assert result is not None
        assert len(result['files']) == 2  # Only Python files
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_directory_empty(self, feature_config, tmp_path):
        """Test analyzing empty directory"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        result = await feature.analyze_directory(str(tmp_path))
        
        assert result is not None
        assert len(result['files']) == 0
        
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_directory_nonexistent(self, feature_config):
        """Test analyzing non-existent directory"""
        feature = CodeReviewFeature(feature_config)
        await feature._initialize()
        
        with pytest.raises(FileNotFoundError):
            await feature.analyze_directory('/nonexistent/directory')
        
        await feature._shutdown()
    
    def test_detect_language_python(self, feature_config):
        """Test language detection for Python files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.py')) == 'python'
        assert feature._detect_language(Path('test.pyw')) == 'python'
    
    def test_detect_language_javascript(self, feature_config):
        """Test language detection for JavaScript files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.js')) == 'javascript'
        assert feature._detect_language(Path('test.jsx')) == 'javascript'
    
    def test_detect_language_typescript(self, feature_config):
        """Test language detection for TypeScript files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.ts')) == 'typescript'
        assert feature._detect_language(Path('test.tsx')) == 'typescript'
    
    def test_detect_language_rust(self, feature_config):
        """Test language detection for Rust files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.rs')) == 'rust'
    
    def test_detect_language_go(self, feature_config):
        """Test language detection for Go files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.go')) == 'go'
    
    def test_detect_language_unknown(self, feature_config):
        """Test language detection for unknown files"""
        feature = CodeReviewFeature(feature_config)
        assert feature._detect_language(Path('test.xyz')) == 'unknown'
    
    def test_generate_formatted_report_text(self, feature_config):
        """Test generating text format report"""
        feature = CodeReviewFeature(feature_config)
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = feature.generate_formatted_report(review, format='text')
        
        assert isinstance(report, str)
        assert "CODE REVIEW REPORT" in report
    
    def test_generate_formatted_report_markdown(self, feature_config):
        """Test generating markdown format report"""
        feature = CodeReviewFeature(feature_config)
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = feature.generate_formatted_report(review, format='markdown')
        
        assert isinstance(report, str)
        assert "# Code Review Report" in report
    
    def test_generate_formatted_report_json(self, feature_config):
        """Test generating JSON format report"""
        feature = CodeReviewFeature(feature_config)
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = feature.generate_formatted_report(review, format='json')
        
        assert isinstance(report, str)
        # Should be valid JSON
        import json
        data = json.loads(report)
        assert 'metadata' in data
    
    def test_generate_formatted_report_html(self, feature_config):
        """Test generating HTML format report"""
        feature = CodeReviewFeature(feature_config)
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = feature.generate_formatted_report(review, format='html')
        
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
    
    def test_generate_formatted_report_invalid_format(self, feature_config):
        """Test generating report with invalid format"""
        feature = CodeReviewFeature(feature_config)
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        with pytest.raises(ValueError, match="Unsupported format"):
            feature.generate_formatted_report(review, format='xml')
    
    def test_get_cli_commands(self, feature_config):
        """Test getting CLI commands"""
        feature = CodeReviewFeature(feature_config)
        commands = feature.get_cli_commands()
        
        assert isinstance(commands, list)
        # CLI commands are defined in cli.py, so this returns empty list
        assert len(commands) == 0
    
    def test_get_tui_components(self, feature_config):
        """Test getting TUI components"""
        feature = CodeReviewFeature(feature_config)
        components = feature.get_tui_components()
        
        assert isinstance(components, list)
        assert len(components) > 0
    
    def test_get_api_endpoints(self, feature_config):
        """Test getting API endpoints"""
        feature = CodeReviewFeature(feature_config)
        endpoints = feature.get_api_endpoints()
        
        assert isinstance(endpoints, list)
        # API endpoints not implemented yet
        assert len(endpoints) == 0


class TestCodeReviewConfig:
    """Tests for CodeReviewConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = CodeReviewConfig()
        
        assert 'python' in config.supported_languages
        assert 'javascript' in config.supported_languages
        assert 'typescript' in config.supported_languages
        
        assert 'critical' in config.severity_levels
        assert 'high' in config.severity_levels
        assert 'medium' in config.severity_levels
        assert 'low' in config.severity_levels
        
        assert 'github' in config.integrations
        assert 'gitlab' in config.integrations
        assert 'bitbucket' in config.integrations
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CodeReviewConfig(
            supported_languages=['python', 'rust'],
            severity_levels=['critical', 'high'],
            integrations=['github']
        )
        
        assert config.supported_languages == ['python', 'rust']
        assert config.severity_levels == ['critical', 'high']
        assert config.integrations == ['github']


class TestCodeReviewFeatureEdgeCases:
    """Tests for edge cases in CodeReviewFeature"""
    
    @pytest.fixture
    async def feature(self):
        """Create feature instance"""
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig()
        )
        feature = CodeReviewFeature(config)
        await feature._initialize()
        yield feature
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_unicode(self, feature, tmp_path):
        """Test analyzing file with Unicode content"""
        test_file = tmp_path / "unicode.py"
        test_file.write_text("# 你好世界\ndef hello():\n    print('Hello 世界')")
        
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert 'issues' in result
    
    @pytest.mark.asyncio
    async def test_analyze_file_empty(self, feature, tmp_path):
        """Test analyzing empty file"""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")
        
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert 'issues' in result
    
    @pytest.mark.asyncio
    async def test_analyze_file_very_large(self, feature, tmp_path):
        """Test analyzing very large file"""
        test_file = tmp_path / "large.py"
        # Create a large file with many lines
        content = "\n".join([f"def func{i}(): pass" for i in range(1000)])
        test_file.write_text(content)
        
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert 'issues' in result
    
    @pytest.mark.asyncio
    async def test_analyze_directory_with_subdirectories(self, feature, tmp_path):
        """Test analyzing directory with subdirectories"""
        # Create nested structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        
        result = await feature.analyze_directory(str(tmp_path))
        
        assert result is not None
        assert len(result['files']) == 2
    
    @pytest.mark.asyncio
    async def test_analyze_directory_with_hidden_files(self, feature, tmp_path):
        """Test analyzing directory with hidden files"""
        (tmp_path / ".hidden.py").write_text("def hidden(): pass")
        (tmp_path / "visible.py").write_text("def visible(): pass")
        
        result = await feature.analyze_directory(str(tmp_path))
        
        # Hidden files should be included
        assert result is not None
        assert len(result['files']) >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_pr_with_no_files(self, feature):
        """Test analyzing PR with no changed files"""
        mock_pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Empty PR',
            'description': 'No files changed',
            'files': [],
            'commits': [],
            'comments': [],
            'state': 'open',
            'author': 'test-user',
            'base_branch': 'main',
            'head_branch': 'feature',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        feature._pr_analyzers['github'].fetch_pr = AsyncMock(return_value=mock_pr_data)
        
        result = await feature.analyze_pr('https://github.com/owner/repo/pull/123', 'github')
        
        assert result is not None
        assert len(result['issues']) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_syntax_errors(self, feature, tmp_path):
        """Test analyzing file with syntax errors"""
        test_file = tmp_path / "syntax_error.py"
        test_file.write_text("def broken(\n    # Missing closing parenthesis")
        
        # Should not crash, just analyze what it can
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert 'issues' in result
    
    @pytest.mark.asyncio
    async def test_analyze_binary_file(self, feature, tmp_path):
        """Test analyzing binary file"""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        # Should handle gracefully
        try:
            result = await feature.analyze_file(str(test_file))
            assert result is not None
        except UnicodeDecodeError:
            # Expected for binary files
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, feature, tmp_path):
        """Test concurrent file analysis"""
        import asyncio
        
        # Create multiple test files
        files = []
        for i in range(5):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}(): pass")
            files.append(str(test_file))
        
        # Analyze concurrently
        tasks = [feature.analyze_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert 'issues' in result


class TestCodeReviewFeatureIntegration:
    """Integration tests for CodeReviewFeature with real components"""
    
    @pytest.fixture
    async def feature(self):
        """Create feature instance"""
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig()
        )
        feature = CodeReviewFeature(config)
        await feature._initialize()
        yield feature
        await feature._shutdown()
    
    @pytest.mark.asyncio
    async def test_full_file_analysis_workflow(self, feature, tmp_path):
        """Test complete file analysis workflow"""
        # Create a file with security issues
        test_file = tmp_path / "vulnerable.py"
        test_file.write_text('''
def authenticate(username, password):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    
    # Hardcoded secret
    api_key = "sk_live_1234567890abcdef"
    
    return True
''')
        
        # Analyze the file
        result = await feature.analyze_file(str(test_file))
        
        # Verify complete analysis
        assert result is not None
        assert 'issues' in result
        assert 'suggestions' in result
        assert 'patterns_detected' in result
        
        # Should detect security issues
        assert len(result['issues']) > 0
        
        # Should have suggestions
        assert len(result['suggestions']) > 0
        
        # Generate report
        report = feature.generate_formatted_report(result, format='text')
        assert "SQL injection" in report or "sqli" in report.lower()
    
    @pytest.mark.asyncio
    async def test_full_directory_analysis_workflow(self, feature, tmp_path):
        """Test complete directory analysis workflow"""
        # Create multiple files with various issues
        (tmp_path / "auth.py").write_text('''
def login(user, pwd):
    cursor.execute("SELECT * FROM users WHERE name = " + user)
''')
        
        (tmp_path / "api.js").write_text('''
function display(msg) {
    document.getElementById('output').innerHTML = msg;
}
''')
        
        # Analyze directory
        result = await feature.analyze_directory(str(tmp_path))
        
        # Verify complete analysis
        assert result is not None
        assert 'files' in result
        assert 'issues' in result
        assert 'suggestions' in result
        
        # Should analyze both files
        assert len(result['files']) == 2
        
        # Should detect issues in both files
        assert len(result['issues']) > 0
        
        # Generate report
        report = feature.generate_formatted_report(result, format='markdown')
        assert "# Code Review Report" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
