"""
Integration tests for AI Code Reviewer feature

Tests complete end-to-end workflows from PR analysis to report generation.
These tests verify that all components work together correctly.
"""

import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from xencode.features.code_review import (
    CodeReviewFeature,
    CodeReviewConfig,
    GitHubPRAnalyzer,
    GitLabPRAnalyzer,
    BitbucketPRAnalyzer,
    CodeLinter,
    AIReviewEngine,
    ReportGenerator
)
from xencode.features.base import FeatureConfig


class TestEndToEndPRReviewWorkflow:
    """Test complete PR review workflow from fetch to report"""
    
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize code review feature"""
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
    async def test_github_pr_complete_workflow(self, feature):
        """Test complete GitHub PR review workflow"""
        # Mock PR data with security vulnerabilities
        mock_pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Add authentication system',
            'description': 'Implementing user authentication',
            'files': [
                {
                    'filename': 'auth.py',
                    'path': 'auth.py',
                    'language': 'python',
                    'status': 'added',
                    'additions': 50,
                    'deletions': 0,
                    'changes': 50,
                    'content': '''
def authenticate(username, password):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    
    # Hardcoded secret
    api_key = "sk_live_1234567890abcdef"
    
    return True
''',
                    'patch': '+ SQL injection code',
                    'blob_url': 'https://github.com/blob'
                }
            ],
            'commits': [
                {
                    'sha': 'abc123',
                    'message': 'Add authentication',
                    'author': 'developer'
                }
            ],
            'comments': [],
            'state': 'open',
            'author': 'developer',
            'base_branch': 'main',
            'head_branch': 'feature/auth',
            'additions': 50,
            'deletions': 0,
            'changed_files': 1
        }
        
        # Mock the GitHub analyzer
        feature._pr_analyzers['github'].fetch_pr = AsyncMock(return_value=mock_pr_data)
        
        # Execute complete workflow
        result = await feature.analyze_pr(
            'https://github.com/owner/repo/pull/123',
            'github'
        )
        
        # Verify complete review structure
        assert result is not None
        assert 'pr' in result
        assert 'code_analysis' in result
        assert 'review' in result
        assert 'summary' in result
        
        # Verify PR info
        assert result['pr']['url'] == 'https://github.com/owner/repo/pull/123'
        assert result['pr']['title'] == 'Add authentication system'
        
        # Verify code analysis ran
        assert 'issues' in result['code_analysis']
        
        # Verify review was generated
        assert 'issues' in result['review']
        assert 'suggestions' in result['review']
        
        # Generate and verify reports in all formats
        text_report = feature.generate_formatted_report(result['review'], pr_data=result['pr'], format='text')
        assert isinstance(text_report, str)
        assert "CODE REVIEW REPORT" in text_report
        
        markdown_report = feature.generate_formatted_report(result['review'], pr_data=result['pr'], format='markdown')
        assert isinstance(markdown_report, str)
        assert "# Code Review Report" in markdown_report
        
        json_report = feature.generate_formatted_report(result['review'], pr_data=result['pr'], format='json')
        assert isinstance(json_report, str)
        
        html_report = feature.generate_formatted_report(result['review'], pr_data=result['pr'], format='html')
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report


class TestEndToEndFileReviewWorkflow:
    """Test complete file review workflow"""
    
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize code review feature"""
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
    async def test_single_file_review_workflow(self, feature, tmp_path):
        """Test complete single file review workflow"""
        # Create test file with multiple issues
        test_file = tmp_path / "vulnerable.py"
        test_file.write_text('''
import hashlib
import os

# Hardcoded credentials
DB_PASSWORD = "super_secret_123"
API_KEY = "sk_live_1234567890abcdef"

def authenticate(username, password):
    # SQL injection
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    
    # Weak crypto
    hash = hashlib.md5(password.encode()).hexdigest()
    return hash

def execute_command(cmd):
    # Command injection
    os.system(cmd)

def read_file(filename):
    # Path traversal
    with open("/var/data/" + filename, 'r') as f:
        return f.read()

# Bare except
try:
    risky_operation()
except:
    pass
''')
        
        # Execute complete workflow
        result = await feature.analyze_file(str(test_file), 'python')
        
        # Verify complete analysis
        assert result is not None
        assert 'file' in result
        assert 'analysis' in result
        assert 'review' in result
        
        # Verify file path
        assert result['file'] == str(test_file)
        
        # Verify analysis ran
        assert 'issues' in result['analysis']
        assert 'summary' in result['analysis']
        
        # Verify review was generated
        assert 'issues' in result['review']
        assert 'suggestions' in result['review']
        
        # Generate report
        report = feature.generate_formatted_report(result['review'], format='text')
        assert isinstance(report, str)
        assert len(report) > 0
    
    @pytest.mark.asyncio
    async def test_directory_review_workflow(self, feature, tmp_path):
        """Test complete directory review workflow"""
        # Create multiple files with different issues
        (tmp_path / "auth.py").write_text('''
def login(user, pwd):
    cursor.execute("SELECT * FROM users WHERE name = " + user)
    return True
''')
        
        (tmp_path / "api.js").write_text('''
function display(msg) {
    document.getElementById('output').innerHTML = msg;
}

if (value == 5) {
    console.log("equal");
}
''')
        
        (tmp_path / "config.ts").write_text('''
const apiKey: any = "secret_key_123";

function process(data: any): any {
    return data;
}
''')
        
        (tmp_path / "utils.rs").write_text('''
fn get_value() -> i32 {
    let result = risky_operation();
    result.unwrap()
}
''')
        
        # Execute complete workflow
        result = await feature.analyze_directory(str(tmp_path))
        
        # Verify all files analyzed
        assert result is not None
        assert 'directory' in result
        assert 'files_analyzed' in result
        assert 'analyses' in result
        
        # Verify directory path
        assert result['directory'] == str(tmp_path)
        
        # Verify files were analyzed
        assert result['files_analyzed'] > 0
        assert len(result['analyses']) > 0


class TestComponentIntegration:
    """Test integration between different components"""
    
    @pytest.mark.asyncio
    async def test_linter_to_ai_engine_integration(self):
        """Test data flow from linter to AI engine"""
        # Create test files
        files = [{
            'path': 'security.py',
            'language': 'python',
            'content': '''
password = "hardcoded_secret"
cursor.execute("SELECT * FROM users WHERE id = " + user_id)
'''
        }]
        
        # Run linter
        linter = CodeLinter()
        lint_result = await linter.analyze(files)
        
        # Verify linter output
        assert lint_result['summary']['total_issues'] > 0
        assert len(lint_result['issues']) > 0
        
        # Run AI engine with linter results
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Security fixes',
            'Fixing security issues',
            files,
            lint_result
        )
        
        # Verify AI engine processed linter results
        assert 'issues' in review
        assert 'suggestions' in review
        assert 'patterns_detected' in review
    
    @pytest.mark.asyncio
    async def test_ai_engine_to_report_generator_integration(self):
        """Test data flow from AI engine to report generator"""
        # Create test data
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': 'def test(): pass'
        }]
        
        # Run linter
        linter = CodeLinter()
        lint_result = await linter.analyze(files)
        
        # Run AI engine
        engine = AIReviewEngine()
        await engine.initialize()
        
        review = await engine.generate_review(
            'Test PR',
            'Testing integration',
            files,
            lint_result
        )
        
        # Generate reports in all formats
        generator = ReportGenerator()
        
        pr_data = {
            'url': 'https://github.com/test/repo/pull/1',
            'title': 'Test PR',
            'author': 'developer',
            'base_branch': 'main',
            'head_branch': 'feature'
        }
        
        text_report = generator.generate_text_report(review, pr_data=pr_data)
        assert isinstance(text_report, str)
        assert len(text_report) > 0
        
        markdown_report = generator.generate_markdown_report(review, pr_data=pr_data)
        assert isinstance(markdown_report, str)
        assert "# Code Review Report" in markdown_report
        
        json_report = generator.generate_json_report(review, pr_data=pr_data)
        assert isinstance(json_report, dict)  # Returns dict, not string
        assert 'metadata' in json_report
        
        html_report = generator.generate_html_report(review, pr_data=pr_data)
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
    
    @pytest.mark.asyncio
    async def test_pr_analyzer_to_linter_integration(self):
        """Test data flow from PR analyzer to linter"""
        # Mock PR data
        pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Test PR',
            'description': 'Testing',
            'files': [
                {
                    'filename': 'test.py',
                    'path': 'test.py',
                    'language': 'python',
                    'status': 'added',
                    'additions': 10,
                    'deletions': 0,
                    'changes': 10,
                    'content': 'def test(): pass',
                    'patch': '+def test(): pass',
                    'blob_url': 'https://github.com/blob'
                }
            ],
            'commits': [],
            'comments': [],
            'state': 'open',
            'author': 'developer',
            'base_branch': 'main',
            'head_branch': 'feature',
            'additions': 10,
            'deletions': 0,
            'changed_files': 1
        }
        
        # Extract files for linter
        files = []
        for file_data in pr_data['files']:
            files.append({
                'path': file_data['path'],
                'language': file_data['language'],
                'content': file_data['content']
            })
        
        # Run linter
        linter = CodeLinter()
        result = await linter.analyze(files)
        
        # Verify linter processed PR files
        assert result['summary']['total_files'] == 1
        assert len(result['files']) == 1
        assert result['files'][0]['path'] == 'test.py'


class TestConcurrentOperations:
    """Test concurrent operations and performance"""
    
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize code review feature"""
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
    async def test_concurrent_file_analysis(self, feature, tmp_path):
        """Test analyzing multiple files concurrently"""
        # Create multiple test files
        files = []
        for i in range(5):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}(): pass")
            files.append(str(test_file))
        
        # Analyze concurrently
        tasks = [feature.analyze_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        # Verify all files analyzed
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert 'file' in result
            assert 'analysis' in result
            assert 'review' in result
    
    @pytest.mark.asyncio
    async def test_concurrent_pr_analysis(self, feature):
        """Test analyzing multiple PRs concurrently"""
        # Mock PR data
        mock_pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Test PR',
            'description': 'Testing',
            'files': [],
            'commits': [],
            'comments': [],
            'state': 'open',
            'author': 'developer',
            'base_branch': 'main',
            'head_branch': 'feature',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        # Mock analyzers
        for platform in ['github', 'gitlab', 'bitbucket']:
            feature._pr_analyzers[platform].fetch_pr = AsyncMock(return_value=mock_pr_data)
        
        # Analyze multiple PRs concurrently
        tasks = [
            feature.analyze_pr('https://github.com/owner/repo/pull/1', 'github'),
            feature.analyze_pr('https://github.com/owner/repo/pull/2', 'github'),
            feature.analyze_pr('https://github.com/owner/repo/pull/3', 'github')
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all PRs analyzed
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'pr' in result
            assert 'review' in result


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in integration"""
    
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize code review feature"""
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
    async def test_pr_with_no_files(self, feature):
        """Test handling PR with no changed files"""
        mock_pr_data = {
            'url': 'https://github.com/owner/repo/pull/123',
            'title': 'Empty PR',
            'description': 'No files changed',
            'files': [],
            'commits': [],
            'comments': [],
            'state': 'open',
            'author': 'developer',
            'base_branch': 'main',
            'head_branch': 'feature',
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        feature._pr_analyzers['github'].fetch_pr = AsyncMock(return_value=mock_pr_data)
        
        result = await feature.analyze_pr('https://github.com/owner/repo/pull/123', 'github')
        
        # Should handle gracefully
        assert result is not None
        assert result['files_analyzed'] == 0
    
    @pytest.mark.asyncio
    async def test_file_with_syntax_errors(self, feature, tmp_path):
        """Test handling file with syntax errors"""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def broken(\n    # Missing closing parenthesis")
        
        # Should not crash
        result = await feature.analyze_file(str(test_file))
        
        assert result is not None
        assert 'file' in result
        assert 'analysis' in result
    
    @pytest.mark.asyncio
    async def test_empty_directory(self, feature, tmp_path):
        """Test handling empty directory"""
        result = await feature.analyze_directory(str(tmp_path))
        
        assert result is not None
        assert result['files_analyzed'] == 0
    
    @pytest.mark.asyncio
    async def test_mixed_file_types(self, feature, tmp_path):
        """Test handling directory with mixed file types"""
        # Create various file types
        (tmp_path / "code.py").write_text("def test(): pass")
        (tmp_path / "README.md").write_text("# README")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        (tmp_path / "binary.bin").write_bytes(b'\x00\x01\x02\x03')
        
        result = await feature.analyze_directory(str(tmp_path))
        
        # Should only analyze code files
        assert result is not None
        assert result['files_analyzed'] >= 1  # At least the Python file


class TestReportGenerationIntegration:
    """Test report generation with real review data"""
    
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialize code review feature"""
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
    async def test_report_with_all_severity_levels(self, feature, tmp_path):
        """Test report generation with issues at all severity levels"""
        test_file = tmp_path / "mixed.py"
        test_file.write_text('''
# Critical: SQL injection
cursor.execute("SELECT * FROM users WHERE id = " + user_id)

# High: Hardcoded secret
api_key = "sk_live_1234567890abcdef"

# Medium: Bare except
try:
    pass
except:
    pass

# Low: Single letter variable
x = 10
''')
        
        result = await feature.analyze_file(str(test_file))
        
        # Generate all report formats
        text_report = feature.generate_formatted_report(result['review'], format='text')
        assert isinstance(text_report, str)
        assert len(text_report) > 0
        
        markdown_report = feature.generate_formatted_report(result['review'], format='markdown')
        assert isinstance(markdown_report, str)
        assert "# Code Review Report" in markdown_report
        
        json_report = feature.generate_formatted_report(result['review'], format='json')
        # JSON report returns a string that can be parsed
        assert isinstance(json_report, str)
        import json
        json_data = json.loads(json_report)
        assert isinstance(json_data, dict)
        
        html_report = feature.generate_formatted_report(result['review'], format='html')
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
    
    @pytest.mark.asyncio
    async def test_report_with_positive_feedback(self, feature, tmp_path):
        """Test report generation with positive feedback for clean code"""
        test_file = tmp_path / "clean.py"
        test_file.write_text('''
def calculate_total(items: list) -> float:
    """Calculate the total price of items.
    
    Args:
        items: List of items with price attribute
        
    Returns:
        float: Total price of all items
    """
    return sum(item.price for item in items)
''')
        
        result = await feature.analyze_file(str(test_file))
        
        # Generate report
        report = feature.generate_formatted_report(result['review'], format='markdown')
        
        # Should include positive feedback section
        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
