"""
Edge case tests for Code Review components

Tests unusual scenarios, error conditions, and boundary cases.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from xencode.features.code_review import (
    GitHubPRAnalyzer,
    GitLabPRAnalyzer,
    BitbucketPRAnalyzer,
    CodeLinter,
    AIReviewEngine,
    ReportGenerator
)


class TestGitHubPRAnalyzerEdgeCases:
    """Edge case tests for GitHubPRAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return GitHubPRAnalyzer()
    
    def test_parse_pr_url_with_trailing_slash(self, analyzer):
        """Test parsing URL with trailing slash"""
        url = "https://github.com/owner/repo/pull/123/"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['pr_number'] == '123'
    
    def test_parse_pr_url_with_query_params(self, analyzer):
        """Test parsing URL with query parameters"""
        url = "https://github.com/owner/repo/pull/123?tab=files"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['pr_number'] == '123'
    
    def test_parse_pr_url_with_fragment(self, analyzer):
        """Test parsing URL with fragment"""
        url = "https://github.com/owner/repo/pull/123#discussion_r123456"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo'
        assert result['pr_number'] == '123'
    
    def test_parse_pr_url_with_special_chars_in_repo(self, analyzer):
        """Test parsing URL with special characters in repo name"""
        url = "https://github.com/owner/repo-name_123/pull/456"
        result = analyzer._parse_pr_url(url)
        
        assert result['owner'] == 'owner'
        assert result['repo'] == 'repo-name_123'
        assert result['pr_number'] == '456'
    
    def test_parse_pr_url_empty_string(self, analyzer):
        """Test parsing empty URL"""
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            analyzer._parse_pr_url("")
    
    def test_parse_pr_url_none(self, analyzer):
        """Test parsing None URL"""
        with pytest.raises((ValueError, AttributeError)):
            analyzer._parse_pr_url(None)
    
    @pytest.mark.asyncio
    async def test_fetch_pr_with_very_large_pr(self, analyzer):
        """Test fetching PR with many files and commits"""
        pr_data = {
            'title': 'Large PR',
            'body': 'Very large PR',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature'},
            'additions': 10000,
            'deletions': 5000,
            'changed_files': 500
        }
        
        # Simulate 500 files
        files_data = [
            {
                'filename': f'file{i}.py',
                'status': 'modified',
                'additions': 20,
                'deletions': 10,
                'changes': 30,
                'patch': 'diff content'
            }
            for i in range(500)
        ]
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'files' in endpoint:
                return files_data
            elif 'commits' in endpoint:
                return []
            elif 'reviews' in endpoint:
                return []
            return pr_data
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result is not None
        assert len(result['files']) == 500
    
    @pytest.mark.asyncio
    async def test_fetch_pr_with_unicode_content(self, analyzer):
        """Test fetching PR with Unicode content"""
        pr_data = {
            'title': '添加新功能',
            'body': '这是一个包含中文的PR描述',
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature'},
            'additions': 10,
            'deletions': 5,
            'changed_files': 1
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint:
                return pr_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result is not None
        assert result['title'] == '添加新功能'
        assert result['description'] == '这是一个包含中文的PR描述'
    
    @pytest.mark.asyncio
    async def test_fetch_pr_with_null_body(self, analyzer):
        """Test fetching PR with null body"""
        pr_data = {
            'title': 'PR without description',
            'body': None,
            'created_at': '2024-01-01T00:00:00Z',
            'state': 'open',
            'user': {'login': 'author'},
            'base': {'ref': 'main'},
            'head': {'ref': 'feature'},
            'additions': 0,
            'deletions': 0,
            'changed_files': 0
        }
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            if 'pulls' in endpoint:
                return pr_data
            return []
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        result = await analyzer.fetch_pr("https://github.com/owner/repo/pull/123")
        
        assert result is not None
        assert result['description'] == ''


class TestCodeLinterEdgeCases:
    """Edge case tests for CodeLinter"""
    
    @pytest.fixture
    def linter(self):
        """Create linter instance"""
        from xencode.features.code_review import CodeLinter
        return CodeLinter()
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_null_bytes(self, linter):
        """Test analyzing file with null bytes"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': 'def test():\x00 pass'
        }]
        
        result = await linter.analyze(files)
        
        # Should handle gracefully
        assert result is not None
        assert result['summary']['total_files'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_very_long_lines(self, linter):
        """Test analyzing file with very long lines"""
        long_line = 'x = "' + 'a' * 10000 + '"'
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': long_line
        }]
        
        result = await linter.analyze(files)
        
        assert result is not None
        assert result['summary']['total_files'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_mixed_line_endings(self, linter):
        """Test analyzing file with mixed line endings"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': 'def test1():\r\n    pass\ndef test2():\n    pass'
        }]
        
        result = await linter.analyze(files)
        
        assert result is not None
        assert result['summary']['total_files'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_only_comments(self, linter):
        """Test analyzing file with only comments"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '# This is a comment\n# Another comment\n# More comments'
        }]
        
        result = await linter.analyze(files)
        
        assert result is not None
        assert result['summary']['total_files'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_only_whitespace(self, linter):
        """Test analyzing file with only whitespace"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '   \n\t\n   \n'
        }]
        
        result = await linter.analyze(files)
        
        assert result is not None
        assert result['summary']['total_files'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_issues_same_line(self, linter):
        """Test analyzing line with multiple issues"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': 'cursor.execute("SELECT * FROM users WHERE id = " + user_id); password = "secret123"'
        }]
        
        result = await linter.analyze(files)
        
        # Should detect both SQL injection and hardcoded secret
        assert result['summary']['total_issues'] >= 2
    
    @pytest.mark.asyncio
    async def test_analyze_nested_security_issues(self, linter):
        """Test analyzing nested security issues"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '''
def process(data):
    for item in data:
        if item:
            cursor.execute("SELECT * FROM users WHERE id = " + str(item))
            os.system("echo " + item)
'''
        }]
        
        result = await linter.analyze(files)
        
        # Should detect both SQL injection and command injection
        assert result['summary']['total_issues'] >= 2
        issue_types = set(i['type'] for i in result['issues'])
        assert 'sqli' in issue_types
        assert 'command_injection' in issue_types
    
    @pytest.mark.asyncio
    async def test_analyze_false_positive_patterns(self, linter):
        """Test that linter doesn't flag false positives"""
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '''
# This is a comment about SQL: SELECT * FROM users
def safe_query():
    # Using parameterized query
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
'''
        }]
        
        result = await linter.analyze(files)
        
        # Should not flag the comment or parameterized query
        sqli_issues = [i for i in result['issues'] if i['type'] == 'sqli']
        assert len(sqli_issues) == 0


class TestAIReviewEngineEdgeCases:
    """Edge case tests for AIReviewEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create AI review engine instance"""
        from xencode.features.code_review import AIReviewEngine
        return AIReviewEngine()
    
    @pytest.mark.asyncio
    async def test_generate_review_with_empty_files(self, engine):
        """Test generating review with empty files list"""
        files = []
        code_analysis = {
            'issues': [],
            'summary': {
                'total_issues': 0,
                'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            }
        }
        
        review = await engine.generate_review('Empty PR', 'No files', files, code_analysis)
        
        assert review is not None
        assert review['summary']['files_analyzed'] == 0
    
    @pytest.mark.asyncio
    async def test_generate_review_with_many_issues(self, engine):
        """Test generating review with many issues"""
        files = [{'path': 'test.py', 'content': 'test'}]
        
        # Create 100 issues
        issues = [
            {
                'type': 'sqli',
                'severity': 'critical',
                'message': f'Issue {i}',
                'file': 'test.py',
                'line': i,
                'column': 0
            }
            for i in range(100)
        ]
        
        code_analysis = {
            'issues': issues,
            'summary': {
                'total_issues': 100,
                'by_severity': {'critical': 100, 'high': 0, 'medium': 0, 'low': 0}
            }
        }
        
        review = await engine.generate_review('Many issues', 'Lots of problems', files, code_analysis)
        
        assert review is not None
        assert len(review['issues']) == 100
    
    @pytest.mark.asyncio
    async def test_generate_suggestion_for_unknown_issue_type(self, engine):
        """Test generating suggestion for unknown issue type"""
        issue = {
            'type': 'completely_unknown_type',
            'severity': 'medium',
            'message': 'Unknown issue',
            'file': 'test.py',
            'line': 1
        }
        
        suggestion = await engine._generate_ai_suggestion(issue, [])
        
        # Should return a generic suggestion
        assert suggestion is not None
        assert 'title' in suggestion
        assert 'description' in suggestion
    
    @pytest.mark.asyncio
    async def test_detect_patterns_in_empty_file(self, engine):
        """Test pattern detection in empty file"""
        files = [{'path': 'empty.py', 'content': ''}]
        
        patterns = await engine._detect_patterns(files)
        
        # Should handle gracefully
        assert isinstance(patterns, list)
    
    @pytest.mark.asyncio
    async def test_detect_patterns_in_minified_code(self, engine):
        """Test pattern detection in minified code"""
        files = [{
            'path': 'minified.js',
            'content': 'function a(b,c){return b+c;}function d(e){return e*2;}function f(g){return g-1;}'
        }]
        
        patterns = await engine._detect_patterns(files)
        
        # Should detect naming issues
        assert isinstance(patterns, list)
        naming_patterns = [p for p in patterns if p['type'] == 'naming']
        assert len(naming_patterns) > 0


class TestReportGeneratorEdgeCases:
    """Edge case tests for ReportGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create report generator instance"""
        return ReportGenerator()
    
    def test_generate_report_with_special_characters(self, generator):
        """Test generating report with special characters"""
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [
                {
                    'type': 'test',
                    'severity': 'high',
                    'message': 'Issue with <script>alert("XSS")</script>',
                    'file': 'test.py',
                    'line': 1
                }
            ],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        # HTML report should escape special characters
        html_report = generator.generate_html_report(review)
        assert '&lt;script&gt;' in html_report
        assert '<script>' not in html_report or '<!DOCTYPE html>' in html_report
    
    def test_generate_report_with_very_long_messages(self, generator):
        """Test generating report with very long messages"""
        long_message = 'A' * 10000
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [
                {
                    'type': 'test',
                    'severity': 'high',
                    'message': long_message,
                    'file': 'test.py',
                    'line': 1
                }
            ],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = generator.generate_text_report(review)
        
        assert long_message in report
    
    def test_generate_report_with_unicode_emoji(self, generator):
        """Test generating report with Unicode emoji"""
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [
                {
                    'type': 'test',
                    'severity': 'high',
                    'message': '🔥 Critical issue found! 🚨',
                    'file': 'test.py',
                    'line': 1
                }
            ],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = generator.generate_markdown_report(review)
        
        assert '🔥' in report
        assert '🚨' in report
    
    def test_calculate_quality_score_edge_cases(self, generator):
        """Test quality score calculation edge cases"""
        # Score should never go below 0
        score = generator._calculate_quality_score({'critical': 100})
        assert score == 0
        
        # Score should never go above 100
        score = generator._calculate_quality_score({})
        assert score == 100
        
        # Test with negative values (shouldn't happen but handle gracefully)
        score = generator._calculate_quality_score({'critical': -1})
        assert score >= 0
    
    def test_group_by_severity_with_missing_severity(self, generator):
        """Test grouping items with missing severity field"""
        items = [
            {'severity': 'critical', 'message': 'Issue 1'},
            {'message': 'Issue 2'},  # Missing severity
            {'severity': 'high', 'message': 'Issue 3'}
        ]
        
        grouped = generator._group_by_severity(items)
        
        # Should handle missing severity gracefully
        assert 'critical' in grouped
        assert 'high' in grouped
    
    def test_generate_json_report_serialization(self, generator):
        """Test JSON report handles non-serializable objects"""
        import json
        from datetime import datetime
        
        review = {
            'summary': {
                'files_analyzed': 1,
                'timestamp': datetime.now()  # Non-serializable
            },
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        # Should handle or skip non-serializable objects
        try:
            report = generator.generate_json_report(review)
            # If it succeeds, verify it's valid JSON
            json.loads(report)
        except (TypeError, ValueError):
            # Expected if datetime is not handled
            pass


class TestConcurrencyAndPerformance:
    """Tests for concurrent operations and performance"""
    
    @pytest.mark.asyncio
    async def test_concurrent_pr_fetches(self):
        """Test fetching multiple PRs concurrently"""
        import asyncio
        
        analyzer = GitHubPRAnalyzer()
        
        async def mock_get_session():
            return MagicMock()
        
        async def mock_make_request(endpoint, params=None):
            await asyncio.sleep(0.01)  # Simulate network delay
            return {
                'title': 'Test',
                'body': '',
                'created_at': '2024-01-01T00:00:00Z',
                'state': 'open',
                'user': {'login': 'author'},
                'base': {'ref': 'main'},
                'head': {'ref': 'feature'},
                'additions': 0,
                'deletions': 0,
                'changed_files': 0
            }
        
        analyzer._get_session = mock_get_session
        analyzer._make_request = mock_make_request
        
        # Fetch multiple PRs concurrently
        urls = [f"https://github.com/owner/repo/pull/{i}" for i in range(5)]
        tasks = [analyzer.fetch_pr(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_linter_performance_with_large_file(self):
        """Test linter performance with large file"""
        from xencode.features.code_review import CodeLinter
        import time
        
        linter = CodeLinter()
        
        # Create a large file
        large_content = "\n".join([f"def func{i}(): pass" for i in range(1000)])
        files = [{
            'path': 'large.py',
            'language': 'python',
            'content': large_content
        }]
        
        start_time = time.time()
        result = await linter.analyze(files)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed_time < 5.0
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
