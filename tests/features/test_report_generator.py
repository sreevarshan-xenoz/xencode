"""
Unit tests for ReportGenerator
"""

import pytest
import json
from xencode.features.code_review import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance"""
        return ReportGenerator()
    
    @pytest.fixture
    def sample_review(self):
        """Sample review data"""
        return {
            'summary': {
                'title': 'Add authentication feature',
                'description': 'Implements user authentication',
                'files_analyzed': 3,
                'ai_summary': 'Good implementation with minor security concerns'
            },
            'issues': [
                {
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'SQL injection vulnerability detected',
                    'file': 'auth.py',
                    'line': 42,
                    'column': 10
                },
                {
                    'type': 'hardcoded_secret',
                    'severity': 'high',
                    'message': 'Hardcoded API key found',
                    'file': 'config.py',
                    'line': 15,
                    'column': 5
                },
                {
                    'type': 'complexity',
                    'severity': 'medium',
                    'message': 'High cyclomatic complexity',
                    'file': 'utils.py',
                    'line': 100,
                    'column': 0
                },
                {
                    'type': 'naming',
                    'severity': 'low',
                    'message': 'Variable name too short',
                    'file': 'helpers.py',
                    'line': 20,
                    'column': 4
                }
            ],
            'suggestions': [
                {
                    'title': 'Use Parameterized Queries',
                    'description': 'Replace string concatenation with parameterized queries',
                    'severity': 'critical',
                    'file': 'auth.py',
                    'line': 42,
                    'example': 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
                },
                {
                    'title': 'Use Environment Variables',
                    'description': 'Store API keys in environment variables',
                    'severity': 'high',
                    'file': 'config.py',
                    'line': 15,
                    'example': 'API_KEY = os.environ.get("API_KEY")'
                }
            ],
            'patterns_detected': [
                {
                    'type': 'complexity',
                    'pattern': 'nested_structure',
                    'file': 'utils.py',
                    'message': 'Deeply nested control structures detected'
                }
            ],
            'semantic_analysis': {
                'analysis': 'The code follows good practices but needs security improvements',
                'confidence': 0.85,
                'consensus_score': 0.92
            },
            'positive_feedback': [
                {
                    'title': 'Good Test Coverage',
                    'message': 'Excellent test coverage for new features',
                    'score': 95
                }
            ]
        }
    
    @pytest.fixture
    def sample_pr_data(self):
        """Sample PR data"""
        return {
            'title': 'Add authentication feature',
            'url': 'https://github.com/owner/repo/pull/123',
            'author': 'developer',
            'head_branch': 'feature/auth',
            'base_branch': 'main'
        }
    
    def test_generate_text_report(self, generator, sample_review, sample_pr_data):
        """Test text report generation"""
        report = generator.generate_text_report(sample_review, sample_pr_data)
        
        # Check header
        assert "CODE REVIEW REPORT" in report
        assert "=" * 80 in report
        
        # Check PR information
        assert "Pull Request Information:" in report
        assert "Add authentication feature" in report
        assert "https://github.com/owner/repo/pull/123" in report
        assert "developer" in report
        assert "feature/auth → main" in report
        
        # Check summary
        assert "Summary:" in report
        assert "Files Analyzed: 3" in report
        assert "Good implementation with minor security concerns" in report
        
        # Check issues by severity
        assert "Issues Found:" in report
        assert "CRITICAL (1 issue(s)):" in report
        assert "HIGH (1 issue(s)):" in report
        assert "MEDIUM (1 issue(s)):" in report
        assert "LOW (1 issue(s)):" in report
        
        # Check issue details
        assert "SQL injection vulnerability detected" in report
        assert "auth.py:42" in report
        assert "Hardcoded API key found" in report
        assert "config.py:15" in report
        
        # Check suggestions
        assert "Suggestions:" in report
        assert "Use Parameterized Queries" in report
        assert "cursor.execute" in report
        
        # Check patterns
        assert "Patterns Detected:" in report
        assert "Deeply nested control structures detected" in report
        
        # Check semantic analysis
        assert "Semantic Analysis:" in report
        assert "The code follows good practices" in report
        assert "Confidence: 85.00%" in report
        assert "Consensus Score: 92.00%" in report
        
        # Check positive feedback
        assert "Positive Feedback:" in report
        assert "Good Test Coverage" in report
        assert "Score: 95/100" in report
    
    def test_generate_text_report_no_issues(self, generator):
        """Test text report with no issues"""
        review = {
            'summary': {'files_analyzed': 2},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = generator.generate_text_report(review)
        
        assert "✓ No issues found!" in report
    
    def test_generate_markdown_report(self, generator, sample_review, sample_pr_data):
        """Test Markdown report generation"""
        report = generator.generate_markdown_report(sample_review, sample_pr_data)
        
        # Check header
        assert "# Code Review Report" in report
        
        # Check PR information
        assert "## Pull Request Information" in report
        assert "- **Title:** Add authentication feature" in report
        assert "- **URL:** https://github.com/owner/repo/pull/123" in report
        assert "- **Author:** developer" in report
        assert "- **Branch:** `feature/auth` → `main`" in report
        
        # Check summary
        assert "## Summary" in report
        assert "- **Files Analyzed:** 3" in report
        
        # Check issues
        assert "## Issues Found" in report
        assert "### 🔴 CRITICAL (1 issue(s))" in report
        assert "### 🟠 HIGH (1 issue(s))" in report
        assert "### 🟡 MEDIUM (1 issue(s))" in report
        assert "### 🟢 LOW (1 issue(s))" in report
        
        # Check issue details
        assert "#### SQLI" in report
        assert "**Message:** SQL injection vulnerability detected" in report
        assert "**Location:** `auth.py:42`" in report
        
        # Check suggestions
        assert "## Suggestions" in report
        assert "#### Use Parameterized Queries" in report
        assert "**Example:**" in report
        assert "```" in report
        
        # Check patterns
        assert "## Patterns Detected" in report
        
        # Check semantic analysis
        assert "## Semantic Analysis" in report
        
        # Check positive feedback
        assert "## ✨ Positive Feedback" in report
        assert "### ✅ Good Test Coverage" in report
    
    def test_generate_markdown_report_no_issues(self, generator):
        """Test Markdown report with no issues"""
        review = {
            'summary': {'files_analyzed': 2},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = generator.generate_markdown_report(review)
        
        assert "## ✅ No Issues Found!" in report
    
    def test_generate_json_report(self, generator, sample_review, sample_pr_data):
        """Test JSON report generation"""
        report = generator.generate_json_report(sample_review, sample_pr_data)
        
        # Check structure
        assert 'metadata' in report
        assert 'generated_at' in report['metadata']
        assert 'report_version' in report['metadata']
        
        assert 'pr_info' in report
        assert report['pr_info']['title'] == 'Add authentication feature'
        
        assert 'summary' in report
        assert report['summary']['files_analyzed'] == 3
        assert report['summary']['total_issues'] == 4
        assert 'severity_counts' in report['summary']
        assert report['summary']['severity_counts']['critical'] == 1
        assert report['summary']['severity_counts']['high'] == 1
        assert report['summary']['severity_counts']['medium'] == 1
        assert report['summary']['severity_counts']['low'] == 1
        
        # Check quality score calculation
        # 100 - (1*20 + 1*10 + 1*5 + 1*2) = 63
        assert report['summary']['quality_score'] == 63
        
        assert 'issues_by_severity' in report
        assert 'critical' in report['issues_by_severity']
        assert len(report['issues_by_severity']['critical']) == 1
        
        assert 'suggestions_by_severity' in report
        assert 'patterns_detected' in report
        assert 'semantic_analysis' in report
        assert 'positive_feedback' in report
    
    def test_generate_html_report(self, generator, sample_review, sample_pr_data):
        """Test HTML report generation"""
        report = generator.generate_html_report(sample_review, sample_pr_data)
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in report
        assert "<html>" in report
        assert "</html>" in report
        assert "<head>" in report
        assert "<body>" in report
        
        # Check title
        assert "<title>Code Review Report</title>" in report
        assert "📋 Code Review Report" in report
        
        # Check CSS
        assert "<style>" in report
        assert ".severity-critical" in report
        assert ".severity-high" in report
        assert ".severity-medium" in report
        assert ".severity-low" in report
        
        # Check PR information
        assert "Pull Request Information" in report
        assert "Add authentication feature" in report
        assert "https://github.com/owner/repo/pull/123" in report
        
        # Check issues
        assert "🔍 Issues Found" in report
        assert "SQL injection vulnerability detected" in report
        
        # Check suggestions
        assert "💡 Suggestions" in report
        assert "Use Parameterized Queries" in report
        
        # Check patterns
        assert "🔎 Patterns Detected" in report
        
        # Check semantic analysis
        assert "🧠 Semantic Analysis" in report
        
        # Check positive feedback
        assert "✨ Positive Feedback" in report
    
    def test_generate_html_report_no_issues(self, generator):
        """Test HTML report with no issues"""
        review = {
            'summary': {'files_analyzed': 2},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        report = generator.generate_html_report(review)
        
        assert "✅ No Issues Found!" in report
    
    def test_html_escape(self, generator):
        """Test HTML escaping"""
        text = '<script>alert("XSS")</script>'
        escaped = generator._html_escape(text)
        
        assert '<' not in escaped
        assert '>' not in escaped
        assert '&lt;' in escaped
        assert '&gt;' in escaped
        assert 'script' in escaped
    
    def test_group_by_severity(self, generator):
        """Test grouping items by severity"""
        items = [
            {'severity': 'critical', 'message': 'Critical issue'},
            {'severity': 'high', 'message': 'High issue'},
            {'severity': 'critical', 'message': 'Another critical'},
            {'severity': 'low', 'message': 'Low issue'}
        ]
        
        grouped = generator._group_by_severity(items)
        
        assert len(grouped['critical']) == 2
        assert len(grouped['high']) == 1
        assert len(grouped['low']) == 1
        assert 'medium' not in grouped
    
    def test_get_severity_emoji(self, generator):
        """Test severity emoji mapping"""
        assert generator._get_severity_emoji('critical') == '🔴'
        assert generator._get_severity_emoji('high') == '🟠'
        assert generator._get_severity_emoji('medium') == '🟡'
        assert generator._get_severity_emoji('low') == '🟢'
        assert generator._get_severity_emoji('unknown') == '⚪'
    
    def test_calculate_quality_score(self, generator):
        """Test quality score calculation"""
        # No issues = 100
        score = generator._calculate_quality_score({})
        assert score == 100
        
        # 1 critical = 80
        score = generator._calculate_quality_score({'critical': 1})
        assert score == 80
        
        # 1 high = 90
        score = generator._calculate_quality_score({'high': 1})
        assert score == 90
        
        # 1 medium = 95
        score = generator._calculate_quality_score({'medium': 1})
        assert score == 95
        
        # 1 low = 98
        score = generator._calculate_quality_score({'low': 1})
        assert score == 98
        
        # Mixed: 1 critical + 2 high + 3 medium + 4 low
        # 100 - (1*20 + 2*10 + 3*5 + 4*2) = 100 - 63 = 37
        score = generator._calculate_quality_score({
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        })
        assert score == 37
        
        # Too many issues = 0 (minimum)
        score = generator._calculate_quality_score({'critical': 10})
        assert score == 0
    
    def test_generate_report_without_pr_data(self, generator, sample_review):
        """Test report generation without PR data"""
        # Text report
        text_report = generator.generate_text_report(sample_review)
        assert "CODE REVIEW REPORT" in text_report
        assert "Pull Request Information:" not in text_report
        
        # Markdown report
        md_report = generator.generate_markdown_report(sample_review)
        assert "# Code Review Report" in md_report
        assert "## Pull Request Information" not in md_report
        
        # HTML report
        html_report = generator.generate_html_report(sample_review)
        assert "Code Review Report" in html_report
        assert "Pull Request Information" not in html_report
        
        # JSON report
        json_report = generator.generate_json_report(sample_review)
        assert json_report['pr_info'] == {}
    
    def test_generate_report_with_empty_sections(self, generator):
        """Test report generation with empty sections"""
        review = {
            'summary': {'files_analyzed': 1},
            'issues': [],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
        
        # Should not crash and should handle empty sections gracefully
        text_report = generator.generate_text_report(review)
        assert "✓ No issues found!" in text_report
        
        md_report = generator.generate_markdown_report(review)
        assert "✅ No Issues Found!" in md_report
        
        html_report = generator.generate_html_report(review)
        assert "✅ No Issues Found!" in html_report
        
        json_report = generator.generate_json_report(review)
        assert json_report['summary']['total_issues'] == 0
        assert json_report['summary']['quality_score'] == 100


class TestCodeReviewFeatureReportGeneration:
    """Tests for report generation in CodeReviewFeature"""
    
    @pytest.fixture
    def feature(self):
        """Create feature instance"""
        from xencode.features.code_review import CodeReviewFeature, CodeReviewConfig
        from xencode.features.base import FeatureConfig
        
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            config=CodeReviewConfig(
                supported_languages=['python', 'javascript', 'typescript'],
                severity_levels=['critical', 'high', 'medium', 'low']
            )
        )
        
        return CodeReviewFeature(config)
    
    @pytest.fixture
    def sample_review(self):
        """Sample review data"""
        return {
            'summary': {
                'title': 'Test PR',
                'description': 'Test description',
                'files_analyzed': 2
            },
            'issues': [
                {
                    'type': 'sqli',
                    'severity': 'critical',
                    'message': 'SQL injection',
                    'file': 'test.py',
                    'line': 10
                }
            ],
            'suggestions': [],
            'patterns_detected': [],
            'semantic_analysis': {},
            'positive_feedback': []
        }
    
    def test_generate_formatted_report_text(self, feature, sample_review):
        """Test text format report generation"""
        report = feature.generate_formatted_report(sample_review, format='text')
        
        assert isinstance(report, str)
        assert "CODE REVIEW REPORT" in report
        assert "SQL injection" in report
    
    def test_generate_formatted_report_markdown(self, feature, sample_review):
        """Test Markdown format report generation"""
        report = feature.generate_formatted_report(sample_review, format='markdown')
        
        assert isinstance(report, str)
        assert "# Code Review Report" in report
        assert "SQL injection" in report
    
    def test_generate_formatted_report_html(self, feature, sample_review):
        """Test HTML format report generation"""
        report = feature.generate_formatted_report(sample_review, format='html')
        
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "SQL injection" in report
    
    def test_generate_formatted_report_json(self, feature, sample_review):
        """Test JSON format report generation"""
        report = feature.generate_formatted_report(sample_review, format='json')
        
        assert isinstance(report, str)
        # Parse to verify it's valid JSON
        data = json.loads(report)
        assert 'metadata' in data
        assert 'summary' in data
        assert data['summary']['total_issues'] == 1
    
    def test_generate_formatted_report_invalid_format(self, feature, sample_review):
        """Test invalid format raises error"""
        with pytest.raises(ValueError, match="Unsupported format"):
            feature.generate_formatted_report(sample_review, format='xml')
    
    def test_generate_formatted_report_with_pr_data(self, feature, sample_review):
        """Test report generation with PR data"""
        pr_data = {
            'title': 'Test PR',
            'url': 'https://github.com/test/repo/pull/1',
            'author': 'tester'
        }
        
        report = feature.generate_formatted_report(sample_review, pr_data, format='text')
        
        assert "Test PR" in report
        assert "https://github.com/test/repo/pull/1" in report
        assert "tester" in report
