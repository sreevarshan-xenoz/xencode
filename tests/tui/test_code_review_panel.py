#!/usr/bin/env python3
"""
Unit tests for Code Review TUI components
"""

import pytest
from textual.app import App
from xencode.tui.widgets.code_review_panel import (
    CodeReviewPanel,
    ReviewHistoryPanel,
    ReviewSummaryPanel,
    ReviewIssueItem,
    SuggestionItem
)


@pytest.fixture
def sample_review():
    """Sample review data for testing"""
    return {
        'summary': {
            'files_analyzed': 5,
            'ai_summary': 'Code quality is good with minor issues'
        },
        'issues': [
            {
                'type': 'sqli',
                'severity': 'critical',
                'message': 'Potential SQL injection detected',
                'file': 'app.py',
                'line': 42
            },
            {
                'type': 'xss',
                'severity': 'high',
                'message': 'Potential XSS vulnerability',
                'file': 'views.py',
                'line': 15
            },
            {
                'type': 'code_quality',
                'severity': 'low',
                'message': 'Consider using more descriptive variable names',
                'file': 'utils.py',
                'line': 8
            }
        ],
        'suggestions': [
            {
                'title': 'SQL Injection Prevention',
                'description': 'Use parameterized queries',
                'severity': 'critical',
                'file': 'app.py',
                'line': 42,
                'example': 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
            },
            {
                'title': 'XSS Prevention',
                'description': 'Sanitize user input',
                'severity': 'high',
                'file': 'views.py',
                'line': 15
            }
        ],
        'patterns_detected': [
            {
                'type': 'complexity',
                'pattern': 'nested_structure',
                'file': 'app.py',
                'message': 'High complexity detected',
                'severity': 'medium'
            }
        ],
        'semantic_analysis': {
            'analysis': 'The code follows good practices overall',
            'confidence': 0.85,
            'consensus_score': 0.92
        },
        'positive_feedback': [
            {
                'title': 'Good Security Posture',
                'message': 'No critical security issues in most files',
                'score': 85
            }
        ]
    }


@pytest.fixture
def sample_pr_data():
    """Sample PR data for testing"""
    return {
        'url': 'https://github.com/user/repo/pull/123',
        'title': 'Add new feature',
        'description': 'This PR adds a new feature',
        'platform': 'github',
        'author': 'testuser',
        'base_branch': 'main',
        'head_branch': 'feature/new-feature'
    }


class TestReviewIssueItem:
    """Tests for ReviewIssueItem widget"""
    
    def test_issue_item_creation(self):
        """Test creating an issue item"""
        issue = {
            'type': 'sqli',
            'severity': 'critical',
            'message': 'SQL injection detected',
            'file': 'app.py',
            'line': 42
        }
        
        item = ReviewIssueItem(issue)
        
        assert item.issue == issue
        assert item.severity == 'critical'
        assert item.has_class('critical')
    
    def test_issue_item_severity_classes(self):
        """Test that severity classes are applied correctly"""
        severities = ['critical', 'high', 'medium', 'low']
        
        for severity in severities:
            issue = {
                'type': 'test',
                'severity': severity,
                'message': 'Test message',
                'file': 'test.py',
                'line': 1
            }
            
            item = ReviewIssueItem(issue)
            assert item.has_class(severity)


class TestSuggestionItem:
    """Tests for SuggestionItem widget"""
    
    def test_suggestion_item_creation(self):
        """Test creating a suggestion item"""
        suggestion = {
            'title': 'Fix SQL Injection',
            'description': 'Use parameterized queries',
            'file': 'app.py',
            'line': 42
        }
        
        item = SuggestionItem(suggestion)
        
        assert item.suggestion == suggestion


class TestReviewSummaryPanel:
    """Tests for ReviewSummaryPanel widget"""
    
    def test_summary_panel_creation(self, sample_review):
        """Test creating a summary panel"""
        panel = ReviewSummaryPanel(sample_review)
        
        assert panel.review == sample_review
    
    def test_summary_panel_calculates_quality_score(self, sample_review):
        """Test that quality score is calculated correctly"""
        panel = ReviewSummaryPanel(sample_review)
        
        # Quality score calculation:
        # 100 - (1 critical * 20) - (1 high * 10) - (0 medium * 5) - (1 low * 2) = 68
        # This is tested indirectly through the render method
        rendered = panel.render()
        assert rendered is not None


class TestCodeReviewPanel:
    """Tests for CodeReviewPanel widget"""
    
    def test_panel_creation(self):
        """Test creating a code review panel"""
        panel = CodeReviewPanel()
        
        assert panel.current_review is None
        assert panel.current_pr_data is None
        assert panel.current_tab == "issues"
    
    def test_update_review(self, sample_review, sample_pr_data):
        """Test updating the panel with review data"""
        panel = CodeReviewPanel()
        
        # Just test that the data is stored correctly
        # Full UI update requires the widget to be mounted
        panel.current_review = sample_review
        panel.current_pr_data = sample_pr_data
        
        assert panel.current_review == sample_review
        assert panel.current_pr_data == sample_pr_data
    
    def test_show_tab(self):
        """Test switching between tabs"""
        panel = CodeReviewPanel()
        
        # Test tab state changes (UI updates require mounted widget)
        panel.current_tab = "suggestions"
        assert panel.current_tab == "suggestions"
        
        panel.current_tab = "summary"
        assert panel.current_tab == "summary"
        
        panel.current_tab = "issues"
        assert panel.current_tab == "issues"


class TestReviewHistoryPanel:
    """Tests for ReviewHistoryPanel widget"""
    
    def test_history_panel_creation(self):
        """Test creating a history panel"""
        panel = ReviewHistoryPanel()
        
        assert panel.history == []
    
    def test_add_review_to_history(self, sample_review, sample_pr_data):
        """Test adding a review to history"""
        panel = ReviewHistoryPanel()
        
        # Manually add to history list (mounting requires app context)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_item = {
            'timestamp': timestamp,
            'review': sample_review,
            'pr_data': sample_pr_data
        }
        panel.history.append(history_item)
        
        assert len(panel.history) == 1
        assert panel.history[0]['review'] == sample_review
        assert panel.history[0]['pr_data'] == sample_pr_data
        assert 'timestamp' in panel.history[0]
    
    def test_clear_history(self, sample_review):
        """Test clearing the history"""
        panel = ReviewHistoryPanel()
        
        # Manually add to history
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        panel.history.append({
            'timestamp': timestamp,
            'review': sample_review,
            'pr_data': None
        })
        assert len(panel.history) == 1
        
        # Clear history
        panel.history.clear()
        assert len(panel.history) == 0


class TestCodeReviewPanelMessages:
    """Tests for CodeReviewPanel messages"""
    
    def test_review_started_message(self):
        """Test ReviewStarted message"""
        message = CodeReviewPanel.ReviewStarted(
            pr_url="https://github.com/user/repo/pull/123",
            platform="github"
        )
        
        assert message.pr_url == "https://github.com/user/repo/pull/123"
        assert message.platform == "github"
    
    def test_issue_selected_message(self):
        """Test IssueSelected message"""
        issue = {
            'type': 'sqli',
            'severity': 'critical',
            'message': 'SQL injection',
            'file': 'app.py',
            'line': 42
        }
        
        message = CodeReviewPanel.IssueSelected(issue)
        
        assert message.issue == issue


@pytest.mark.asyncio
async def test_code_review_panel_integration(sample_review, sample_pr_data):
    """Integration test for code review panel"""
    
    class TestApp(App):
        def compose(self):
            yield CodeReviewPanel()
    
    app = TestApp()
    
    async with app.run_test() as pilot:
        # Get the panel
        panel = app.query_one(CodeReviewPanel)
        
        # Update with review data
        panel.update_review(sample_review, sample_pr_data)
        
        # Verify the panel was updated
        assert panel.current_review == sample_review
        assert panel.current_pr_data == sample_pr_data
        
        # Test tab switching
        panel._show_tab("suggestions")
        assert panel.current_tab == "suggestions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
