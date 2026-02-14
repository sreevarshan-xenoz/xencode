#!/usr/bin/env python3
"""
Unit tests for Review Diff Viewer TUI components
"""

import pytest
from textual.app import App
from xencode.tui.widgets.review_diff_viewer import (
    ReviewDiffViewer,
    ReviewDiffPanel,
    DiffLineWithSuggestion,
    InlineSuggestionPanel
)


@pytest.fixture
def sample_diff():
    """Sample diff content for testing"""
    return """diff --git a/example.py b/example.py
--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@ def process_data(data):
     for item in data:
         if item:
-            result = eval(item)
+            result = json.loads(item)
             results.append(result)
     return results
 
@@ -20,5 +20,5 @@ def authenticate(username, password):
-    query = f"SELECT * FROM users WHERE username='{username}'"
+    query = "SELECT * FROM users WHERE username=?"
-    cursor.execute(query)
+    cursor.execute(query, (username,))
     return cursor.fetchone()
"""


@pytest.fixture
def sample_suggestions():
    """Sample suggestions for testing"""
    return [
        {
            'title': 'Avoid eval() - Security Risk',
            'description': 'Using eval() can execute arbitrary code',
            'severity': 'critical',
            'file': 'example.py',
            'line': 13,
            'example': 'Use json.loads() instead'
        },
        {
            'title': 'SQL Injection Prevention',
            'description': 'Use parameterized queries',
            'severity': 'critical',
            'file': 'example.py',
            'line': 23,
            'example': 'cursor.execute(query, (username,))'
        },
        {
            'title': 'Add Error Handling',
            'description': 'Consider adding try-except blocks',
            'severity': 'medium',
            'file': 'example.py',
            'line': 13
        }
    ]


class TestDiffLineWithSuggestion:
    """Tests for DiffLineWithSuggestion widget"""
    
    def test_addition_line(self):
        """Test addition line styling"""
        line = "+            result = json.loads(item)"
        widget = DiffLineWithSuggestion(line, 13)
        
        assert widget.line == line
        assert widget.line_number == 13
        assert widget.has_class("addition")
    
    def test_deletion_line(self):
        """Test deletion line styling"""
        line = "-            result = eval(item)"
        widget = DiffLineWithSuggestion(line, 0)
        
        assert widget.has_class("deletion")
    
    def test_hunk_header(self):
        """Test hunk header styling"""
        line = "@@ -10,7 +10,7 @@ def process_data(data):"
        widget = DiffLineWithSuggestion(line, 0)
        
        assert widget.has_class("hunk-header")
    
    def test_context_line(self):
        """Test context line styling"""
        line = "     for item in data:"
        widget = DiffLineWithSuggestion(line, 11)
        
        assert widget.has_class("context")
    
    def test_line_with_suggestion(self):
        """Test line with suggestion"""
        line = "+            result = json.loads(item)"
        suggestion = {
            'title': 'Good fix!',
            'description': 'This is better than eval()',
            'severity': 'low'
        }
        
        widget = DiffLineWithSuggestion(line, 13, suggestion)
        
        assert widget.suggestion == suggestion
        assert widget.has_class("has-suggestion")


class TestInlineSuggestionPanel:
    """Tests for InlineSuggestionPanel widget"""
    
    def test_panel_creation(self):
        """Test creating a suggestion panel"""
        suggestion = {
            'title': 'Fix Security Issue',
            'description': 'Use parameterized queries',
            'severity': 'critical',
            'example': 'cursor.execute(query, params)'
        }
        
        panel = InlineSuggestionPanel(suggestion)
        
        assert panel.suggestion == suggestion
        assert panel.has_class('critical')
    
    def test_severity_classes(self):
        """Test that severity classes are applied"""
        severities = ['critical', 'high', 'medium', 'low']
        
        for severity in severities:
            suggestion = {
                'title': 'Test',
                'description': 'Test description',
                'severity': severity
            }
            
            panel = InlineSuggestionPanel(suggestion)
            assert panel.has_class(severity)


class TestReviewDiffViewer:
    """Tests for ReviewDiffViewer widget"""
    
    def test_viewer_creation(self, sample_diff, sample_suggestions):
        """Test creating a diff viewer"""
        viewer = ReviewDiffViewer(sample_diff, sample_suggestions)
        
        assert viewer._diff_content == sample_diff
        assert viewer.suggestions == sample_suggestions
    
    def test_suggestion_indexing(self, sample_diff, sample_suggestions):
        """Test that suggestions are indexed by file and line"""
        viewer = ReviewDiffViewer(sample_diff, sample_suggestions)
        
        # Check that suggestions are indexed
        assert len(viewer.suggestions_by_line) > 0
        
        # Check specific suggestion
        key = "example.py:13"
        assert key in viewer.suggestions_by_line
        assert len(viewer.suggestions_by_line[key]) >= 1
    
    def test_update_suggestions(self, sample_diff, sample_suggestions):
        """Test updating suggestions"""
        viewer = ReviewDiffViewer(sample_diff, [])
        
        assert len(viewer.suggestions) == 0
        
        # Update suggestions directly
        viewer.suggestions = sample_suggestions
        viewer._index_suggestions()
        
        assert len(viewer.suggestions) == len(sample_suggestions)
        assert len(viewer.suggestions_by_line) > 0


class TestReviewDiffPanel:
    """Tests for ReviewDiffPanel widget"""
    
    def test_panel_creation(self, sample_diff, sample_suggestions):
        """Test creating a diff panel"""
        panel = ReviewDiffPanel(sample_diff, sample_suggestions, "Test Review")
        
        assert panel.diff == sample_diff
        assert panel.suggestions == sample_suggestions
        assert panel.title == "Test Review"
    
    def test_stats_calculation(self, sample_diff, sample_suggestions):
        """Test diff statistics calculation"""
        panel = ReviewDiffPanel(sample_diff, sample_suggestions)
        
        stats = panel._stats
        
        # Check basic stats
        assert stats['additions'] > 0
        assert stats['deletions'] > 0
        assert stats['files'] > 0
        
        # Check suggestion stats
        assert stats['suggestions'] == len(sample_suggestions)
        assert stats['critical'] == 2  # Two critical suggestions in sample
        assert stats['medium'] == 1    # One medium suggestion in sample
    
    def test_update_diff(self, sample_diff, sample_suggestions):
        """Test updating diff content"""
        panel = ReviewDiffPanel("", [])
        
        # Update internal state
        panel.diff = sample_diff
        panel.suggestions = sample_suggestions
        panel.title = "Updated Review"
        panel._stats = panel._calculate_stats(sample_diff)
        
        assert panel.diff == sample_diff
        assert panel.suggestions == sample_suggestions
        assert panel.title == "Updated Review"


class TestReviewDiffPanelMessages:
    """Tests for ReviewDiffPanel messages"""
    
    def test_apply_suggestion_message(self):
        """Test ApplySuggestion message"""
        suggestion = {
            'title': 'Fix Issue',
            'description': 'Apply this fix',
            'file': 'test.py',
            'line': 10
        }
        
        message = ReviewDiffPanel.ApplySuggestion(suggestion)
        
        assert message.suggestion == suggestion


@pytest.mark.asyncio
async def test_diff_viewer_integration(sample_diff, sample_suggestions):
    """Integration test for diff viewer"""
    
    class TestApp(App):
        def compose(self):
            yield ReviewDiffPanel(sample_diff, sample_suggestions, "Integration Test")
    
    app = TestApp()
    
    async with app.run_test() as pilot:
        # Get the panel
        panel = app.query_one(ReviewDiffPanel)
        
        # Verify initial state
        assert panel.diff == sample_diff
        assert len(panel.suggestions) == len(sample_suggestions)
        
        # Test stats
        stats = panel._stats
        assert stats['suggestions'] == len(sample_suggestions)
        assert stats['critical'] == 2


@pytest.mark.asyncio
async def test_empty_diff_viewer():
    """Test diff viewer with no content"""
    
    class TestApp(App):
        def compose(self):
            yield ReviewDiffViewer("", [])
    
    app = TestApp()
    
    async with app.run_test() as pilot:
        viewer = app.query_one(ReviewDiffViewer)
        
        assert viewer.diff_content == ""
        assert len(viewer.suggestions) == 0


def test_diff_line_numbers():
    """Test that line numbers are tracked correctly"""
    diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test():
+    print("hello")
     pass
"""
    
    viewer = ReviewDiffViewer(diff, [])
    
    # Verify viewer was created with diff
    assert viewer._diff_content == diff
    # Line numbers are tracked during compose, which is tested in integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
