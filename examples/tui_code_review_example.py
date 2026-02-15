#!/usr/bin/env python3
"""
Example: Using Code Review TUI Components

This example demonstrates how to integrate the code review TUI components
into a Textual application.
"""

import asyncio
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Header, Footer

# Import code review components
from xencode.tui.widgets.code_review_panel import CodeReviewPanel, ReviewHistoryPanel
from xencode.tui.widgets.review_diff_viewer import ReviewDiffPanel

# Import code review feature
from xencode.features.code_review import CodeReviewFeature
from xencode.features.base import FeatureConfig


class CodeReviewApp(App):
    """Example TUI app with code review components"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
    }
    
    #left-panel {
        width: 60%;
    }
    
    #right-panel {
        width: 40%;
    }
    """
    
    TITLE = "Code Review TUI Example"
    
    def __init__(self):
        super().__init__()
        
        # Initialize code review feature
        config = FeatureConfig(
            name="code_review",
            enabled=True,
            settings={}
        )
        self.code_review = CodeReviewFeature(config)
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        
        with Container(id="main-container"):
            with Horizontal():
                # Left panel: Main review interface
                with Container(id="left-panel"):
                    self.review_panel = CodeReviewPanel()
                    yield self.review_panel
                
                # Right panel: History
                with Container(id="right-panel"):
                    self.history_panel = ReviewHistoryPanel()
                    yield self.history_panel
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Called when app is mounted"""
        # Initialize the code review feature
        await self.code_review.initialize()
        
        self.notify("Code Review TUI ready! Enter a PR URL to start.", severity="information")
    
    async def on_code_review_panel_review_started(
        self, event: CodeReviewPanel.ReviewStarted
    ) -> None:
        """Handle review start event"""
        pr_url = event.pr_url
        platform = event.platform
        
        self.notify(f"Analyzing {platform} PR: {pr_url}", severity="information")
        
        try:
            # Analyze the PR
            result = await self.code_review.analyze_pr(pr_url, platform)
            
            # Extract review data
            review = result.get('review', {})
            pr_data = result.get('pr', {})
            
            # Update the review panel
            self.review_panel.update_review(review, pr_data)
            
            # Add to history
            self.history_panel.add_review(review, pr_data)
            
            # Show diff if available
            if 'files' in pr_data:
                # Generate a simple diff representation
                diff_lines = []
                for file_data in pr_data['files'][:5]:  # Limit to first 5 files
                    diff_lines.append(f"diff --git a/{file_data['filename']} b/{file_data['filename']}")
                    diff_lines.append(f"--- a/{file_data['filename']}")
                    diff_lines.append(f"+++ b/{file_data['filename']}")
                    
                    if 'patch' in file_data:
                        diff_lines.append(file_data['patch'])
                
                diff_content = '\n'.join(diff_lines)
                
                # Create diff panel (you could mount this dynamically)
                # For this example, we'll just show a notification
                self.notify(f"Review complete! Found {len(review.get('issues', []))} issues", 
                           severity="success")
        
        except Exception as e:
            self.notify(f"Error analyzing PR: {e}", severity="error")


class FileDiffReviewApp(App):
    """Example app showing diff viewer with suggestions"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    """
    
    TITLE = "Diff Viewer with AI Suggestions"
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        
        # Sample diff content
        sample_diff = """diff --git a/example.py b/example.py
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
-    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
+    query = "SELECT * FROM users WHERE username=? AND password=?"
-    cursor.execute(query)
+    cursor.execute(query, (username, password))
     return cursor.fetchone()
"""
        
        # Sample suggestions
        sample_suggestions = [
            {
                'title': 'Avoid eval() - Security Risk',
                'description': 'Using eval() can execute arbitrary code. Use json.loads() instead.',
                'severity': 'critical',
                'file': 'example.py',
                'line': 13,
                'example': '''# Bad
result = eval(item)

# Good
import json
result = json.loads(item)'''
            },
            {
                'title': 'SQL Injection Prevention',
                'description': 'Use parameterized queries to prevent SQL injection attacks.',
                'severity': 'critical',
                'file': 'example.py',
                'line': 23,
                'example': '''# Bad
query = f"SELECT * FROM users WHERE username='{username}'"
cursor.execute(query)

# Good
query = "SELECT * FROM users WHERE username=?"
cursor.execute(query, (username,))'''
            }
        ]
        
        yield ReviewDiffPanel(
            diff=sample_diff,
            suggestions=sample_suggestions,
            title="Security Review - example.py"
        )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted"""
        self.notify("Diff viewer loaded with 2 critical security issues", severity="warning")


def main():
    """Run the example app"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        # Show diff viewer example
        app = FileDiffReviewApp()
    else:
        # Show full code review example
        app = CodeReviewApp()
    
    app.run()


if __name__ == "__main__":
    main()
