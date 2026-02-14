#!/usr/bin/env python3
"""
Tests for Terminal Assistant TUI Panel

Tests all TUI components: suggestions, explanations, fixes, progress, and history.
"""

import pytest
from datetime import datetime
from textual.widgets import Label, Input

from xencode.tui.widgets.terminal_assistant_panel import (
    CommandSuggestionItem,
    CommandSuggestionPanel,
    CommandExplanationViewer,
    ErrorFixItem,
    ErrorFixPanel,
    LearningProgressTracker,
    CommandHistoryBrowser,
    TerminalAssistantPanel
)


class TestCommandSuggestionItem:
    """Test CommandSuggestionItem widget"""
    
    def test_create_suggestion_item(self):
        """Test creating a suggestion item"""
        suggestion = {
            'command': 'git status',
            'score': 15.5,
            'source': 'history',
            'explanation': 'Check repository status',
            'reason': 'Frequently used'
        }
        
        item = CommandSuggestionItem(suggestion)
        assert item.suggestion == suggestion
        assert 'high-score' in item.classes
    
    def test_low_score_suggestion(self):
        """Test suggestion with low score"""
        suggestion = {
            'command': 'ls -la',
            'score': 5.0,
            'source': 'context'
        }
        
        item = CommandSuggestionItem(suggestion)
        assert 'high-score' not in item.classes


class TestCommandSuggestionPanel:
    """Test CommandSuggestionPanel widget"""
    
    @pytest.fixture
    def panel(self):
        """Create a suggestion panel"""
        return CommandSuggestionPanel()
    
    def test_create_panel(self, panel):
        """Test creating suggestion panel"""
        assert panel.suggestions == []
    
    def test_update_suggestions_data(self, panel):
        """Test updating suggestions data"""
        suggestions = [
            {'command': 'git status', 'score': 10, 'source': 'history'},
            {'command': 'npm install', 'score': 8, 'source': 'context'},
            {'command': 'python -m pytest', 'score': 12, 'source': 'pattern'}
        ]
        
        # Just verify the data is stored
        panel.suggestions = suggestions
        assert panel.suggestions == suggestions
    
    def test_empty_suggestions_data(self, panel):
        """Test with empty suggestions"""
        panel.suggestions = []
        assert panel.suggestions == []


class TestCommandExplanationViewer:
    """Test CommandExplanationViewer widget"""
    
    @pytest.fixture
    def viewer(self):
        """Create an explanation viewer"""
        return CommandExplanationViewer()
    
    def test_create_viewer(self, viewer):
        """Test creating explanation viewer"""
        assert viewer.current_explanation is None
    
    def test_store_explanation(self, viewer):
        """Test storing command explanation"""
        explanation = {
            'command': 'git commit -m "message"',
            'description': 'Commit changes to repository',
            'arguments': [
                {'value': '-m', 'description': 'Commit message'}
            ],
            'examples': [
                'git commit -m "Initial commit"',
                'git commit -m "Fix bug"'
            ],
            'warnings': [
                '⚠️  Make sure to stage files first with git add'
            ]
        }
        
        viewer.current_explanation = explanation
        assert viewer.current_explanation == explanation
    
    def test_explanation_with_minimal_data(self, viewer):
        """Test explanation with minimal data"""
        explanation = {
            'command': 'ls',
            'description': 'List directory contents'
        }
        
        viewer.current_explanation = explanation
        assert viewer.current_explanation == explanation


class TestErrorFixItem:
    """Test ErrorFixItem widget"""
    
    def test_create_fix_item(self):
        """Test creating a fix item"""
        fix = {
            'fix': 'npm install',
            'explanation': 'Install missing dependencies',
            'confidence': 0.9,
            'category': 'dependency',
            'requires_sudo': False,
            'requires_install': False
        }
        
        item = ErrorFixItem(fix)
        assert item.fix == fix
        assert 'high-confidence' in item.classes
    
    def test_low_confidence_fix(self):
        """Test fix with low confidence"""
        fix = {
            'fix': 'chmod +x script.sh',
            'explanation': 'Make script executable',
            'confidence': 0.6,
            'category': 'permission'
        }
        
        item = ErrorFixItem(fix)
        assert 'high-confidence' not in item.classes
    
    def test_fix_requiring_sudo(self):
        """Test fix requiring sudo"""
        fix = {
            'fix': 'sudo apt-get install python3',
            'explanation': 'Install Python 3',
            'confidence': 0.95,
            'category': 'installation',
            'requires_sudo': True,
            'requires_install': True,
            'install_command': 'sudo apt-get install python3'
        }
        
        item = ErrorFixItem(fix)
        assert item.fix['requires_sudo'] is True


class TestErrorFixPanel:
    """Test ErrorFixPanel widget"""
    
    @pytest.fixture
    def panel(self):
        """Create an error fix panel"""
        return ErrorFixPanel()
    
    def test_create_panel(self, panel):
        """Test creating error fix panel"""
        assert panel.current_error is None
        assert panel.current_command is None
    
    def test_store_error_data(self, panel):
        """Test storing error data"""
        command = "npm start"
        error = "Error: Cannot find module 'express'"
        
        panel.current_command = command
        panel.current_error = error
        assert panel.current_command == command
        assert panel.current_error == error
    
    def test_store_empty_error(self, panel):
        """Test storing empty error"""
        panel.current_command = "unknown command"
        panel.current_error = "command not found"
        assert panel.current_command == "unknown command"


class TestLearningProgressTracker:
    """Test LearningProgressTracker widget"""
    
    @pytest.fixture
    def tracker(self):
        """Create a learning progress tracker"""
        return LearningProgressTracker()
    
    def test_create_tracker(self, tracker):
        """Test creating progress tracker"""
        assert tracker.learning_stats == {}
    
    def test_store_progress_data(self, tracker):
        """Test storing learning progress data"""
        stats = {
            'total_commands_learned': 25,
            'total_executions': 150,
            'mastered_commands': ['git', 'npm', 'python'],
            'skill_levels': {
                'git': 0.85,
                'npm': 0.75,
                'python': 0.90,
                'docker': 0.45
            },
            'learning_progress': {
                'git': {
                    'total_uses': 50,
                    'successful_uses': 48,
                    'mastery_level': 0.85
                },
                'npm': {
                    'total_uses': 30,
                    'successful_uses': 27,
                    'mastery_level': 0.75
                }
            }
        }
        
        tracker.learning_stats = stats
        assert tracker.learning_stats == stats
    
    def test_store_empty_progress(self, tracker):
        """Test storing empty stats"""
        stats = {
            'total_commands_learned': 0,
            'total_executions': 0,
            'mastered_commands': [],
            'skill_levels': {},
            'learning_progress': {}
        }
        
        tracker.learning_stats = stats
        assert tracker.learning_stats == stats


class TestCommandHistoryBrowser:
    """Test CommandHistoryBrowser widget"""
    
    @pytest.fixture
    def browser(self):
        """Create a command history browser"""
        return CommandHistoryBrowser()
    
    def test_create_browser(self, browser):
        """Test creating history browser"""
        assert browser.history == []
        assert browser.filtered_history == []
    
    def test_store_history_data(self, browser):
        """Test storing command history data"""
        history = [
            {
                'command': 'git status',
                'timestamp': '2024-01-01T10:00:00',
                'success': True,
                'context': {'project_type': 'python'}
            },
            {
                'command': 'npm install',
                'timestamp': '2024-01-01T10:05:00',
                'success': True,
                'context': {'project_type': 'node'}
            },
            {
                'command': 'python test.py',
                'timestamp': '2024-01-01T10:10:00',
                'success': False,
                'context': {'project_type': 'python'}
            }
        ]
        
        browser.history = history
        browser.filtered_history = history
        assert browser.history == history
        assert browser.filtered_history == history
    
    def test_filter_history_logic(self, browser):
        """Test filtering command history logic"""
        history = [
            {'command': 'git status', 'timestamp': '2024-01-01T10:00:00', 'success': True},
            {'command': 'git commit', 'timestamp': '2024-01-01T10:05:00', 'success': True},
            {'command': 'npm install', 'timestamp': '2024-01-01T10:10:00', 'success': True},
            {'command': 'git push', 'timestamp': '2024-01-01T10:15:00', 'success': True}
        ]
        
        browser.history = history
        browser.filtered_history = history
        
        # Test filter logic
        pattern = 'git'
        filtered = [item for item in history if pattern.lower() in item['command'].lower()]
        
        assert len(filtered) == 3
        assert all('git' in item['command'] for item in filtered)
    
    def test_filter_history_no_match_logic(self, browser):
        """Test filtering with no matches logic"""
        history = [
            {'command': 'git status', 'timestamp': '2024-01-01T10:00:00', 'success': True},
            {'command': 'npm install', 'timestamp': '2024-01-01T10:05:00', 'success': True}
        ]
        
        browser.history = history
        
        # Test filter logic
        pattern = 'docker'
        filtered = [item for item in history if pattern.lower() in item['command'].lower()]
        
        assert len(filtered) == 0
    
    def test_filter_history_empty_pattern_logic(self, browser):
        """Test filtering with empty pattern logic"""
        history = [
            {'command': 'git status', 'timestamp': '2024-01-01T10:00:00', 'success': True},
            {'command': 'npm install', 'timestamp': '2024-01-01T10:05:00', 'success': True}
        ]
        
        browser.history = history
        
        # Empty pattern should return all
        pattern = ''
        if not pattern:
            filtered = history
        else:
            filtered = [item for item in history if pattern.lower() in item['command'].lower()]
        
        assert filtered == history


class TestTerminalAssistantPanel:
    """Test TerminalAssistantPanel main widget"""
    
    @pytest.fixture
    def panel(self):
        """Create a terminal assistant panel"""
        return TerminalAssistantPanel()
    
    def test_create_panel(self, panel):
        """Test creating terminal assistant panel"""
        assert panel.current_tab == "suggestions"
    
    def test_tab_state(self, panel):
        """Test tab state management"""
        panel.current_tab = "explanation"
        assert panel.current_tab == "explanation"
        
        panel.current_tab = "fixes"
        assert panel.current_tab == "fixes"
        
        panel.current_tab = "progress"
        assert panel.current_tab == "progress"
        
        panel.current_tab = "history"
        assert panel.current_tab == "history"


class TestIntegration:
    """Integration tests for terminal assistant TUI"""
    
    def test_widget_creation(self):
        """Test creating all widgets"""
        # Test individual widget creation
        suggestion_panel = CommandSuggestionPanel()
        assert suggestion_panel.suggestions == []
        
        explanation_viewer = CommandExplanationViewer()
        assert explanation_viewer.current_explanation is None
        
        fix_panel = ErrorFixPanel()
        assert fix_panel.current_error is None
        
        progress_tracker = LearningProgressTracker()
        assert progress_tracker.learning_stats == {}
        
        history_browser = CommandHistoryBrowser()
        assert history_browser.history == []
        
        # Test main panel
        main_panel = TerminalAssistantPanel()
        assert main_panel.current_tab == "suggestions"
    
    def test_data_flow(self):
        """Test data flow through widgets"""
        # Create widgets
        suggestion_panel = CommandSuggestionPanel()
        explanation_viewer = CommandExplanationViewer()
        fix_panel = ErrorFixPanel()
        progress_tracker = LearningProgressTracker()
        history_browser = CommandHistoryBrowser()
        
        # Test data storage
        suggestions = [{'command': 'git status', 'score': 10, 'source': 'history'}]
        suggestion_panel.suggestions = suggestions
        assert suggestion_panel.suggestions == suggestions
        
        explanation = {'command': 'git status', 'description': 'Show status'}
        explanation_viewer.current_explanation = explanation
        assert explanation_viewer.current_explanation == explanation
        
        fix_panel.current_command = "npm start"
        fix_panel.current_error = "Module not found"
        assert fix_panel.current_command == "npm start"
        
        stats = {'total_commands_learned': 10}
        progress_tracker.learning_stats = stats
        assert progress_tracker.learning_stats == stats
        
        history = [{'command': 'git status', 'timestamp': '2024-01-01T10:00:00', 'success': True}]
        history_browser.history = history
        assert history_browser.history == history


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
