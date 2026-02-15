"""
Tests for Terminal Assistant CLI Commands

Tests all CLI commands for the Terminal Assistant feature:
- xencode terminal suggest
- xencode terminal explain
- xencode terminal fix
- xencode terminal history
- xencode terminal learn
- xencode terminal statistics
- xencode terminal patterns
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from click.testing import CliRunner
from xencode.cli import cli


@pytest.fixture
def cli_runner():
    """Create a CLI runner"""
    return CliRunner()


@pytest.fixture
def mock_terminal_feature():
    """Create a mock Terminal Assistant feature"""
    feature = Mock()
    feature._initialize = AsyncMock()
    feature._shutdown = AsyncMock()
    feature.suggest_commands = AsyncMock(return_value=[
        {
            'command': 'git status',
            'score': 10.0,
            'source': 'context',
            'explanation': 'Check repository status'
        },
        {
            'command': 'git pull',
            'score': 9.0,
            'source': 'frequency',
            'explanation': 'Update local repository'
        }
    ])
    feature.explain_command = AsyncMock(return_value={
        'command': 'git commit -m "message"',
        'description': 'Record changes to the repository',
        'arguments': [
            {'value': '-m', 'description': 'Commit message flag'},
            {'value': '"message"', 'description': 'Commit message text'}
        ],
        'examples': [
            'git commit -m "Initial commit"',
            'git commit -m "Fix bug"'
        ],
        'warnings': []
    })
    feature.fix_error = AsyncMock(return_value=[
        {
            'fix': 'npm install',
            'explanation': 'Install missing dependencies',
            'confidence': 0.95,
            'category': 'missing_dependency',
            'requires_sudo': False,
            'requires_install': False,
            'install_command': None,
            'documentation_url': 'https://docs.npmjs.com',
            'alternative_commands': ['yarn install']
        }
    ])
    feature.search_history = AsyncMock(return_value=[
        {
            'command': 'git commit -m "test"',
            'timestamp': '2024-01-15T10:30:00',
            'success': True,
            'context': {}
        },
        {
            'command': 'git push origin main',
            'timestamp': '2024-01-15T10:31:00',
            'success': True,
            'context': {}
        }
    ])
    feature.get_statistics = AsyncMock(return_value={
        'total_commands': 150,
        'unique_commands': 45,
        'most_frequent': [
            ('git status', 25),
            ('git commit', 20),
            ('npm install', 15)
        ],
        'success_rate': 0.92,
        'patterns_detected': 12
    })
    feature.analyze_patterns = AsyncMock(return_value={
        'command_patterns': {
            'git': {
                'count': 5,
                'examples': ['git status', 'git commit', 'git push']
            }
        },
        'sequence_patterns': [
            {'from': 'git add', 'to': 'git commit', 'frequency': 10}
        ],
        'temporal_patterns': {
            'hour_9': [('git status', 5), ('npm install', 3)]
        },
        'context_patterns': {
            'python': [('pytest', 8), ('pip install', 6)]
        }
    })
    feature.learning_engine = Mock()
    feature.learning_engine.user_skill_level = {
        'git': 0.85,
        'npm': 0.70,
        'docker': 0.45
    }
    return feature


class TestTerminalSuggestCommand:
    """Test 'xencode terminal suggest' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test basic command suggestion"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest'])
        
        assert result.exit_code == 0
        assert 'Command Suggestions' in result.output
        assert 'git status' in result.output
        assert 'git pull' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_with_partial(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command suggestion with partial input"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest', '--partial', 'git'])
        
        assert result.exit_code == 0
        assert 'Command Suggestions' in result.output
        mock_terminal_feature.suggest_commands.assert_called_once()
        call_args = mock_terminal_feature.suggest_commands.call_args
        assert call_args[1]['partial'] == 'git'
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_with_context(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command suggestion with context"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest', 'python project'])
        
        assert result.exit_code == 0
        mock_terminal_feature.suggest_commands.assert_called_once()
        call_args = mock_terminal_feature.suggest_commands.call_args
        assert call_args[1]['context'] == 'python project'
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_with_limit(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command suggestion with custom limit"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest', '--limit', '10'])
        
        assert result.exit_code == 0
        # Verify config was created with correct limit
        mock_feature_class.assert_called_once()
        config = mock_feature_class.call_args[0][0]
        assert config.config['suggestion_limit'] == 10
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_no_results(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command suggestion with no results"""
        mock_terminal_feature.suggest_commands = AsyncMock(return_value=[])
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest'])
        
        assert result.exit_code == 0
        assert 'No suggestions found' in result.output


class TestTerminalExplainCommand:
    """Test 'xencode terminal explain' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_explain_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test basic command explanation"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'explain', 'git commit -m "message"'])
        
        assert result.exit_code == 0
        assert 'Command Explanation' in result.output
        assert 'git commit -m "message"' in result.output
        assert 'Record changes to the repository' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_explain_with_arguments(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command explanation shows arguments"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'explain', 'git commit -m "test"'])
        
        assert result.exit_code == 0
        assert 'Arguments:' in result.output
        assert '-m' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_explain_with_examples(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command explanation shows examples"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'explain', 'git commit'])
        
        assert result.exit_code == 0
        assert 'Examples:' in result.output
        assert 'Initial commit' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_explain_with_warnings(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test command explanation shows warnings for dangerous commands"""
        mock_terminal_feature.explain_command = AsyncMock(return_value={
            'command': 'rm -rf /',
            'description': 'Remove files recursively',
            'arguments': [],
            'examples': [],
            'warnings': ['⚠️  This will permanently delete files without confirmation']
        })
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'explain', 'rm -rf /'])
        
        assert result.exit_code == 0
        assert 'Warnings:' in result.output
        assert 'permanently delete' in result.output


class TestTerminalFixCommand:
    """Test 'xencode terminal fix' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_fix_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test basic error fix suggestion"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'fix', 'command not found: npm'])
        
        assert result.exit_code == 0
        assert 'fix suggestions' in result.output
        assert 'npm install' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_fix_with_command(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test error fix with original command"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, [
            'terminal', 'fix', 'Permission denied',
            '--command', 'rm file.txt'
        ])
        
        assert result.exit_code == 0
        mock_terminal_feature.fix_error.assert_called_once()
        call_args = mock_terminal_feature.fix_error.call_args
        assert call_args[0][0] == 'rm file.txt'
        assert call_args[0][1] == 'Permission denied'
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_fix_shows_confidence(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test fix suggestion shows confidence score"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'fix', 'error message'])
        
        assert result.exit_code == 0
        assert 'Confidence:' in result.output
        assert '95' in result.output  # 0.95 = 95%
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_fix_shows_alternatives(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test fix suggestion shows alternative commands"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'fix', 'error'])
        
        assert result.exit_code == 0
        assert 'Alternatives:' in result.output
        assert 'yarn install' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_fix_with_limit(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test fix suggestion with custom limit"""
        mock_terminal_feature.fix_error = AsyncMock(return_value=[
            {'fix': f'fix{i}', 'explanation': f'exp{i}', 'confidence': 0.9,
             'category': 'test', 'requires_sudo': False, 'requires_install': False}
            for i in range(10)
        ])
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'fix', 'error', '--limit', '3'])
        
        assert result.exit_code == 0
        # Should only show 3 fixes
        assert 'Fix #1' in result.output
        assert 'Fix #2' in result.output
        assert 'Fix #3' in result.output
        assert 'Fix #4' not in result.output


class TestTerminalHistoryCommand:
    """Test 'xencode terminal history' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_history_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test basic history search"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'history', 'git'])
        
        assert result.exit_code == 0
        assert 'Command History' in result.output
        assert 'git commit' in result.output
        assert 'git push' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_history_shows_timestamps(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test history shows timestamps"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'history', 'git'])
        
        assert result.exit_code == 0
        assert '2024-01-15' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_history_shows_status(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test history shows success/failure status"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'history', 'git'])
        
        assert result.exit_code == 0
        assert 'Success' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_history_with_limit(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test history with custom limit"""
        # Create 30 results
        mock_terminal_feature.search_history = AsyncMock(return_value=[
            {
                'command': f'command{i}',
                'timestamp': '2024-01-15T10:30:00',
                'success': True
            }
            for i in range(30)
        ])
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'history', 'test', '--limit', '5'])
        
        assert result.exit_code == 0
        assert 'Showing 5 of 30 results' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_history_no_results(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test history with no matching commands"""
        mock_terminal_feature.search_history = AsyncMock(return_value=[])
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'history', 'nonexistent'])
        
        assert result.exit_code == 0
        assert 'No matching commands found' in result.output


class TestTerminalLearnCommand:
    """Test 'xencode terminal learn' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_learn_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test learning mode activation"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'learn'])
        
        assert result.exit_code == 0
        assert 'Learning Mode' in result.output
        assert 'Welcome to Terminal Assistant Learning Mode' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_learn_shows_skill_levels(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test learning mode shows current skill levels"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'learn'])
        
        assert result.exit_code == 0
        assert 'Your Current Skill Levels' in result.output
        assert 'git' in result.output
        assert 'npm' in result.output
        assert 'docker' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_learn_shows_progress_bars(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test learning mode shows progress bars"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'learn'])
        
        assert result.exit_code == 0
        # Progress bars use █ and ░ characters
        assert '█' in result.output or 'Progress' in result.output


class TestTerminalStatisticsCommand:
    """Test 'xencode terminal statistics' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_statistics_overall(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test overall statistics"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'statistics'])
        
        assert result.exit_code == 0
        assert 'Overall Statistics' in result.output
        assert 'Total Commands: 150' in result.output
        assert 'Unique Commands: 45' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_statistics_shows_most_frequent(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test statistics shows most frequent commands"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'statistics'])
        
        assert result.exit_code == 0
        assert 'Most Frequent Commands' in result.output
        assert 'git status' in result.output
        assert 'git commit' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_statistics_specific_command(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test statistics for specific command"""
        mock_terminal_feature.get_statistics = AsyncMock(return_value={
            'command': 'git commit',
            'frequency': 20,
            'success_rate': 0.95,
            'last_used': '2024-01-15T10:30:00',
            'common_sequences': [('git push', 15), ('git status', 10)],
            'temporal_usage': {'hour_9': 5, 'hour_14': 8}
        })
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'statistics', '--command', 'git commit'])
        
        assert result.exit_code == 0
        assert "Statistics for 'git commit'" in result.output
        assert 'Frequency: 20 times' in result.output
        assert 'Success Rate: 95' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_statistics_shows_sequences(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test statistics shows common command sequences"""
        mock_terminal_feature.get_statistics = AsyncMock(return_value={
            'command': 'git add',
            'frequency': 25,
            'success_rate': 1.0,
            'last_used': '2024-01-15T10:30:00',
            'common_sequences': [('git commit', 20), ('git status', 5)],
            'temporal_usage': {}
        })
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'statistics', '-c', 'git add'])
        
        assert result.exit_code == 0
        assert 'Common Sequences' in result.output
        assert 'git commit' in result.output


class TestTerminalPatternsCommand:
    """Test 'xencode terminal patterns' command"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_patterns_basic(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test basic pattern analysis"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'patterns'])
        
        assert result.exit_code == 0
        assert 'Analyzing command patterns' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_patterns_shows_command_patterns(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test patterns shows command patterns"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'patterns'])
        
        assert result.exit_code == 0
        assert 'Command Patterns' in result.output
        assert 'git' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_patterns_shows_sequences(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test patterns shows sequence patterns"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'patterns'])
        
        assert result.exit_code == 0
        assert 'Sequence Patterns' in result.output
        assert 'git add' in result.output
        assert 'git commit' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_patterns_shows_temporal(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test patterns shows temporal patterns"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'patterns'])
        
        assert result.exit_code == 0
        assert 'Temporal Patterns' in result.output
        assert 'hour_9' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_patterns_shows_context(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test patterns shows context patterns"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'patterns'])
        
        assert result.exit_code == 0
        assert 'Context Patterns' in result.output
        assert 'python' in result.output
        assert 'pytest' in result.output


class TestCLIErrorHandling:
    """Test CLI error handling"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_suggest_handles_exception(self, mock_feature_class, cli_runner):
        """Test suggest command handles exceptions gracefully"""
        mock_feature = Mock()
        mock_feature._initialize = AsyncMock(side_effect=Exception("Test error"))
        mock_feature_class.return_value = mock_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest'])
        
        assert result.exit_code == 1
        assert 'Suggestion failed' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_explain_handles_exception(self, mock_feature_class, cli_runner):
        """Test explain command handles exceptions gracefully"""
        mock_feature = Mock()
        mock_feature._initialize = AsyncMock(side_effect=Exception("Test error"))
        mock_feature_class.return_value = mock_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'explain', 'test'])
        
        assert result.exit_code == 1
        assert 'Explanation failed' in result.output
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_verbose_mode_shows_traceback(self, mock_feature_class, cli_runner):
        """Test verbose mode shows full traceback"""
        mock_feature = Mock()
        mock_feature._initialize = AsyncMock(side_effect=Exception("Test error"))
        mock_feature_class.return_value = mock_feature
        
        result = cli_runner.invoke(cli, ['--verbose', 'terminal', 'suggest'])
        
        assert result.exit_code == 1
        # In verbose mode, should show traceback
        assert 'Traceback' in result.output or 'Test error' in result.output


class TestCLIIntegration:
    """Integration tests for CLI commands"""
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_full_workflow(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test a full workflow: suggest -> explain -> statistics"""
        mock_feature_class.return_value = mock_terminal_feature
        
        # Get suggestions
        result1 = cli_runner.invoke(cli, ['terminal', 'suggest'])
        assert result1.exit_code == 0
        
        # Explain a command
        result2 = cli_runner.invoke(cli, ['terminal', 'explain', 'git status'])
        assert result2.exit_code == 0
        
        # Check statistics
        result3 = cli_runner.invoke(cli, ['terminal', 'statistics'])
        assert result3.exit_code == 0
    
    @patch('xencode.features.terminal_assistant.TerminalAssistantFeature')
    def test_feature_initialization_and_shutdown(self, mock_feature_class, cli_runner, mock_terminal_feature):
        """Test feature is properly initialized and shut down"""
        mock_feature_class.return_value = mock_terminal_feature
        
        result = cli_runner.invoke(cli, ['terminal', 'suggest'])
        
        assert result.exit_code == 0
        mock_terminal_feature._initialize.assert_called_once()
        mock_terminal_feature._shutdown.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
