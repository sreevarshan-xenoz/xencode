"""
Unit tests for Terminal Assistant command history analysis
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from pathlib import Path
from collections import Counter

from xencode.features.terminal_assistant import (
    CommandPredictor,
    TerminalAssistantFeature,
    TerminalAssistantConfig
)
from xencode.features.base import FeatureConfig


class TestCommandHistoryAnalysis:
    """Test command history analysis features"""
    
    @pytest.fixture
    def predictor(self):
        """Create a CommandPredictor instance"""
        return CommandPredictor(history_size=100, suggestion_limit=5)
    
    @pytest_asyncio.fixture
    async def populated_predictor(self, predictor):
        """Create a predictor with sample history"""
        # Add sample commands
        commands = [
            ('git status', True, {'project_type': 'git'}),
            ('git add .', True, {'project_type': 'git'}),
            ('git commit -m "test"', True, {'project_type': 'git'}),
            ('git push', True, {'project_type': 'git'}),
            ('python test.py', True, {'project_type': 'python'}),
            ('python -m pytest', True, {'project_type': 'python'}),
            ('npm install', True, {'project_type': 'node'}),
            ('npm test', True, {'project_type': 'node'}),
            ('docker ps', True, None),
            ('docker build -t test .', False, None),
        ]
        
        for cmd, success, context in commands:
            await predictor.record(cmd, success, context)
        
        return predictor
    
    @pytest.mark.asyncio
    async def test_basic_recording(self, predictor):
        """Test basic command recording"""
        await predictor.record('ls -la', True)
        
        assert len(predictor.history) == 1
        assert predictor.history[0]['command'] == 'ls -la'
        assert predictor.history[0]['success'] is True
        assert predictor.command_frequency['ls -la'] == 1
    
    @pytest.mark.asyncio
    async def test_frequency_tracking(self, predictor):
        """Test command frequency tracking"""
        await predictor.record('git status', True)
        await predictor.record('git status', True)
        await predictor.record('git status', True)
        await predictor.record('git add .', True)
        
        assert predictor.command_frequency['git status'] == 3
        assert predictor.command_frequency['git add .'] == 1
    
    @pytest.mark.asyncio
    async def test_success_rate_tracking(self, predictor):
        """Test success rate tracking"""
        await predictor.record('test_cmd', True)
        await predictor.record('test_cmd', True)
        await predictor.record('test_cmd', False)
        
        stats = predictor.success_rates['test_cmd']
        assert stats['success'] == 2
        assert stats['failure'] == 1
        
        success_rate = predictor._calculate_success_rate('test_cmd')
        assert success_rate == pytest.approx(2/3)
    
    @pytest.mark.asyncio
    async def test_sequence_detection(self, predictor):
        """Test command sequence detection"""
        await predictor.record('git add .', True)
        await predictor.record('git commit -m "test"', True)
        await predictor.record('git add .', True)
        await predictor.record('git commit -m "test"', True)
        
        sequences = predictor.command_sequences['git add .']
        assert sequences['git commit -m "test"'] == 2
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, predictor):
        """Test command pattern detection"""
        await predictor.record('git status', True)
        await predictor.record('git add .', True)
        await predictor.record('git commit -m "test"', True)
        
        assert 'git' in predictor.command_patterns
        assert 'git status' in predictor.command_patterns['git']
        assert 'git add .' in predictor.command_patterns['git']
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_tracking(self, predictor):
        """Test temporal pattern tracking"""
        timestamp = datetime.now().isoformat()
        await predictor.record('morning_cmd', True)
        
        current_hour = datetime.now().hour
        hour_key = f"hour_{current_hour}"
        
        assert hour_key in predictor.temporal_patterns
        assert 'morning_cmd' in predictor.temporal_patterns[hour_key]
    
    @pytest.mark.asyncio
    async def test_context_pattern_tracking(self, predictor):
        """Test context-based pattern tracking"""
        context = {'project_type': 'python', 'directory': '/test/project'}
        await predictor.record('python test.py', True, context)
        
        assert 'python' in predictor.context_patterns
        assert 'python test.py' in predictor.context_patterns['python']
    
    @pytest.mark.asyncio
    async def test_prediction_with_partial(self, populated_predictor):
        """Test command prediction with partial input"""
        suggestions = await populated_predictor.predict(partial='git')
        
        assert len(suggestions) > 0
        assert all(s['command'].startswith('git') for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_prediction_with_context(self, populated_predictor):
        """Test command prediction with context"""
        context = {'project_type': 'python'}
        suggestions = await populated_predictor.predict(context=context)
        
        assert len(suggestions) > 0
        # Should include python-related commands
        python_cmds = [s for s in suggestions if 'python' in s['command'].lower()]
        assert len(python_cmds) > 0
    
    @pytest.mark.asyncio
    async def test_sequence_suggestions(self, populated_predictor):
        """Test sequence-based suggestions"""
        suggestions = populated_predictor._get_sequence_suggestions()
        
        # Should suggest commands that follow recent commands
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert 'command' in suggestion
            assert 'score' in suggestion
            assert suggestion['source'] == 'sequence'
    
    @pytest.mark.asyncio
    async def test_temporal_suggestions(self, populated_predictor):
        """Test temporal pattern suggestions"""
        suggestions = populated_predictor._get_temporal_suggestions()
        
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert 'command' in suggestion
            assert 'source' in suggestion
            assert suggestion['source'] == 'temporal'
    
    @pytest.mark.asyncio
    async def test_pattern_suggestions(self, populated_predictor):
        """Test pattern-based suggestions"""
        suggestions = populated_predictor._get_pattern_suggestions(partial='git')
        
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert suggestion['command'].startswith('git')
            assert suggestion['source'] == 'pattern'
    
    @pytest.mark.asyncio
    async def test_command_score_calculation(self, populated_predictor):
        """Test comprehensive command scoring"""
        context = {'project_type': 'git'}
        score = populated_predictor._calculate_command_score('git status', context)
        
        assert score > 0
        # Should have higher score due to frequency and context
        assert score > populated_predictor._calculate_command_score('docker ps', context)
    
    @pytest.mark.asyncio
    async def test_command_statistics(self, populated_predictor):
        """Test command statistics retrieval"""
        stats = await populated_predictor.get_command_statistics('git status')
        
        assert stats['command'] == 'git status'
        assert stats['frequency'] > 0
        assert 'success_rate' in stats
        assert 'last_used' in stats
        assert 'common_sequences' in stats
    
    @pytest.mark.asyncio
    async def test_overall_statistics(self, populated_predictor):
        """Test overall statistics"""
        stats = await populated_predictor.get_command_statistics()
        
        assert 'total_commands' in stats
        assert 'unique_commands' in stats
        assert 'most_frequent' in stats
        assert 'success_rate' in stats
        assert stats['total_commands'] == 10
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self, populated_predictor):
        """Test pattern analysis"""
        analysis = await populated_predictor.analyze_patterns()
        
        assert 'command_patterns' in analysis
        assert 'sequence_patterns' in analysis
        assert 'temporal_patterns' in analysis
        assert 'context_patterns' in analysis
        
        # Should detect git patterns
        assert 'git' in analysis['command_patterns']
    
    @pytest.mark.asyncio
    async def test_history_search(self, populated_predictor):
        """Test history search"""
        results = await populated_predictor.search_history('git')
        
        assert len(results) > 0
        assert all('git' in r['command'].lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_history_size_limit(self, predictor):
        """Test history size limiting"""
        predictor.history_size = 5
        
        for i in range(10):
            await predictor.record(f'cmd_{i}', True)
        
        assert len(predictor.history) == 5
        # Should keep most recent commands
        assert predictor.history[-1]['command'] == 'cmd_9'
    
    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, predictor):
        """Test success rate calculation"""
        await predictor.record('cmd1', True)
        await predictor.record('cmd1', True)
        await predictor.record('cmd1', False)
        
        rate = predictor._calculate_success_rate('cmd1')
        assert rate == pytest.approx(2/3)
        
        # Test with no history
        rate = predictor._calculate_success_rate('nonexistent')
        assert rate == 0.0
    
    @pytest.mark.asyncio
    async def test_overall_success_rate(self, populated_predictor):
        """Test overall success rate calculation"""
        rate = populated_predictor._calculate_overall_success_rate()
        
        assert 0.0 <= rate <= 1.0
        # 9 successes out of 10 commands
        assert rate == pytest.approx(0.9)
    
    @pytest.mark.asyncio
    async def test_last_used_timestamp(self, populated_predictor):
        """Test last used timestamp retrieval"""
        timestamp = populated_predictor._get_last_used('git status')
        
        assert timestamp is not None
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(timestamp)
    
    @pytest.mark.asyncio
    async def test_common_sequences(self, predictor):
        """Test common sequence retrieval"""
        await predictor.record('cmd1', True)
        await predictor.record('cmd2', True)
        await predictor.record('cmd1', True)
        await predictor.record('cmd2', True)
        await predictor.record('cmd1', True)
        await predictor.record('cmd3', True)
        
        sequences = predictor._get_common_sequences('cmd1')
        
        assert len(sequences) > 0
        assert sequences[0][0] == 'cmd2'  # Most common
        assert sequences[0][1] == 2  # Frequency


class TestTerminalAssistantIntegration:
    """Test Terminal Assistant integration with history analysis"""
    
    @pytest_asyncio.fixture
    async def terminal_assistant(self):
        """Create a Terminal Assistant instance"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={
                'history_size': 100,
                'context_aware': True,
                'learning_enabled': True
            }
        )
        assistant = TerminalAssistantFeature(config)
        await assistant.initialize()
        return assistant
    
    @pytest.mark.asyncio
    async def test_record_and_retrieve_statistics(self, terminal_assistant):
        """Test recording commands and retrieving statistics"""
        await terminal_assistant.record_command('test_cmd', True, {'project_type': 'test'})
        
        stats = await terminal_assistant.get_statistics('test_cmd')
        
        assert stats['command'] == 'test_cmd'
        assert stats['frequency'] == 1
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_integration(self, terminal_assistant):
        """Test pattern analysis through Terminal Assistant"""
        # Record some commands
        commands = [
            'git status',
            'git add .',
            'git commit -m "test"',
            'python test.py'
        ]
        
        for cmd in commands:
            await terminal_assistant.record_command(cmd, True)
        
        analysis = await terminal_assistant.analyze_patterns()
        
        assert 'command_patterns' in analysis
        assert 'git' in analysis['command_patterns']
    
    @pytest.mark.asyncio
    async def test_suggestions_with_history(self, terminal_assistant):
        """Test command suggestions with recorded history"""
        # Record some commands
        await terminal_assistant.record_command('git status', True, {'project_type': 'git'})
        await terminal_assistant.record_command('git add .', True, {'project_type': 'git'})
        
        # Get suggestions
        suggestions = await terminal_assistant.suggest_commands(partial='git')
        
        assert len(suggestions) > 0
        assert any('git' in s['command'] for s in suggestions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
