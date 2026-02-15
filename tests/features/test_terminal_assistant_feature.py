"""
Unit Tests for Terminal Assistant Feature

Tests the TerminalAssistantFeature class including:
- Feature initialization and lifecycle
- Integration between all components (CommandPredictor, ContextAnalyzer, LearningEngine, ErrorHandler)
- Configuration handling
- Edge cases and error conditions
- API endpoint functionality
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from xencode.features.terminal_assistant import (
    TerminalAssistantFeature,
    TerminalAssistantConfig,
    CommandPredictor,
    ContextAnalyzer,
    LearningEngine
)
from xencode.features.base import FeatureConfig
from xencode.features.error_handler_enhanced import EnhancedErrorHandler


@pytest.fixture
def feature_config():
    """Create a basic feature configuration"""
    return FeatureConfig(
        name='terminal_assistant',
        enabled=True,
        config={
            'history_size': 100,
            'context_aware': True,
            'learning_enabled': True,
            'suggestion_limit': 5,
            'error_fix_enabled': True,
            'shell_type': 'bash'
        }
    )


@pytest.fixture
def minimal_config():
    """Create minimal feature configuration"""
    return FeatureConfig(
        name='terminal_assistant',
        enabled=True,
        config={}
    )


@pytest_asyncio.fixture
async def terminal_feature(feature_config):
    """Create and initialize a Terminal Assistant feature"""
    feature = TerminalAssistantFeature(feature_config)
    await feature.initialize()
    yield feature
    await feature.shutdown()


class TestTerminalAssistantConfig:
    """Test TerminalAssistantConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TerminalAssistantConfig()
        
        assert config.history_size == 1000
        assert config.context_aware is True
        assert config.learning_enabled is True
        assert config.suggestion_limit == 5
        assert config.error_fix_enabled is True
        assert config.shell_type == 'bash'
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TerminalAssistantConfig(
            history_size=500,
            context_aware=False,
            learning_enabled=False,
            suggestion_limit=10,
            error_fix_enabled=False,
            shell_type='zsh'
        )
        
        assert config.history_size == 500
        assert config.context_aware is False
        assert config.learning_enabled is False
        assert config.suggestion_limit == 10
        assert config.error_fix_enabled is False
        assert config.shell_type == 'zsh'
    
    def test_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            'history_size': 200,
            'context_aware': False,
            'learning_enabled': True,
            'suggestion_limit': 3,
            'error_fix_enabled': True,
            'shell_type': 'fish'
        }
        
        config = TerminalAssistantConfig.from_dict(data)
        
        assert config.history_size == 200
        assert config.context_aware is False
        assert config.learning_enabled is True
        assert config.suggestion_limit == 3
        assert config.error_fix_enabled is True
        assert config.shell_type == 'fish'
    
    def test_from_dict_partial(self):
        """Test creating config from partial dictionary"""
        data = {'history_size': 300}
        
        config = TerminalAssistantConfig.from_dict(data)
        
        assert config.history_size == 300
        assert config.context_aware is True  # Default
        assert config.learning_enabled is True  # Default
    
    def test_from_dict_empty(self):
        """Test creating config from empty dictionary"""
        config = TerminalAssistantConfig.from_dict({})
        
        assert config.history_size == 1000  # Default
        assert config.context_aware is True  # Default


class TestFeatureInitialization:
    """Test feature initialization and lifecycle"""
    
    @pytest.mark.asyncio
    async def test_feature_creation(self, feature_config):
        """Test creating feature instance"""
        feature = TerminalAssistantFeature(feature_config)
        
        assert feature.name == 'terminal_assistant'
        assert feature.description == 'Context-aware command suggestions and intelligent error handling'
        assert feature.command_predictor is None
        assert feature.context_analyzer is None
        assert feature.learning_engine is None
        assert feature.error_handler is None
    
    @pytest.mark.asyncio
    async def test_feature_initialization(self, feature_config):
        """Test feature initialization"""
        feature = TerminalAssistantFeature(feature_config)
        await feature.initialize()
        
        # Verify all components are initialized
        assert feature.command_predictor is not None
        assert isinstance(feature.command_predictor, CommandPredictor)
        assert feature.context_analyzer is not None
        assert isinstance(feature.context_analyzer, ContextAnalyzer)
        assert feature.learning_engine is not None
        assert isinstance(feature.learning_engine, LearningEngine)
        assert feature.error_handler is not None
        assert isinstance(feature.error_handler, EnhancedErrorHandler)
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_feature_initialization_with_minimal_config(self, minimal_config):
        """Test initialization with minimal config"""
        feature = TerminalAssistantFeature(minimal_config)
        await feature.initialize()
        
        # Should use defaults
        assert feature.command_predictor is not None
        assert feature.context_analyzer is not None
        assert feature.learning_engine is not None
        assert feature.error_handler is not None
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_feature_shutdown(self, terminal_feature):
        """Test feature shutdown"""
        # Feature is already initialized via fixture
        assert terminal_feature.command_predictor is not None
        
        # Shutdown should save state
        await terminal_feature.shutdown()
        
        # Components should still exist but state should be saved
        assert terminal_feature.command_predictor is not None
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, feature_config):
        """Test that double initialization doesn't break"""
        feature = TerminalAssistantFeature(feature_config)
        
        await feature.initialize()
        await feature.initialize()  # Second init
        
        # Should still work
        assert feature.command_predictor is not None
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self, feature_config):
        """Test shutdown without initialization"""
        feature = TerminalAssistantFeature(feature_config)
        
        # Should not raise error
        await feature.shutdown()


class TestComponentIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_suggest_commands_integration(self, terminal_feature):
        """Test command suggestion integrates all components"""
        # Record some commands first
        await terminal_feature.record_command('git status', True, {'project_type': 'python'})
        await terminal_feature.record_command('python test.py', True, {'project_type': 'python'})
        
        # Get suggestions
        suggestions = await terminal_feature.suggest_commands(
            context='python project',
            partial='git'
        )
        
        # Should return suggestions
        assert isinstance(suggestions, list)
        # Suggestions should be enhanced by learning engine
        if suggestions:
            assert 'command' in suggestions[0]
            assert 'score' in suggestions[0]
    
    @pytest.mark.asyncio
    async def test_explain_command_basic(self, terminal_feature):
        """Test command explanation"""
        explanation = await terminal_feature.explain_command('git status')
        
        assert explanation['command'] == 'git status'
        assert 'description' in explanation
        assert 'arguments' in explanation
        assert 'examples' in explanation
        assert 'warnings' in explanation
    
    @pytest.mark.asyncio
    async def test_explain_command_with_flags(self, terminal_feature):
        """Test explaining command with flags"""
        explanation = await terminal_feature.explain_command('ls -la')
        
        assert explanation['command'] == 'ls -la'
        assert len(explanation['arguments']) > 0
    
    @pytest.mark.asyncio
    async def test_explain_dangerous_command(self, terminal_feature):
        """Test explaining dangerous command shows warnings"""
        explanation = await terminal_feature.explain_command('rm -rf /')
        
        assert len(explanation['warnings']) > 0
        assert any('permanently delete' in w.lower() for w in explanation['warnings'])
    
    @pytest.mark.asyncio
    async def test_fix_error_integration(self, terminal_feature):
        """Test error fixing integrates error handler"""
        command = 'pyhton script.py'
        error = 'bash: pyhton: command not found'
        
        fixes = await terminal_feature.fix_error(command, error)
        
        assert isinstance(fixes, list)
        if fixes:
            assert 'fix' in fixes[0]
            assert 'explanation' in fixes[0]
            assert 'confidence' in fixes[0]
    
    @pytest.mark.asyncio
    async def test_fix_error_with_context(self, terminal_feature):
        """Test error fixing with context"""
        command = 'python script.py'
        error = 'ModuleNotFoundError: No module named "requests"'
        context = {'project_type': 'python'}
        
        fixes = await terminal_feature.fix_error(command, error, context)
        
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_record_command_updates_all_components(self, terminal_feature):
        """Test recording command updates all components"""
        command = 'git commit -m "test"'
        context = {'project_type': 'git'}
        
        await terminal_feature.record_command(command, True, context)
        
        # Verify command was recorded in predictor
        assert command in terminal_feature.command_predictor.command_frequency
        
        # Verify learning engine was updated
        assert command in terminal_feature.learning_engine.patterns
    
    @pytest.mark.asyncio
    async def test_record_successful_fix(self, terminal_feature):
        """Test recording successful fix"""
        original = 'pyhton script.py'
        error = 'command not found'
        fix = 'python script.py'
        
        await terminal_feature.record_successful_fix(original, error, fix)
        
        # Verify fix was recorded
        error_key = error[:100]
        assert error_key in terminal_feature.error_handler.successful_fixes
    
    @pytest.mark.asyncio
    async def test_search_history(self, terminal_feature):
        """Test searching command history"""
        # Record some commands
        await terminal_feature.record_command('git status', True)
        await terminal_feature.record_command('git commit', True)
        await terminal_feature.record_command('npm install', True)
        
        # Search for git commands
        results = await terminal_feature.search_history('git')
        
        assert isinstance(results, list)
        assert all('git' in r['command'].lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_get_statistics_overall(self, feature_config, tmp_path):
        """Test getting overall statistics"""
        # Create fresh feature instance with isolated storage
        with patch('pathlib.Path.home', return_value=tmp_path):
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            # Record some commands
            await feature.record_command('git status', True)
            await feature.record_command('git status', True)
            await feature.record_command('npm install', False)
            
            stats = await feature.get_statistics()
            
            assert 'total_commands' in stats
            assert 'unique_commands' in stats
            assert 'most_frequent' in stats
            assert 'success_rate' in stats
            assert stats['total_commands'] == 3
            assert stats['unique_commands'] >= 1  # At least git status
            
            await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_statistics_specific_command(self, feature_config, tmp_path):
        """Test getting statistics for specific command"""
        # Create fresh feature instance with isolated storage
        with patch('pathlib.Path.home', return_value=tmp_path):
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            # Record commands
            await feature.record_command('git status', True)
            await feature.record_command('git status', True)
            await feature.record_command('git status', False)
            
            stats = await feature.get_statistics('git status')
            
            assert stats['command'] == 'git status'
            assert stats['frequency'] == 2  # Only successful ones counted
            assert 'success_rate' in stats
            assert 'last_used' in stats
            
            await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_patterns(self, terminal_feature):
        """Test pattern analysis"""
        # Record commands with patterns
        await terminal_feature.record_command('git add .', True)
        await terminal_feature.record_command('git commit -m "test"', True)
        await terminal_feature.record_command('git push', True)
        
        analysis = await terminal_feature.analyze_patterns()
        
        assert 'command_patterns' in analysis
        assert 'sequence_patterns' in analysis
        assert 'temporal_patterns' in analysis
        assert 'context_patterns' in analysis


class TestCommandParsing:
    """Test command parsing functionality"""
    
    @pytest.mark.asyncio
    async def test_parse_simple_command(self, terminal_feature):
        """Test parsing simple command"""
        parsed = terminal_feature._parse_command('ls')
        
        assert parsed['base'] == 'ls'
        assert parsed['args'] == []
        assert parsed['flags'] == []
    
    @pytest.mark.asyncio
    async def test_parse_command_with_args(self, terminal_feature):
        """Test parsing command with arguments"""
        parsed = terminal_feature._parse_command('cd /home/user')
        
        assert parsed['base'] == 'cd'
        assert '/home/user' in parsed['args']
    
    @pytest.mark.asyncio
    async def test_parse_command_with_flags(self, terminal_feature):
        """Test parsing command with flags"""
        parsed = terminal_feature._parse_command('ls -la')
        
        assert parsed['base'] == 'ls'
        assert '-la' in parsed['flags']
    
    @pytest.mark.asyncio
    async def test_parse_command_with_mixed(self, terminal_feature):
        """Test parsing command with flags and arguments"""
        parsed = terminal_feature._parse_command('git commit -m "test message"')
        
        assert parsed['base'] == 'git'
        assert '-m' in parsed['flags']
        assert 'commit' in parsed['args']
    
    @pytest.mark.asyncio
    async def test_parse_empty_command(self, terminal_feature):
        """Test parsing empty command"""
        parsed = terminal_feature._parse_command('')
        
        assert parsed['base'] == ''
        assert parsed['args'] == []
        assert parsed['flags'] == []


class TestCommandDescriptions:
    """Test command description functionality"""
    
    @pytest.mark.asyncio
    async def test_get_known_command_description(self, terminal_feature):
        """Test getting description for known command"""
        desc = terminal_feature._get_command_description('git')
        assert 'version control' in desc.lower()
    
    @pytest.mark.asyncio
    async def test_get_unknown_command_description(self, terminal_feature):
        """Test getting description for unknown command"""
        desc = terminal_feature._get_command_description('unknown_cmd')
        assert 'execute' in desc.lower()
        assert 'unknown_cmd' in desc.lower()
    
    @pytest.mark.asyncio
    async def test_explain_common_flags(self, terminal_feature):
        """Test explaining common flags"""
        flag_desc = terminal_feature._explain_flag('ls', '-l')
        assert 'long format' in flag_desc.lower()
        
        flag_desc = terminal_feature._explain_flag('ls', '-a')
        assert 'all' in flag_desc.lower() or 'hidden' in flag_desc.lower()
    
    @pytest.mark.asyncio
    async def test_explain_unknown_flag(self, terminal_feature):
        """Test explaining unknown flag"""
        flag_desc = terminal_feature._explain_flag('cmd', '--unknown')
        assert '--unknown' in flag_desc
    
    @pytest.mark.asyncio
    async def test_get_command_examples(self, terminal_feature):
        """Test getting command examples"""
        examples = terminal_feature._get_command_examples('git')
        assert len(examples) > 0
        assert any('git status' in ex for ex in examples)
    
    @pytest.mark.asyncio
    async def test_get_examples_unknown_command(self, terminal_feature):
        """Test getting examples for unknown command"""
        examples = terminal_feature._get_command_examples('unknown_cmd')
        assert examples == []


class TestWarnings:
    """Test command warning functionality"""
    
    @pytest.mark.asyncio
    async def test_warning_for_rm_rf(self, terminal_feature):
        """Test warning for rm -rf command"""
        parsed = terminal_feature._parse_command('rm -rf /path')
        warnings = terminal_feature._get_command_warnings(parsed)
        
        assert len(warnings) > 0
        assert any('permanently delete' in w.lower() for w in warnings)
    
    @pytest.mark.asyncio
    async def test_warning_for_sudo(self, terminal_feature):
        """Test warning for sudo command"""
        parsed = terminal_feature._parse_command('sudo apt-get install')
        warnings = terminal_feature._get_command_warnings(parsed)
        
        assert len(warnings) > 0
        assert any('elevated privileges' in w.lower() for w in warnings)
    
    @pytest.mark.asyncio
    async def test_warning_for_chmod_777(self, terminal_feature):
        """Test warning for chmod 777"""
        parsed = terminal_feature._parse_command('chmod 777 file.txt')
        warnings = terminal_feature._get_command_warnings(parsed)
        
        assert len(warnings) > 0
        assert any('security risk' in w.lower() for w in warnings)
    
    @pytest.mark.asyncio
    async def test_no_warning_for_safe_command(self, terminal_feature):
        """Test no warning for safe command"""
        parsed = terminal_feature._parse_command('ls -la')
        warnings = terminal_feature._get_command_warnings(parsed)
        
        assert len(warnings) == 0


class TestAPIEndpoints:
    """Test API endpoint definitions"""
    
    @pytest.mark.asyncio
    async def test_get_api_endpoints(self, terminal_feature):
        """Test getting API endpoints"""
        endpoints = terminal_feature.get_api_endpoints()
        
        assert isinstance(endpoints, list)
        assert len(endpoints) > 0
        
        # Verify endpoint structure
        for endpoint in endpoints:
            assert 'path' in endpoint
            assert 'method' in endpoint
            assert 'handler' in endpoint
    
    @pytest.mark.asyncio
    async def test_api_endpoint_paths(self, terminal_feature):
        """Test API endpoint paths"""
        endpoints = terminal_feature.get_api_endpoints()
        paths = [ep['path'] for ep in endpoints]
        
        assert '/api/terminal/suggest' in paths
        assert '/api/terminal/explain' in paths
        assert '/api/terminal/fix' in paths
        assert '/api/terminal/history' in paths
        assert '/api/terminal/statistics' in paths
        assert '/api/terminal/patterns' in paths
    
    @pytest.mark.asyncio
    async def test_api_endpoint_handlers(self, terminal_feature):
        """Test API endpoint handlers are callable"""
        endpoints = terminal_feature.get_api_endpoints()
        
        for endpoint in endpoints:
            handler = endpoint['handler']
            assert callable(handler)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_suggest_with_no_history(self, terminal_feature):
        """Test suggestions with no command history"""
        suggestions = await terminal_feature.suggest_commands()
        
        # Should return context-based suggestions
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_suggest_with_empty_partial(self, terminal_feature):
        """Test suggestions with empty partial"""
        suggestions = await terminal_feature.suggest_commands(partial='')
        
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_explain_empty_command(self, terminal_feature):
        """Test explaining empty command"""
        explanation = await terminal_feature.explain_command('')
        
        assert explanation['command'] == ''
        assert 'description' in explanation
    
    @pytest.mark.asyncio
    async def test_fix_error_empty_command(self, terminal_feature):
        """Test fixing error with empty command"""
        fixes = await terminal_feature.fix_error('', 'some error')
        
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_fix_error_empty_error(self, terminal_feature):
        """Test fixing error with empty error message"""
        fixes = await terminal_feature.fix_error('some command', '')
        
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_search_history_no_matches(self, terminal_feature):
        """Test searching history with no matches"""
        await terminal_feature.record_command('git status', True)
        
        results = await terminal_feature.search_history('nonexistent')
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_history_empty_pattern(self, terminal_feature):
        """Test searching history with empty pattern"""
        await terminal_feature.record_command('git status', True)
        
        results = await terminal_feature.search_history('')
        
        # Empty pattern should match all
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_statistics_no_commands(self, feature_config, tmp_path):
        """Test statistics with no recorded commands"""
        # Create fresh feature instance with isolated storage
        with patch('pathlib.Path.home', return_value=tmp_path):
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            stats = await feature.get_statistics()
            
            assert stats['total_commands'] == 0
            assert stats['unique_commands'] == 0
            
            await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_statistics_nonexistent_command(self, terminal_feature):
        """Test statistics for nonexistent command"""
        stats = await terminal_feature.get_statistics('nonexistent_command')
        
        assert stats['command'] == 'nonexistent_command'
        assert stats['frequency'] == 0
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_no_data(self, terminal_feature):
        """Test pattern analysis with no data"""
        analysis = await terminal_feature.analyze_patterns()
        
        assert 'command_patterns' in analysis
        assert 'sequence_patterns' in analysis


class TestConfigurationVariations:
    """Test different configuration variations"""
    
    @pytest.mark.asyncio
    async def test_disabled_context_awareness(self):
        """Test with context awareness disabled"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'context_aware': False}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        # Context analyzer should be disabled
        assert feature.context_analyzer.enabled is False
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_disabled_learning(self):
        """Test with learning disabled"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'learning_enabled': False}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        # Learning engine should be disabled
        assert feature.learning_engine.enabled is False
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_disabled_error_fixing(self):
        """Test with error fixing disabled"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'error_fix_enabled': False}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        # Error handler should be disabled
        assert feature.error_handler.enabled is False
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_custom_history_size(self):
        """Test with custom history size"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'history_size': 50}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        assert feature.command_predictor.history_size == 50
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_custom_suggestion_limit(self):
        """Test with custom suggestion limit"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'suggestion_limit': 10}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        assert feature.command_predictor.suggestion_limit == 10
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_different_shell_types(self):
        """Test with different shell types"""
        for shell in ['bash', 'zsh', 'fish', 'powershell']:
            config = FeatureConfig(
                name='terminal_assistant',
                enabled=True,
                config={'shell_type': shell}
            )
            
            feature = TerminalAssistantFeature(config)
            await feature.initialize()
            
            assert feature.ta_config.shell_type == shell
            
            await feature.shutdown()


class TestConcurrency:
    """Test concurrent operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_suggestions(self, terminal_feature):
        """Test concurrent suggestion requests"""
        # Make multiple concurrent requests
        tasks = [
            terminal_feature.suggest_commands(partial='git'),
            terminal_feature.suggest_commands(partial='npm'),
            terminal_feature.suggest_commands(partial='python')
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_recordings(self, feature_config, tmp_path):
        """Test concurrent command recordings"""
        # Create fresh feature instance with isolated storage
        with patch('pathlib.Path.home', return_value=tmp_path):
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            # Record multiple commands concurrently
            tasks = [
                feature.record_command('git status', True),
                feature.record_command('npm install', True),
                feature.record_command('python test.py', True)
            ]
            
            await asyncio.gather(*tasks)
            
            # All commands should be recorded
            stats = await feature.get_statistics()
            assert stats['total_commands'] == 3
            
            await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, terminal_feature):
        """Test concurrent mixed operations"""
        # Mix of different operations
        tasks = [
            terminal_feature.record_command('git status', True),
            terminal_feature.suggest_commands(partial='git'),
            terminal_feature.explain_command('ls -la'),
            terminal_feature.get_statistics()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should complete
        assert len(results) == 4


class TestPersistence:
    """Test data persistence"""
    
    @pytest.mark.asyncio
    async def test_history_persistence(self, feature_config, tmp_path):
        """Test command history is persisted"""
        with patch('pathlib.Path.home', return_value=tmp_path):
            # Create feature and record commands
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            await feature.record_command('git status', True)
            await feature.record_command('npm install', True)
            
            await feature.shutdown()
            
            # Create new feature instance
            feature2 = TerminalAssistantFeature(feature_config)
            await feature2.initialize()
            
            # History should be loaded
            stats = await feature2.get_statistics()
            assert stats['total_commands'] == 2
            
            await feature2.shutdown()
    
    @pytest.mark.asyncio
    async def test_preferences_persistence(self, feature_config, tmp_path):
        """Test user preferences are persisted"""
        with patch('pathlib.Path.home', return_value=tmp_path):
            # Create feature and learn patterns
            feature = TerminalAssistantFeature(feature_config)
            await feature.initialize()
            
            await feature.record_command('git status', True, {'project_type': 'python'})
            
            await feature.shutdown()
            
            # Create new feature instance
            feature2 = TerminalAssistantFeature(feature_config)
            await feature2.initialize()
            
            # Patterns should be loaded
            assert len(feature2.learning_engine.patterns) > 0
            
            await feature2.shutdown()


class TestMemoryManagement:
    """Test memory management and limits"""
    
    @pytest.mark.asyncio
    async def test_history_size_limit(self):
        """Test history respects size limit"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'history_size': 10}
        )
        
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        
        # Record more commands than limit
        for i in range(20):
            await feature.record_command(f'command_{i}', True)
        
        # History should be limited
        assert len(feature.command_predictor.history) <= 10
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_context_history_limit(self, terminal_feature):
        """Test context history is limited"""
        # Record many commands
        for i in range(100):
            await terminal_feature.record_command('git status', True, {'project_type': 'python'})
        
        # Context history should be limited to 50
        contexts = terminal_feature.learning_engine.command_contexts.get('git status', [])
        assert len(contexts) <= 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
