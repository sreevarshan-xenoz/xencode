"""
Integration Tests for Terminal Assistant Feature

Tests end-to-end workflows and multi-component interactions:
- Complete user workflows (suggest → explain → execute → fix → learn)
- CLI command integration
- TUI component integration
- Multi-component interactions
- Real-world usage scenarios
- Performance under load
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

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
def temp_workspace(tmp_path):
    """Create a temporary workspace directory"""
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    return workspace


@pytest_asyncio.fixture
async def terminal_feature(tmp_path):
    """Create a fully initialized Terminal Assistant feature"""
    config = FeatureConfig(
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
    
    with patch('pathlib.Path.home', return_value=tmp_path):
        feature = TerminalAssistantFeature(config)
        await feature.initialize()
        yield feature
        await feature.shutdown()


class TestCompleteUserWorkflows:
    """Test complete end-to-end user workflows"""
    
    @pytest.mark.asyncio
    async def test_suggest_explain_execute_workflow(self, terminal_feature, temp_workspace):
        """Test: User gets suggestion → explains it → executes it → records success"""
        # First, record some git commands to build history
        await terminal_feature.record_command('git status', success=True)
        await terminal_feature.record_command('git add .', success=True)
        
        # Step 1: Get command suggestions
        suggestions = await terminal_feature.suggest_commands(
            context='python project',
            partial='git'
        )
        
        assert len(suggestions) > 0
        suggested_command = suggestions[0]['command']
        
        # Step 2: Explain the suggested command
        explanation = await terminal_feature.explain_command(suggested_command)
        
        assert explanation['command'] == suggested_command
        assert 'description' in explanation
        assert 'arguments' in explanation
        
        # Step 3: Record successful execution
        await terminal_feature.record_command(
            suggested_command,
            success=True,
            context={'project_type': 'python', 'directory': str(temp_workspace)}
        )
        
        # Step 4: Verify command was learned
        stats = await terminal_feature.get_statistics(suggested_command)
        assert stats['frequency'] >= 1
        
        # Step 5: Get suggestions again - should rank this command higher
        new_suggestions = await terminal_feature.suggest_commands(
            context='python project',
            partial='git'
        )
        
        # The executed command should appear in suggestions
        command_found = any(s['command'] == suggested_command for s in new_suggestions)
        assert command_found
    
    @pytest.mark.asyncio
    async def test_error_fix_learn_workflow(self, terminal_feature):
        """Test: User encounters error → gets fix → applies fix → records success"""
        # Step 1: User tries a command that fails
        failed_command = 'pyhton script.py'
        error_message = 'bash: pyhton: command not found'
        
        await terminal_feature.record_command(failed_command, success=False)
        
        # Step 2: Get fix suggestions
        fixes = await terminal_feature.fix_error(failed_command, error_message)
        
        assert len(fixes) > 0
        best_fix = fixes[0]
        assert 'fix' in best_fix
        assert 'explanation' in best_fix
        assert 'confidence' in best_fix
        
        # Step 3: Apply the fix and record success
        fixed_command = best_fix['fix']
        await terminal_feature.record_command(fixed_command, success=True)
        
        # Step 4: Record that this fix worked
        await terminal_feature.record_successful_fix(
            failed_command,
            error_message,
            fixed_command
        )
        
        # Step 5: Encounter same error again - should suggest the successful fix
        new_fixes = await terminal_feature.fix_error(failed_command, error_message)
        
        # The successful fix should be suggested with high confidence
        assert any(f['fix'] == fixed_command for f in new_fixes)
    
    @pytest.mark.asyncio
    async def test_learning_progression_workflow(self, terminal_feature):
        """Test: User learns a new tool through repeated use"""
        # Simulate learning git over time
        git_commands = [
            'git status',
            'git add .',
            'git commit -m "test"',
            'git push',
            'git pull',
            'git branch',
            'git checkout -b feature',
            'git merge main'
        ]
        
        # Step 1: Execute commands multiple times (simulating learning)
        for _ in range(3):
            for cmd in git_commands:
                await terminal_feature.record_command(
                    cmd,
                    success=True,
                    context={'project_type': 'git'}
                )
        
        # Step 2: Check learning progress
        stats = await terminal_feature.learning_engine.get_learning_stats()
        
        assert stats['total_commands_learned'] > 0
        assert stats['total_executions'] >= len(git_commands) * 3
        
        # Step 3: Verify skill level increased
        if 'git' in stats['skill_levels']:
            assert stats['skill_levels']['git'] > 0.5
        
        # Step 4: Get suggestions - should be personalized based on learning
        suggestions = await terminal_feature.suggest_commands(
            context='git project',
            partial='git'
        )
        
        # Should suggest git commands with high scores
        assert len(suggestions) > 0
        assert all('git' in s['command'] for s in suggestions)

    @pytest.mark.asyncio
    async def test_context_aware_workflow(self, terminal_feature, temp_workspace):
        """Test: Commands are suggested based on project context"""
        # Step 1: Create Python project context
        python_project = temp_workspace / 'python_project'
        python_project.mkdir()
        (python_project / 'requirements.txt').touch()
        (python_project / 'main.py').touch()
        
        # Step 2: Record Python-specific commands
        python_commands = [
            'python -m venv venv',
            'pip install -r requirements.txt',
            'python main.py',
            'pytest tests/'
        ]
        
        for cmd in python_commands:
            await terminal_feature.record_command(
                cmd,
                success=True,
                context={'project_type': 'python', 'directory': str(python_project)}
            )
        
        # Step 3: Create Node.js project context
        node_project = temp_workspace / 'node_project'
        node_project.mkdir()
        (node_project / 'package.json').write_text('{"name": "test"}')
        
        # Step 4: Record Node-specific commands
        node_commands = [
            'npm install',
            'npm run dev',
            'npm test'
        ]
        
        for cmd in node_commands:
            await terminal_feature.record_command(
                cmd,
                success=True,
                context={'project_type': 'node', 'directory': str(node_project)}
            )
        
        # Step 5: Get suggestions for Python context
        python_suggestions = await terminal_feature.suggest_commands(
            context=f'python project in {python_project}'
        )
        
        # Should suggest Python commands
        python_cmd_count = sum(1 for s in python_suggestions if 'python' in s['command'] or 'pip' in s['command'])
        assert python_cmd_count > 0
        
        # Step 6: Get suggestions for Node context
        node_suggestions = await terminal_feature.suggest_commands(
            context=f'node project in {node_project}'
        )
        
        # Should suggest Node commands
        node_cmd_count = sum(1 for s in node_suggestions if 'npm' in s['command'])
        assert node_cmd_count > 0
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_workflow(self, terminal_feature):
        """Test: System recognizes and suggests command patterns"""
        # Step 1: Execute a common workflow pattern multiple times
        workflow_pattern = [
            'git add .',
            'git commit -m "update"',
            'git push origin main'
        ]
        
        # Execute pattern 5 times
        for _ in range(5):
            for cmd in workflow_pattern:
                await terminal_feature.record_command(cmd, success=True)
                await asyncio.sleep(0.01)  # Small delay to maintain sequence
        
        # Step 2: Analyze patterns
        patterns = await terminal_feature.analyze_patterns()
        
        assert 'sequence_patterns' in patterns
        assert 'command_patterns' in patterns
        
        # Step 3: Execute first command in pattern
        await terminal_feature.record_command('git add .', success=True)
        
        # Step 4: Get suggestions - should suggest next command in pattern
        suggestions = await terminal_feature.suggest_commands()
        
        # Should suggest 'git commit' as it follows 'git add .'
        commit_suggested = any('git commit' in s['command'] for s in suggestions)
        assert commit_suggested


class TestMultiComponentIntegration:
    """Test integration between multiple components"""
    
    @pytest.mark.asyncio
    async def test_predictor_context_learning_integration(self, terminal_feature):
        """Test CommandPredictor + ContextAnalyzer + LearningEngine integration"""
        # Record commands with context
        commands_with_context = [
            ('git status', {'project_type': 'python', 'directory': '/project'}),
            ('python test.py', {'project_type': 'python', 'directory': '/project'}),
            ('npm install', {'project_type': 'node', 'directory': '/webapp'}),
        ]
        
        for cmd, ctx in commands_with_context:
            await terminal_feature.record_command(cmd, success=True, context=ctx)
        
        # Get suggestions with context
        suggestions = await terminal_feature.suggest_commands(
            context='python project'
        )
        
        # Verify all components contributed
        assert len(suggestions) > 0
        
        # Check that suggestions have enhanced data from learning engine
        for suggestion in suggestions:
            assert 'command' in suggestion
            assert 'score' in suggestion
            # Learning engine should add explanation and difficulty
            if 'explanation' in suggestion:
                assert isinstance(suggestion['explanation'], str)
    
    @pytest.mark.asyncio
    async def test_error_handler_learning_integration(self, terminal_feature):
        """Test ErrorHandler + LearningEngine integration"""
        # Record a common error and its fix
        error_scenarios = [
            ('pyhton script.py', 'command not found', 'python script.py'),
            ('npm satrt', 'command not found', 'npm start'),
            ('git psuh', 'command not found', 'git push'),
        ]
        
        for original, error, fix in error_scenarios:
            # Record the error
            await terminal_feature.record_command(original, success=False)
            
            # Get fix suggestions
            fixes = await terminal_feature.fix_error(original, error)
            
            # Record successful fix
            await terminal_feature.record_successful_fix(original, error, fix)
            await terminal_feature.record_command(fix, success=True)
        
        # Test that learning improved error fixing
        new_error = 'pyhton test.py'
        new_error_msg = 'command not found'
        
        fixes = await terminal_feature.fix_error(new_error, new_error_msg)
        
        # Should suggest 'python test.py' with high confidence
        assert len(fixes) > 0
        best_fix = fixes[0]
        assert 'python' in best_fix['fix']
        assert best_fix['confidence'] > 0.7
    
    @pytest.mark.asyncio
    async def test_history_statistics_patterns_integration(self, terminal_feature):
        """Test CommandPredictor history + statistics + pattern analysis integration"""
        # Build up a rich command history
        commands = [
            'git status', 'git status', 'git status',  # Frequent
            'git add .', 'git commit -m "test"', 'git push',  # Sequence
            'npm install', 'npm test',  # Different tool
            'docker ps', 'docker logs container',  # Another tool
        ]
        
        for cmd in commands:
            await terminal_feature.record_command(cmd, success=True)
        
        # Test statistics
        overall_stats = await terminal_feature.get_statistics()
        assert overall_stats['total_commands'] == len(commands)
        assert overall_stats['unique_commands'] > 0
        
        # Test specific command statistics
        git_status_stats = await terminal_feature.get_statistics('git status')
        assert git_status_stats['frequency'] == 3
        assert git_status_stats['success_rate'] == 1.0
        
        # Test pattern analysis
        patterns = await terminal_feature.analyze_patterns()
        assert 'git' in patterns['command_patterns']
        assert len(patterns['sequence_patterns']) > 0
        
        # Test history search
        git_history = await terminal_feature.search_history('git')
        assert len(git_history) == 6  # All git commands


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""
    
    @pytest.mark.asyncio
    async def test_new_developer_onboarding(self, terminal_feature, temp_workspace):
        """Scenario: New developer learning a project"""
        # Day 1: Basic exploration
        day1_commands = [
            'ls -la',
            'cat README.md',
            'git status',
            'git log',
        ]
        
        for cmd in day1_commands:
            await terminal_feature.record_command(cmd, success=True)
        
        # Day 2: Setup and first changes
        day2_commands = [
            'python -m venv venv',
            'source venv/bin/activate',
            'pip install -r requirements.txt',
            'python main.py',
            'git checkout -b feature/my-first-change',
        ]
        
        for cmd in day2_commands:
            await terminal_feature.record_command(cmd, success=True)
        
        # Day 3: Making mistakes and learning
        mistakes = [
            ('pyhton test.py', 'command not found', 'python test.py'),
            ('git comit', 'command not found', 'git commit'),
        ]
        
        for original, error, fix in mistakes:
            await terminal_feature.record_command(original, success=False)
            fixes = await terminal_feature.fix_error(original, error)
            await terminal_feature.record_successful_fix(original, error, fix)
            await terminal_feature.record_command(fix, success=True)
        
        # Check learning progress
        stats = await terminal_feature.learning_engine.get_learning_stats()
        assert stats['total_commands_learned'] > 5
        
        # Get suggestions - should be helpful for a beginner
        suggestions = await terminal_feature.suggest_commands(context='python project')
        assert len(suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_experienced_developer_workflow(self, terminal_feature):
        """Scenario: Experienced developer with established patterns"""
        # Simulate experienced developer's typical workflow
        typical_workflow = [
            # Morning routine
            'git pull origin main',
            'git checkout -b feature/new-feature',
            
            # Development cycle (repeated)
            'python -m pytest tests/',
            'git add .',
            'git commit -m "Add feature"',
            'python -m pytest tests/',
            
            # Code review prep
            'git push origin feature/new-feature',
            'git log --oneline -10',
        ]
        
        # Execute workflow multiple times
        for _ in range(3):
            for cmd in typical_workflow:
                await terminal_feature.record_command(cmd, success=True)
        
        # Analyze patterns
        patterns = await terminal_feature.analyze_patterns()
        
        # Should detect strong sequence patterns
        assert len(patterns['sequence_patterns']) > 0
        
        # Get suggestions mid-workflow
        await terminal_feature.record_command('git add .', success=True)
        suggestions = await terminal_feature.suggest_commands()
        
        # Should suggest 'git commit' as next step
        commit_suggested = any('git commit' in s['command'] for s in suggestions)
        assert commit_suggested
    
    @pytest.mark.asyncio
    async def test_debugging_session(self, terminal_feature):
        """Scenario: Developer debugging an issue"""
        # Debugging workflow
        debug_commands = [
            'python -m pytest tests/test_feature.py -v',  # Run specific test
            'python -m pytest tests/test_feature.py -v -s',  # With output
            'python -m pdb main.py',  # Start debugger
            'grep -r "error_message" .',  # Search for error
            'git log --grep="feature"',  # Check git history
            'git diff HEAD~1',  # Check recent changes
            'python -m pytest tests/test_feature.py::test_specific -v',  # Narrow down
        ]
        
        for cmd in debug_commands:
            await terminal_feature.record_command(cmd, success=True)
        
        # Get suggestions for debugging context
        suggestions = await terminal_feature.suggest_commands(
            context='debugging python test failure'
        )
        
        # Should suggest debugging-related commands
        assert len(suggestions) > 0
        debug_related = any(
            'pytest' in s['command'] or 'pdb' in s['command'] or 'grep' in s['command']
            for s in suggestions
        )
        assert debug_related
    
    @pytest.mark.asyncio
    async def test_deployment_workflow(self, terminal_feature):
        """Scenario: Developer deploying application"""
        # Deployment workflow
        deploy_commands = [
            'git checkout main',
            'git pull origin main',
            'python -m pytest',  # Run all tests
            'docker build -t myapp:latest .',
            'docker tag myapp:latest myapp:v1.0.0',
            'docker push myapp:v1.0.0',
            'kubectl apply -f deployment.yaml',
            'kubectl rollout status deployment/myapp',
        ]
        
        for cmd in deploy_commands:
            await terminal_feature.record_command(cmd, success=True)
        
        # Analyze deployment patterns
        patterns = await terminal_feature.analyze_patterns()
        
        # Should recognize docker and kubectl patterns
        assert 'docker' in patterns['command_patterns'] or 'kubectl' in patterns['command_patterns']
        
        # Get suggestions for deployment context
        suggestions = await terminal_feature.suggest_commands(
            context='deploying application'
        )
        
        assert len(suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_multi_project_context_switching(self, terminal_feature, temp_workspace):
        """Scenario: Developer working on multiple projects"""
        # Project A: Python backend
        project_a = temp_workspace / 'backend'
        project_a.mkdir()
        (project_a / 'requirements.txt').touch()
        
        backend_commands = [
            'python -m venv venv',
            'pip install -r requirements.txt',
            'python manage.py runserver',
            'python -m pytest',
        ]
        
        for cmd in backend_commands:
            await terminal_feature.record_command(
                cmd,
                success=True,
                context={'project_type': 'python', 'directory': str(project_a)}
            )
        
        # Project B: React frontend
        project_b = temp_workspace / 'frontend'
        project_b.mkdir()
        (project_b / 'package.json').write_text('{"name": "frontend"}')
        
        frontend_commands = [
            'npm install',
            'npm run dev',
            'npm test',
            'npm run build',
        ]
        
        for cmd in frontend_commands:
            await terminal_feature.record_command(
                cmd,
                success=True,
                context={'project_type': 'node', 'directory': str(project_b)}
            )
        
        # Switch to backend - should suggest Python commands
        backend_suggestions = await terminal_feature.suggest_commands(
            context=f'python project in {project_a}'
        )
        
        python_count = sum(1 for s in backend_suggestions if 'python' in s['command'] or 'pip' in s['command'])
        assert python_count > 0
        
        # Switch to frontend - should suggest Node commands
        frontend_suggestions = await terminal_feature.suggest_commands(
            context=f'node project in {project_b}'
        )
        
        npm_count = sum(1 for s in frontend_suggestions if 'npm' in s['command'])
        assert npm_count > 0


class TestPerformanceUnderLoad:
    """Test performance with high load"""
    
    @pytest.mark.asyncio
    async def test_large_command_history(self, terminal_feature):
        """Test performance with large command history"""
        # Record 500 commands
        commands = [f'command_{i}' for i in range(500)]
        
        start_time = asyncio.get_event_loop().time()
        
        for cmd in commands:
            await terminal_feature.record_command(cmd, success=True)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        
        # Verify history is limited
        assert len(terminal_feature.command_predictor.history) <= terminal_feature.command_predictor.history_size
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, terminal_feature):
        """Test concurrent operations don't cause issues"""
        # Create multiple concurrent operations
        tasks = []
        
        # 10 concurrent suggestion requests
        for i in range(10):
            tasks.append(terminal_feature.suggest_commands(partial=f'cmd{i}'))
        
        # 10 concurrent command recordings
        for i in range(10):
            tasks.append(terminal_feature.record_command(f'command_{i}', success=True))
        
        # 5 concurrent explanations
        for i in range(5):
            tasks.append(terminal_feature.explain_command(f'command_{i}'))
        
        # Execute all concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 3.0
        
        # No exceptions should occur
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

    @pytest.mark.asyncio
    async def test_rapid_command_execution(self, terminal_feature):
        """Test rapid command execution (simulating fast typing)"""
        commands = ['git status', 'git add .', 'git commit', 'git push'] * 10
        
        start_time = asyncio.get_event_loop().time()
        
        for cmd in commands:
            await terminal_feature.record_command(cmd, success=True)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should handle 40 commands quickly
        assert elapsed < 2.0
        
        # Verify all commands were recorded
        stats = await terminal_feature.get_statistics()
        assert stats['total_commands'] == len(commands)
    
    @pytest.mark.asyncio
    async def test_complex_pattern_analysis(self, terminal_feature):
        """Test pattern analysis with complex command history"""
        # Create complex patterns
        patterns = [
            ['git status', 'git add .', 'git commit -m "msg"', 'git push'],
            ['npm install', 'npm test', 'npm run build'],
            ['docker build', 'docker tag', 'docker push'],
            ['python -m pytest', 'python main.py'],
        ]
        
        # Execute each pattern multiple times
        for _ in range(10):
            for pattern in patterns:
                for cmd in pattern:
                    await terminal_feature.record_command(cmd, success=True)
        
        # Analyze patterns
        start_time = asyncio.get_event_loop().time()
        analysis = await terminal_feature.analyze_patterns()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should complete analysis quickly
        assert elapsed < 1.0
        
        # Should detect multiple patterns
        assert len(analysis['command_patterns']) >= 3
        assert len(analysis['sequence_patterns']) > 0


class TestDataPersistence:
    """Test data persistence across sessions"""
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, tmp_path):
        """Test data persists across feature shutdown and restart"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'history_size': 100}
        )
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            # Session 1: Record commands
            feature1 = TerminalAssistantFeature(config)
            await feature1.initialize()
            
            commands = ['git status', 'npm install', 'python test.py']
            for cmd in commands:
                await feature1.record_command(cmd, success=True)
            
            stats1 = await feature1.get_statistics()
            total_commands_1 = stats1['total_commands']
            
            await feature1.shutdown()
            
            # Session 2: Load and verify
            feature2 = TerminalAssistantFeature(config)
            await feature2.initialize()
            
            stats2 = await feature2.get_statistics()
            assert stats2['total_commands'] == total_commands_1
            
            # Add more commands
            await feature2.record_command('docker ps', success=True)
            
            stats3 = await feature2.get_statistics()
            assert stats3['total_commands'] == total_commands_1 + 1
            
            await feature2.shutdown()
    
    @pytest.mark.asyncio
    async def test_learning_persistence(self, tmp_path):
        """Test learning data persists across sessions"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'learning_enabled': True}
        )
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            # Session 1: Build learning data
            feature1 = TerminalAssistantFeature(config)
            await feature1.initialize()
            
            for _ in range(10):
                await feature1.record_command('git status', success=True)
            
            stats1 = await feature1.learning_engine.get_learning_stats()
            
            await feature1.shutdown()
            
            # Session 2: Verify learning data persisted
            feature2 = TerminalAssistantFeature(config)
            await feature2.initialize()
            
            stats2 = await feature2.learning_engine.get_learning_stats()
            
            # Learning data should be preserved
            assert stats2['total_commands_learned'] >= stats1['total_commands_learned']
            assert stats2['total_executions'] >= stats1['total_executions']
            
            await feature2.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_fix_persistence(self, tmp_path):
        """Test error fixes persist across sessions"""
        config = FeatureConfig(
            name='terminal_assistant',
            enabled=True,
            config={'error_fix_enabled': True}
        )
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            # Session 1: Record successful fix
            feature1 = TerminalAssistantFeature(config)
            await feature1.initialize()
            
            await feature1.record_successful_fix(
                'pyhton script.py',
                'command not found',
                'python script.py'
            )
            
            await feature1.shutdown()
            
            # Session 2: Verify fix is remembered
            feature2 = TerminalAssistantFeature(config)
            await feature2.initialize()
            
            fixes = await feature2.fix_error('pyhton test.py', 'command not found')
            
            # Should suggest the learned fix
            assert len(fixes) > 0
            python_fix = any('python' in f['fix'] for f in fixes)
            assert python_fix
            
            await feature2.shutdown()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_empty_history_suggestions(self, terminal_feature):
        """Test suggestions work with empty history"""
        suggestions = await terminal_feature.suggest_commands()
        
        # Should return some default suggestions
        assert isinstance(suggestions, list)
    
    @pytest.mark.asyncio
    async def test_malformed_command_handling(self, terminal_feature):
        """Test handling of malformed commands"""
        malformed_commands = [
            'cmd\x00with\x00nulls',
        ]
        
        for cmd in malformed_commands:
            # Should not raise exceptions
            await terminal_feature.record_command(cmd, success=False)
            explanation = await terminal_feature.explain_command(cmd)
            assert 'command' in explanation
        
        # Test empty/whitespace commands separately (they may be filtered)
        empty_commands = ['', '   ', '\n\n']
        for cmd in empty_commands:
            # These should be handled gracefully
            explanation = await terminal_feature.explain_command(cmd)
            assert 'command' in explanation
    
    @pytest.mark.asyncio
    async def test_very_long_command(self, terminal_feature):
        """Test handling of very long commands"""
        long_command = 'echo ' + 'a' * 10000
        
        await terminal_feature.record_command(long_command, success=True)
        explanation = await terminal_feature.explain_command(long_command)
        
        assert explanation['command'] == long_command
    
    @pytest.mark.asyncio
    async def test_special_characters_in_commands(self, terminal_feature):
        """Test commands with special characters"""
        special_commands = [
            'grep "pattern" file.txt',
            "echo 'hello world'",
            'find . -name "*.py"',
            'sed -i "s/old/new/g" file.txt',
        ]
        
        for cmd in special_commands:
            await terminal_feature.record_command(cmd, success=True)
            explanation = await terminal_feature.explain_command(cmd)
            assert explanation['command'] == cmd
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, terminal_feature):
        """Test concurrent reads and writes don't cause corruption"""
        # Mix of read and write operations
        tasks = []
        
        for i in range(20):
            if i % 2 == 0:
                tasks.append(terminal_feature.record_command(f'cmd_{i}', success=True))
            else:
                tasks.append(terminal_feature.suggest_commands())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # No exceptions should occur
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
        
        # Data should be consistent
        stats = await terminal_feature.get_statistics()
        assert stats['total_commands'] >= 10


class TestCLIIntegration:
    """Test CLI command integration"""
    
    @pytest.mark.asyncio
    async def test_cli_workflow_simulation(self, terminal_feature):
        """Simulate a complete CLI workflow"""
        # First, record some commands to build history
        await terminal_feature.record_command('git status', success=True)
        await terminal_feature.record_command('git commit', success=True)
        
        # Simulate: xencode terminal suggest
        suggestions = await terminal_feature.suggest_commands(partial='git')
        assert len(suggestions) > 0
        
        # Simulate: xencode terminal explain <command>
        if suggestions:
            explanation = await terminal_feature.explain_command(suggestions[0]['command'])
            assert 'description' in explanation
        
        # Simulate: user executes command
        await terminal_feature.record_command('git status', success=True)
        
        # Simulate: xencode terminal history git
        history = await terminal_feature.search_history('git')
        assert len(history) > 0
        
        # Simulate: xencode terminal statistics
        stats = await terminal_feature.get_statistics()
        assert stats['total_commands'] > 0
        
        # Simulate: xencode terminal patterns
        patterns = await terminal_feature.analyze_patterns()
        assert 'command_patterns' in patterns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
