"""
Tests for Enhanced Error Handler

Tests comprehensive intelligent error handling including:
- Advanced error pattern recognition
- Context-aware fix suggestions
- Multiple fix alternatives with confidence scores
- Learning from successful fixes
"""

import pytest
import asyncio
from pathlib import Path
from xencode.features.error_handler_enhanced import (
    EnhancedErrorHandler,
    ErrorFix,
    ErrorPattern
)


@pytest.fixture
def error_handler():
    """Create error handler instance"""
    return EnhancedErrorHandler(enabled=True)


@pytest.fixture
def error_handler_with_history():
    """Create error handler with command history"""
    history = [
        {'command': 'python script.py', 'success': True},
        {'command': 'git status', 'success': True},
        {'command': 'npm install', 'success': True},
        {'command': 'docker ps', 'success': True},
    ]
    return EnhancedErrorHandler(enabled=True, command_history=history)


class TestCommandNotFound:
    """Test command not found error handling"""
    
    @pytest.mark.asyncio
    async def test_typo_correction(self, error_handler):
        """Test typo correction for common commands"""
        command = "pyhton script.py"
        error = "bash: pyhton: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        assert any('python' in fix.explanation.lower() for fix in fixes)
        assert any(fix.confidence > 0.7 for fix in fixes)
    
    @pytest.mark.asyncio
    async def test_installation_suggestion(self, error_handler):
        """Test installation suggestions for missing commands"""
        command = "docker ps"
        error = "bash: docker: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        install_fix = next((f for f in fixes if f.requires_install), None)
        assert install_fix is not None
        assert install_fix.install_command is not None
        assert 'docker' in install_fix.explanation.lower()
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching(self, error_handler):
        """Test fuzzy matching for similar commands"""
        command = "gti status"
        error = "bash: gti: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        assert any('git' in fix.explanation.lower() for fix in fixes)
    
    @pytest.mark.asyncio
    async def test_history_based_suggestion(self, error_handler_with_history):
        """Test suggestions based on command history"""
        command = "pythn script.py"
        error = "bash: pythn: command not found"
        
        fixes = await error_handler_with_history.suggest_fixes(command, error)
        
        # Should suggest similar commands from history
        assert len(fixes) > 0


class TestPermissionDenied:
    """Test permission denied error handling"""
    
    @pytest.mark.asyncio
    async def test_sudo_suggestion(self, error_handler):
        """Test sudo suggestion for permission errors"""
        command = "apt-get install python3"
        error = "E: Could not open lock file - open (13: Permission denied)"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        sudo_fix = next((f for f in fixes if f.requires_sudo), None)
        assert sudo_fix is not None
        assert 'sudo' in sudo_fix.fix_command
    
    @pytest.mark.asyncio
    async def test_file_permission_fix(self, error_handler):
        """Test file permission fixes"""
        command = "./script.sh"
        error = "bash: ./script.sh: Permission denied"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        chmod_fix = next((f for f in fixes if 'chmod' in (f.fix_command or '')), None)
        assert chmod_fix is not None
    
    @pytest.mark.asyncio
    async def test_docker_permission_fix(self, error_handler):
        """Test Docker-specific permission fixes"""
        command = "docker ps"
        error = "Got permission denied while trying to connect to the Docker daemon socket"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        docker_fix = next((f for f in fixes if 'docker' in f.explanation.lower() and 'group' in f.explanation.lower()), None)
        assert docker_fix is not None


class TestFileNotFound:
    """Test file not found error handling"""
    
    @pytest.mark.asyncio
    async def test_file_creation_suggestion(self, error_handler):
        """Test file creation suggestions"""
        command = "cat missing.txt"
        error = "cat: missing.txt: No such file or directory"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        # Should suggest creating file or checking path
        assert any('create' in fix.explanation.lower() or 'check' in fix.explanation.lower() 
                  for fix in fixes)
    
    @pytest.mark.asyncio
    async def test_directory_creation_suggestion(self, error_handler):
        """Test directory creation suggestions"""
        command = "cd /nonexistent/path"
        error = "bash: cd: /nonexistent/path: No such file or directory"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0


class TestSyntaxError:
    """Test syntax error handling"""
    
    @pytest.mark.asyncio
    async def test_quote_mismatch_detection(self, error_handler):
        """Test detection of quote mismatches"""
        command = 'echo "hello world'
        error = "bash: unexpected EOF while looking for matching `\"'"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        assert any('quote' in fix.explanation.lower() for fix in fixes)
    
    @pytest.mark.asyncio
    async def test_command_help_suggestion(self, error_handler):
        """Test command help suggestions"""
        command = "git --invalid-flag"
        error = "error: unknown option `invalid-flag'"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        help_fix = next((f for f in fixes if '--help' in (f.fix_command or '')), None)
        assert help_fix is not None


class TestPortInUse:
    """Test port already in use error handling"""
    
    @pytest.mark.asyncio
    async def test_port_kill_suggestion(self, error_handler):
        """Test suggestion to kill process using port"""
        command = "npm start"
        error = "Error: listen EADDRINUSE: address already in use :::3000"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        kill_fix = next((f for f in fixes if 'kill' in (f.fix_command or '').lower()), None)
        assert kill_fix is not None
        assert '3000' in kill_fix.fix_command
    
    @pytest.mark.asyncio
    async def test_alternative_port_suggestion(self, error_handler):
        """Test suggestion to use alternative port"""
        command = "python -m http.server 8000"
        error = "OSError: [Errno 98] Address already in use"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0


class TestModuleNotFound:
    """Test module/package not found error handling"""
    
    @pytest.mark.asyncio
    async def test_python_package_install(self, error_handler):
        """Test Python package installation suggestion"""
        command = "python script.py"
        error = "ModuleNotFoundError: No module named 'requests'"
        context = {'project_type': 'python'}
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        assert len(fixes) > 0
        pip_fix = next((f for f in fixes if 'pip install' in (f.fix_command or '')), None)
        assert pip_fix is not None
        assert 'requests' in pip_fix.fix_command
        assert pip_fix.requires_install
    
    @pytest.mark.asyncio
    async def test_node_package_install(self, error_handler):
        """Test Node.js package installation suggestion"""
        command = "node app.js"
        error = "Error: Cannot find module 'express'"
        context = {'project_type': 'node'}
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        assert len(fixes) > 0
        # Should have a fix with the specific module name
        express_fix = next((f for f in fixes if f.fix_command and 'express' in f.fix_command), None)
        assert express_fix is not None
        assert 'npm install express' in express_fix.fix_command
    
    @pytest.mark.asyncio
    async def test_alternative_install_commands(self, error_handler):
        """Test alternative installation commands are provided"""
        command = "python script.py"
        error = "ModuleNotFoundError: No module named 'numpy'"
        context = {'project_type': 'python'}
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        pip_fix = next((f for f in fixes if 'pip install' in (f.fix_command or '')), None)
        assert pip_fix is not None
        assert len(pip_fix.alternative_commands) > 0
        assert any('pip3' in cmd or 'poetry' in cmd for cmd in pip_fix.alternative_commands)


class TestGitErrors:
    """Test Git-related error handling"""
    
    @pytest.mark.asyncio
    async def test_not_git_repo(self, error_handler):
        """Test 'not a git repository' error"""
        command = "git status"
        error = "fatal: not a git repository (or any of the parent directories): .git"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        init_fix = next((f for f in fixes if 'git init' in (f.fix_command or '')), None)
        assert init_fix is not None


class TestNetworkErrors:
    """Test network-related error handling"""
    
    @pytest.mark.asyncio
    async def test_connection_refused(self, error_handler):
        """Test connection refused error"""
        command = "curl http://localhost:8000"
        error = "curl: (7) Failed to connect to localhost port 8000: Connection refused"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        # Should suggest checking connectivity
        assert any('ping' in (f.fix_command or '') for f in fixes)
    
    @pytest.mark.asyncio
    async def test_dns_resolution_error(self, error_handler):
        """Test DNS resolution error"""
        command = "ping example.invalid"
        error = "ping: example.invalid: Name or service not known"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        dns_fix = next((f for f in fixes if 'nslookup' in (f.fix_command or '')), None)
        assert dns_fix is not None


class TestDiskSpaceErrors:
    """Test disk space error handling"""
    
    @pytest.mark.asyncio
    async def test_no_space_left(self, error_handler):
        """Test 'no space left on device' error"""
        command = "cp large_file.iso /tmp/"
        error = "cp: error writing '/tmp/large_file.iso': No space left on device"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        df_fix = next((f for f in fixes if 'df' in (f.fix_command or '')), None)
        assert df_fix is not None


class TestDockerErrors:
    """Test Docker-related error handling"""
    
    @pytest.mark.asyncio
    async def test_docker_not_running(self, error_handler):
        """Test Docker daemon not running error"""
        command = "docker ps"
        error = "Cannot connect to the Docker daemon. Is the docker daemon running?"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        assert len(fixes) > 0
        start_fix = next((f for f in fixes if 'start docker' in (f.fix_command or '').lower()), None)
        assert start_fix is not None
        assert start_fix.requires_sudo


class TestContextAwareFixes:
    """Test context-aware fix suggestions"""
    
    @pytest.mark.asyncio
    async def test_python_project_context(self, error_handler):
        """Test fixes in Python project context"""
        command = "python app.py"
        error = "ModuleNotFoundError: No module named 'flask'"
        context = {
            'project_type': 'python',
            'directory': '/home/user/myproject'
        }
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        assert len(fixes) > 0
        # Should suggest installing from requirements.txt
        req_fix = next((f for f in fixes if 'requirements.txt' in (f.fix_command or '')), None)
        assert req_fix is not None
    
    @pytest.mark.asyncio
    async def test_node_project_context(self, error_handler):
        """Test fixes in Node.js project context"""
        command = "node server.js"
        error = "Error: Cannot find module 'express'"
        context = {
            'project_type': 'node',
            'directory': '/home/user/webapp'
        }
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        assert len(fixes) > 0
        # Should suggest npm install
        npm_fix = next((f for f in fixes if 'npm install' == f.fix_command), None)
        assert npm_fix is not None
    
    @pytest.mark.asyncio
    async def test_git_repo_context(self, error_handler):
        """Test fixes in Git repository context"""
        command = "git checkout feature-branch"
        error = "error: pathspec 'feature-branch' did not match any file(s) known to git"
        context = {
            'git_info': {'is_repo': True, 'branch': 'main'}
        }
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        assert len(fixes) > 0
        branch_fix = next((f for f in fixes if 'git branch' in (f.fix_command or '')), None)
        assert branch_fix is not None


class TestLearningCapabilities:
    """Test learning from successful fixes"""
    
    @pytest.mark.asyncio
    async def test_record_successful_fix(self, error_handler):
        """Test recording successful fixes"""
        command = "pyhton script.py"
        error = "bash: pyhton: command not found"
        fix_command = "python script.py"
        
        # Get initial count
        error_key = error[:100]
        initial_count = error_handler.successful_fixes[error_key].get(fix_command, 0)
        
        await error_handler.record_successful_fix(command, error, fix_command)
        
        # Check that fix was recorded
        assert error_key in error_handler.successful_fixes
        assert error_handler.successful_fixes[error_key][fix_command] == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_learned_fix_suggestion(self, error_handler):
        """Test that learned fixes are suggested"""
        command = "pyhton script.py"
        error = "bash: pyhton: command not found"
        fix_command = "python script.py"
        
        # Record successful fix multiple times
        for _ in range(3):
            await error_handler.record_successful_fix(command, error, fix_command)
        
        # Get suggestions
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Should include learned fix with high confidence
        learned_fix = next((f for f in fixes if f.fix_command == fix_command), None)
        assert learned_fix is not None
        assert learned_fix.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_confidence_boost_from_history(self, error_handler):
        """Test confidence boost for historically successful fixes"""
        command = "gti status"
        error = "bash: gti: command not found"
        fix_command = "git status"
        
        # Record successful fix
        await error_handler.record_successful_fix(command, error, fix_command)
        
        # Get suggestions
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Confidence should be boosted
        git_fix = next((f for f in fixes if f.fix_command == fix_command), None)
        assert git_fix is not None
        # Confidence should be higher than base
        assert git_fix.confidence > 0.7


class TestErrorStatistics:
    """Test error statistics and reporting"""
    
    @pytest.mark.asyncio
    async def test_error_statistics(self, error_handler):
        """Test getting error statistics"""
        # Generate some errors
        await error_handler.suggest_fixes("pyhton", "command not found")
        await error_handler.suggest_fixes("docker ps", "permission denied")
        
        stats = error_handler.get_error_statistics()
        
        assert 'total_errors_seen' in stats
        assert 'unique_errors' in stats
        assert 'patterns_registered' in stats
        assert stats['total_errors_seen'] >= 2
        assert stats['patterns_registered'] > 0


class TestMultipleFixAlternatives:
    """Test multiple fix alternatives with confidence scores"""
    
    @pytest.mark.asyncio
    async def test_multiple_fixes_returned(self, error_handler):
        """Test that multiple fix alternatives are returned"""
        command = "pyhton script.py"
        error = "bash: pyhton: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Should return multiple alternatives
        assert len(fixes) >= 2
    
    @pytest.mark.asyncio
    async def test_fixes_sorted_by_confidence(self, error_handler):
        """Test that fixes are sorted by confidence"""
        command = "docker ps"
        error = "bash: docker: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Check that fixes are sorted by confidence (descending)
        confidences = [fix.confidence for fix in fixes]
        assert confidences == sorted(confidences, reverse=True)
    
    @pytest.mark.asyncio
    async def test_alternative_commands_provided(self, error_handler):
        """Test that alternative commands are provided"""
        command = "python script.py"
        error = "ModuleNotFoundError: No module named 'requests'"
        context = {'project_type': 'python'}
        
        fixes = await error_handler.suggest_fixes(command, error, context)
        
        # Find fix with alternatives
        fix_with_alts = next((f for f in fixes if f.alternative_commands), None)
        assert fix_with_alts is not None
        assert len(fix_with_alts.alternative_commands) > 0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_command(self, error_handler):
        """Test handling of empty command"""
        fixes = await error_handler.suggest_fixes("", "some error")
        
        # Should not crash, may return empty or generic fixes
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_empty_error(self, error_handler):
        """Test handling of empty error message"""
        fixes = await error_handler.suggest_fixes("some command", "")
        
        # Should not crash
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_disabled_handler(self):
        """Test that disabled handler returns empty list"""
        handler = EnhancedErrorHandler(enabled=False)
        fixes = await handler.suggest_fixes("command", "error")
        
        assert fixes == []
    
    @pytest.mark.asyncio
    async def test_unknown_error_pattern(self, error_handler):
        """Test handling of unknown error patterns"""
        command = "some_command"
        error = "This is a completely unknown error pattern xyz123"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Should not crash, may return empty or generic fixes
        assert isinstance(fixes, list)
    
    @pytest.mark.asyncio
    async def test_deduplication(self, error_handler):
        """Test that duplicate fixes are removed"""
        command = "pyhton script.py"
        error = "bash: pyhton: command not found"
        
        fixes = await error_handler.suggest_fixes(command, error)
        
        # Check for duplicates
        fix_keys = [(f.fix_command, f.explanation) for f in fixes]
        assert len(fix_keys) == len(set(fix_keys))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
