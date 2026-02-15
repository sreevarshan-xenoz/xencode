"""
Tests for Terminal Assistant Context Analyzer

Tests the enhanced ContextAnalyzer class functionality including:
- Advanced project type detection
- Git repository analysis
- Environment variable analysis
- File system context
- Process context
- Network context
"""

import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from xencode.features.terminal_assistant import ContextAnalyzer


@pytest.fixture
def context_analyzer():
    """Create a ContextAnalyzer instance"""
    return ContextAnalyzer(enabled=True)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestProjectTypeDetection:
    """Test advanced project type detection"""
    
    @pytest.mark.asyncio
    async def test_detect_python_poetry(self, context_analyzer, temp_project_dir):
        """Test detection of Python Poetry projects"""
        (temp_project_dir / 'pyproject.toml').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'python-poetry'
    
    @pytest.mark.asyncio
    async def test_detect_python_setuptools(self, context_analyzer, temp_project_dir):
        """Test detection of Python setuptools projects"""
        (temp_project_dir / 'setup.py').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'python-setuptools'
    
    @pytest.mark.asyncio
    async def test_detect_python_requirements(self, context_analyzer, temp_project_dir):
        """Test detection of Python projects with requirements.txt"""
        (temp_project_dir / 'requirements.txt').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'python'
    
    @pytest.mark.asyncio
    async def test_detect_node_react(self, context_analyzer, temp_project_dir):
        """Test detection of React projects"""
        package_json = {
            'dependencies': {
                'react': '^18.0.0'
            }
        }
        (temp_project_dir / 'package.json').write_text(json.dumps(package_json))
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'node-react'
    
    @pytest.mark.asyncio
    async def test_detect_node_vue(self, context_analyzer, temp_project_dir):
        """Test detection of Vue projects"""
        package_json = {
            'dependencies': {
                'vue': '^3.0.0'
            }
        }
        (temp_project_dir / 'package.json').write_text(json.dumps(package_json))
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'node-vue'
    
    @pytest.mark.asyncio
    async def test_detect_rust(self, context_analyzer, temp_project_dir):
        """Test detection of Rust projects"""
        (temp_project_dir / 'Cargo.toml').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'rust'
    
    @pytest.mark.asyncio
    async def test_detect_go(self, context_analyzer, temp_project_dir):
        """Test detection of Go projects"""
        (temp_project_dir / 'go.mod').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'go'
    
    @pytest.mark.asyncio
    async def test_detect_java_maven(self, context_analyzer, temp_project_dir):
        """Test detection of Java Maven projects"""
        (temp_project_dir / 'pom.xml').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'java-maven'
    
    @pytest.mark.asyncio
    async def test_detect_docker(self, context_analyzer, temp_project_dir):
        """Test detection of Docker projects"""
        (temp_project_dir / 'Dockerfile').touch()
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type == 'docker'
    
    @pytest.mark.asyncio
    async def test_detect_no_project_type(self, context_analyzer, temp_project_dir):
        """Test detection when no project type is found"""
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        assert project_type is None


class TestGitAnalysis:
    """Test Git repository analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_non_git_repo(self, context_analyzer, temp_project_dir):
        """Test analysis of non-git directory"""
        git_info = await context_analyzer._analyze_git_repo(temp_project_dir)
        
        assert git_info['is_repo'] is False
        assert git_info['branch'] is None
        assert git_info['remotes'] == []
    
    @pytest.mark.asyncio
    async def test_analyze_git_repo_with_branch(self, context_analyzer, temp_project_dir):
        """Test analysis of git repo with branch"""
        # Create minimal git structure
        git_dir = temp_project_dir / '.git'
        git_dir.mkdir()
        
        # Create HEAD file with branch reference
        (git_dir / 'HEAD').write_text('ref: refs/heads/main\n')
        
        git_info = await context_analyzer._analyze_git_repo(temp_project_dir)
        
        assert git_info['is_repo'] is True
        assert git_info['branch'] == 'main'
    
    @pytest.mark.asyncio
    async def test_analyze_git_repo_with_remotes(self, context_analyzer, temp_project_dir):
        """Test analysis of git repo with remotes"""
        # Create minimal git structure
        git_dir = temp_project_dir / '.git'
        git_dir.mkdir()
        
        # Create config file with remotes
        config_content = '''[remote "origin"]
    url = https://github.com/user/repo.git
[remote "upstream"]
    url = https://github.com/upstream/repo.git
'''
        (git_dir / 'config').write_text(config_content)
        (git_dir / 'HEAD').write_text('ref: refs/heads/main\n')
        
        git_info = await context_analyzer._analyze_git_repo(temp_project_dir)
        
        assert git_info['is_repo'] is True
        assert 'origin' in git_info['remotes']
        assert 'upstream' in git_info['remotes']


class TestEnvironmentAnalysis:
    """Test environment variable analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_environment_paths(self, context_analyzer):
        """Test analysis of PATH variables"""
        with patch.dict(os.environ, {'PATH': '/usr/bin:/usr/local/bin', 'PYTHONPATH': '/opt/python'}):
            env_info = context_analyzer._analyze_environment()
            
            assert 'paths' in env_info
            assert 'PATH' in env_info['paths']
            assert 'PYTHONPATH' in env_info['paths']
    
    @pytest.mark.asyncio
    async def test_analyze_environment_development(self, context_analyzer):
        """Test analysis of development variables"""
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/path/to/venv', 'JAVA_HOME': '/usr/lib/jvm/java'}):
            env_info = context_analyzer._analyze_environment()
            
            assert 'development' in env_info
            assert 'VIRTUAL_ENV' in env_info['development']
            assert 'JAVA_HOME' in env_info['development']
    
    @pytest.mark.asyncio
    async def test_analyze_environment_cloud(self, context_analyzer):
        """Test analysis of cloud provider variables"""
        with patch.dict(os.environ, {'AWS_PROFILE': 'default', 'AWS_REGION': 'us-east-1'}):
            env_info = context_analyzer._analyze_environment()
            
            assert 'cloud' in env_info
            assert 'AWS_PROFILE' in env_info['cloud']
            assert 'AWS_REGION' in env_info['cloud']
    
    @pytest.mark.asyncio
    async def test_analyze_environment_custom(self, context_analyzer):
        """Test analysis of custom project variables"""
        with patch.dict(os.environ, {'PROJECT_NAME': 'myapp', 'API_KEY': 'secret'}):
            env_info = context_analyzer._analyze_environment()
            
            assert 'custom' in env_info
            assert 'PROJECT_NAME' in env_info['custom']
            assert 'API_KEY' in env_info['custom']


class TestFilesystemAnalysis:
    """Test file system context analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_filesystem_disk_usage(self, context_analyzer, temp_project_dir):
        """Test disk usage analysis"""
        fs_info = context_analyzer._analyze_filesystem(temp_project_dir)
        
        assert 'disk_usage' in fs_info
        assert 'total' in fs_info['disk_usage']
        assert 'used' in fs_info['disk_usage']
        assert 'free' in fs_info['disk_usage']
        assert 'percent_used' in fs_info['disk_usage']
    
    @pytest.mark.asyncio
    async def test_analyze_filesystem_permissions(self, context_analyzer, temp_project_dir):
        """Test permissions analysis"""
        fs_info = context_analyzer._analyze_filesystem(temp_project_dir)
        
        assert 'permissions' in fs_info
        assert 'readable' in fs_info['permissions']
        assert 'writable' in fs_info['permissions']
        assert 'executable' in fs_info['permissions']
    
    @pytest.mark.asyncio
    async def test_analyze_filesystem_file_counts(self, context_analyzer, temp_project_dir):
        """Test file count analysis"""
        # Create some test files
        (temp_project_dir / 'test1.py').touch()
        (temp_project_dir / 'test2.py').touch()
        (temp_project_dir / 'test.js').touch()
        
        fs_info = context_analyzer._analyze_filesystem(temp_project_dir)
        
        assert 'file_counts' in fs_info
        assert '.py' in fs_info['file_counts']
        assert fs_info['file_counts']['.py'] == 2


class TestProcessAnalysis:
    """Test process context analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_processes_without_psutil(self, context_analyzer):
        """Test process analysis when psutil is not available"""
        # Mock the import to raise ImportError
        import sys
        with patch.dict(sys.modules, {'psutil': None}):
            process_info = context_analyzer._analyze_processes()
            
            assert 'development_servers' in process_info
            assert 'databases' in process_info
            assert 'containers' in process_info
            assert process_info['development_servers'] == []
    
    @pytest.mark.asyncio
    async def test_analyze_processes_with_psutil(self, context_analyzer):
        """Test process analysis with psutil available"""
        try:
            import psutil
            # If psutil is available, test with real module
            process_info = context_analyzer._analyze_processes()
            
            assert 'development_servers' in process_info
            assert 'databases' in process_info
            assert 'containers' in process_info
            # Results may vary based on actual running processes
        except ImportError:
            # If psutil is not available, skip this test
            pytest.skip("psutil not available")


class TestNetworkAnalysis:
    """Test network context analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_network_without_psutil(self, context_analyzer):
        """Test network analysis when psutil is not available"""
        # Mock the import to raise ImportError
        import sys
        with patch.dict(sys.modules, {'psutil': None}):
            network_info = context_analyzer._analyze_network()
            
            assert 'localhost_ports' in network_info
            assert 'vpn_active' in network_info
            assert 'network_interfaces' in network_info
            assert network_info['localhost_ports'] == []
    
    @pytest.mark.asyncio
    async def test_analyze_network_with_listening_ports(self, context_analyzer):
        """Test network analysis with listening ports"""
        try:
            import psutil
            # If psutil is available, test with real module
            network_info = context_analyzer._analyze_network()
            
            assert 'localhost_ports' in network_info
            assert 'vpn_active' in network_info
            assert 'network_interfaces' in network_info
            # Results may vary based on actual network state
        except ImportError:
            # If psutil is not available, skip this test
            pytest.skip("psutil not available")


class TestComprehensiveAnalysis:
    """Test comprehensive context analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_comprehensive(self, context_analyzer, temp_project_dir):
        """Test comprehensive context analysis"""
        # Create a Python project
        (temp_project_dir / 'requirements.txt').touch()
        (temp_project_dir / 'main.py').touch()
        
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_project_dir)
            
            context_info = await context_analyzer.analyze()
            
            # Verify all expected keys are present
            assert 'directory' in context_info
            assert 'project_type' in context_info
            assert 'git_info' in context_info
            assert 'environment' in context_info
            assert 'filesystem' in context_info
            assert 'processes' in context_info
            assert 'network' in context_info
            assert 'os' in context_info
            assert 'files' in context_info
            
            # Verify project type detection
            assert context_info['project_type'] == 'python'
            
            # Verify git info structure
            assert 'is_repo' in context_info['git_info']
            
            # Verify environment structure
            assert 'paths' in context_info['environment']
            assert 'development' in context_info['environment']
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_analyze_disabled(self, temp_project_dir):
        """Test that disabled analyzer returns empty dict"""
        analyzer = ContextAnalyzer(enabled=False)
        
        context_info = await analyzer.analyze()
        
        assert context_info == {}
    
    @pytest.mark.asyncio
    async def test_analyze_with_git_repo(self, context_analyzer, temp_project_dir):
        """Test analysis of directory with git repo"""
        # Create git structure
        git_dir = temp_project_dir / '.git'
        git_dir.mkdir()
        (git_dir / 'HEAD').write_text('ref: refs/heads/main\n')
        
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_project_dir)
            
            context_info = await context_analyzer.analyze()
            
            assert context_info['git_info']['is_repo'] is True
            assert context_info['git_info']['branch'] == 'main'
            
        finally:
            os.chdir(original_cwd)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_analyze_with_invalid_package_json(self, context_analyzer, temp_project_dir):
        """Test handling of invalid package.json"""
        (temp_project_dir / 'package.json').write_text('invalid json{')
        
        project_type = context_analyzer._detect_project_type(temp_project_dir)
        # Should fall back to 'node' when JSON parsing fails
        assert project_type == 'node'
    
    @pytest.mark.asyncio
    async def test_analyze_with_permission_errors(self, context_analyzer, temp_project_dir):
        """Test handling of permission errors"""
        # This test verifies graceful handling of permission errors
        # The actual implementation catches exceptions
        fs_info = context_analyzer._analyze_filesystem(temp_project_dir)
        
        # Should return structure even if some operations fail
        assert 'disk_usage' in fs_info
        assert 'permissions' in fs_info
    
    @pytest.mark.asyncio
    async def test_get_relevant_files_with_many_files(self, context_analyzer, temp_project_dir):
        """Test that file listing is limited"""
        # Create 30 files
        for i in range(30):
            (temp_project_dir / f'file{i}.txt').touch()
        
        files = context_analyzer._get_relevant_files(temp_project_dir)
        
        # Should be limited to 20 files
        assert len(files) <= 20
