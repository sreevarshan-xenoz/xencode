"""
Unit tests for Code Review CLI commands
"""

import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch
from xencode.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner"""
    return CliRunner()


@pytest.fixture
def mock_feature():
    """Create a mock CodeReviewFeature"""
    feature = MagicMock()
    feature._initialize = AsyncMock()
    feature._shutdown = AsyncMock()
    feature.analyze_pr = AsyncMock(return_value={
        'issues': [
            {'severity': 'high', 'message': 'Test issue 1'},
            {'severity': 'low', 'message': 'Test issue 2'}
        ]
    })
    feature.analyze_file = AsyncMock(return_value={
        'issues': [
            {'severity': 'critical', 'message': 'Critical issue'}
        ]
    })
    feature.analyze_directory = AsyncMock(return_value={
        'files': [
            {'path': 'test.py', 'language': 'python'}
        ],
        'issues': [
            {'severity': 'medium', 'message': 'Medium issue'}
        ]
    })
    feature.generate_formatted_report = MagicMock(return_value="Test Report")
    return feature


class TestReviewPRCommand:
    """Tests for 'xencode review pr' command"""
    
    def test_pr_help(self, runner):
        """Test that PR command help works"""
        result = runner.invoke(cli, ['review', 'pr', '--help'])
        assert result.exit_code == 0
        assert 'Review a pull request' in result.output
        assert '--platform' in result.output
        assert '--format' in result.output
        assert '--severity' in result.output
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_pr_basic(self, mock_class, runner, mock_feature):
        """Test basic PR review"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'pr', 
            'https://github.com/owner/repo/pull/123'
        ])
        
        assert result.exit_code == 0
        assert 'Analyzing pull request' in result.output
        mock_feature.analyze_pr.assert_called_once()
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_pr_with_platform(self, mock_class, runner, mock_feature):
        """Test PR review with platform specified"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'pr',
            'https://gitlab.com/owner/repo/-/merge_requests/45',
            '--platform', 'gitlab'
        ])
        
        assert result.exit_code == 0
        mock_feature.analyze_pr.assert_called_once()
        call_args = mock_feature.analyze_pr.call_args
        assert call_args[0][1] == 'gitlab'
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_pr_with_severity_filter(self, mock_class, runner, mock_feature):
        """Test PR review with severity filter"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'pr',
            'https://github.com/owner/repo/pull/123',
            '--severity', 'high'
        ])
        
        assert result.exit_code == 0
        # Should filter out low severity issues
        mock_feature.generate_formatted_report.assert_called_once()
    
    @patch('xencode.cli.Path')
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_pr_with_output_file(self, mock_class, mock_path, runner, mock_feature):
        """Test PR review with output file"""
        mock_class.return_value = mock_feature
        mock_file = MagicMock()
        mock_path.return_value = mock_file
        
        result = runner.invoke(cli, [
            'review', 'pr',
            'https://github.com/owner/repo/pull/123',
            '--output', 'report.md'
        ])
        
        assert result.exit_code == 0
        mock_file.write_text.assert_called_once()
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_pr_with_format(self, mock_class, runner, mock_feature):
        """Test PR review with different formats"""
        mock_class.return_value = mock_feature
        
        for fmt in ['text', 'markdown', 'json', 'html']:
            result = runner.invoke(cli, [
                'review', 'pr',
                'https://github.com/owner/repo/pull/123',
                '--format', fmt
            ])
            
            assert result.exit_code == 0
            call_args = mock_feature.generate_formatted_report.call_args
            assert call_args[0][1] == fmt


class TestReviewFileCommand:
    """Tests for 'xencode review file' command"""
    
    def test_file_help(self, runner):
        """Test that file command help works"""
        result = runner.invoke(cli, ['review', 'file', '--help'])
        assert result.exit_code == 0
        assert 'Review a specific file' in result.output
        assert '--language' in result.output
        assert '--format' in result.output
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_file_basic(self, mock_class, runner, mock_feature, tmp_path):
        """Test basic file review"""
        mock_class.return_value = mock_feature
        
        # Create a temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        
        result = runner.invoke(cli, [
            'review', 'file',
            str(test_file)
        ])
        
        assert result.exit_code == 0
        assert 'Analyzing file' in result.output
        mock_feature.analyze_file.assert_called_once()
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_file_with_language(self, mock_class, runner, mock_feature, tmp_path):
        """Test file review with language specified"""
        mock_class.return_value = mock_feature
        
        test_file = tmp_path / "test.js"
        test_file.write_text("console.log('hello');")
        
        result = runner.invoke(cli, [
            'review', 'file',
            str(test_file),
            '--language', 'javascript'
        ])
        
        assert result.exit_code == 0
        call_args = mock_feature.analyze_file.call_args
        assert call_args[0][1] == 'javascript'
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_file_with_severity_filter(self, mock_class, runner, mock_feature, tmp_path):
        """Test file review with severity filter"""
        mock_class.return_value = mock_feature
        
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        
        result = runner.invoke(cli, [
            'review', 'file',
            str(test_file),
            '--severity', 'critical'
        ])
        
        assert result.exit_code == 0
        mock_feature.generate_formatted_report.assert_called_once()


class TestReviewDirectoryCommand:
    """Tests for 'xencode review directory' command"""
    
    def test_directory_help(self, runner):
        """Test that directory command help works"""
        result = runner.invoke(cli, ['review', 'directory', '--help'])
        assert result.exit_code == 0
        assert 'Review an entire directory' in result.output
        assert '--patterns' in result.output
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_directory_basic(self, mock_class, runner, mock_feature, tmp_path):
        """Test basic directory review"""
        mock_class.return_value = mock_feature
        
        # Create a temporary directory with files
        (tmp_path / "test.py").write_text("print('hello')")
        
        result = runner.invoke(cli, [
            'review', 'directory',
            str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert 'Analyzing directory' in result.output
        mock_feature.analyze_directory.assert_called_once()
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_directory_with_patterns(self, mock_class, runner, mock_feature, tmp_path):
        """Test directory review with file patterns"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'directory',
            str(tmp_path),
            '--patterns', '*.py',
            '--patterns', '*.js'
        ])
        
        assert result.exit_code == 0
        call_args = mock_feature.analyze_directory.call_args
        assert call_args[0][1] == ['*.py', '*.js']
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_directory_with_language_filter(self, mock_class, runner, mock_feature, tmp_path):
        """Test directory review with language filter"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'directory',
            str(tmp_path),
            '--language', 'python'
        ])
        
        assert result.exit_code == 0
        mock_feature.generate_formatted_report.assert_called_once()
    
    @patch('xencode.features.code_review.CodeReviewFeature')
    def test_directory_with_severity_filter(self, mock_class, runner, mock_feature, tmp_path):
        """Test directory review with severity filter"""
        mock_class.return_value = mock_feature
        
        result = runner.invoke(cli, [
            'review', 'directory',
            str(tmp_path),
            '--severity', 'high'
        ])
        
        assert result.exit_code == 0
        mock_feature.generate_formatted_report.assert_called_once()


class TestReviewCommandGroup:
    """Tests for the review command group"""
    
    def test_review_help(self, runner):
        """Test that review command group help works"""
        result = runner.invoke(cli, ['review', '--help'])
        assert result.exit_code == 0
        assert 'AI Code Review commands' in result.output
        assert 'pr' in result.output
        assert 'file' in result.output
        assert 'directory' in result.output
    
    def test_review_no_subcommand(self, runner):
        """Test review command without subcommand shows help"""
        result = runner.invoke(cli, ['review'])
        # Click returns exit code 0 for help display
        assert 'AI Code Review commands' in result.output
