"""
Integration tests for CodeLinter with CodeReviewFeature
"""

import pytest
from xencode.features.code_review import CodeReviewFeature, CodeLinter
from xencode.features.base import FeatureConfig


class TestCodeLinterIntegration:
    """Integration tests for CodeLinter"""
    
    @pytest.mark.asyncio
    async def test_linter_standalone_usage(self):
        """Test using CodeLinter directly without file system"""
        linter = CodeLinter()
        
        files = [{
            'path': 'test.py',
            'language': 'python',
            'content': '''
password = "hardcoded_secret"
cursor.execute("SELECT * FROM users WHERE id = " + user_id)
'''
        }]
        
        result = await linter.analyze(files)
        
        # Should have analyzed the file and found issues
        assert result['summary']['total_files'] == 1
        assert result['summary']['total_issues'] > 0
        assert 'issues' in result
        assert len(result['issues']) > 0
    
    @pytest.mark.asyncio
    async def test_linter_detects_multiple_issue_types(self):
        """Test that linter can detect multiple types of issues in one file"""
        linter = CodeLinter()
        
        files = [{
            'path': 'vulnerable.py',
            'language': 'python',
            'content': '''
import hashlib

# Hardcoded password
password = "super_secret_123"

# SQL injection
def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id = " + user_id)
    return cursor.fetchone()

# Insecure hash
def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()

# Command injection
import os
def run_cmd(cmd):
    os.system(cmd)

# Bare except
try:
    risky_operation()
except:
    pass
'''
        }]
        
        result = await linter.analyze(files)
        
        # Should detect multiple issue types
        assert result['summary']['total_issues'] > 0
        
        issue_types = set(issue['type'] for issue in result['issues'])
        
        # Should have detected various security issues
        assert 'hardcoded_secrets' in issue_types
        assert 'sqli' in issue_types
        assert 'insecure_crypto' in issue_types
        assert 'command_injection' in issue_types
        assert 'code_quality' in issue_types
        
        # Should have critical, high, and medium severity issues
        severities = set(issue['severity'] for issue in result['issues'])
        assert 'critical' in severities
        assert len(severities) > 1  # Multiple severity levels
    
    @pytest.mark.asyncio
    async def test_linter_language_specific_checks(self):
        """Test that linter applies language-specific checks"""
        linter = CodeLinter()
        
        files = [
            {
                'path': 'test.py',
                'language': 'python',
                'content': 'try:\n    pass\nexcept:\n    pass'
            },
            {
                'path': 'test.js',
                'language': 'javascript',
                'content': 'if (x == 5) { console.log("equal"); }'
            },
            {
                'path': 'test.ts',
                'language': 'typescript',
                'content': 'function test(data: any) { return data; }'
            },
            {
                'path': 'test.rs',
                'language': 'rust',
                'content': 'let value = result.unwrap();'
            }
        ]
        
        result = await linter.analyze(files)
        
        # Each file should have language-specific issues
        assert result['summary']['total_files'] == 4
        assert result['summary']['total_issues'] > 0
        
        # Check that each file was analyzed
        for file_result in result['files']:
            assert 'language' in file_result
            assert 'issues' in file_result
    
    @pytest.mark.asyncio
    async def test_linter_severity_distribution(self):
        """Test that linter correctly categorizes issues by severity"""
        linter = CodeLinter()
        
        files = [{
            'path': 'mixed.py',
            'language': 'python',
            'content': '''
# Critical: hardcoded secret
api_key = "sk_live_1234567890abcdef"

# Critical: SQL injection
cursor.execute("SELECT * FROM users WHERE id = " + user_id)

# High: XSS
element.innerHTML = user_input

# Medium: bare except
try:
    pass
except:
    pass

# Low: import check
import unused_module
'''
        }]
        
        result = await linter.analyze(files)
        
        # Should have issues at multiple severity levels
        by_severity = result['summary']['by_severity']
        assert by_severity['critical'] > 0
        assert by_severity['high'] >= 0
        assert by_severity['medium'] > 0
        assert by_severity['low'] >= 0
        
        # Total issues should match sum of severities
        total = sum(by_severity.values())
        assert total == result['summary']['total_issues']
