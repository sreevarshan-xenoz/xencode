#!/usr/bin/env python3
"""
Tests for Enhanced Security Analyzer with Bandit Integration
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from xencode.analyzers.security_analyzer import (
    SecurityAnalyzer,
    BanditIntegration,
    VulnerabilityDatabase,
    SecurityReportGenerator
)
from xencode.models.code_analysis import (
    AnalysisIssue,
    AnalysisType,
    CodeLocation,
    Language,
    SecurityIssue,
    SeverityLevel
)


class TestBanditIntegration:
    """Test Bandit integration functionality"""
    
    def test_bandit_availability_check(self):
        """Test Bandit availability detection"""
        bandit = BanditIntegration()
        # Should not raise exception
        assert isinstance(bandit.bandit_available, bool)
    
    @pytest.mark.asyncio
    async def test_scan_python_code_no_bandit(self):
        """Test Python code scanning when Bandit is not available"""
        bandit = BanditIntegration()
        bandit.bandit_available = False
        
        code = "import os\nos.system('ls')"
        results = await bandit.scan_python_code(code)
        
        assert results == []
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    @patch('asyncio.create_subprocess_exec')
    async def test_scan_python_code_with_bandit(self, mock_subprocess, mock_run):
        """Test Python code scanning with Bandit available"""
        # Mock Bandit availability check
        mock_run.return_value.returncode = 0
        
        # Mock Bandit subprocess execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b'{"results": [{"issue_severity": "HIGH", "issue_confidence": "HIGH", "issue_text": "Test issue", "line_number": 1, "test_id": "B101"}]}',
            b''
        )
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process
        
        bandit = BanditIntegration()
        bandit.bandit_available = True
        
        code = "assert False"
        results = await bandit.scan_python_code(code)
        
        assert len(results) == 1
        assert results[0]['issue_severity'] == 'HIGH'
        assert results[0]['test_id'] == 'B101'
    
    def test_convert_bandit_to_analysis_issues(self):
        """Test conversion of Bandit results to AnalysisIssue objects"""
        bandit = BanditIntegration()
        
        bandit_results = [
            {
                'issue_severity': 'HIGH',
                'issue_confidence': 'MEDIUM',
                'issue_text': 'Use of assert detected',
                'line_number': 5,
                'col_offset': 10,
                'code': 'assert False',
                'test_id': 'B101',
                'test_name': 'assert_used'
            }
        ]
        
        issues = bandit.convert_bandit_to_analysis_issues(bandit_results, "test.py")
        
        assert len(issues) == 1
        issue = issues[0]
        assert issue.analysis_type == AnalysisType.SECURITY
        assert issue.severity == SeverityLevel.ERROR
        assert issue.location.line == 5
        assert issue.location.column == 10
        assert issue.rule_id == 'B101'
        assert issue.confidence == 0.7


class TestVulnerabilityDatabase:
    """Test vulnerability database functionality"""
    
    def test_initialization(self):
        """Test vulnerability database initialization"""
        vuln_db = VulnerabilityDatabase()
        
        assert hasattr(vuln_db, 'cve_patterns')
        assert hasattr(vuln_db, 'owasp_top10')
        assert Language.PYTHON in vuln_db.cve_patterns
    
    @pytest.mark.asyncio
    async def test_check_vulnerabilities_python(self):
        """Test vulnerability checking for Python code"""
        vuln_db = VulnerabilityDatabase()
        
        code = """
import pickle
data = pickle.loads(user_input)
config = yaml.load(config_file)
"""
        
        issues = await vuln_db.check_vulnerabilities(code, Language.PYTHON)
        
        assert len(issues) >= 2  # Should find pickle and yaml issues
        
        # Check for pickle vulnerability
        pickle_issues = [issue for issue in issues if 'pickle' in issue.description.lower()]
        assert len(pickle_issues) > 0
        
        # Check for YAML vulnerability
        yaml_issues = [issue for issue in issues if 'yaml' in issue.description.lower()]
        assert len(yaml_issues) > 0
    
    @pytest.mark.asyncio
    async def test_check_vulnerabilities_javascript(self):
        """Test vulnerability checking for JavaScript code"""
        vuln_db = VulnerabilityDatabase()
        
        code = """
var result = JSON.parse(userInput + extraData);
var func = new Function(userCode);
window.location = baseUrl + userPath;
"""
        
        issues = await vuln_db.check_vulnerabilities(code, Language.JAVASCRIPT)
        
        assert len(issues) >= 2  # Should find JSON and Function issues


class TestSecurityReportGenerator:
    """Test security report generation"""
    
    def test_initialization(self):
        """Test report generator initialization"""
        generator = SecurityReportGenerator()
        
        assert 'summary' in generator.report_templates
        assert 'detailed' in generator.report_templates
        assert 'executive' in generator.report_templates
    
    @pytest.mark.asyncio
    async def test_generate_summary_report(self):
        """Test summary report generation"""
        generator = SecurityReportGenerator()
        
        # Create sample issues
        issues = [
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.ERROR,
                message="Critical security issue",
                description="Test critical issue",
                location=CodeLocation(line=1, column=1),
                code_snippet="test code",
                rule_id="test_critical",
                rule_name="Critical Test",
                confidence=0.9
            ),
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.WARNING,
                message="High security issue",
                description="Test high issue",
                location=CodeLocation(line=2, column=1),
                code_snippet="test code 2",
                rule_id="test_high",
                rule_name="High Test",
                confidence=0.8
            )
        ]
        
        report = await generator.generate_report(issues, 'summary')
        
        assert "Security Analysis Summary" in report
        assert "**Total Issues Found**: 2" in report
        assert "**Critical Issues**: 1" in report
        assert "**High Severity**: 1" in report
        assert "CRITICAL" in report  # Risk level
    
    @pytest.mark.asyncio
    async def test_generate_detailed_report(self):
        """Test detailed report generation"""
        generator = SecurityReportGenerator()
        
        issues = [
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.ERROR,
                message="SQL Injection vulnerability",
                description="Potential SQL injection",
                location=CodeLocation(line=10, column=5),
                code_snippet="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
                rule_id="sql_injection",
                rule_name="SQL Injection",
                confidence=0.95
            )
        ]
        
        report = await generator.generate_report(issues, 'detailed')
        
        assert "Detailed Security Analysis Report" in report
        assert "Executive Summary" in report
        assert "Methodology" in report
        assert "Bandit integration" in report
        assert "Vulnerability Details" in report


class TestSecurityAnalyzer:
    """Test main security analyzer functionality"""
    
    def test_initialization(self):
        """Test security analyzer initialization"""
        analyzer = SecurityAnalyzer()
        
        assert hasattr(analyzer, 'bandit_integration')
        assert hasattr(analyzer, 'vulnerability_db')
        assert hasattr(analyzer, 'report_generator')
        assert hasattr(analyzer, 'security_patterns')
    
    @pytest.mark.asyncio
    async def test_analyze_security_python_basic(self):
        """Test basic Python security analysis"""
        analyzer = SecurityAnalyzer()
        
        code = """
import os
password = "hardcoded123"
os.system("rm -rf /")
eval(user_input)
"""
        
        analysis_issues, security_issues = await analyzer.analyze_security(
            code, Language.PYTHON, "test.py"
        )
        
        assert len(analysis_issues) > 0
        assert len(security_issues) > 0
        
        # Check for specific vulnerabilities
        issue_messages = [issue.message for issue in analysis_issues]
        assert any('eval' in msg.lower() for msg in issue_messages)
        assert any('password' in msg.lower() for msg in issue_messages)
    
    @pytest.mark.asyncio
    async def test_analyze_security_javascript(self):
        """Test JavaScript security analysis"""
        analyzer = SecurityAnalyzer()
        
        code = """
var userInput = getInput();
eval(userInput);
document.getElementById('content').innerHTML = userInput;
var apiKey = "sk-1234567890abcdef";
"""
        
        analysis_issues, security_issues = await analyzer.analyze_security(
            code, Language.JAVASCRIPT, "test.js"
        )
        
        assert len(analysis_issues) > 0
        assert len(security_issues) > 0
        
        # Check for specific vulnerabilities
        issue_messages = [issue.message for issue in analysis_issues]
        assert any('eval' in msg.lower() for msg in issue_messages)
        assert any('innerHTML' in msg.lower() or 'xss' in msg.lower() for msg in issue_messages)
    
    @pytest.mark.asyncio
    async def test_analyze_security_unsupported_language(self):
        """Test security analysis for unsupported language"""
        analyzer = SecurityAnalyzer()
        
        code = "some code in unsupported language"
        
        analysis_issues, security_issues = await analyzer.analyze_security(
            code, Language.GO, "test.go"  # Assuming GO is not supported
        )
        
        # Should still run vulnerability database checks
        assert isinstance(analysis_issues, list)
        assert isinstance(security_issues, list)
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_scan(self):
        """Test comprehensive security scan"""
        analyzer = SecurityAnalyzer()
        
        code = """
import pickle
import hashlib

password = "secret123"
data = pickle.loads(user_data)
hash_value = hashlib.md5(password.encode()).hexdigest()
eval(user_input)
"""
        
        result = await analyzer.run_comprehensive_scan(
            code, Language.PYTHON, "test.py", generate_report=True
        )
        
        assert 'analysis_issues' in result
        assert 'security_issues' in result
        assert 'metrics' in result
        assert 'report' in result
        assert 'scan_timestamp' in result
        assert 'bandit_available' in result
        
        # Check metrics
        metrics = result['metrics']
        assert 'total_issues' in metrics
        assert 'security_score' in metrics
        assert 'risk_level' in metrics
        assert 'compliance_percentage' in metrics
        
        # Should have found multiple issues
        assert metrics['total_issues'] > 0
        assert metrics['security_score'] < 100  # Should have deductions
    
    @pytest.mark.asyncio
    async def test_generate_security_report(self):
        """Test security report generation"""
        analyzer = SecurityAnalyzer()
        
        # Create sample analysis issues
        issues = [
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.ERROR,
                message="Code injection vulnerability",
                description="Use of eval() detected",
                location=CodeLocation(line=5, column=1),
                code_snippet="eval(user_input)",
                rule_id="code_injection",
                rule_name="Code Injection",
                confidence=0.9
            )
        ]
        
        report = await analyzer.generate_security_report(issues, 'summary')
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Security Analysis Summary" in report
    
    def test_extract_cwe_from_bandit(self):
        """Test CWE extraction from Bandit rule IDs"""
        analyzer = SecurityAnalyzer()
        
        # Test known Bandit rule mappings
        assert analyzer._extract_cwe_from_bandit('B101') == 'CWE-78'
        assert analyzer._extract_cwe_from_bandit('B105') == 'CWE-798'
        assert analyzer._extract_cwe_from_bandit('B301') == 'CWE-502'
        assert analyzer._extract_cwe_from_bandit('B307') == 'CWE-94'
        
        # Test unknown rule
        assert analyzer._extract_cwe_from_bandit('B999') is None
    
    def test_get_bandit_mitigation(self):
        """Test Bandit mitigation advice"""
        analyzer = SecurityAnalyzer()
        
        # Test known mitigations
        mitigation = analyzer._get_bandit_mitigation('B105')
        assert 'environment variables' in mitigation.lower()
        
        mitigation = analyzer._get_bandit_mitigation('B307')
        assert 'eval' in mitigation.lower()
        
        # Test unknown rule
        mitigation = analyzer._get_bandit_mitigation('B999')
        assert 'security best practices' in mitigation.lower()
    
    def test_calculate_security_metrics(self):
        """Test security metrics calculation"""
        analyzer = SecurityAnalyzer()
        
        issues = [
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.ERROR,
                message="Critical issue",
                description="Test",
                location=CodeLocation(line=1, column=1),
                code_snippet="test",
                rule_id="test1",
                rule_name="Test Rule 1",
                confidence=0.9
            ),
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.WARNING,
                message="High issue",
                description="Test",
                location=CodeLocation(line=2, column=1),
                code_snippet="test",
                rule_id="test2",
                rule_name="Test Rule 2",
                confidence=0.8
            ),
            AnalysisIssue(
                analysis_type=AnalysisType.SECURITY,
                severity=SeverityLevel.INFO,
                message="Medium issue",
                description="Test",
                location=CodeLocation(line=3, column=1),
                code_snippet="test",
                rule_id="test3",
                rule_name="Test Rule 3",
                confidence=0.7
            )
        ]
        
        metrics = analyzer._calculate_security_metrics(issues)
        
        assert metrics['total_issues'] == 3
        assert metrics['critical_count'] == 1
        assert metrics['high_count'] == 1
        assert metrics['medium_count'] == 1
        assert metrics['security_score'] == 60  # 100 - (1*25 + 1*10 + 1*5)
        assert metrics['risk_level'] == 'CRITICAL'
        assert metrics['compliance_percentage'] == 60
        assert 'Test Rule 1' in metrics['issue_categories']
        assert 'Test Rule 2' in metrics['issue_categories']
        assert 'Test Rule 3' in metrics['issue_categories']


class TestIntegration:
    """Integration tests for the enhanced security analyzer"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_analysis(self):
        """Test complete end-to-end security analysis workflow"""
        analyzer = SecurityAnalyzer()
        
        # Sample vulnerable Python code
        vulnerable_code = """
import os
import pickle
import hashlib
from flask import Flask, request

app = Flask(__name__)
app.config['DEBUG'] = True

# Hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"

@app.route('/execute')
def execute_command():
    cmd = request.args.get('cmd')
    # Command injection vulnerability
    os.system(cmd)
    return "Command executed"

@app.route('/load_data')
def load_data():
    data = request.args.get('data')
    # Deserialization vulnerability
    result = pickle.loads(data.encode())
    return str(result)

@app.route('/hash')
def hash_password():
    password = request.args.get('password')
    # Weak cryptography
    hash_value = hashlib.md5(password.encode()).hexdigest()
    return hash_value

@app.route('/eval')
def eval_code():
    code = request.args.get('code')
    # Code injection vulnerability
    result = eval(code)
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
        
        # Run comprehensive scan
        result = await analyzer.run_comprehensive_scan(
            vulnerable_code, Language.PYTHON, "vulnerable_app.py", generate_report=True
        )
        
        # Verify results
        assert len(result['analysis_issues']) > 5  # Should find multiple vulnerabilities
        assert result['metrics']['total_issues'] > 5
        assert result['metrics']['security_score'] < 50  # Should be low due to many issues
        assert result['metrics']['risk_level'] in ['CRITICAL', 'HIGH']
        
        # Verify specific vulnerabilities were found
        issue_messages = [issue.message.lower() for issue in result['analysis_issues']]
        
        # Should detect various vulnerability types
        vulnerability_types = [
            'hardcoded',  # Hardcoded credentials
            'command',    # Command injection
            'pickle',     # Deserialization
            'md5',        # Weak crypto
            'eval',       # Code injection
            'debug'       # Debug mode
        ]
        
        found_types = []
        for vuln_type in vulnerability_types:
            if any(vuln_type in msg for msg in issue_messages):
                found_types.append(vuln_type)
        
        assert len(found_types) >= 4  # Should find at least 4 different vulnerability types
        
        # Verify report was generated
        assert result['report'] is not None
        assert len(result['report']) > 1000  # Should be a substantial report
        assert 'Security Analysis' in result['report']
    
    @pytest.mark.asyncio
    async def test_performance_with_large_codebase(self):
        """Test performance with larger codebase"""
        analyzer = SecurityAnalyzer()
        
        # Generate larger code sample
        large_code = """
import os
import sys
import json
import hashlib
from typing import Dict, List, Optional

class SecurityTest:
    def __init__(self):
        self.password = "hardcoded_password"
        self.api_key = "sk-test123"
    
    def process_data(self, user_input: str):
        # Multiple potential issues
        eval(user_input)
        os.system(f"echo {user_input}")
        hash_val = hashlib.md5(user_input.encode()).hexdigest()
        return hash_val
    
    def load_config(self, config_data: str):
        import pickle
        return pickle.loads(config_data.encode())

""" * 10  # Repeat to make it larger
        
        import time
        start_time = time.time()
        
        result = await analyzer.run_comprehensive_scan(
            large_code, Language.PYTHON, "large_test.py", generate_report=False
        )
        
        end_time = time.time()
        scan_duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert scan_duration < 30  # 30 seconds max
        
        # Should still find issues
        assert len(result['analysis_issues']) > 0
        assert result['metrics']['total_issues'] > 0


if __name__ == '__main__':
    pytest.main([__file__])