"""
Tests for Security Auditor feature
"""

import asyncio
import pytest
from pathlib import Path
from xencode.features.security_auditor import (
    SecurityAuditor,
    VulnerabilityScanner,
    DependencyAnalyzer,
    SecurityReportGenerator,
    SecurityToolIntegration,
    VulnerabilityType,
    RiskLevel,
    SecurityAuditorConfig
)
from xencode.features.base import FeatureConfig


@pytest.fixture
def security_config():
    """Create test security auditor config"""
    return FeatureConfig(
        name="security_auditor",
        enabled=True,
        config={
            'enabled': True,
            'checks': ['owasp_top_10', 'dependency_vulnerabilities'],
            'tools': ['bandit', 'snyk'],
            'exclude_patterns': ['*/tests/*', '*/.venv/*']
        }
    )


@pytest.fixture
def security_auditor(security_config):
    """Create security auditor instance"""
    return SecurityAuditor(security_config)


@pytest.fixture
def temp_vulnerable_file(tmp_path):
    """Create a temporary file with vulnerabilities"""
    file_path = tmp_path / "vulnerable.py"
    content = """
import os

# Hardcoded password - security issue
password = "secret123"

# SQL injection vulnerability
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    execute(query)

# Command injection
def run_command(cmd):
    os.system("ls " + cmd)

# Weak crypto
import hashlib
def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def temp_requirements_file(tmp_path):
    """Create a temporary requirements.txt with vulnerable dependencies"""
    file_path = tmp_path / "requirements.txt"
    content = """
requests==2.20.0
django==3.0.0
flask==2.0.0
pyyaml==5.1
"""
    file_path.write_text(content)
    return file_path


class TestVulnerabilityScanner:
    """Test vulnerability scanner"""
    
    @pytest.mark.asyncio
    async def test_scan_file_with_vulnerabilities(self, temp_vulnerable_file):
        """Test scanning a file with vulnerabilities"""
        scanner = VulnerabilityScanner()
        vulnerabilities = await scanner.scan(str(temp_vulnerable_file))
        
        assert len(vulnerabilities) > 0
        
        # Check for hardcoded password
        password_vulns = [v for v in vulnerabilities 
                         if v.type == VulnerabilityType.HARDCODED_SECRETS]
        assert len(password_vulns) > 0
        
        # Check for injection vulnerabilities
        injection_vulns = [v for v in vulnerabilities 
                          if v.type == VulnerabilityType.INJECTION]
        assert len(injection_vulns) > 0
        
        # Check for weak crypto
        crypto_vulns = [v for v in vulnerabilities 
                       if v.type == VulnerabilityType.WEAK_CRYPTO]
        assert len(crypto_vulns) > 0
    
    @pytest.mark.asyncio
    async def test_scan_directory(self, tmp_path, temp_vulnerable_file):
        """Test scanning a directory"""
        scanner = VulnerabilityScanner()
        vulnerabilities = await scanner.scan(str(tmp_path))
        
        assert len(vulnerabilities) > 0
    
    @pytest.mark.asyncio
    async def test_exclude_patterns(self, tmp_path):
        """Test file exclusion patterns"""
        # Create test file in excluded directory
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_file.py"
        test_file.write_text("password = 'secret'")
        
        scanner = VulnerabilityScanner(exclude_patterns=['*/tests/*'])
        vulnerabilities = await scanner.scan(str(tmp_path))
        
        # Should not find vulnerabilities in excluded directory
        assert len(vulnerabilities) == 0


class TestDependencyAnalyzer:
    """Test dependency analyzer"""
    
    @pytest.mark.asyncio
    async def test_analyze_requirements(self, temp_requirements_file):
        """Test analyzing requirements.txt"""
        analyzer = DependencyAnalyzer()
        vulnerabilities = await analyzer.analyze(str(temp_requirements_file))
        
        # Should find some vulnerable dependencies
        assert len(vulnerabilities) > 0
        
        # Check vulnerability structure
        for vuln in vulnerabilities:
            assert vuln.package_name
            assert vuln.cve_id
            assert vuln.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, 
                                      RiskLevel.MEDIUM, RiskLevel.LOW]
    
    @pytest.mark.asyncio
    async def test_analyze_directory(self, tmp_path, temp_requirements_file):
        """Test analyzing a directory with dependency files"""
        analyzer = DependencyAnalyzer()
        vulnerabilities = await analyzer.analyze(str(tmp_path))
        
        assert len(vulnerabilities) > 0


class TestSecurityReportGenerator:
    """Test security report generator"""
    
    @pytest.mark.asyncio
    async def test_generate_report(self, temp_vulnerable_file, temp_requirements_file):
        """Test generating security report"""
        scanner = VulnerabilityScanner()
        analyzer = DependencyAnalyzer()
        generator = SecurityReportGenerator()
        
        vulnerabilities = await scanner.scan(str(temp_vulnerable_file))
        dep_vulnerabilities = await analyzer.analyze(str(temp_requirements_file))
        
        report = await generator.generate(
            scan_path=str(temp_vulnerable_file.parent),
            vulnerabilities=vulnerabilities,
            dependency_vulnerabilities=dep_vulnerabilities
        )
        
        assert report.timestamp
        assert report.scan_path
        assert len(report.vulnerabilities) > 0
        assert len(report.dependency_vulnerabilities) > 0
        assert report.summary['total_vulnerabilities'] > 0
        assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_generate_markdown_report(self, temp_vulnerable_file):
        """Test generating markdown report"""
        scanner = VulnerabilityScanner()
        generator = SecurityReportGenerator()
        
        vulnerabilities = await scanner.scan(str(temp_vulnerable_file))
        
        report = await generator.generate(
            scan_path=str(temp_vulnerable_file),
            vulnerabilities=vulnerabilities,
            dependency_vulnerabilities=[]
        )
        
        markdown = await generator.generate_markdown(report)
        
        assert "# Security Audit Report" in markdown
        assert "## Summary" in markdown
        assert "## Code Vulnerabilities" in markdown
    
    @pytest.mark.asyncio
    async def test_generate_html_report(self, temp_vulnerable_file):
        """Test generating HTML report"""
        scanner = VulnerabilityScanner()
        generator = SecurityReportGenerator()
        
        vulnerabilities = await scanner.scan(str(temp_vulnerable_file))
        
        report = await generator.generate(
            scan_path=str(temp_vulnerable_file),
            vulnerabilities=vulnerabilities,
            dependency_vulnerabilities=[]
        )
        
        html = await generator.generate_html(report)
        
        assert "<!DOCTYPE html>" in html
        assert "Security Audit Report" in html
        assert "Summary" in html


class TestSecurityToolIntegration:
    """Test security tool integration"""
    
    def test_check_tools(self):
        """Test checking for available tools"""
        integration = SecurityToolIntegration()
        
        assert isinstance(integration.tools_available, dict)
        assert 'bandit' in integration.tools_available
        assert 'snyk' in integration.tools_available


class TestSecurityAuditor:
    """Test main security auditor class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, security_auditor):
        """Test security auditor initialization"""
        await security_auditor.initialize()
        
        assert security_auditor.name == "security_auditor"
        assert security_auditor.description
        assert security_auditor.scanner
        assert security_auditor.dependency_analyzer
        assert security_auditor.report_generator
        
        await security_auditor.shutdown()
    
    @pytest.mark.asyncio
    async def test_scan(self, security_auditor, temp_vulnerable_file):
        """Test scanning with security auditor"""
        await security_auditor.initialize()
        
        result = await security_auditor.scan(
            str(temp_vulnerable_file),
            use_external_tools=False
        )
        
        assert result['success']
        assert 'report' in result
        assert 'summary' in result
        assert 'recommendations' in result
        assert result['summary']['total_vulnerabilities'] > 0
        
        await security_auditor.shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze(self, security_auditor, temp_requirements_file):
        """Test dependency analysis"""
        await security_auditor.initialize()
        
        result = await security_auditor.analyze(str(temp_requirements_file))
        
        assert result['success']
        assert 'vulnerabilities' in result
        assert 'count' in result
        
        await security_auditor.shutdown()
    
    @pytest.mark.asyncio
    async def test_report_markdown(self, security_auditor, temp_vulnerable_file):
        """Test generating markdown report"""
        await security_auditor.initialize()
        
        result = await security_auditor.report(
            str(temp_vulnerable_file),
            format='markdown'
        )
        
        assert result['success']
        assert result['format'] == 'markdown'
        assert '# Security Audit Report' in result['content']
        
        await security_auditor.shutdown()
    
    @pytest.mark.asyncio
    async def test_report_html(self, security_auditor, temp_vulnerable_file):
        """Test generating HTML report"""
        await security_auditor.initialize()
        
        result = await security_auditor.report(
            str(temp_vulnerable_file),
            format='html'
        )
        
        assert result['success']
        assert result['format'] == 'html'
        assert '<!DOCTYPE html>' in result['content']
        
        await security_auditor.shutdown()
    
    @pytest.mark.asyncio
    async def test_cli_commands(self, security_auditor):
        """Test CLI commands are available"""
        commands = security_auditor.get_cli_commands()
        
        assert len(commands) > 0
        assert commands[0].name == 'security'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
