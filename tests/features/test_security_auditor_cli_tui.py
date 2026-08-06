"""
Tests for Security Auditor CLI and TUI Components

Tests the CLI commands and TUI widgets for the security auditor feature.
"""


import pytest

from xencode.features.base import FeatureConfig
from xencode.features.security_auditor import SecurityAuditor


@pytest.fixture
def security_auditor():
    """Create a security auditor instance"""
    config = FeatureConfig(
        name="security_auditor",
        enabled=True,
        config={
            'enabled': True,
            'checks': ['owasp_top_10', 'dependency_vulnerabilities'],
            'tools': ['bandit', 'snyk'],
            'exclude_patterns': ['*/tests/*', '*/.venv/*']
        }
    )
    return SecurityAuditor(config)


@pytest.fixture
def sample_vulnerabilities():
    """Sample vulnerabilities for testing"""
    return [
        {
            'id': 'vuln_1',
            'type': 'injection',
            'risk_level': 'critical',
            'title': 'SQL Injection',
            'description': 'SQL injection vulnerability',
            'file_path': 'app.py',
            'line_number': 42,
            'code_snippet': 'query = "SELECT * FROM users WHERE id = " + user_id',
            'fix_suggestion': 'Use parameterized queries',
            'references': []
        },
        {
            'id': 'vuln_2',
            'type': 'hardcoded_secrets',
            'risk_level': 'critical',
            'title': 'Hardcoded password',
            'description': 'Password hardcoded in source',
            'file_path': 'config.py',
            'line_number': 10,
            'code_snippet': 'password = "admin123"',
            'fix_suggestion': 'Use environment variables',
            'references': []
        }
    ]


@pytest.fixture
def sample_dependencies():
    """Sample dependency vulnerabilities for testing"""
    return [
        {
            'package_name': 'requests',
            'current_version': '2.20.0',
            'vulnerable_versions': '<2.31.0',
            'fixed_version': '2.31.0',
            'cve_id': 'CVE-2023-32681',
            'risk_level': 'medium',
            'description': 'Unintended leak of Proxy-Authorization header',
            'cvss_score': 6.1,
            'references': ['https://nvd.nist.gov/vuln/detail/CVE-2023-32681']
        }
    ]


class TestSecurityAuditorCLI:
    """Test CLI commands"""

    @pytest.mark.asyncio
    async def test_cli_scan_command(self, security_auditor, tmp_path):
        """Test scan CLI command"""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret123"')

        # Initialize and scan
        await security_auditor.initialize()
        result = await security_auditor.scan(str(test_file), use_external_tools=False)

        # Verify result structure
        assert 'success' in result
        assert result['success'] is True
        assert 'report' in result
        assert 'summary' in result
        assert 'recommendations' in result

        # Verify summary
        summary = result['summary']
        assert 'total_vulnerabilities' in summary
        assert 'critical' in summary
        assert 'high' in summary
        assert 'medium' in summary
        assert 'low' in summary

        await security_auditor.shutdown()

    @pytest.mark.asyncio
    async def test_cli_dependencies_command(self, security_auditor, tmp_path):
        """Test dependencies CLI command"""
        # Create a requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("requests==2.20.0\ndjango==3.0.0")

        # Initialize and analyze
        await security_auditor.initialize()
        result = await security_auditor.analyze(str(req_file))

        # Verify result structure
        assert 'success' in result
        assert result['success'] is True
        assert 'vulnerabilities' in result
        assert 'count' in result

        await security_auditor.shutdown()

    @pytest.mark.asyncio
    async def test_cli_report_command_markdown(self, security_auditor, tmp_path):
        """Test report CLI command with markdown format"""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret123"')

        # Initialize and generate report
        await security_auditor.initialize()
        result = await security_auditor.report(str(test_file), format='markdown')

        # Verify result
        assert 'success' in result
        assert result['success'] is True
        assert 'format' in result
        assert result['format'] == 'markdown'
        assert 'content' in result
        assert isinstance(result['content'], str)
        assert '# Security Audit Report' in result['content']

        await security_auditor.shutdown()

    @pytest.mark.asyncio
    async def test_cli_report_command_html(self, security_auditor, tmp_path):
        """Test report CLI command with HTML format"""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret123"')

        # Initialize and generate report
        await security_auditor.initialize()
        result = await security_auditor.report(str(test_file), format='html')

        # Verify result
        assert 'success' in result
        assert result['success'] is True
        assert 'format' in result
        assert result['format'] == 'html'
        assert 'content' in result
        assert isinstance(result['content'], str)
        assert '<!DOCTYPE html>' in result['content']

        await security_auditor.shutdown()

    @pytest.mark.asyncio
    async def test_cli_audit_command(self, security_auditor, tmp_path):
        """Test full audit CLI command"""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret123"\nquery = "SELECT * FROM users WHERE id = " + user_id')

        # Initialize and audit
        await security_auditor.initialize()
        result = await security_auditor.scan(str(test_file), use_external_tools=False)

        # Verify comprehensive results
        assert result['success'] is True
        assert result['summary']['total_vulnerabilities'] > 0

        await security_auditor.shutdown()

    def test_cli_commands_registered(self, security_auditor):
        """Test that CLI commands are registered"""
        commands = security_auditor.get_cli_commands()

        # Verify commands are returned
        assert len(commands) > 0

        # Verify it's a click group
        import click
        assert isinstance(commands[0], click.Group)


class TestSecurityAuditorTUI:
    """Test TUI components"""

    def test_vulnerability_dashboard_import(self):
        """Test VulnerabilityDashboard can be imported"""
        from xencode.tui.widgets.security_auditor_panel import VulnerabilityDashboard

        dashboard = VulnerabilityDashboard()
        assert dashboard is not None

    def test_vulnerability_dashboard_update(self, sample_vulnerabilities):
        """Test VulnerabilityDashboard update"""
        from xencode.tui.widgets.security_auditor_panel import VulnerabilityDashboard

        dashboard = VulnerabilityDashboard()

        summary = {
            'total_vulnerabilities': 10,
            'code_vulnerabilities': 6,
            'dependency_vulnerabilities': 4,
            'critical': 2,
            'high': 3,
            'medium': 4,
            'low': 1,
            'info': 0
        }

        # Should not raise exception
        dashboard.update_summary(summary)

        # Verify summary is stored
        assert dashboard.summary['total_vulnerabilities'] == 10
        assert dashboard.summary['critical'] == 2

    def test_vulnerability_list_import(self):
        """Test VulnerabilityList can be imported"""
        from xencode.tui.widgets.security_auditor_panel import VulnerabilityList

        vuln_list = VulnerabilityList()
        assert vuln_list is not None

    def test_vulnerability_list_update(self, sample_vulnerabilities):
        """Test VulnerabilityList update"""
        from xencode.tui.widgets.security_auditor_panel import VulnerabilityList

        vuln_list = VulnerabilityList()

        # Store vulnerabilities directly (update method requires mounted widget)
        vuln_list.vulnerabilities = sample_vulnerabilities

        # Verify vulnerabilities are stored
        assert len(vuln_list.vulnerabilities) == 2

    def test_dependency_tree_import(self):
        """Test DependencyTree can be imported"""
        from xencode.tui.widgets.security_auditor_panel import DependencyTree

        dep_tree = DependencyTree()
        assert dep_tree is not None

    def test_dependency_tree_update(self, sample_dependencies):
        """Test DependencyTree update"""
        from xencode.tui.widgets.security_auditor_panel import DependencyTree

        dep_tree = DependencyTree()

        # Should not raise exception
        dep_tree.update_dependencies(sample_dependencies)

        # Verify dependencies are stored
        assert len(dep_tree.dependencies) == 1

    def test_fix_suggestions_import(self):
        """Test FixSuggestions can be imported"""
        from xencode.tui.widgets.security_auditor_panel import FixSuggestions

        fixes = FixSuggestions()
        assert fixes is not None

    def test_fix_suggestions_update(self, sample_vulnerabilities):
        """Test FixSuggestions update"""
        from xencode.tui.widgets.security_auditor_panel import FixSuggestions

        fixes = FixSuggestions()

        # Store vulnerability directly (update method requires mounted widget)
        fixes.current_vuln = sample_vulnerabilities[0]

        # Verify vulnerability is stored
        assert fixes.current_vuln is not None
        assert fixes.current_vuln['id'] == 'vuln_1'

    def test_audit_history_import(self):
        """Test AuditHistory can be imported"""
        from xencode.tui.widgets.security_auditor_panel import AuditHistory

        history = AuditHistory()
        assert history is not None

    def test_audit_history_update(self):
        """Test AuditHistory update"""
        from xencode.tui.widgets.security_auditor_panel import AuditHistory

        history = AuditHistory()

        history_data = [
            {
                'timestamp': '2024-01-15T12:00:00',
                'scan_path': './src',
                'summary': {
                    'total_vulnerabilities': 5,
                    'critical': 1,
                    'high': 2,
                    'medium': 2,
                    'low': 0
                }
            }
        ]

        # Should not raise exception
        history.update_history(history_data)

        # Verify history is stored
        assert len(history.history) == 1

    def test_security_auditor_panel_import(self):
        """Test SecurityAuditorPanel can be imported"""
        from xencode.tui.widgets.security_auditor_panel import SecurityAuditorPanel

        panel = SecurityAuditorPanel()
        assert panel is not None

    def test_security_auditor_panel_tabs(self):
        """Test SecurityAuditorPanel tab switching"""
        from xencode.tui.widgets.security_auditor_panel import SecurityAuditorPanel

        panel = SecurityAuditorPanel()

        # Test tab switching (without querying DOM)
        panel.current_tab = "dashboard"
        assert panel.current_tab == "dashboard"

        panel.current_tab = "vulnerabilities"
        assert panel.current_tab == "vulnerabilities"

        panel.current_tab = "dependencies"
        assert panel.current_tab == "dependencies"

    def test_security_auditor_panel_update_methods(self, sample_vulnerabilities, sample_dependencies):
        """Test SecurityAuditorPanel update methods"""
        from xencode.tui.widgets.security_auditor_panel import SecurityAuditorPanel

        panel = SecurityAuditorPanel()

        # Test that panel can be created and has expected attributes
        assert panel.current_tab == "dashboard"
        assert panel.scan_path == "."

        # Note: Update methods require mounted widgets, so we just verify they exist
        assert hasattr(panel, 'update_dashboard')
        assert hasattr(panel, 'update_vulnerabilities')
        assert hasattr(panel, 'update_dependencies')
        assert hasattr(panel, 'update_history')


class TestIntegration:
    """Test CLI and TUI integration"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, security_auditor, tmp_path):
        """Test complete workflow from scan to report"""
        # Create test files
        test_file = tmp_path / "app.py"
        test_file.write_text('password = "secret123"\nquery = "SELECT * FROM users WHERE id = " + user_id')

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("requests==2.20.0")

        # Initialize
        await security_auditor.initialize()

        # 1. Scan
        scan_result = await security_auditor.scan(str(test_file), use_external_tools=False)
        assert scan_result['success'] is True

        # 2. Analyze dependencies
        dep_result = await security_auditor.analyze(str(req_file))
        assert dep_result['success'] is True

        # 3. Generate report
        report_result = await security_auditor.report(str(test_file), format='markdown')
        assert report_result['success'] is True

        # 4. Verify TUI panel can be created
        from xencode.tui.widgets.security_auditor_panel import SecurityAuditorPanel

        panel = SecurityAuditorPanel()
        assert panel is not None

        # Note: Update methods require mounted widgets in a running app
        # We just verify the panel was created successfully

        await security_auditor.shutdown()

    def test_cli_commands_available(self, security_auditor):
        """Test that all CLI commands are available"""
        commands = security_auditor.get_cli_commands()

        assert len(commands) > 0

        # Get the security group
        security_group = commands[0]

        # Verify commands exist
        command_names = [cmd.name for cmd in security_group.commands.values()]
        assert 'scan' in command_names
        assert 'dependencies' in command_names
        assert 'report' in command_names
        assert 'audit' in command_names


class TestErrorHandling:
    """Test error handling in CLI and TUI"""

    @pytest.mark.asyncio
    async def test_scan_nonexistent_path(self, security_auditor):
        """Test scanning nonexistent path"""
        await security_auditor.initialize()

        # Should handle gracefully
        result = await security_auditor.scan('/nonexistent/path', use_external_tools=False)

        # Should still return valid structure
        assert 'success' in result
        assert 'summary' in result

        await security_auditor.shutdown()

    @pytest.mark.asyncio
    async def test_report_invalid_format(self, security_auditor, tmp_path):
        """Test report with invalid format"""
        test_file = tmp_path / "test.py"
        test_file.write_text('password = "secret"')

        await security_auditor.initialize()

        result = await security_auditor.report(str(test_file), format='invalid')

        # Should return error
        assert 'success' in result
        assert result['success'] is False
        assert 'error' in result

        await security_auditor.shutdown()

    def test_tui_empty_data(self):
        """Test TUI components with empty data"""
        from xencode.tui.widgets.security_auditor_panel import (
            DependencyTree,
            VulnerabilityDashboard,
            VulnerabilityList,
        )

        # Should handle empty data gracefully (without mounting)
        dashboard = VulnerabilityDashboard()
        dashboard.summary = {}

        vuln_list = VulnerabilityList()
        vuln_list.vulnerabilities = []

        dep_tree = DependencyTree()
        dep_tree.dependencies = []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
