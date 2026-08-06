"""
Security Auditor CLI and TUI Demo

Demonstrates the security auditor CLI commands and TUI components.
"""

import asyncio
from pathlib import Path

from xencode.features.base import FeatureConfig
from xencode.features.security_auditor import SecurityAuditor


async def demo_cli_usage():
    """Demonstrate CLI-style usage"""

    print("=" * 70)
    print("Security Auditor CLI Demo")
    print("=" * 70)

    # Create configuration
    config = FeatureConfig(
        name="security_auditor",
        enabled=True,
        config={
            'enabled': True,
            'checks': ['owasp_top_10', 'dependency_vulnerabilities', 'code_patterns'],
            'tools': ['bandit', 'snyk'],
            'exclude_patterns': ['*/tests/*', '*/.venv/*', '*/node_modules/*']
        }
    )

    # Initialize security auditor
    auditor = SecurityAuditor(config)
    await auditor.initialize()

    # Example 1: Scan command
    print("\n1. xencode security scan <path>")
    print("-" * 70)

    # Create a demo vulnerable file
    demo_file = Path("demo_vulnerable.py")
    demo_file.write_text("""
# Example vulnerable code
API_KEY = "sk-1234567890abcdef"
password = "admin123"

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute(query)

import hashlib
def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()
""")

    try:
        result = await auditor.scan(str(demo_file), use_external_tools=False)

        print("\n🔒 Security Scan Results")
        print(f"{'=' * 50}")
        print(f"Path: {result['report']['scan_path']}")
        print("\n📊 Summary:")
        print(f"  Total Vulnerabilities: {result['summary']['total_vulnerabilities']}")
        print(f"  🔴 Critical: {result['summary']['critical']}")
        print(f"  🟠 High: {result['summary']['high']}")
        print(f"  🟡 Medium: {result['summary']['medium']}")
        print(f"  🟢 Low: {result['summary']['low']}")

        if result['report']['vulnerabilities']:
            print("\n🐛 Code Vulnerabilities:")
            for vuln in result['report']['vulnerabilities'][:3]:
                print(f"  - [{vuln['risk_level'].upper()}] {vuln['title']}")
                print(f"    Location: {vuln['file_path']}:{vuln['line_number']}")

        print("\n💡 Recommendations:")
        for rec in result['recommendations'][:3]:
            print(f"  {rec}")

    finally:
        if demo_file.exists():
            demo_file.unlink()

    # Example 2: Dependencies command
    print("\n\n2. xencode security dependencies <path>")
    print("-" * 70)

    # Create a demo requirements file
    req_file = Path("demo_requirements.txt")
    req_file.write_text("""
requests==2.20.0
django==3.0.0
flask==2.0.0
pyyaml==5.1
""")

    try:
        result = await auditor.analyze(str(req_file))

        print("\n📦 Dependency Analysis")
        print(f"{'=' * 50}")
        print(f"Found {result['count']} vulnerable dependencies\n")

        for vuln in result['vulnerabilities'][:3]:
            print(f"Package: {vuln['package_name']}")
            print(f"  Risk: {vuln['risk_level'].upper()}")
            print(f"  Current: {vuln['current_version']}")
            print(f"  Fixed: {vuln['fixed_version']}")
            print(f"  CVE: {vuln['cve_id']}")
            print(f"  Description: {vuln['description']}\n")

    finally:
        if req_file.exists():
            req_file.unlink()

    # Example 3: Report command
    print("\n3. xencode security report <path> --format markdown")
    print("-" * 70)

    # Create demo file again
    demo_file.write_text("""
password = "secret123"
api_key = "1234567890"
""")

    try:
        result = await auditor.report(str(demo_file), format='markdown')

        if result['success']:
            print("✅ Report generated!")
            print("\nReport preview (first 500 chars):")
            print(result['content'][:500])
            print("...")

            # Save report
            report_path = Path("security_report_demo.md")
            report_path.write_text(result['content'], encoding='utf-8')
            print(f"\n✓ Full report saved to: {report_path}")

    finally:
        if demo_file.exists():
            demo_file.unlink()

    # Example 4: Audit command
    print("\n\n4. xencode security audit <path>")
    print("-" * 70)

    # Create demo file
    demo_file.write_text("""
password = "secret123"
query = "SELECT * FROM users WHERE id = " + user_id
""")

    try:
        print("🔍 Running full security audit...")
        print("This may take a few minutes...\n")

        result = await auditor.scan(str(demo_file), use_external_tools=False)

        print("✅ Audit complete!")
        print("\n📊 Results:")
        print(f"  Total Issues: {result['summary']['total_vulnerabilities']}")
        print(f"  Critical: {result['summary']['critical']}")
        print(f"  High: {result['summary']['high']}")
        print(f"  Medium: {result['summary']['medium']}")
        print(f"  Low: {result['summary']['low']}")

        if result['summary']['critical'] > 0:
            print(f"\n⚠️  WARNING: {result['summary']['critical']} critical vulnerabilities found!")
            print("Address these immediately before deploying to production.")

    finally:
        if demo_file.exists():
            demo_file.unlink()

    # Shutdown
    await auditor.shutdown()

    print("\n" + "=" * 70)
    print("CLI Demo Complete!")
    print("=" * 70)


async def demo_tui_components():
    """Demonstrate TUI component usage"""

    print("\n\n" + "=" * 70)
    print("Security Auditor TUI Components Demo")
    print("=" * 70)

    # Import TUI components
    try:
        from xencode.tui.widgets.security_auditor_panel import (
            AuditHistory,
            DependencyTree,
            FixSuggestions,
            SecurityAuditorPanel,
            VulnerabilityDashboard,
            VulnerabilityList,
        )

        print("\n✅ TUI components imported successfully!")
        print("\nAvailable Components:")
        print("  • VulnerabilityDashboard - Shows vulnerability summary")
        print("  • VulnerabilityList - Lists detected vulnerabilities")
        print("  • DependencyTree - Shows dependency vulnerabilities")
        print("  • FixSuggestions - Displays fix recommendations")
        print("  • AuditHistory - Shows audit history")
        print("  • SecurityAuditorPanel - Main integrated panel")

        print("\n📝 Component Usage Examples:")

        # Example 1: VulnerabilityDashboard
        print("\n1. VulnerabilityDashboard")
        print("-" * 70)
        print("""
dashboard = VulnerabilityDashboard()
dashboard.update_summary({
    'total_vulnerabilities': 10,
    'code_vulnerabilities': 6,
    'dependency_vulnerabilities': 4,
    'critical': 2,
    'high': 3,
    'medium': 4,
    'low': 1
})
""")

        # Example 2: VulnerabilityList
        print("\n2. VulnerabilityList")
        print("-" * 70)
        print("""
vuln_list = VulnerabilityList()
vuln_list.update_vulnerabilities([
    {
        'id': 'vuln_1',
        'title': 'SQL Injection',
        'risk_level': 'critical',
        'type': 'injection',
        'file_path': 'app.py',
        'line_number': 42
    }
])
""")

        # Example 3: DependencyTree
        print("\n3. DependencyTree")
        print("-" * 70)
        print("""
dep_tree = DependencyTree()
dep_tree.update_dependencies([
    {
        'package_name': 'requests',
        'current_version': '2.20.0',
        'fixed_version': '2.31.0',
        'cve_id': 'CVE-2023-32681',
        'risk_level': 'medium'
    }
])
""")

        # Example 4: FixSuggestions
        print("\n4. FixSuggestions")
        print("-" * 70)
        print("""
fixes = FixSuggestions()
fixes.update_fix_suggestions({
    'title': 'SQL Injection',
    'type': 'injection',
    'risk_level': 'critical',
    'code_snippet': 'query = "SELECT * FROM users WHERE id = " + user_id',
    'fix_suggestion': 'Use parameterized queries',
    'references': ['https://owasp.org/...']
})
""")

        # Example 5: SecurityAuditorPanel
        print("\n5. SecurityAuditorPanel (Main Panel)")
        print("-" * 70)
        print("""
panel = SecurityAuditorPanel()

# Update components
panel.update_dashboard(summary)
panel.update_vulnerabilities(vulnerabilities)
panel.update_dependencies(dependencies)
panel.update_history(history)

# Keybindings:
# Ctrl+S - Scan
# Ctrl+D - Dependencies
# Ctrl+R - Report
# 1-5 - Switch tabs
""")

        print("\n✅ TUI components ready for integration!")

    except ImportError as e:
        print(f"\n⚠️  TUI components not available: {e}")
        print("Install textual: pip install textual")


async def demo_integration():
    """Demonstrate full integration"""

    print("\n\n" + "=" * 70)
    print("Full Integration Demo")
    print("=" * 70)

    print("\n📋 Integration Steps:")
    print("\n1. Enable the feature:")
    print("   xencode features enable security_auditor")

    print("\n2. Run security scan:")
    print("   xencode security scan .")

    print("\n3. Analyze dependencies:")
    print("   xencode security dependencies .")

    print("\n4. Generate report:")
    print("   xencode security report . --output security_report.md")

    print("\n5. Run full audit:")
    print("   xencode security audit .")

    print("\n📊 CI/CD Integration:")
    print("""
# .github/workflows/security.yml
name: Security Audit

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Xencode
        run: pip install xencode
      - name: Run Security Audit
        run: xencode security audit .
      - name: Generate Report
        run: xencode security report . --output security_report.md
""")

    print("\n🎯 Best Practices:")
    print("  • Run scans before commits")
    print("  • Integrate into CI/CD pipeline")
    print("  • Fix critical issues immediately")
    print("  • Keep dependencies updated")
    print("  • Review reports regularly")
    print("  • Document security decisions")


async def main():
    """Run all demos"""

    print("\n" + "=" * 70)
    print("SECURITY AUDITOR CLI AND TUI DEMO")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  • CLI commands for security scanning")
    print("  • TUI components for visual feedback")
    print("  • Integration examples")
    print("  • Best practices")

    # Run CLI demo
    await demo_cli_usage()

    # Run TUI demo
    await demo_tui_components()

    # Run integration demo
    await demo_integration()

    print("\n\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\n📚 Next Steps:")
    print("  1. Review the CLI/TUI guide: xencode/features/security_auditor/CLI_TUI_GUIDE.md")
    print("  2. Try the CLI commands: xencode security --help")
    print("  3. Explore the TUI: xencode (then navigate to Security Auditor)")
    print("  4. Integrate into your workflow")
    print("\n💡 Tips:")
    print("  • Install Bandit for Python scanning: pip install bandit")
    print("  • Install Snyk for dependency scanning: npm install -g snyk")
    print("  • Configure exclusion patterns for your project")
    print("  • Set up CI/CD integration for automated scanning")
    print("\n🔒 Stay secure!")


if __name__ == '__main__':
    asyncio.run(main())
