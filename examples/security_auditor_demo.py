"""
Security Auditor Demo

Demonstrates the security auditor feature for vulnerability scanning.
"""

import asyncio
from pathlib import Path

from xencode.features.base import FeatureConfig
from xencode.features.security_auditor import SecurityAuditor


async def main():
    """Run security auditor demo"""

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

    print("=" * 70)
    print("Security Auditor Demo")
    print("=" * 70)

    # Example 1: Scan a file
    print("\n1. Scanning a vulnerable file...")
    print("-" * 70)

    # Create a temporary vulnerable file for demo
    demo_file = Path("demo_vulnerable.py")
    demo_file.write_text("""
# Example vulnerable code
import os

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
password = "admin123"

# SQL injection vulnerability
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute(query)

# Command injection
def run_backup(path):
    os.system("tar -czf backup.tar.gz " + path)

# Weak cryptography
import hashlib
def hash_password(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()
""")

    try:
        result = await auditor.scan(str(demo_file), use_external_tools=False)

        print("✓ Scan complete!")
        print(f"  Total vulnerabilities: {result['summary']['total_vulnerabilities']}")
        print(f"  🔴 Critical: {result['summary']['critical']}")
        print(f"  🟠 High: {result['summary']['high']}")
        print(f"  🟡 Medium: {result['summary']['medium']}")
        print(f"  🟢 Low: {result['summary']['low']}")

        # Show first few vulnerabilities
        if result['report']['vulnerabilities']:
            print("\n  Top vulnerabilities:")
            for vuln in result['report']['vulnerabilities'][:3]:
                print(f"    - [{vuln['risk_level'].upper()}] {vuln['title']}")
                print(f"      Line {vuln['line_number']}: {vuln['code_snippet'][:50]}...")

    finally:
        # Clean up demo file
        if demo_file.exists():
            demo_file.unlink()

    # Example 2: Analyze dependencies
    print("\n2. Analyzing dependencies...")
    print("-" * 70)

    # Create a temporary requirements file
    req_file = Path("demo_requirements.txt")
    req_file.write_text("""
requests==2.20.0
django==3.0.0
flask==2.0.0
pyyaml==5.1
""")

    try:
        result = await auditor.analyze(str(req_file))

        print("✓ Analysis complete!")
        print(f"  Vulnerable dependencies: {result['count']}")

        if result['vulnerabilities']:
            print("\n  Vulnerable packages:")
            for vuln in result['vulnerabilities'][:3]:
                print(f"    - {vuln['package_name']} {vuln['current_version']}")
                print(f"      CVE: {vuln['cve_id']}")
                print(f"      Fix: Upgrade to {vuln['fixed_version']}")

    finally:
        # Clean up demo file
        if req_file.exists():
            req_file.unlink()

    # Example 3: Generate security report
    print("\n3. Generating security report...")
    print("-" * 70)

    # Create demo files again for report
    demo_file.write_text("""
password = "secret123"
api_key = "1234567890"
""")

    try:
        result = await auditor.report(str(demo_file), format='markdown')

        if result['success']:
            print("✓ Report generated!")
            print("\nReport preview (first 500 chars):")
            print(result['content'][:500])
            print("...")

            # Save report to file
            report_path = Path("security_report.md")
            report_path.write_text(result['content'], encoding='utf-8')
            print(f"\n✓ Full report saved to: {report_path}")

    finally:
        # Clean up demo file
        if demo_file.exists():
            demo_file.unlink()

    # Example 4: Check available security tools
    print("\n4. Available security tools...")
    print("-" * 70)

    tools = auditor.tool_integration.tools_available
    print(f"  Bandit: {'✓ Available' if tools.get('bandit') else '✗ Not installed'}")
    print(f"  Snyk: {'✓ Available' if tools.get('snyk') else '✗ Not installed'}")

    if not tools.get('bandit'):
        print("\n  To install Bandit: pip install bandit")
    if not tools.get('snyk'):
        print("  To install Snyk: npm install -g snyk")

    # Shutdown
    await auditor.shutdown()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("  • OWASP Top 10 vulnerability detection")
    print("  • Dependency CVE scanning")
    print("  • Integration with Bandit and Snyk")
    print("  • Markdown and HTML report generation")
    print("  • Configurable exclude patterns")
    print("  • Risk level classification")
    print("\nCLI Usage:")
    print("  xencode security scan <path>")
    print("  xencode security dependencies <path>")
    print("  xencode security report <path> --format markdown")
    print("  xencode security audit <path>")


if __name__ == '__main__':
    asyncio.run(main())
