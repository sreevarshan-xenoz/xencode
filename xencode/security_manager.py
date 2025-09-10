#!/usr/bin/env python3
"""
Security Manager for Xencode Phase 2 Integration

Provides comprehensive security protection including:
- AST-based Python code analysis
- Path sanitization and symlink detection
- Malicious content validation
- .git directory security layer
- Commit message sanitization

Requirements: 1.6, 1.7, 3.17, 3.18, 3.19
"""

import ast
import os
import re
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Security risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of security violations"""
    SYMLINK = "symlink"
    EXECUTABLE = "executable"
    SENSITIVE = "sensitive"
    OBFUSCATED = "obfuscated"
    PATH_TRAVERSAL = "path_traversal"
    GIT_EXPLOIT = "git_exploit"


@dataclass
class SecurityViolation:
    """Represents a security violation found during scanning"""
    file_path: str
    violation_type: ViolationType
    risk_level: RiskLevel
    description: str
    detected_patterns: List[str]


@dataclass
class SecurityReport:
    """Summary of security scan results"""
    symlinks: List[str]
    executables: List[str]
    sensitive_files: List[str]
    git_risks: List[str]
    total_excluded: int
    violations: List[SecurityViolation]


class SecurityManager:
    """
    Main security manager that coordinates all security validations
    """
    
    def __init__(self):
        self.dangerous_patterns = self._load_dangerous_patterns()
        self.sensitive_patterns = self._load_sensitive_patterns()
        self.executable_extensions = {'.exe', '.bat', '.cmd', '.sh', '.ps1', '.scr', '.com'}
        
    def _load_dangerous_patterns(self) -> List[str]:
        """Load patterns that indicate dangerous code"""
        return [
            r'__import__\s*\(\s*[\'"]os[\'"]',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'shell=True',
            r'rm\s+-rf',
            r'del\s+/[sq]',
            r'format\s*\(\s*[\'"].*\{.*\}',
            r'\.format\s*\(',
        ]
    
    def _load_sensitive_patterns(self) -> List[str]:
        """Load patterns that indicate sensitive content"""
        return [
            r'password\s*=',
            r'secret\s*=',
            r'secret_key\s*=',
            r'api_key\s*=',
            r'token\s*=',
            r'-----BEGIN.*KEY-----',
            r'-----BEGIN.*CERTIFICATE-----',
            r'ssh-rsa\s+',
            r'ssh-ed25519\s+',
        ]
    
    def validate_project_path(self, path: str) -> bool:
        """
        Validate that a project path is safe to scan
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve to absolute path
            abs_path = os.path.abspath(path)
            
            # Check if path exists
            if not os.path.exists(abs_path):
                return False
                
            # Check if it's a directory
            if not os.path.isdir(abs_path):
                return False
                
            # Check for path traversal attempts in relative paths
            if '..' in path:
                return False
                
            return True
            
        except (OSError, ValueError):
            return False
    
    def sanitize_file_path(self, file_path: str, project_root: str) -> Optional[str]:
        """
        Sanitize and validate a file path within project boundaries
        
        Args:
            file_path: File path to sanitize
            project_root: Root directory of the project
            
        Returns:
            Sanitized path if safe, None if unsafe
        """
        try:
            # Resolve both paths to absolute
            abs_file = os.path.abspath(file_path)
            abs_root = os.path.abspath(project_root)
            
            # Use realpath to resolve symlinks
            real_file = os.path.realpath(abs_file)
            real_root = os.path.realpath(abs_root)
            
            # Check if file is within project root
            if not real_file.startswith(real_root):
                return None
                
            # Check for suspicious patterns in path
            if self._has_suspicious_path_patterns(file_path):
                return None
                
            return abs_file
            
        except (OSError, ValueError):
            return None
    
    def _has_suspicious_path_patterns(self, path: str) -> bool:
        """Check for suspicious patterns in file paths"""
        suspicious_patterns = [
            r'\.\./',
            r'/etc/',
            r'/proc/',
            r'/sys/',
            r'~/',
            r'\$\{',
            r'%[A-Z_]+%',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, path):
                return True
        return False
    
    def detect_symlink_risks(self, file_path: str, project_root: str) -> Optional[SecurityViolation]:
        """
        Detect if a file is a symlink pointing outside project root
        
        Args:
            file_path: Path to check
            project_root: Project root directory
            
        Returns:
            SecurityViolation if risky symlink detected, None otherwise
        """
        try:
            if not os.path.islink(file_path):
                return None
                
            # Get the target of the symlink
            link_target = os.readlink(file_path)
            
            # Resolve to absolute path
            abs_target = os.path.abspath(os.path.join(os.path.dirname(file_path), link_target))
            abs_root = os.path.abspath(project_root)
            
            # Check if symlink points outside project
            if not abs_target.startswith(abs_root):
                return SecurityViolation(
                    file_path=file_path,
                    violation_type=ViolationType.SYMLINK,
                    risk_level=RiskLevel.HIGH,
                    description=f"Symlink points outside project root: {link_target}",
                    detected_patterns=[f"symlink -> {link_target}"]
                )
                
            return None
            
        except (OSError, ValueError):
            # If we can't read the symlink, treat it as suspicious
            return SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.SYMLINK,
                risk_level=RiskLevel.MEDIUM,
                description="Unreadable or broken symlink",
                detected_patterns=["broken_symlink"]
            )
    
    def validate_file_content(self, file_path: str, content: str) -> List[SecurityViolation]:
        """
        Validate file content for security risks
        
        Args:
            file_path: Path to the file
            content: File content to analyze
            
        Returns:
            List of security violations found
        """
        violations = []
        
        # Check for null bytes (binary files or injection attempts)
        if '\x00' in content:
            violations.append(SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.EXECUTABLE,
                risk_level=RiskLevel.HIGH,
                description="File contains null bytes (binary/executable content)",
                detected_patterns=["null_bytes"]
            ))
            return violations
        
        # Check for sensitive content patterns
        sensitive_matches = self._find_sensitive_patterns(content)
        if sensitive_matches:
            violations.append(SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.SENSITIVE,
                risk_level=RiskLevel.MEDIUM,
                description="File contains sensitive information patterns",
                detected_patterns=sensitive_matches
            ))
        
        # Check for dangerous code patterns
        dangerous_matches = self._find_dangerous_patterns(content)
        if dangerous_matches:
            violations.append(SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.EXECUTABLE,
                risk_level=RiskLevel.HIGH,
                description="File contains potentially dangerous code patterns",
                detected_patterns=dangerous_matches
            ))
        
        # Python-specific AST analysis
        if file_path.endswith('.py'):
            ast_violations = self.analyze_python_ast(file_path, content)
            violations.extend(ast_violations)
        
        return violations
    
    def _find_sensitive_patterns(self, content: str) -> List[str]:
        """Find sensitive information patterns in content"""
        matches = []
        content_lower = content.lower()
        
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                matches.append(pattern)
        
        return matches
    
    def _find_dangerous_patterns(self, content: str) -> List[str]:
        """Find dangerous code patterns in content"""
        matches = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(pattern)
        
        return matches
    
    def analyze_python_ast(self, file_path: str, content: str) -> List[SecurityViolation]:
        """
        Perform AST-based analysis of Python code for security risks
        
        Args:
            file_path: Path to the Python file
            content: Python code content
            
        Returns:
            List of security violations found
        """
        violations = []
        
        try:
            tree = ast.parse(content)
            
            # Walk through all nodes in the AST
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    violation = self._analyze_function_call(node, file_path)
                    if violation:
                        violations.append(violation)
                
                # Check for dangerous imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    violation = self._analyze_import(node, file_path)
                    if violation:
                        violations.append(violation)
                
                # Check for exec/eval usage
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, 'id') and node.value.func.id in ['exec', 'eval']:
                        violations.append(SecurityViolation(
                            file_path=file_path,
                            violation_type=ViolationType.EXECUTABLE,
                            risk_level=RiskLevel.CRITICAL,
                            description=f"Use of {node.value.func.id}() function detected",
                            detected_patterns=[f"{node.value.func.id}()"]
                        ))
        
        except SyntaxError as e:
            # Invalid syntax could indicate obfuscation
            violations.append(SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.OBFUSCATED,
                risk_level=RiskLevel.MEDIUM,
                description=f"Invalid Python syntax - potential obfuscation: {str(e)}",
                detected_patterns=["syntax_error"]
            ))
        
        except Exception as e:
            # Other parsing errors
            violations.append(SecurityViolation(
                file_path=file_path,
                violation_type=ViolationType.OBFUSCATED,
                risk_level=RiskLevel.LOW,
                description=f"AST parsing failed: {str(e)}",
                detected_patterns=["parse_error"]
            ))
        
        return violations
    
    def _analyze_function_call(self, node: ast.Call, file_path: str) -> Optional[SecurityViolation]:
        """Analyze a function call node for security risks"""
        try:
            # Get function name
            func_name = None
            if hasattr(node.func, 'id'):
                func_name = node.func.id
            elif hasattr(node.func, 'attr'):
                if hasattr(node.func.value, 'id'):
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                else:
                    func_name = node.func.attr
            
            if not func_name:
                return None
            
            # Check for dangerous functions
            dangerous_functions = [
                'exec', 'eval', 'compile',
                'os.system', 'os.popen', 'os.execv', 'os.execve',
                'subprocess.call', 'subprocess.run', 'subprocess.Popen',
                '__import__'
            ]
            
            if func_name in dangerous_functions:
                return SecurityViolation(
                    file_path=file_path,
                    violation_type=ViolationType.EXECUTABLE,
                    risk_level=RiskLevel.HIGH,
                    description=f"Dangerous function call: {func_name}",
                    detected_patterns=[func_name]
                )
            
            # Check for shell=True in subprocess calls
            if 'subprocess' in func_name:
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and hasattr(keyword.value, 'value') and keyword.value.value:
                        return SecurityViolation(
                            file_path=file_path,
                            violation_type=ViolationType.EXECUTABLE,
                            risk_level=RiskLevel.CRITICAL,
                            description="subprocess call with shell=True detected",
                            detected_patterns=["shell=True"]
                        )
            
            return None
            
        except Exception:
            return None
    
    def _analyze_import(self, node: ast.Import | ast.ImportFrom, file_path: str) -> Optional[SecurityViolation]:
        """Analyze import statements for security risks"""
        try:
            dangerous_modules = [
                'os', 'subprocess', 'sys', 'ctypes', 'marshal', 'pickle',
                'socket', 'urllib', 'http', 'ftplib', 'telnetlib'
            ]
            
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        return SecurityViolation(
                            file_path=file_path,
                            violation_type=ViolationType.EXECUTABLE,
                            risk_level=RiskLevel.MEDIUM,
                            description=f"Import of potentially dangerous module: {alias.name}",
                            detected_patterns=[f"import {alias.name}"]
                        )
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in dangerous_modules:
                    return SecurityViolation(
                        file_path=file_path,
                        violation_type=ViolationType.EXECUTABLE,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"Import from potentially dangerous module: {node.module}",
                        detected_patterns=[f"from {node.module}"]
                    )
            
            return None
            
        except Exception:
            return None
    
    def validate_git_directory(self, git_path: str) -> List[SecurityViolation]:
        """
        Validate .git directory for security risks
        
        Args:
            git_path: Path to .git directory
            
        Returns:
            List of security violations found
        """
        violations = []
        
        if not os.path.exists(git_path) or not os.path.isdir(git_path):
            return violations
        
        # Check for risky files in .git directory
        risky_git_files = [
            'COMMIT_EDITMSG',
            'hooks/pre-commit',
            'hooks/post-commit',
            'hooks/pre-push',
            'config'
        ]
        
        for risky_file in risky_git_files:
            file_path = os.path.join(git_path, risky_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for injection attempts in git files
                    if self._has_git_injection_patterns(content):
                        violations.append(SecurityViolation(
                            file_path=file_path,
                            violation_type=ViolationType.GIT_EXPLOIT,
                            risk_level=RiskLevel.HIGH,
                            description=f"Potential injection in git file: {risky_file}",
                            detected_patterns=["git_injection"]
                        ))
                
                except (OSError, UnicodeDecodeError):
                    violations.append(SecurityViolation(
                        file_path=file_path,
                        violation_type=ViolationType.GIT_EXPLOIT,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"Unreadable git file: {risky_file}",
                        detected_patterns=["unreadable_git_file"]
                    ))
        
        return violations
    
    def _has_git_injection_patterns(self, content: str) -> bool:
        """Check for git injection patterns in content"""
        injection_patterns = [
            r'\$\{.*\}',  # Variable expansion
            r'`.*`',      # Command substitution
            r'\$\(.*\)',  # Command substitution
            r'rm\s+-rf',  # Dangerous commands
            r'curl\s+',   # Network access
            r'wget\s+',   # Network access
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def sanitize_commit_message(self, message: str) -> str:
        """
        Sanitize git commit message to prevent injection attacks
        
        Args:
            message: Raw commit message
            
        Returns:
            Sanitized commit message
        """
        if not message:
            return ""
        
        # Remove null bytes
        message = message.replace('\x00', '')
        
        # Remove or escape dangerous patterns
        dangerous_patterns = [
            (r'\$\{.*?\}', '${SANITIZED}'),  # Variable expansion
            (r'`([^`]*)`', r'"\1"'),         # Command substitution
            (r'\$\([^)]*\)', '$(SANITIZED)'), # Command substitution
            (r'[;&|><]', ' '),               # Shell operators
        ]
        
        for pattern, replacement in dangerous_patterns:
            message = re.sub(pattern, replacement, message)
        
        # Limit length
        if len(message) > 500:
            message = message[:497] + "..."
        
        # Ensure it's valid UTF-8
        try:
            message.encode('utf-8')
        except UnicodeEncodeError:
            message = message.encode('utf-8', errors='replace').decode('utf-8')
        
        return message.strip()
    
    def scan_for_security_risks(self, files: List[str], project_root: str) -> SecurityReport:
        """
        Perform comprehensive security scan of files
        
        Args:
            files: List of file paths to scan
            project_root: Root directory of the project
            
        Returns:
            SecurityReport with all findings
        """
        report = SecurityReport(
            symlinks=[],
            executables=[],
            sensitive_files=[],
            git_risks=[],
            total_excluded=0,
            violations=[]
        )
        
        for file_path in files:
            try:
                # Skip if file doesn't exist
                if not os.path.exists(file_path):
                    continue
                
                # Check for symlink risks
                symlink_violation = self.detect_symlink_risks(file_path, project_root)
                if symlink_violation:
                    report.violations.append(symlink_violation)
                    report.symlinks.append(file_path)
                    report.total_excluded += 1
                    continue
                
                # Check file extension for executables
                if any(file_path.lower().endswith(ext) for ext in self.executable_extensions):
                    report.executables.append(file_path)
                    report.total_excluded += 1
                    continue
                
                # Check for sensitive filenames
                filename = os.path.basename(file_path).lower()
                if any(pattern in filename for pattern in ['.env', 'secret', 'password', '.key', '.pem']):
                    report.sensitive_files.append(file_path)
                    report.total_excluded += 1
                    continue
                
                # Read and analyze file content
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(10000)  # Limit read size for performance
                    
                    # Validate content
                    content_violations = self.validate_file_content(file_path, content)
                    if content_violations:
                        report.violations.extend(content_violations)
                        
                        # Categorize violations
                        for violation in content_violations:
                            if violation.violation_type == ViolationType.SENSITIVE:
                                report.sensitive_files.append(file_path)
                                report.total_excluded += 1
                            elif violation.violation_type in [ViolationType.EXECUTABLE, ViolationType.OBFUSCATED]:
                                report.executables.append(file_path)
                                report.total_excluded += 1
                
                except (OSError, UnicodeDecodeError):
                    # Treat unreadable files as potentially risky
                    report.executables.append(file_path)
                    report.total_excluded += 1
            
            except Exception as e:
                # Log error but continue scanning
                continue
        
        # Check .git directory if present
        git_path = os.path.join(project_root, '.git')
        if os.path.exists(git_path):
            git_violations = self.validate_git_directory(git_path)
            if git_violations:
                report.violations.extend(git_violations)
                report.git_risks = [v.file_path for v in git_violations]
        
        return report
    
    def generate_security_summary(self, report: SecurityReport) -> str:
        """
        Generate a human-readable security summary
        
        Args:
            report: SecurityReport to summarize
            
        Returns:
            Formatted security summary string
        """
        if report.total_excluded == 0 and not report.violations:
            return "✅ Security scan complete: No risks detected"
        
        summary_parts = []
        
        if report.total_excluded > 0:
            summary_parts.append(f"❌ Security alert: Excluded {report.total_excluded} risky files")
            
            details = []
            if report.symlinks:
                details.append(f"{len(report.symlinks)} symlinks")
            if report.executables:
                details.append(f"{len(report.executables)} executables")
            if report.sensitive_files:
                details.append(f"{len(report.sensitive_files)} sensitive")
            if report.git_risks:
                details.append(f"{len(report.git_risks)} git risks")
            
            if details:
                summary_parts.append(f"({', '.join(details)})")
        
        # Add high-risk violation details
        critical_violations = [v for v in report.violations if v.risk_level == RiskLevel.CRITICAL]
        if critical_violations:
            summary_parts.append(f"🚨 {len(critical_violations)} critical security issues found")
        
        return " ".join(summary_parts)
    
    def is_safe_to_scan(self, file_path: str, project_root: str) -> bool:
        """
        Quick check if a file is safe to scan
        
        Args:
            file_path: File to check
            project_root: Project root directory
            
        Returns:
            True if safe to scan, False otherwise
        """
        # Sanitize path
        sanitized_path = self.sanitize_file_path(file_path, project_root)
        if not sanitized_path:
            return False
        
        # Check for symlinks
        if os.path.islink(file_path):
            violation = self.detect_symlink_risks(file_path, project_root)
            return violation is None
        
        # Check file extension
        if any(file_path.lower().endswith(ext) for ext in self.executable_extensions):
            return False
        
        return True


def main():
    """Demo/test function for SecurityManager"""
    security_manager = SecurityManager()
    
    # Test path validation
    print("Testing path validation:")
    test_paths = [
        "/tmp/test",
        "../etc/passwd",
        "normal/path",
        "/etc/passwd"
    ]
    
    for path in test_paths:
        is_valid = security_manager.validate_project_path(path)
        print(f"  {path}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    print("\nTesting commit message sanitization:")
    test_messages = [
        "Normal commit message",
        "Fix bug with $(rm -rf /)",
        "Add feature `curl evil.com`",
        "Update ${HOME}/.bashrc"
    ]
    
    for message in test_messages:
        sanitized = security_manager.sanitize_commit_message(message)
        print(f"  Original: {message}")
        print(f"  Sanitized: {sanitized}")
        print()


if __name__ == "__main__":
    main()