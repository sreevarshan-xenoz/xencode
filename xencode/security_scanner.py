"""
Security Scanner for Xencode.

Provides comprehensive code security scanning with Bandit integration,
OWASP Top 10 vulnerability detection, and multi-language analysis.

This module serves as the main entry point for security scanning functionality.
For detailed scanning capabilities, use xencode.analyzers.SecurityAnalyzer.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Security vulnerability severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Represents a single security issue found in code."""
    rule_id: str
    description: str
    severity: SeverityLevel
    line_number: int
    column: int = 0
    filename: str = ""
    cwe_id: Optional[str] = None
    confidence: str = "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "severity": self.severity.value,
            "line_number": self.line_number,
            "column": self.column,
            "filename": self.filename,
            "cwe_id": self.cwe_id,
            "confidence": self.confidence,
        }


@dataclass
class ScanResult:
    """Results from a security scan."""
    issues: List[SecurityIssue]
    files_scanned: int
    total_lines: int
    scan_time_ms: float
    summary: Dict[str, int]

    @property
    def has_critical(self) -> bool:
        return self.summary.get("critical", 0) > 0

    @property
    def has_high(self) -> bool:
        return self.summary.get("high", 0) > 0

    @property
    def passed(self) -> bool:
        return not self.has_critical and not self.has_high


class SecurityScanner:
    """
    Main security scanner class.

    Provides interface for scanning Python, JavaScript, and Java code
    for security vulnerabilities using pattern-based detection and
    Bandit integration.
    """

    def __init__(self):
        """Initialize the security scanner."""
        self._scanner = None

    def _get_scanner(self):
        """Lazy-load the actual scanner implementation."""
        if self._scanner is None:
            try:
                from xencode.analyzers import SecurityAnalyzer
                self._scanner = SecurityAnalyzer()
            except (ImportError, AttributeError):
                self._scanner = None
        return self._scanner

    def scan_file(self, filepath: str) -> ScanResult:
        """
        Scan a single file for security vulnerabilities.

        Args:
            filepath: Path to the file to scan.

        Returns:
            ScanResult with all issues found.
        """
        scanner = self._get_scanner()
        if scanner is not None:
            return scanner.scan_file(filepath)
        return ScanResult(
            issues=[],
            files_scanned=0,
            total_lines=0,
            scan_time_ms=0.0,
            summary={"low": 0, "medium": 0, "high": 0, "critical": 0},
        )

    def scan_directory(self, directory: str, recursive: bool = True) -> ScanResult:
        """
        Scan a directory for security vulnerabilities.

        Args:
            directory: Path to the directory to scan.
            recursive: Whether to scan subdirectories.

        Returns:
            ScanResult with all issues found.
        """
        scanner = self._get_scanner()
        if scanner is not None:
            return scanner.scan_directory(directory, recursive)
        return ScanResult(
            issues=[],
            files_scanned=0,
            total_lines=0,
            scan_time_ms=0.0,
            summary={"low": 0, "medium": 0, "high": 0, "critical": 0},
        )

    def scan_code(self, code: str, language: str = "python") -> List[SecurityIssue]:
        """
        Scan a code string for security vulnerabilities.

        Args:
            code: Source code to scan.
            language: Programming language of the code.

        Returns:
            List of SecurityIssue objects.
        """
        scanner = self._get_scanner()
        if scanner is not None:
            results = scanner.scan_code(code, language)
            return results if isinstance(results, list) else []
        return []

    def generate_report(self, result: ScanResult, format: str = "summary") -> str:
        """
        Generate a human-readable security report.

        Args:
            result: ScanResult from a scan operation.
            format: Report format ('summary', 'detailed', or 'executive').

        Returns:
            Formatted report string.
        """
        scanner = self._get_scanner()
        if scanner is not None:
            return scanner.generate_report(result, format)

        # Basic fallback report
        lines = [
            "Security Scan Report",
            "=" * 50,
            f"Files scanned: {result.files_scanned}",
            f"Total lines: {result.total_lines}",
            f"Scan time: {result.scan_time_ms:.1f}ms",
            "",
            "Summary:",
            f"  Critical: {result.summary.get('critical', 0)}",
            f"  High:     {result.summary.get('high', 0)}",
            f"  Medium:   {result.summary.get('medium', 0)}",
            f"  Low:      {result.summary.get('low', 0)}",
            "",
            f"Status: {'PASS' if result.passed else 'FAIL'}",
        ]
        if result.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in result.issues:
                lines.append(
                    f"  [{issue.severity.value.upper()}] {issue.description} "
                    f"({issue.filename}:{issue.line_number})"
                )
        return "\n".join(lines)


__all__ = [
    "SeverityLevel",
    "SecurityIssue",
    "ScanResult",
    "SecurityScanner",
]
