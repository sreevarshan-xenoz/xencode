#!/usr/bin/env python3
"""
Failure Analysis Engine

Analyzes test failures, classifies them by type, suggests fixes,
and supports automatic retry with fixes.

Features:
- Classify test failures (assertion, error, timeout, etc.)
- Suggest fixes based on failure type
- Auto-retry with fixes
- Failure pattern recognition
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .test_runner import TestExecutionResult, TestResult, TestStatus


class FailureType(Enum):
    """Types of test failures"""
    ASSERTION_ERROR = "assertion_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    ZERO_DIVISION_ERROR = "zero_division_error"
    FILE_NOT_FOUND_ERROR = "file_not_found_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    RECURSION_ERROR = "recursion_error"
    SYNTAX_ERROR = "syntax_error"
    INDENTATION_ERROR = "indentation_error"
    MOCK_ERROR = "mock_error"
    FIXTURE_ERROR = "fixture_error"
    UNKNOWN_ERROR = "unknown_error"


class FailureSeverity(Enum):
    """Severity levels for failures"""
    CRITICAL = "critical"  # Test infrastructure issues
    HIGH = "high"  # Test logic errors
    MEDIUM = "medium"  # Assertion failures
    LOW = "low"  # Minor issues, warnings


class FixStrategy(Enum):
    """Available fix strategies"""
    RETRY = "retry"
    UPDATE_ASSERTION = "update_assertion"
    FIX_MOCK = "fix_mock"
    FIX_FIXTURE = "fix_fixture"
    UPDATE_EXPECTED_VALUE = "update_expected_value"
    ADD_ERROR_HANDLING = "add_error_handling"
    FIX_IMPORT = "fix_import"
    FIX_SYNTAX = "fix_syntax"
    INCREASE_TIMEOUT = "increase_timeout"
    SKIP_TEST = "skip_test"
    MANUAL_REVIEW = "manual_review"


@dataclass
class FailureInfo:
    """Detailed information about a test failure"""
    failure_id: str
    test_result: TestResult
    failure_type: FailureType
    severity: FailureSeverity
    error_message: str
    traceback: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FixSuggestion:
    """Suggested fix for a failure"""
    suggestion_id: str
    failure_id: str
    strategy: FixStrategy
    description: str
    confidence: float  # 0.0 to 1.0
    code_change: Optional[str] = None
    explanation: str = ""
    auto_applicable: bool = False
    requires_review: bool = True


@dataclass
class FailurePattern:
    """Recognized failure pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    failure_type: FailureType
    regex_pattern: str
    fix_strategy: FixStrategy
    occurrence_count: int = 0
    last_seen: Optional[datetime] = None


@dataclass
class AnalysisResult:
    """Result of failure analysis"""
    analysis_id: str
    execution_result: TestExecutionResult
    failures: List[FailureInfo]
    suggestions: List[FixSuggestion]
    patterns_detected: List[FailurePattern]
    summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class FailureClassifier:
    """Classifies test failures by type"""

    # Error type patterns
    ERROR_PATTERNS: Dict[FailureType, str] = {
        FailureType.ASSERTION_ERROR: r"(AssertionError|assert\s|assert\s+.*==)",
        FailureType.TYPE_ERROR: r"TypeError:",
        FailureType.VALUE_ERROR: r"ValueError:",
        FailureType.ATTRIBUTE_ERROR: r"AttributeError:",
        FailureType.IMPORT_ERROR: r"(ImportError|ModuleNotFoundError):",
        FailureType.NAME_ERROR: r"NameError:",
        FailureType.KEY_ERROR: r"KeyError:",
        FailureType.INDEX_ERROR: r"IndexError:",
        FailureType.ZERO_DIVISION_ERROR: r"ZeroDivisionError:",
        FailureType.FILE_NOT_FOUND_ERROR: r"FileNotFoundError:",
        FailureType.TIMEOUT_ERROR: r"(TimeoutError|timeout|timed out)",
        FailureType.MEMORY_ERROR: r"MemoryError:",
        FailureType.RECURSION_ERROR: r"RecursionError:",
        FailureType.SYNTAX_ERROR: r"SyntaxError:",
        FailureType.INDENTATION_ERROR: r"IndentationError:",
        FailureType.MOCK_ERROR: r"(Mock|magicmock|patch)",
        FailureType.FIXTURE_ERROR: r"(fixture|@pytest\.fixture)",
    }

    # Severity mappings
    SEVERITY_MAP: Dict[FailureType, FailureSeverity] = {
        FailureType.ASSERTION_ERROR: FailureSeverity.MEDIUM,
        FailureType.TYPE_ERROR: FailureSeverity.HIGH,
        FailureType.VALUE_ERROR: FailureSeverity.MEDIUM,
        FailureType.ATTRIBUTE_ERROR: FailureSeverity.HIGH,
        FailureType.IMPORT_ERROR: FailureSeverity.CRITICAL,
        FailureType.NAME_ERROR: FailureSeverity.HIGH,
        FailureType.KEY_ERROR: FailureSeverity.MEDIUM,
        FailureType.INDEX_ERROR: FailureSeverity.MEDIUM,
        FailureType.ZERO_DIVISION_ERROR: FailureSeverity.MEDIUM,
        FailureType.FILE_NOT_FOUND_ERROR: FailureSeverity.HIGH,
        FailureType.TIMEOUT_ERROR: FailureSeverity.HIGH,
        FailureType.MEMORY_ERROR: FailureSeverity.CRITICAL,
        FailureType.RECURSION_ERROR: FailureSeverity.HIGH,
        FailureType.SYNTAX_ERROR: FailureSeverity.CRITICAL,
        FailureType.INDENTATION_ERROR: FailureSeverity.CRITICAL,
        FailureType.MOCK_ERROR: FailureSeverity.HIGH,
        FailureType.FIXTURE_ERROR: FailureSeverity.HIGH,
        FailureType.UNKNOWN_ERROR: FailureSeverity.MEDIUM,
    }

    def classify(self, error_message: str, traceback: str = "") -> Tuple[FailureType, FailureSeverity]:
        """Classify a failure by type and severity"""
        combined_text = f"{error_message}\n{traceback}"

        for failure_type, pattern in self.ERROR_PATTERNS.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                severity = self.SEVERITY_MAP.get(failure_type, FailureSeverity.MEDIUM)
                return failure_type, severity

        return FailureType.UNKNOWN_ERROR, FailureSeverity.MEDIUM

    def extract_line_number(self, traceback: str) -> Optional[int]:
        """Extract line number from traceback"""
        # Pattern: File "path", line 42, in function
        match = re.search(r'line\s+(\d+)', traceback)
        if match:
            return int(match.group(1))
        return None

    def extract_file_path(self, traceback: str) -> Optional[str]:
        """Extract file path from traceback"""
        # Pattern: File "path/to/file.py", line 42
        match = re.search(r'File\s+"([^"]+)"', traceback)
        if match:
            return match.group(1)
        return None

    def extract_function_name(self, traceback: str) -> Optional[str]:
        """Extract function name from traceback"""
        # Pattern: in function_name
        match = re.search(r'in\s+(\w+)', traceback)
        if match:
            return match.group(1)
        return None


class FixSuggester:
    """Suggests fixes for test failures"""

    def __init__(self):
        self.classifier = FailureClassifier()

    def suggest(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Generate fix suggestions for a failure"""
        suggestions = []

        # Get suggestions based on failure type
        if failure.failure_type == FailureType.ASSERTION_ERROR:
            suggestions.extend(self._suggest_assertion_fixes(failure, test_code))
        elif failure.failure_type == FailureType.TYPE_ERROR:
            suggestions.extend(self._suggest_type_error_fixes(failure, test_code))
        elif failure.failure_type == FailureType.IMPORT_ERROR:
            suggestions.extend(self._suggest_import_fixes(failure, test_code))
        elif failure.failure_type == FailureType.MOCK_ERROR:
            suggestions.extend(self._suggest_mock_fixes(failure, test_code))
        elif failure.failure_type == FailureType.FIXTURE_ERROR:
            suggestions.extend(self._suggest_fixture_fixes(failure, test_code))
        elif failure.failure_type == FailureType.TIMEOUT_ERROR:
            suggestions.extend(self._suggest_timeout_fixes(failure, test_code))
        elif failure.failure_type in (FailureType.SYNTAX_ERROR, FailureType.INDENTATION_ERROR):
            suggestions.extend(self._suggest_syntax_fixes(failure, test_code))
        else:
            suggestions.append(self._suggest_manual_review(failure))

        return suggestions

    def _suggest_assertion_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for assertion errors"""
        suggestions = []
        error_msg = failure.error_message

        # Check for common assertion patterns
        if "assert" in error_msg.lower():
            # Extract actual and expected values if possible
            match = re.search(r'assert\s+(.+)\s+==\s+(.+)', error_msg)
            if match:
                actual = match.group(1).strip()
                expected = match.group(2).strip()

                suggestions.append(FixSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    failure_id=failure.failure_id,
                    strategy=FixStrategy.UPDATE_EXPECTED_VALUE,
                    description=f"Update expected value from {expected} to match actual {actual}",
                    confidence=0.7,
                    explanation="The assertion failed because the actual value differs from expected",
                    auto_applicable=False,
                    requires_review=True
                ))

            suggestions.append(FixSuggestion(
                suggestion_id=str(uuid.uuid4()),
                failure_id=failure.failure_id,
                strategy=FixStrategy.UPDATE_ASSERTION,
                description="Review and update the assertion logic",
                confidence=0.8,
                explanation="The assertion may need to be updated based on the actual behavior",
                auto_applicable=False,
                requires_review=True
            ))

        # Retry suggestion
        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.RETRY,
            description="Retry the test (may be a flaky test)",
            confidence=0.5,
            explanation="Some tests fail intermittently due to timing or race conditions",
            auto_applicable=True,
            requires_review=False
        ))

        return suggestions

    def _suggest_type_error_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for type errors"""
        suggestions = []
        error_msg = failure.error_message

        # Extract type information
        match = re.search(r"'(\w+)' object", error_msg)
        if match:
            actual_type = match.group(1)
            suggestions.append(FixSuggestion(
                suggestion_id=str(uuid.uuid4()),
                failure_id=failure.failure_id,
                strategy=FixStrategy.UPDATE_ASSERTION,
                description=f"Fix type mismatch - got {actual_type}",
                confidence=0.8,
                explanation=f"The code received an unexpected type: {actual_type}",
                auto_applicable=False,
                requires_review=True
            ))

        # Add error handling suggestion
        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.ADD_ERROR_HANDLING,
            description="Add type checking or conversion before the operation",
            confidence=0.6,
            explanation="Adding type validation can prevent type errors",
            auto_applicable=False,
            requires_review=True
        ))

        return suggestions

    def _suggest_import_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for import errors"""
        suggestions = []
        error_msg = failure.error_message

        # Extract missing module name
        match = re.search(r"No module named ['\"]?(\w+)['\"]?", error_msg)
        if match:
            module_name = match.group(1)
            suggestions.append(FixSuggestion(
                suggestion_id=str(uuid.uuid4()),
                failure_id=failure.failure_id,
                strategy=FixStrategy.FIX_IMPORT,
                description=f"Install or import the missing module: {module_name}",
                confidence=0.9,
                explanation=f"The module '{module_name}' is not installed or not in PYTHONPATH",
                auto_applicable=False,
                requires_review=True
            ))

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.FIX_IMPORT,
            description="Check import statements and module paths",
            confidence=0.8,
            explanation="Import errors often indicate missing dependencies or incorrect paths",
            auto_applicable=False,
            requires_review=True
        ))

        return suggestions

    def _suggest_mock_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for mock-related errors"""
        suggestions = []

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.FIX_MOCK,
            description="Review mock setup and patch paths",
            confidence=0.7,
            explanation="Mock errors often indicate incorrect patch paths or missing return values",
            auto_applicable=False,
            requires_review=True
        ))

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.RETRY,
            description="Retry with different mock configuration",
            confidence=0.5,
            explanation="Mock behavior may need adjustment",
            auto_applicable=True,
            requires_review=False
        ))

        return suggestions

    def _suggest_fixture_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for fixture errors"""
        suggestions = []

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.FIX_FIXTURE,
            description="Check fixture definition and scope",
            confidence=0.8,
            explanation="Fixture errors indicate issues with test setup",
            auto_applicable=False,
            requires_review=True
        ))

        return suggestions

    def _suggest_timeout_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for timeout errors"""
        suggestions = []

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.INCREASE_TIMEOUT,
            description="Increase test timeout threshold",
            confidence=0.6,
            explanation="The test may need more time to complete",
            auto_applicable=True,
            requires_review=True
        ))

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.ADD_ERROR_HANDLING,
            description="Add timeout handling or optimize test code",
            confidence=0.5,
            explanation="Consider optimizing slow operations or adding proper timeout handling",
            auto_applicable=False,
            requires_review=True
        ))

        return suggestions

    def _suggest_syntax_fixes(
        self,
        failure: FailureInfo,
        test_code: Optional[str] = None
    ) -> List[FixSuggestion]:
        """Suggest fixes for syntax errors"""
        suggestions = []

        suggestions.append(FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.FIX_SYNTAX,
            description="Fix syntax error at the indicated location",
            confidence=0.9,
            explanation="Syntax errors must be fixed before tests can run",
            auto_applicable=False,
            requires_review=True
        ))

        return suggestions

    def _suggest_manual_review(self, failure: FailureInfo) -> FixSuggestion:
        """Suggest manual review for unknown errors"""
        return FixSuggestion(
            suggestion_id=str(uuid.uuid4()),
            failure_id=failure.failure_id,
            strategy=FixStrategy.MANUAL_REVIEW,
            description="Manual review required for this failure",
            confidence=0.3,
            explanation="This failure type requires human analysis",
            auto_applicable=False,
            requires_review=True
        )


class FailurePatternRecognizer:
    """Recognizes patterns in test failures"""

    def __init__(self):
        self.patterns: List[FailurePattern] = self._initialize_patterns()
        self.failure_history: List[FailureInfo] = []

    def _initialize_patterns(self) -> List[FailurePattern]:
        """Initialize known failure patterns"""
        return [
            FailurePattern(
                pattern_id="P001",
                pattern_name="Flaky Test",
                description="Test passes and fails intermittently",
                failure_type=FailureType.ASSERTION_ERROR,
                regex_pattern=r"(intermittent|flaky|sometimes|random)",
                fix_strategy=FixStrategy.RETRY
            ),
            FailurePattern(
                pattern_id="P002",
                pattern_name="Missing Dependency",
                description="Required module or package is not installed",
                failure_type=FailureType.IMPORT_ERROR,
                regex_pattern=r"No module named|cannot import name",
                fix_strategy=FixStrategy.FIX_IMPORT
            ),
            FailurePattern(
                pattern_id="P003",
                pattern_name="Mock Configuration Error",
                description="Mock is not properly configured",
                failure_type=FailureType.MOCK_ERROR,
                regex_pattern=r"expected.*called|not.*called",
                fix_strategy=FixStrategy.FIX_MOCK
            ),
            FailurePattern(
                pattern_id="P004",
                pattern_name="Fixture Scope Issue",
                description="Fixture scope is incorrect for the test",
                failure_type=FailureType.FIXTURE_ERROR,
                regex_pattern=r"fixture.*not found|scope",
                fix_strategy=FixStrategy.FIX_FIXTURE
            ),
            FailurePattern(
                pattern_id="P005",
                pattern_name="Type Mismatch",
                description="Function receives unexpected type",
                failure_type=FailureType.TYPE_ERROR,
                regex_pattern=r"expected.*got|cannot.*type",
                fix_strategy=FixStrategy.UPDATE_ASSERTION
            ),
            FailurePattern(
                pattern_id="P006",
                pattern_name="Timeout Pattern",
                description="Test consistently times out",
                failure_type=FailureType.TIMEOUT_ERROR,
                regex_pattern=r"timeout|timed out|exceeded",
                fix_strategy=FixStrategy.INCREASE_TIMEOUT
            ),
        ]

    def recognize(self, failure: FailureInfo) -> List[FailurePattern]:
        """Recognize patterns in a failure"""
        matched_patterns = []
        combined_text = f"{failure.error_message}\n{failure.traceback}"

        for pattern in self.patterns:
            if re.search(pattern.regex_pattern, combined_text, re.IGNORECASE):
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now()
                matched_patterns.append(pattern)

        # Add to history
        self.failure_history.append(failure)

        return matched_patterns

    def get_recurring_patterns(self, min_occurrences: int = 3) -> List[FailurePattern]:
        """Get patterns that occur frequently"""
        return [p for p in self.patterns if p.occurrence_count >= min_occurrences]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns"""
        return {
            "total_patterns": len(self.patterns),
            "total_failures_analyzed": len(self.failure_history),
            "recurring_patterns": [
                {
                    "name": p.pattern_name,
                    "count": p.occurrence_count,
                    "last_seen": p.last_seen.isoformat() if p.last_seen else None
                }
                for p in self.get_recurring_patterns()
            ]
        }


class FailureAnalyzer:
    """Main failure analysis engine"""

    def __init__(self):
        self.classifier = FailureClassifier()
        self.suggester = FixSuggester()
        self.pattern_recognizer = FailurePatternRecognizer()
        self.analysis_history: List[AnalysisResult] = []

    def analyze(
        self,
        execution_result: TestExecutionResult,
        test_codes: Optional[Dict[str, str]] = None
    ) -> AnalysisResult:
        """Analyze all failures in an execution result"""
        analysis_id = str(uuid.uuid4())
        failures = []
        suggestions = []
        patterns_detected = []

        # Analyze each failed test
        for test_result in execution_result.test_results:
            if test_result.status in (TestStatus.FAILED, TestStatus.ERROR):
                # Create failure info
                failure_type, severity = self.classifier.classify(
                    test_result.error_message or "",
                    test_result.error_traceback or ""
                )

                failure = FailureInfo(
                    failure_id=str(uuid.uuid4()),
                    test_result=test_result,
                    failure_type=failure_type,
                    severity=severity,
                    error_message=test_result.error_message or "",
                    traceback=test_result.error_traceback or "",
                    line_number=self.classifier.extract_line_number(test_result.error_traceback or ""),
                    file_path=self.classifier.extract_file_path(test_result.error_traceback or ""),
                    function_name=self.classifier.extract_function_name(test_result.error_traceback or ""),
                    context={
                        "duration_ms": test_result.duration_ms,
                        "retry_count": test_result.retry_count
                    }
                )
                failures.append(failure)

                # Get fix suggestions
                test_code = test_codes.get(test_result.test_file) if test_codes else None
                test_suggestions = self.suggester.suggest(failure, test_code)
                suggestions.extend(test_suggestions)

                # Recognize patterns
                patterns = self.pattern_recognizer.recognize(failure)
                patterns_detected.extend(patterns)

        # Create summary
        summary = self._create_summary(execution_result, failures, suggestions)

        # Create analysis result
        result = AnalysisResult(
            analysis_id=analysis_id,
            execution_result=execution_result,
            failures=failures,
            suggestions=suggestions,
            patterns_detected=patterns_detected,
            summary=summary
        )

        self.analysis_history.append(result)

        return result

    def _create_summary(
        self,
        execution_result: TestExecutionResult,
        failures: List[FailureInfo],
        suggestions: List[FixSuggestion]
    ) -> Dict[str, Any]:
        """Create analysis summary"""
        failure_type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}

        for failure in failures:
            type_key = failure.failure_type.value
            failure_type_counts[type_key] = failure_type_counts.get(type_key, 0) + 1

            severity_key = failure.severity.value
            severity_counts[severity_key] = severity_counts.get(severity_key, 0) + 1

        auto_fixable = sum(1 for s in suggestions if s.auto_applicable)

        return {
            "total_failures": len(failures),
            "failure_types": failure_type_counts,
            "severity_distribution": severity_counts,
            "total_suggestions": len(suggestions),
            "auto_fixable_count": auto_fixable,
            "patterns_detected": len({p.pattern_id for p in self.pattern_recognizer.patterns if p.occurrence_count > 0}),
            "critical_failures": severity_counts.get("critical", 0),
            "high_severity_failures": severity_counts.get("high", 0)
        }

    def get_auto_retry_suggestions(self, analysis_result: AnalysisResult) -> List[FixSuggestion]:
        """Get suggestions that can be auto-applied"""
        return [s for s in analysis_result.suggestions if s.auto_applicable]

    def get_critical_failures(self, analysis_result: AnalysisResult) -> List[FailureInfo]:
        """Get critical severity failures"""
        return [f for f in analysis_result.failures if f.severity == FailureSeverity.CRITICAL]


class AutoRetryEngine:
    """Automatically retries tests with fixes"""

    def __init__(
        self,
        analyzer: Optional[FailureAnalyzer] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0
    ):
        self.analyzer = analyzer or FailureAnalyzer()
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_history: List[Dict[str, Any]] = []

    def retry_with_fixes(
        self,
        execution_result: TestExecutionResult,
        apply_fix_fn: Callable[[FixSuggestion], bool],
        run_tests_fn: Callable[[], TestExecutionResult]
    ) -> Dict[str, Any]:
        """
        Retry failed tests with automatic fixes.

        Args:
            execution_result: Original execution result
            apply_fix_fn: Function to apply a fix suggestion
            run_tests_fn: Function to re-run tests

        Returns:
            Dictionary with retry results
        """
        results = {
            "success": False,
            "retries_attempted": 0,
            "fixes_applied": 0,
            "final_result": None,
            "history": []
        }

        current_result = execution_result

        for retry_num in range(self.max_retries):
            # Analyze failures
            analysis = self.analyzer.analyze(current_result)

            # Get auto-fixable suggestions
            auto_fixes = self.analyzer.get_auto_retry_suggestions(analysis)

            if not auto_fixes:
                results["history"].append({
                    "retry": retry_num + 1,
                    "status": "no_auto_fixes",
                    "message": "No automatic fixes available"
                })
                break

            # Apply fixes
            fixes_applied = 0
            for suggestion in auto_fixes:
                if apply_fix_fn(suggestion):
                    fixes_applied += 1

            if fixes_applied == 0:
                results["history"].append({
                    "retry": retry_num + 1,
                    "status": "no_fixes_applied",
                    "message": "Could not apply any fixes"
                })
                break

            results["fixes_applied"] += fixes_applied

            # Wait before retry
            import time
            time.sleep(self.retry_delay_seconds)

            # Re-run tests
            current_result = run_tests_fn()

            results["retries_attempted"] += 1
            results["history"].append({
                "retry": retry_num + 1,
                "status": "completed",
                "passed": current_result.passed,
                "failed": current_result.failed,
                "fixes_applied": fixes_applied
            })

            # Check if all tests passed
            if current_result.failed == 0 and current_result.errors == 0:
                results["success"] = True
                break

        results["final_result"] = {
            "passed": current_result.passed,
            "failed": current_result.failed,
            "errors": current_result.errors,
            "skipped": current_result.skipped
        }

        self.retry_history.append(results)

        return results


def create_failure_analyzer() -> FailureAnalyzer:
    """Factory function to create a failure analyzer"""
    return FailureAnalyzer()


def create_auto_retry_engine(
    analyzer: Optional[FailureAnalyzer] = None,
    max_retries: int = 3
) -> AutoRetryEngine:
    """Factory function to create an auto-retry engine"""
    return AutoRetryEngine(analyzer, max_retries)


def analyze_failures(
    execution_result: TestExecutionResult,
    test_codes: Optional[Dict[str, str]] = None
) -> AnalysisResult:
    """Convenience function to analyze failures"""
    analyzer = FailureAnalyzer()
    return analyzer.analyze(execution_result, test_codes)


if __name__ == "__main__":
    # Example usage
    print("Failure Analyzer - Example Usage")
    print("=" * 50)

    # Create sample test result
    sample_result = TestResult(
        test_id="test-123",
        test_name="test_sample",
        test_file="tests/test_sample.py",
        status=TestStatus.FAILED,
        duration_ms=150.0,
        error_message="AssertionError: assert 1 == 2",
        error_traceback='File "tests/test_sample.py", line 10, in test_sample\n    assert 1 == 2'
    )

    execution_result = TestExecutionResult(
        execution_id="exec-456",
        total_tests=1,
        passed=0,
        failed=1,
        skipped=0,
        errors=0,
        duration_ms=150.0,
        test_results=[sample_result]
    )

    # Analyze
    analyzer = FailureAnalyzer()
    analysis = analyzer.analyze(execution_result)

    print(f"Analysis ID: {analysis.analysis_id}")
    print(f"Total failures: {len(analysis.failures)}")
    print(f"Total suggestions: {len(analysis.suggestions)}")
    print(f"Summary: {analysis.summary}")
