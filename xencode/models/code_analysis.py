#!/usr/bin/env python3
"""
Code Analysis Data Models

Defines data models for code analysis results, including syntax analysis,
error detection, complexity metrics, and refactoring suggestions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class AnalysisType(str, Enum):
    """Types of code analysis"""
    SYNTAX = "syntax"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    REFACTORING = "refactoring"
    DEPENDENCIES = "dependencies"


class SeverityLevel(str, Enum):
    """Severity levels for analysis issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"
    UNKNOWN = "unknown"


@dataclass
class CodeLocation:
    """Location information for code elements"""
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'line': self.line,
            'column': self.column,
            'end_line': self.end_line,
            'end_column': self.end_column
        }


@dataclass
class AnalysisIssue:
    """Represents an issue found during code analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_type: AnalysisType = AnalysisType.SYNTAX
    severity: SeverityLevel = SeverityLevel.INFO
    message: str = ""
    description: str = ""
    location: Optional[CodeLocation] = None
    
    # Code context
    code_snippet: Optional[str] = None
    affected_code: Optional[str] = None
    
    # Suggestions
    suggested_fix: Optional[str] = None
    refactoring_suggestion: Optional[str] = None
    
    # Metadata
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'analysis_type': self.analysis_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'description': self.description,
            'location': self.location.to_dict() if self.location else None,
            'code_snippet': self.code_snippet,
            'affected_code': self.affected_code,
            'suggested_fix': self.suggested_fix,
            'refactoring_suggestion': self.refactoring_suggestion,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'confidence': self.confidence
        }


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    logical_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    
    # Function/method metrics
    function_count: int = 0
    class_count: int = 0
    max_function_complexity: int = 0
    avg_function_complexity: float = 0.0
    
    # Nesting and depth
    max_nesting_depth: int = 0
    avg_nesting_depth: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'lines_of_code': self.lines_of_code,
            'logical_lines_of_code': self.logical_lines_of_code,
            'comment_lines': self.comment_lines,
            'blank_lines': self.blank_lines,
            'function_count': self.function_count,
            'class_count': self.class_count,
            'max_function_complexity': self.max_function_complexity,
            'avg_function_complexity': self.avg_function_complexity,
            'max_nesting_depth': self.max_nesting_depth,
            'avg_nesting_depth': self.avg_nesting_depth
        }


@dataclass
class SecurityIssue:
    """Security-specific analysis issue"""
    vulnerability_type: str = ""
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    owasp_category: Optional[str] = None
    risk_level: SeverityLevel = SeverityLevel.INFO
    exploit_scenario: Optional[str] = None
    mitigation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'vulnerability_type': self.vulnerability_type,
            'cwe_id': self.cwe_id,
            'owasp_category': self.owasp_category,
            'risk_level': self.risk_level.value,
            'exploit_scenario': self.exploit_scenario,
            'mitigation': self.mitigation
        }


@dataclass
class PerformanceIssue:
    """Performance-specific analysis issue"""
    performance_category: str = ""  # e.g., "memory", "cpu", "io", "algorithm"
    impact_level: SeverityLevel = SeverityLevel.INFO
    estimated_impact: Optional[str] = None
    optimization_suggestion: Optional[str] = None
    benchmark_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'performance_category': self.performance_category,
            'impact_level': self.impact_level.value,
            'estimated_impact': self.estimated_impact,
            'optimization_suggestion': self.optimization_suggestion,
            'benchmark_data': self.benchmark_data
        }


@dataclass
class RefactoringSuggestion:
    """Refactoring suggestion with before/after code"""
    refactoring_type: str = ""  # e.g., "extract_method", "rename_variable"
    description: str = ""
    before_code: str = ""
    after_code: str = ""
    benefits: List[str] = field(default_factory=list)
    effort_level: str = "medium"  # "low", "medium", "high"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'refactoring_type': self.refactoring_type,
            'description': self.description,
            'before_code': self.before_code,
            'after_code': self.after_code,
            'benefits': self.benefits,
            'effort_level': self.effort_level,
            'confidence': self.confidence
        }


@dataclass
class DependencyInfo:
    """Information about code dependencies"""
    name: str = ""
    version: Optional[str] = None
    import_path: str = ""
    usage_count: int = 0
    is_external: bool = True
    is_deprecated: bool = False
    security_issues: List[str] = field(default_factory=list)
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'import_path': self.import_path,
            'usage_count': self.usage_count,
            'is_external': self.is_external,
            'is_deprecated': self.is_deprecated,
            'security_issues': self.security_issues,
            'license': self.license
        }


@dataclass
class CodeAnalysisResult:
    """Complete code analysis result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    language: Language = Language.UNKNOWN
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_duration_ms: int = 0
    
    # Analysis results
    issues: List[AnalysisIssue] = field(default_factory=list)
    complexity_metrics: Optional[ComplexityMetrics] = None
    security_issues: List[SecurityIssue] = field(default_factory=list)
    performance_issues: List[PerformanceIssue] = field(default_factory=list)
    refactoring_suggestions: List[RefactoringSuggestion] = field(default_factory=list)
    dependencies: List[DependencyInfo] = field(default_factory=list)
    
    # Summary statistics
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    
    # Quality score (0-100)
    quality_score: float = 0.0
    maintainability_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    
    def __post_init__(self):
        """Calculate summary statistics after initialization"""
        self._calculate_summary_stats()
    
    def add_issue(self, issue: AnalysisIssue) -> None:
        """Add an analysis issue"""
        self.issues.append(issue)
        self._calculate_summary_stats()
    
    def add_security_issue(self, security_issue: SecurityIssue, base_issue: AnalysisIssue) -> None:
        """Add a security issue with base analysis issue"""
        base_issue.analysis_type = AnalysisType.SECURITY
        self.issues.append(base_issue)
        self.security_issues.append(security_issue)
        self._calculate_summary_stats()
    
    def add_performance_issue(self, perf_issue: PerformanceIssue, base_issue: AnalysisIssue) -> None:
        """Add a performance issue with base analysis issue"""
        base_issue.analysis_type = AnalysisType.PERFORMANCE
        self.issues.append(base_issue)
        self.performance_issues.append(perf_issue)
        self._calculate_summary_stats()
    
    def add_refactoring_suggestion(self, suggestion: RefactoringSuggestion) -> None:
        """Add a refactoring suggestion"""
        self.refactoring_suggestions.append(suggestion)
    
    def _calculate_summary_stats(self) -> None:
        """Calculate summary statistics"""
        self.total_issues = len(self.issues)
        self.critical_issues = sum(1 for issue in self.issues if issue.severity == SeverityLevel.CRITICAL)
        self.error_issues = sum(1 for issue in self.issues if issue.severity == SeverityLevel.ERROR)
        self.warning_issues = sum(1 for issue in self.issues if issue.severity == SeverityLevel.WARNING)
        self.info_issues = sum(1 for issue in self.issues if issue.severity == SeverityLevel.INFO)
        
        # Calculate quality scores
        self._calculate_quality_scores()
    
    def _calculate_quality_scores(self) -> None:
        """Calculate quality scores based on issues and metrics"""
        base_score = 100.0
        
        # Deduct points for issues
        base_score -= self.critical_issues * 20
        base_score -= self.error_issues * 10
        base_score -= self.warning_issues * 5
        base_score -= self.info_issues * 1
        
        self.quality_score = max(0.0, base_score)
        
        # Calculate specific scores
        self.security_score = max(0.0, 100.0 - len(self.security_issues) * 15)
        self.performance_score = max(0.0, 100.0 - len(self.performance_issues) * 10)
        
        # Maintainability based on complexity
        if self.complexity_metrics:
            complexity_penalty = min(self.complexity_metrics.cyclomatic_complexity * 2, 50)
            self.maintainability_score = max(0.0, 100.0 - complexity_penalty)
        else:
            self.maintainability_score = self.quality_score
    
    def get_issues_by_severity(self, severity: SeverityLevel) -> List[AnalysisIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_type(self, analysis_type: AnalysisType) -> List[AnalysisIssue]:
        """Get issues filtered by analysis type"""
        return [issue for issue in self.issues if issue.analysis_type == analysis_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'file_path': self.file_path,
            'language': self.language.value,
            'analyzed_at': self.analyzed_at.isoformat(),
            'analysis_duration_ms': self.analysis_duration_ms,
            'issues': [issue.to_dict() for issue in self.issues],
            'complexity_metrics': self.complexity_metrics.to_dict() if self.complexity_metrics else None,
            'security_issues': [issue.to_dict() for issue in self.security_issues],
            'performance_issues': [issue.to_dict() for issue in self.performance_issues],
            'refactoring_suggestions': [suggestion.to_dict() for suggestion in self.refactoring_suggestions],
            'dependencies': [dep.to_dict() for dep in self.dependencies],
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'error_issues': self.error_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
            'quality_score': self.quality_score,
            'maintainability_score': self.maintainability_score,
            'security_score': self.security_score,
            'performance_score': self.performance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeAnalysisResult':
        """Create CodeAnalysisResult from dictionary"""
        
        # Parse issues
        issues = []
        for issue_data in data.get('issues', []):
            location = None
            if issue_data.get('location'):
                location = CodeLocation(**issue_data['location'])
            
            issue = AnalysisIssue(
                id=issue_data.get('id', str(uuid.uuid4())),
                analysis_type=AnalysisType(issue_data.get('analysis_type', AnalysisType.SYNTAX)),
                severity=SeverityLevel(issue_data.get('severity', SeverityLevel.INFO)),
                message=issue_data.get('message', ''),
                description=issue_data.get('description', ''),
                location=location,
                code_snippet=issue_data.get('code_snippet'),
                affected_code=issue_data.get('affected_code'),
                suggested_fix=issue_data.get('suggested_fix'),
                refactoring_suggestion=issue_data.get('refactoring_suggestion'),
                rule_id=issue_data.get('rule_id'),
                rule_name=issue_data.get('rule_name'),
                confidence=issue_data.get('confidence', 1.0)
            )
            issues.append(issue)
        
        # Parse complexity metrics
        complexity_metrics = None
        if data.get('complexity_metrics'):
            complexity_metrics = ComplexityMetrics(**data['complexity_metrics'])
        
        # Parse security issues
        security_issues = [SecurityIssue(**issue_data) for issue_data in data.get('security_issues', [])]
        
        # Parse performance issues
        performance_issues = [PerformanceIssue(**issue_data) for issue_data in data.get('performance_issues', [])]
        
        # Parse refactoring suggestions
        refactoring_suggestions = [RefactoringSuggestion(**suggestion_data) for suggestion_data in data.get('refactoring_suggestions', [])]
        
        # Parse dependencies
        dependencies = [DependencyInfo(**dep_data) for dep_data in data.get('dependencies', [])]
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            file_path=data.get('file_path', ''),
            language=Language(data.get('language', Language.UNKNOWN)),
            analyzed_at=datetime.fromisoformat(data.get('analyzed_at', datetime.now().isoformat())),
            analysis_duration_ms=data.get('analysis_duration_ms', 0),
            issues=issues,
            complexity_metrics=complexity_metrics,
            security_issues=security_issues,
            performance_issues=performance_issues,
            refactoring_suggestions=refactoring_suggestions,
            dependencies=dependencies,
            total_issues=data.get('total_issues', 0),
            critical_issues=data.get('critical_issues', 0),
            error_issues=data.get('error_issues', 0),
            warning_issues=data.get('warning_issues', 0),
            info_issues=data.get('info_issues', 0),
            quality_score=data.get('quality_score', 0.0),
            maintainability_score=data.get('maintainability_score', 0.0),
            security_score=data.get('security_score', 0.0),
            performance_score=data.get('performance_score', 0.0)
        )


# Utility functions
def detect_language_from_extension(file_path: Union[str, Path]) -> Language:
    """Detect programming language from file extension"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    extension = file_path.suffix.lower()
    
    language_mapping = {
        '.py': Language.PYTHON,
        '.js': Language.JAVASCRIPT,
        '.ts': Language.TYPESCRIPT,
        '.java': Language.JAVA,
        '.cpp': Language.CPP,
        '.cc': Language.CPP,
        '.cxx': Language.CPP,
        '.c': Language.C,
        '.cs': Language.CSHARP,
        '.go': Language.GO,
        '.rs': Language.RUST,
        '.php': Language.PHP,
        '.rb': Language.RUBY,
        '.html': Language.HTML,
        '.htm': Language.HTML,
        '.css': Language.CSS,
        '.sql': Language.SQL,
        '.sh': Language.BASH,
        '.bash': Language.BASH,
        '.ps1': Language.POWERSHELL,
    }
    
    return language_mapping.get(extension, Language.UNKNOWN)


def get_tree_sitter_language_name(language: Language) -> Optional[str]:
    """Get tree-sitter language name for a Language enum"""
    mapping = {
        Language.PYTHON: 'python',
        Language.JAVASCRIPT: 'javascript',
        Language.TYPESCRIPT: 'typescript',
        Language.JAVA: 'java',
        Language.CPP: 'cpp',
        Language.C: 'c',
        Language.CSHARP: 'c_sharp',
        Language.GO: 'go',
        Language.RUST: 'rust',
        Language.PHP: 'php',
        Language.RUBY: 'ruby',
        Language.HTML: 'html',
        Language.CSS: 'css',
        Language.BASH: 'bash',
    }
    
    return mapping.get(language)