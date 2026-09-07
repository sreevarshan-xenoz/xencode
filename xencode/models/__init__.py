"""
Data models package for Xencode.

Contains Pydantic models for documents, users, workspaces, and code analysis.
"""

try:
    from .document import DocumentType, DocumentMetadata, ProcessedDocument
except ImportError:
    DocumentType = DocumentMetadata = ProcessedDocument = None

try:
    from .user import User, UserRole
except ImportError:
    User = UserRole = None

try:
    from .workspace import WorkspaceConfig, WorkspaceFile
except ImportError:
    WorkspaceConfig = WorkspaceFile = None

try:
    from .code_analysis import AnalysisIssue, SeverityLevel, ComplexityMetrics
except ImportError:
    AnalysisIssue = SeverityLevel = ComplexityMetrics = None

__all__ = [
    "DocumentType",
    "DocumentMetadata",
    "ProcessedDocument",
    "User",
    "UserRole",
    "WorkspaceConfig",
    "WorkspaceFile",
    "AnalysisIssue",
    "SeverityLevel",
    "ComplexityMetrics",
]
