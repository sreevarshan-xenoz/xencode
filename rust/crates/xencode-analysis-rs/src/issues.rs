use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of code issues detected during analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum IssueType {
    SyntaxError,
    StyleIssue,
    PotentialBug,
    Performance,
    Security,
    Maintainability,
    Documentation,
}

impl IssueType {
    pub fn description(&self) -> &str {
        match self {
            IssueType::SyntaxError => "Syntax or parsing error",
            IssueType::StyleIssue => "Code style or formatting concern",
            IssueType::PotentialBug => "Likely bug or logic error",
            IssueType::Performance => "Performance optimization opportunity",
            IssueType::Security => "Security vulnerability",
            IssueType::Maintainability => "Code maintainability concern",
            IssueType::Documentation => "Missing or insufficient documentation",
        }
    }
}

/// Severity levels for code issues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn label(&self) -> &str {
        match self {
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        }
    }
}

/// A single code issue found during analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIssue {
    pub issue_type: IssueType,
    pub severity: Severity,
    pub message: String,
    pub file_path: String,
    pub line_number: u32,
    pub column: u32,
    pub suggestion: String,
    pub code_snippet: String,
}

impl CodeIssue {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        issue_type: IssueType,
        severity: Severity,
        message: impl Into<String>,
        file_path: impl Into<String>,
        line_number: u32,
        column: u32,
        suggestion: impl Into<String>,
        code_snippet: impl Into<String>,
    ) -> Self {
        Self {
            issue_type,
            severity,
            message: message.into(),
            file_path: file_path.into(),
            line_number,
            column,
            suggestion: suggestion.into(),
            code_snippet: code_snippet.into(),
        }
    }
}

/// Summary statistics for an analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub total_issues: u32,
    pub by_severity: HashMap<Severity, u32>,
    pub by_type: HashMap<IssueType, u32>,
}

impl AnalysisSummary {
    pub fn new(issues: &[CodeIssue]) -> Self {
        let mut by_severity = HashMap::new();
        let mut by_type = HashMap::new();

        for issue in issues {
            *by_severity.entry(issue.severity.clone()).or_insert(0) += 1;
            *by_type.entry(issue.issue_type.clone()).or_insert(0) += 1;
        }

        Self {
            total_issues: issues.len() as u32,
            by_severity,
            by_type,
        }
    }
}

/// Complete analysis report for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub file_path: String,
    pub issues: Vec<CodeIssue>,
    pub summary: AnalysisSummary,
}

impl AnalysisReport {
    pub fn new(file_path: impl Into<String>, issues: Vec<CodeIssue>) -> Self {
        let summary = AnalysisSummary::new(&issues);
        Self {
            file_path: file_path.into(),
            issues,
            summary,
        }
    }
}

/// A security vulnerability finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFinding {
    pub finding_type: String,
    pub severity: Severity,
    pub message: String,
    pub file_path: String,
    pub line_number: u32,
    pub line_content: String,
    pub recommendation: String,
    pub cwe_id: Option<String>,
}

impl SecurityFinding {
    pub fn new(
        finding_type: impl Into<String>,
        severity: Severity,
        message: impl Into<String>,
        file_path: impl Into<String>,
        line_number: u32,
        line_content: impl Into<String>,
        recommendation: impl Into<String>,
    ) -> Self {
        Self {
            finding_type: finding_type.into(),
            severity,
            message: message.into(),
            file_path: file_path.into(),
            line_number,
            line_content: line_content.into(),
            recommendation: recommendation.into(),
            cwe_id: None,
        }
    }

    pub fn with_cwe(mut self, cwe: impl Into<String>) -> Self {
        self.cwe_id = Some(cwe.into());
        self
    }
}
