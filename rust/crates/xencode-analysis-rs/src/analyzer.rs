use crate::issues::{CodeIssue, IssueType, Severity};
use regex::Regex;
use std::path::Path;

/// Language-aware code analyzer.
pub struct CodeAnalyzer;

impl CodeAnalyzer {
    /// Analyze a single file, auto-detecting language from extension.
    pub fn analyze_file(path: &Path) -> Result<Vec<CodeIssue>, AnalysisError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let source = std::fs::read_to_string(path)
            .map_err(|e| AnalysisError::ReadError(path.display().to_string(), e.to_string()))?;

        let file_path = path.display().to_string();
        let issues = match ext.as_str() {
            "py" => Self::analyze_python(&source, &file_path),
            "js" | "jsx" | "ts" | "tsx" => Self::analyze_javascript(&source, &file_path),
            "rs" => Self::analyze_rust(&source, &file_path),
            _ => Self::analyze_generic(&source, &file_path),
        };

        Ok(issues)
    }

    /// Analyze Python source for common issues using regex patterns.
    fn analyze_python(source: &str, path: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let lineno = (i + 1) as u32;

            // Check for bare except
            if line.trim().starts_with("except:") {
                issues.push(CodeIssue::new(
                    IssueType::PotentialBug,
                    Severity::Medium,
                    "Bare except clause catches all exceptions, including SystemExit and KeyboardInterrupt",
                    path, lineno, 0,
                    "Use 'except Exception:' to catch only expected exceptions",
                    line.trim(),
                ));
            }

            // Check for print statements (potential debug left-overs)
            if line.trim().starts_with("print(") && !source.contains("import __future__") {
                issues.push(CodeIssue::new(
                    IssueType::StyleIssue,
                    Severity::Low,
                    "Print statement found; consider using logging instead",
                    path,
                    lineno,
                    0,
                    "Replace with logging.debug() or similar",
                    line.trim(),
                ));
            }

            // Check for mutable default arguments
            if line.contains("=[]") || line.contains("={}") {
                issues.push(CodeIssue::new(
                    IssueType::PotentialBug,
                    Severity::High,
                    "Mutable default argument is shared across all calls",
                    path,
                    lineno,
                    0,
                    "Use None as default and initialize inside the function",
                    line.trim(),
                ));
            }

            // Check for long lines (>100 chars)
            if line.len() > 100 {
                issues.push(CodeIssue::new(
                    IssueType::StyleIssue,
                    Severity::Low,
                    "Line too long",
                    path,
                    lineno,
                    100,
                    "Break line into multiple lines",
                    &line[..100.min(line.len())],
                ));
            }

            // Check for TODO/FIXME comments
            if line.contains("TODO") || line.contains("FIXME") || line.contains("HACK") {
                issues.push(CodeIssue::new(
                    IssueType::Maintainability,
                    Severity::Low,
                    "TODO/FIXME/HACK marker found",
                    path,
                    lineno,
                    0,
                    "Address the marked item",
                    line.trim(),
                ));
            }
        }

        // Check for missing docstrings in functions
        let docstring_re = Regex::new(r#"def \w+\(.*\):"#).unwrap();
        for cap in docstring_re.find_iter(source) {
            let line_start = source[..cap.start()].matches('\n').count();
            let lineno = (line_start + 1) as u32;
            // Look for docstring on the next line
            let next_line_offset = source[cap.end()..].find('\n').map(|o| cap.end() + o + 1);
            let has_docstring = next_line_offset
                .and_then(|off| source[off..].lines().next())
                .map(|l| l.trim().starts_with("\"\"\"") || l.trim().starts_with("'''"))
                .unwrap_or(false);

            if !has_docstring {
                issues.push(CodeIssue::new(
                    IssueType::Documentation,
                    Severity::Low,
                    "Function missing docstring",
                    path,
                    lineno,
                    0,
                    "Add a docstring describing purpose, args, and returns",
                    &source[cap.start()..cap.end()],
                ));
            }
        }

        issues
    }

    /// Analyze JavaScript/TypeScript source.
    fn analyze_javascript(source: &str, path: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let lineno = (i + 1) as u32;

            // Check for console.log
            if line.contains("console.log") {
                issues.push(CodeIssue::new(
                    IssueType::StyleIssue,
                    Severity::Low,
                    "Console log statement found",
                    path,
                    lineno,
                    0,
                    "Remove or replace with proper logging",
                    line.trim(),
                ));
            }

            // Check for == instead of ===
            if line.contains("==") && !line.contains("===") && !line.contains("=>") {
                issues.push(CodeIssue::new(
                    IssueType::PotentialBug,
                    Severity::Medium,
                    "Loose equality operator may cause type coercion issues",
                    path,
                    lineno,
                    0,
                    "Use === instead of ==",
                    line.trim(),
                ));
            }

            // Check for var usage
            if line.trim().starts_with("var ") {
                issues.push(CodeIssue::new(
                    IssueType::StyleIssue,
                    Severity::Medium,
                    "'var' declaration may cause scoping issues",
                    path,
                    lineno,
                    0,
                    "Use 'let' or 'const' instead of 'var'",
                    line.trim(),
                ));
            }
        }

        issues
    }

    /// Analyze Rust source.
    fn analyze_rust(source: &str, path: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let lineno = (i + 1) as u32;

            // Check for unwrap()
            if line.contains(".unwrap()") {
                issues.push(CodeIssue::new(
                    IssueType::PotentialBug,
                    Severity::Medium,
                    "Unwrap may cause panic on None/Err",
                    path,
                    lineno,
                    0,
                    "Use proper error handling with match or ? operator",
                    line.trim(),
                ));
            }

            // Check for TODO/FIXME
            if line.contains("TODO") || line.contains("FIXME") || line.contains("HACK") {
                issues.push(CodeIssue::new(
                    IssueType::Maintainability,
                    Severity::Low,
                    "TODO/FIXME/HACK marker found",
                    path,
                    lineno,
                    0,
                    "Address the marked item",
                    line.trim(),
                ));
            }

            // Check for missing documentation on public items
            if line.trim().starts_with("pub fn")
                || line.trim().starts_with("pub struct")
                || line.trim().starts_with("pub enum")
                || line.trim().starts_with("pub trait")
            {
                // Check if previous non-blank line has a doc comment
                let prev_line = if i > 0 { lines[i - 1].trim() } else { "" };
                if !prev_line.starts_with("///") && !prev_line.starts_with("#[doc") {
                    issues.push(CodeIssue::new(
                        IssueType::Documentation,
                        Severity::Low,
                        "Public item missing documentation",
                        path,
                        lineno,
                        0,
                        "Add /// doc comment explaining purpose and usage",
                        line.trim(),
                    ));
                }
            }
        }

        issues
    }

    /// Analyze generic source files for basic issues.
    fn analyze_generic(source: &str, path: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let lineno = (i + 1) as u32;

            if line.len() > 120 {
                issues.push(CodeIssue::new(
                    IssueType::StyleIssue,
                    Severity::Low,
                    "Line too long",
                    path,
                    lineno,
                    120,
                    "Break line into multiple lines",
                    &line[..120.min(line.len())],
                ));
            }

            if line.contains("TODO") || line.contains("FIXME") {
                issues.push(CodeIssue::new(
                    IssueType::Maintainability,
                    Severity::Low,
                    "TODO/FIXME marker found",
                    path,
                    lineno,
                    0,
                    "Address the marked item",
                    line.trim(),
                ));
            }
        }

        issues
    }
}

/// Error type for analysis operations.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Failed to read file {0}: {1}")]
    ReadError(String, String),
}
