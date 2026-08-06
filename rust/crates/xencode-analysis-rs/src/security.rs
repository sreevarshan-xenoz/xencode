use crate::issues::{SecurityFinding, Severity};
use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;

/// Pattern-based vulnerability scanner focused on OWASP Top 10.
pub struct VulnerabilityScanner;

impl VulnerabilityScanner {
    /// Scan a single file for security vulnerabilities.
    pub fn scan_file(path: &Path) -> Result<Vec<SecurityFinding>, ScanError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| ScanError::ReadError(path.display().to_string(), e.to_string()))?;

        let file_path = path.display().to_string();
        let mut findings = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let lineno = (i + 1) as u32;

            if let Some(finding) = Self::check_hardcoded_secrets(line, &file_path, lineno) {
                findings.push(finding);
            }
            if let Some(finding) = Self::check_injection(line, &file_path, lineno) {
                findings.push(finding);
            }
            if let Some(finding) = Self::check_weak_crypto(line, &file_path, lineno) {
                findings.push(finding);
            }
            if let Some(finding) = Self::check_path_traversal(line, &file_path, lineno) {
                findings.push(finding);
            }
            if let Some(finding) = Self::check_ssrf(line, &file_path, lineno) {
                findings.push(finding);
            }
        }

        Ok(findings)
    }

    /// Check for hardcoded secrets (passwords, API keys, tokens).
    fn check_hardcoded_secrets(line: &str, file_path: &str, lineno: u32) -> Option<SecurityFinding> {
        static PASSWORD_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#"(?i)(password|passwd|pwd|secret|api[-_]?key|api[-_]?secret|access[-_]?token)\s*[=:]\s*['\"][^'"]{8,}['"]"#).unwrap()
        });
        static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36,}|AKIA[0-9A-Z]{16})").unwrap()
        });

        if PASSWORD_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "hardcoded-secret",
                    Severity::Critical,
                    "Hardcoded credential detected - use environment variables or a secrets manager",
                    file_path, lineno, line,
                    "Remove the hardcoded value and use environment variables or a vault service",
                )
                .with_cwe("CWE-798"),
            );
        }

        if TOKEN_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "hardcoded-token",
                    Severity::Critical,
                    "Hardcoded API token or key detected",
                    file_path, lineno, line,
                    "Revoke this token and use environment variables instead",
                )
                .with_cwe("CWE-798"),
            );
        }

        None
    }

    /// Check for injection patterns (SQL, command, eval).
    fn check_injection(line: &str, file_path: &str, lineno: u32) -> Option<SecurityFinding> {
        // SQL injection: string concatenation in SQL queries
        static SQL_INJECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#"(?i)(execute|exec|query|raw_sql|run)\s*\([^)]*['\"].*\+.*['\"]"#).unwrap()
        });

        // Command injection: user input in shell commands
        static CMD_INJECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#"(?i)(os\.system|subprocess\.call|subprocess\.run|exec|eval|`)\s*\([^)]*\+"#).unwrap()
        });

        // Eval usage
        static EVAL_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)\beval\s*\(").unwrap()
        });

        if SQL_INJECTION_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "sql-injection",
                    Severity::Critical,
                    "Potential SQL injection: string concatenation in database query",
                    file_path, lineno, line,
                    "Use parameterized queries or an ORM instead of string concatenation",
                )
                .with_cwe("CWE-89"),
            );
        }

        if CMD_INJECTION_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "command-injection",
                    Severity::Critical,
                    "Potential command injection: user input in shell command",
                    file_path, lineno, line,
                    "Use subprocess with argument list instead of shell=True or string formatting",
                )
                .with_cwe("CWE-78"),
            );
        }

        if EVAL_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "dangerous-eval",
                    Severity::High,
                    "Use of eval() can execute arbitrary code",
                    file_path, lineno, line,
                    "Avoid eval(); use safer alternatives like ast.literal_eval()",
                )
                .with_cwe("CWE-95"),
            );
        }

        None
    }

    /// Check for weak cryptography.
    fn check_weak_crypto(line: &str, file_path: &str, lineno: u32) -> Option<SecurityFinding> {
        static WEAK_HASH_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)(md5|sha1)\s*\(").unwrap()
        });

        static WEAK_CIPHER_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"(?i)(DES|RC2|RC4|Blowfish)\s*\(").unwrap()
        });

        if let Some(cap) = WEAK_HASH_RE.find(line) {
            return Some(
                SecurityFinding::new(
                    "weak-crypto-hash",
                    Severity::High,
                    format!("Weak cryptographic hash function: {}", cap.as_str()),
                    file_path, lineno, line,
                    "Use SHA-256 or SHA-3 instead of MD5/SHA1",
                )
                .with_cwe("CWE-327"),
            );
        }

        if let Some(cap) = WEAK_CIPHER_RE.find(line) {
            return Some(
                SecurityFinding::new(
                    "weak-crypto-cipher",
                    Severity::High,
                    format!("Weak cipher algorithm: {}", cap.as_str()),
                    file_path, lineno, line,
                    "Use AES-256-GCM instead of DES/RC2/RC4",
                )
                .with_cwe("CWE-327"),
            );
        }

        None
    }

    /// Check for path traversal vulnerabilities.
    fn check_path_traversal(line: &str, file_path: &str, lineno: u32) -> Option<SecurityFinding> {
        static PATH_TRAVERSAL_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#"(?i)(open|read_text|read_to_string|Path::new)\s*\([^)]*user|input|param|filename"#).unwrap()
        });

        if PATH_TRAVERSAL_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "path-traversal",
                    Severity::High,
                    "Potential path traversal: user-controlled path used in file operation",
                    file_path, lineno, line,
                    "Validate and sanitize user input; use a allowlist of permitted paths",
                )
                .with_cwe("CWE-22"),
            );
        }

        None
    }

    /// Check for Server-Side Request Forgery (SSRF).
    fn check_ssrf(line: &str, file_path: &str, lineno: u32) -> Option<SecurityFinding> {
        static SSRF_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#"(?i)(requests\.(get|post|put|delete)|reqwest::(get|post)|fetch)\s*\([^)]*input|url|user|param"#).unwrap()
        });

        if SSRF_RE.is_match(line) {
            return Some(
                SecurityFinding::new(
                    "ssrf",
                    Severity::Medium,
                    "Potential SSRF: user-controlled URL used in HTTP request",
                    file_path, lineno, line,
                    "Validate URLs against an allowlist of permitted hosts",
                )
                .with_cwe("CWE-918"),
            );
        }

        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("Failed to read file {0}: {1}")]
    ReadError(String, String),
}
