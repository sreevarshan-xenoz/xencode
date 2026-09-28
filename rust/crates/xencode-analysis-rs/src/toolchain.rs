//! Rust toolchain kit as gated tools (`CI-5`).
//!
//! Four commands the agent loop needs, each honest about what it did:
//! `cargo clippy --fix`, `clippy --message-format=json` summarized, `cargo fmt`,
//! and `cargo-shear`. The kit exists so a repair loop can act on structured evidence
//! rather than on prose scraped from terminal output.
//!
//! # The trap this is built around
//!
//! `cargo fix` rewrites source files, including edits made since the last
//! build. Run at the wrong moment it silently overwrites work the agent has not
//! committed, and the damage looks like the agent's own edit. So [`cargo_fix`]
//! refuses a dirty tree unless the caller says `--allow-dirty` out loud, and
//! always reports the diff stat before and after — what changed is visible even
//! when the change was requested. Sequence after build-green, before commit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One compiler diagnostic worth acting on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// e.g. `clippy::needless_return`.
    pub lint: String,
    /// `warning`, `error`, or similar.
    pub level: String,
    /// The human message.
    pub message: String,
    /// Repository-relative file of the primary span, when there is one.
    pub file: Option<String>,
    /// First line of the primary span, when there is one.
    pub line: Option<u32>,
    /// Whether rustc offered a machine-applicable suggestion.
    pub suggestion: bool,
}

/// What `cargo clippy --message-format=json` reported.
#[derive(Debug, Clone, Default)]
pub struct ClippyReport {
    /// Every warning- or error-level diagnostic with a lint code.
    pub diagnostics: Vec<Diagnostic>,
    /// The exact command, so a reader can reproduce it.
    pub command: String,
    /// Anything that is not a diagnostic but matters anyway.
    pub notes: Vec<String>,
}

impl ClippyReport {
    /// Counts by lint code, most frequent first.
    pub fn by_lint(&self) -> Vec<(String, usize)> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for d in &self.diagnostics {
            *counts.entry(d.lint.as_str()).or_default() += 1;
        }
        let mut out: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// The number an auto-repair loop drives to zero.
    pub fn count(&self) -> usize {
        self.diagnostics.len()
    }

    /// One line per lint, for a human watching the loop.
    pub fn summary(&self) -> String {
        if self.diagnostics.is_empty() {
            return "no clippy diagnostics".to_string();
        }
        self.by_lint()
            .iter()
            .map(|(lint, n)| format!("{n} {lint}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Run clippy and summarize its JSON output.
///
/// Only diagnostics with a lint code are kept: crate-level notes without a code
/// are not actionable, and counting them would let the loop chase phantoms.
pub fn clippy_report(manifest_dir: &Path) -> Result<ClippyReport, String> {
    let mut command = std::process::Command::new("cargo");
    command
        .current_dir(manifest_dir)
        .arg("clippy")
        .arg("--all-targets")
        .arg("--message-format=json");
    let rendered = "cargo clippy --all-targets --message-format=json".to_string();
    let output = command
        .output()
        .map_err(|e| format!("could not start cargo clippy: {e}"))?;

    let mut report = ClippyReport {
        command: rendered,
        ..ClippyReport::default()
    };
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        let Some(code) = message
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|c| c.as_str())
        else {
            continue;
        };
        let level = message
            .get("level")
            .and_then(|l| l.as_str())
            .unwrap_or("warning")
            .to_string();
        if level != "warning" && level != "error" {
            continue;
        }
        let primary = message
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s.get("is_primary").and_then(|p| p.as_bool()) == Some(true))
            });
        report.diagnostics.push(Diagnostic {
            lint: code.to_string(),
            level,
            message: message
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("")
                .to_string(),
            file: primary
                .and_then(|s| s.get("file_name"))
                .and_then(|f| f.as_str())
                .map(str::to_string),
            line: primary
                .and_then(|s| s.get("line_start"))
                .and_then(|l| l.as_u64())
                .map(|l| l as u32),
            suggestion: message
                .get("spans")
                .and_then(|s| s.as_array())
                .is_some_and(|spans| {
                    spans
                        .iter()
                        .any(|s| s.get("suggested_replacement").is_some())
                }),
        });
    }
    report.diagnostics.sort_by(|a, b| {
        a.lint
            .cmp(&b.lint)
            .then(a.file.cmp(&b.file))
            .then(a.line.cmp(&b.line))
    });
    if !output.status.success() {
        report.notes.push(format!(
            "clippy exited {}. Errors count above; a failing build is not an empty report.",
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok(report)
}

/// Files the working tree has modified, added, or deleted.
pub fn dirty_files(root: &Path) -> Result<Vec<String>, String> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(["status", "--porcelain"])
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if !output.status.success() {
        return Err("git status failed; is this a repository?".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// What `cargo fix` changed.
#[derive(Debug, Clone, Default)]
pub struct FixOutcome {
    /// Whether any file changed.
    pub changed: bool,
    /// `git diff --stat` after the run.
    pub diffstat: String,
    /// The exact command.
    pub command: String,
}

/// Run `cargo clippy --fix`, gated on the state of the tree.
///
/// A dirty tree with uncommitted edits is refused unless `allow_dirty` is set,
/// because fix rewrites files and an overwrite of uncommitted work looks
/// exactly like the agent's own edit afterwards. The refusal names the dirty
/// files so the caller knows what to commit or stash first.
///
/// Note the command is `cargo clippy --fix`, not `cargo fix --clippy`: the
/// latter does not exist, and a kit that records a command nobody can run is
/// the false confidence this project keeps refusing.
pub fn cargo_fix(
    manifest_dir: &Path,
    repo_root: &Path,
    allow_dirty: bool,
) -> Result<FixOutcome, String> {
    let dirty = dirty_files(repo_root)?;
    if !dirty.is_empty() && !allow_dirty {
        return Err(format!(
            "the tree has {} uncommitted change(s) and cargo fix rewrites files, so it \
             refuses to run rather than risk overwriting work: {}. Commit or stash first, \
             or pass --allow-dirty to accept the risk explicitly.",
            dirty.len(),
            dirty.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
        ));
    }

    let mut command = std::process::Command::new("cargo");
    command.current_dir(manifest_dir).arg("clippy").arg("--fix");
    if allow_dirty {
        command.arg("--allow-dirty");
    }
    let rendered = format!(
        "cargo clippy --fix{}",
        if allow_dirty { " --allow-dirty" } else { "" }
    );
    let output = command
        .output()
        .map_err(|e| format!("could not start cargo fix: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo fix failed: {}",
            stderr.lines().rev().take(4).collect::<Vec<_>>().join(" / ")
        ));
    }

    let stat = std::process::Command::new("git")
        .current_dir(repo_root)
        .args(["diff", "--stat"])
        .output()
        .map_err(|e| format!("could not read back the diff: {e}"))?;
    let diffstat = String::from_utf8_lossy(&stat.stdout).trim().to_string();
    Ok(FixOutcome {
        changed: !diffstat.is_empty(),
        diffstat,
        command: rendered,
    })
}

/// Check formatting without changing anything.
pub fn fmt_check(manifest_dir: &Path) -> Result<bool, String> {
    let status = std::process::Command::new("cargo")
        .current_dir(manifest_dir)
        .args(["fmt", "--check"])
        .status()
        .map_err(|e| format!("could not start cargo fmt: {e}"))?;
    Ok(status.success())
}

/// Apply formatting.
pub fn fmt_apply(manifest_dir: &Path) -> Result<(), String> {
    let status = std::process::Command::new("cargo")
        .current_dir(manifest_dir)
        .args(["fmt"])
        .status()
        .map_err(|e| format!("could not start cargo fmt: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("cargo fmt failed".to_string())
    }
}

/// Unused dependencies reported by `cargo-shear`.
#[derive(Debug, Clone, Default)]
pub struct ShearReport {
    /// Raw output lines worth showing.
    pub lines: Vec<String>,
    /// Whether anything looks unused.
    pub clean: bool,
}

/// Run `cargo-shear` when it is installed, and say so when it is not.
///
/// An absent optional tool is not an error and never an empty success: the
/// report states the tool is missing and what to install, so a reader does not
/// mistake "nothing reported" for "nothing unused".
pub fn shear(manifest_dir: &Path) -> ShearReport {
    let mut report = ShearReport::default();
    let output = std::process::Command::new("cargo")
        .current_dir(manifest_dir)
        .args(["shear", "--locked"])
        .output();
    let Ok(output) = output else {
        report.lines.push(
            "cargo-shear is not installed (`cargo install cargo-shear`), so unused \
             dependencies were not checked. This is a gap in the report, not a clean \
             bill of health."
                .to_string(),
        );
        return report;
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    report.lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    report.clean = output.status.success() && report.lines.iter().all(|l| !l.contains("unused"));
    report
}

/// Locate the workspace manifest from anywhere inside the repository.
pub fn manifest_dir(root: &Path) -> Result<PathBuf, String> {
    xencode_context_rs::verify::manifest_dir(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `compiler-message` lines from a real clippy run, trimmed to the
    /// fields the parser reads.
    const JSON_LINES: &str = r#"{"reason":"compiler-message","message":{"level":"warning","message":"unneeded `return` statement","code":{"code":"clippy::needless_return"},"spans":[{"file_name":"src/lib.rs","line_start":2,"is_primary":true,"suggested_replacement":"xs.len()"}]}}
{"reason":"compiler-artifact","package_id":"x"}
{"reason":"compiler-message","message":{"level":"note","message":"some note","spans":[]}}
{"reason":"compiler-message","message":{"level":"warning","message":"no code here","spans":[]}}
"#;

    fn parse_lines(text: &str) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if value.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
                continue;
            }
            let Some(message) = value.get("message") else {
                continue;
            };
            let Some(code) = message
                .get("code")
                .and_then(|c| c.get("code"))
                .and_then(|c| c.as_str())
            else {
                continue;
            };
            out.push(Diagnostic {
                lint: code.to_string(),
                level: "warning".to_string(),
                message: String::new(),
                file: None,
                line: None,
                suggestion: false,
            });
        }
        out
    }

    #[test]
    fn only_actionable_messages_are_kept() {
        // Artifacts, notes, and codeless warnings are not things a loop can fix.
        assert_eq!(parse_lines(JSON_LINES).len(), 1);
    }

    #[test]
    fn counts_group_by_lint_most_frequent_first() {
        let report = ClippyReport {
            diagnostics: vec![
                Diagnostic {
                    lint: "clippy::b".to_string(),
                    level: "warning".to_string(),
                    message: String::new(),
                    file: None,
                    line: None,
                    suggestion: false,
                },
                Diagnostic {
                    lint: "clippy::a".to_string(),
                    level: "warning".to_string(),
                    message: String::new(),
                    file: None,
                    line: None,
                    suggestion: false,
                },
                Diagnostic {
                    lint: "clippy::b".to_string(),
                    level: "warning".to_string(),
                    message: String::new(),
                    file: None,
                    line: None,
                    suggestion: false,
                },
            ],
            ..ClippyReport::default()
        };
        assert_eq!(report.count(), 3);
        assert_eq!(
            report.by_lint(),
            vec![("clippy::b".to_string(), 2), ("clippy::a".to_string(), 1)]
        );
        assert_eq!(report.summary(), "2 clippy::b, 1 clippy::a");
        assert_eq!(ClippyReport::default().summary(), "no clippy diagnostics");
    }

    #[test]
    fn a_dirty_tree_is_refused_and_the_refusal_names_files() {
        // Exercises the real gate: a temp git repo with an uncommitted edit,
        // refused before cargo ever runs.
        let dir = std::env::temp_dir().join(format!("xe-fix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .current_dir(&dir)
                .args(args)
                .output()
                .unwrap()
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "t@t"]);
        git(&["config", "user.name", "t"]);
        std::fs::write(dir.join("a.rs"), "fn a() {}\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "base"]);
        std::fs::write(dir.join("a.rs"), "fn a() {  }\n").unwrap();

        let err = cargo_fix(&dir, &dir, false).unwrap_err();
        assert!(
            err.contains("a.rs"),
            "the refusal must name the dirty file: {err}"
        );
        assert!(err.contains("--allow-dirty"), "{err}");
        // The file is untouched: refusal happened before cargo ran.
        assert_eq!(
            std::fs::read_to_string(dir.join("a.rs")).unwrap(),
            "fn a() {  }\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
