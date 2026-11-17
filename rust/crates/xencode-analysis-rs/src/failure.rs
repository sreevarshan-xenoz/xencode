//! Turn real command output into structured, actionable failure diagnostics (EC-1).
//!
//! The agent loop already has two narrower readers of tool output, and this
//! module deliberately does not replace either of them:
//!
//! * `xencode_core_rs::rustc_json` rewrites a *plain* `cargo build`/`cargo check`
//!   into rustc's own machine-readable account, with the error index and the
//!   compiler's fix. That path only fires for a single, uncomposed build command.
//! * `reprogate` in the TUI crate reads panic sites to decide whether a
//!   red-to-green reproduction was witnessed. It records evidence for a gate;
//!   it does not suggest a fix.
//!
//! This classifier covers the case neither handles: the rendered text the model
//! actually sees when *any* command fails (`cargo test`, a composed command, a
//! bare binary, a shell step). It is a pure function over that text — no process
//! is run and no file is read — so the map from output shape to suggestion is
//! testable with captured output. It recognises Rust and cargo output shapes
//! only; it is not a general error parser for other languages or providers.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The kind of failure, in the vocabulary the repair loop acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Category {
    CompileError,
    TestFailure,
    Panic,
    Io,
    Permission,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Self::CompileError => "compile error",
            Self::TestFailure => "test failure",
            Self::Panic => "panic",
            Self::Io => "I/O error",
            Self::Permission => "permission error",
        }
    }
}

/// How serious the diagnostic is. Everything that made a command fail is an
/// `Error`; the field exists so a caller can tell a real failure from a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureSeverity {
    Error,
    Warning,
}

impl FailureSeverity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
        }
    }
}

/// One structured diagnostic extracted from command output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDiagnostic {
    pub category: Category,
    pub severity: FailureSeverity,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub message: String,
    pub suggested_action: String,
}

impl FailureDiagnostic {
    /// `src/lib.rs:12` when a position is known, the bare file when only a path
    /// is known, or an honest placeholder when neither is.
    pub fn location(&self) -> String {
        match (&self.file, self.line) {
            (Some(file), Some(line)) => format!("{file}:{line}"),
            (Some(file), None) => file.clone(),
            _ => "unknown location".to_string(),
        }
    }
}

fn compile_bracket_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"error\[(E\d{4})\]:\s*([^\r\n]+)").expect("compile-error regex"))
}

fn compile_rendered_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"error\s+(E\d{4}):\s*([^\r\n]+)").expect("rendered-error regex"))
}

fn location_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?:-->|—)\s*([^\r\n:]+):(\d+):(\d+)").expect("source-location regex")
    })
}

fn help_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"help:\s*([^\r\n]+)").expect("help regex"))
}

fn panic_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"panicked at ([^\r\n'"]+?):(\d+):(\d+):"#).expect("panic-site regex")
    })
}

fn failed_test_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*test\s+(\S+)\s+\.\.\.\s+FAILED|test result:\s*FAILED")
            .expect("failed-test regex")
    })
}

fn permission_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)permission denied|operation not permitted|os error 13")
            .expect("permission regex")
    })
}

fn io_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)no such file or directory|os error 2|not a directory|is a directory|os error 20|os error 21")
            .expect("I/O regex")
    })
}

fn path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?:cannot\s+\w+\s+|for\s+)([^\s:]+)").expect("path regex"))
}

/// The first `--> file:line:col` or `— file:line:col` in `text`. Windows paths
/// are normalised to forward slashes so a reported location reads the same on
/// every host.
fn find_location(text: &str) -> Option<(String, u32, u32)> {
    let caps = location_re().captures(text)?;
    let file = caps.get(1)?.as_str().trim().replace('\\', "/");
    if file.is_empty() {
        return None;
    }
    let line = caps.get(2)?.as_str().parse().ok()?;
    let column = caps.get(3)?.as_str().parse().ok()?;
    Some((file, line, column))
}

/// Keep the text after an error, but stop before the next error so a suggestion
/// cannot be borrowed from a different diagnostic.
fn until_next_error(text: &str) -> &str {
    let mut end = text.len();
    for marker in ["\nerror[", "\nerror E"] {
        if let Some(at) = text.find(marker) {
            end = end.min(at);
        }
    }
    &text[..end]
}

fn compile_errors(output: &str) -> Vec<FailureDiagnostic> {
    let mut out = Vec::new();
    let bracket: Vec<_> = compile_bracket_re().captures_iter(output).collect();
    let rendered: Vec<_> = compile_rendered_re().captures_iter(output).collect();

    for caps in bracket {
        let Some(whole) = caps.get(0) else { continue };
        let code = caps.get(1).map_or("", |m| m.as_str());
        let message = caps.get(2).map_or("", |m| m.as_str()).trim().to_string();
        let after = until_next_error(&output[whole.end()..]);
        let (file, line, column) = find_location(after)
            .or_else(|| find_location(whole.as_str()))
            .map(|(f, l, c)| (Some(f), Some(l), Some(c)))
            .unwrap_or((None, None, None));
        let hint = help_re()
            .captures(after)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string());
        out.push(compile_diag(code, &message, file, line, column, hint));
    }

    for caps in rendered {
        let Some(whole) = caps.get(0) else { continue };
        let code = caps.get(1).map_or("", |m| m.as_str());
        let raw = caps.get(2).map_or("", |m| m.as_str()).trim();
        // The rendered form carries the location on the same line, after an
        // em dash: `error E0308: mismatched types — src/lib.rs:1:27`.
        let message = raw.split(" — ").next().unwrap_or(raw).trim().to_string();
        let (file, line, column) = find_location(whole.as_str())
            .map(|(f, l, c)| (Some(f), Some(l), Some(c)))
            .unwrap_or((None, None, None));
        let after = until_next_error(&output[whole.end()..]);
        let hint = help_re()
            .captures(after)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string());
        out.push(compile_diag(code, &message, file, line, column, hint));
    }
    out
}

fn compile_diag(
    code: &str,
    message: &str,
    file: Option<String>,
    line: Option<u32>,
    column: Option<u32>,
    hint: Option<String>,
) -> FailureDiagnostic {
    let placeholder = "the reported line".to_string();
    let at = match (&file, line) {
        (Some(f), Some(l)) => format!("{f}:{l}"),
        (Some(f), None) => f.clone(),
        _ => placeholder.clone(),
    };
    let suggested_action = match hint {
        Some(help) => format!("Follow the compiler's hint at {at}: {help}"),
        None => format!("Correct the {code} reported at {at}: {message}"),
    };
    let _ = column; // kept on the diagnostic; the action only needs file:line
    FailureDiagnostic {
        category: Category::CompileError,
        severity: FailureSeverity::Error,
        file,
        line,
        column,
        message: format!("{code}: {message}"),
        suggested_action,
    }
}

fn path_after_verb(line: &str) -> Option<String> {
    path_re()
        .captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim_end_matches(':').replace('\\', "/"))
        .filter(|p| !p.is_empty())
}

fn permission_and_io(output: &str) -> Vec<FailureDiagnostic> {
    let mut out = Vec::new();
    for line in output.lines() {
        let text = line.trim();
        if text.is_empty() || text.starts_with("error[") || text.starts_with("error E") {
            continue;
        }
        let permission = permission_re().is_match(text);
        let io = !permission && io_re().is_match(text);
        if !permission && !io {
            continue;
        }
        let category = if permission {
            Category::Permission
        } else {
            Category::Io
        };
        let path = path_after_verb(text);
        let suggested_action = match (category, &path) {
            (Category::Permission, Some(p)) => format!(
                "Check the read/write rights for {p} and retry with the needed permission"
            ),
            (Category::Permission, None) => {
                "Check the file or directory permissions for this operation, then retry with the needed rights".to_string()
            }
            (Category::Io, Some(p)) => {
                format!("Verify that {p} exists and the path is correct, then retry")
            }
            (Category::Io, None) => {
                "Verify the path exists and is correct, then retry".to_string()
            }
            (Category::CompileError | Category::TestFailure | Category::Panic, _) => {
                "Inspect the reported output and retry after correcting it".to_string()
            }
        };
        out.push(FailureDiagnostic {
            category,
            severity: FailureSeverity::Error,
            file: path,
            line: None,
            column: None,
            message: text.to_string(),
            suggested_action,
        });
    }
    out
}

/// `(file, line, column, message)` for each `panicked at file:line:col:` in the
/// output. The message is the lines after the header, up to the next blank line,
/// note or panic — the shape Rust's test harness and bare binaries both print.
fn panic_sites(output: &str) -> Vec<(String, u32, u32, String)> {
    let mut sites = Vec::new();
    for caps in panic_re().captures_iter(output) {
        let (Some(file), Some(line), Some(col), Some(whole)) =
            (caps.get(1), caps.get(2), caps.get(3), caps.get(0))
        else {
            continue;
        };
        let file = file.as_str().replace('\\', "/");
        if file.is_empty() {
            continue;
        }
        let line = line.as_str().parse().unwrap_or(0);
        let col = col.as_str().parse().unwrap_or(0);
        let after = &output[whole.end()..];
        let mut message = after
            .lines()
            .skip(1)
            .take_while(|l| {
                !l.trim().is_empty() && !l.contains("panicked at") && !l.starts_with("note:")
            })
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ");
        if message.is_empty() {
            message = after.lines().next().unwrap_or("").trim().to_string();
        }
        sites.push((file, line, col, message));
    }
    sites
}

fn failed_tests(output: &str) -> Vec<String> {
    failed_test_re()
        .captures_iter(output)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

fn test_failure(
    name: Option<&str>,
    site: Option<&(String, u32, u32, String)>,
) -> FailureDiagnostic {
    let (file, line, column) = match site {
        Some((f, l, c, _)) => (Some(f.clone()), Some(*l), Some(*c)),
        None => (None, None, None),
    };
    let at = match (&file, line) {
        (Some(f), Some(l)) => format!("{f}:{l}"),
        (Some(f), None) => f.clone(),
        _ => "the failing test".to_string(),
    };
    let message = match (name, site) {
        (Some(test), Some((_, _, _, panic))) => format!("{test}: {panic}"),
        (Some(test), None) => format!("{test} failed"),
        (None, Some((_, _, _, panic))) => panic.clone(),
        (None, None) => "test result: FAILED".to_string(),
    };
    let suggested_action = match name {
        Some(test) => format!(
            "Fix the failing assertion at {at} so {test} passes; compare the expected and actual values in the panic message"
        ),
        None => format!(
            "Fix the failing assertion at {at}; compare the expected and actual values in the panic message"
        ),
    };
    FailureDiagnostic {
        category: Category::TestFailure,
        severity: FailureSeverity::Error,
        file,
        line,
        column,
        message,
        suggested_action,
    }
}

fn panic_failure(site: &(String, u32, u32, String)) -> FailureDiagnostic {
    let (file, line, column, panic) = site;
    let at = format!("{}:{}", file, line);
    FailureDiagnostic {
        category: Category::Panic,
        severity: FailureSeverity::Error,
        file: Some(file.clone()),
        line: Some(*line),
        column: Some(*column),
        message: panic.clone(),
        suggested_action: format!(
            "Handle the panic at {at} so it cannot be reached with these inputs; check the unwrap, expect or index that triggered it"
        ),
    }
}

/// Classify the combined stdout+stderr of a failed command.
///
/// Pure: it reads only `output`, runs nothing, and returns an empty vector when
/// no Rust/cargo failure shape is recognised. Diagnostics are ordered by
/// category so the result is stable for the same input.
pub fn classify_command_output(output: &str) -> Vec<FailureDiagnostic> {
    let mut out = compile_errors(output);
    out.extend(permission_and_io(output));

    let sites = panic_sites(output);
    let tests = failed_tests(output);
    let test_run_failed = output.contains("test result: FAILED") || !tests.is_empty();
    let mut used = vec![false; sites.len()];

    if test_run_failed {
        if tests.is_empty() {
            if let Some(site) = sites.first() {
                used[0] = true;
                out.push(test_failure(None, Some(site)));
            } else {
                out.push(test_failure(None, None));
            }
        } else {
            for (i, test) in tests.iter().enumerate() {
                if let Some(site) = sites.get(i) {
                    used[i] = true;
                    out.push(test_failure(Some(test), Some(site)));
                } else {
                    out.push(test_failure(Some(test), None));
                }
            }
        }
    }

    for (i, site) in sites.iter().enumerate() {
        if !used[i] {
            out.push(panic_failure(site));
        }
    }

    out.sort_by_key(|d| d.category);
    out
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut short: String = text.chars().take(max).collect();
    short.push('…');
    short
}

/// Render diagnostics as a short block to append to a failing command's result,
/// so the model sees a targeted next step beside the raw output. Empty when
/// nothing was classified.
pub fn render_suggestions(diagnostics: &[FailureDiagnostic]) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }
    let mut rendered = String::from("suggested fix:");
    for diag in diagnostics {
        rendered.push_str(&format!(
            "\n  - [{}] {}: {} — {}",
            diag.category.label(),
            diag.location(),
            clip(&diag.message, 120),
            diag.suggested_action
        ));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured verbatim from real runs in a scratch crate (rustc 1.100.0):
    //   cargo test --no-run, cargo test, a bare binary reading a locked file,
    //   the same binary on a missing path, and a bare out-of-bounds panic.
    const COMPILE_OUTPUT: &str = r#"   Compiling compile_probe v0.1.0 (/tmp/opencode/ec1-fixtures/compile_probe)
error[E0308]: mismatched types
 --> src/lib.rs:7:15
  |
7 |     needs_u64(n)
  |     --------- ^ expected `u64`, found `u32`
  |     |
  |     arguments to this function are incorrect
  |
note: function defined here
 --> src/lib.rs:1:8
  |
1 | pub fn needs_u64(x: u64) -> u64 {
  |        ^^^^^^^^^ ------
help: you can convert a `u32` to a `u64`
  |
7 |     needs_u64(n.into())
  |                +++++++

For more information about this error, try `rustc --explain E0308`.
error: could not compile `compile_probe` (lib) due to 1 previous error
"#;

    const TEST_OUTPUT: &str = r#"   Compiling test_probe v0.1.0 (/tmp/opencode/ec1-fixtures/test_probe)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.22s
     Running unittests src/lib.rs (/tmp/opencode/ec1-fixtures/target/debug/build/test_probe/d63efb8e7b3333b6/out/test_probe-d63efb8e7b3333b6)

running 1 test
test tests::doubles_input ... FAILED

failures:

---- tests::doubles_input stdout ----

thread 'tests::doubles_input' (2136622) panicked at src/lib.rs:12:9:
assertion `left == right` failed: doubling must be exact
  left: 3
 right: 4
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::doubles_input

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
"#;

    const PERMISSION_OUTPUT: &str =
        "cannot read /tmp/opencode/ec1-fixtures/locked.txt: Permission denied (os error 13)\n";

    const IO_OUTPUT: &str =
        "cannot read /tmp/opencode/ec1-fixtures/does-not-exist.txt: No such file or directory (os error 2)\n";

    const PANIC_OUTPUT: &str = r#"
thread 'main' (2137401) panicked at src/main.rs:3:37:
index out of bounds: the len is 3 but the index is 5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
"#;

    // The shape `run_foreground` shows for a plain `cargo build` after rustc's
    // JSON has been rendered; quoted from CLI_GUIDE.md's measured example.
    const RENDERED_OUTPUT: &str = r#"1 error(s), 0 warning(s) from rustc:
  error E0308: mismatched types — src/lib.rs:1:27
      help: you can convert a `u32` to a `u64` ⇒ .into()

error: could not compile `probe` (lib) due to 1 previous error
"#;

    #[test]
    fn compile_error_reports_code_location_category_and_hint() {
        let diags = classify_command_output(COMPILE_OUTPUT);
        let compile: Vec<_> = diags
            .iter()
            .filter(|d| d.category == Category::CompileError)
            .collect();
        assert_eq!(compile.len(), 1, "one compiler error, got {diags:?}");
        let diag = compile[0];
        assert_eq!(diag.file.as_deref(), Some("src/lib.rs"));
        assert_eq!(diag.line, Some(7));
        assert_eq!(diag.location(), "src/lib.rs:7");
        assert_eq!(diag.severity, FailureSeverity::Error);
        assert!(diag.message.contains("E0308"));
        assert!(diag.message.contains("mismatched types"));
        assert!(
            diag.suggested_action.contains("convert a `u32` to a `u64`"),
            "the compiler's hint should reach the suggestion: {}",
            diag.suggested_action
        );
    }

    #[test]
    fn test_failure_with_panic_is_one_diagnostic_at_the_panic_site() {
        let diags = classify_command_output(TEST_OUTPUT);
        assert_eq!(
            diags.len(),
            1,
            "the panic is the test failure, not a second issue: {diags:?}"
        );
        let diag = &diags[0];
        assert_eq!(diag.category, Category::TestFailure);
        assert_eq!(diag.file.as_deref(), Some("src/lib.rs"));
        assert_eq!(diag.line, Some(12));
        assert_eq!(diag.location(), "src/lib.rs:12");
        assert!(diag.message.contains("tests::doubles_input"));
        assert!(diag.message.contains("doubling must be exact"));
        assert!(diag.message.contains("left: 3"));
        assert!(diag.suggested_action.contains("tests::doubles_input"));
        assert!(diag.suggested_action.contains("expected and actual"));
    }

    #[test]
    fn permission_error_is_classified_with_its_path() {
        let diags = classify_command_output(PERMISSION_OUTPUT);
        assert_eq!(diags.len(), 1, "{diags:?}");
        let diag = &diags[0];
        assert_eq!(diag.category, Category::Permission);
        assert_eq!(diag.line, None);
        assert_eq!(
            diag.file.as_deref(),
            Some("/tmp/opencode/ec1-fixtures/locked.txt")
        );
        assert!(diag.message.contains("Permission denied (os error 13)"));
        assert!(diag.suggested_action.contains("rights"));
    }

    #[test]
    fn missing_path_is_classified_as_io() {
        let diags = classify_command_output(IO_OUTPUT);
        assert_eq!(diags.len(), 1, "{diags:?}");
        let diag = &diags[0];
        assert_eq!(diag.category, Category::Io);
        assert!(diag
            .message
            .contains("No such file or directory (os error 2)"));
        assert!(diag.suggested_action.contains("exists"));
    }

    #[test]
    fn a_bare_panic_without_a_test_runner_is_a_panic() {
        let diags = classify_command_output(PANIC_OUTPUT);
        assert_eq!(diags.len(), 1, "{diags:?}");
        let diag = &diags[0];
        assert_eq!(diag.category, Category::Panic);
        assert_eq!(diag.file.as_deref(), Some("src/main.rs"));
        assert_eq!(diag.line, Some(3));
        assert!(diag.message.contains("index out of bounds"));
        assert!(diag.suggested_action.contains("unwrap, expect or index"));
    }

    #[test]
    fn rendered_rustc_report_is_classified_too() {
        let diags = classify_command_output(RENDERED_OUTPUT);
        assert_eq!(diags.len(), 1, "{diags:?}");
        let diag = &diags[0];
        assert_eq!(diag.category, Category::CompileError);
        assert_eq!(diag.location(), "src/lib.rs:1");
        assert!(diag.message.contains("mismatched types"));
        assert!(diag.suggested_action.contains("convert"));
    }

    #[test]
    fn unrecognised_output_yields_nothing_to_render() {
        assert!(classify_command_output("running 1 test\ntest ok ... ok\n").is_empty());
        assert!(render_suggestions(&[]).is_empty());
    }

    #[test]
    fn rendering_names_category_location_and_action() {
        let diags = classify_command_output(COMPILE_OUTPUT);
        let rendered = render_suggestions(&diags);
        assert!(rendered.starts_with("suggested fix:"));
        assert!(rendered.contains("[compile error] src/lib.rs:7"));
        assert!(rendered.contains("convert a `u32` to a `u64`"));
    }
}
