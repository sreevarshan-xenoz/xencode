//! Reading a build failure from rustc's own machine-readable output.
//!
//! A failing `cargo build` used to reach the model as a stderr dump: the
//! rendered text, truncated from the tail, so on a real workspace the part that
//! survived was the last few hundred bytes rather than the explanation of what
//! went wrong. rustc has carried a structured account of the same event since
//! long ago — an error code, the file and line, the help children with the exact
//! replacement it suggests, and for every `E`-code the full text of its entry in
//! the error index, shipped inside the compiler. This parses that and renders it
//! back as prose, so the thing the model reads is the diagnosis instead of the
//! tail of a log.
//!
//! Two shapes of the stream are worth knowing before reading the code. With
//! `--message-format=json` cargo writes the JSON to **stdout** and keeps its
//! progress and its own one-line summary on stderr, so the two are handled
//! apart. And a replayed cached failure prints the summary with no JSON at all,
//! which is why [`parse`] returns `None` rather than an empty report when the
//! text holds no diagnostics — the caller then shows the ordinary output.

use serde_json::Value;

/// What a machine-applicable replacement would put in place of a span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    pub replacement: String,
    pub applicability: String,
    pub at: String,
}

/// One rustc diagnostic, read from the fields rather than from `rendered`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: String,
    pub message: String,
    pub code: Option<String>,
    pub location: Option<String>,
    pub helps: Vec<String>,
    pub suggestion: Option<Suggestion>,
}

/// Everything one build said, with the error-index text kept once per code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildReport {
    pub diagnostics: Vec<Diagnostic>,
    pub explanations: Vec<(String, String)>,
}

/// The flag that turns the dump into the account.
pub const CARGO_JSON_FLAG: &str = "--message-format=json";
/// rustc's error-index entries run to several kilobytes; the first screens are
/// the ones that answer "why", so the rest is cut rather than dropped.
pub const EXPLANATION_CAP: usize = 1_200;
pub const MAX_EXPLANATIONS: usize = 3;
pub const MAX_HELPS: usize = 3;
pub const MAX_DIAGNOSTICS: usize = 20;
/// The whole account, kept under the output ceiling the calling tool applies
/// (`COMMAND_OUTPUT_CAP` in `xencode-tui-rs`), so that what is dropped for
/// length is decided here and named there rather than silently cut from the
/// front by a tail cap.
pub const ACCOUNT_CAP: usize = 6 * 1024;
/// A replacement longer than this is not a hint any more.
pub const SUGGESTION_CAP: usize = 120;

/// Ask for the machine-readable form, when it is safe to append the flag.
///
/// Only a single `cargo build` or `cargo check` is rewritten. A command that
/// composes would take the flag on the wrong word (`cargo build && cargo test`
/// would put it on the test), everything after `--` belongs to rustc rather than
/// cargo, and a caller that already chose a format knows what it wants.
pub fn cargo_json_command(command: &str) -> Option<String> {
    let trimmed = command.trim();
    if trimmed.chars().any(|c| {
        matches!(
            c,
            '&' | ';' | '|' | '<' | '>' | '$' | '`' | '\n' | '(' | ')'
        )
    }) {
        return None;
    }
    let mut words = trimmed.split_whitespace();
    if words.next()? != "cargo" {
        return None;
    }
    if !matches!(words.next()?, "build" | "check") {
        return None;
    }
    if trimmed.contains(" -- ") || trimmed.ends_with(" --") {
        return None;
    }
    if trimmed.contains("--message-format") || trimmed.contains("--format") {
        return None;
    }
    Some(format!("{trimmed} {CARGO_JSON_FLAG}"))
}

/// Read cargo's JSON stream. `None` means this is not a machine-readable build:
/// no line parsed with a `reason`, so the ordinary output is the better answer.
pub fn parse(stdout: &str) -> Option<BuildReport> {
    let mut report = BuildReport::default();
    let mut saw_json = false;
    for line in stdout.lines() {
        let line = line.trim_start();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(reason) = value.get("reason").and_then(Value::as_str) else {
            continue;
        };
        saw_json = true;
        if reason != "compiler-message" {
            continue;
        }
        let message = value.get("message")?;
        if let Some(diagnostic) = diagnostic(message) {
            if let Some(explanation) = explanation(message) {
                let code = diagnostic.code.clone().unwrap_or_else(|| "an error".into());
                let known = report.explanations.iter().any(|(c, _)| *c == code);
                if !known && report.explanations.len() < MAX_EXPLANATIONS {
                    report.explanations.push((code, explanation));
                }
            }
            report.diagnostics.push(diagnostic);
        }
    }
    saw_json.then_some(report)
}

fn diagnostic(message: &Value) -> Option<Diagnostic> {
    let level = message.get("level").and_then(Value::as_str)?.to_string();
    // A `failure-note` is cargo's own pointer at the end of a run, not a
    // diagnosis: "For more information about this error, try …". The line that
    // answers the question is the diagnostic it refers to.
    if level == "failure-note" {
        return None;
    }
    let mut diagnostic = Diagnostic {
        level,
        message: message
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        code: message
            .get("code")
            .and_then(code_name)
            .map(|code| code.to_string()),
        location: message
            .get("spans")
            .and_then(primary_span)
            .map(span_location),
        helps: Vec::new(),
        suggestion: None,
    };
    for child in message
        .get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if child.get("level").and_then(Value::as_str) != Some("help") {
            continue;
        }
        if diagnostic.helps.len() >= MAX_HELPS {
            break;
        }
        let text = child
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if let Some(suggestion) = child
            .get("spans")
            .and_then(primary_span)
            .and_then(substitute)
        {
            if diagnostic.suggestion.is_none() {
                diagnostic.suggestion = Some(suggestion.clone());
            }
            diagnostic.helps.push(format!(
                "{text} ⇒ {}",
                one_line(&suggestion.replacement, SUGGESTION_CAP)
            ));
        } else {
            diagnostic.helps.push(text);
        }
    }
    Some(diagnostic)
}

fn explanation(message: &Value) -> Option<String> {
    let code = message.get("code")?;
    let text = code.get("explanation").and_then(Value::as_str)?;
    (!text.trim().is_empty()).then(|| text.trim().to_string())
}

fn code_name(code: &Value) -> Option<&str> {
    match code {
        Value::String(code) => Some(code),
        Value::Object(object) => object.get("code").and_then(Value::as_str),
        _ => None,
    }
}

fn primary_span(spans: &Value) -> Option<&Value> {
    let spans = spans.as_array()?;
    spans
        .iter()
        .find(|span| span.get("is_primary").and_then(Value::as_bool) == Some(true))
        .or_else(|| spans.first())
}

fn span_location(span: &Value) -> String {
    let file = span
        .get("file_name")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    let line = span.get("line_start").and_then(Value::as_u64).unwrap_or(0);
    let column = span
        .get("column_start")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("{file}:{line}:{column}")
}

fn substitute(span: &Value) -> Option<Suggestion> {
    let replacement = span.get("suggested_replacement").and_then(Value::as_str)?;
    let applicability = span
        .get("suggestion_applicability")
        .and_then(Value::as_str)
        .unwrap_or("unspecified")
        .to_string();
    Some(Suggestion {
        replacement: replacement.to_string(),
        applicability,
        at: span_location(span),
    })
}

fn one_line(text: &str, cap: usize) -> String {
    let flat = text.replace('\n', " ⏎ ");
    if flat.len() <= cap {
        return flat;
    }
    let mut end = cap;
    while !flat.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &flat[..end])
}

impl BuildReport {
    /// The account a model reads: each error with its position and rustc's own
    /// fix, then the error-index text once per code.
    pub fn render(&self) -> String {
        let errors = self
            .diagnostics
            .iter()
            .filter(|d| d.level == "error")
            .count();
        let warnings = self.diagnostics.len() - errors;
        if self.diagnostics.is_empty() {
            return "rustc reported no errors and no warnings.".to_string();
        }
        let mut out = format!("{} error(s), {} warning(s) from rustc:", errors, warnings);
        // The tool that shows this caps what it keeps from the end, so an
        // account that runs past its own budget would lose the errors and keep
        // the textbook. It is cut here instead, where what was cut is said.
        let mut listed = 0;
        for diagnostic in self.diagnostics.iter().take(MAX_DIAGNOSTICS) {
            if listed > 0 && out.len() > ACCOUNT_CAP - EXPLANATION_CAP {
                break;
            }
            listed += 1;
            out.push('\n');
            out.push_str(&format!(
                "  {} {}{}",
                diagnostic.level,
                diagnostic
                    .code
                    .as_deref()
                    .map(|code| format!("{code}: "))
                    .unwrap_or_default(),
                diagnostic.message
            ));
            if let Some(location) = &diagnostic.location {
                out.push_str(&format!(" — {location}"));
            }
            for help in &diagnostic.helps {
                out.push_str(&format!("\n      help: {help}"));
            }
            if let Some(suggestion) = &diagnostic.suggestion {
                if suggestion.applicability == "MachineApplicable" {
                    out.push_str(&format!(
                        "\n      rustc can apply this itself ({}): {}",
                        suggestion.at,
                        one_line(&suggestion.replacement, SUGGESTION_CAP)
                    ));
                }
            }
        }
        if self.diagnostics.len() > listed {
            out.push_str(&format!(
                "\n  … {} more not listed; ask for the same command without the JSON format to see them",
                self.diagnostics.len() - listed
            ));
        }
        let mut explained = 0;
        for (code, text) in &self.explanations {
            if out.len() + EXPLANATION_CAP > ACCOUNT_CAP {
                break;
            }
            out.push_str(&format!(
                "\n\nWhat rustc's own error index says about {code}:\n{}",
                cut(text, EXPLANATION_CAP)
            ));
            explained += 1;
        }
        if explained < self.explanations.len() {
            out.push_str(&format!(
                "\n\n({} of these error codes were left out for length — `rustc --explain {}` answers one at a time.)",
                self.explanations.len() - explained,
                self.explanations[explained].0
            ));
        }
        out
    }
}

fn cut(text: &str, cap: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= cap {
        return trimmed.to_string();
    }
    let mut end = cap;
    while !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = trimmed[..end].to_string();
    if let Some(at) = out.rfind('\n') {
        if at > cap / 2 {
            out.truncate(at);
        }
    }
    out.push_str("\n… (explanation cut)");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `E0308` as cargo really prints it, from a live build: the fields the
    /// renderer uses, with `rendered` alongside to prove it is ignored.
    const E0308: &str = r#"{"reason":"compiler-message","package_id":"path+file:///tmp/p#p@0.1.0","manifest_path":"/tmp/p/Cargo.toml","target":{"kind":["lib"],"crate_types":["lib"],"name":"p","src_path":"/tmp/p/src/lib.rs","edition":"2021"},"message":{"rendered":"error[E0308]: mismatched types\n","$message_type":"diagnostic","children":[{"children":[],"code":null,"level":"help","message":"you can convert a `u32` to a `u64`","rendered":null,"spans":[{"byte_end":34,"byte_start":31,"column_end":35,"column_start":32,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":1,"line_start":1,"suggested_replacement":"u64::from(u32)","suggestion_applicability":"MachineApplicable","text":[{"highlight_end":35,"highlight_start":32,"text":"x }"}]}]}],"code":{"code":"E0308","explanation":"This error occurs when a type does not match the type that is expected.\n"},"level":"error","message":"mismatched types","spans":[{"byte_end":32,"byte_start":31,"column_end":33,"column_start":32,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":"expected `u64`, found `u32`","line_end":1,"line_start":1,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

    #[test]
    fn a_plain_build_or_check_is_asked_for_its_machine_readable_form() {
        assert_eq!(
            cargo_json_command("cargo build").as_deref(),
            Some("cargo build --message-format=json")
        );
        assert_eq!(
            cargo_json_command("  cargo check --release  ").as_deref(),
            Some("cargo check --release --message-format=json")
        );
    }

    #[test]
    fn a_command_that_composes_is_left_exactly_as_it_was() {
        for command in [
            "cargo build && cargo test",
            "cd rust && cargo build",
            "cargo test",
            "cargo build -- --features nightly",
            "cargo build --message-format=json",
            "make && cargo build",
            "echo $(cargo build)",
            "cargo",
        ] {
            assert_eq!(cargo_json_command(command), None, "{command:?}");
        }
    }

    #[test]
    fn a_diagnostic_is_read_from_rustcs_own_fields() {
        let report = parse(E0308).expect("a cargo JSON stream is recognised");
        assert_eq!(report.diagnostics.len(), 1);
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.level, "error");
        assert_eq!(diagnostic.code.as_deref(), Some("E0308"));
        assert_eq!(diagnostic.location.as_deref(), Some("src/lib.rs:1:32"));
        assert_eq!(
            diagnostic.helps,
            vec!["you can convert a `u32` to a `u64` ⇒ u64::from(u32)".to_string()]
        );
        let suggestion = diagnostic
            .suggestion
            .as_ref()
            .expect("the help carries a replacement");
        assert_eq!(suggestion.applicability, "MachineApplicable");
        assert_eq!(suggestion.at, "src/lib.rs:1:32");
        assert_eq!(
            report.explanations,
            vec![(
                "E0308".to_string(),
                "This error occurs when a type does not match the type that is expected."
                    .to_string()
            )]
        );
    }

    #[test]
    fn the_rendered_account_names_the_code_the_position_and_the_fix() {
        let report = parse(E0308).expect("parsed");
        let rendered = report.render();
        assert!(
            rendered.contains("error E0308: mismatched types — src/lib.rs:1:32"),
            "{rendered}"
        );
        assert!(rendered.contains("help: you can convert"), "{rendered}");
        assert!(
            rendered.contains("rustc can apply this itself (src/lib.rs:1:32): u64::from(u32)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("What rustc's own error index says about E0308:"),
            "{rendered}"
        );
        assert!(!rendered.contains("rendered"), "the dump is not re-used");
    }

    #[test]
    fn an_output_that_is_not_a_machine_readable_build_changes_nothing() {
        assert!(parse("error: could not compile `p` (lib) due to 1 previous error\n").is_none());
        assert!(parse("").is_none());
        assert!(parse("not json at all\n").is_none());
    }

    #[test]
    fn a_build_with_nothing_to_say_says_so_instead_of_printing_nothing() {
        let report = parse(r#"{"reason":"compiler-artifact","package_id":"a","filenames":[],"fresh":false,"message":{}}"#)
            .expect("cargo JSON without diagnostics is still a machine-readable build");
        assert_eq!(report.render(), "rustc reported no errors and no warnings.");
    }

    #[test]
    fn cargo_s_own_note_at_the_end_is_not_reported_as_an_error() {
        let line = r#"{"reason":"compiler-message","message":{"level":"failure-note","message":"For more information about this error, try `rustc --explain E0308`.","children":[],"code":null,"spans":[]}}"#;
        let report = parse(line).expect("still cargo JSON");
        assert!(report.diagnostics.is_empty(), "{:?}", report.diagnostics);
    }

    #[test]
    fn a_long_run_is_summarised_rather_than_pasted_and_says_what_it_hid() {
        let mut stdout = String::new();
        for _ in 0..25 {
            stdout.push_str(E0308);
            stdout.push('\n');
        }
        let report = parse(&stdout).expect("parsed");
        assert_eq!(report.diagnostics.len(), 25);
        assert_eq!(
            report.explanations.len(),
            1,
            "the same code explains itself once"
        );
        let rendered = report.render();
        assert!(rendered.contains("25 error(s)"), "{rendered}");
        assert!(
            rendered.contains("more not listed"),
            "the hidden tail is admitted, not hidden: {rendered}"
        );
        assert!(rendered.len() <= ACCOUNT_CAP, "{}", rendered.len());
    }

    /// A build whose diagnostics are individually long has to stop by size as
    /// well as by count, and still say what it stopped at.
    #[test]
    fn a_budget_as_tight_as_the_tools_is_applied_before_the_list_is() {
        let template = r#"{"reason":"compiler-message","message":{"level":"error","message":"@LONG@","code":{"code":"E0308","explanation":"because"},"spans":[]}}"#;
        let long = "x".repeat(600);
        let mut stdout = String::new();
        for _ in 0..25 {
            stdout.push_str(&template.replace("@LONG@", &long));
            stdout.push('\n');
        }
        let report = parse(&stdout).expect("parsed");
        let rendered = report.render();
        assert!(rendered.len() <= ACCOUNT_CAP, "{}", rendered.len());
        assert!(
            rendered.contains("more not listed"),
            "the size cut is named too: {rendered}"
        );
        assert!(
            rendered.contains("25 error(s)"),
            "the count still says what really happened: {rendered}"
        );
    }

    #[test]
    fn an_overlong_explanation_is_cut_at_a_line_and_says_so() {
        let long = format!("{}\n", "e".repeat(EXPLANATION_CAP + 500));
        let truncated = cut(&long, EXPLANATION_CAP);
        assert!(
            truncated.len() <= EXPLANATION_CAP + 40,
            "{}",
            truncated.len()
        );
        assert!(truncated.ends_with("… (explanation cut)"), "{truncated}");
        assert_eq!(cut("short", EXPLANATION_CAP), "short");
    }

    /// The end of the line this module has to stop at: three different codes
    /// only, because a build with more is not helped by three textbook pages.
    #[test]
    fn only_the_first_few_error_codes_get_their_text() {
        let template = r#"{"reason":"compiler-message","message":{"level":"error","message":"m","code":{"code":"@CODE@","explanation":"because"},"spans":[]}}"#;
        let mut stdout = String::new();
        for code in ["E0308", "E0425", "E0502", "E0716"] {
            stdout.push_str(&template.replace("@CODE@", code));
            stdout.push('\n');
        }
        let report = parse(&stdout).expect("parsed");
        assert_eq!(report.diagnostics.len(), 4);
        let codes: Vec<&str> = report
            .explanations
            .iter()
            .map(|(code, _)| code.as_str())
            .collect();
        assert_eq!(codes, vec!["E0308", "E0425", "E0502"]);
    }

    /// Real cargo and real rustc, no fixture in sight: a scratch crate that does
    /// not compile, built exactly the way the tool now builds it, so the field
    /// names above are checked against the compiler that is installed here.
    #[test]
    #[ignore = "compiles a scratch crate with the installed cargo"]
    fn a_live_build_of_a_broken_crate_says_what_rustc_knows() {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-rustc-json-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname=\"probe\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\n[workspace]\n",
        )
        .unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn f(x: u32) -> u64 { x }\n").unwrap();

        let asked = cargo_json_command("cargo build").expect("a plain build is worth asking");
        assert_eq!(asked, format!("cargo build {CARGO_JSON_FLAG}"));
        let ran = std::process::Command::new("cargo")
            .args(["build", CARGO_JSON_FLAG])
            .current_dir(&dir)
            .output()
            .expect("cargo runs here");
        assert!(!ran.status.success(), "the crate is meant to fail");

        let stdout = String::from_utf8_lossy(&ran.stdout);
        let report = parse(&stdout).expect("cargo's JSON arrives on stdout");
        let rendered = report.render();
        println!("── rustc's account, as the model now sees it ──\n{rendered}");
        println!(
            "── cargo's own stderr, kept alongside ──\n{}",
            String::from_utf8_lossy(&ran.stderr)
        );

        assert_eq!(report.diagnostics.len(), 1, "{rendered}");
        let diagnostic = &report.diagnostics[0];
        assert_eq!(diagnostic.code.as_deref(), Some("E0308"));
        assert_eq!(diagnostic.location.as_deref(), Some("src/lib.rs:1:27"));
        assert!(
            diagnostic
                .helps
                .iter()
                .any(|help| help.starts_with("you can convert")),
            "{:?}",
            diagnostic.helps
        );
        assert_eq!(report.explanations.len(), 1);
        let explanation = &report.explanations[0].1;
        println!("── explanation length: {} characters ──", explanation.len());
        assert!(
            explanation.contains("mismatched types") || explanation.contains("type"),
            "{explanation}"
        );

        // A second failure, the kind rustc answers at length: an unmet trait
        // bound, whose rendered text lists every type that implements the trait.
        // The same error is built twice — once the ordinary way, once for the
        // account — with a rewritten file in between, because a cached failure
        // replays without any JSON at all.
        let broken = "pub fn f(x: u32) -> u32 { x + \"no\" }\n";
        std::fs::write(dir.join("src/lib.rs"), broken).unwrap();
        let plain = std::process::Command::new("cargo")
            .arg("build")
            .current_dir(&dir)
            .output()
            .expect("cargo runs here");
        let ordinary = String::from_utf8_lossy(&plain.stderr).into_owned();
        std::fs::write(
            dir.join("src/lib.rs"),
            format!("{broken}// rebuilt on purpose\n"),
        )
        .unwrap();
        let ran = std::process::Command::new("cargo")
            .args(["build", CARGO_JSON_FLAG])
            .current_dir(&dir)
            .output()
            .expect("cargo runs here");
        let stdout = String::from_utf8_lossy(&ran.stdout);
        let report = parse(&stdout).expect("a rewritten source recompiles, so the JSON comes back");
        let rendered = report.render();
        println!("── the same error, three ways ──");
        println!("rustc's rendered text on stderr: {} bytes", ordinary.len());
        println!("cargo's JSON stream on stdout: {} bytes", stdout.len());
        println!("the account the model is handed: {} bytes", rendered.len());
        println!("{rendered}");
        assert_eq!(
            report.diagnostics[0].code.as_deref(),
            Some("E0277"),
            "{rendered}"
        );
        assert!(
            rendered.len() * 4 < stdout.len(),
            "the account has to be a fraction of the stream it replaces: {}",
            rendered.len()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
