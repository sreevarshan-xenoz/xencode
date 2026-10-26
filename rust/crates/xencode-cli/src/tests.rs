//! Unit tests for the command line in `main.rs`, kept in their own file.

use super::{
    attach_images_to_final_user_message, compute_advise, doc, format_image_text, human_duration,
    join, parse_comma_list, plural_count, resolve_audit_path, resolve_bind, Cli, Commands,
    GenerateShell, OutputFormat, PerfAction, PriceAction, RunsAction, SessionAction,
};
use clap::Parser;
use xencode_analysis_rs::images::{ImageFormat, ImageMeta};
use xencode_providers_rs::{ChatMessage, ContentPart, MessageContent};

/// Minimal standard-alphabet base64 decoder, so a test can check what the
/// encoder put on the wire without adding a dependency for one assertion.
fn decode_base64(encoded: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0;
    for c in encoded.bytes() {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\r' | b'\n' => continue,
            other => panic!("unexpected byte {other:#x} in base64"),
        } as u32;
        acc = (acc << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

/// A 1x1 opaque PNG — the smallest file that is genuinely an image, so the
/// intake path is exercised against real bytes.
const ONE_PIXEL_PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0xf8, 0xcf, 0xc0,
    0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xdd, 0x8d, 0xb0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

/// The three repair shapes the mutation run found survive, decided by the
/// pure gate so each one is pinned by a test rather than by a re-run.
fn repair_input(patch: &str, before: usize, after: usize, rerun_verdict: Option<&str>) -> String {
    let mutants = serde_json::json!([{
        "file": "src/lib.rs",
        "name": "src/lib.rs:2:5: replace is_even -> bool with true",
    }]);
    let mut value = serde_json::json!({
        "patch": patch,
        "assertions_before": { "tests/lib.rs": before },
        "assertions_after": { "tests/lib.rs": after },
        "targeted": mutants,
    });
    if let Some(verdict) = rerun_verdict {
        value["rerun"] = serde_json::json!({
            "mutants": [{
                "file": "src/lib.rs",
                "name": "src/lib.rs:2:5: replace is_even -> bool with true",
                "verdict": verdict,
            }],
        });
    }
    serde_json::to_string_pretty(&value).unwrap()
}

fn judge(text: &str) -> Result<(), String> {
    let path = std::env::temp_dir().join(format!(
        "xe-repair-{}-{:?}.json",
        std::process::id(),
        text.len()
    ));
    std::fs::write(&path, text).unwrap();
    let out = super::judge_repair(path.clone(), super::OutputFormat::Text);
    let _ = std::fs::remove_file(&path);
    out
}

#[test]
fn a_tautological_repair_is_rejected() {
    // The measured fake fix: the negative assertion is replaced by
    // `x || !x`, which is true for every value, and the count goes *up*.
    let text = repair_input(
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,1 +4,2 @@\n-    assert!(!is_even(3));\n+    assert!(x || !x);\n",
            2, 3,
            Some("missed"),
        );
    let err = judge(&text).unwrap_err();
    assert!(err.contains("rejected"), "{err}");
}

#[test]
fn a_repair_that_edits_the_code_under_mutation_is_rejected() {
    let text = repair_input(
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -2,1 +2,1 @@\n-    n % 2 == 0\n+    true\n",
        1,
        4,
        Some("caught"),
    );
    assert!(judge(&text).is_err());
}

#[test]
fn a_repair_with_no_rerun_is_rejected() {
    let text = repair_input(
        "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,0 +5 @@\n+    assert!(!is_even(3));\n",
        1,
        2,
        None,
    );
    assert!(judge(&text).is_err());
}

#[test]
fn a_real_repair_is_accepted() {
    let text = repair_input(
        "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,0 +5 @@\n+    assert!(!is_even(3));\n",
        1,
        2,
        Some("caught"),
    );
    assert!(judge(&text).is_ok());
}

fn generated_artifact(kind: &str, shell: Option<GenerateShell>) -> String {
    use clap::CommandFactory;
    let mut cmd = super::Cli::command();
    let mut out = Vec::new();
    match kind {
        "man" => {
            clap_mangen::Man::new(cmd).render(&mut out).unwrap();
        }
        _ => {
            let shell = match shell {
                Some(GenerateShell::Fish) => clap_complete::Shell::Fish,
                Some(GenerateShell::Zsh) => clap_complete::Shell::Zsh,
                _ => clap_complete::Shell::Bash,
            };
            clap_complete::generate(shell, &mut cmd, "xencode", &mut out);
        }
    }
    String::from_utf8(out).unwrap()
}

#[test]
fn the_committed_completions_are_generated_never_hand_written() {
    // The WF-6 trap is drift: these files must equal what clap emits now.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs");
    for (file, kind, shell) in [
        (
            "completions/xencode.fish",
            "completions",
            Some(GenerateShell::Fish),
        ),
        (
            "completions/xencode.zsh",
            "completions",
            Some(GenerateShell::Zsh),
        ),
        (
            "completions/xencode.bash",
            "completions",
            Some(GenerateShell::Bash),
        ),
        ("man/xencode.1", "man", None),
    ] {
        let committed =
            std::fs::read_to_string(root.join(file)).unwrap_or_else(|_| panic!("missing {file}"));
        assert_eq!(
            generated_artifact(kind, shell),
            committed,
            "{file} drifted from what clap generates — regenerate it, do not edit it"
        );
    }
}

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
fn the_release_pipeline_is_dist_generated_not_hand_written() {
    // WF-5's trap is keys in CI and drift in the release files. The workflow
    // must be dist's output, and must reference no secret but the automatic
    // token — anything else is a key waiting to leak.
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
        .expect("release workflow");
    assert!(
        workflow.contains("autogenerated by dist"),
        "release.yml must be dist's output, not a hand edit"
    );
    for line in workflow.lines() {
        if line.contains("secrets.") {
            assert!(
                line.contains("secrets.GITHUB_TOKEN"),
                "only the automatic token may appear: {line}"
            );
        }
    }
    assert!(!workflow.contains("GPG"), "no signing keys in CI");
}

#[test]
fn dist_config_names_targets_installers_and_repository() {
    let config =
        std::fs::read_to_string(repo_root().join("rust/dist-workspace.toml")).expect("dist config");
    for key in ["cargo-dist-version", "targets", "installers", "hosting"] {
        assert!(config.contains(key), "dist config lacks {key}");
    }
    assert!(config.contains("shell"), "no shell installer");
    let manifest = std::fs::read_to_string(repo_root().join("rust/crates/xencode-cli/Cargo.toml"))
        .expect("cli manifest");
    assert!(
        manifest.contains("repository = \"https://github.com/"),
        "dist needs a repository URL and will not guess one"
    );
}

#[test]
fn completions_name_real_subcommands() {
    let fish = generated_artifact("completions", Some(GenerateShell::Fish));
    for subcommand in [
        "toolchain",
        "test",
        "cov",
        "mutants",
        "anchor",
        "generate",
        "session",
        "envcheck",
        "paths",
        "migrate",
    ] {
        assert!(fish.contains(subcommand), "completions omit {subcommand}");
    }
}

#[test]
fn agents_contract_parses() {
    let cli = Cli::try_parse_from(["xencode", "agents", "--contract"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Agents { contract: true, .. })
    ));
}

#[test]
fn agents_health_parses() {
    let cli = Cli::try_parse_from(["xencode", "agents", "--health", "--agent", "claude"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Agents {
            health: true,
            agent: Some(ref a),
            ..
        }) if a == "claude"
    ));
}

#[test]
fn agents_parses() {
    let cli = Cli::try_parse_from(["xencode", "agents"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Agents { .. })));
}

#[test]
fn agents_package_and_resume_parse() {
    let cli = Cli::try_parse_from(["xencode", "agents", "--build-package", "task-123"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Agents {
            build_package: Some(ref t),
            ..
        }) if t == "task-123"
    ));

    let cli = Cli::try_parse_from([
        "xencode",
        "agents",
        "--package",
        "pkg.json",
        "--resume",
        "--agent",
        "claude",
    ])
    .unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Agents {
            package: Some(ref p),
            resume: true,
            agent: Some(ref a),
            ..
        }) if p.to_str() == Some("pkg.json") && a == "claude"
    ));
}

#[test]
fn agents_route_parses() {
    let cli = Cli::try_parse_from([
        "xencode",
        "agents",
        "--route",
        "task-456",
        "--require-cap",
        "acp",
        "--require-cap",
        "stream",
        "--max-cost",
        "1.50",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Agents {
            route_task: Some(ref t),
            ref require_caps,
            max_cost: Some(cost),
            ..
        }) => {
            assert_eq!(t, "task-456");
            assert_eq!(require_caps, &["acp", "stream"]);
            assert_eq!(cost, 1.50);
        }
        _ => panic!("expected agents --route"),
    }
}

#[test]
fn agents_redispatch_parses() {
    let cli = Cli::try_parse_from([
        "xencode",
        "agents",
        "--redispatch",
        "task-789",
        "--agent",
        "initial-worker",
        "--replacement-agent",
        "second-worker",
        "--stop-reason",
        "signal:9",
        "--test-cmd",
        "cargo test",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Agents {
            redispatch_task: Some(ref t),
            agent: Some(ref a),
            replacement_agent: Some(ref rep),
            stop_reason: Some(ref s),
            ref test_cmds,
            ..
        }) => {
            assert_eq!(t, "task-789");
            assert_eq!(a, "initial-worker");
            assert_eq!(rep, "second-worker");
            assert_eq!(s, "signal:9");
            assert_eq!(test_cmds, &["cargo test"]);
        }
        _ => panic!("expected agents --redispatch"),
    }
}

#[test]
fn paths_and_migrate_parse() {
    let cli = Cli::try_parse_from(["xencode", "paths"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Paths { .. })));
    let cli = Cli::try_parse_from(["xencode", "migrate"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Migrate { dry_run: false })
    ));
    let cli = Cli::try_parse_from(["xencode", "migrate", "--dry-run"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Migrate { dry_run: true })
    ));
}

#[test]
fn hotspots_parses_with_limit() {
    let cli = Cli::try_parse_from(["xencode", "hotspots"]).unwrap();
    match cli.command {
        Some(Commands::Hotspots { limit, .. }) => assert_eq!(limit, 10),
        _ => panic!("expected hotspots"),
    }
    let cli = Cli::try_parse_from(["xencode", "hotspots", "--limit", "3"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Hotspots { limit: 3, .. })
    ));
}

#[test]
fn impact_parses_a_file_and_optional_limit() {
    // The file is positional; without it the command must not parse.
    assert!(Cli::try_parse_from(["xencode", "impact"]).is_err());
    let cli = Cli::try_parse_from(["xencode", "impact", "src/lib.rs"]).unwrap();
    match cli.command {
        Some(Commands::Impact { file, limit, .. }) => {
            assert_eq!(file, "src/lib.rs");
            assert_eq!(limit, 15, "default section size");
        }
        _ => panic!("expected impact"),
    }
    let cli = Cli::try_parse_from([
        "xencode",
        "impact",
        "src/lib.rs",
        "--limit",
        "4",
        "--format",
        "json",
    ])
    .unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Impact { file, limit: 4, format: super::OutputFormat::Json, .. })
            if file == "src/lib.rs"
    ));
}

#[test]
fn removal_parses_a_file_and_optional_limit() {
    // The file is positional; without it the command must not parse.
    assert!(Cli::try_parse_from(["xencode", "removal"]).is_err());
    let cli = Cli::try_parse_from(["xencode", "removal", "src/leaf.rs"]).unwrap();
    match cli.command {
        Some(Commands::Removal { file, limit, .. }) => {
            assert_eq!(file, "src/leaf.rs");
            assert_eq!(limit, 15, "default section size");
        }
        _ => panic!("expected removal"),
    }
    let cli =
        Cli::try_parse_from(["xencode", "removal", "src/leaf.rs", "--format", "json"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Removal { file, limit: 15, format: super::OutputFormat::Json, .. })
            if file == "src/leaf.rs"
    ));
}

#[test]
fn doctor_selfcheck_parses() {
    let cli = Cli::try_parse_from(["xencode", "doctor", "--selfcheck"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Doctor {
            selfcheck: true,
            ..
        })
    ));
}

#[test]
fn doctor_deps_parses() {
    let cli = Cli::try_parse_from(["xencode", "doctor", "--deps"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Doctor { deps: true, .. })
    ));
}

#[test]
fn doctor_env_parses() {
    let cli = Cli::try_parse_from(["xencode", "doctor", "--env"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Doctor { env: true, .. })
    ));
    let cli = Cli::try_parse_from(["xencode", "doctor"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Doctor { env: false, .. })
    ));
}

#[test]
fn session_actions_parse() {
    let cli = Cli::try_parse_from(["xencode", "session", "resolve", "latest"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Session {
            action: SessionAction::Resolve { .. }
        })
    ));
    let cli = Cli::try_parse_from(["xencode", "session", "export", "demo", "--redacted"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Session {
            action: SessionAction::Export { .. }
        })
    ));
    let cli = Cli::try_parse_from(["xencode", "session", "name", "abc123", "demo"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Session {
            action: SessionAction::Name { .. }
        })
    ));
}

#[test]
fn envcheck_parses_with_optional_format() {
    let cli = Cli::try_parse_from(["xencode", "envcheck"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Envcheck { .. })));
    let cli = Cli::try_parse_from(["xencode", "envcheck", "--format", "json"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Envcheck { .. })));
}

#[test]
fn line_ranges_read_as_ranges() {
    assert_eq!(join(&[1, 2, 3, 7, 9, 10]), "1-3, 7, 9-10");
    assert_eq!(join(&[5]), "5");
    assert_eq!(join(&[]), "");
}

#[test]
fn the_new_verification_subcommands_parse() {
    // These are the arms `xencode cov` coverage found unexercised, so
    // parsing them is the cheapest honest check that the wiring is right.
    let cli = Cli::try_parse_from(["xencode", "cov", "--base", "main"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Cov { base: Some(_), .. })
    ));

    let cli = Cli::try_parse_from(["xencode", "cov", "--show-missing-lines"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Cov {
            show_missing_lines: true,
            ..
        })
    ));

    // Retries default to 0, and a retry-pass is not a pass.
    let cli = Cli::try_parse_from(["xencode", "test"]).unwrap();
    match cli.command {
        Some(Commands::Test {
            retries,
            stress_count,
            packages,
            isolate,
            base,
            ..
        }) => {
            assert_eq!(retries, 0, "a retry must be asked for, not assumed");
            assert_eq!(stress_count, 0);
            assert!(packages.is_empty());
            assert!(isolate.is_none());
            assert_eq!(base, "HEAD");
        }
        _ => panic!("expected the test subcommand"),
    }

    let cli = Cli::try_parse_from([
        "xencode",
        "test",
        "--isolate",
        "old_broken",
        "--base",
        "main",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Test {
            isolate,
            base,
            repeat,
            ..
        }) => {
            assert_eq!(isolate.as_deref(), Some("old_broken"));
            assert_eq!(base, "main");
            assert_eq!(repeat, 3);
        }
        _ => panic!("expected the test subcommand"),
    }

    let cli = Cli::try_parse_from([
        "xencode",
        "test",
        "--retries",
        "3",
        "--package",
        "a",
        "--package",
        "b",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Test {
            retries, packages, ..
        }) => {
            assert_eq!(retries, 3);
            assert_eq!(packages, vec!["a".to_string(), "b".to_string()]);
        }
        _ => panic!("expected the test subcommand"),
    }
}

#[test]
fn cov_takes_a_test_command_and_the_anchor_command_is_the_default() {
    let cli = Cli::try_parse_from(["xencode", "cov", "--test", "cargo test"]).unwrap();
    match cli.command {
        Some(Commands::Cov { test, .. }) => assert_eq!(test.as_deref(), Some("cargo test")),
        _ => panic!("expected cov"),
    }
    let cli = Cli::try_parse_from(["xencode", "cov"]).unwrap();
    match cli.command {
        // None means "use the repository's own verified test command".
        Some(Commands::Cov { test, base, .. }) => {
            assert!(test.is_none());
            assert!(base.is_none());
        }
        _ => panic!("expected cov"),
    }
}

/// QO-4's two thresholds are product decisions, so the defaults are pinned
/// here as well as in the module: alert at 5% of the baseline, judge at
/// α = 0.05.
#[test]
fn perf_check_defaults_to_a_five_percent_alert_judged_at_five_percent_significance() {
    let cli = Cli::try_parse_from(["xencode", "perf", "check"]).unwrap();
    match cli.command {
        Some(Commands::Perf {
            action:
                PerfAction::Check {
                    filter,
                    alert_pct,
                    alpha,
                    ..
                },
        }) => {
            assert!(filter.is_none());
            assert_eq!(alert_pct, 5.0);
            assert_eq!(alpha, 0.05);
        }
        _ => panic!("expected perf check"),
    }

    let cli = Cli::try_parse_from([
        "xencode",
        "perf",
        "check",
        "--filter",
        "retrieve",
        "--alert-pct",
        "2.5",
        "--alpha",
        "0.01",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Perf {
            action:
                PerfAction::Check {
                    filter,
                    alert_pct,
                    alpha,
                    ..
                },
        }) => {
            assert_eq!(filter.as_deref(), Some("retrieve"));
            assert_eq!(alert_pct, 2.5);
            assert_eq!(alpha, 0.01);
        }
        _ => panic!("expected perf check"),
    }
}

#[test]
fn perf_record_takes_no_filter_because_a_partial_baseline_is_not_a_baseline() {
    let cli = Cli::try_parse_from(["xencode", "perf", "record", "--force"]).unwrap();
    match cli.command {
        Some(Commands::Perf {
            action: PerfAction::Record { force, .. },
        }) => assert!(force),
        _ => panic!("expected perf record"),
    }
    let rejected = Cli::try_parse_from(["xencode", "perf", "record", "--filter", "retrieve"]);
    assert!(
        rejected.is_err(),
        "recording three of seven paths would replace the baseline with three of seven \
             paths, so the flag is not offered"
    );

    let cli = Cli::try_parse_from(["xencode", "perf", "show"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Perf {
            action: PerfAction::Show { .. }
        })
    ));
}

#[test]
fn a_measured_path_reads_in_the_unit_its_number_belongs_in() {
    // The seven paths span five orders of magnitude, so a fixed unit would
    // print either 0.016 or 801449.095.
    assert_eq!(human_duration(999.0), "999.000 ns");
    assert_eq!(human_duration(15_842.0), "15.842 µs");
    assert_eq!(human_duration(1_038_604.0), "1.039 ms");
    assert_eq!(human_duration(801_449_095.0), "801.449 ms");
    assert_eq!(human_duration(1_500_000_000.0), "1.500 s");
}

/// QO-6 — the whole flag set, parsed.
#[test]
fn release_notes_parses_every_flag_it_offers() {
    let cli = Cli::try_parse_from([
        "xencode",
        "release-notes",
        "--from",
        "v2.1.0",
        "--to",
        "main",
        "--release",
        "2.2.0",
        "--out",
        "draft/notes.md",
        "--force",
        "--format",
        "json",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::ReleaseNotes {
            from,
            to,
            release,
            out,
            force,
            format,
        }) => {
            assert_eq!(from.as_deref(), Some("v2.1.0"));
            assert_eq!(to.as_deref(), Some("main"));
            assert_eq!(release.as_deref(), Some("2.2.0"));
            assert_eq!(out, Some(path("draft/notes.md")));
            assert!(force);
            assert!(matches!(format, OutputFormat::Json));
        }
        _ => panic!("expected release-notes"),
    }
}

#[test]
fn release_notes_without_flags_drafts_head_as_an_unreleased_block() {
    let cli = Cli::try_parse_from(["xencode", "release-notes"]).unwrap();
    match cli.command {
        Some(Commands::ReleaseNotes {
            from,
            to,
            release,
            out,
            force,
            format,
        }) => {
            assert!(from.is_none(), "the newest tag picks the range");
            assert!(to.is_none(), "the default upper end is HEAD");
            assert!(
                release.is_none(),
                "no version is invented for the heading: it stays [Unreleased]"
            );
            assert!(out.is_none(), "the draft goes to standard output");
            assert!(!force, "an existing file is not replaced by default");
            assert!(matches!(format, OutputFormat::Text));
        }
        _ => panic!("expected release-notes"),
    }
}

#[test]
fn a_count_next_to_its_noun_reads_as_english() {
    assert_eq!(plural_count(1, "entry", "entries"), "1 entry");
    assert_eq!(plural_count(0, "entry", "entries"), "0 entries");
    assert_eq!(plural_count(131, "entry", "entries"), "131 entries");
}

/// `xencode prices` with nothing after it is the listing of what a cost report
/// would read — the fetch is the part that dials out, so it is never implied.
#[test]
fn prices_alone_shows_the_documents_and_fetches_nothing() {
    let cli = Cli::try_parse_from(["xencode", "prices"]).unwrap();
    match cli.command {
        Some(Commands::Prices { action }) => {
            assert!(action.is_none(), "no action means the showing one");
        }
        _ => panic!("expected prices"),
    }
}

#[test]
fn prices_show_takes_a_format_and_fetch_takes_a_url() {
    let cli = Cli::try_parse_from(["xencode", "prices", "show", "--format", "json"]).unwrap();
    match cli.command {
        Some(Commands::Prices {
            action: Some(PriceAction::Show { format }),
        }) => assert!(matches!(format, OutputFormat::Json)),
        _ => panic!("expected prices show"),
    }
    let cli = Cli::try_parse_from([
        "xencode",
        "prices",
        "fetch",
        "--url",
        "http://127.0.0.1:9/models",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Prices {
            action: Some(PriceAction::Fetch { url }),
        }) => assert_eq!(url.as_deref(), Some("http://127.0.0.1:9/models")),
        _ => panic!("expected prices fetch"),
    }
}

/// Bare `xencode runs` lists, the way bare `xencode prices` shows: the
/// common question needs no verb, and nothing here dials out.
#[test]
fn runs_alone_lists_and_show_takes_an_id() {
    let cli = Cli::try_parse_from(["xencode", "runs"]).unwrap();
    match cli.command {
        Some(Commands::Runs { action }) => {
            assert!(action.is_none(), "no action means the listing one");
        }
        _ => panic!("expected runs"),
    }
    let cli = Cli::try_parse_from(["xencode", "runs", "show", "1700000000-aaaa1111"]).unwrap();
    match cli.command {
        Some(Commands::Runs {
            action: Some(RunsAction::Show { run_id, .. }),
        }) => assert_eq!(run_id, "1700000000-aaaa1111"),
        _ => panic!("expected runs show"),
    }
    let cli = Cli::try_parse_from(["xencode", "runs", "trailer", "1700000000-aaaa"]).unwrap();
    match cli.command {
        Some(Commands::Runs {
            action: Some(RunsAction::Trailer { run_id }),
        }) => assert_eq!(run_id, "1700000000-aaaa"),
        _ => panic!("expected runs trailer"),
    }
}

/// `xencode run` takes one action per invocation: a prompt with `--log`
/// is refused by the dispatcher, not guessed at.
#[test]
fn run_takes_a_prompt_and_caps_or_one_reporting_flag() {
    let cli = Cli::try_parse_from([
        "xencode",
        "run",
        "fix the typo",
        "--detach",
        "--max-rounds",
        "4",
        "--max-minutes",
        "10",
        "--max-cost",
        "0.5",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Run {
            prompt,
            detach,
            max_rounds,
            max_minutes,
            max_cost,
            ..
        }) => {
            assert_eq!(prompt.as_deref(), Some("fix the typo"));
            assert!(detach);
            assert_eq!(max_rounds, Some(4));
            assert_eq!(max_minutes, Some(10.0));
            assert_eq!(max_cost, Some(0.5));
        }
        _ => panic!("expected run"),
    }
    let cli = Cli::try_parse_from(["xencode", "run", "--resume", "1700000000-aaaa"]).unwrap();
    match cli.command {
        Some(Commands::Run { resume, .. }) => {
            assert_eq!(resume.as_deref(), Some("1700000000-aaaa"))
        }
        _ => panic!("expected run --resume"),
    }
    // The worker is forked, never typed — but it parses, because the
    // forked child is this same binary.
    let cli = Cli::try_parse_from(["xencode", "run", "--child", "1700000000-aaaa"]).unwrap();
    match cli.command {
        Some(Commands::Run { child, .. }) => {
            assert_eq!(child.as_deref(), Some("1700000000-aaaa"))
        }
        _ => panic!("expected run --child"),
    }
}

fn path(p: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(p)
}

/// `config set agent_fallback_models "a, b,, c"` yields the ordered,
/// trimmed, non-empty list — the order is the try order (I4-01).
#[test]
fn comma_list_parses_ordered_trimmed_and_drops_empties() {
    assert_eq!(parse_comma_list("a, b,, c"), vec!["a", "b", "c"]);
    assert_eq!(
        parse_comma_list(" qwen2.5:14b ,gemini:gemini-2.0-flash , "),
        vec!["qwen2.5:14b", "gemini:gemini-2.0-flash"]
    );
    assert!(parse_comma_list("").is_empty());
    assert!(parse_comma_list("  , ,").is_empty());
}

#[test]
fn loopback_binds_are_accepted_without_tls_or_warning() {
    // "::1" is the case a `starts_with("127.")` check gets wrong.
    for host in ["127.0.0.1", "localhost", "::1", "127.0.0.53"] {
        let bind = resolve_bind(host, 8765, None, None, false).unwrap();
        assert!(bind.addr.ip().is_loopback(), "{host} misclassified");
        assert_eq!(bind.addr.port(), 8765);
        assert!(bind.tls.is_none());
        assert!(bind.warning.is_none());
    }
}

#[test]
fn a_public_plain_bind_is_refused_and_the_error_names_both_ways_out() {
    let err = resolve_bind("0.0.0.0", 8765, None, None, false).unwrap_err();
    assert!(err.contains("--cert") && err.contains("--key"));
    assert!(err.contains("--allow-insecure-public"));
    // IPv6 public bind: same refusal.
    assert!(resolve_bind("::", 8765, None, None, false).is_err());
}

#[test]
fn the_escape_hatch_binds_but_warns_loudly() {
    let bind = resolve_bind("0.0.0.0", 9000, None, None, true).unwrap();
    assert_eq!(bind.addr.port(), 9000);
    assert!(bind.tls.is_none());
    let warning = bind.warning.expect("plain public bind must warn");
    assert!(warning.contains("WITHOUT TLS"));
}

#[test]
fn cert_and_key_must_come_as_a_pair() {
    assert!(resolve_bind(
        "127.0.0.1",
        8765,
        Some(path("c.pem").as_path()),
        None,
        false
    )
    .unwrap_err()
    .contains("--key"));
    assert!(resolve_bind(
        "127.0.0.1",
        8765,
        None,
        Some(path("k.pem").as_path()),
        false
    )
    .unwrap_err()
    .contains("--cert"));
}

#[test]
fn certs_win_over_the_escape_hatch_and_speak() {
    let bind = resolve_bind(
        "0.0.0.0",
        8765,
        Some(path("c.pem").as_path()),
        Some(path("k.pem").as_path()),
        true,
    )
    .unwrap();
    assert_eq!(
        bind.tls,
        Some((path("c.pem"), path("k.pem"))),
        "TLS must be on, not the plain hatch"
    );
    assert!(bind.warning.is_none());
}

#[test]
fn tls_on_loopback_needs_no_hatch() {
    let bind = resolve_bind(
        "::1",
        8765,
        Some(path("c.pem").as_path()),
        Some(path("k.pem").as_path()),
        false,
    )
    .unwrap();
    assert!(bind.tls.is_some());
}

#[test]
fn invalid_hosts_are_rejected_not_resolved_lazily() {
    // Hostnames would need a DNS lookup — run_server must not pretend.
    assert!(resolve_bind("example.com", 8765, None, None, true).is_err());
    assert!(resolve_bind("0.0.0.256", 8765, None, None, true).is_err());
    assert!(resolve_bind("", 8765, None, None, true).is_err());
}

#[test]
fn audit_path_none_disables_explicit_paths_pass_through() {
    assert_eq!(resolve_audit_path(Some("none")).unwrap(), None);
    assert_eq!(
        resolve_audit_path(Some("/tmp/x.jsonl")).unwrap(),
        Some(path("/tmp/x.jsonl"))
    );
    // The default lands in the directory xencode keeps its records in,
    // never the repo.
    let default = resolve_audit_path(None).unwrap().unwrap();
    assert_eq!(default.file_name().unwrap(), "audit.jsonl");
    assert_eq!(
        default.parent().unwrap(),
        xencode_config_rs::paths::state_dir().unwrap()
    );
}

#[test]
fn compute_advise_reads_snapshot_filters_and_errors() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use xencode_context_rs::AdviceKind;
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "xencode-advise-cli-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\nmod c;\n").unwrap();
    std::fs::write(root.join("src/a.rs"), "use crate::b::bee;\n").unwrap();
    std::fs::write(root.join("src/b.rs"), "use crate::a::ay;\n").unwrap();
    std::fs::write(root.join("src/c.rs"), "pub fn cc() {}\n").unwrap();
    // Nothing declares or imports this one, so it is the only real orphan in
    // the tree: `mod c;` in lib.rs connects c.rs, which the graph could not
    // see until module declarations counted as edges.
    std::fs::write(root.join("src/unreferenced.rs"), "pub fn u() {}\n").unwrap();
    std::fs::File::create(root.join("Cargo.toml")).unwrap();
    xencode_context_rs::init_project(&root, std::sync::Arc::new(AtomicBool::new(false)), |_| {})
        .expect("init");

    let items = compute_advise(&root, None).unwrap();
    assert!(items
        .iter()
        .any(|i| i.kind == AdviceKind::Cycle && i.file == "src/a.rs"));
    assert!(items
        .iter()
        .any(|i| i.kind == AdviceKind::Orphan && i.file == "src/unreferenced.rs"));
    assert!(!items
        .iter()
        .any(|i| i.kind == AdviceKind::Orphan && i.file == "src/c.rs"));

    let filtered = compute_advise(&root, Some("unreferenced.rs")).unwrap();
    assert!(!filtered.is_empty());
    assert!(filtered.iter().all(|i| i.file.contains("unreferenced.rs")));
    assert!(compute_advise(&root, Some("nope")).unwrap().is_empty());

    // No snapshot at all → actionable error, not an empty report.
    let err = compute_advise(std::path::Path::new("/definitely-not-a-repo-xencode"), None)
        .expect_err("should fail");
    assert!(err.contains("no project index"), "{err}");

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn image_text_line_shows_mime_dimensions_and_bytes() {
    let meta = ImageMeta {
        path: "assets/logo.png".to_string(),
        format: ImageFormat::Png,
        width: Some(800),
        height: Some(600),
        bytes: 12345,
    };
    assert_eq!(
        format_image_text(&meta),
        "assets/logo.png — image/png 800x600 · 12345 bytes"
    );
}

#[test]
fn image_text_line_marks_vector_without_dimensions() {
    let meta = ImageMeta {
        path: "assets/icon.svg".to_string(),
        format: ImageFormat::Svg,
        width: None,
        height: None,
        bytes: 512,
    };
    assert_eq!(
        format_image_text(&meta),
        "assets/icon.svg — image/svg+xml vector · 512 bytes"
    );
}

#[test]
fn fetch_text_line_shows_url_title_and_body() {
    use xencode_analysis_rs::web::FetchedPage;
    let page = FetchedPage {
        url: "https://example.com/a".to_string(),
        title: Some("Example".to_string()),
        text: "hello web".to_string(),
        bytes: 128,
    };
    assert_eq!(
        super::format_fetch_text(&page),
        "https://example.com/a\nExample (128 bytes)\n\nhello web"
    );
    let untitled = FetchedPage {
        title: None,
        ..page
    };
    assert!(super::format_fetch_text(&untitled).contains("(no title)"));
}

#[test]
fn fetch_text_truncates_huge_pages_with_notice() {
    use xencode_analysis_rs::web::FetchedPage;
    let page = FetchedPage {
        url: "https://example.com/big".to_string(),
        title: Some("Big".to_string()),
        text: "z".repeat(super::FETCH_TEXT_CAP_CHARS + 10),
        bytes: 99999,
    };
    let out = super::format_fetch_text(&page);
    assert!(out.contains("…[truncated to"), "{out}");
    assert!(out.contains("--format json"), "{out}");
    assert!(!out.contains(&"z".repeat(super::FETCH_TEXT_CAP_CHARS + 10)));
}

#[test]
fn json_schema_rejects_invalid_json() {
    let ok = super::parse_json_schema(Some(r#"{"type":"object"}"#.to_string())).unwrap();
    assert_eq!(ok, Some(serde_json::json!({"type": "object"})));
    assert_eq!(super::parse_json_schema(None).unwrap(), None);
    let err = super::parse_json_schema(Some("{broken".to_string())).unwrap_err();
    assert!(err.contains("invalid --json-schema"), "{err}");
}

#[test]
fn a_declared_schema_decides_what_counts_as_an_answer() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["files"],
        "properties": {"files": {"type": "array", "items": {"type": "string"}}}
    });
    assert!(super::structured_answer_fits(
        "{\"files\": [\"a.rs\"]}\n",
        Some(&schema)
    ));
    // Nothing declared, nothing to fit.
    assert!(super::structured_answer_fits("any prose at all", None));
    // Prose, the wrong shape of JSON, and an empty reply all fail.
    assert!(!super::structured_answer_fits(
        "here you go: {\"files\": \"a.rs\"}",
        Some(&schema)
    ));
    assert!(!super::structured_answer_fits(
        r#"{"files": "a.rs"}"#,
        Some(&schema)
    ));
    assert!(!super::structured_answer_fits("", Some(&schema)));
}

#[test]
fn review_text_summarizes_files_and_notes() {
    let files = vec![
        super::ReviewedFile {
            path: "src/a.rs".to_string(),
            added: Some(10),
            deleted: Some(2),
            issues: vec![],
            note: None,
        },
        super::ReviewedFile {
            path: "assets/logo.png".to_string(),
            added: None,
            deleted: None,
            issues: vec![],
            note: Some("binary".to_string()),
        },
    ];
    let out = super::format_review_text("main", None, &files);
    assert!(
        out.contains("Review of diff main...HEAD (2 files)"),
        "{out}"
    );
    assert!(out.contains("src/a.rs (+10 -2)"), "{out}");
    assert!(out.contains("assets/logo.png (binary)"), "{out}");

    let suffixed = super::format_review_text(
        "origin/main",
        Some(" [base resolved from origin/HEAD]"),
        &files,
    );
    assert!(
        suffixed.contains(
            "Review of diff origin/main...HEAD (2 files) [base resolved from origin/HEAD]"
        ),
        "{suffixed}"
    );
}

#[test]
fn resolve_review_base_explicit_and_fallbacks() {
    let dir = std::env::temp_dir().join(format!(
        "xencode-cli-base-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    // 1. Explicit base always wins
    let (base, src) = super::resolve_review_base(&dir, Some("custom-branch"));
    assert_eq!(base, "custom-branch");
    assert_eq!(src, super::BaseSource::Explicit);
    assert_eq!(src.header_suffix(), "");

    // Outside a repo: falls back to "main"
    let (base, src) = super::resolve_review_base(&dir, None);
    assert_eq!(base, "main");
    assert_eq!(src, super::BaseSource::DefaultFallback);
    assert_eq!(
        src.header_suffix(),
        " [no remote; fell back to default 'main']"
    );

    // Inside a repo with only master branch:
    let git = |args: &[&str]| {
        assert!(std::process::Command::new("git")
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap()
            .status
            .success());
    };
    git(&["init", "-q", "-b", "master"]);
    git(&["config", "user.email", "test@xencode.local"]);
    git(&["config", "user.name", "Xencode Test"]);
    git(&["commit", "--allow-empty", "-q", "-m", "initial"]);

    let (base, src) = super::resolve_review_base(&dir, None);
    assert_eq!(base, "master");
    assert_eq!(
        src,
        super::BaseSource::InitDefaultBranch("master".to_string())
    );
    assert_eq!(
        src.header_suffix(),
        " [no remote; fell back to init.defaultBranch (master)]"
    );

    // With origin/HEAD pointing to origin/main:
    git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    git(&[
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
    ]);
    let (base, src) = super::resolve_review_base(&dir, None);
    assert_eq!(base, "origin/main");
    assert_eq!(src, super::BaseSource::OriginHead);
    assert_eq!(src.header_suffix(), " [base resolved from origin/HEAD]");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn review_file_notes_missing_and_binary() {
    let dir = std::env::temp_dir().join(format!(
        "xencode-cli-review-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // Deleted from the working tree: note, not silence, not panic.
    let gone = super::review_file(
        &dir,
        &xencode_context_rs::DiffFile {
            path: "gone.rs".to_string(),
            added: Some(1),
            deleted: Some(1),
        },
    );
    assert!(gone.note.is_some(), "{gone:?}");
    // Binary content: analyzer cannot read it → note.
    std::fs::write(dir.join("blob.o"), [0xFF, 0xFE, 0x00]).unwrap();
    let bin = super::review_file(
        &dir,
        &xencode_context_rs::DiffFile {
            path: "blob.o".to_string(),
            added: None,
            deleted: None,
        },
    );
    assert!(bin.note.is_some(), "{bin:?}");
    // Plain code analyzes.
    std::fs::write(dir.join("ok.rs"), "pub fn f() {}\n").unwrap();
    let ok = super::review_file(
        &dir,
        &xencode_context_rs::DiffFile {
            path: "ok.rs".to_string(),
            added: Some(1),
            deleted: Some(0),
        },
    );
    assert!(ok.note.is_none(), "{ok:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn review_root_prefers_git_toplevel() {
    let dir = std::env::temp_dir().join(format!(
        "xencode-cli-root-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    // Outside any repo: falls back to cwd itself.
    assert_eq!(
        super::resolve_review_root(&dir.join("sub")),
        dir.join("sub")
    );
    assert_eq!(super::resolve_review_root(&dir), dir);
    // Inside a repo: the toplevel, even from a subdirectory.
    let git = |args: &[&str]| {
        assert!(std::process::Command::new("git")
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap()
            .status
            .success());
    };
    git(&["init", "-q"]);
    let toplevel = super::resolve_review_root(&dir.join("sub"));
    let canon = |p: std::path::PathBuf| p.canonicalize().unwrap();
    assert_eq!(canon(toplevel), canon(dir.clone()));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A stream line as a script receives it: the rendered text must hold
/// exactly one JSON object, whatever the answer contained.
fn stream_line(event: &serde_json::Value) -> serde_json::Value {
    let text = event.to_string();
    assert!(
        !text.contains('\n'),
        "a stream line held a newline: {text:?}"
    );
    serde_json::from_str(&text).expect("a stream line is not valid JSON")
}

#[test]
fn every_stream_event_names_the_schema_version_and_its_kind() {
    let events = vec![
        super::query_stream::start("llama:", "llamacpp", "local", None),
        super::query_stream::token("hi"),
        super::query_stream::done("hi", false, 5, None, None),
        super::query_stream::error("Query failed: nobody is listening"),
    ];
    for event in &events {
        let parsed = stream_line(event);
        assert_eq!(parsed["v"], super::query_stream::VERSION, "{parsed}");
        assert!(parsed["type"].is_string(), "{parsed}");
    }
    let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, vec!["start", "token", "done", "error"]);
}

#[test]
fn an_answer_full_of_newlines_and_quotes_still_fits_on_one_line() {
    // The reason a script can read this stream by splitting on newlines.
    let answer = "two\nlines \"quoted\"\ttab 日本語 \\ backslash";
    assert_eq!(
        stream_line(&super::query_stream::token(answer))["text"],
        answer
    );
    assert_eq!(
        stream_line(&super::query_stream::done(answer, false, 1, None, None))["answer"],
        answer
    );
}

#[test]
fn token_lines_rebuilt_in_order_are_the_answer_the_done_line_reports() {
    // The invariant a piping script depends on, including for an answer
    // the cache served without generating anything.
    let pieces = ["Hel", "lo\n", "世界"];
    let answer: String = pieces.concat();
    let rebuilt: String = pieces
        .iter()
        .map(|piece| {
            stream_line(&super::query_stream::token(piece))["text"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(rebuilt, answer);
}

#[test]
fn the_start_line_says_which_route_was_chosen_and_says_so_when_there_is_no_session() {
    let with_session = stream_line(&super::query_stream::start(
        "qwen2.5:7b",
        "ollama",
        "local",
        Some("s-1"),
    ));
    assert_eq!(with_session["model"], "qwen2.5:7b");
    assert_eq!(with_session["provider"], "ollama");
    assert_eq!(with_session["session"], "s-1");
    let without = stream_line(&super::query_stream::start(
        "anthropic:claude",
        "anthropic",
        "cloud",
        None,
    ));
    assert!(
        without["session"].is_null(),
        "a run with no conversation said otherwise: {without}"
    );
}

#[test]
fn a_done_line_leaves_out_the_counts_no_route_reported() {
    let measured = stream_line(&super::query_stream::done(
        "hi",
        false,
        1_537,
        Some(42),
        Some(6.5432),
    ));
    assert_eq!(measured["tokens_generated"], 42);
    assert_eq!(
        measured["tokens_per_second"], 6.54,
        "rate should round plainly"
    );
    assert_eq!(measured["elapsed_ms"], 1_537);
    assert_eq!(measured["cached"], false);
    let unmeasured = stream_line(&super::query_stream::done("hi", true, 3, None, None));
    assert!(unmeasured["tokens_generated"].is_null(), "{unmeasured}");
    assert!(unmeasured["tokens_per_second"].is_null(), "{unmeasured}");
    assert_eq!(unmeasured["cached"], true);
}

/// Images go out as parts on the final user turn, with the prompt text
/// ahead of them — the shape vision-capable endpoints accept.
#[test]
fn images_attach_as_parts_on_the_final_user_message() {
    let mut msgs = vec![
        ChatMessage::text("system", "be brief"),
        ChatMessage::text("user", "what is this?"),
    ];
    assert!(attach_images_to_final_user_message(
        &mut msgs,
        vec!["data:image/jpeg;base64,AAAA".to_string()]
    ));
    assert_eq!(
        msgs[1].content,
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what is this?".to_string()
            },
            ContentPart::ImageUrl {
                image_url: xencode_providers_rs::ImageUrlPart {
                    url: "data:image/jpeg;base64,AAAA".to_string(),
                    detail: None,
                }
            },
        ])
    );
    assert_eq!(
        msgs[0].content,
        MessageContent::Text("be brief".to_string())
    );
}

/// Several images keep their order after the text.
#[test]
fn several_images_keep_their_order() {
    let mut msgs = vec![ChatMessage::text("user", "compare")];
    assert!(attach_images_to_final_user_message(
        &mut msgs,
        vec![
            "data:image/png;base64,ONE".to_string(),
            "data:image/png;base64,TWO".to_string()
        ]
    ));
    match &msgs[0].content {
        MessageContent::Parts(parts) => {
            assert_eq!(parts.len(), 3);
            assert_eq!(
                msgs[0].image_urls(),
                vec!["data:image/png;base64,ONE", "data:image/png;base64,TWO"]
            );
            assert_eq!(msgs[0].text_content(), "compare");
        }
        other => panic!("expected parts, got {other:?}"),
    }
}

/// A turn with no user message cannot carry images. Reported, not silently
/// dropped — a request that quietly loses the images would be answered from
/// the prompt alone and look like the model ignored the picture.
#[test]
fn images_are_refused_rather_than_dropped_when_there_is_no_user_turn() {
    let mut msgs = vec![ChatMessage::text("system", "be brief")];
    assert!(!attach_images_to_final_user_message(
        &mut msgs,
        vec!["data:image/png;base64,AAAA".to_string()]
    ));
    let mut empty: Vec<ChatMessage> = Vec::new();
    assert!(!attach_images_to_final_user_message(
        &mut empty,
        vec!["data:image/png;base64,AAAA".to_string()]
    ));
}

/// An empty prompt alongside images still sends the images.
#[test]
fn images_go_out_even_when_the_text_part_is_empty() {
    let mut msgs = vec![ChatMessage::text("user", "")];
    assert!(attach_images_to_final_user_message(
        &mut msgs,
        vec!["data:image/png;base64,AAAA".to_string()]
    ));
    assert_eq!(msgs[0].image_urls(), vec!["data:image/png;base64,AAAA"]);
    assert_eq!(msgs[0].text_content(), "");
}

/// The flag parses, repeats, and is absent unless asked for.
#[test]
fn the_image_flag_parses_and_repeats() {
    let image_paths = |args: &[&str]| -> Vec<String> {
        match Cli::parse_from(args.iter().copied()).command {
            Some(Commands::Query { images, .. }) => images,
            _ => panic!("expected a query command"),
        }
    };
    assert_eq!(
        image_paths(&["xencode", "query", "--image", "a.png", "hello"]),
        vec!["a.png".to_string()]
    );
    assert_eq!(
        image_paths(&["xencode", "query", "--image", "a.png", "--image", "b.jpg", "hello"]),
        vec!["a.png".to_string(), "b.jpg".to_string()]
    );
    assert!(image_paths(&["xencode", "query", "hello"]).is_empty());
}

/// A file that is not an image, or cannot be read, is named and refused
/// before any request goes out.
#[test]
fn an_unusable_image_is_named_and_refused() {
    let dir = std::env::temp_dir().join(format!("xe-img-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("fake.png");
    std::fs::write(&fake, b"not an image").unwrap();
    let reason = super::encode_query_image(fake.to_str().unwrap()).unwrap_err();
    assert_eq!(
        reason,
        format!("{} is not a recognized image", fake.display())
    );

    let gone = dir.join("gone.png");
    let reason = super::encode_query_image(gone.to_str().unwrap()).unwrap_err();
    assert!(
        reason.starts_with(&format!("cannot read image {}", gone.display())),
        "{reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A real image becomes a data URL the vision path can send.
#[test]
fn a_real_image_becomes_a_data_url() {
    let dir = std::env::temp_dir().join(format!("xe-img-ok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shot.png");
    std::fs::write(&path, ONE_PIXEL_PNG).unwrap();
    let url = super::encode_query_image(path.to_str().unwrap()).unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "{url}");
    // The payload decodes back to real image bytes, not a header alone: the
    // PNG magic is in there and the format is still recognisable. Checked on
    // the encoded text so the assertion is about what goes on the wire.
    let (_, encoded) = url.split_once(";base64,").unwrap();
    let raw = decode_base64(encoded);
    assert_eq!(&raw[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(
        xencode_analysis_rs::images::detect_format(&raw),
        Some(ImageFormat::Png)
    );
    assert!(raw.len() > 8, "only a signature was sent");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_source_word_is_the_same_spelling_the_metrics_rows_use() {
    use xencode_context_rs::MetricSource;
    use xencode_providers_rs::Egress;
    let spelled = |source: MetricSource| serde_json::to_value(source).unwrap();
    assert_eq!(
        spelled(MetricSource::Local),
        super::source_word(Egress::Local),
        "the stream and cache/metrics.jsonl disagree about what a local route is called"
    );
    assert_eq!(
        spelled(MetricSource::Cloud),
        super::source_word(Egress::Cloud),
        "the stream and cache/metrics.jsonl disagree about what an off-machine route is called"
    );
}

/// The `nvidia` endpoint is resolved from `NVIDIA_NIM_API_KEY` as well as
/// from the configuration file, so a test cannot pin it without editing the
/// environment every other test reads. The rest are asserted on in full.
fn endpoint_names(config: &xencode_config_rs::XencodeConfig) -> Vec<&'static str> {
    super::endpoints(config)
        .iter()
        .filter(|endpoint| endpoint.name != "nvidia")
        .map(|endpoint| endpoint.name)
        .collect()
}

#[test]
fn a_config_with_no_keys_dials_only_the_two_local_servers() {
    let config = xencode_config_rs::XencodeConfig::default();
    assert_eq!(endpoint_names(&config), ["ollama", "llamacpp"]);

    // The start command is the row's fix, so it has to be the real one.
    let list = super::endpoints(&config);
    assert_eq!(list[0].start_command, Some("ollama serve"));
    assert_eq!(
        list[1].start_command,
        Some("llama-server --model <path>"),
        "the report must tell the reader to run the command that loads a model"
    );
    assert_eq!(list[0].url, config.ollama_url);
    assert_eq!(list[1].url, config.llama_cpp_url);
}

#[test]
fn a_configured_remote_and_a_keyed_cloud_join_the_dial_list() {
    let mut config = xencode_config_rs::XencodeConfig {
        remote_base_url: "http://127.0.0.1:18000/v1".to_string(),
        ..Default::default()
    };
    assert_eq!(endpoint_names(&config), ["ollama", "llamacpp", "remote"]);
    // A service the person may not have running is not something this
    // machine can be told to start, so there is no command to name.
    assert_eq!(super::endpoints(&config)[2].start_command, None);

    config.api_keys.openai_api_key = Some("sk-not-a-real-key".to_string());
    config.api_keys.google_gemini_api_key = Some("not-a-real-key".to_string());
    assert_eq!(
        endpoint_names(&config),
        ["ollama", "llamacpp", "remote", "openai", "gemini"],
        "clouds come after the local endpoints, in the order the code lists them"
    );
    let clouds: Vec<String> = super::endpoints(&config)
        .iter()
        .skip(3)
        .map(|endpoint| endpoint.url.clone())
        .collect();
    assert_eq!(
        clouds,
        [
            "https://api.openai.com".to_string(),
            "https://generativelanguage.googleapis.com".to_string()
        ],
        "the address dialled is the provider's own host, not one the report guessed"
    );
}

#[test]
fn a_key_that_is_only_whitespace_does_not_open_a_cloud_row() {
    let config = xencode_config_rs::XencodeConfig {
        api_keys: xencode_config_rs::ApiKeys {
            openrouter_api_key: Some("   ".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        endpoint_names(&config),
        ["ollama", "llamacpp"],
        "a saved blank key is not a configured provider"
    );
}

#[test]
fn the_bridge_rows_are_named_once_inside_the_report() {
    let report = xencode_colab_rs::PreflightReport {
        checks: vec![
            xencode_colab_rs::Check {
                name: "colab CLI",
                ok: true,
                detail: "/home/someone/.local/bin/colab".to_string(),
                fix: None,
            },
            xencode_colab_rs::Check {
                name: "Colab API",
                ok: false,
                detail: "`colab sessions` failed — not signed in".to_string(),
                fix: Some("gcloud auth application-default login".to_string()),
            },
            xencode_colab_rs::Check {
                name: "OpenSSH",
                ok: true,
                detail: "ssh at /usr/bin/ssh".to_string(),
                fix: None,
            },
        ],
    };
    let rows = super::bridge_rows(report);
    let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
    assert_eq!(names, ["colab:CLI", "colab:API", "colab:OpenSSH"]);
    assert_eq!(rows[1].state, "fail");
    assert_eq!(
        rows[1].fix.as_deref(),
        Some("gcloud auth application-default login"),
        "the remediation the bridge owns rides through unchanged"
    );
}

#[test]
fn a_refused_model_is_answered_with_the_command_that_would_work() {
    // The three ways Ollama can turn the default model away, each answered
    // with what the reader should actually run.
    use xencode_models_rs::OllamaError;
    assert_eq!(
        super::ollama_fix(
            &OllamaError::NotRunning("error sending request".to_string()),
            "ollama:qwen2.5:7b"
        ),
        "ollama serve"
    );
    // An id carrying the provider prefix is what the configuration can hold
    // and Ollama cannot serve: `ollama pull ollama:qwen2.5:7b` would fetch
    // nothing, so the row names the bare model instead.
    let prefixed = super::ollama_fix(
        &OllamaError::ModelNotFound("ollama:qwen2.5:7b".to_string()),
        "ollama:qwen2.5:7b",
    );
    assert!(
        prefixed.contains("set the default to `qwen2.5:7b`"),
        "{prefixed}"
    );
    assert!(prefixed.contains("`ollama pull qwen2.5:7b`"), "{prefixed}");
    assert!(
        !prefixed.contains("pull ollama:"),
        "the fix must not tell the reader to pull a prefixed name: {prefixed}"
    );
    assert_eq!(
        super::ollama_fix(
            &OllamaError::ModelNotFound("qwen3:4b".to_string()),
            "qwen3:4b"
        ),
        "ollama pull qwen3:4b"
    );
    // Anything else is the Ollama client's own problem, and the command that
    // explains it in full is the health check.
    assert_eq!(
        super::ollama_fix(&OllamaError::Timeout("20 s".to_string()), "qwen3:4b"),
        "xencode model health qwen3:4b"
    );
}

#[test]
fn the_json_report_omits_the_fix_key_only_where_there_is_nothing_to_fix() {
    let passing = doc::check_model(true, "qwen3:4b is loaded".to_string(), None);
    let json = serde_json::to_value(&passing).unwrap();
    assert!(
        json.get("fix").is_none(),
        "a reader counting failures should not have to tell a null from an action: {json}"
    );

    let failing = doc::check_model(
        false,
        "nothing listening on 11434".to_string(),
        Some("ollama serve".to_string()),
    );
    let json = serde_json::to_value(&failing).unwrap();
    assert_eq!(json["fix"], serde_json::json!("ollama serve"));
    assert_eq!(json["state"], serde_json::json!("fail"));
}

#[test]
fn the_colab_preview_names_the_keys_a_bring_up_would_rewrite() {
    let config = xencode_config_rs::XencodeConfig::default();
    let lines = super::colab_up_config_preview(&config, "llama.cpp", 18000);
    let text = lines.join("\n");
    assert!(
        text.contains("llama_cpp_url: http://localhost:8080 → http://127.0.0.1:18000"),
        "{text}"
    );
    assert!(
        text.contains("remote_base_url:  → http://127.0.0.1:18000/v1"),
        "{text}"
    );
    assert!(
        !text.contains("ollama_url"),
        "a llama.cpp forward does not move Ollama's endpoint: {text}"
    );

    let ollama = super::colab_up_config_preview(&config, "ollama", 19000);
    let text = ollama.join("\n");
    assert!(text.contains("ollama_url"), "{text}");
    assert!(text.contains("http://127.0.0.1:19000"), "{text}");
}

#[test]
fn the_colab_preview_says_so_when_the_config_already_points_at_the_forward() {
    let mut config = xencode_config_rs::XencodeConfig::default();
    xencode_colab_rs::point_config_at_forward(&mut config, "llama.cpp", "http://127.0.0.1:18000");
    let lines = super::colab_up_config_preview(&config, "llama.cpp", 18000);
    assert_eq!(
        lines,
        vec!["config.json would not change".to_string()],
        "an up that changes nothing says that, instead of printing a diff of nothing"
    );
}

#[test]
fn compete_run_parses_arms_edits_and_commands() {
    let cli = Cli::try_parse_from([
        "xencode",
        "compete",
        "run",
        "Loop or recurse?",
        "--arm",
        "loop=Iterative",
        "--arm",
        "recur",
        "--edit",
        "loop",
        "src/lib.rs",
        "pub fn a() { while false {} }",
        "--command",
        "recur",
        "touch READY",
        "--skip",
        "test",
        "--timeout",
        "30",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Compete {
            action:
                super::CompeteAction::Run {
                    prompt,
                    arms,
                    edits,
                    commands,
                    skip,
                    timeout,
                    format,
                },
        }) => {
            assert_eq!(prompt, "Loop or recurse?");
            assert_eq!(arms, ["loop=Iterative", "recur"]);
            assert_eq!(
                edits,
                ["loop", "src/lib.rs", "pub fn a() { while false {} }"]
            );
            assert_eq!(commands, ["recur", "touch READY"]);
            assert_eq!(skip, ["test"]);
            assert_eq!(timeout, 30);
            assert!(matches!(format, OutputFormat::Text));
        }
        _ => panic!("expected compete run"),
    }

    let specs = super::build_candidate_arms(
        &["loop=Iterative".to_string(), "recur".to_string()],
        &[
            "loop".to_string(),
            "src/lib.rs".to_string(),
            "fn a() {}".to_string(),
        ],
        &["recur".to_string(), "touch READY".to_string()],
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].arm_id, "loop");
    assert_eq!(specs[0].label, "Iterative");
    assert_eq!(
        specs[0].file_edits[0].0,
        std::path::PathBuf::from("src/lib.rs")
    );
    assert_eq!(specs[1].arm_id, "recur");
    // An arm named without a label is called by its id, not left blank.
    assert_eq!(specs[1].label, "recur");
    assert_eq!(specs[1].command.as_deref(), Some("touch READY"));

    // Omitting --arm competes the two default arms.
    let defaults = super::build_candidate_arms(&[], &[], &[]).unwrap();
    assert_eq!(
        defaults
            .iter()
            .map(|s| s.arm_id.as_str())
            .collect::<Vec<_>>(),
        ["arm-a", "arm-b"]
    );
}

/// An edit is a write into a worktree, so the paths it names are checked
/// before anything is built: no escaping the arm's own tree, no writing to
/// an arm nobody declared.
#[test]
fn compete_refuses_edits_that_escape_or_name_no_arm() {
    let escape = super::build_candidate_arms(
        &["a".to_string(), "b".to_string()],
        &[
            "a".to_string(),
            "../escape.txt".to_string(),
            "x".to_string(),
        ],
        &[],
    )
    .unwrap_err();
    assert!(escape.contains("escapes the worktree"), "{escape}");

    let absolute = super::build_candidate_arms(
        &["a".to_string(), "b".to_string()],
        &["a".to_string(), "/etc/passwd".to_string(), "x".to_string()],
        &[],
    )
    .unwrap_err();
    assert!(absolute.contains("escapes the worktree"), "{absolute}");

    let unknown = super::build_candidate_arms(
        &["a".to_string(), "b".to_string()],
        &["c".to_string(), "src/lib.rs".to_string(), "x".to_string()],
        &[],
    )
    .unwrap_err();
    assert!(unknown.contains("known arms: a, b"), "{unknown}");

    let dupe = super::build_candidate_arms(
        &["a".to_string(), "a".to_string()],
        &[],
        &["a".to_string(), "true".to_string()],
    )
    .unwrap_err();
    assert!(dupe.contains("given twice"), "{dupe}");

    let two_cmds = super::build_candidate_arms(
        &["a".to_string(), "b".to_string()],
        &[],
        &[
            "a".to_string(),
            "touch one".to_string(),
            "a".to_string(),
            "touch two".to_string(),
        ],
    )
    .unwrap_err();
    assert!(two_cmds.contains("already has a command"), "{two_cmds}");

    let blank = super::build_candidate_arms(&["=no id".to_string(), "b".to_string()], &[], &[])
        .unwrap_err();
    assert!(blank.contains("has no id"), "{blank}");
}

#[test]
fn compete_list_show_and_pick_parse_their_ids() {
    let cli = Cli::try_parse_from(["xencode", "compete", "list"]).unwrap();
    assert!(matches!(
        cli.command,
        Some(Commands::Compete {
            action: super::CompeteAction::List { .. }
        })
    ));

    let cli = Cli::try_parse_from([
        "xencode",
        "compete",
        "show",
        "compete-1",
        "--format",
        "json",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Compete {
            action: super::CompeteAction::Show { run_id, format },
        }) => {
            assert_eq!(run_id, "compete-1");
            assert!(matches!(format, OutputFormat::Json));
        }
        _ => panic!("expected compete show"),
    }

    let cli = Cli::try_parse_from(["xencode", "compete", "pick", "compete-1", "loop"]).unwrap();
    match cli.command {
        Some(Commands::Compete {
            action: super::CompeteAction::Pick { run_id, arm_id },
        }) => {
            assert_eq!(run_id, "compete-1");
            assert_eq!(arm_id, "loop");
        }
        _ => panic!("expected compete pick"),
    }

    // `run` cannot be invoked without the question it is answering.
    assert!(Cli::try_parse_from(["xencode", "compete", "run"]).is_err());
}

#[test]
fn cli_merge_parsing() {
    use super::MergeAction;

    // Precheck
    let cli = Cli::try_parse_from(["xencode", "merge", "precheck", "feat-1", "--base", "master"])
        .unwrap();
    match cli.command {
        Some(Commands::Merge {
            action: MergeAction::Precheck { branch, base, .. },
        }) => {
            assert_eq!(branch, "feat-1");
            assert_eq!(base, "master");
        }
        _ => panic!("expected merge precheck"),
    }

    // Plan
    let cli = Cli::try_parse_from([
        "xencode", "merge", "plan", "--branch", "feat-1", "--branch", "feat-2", "--base", "master",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Merge {
            action: MergeAction::Plan { branches, base, .. },
        }) => {
            assert_eq!(branches, vec!["feat-1", "feat-2"]);
            assert_eq!(base, "master");
        }
        _ => panic!("expected merge plan"),
    }

    // Land
    let cli = Cli::try_parse_from([
        "xencode",
        "merge",
        "land",
        "--branch",
        "feat-1",
        "--branch",
        "feat-2",
        "--approved-by",
        "Alice",
        "--test-cmd",
        "cargo test",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Merge {
            action:
                MergeAction::Land {
                    branches,
                    approved_by,
                    test_cmds,
                    ..
                },
        }) => {
            assert_eq!(branches, vec!["feat-1", "feat-2"]);
            assert_eq!(approved_by, "Alice");
            assert_eq!(test_cmds, vec!["cargo test"]);
        }
        _ => panic!("expected merge land"),
    }

    // Land without approved-by fails (required human gate)
    assert!(Cli::try_parse_from(["xencode", "merge", "land", "--branch", "feat-1",]).is_err());
}
