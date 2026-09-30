//! `xencode plugin install <git-url>`, `update` and `remove`, driven as a user
//! drives them: a separate process running the real binary against a real
//! repository it cloned over the `file://` transport.
//!
//! What is asserted is the order things are said in, not only that they are
//! said. A plugin's prompt prefix reaches the model on every turn, so the
//! install must show that text before the copy lands where the loader scans, and
//! an update must show the difference before it is applied rather than instead
//! of.

use std::path::{Path, PathBuf};
use std::process::Command;

use xencode_plugin_rs::SOURCE_FILE;

fn git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args([
            "-c",
            "user.name=Test Author",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git ran");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A committed plugin repository holding `manifest`, at `path`.
fn make_repo(dir: &Path, manifest: &str) -> PathBuf {
    let repo = dir.join("guardrails-src");
    std::fs::create_dir_all(&repo).expect("repo dir");
    std::fs::write(repo.join("plugin.json"), manifest).expect("manifest");
    git(&repo, &["init", "-q", "-b", "main", "."]);
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-qm", "initial"]);
    repo
}

fn commit_manifest(repo: &Path, manifest: &str) {
    std::fs::write(repo.join("plugin.json"), manifest).expect("manifest");
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-qm", "revision"]);
}

fn head(repo: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .expect("git ran");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Run the built binary with the plugin directory pointed at `plugins`.
fn xencode(plugins: &Path, args: &[&str]) -> (String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_PLUGIN_DIR", plugins)
        .output()
        .expect("the xencode binary ran");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (text, output.status.success())
}

/// A manifest with `prefix` as its prompt text. The prefix is escaped the way a
/// hand-written manifest would escape it, so a two-line prompt is a JSON
/// `\\n` inside the string rather than a raw newline, which JSON forbids.
fn manifest(version: &str, prefix: &str) -> String {
    let prefix = prefix
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!(
        r#"{{
  "name": "guardrails",
  "version": "{version}",
  "description": "Refuses the sloppy answer",
  "author": "Test Author",
  "license": "MIT",
  "dependencies": [],
  "xencode_version": "*",
  "permissions": ["prompt"],
  "prompt_prefix": "{prefix}"
}}"#
    )
}

#[test]
fn an_install_declares_before_it_copies_and_names_the_commit_it_pinned() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plugins = tmp.path().join("plugins");
    let repo = make_repo(
        tmp.path(),
        &manifest("1.0.0", "Run the tests before answering."),
    );

    let (text, ok) = xencode(
        &plugins,
        &["plugin", "install", &format!("file://{}", repo.display())],
    );
    assert!(ok, "{text}");
    // The declaration of what reaches the model comes first, in the same output.
    let declared = text.find("What it declares:").expect(&text);
    let installed = text.find("✅ Installed").expect(&text);
    assert!(declared < installed, "{text}");
    assert!(text.contains("| Run the tests before answering."), "{text}");
    // Unpinned by the user, pinned by the installer: the commit is named.
    let pinned = head(&repo);
    assert!(
        text.contains(&format!("Pinned to commit {}…", &pinned[..7])),
        "{text}"
    );

    // The full SHA is what is recorded, and no git metadata was copied.
    let record = std::fs::read_to_string(plugins.join("guardrails").join(SOURCE_FILE))
        .expect("source record");
    assert!(record.contains(&pinned), "{record}");
    assert!(
        !plugins.join("guardrails").join(".git").exists(),
        "the clone's .git was installed as part of the plugin"
    );

    // Loading it the way the TUI does shows the prefix and where it came from.
    let (list, ok) = xencode(&plugins, &["plugin", "list"]);
    assert!(ok, "{list}");
    assert!(list.contains("guardrails v1.0.0 — loaded."), "{list}");
    assert!(list.contains("| Run the tests before answering."), "{list}");
    assert!(
        list.contains(&format!(
            "from file://{} @ {}",
            repo.display(),
            &pinned[..7]
        )),
        "{list}"
    );
}

#[test]
fn an_update_that_changes_the_prompt_is_shown_as_a_diff_before_it_is_applied() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plugins = tmp.path().join("plugins");
    let repo = make_repo(
        tmp.path(),
        &manifest("1.0.0", "Run the tests before answering."),
    );
    xencode(
        &plugins,
        &["plugin", "install", &format!("file://{}", repo.display())],
    );
    let first = head(&repo);
    let installed_path = plugins.join("guardrails").join("plugin.json");
    let before = std::fs::read_to_string(&installed_path).expect("installed manifest");

    commit_manifest(
        &repo,
        &manifest(
            "1.1.0",
            "Run the tests before answering.\nNever widen a permission.",
        ),
    );
    let second = head(&repo);
    assert_ne!(first, second);

    // Show me, don't apply it.
    let (text, ok) = xencode(&plugins, &["plugin", "update", "guardrails"]);
    assert!(ok, "{text}");
    assert!(text.contains("NOT APPLIED"), "{text}");
    assert!(
        text.contains("changes its prompt prefix (1 line(s) → 2)"),
        "{text}"
    );
    assert!(
        text.contains("-  \"prompt_prefix\": \"Run the tests before answering.\""),
        "{text}"
    );
    assert!(
        text.contains("+  \"prompt_prefix\": \"Run the tests before answering.\\nNever widen"),
        "{text}"
    );
    assert_eq!(
        std::fs::read_to_string(&installed_path).expect("still installed"),
        before,
        "the update was applied without being acknowledged"
    );

    // Acknowledged.
    let (text, ok) = xencode(&plugins, &["plugin", "update", "guardrails", "--yes"]);
    assert!(ok, "{text}");
    assert!(text.contains("✅ Applied: v1.0.0 → v1.1.0"), "{text}");
    let after = std::fs::read_to_string(&installed_path).expect("updated manifest");
    assert!(after.contains("Never widen a permission."), "{after}");
    let record =
        std::fs::read_to_string(plugins.join("guardrails").join(SOURCE_FILE)).expect("record");
    assert!(record.contains(&second), "{record}");
    // The swap left no backup directory behind.
    assert!(
        std::fs::read_dir(&plugins)
            .expect("plugins dir")
            .filter_map(|entry| entry.ok())
            .all(|entry| !entry.file_name().to_string_lossy().starts_with('.')),
        "a staging directory is still in the plugin directory"
    );

    // Asking again reports the truth: there is nothing new.
    let (text, ok) = xencode(&plugins, &["plugin", "update", "guardrails"]);
    assert!(ok, "{text}");
    assert!(text.contains("Already up to date"), "{text}");

    // Removal names what was being contributed, and then it is gone.
    let (text, ok) = xencode(&plugins, &["plugin", "remove", "guardrails"]);
    assert!(ok, "{text}");
    assert!(text.contains("It was contributing:"), "{text}");
    assert!(text.contains("| Never widen a permission."), "{text}");
    assert!(!plugins.join("guardrails").exists());
    let (text, ok) = xencode(&plugins, &["plugin", "list"]);
    assert!(ok, "{text}");
    assert!(text.contains("No plugins installed"), "{text}");
}

#[test]
fn a_repository_that_is_not_a_plugin_and_a_second_copy_of_one_are_both_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plugins = tmp.path().join("plugins");

    let plain = tmp.path().join("plain-src");
    std::fs::create_dir_all(&plain).expect("repo dir");
    std::fs::write(plain.join("README.md"), "nothing to load").expect("readme");
    git(&plain, &["init", "-q", "-b", "main", "."]);
    git(&plain, &["add", "-A"]);
    git(&plain, &["commit", "-qm", "initial"]);

    let (text, ok) = xencode(
        &plugins,
        &["plugin", "install", &format!("file://{}", plain.display())],
    );
    assert!(!ok, "{text}");
    assert!(text.contains("that repository is not a plugin"), "{text}");
    assert!(
        std::fs::read_dir(&plugins)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(true),
        "a refused install still wrote into the plugin directory"
    );

    let repo = make_repo(
        tmp.path(),
        &manifest("1.0.0", "Run the tests before answering."),
    );
    let url = format!("file://{}", repo.display());
    let (text, ok) = xencode(&plugins, &["plugin", "install", &url]);
    assert!(ok, "{text}");
    let (text, ok) = xencode(&plugins, &["plugin", "install", &url]);
    assert!(!ok, "installing the same plugin twice succeeded: {text}");
    assert!(text.contains("already installed"), "{text}");
    assert!(text.contains("plugin update guardrails"), "{text}");
}

#[test]
fn an_update_to_a_named_commit_pins_the_plugin_there_and_a_branch_moves_it() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plugins = tmp.path().join("plugins");
    let repo = make_repo(
        tmp.path(),
        &manifest("1.0.0", "Run the tests before answering."),
    );
    let first = head(&repo);

    let (text, ok) = xencode(
        &plugins,
        &[
            "plugin",
            "install",
            &format!("file://{}", repo.display()),
            "--rev",
            &first,
        ],
    );
    assert!(ok, "{text}");
    // Pinned to the exact commit named on the command line.
    assert!(
        text.contains(&format!("Pinned to commit {}…", &first[..7])),
        "{text}"
    );

    commit_manifest(
        &repo,
        &manifest(
            "1.1.0",
            "Run the tests before answering.\nNever widen a permission.",
        ),
    );

    // A plugin pinned to a commit stays there when the branch moves on.
    let (text, ok) = xencode(&plugins, &["plugin", "update", "guardrails"]);
    assert!(ok, "{text}");
    assert!(text.contains("Already up to date"), "{text}");

    // Naming the branch is what asks for the newer commit.
    let (text, ok) = xencode(
        &plugins,
        &["plugin", "update", "guardrails", "--rev", "main"],
    );
    assert!(ok, "{text}");
    assert!(text.contains("NOT APPLIED"), "{text}");
    assert!(
        text.contains(&format!("at {}…", &head(&repo)[..7])),
        "{text}"
    );

    // A rev that names nothing at all is refused with the repository in the
    // message, not an internal git invocation.
    let (text, ok) = xencode(
        &plugins,
        &["plugin", "install", &url_for(&repo), "--rev", "nope"],
    );
    assert!(!ok, "{text}");
    assert!(
        text.contains("has no branch, tag or commit named \"nope\""),
        "{text}"
    );
    assert!(!text.contains("git -C"), "{text}");
}

fn url_for(repo: &Path) -> String {
    format!("file://{}", repo.display())
}
