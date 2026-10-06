//! Build/test autodiscovery (`WF-4`).
//!
//! Probes a repository for the commands that build, check and test it, then
//! *runs* the candidates and records only the ones that exited zero. The
//! verified result becomes `.xencode/anchor.md`.
//!
//! Two properties of that destination drive the whole design:
//!
//! - **The output must be deterministic.** `anchor.md` is read into the
//!   byte-stable head ([`crate::context::stable_system_text`]) that every
//!   request begins with, and that head is what makes a llama.cpp KV prefix
//!   reusable. A timestamp, an absolute path or a duration in this file would
//!   change the head on every run and quietly destroy the cache it exists to
//!   protect. So the render contains no clock, no machine-specific path, and no
//!   ordering that depends on the filesystem.
//! - **Nothing is claimed before it is run.** Discovery finds candidates; only
//!   [`prove`] promotes one. A recipe that is merely *declared* is rendered as
//!   declared, because the failure this guards against is the one this project
//!   keeps refusing to accept: a green tick standing in for a command nobody
//!   executed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::init::{ContextError, XENCODE_DIR};

/// What a discovered command is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Formatting check.
    Format,
    /// Linter or type check.
    Lint,
    /// Compilation.
    Build,
    /// The suite. This is the one `WF-4`'s done-when is about.
    Test,
}

impl Kind {
    /// The heading used in `anchor.md`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Format => "format",
            Self::Lint => "lint",
            Self::Build => "build",
            Self::Test => "test",
        }
    }

    /// Classify a command line by what it runs, longest-signal-first so
    /// `cargo clippy` is lint and not build.
    pub fn classify(command: &str) -> Option<Self> {
        let c = command.to_ascii_lowercase();
        let has = |needles: &[&str]| needles.iter().any(|n| c.contains(n));
        if has(&[
            "cargo fmt",
            "rustfmt",
            "gofmt",
            "prettier --check",
            "npm run format",
        ]) {
            Some(Self::Format)
        } else if has(&[
            "clippy",
            "eslint",
            "ruff ",
            "golangci",
            "cargo check",
            "tsc ",
        ]) {
            Some(Self::Lint)
        } else if has(&[
            "cargo test",
            "pytest",
            "npm test",
            "yarn test",
            "pnpm test",
            "go test",
            "make test",
            "just test",
            "vitest",
            "jest",
        ]) {
            Some(Self::Test)
        } else if has(&[
            "cargo build",
            "cargo check --all",
            "npm run build",
            "go build",
            "make build",
            "just build",
        ]) {
            Some(Self::Build)
        } else {
            None
        }
    }
}

/// Where a candidate came from. The order is the order of trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// Read out of CI configuration. The strongest signal available, because
    /// it is the command the repository already refuses to merge without.
    ContinuousIntegration,
    /// A task runner declares it as a target (`justfile`, `Makefile`, `mise`).
    TaskRunner,
    /// A manifest implies it (`package.json` scripts, a Cargo workspace).
    Manifest,
    /// Found as a fenced command block in the README.
    Readme,
    /// Nothing in the repository said so; derived from the project type alone.
    Inferred,
}

impl Provenance {
    /// How the claim should be read in the rendered anchor.
    pub fn caveat(self) -> &'static str {
        match self {
            Self::ContinuousIntegration => "read from CI, which already gates this repository",
            Self::TaskRunner => "declared by a task runner",
            Self::Manifest => "implied by a project manifest",
            Self::Readme => "found in the README",
            Self::Inferred => "not stated anywhere in the repository",
        }
    }
}

/// A command that might build or check the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    /// What it is for.
    pub kind: Kind,
    /// The command line to run.
    pub command: String,
    /// Repository-relative file it was read from, or `inferred`.
    pub source: String,
    /// How much the source is worth.
    pub provenance: Provenance,
    /// Filled in by [`prove`]. `None` means the command has not been run.
    pub verdict: Option<Verdict>,
}

impl Recipe {
    /// `true` only when the command was executed and returned success.
    pub fn is_verified(&self) -> bool {
        matches!(self.verdict, Some(Verdict::Passed))
    }
}

/// The result of actually running a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Exited zero. The only outcome that earns the word "works".
    Passed,
    /// Exited non-zero.
    Failed(i32),
    /// Still running when the budget ran out. Deliberately *not* a pass: a
    /// command nobody saw finish has proved nothing, and reporting it as green
    /// is the exact false confidence this item exists to prevent.
    TimedOut,
    /// The command could not be started at all.
    NotLaunchable(String),
}

impl Verdict {
    /// The line a reader needs, including the reason.
    pub fn describe(&self) -> String {
        match self {
            Self::Passed => "verified — ran it, it exited 0".to_string(),
            Self::Failed(code) => format!("FAILED — exited {code}"),
            Self::TimedOut => {
                "not verified — still running when the time budget ran out".to_string()
            }
            Self::NotLaunchable(why) => format!("not verified — {why}"),
        }
    }
}

/// Everything discovery and proof found.
#[derive(Debug, Clone, Default)]
pub struct Discovery {
    /// One entry per distinct command, in a stable order.
    pub recipes: Vec<Recipe>,
    /// Files that were read while looking, in a stable order.
    pub probed: Vec<String>,
}

impl Discovery {
    /// The verified recipe for `kind`, if one was proven.
    pub fn verified(&self, kind: Kind) -> Option<&Recipe> {
        self.recipes
            .iter()
            .find(|r| r.kind == kind && r.is_verified())
    }

    /// The best candidate for `kind` even when unproven, so the caller can
    /// report what was found without implying it works.
    pub fn candidate(&self, kind: Kind) -> Option<&Recipe> {
        self.recipes.iter().find(|r| r.kind == kind)
    }

    /// Whether anything at all was proven. An anchor claiming a test command
    /// with nothing behind it is worse than no anchor.
    pub fn has_proof(&self) -> bool {
        self.recipes.iter().any(Recipe::is_verified)
    }
}

/// Probe `root` for build, check and test commands.
///
/// Reads CI configuration, task runners, manifests and the README. Additions
/// are deduped by command, keeping the strongest provenance, and the result is
/// sorted so the same repository always yields the same ordering.
pub fn discover(root: &Path) -> Discovery {
    let mut found: BTreeMap<String, Recipe> = BTreeMap::new();
    let mut probed: Vec<String> = Vec::new();

    let mut take = |command: &str, source: &str, provenance: Provenance| {
        let command = command.trim();
        if command.is_empty() || command.starts_with('#') || command.contains('\n') {
            return;
        }
        let Some(kind) = Kind::classify(command) else {
            return;
        };
        found.entry(command.to_string()).or_insert_with(|| Recipe {
            kind,
            command: command.to_string(),
            source: source.to_string(),
            provenance,
            verdict: None,
        });
    };

    for (dir, file, provenance, extract) in SOURCES {
        let path = root.join(dir).join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        probed.push(format!("{dir}/{file}"));
        for command in extract(&text) {
            take(&command, &format!("{dir}/{file}"), *provenance);
        }
    }

    if let Some(dir) = read_dir_sorted(root.join(".github").join("workflows")) {
        for file in dir {
            let path = root.join(".github").join("workflows").join(&file);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            probed.push(format!(".github/workflows/{file}"));
            for command in run_lines(&text) {
                take(
                    &command,
                    &format!(".github/workflows/{file}"),
                    Provenance::ContinuousIntegration,
                );
            }
        }
    }

    for readme in ["README.md", "readme.md", "README.rst"] {
        let path = root.join(readme);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        probed.push(readme.to_string());
        for command in fenced_commands(&text) {
            take(&command, readme, Provenance::Readme);
        }
    }

    for (manifest, source) in inferred_from_manifests(root) {
        take(&manifest, &source, Provenance::Inferred);
    }

    Discovery {
        recipes: found.into_values().collect(),
        probed: probed.into_iter().collect(),
    }
}

/// One probe source: where the file is, how much it is trusted, and the
/// extractor that pulls command lines out of it.
type Source = (
    &'static str,
    &'static str,
    Provenance,
    fn(&str) -> Vec<String>,
);

/// The fixed probe order. Deliberately a constant: discovery must not depend on
/// which optional files happen to exist in a directory listing.
const SOURCES: &[Source] = &[
    ("", "justfile", Provenance::TaskRunner, just_recipes),
    ("", "Justfile", Provenance::TaskRunner, just_recipes),
    ("", "makefile", Provenance::TaskRunner, make_targets),
    ("", "Makefile", Provenance::TaskRunner, make_targets),
    ("", "mise.toml", Provenance::TaskRunner, mise_tasks),
    ("", ".mise.toml", Provenance::TaskRunner, mise_tasks),
    (
        ".gitlab",
        "ci.yml",
        Provenance::ContinuousIntegration,
        run_lines,
    ),
];

/// Recipe names and their command bodies from a `justfile`.
///
/// A `justfile` recipe is a header line followed by an indented body, and the
/// body is the command — the header alone says nothing, so reading only the
/// header is how a `justfile` ends up looking empty when it is not.
fn just_recipes(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with(char::is_whitespace)
        {
            continue;
        }
        let Some((header, inline)) = line.split_once(':') else {
            continue;
        };
        let name = header.split_whitespace().next().unwrap_or_default().trim();
        if name.is_empty() || name.contains(char::is_whitespace) {
            continue;
        }
        let body = collect_body(&mut lines);
        let inline = inline.trim();
        if !inline.is_empty() && !inline.starts_with("//") {
            out.push(inline.to_string());
        } else if !body.is_empty() {
            out.push(body);
        } else if matches!(name, "test" | "check" | "lint" | "build" | "fmt") {
            out.push(format!("just {name}"));
        }
    }
    out
}

/// Targets and their recipes from a `Makefile`.
fn make_targets(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with('\t') {
            continue;
        }
        let Some((targets, inline)) = line.split_once(':') else {
            continue;
        };
        if inline.trim_start().starts_with('=') {
            continue;
        }
        let body = collect_body(&mut lines);
        let inline = inline.trim();
        for target in targets.split_whitespace() {
            if target.is_empty() || target.contains('=') || target.starts_with('.') {
                continue;
            }
            if !inline.is_empty() {
                out.push(inline.to_string());
            } else if !body.is_empty() {
                out.push(body.clone());
            } else if matches!(target, "test" | "check" | "lint" | "build" | "fmt") {
                out.push(format!("make {target}"));
            }
        }
    }
    out
}

/// The indented command lines directly under a task-runner header, joined into
/// one shell command. Continuation lines are joined with `&&` so a multi-line
/// recipe still runs as a whole rather than as its first line alone.
fn collect_body<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>) -> String {
    let mut parts: Vec<String> = Vec::new();
    while let Some(next) = lines.peek() {
        if next.trim().is_empty() || !next.starts_with(char::is_whitespace) {
            break;
        }
        let line = lines.next().unwrap_or_default();
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        parts.push(line.trim_start_matches('-').trim().to_string());
    }
    parts.retain(|p| !p.is_empty());
    parts.join(" && ")
}

/// Command bodies from a `mise.toml`.
fn mise_tasks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_task = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_task = trimmed.contains("task") || trimmed.contains("tasks");
            continue;
        }
        if in_task {
            if let Some((_, value)) = trimmed.split_once('=') {
                let value = value.trim().trim_matches('"');
                if !value.is_empty() {
                    out.push(value.to_string());
                }
            }
        }
    }
    out
}

/// `run:` command bodies from a CI file.
///
/// A step's keys can come in any order — in practice `working-directory` is
/// written *after* the `run` it applies to — so this reads a step as a unit and
/// resolves its keys together rather than streaming them. Three shapes have to
/// be handled, and getting any one wrong yields a command that looks right and
/// fails:
///
/// - the sequence marker is YAML, not part of the command (`- run: cargo test`);
/// - a block scalar moves the body onto following, deeper-indented lines;
/// - `working-directory` decides *where* the command runs, and dropping it turns
///   a command that passes in CI into one that fails from the repository root.
fn run_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for step in steps(text) {
        let Some(dir) = step.iter().find_map(|l| yaml_value(l, "working-directory")) else {
            for command in step_run_bodies(&step) {
                out.push(command);
            }
            continue;
        };
        let prefix = if dir.is_empty() || dir.contains(char::is_whitespace) {
            String::new()
        } else {
            format!("cd {dir} && ")
        };
        for command in step_run_bodies(&step) {
            out.push(format!("{prefix}{command}"));
        }
    }
    out
}

/// Split a CI file into its list items, keeping each item's indented body.
///
/// A line beginning `- ` opens an item; lines indented further than that marker
/// belong to it. A key that is not a list item is treated as a one-line item of
/// its own so top-level `run:` keys are still found.
fn steps(text: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut indent = 0usize;
    for line in text.lines() {
        let raw = line.trim_end();
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let width = raw.len() - raw.trim_start().len();
        let is_item = raw.trim_start().starts_with("- ");
        if is_item || out.is_empty() || width <= indent {
            indent = width;
            out.push(Vec::new());
        }
        if let Some(last) = out.last_mut() {
            last.push(raw.to_string());
        }
    }
    out
}

/// The value of `key: value` on one line, unquoted.
fn yaml_value(line: &str, key: &str) -> Option<String> {
    let body = line
        .trim_start()
        .strip_prefix("- ")
        .unwrap_or_else(|| line.trim_start());
    let rest = body.strip_prefix(key)?.strip_prefix(':')?;
    Some(rest.trim().trim_matches('"').trim_matches('\'').to_string())
}

/// Every command a single step's `run:` key declares, block scalars included.
fn step_run_bodies(step: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut run_indent: Option<usize> = None;
    for line in step {
        let raw = line.trim_end();
        let trimmed = raw.trim_start();
        let width = raw.len() - trimmed.len();
        // `- ` is YAML's sequence marker, not part of the key.
        let body = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        if let Some(rest) = body.strip_prefix("run:") {
            let value = rest.trim();
            // A bare `|` or `>` is a block-scalar marker: the body follows on
            // deeper-indented lines, so this key has no value of its own yet.
            let is_block =
                !value.is_empty() && value.chars().all(|c| matches!(c, '|' | '>' | '-' | '+'));
            if is_block {
                run_indent = Some(width);
                continue;
            }
            run_indent = None;
            let value = value.trim_matches(['|', '>', '-', '+']).trim();
            if !value.is_empty() && !value.starts_with('#') {
                out.push(value.to_string());
            }
        } else if let Some(key_indent) = run_indent {
            let continues = width > key_indent && !looks_like_yaml_key(body);
            if continues {
                let joined = match out.pop() {
                    Some(prev) => format!("{prev} && {}", next_of(body)),
                    None => next_of(body).to_string(),
                };
                out.push(joined);
            } else {
                run_indent = None;
            }
        }
    }
    out.into_iter()
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

/// A continuation line with its sequence marker removed.
fn next_of(body: &str) -> &str {
    body.strip_prefix("- ").unwrap_or(body).trim()
}

/// Whether a line reads as `key: value`, which ends a block scalar rather than
/// continuing it.
fn looks_like_yaml_key(line: &str) -> bool {
    match line.split_once(':') {
        Some((key, _)) => !key.is_empty() && !key.contains(char::is_whitespace),
        None => false,
    }
}

/// Shell commands from fenced code blocks in a README.
///
/// Only a fence holding exactly one command becomes a candidate. A fence with
/// several lines is prose showing a sequence — comment lines, a `cd` then a
/// build, a list of alternatives — and joining those into one string produces a
/// command nobody would type, so it is left alone. Guessing here would be the
/// same mistake as guessing a test command at all.
fn fenced_commands(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    let mut body: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if inside {
                let mut kept: Vec<&str> = body
                    .iter()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .collect();
                // Strip a leading shell prompt, but only one that covers the
                // whole line, so `cd x && cargo build` is kept intact.
                if kept.len() == 1 {
                    let one = kept[0];
                    let stripped = one
                        .strip_prefix("$ ")
                        .or_else(|| one.strip_prefix("> "))
                        .unwrap_or(one)
                        .trim();
                    kept = vec![stripped];
                }
                if kept.len() == 1 && !kept[0].is_empty() {
                    out.push(kept[0].to_string());
                }
                body.clear();
            }
            inside = !inside;
            continue;
        }
        if inside {
            body.push(trimmed.to_string());
        }
    }
    out
}

/// Commands implied by the project type, for a repository that declares nothing.
fn inferred_from_manifests(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Ok(text) = std::fs::read_to_string(root.join("Cargo.toml")) {
        if text.contains("[workspace]") || text.contains("[package]") {
            let test = if text.contains("[workspace]") {
                "cargo test --workspace"
            } else {
                "cargo test"
            };
            out.push((test.to_string(), "Cargo.toml".to_string()));
            out.push(("cargo build".to_string(), "Cargo.toml".to_string()));
        }
    }
    if root.join("package.json").is_file() {
        out.push(("npm test".to_string(), "package.json".to_string()));
        out.push(("npm run build".to_string(), "package.json".to_string()));
    }
    if root.join("go.mod").is_file() {
        out.push(("go test ./...".to_string(), "go.mod".to_string()));
    }
    if root.join("pyproject.toml").is_file() || root.join("pytest.ini").is_file() {
        out.push(("pytest".to_string(), "pyproject.toml".to_string()));
    }
    out
}

/// Directory entries in a stable order, so discovery is reproducible.
fn read_dir_sorted(dir: PathBuf) -> Option<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".yml") || name.ends_with(".yaml"))
        .collect();
    names.sort();
    Some(names)
}

/// Run one recipe and record what happened.
///
/// A recipe that times out is deliberately recorded as unverified rather than
/// as a pass. The alternative — reporting a command green that was still
/// running when the budget expired — is the false confidence `WF-4` names as
/// its trap, and a wrong green tick is worse than an admitted gap because it
/// stops anyone from looking.
pub fn prove(root: &Path, recipe: &mut Recipe, budget: Duration) -> Verdict {
    let mut command = shell_command(&recipe.command);
    command
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            let verdict = Verdict::NotLaunchable(format!("could not start it: {e}"));
            recipe.verdict = Some(verdict.clone());
            return verdict;
        }
    };

    let deadline = Instant::now() + budget;
    let verdict = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                break if status.success() {
                    Verdict::Passed
                } else {
                    Verdict::Failed(status.code().unwrap_or(-1))
                };
            }
            Ok(None) => {}
            Err(e) => break Verdict::NotLaunchable(format!("could not wait for it: {e}")),
        }
        if Instant::now() >= deadline {
            break Verdict::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    if verdict == Verdict::TimedOut {
        let _ = child.kill();
        let _ = child.wait();
    }
    recipe.verdict = Some(verdict.clone());
    verdict
}

fn shell_command(command: &str) -> std::process::Command {
    let mut c = if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.arg("/C");
        c
    } else {
        let mut c = std::process::Command::new("sh");
        c.arg("-c");
        c
    };
    c.arg(command);
    c
}

/// Render the anchor document.
///
/// Deterministic by construction: sorted input, no clock, no absolute path, no
/// duration. Two runs over the same repository produce identical bytes, which
/// is what lets the file sit in the stable head without churning the KV prefix.
pub fn render(discovery: &Discovery) -> String {
    let mut out = String::from(
        "# Build and test\n\
         \n\
         Commands for this repository, as read from its own files. Run these rather\n\
         than guessing; a command that works is the fastest way to find out what\n\
         is true here.\n",
    );
    if !discovery.has_proof() {
        out.push_str(
            "\n**Nothing here has been verified.** No candidate command was run to\n\
             completion, so treat the list below as a claim by the repository's own\n\
             files, not as fact. Run a command and check its exit code before\n\
             believing it.\n",
        );
    }
    for kind in [Kind::Format, Kind::Lint, Kind::Build, Kind::Test] {
        let matching: Vec<&Recipe> = discovery
            .recipes
            .iter()
            .filter(|r| r.kind == kind)
            .collect();
        if matching.is_empty() {
            continue;
        }
        out.push_str(&format!("\n## {}\n\n", kind.label()));
        for recipe in matching {
            match &recipe.verdict {
                Some(verdict) => {
                    out.push_str(&format!(
                        "- `{}`\n  — {}, {}\n",
                        recipe.command,
                        verdict.describe(),
                        recipe.provenance.caveat()
                    ));
                }
                None => out.push_str(&format!(
                    "- `{}`\n  — **not run**, {}\n",
                    recipe.command,
                    recipe.provenance.caveat()
                )),
            }
        }
    }
    if !discovery.probed.is_empty() {
        out.push_str(&format!("\nRead from: {}\n", discovery.probed.join(", ")));
    }
    out
}

/// Write the anchor into `.xencode/anchor.md`, atomically.
///
/// A half-written anchor would sit inside the stable head, so a truncated file
/// is worse here than in most places: it would change the prefix for every
/// subsequent request. Write beside the target and rename.
pub fn write_anchor(root: &Path, text: &str) -> Result<PathBuf, ContextError> {
    let dir = root.join(XENCODE_DIR);
    std::fs::create_dir_all(&dir).map_err(|source| ContextError::Io {
        path: dir.clone(),
        source,
    })?;
    let target = dir.join("anchor.md");
    let temp = dir.join("anchor.md.tmp");
    std::fs::write(&temp, text).map_err(|source| ContextError::Io {
        path: temp.clone(),
        source,
    })?;
    std::fs::rename(&temp, &target).map_err(|source| ContextError::Io {
        path: target.clone(),
        source,
    })?;
    Ok(target)
}

/// Whether the existing anchor already says exactly this, so an unchanged
/// repository is not rewritten and the stable head keeps its bytes.
pub fn is_current(root: &Path, text: &str) -> bool {
    std::fs::read_to_string(root.join(XENCODE_DIR).join("anchor.md")).is_ok_and(|a| a == text)
}

/// Sidecar metadata recording anchor proof provenance and timestamp (`AB-1`).
///
/// Written to `.xencode/anchor.meta` when recipes are proven by `xencode anchor`.
/// Kept outside `anchor.md` so that recording verification timestamps does not
/// alter the rendered markdown bytes or break llama.cpp KV prefix caching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnchorMeta {
    /// Timestamp (seconds since UNIX epoch) when recipes were verified.
    pub proved_at_unix_s: u64,
    /// Number of recipe candidates discovered.
    pub candidates: usize,
    /// Number of recipe candidates that exited 0 when run.
    pub verified: usize,
}

/// The filename for the anchor metadata sidecar.
pub const ANCHOR_META_FILE: &str = "anchor.meta";

/// Freshness threshold in days after which an anchor proof is considered aged.
pub const ANCHOR_STALE_AGE_DAYS: u64 = 14;

/// Calculate the age of an anchor proof in whole days.
pub fn anchor_age_days(proved_at_unix_s: u64, now_unix_s: u64) -> u64 {
    now_unix_s.saturating_sub(proved_at_unix_s) / 86400
}

/// Atomically write the anchor proof metadata sidecar into `.xencode/anchor.meta`.
pub fn write_anchor_meta(root: &Path, meta: &AnchorMeta) -> Result<PathBuf, ContextError> {
    let dir = if root.ends_with(XENCODE_DIR) {
        root.to_path_buf()
    } else {
        root.join(XENCODE_DIR)
    };
    std::fs::create_dir_all(&dir).map_err(|source| ContextError::Io {
        path: dir.clone(),
        source,
    })?;
    let target = dir.join(ANCHOR_META_FILE);
    let temp = dir.join(format!("{ANCHOR_META_FILE}.tmp"));
    let json = serde_json::to_string_pretty(meta).map_err(|e| ContextError::Io {
        path: temp.clone(),
        source: std::io::Error::other(e.to_string()),
    })?;
    std::fs::write(&temp, json).map_err(|source| ContextError::Io {
        path: temp.clone(),
        source,
    })?;
    std::fs::rename(&temp, &target).map_err(|source| ContextError::Io {
        path: target.clone(),
        source,
    })?;
    Ok(target)
}

/// Read the anchor metadata sidecar from a project root or `.xencode` directory.
pub fn read_anchor_meta(root: &Path) -> Option<AnchorMeta> {
    read_anchor_meta_from_dir(root)
}

/// Read the anchor metadata sidecar from either the project root or `.xencode` directory.
pub fn read_anchor_meta_from_dir(dir: &Path) -> Option<AnchorMeta> {
    let candidate1 = dir.join(XENCODE_DIR).join(ANCHOR_META_FILE);
    let candidate2 = dir.join(ANCHOR_META_FILE);
    let target = if candidate1.is_file() {
        candidate1
    } else if candidate2.is_file() {
        candidate2
    } else {
        return None;
    };
    let text = std::fs::read_to_string(target).ok()?;
    serde_json::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Tree(PathBuf);

    impl Tree {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("xe-anchor-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(self, rel: &str, body: &str) -> Self {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(body.as_bytes()).unwrap();
            drop(f);
            self
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn classify_prefers_the_specific_signal() {
        assert_eq!(Kind::classify("cargo fmt --check"), Some(Kind::Format));
        assert_eq!(Kind::classify("cargo clippy --workspace"), Some(Kind::Lint));
        assert_eq!(Kind::classify("cargo test --workspace"), Some(Kind::Test));
        assert_eq!(Kind::classify("cargo build"), Some(Kind::Build));
        assert_eq!(Kind::classify("echo hello"), None);
    }

    #[test]
    fn reads_the_command_ci_actually_runs() {
        let tree = Tree::new("ci").file(
            ".github/workflows/ci.yml",
            "jobs:\n  test:\n    steps:\n      - run: cargo fmt --check\n      - run: cargo test --workspace --verbose\n",
        );
        let d = discover(tree.path());
        let test = d.candidate(Kind::Test).expect("a test command");
        assert_eq!(test.command, "cargo test --workspace --verbose");
        assert_eq!(test.provenance, Provenance::ContinuousIntegration);
    }

    #[test]
    fn joins_a_block_scalared_run_body() {
        let lines = run_lines(
            "steps:\n  - run: |\n      cargo fmt --check\n      cargo test --workspace\n",
        );
        assert!(
            lines.iter().any(|l| l.contains("cargo test --workspace")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_justfile_yields_its_recipe_bodies() {
        let tree = Tree::new("just").file(
            "justfile",
            "test:\n    cargo test --workspace\n\nbuild:\n    cargo build --release\n",
        );
        let d = discover(tree.path());
        assert!(d
            .recipes
            .iter()
            .any(|r| r.command == "cargo test --workspace"));
    }

    #[test]
    fn a_makefile_target_is_discovered() {
        let tree = Tree::new("make").file(
            "Makefile",
            "test:\n\tcargo test --workspace\n\n.PHONY: test\n",
        );
        let d = discover(tree.path());
        assert!(d
            .recipes
            .iter()
            .any(|r| r.command == "cargo test --workspace"));
    }

    #[test]
    fn a_working_directory_becomes_part_of_the_command() {
        // CI runs this from `rust/`. Without the `cd` the anchor would record a
        // command that fails from the repository root, while looking correct.
        let lines = run_lines(
            "    - name: Check formatting\n      run: cargo fmt --check\n      working-directory: rust\n",
        );
        assert_eq!(lines, vec!["cd rust && cargo fmt --check"], "{lines:?}");
    }

    #[test]
    fn a_working_directory_applies_to_a_block_scalar_body() {
        let lines = run_lines("    - run: |\n        cargo test\n      working-directory: rust\n");
        assert_eq!(lines, vec!["cd rust && cargo test"], "{lines:?}");
    }

    #[test]
    fn a_working_directory_does_not_leak_into_the_next_step() {
        let lines = run_lines(
            "    - run: cargo build\n      working-directory: rust\n    - run: cargo deploy\n",
        );
        assert!(
            lines.contains(&"cargo deploy".to_string()),
            "the next step must not inherit the previous directory: {lines:?}"
        );
        assert!(
            !lines
                .iter()
                .any(|l| l.starts_with("cd rust && cargo deploy")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_block_scalar_stops_at_the_next_yaml_key() {
        // A `with:` block and a following `name:` must not be swallowed into the
        // command; that produced entries like `name: Run cargo test`.
        let lines = run_lines(
            "    - name: cache\n      with:\n        path: |\n          ~/.cargo\n      run: cargo build\n",
        );
        assert!(
            !lines.iter().any(|l| l.contains("name:")),
            "a YAML key leaked into a command: {lines:?}"
        );
    }

    #[test]
    fn a_readme_fence_with_several_commands_is_not_one_recipe() {
        let readme = "```bash\ncd rust\ncargo test --workspace\ncargo build\n```\n";
        assert!(fenced_commands(readme).is_empty(), "prose is not a command");
    }

    #[test]
    fn a_readme_fence_holding_one_command_is_kept() {
        assert_eq!(
            fenced_commands("```bash\ncargo test --workspace\n```\n"),
            vec!["cargo test --workspace"]
        );
        assert_eq!(
            fenced_commands("```\n$ cargo test\n```\n"),
            vec!["cargo test"],
            "a lone shell prompt is stripped"
        );
    }

    #[test]
    fn a_workspace_is_implied_when_nothing_declares_anything() {
        let tree = Tree::new("cargo").file("Cargo.toml", "[workspace]\nmembers = [\"a\"]\n");
        let d = discover(tree.path());
        let test = d.candidate(Kind::Test).expect("inferred test");
        assert_eq!(test.command, "cargo test --workspace");
        assert_eq!(test.provenance, Provenance::Inferred);
    }

    #[test]
    fn nothing_is_claimed_before_it_runs() {
        let tree = Tree::new("unproven").file("Cargo.toml", "[workspace]\n");
        let d = discover(tree.path());
        assert!(!d.has_proof(), "discovery alone is not proof");
        assert!(d.candidate(Kind::Test).is_some());
        let anchor = render(&d);
        assert!(anchor.contains("**not run**"), "{anchor}");
        assert!(
            anchor.contains("Nothing here has been verified"),
            "{anchor}"
        );
    }

    fn recipe(command: &str) -> Recipe {
        Recipe {
            kind: Kind::Test,
            command: command.to_string(),
            source: "test fixture".to_string(),
            provenance: Provenance::Inferred,
            verdict: None,
        }
    }

    #[test]
    fn a_command_that_exits_zero_is_verified() {
        let tree = Tree::new("pass");
        let mut recipe = recipe("true");
        assert_eq!(
            prove(tree.path(), &mut recipe, Duration::from_secs(30)),
            Verdict::Passed
        );
        assert!(recipe.is_verified());
    }

    #[test]
    fn a_command_that_fails_is_not_verified() {
        let tree = Tree::new("fail");
        let mut recipe = recipe("exit 3");
        let verdict = prove(tree.path(), &mut recipe, Duration::from_secs(30));
        assert_eq!(verdict, Verdict::Failed(3), "{verdict:?}");
        assert!(!recipe.is_verified());
        let d = Discovery {
            recipes: vec![recipe],
            probed: vec![],
        };
        assert!(
            render(&d).contains("FAILED"),
            "the anchor must admit the failure"
        );
    }

    #[test]
    fn a_timeout_is_never_reported_as_success() {
        let tree = Tree::new("slow");
        let mut recipe = recipe("sleep 30");
        assert_eq!(
            prove(tree.path(), &mut recipe, Duration::from_millis(200)),
            Verdict::TimedOut
        );
        assert!(
            !recipe.is_verified(),
            "an unfinished command proves nothing"
        );
    }

    #[test]
    fn a_missing_program_is_reported_not_ignored() {
        let tree = Tree::new("missing");
        let mut recipe = recipe("this-binary-does-not-exist --all");
        let verdict = prove(tree.path(), &mut recipe, Duration::from_secs(10));
        assert!(!recipe.is_verified());
        assert!(
            matches!(verdict, Verdict::Failed(_) | Verdict::NotLaunchable(_)),
            "{verdict:?}"
        );
    }

    #[test]
    fn a_failing_discovery_renders_as_unverified_end_to_end() {
        // Classifies as a test command, so discovery keeps it, and fails
        // immediately, so proof has something real to reject.
        let tree = Tree::new("e2e").file(
            "Makefile",
            "test:\n\tcargo test --flag-that-does-not-exist\n",
        );
        let mut d = discover(tree.path());
        let candidate = d.recipes.first_mut().expect("a classified candidate");
        assert_eq!(candidate.kind, Kind::Test);
        prove(tree.path(), candidate, Duration::from_secs(60));
        assert!(!d.has_proof(), "a failing command is not a proof");
        let anchor = render(&d);
        assert!(
            anchor.contains("Nothing here has been verified"),
            "{anchor}"
        );
        assert!(
            anchor.contains("not verified") || anchor.contains("FAILED"),
            "{anchor}"
        );
    }

    #[test]
    fn the_anchor_is_byte_identical_across_renders() {
        let tree = Tree::new("stable")
            .file("Cargo.toml", "[workspace]\n")
            .file(
                ".github/workflows/ci.yml",
                "steps:\n  - run: cargo test --workspace\n",
            );
        let first = render(&discover(tree.path()));
        let second = render(&discover(tree.path()));
        assert_eq!(first, second, "the stable head cannot tolerate churn");
    }

    #[test]
    fn the_anchor_carries_no_clock_or_absolute_path() {
        let tree = Tree::new("clean").file("Cargo.toml", "[workspace]\n").file(
            ".github/workflows/ci.yml",
            "steps:\n  - run: cargo test --workspace\n",
        );
        let anchor = render(&discover(tree.path()));
        for banned in [
            tree.path().to_string_lossy().as_ref(),
            "Generated",
            "seconds",
        ] {
            assert!(
                !anchor.contains(banned),
                "the anchor must not contain {banned:?}"
            );
        }
    }

    #[test]
    fn the_anchor_lands_in_xencode_and_is_recognised_as_current() {
        let tree = Tree::new("write");
        let d = discover(tree.path());
        let text = render(&d);
        let written = write_anchor(tree.path(), &text).unwrap();
        assert!(written.ends_with("anchor.md"));
        assert!(is_current(tree.path(), &text));
        assert_eq!(std::fs::read_to_string(&written).unwrap(), text);
        assert!(!is_current(tree.path(), "different"));
    }

    #[test]
    fn discovery_is_reproducible() {
        let tree = Tree::new("repro")
            .file("Cargo.toml", "[workspace]\n")
            .file("justfile", "test:\n    cargo test --workspace\n")
            .file(
                ".github/workflows/ci.yml",
                "steps:\n  - run: cargo fmt --check\n",
            );
        let first = discover(tree.path());
        let second = discover(tree.path());
        assert_eq!(first.recipes, second.recipes);
        assert_eq!(first.probed, second.probed);
    }

    #[test]
    fn anchor_metadata_sidecar_round_trips_and_leaves_anchor_md_untouched() {
        let tree = Tree::new("meta");
        let d = discover(tree.path());
        let text = render(&d);
        let anchor_file = write_anchor(tree.path(), &text).unwrap();

        // Writing anchor.md creates no sidecar file automatically.
        assert!(!tree.path().join(XENCODE_DIR).join(ANCHOR_META_FILE).exists());
        assert_eq!(read_anchor_meta(tree.path()), None);

        let meta = AnchorMeta {
            proved_at_unix_s: 1_700_000_000,
            candidates: 4,
            verified: 3,
        };
        let meta_path = write_anchor_meta(tree.path(), &meta).unwrap();
        assert!(meta_path.ends_with("anchor.meta"));

        // anchor.md was not touched or modified by the sidecar write.
        assert_eq!(std::fs::read_to_string(&anchor_file).unwrap(), text);

        let read_back = read_anchor_meta(tree.path()).expect("anchor.meta must parse");
        assert_eq!(read_back, meta);

        // Reading directly from .xencode dir also works.
        let from_dir = read_anchor_meta_from_dir(&tree.path().join(XENCODE_DIR)).unwrap();
        assert_eq!(from_dir, meta);

        // Age calculation in days.
        assert_eq!(anchor_age_days(1_700_000_000, 1_700_000_000), 0);
        assert_eq!(anchor_age_days(1_700_000_000, 1_700_086_400), 1);
        assert_eq!(anchor_age_days(1_700_000_000, 1_700_086_399), 0);
        assert_eq!(anchor_age_days(1_700_000_000, 1_701_209_600), 14);
        // Clock skew / past timestamp saturates at 0 instead of panicking.
        assert_eq!(anchor_age_days(1_700_000_000, 1_699_000_000), 0);
    }
}
