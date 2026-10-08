//! `OR-1` — the planner that proposes a split, and the facts a split is judged
//! against.
//!
//! [`xencode_core_rs::decompose`] holds the split, the score and the refusal.
//! This file holds the two things that need a repository and a model:
//!
//! - [`reference_from_paths`] builds the thing a split is compared to out of
//!   observed facts: the files the change actually touches, and which of them the
//!   build refuses to compile out of order. The order is not an opinion — it is
//!   `cargo metadata`'s own dependency graph, so "this crate must land first" is
//!   a statement about the compiler rather than about the planner, and a reader
//!   can run the same command. A pair the build does not force is left out of the
//!   reference entirely: a split is never punished for not stating an order that
//!   nothing enforces.
//! - [`ask_planner`] asks a model for the split. It gets the task and the
//!   workspace's crate layout, and nothing else — in particular it does not get
//!   the reference's file list, because then the measurement would be a copy.
//!
//! Both halves keep their distance from each other on purpose. The score is
//! computed by the pure layer from the two of them, so a run that cannot reach a
//! model still produces a reference, and a reference built for somebody else's
//! task still scores honestly.
//!
//! [`commit_change`] is how a task gets stated without a person writing it out:
//! the message of a real commit is the task, and its diff is the file set. That
//! pairing matters — the only split worth measuring is one that was never shown
//! the answer.

use std::path::{Path, PathBuf};

use xencode_core_rs::{Reference, Split};

/// What the planner said, kept with the prompt that asked for it: a number
/// without the text behind it is not evidence.
#[derive(Debug, Clone)]
pub struct PlannerRun {
    pub model: String,
    pub prompt: String,
    /// The answer exactly as it came back, before anything was read out of it.
    pub raw: String,
    pub split: Split,
    pub elapsed_ms: u128,
}

/// The workspace's own crates, as `name — directory` lines relative to the
/// repository, which is the only map a planner can be given without handing it
/// the answer. Sorted, so the same repository produces the same prompt and the
/// same sampling seed means the same thing twice.
pub fn workspace_layout(root: &Path) -> Result<Vec<String>, String> {
    let graph = crate_graph(root)?;
    let prefix = root.to_path_buf();
    let mut lines = Vec::new();
    for name in &graph.members {
        let dir = graph
            .member_dir
            .get(name)
            .map(|d| d.strip_prefix(&prefix).unwrap_or(d).to_path_buf())
            .unwrap_or_else(|| PathBuf::from(name));
        lines.push(format!("{name} — {}", dir.display()));
    }
    Ok(lines)
}

/// The reference a split is scored against, from the files a change really
/// touched and the order the workspace's own build graph enforces.
///
/// `verification` is the command every node is graded by — the same one for all
/// of them, so neither a planner nor a baseline can pick an easier test.
pub fn reference_from_paths(
    root: &Path,
    paths: &[String],
    verification: &str,
    source: &str,
) -> Result<Reference, String> {
    let graph = crate_graph(root)?;
    let mut reference = Reference::new(source, verification);
    reference.paths = paths.to_vec();

    // Which crate each file sits in. A file no member claims (a top-level README,
    // the workspace manifest itself) belongs to no crate, so it takes part in no
    // ordering claim — it stays in the file set, where a missing owner is still
    // counted against the split.
    let crates: Vec<Option<String>> = paths
        .iter()
        .map(|path| xencode_context_rs::crate_of_file(&graph, &root.join(path)))
        .collect();

    for i in 0..paths.len() {
        for j in 0..paths.len() {
            if i == j {
                continue;
            }
            let (Some(a), Some(b)) = (&crates[i], &crates[j]) else {
                continue;
            };
            if a == b {
                // Two files in one crate: the build says nothing about which of
                // them lands first. `reverse_closure` never returns the crate it
                // started from, so this would be inferred — it is stated here so
                // the rule is visible and the closure walk is not run for it.
                continue;
            }
            // `b` depending on `a`, transitively, is the compiler's own statement
            // that `a`'s crate is built before `b`'s.
            if graph.reverse_closure(a).iter().any(|(name, _)| name == b) {
                let pair = [paths[i].clone(), paths[j].clone()];
                if !reference.before.contains(&pair) {
                    reference.before.push(pair);
                }
            }
        }
    }
    reference.before.sort();
    Ok(reference)
}

/// The file set a split named for itself, for the case where nobody has a diff to
/// compare it with. The ordering half of the score is still read off the build and
/// is still a measurement; the ownership half becomes the answer agreeing with
/// itself, and `source` says so, because a figure that cannot be wrong is not
/// evidence and must never be quoted as if it were.
pub fn reference_from_split(
    split: &Split,
    root: &Path,
    verification: &str,
) -> Result<Reference, String> {
    let mut paths: Vec<String> = split
        .subtasks
        .iter()
        .flat_map(|n| n.paths.clone())
        .collect();
    paths.sort();
    paths.dedup();
    reference_from_paths(
        root,
        &paths,
        verification,
        "the split's own file list — the order is measured against the build, \
         the ownership is the answer agreeing with itself",
    )
}

/// The same metadata the impact report reads: first-party crates and their edges,
/// from `cargo metadata --no-deps`, which is offline and touches no lock file.
fn crate_graph(root: &Path) -> Result<xencode_context_rs::CrateGraph, String> {
    let metadata = xencode_context_rs::cargo_metadata(root)
        .map_err(|e| format!("cannot read the workspace's crates: {e}"))?;
    xencode_context_rs::parse_crate_graph(&metadata)
}

/// Where a commit's own message and diff are read back as the task and the file
/// set, which is the only way to score a split without letting the split choose
/// what it is scored against.
///
/// Git names files relative to the repository; the reference and the planner's
/// answer are both named relative to the Rust workspace, so the workspace's own
/// position inside the repository — `git rev-parse --show-prefix`, asked of git
/// rather than guessed from the directory name — is stripped here. A path outside
/// the workspace stays in the file set: a file the change touched is a file the
/// change touched. It belongs to no crate, so it asserts no order, and a split
/// that ignores it is still short of the change.
#[derive(Debug, Clone)]
pub struct CommitChange {
    pub sha: String,
    pub subject: String,
    pub body: String,
    pub paths: Vec<String>,
    /// How many of those paths sit outside this workspace.
    pub outside: usize,
}

impl CommitChange {
    /// The task text handed to the planner: what the commit set out to do, in the
    /// words its author wrote, and nothing else. Notably it does not name the
    /// files, which is what keeps the measurement from being a copy.
    pub fn task(&self) -> String {
        let body = self.body.trim();
        if body.is_empty() {
            self.subject.clone()
        } else {
            format!("{}\n\n{body}", self.subject)
        }
    }

    /// Where the reference came from, in the words a reader can check.
    pub fn source(&self) -> String {
        let outside = if self.outside == 0 {
            String::new()
        } else {
            format!(", {} of them outside this workspace", self.outside)
        };
        format!(
            "commit {} — `{}` ({} file{}){outside}",
            &self.sha[..10.min(self.sha.len())],
            self.subject,
            self.paths.len(),
            if self.paths.len() == 1 { "" } else { "s" }
        )
    }
}

fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("could not start `git {}`: {e}", args.join(" ")))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub fn commit_change(root: &Path, sha: &str) -> Result<CommitChange, String> {
    let toplevel = git(root, &["rev-parse", "--show-toplevel"])
        .map_err(|e| format!("{} is not in a git repository: {e}", root.display()))?;
    let toplevel = PathBuf::from(toplevel.trim());
    let prefix = git(root, &["rev-parse", "--show-prefix"])
        .unwrap_or_default()
        .trim()
        .to_string();

    // `%x1f` cannot appear in a commit message, so one call yields the full sha,
    // the subject and the body without parsing a format a quote could break.
    let shown = git(&toplevel, &["show", "-s", "--format=%H%x1f%s%x1f%b", sha])
        .map_err(|e| format!("git has no commit `{sha}` here: {e}"))?;
    let mut fields = shown.split('\u{1f}');
    let (Some(full), Some(subject), Some(body)) = (fields.next(), fields.next(), fields.next())
    else {
        return Err(format!("cannot read the message of commit `{sha}`"));
    };
    let full = full.trim().to_string();

    // `diff.relative` would name the files against this directory instead of the
    // repository, so it is pinned off rather than trusted to be unset.
    let listing = git(
        &toplevel,
        &[
            "-c",
            "diff.relative=false",
            "show",
            "--name-only",
            "--format=",
            sha,
        ],
    )
    .map_err(|e| format!("cannot list what commit `{sha}` touched: {e}"))?;
    let mut paths = Vec::new();
    let mut outside = 0usize;
    for line in listing.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match line.strip_prefix(&prefix) {
            Some(rel) if !rel.is_empty() => paths.push(rel.to_string()),
            _ => {
                outside += 1;
                paths.push(line.to_string());
            }
        }
    }
    if paths.is_empty() {
        return Err(format!(
            "commit {} names no files, so there is nothing to score a split \
             against — a merge shows no diff unless one side is named",
            &full[..10.min(full.len())]
        ));
    }
    Ok(CommitChange {
        sha: full,
        subject: subject.trim().to_string(),
        body: body.to_string(),
        paths,
        outside,
    })
}

/// The shape the planner must answer in. `additionalProperties: false` is the
/// point of the whole schema: a unit that writes `depends_on` instead of `needs`
/// would otherwise be read as a unit that waits for nothing, and the scheduler
/// would start it immediately.
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "subtasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "goal": {"type": "string"},
                        "paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                        "needs": {"type": "array", "items": {"type": "string"}},
                        "verify": {"type": "string"},
                    },
                    "required": ["id", "goal", "paths", "needs", "verify"],
                    "additionalProperties": false,
                }
            }
        },
        "required": ["task", "subtasks"],
        "additionalProperties": false,
    })
}

/// The request. Deliberately short of the answer: the task as a person stated it,
/// the crates that exist, and what happens to a unit that says nothing about
/// order.
pub fn prompt(task: &str, layout: &[String]) -> String {
    let mut out = String::from(
        "You are planning one change to a Rust workspace. Split it into the units \
         a worker could each finish on its own, and say which units must come \
         before which.\n\nThe workspace's crates are:\n",
    );
    for line in layout {
        out.push_str(&format!("  {line}\n"));
    }
    out.push_str(&format!(
        "\nThe task:\n{task}\n\nAnswer with JSON only, in exactly this shape, with no \
         field added and none left out:\n\
         {{\"task\": \"the task in your own words\", \"subtasks\": [{{\"id\": \
         \"short-name\", \"goal\": \"what this unit is for\", \"paths\": \
         [\"crates/x/src/lib.rs\"], \"needs\": [\"ids whose work this unit \
         reads\"], \"verify\": \"one command that decides it is done\"}}]}}\n\n\
         Rules: write every path relative to this workspace root, in the same form \
         the crate list above uses, so a file in the crate listed as \
         `xencode-foo — crates/xencode-foo` is \
         `crates/xencode-foo/src/lib.rs`. Every unit names at least one file or \
         directory it changes and one command that proves it. `needs` must be \
         present even when empty — a unit with nothing in `needs` is started \
         immediately, alongside every other unit that is also waiting for nothing, \
         so leaving a real dependency out is not caution, it is two workers \
         editing the same file at once.\n"
    ));
    out
}

/// Ask the model. A refusal, a timeout or an answer that is not the shape asked
/// for comes back as `Err` naming what happened; nothing here invents a split to
/// keep a run going.
pub async fn ask_planner(
    task: &str,
    layout: &[String],
    model: &str,
    options: &PlannerOptions,
) -> Result<PlannerRun, String> {
    use xencode_models_rs::LlamaCppOptions;
    use xencode_providers_rs::{ChatMessage, ProviderManager};

    let prompt = prompt(task, layout);
    let started = std::time::Instant::now();
    let manager = ProviderManager::new(
        xencode_models_rs::OllamaClient::new(&options.ollama_url, options.timeout_secs),
        None,
        None,
        None,
        None,
    )
    .with_llama_cpp(xencode_models_rs::LlamaCppClient::new(
        &options.llama_cpp_url,
        options.timeout_secs,
    ))
    .with_request_timeout(options.timeout_secs);

    let sampling = LlamaCppOptions {
        temperature: Some(options.temperature),
        seed: Some(options.seed),
        max_tokens: Some(options.max_tokens),
        json_schema: Some(schema()),
        ..Default::default()
    };
    let raw = manager
        .generate_with_options(
            model,
            &[ChatMessage::text("user", prompt.clone())],
            Some(&sampling),
        )
        .await
        .map_err(|e| {
            // Which endpoint answered is the manager's business, not this caller's:
            // the model's own name routes it, so the failure is reported against the
            // model that was asked rather than against one of the two URLs below.
            format!("the planner could not be asked (`{model}` is not reachable): {e}")
        })?;

    let value = xencode_providers_rs::schema::read_answer(&raw, &schema()).map_err(|e| {
        // A split that ran past the cap comes back as a half-written document, and
        // "not JSON" alone hides the one fact that says what to do about it. Measured
        // here: an answer to an eleven-file change stopped mid-line at 1024 tokens.
        format!(
            "the planner's answer was not a split ({} characters came back against a {}-token \
             cap): {e}",
            raw.chars().count(),
            options.max_tokens
        )
    })?;
    let split = Split::from_value(&value).map_err(|e| format!("the answer is not a split: {e}"))?;
    Ok(PlannerRun {
        model: model.to_string(),
        prompt,
        raw,
        split,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Where to ask, and how. Both endpoints are this machine's — the planner is the
/// local model, which is the reason `OR-1` has to be measured before anything
/// downstream trusts it (§S-4.2). Sampling is pinned by default for the same
/// reason `EV-1` pins it: a split measured twice must be the same split.
#[derive(Debug, Clone)]
pub struct PlannerOptions {
    pub ollama_url: String,
    pub llama_cpp_url: String,
    pub temperature: f64,
    pub seed: i64,
    pub max_tokens: u32,
    pub timeout_secs: u64,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        PlannerOptions {
            ollama_url: "http://localhost:11434".to_string(),
            llama_cpp_url: "http://localhost:8080".to_string(),
            temperature: 0.0,
            seed: 42,
            // Measured against two real changes in this repository's history: a
            // nine-unit answer took 705 tokens, and an answer for a twenty-file
            // change was cut off mid-line at 1024 and came back as unreadable JSON.
            // The cap has to sit above the largest answer, not below it.
            max_tokens: 2048,
            // This is the cap's cost on the machine these runs were measured on: CPU
            // inference at ~4 tokens/s turns 2048 tokens into eight and a half
            // minutes, so a timeout shorter than that only converts one failure
            // into another. A faster model finishes well inside it.
            timeout_secs: 600,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_core_rs::{decide, Split, Subtask};

    /// A three-crate workspace built for real on disk: `leaf` is read by `middle`,
    /// which is read by `top`. `cargo metadata` is the only thing here that decides
    /// the order, so this test measures the reference rather than asserting it.
    fn workspace(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir.join("crates/leaf/src")).unwrap();
        std::fs::create_dir_all(dir.join("crates/middle/src")).unwrap();
        std::fs::create_dir_all(dir.join("crates/top/src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nresolver = \"2\"\nmembers = [\"crates/leaf\", \
             \"crates/middle\", \"crates/top\"]\n",
        )
        .unwrap();
        for (name, dep) in [
            ("leaf", None),
            ("middle", Some(("xleaf", "leaf"))),
            ("top", Some(("xmiddle", "middle"))),
        ] {
            let manifest = match dep {
                None => format!(
                    "[package]\nname = \"x{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"
                ),
                Some((dep_name, dep_dir)) => format!(
                    "[package]\nname = \"x{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
                     [dependencies]\n{dep_name} = {{ path = \"../{dep_dir}\" }}\n"
                ),
            };
            std::fs::write(dir.join(format!("crates/{name}/Cargo.toml")), manifest).unwrap();
            std::fs::write(dir.join(format!("crates/{name}/src/lib.rs")), "// empty\n").unwrap();
        }
        dir.to_path_buf()
    }

    /// This repository's actual shape: the Rust workspace sits one directory
    /// below the repository root, so git's names and the workspace's names differ
    /// by exactly that prefix. Committed for real, because the point of the
    /// function is that it reads what `git show` answers.
    fn repo(dir: &Path) -> PathBuf {
        let root = dir.join("repo");
        workspace(&root.join("rust"));
        std::fs::write(root.join("README.md"), "# a repository\n").unwrap();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?} failed to start: {e}"));
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q", "."]);
        git(&["config", "user.name", "tester"]);
        git(&["config", "user.email", "tester@example.invalid"]);
        git(&["add", "-A"]);
        git(&[
            "commit",
            "-qm",
            "add a record type, its store and the command",
            "-m",
            "leaf first.",
        ]);
        root
    }

    #[test]
    fn a_commit_names_its_own_change_in_workspace_relative_words() {
        let temp = tempfile::tempdir().unwrap();
        let root = repo(temp.path());
        let workspace_root = root.join("rust");
        let change = commit_change(&workspace_root, "HEAD").unwrap();
        assert_eq!(
            change.subject,
            "add a record type, its store and the command"
        );
        assert!(change.body.contains("leaf first."));
        assert_eq!(change.sha.len(), 40, "{}", change.sha);
        // Every path is named against the workspace, not the repository: the
        // `rust/` the diff prints is the directory we are standing in.
        assert!(
            change.paths.contains(&"crates/leaf/src/lib.rs".to_string()),
            "{:?}",
            change.paths
        );
        assert!(!change.paths.iter().any(|p| p.starts_with("rust/")));
        // The README is outside the workspace and stays in, counted as such.
        assert_eq!(change.outside, 1, "{:?}", change.paths);
        assert!(change.paths.contains(&"README.md".to_string()));
        // The task text is the message alone. Handing the planner the file list
        // would make the measurement a copy, so it is not in there to be copied.
        assert!(!change.task().contains("crates/leaf/src/lib.rs"));
        assert!(change.source().contains("outside this workspace"));
    }

    #[test]
    fn a_commit_that_changed_nothing_is_refused_rather_than_scored_against_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let root = repo(temp.path());
        std::process::Command::new("git")
            .args(["commit", "-qm", "nothing", "--allow-empty"])
            .current_dir(&root)
            .output()
            .unwrap();
        let error = commit_change(&root.join("rust"), "HEAD").unwrap_err();
        assert!(error.contains("names no files"), "{error}");
    }

    #[test]
    fn a_file_outside_the_workspace_is_in_the_reference_and_asserts_no_order() {
        // The README belongs to no crate, so the build says nothing about when it
        // lands. It must still be owned by some node — a split that left it out is
        // short of the change — but it cannot make that split look wrong.
        let temp = tempfile::tempdir().unwrap();
        let root = workspace(temp.path());
        let reference = reference_from_paths(
            &root,
            &[
                "README.md".to_string(),
                "crates/leaf/src/lib.rs".to_string(),
                "crates/top/src/lib.rs".to_string(),
            ],
            "cargo check --offline",
            "with one file outside the workspace",
        )
        .unwrap();
        assert_eq!(reference.paths.len(), 3);
        // Only the crate pair the build forces: leaf before top.
        assert_eq!(reference.before.len(), 1, "{:?}", reference.before);
        assert!(!reference.before.iter().any(|p| p[0] == *"README.md"));
        assert!(!reference.before.iter().any(|p| p[1] == *"README.md"));
    }

    #[test]
    fn the_reference_order_comes_from_what_the_build_actually_requires() {
        let temp = tempfile::tempdir().unwrap();
        let root = workspace(temp.path());
        let reference = reference_from_paths(
            &root,
            &[
                "crates/leaf/src/lib.rs".to_string(),
                "crates/middle/src/lib.rs".to_string(),
                "crates/top/src/lib.rs".to_string(),
            ],
            "cargo check --offline",
            "a three-crate change",
        )
        .unwrap();
        // leaf before middle, middle before top, and leaf before top by
        // transitivity — every one of them a dependency edge cargo reports.
        assert_eq!(reference.before.len(), 3, "{:?}", reference.before);
        assert!(reference.before.contains(&[
            "crates/leaf/src/lib.rs".to_string(),
            "crates/top/src/lib.rs".to_string()
        ]));
        // Nothing forces `top` before `leaf`, and the reference does not pretend it does.
        assert!(!reference.before.contains(&[
            "crates/top/src/lib.rs".to_string(),
            "crates/leaf/src/lib.rs".to_string()
        ]));
    }

    #[test]
    fn two_files_in_one_crate_assert_no_order_at_all() {
        // The honest half: the build graph is silent about two files in the same
        // package, so a split is not asked to state an order nobody enforces.
        let temp = tempfile::tempdir().unwrap();
        let root = workspace(temp.path());
        let reference = reference_from_paths(
            &root,
            &[
                "crates/leaf/src/lib.rs".to_string(),
                "crates/leaf/Cargo.toml".to_string(),
            ],
            "cargo check --offline",
            "two files in one crate",
        )
        .unwrap();
        assert!(reference.before.is_empty(), "{:?}", reference.before);
        assert_eq!(reference.paths.len(), 2);
    }

    #[test]
    fn a_planner_that_omits_needs_is_refused_by_the_schema_it_was_given() {
        // The schema is the guard, and it has to be checked here rather than only
        // handed to a server: not every endpoint honours `response_format`, and a
        // missing `needs` arriving as an empty list would read as "start at once".
        let missing = serde_json::json!({
            "task": "t",
            "subtasks": [{"id": "a", "goal": "g", "paths": ["a.rs"], "verify": "cargo build"}]
        });
        let reason = xencode_providers_rs::schema::check(&schema(), &missing).unwrap_err();
        assert!(reason.contains("needs"), "{reason}");

        let invented = serde_json::json!({
            "task": "t",
            "subtasks": [{"id": "a", "goal": "g", "paths": ["a.rs"], "needs": [],
                          "verify": "cargo build", "depends_on": []}]
        });
        let reason = xencode_providers_rs::schema::check(&schema(), &invented).unwrap_err();
        assert!(reason.contains("depends_on"), "{reason}");

        let good = serde_json::json!({
            "task": "t",
            "subtasks": [{"id": "a", "goal": "g", "paths": ["a.rs"], "needs": [],
                          "verify": "cargo build"}]
        });
        xencode_providers_rs::schema::check(&schema(), &good).expect("the shape should pass");
    }

    #[test]
    fn the_prompt_shows_the_crates_and_never_the_file_list() {
        let layout = vec![
            "leaf — crates/leaf".to_string(),
            "top — crates/top".to_string(),
        ];
        let text = prompt("make the store retry", &layout);
        assert!(text.contains("leaf — crates/leaf"));
        assert!(text.contains("make the store retry"));
        // The instruction that makes silence expensive, because a unit that names
        // no dependency is started immediately.
        assert!(text.contains("two workers editing the same file"));
        assert!(text.contains("\"needs\""));
        // The path form has to be the one the score reads. Both local runs measured
        // here answered in a form that owned nothing — one wrote `crates/x/...`,
        // which is right, the other wrote bare `src/routing.rs`, which the example
        // in this prompt invited — so the example is now the same shape as the
        // crate list, and the wrong shape is named as wrong.
        assert!(text.contains("relative to this workspace root"), "{text}");
        assert!(text.contains("\"crates/x/src/lib.rs\""), "{text}");
        assert!(!text.contains("rust/crates"), "{text}");
    }

    #[test]
    fn a_split_read_from_a_planners_answer_scores_against_the_real_reference() {
        // End to end without a model: the answer below is what a planner that got
        // the order right would have produced for the workspace above.
        let temp = tempfile::tempdir().unwrap();
        let root = workspace(temp.path());
        let paths = vec![
            "crates/leaf/src/lib.rs".to_string(),
            "crates/middle/src/lib.rs".to_string(),
            "crates/top/src/lib.rs".to_string(),
        ];
        let reference =
            reference_from_paths(&root, &paths, "cargo check --offline", "measured").unwrap();
        let answer = serde_json::json!({
            "task": "a change across three crates",
            "subtasks": [
                {"id": "leaf", "goal": "the type", "paths": ["crates/leaf/src/lib.rs"],
                 "needs": [], "verify": "cargo check --offline"},
                {"id": "middle", "goal": "the store", "paths": ["crates/middle/src/lib.rs"],
                 "needs": ["leaf"], "verify": "cargo check --offline"},
                {"id": "top", "goal": "the command", "paths": ["crates/top/src/lib.rs"],
                 "needs": ["middle"], "verify": "cargo check --offline"},
            ]
        });
        let split = Split::from_value(&answer).unwrap();
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(
            decision.schedulable,
            "a correct split was refused: {:?}",
            decision.reasons
        );
        assert_eq!(decision.score.agreed, 3);
        assert_eq!(decision.baseline.agreed, 0);
    }

    #[test]
    fn a_planners_split_that_invents_an_order_is_refused_against_the_build() {
        // The same answer with the chain backwards: `leaf` now waits on `top`,
        // which the compiler will not do.
        let temp = tempfile::tempdir().unwrap();
        let root = workspace(temp.path());
        let paths = vec![
            "crates/leaf/src/lib.rs".to_string(),
            "crates/middle/src/lib.rs".to_string(),
            "crates/top/src/lib.rs".to_string(),
        ];
        let reference =
            reference_from_paths(&root, &paths, "cargo check --offline", "measured").unwrap();
        let mut split = Split::new("backwards");
        split.push(Subtask {
            id: "top".into(),
            goal: "the command".into(),
            paths: vec!["crates/top/src/lib.rs".into()],
            needs: vec![],
            verify: "cargo check --offline".into(),
        });
        split.push(Subtask {
            id: "middle".into(),
            goal: "the store".into(),
            paths: vec!["crates/middle/src/lib.rs".into()],
            needs: vec!["top".into()],
            verify: "cargo check --offline".into(),
        });
        split.push(Subtask {
            id: "leaf".into(),
            goal: "the type".into(),
            paths: vec!["crates/leaf/src/lib.rs".into()],
            needs: vec!["middle".into()],
            verify: "cargo check --offline".into(),
        });
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert_eq!(decision.score.contradicted.len(), 3, "{:?}", decision.score);
    }
}
