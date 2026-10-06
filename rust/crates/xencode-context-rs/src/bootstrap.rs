//! Bootstrap the files a project xencode has never seen would rather already have.
//!
//! A fresh clone of somebody else's repository has none of the things this
//! product reads: no `AGENTS.md`, no `.xencode/anchor.md`, and — if the person
//! just sat down at this machine — no settings file to look at to find out which
//! keys even exist. `QK-9` is that first minute.
//!
//! **It is declarative, which is the whole design.** Nothing here calls a model,
//! runs a build, or executes a project's own tooling. The reason is on this
//! repository's own record: an agent inventing plausible configuration for a
//! project it had not read is how 9,371 lines of fiction came to be deleted. So
//! every line of a written file is either a fact read off the disk — a branch
//! name, a revision, a file that is there, a count of extensions — or a question
//! left blank for a person. `AGENTS.md` therefore names no build command, even in
//! a repository holding `Cargo.toml`: "this file is here" is a fact and "run
//! `cargo test`" is a guess about how this particular project is checked.
//!
//! **A file that exists is never written**, and there is no flag to make it
//! happen. `AGENTS.md` and the anchor belong to a human once a human has opened
//! them, and [`super::factgc`] already treats the durable facts in `state.md` as
//! things to disable rather than delete. `--check` reports the same list having
//! created nothing, `.xencode/` included.
//!
//! The settings template is generated from the struct this binary loads rather
//! than transcribed by hand, so a key that does not exist cannot appear in it and
//! a key that does cannot go missing. Its nine credential fields are what they
//! ship as: absent. A bootstrap that copied somebody's live `config.json` in
//! order to be helpful would put a key into a repository other people read and
//! other tools index.
//!
//! Two traps are worth naming because neither shows up in a diff. A file *name*
//! is somebody else's string, and the anchor quotes names, so every body passes
//! through the same scrub the transcript and the tombstone queue use, and the
//! report says when it had to. And the two words this capability might have been
//! called are already in the codebase for other things: [`super::seeds`]
//! generates the repositories an agent is graded on, and
//! `xencode-colab-rs/src/bootstrap.rs` writes the script pushed to a Colab VM.
//! Neither is a command a person can type, so this one takes the name the
//! research gave it — a bootstrap for a project — and the manuals say so.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use serde::Serialize;

use crate::gitinfo::{current_git_info, git_file_set, is_git_repo};

/// How many of each kind of line a person wants to read without scrolling.
const ROOT_FILES_SHOWN: usize = 12;
const EXTENSIONS_SHOWN: usize = 5;

/// What the run could see on disk, before any file is written.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ProjectFacts {
    /// Whether this directory is inside a git repository at all.
    pub is_git: bool,
    /// The checked-out branch, absent on a detached head.
    pub branch: Option<String>,
    /// The commit everything below is counted at, or `(unborn HEAD)`.
    pub revision: Option<String>,
    /// How many files git reports, where a repository exists to report them.
    pub files: Option<usize>,
    /// Names sitting at the repository root, sorted.
    pub root_files: Vec<String>,
    /// Filename extensions by count, most first, ties broken by name.
    pub extensions: Vec<(String, usize)>,
}

impl ProjectFacts {
    /// Read the repository. Nothing in here runs in it: every field is a name, a
    /// number, or the reason there is no number.
    pub fn gather(root: &Path) -> Self {
        let mut facts = ProjectFacts {
            is_git: is_git_repo(root),
            ..Default::default()
        };
        if !facts.is_git {
            return facts;
        }
        if let Some(git) = current_git_info(root) {
            // Read the label before moving the branch out of the same struct.
            let revision = git.revision_label();
            facts.branch = (!git.branch.is_empty()).then_some(git.branch);
            facts.revision = Some(revision);
        }
        let Some(set) = git_file_set(root) else {
            return facts;
        };
        facts.files = Some(set.len());
        let mut roots: Vec<String> = set
            .iter()
            .filter(|path| !path.contains('/'))
            .cloned()
            .collect();
        roots.sort();
        facts.root_files = roots;

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for path in &set {
            let name = path.rsplit('/').next().unwrap_or(path);
            if let Some((_, ext)) = name.rsplit_once('.') {
                let ext = ext.to_lowercase();
                if !ext.is_empty() {
                    *counts.entry(ext).or_default() += 1;
                }
            }
        }
        let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
        // Count first, name second: a tie has to land the same way on two
        // machines, or the anchor is not the byte-stable tier it is written into.
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        facts.extensions = ranked;
        facts
    }

    /// The one phrase every written file uses for where it came from.
    fn where_from(&self) -> String {
        // `revision_label` answers "(unborn HEAD)" for a repository with no
        // commit, and a branch name beside that reads like a revision exists.
        if self.revision.as_deref() == Some("(unborn HEAD)") {
            return "a git repository with no commit yet".to_string();
        }
        match (
            self.is_git,
            self.branch.as_deref(),
            self.revision.as_deref(),
        ) {
            (false, _, _) => "not a git repository".to_string(),
            (true, Some(branch), Some(rev)) => format!("git branch `{branch}` at `{rev}`"),
            (true, None, Some(rev)) => format!("detached head at `{rev}`"),
            (true, _, None) => "a git repository with no commit yet".to_string(),
        }
    }
}

/// One of the three files a bootstrap offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapKind {
    /// The instruction file this product reads into every prompt.
    Agents,
    /// The last tier of the byte-stable prompt head.
    Anchor,
    /// Every setting the binary knows, at its default, with no credentials.
    SettingsExample,
}

impl BootstrapKind {
    /// Where this file goes, relative to the project root.
    ///
    /// The anchor is `.xencode/anchor.md` because that is the only path the
    /// prompt reader looks at — a root-level `anchor.md` is never read, so
    /// writing one would leave a file nothing consumes.
    pub fn rel_path(self) -> &'static str {
        match self {
            BootstrapKind::Agents => "AGENTS.md",
            BootstrapKind::Anchor => ".xencode/anchor.md",
            BootstrapKind::SettingsExample => ".xencode.example.json",
        }
    }

    /// Whether this body has to pass the credential scrub on its way out.
    ///
    /// `AGENTS.md` and the anchor quote names off somebody's disk, and a file
    /// name is somebody else's string. The settings template quotes nothing: it
    /// is generated from a struct whose credential fields are absent, so there
    /// is nothing here to scrub — and scrubbing it would be a defect, because
    /// `redact_secrets` replaces the *value* beside a key that looks like a
    /// credential, turning `"openai_api_key": null` into a string a person could
    /// copy out as if it were a value. That rewrite was watched happening.
    fn carries_disk_text(self) -> bool {
        !matches!(self, BootstrapKind::SettingsExample)
    }

    /// Why the file is offered, in the words the command prints.
    fn purpose(self) -> &'static str {
        match self {
            BootstrapKind::Agents => "questions only: nothing ran, so no command is guessed",
            BootstrapKind::Anchor => "what was read off this disk, with no build or model in it",
            BootstrapKind::SettingsExample => {
                "every key this binary reads, at its default, credentials absent"
            }
        }
    }
}

/// What a run decided about one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BootstrapEntry {
    /// Project-relative path, always with `/` separators.
    pub path: String,
    /// `write` for a file that was not there, `keep` for one that was and stays.
    pub action: &'static str,
    /// The sentence naming why, for a person reading the terminal.
    pub reason: String,
}

/// The result of one bootstrap run, whether it wrote or only reported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BootstrapReport {
    /// The directory, resolved to an absolute path.
    pub root: String,
    /// What the disk said.
    pub facts: ProjectFacts,
    /// One entry per offered file, in the order they are offered.
    pub entries: Vec<BootstrapEntry>,
    /// Whether a body built from names on this disk had something
    /// credential-shaped taken out of it on the way here.
    pub scrubbed: bool,
    /// Whether this was a report only.
    pub check_only: bool,
}

impl BootstrapReport {
    /// The files that were not there. Under `--check` these are the ones only
    /// offered; `check_only` is what says whether any byte moved.
    pub fn writing(&self) -> impl Iterator<Item = &BootstrapEntry> {
        self.entries.iter().filter(|e| e.action == "write")
    }

    /// The files left exactly as they were.
    pub fn keeping(&self) -> impl Iterator<Item = &BootstrapEntry> {
        self.entries.iter().filter(|e| e.action == "keep")
    }
}

/// One offered file: where it goes and what it would say.
pub struct OfferedFile {
    /// Which of the three this is.
    pub kind: BootstrapKind,
    /// The exact bytes, before the credential scrub.
    pub body: String,
}

/// The `AGENTS.md` a bootstrap leaves behind: headings this product enforces,
/// and no answers.
///
/// The blank sections are the point. A written instruction file that said
/// "follow the existing style" would hand the model a line to obey that no
/// person ever wrote — which is exactly what `/lesson approve` exists to stop.
fn agents_text(facts: &ProjectFacts) -> String {
    let seen = if facts.files.is_some() {
        "It read the file list git gives, and nothing else."
    } else {
        "This directory is not a git repository, so no file list was read."
    };
    format!(
        concat!(
            "# Agent instructions\n\n",
            "`xencode bootstrap` put these headings here, because this product reads\n",
            "this file at the start of every prompt and this repository had none. {seen}\n",
            "Nothing below is a claim about this project: no build ran, no test ran, no\n",
            "model was asked, so it does not know a command, a convention, or a path to\n",
            "stay out of. Answer the questions, and delete any heading you would not\n",
            "enforce. What is written here is what an agent is held to.\n\n",
            "Repository as of this file: {here}.\n\n",
            "## What this project is, and what it is for\n\n",
            "## How to check your own work\n",
            "<!-- One command, that an agent can run and be told the answer by. A check\n",
            "     that is not written here is a check that never happens. -->\n\n",
            "## Conventions to follow\n\n",
            "## What not to touch\n",
            "<!-- Generated files, vendored code, a branch nobody should rewrite. Said\n",
            "     once here instead of once per session. -->\n",
        ),
        seen = seen,
        here = facts.where_from(),
    )
}

/// The `.xencode/anchor.md` a bootstrap leaves behind.
///
/// This text is the tail of the prompt head that stays byte-identical between
/// turns, so it carries no clock and no absolute path — a date in it would make
/// every later request pay for a cache miss. `xencode anchor` renders its own
/// version into the same path from what it discovers and proves in the code, and
/// running that replaces these lines; that is a person asking, so it is allowed.
fn anchor_text(facts: &ProjectFacts) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "# Anchor\n\n\
         Written by `xencode bootstrap` from what is on disk. Every line below is a name,\n\
         a number, or the reason there is no number: no build ran, no test ran, no model\n\
         was asked. Nothing here says how this project is built or checked, and a fact in\n\
         this file is not evidence — it is what the tree looked like when this was\n\
         written. A durable fact the code contradicts goes to `xencode memory gc`.\n\n",
    );
    let _ = writeln!(out, "- repository: {}", facts.where_from());
    match facts.files {
        Some(n) => {
            let _ = writeln!(out, "- files git reports here: {n}");
        }
        None => out.push_str(
            "- files: none listed; the bootstrap gets this from git and did not get it\n",
        ),
    }
    if !facts.root_files.is_empty() {
        let shown: Vec<String> = facts
            .root_files
            .iter()
            .take(ROOT_FILES_SHOWN)
            .map(|name| format!("`{name}`"))
            .collect();
        let more = facts.root_files.len() - shown.len();
        let _ = writeln!(
            out,
            "- files at the repository root: {}{}",
            shown.join(", "),
            if more == 0 {
                String::new()
            } else {
                format!(", and {more} more")
            }
        );
    }
    if !facts.extensions.is_empty() {
        let shown: Vec<String> = facts
            .extensions
            .iter()
            .take(EXTENSIONS_SHOWN)
            .map(|(ext, n)| format!(".{ext} ({n})"))
            .collect();
        let _ = writeln!(out, "- extensions by count: {}", shown.join(", "));
    }
    out.push_str(
        "\nRecognising a file name is not a claim about it. That `Cargo.toml` is here\n\
         means the file is here, not that `cargo` is how this project is checked.\n",
    );
    out
}

/// The three files and their bytes, from the facts already gathered.
///
/// `settings_template` arrives from the caller because the config structs live
/// in another crate: the CLI generates it from the real `XencodeConfig`, and this
/// module decides where the bytes go.
pub fn offered_files(facts: &ProjectFacts, settings_template: &str) -> Vec<OfferedFile> {
    let mut template = settings_template.to_string();
    if !template.ends_with('\n') {
        template.push('\n');
    }
    vec![
        OfferedFile {
            kind: BootstrapKind::Agents,
            body: agents_text(facts),
        },
        OfferedFile {
            kind: BootstrapKind::Anchor,
            body: anchor_text(facts),
        },
        OfferedFile {
            kind: BootstrapKind::SettingsExample,
            body: template,
        },
    ]
}

/// Write what is missing, or report it and touch nothing.
pub fn bootstrap(root: &Path, settings_template: &str, check: bool) -> io::Result<BootstrapReport> {
    let root = root.canonicalize()?;
    let facts = ProjectFacts::gather(&root);
    let mut entries = Vec::new();
    let mut scrubbed = false;

    for offered in offered_files(&facts, settings_template) {
        let rel = offered.kind.rel_path();
        let body = if offered.kind.carries_disk_text() {
            let clean = crate::trace::redact_secrets(&offered.body);
            if clean != offered.body {
                scrubbed = true;
            }
            clean
        } else {
            offered.body
        };
        if root.join(rel).exists() {
            entries.push(BootstrapEntry {
                path: rel.to_string(),
                action: "keep",
                reason: "already here, and this command does not replace a person's file"
                    .to_string(),
            });
            continue;
        }
        if !check {
            match offered.kind {
                // The anchor has an owner already; a second writer for one path
                // would be two places to change when the stable head moves.
                BootstrapKind::Anchor => {
                    crate::anchor::write_anchor(&root, &body)
                        .map_err(|problem| io::Error::other(problem.to_string()))?;
                }
                _ => {
                    xencode_core_rs::write_atomic(&root.join(rel), body.as_bytes())?;
                }
            }
        }
        entries.push(BootstrapEntry {
            path: rel.to_string(),
            action: "write",
            reason: offered.kind.purpose().to_string(),
        });
    }

    Ok(BootstrapReport {
        root: root.display().to_string(),
        facts,
        entries,
        scrubbed,
        check_only: check,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The anchor is read into every prompt, so a repository with four hundred
    /// root files must not put four hundred names there. What is cut has to be
    /// said, not hidden.
    #[test]
    fn a_long_file_list_is_cut_short_and_says_it_was() {
        let facts = ProjectFacts {
            is_git: true,
            branch: Some("main".to_string()),
            revision: Some("1a2b3c4d".to_string()),
            files: Some(400),
            root_files: (0..15).map(|i| format!("f{i}.rs")).collect(),
            extensions: (0..8).map(|i| (format!("e{i}"), 30 - i)).collect(),
        };
        let text = anchor_text(&facts);
        assert!(text.contains("`f0.rs`"), "{text}");
        assert!(
            text.contains("and 3 more"),
            "root files shown: {ROOT_FILES_SHOWN}"
        );
        assert!(!text.contains("`f14.rs`"));
        assert!(text.contains(".e0 (30), .e1 (29), .e2 (28), .e3 (27), .e4 (26)"));
        assert!(!text.contains(".e5"));
    }

    /// The written `AGENTS.md` must not contain the sentence it would be famous
    /// for inventing: a command.
    #[test]
    fn the_written_instructions_guess_no_command() {
        let facts = ProjectFacts {
            is_git: true,
            branch: Some("main".to_string()),
            revision: Some("1a2b3c4d".to_string()),
            files: Some(214),
            root_files: vec!["Cargo.toml".to_string()],
            extensions: vec![("rs".to_string(), 120)],
        };
        let text = agents_text(&facts);
        for forbidden in [
            "cargo test",
            "cargo build",
            "cargo check",
            "npm run",
            "npm test",
            "pnpm",
            "yarn ",
            "pytest",
            "make test",
            "go test",
        ] {
            assert!(
                !text.contains(forbidden),
                "the written file tells the model to run {forbidden:?}, which nothing checked"
            );
        }
        assert!(text.contains("no build ran, no test ran"));
        assert!(text.contains("What is written here is what an agent is held to."));
    }

    /// No clock, no absolute path: this file sits inside the byte-stable head.
    #[test]
    fn the_written_anchor_holds_no_clock_and_no_absolute_path() {
        let facts = ProjectFacts {
            is_git: true,
            branch: Some("main".to_string()),
            revision: Some("1a2b3c4d".to_string()),
            files: Some(3),
            root_files: vec!["Cargo.toml".to_string(), "README.md".to_string()],
            extensions: vec![("rs".to_string(), 2), ("md".to_string(), 1)],
        };
        let text = anchor_text(&facts);
        assert!(
            !text.contains("/home/"),
            "an absolute path does not belong here"
        );
        assert!(text.contains("- repository: git branch `main` at `1a2b3c4d`"));
        assert!(text.contains("- files at the repository root: `Cargo.toml`, `README.md`"));
        assert!(text.contains("- files git reports here: 3"));
        assert!(text.contains("Recognising a file name is not a claim about it"));
    }

    /// A detached head and an unborn one are both real states, and each has to be
    /// said rather than left blank.
    #[test]
    fn a_head_with_no_branch_or_no_commit_is_still_described() {
        let detached = ProjectFacts {
            is_git: true,
            branch: None,
            revision: Some("1a2b3c4d".to_string()),
            ..Default::default()
        };
        assert_eq!(detached.where_from(), "detached head at `1a2b3c4d`");
        let unborn = ProjectFacts {
            is_git: true,
            branch: Some("main".to_string()),
            revision: Some("(unborn HEAD)".to_string()),
            ..Default::default()
        };
        assert!(anchor_text(&unborn).contains("- repository: a git repository with no commit yet"));
        assert!(anchor_text(&unborn).contains("- files: none listed"));
    }

    #[test]
    fn a_directory_that_is_not_a_repository_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let facts = ProjectFacts::gather(dir.path());
        assert!(!facts.is_git);
        assert_eq!(facts.files, None);
        assert!(anchor_text(&facts).contains("not a git repository"));
        assert!(agents_text(&facts).contains("no file list was read"));
    }

    #[test]
    fn the_three_offered_paths_are_the_three_this_product_reads() {
        assert_eq!(BootstrapKind::Agents.rel_path(), "AGENTS.md");
        assert_eq!(BootstrapKind::Anchor.rel_path(), ".xencode/anchor.md");
        assert_eq!(
            BootstrapKind::SettingsExample.rel_path(),
            ".xencode.example.json"
        );
    }

    /// A template with no final newline would end the file mid-sentence for
    /// whatever reads it next, and a diff of a file without a trailing newline is
    /// its own small nuisance.
    #[test]
    fn a_template_loses_no_byte_and_gains_the_one_it_needs() {
        let facts = ProjectFacts::default();
        let offered = offered_files(&facts, "{\"a\":1}");
        let template = &offered[2];
        assert_eq!(template.body, "{\"a\":1}\n");
        let offered = offered_files(&facts, "{\"a\":1}\n");
        assert_eq!(offered[2].body, "{\"a\":1}\n");
        assert_eq!(offered.len(), 3);
    }

    /// A file name is somebody else's string, and the anchor quotes names into a
    /// file meant to be committed. The end of this, over a real repository, is in
    /// `tests/bootstrap.rs`.
    #[test]
    fn a_credential_shaped_file_name_does_not_survive_the_scrub() {
        let facts = ProjectFacts {
            is_git: true,
            branch: Some("main".to_string()),
            revision: Some("1a2b3c4d".to_string()),
            files: Some(1),
            root_files: vec!["sk-FAKE-NOT-A-REAL-TEST-KEY.txt".to_string()],
            extensions: vec![("txt".to_string(), 1)],
        };
        let scrubbed = crate::trace::redact_secrets(&anchor_text(&facts));
        assert!(scrubbed.contains("[redacted]"), "{scrubbed}");
        assert!(!scrubbed.contains("sk-FAKE"));
    }
}
