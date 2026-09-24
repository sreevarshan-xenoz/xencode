//! Seeded task repositories (QA-5) — the small programs an agent is graded on.
//!
//! EV-1 measures whether the agent can fix something. To measure that, the
//! something has to exist before the run, be written by nobody in a hurry, and
//! fail for exactly one reason. So these repositories are generated from a
//! declared description of each case — the shape of the defect, the files, the
//! symptom the task states, the behaviour the grader checks, and the smallest
//! change that makes it pass. No model is asked to invent a bug: everything
//! here is data in this file, which means a measurement taken twice is taken
//! against the same code.
//!
//! Eight shapes, chosen because they are the ones an agent gets wrong in
//! practice and an exit code can tell apart:
//!
//! - [`BugShape::OffByOne`] — a loop stops one short
//! - [`BugShape::MissingValue`] — a lookup that should fall back stops the program
//! - [`BugShape::EarlyReturn`] — an answer is given before the better rule is read
//! - [`BugShape::SwallowedError`] — bad input is skipped instead of reported
//! - [`BugShape::InvertedCondition`] — a comparison points the wrong way
//! - [`BugShape::IgnoredResult`] — a result that says whether it worked is dropped
//! - [`BugShape::LostUpdate`] — two workers read the same value and one write is eaten
//! - [`BugShape::StaleCache`] — a remembered value outlives the thing it came from
//!
//! Two rules the generator exists to keep:
//!
//! - **A fresh repository per case.** [`write_seed`] runs `git init` in the new
//!   directory and commits everything, so the tree starts clean and its history
//!   is one commit long. Retrieval seeds part of its ranking from the files git
//!   reports as changed (`retrieve.rs`), so a case unpacked into somebody else's
//!   repository would be indexed differently for reasons that have nothing to do
//!   with the defect — and the difference would be invisible in the result.
//! - **The answer stays out of the tree.** A case carries its own fix as data,
//!   and that text is written nowhere inside the repository: not in `task.md`,
//!   not in a comment, not in a file the grader reads. An agent can still open
//!   the grader and read the expected values, which is a real limit of this
//!   design rather than something the layout fixes — what it cannot do is read
//!   the change that was actually made.
//!
//! Grading is an exit code from `cargo test --offline` inside the case. The
//! generated programs use nothing outside the standard library, so a suite of
//! them runs with no network and no registry.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::gitinfo::git_stdout;

/// The defect a seeded case is built around.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BugShape {
    OffByOne,
    MissingValue,
    EarlyReturn,
    SwallowedError,
    InvertedCondition,
    IgnoredResult,
    LostUpdate,
    StaleCache,
}

/// Every shape, in the order a suite walks them.
pub const SEED_SHAPES: [BugShape; 8] = [
    BugShape::OffByOne,
    BugShape::MissingValue,
    BugShape::EarlyReturn,
    BugShape::SwallowedError,
    BugShape::InvertedCondition,
    BugShape::IgnoredResult,
    BugShape::LostUpdate,
    BugShape::StaleCache,
];

impl BugShape {
    /// All eight, as a slice for callers that want to iterate.
    pub fn all() -> &'static [BugShape] {
        &SEED_SHAPES
    }

    /// Directory name, package name and the slug a report prints.
    pub fn slug(self) -> &'static str {
        match self {
            BugShape::OffByOne => "off-by-one",
            BugShape::MissingValue => "missing-value",
            BugShape::EarlyReturn => "early-return",
            BugShape::SwallowedError => "swallowed-error",
            BugShape::InvertedCondition => "inverted-condition",
            BugShape::IgnoredResult => "ignored-result",
            BugShape::LostUpdate => "lost-update",
            BugShape::StaleCache => "stale-cache",
        }
    }

    /// One line naming the defect for a report header. Plain words, because it
    /// is printed next to a pass rate that has to mean something to a stranger.
    pub fn title(self) -> &'static str {
        match self {
            BugShape::OffByOne => "a total leaves out the last reading",
            BugShape::MissingValue => "a missing setting stops the program",
            BugShape::EarlyReturn => "a price answers before the better rule",
            BugShape::SwallowedError => "an unreadable line is skipped in silence",
            BugShape::InvertedCondition => "a licence date is compared the wrong way",
            BugShape::IgnoredResult => "a rejected name is counted as added",
            BugShape::LostUpdate => "two workers, one of the additions lost",
            BugShape::StaleCache => "a renamed report still shows the old title",
        }
    }
}

/// The crate name a case's test file imports.
fn crate_name(slug: &str) -> String {
    format!("seed-{}", slug)
}

fn rust_name(slug: &str) -> String {
    slug.replace('-', "_")
}

/// One file inside a seeded repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedFile {
    pub path: String,
    pub body: String,
}

/// The change that makes a case pass: text to find, text to put in its place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedEdit {
    pub path: String,
    pub find: String,
    pub replace: String,
}

/// A case as declared, before anything is written to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedCase {
    pub shape: BugShape,
    /// Where the repository goes: `<parent>/<slug>`.
    pub slug: String,
    pub title: String,
    /// The words the agent is given. A symptom, never the change.
    pub task: String,
    pub files: Vec<SeedFile>,
    /// The reference change, held by the harness and written nowhere in the tree.
    pub edits: Vec<SeedEdit>,
    /// The files a correct answer is expected to touch, for grading the diff
    /// rather than the chat.
    pub expected_files: Vec<String>,
    /// The command whose exit code decides the case.
    pub grader: Vec<String>,
}

/// The declared description of one shape.
pub fn seed_case(shape: BugShape) -> SeedCase {
    let slug = shape.slug().to_string();
    let package = crate_name(&slug);
    let import = format!("seed_{}", rust_name(&slug));
    let (task, source, test, edits): (String, String, String, Vec<(&str, &str, &str)>) = match shape
    {
        BugShape::OffByOne => (
            "The daily report built from `total` comes out short, and a day with no \
             readings at all makes the job stop instead of reporting zero. Make both \
             cases answer correctly. \
             Don't change what a reading is; change what the function does with them."
                .to_string(),
            r#"//! Adds up the readings a sensor reported.

/// Total of the readings, in the order they arrived.
pub fn total(readings: &[i32]) -> i32 {
    let mut sum = 0;
    for index in 0..readings.len() - 1 {
        sum += readings[index];
    }
    sum
}
"#
            .to_string(),
            r#"use __CRATE__::total;

#[test]
fn adds_every_reading() {
    assert_eq!(total(&[3, 4, 5]), 12);
}

#[test]
fn a_day_with_no_readings_is_zero() {
    assert_eq!(total(&[]), 0);
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "    for index in 0..readings.len() - 1 {",
                "    for index in 0..readings.len() {",
            )],
        ),
        BugShape::MissingValue => (
            "Starting with a config that leaves `port` out stops the program on its \
             face, and so does a config that spells the port badly. Neither should be \
             fatal: a sensible default exists in this crate already. Keep honouring a \
             port that is written correctly."
                .to_string(),
            r#"//! Reads the settings a server starts from.

use std::collections::HashMap;

/// What the server listens on when nothing is configured.
pub const DEFAULT_PORT: u16 = 8080;

/// The port to bind, from the settings a caller was given.
pub fn port(settings: &HashMap<String, String>) -> u16 {
    settings.get("port").unwrap().parse::<u16>().unwrap()
}
"#
            .to_string(),
            r#"use std::collections::HashMap;

use __CRATE__::{port, DEFAULT_PORT};

fn settings(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn uses_a_port_that_is_written_out() {
    let given = settings(&[("port", "8123")]);
    assert_eq!(port(&given), 8123);
}

#[test]
fn absent_port_starts_the_default() {
    assert_eq!(port(&settings(&[("host", "localhost")])), DEFAULT_PORT);
}

#[test]
fn a_port_that_is_not_a_number_starts_the_default() {
    assert_eq!(port(&settings(&[("port", "http")])), DEFAULT_PORT);
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "    settings.get(\"port\").unwrap().parse::<u16>().unwrap()",
                "    settings\n        .get(\"port\")\n        .and_then(|value| value.parse::<u16>().ok())\n        .unwrap_or(DEFAULT_PORT)",
            )],
        ),
        BugShape::EarlyReturn => (
            "Members pay less than anyone else, whatever day it is. On a weekend the \
             shop also takes a tenth off for everybody. Some customers are being \
             charged the wrong amount, and it is the ones who should be paying least."
                .to_string(),
            r#"//! The price of an item after the shop's discounts.

/// Price in cents after the shop's discounts.
pub fn price_cents(base: u32, member: bool, weekend: bool) -> u32 {
    if weekend {
        return base * 9 / 10;
    }
    if member {
        return base * 8 / 10;
    }
    base
}
"#
            .to_string(),
            r#"use __CRATE__::price_cents;

#[test]
fn a_member_pays_the_member_price_on_a_weekday() {
    assert_eq!(price_cents(1000, true, false), 800);
}

#[test]
fn a_member_keeps_the_member_price_on_a_weekend() {
    assert_eq!(price_cents(1000, true, true), 800);
}

#[test]
fn a_weekend_takes_a_tenth_off_for_everybody_else() {
    assert_eq!(price_cents(1000, false, true), 900);
}

#[test]
fn nobody_is_discounted_on_a_weekday() {
    assert_eq!(price_cents(1000, false, false), 1000);
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "    if weekend {\n        return base * 9 / 10;\n    }\n    if member {\n        return base * 8 / 10;\n    }",
                "    if member {\n        return base * 8 / 10;\n    }\n    if weekend {\n        return base * 9 / 10;\n    }",
            )],
        ),
        BugShape::SwallowedError => (
            "A settings file with a line that is not `key=value` should be reported to \
             whoever asked for it, because right now it is dropped and the program \
             starts without the option the author thought they had written. Lines that \
             can be read still have to be returned."
                .to_string(),
            r#"//! Reads `key=value` lines into settings.

/// Parse the lines of a settings file.
pub fn parse(lines: &[&str]) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for line in lines {
        if let Some((key, value)) = line.split_once('=') {
            pairs.push((key.to_string(), value.to_string()));
        }
    }
    Ok(pairs)
}
"#
            .to_string(),
            r#"use __CRATE__::parse;

#[test]
fn reads_the_lines_that_are_written_correctly() {
    let pairs = parse(&["host = localhost", "port = 8080"]).expect("both lines are readable");
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, "host ");
}

#[test]
fn a_line_that_is_not_key_value_is_reported() {
    assert!(parse(&["port = 8080", "nonsense"]).is_err());
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "        if let Some((key, value)) = line.split_once('=') {\n            pairs.push((key.to_string(), value.to_string()));\n        }",
                "        let Some((key, value)) = line.split_once('=') else {\n            return Err(format!(\"this line is not key=value: {line}\"));\n        };\n        pairs.push((key.to_string(), value.to_string()));",
            )],
        ),
        BugShape::InvertedCondition => (
            "A licence is valid on its last day and not the day after it. Drivers are \
             being told they may not drive on days they may, and may drive on a day \
             their licence has just run out."
                .to_string(),
            r#"//! Whether a driver may still be on the road.

/// Whether the licence is still valid today. Its last day counts.
pub fn may_drive(valid_until_day: u32, today: u32) -> bool {
    today > valid_until_day
}
"#
            .to_string(),
            r#"use __CRATE__::may_drive;

#[test]
fn a_licence_held_for_years_is_still_good() {
    assert!(may_drive(10, 3));
}

#[test]
fn the_last_valid_day_is_valid() {
    assert!(may_drive(10, 10));
}

#[test]
fn the_day_after_the_last_one_is_not() {
    assert!(!may_drive(10, 11));
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "    today > valid_until_day",
                "    today <= valid_until_day",
            )],
        ),
        BugShape::IgnoredResult => (
            "The count `add_all` hands back is supposed to say how many names made it \
             onto the list. It does not, for some names, and the list itself is right \
             even when the number is not."
                .to_string(),
            r#"//! A guest list that turns some names away.

/// Append a name to the guest list. A name with a space in it is turned away.
#[must_use = "whether the name made it onto the list"]
pub fn add(guests: &mut Vec<String>, name: &str) -> Result<(), String> {
    if name.contains(' ') {
        return Err(format!("not one name: {name}"));
    }
    guests.push(name.to_string());
    Ok(())
}

/// Add every name and report how many are now on the list.
pub fn add_all(guests: &mut Vec<String>, names: &[&str]) -> usize {
    let mut added = 0;
    for name in names {
        add(guests, name);
        added += 1;
    }
    added
}
"#
            .to_string(),
            r#"use __CRATE__::{add_all, add};

#[test]
fn names_that_are_accepted_are_counted() {
    let mut guests = Vec::new();
    assert_eq!(add_all(&mut guests, &["ana", "bo"]), 2);
    assert_eq!(guests.len(), 2);
}

#[test]
fn a_name_that_is_turned_away_is_not_counted() {
    let mut guests = Vec::new();
    assert_eq!(add_all(&mut guests, &["ana", "bob smith"]), 1);
    assert_eq!(guests, vec!["ana".to_string()]);
}

#[test]
fn a_single_name_still_lands_on_the_list() {
    let mut guests = Vec::new();
    assert!(add(&mut guests, "zoe").is_ok());
    assert_eq!(guests, vec!["zoe".to_string()]);
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "        add(guests, name);\n        added += 1;",
                "        if add(guests, name).is_ok() {\n            added += 1;\n        }",
            )],
        ),
        BugShape::LostUpdate => (
            "Two workers each add one to the same counter, and the finished count comes \
             out as one. The count is the problem; the workers are doing what they were \
             asked."
                .to_string(),
            r#"//! Two workers adding to one counter.

use std::sync::{Arc, Mutex};
use std::thread;

/// Add one to the shared counter, as one worker.
pub fn bump(counters: &Arc<Mutex<usize>>, worker: usize) {
    let seen = { *counters.lock().unwrap() };
    thread::sleep(std::time::Duration::from_millis(40 * (worker as u64 + 1)));
    let updated = seen + 1;
    *counters.lock().unwrap() = updated;
}

/// Run two workers over one counter and report what it holds when they finish.
pub fn bump_twice(counters: Arc<Mutex<usize>>) -> usize {
    let first = counters.clone();
    let second = counters.clone();
    let a = thread::spawn(move || bump(&first, 0));
    let b = thread::spawn(move || bump(&second, 1));
    a.join().expect("worker one finished");
    b.join().expect("worker two finished");
    *counters.lock().unwrap()
}
"#
            .to_string(),
            r#"use std::sync::{Arc, Mutex};

use __CRATE__::bump_twice;

#[test]
fn two_workers_add_two() {
    let counters = Arc::new(Mutex::new(0usize));
    assert_eq!(bump_twice(counters.clone()), 2);
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "    let seen = { *counters.lock().unwrap() };\n    thread::sleep(std::time::Duration::from_millis(40 * (worker as u64 + 1)));\n    let updated = seen + 1;\n    *counters.lock().unwrap() = updated;",
                "    let mut seen = counters.lock().unwrap();\n    let before = *seen;\n    thread::sleep(std::time::Duration::from_millis(40 * (worker as u64 + 1)));\n    *seen = before + 1;",
            )],
        ),
        BugShape::StaleCache => (
            "A report is expensive to render, so it keeps the last one it made. \
             Renaming the report changes what later renders show for nobody else, and \
             for the person who asked for the rename either. Rendering itself is fine."
                .to_string(),
            r#"//! A rendered report that remembers its last answer.

use std::cell::RefCell;

/// A titled thing whose rendering is worth remembering.
pub struct Report {
    title: RefCell<String>,
    cached: RefCell<Option<String>>,
}

impl Report {
    pub fn new(title: &str) -> Self {
        Report {
            title: RefCell::new(title.to_string()),
            cached: RefCell::new(None),
        }
    }

    /// Give the report a new title.
    pub fn rename(&self, title: &str) {
        *self.title.borrow_mut() = title.to_string();
    }

    /// The line the report prints.
    pub fn render(&self) -> String {
        let mut cached = self.cached.borrow_mut();
        if let Some(text) = cached.as_ref() {
            return text.clone();
        }
        let text = format!("-- {} --", self.title.borrow());
        *cached = Some(text.clone());
        text
    }
}
"#
            .to_string(),
            r#"use __CRATE__::Report;

#[test]
fn a_report_renders_its_title() {
    assert_eq!(Report::new("alpha").render(), "-- alpha --");
}

#[test]
fn a_renamed_report_renders_the_new_title() {
    let report = Report::new("alpha");
    assert_eq!(report.render(), "-- alpha --");
    report.rename("omega");
    assert_eq!(report.render(), "-- omega --");
}

#[test]
fn rendering_the_same_title_twice_still_agrees_with_itself() {
    let report = Report::new("alpha");
    assert_eq!(report.render(), report.render());
}
"#
            .to_string(),
            vec![(
                "src/lib.rs",
                "        *self.title.borrow_mut() = title.to_string();",
                "        *self.title.borrow_mut() = title.to_string();\n        *self.cached.borrow_mut() = None;",
            )],
        ),
    };

    let manifest = format!(
        r#"[package]
name = "{package}"
version = "0.0.0"
edition = "2021"

# Its own workspace root, so a case generated inside another tree is not read
# as a member of that tree's build.
[workspace]

[dependencies]
"#
    );

    let task_page = format!("# {}\n\n{}\n", shape.title(), task);

    SeedCase {
        shape,
        slug: slug.clone(),
        title: shape.title().to_string(),
        task,
        files: vec![
            SeedFile {
                path: "Cargo.toml".to_string(),
                body: manifest,
            },
            SeedFile {
                path: ".gitignore".to_string(),
                body: "/target\nCargo.lock\n".to_string(),
            },
            SeedFile {
                path: "task.md".to_string(),
                body: task_page,
            },
            SeedFile {
                path: "src/lib.rs".to_string(),
                body: source,
            },
            SeedFile {
                path: "tests/behaviour.rs".to_string(),
                body: test.replace("__CRATE__", &import),
            },
        ],
        edits: edits
            .into_iter()
            .map(|(path, find, replace)| SeedEdit {
                path: path.to_string(),
                find: find.to_string(),
                replace: replace.to_string(),
            })
            .collect(),
        expected_files: vec!["src/lib.rs".to_string()],
        grader: vec![
            "cargo".to_string(),
            "test".to_string(),
            "--offline".to_string(),
        ],
    }
}

/// Every case the generator knows, in [`SEED_SHAPES`] order.
pub fn all_seed_cases() -> Vec<SeedCase> {
    BugShape::all().iter().copied().map(seed_case).collect()
}

/// What one grader run reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraderRun {
    /// The command, joined for a report line.
    pub command: String,
    pub passed: bool,
    /// `None` when the command could not be started at all.
    pub exit_code: Option<i32>,
    /// The last lines of what the grader wrote, so a failure can be read.
    pub tail: String,
    pub elapsed_ms: u64,
}

/// A case as it exists on disk, with the reference change still held aside.
#[derive(Debug, Clone)]
pub struct SeededTask {
    pub case: SeedCase,
    pub path: PathBuf,
}

impl SeededTask {
    /// The reference change, applied in place. Returns the files it touched.
    pub fn apply_reference_fix(&self) -> Result<Vec<String>, String> {
        let mut touched = Vec::new();
        for edit in &self.case.edits {
            let full = self.path.join(&edit.path);
            let text = std::fs::read_to_string(&full)
                .map_err(|e| format!("could not read {}: {e}", edit.path))?;
            let hits = text.matches(edit.find.as_str()).count();
            if hits != 1 {
                return Err(format!(
                    "{} matches the reference change in {} {}, not once",
                    edit.path,
                    self.case.slug,
                    if hits == 0 { "nowhere" } else { "twice" }
                ));
            }
            let fixed = text.replacen(&edit.find, &edit.replace, 1);
            write_atomically(&full, &fixed)?;
            if !touched.contains(&edit.path) {
                touched.push(edit.path.clone());
            }
        }
        Ok(touched)
    }

    /// Run the case's grader and report its exit code.
    pub fn run_grader(&self) -> Result<GraderRun, String> {
        let started = Instant::now();
        let output = Command::new(&self.case.grader[0])
            .args(&self.case.grader[1..])
            .current_dir(&self.path)
            .env("CARGO_NET_OFFLINE", "true")
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .map_err(|e| format!("the grader could not be started: {e}"))?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        let tail = text
            .lines()
            .rev()
            .take(12)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        Ok(GraderRun {
            command: self.case.grader.join(" "),
            passed: output.status.success(),
            exit_code: output.status.code(),
            tail,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// Paths git considers changed, which is what a diff is graded from.
    pub fn changed_paths(&self) -> Vec<String> {
        crate::gitinfo::dirty_paths(&self.path)
    }

    /// How many commits the repository has. A seeded case starts at one.
    pub fn commit_count(&self) -> Result<usize, String> {
        let out = git_stdout(&self.path, &["rev-list", "--count", "HEAD"])?;
        Ok(out.trim().parse().unwrap_or(0))
    }
}

/// Write one case out under `parent`, as its own repository with one commit.
pub fn write_seed(parent: &Path, shape: BugShape) -> Result<SeededTask, String> {
    let case = seed_case(shape);
    let path = parent.join(&case.slug);
    if path.exists() {
        return Err(format!(
            "{} already exists; a case is written into an empty directory so its \
             repository history is its own",
            path.display()
        ));
    }
    for file in &case.files {
        let full = path.join(&file.path);
        if let Some(dir) = full.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("could not create {:?}: {e}", dir))?;
        }
        std::fs::write(&full, &file.body).map_err(|e| format!("could not write: {e}"))?;
    }
    init_repository(&path, &case.slug)?;
    Ok(SeededTask { case, path })
}

/// A fresh repository with everything committed, so a case begins clean and its
/// own history. The identity and signing options are passed per command rather
/// than read from whoever's `git config` this happens to run under, and the seed
/// commit skips hooks: a machine with `core.hooksPath` set globally would
/// otherwise run its own project hooks against a throwaway repository.
fn init_repository(path: &Path, slug: &str) -> Result<(), String> {
    git_stdout(path, &["init", "--quiet", "."])?;
    git_stdout(path, &["add", "-A"])?;
    git_stdout(
        path,
        &[
            "-c",
            "user.name=xencode-seed",
            "-c",
            "user.email=seed@xencode.local",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--no-verify",
            "-m",
            &format!("seed: {slug}"),
        ],
    )?;
    Ok(())
}

fn write_atomically(path: &Path, text: &str) -> Result<(), String> {
    let temp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    std::fs::write(&temp, text).map_err(|e| format!("could not write {:?}: {e}", temp))?;
    std::fs::rename(&temp, path).map_err(|e| format!("could not replace {:?}: {e}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-seeds-{}-{}-{}",
            name,
            std::process::id(),
            std::time::UNIX_EPOCH.elapsed().unwrap().subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Every shape has a task, files, a change that applies exactly once, and a
    /// grader — the generator is one code path for all eight, so a shape that
    /// was wired up wrong would otherwise sit unnoticed in a report.
    #[test]
    fn every_shape_declares_a_case_the_generator_can_write() {
        let cases = all_seed_cases();
        assert_eq!(cases.len(), 8, "the eight shapes the plan named");
        let mut slugs = std::collections::HashSet::new();
        for case in &cases {
            assert!(!case.task.is_empty(), "{:?} has nothing to say", case.shape);
            assert!(
                case.files.iter().any(|f| f.path == "src/lib.rs"),
                "{:?} has no source to be wrong in",
                case.shape
            );
            assert!(
                case.files.iter().any(|f| f.path == "tests/behaviour.rs"),
                "{:?} has nothing to grade it",
                case.shape
            );
            assert!(!case.edits.is_empty(), "{:?} has no fix", case.shape);
            assert_eq!(case.grader.join(" "), "cargo test --offline");
            assert!(
                slugs.insert(case.slug.clone()),
                "two shapes share the directory {}",
                case.slug
            );
        }
    }

    /// The reference change has to land on the code that was seeded, or the
    /// suite is measuring nothing. Checked without compiling: text only.
    #[test]
    fn the_reference_change_applies_to_what_was_seeded() {
        for shape in BugShape::all() {
            let dir = scratch_dir("applies");
            let task = write_seed(&dir, *shape).expect("seed written");
            let touched = task.apply_reference_fix().expect("the fix applies once");
            assert_eq!(touched, task.case.expected_files, "{shape:?}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The trap this design cannot fully close is an agent reading the grader's
    /// expected values. The one it can close is the answer sitting in the tree,
    /// so the change as it is written is asserted absent from every file the
    /// agent is handed — including `task.md`.
    #[test]
    fn the_reference_change_is_nowhere_in_the_seeded_tree() {
        for shape in BugShape::all() {
            let case = seed_case(*shape);
            for edit in &case.edits {
                for file in &case.files {
                    assert!(
                        !file.body.contains(&edit.replace),
                        "{:?}: the fix leaked into {}",
                        shape,
                        file.path
                    );
                }
            }
        }
    }

    /// A case is its own repository: one commit, nothing pending, and a
    /// different history from the case beside it. This is the hard rule the
    /// item carries, because retrieval seeds part of its ranking from the files
    /// git reports as changed.
    #[test]
    fn each_case_starts_as_its_own_clean_repository() {
        let dir = scratch_dir("repo");
        let first = write_seed(&dir, BugShape::OffByOne).unwrap();
        let second = write_seed(&dir, BugShape::StaleCache).unwrap();
        assert_eq!(first.commit_count().unwrap(), 1);
        assert_eq!(second.commit_count().unwrap(), 1);
        assert!(
            first.changed_paths().is_empty(),
            "a seeded case must not start with work pending: {:?}",
            first.changed_paths()
        );
        let first_head = git_stdout(&first.path, &["rev-parse", "HEAD"]).unwrap();
        let second_head = git_stdout(&second.path, &["rev-parse", "HEAD"]).unwrap();
        assert_ne!(first_head, second_head, "two cases share a history");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A case is not written on top of anything: an old tree would carry its own
    /// changes into a measurement that thinks it is starting fresh.
    #[test]
    fn a_case_refuses_a_directory_that_is_already_occupied() {
        let dir = scratch_dir("occupied");
        std::fs::create_dir(dir.join("off-by-one")).unwrap();
        std::fs::write(dir.join("off-by-one").join("notes.txt"), "mine").unwrap();
        let err = write_seed(&dir, BugShape::OffByOne).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(
            std::fs::read_to_string(dir.join("off-by-one").join("notes.txt")).unwrap(),
            "mine",
            "the existing tree was left alone"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Generation is data, not a model's mood: the same shape twice is the same
    /// bytes, which is what lets a pass rate mean the same thing next month.
    #[test]
    fn one_shape_always_writes_the_same_files() {
        for shape in BugShape::all() {
            let a = seed_case(*shape);
            let b = seed_case(*shape);
            assert_eq!(a, b, "{shape:?} generated twice, differently");
        }
    }

    /// The measurement itself: the seeded case fails its grader, and the case
    /// with the reference change applied passes it. Runs the real toolchain, so
    /// it is the slowest thing here and covers two shapes rather than eight;
    /// `all_cases_pass_their_grader_once_fixed` walks the rest on demand.
    #[test]
    fn a_seeded_case_fails_and_the_same_case_with_the_change_applied_passes() {
        for shape in [BugShape::OffByOne, BugShape::LostUpdate] {
            let dir = scratch_dir("grade");
            let task = write_seed(&dir, shape).unwrap();
            let before = task.run_grader().expect("the grader ran");
            assert!(
                !before.passed,
                "{shape:?} passed its grader while still seeded:\n{}",
                before.tail
            );
            task.apply_reference_fix().unwrap();
            let after = task.run_grader().expect("the grader ran again");
            assert!(
                after.passed,
                "{shape:?} still fails after the reference change:\n{}",
                after.tail
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// All eight, graded both ways. Ignored because it compiles sixteen small
    /// crates; run it with
    /// `cargo test -p xencode-context-rs --lib seeds::tests::all_cases -- --ignored --nocapture`.
    #[test]
    #[ignore = "compiles all eight seeded cases twice; run it when a case definition changes"]
    fn all_cases_pass_their_grader_once_fixed() {
        let dir = scratch_dir("all");
        for shape in BugShape::all() {
            let task = write_seed(&dir, *shape).unwrap();
            let seeded = task.run_grader().expect("the grader ran");
            assert!(!seeded.passed, "{shape:?} was not broken");
            task.apply_reference_fix().unwrap();
            let fixed = task.run_grader().expect("the grader ran again");
            assert!(fixed.passed, "{shape:?}: {}", fixed.tail);
            println!("{shape:?}: {seeded:?} -> {fixed:?}");
            let _ = std::fs::remove_dir_all(&task.path);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
