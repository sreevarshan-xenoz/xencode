//! `OR-10` — what an approved team run took, so the next plan can quote a
//! measurement instead of a guess.
//!
//! [`crate::team`] says what a team *is*; `OR-2`'s [`crate::scheduler`] says what
//! it would take to run one. This module holds the third thing: the record a run
//! leaves behind when it actually ran, and the reading of those records that turns
//! one into an estimate. Nothing here launches anything.
//!
//! The estimate is measured or it is absent. A plan for a recipe that has never
//! run on this machine says so in as many words, rather than printing a number
//! with no run behind it — which is why [`fingerprint`] exists: it identifies the
//! recipe a record was made from, so a wall clock only ever gets quoted against
//! the same roles, commands, worker assignments and capacity that produced it.
//! Edit one command and the fingerprint moves, and the plan goes back to saying it
//! does not know.
//!
//! Records are stored per project under `.xencode/team-runs/`, one JSON file per
//! run. They are local machine state, not committed project data: the watt-hours
//! in a record are this laptop's, and pricing them at another machine's tariff
//! would be a different number that means the same thing badly.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::scheduler::ScheduleReport;
use crate::team::TeamRecipe;

/// The directory run records live in, under a project's `.xencode`. Kept out of
/// [`crate::team::RECIPES_DIR`] on purpose: that directory is committed project
/// data, and these are measurements of one machine.
pub const RUNS_DIR: &str = "team-runs";

/// A stable identity for the exact recipe a run was made from.
///
/// This is a fingerprint for cache identity, not a security digest: it exists so
/// an estimate cannot be carried over from a different team, and nothing here
/// needs to resist a hostile recipe author. Two recipes that differ in any field
/// a schedule would notice must differ in this string; a collision would only
/// quote the wrong seconds, not grant anything.
///
/// Fields are length-prefixed before being concatenated, so `ab` + `c` and `a` +
/// `bc` cannot produce the same bytes — a real risk when a role's command is free
/// text that a `needs` list or a gate name sits beside.
pub fn fingerprint(recipe: &TeamRecipe) -> String {
    let mut canonical = String::new();
    push(&mut canonical, &recipe.name);
    push(&mut canonical, &recipe.capacity.workers.to_string());
    push(
        &mut canonical,
        &recipe.capacity.verification_throughput.to_string(),
    );
    for role in &recipe.roles {
        push(&mut canonical, &role.name);
        push(&mut canonical, &role.worker);
        push(&mut canonical, &role.command);
        for gate in &role.gate {
            push(&mut canonical, gate);
        }
        // A list's own length goes in as a separator, so a role with one gate
        // cannot read like a role whose single gate contains a comma.
        canonical.push('|');
        for need in &role.needs {
            push(&mut canonical, need);
        }
        canonical.push('/');
    }
    fn push(buf: &mut String, field: &str) {
        buf.push_str(&field.len().to_string());
        buf.push('#');
        buf.push_str(field);
        buf.push(';');
    }
    // FNV-1a, 64-bit. Chosen because it is four lines, deterministic across
    // platforms and versions, and the only requirement is that different inputs
    // land differently — the same job a checksum of a config does.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// What one role of an approved run ended as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleRun {
    pub name: String,
    /// The task status label as the queue reported it: `exited(0)`, `killed`,
    /// `timed out`. Kept as the word a person reads, because a record is read by
    /// people before it is read by anything else.
    pub status: String,
    /// Milliseconds from the start of the run to this role being launched.
    pub started_ms: u64,
    /// Milliseconds from the start to this role finishing; `0` for a role that
    /// never finished, which its `status` says out loud.
    pub finished_ms: u64,
}

/// One run, as it was measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamRun {
    /// `<recipe slug>-<milliseconds>`: recognisable in a directory listing and
    /// ordered by when it ran.
    pub run_id: String,
    pub recipe: String,
    /// The recipe file this was run from, kept for the reader — a name that now
    /// points at two files is answerable.
    pub recipe_file: String,
    pub fingerprint: String,
    /// The human who approved it. Required, because a run record with no name on
    /// it is a run nobody agreed to.
    pub approved_by: String,
    pub started_at_unix_ms: u64,
    pub elapsed_ms: u64,
    pub capacity: usize,
    pub binding: String,
    pub peak_concurrency: usize,
    pub launch_order: Vec<String>,
    pub roles: Vec<RoleRun>,
    /// What the machine drew while the run lasted, or `None` when it reports no
    /// energy counter. Stored as watt-hours rather than as a price so the cost of
    /// a past run can be quoted at today's tariff instead of frozen at yesterday's.
    pub watt_hours: Option<f64>,
    /// The tariff set when the run happened. Kept so a record can be read back
    /// years later and still say what it was priced at then.
    pub cents_per_kwh: Option<f64>,
}

impl TeamRun {
    /// Build the record of a run from what the queue reported and what the
    /// machine saw around it.
    pub fn from_report(
        recipe: &TeamRecipe,
        report: &ScheduleReport,
        observed: &RunObservation<'_>,
    ) -> Self {
        let roles = report
            .launch_order
            .iter()
            .filter_map(|id| {
                let outcome = report.outcome(id)?;
                Some(RoleRun {
                    name: id.clone(),
                    status: outcome.status.label(),
                    started_ms: outcome.started_ms,
                    finished_ms: outcome.finished_ms,
                })
            })
            .collect();
        Self {
            run_id: format!("{}-{}", slug(&recipe.name), observed.started_at_unix_ms),
            recipe: recipe.name.clone(),
            recipe_file: observed.recipe_file.display().to_string(),
            fingerprint: observed.fingerprint.to_string(),
            approved_by: observed.approved_by.to_string(),
            started_at_unix_ms: observed.started_at_unix_ms,
            elapsed_ms: report.elapsed_ms,
            capacity: report.capacity,
            binding: report.binding.label().to_string(),
            peak_concurrency: report.peak_concurrency,
            launch_order: report.launch_order.clone(),
            roles,
            watt_hours: observed.watt_hours,
            cents_per_kwh: observed.cents_per_kwh,
        }
    }

    /// What this run's measured energy costs at `cents_per_kwh`.
    ///
    /// A watt-hour at `cents` cents is `cents * 10` micro-dollars — the same
    /// arithmetic `xencode-context-rs`'s `PowerUse::cost_micros` uses, re-derived
    /// here because a record holds the watt-hours and not the open window.
    /// `None` on either side means the number is unknown, never zero.
    pub fn cost_micros(&self, cents_per_kwh: Option<f64>) -> Option<u64> {
        let wh = self.watt_hours?;
        let cents = cents_per_kwh?;
        if !cents.is_finite() || cents < 0.0 {
            return None;
        }
        Some((wh * cents * 10.0).round() as u64)
    }

    /// Write the record into `dir`, one file per run, and say where it went.
    pub fn write(&self, dir: &Path) -> Result<PathBuf, String> {
        let path = dir.join(format!("{}.json", self.run_id));
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| format!("cannot write the run record as JSON: {e}"))?;
        crate::atomic::write_atomic(&path, format!("{text}\n").as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(path)
    }
}

/// What the machine saw about one approved run, beside what the queue reported.
#[derive(Debug, Clone)]
pub struct RunObservation<'a> {
    pub recipe_file: &'a Path,
    pub fingerprint: &'a str,
    pub approved_by: &'a str,
    pub started_at_unix_ms: u64,
    pub watt_hours: Option<f64>,
    pub cents_per_kwh: Option<f64>,
}

/// A name as it may appear in a file name. A recipe's `name` is free text written
/// by a person, and a name containing `/` would otherwise make the record path
/// reach into a subdirectory — or above the directory, with `..`. Dots go too, so
/// no component of a record's name can ever read as a traversal or as a hidden
/// file; the only dot in the path is the one on `.json`.
fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        return "recipe".to_string();
    }
    trimmed.chars().take(40).collect()
}

/// One file in the runs directory, with whatever it turned out to hold. A record
/// that cannot be read is carried beside the ones that can, so a truncated file
/// cannot silently remove a team's measured history.
#[derive(Debug, Clone)]
pub struct RunFile {
    pub path: PathBuf,
    pub run: Result<TeamRun, String>,
}

/// Read every recorded run. A missing directory is an empty answer, not an error:
/// a project that has never run a team has no history, and asking about it must
/// not create any.
pub fn load_runs(dir: &Path) -> Result<Vec<RunFile>, std::io::Error> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let run = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))
            .and_then(|text| {
                serde_json::from_str::<TeamRun>(&text)
                    .map_err(|e| format!("{}: {e}", path.display()))
            });
        files.push(RunFile { path, run });
    }
    Ok(files)
}

/// The wall clock and energy a previous run of this exact recipe measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    pub run_id: String,
    pub approved_by: String,
    pub started_at_unix_ms: u64,
    pub wall_clock_ms: u64,
    pub peak_concurrency: usize,
    pub watt_hours: Option<f64>,
    pub roles: usize,
}

impl Estimate {
    /// What the measured energy would cost at the tariff set now.
    pub fn cost_micros(&self, cents_per_kwh: Option<f64>) -> Option<u64> {
        let cents = cents_per_kwh?;
        if !cents.is_finite() || cents < 0.0 {
            return None;
        }
        self.watt_hours.map(|wh| (wh * cents * 10.0).round() as u64)
    }
}

/// The estimate for a recipe: the most recent recorded run that carries the same
/// fingerprint. `None` means no run of this recipe has been measured here, which
/// the caller has to say out loud rather than fill in with a typical number.
pub fn estimate(runs: &[RunFile], fingerprint: &str) -> Option<Estimate> {
    let newest = runs
        .iter()
        .filter_map(|file| file.run.as_ref().ok())
        .filter(|run| run.fingerprint == fingerprint)
        .max_by_key(|run| (run.started_at_unix_ms, run.run_id.as_str()))?;
    Some(Estimate {
        run_id: newest.run_id.clone(),
        approved_by: newest.approved_by.clone(),
        started_at_unix_ms: newest.started_at_unix_ms,
        wall_clock_ms: newest.elapsed_ms,
        peak_concurrency: newest.peak_concurrency,
        watt_hours: newest.watt_hours,
        roles: newest.roles.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::TaskManager;
    use crate::team::{Capacity, RoleSpec};

    const RECIPE: &str = r#"
name = "rust-fix"

[[roles]]
name = "survey"
worker = "opencode"
gate = ["test"]
command = "true"

[[roles]]
name = "harden"
worker = "codex"
gate = ["lint", "test"]
command = "true"
needs = ["survey"]

[capacity]
workers = 2
verification_throughput = 2
"#;

    fn recipe(text: &str) -> TeamRecipe {
        TeamRecipe::parse(Path::new("rust-fix.toml"), text).expect("the test recipe parses")
    }

    /// A record made the way the command makes one: from a report the scheduler
    /// produced over real children.
    async fn recorded(dir: &Path) -> TeamRun {
        let r = recipe(RECIPE);
        let graph = r.to_task_graph().unwrap();
        let mut manager = TaskManager::new();
        let report = r
            .scheduler()
            .run(&graph, &mut manager)
            .await
            .expect("the schedule completes");
        TeamRun::from_report(
            &r,
            &report,
            &RunObservation {
                recipe_file: dir,
                fingerprint: &fingerprint(&r),
                approved_by: "Sree",
                started_at_unix_ms: 1_700_000_000_000,
                watt_hours: Some(0.5),
                cents_per_kwh: Some(12.0),
            },
        )
    }

    #[test]
    fn the_same_recipe_read_twice_fingerprints_the_same() {
        assert_eq!(fingerprint(&recipe(RECIPE)), fingerprint(&recipe(RECIPE)));
    }

    #[test]
    fn editing_a_role_command_moves_the_fingerprint_because_the_run_it_estimated_is_gone() {
        let edited = RECIPE.replace("command = \"true\"", "command = \"cargo build\"");
        assert_ne!(fingerprint(&recipe(RECIPE)), fingerprint(&recipe(&edited)));
    }

    #[test]
    fn the_fingerprint_notices_who_plays_a_role_and_how_wide_the_team_runs() {
        // A different worker is a different wall clock, and so is a team allowed
        // to run twice as wide; an estimate must not survive either edit.
        for change in [
            (r#"worker = "codex""#, r#"worker = "claude""#),
            ("workers = 2", "workers = 4"),
            ("verification_throughput = 2", "verification_throughput = 3"),
            ("needs = [\"survey\"]", "needs = []"),
            ("gate = [\"lint\", \"test\"]", "gate = [\"lint\"]"),
            ("name = \"rust-fix\"", "name = \"rust-fix-v2\""),
        ] {
            let text = RECIPE.replace(change.0, change.1);
            assert_ne!(
                fingerprint(&recipe(RECIPE)),
                fingerprint(&recipe(&text)),
                "a recipe changed at {change:?} read as the same team"
            );
        }
    }

    #[test]
    fn moving_a_field_boundary_does_not_read_as_the_same_recipe() {
        // Without the length prefixes these two would concatenate identically.
        let mut a = TeamRecipe {
            name: "x".to_string(),
            roles: vec![RoleSpec {
                name: "ab".to_string(),
                worker: "w".to_string(),
                gate: vec![],
                command: "c".to_string(),
                needs: vec![],
            }],
            capacity: Capacity {
                workers: 1,
                verification_throughput: 1,
            },
        };
        let first = fingerprint(&a);
        a.roles[0].name = "a".to_string();
        a.roles[0].command = "bc".to_string();
        assert_ne!(first, fingerprint(&a), "`ab`+`c` and `a`+`bc` collided");
    }

    #[tokio::test]
    async fn a_run_written_is_a_run_read_back_with_the_same_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let run = recorded(dir.path()).await;
        let path = run.write(dir.path()).unwrap();
        assert!(path.exists(), "the record went somewhere else: {path:?}");
        let files = load_runs(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        let read = files[0].run.as_ref().expect("the record reads back");
        assert_eq!(read, &run);
        assert_eq!(read.approved_by, "Sree");
        assert_eq!(read.roles.len(), 2, "one entry per role that ran");
        assert_eq!(
            read.roles
                .iter()
                .map(|r| r.status.as_str())
                .collect::<Vec<_>>(),
            vec!["exited(0)", "exited(0)"],
            "these were real children and they really exited 0"
        );
    }

    #[tokio::test]
    async fn a_record_built_from_a_real_schedule_carries_the_times_that_run_reported() {
        // The wall clock in a record is the one the queue measured, not a number
        // this test could write in by hand.
        let dir = tempfile::tempdir().unwrap();
        let run = recorded(dir.path()).await;
        assert_eq!(run.launch_order, vec!["survey", "harden"]);
        assert_eq!(run.capacity, 2);
        assert_eq!(run.binding, "workers and verification (equal)");
        assert!(
            run.elapsed_ms > 0,
            "a run over real children took no measurable time: {}",
            run.elapsed_ms
        );
        let harden = run.roles.iter().find(|r| r.name == "harden").unwrap();
        let survey = run.roles.iter().find(|r| r.name == "survey").unwrap();
        assert!(
            harden.started_ms >= survey.finished_ms,
            "harden started at {} before survey finished at {}",
            harden.started_ms,
            survey.finished_ms
        );
    }

    #[test]
    fn a_missing_runs_directory_is_an_empty_answer_and_stays_missing() {
        let dir = tempfile::tempdir().unwrap();
        let runs = dir.path().join("team-runs");
        let files = load_runs(&runs).unwrap();
        assert!(files.is_empty());
        assert!(
            !runs.exists(),
            "reading the history created a history directory"
        );
    }

    #[tokio::test]
    async fn an_unreadable_record_is_reported_beside_the_good_one_instead_of_replacing_it() {
        let dir = tempfile::tempdir().unwrap();
        let run = recorded(dir.path()).await;
        run.write(dir.path()).unwrap();
        std::fs::write(dir.path().join("broken.json"), "{ not json").unwrap();
        let files = load_runs(dir.path()).unwrap();
        assert_eq!(files.len(), 2, "both files were seen");
        assert!(
            files.iter().any(|f| f.run.is_err()),
            "the broken record was not reported"
        );
        assert!(files.iter().any(|f| f.run.as_ref().ok() == Some(&run)));
    }

    #[tokio::test]
    async fn the_estimate_is_the_newest_run_of_the_same_fingerprint_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let mine = recorded(dir.path()).await;
        let older = TeamRun {
            run_id: "rust-fix-1".to_string(),
            started_at_unix_ms: 1,
            elapsed_ms: 111,
            ..mine.clone()
        };
        let newer = TeamRun {
            run_id: "rust-fix-2".to_string(),
            started_at_unix_ms: mine.started_at_unix_ms + 5_000,
            elapsed_ms: 999,
            ..mine.clone()
        };
        older.write(dir.path()).unwrap();
        newer.write(dir.path()).unwrap();
        let files = load_runs(dir.path()).unwrap();
        let est = estimate(&files, &mine.fingerprint).expect("two runs of it are recorded");
        assert_eq!(est.run_id, newer.run_id, "the newest run is the estimate");
        assert_eq!(est.wall_clock_ms, 999);
        assert_eq!(est.approved_by, "Sree");

        // A recipe nobody here has run gets no number, rather than a nearby one.
        assert!(estimate(&files, "0000000000000000").is_none());
        let other = RECIPE.replace("command = \"true\"", "command = \"sleep 1\"");
        assert!(estimate(&files, &fingerprint(&recipe(&other))).is_none());
    }

    #[tokio::test]
    async fn a_recipe_name_that_looks_like_a_path_cannot_put_a_record_outside_the_runs_directory() {
        let dir = tempfile::tempdir().unwrap();
        let escapes = recipe(&RECIPE.replace("name = \"rust-fix\"", "name = \"../../escape\""));
        let run = TeamRun {
            run_id: format!("{}-123", slug(&escapes.name)),
            recipe: escapes.name.clone(),
            ..recorded(dir.path()).await
        };
        let path = run.write(dir.path()).unwrap();
        assert_eq!(
            path.parent().unwrap().canonicalize().unwrap(),
            dir.path().canonicalize().unwrap(),
            "the record was written outside the runs directory"
        );
        assert!(!path.display().to_string().contains(".."), "{path:?}");
        assert!(!run.run_id.contains('/') && !run.run_id.contains(".."));
        // A name with nothing writable in it still gets a file, not a panic, and
        // a dot in a recipe name is not allowed to become a hidden file.
        assert_eq!(slug("///"), "recipe");
        assert_eq!(slug("rust fix"), "rust-fix");
        assert_eq!(slug("v1.2"), "v1-2");
    }

    #[tokio::test]
    async fn an_unknown_energy_or_an_unset_tariff_prices_nothing_and_never_reads_as_free() {
        let dir = tempfile::tempdir().unwrap();
        let mut run = recorded(dir.path()).await;
        assert!(run.cost_micros(Some(12.0)).unwrap() > 0);
        assert_eq!(
            run.cost_micros(None),
            None,
            "no tariff set is not a free run"
        );
        run.watt_hours = None;
        assert_eq!(
            run.cost_micros(Some(12.0)),
            None,
            "nothing measured is not zero"
        );
        // A machine whose counter rolled to a stop reads as no cost, not as a
        // run that drew nothing.
        run.watt_hours = Some(0.0);
        assert_eq!(run.cost_micros(Some(12.0)), Some(0));
    }

    #[tokio::test]
    async fn a_role_that_failed_is_recorded_with_the_exit_code_the_os_reported() {
        // A record that only ever says `exited(0)` would be a report of what the
        // queue hopes happened. This is the status the child actually returned.
        let dir = tempfile::tempdir().unwrap();
        let failing = RECIPE.replace("command = \"true\"\nneeds", "command = \"exit 3\"\nneeds");
        let r = recipe(&failing);
        let graph = r.to_task_graph().unwrap();
        let mut manager = TaskManager::new();
        let report = r
            .scheduler()
            .run(&graph, &mut manager)
            .await
            .expect("a failing role still completes the schedule");
        let run = TeamRun::from_report(
            &r,
            &report,
            &RunObservation {
                recipe_file: dir.path(),
                fingerprint: &fingerprint(&r),
                approved_by: "Sree",
                started_at_unix_ms: 2_000,
                watt_hours: None,
                cents_per_kwh: None,
            },
        );
        let harden = run.roles.iter().find(|role| role.name == "harden").unwrap();
        assert_eq!(harden.status, "exited(3)", "{harden:?}");
        assert_eq!(run.cost_micros(Some(12.0)), None, "nothing measured");
    }
}
