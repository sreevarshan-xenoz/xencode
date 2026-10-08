//! `OR-14` — the orchestrator's own command surface. Nine of its eleven verbs are
//! readings of state that already exists, and the item's done-when is that a reading
//! leaves that state exactly as it was found; the other two start or stop one real
//! process on this machine. So every case here runs the real `xencode` binary in a
//! real directory against real recorded data — a team run that really launched its
//! roles, a background task that really has a pid — and asserts the words it
//! printed. Nothing is mocked, and no vendor's program is started by a test: the
//! handover cases are all refusals, which is what `attach` is allowed to do unasked.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

/// Three roles, two of which can run at once. The worker is `xencode-loop`, which
/// is not a name on the agent roster, so these runs are allowed by the shipped
/// posture and nothing here depends on a vendor's program being installed.
fn recipe(name: &str, command: &str) -> String {
    format!(
        r#"name = "{name}"

[[roles]]
name = "survey"
worker = "xencode-loop"
gate = []
command = "echo surveyed"

[[roles]]
name = "harden"
worker = "xencode-loop"
gate = ["test"]
command = "{command}"

[[roles]]
name = "integrate"
worker = "xencode-loop"
gate = ["lint", "test"]
command = "echo integrated"
needs = ["survey", "harden"]

[capacity]
workers = 2
verification_throughput = 2
"#
    )
}

/// A recipe naming a roster agent as its worker, which is the one shape the
/// shipped posture is built to refuse.
fn external_recipe() -> String {
    r#"name = "external"

[[roles]]
name = "ask-claude"
worker = "claude"
gate = []
command = "touch SHOULD-NOT-EXIST"

[capacity]
workers = 1
verification_throughput = 1
"#
    .to_string()
}

/// A project whose `.xencode/teams` holds exactly these files. The caller keeps
/// the `TempDir` alive, so nothing here is cleaned up early or leaks.
fn project(files: &[(&str, String)]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let teams = dir.path().join(".xencode").join("teams");
    fs::create_dir_all(&teams).unwrap();
    for (name, text) in files {
        fs::write(teams.join(name), text).unwrap();
    }
    dir
}

/// A home directory of its own, so the approval mode and the posture these
/// readings quote are whatever this test set rather than whatever the person
/// running it has in their settings. Set with `xencode config set`, the way a
/// person sets it.
fn sandbox_home(settings: &[(&str, &str)]) -> TempDir {
    let home = TempDir::new().unwrap();
    for (key, value) in settings {
        let out = Command::new(xencode_bin())
            .args(["config", "set", key, value])
            .current_dir(home.path())
            .env("HOME", home.path())
            .env("XDG_CONFIG_HOME", home.path().join(".config"))
            .output()
            .expect("xencode ran");
        assert!(
            out.status.success(),
            "setting {key} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    home
}

fn run_in(dir: &Path, home: &Path, args: &[&str]) -> Output {
    Command::new(xencode_bin())
        .args(args)
        .current_dir(dir)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .output()
        .expect("xencode ran")
}

/// A command that has to succeed; its stdout, so a failing case prints what the
/// binary said instead of only its exit code.
fn succeeds(dir: &Path, home: &Path, args: &[&str]) -> String {
    let out = run_in(dir, home, args);
    assert!(
        out.status.success(),
        "`xencode {}` failed: {}{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// A command that has to fail, because these refusals are the product. Its whole
/// output: the error goes to stderr and the person reads both.
fn refuses(dir: &Path, home: &Path, args: &[&str]) -> String {
    let out = run_in(dir, home, args);
    assert!(
        !out.status.success(),
        "`xencode {}` was expected to refuse and exited {}: {}{}",
        args.join(" "),
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Every path under `root`, with the bytes of each file. A reading that touches
/// one byte anywhere in the project or the config is caught by comparing this.
fn fingerprint(root: &Path) -> Vec<(std::path::PathBuf, Option<Vec<u8>>)> {
    fn walk(dir: &Path, out: &mut Vec<(std::path::PathBuf, Option<Vec<u8>>)>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.push((path.clone(), None));
                walk(&path, out);
            } else {
                out.push((path.clone(), fs::read(&path).ok()));
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    found.sort();
    found
}

/// What moved between two readings of the same tree: which paths were created,
/// removed, or hold different bytes. Names only — a failure has to say which
/// file a reading touched, not print every file in the project.
fn differences(
    label: &str,
    before: &[(std::path::PathBuf, Option<Vec<u8>>)],
    after: &[(std::path::PathBuf, Option<Vec<u8>>)],
) -> Vec<String> {
    let mut moved = Vec::new();
    for (path, bytes) in after {
        match before.iter().find(|(found, _)| found == path) {
            None => moved.push(format!("{label}: created {}", path.display())),
            Some((_, found)) if found != bytes => {
                moved.push(format!("{label}: changed {}", path.display()))
            }
            Some(_) => {}
        }
    }
    for (path, _) in before {
        if !after.iter().any(|(found, _)| found == path) {
            moved.push(format!("{label}: removed {}", path.display()));
        }
    }
    moved
}

/// The run records this project holds, by id.
fn record_ids(dir: &Path) -> Vec<String> {
    let runs = dir.join(".xencode").join("team-runs");
    let mut ids: Vec<String> = match fs::read_dir(&runs) {
        Ok(entries) => entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .map(|n| n.trim_end_matches(".json").to_string())
            .collect(),
        Err(_) => return Vec::new(),
    };
    ids.sort();
    ids
}

/// A real approved run of the demo recipe, so the readings below have recorded
/// data to quote rather than a file this test wrote by hand.
fn record_a_run(dir: &Path, home: &Path) -> String {
    let out = succeeds(
        dir,
        home,
        &["team", "run", "demo", "--approved-by", "Tester"],
    );
    assert!(
        out.contains("Every role exited 0"),
        "the run this test depends on did not succeed:\n{out}"
    );
    let ids = record_ids(dir);
    assert_eq!(ids.len(), 1, "the approved run recorded exactly once");
    ids[0].clone()
}

/// The pids whose command line is exactly these arguments, read off `/proc` the
/// way the task registry reads a pid's state.
fn pids_running(argv: &[&str]) -> Vec<u32> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return found;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(raw) = fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        let args: Vec<String> = raw
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).to_string())
            .collect();
        if args.iter().map(String::as_str).eq(argv.iter().copied()) {
            found.push(pid);
        }
    }
    found
}

/// Whether a pid is still a running process. Existence of `/proc/<pid>` is not
/// the same question: a signalled process stays visible as a zombie until its
/// parent reaps it, and a machine loaded by the rest of the suite widens that
/// window enough to be mistaken for a process that ignored the signal. This
/// reads the task state the kernel reports, so `Z` counts as dead.
fn alive(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => match stat.rfind(')') {
            Some(close) => stat.as_bytes().get(close + 2) != Some(&(b'Z')),
            None => true,
        },
        Err(_) => false,
    }
}

#[test]
fn a_reading_leaves_the_project_and_the_config_exactly_as_it_was_found() {
    // The done-when, stated as a check over bytes: the nine reading verbs change
    // nothing — not in the project, not in the config they only report on, not by
    // creating a directory that was not there.
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);
    let run_id = record_a_run(dir.path(), home.path());
    succeeds(
        dir.path(),
        home.path(),
        &["tasks", "start", "echo task-output", "--name", "listed"],
    );
    // Wait for that child to be reaped: its output file and exit file are written
    // by the child itself, and a snapshot taken mid-write would blame the reading.
    for _ in 0..50 {
        let json: serde_json::Value = serde_json::from_str(&succeeds(
            dir.path(),
            home.path(),
            &["orchestrator", "tasks", "--format", "json"],
        ))
        .unwrap();
        if json["tasks"][0]["status"] != "running" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let before = fingerprint(dir.path());
    let home_before = fingerprint(home.path());

    for verb in [
        vec!["status"],
        vec!["agents"],
        vec!["tasks"],
        vec!["graph"],
        vec!["graph", "demo"],
        vec!["logs"],
        vec!["permissions"],
        vec!["costs"],
        vec!["inspect", "demo"],
        vec!["inspect", &run_id],
    ] {
        let mut args: Vec<&str> = vec!["orchestrator"];
        args.extend(verb.iter().copied());
        let text = succeeds(dir.path(), home.path(), &args);
        args.push("--format");
        args.push("json");
        let out = succeeds(dir.path(), home.path(), &args);
        assert!(
            serde_json::from_str::<serde_json::Value>(&out).is_ok(),
            "`xencode orchestrator {} --format json` is not JSON",
            verb.join(" ")
        );
        assert!(
            !text.contains("error:"),
            "`xencode orchestrator {}` failed in its text form",
            verb.join(" ")
        );
    }
    // The readings that refuse: a miss must not leave anything behind either.
    for args in [
        vec!["orchestrator", "logs", "no-such-run"],
        vec!["orchestrator", "inspect", "no-such-thing"],
        vec!["orchestrator", "attach", "cline"],
        vec!["orchestrator", "retry", "demo", "no-such-role"],
        vec!["orchestrator", "stop", "no-such-run"],
    ] {
        refuses(dir.path(), home.path(), &args);
    }

    let mut moved = differences("project", &before, &fingerprint(dir.path()));
    moved.extend(differences(
        "config",
        &home_before,
        &fingerprint(home.path()),
    ));
    assert!(
        moved.is_empty(),
        "a reading changed something, when every one of them is meant to leave it as found:\n{}",
        moved.join("\n")
    );
    assert_eq!(
        record_ids(dir.path()).len(),
        1,
        "a reading wrote a run record"
    );
}

#[test]
fn status_quotes_the_posture_and_counts_only_what_it_could_read() {
    let dir = project(&[
        ("demo.toml", recipe("demo", "echo hardened")),
        ("external.toml", external_recipe()),
    ]);
    let home = sandbox_home(&[]);

    // Nothing recorded yet: the empty case names the directory it looked in, and
    // does not read as "checked and clear".
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "status"]);
    assert!(
        out.contains(&format!(
            "none in {} — every figure below is therefore unmeasured",
            dir.path().join(".xencode").join("team-runs").display()
        )),
        "says where it looked: {out}"
    );
    assert!(
        out.contains("0 — none in `cache/detached`"),
        "and where it looked for detached runs: {out}"
    );
    assert!(
        out.contains("work may go to:      Local Only"),
        "one aligned label column, with the shipped posture at the top: {out}"
    );
    assert!(
        out.contains("roster refused:      10 of the 10 agents on the roster"),
        "the posture's answer covers the whole roster: {out}"
    );

    let run_id = record_a_run(dir.path(), home.path());
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "status"]);
    assert!(
        out.contains(&format!(
            "recorded runs:       1 · newest {run_id} (recipe demo, 3 roles, approved by Tester,"
        )),
        "the newest record is quoted by its own id: {out}"
    );
    assert!(
        out.contains("recipes:             2 in "),
        "both recipe files are counted: {out}"
    );

    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "status", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["posture"], "Local Only");
    assert_eq!(json["recipes"], 2);
    assert_eq!(json["recipes_unreadable"], 0);
    assert_eq!(json["recorded_runs"], 1);
    assert_eq!(json["newest_run"]["approved_by"], "Tester");
    assert_eq!(json["newest_run"]["roles"], 3);
    assert_eq!(json["refused_agents"].as_array().unwrap().len(), 10);
    assert_eq!(json["detached_runs"], 0);

    // Opening the roster to external workers removes the refusal, and the screen
    // stops claiming a refusal that no longer happens.
    succeeds(
        dir.path(),
        home.path(),
        &["config", "set", "allow_external_workers", "true"],
    );
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "status"]);
    assert!(
        !out.contains("roster refused"),
        "nothing is refused, so the row is gone rather than reading 0: {out}"
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "status", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["refused_agents"].as_array().unwrap().len(), 0);
}

#[test]
fn agents_shows_a_handover_verb_only_where_the_vendor_documents_one() {
    let dir = project(&[]);
    let home = sandbox_home(&[]);
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "agents"]);
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "agents", "--format", "json"],
    ))
    .unwrap();

    let rows = json["agents"].as_array().unwrap();
    let with_verb: Vec<&str> = rows
        .iter()
        .filter(|row| row["handover"].is_string())
        .map(|row| row["agent"].as_str().unwrap())
        .collect();
    let without = rows.len() - with_verb.len();
    assert!(
        !with_verb.is_empty() && without > 0,
        "the roster has to hold both kinds for this screen to mean anything: {out}"
    );
    // The table and the JSON agree row by row: a row with no verb says so in words
    // instead of leaving the cell blank, which would read as "not checked".
    for row in rows {
        let name = row["agent"].as_str().unwrap();
        let line = out
            .lines()
            .find(|l| l.starts_with(name))
            .unwrap_or_else(|| panic!("no table row for {name}: {out}"));
        match row["handover"].as_str() {
            Some(template) => assert!(
                line.contains(template),
                "{name} carries `{template}` in JSON and not in the table: {line}"
            ),
            None => assert!(
                line.contains("no attach verb read from its help"),
                "{name} has no handover verb and the row says so: {line}"
            ),
        }
    }
    assert!(
        out.contains(&format!(
            "{without} of the {} rows here have none",
            rows.len()
        )),
        "the count in the prose is the count of the rows: {out}"
    );
    assert_eq!(json["posture"], "Local Only");
    assert!(
        rows.iter()
            .all(|row| row["refused_by_posture"] == serde_json::json!(true)),
        "the shipped posture refuses every name on the roster"
    );
    assert!(
        rows.iter().all(|row| row["cells_read_on"].is_string()),
        "every cell names the date it was read from: {json}"
    );
    assert!(
        out.contains("nothing here is a claim that a probe ran the agent"),
        "and the screen says what it is not: {out}"
    );
}

#[test]
fn graph_names_the_waves_and_puts_the_recorded_timings_beside_them() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);

    // Before any run, the shape is shown and the numbers are declared missing.
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "graph", "demo"]);
    assert!(
        out.contains("capacity: 2 at once, limited by workers and verification (equal)"),
        "{out}"
    );
    assert!(out.contains("wave 1:  survey  +  harden"), "{out}");
    assert!(
        out.contains("integrate ← waits on survey, harden"),
        "what each role waits on: {out}"
    );
    assert!(out.contains("critical path: survey → integrate"), "{out}");
    assert!(out.contains("bottleneck: integrate"), "{out}");
    assert!(
        out.contains("none, so nothing above has been measured on this machine"),
        "no run yet, and it says that rather than estimating: {out}"
    );

    let run_id = record_a_run(dir.path(), home.path());
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "graph", "demo"]);
    assert!(
        out.contains("Recorded runs of this exact recipe: 1"),
        "{out}"
    );
    assert!(
        out.contains(&format!("from run {run_id} — peak 2,")),
        "the timings below come from a run that happened: {out}"
    );
    for role in ["survey", "harden", "integrate"] {
        assert!(
            out.contains(&format!("{role:<14} exited(0)")),
            "{role}'s recorded end is not in the graph: {out}"
        );
    }
    assert!(
        !out.contains("did not exit 0"),
        "every role exited 0, so no failure list: {out}"
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "graph", "demo", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["recorded_runs"], 1);
    assert_eq!(json["last_run"]["run_id"], run_id);
    assert_eq!(json["failed_roles_on_record"].as_array().unwrap().len(), 0);
    assert_eq!(json["capacity"], 2);

    // The failure case, read off a second recipe's own record.
    let dir2 = project(&[("bad.toml", recipe("bad", "exit 3"))]);
    let home2 = sandbox_home(&[]);
    let out = run_in(
        dir2.path(),
        home2.path(),
        &["team", "run", "bad", "--approved-by", "Tester"],
    );
    assert!(
        !out.status.success(),
        "a role exiting 3 has to fail the run: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let graph = succeeds(dir2.path(), home2.path(), &["orchestrator", "graph", "bad"]);
    assert!(
        graph.contains("Roles that did not exit 0 on record:"),
        "and the graph names it: {graph}"
    );
    assert!(graph.contains("exited(3)"), "{graph}");

    // With no recipe named it is every run this project recorded, in the words the
    // worker panel uses, because both read the same files.
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "graph"]);
    assert!(
        out.contains(&format!(
            "Recorded graphs · {}",
            dir.path().join(".xencode").join("team-runs").display()
        )),
        "{out}"
    );
    assert!(out.contains(&run_id), "the recorded run is a row: {out}");

    let missing = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "graph", "nosuchrecipe"],
    );
    assert!(missing.contains("nosuchrecipe"), "{missing}");
}

#[test]
fn inspect_opens_the_thing_that_was_named_not_a_run_that_shares_its_prefix() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);
    let run_id = record_a_run(dir.path(), home.path());
    assert!(
        run_id.starts_with("demo-"),
        "a run id begins with its recipe's name, which is what this check is about: {run_id}"
    );

    // `demo` is both the recipe's name and the prefix of its run ids. The recipe
    // is what was asked for, and opening a run instead would show timings for a
    // graph nobody named.
    let out = succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", "demo"],
    );
    assert!(
        out.starts_with("Recipe demo would run 3 roles"),
        "the plan of the recipe, not one of its runs: {out}"
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", "demo", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["recipe"], "demo");

    // The full id opens the record, and names the file it came from.
    let out = succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", &run_id],
    );
    assert!(out.contains("Recorded team run ·"), "{out}");
    assert!(
        out.contains(&format!(
            "{}",
            dir.path().join(".xencode").join("team-runs").display()
        )),
        "the record's own path is quoted: {out}"
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", &run_id, "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["run_id"], run_id);
    assert_eq!(json["approved_by"], "Tester");
    assert_eq!(json["roles"].as_array().unwrap().len(), 3);

    // A prefix of a run id still reaches the same record.
    let prefix: String = run_id.chars().take(6).collect();
    let out = succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", &prefix],
    );
    assert!(out.contains("Recorded team run ·"), "{out}");

    // And a name that belongs to nothing says every place it checked.
    let missing = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "inspect", "zz-nothing"],
    );
    for place in [
        ".xencode/tasks",
        ".xencode/teams",
        ".xencode/cache/detached",
        ".xencode/team-runs",
    ] {
        assert!(
            missing.contains(place),
            "the refusal names {place}: {missing}"
        );
    }
    assert!(
        missing.contains("nothing here is named zz-nothing"),
        "{missing}"
    );
}

#[test]
fn logs_tells_a_run_that_kept_a_log_from_one_that_never_captured_anything() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "logs"]);
    assert!(
        out.contains("Nothing was named, so here is what has a log to read:"),
        "it lists instead of choosing one for you: {out}"
    );
    assert!(out.contains("detached runs: none in"), "{out}");
    assert!(out.contains("team runs:     none in"), "{out}");

    let run_id = record_a_run(dir.path(), home.path());
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "logs"]);
    assert!(
        out.contains(&format!("  {run_id}  demo")),
        "the recorded run is listed by id and recipe: {out}"
    );
    assert!(
        out.contains("team runs (timings and exit status, no captured output):"),
        "{out}"
    );

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "logs", &run_id]);
    assert!(out.contains(&format!("Team run {run_id}")), "{out}");
    assert!(
        out.contains("There is no output above because a team run never captured it"),
        "an empty block would read as a quiet child: {out}"
    );
    assert!(out.contains("recipe demo — approved by Tester"), "{out}");
    assert!(
        out.contains("survey         exited(0)"),
        "what the record does keep: {out}"
    );

    // A prefix works, because the ids are long and machine-made.
    let prefix: String = run_id.chars().take(9).collect();
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "logs", &prefix]);
    assert!(out.contains(&format!("Team run {run_id}")), "{out}");

    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "logs", &run_id, "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["run_id"], run_id);
    assert_eq!(json["roles"].as_array().unwrap().len(), 3);

    let missing = refuses(dir.path(), home.path(), &["orchestrator", "logs", "zz-run"]);
    assert!(
        missing.contains("no run zz-run — not a detached run in")
            && missing.contains("and not a recorded team run in"),
        "{missing}"
    );
    assert!(
        missing.contains("`xencode orchestrator logs` with no name lists both"),
        "{missing}"
    );
}

#[test]
fn permissions_builds_each_launch_with_the_same_function_a_launch_uses() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);

    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "permissions", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["config_key"], "agent_approval");
    assert_eq!(json["config_value"], "ask");
    assert_eq!(json["mode"], "Ask");
    let launches = json["launches"].as_array().unwrap();
    assert!(!launches.is_empty());
    // Under `ask`, no launch carries an autonomy flag of any kind, and a worker
    // asking for its own changes nothing about the line.
    for row in launches {
        assert_eq!(row["grant"], "Nothing", "{}", row["agent"]);
        assert_eq!(row["grant_words"], "no autonomy flag", "{}", row["agent"]);
        assert_eq!(
            row["argv"], row["argv_if_worker_asked_for_its_own"],
            "{}: a worker changed its own launch line",
            row["agent"]
        );
        assert!(
            !row["argv"].as_array().unwrap().iter().any(|t| t
                .as_str()
                .is_some_and(|s| s.contains("yolo") || s.contains("permission-mode"))),
            "{}: an autonomy flag reached a launch under `ask`: {:?}",
            row["agent"],
            row["argv"]
        );
    }
    assert_eq!(json["approval_history"]["runs_on_record"], 0);
    assert_eq!(json["approval_history"]["calls_asked"], 0);

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "permissions"]);
    assert!(
        out.contains("Permissions · agent_approval = \"ask\" → Ask"),
        "{out}"
    );
    assert!(
        out.contains(&format!(
            "{n} of {n} came back the line above, unchanged",
            n = launches.len()
        )),
        "the count it prints is the count of the rows it drew: {out}"
    );
    assert!(out.contains("Answered here, on record: 0 run(s)"), "{out}");
    assert!(
        out.contains("A `no autonomy flag` row is the strictest of the three grants"),
        "{out}"
    );

    // One agent asked for, and the same answer narrowed to it.
    let out = succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "permissions", "claude"],
    );
    assert_eq!(
        out.lines()
            .filter(|l| l.starts_with("claude") || l.starts_with("opencode"))
            .count(),
        1,
        "asking for one agent shows one row: {out}"
    );

    // The same screen with the operator's own permission widened: the flag now
    // appears, and it appears because the mode said so.
    succeeds(
        dir.path(),
        home.path(),
        &["config", "set", "agent_approval", "all-allow"],
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "permissions", "claude", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["config_value"], "all-allow");
    assert_eq!(json["mode"], "AllAllow");
    assert_eq!(json["launches"][0]["grant"], "Bypass");
    assert_eq!(json["launches"][0]["grant_words"], "full autonomy");
    assert!(
        json["launches"][0]["argv"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "--permission-mode"),
        "{json}"
    );
    assert_eq!(
        json["launches"][0]["argv"], json["launches"][0]["argv_if_worker_asked_for_its_own"],
        "even the widened line is the mode's choice, not the worker's"
    );
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "permissions"]);
    assert!(
        !out.contains("A `no autonomy flag` row is the strictest"),
        "no row is that grant now, so nothing says it: {out}"
    );

    let missing = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "permissions", "nosuchagent"],
    );
    assert!(
        missing.contains("no agent named nosuchagent on the roster"),
        "{missing}"
    );
}

#[test]
fn costs_prices_only_what_has_a_price_and_counts_only_what_the_counter_saw() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened"))]);
    let home = sandbox_home(&[]);

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "costs"]);
    assert!(
        out.contains("nothing recorded, so no token count exists to price"),
        "{out}"
    );
    assert!(
        out.contains(&format!(
            "pricing from {}, which is not there",
            dir.path().join(".xencode").join("pricing.json").display()
        )),
        "the pricing file it wanted, named: {out}"
    );
    assert!(
        out.contains("team energy:     nothing measured — no run is recorded in"),
        "{out}"
    );

    record_a_run(dir.path(), home.path());
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "costs", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["metric_records"], 0);
    assert_eq!(json["known_micros"], 0);
    assert_eq!(json["models"].as_array().unwrap().len(), 0);
    assert_eq!(json["pricing_file"]["present"], false);
    assert_eq!(json["pricing_file"]["lookup_enabled"], false);
    assert_eq!(json["team_runs"]["recorded"], 1);
    assert_eq!(json["team_runs"]["with_power_counter"], 1);
    assert!(
        json["team_runs"]["watt_hours"]
            .as_f64()
            .is_some_and(|wh| wh > 0.0),
        "a real run drew real power and the counter read it: {json}"
    );
    assert_eq!(json["team_runs"]["cents_per_kwh"], serde_json::Value::Null);

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "costs"]);
    assert!(
        out.contains("from 1 of 1 recorded run(s) — every one of them left a counter reading."),
        "nothing is said about runs that are not there: {out}"
    );
    assert!(
        out.contains("≈ $0"),
        "no tariff set, so no dollars claimed: {out}"
    );

    // With a tariff set the same run is priced in money, and the number is the
    // record's own watt-hours at that rate rather than a restatement.
    succeeds(
        dir.path(),
        home.path(),
        &["config", "set", "power_cents_per_kwh", "30"],
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "costs", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["team_runs"]["cents_per_kwh"], 30.0);
    let _ = out;
}

#[test]
fn retry_without_a_name_launches_nothing_and_an_empty_name_is_refused() {
    let marker = "RETRY-MUST-NOT-EXIST";
    let dir = project(&[("demo.toml", recipe("demo", &format!("touch {marker}")))]);
    let home = sandbox_home(&[]);

    let out = succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "retry", "demo", "harden"],
    );
    assert!(out.contains("Nothing has run."), "{out}");
    assert!(out.contains("One role replayed is not the team"), "{out}");
    assert!(
        out.contains("To do it: xencode orchestrator retry demo harden --approved-by <your name>"),
        "and what would do it: {out}"
    );
    assert!(
        !dir.path().join(marker).exists(),
        "the unapproved reading executed the role's command"
    );
    assert!(
        !dir.path().join(".xencode").join("team-runs").exists(),
        "the unapproved reading wrote a record"
    );
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &[
            "orchestrator",
            "retry",
            "demo",
            "harden",
            "--format",
            "json",
        ],
    ))
    .unwrap();
    assert_eq!(json["approval_required"], true);
    assert_eq!(json["command"], format!("touch {marker}"));
    assert!(
        !dir.path().join(marker).exists(),
        "the JSON form executed the role either"
    );

    let said = refuses(
        dir.path(),
        home.path(),
        &[
            "orchestrator",
            "retry",
            "demo",
            "harden",
            "--approved-by",
            "",
        ],
    );
    assert!(said.contains("`--approved-by` needs a name"), "{said}");

    let said = refuses(
        dir.path(),
        home.path(),
        &[
            "orchestrator",
            "retry",
            "demo",
            "nosuchrole",
            "--approved-by",
            "Tester",
        ],
    );
    assert!(
        said.contains("demo has no role named `nosuchrole`. It has: survey, harden, integrate"),
        "{said}"
    );

    // The shipped posture, on a role that names a vendor's agent.
    let dir2 = project(&[("external.toml", external_recipe())]);
    let home2 = sandbox_home(&[]);
    let said = refuses(
        dir2.path(),
        home2.path(),
        &[
            "orchestrator",
            "retry",
            "external",
            "ask-claude",
            "--approved-by",
            "Tester",
        ],
    );
    assert!(
        said.contains(
            "the Local Only posture refuses the worker this role names, so nothing was launched"
        ),
        "{said}"
    );
    assert!(
        !dir2.path().join("SHOULD-NOT-EXIST").exists(),
        "a refused role still ran its command"
    );
}

#[test]
fn retry_re_runs_one_role_for_real_and_writes_no_run_record() {
    let dir = project(&[("demo.toml", recipe("demo", "echo hardened-again"))]);
    let home = sandbox_home(&[]);
    record_a_run(dir.path(), home.path());

    let out = succeeds(
        dir.path(),
        home.path(),
        &[
            "orchestrator",
            "retry",
            "demo",
            "harden",
            "--approved-by",
            "Tester",
        ],
    );
    assert!(
        out.contains("Re-ran harden of demo · approved by Tester"),
        "{out}"
    );
    assert!(out.contains("command:  echo hardened-again"), "{out}");
    assert!(out.contains("exited(0)"), "the child ran and ended: {out}");
    assert!(
        out.contains("hardened-again"),
        "its output is quoted: {out}"
    );
    assert!(
        out.contains("It waits on nothing, so the graph above it is not in question."),
        "{out}"
    );
    assert!(
        out.contains("Not re-run after it: integrate"),
        "what it did not do is said: {out}"
    );
    assert!(out.contains("No run record was written"), "{out}");

    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &[
            "orchestrator",
            "retry",
            "demo",
            "harden",
            "--approved-by",
            "Tester",
            "--format",
            "json",
        ],
    ))
    .unwrap();
    assert_eq!(json["run_record_written"], false);
    assert_eq!(json["approved_by"], "Tester");
    assert_eq!(json["status"], "exited(0)");
    assert_eq!(json["output_tail"], serde_json::json!(["hardened-again"]));
    assert_eq!(
        json["not_run"]["needed_by_this_role"],
        serde_json::json!([])
    );
    assert_eq!(
        json["not_run"]["roles_that_were_waiting_on_this_one"],
        serde_json::json!(["integrate"])
    );
    assert!(json["watt_hours"].as_f64().is_some());

    // The record count is the proof, not the sentence above it.
    assert_eq!(
        record_ids(dir.path()).len(),
        1,
        "a single-role replay wrote itself into the history a plan quotes"
    );
}

#[test]
fn tasks_lists_both_registries_and_stop_reaches_only_the_recorded_pid() {
    let dir = project(&[]);
    let home = sandbox_home(&[]);

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "tasks"]);
    assert!(
        out.contains("none recorded — the registry answered the read and held no rows."),
        "{out}"
    );
    assert!(
        out.contains("none — `xencode run \"the task\" --detach` starts one."),
        "{out}"
    );

    // A real child of a real `sh` wrapper, asleep long enough for this test to find
    // it and short enough that a failure here cannot leave it behind.
    succeeds(
        dir.path(),
        home.path(),
        &["tasks", "start", "sleep 9.7", "--name", "keeper"],
    );
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "tasks"]);
    assert!(out.contains("ID  STATUS            PID"), "{out}");
    assert!(out.contains("keeper"), "{out}");
    assert!(out.contains("running"), "the row reads as running: {out}");
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "tasks", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["tasks"].as_array().unwrap().len(), 1);
    assert_eq!(json["tasks"][0]["name"], "keeper");
    assert_eq!(json["tasks"][0]["command"], "sleep 9.7");
    assert_eq!(json["tasks"][0]["status"], "running");
    assert_eq!(json["tasks"][0]["killed"], false);
    assert_eq!(json["tasks"][0]["source"], serde_json::Value::Null);
    let pid = json["tasks"][0]["pid"].as_u64().unwrap() as u32;
    assert!(alive(pid), "the registry's pid is a live process: {pid}");
    let started = json["tasks"][0]["started_at_unix_secs"].as_u64().unwrap();
    assert!(
        started > 1_700_000_000,
        "the registry's stamp, in whole seconds: {started}"
    );

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "inspect", "1"]);
    assert!(out.contains("Background task #1 ·"), "{out}");
    assert!(out.contains("sleep 9.7"), "{out}");
    assert!(out.contains("(running)"), "{out}");
    assert!(
        out.contains("the registry's own stamp, in whole seconds since the epoch"),
        "{out}"
    );

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "stop", "1"]);
    assert!(
        out.contains(&format!("Stopped task #1 (keeper, pid {pid}).")),
        "{out}"
    );
    assert!(
        out.contains("The pid the registry holds is the `sh -c` wrapper"),
        "and the limit on what that means, said out loud: {out}"
    );
    assert!(!alive(pid), "the recorded wrapper is dead");
    // That sentence is the claim this check watches: the wrapper died and the
    // `sleep` it spawned did not die with it.
    let orphans = pids_running(&["sleep", "9.7"]);
    assert!(
        !orphans.is_empty(),
        "the wrapper's child went down with it, which the message did not expect"
    );
    for orphan in &orphans {
        let _ = Command::new("kill").arg(orphan.to_string()).status();
    }
    assert!(
        pids_running(&["sleep", "9.7"]).is_empty(),
        "this test left a stray process behind"
    );

    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "tasks", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["tasks"][0]["killed"], true, "the row keeps the stop");
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "inspect", "1"]);
    assert!(out.contains("yes — this xencode stopped it"), "{out}");

    let said = refuses(dir.path(), home.path(), &["orchestrator", "stop", "1"]);
    assert!(
        said.contains("not running — there is nothing to stop"),
        "{said}"
    );
    let said = refuses(dir.path(), home.path(), &["orchestrator", "stop", "42"]);
    assert!(said.contains("no background task #42 in"), "{said}");
    let said = refuses(dir.path(), home.path(), &["orchestrator", "stop", "zz-id"]);
    assert!(said.contains("no detached run zz-id"), "{said}");
    assert!(
        said.contains("`xencode orchestrator tasks` names both kinds that can be stopped"),
        "{said}"
    );
}

#[test]
fn attach_refuses_every_case_that_would_have_to_guess() {
    let dir = project(&[]);
    let home = sandbox_home(&[]);
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["orchestrator", "agents", "--format", "json"],
    ))
    .unwrap();
    let rows = json["agents"].as_array().unwrap();
    let name_of = |row: &serde_json::Value| row["agent"].as_str().unwrap().to_string();
    let row_without = rows
        .iter()
        .find(|row| row["handover"].is_null())
        .map(name_of)
        .expect("at least one roster row documents no handover verb");
    let with_verb = rows
        .iter()
        .find(|row| row["handover"].is_string())
        .cloned()
        .expect("at least one roster row documents a handover verb");
    let row_with = name_of(&with_verb);
    let template = with_verb["handover"].as_str().unwrap().to_string();

    // A row with no verb: refusing is the only honest answer, and the refusal
    // quotes the one command that vendor's help does document.
    let said = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "attach", &row_without],
    );
    assert!(
        said.contains(&format!(
            "{row_without} has no command that takes over a session it already has"
        )),
        "{said}"
    );
    assert!(said.contains("refuses instead of pretending"), "{said}");

    // A row with a verb but no session named: xencode does not pick one, and does
    // not run the vendor's own listing command to find one either.
    let said = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "attach", &row_with],
    );
    assert!(
        said.contains(&format!(
            "attach {row_with} needs the session to hand over, given as the vendor's own usage \
             line `{template}`"
        )),
        "{said}"
    );
    assert!(said.contains("does not list it for you"), "{said}");

    // A name that is not on the roster at all.
    let said = refuses(
        dir.path(),
        home.path(),
        &["orchestrator", "attach", "nosuchagent"],
    );
    assert!(
        said.contains("nosuchagent is not an agent xencode has a roster row for")
            && said.contains(&format!("lists the {} rows there are", rows.len())),
        "{said}"
    );

    // With nothing of the vendor's on PATH, even a named session reaches no
    // process — the last refusal before xencode would start one.
    let empty_path = dir.path().join("empty-bin");
    fs::create_dir_all(&empty_path).unwrap();
    let out = Command::new(xencode_bin())
        .args(["orchestrator", "attach", &row_with, "http://127.0.0.1:1"])
        .current_dir(dir.path())
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path().join(".config"))
        .env("PATH", &empty_path)
        .output()
        .expect("xencode ran");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "attach found a binary that is not on PATH: {said}"
    );
    assert!(
        said.contains(
            "no binary for it is on PATH here, so there is nothing to hand the terminal to"
        ),
        "{said}"
    );

    // And where a roster agent really is installed, the guard is that there is no
    // terminal here to hand over — the other half of the same rule. Standard
    // output of a test's `Command` is a pipe, never a terminal.
    let installed_with_verb = rows
        .iter()
        .find(|row| row["handover"].is_string() && row["installed"] == serde_json::json!(true))
        .map(name_of);
    match installed_with_verb {
        Some(agent) => {
            let said = refuses(
                dir.path(),
                home.path(),
                &["orchestrator", "attach", &agent, "http://127.0.0.1:1"],
            );
            assert!(
                said.contains("standard output here is not a terminal")
                    && said.contains("There is no terminal to hand over, so nothing was started."),
                "{said}"
            );
        }
        None => {
            // Nothing on this machine to hand a terminal to: the PATH refusal above
            // is the reached case, and this says so rather than passing quietly.
            let said = refuses(
                dir.path(),
                home.path(),
                &["orchestrator", "attach", &row_with, "http://127.0.0.1:1"],
            );
            assert!(said.contains("no binary for it is on PATH here"), "{said}");
        }
    }
    assert!(
        !dir.path().join("SHOULD-NOT-EXIST").exists(),
        "attach started something it should only have described"
    );
}

#[test]
fn the_surface_itself_is_named_and_every_reading_says_it_changed_nothing() {
    let dir = project(&[]);
    let home = sandbox_home(&[]);
    let out = succeeds(dir.path(), home.path(), &["orchestrator", "--help"]);
    for verb in [
        "status",
        "agents",
        "tasks",
        "graph",
        "logs",
        "permissions",
        "costs",
        "inspect",
        "retry",
        "stop",
        "attach",
    ] {
        assert!(
            out.contains(&format!("  {verb}")),
            "`{verb}` is not in the help:\n{out}"
        );
    }
    assert!(!run_in(dir.path(), home.path(), &["orchestrator"])
        .status
        .success());

    let out = succeeds(dir.path(), home.path(), &["orchestrator", "agents"]);
    assert!(
        out.contains("Nothing was launched, stopped or changed by this reading."),
        "{out}"
    );
    assert!(
        out.contains("`xencode team run <recipe> --approved-by <name>` runs a recipe"),
        "and what would change it: {out}"
    );
}
