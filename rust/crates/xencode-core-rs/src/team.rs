//! `OR-9` — a team as a file, not as an engine.
//!
//! A recipe is one TOML file naming the roles on a team, which worker plays each
//! role, and which checks gate that role's output. Nothing in this module
//! launches or supervises anything: [`TeamRecipe::to_task_graph`] hands the shape
//! to `OR-2`'s [`Scheduler`], and that is the whole of the machinery a recipe
//! drives. Dependency faults — a duplicate role, a need that names nothing, a
//! cycle — are detected there, not re-derived here.
//!
//! Two properties the item exists to hold:
//!
//! - **Readable and diffable.** The file is the feature. Every field a recipe can
//!   carry is one a person wrote, and an unknown key is refused rather than
//!   ignored, so a typing mistake cannot quietly turn into a team of nobody.
//! - **Removal takes nothing else with it.** Recipes are read from a directory of
//!   independent files. A recipe holds no state elsewhere — not in settings, not
//!   on disk beside it, not in another recipe's fields — so deleting one file
//!   deletes one recipe, and the directory being absent is a normal reading, not
//!   an error.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::scheduler::{GraphError, Scheduler, TaskGraph, TaskNode};

/// The checks a gate may name. These are the three slots the verification
/// checklist runs — `xencode-analysis-rs::toolchain::run_checklist` takes exactly
/// this vocabulary as its skip list — so a gate naming a fourth would be
/// promising a check nobody performs.
pub const GATE_CHECKS: [&str; 3] = ["fmt", "lint", "test"];

/// The directory recipes live in, under a project's `.xencode`.
pub const RECIPES_DIR: &str = "teams";

/// Why a file is not a recipe that can be scheduled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeError {
    /// Not TOML, a key a recipe does not have, or a key a recipe requires that
    /// is missing. Carries the path, because the person who has to fix it wrote
    /// one of several files.
    Parse { path: String, problem: String },
    /// Something that must be named or non-empty was left blank. A role with no
    /// command would launch `sh -c` on an empty string and exit zero, which
    /// reads as a pass for work nobody did.
    Blank { recipe: String, what: String },
    /// The recipe has a name and a capacity but no roles, so it schedules
    /// nothing. Almost always a mistyped `[[roles]]` header that serde could not
    /// see because the table was empty.
    NoRoles { recipe: String },
    /// A gate named a check the verification checklist does not run.
    UnknownGate {
        recipe: String,
        role: String,
        gate: String,
    },
    /// The roles do not form a schedulable graph. Detection is `OR-2`'s; this
    /// wrapper exists so the message says "role" and names the recipe.
    Graph { recipe: String, problem: GraphError },
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecipeError::Parse { path, problem } => {
                write!(f, "{path} is not a readable team recipe: {problem}")
            }
            RecipeError::Blank { recipe, what } => {
                write!(f, "recipe `{recipe}`: {what}")
            }
            RecipeError::NoRoles { recipe } => write!(
                f,
                "recipe `{recipe}` names no roles, so it would schedule nothing"
            ),
            RecipeError::UnknownGate { recipe, role, gate } => write!(
                f,
                "recipe `{recipe}`, role `{role}`: gate `{gate}` is not a check that exists. \
                 A gate is made of {available}",
                available = GATE_CHECKS.join(", ")
            ),
            RecipeError::Graph { recipe, problem } => {
                write!(f, "recipe `{recipe}`: {}", role_words(problem))
            }
        }
    }
}

impl std::error::Error for RecipeError {}

/// Say `OR-2`'s structural faults in the words a person writing a recipe used.
/// The check itself is not duplicated here — only its vocabulary is translated.
fn role_words(problem: &GraphError) -> String {
    match problem {
        GraphError::DuplicateNode(id) => format!("two roles are both named `{id}`"),
        GraphError::UnknownDependency { node, needs } => {
            format!("role `{node}` needs `{needs}`, which is not a role in this recipe")
        }
        GraphError::SelfDependency(id) => format!("role `{id}` needs itself"),
        GraphError::Cycle(ids) => {
            format!("these roles form a dependency cycle: {}", ids.join(" → "))
        }
    }
}

/// How wide the team may run at once. Both numbers come from a person, and the
/// queue's real capacity is the smaller of the two — see [`Scheduler::capacity`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capacity {
    /// How many workers the team could have running at once.
    pub workers: usize,
    /// How many finished results can be verified at once.
    pub verification_throughput: usize,
}

/// One role: a worker, a gate, a command, and what it waits on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleSpec {
    pub name: String,
    /// Which agent plays this role. A recipe says who is on the team; choosing a
    /// worker by probed capability is a different thing entirely (`OR-6`).
    pub worker: String,
    /// The checks that must pass for this role's output. `[]` is legal and means
    /// no check gates this role — which is reported as such rather than hidden
    /// behind a green-looking line.
    pub gate: Vec<String>,
    /// The command the role runs. One node, one `sh -c` child, exactly as `OR-2`
    /// schedules it.
    pub command: String,
    /// Roles whose completion this one waits on.
    #[serde(default)]
    pub needs: Vec<String>,
}

/// A team recipe: what was read out of one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamRecipe {
    pub name: String,
    pub roles: Vec<RoleSpec>,
    pub capacity: Capacity,
}

/// One file in the recipes directory, with whatever it turned out to hold. A
/// recipe that does not parse is carried beside the ones that do so that one
/// typing mistake cannot hide a team somebody else wrote.
#[derive(Debug, Clone)]
pub struct RecipeFile {
    pub path: PathBuf,
    pub recipe: Result<TeamRecipe, RecipeError>,
}

impl TeamRecipe {
    /// Read a recipe from text. `path` is only ever used to name the file in an
    /// error, so a caller with no file can pass something recognizable.
    pub fn parse(path: &Path, text: &str) -> Result<Self, RecipeError> {
        toml::from_str(text).map_err(|e| RecipeError::Parse {
            path: path.display().to_string(),
            problem: e.to_string(),
        })
    }

    pub fn from_file(path: &Path) -> Result<Self, RecipeError> {
        let text = std::fs::read_to_string(path).map_err(|e| RecipeError::Parse {
            path: path.display().to_string(),
            problem: e.to_string(),
        })?;
        Self::parse(path, &text)
    }

    /// Everything a recipe has to get right on its own, before `OR-2` looks at
    /// the shape of the graph: names present, at least one role, and only gates
    /// that name a check the verification checklist really runs.
    pub fn validate(&self) -> Result<(), RecipeError> {
        let blank = |what: String| RecipeError::Blank {
            recipe: self.name.clone(),
            what,
        };
        if self.name.trim().is_empty() {
            return Err(blank("the recipe has no name".to_string()));
        }
        if self.roles.is_empty() {
            return Err(RecipeError::NoRoles {
                recipe: self.name.clone(),
            });
        }
        // `Scheduler::capacity` floors at one, so a zero here would not run zero
        // roles — it would read as one. Refusing it keeps the file honest about
        // itself.
        if self.capacity.workers == 0 {
            return Err(blank(
                "[capacity] workers is 0, so no role could ever start".to_string(),
            ));
        }
        if self.capacity.verification_throughput == 0 {
            return Err(blank(
                "[capacity] verification_throughput is 0, so nothing could be checked".to_string(),
            ));
        }
        for role in &self.roles {
            if role.name.trim().is_empty() {
                return Err(blank("a role has no name".to_string()));
            }
            if role.worker.trim().is_empty() {
                return Err(blank(format!("role `{}` has no worker", role.name)));
            }
            if role.command.trim().is_empty() {
                return Err(blank(format!(
                    "role `{}` has no command — it would run nothing and exit zero",
                    role.name
                )));
            }
            for gate in &role.gate {
                if !GATE_CHECKS.contains(&gate.as_str()) {
                    return Err(RecipeError::UnknownGate {
                        recipe: self.name.clone(),
                        role: role.name.clone(),
                        gate: gate.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Compile to the graph `OR-2` schedules, in role order so that readiness is
    /// deterministic — the same file always launches the same role first when two
    /// are ready at once.
    pub fn to_task_graph(&self) -> Result<TaskGraph, RecipeError> {
        self.validate()?;
        let mut graph = TaskGraph::new();
        for role in &self.roles {
            graph.add(TaskNode {
                id: role.name.clone(),
                command: role.command.clone(),
                needs: role.needs.clone(),
            });
        }
        graph.validate().map_err(|problem| RecipeError::Graph {
            recipe: self.name.clone(),
            problem,
        })?;
        Ok(graph)
    }

    /// The scheduler this recipe describes: its capacity numbers and nothing
    /// else. Building it launches nothing.
    pub fn scheduler(&self) -> Scheduler {
        Scheduler::new(self.capacity.workers, self.capacity.verification_throughput)
    }
}

/// Read every `*.toml` in `dir`. A directory that is not there is an empty
/// answer, not a failure: removing the recipes is supposed to take nothing else
/// with it, including an error.
pub fn load_recipes(dir: &Path) -> Result<Vec<RecipeFile>, std::io::Error> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") || !path.is_file() {
            continue;
        }
        let recipe = TeamRecipe::from_file(&path);
        files.push(RecipeFile { path, recipe });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{TaskManager, TaskStatus};
    use std::collections::BTreeSet;
    use tempfile::tempdir;

    /// The shape `OR-2`'s own done-when describes: two independent branches, one
    /// waiting on the other two, and a join — written the way a person writes it.
    const FOUR_ROLE: &str = r#"
name = "rust-fix"

[[roles]]
name = "survey"
worker = "opencode"
gate = []
command = "true"

[[roles]]
name = "harden"
worker = "claude"
gate = ["test"]
command = "true"

[[roles]]
name = "integrate"
worker = "opencode"
gate = ["lint", "test"]
command = "true"
needs = ["survey", "harden"]

[capacity]
workers = 2
verification_throughput = 2
"#;

    fn recipe(text: &str) -> TeamRecipe {
        TeamRecipe::parse(Path::new("test.toml"), text).expect(text)
    }

    #[test]
    fn a_recipe_is_a_file_a_person_can_read_and_diff() {
        let r = recipe(FOUR_ROLE);
        assert_eq!(r.name, "rust-fix");
        assert_eq!(r.roles.len(), 3);
        assert_eq!(r.roles[0].worker, "opencode");
        assert_eq!(r.roles[1].gate, vec!["test".to_string()]);
        assert!(
            r.roles[0].needs.is_empty(),
            "a branch head waits on nothing"
        );
        assert_eq!(r.roles[2].needs, vec!["survey", "harden"]);
        assert_eq!(r.capacity.workers, 2);
        assert_eq!(r.capacity.verification_throughput, 2);
    }

    #[test]
    fn the_recipe_compiles_into_the_graph_or_2_schedules() {
        let graph = recipe(FOUR_ROLE).to_task_graph().expect("schedulable");
        let ids: Vec<&str> = graph.nodes().iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["survey", "harden", "integrate"]);
        let none_done = BTreeSet::new();
        let ready: Vec<&str> = graph
            .ready(&none_done)
            .iter()
            .map(|n| n.id.as_str())
            .collect();
        assert_eq!(
            ready,
            vec!["survey", "harden"],
            "the two branches are ready together and the join waits"
        );
    }

    #[tokio::test]
    async fn a_recipe_runs_on_real_children_through_that_graph() {
        // Not a stand-in: these are `sh -c` children of this test process, and
        // every node's exit status is the one the OS reported.
        let r = recipe(FOUR_ROLE);
        let graph = r.to_task_graph().unwrap();
        let mut manager = TaskManager::new();
        let report = r
            .scheduler()
            .run(&graph, &mut manager)
            .await
            .expect("the schedule completes");
        assert_eq!(report.capacity, 2);
        assert_eq!(report.launch_order[0], "survey");
        assert_eq!(report.launch_order[1], "harden");
        assert_eq!(report.bottleneck.as_deref(), Some("integrate"));
        for id in ["survey", "harden", "integrate"] {
            let outcome = report
                .outcome(id)
                .unwrap_or_else(|| panic!("{id} never ran"));
            assert_eq!(outcome.status, TaskStatus::Exited(0), "{id} did not exit 0");
        }
    }

    #[test]
    fn the_smaller_of_the_two_capacity_numbers_is_the_one_that_binds() {
        // Eight agents, one verifier: the queue is one deep and the report says
        // verification is what limited it, not the worker count.
        let text = FOUR_ROLE.replace(
            "[capacity]\nworkers = 2\nverification_throughput = 2",
            "[capacity]\nworkers = 8\nverification_throughput = 1",
        );
        let r = recipe(&text);
        assert_eq!(r.scheduler().capacity(), 1);
        assert_eq!(r.scheduler().binding().label(), "verification throughput");
    }

    #[test]
    fn a_key_that_is_not_part_of_a_recipe_is_refused_not_ignored() {
        // `[[role]]` instead of `[[roles]]` would otherwise read as a team of
        // nobody, and the file would look fine to the person who wrote it.
        let bad = FOUR_ROLE.replace("[[roles]]", "[[role]]");
        let err = TeamRecipe::parse(Path::new("rust-fix.toml"), &bad).unwrap_err();
        assert!(matches!(err, RecipeError::Parse { .. }), "{err}");
        assert!(err.to_string().contains("rust-fix.toml"), "{err}");
        assert!(err.to_string().contains("role"), "{err}");
    }

    #[test]
    fn a_missing_capacity_is_refused_because_no_default_would_be_someone_s_guess() {
        let bad = FOUR_ROLE.replace("[capacity]\nworkers = 2\nverification_throughput = 2", "");
        let err = TeamRecipe::parse(Path::new("rust-fix.toml"), &bad).unwrap_err();
        assert!(err.to_string().contains("capacity"), "{err}");
    }

    #[test]
    fn a_gate_naming_a_check_that_does_not_exist_is_refused_with_the_alternatives() {
        let bad = FOUR_ROLE.replace("gate = [\"lint\", \"test\"]", "gate = [\"bench\"]");
        let r = recipe(&bad);
        let err = r.validate().unwrap_err();
        assert!(
            matches!(err, RecipeError::UnknownGate { .. }),
            "expected a refused gate, got {err}"
        );
        let said = err.to_string();
        assert!(said.contains("integrate"), "names the role: {said}");
        assert!(said.contains("bench"), "names what was asked for: {said}");
        assert!(
            said.contains("fmt, lint, test"),
            "names the real set: {said}"
        );
    }

    #[test]
    fn a_role_with_no_command_is_refused_before_it_could_pass_at_nothing() {
        let bad = FOUR_ROLE.replace("command = \"true\"\nneeds =", "command = \"\"\nneeds =");
        let err = recipe(&bad).validate().unwrap_err();
        assert!(
            matches!(err, RecipeError::Blank { .. }),
            "an empty command should not schedule: {err}"
        );
        assert!(err.to_string().contains("integrate"), "{err}");
    }

    #[test]
    fn a_recipe_with_no_roles_is_refused() {
        // Built directly: a file that omitted `roles` would not parse at all, so
        // the case this guards is a recipe that parses and still schedules nothing.
        let r = TeamRecipe {
            name: "empty".to_string(),
            roles: Vec::new(),
            capacity: Capacity {
                workers: 1,
                verification_throughput: 1,
            },
        };
        let err = r.validate().unwrap_err();
        assert!(matches!(err, RecipeError::NoRoles { .. }), "{err}");
    }

    #[test]
    fn a_capacity_of_zero_is_refused_because_the_queue_would_not_run_zero_of_anything() {
        // `Scheduler::capacity` floors at one, so a file saying `workers = 0`
        // would be reported as a team of one. The file is refused instead.
        for (key, word) in [
            ("workers", "no role could ever start"),
            ("verification_throughput", "nothing could be checked"),
        ] {
            let zero = FOUR_ROLE.replace(&format!("{key} = 2"), &format!("{key} = 0"));
            let err = recipe(&zero).validate().unwrap_err();
            assert!(matches!(err, RecipeError::Blank { .. }), "{key} = 0: {err}");
            let said = err.to_string();
            assert!(said.contains("rust-fix"), "names the recipe: {said}");
            assert!(said.contains(word), "names the consequence: {said}");
        }
    }

    #[test]
    fn a_cycle_among_roles_is_caught_before_anything_could_wait_on_it() {
        let text = r#"
name = "loop"

[[roles]]
name = "a"
worker = "opencode"
gate = []
command = "true"
needs = ["b"]

[[roles]]
name = "b"
worker = "opencode"
gate = []
command = "true"
needs = ["a"]

[capacity]
workers = 2
verification_throughput = 2
"#;
        let err = recipe(text).to_task_graph().unwrap_err();
        assert!(matches!(err, RecipeError::Graph { .. }), "{err}");
        assert!(err.to_string().contains("cycle"), "{err}");
        assert!(err.to_string().contains("a → b"), "names the roles: {err}");
    }

    #[test]
    fn a_need_naming_a_role_that_is_not_there_is_refused_by_name() {
        let text = FOUR_ROLE.replace("needs = [\"survey\", \"harden\"]", "needs = [\"review\"]");
        let err = recipe(&text).to_task_graph().unwrap_err();
        assert!(
            err.to_string()
                .contains("role `integrate` needs `review`, which is not a role"),
            "{err}"
        );
    }

    #[test]
    fn one_unreadable_recipe_does_not_hide_the_others() {
        let dir = tempdir().unwrap();
        write(dir.path(), "good.toml", FOUR_ROLE);
        write(dir.path(), "broken.toml", "name = = 3\n");
        let files = load_recipes(dir.path()).unwrap();
        assert_eq!(files.len(), 2, "both files are reported");
        assert_eq!(files[0].path.file_name().unwrap(), "broken.toml");
        assert!(files[0].recipe.is_err(), "the bad one says so");
        assert_eq!(files[1].recipe.as_ref().unwrap().name, "rust-fix");
    }

    #[test]
    fn removing_a_recipe_takes_nothing_else_with_it() {
        // The other file is not edited, not renumbered, not invalidated: it reads
        // back the same team it did before.
        let dir = tempdir().unwrap();
        write(dir.path(), "one.toml", FOUR_ROLE);
        write(
            dir.path(),
            "two.toml",
            &FOUR_ROLE.replace("rust-fix", "docs-pass"),
        );
        let before = load_recipes(dir.path()).unwrap();
        assert_eq!(before.len(), 2);
        std::fs::remove_file(dir.path().join("one.toml")).unwrap();
        let after = load_recipes(dir.path()).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0].recipe, before[1].recipe,
            "the surviving recipe changed when an unrelated one was deleted"
        );
        assert_eq!(after[0].recipe.as_ref().unwrap().name, "docs-pass");
    }

    #[test]
    fn no_recipes_directory_at_all_is_an_empty_answer_not_a_failure() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("teams");
        assert!(load_recipes(&missing).unwrap().is_empty());
    }

    #[test]
    fn only_dot_toml_files_are_recipes() {
        let dir = tempdir().unwrap();
        write(dir.path(), "rust-fix.toml", FOUR_ROLE);
        write(dir.path(), "notes.md", "# not a recipe\n");
        write(
            dir.path(),
            "rust-fix.toml.bak",
            "this is not even close to toml = =\n",
        );
        let files = load_recipes(dir.path()).unwrap();
        assert_eq!(
            files.len(),
            1,
            "{:?}",
            files.iter().map(|f| &f.path).collect::<Vec<_>>()
        );
    }

    fn write(dir: &Path, name: &str, text: &str) {
        std::fs::write(dir.join(name), text).unwrap();
    }
}
