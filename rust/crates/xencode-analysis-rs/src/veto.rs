//! OR-17 — the veto.
//!
//! A review, or a verification run that came back red, can say *this branch
//! does not land*. That decision has to outlive the pipeline it came out of, so
//! a veto is kept in the repository (`.xencode/vetoes.json`), re-read from disk
//! by [`crate::merge_decision::execute_merge`] rather than trusted from whatever
//! the plan happened to carry, and lifted only by an identity that is not the
//! worker it blocks.
//!
//! ## What is enforced, and what is not
//!
//! **Enforced:** nothing the blocked worker emits can clear its own veto. A
//! check result, a claim in prose, a rerun that happens to go green — none of
//! them reach the clearing path, because clearing is a separate operation that
//! takes a separate identity and refuses that identity by name. The worker's own
//! artifacts are inputs to *raising* a veto and never to removing one.
//!
//! **Not enforced:** the CLI takes a name (`--by`), it does not authenticate
//! one. Someone — or an agent holding a shell — can type a name that is not
//! theirs, exactly as they can today with `merge land --approved-by`. This is a
//! repository of files on a machine that runs the code; there is no key here to
//! check a signature against. What that leaves is the part that matters anyway:
//! a clearing is impossible to do *without* naming someone, and naming someone
//! writes a record into the chained audit trail that `xencode audit verify`
//! walks. The guard is against an outcome laundering itself, not against a
//! determined user of this machine.
//!
//! ## Why clearing needs the log and recording does not
//!
//! The two fail in opposite directions on purpose. Recording a veto while the
//! audit trail is unwritable keeps the veto — a block that exists but is
//! unlogged is still a block, and dropping it because bookkeeping failed would
//! remove protection. Clearing one while the trail is unwritable refuses, and
//! the veto stays: lifting a block without leaving a trace is the exact failure
//! this file exists to prevent. The caller therefore writes the record first and
//! only then persists the clearance.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Where the vetoes for one repository live, under its `.xencode` directory.
pub const VETO_FILE: &str = "vetoes.json";

/// What kind of outcome the block came out of. Kept because a human reading a
/// blocked plan wants to know whether a person looked at the code or a command
/// exited nonzero — the two call for different follow-ups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VetoSource {
    /// A person read the change and said no.
    Review,
    /// A check that ran came back failed, or never ran.
    Verification,
}

impl VetoSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Review => "review",
            Self::Verification => "verification",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "review" => Some(Self::Review),
            "verification" => Some(Self::Verification),
            _ => None,
        }
    }
}

/// Who took a veto off, and under what authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VetoClearance {
    pub cleared_by: String,
    pub at_ms: u64,
    /// A policy statement written out in full, naming the veto it clears.
    /// `None` means a person cleared it on their own judgement.
    pub policy: Option<String>,
}

/// One block on one branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Veto {
    pub id: String,
    pub branch: String,
    /// The worker whose branch this is. It is the identity a clearing is
    /// refused for, so this is recorded, not looked up at the moment of use.
    pub worker: String,
    pub source: VetoSource,
    /// Who or what raised it.
    pub raised_by: String,
    pub reason: String,
    pub at_ms: u64,
    pub cleared: Option<VetoClearance>,
}

impl Veto {
    pub fn is_open(&self) -> bool {
        self.cleared.is_none()
    }

    /// Two identities that mean the same person or agent. Compared on the
    /// trimmed, case-folded name because a worker recorded as `Codex` and a
    /// reviewer typing `codex` to clear the block are the same party.
    fn same_identity(one: &str, other: &str) -> bool {
        one.trim().eq_ignore_ascii_case(other.trim())
    }

    /// Whether `name` is the party this veto blocks, and so may not lift it.
    pub fn is_blocked_party(&self, name: &str) -> bool {
        Self::same_identity(name, &self.worker)
    }

    /// What it would take to lift this, in the words a human is shown. Names
    /// the blocked party so the reader can see who is excluded.
    pub fn who_may_clear(&self) -> String {
        format!(
            "a named reviewer or the human at the keyboard — not {}, which is the \
             worker this veto blocks",
            self.worker
        )
    }

    /// Who lifted this, once it has been lifted.
    pub fn cleared_by(&self) -> Option<&str> {
        self.cleared.as_ref().map(|c| c.cleared_by.as_str())
    }

    /// Where this record is kept, as a pointer an auditor can follow.
    pub fn evidence_ref(&self) -> String {
        format!(
            "{}/{VETO_FILE}#{}",
            xencode_context_rs::XENCODE_DIR,
            self.id
        )
    }

    /// One line, for plan output and for a refusal message.
    pub fn summary_line(&self) -> String {
        format!(
            "{}: {} veto from {} — {}",
            self.id,
            self.source.label(),
            self.raised_by,
            self.reason
        )
    }
}

/// The whole veto file. A named type so the JSON on disk is an object with a
/// version-shape rather than a bare array someone can mistake for anything.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VetoStore {
    pub vetoes: Vec<Veto>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn veto_path(repo_root: &Path) -> PathBuf {
    repo_root
        .join(xencode_context_rs::XENCODE_DIR)
        .join(VETO_FILE)
}

/// Read every veto on record.
///
/// A file that exists but does not parse is an error, not an empty list. Every
/// other caller of this crate treats unreadable state as absent state; here
/// that would silently un-block a branch, which is the one mistake a check like
/// this cannot be trusted to make.
pub fn load_vetoes(repo_root: &Path) -> Result<Vec<Veto>, String> {
    let path = veto_path(repo_root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read the veto file {}: {e}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let store: VetoStore = serde_json::from_str(&text).map_err(|e| {
        format!(
            "could not parse the veto file {}: {e} — an unreadable veto list is not an empty one",
            path.display()
        )
    })?;
    Ok(store.vetoes)
}

fn save_vetoes(repo_root: &Path, vetoes: &[Veto]) -> Result<(), String> {
    let path = veto_path(repo_root);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| {
            format!(
                "could not create the directory {} for the veto file: {e}",
                dir.display()
            )
        })?;
    }
    let store = VetoStore {
        vetoes: vetoes.to_vec(),
    };
    xencode_context_rs::index::write_atomic(&path, &store)
        .map_err(|e| format!("could not write the veto file {}: {e}", path.display()))
}

/// The next veto id: one past the highest numbered one on record. Readable in a
/// refusal message and stable enough to quote in a policy statement.
fn next_id(vetoes: &[Veto]) -> String {
    let mut highest = 0u64;
    for veto in vetoes {
        if let Some(digits) = veto.id.strip_prefix("veto-") {
            if let Ok(number) = digits.parse::<u64>() {
                highest = highest.max(number);
            }
        }
    }
    format!("veto-{:04}", highest + 1)
}

/// Record that a review or a verification outcome blocks `branch`.
///
/// `worker` is the party being blocked and so the party that may not lift the
/// block; the caller passes the one it observed (the CLI reads it from the
/// branch's own commit author) rather than a name the worker chose.
pub fn record_veto(
    repo_root: &Path,
    branch: &str,
    worker: &str,
    source: VetoSource,
    raised_by: &str,
    reason: &str,
) -> Result<Veto, String> {
    for (field, value) in [
        ("branch", branch),
        ("worker", worker),
        ("raised-by", raised_by),
        ("reason", reason),
    ] {
        if value.trim().is_empty() {
            return Err(format!("a veto needs a {field}, and none was given"));
        }
    }
    let mut vetoes = load_vetoes(repo_root)?;
    let veto = Veto {
        id: next_id(&vetoes),
        branch: branch.trim().to_string(),
        worker: worker.trim().to_string(),
        source,
        raised_by: raised_by.trim().to_string(),
        reason: reason.trim().to_string(),
        at_ms: now_ms(),
        cleared: None,
    };
    vetoes.push(veto.clone());
    save_vetoes(repo_root, &vetoes)?;
    Ok(veto)
}

/// Decide whether a clearance is allowed, and return the veto as it would read
/// afterwards, **without writing anything**.
///
/// Split from [`clear_veto`] so that a caller can put the decision into the audit
/// trail before it exists on disk. Order matters here: a clearance that is logged
/// and then fails to persist leaves the branch blocked and the trail one record
/// ahead of reality, which is readable and safe; a clearance persisted first and
/// logged second leaves a window where the branch is open and nothing says so,
/// which is the exact failure this module exists to prevent.
///
/// Refuses when `cleared_by` is the blocked worker, when the veto is already
/// closed, and when a `policy` is offered that does not name the veto it is
/// clearing — a policy that says "clear anything" says out loud nothing.
pub fn check_clearance(
    veto: &Veto,
    cleared_by: &str,
    policy: Option<&str>,
) -> Result<Veto, String> {
    if cleared_by.trim().is_empty() {
        return Err("a veto is cleared by a name, and none was given".to_string());
    }
    if !veto.is_open() {
        let clearance = veto
            .cleared
            .as_ref()
            .map(|c| format!("'{}'", c.cleared_by))
            .unwrap_or_default();
        return Err(format!(
            "veto '{}' was already cleared by {clearance} — a veto clears once, and re-running \
             the clear does not make it open again",
            veto.id
        ));
    }
    if veto.is_blocked_party(cleared_by) {
        return Err(format!(
            "veto '{}' blocks '{}', the worker whose branch it is. A worker cannot lift its own \
             block; it has to be cleared by {}.",
            veto.id,
            veto.worker,
            veto.who_may_clear()
        ));
    }
    let policy = match policy {
        Some(text) => {
            let text = text.trim();
            if text.is_empty() {
                return Err(
                    "a policy clearance has to state the policy; --policy with nothing in it \
                     clears nothing"
                        .to_string(),
                );
            }
            if !text.contains(&veto.id) {
                return Err(format!(
                    "the policy has to say out loud what it clears: the statement does not name \
                     '{}'",
                    veto.id
                ));
            }
            Some(text.to_string())
        }
        None => None,
    };
    let mut cleared = veto.clone();
    cleared.cleared = Some(VetoClearance {
        cleared_by: cleared_by.trim().to_string(),
        at_ms: now_ms(),
        policy,
    });
    Ok(cleared)
}

/// Write a clearance produced by [`check_clearance`] back to the repository.
pub fn persist_clearance(repo_root: &Path, cleared: &Veto) -> Result<(), String> {
    let mut vetoes = load_vetoes(repo_root)?;
    let index = vetoes
        .iter()
        .position(|v| v.id == cleared.id)
        .ok_or_else(|| format!("no veto '{}' is on record in this repository", cleared.id))?;
    vetoes[index] = cleared.clone();
    save_vetoes(repo_root, &vetoes)
}

/// Lift one open veto, checking and persisting in one step.
///
/// This is the whole operation for a caller that does not keep an audit trail. A
/// caller that does should use [`check_clearance`], record the decision, and only
/// then [`persist_clearance`], so the record comes first.
pub fn clear_veto(
    repo_root: &Path,
    id: &str,
    cleared_by: &str,
    policy: Option<&str>,
) -> Result<Veto, String> {
    let mut vetoes = load_vetoes(repo_root)?;
    let index = vetoes
        .iter()
        .position(|v| v.id == id.trim())
        .ok_or_else(|| format!("no veto '{id}' is on record in this repository"))?;
    let cleared = check_clearance(&vetoes[index], cleared_by, policy)?;
    vetoes[index] = cleared.clone();
    save_vetoes(repo_root, &vetoes)?;
    Ok(cleared)
}

/// Every veto still blocking `branch`.
pub fn open_vetoes_for(repo_root: &Path, branch: &str) -> Result<Vec<Veto>, String> {
    Ok(load_vetoes(repo_root)?
        .into_iter()
        .filter(|veto| veto.is_open() && veto.branch == branch.trim())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn a_recorded_veto_is_open_and_names_who_may_lift_it() {
        let dir = repo();
        let veto = record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            "the retry swallows the error",
        )
        .unwrap();
        assert_eq!(veto.id, "veto-0001");
        assert!(veto.is_open());
        assert!(veto.who_may_clear().contains("codex"));
        let open = open_vetoes_for(dir.path(), "arm-1").unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].reason, "the retry swallows the error");
        assert_eq!(
            open[0].evidence_ref(),
            ".xencode/vetoes.json#veto-0001",
            "the pointer an auditor follows"
        );
    }

    #[test]
    fn ids_count_up_and_other_branches_are_untouched() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Verification,
            "ci",
            "clippy exited 101",
        )
        .unwrap();
        let second = record_veto(
            dir.path(),
            "arm-2",
            "claude",
            VetoSource::Review,
            "Alice",
            "needs a test",
        )
        .unwrap();
        assert_eq!(second.id, "veto-0002");
        assert!(!open_vetoes_for(dir.path(), "arm-1").unwrap().is_empty());
        assert_eq!(open_vetoes_for(dir.path(), "arm-2").unwrap().len(), 1);
        assert!(open_vetoes_for(dir.path(), "arm-3").unwrap().is_empty());
    }

    /// The done-when in one test: the blocked worker cannot clear its own block,
    /// and the refusal says who can.
    #[test]
    fn the_blocked_worker_cannot_clear_its_own_veto() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            "wrong",
        )
        .unwrap();

        let refused = clear_veto(dir.path(), "veto-0001", "codex", None).unwrap_err();
        assert!(refused.contains("cannot lift its own block"), "{refused}");
        assert!(open_vetoes_for(dir.path(), "arm-1").unwrap().len() == 1);

        // The same name with different case or padding is the same party.
        assert!(clear_veto(dir.path(), "veto-0001", "  CodEX ", None).is_err());
        // An empty name is not a clearance either.
        assert!(clear_veto(dir.path(), "veto-0001", "   ", None).is_err());

        let cleared = clear_veto(dir.path(), "veto-0001", "Alice", None).unwrap();
        assert_eq!(cleared.cleared_by(), Some("Alice"));
        assert!(open_vetoes_for(dir.path(), "arm-1").unwrap().is_empty());
    }

    /// A policy is only a clearance if it says what it clears.
    #[test]
    fn a_policy_must_name_the_veto_it_clears() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Verification,
            "ci",
            "tests red",
        )
        .unwrap();

        let vague = clear_veto(
            dir.path(),
            "veto-0001",
            "Alice",
            Some("docs-only changes land"),
        )
        .unwrap_err();
        assert!(vague.contains("does not name"), "{vague}");
        assert!(clear_veto(dir.path(), "veto-0001", "Alice", Some("")).is_err());
        assert!(
            open_vetoes_for(dir.path(), "arm-1").unwrap().len() == 1,
            "a refused policy left the block in place"
        );

        let cleared = clear_veto(
            dir.path(),
            "veto-0001",
            "Alice",
            Some("veto-0001 is cleared because the tests were re-run on the integrated tree"),
        )
        .unwrap();
        assert!(cleared
            .cleared
            .as_ref()
            .unwrap()
            .policy
            .as_ref()
            .unwrap()
            .contains("veto-0001"));
    }

    #[test]
    fn a_veto_clears_once_and_a_second_clear_says_so() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            "x",
        )
        .unwrap();
        clear_veto(dir.path(), "veto-0001", "Bob", None).unwrap();
        let again = clear_veto(dir.path(), "veto-0001", "Carol", None).unwrap_err();
        assert!(again.contains("already cleared by 'Bob'"), "{again}");
        assert!(clear_veto(dir.path(), "veto-9999", "Bob", None)
            .unwrap_err()
            .contains("no veto"));
    }

    /// Deciding a clearance and writing it are separate calls, so that the
    /// audit trail can take the decision before the veto file changes. A refused
    /// or undecided clearance must leave the file byte-for-byte alone.
    #[test]
    fn checking_a_clearance_decides_without_writing_anything() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            "the auth path is untested",
        )
        .unwrap();
        let path = veto_path(dir.path());
        let before = std::fs::read(&path).unwrap();

        let stored = load_vetoes(dir.path()).unwrap();
        let veto = stored.iter().find(|v| v.id == "veto-0001").unwrap();

        // A refusal: the blocked worker asks. Nothing is written, and the veto
        // is still open on disk rather than cleared-and-unlogged.
        let refused = check_clearance(veto, "CODEX", None).unwrap_err();
        assert!(refused.contains("cannot lift its own block"), "{refused}");
        assert_eq!(std::fs::read(&path).unwrap(), before, "refusal wrote");

        // An acceptance, still not written: the caller can go and log it first.
        let cleared = check_clearance(veto, "Bob", None).unwrap();
        assert_eq!(cleared.cleared_by(), Some("Bob"));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "checking alone wrote to the veto file"
        );
        assert!(
            !load_vetoes(dir.path())
                .unwrap()
                .iter()
                .any(|v| v.id == "veto-0001" && !v.is_open()),
            "the veto reads as cleared before it was persisted"
        );

        // Only persisting changes it.
        persist_clearance(dir.path(), &cleared).unwrap();
        assert!(open_vetoes_for(dir.path(), "arm-1").unwrap().is_empty());
        assert_ne!(std::fs::read(&path).unwrap(), before);
    }

    /// A veto file that will not parse must not read as an empty one: the merge
    /// refuses rather than assuming nothing objects.
    #[test]
    fn an_unreadable_veto_file_is_an_error_not_a_clean_slate() {
        let dir = repo();
        record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            "x",
        )
        .unwrap();
        std::fs::write(veto_path(dir.path()), "{ not json").unwrap();
        let err = load_vetoes(dir.path()).unwrap_err();
        assert!(
            err.contains("an unreadable veto list is not an empty one"),
            "{err}"
        );
        assert!(record_veto(dir.path(), "arm-2", "codex", VetoSource::Review, "Al", "y").is_err());
        assert!(clear_veto(dir.path(), "veto-0001", "Al", None).is_err());
    }

    #[test]
    fn a_veto_needs_every_field_that_makes_it_attributable() {
        let dir = repo();
        for missing in ["", "  "] {
            assert!(record_veto(
                dir.path(),
                missing,
                "codex",
                VetoSource::Review,
                "Alice",
                "reason"
            )
            .is_err());
        }
        assert!(record_veto(
            dir.path(),
            "arm-1",
            "codex",
            VetoSource::Review,
            "Alice",
            ""
        )
        .is_err());
    }

    #[test]
    fn a_source_parses_only_the_two_words_it_knows() {
        assert_eq!(VetoSource::parse("Review"), Some(VetoSource::Review));
        assert_eq!(
            VetoSource::parse(" verification "),
            Some(VetoSource::Verification)
        );
        assert_eq!(VetoSource::parse("vibes"), None);
    }
}
