//! Where xencode keeps the files that belong to the person rather than to a
//! project: settings, the record of past sessions, downloaded data, and the
//! cache that can be thrown away.
//!
//! The four kinds of file do not want the same treatment. A settings file is
//! edited by hand and backed up. A session record grows with use and is not
//! interesting once the session is over. A cache is disposable by definition,
//! and a machine-wide cache cleaner is expected to delete it. A downloaded model
//! weighs gigabytes and is expensive to fetch again. Putting all four in one
//! directory means a person cannot do the useful thing — clean the cache —
//! without risking the other three, so they go where the XDG base directory
//! spec says each of them goes:
//!
//! | kind | resolves to | holds |
//! |---|---|---|
//! | settings | `$XDG_CONFIG_HOME/xencode`, else `~/.config/xencode` | `config.json` and its backups, `layout.json`, `model_advice.json`, `remotes/<host>.json`, `skills/`, the Colab keypair |
//! | state | `$XDG_STATE_HOME/xencode`, else `~/.local/state/xencode` | `conversation_memory.json`, `colab.json`, `audit.jsonl`, `last_panic.log`, `llamaserver.pid` |
//! | cache | `$XDG_CACHE_HOME/xencode`, else `~/.cache/xencode` | cached model responses, `advisories/` |
//! | data | `$XDG_DATA_HOME/xencode`, else `~/.local/share/xencode` | downloaded GGUF weights |
//!
//! `$XCODE_CONFIG_DIR` overrides all four and puts every kind back under one
//! directory, which is what a portable install and every test that isolates a
//! run needs: an override that moved only the settings file would leave the
//! response cache writing into the real home directory.
//!
//! **A directory that already exists wins over the legacy one.** Before these
//! locations existed, all four kinds lived in `~/.xencode`, and a person with
//! that directory still has their settings, keypair and history in it. Repoint
//! the paths and leave the files, and their configuration is untouched — xencode
//! simply starts a new, empty life somewhere else, and the checkpoints, memory
//! and key they had are gone from under them without a word. So each kind reads
//! from `~/.xencode` until something is actually written at the new location,
//! and [`migrate`] is the command that does the moving and says what it moved.

use std::path::{Path, PathBuf};

use crate::config::ConfigError;

/// The directory each kind of file is put under inside its base directory.
const APP_DIR: &str = "xencode";

/// What the layout was before these locations existed, and what a person who
/// has never run [`migrate`] still uses.
const LEGACY_DIR: &str = ".xencode";

/// The four kinds of file xencode keeps in the person's home.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Files {
    /// Read, edited and backed up by hand: `config.json` and friends.
    Settings,
    /// Written by xencode about what has happened: session memory, the audit
    /// trail, a crash record, the pid of a background server.
    State,
    /// Derivable by fetching or computing again, and fair game for a cleaner.
    Cache,
    /// Fetched once and expensive to fetch twice: model weights.
    Data,
}

impl Files {
    /// The name a person sees for this kind of file, in a path listing or a
    /// migration report.
    pub fn label(self) -> &'static str {
        match self {
            Self::Settings => "settings",
            Self::State => "state",
            Self::Cache => "cache",
            Self::Data => "downloaded models",
        }
    }

    /// The subdirectory this kind of file keeps under a single-directory
    /// layout: `~/.xencode`, or an `$XCODE_CONFIG_DIR` root. The settings and
    /// state kinds have always sat directly at the root, so they get no
    /// subdirectory in either layout, and the two that had one — the response
    /// cache and the downloaded weights — keep the same name in both.
    fn subdirectory(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::Data => "models",
            _ => "",
        }
    }
}

/// The directory the spec names for a kind of file, before the `xencode`
/// segment. `None` when the platform has no answer for it — which is also what
/// happens when there is no home directory to build a fallback from.
fn base(kind: Files) -> Option<PathBuf> {
    match kind {
        Files::Settings => dirs::config_dir(),
        // A platform without a state directory puts state in the config
        // directory, which is what the spec says and what `dirs` does for the
        // other bases.
        Files::State => dirs::state_dir().or_else(dirs::config_dir),
        Files::Cache => dirs::cache_dir(),
        Files::Data => dirs::data_dir(),
    }
}

fn modern(kind: Files) -> Option<PathBuf> {
    base(kind).map(|dir| dir.join(APP_DIR))
}

/// `~/.xencode`, or the subdirectory of it a kind of file used to live in.
fn legacy(kind: Files) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let dir = home.join(LEGACY_DIR);
    let suffix = kind.subdirectory();
    if suffix.is_empty() {
        Some(dir)
    } else {
        Some(dir.join(suffix))
    }
}

/// What `$XCODE_CONFIG_DIR` names, when it names anything. An empty value is not
/// an override, which is how it has behaved since before this module existed.
/// A relative name is resolved against the working directory, because every
/// guard that asks "is this path inside xencode's own files" compares prefixes
/// and a relative answer would not match an absolute one.
fn overridden(kind: Files) -> Option<PathBuf> {
    let root = std::env::var_os("XCODE_CONFIG_DIR")
        .map(PathBuf::from)
        .filter(|root| !root.as_os_str().is_empty())?;
    let root = if root.is_absolute() {
        root
    } else {
        std::env::current_dir().ok()?.join(root)
    };
    let suffix = kind.subdirectory();
    if suffix.is_empty() {
        Some(root)
    } else {
        Some(root.join(suffix))
    }
}

/// Where a kind of file lives: the override, else the XDG location if something
/// has been written there, else the legacy directory if the person still has
/// one, else the XDG location, which the first write creates.
///
/// Returns [`ConfigError::NoHomeDir`] only when there is neither an XDG base nor
/// a home directory to derive one from, because then there is no answer.
pub fn dir(kind: Files) -> Result<PathBuf, ConfigError> {
    if let Some(overridden) = overridden(kind) {
        return Ok(overridden);
    }
    let modern = modern(kind);
    if modern.as_deref().is_some_and(Path::exists) {
        return Ok(modern.expect("the `exists` check just read it"));
    }
    let legacy = legacy(kind);
    if legacy.as_deref().is_some_and(Path::exists) {
        return Ok(legacy.expect("the `exists` check just read it"));
    }
    modern.or(legacy).ok_or(ConfigError::NoHomeDir)
}

/// Where settings live. See the module documentation for what sits here.
pub fn settings_dir() -> Result<PathBuf, ConfigError> {
    dir(Files::Settings)
}

/// Where the record of what has happened lives, including the conversation
/// memory and the audit trail.
pub fn state_dir() -> Result<PathBuf, ConfigError> {
    dir(Files::State)
}

/// Where the throwaway copies live: cached responses and the advisory corpora.
pub fn cache_dir() -> Result<PathBuf, ConfigError> {
    dir(Files::Cache)
}

/// Where a downloaded model goes when nothing else was asked for.
pub fn data_dir() -> Result<PathBuf, ConfigError> {
    dir(Files::Data)
}

/// Every kind, in the order a path listing prints them.
pub const ALL: [Files; 4] = [Files::Settings, Files::State, Files::Cache, Files::Data];

/// Whether a path is inside any of the directories xencode keeps in the person's
/// home. A tool that may write into a project must not write into these, and
/// asking only about the settings directory would miss a cache or a state file
/// that has since moved out of it.
pub fn is_internal(path: &Path) -> bool {
    ALL.iter().filter_map(|kind| dir(*kind).ok()).any(|kept| {
        let kept = kept.as_path();
        path == kept || path.starts_with(kept)
    })
}

/// Where each kind of file would live if this person were starting now, whether
/// it is there, and where it actually is read from. This is what a migration
/// report is built from, and what `xencode paths` prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    /// Which kind of file.
    pub kind: Files,
    /// The directory the paths module resolves for it today.
    pub in_use: PathBuf,
    /// The XDG directory it would move to, or the same path if it is already
    /// there.
    pub modern: PathBuf,
    /// Whether `in_use` is the legacy `~/.xencode` directory.
    pub legacy: bool,
}

/// Describe where every kind of file lives right now.
pub fn locations() -> Vec<Location> {
    ALL.iter()
        .filter_map(|kind| {
            let in_use = dir(*kind).ok()?;
            let modern = modern(*kind).unwrap_or_else(|| in_use.clone());
            let legacy = legacy(*kind).is_some_and(|dir| in_use == dir);
            Some(Location {
                kind: *kind,
                in_use,
                modern,
                legacy,
            })
        })
        .collect()
}

/// Whether any kind of file is still being read from `~/.xencode`.
pub fn legacy_in_use() -> bool {
    locations().iter().any(|place| place.legacy)
}

/// What `$XCODE_CONFIG_DIR` names, when it names anything: the single directory
/// every kind of file is being read from. A listing has to say this, because
/// neither the XDG locations nor `~/.xencode` are in use while it is set, and a
/// migration of the real directories is refused.
pub fn override_root() -> Option<PathBuf> {
    overridden(Files::Settings)
}

/// The records of what happened, by the name each has in the old single
/// directory. Everything else in `~/.xencode` is sorted as a setting, which is
/// where the settings directory's table in the module documentation lists it:
/// an entry a newer xencode has added goes to settings rather than being left
/// behind in a directory the migration is trying to remove.
const STATE_ENTRIES: &[&str] = &[
    "conversation_memory.json",
    "colab.json",
    "audit.jsonl",
    "last_panic.log",
    "llamaserver.pid",
];

/// The kinds that had their own subdirectory in the old layout, with the name
/// each had there. These are the entries a migration can move whole, because
/// the new location for the kind is that same subdirectory under a new base.
const WHOLE_DIRECTORIES: [(Files, &str); 2] = [(Files::Cache, "cache"), (Files::Data, "models")];

/// Which kind of file an entry in the old directory is, by name.
///
/// `config.json.bak.<time>` is not a name anyone has to recognise: a backup of
/// the settings file is a setting, and that is where an unlisted name goes.
pub fn category_of(name: &str) -> Files {
    if STATE_ENTRIES.contains(&name) {
        Files::State
    } else {
        Files::Settings
    }
}

/// One file or directory the migration relocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Moved {
    /// Where it was.
    pub from: PathBuf,
    /// Where it is now.
    pub to: PathBuf,
    /// Which kind of file it was sorted as.
    pub kind: Files,
}

/// One entry the migration left where it was, with the reason in the terms a
/// person can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Left {
    /// The entry that stayed.
    pub path: PathBuf,
    /// Why it stayed.
    pub reason: String,
}

/// What a migration did. Every field is printed: a move that relocated a
/// keypair is not something to do quietly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Migration {
    /// The entries that moved.
    pub moved: Vec<Moved>,
    /// The entries that stayed, and why.
    pub left: Vec<Left>,
    /// The old directory the migration worked from.
    pub old_directory: PathBuf,
    /// Whether that directory is gone. It is only removed when it is empty, so
    /// anything [`Left`] describes kept it too.
    pub old_directory_removed: bool,
}

impl Migration {
    /// Whether there was nothing to do: no old directory, or an empty one.
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty() && self.left.is_empty()
    }

    /// Record an entry that stayed where it was, with what to tell the person.
    fn leave(&mut self, path: &Path, reason: impl Into<String>) {
        self.left.push(Left {
            path: path.to_path_buf(),
            reason: reason.into(),
        });
    }
}

/// Why a migration could not be attempted at all, as opposed to an entry that
/// was refused, which the report carries.
#[derive(Debug)]
pub enum MigrationError {
    /// `$XCODE_CONFIG_DIR` points this run at another tree, so it is not reading
    /// the directories a migration would move.
    Overridden,
    /// There is no home directory to ask about.
    NoHomeDir,
    /// The old directory could not be listed.
    Io(std::io::Error),
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overridden => write!(
                f,
                "$XCODE_CONFIG_DIR is set, so this run reads its files from there rather than \
                 from ~/.xencode — unset it to move the real directories"
            ),
            Self::NoHomeDir => write!(f, "could not determine home directory"),
            Self::Io(source) => write!(f, "could not read the old xencode directory: {source}"),
        }
    }
}

impl std::error::Error for MigrationError {}

/// Move a person's files from `~/.xencode` to the four directories they belong
/// in, and report what happened.
///
/// Nothing here is automatic. Changing where the paths point, and leaving the
/// files, is the failure this module exists to prevent: the settings, the
/// keypair and the conversation history would still be on disk and xencode would
/// stop reading any of them. So [`dir`] keeps reading the old directory until
/// something is written at the new one, and the move is a decision the person
/// takes with `xencode migrate`.
///
/// The rules that make a partial run survivable: an entry is never overwritten,
/// because two files with the same name in the same category means a newer
/// xencode already wrote there and guessing which one to keep is not this
/// command's call; the old directory is removed only when it is empty; and a
/// directory that will not move is reported as it is, not silently skipped.
pub fn migrate() -> Result<Migration, MigrationError> {
    migration(false)
}

/// The same walk, reporting what `xencode migrate` would do without touching
/// anything. A person can read this before agreeing to the move.
pub fn plan_migration() -> Result<Migration, MigrationError> {
    migration(true)
}

fn migration(dry_run: bool) -> Result<Migration, MigrationError> {
    if overridden(Files::Settings).is_some() {
        return Err(MigrationError::Overridden);
    }
    let home = dirs::home_dir().ok_or(MigrationError::NoHomeDir)?;
    let old = home.join(LEGACY_DIR);
    let mut report = Migration {
        old_directory: old.clone(),
        ..Default::default()
    };
    if !old.exists() {
        return Ok(report);
    }
    let entries = std::fs::read_dir(&old).map_err(MigrationError::Io)?;
    for entry in entries {
        let entry = entry.map_err(MigrationError::Io)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let from = entry.path();
        // The cache and the weights each had their own subdirectory in the old
        // layout, and the new location for that kind *is* that subdirectory
        // under a new base, so the whole directory moves when nothing is in the
        // way of it. If something already lives at the new location, move what
        // is inside rather than refusing a directory whose every entry is still
        // wanted.
        let whole = WHOLE_DIRECTORIES
            .iter()
            .find(|(_, sub)| *sub == name.as_ref())
            .map(|(kind, _)| *kind);
        let kind = whole.unwrap_or_else(|| category_of(&name));
        let Some(target) = modern(kind) else {
            report.leave(&from, "this platform has no directory to move it to");
            continue;
        };
        if whole.is_some() {
            if target.exists() {
                move_contents(&from, &target, kind, dry_run, &mut report);
            } else {
                move_entry(&from, &target, &old, kind, dry_run, &mut report);
            }
        } else {
            move_entry(
                &from,
                &target.join(name.as_ref()),
                &old,
                kind,
                dry_run,
                &mut report,
            );
        }
    }
    if dry_run {
        // Nothing was emptied, because nothing was moved, so the old directory
        // is reported as staying.
        return Ok(report);
    }
    // The two subdirectories are now empty, or still hold what was refused;
    // remove only what is genuinely empty.
    for (_, sub) in WHOLE_DIRECTORIES {
        let dir = old.join(sub);
        if dir.is_dir() && is_empty_dir(&dir) {
            let _ = std::fs::remove_dir(&dir);
        }
    }
    if is_empty_dir(&old) && std::fs::remove_dir(&old).is_ok() {
        report.old_directory_removed = true;
    }
    Ok(report)
}

/// Move one entry, refusing it if the destination already holds something.
///
/// `mode_from` is the directory this entry came from, whose permissions the
/// destination's parents are given so a settings tree that was `0700` does not
/// become group-readable on the way.
fn move_entry(
    from: &Path,
    to: &Path,
    mode_from: &Path,
    kind: Files,
    dry_run: bool,
    report: &mut Migration,
) {
    if to.exists() {
        report.leave(
            from,
            format!(
                "{} is already there and nothing is overwritten: xencode reads the file that was \
                 already at the destination, and this one stays where it is",
                to.display()
            ),
        );
        return;
    }
    if dry_run {
        report.moved.push(Moved {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            kind,
        });
        return;
    }
    if let Some(parent) = to.parent() {
        if let Err(source) = prepare_dir(parent, mode_from) {
            report.leave(
                from,
                format!("could not create {}: {source}", parent.display()),
            );
            return;
        }
    }
    if std::fs::rename(from, to).is_ok() {
        report.moved.push(Moved {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            kind,
        });
        return;
    }
    // A different filesystem — a separate home partition, or a cache directory
    // on another mount — is the one case a rename cannot answer, so copy it over
    // and only then remove the original.
    match copy_all(from, to).and_then(|()| remove(from)) {
        Ok(()) => report.moved.push(Moved {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            kind,
        }),
        Err(source) => {
            // The copy that got this far created part of the destination. A new
            // location that exists is the one thing that makes xencode stop
            // reading the old directory, so a half-written copy would orphan
            // every entry that has not moved yet; take it back down.
            let _ = remove(to);
            report.leave(
                from,
                format!("could not be copied to {}: {source}", to.display()),
            );
        }
    }
}

/// Move the children of one directory into another, then leave the emptied
/// source for the caller to remove.
fn move_contents(from: &Path, to: &Path, kind: Files, dry_run: bool, report: &mut Migration) {
    let entries = match std::fs::read_dir(from) {
        Ok(entries) => entries,
        Err(source) => {
            report.leave(from, format!("could not be read: {source}"));
            return;
        }
    };
    for entry in entries {
        match entry {
            Ok(entry) => {
                let target = to.join(entry.file_name());
                move_entry(&entry.path(), &target, from, kind, dry_run, report);
            }
            Err(source) => report.leave(
                from,
                format!("its contents could not all be listed: {source}"),
            ),
        }
    }
}

/// Create a directory, giving it the mode the directory it came from has — but
/// never an existing directory a different one, since a destination that is
/// already there may be deliberately tighter than the old layout was.
fn prepare_dir(target: &Path, mode_from: &Path) -> std::io::Result<()> {
    if target.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(target)?;
    copy_mode(target, mode_from)
}

#[cfg(unix)]
fn copy_mode(target: &Path, mode_from: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(mode_from)?.permissions().mode();
    std::fs::set_permissions(target, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn copy_mode(_target: &Path, _mode_from: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Copy a file or a whole directory. A symbolic link is followed and what it
/// points at is copied, which loses the link; nothing xencode writes is a link,
/// and refusing to move a real file because of one would be worse.
fn copy_all(from: &Path, to: &Path) -> std::io::Result<()> {
    if from.is_dir() {
        std::fs::create_dir_all(to)?;
        copy_mode(to, from)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            copy_all(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        std::fs::copy(from, to).map(|_| ())
    }
}

fn remove(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

/// Whether a directory has no entries in it. A directory that cannot be read is
/// not empty, and the caller leaves it alone.
fn is_empty_dir(path: &Path) -> bool {
    match std::fs::read_dir(path) {
        Ok(mut entries) => entries.next().is_none(),
        Err(_) => false,
    }
}

/// The environment is process-global, so every test in this binary that changes
/// `HOME`, an `XDG_*_HOME` or `XCODE_CONFIG_DIR` holds this first — including the
/// one in [`crate::config`], which is why it lives here rather than in a test
/// module.
#[cfg(test)]
pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Every test here fakes a home by setting `HOME` and the `XDG_*_HOME`
// variables. Windows resolves these directories through the system instead, so
// the fake would not hold there: the migration tests would read and move the
// real profile's files. They run on Unix only.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::path::Path;

    /// Point `HOME` and every XDG base at one empty directory, so a test sees a
    /// home that belongs to nobody else and cannot read the real one's files.
    fn with_fake_home<T>(body: impl FnOnce(&Path) -> T) -> T {
        let _guard = env_lock();
        let home = std::env::temp_dir().join(format!(
            "xencode-paths-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        let previous = std::env::var_os("HOME");
        let previous_xdg = [
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
        ]
        .map(|name| (name, std::env::var_os(name)));
        let previous_override = std::env::var_os("XCODE_CONFIG_DIR");
        std::env::set_var("HOME", &home);
        for (name, _) in &previous_xdg {
            std::env::set_var(name, home.join(name.to_lowercase()));
        }
        let body_result = body(&home);
        // Restore before anything can be dropped on the way out of a failed
        // assert, so one bad test cannot poison every later one.
        match previous {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        for (name, value) in previous_xdg {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        match previous_override {
            Some(value) => std::env::set_var("XCODE_CONFIG_DIR", value),
            None => std::env::remove_var("XCODE_CONFIG_DIR"),
        }
        let _ = std::fs::remove_dir_all(&home);
        body_result
    }

    /// A home with no `~/.xencode` in it is a new installation: every kind goes
    /// to its own XDG directory, and none of them is created by asking.
    #[test]
    fn a_fresh_home_resolves_every_kind_to_its_own_directory() {
        with_fake_home(|home| {
            assert!(!legacy_in_use());
            assert_eq!(
                settings_dir().unwrap(),
                home.join("xdg_config_home").join("xencode")
            );
            assert_eq!(
                state_dir().unwrap(),
                home.join("xdg_state_home").join("xencode")
            );
            assert_eq!(
                cache_dir().unwrap(),
                home.join("xdg_cache_home").join("xencode")
            );
            assert_eq!(
                data_dir().unwrap(),
                home.join("xdg_data_home").join("xencode")
            );
            for dir in [
                settings_dir().unwrap(),
                state_dir().unwrap(),
                cache_dir().unwrap(),
                data_dir().unwrap(),
            ] {
                assert!(!dir.exists(), "asking must not create {dir:?}");
            }
        });
    }

    /// The trap this module exists to avoid: with `~/.xencode` still full and
    /// the new locations empty, reading must keep finding the real files rather
    /// than starting an untouched life somewhere else.
    #[test]
    fn a_person_who_has_never_migrated_keeps_reading_their_old_directory() {
        with_fake_home(|home| {
            std::fs::create_dir_all(home.join(".xencode/cache")).unwrap();
            std::fs::write(home.join(".xencode/config.json"), "{}").unwrap();
            assert!(legacy_in_use());
            assert_eq!(settings_dir().unwrap(), home.join(".xencode"));
            assert_eq!(state_dir().unwrap(), home.join(".xencode"));
            assert_eq!(cache_dir().unwrap(), home.join(".xencode/cache"));
            // Nothing was ever written to the old models directory, so there is
            // nothing to keep reading from it: only a kind that has files on the
            // old layout stays there.
            assert_eq!(data_dir().unwrap(), home.join("xdg_data_home/xencode"));
            std::fs::create_dir_all(home.join(".xencode/models")).unwrap();
            assert_eq!(data_dir().unwrap(), home.join(".xencode/models"));
        });
    }

    /// Once something has been written at the new location it wins, so a
    /// migration that left one file behind cannot pull the paths back to the
    /// old directory and split the same kind of file over two places.
    #[test]
    fn a_new_directory_that_exists_wins_over_the_old_one() {
        with_fake_home(|home| {
            std::fs::create_dir_all(home.join(".xencode")).unwrap();
            std::fs::create_dir_all(home.join("xdg_config_home/xencode")).unwrap();
            assert_eq!(
                settings_dir().unwrap(),
                home.join("xdg_config_home/xencode")
            );
            // Only this kind moved, so only this kind reads from the new place.
            assert_eq!(state_dir().unwrap(), home.join(".xencode"));
        });
    }

    /// The override is what lets a test run against a real home directory, so it
    /// has to win over both the XDG bases and `~/.xencode` — and it has to keep
    /// every kind inside one tree, because an override that moved only the
    /// settings file would leave a cache write in the person's real directory.
    #[test]
    fn the_override_wins_and_keeps_every_kind_in_one_tree() {
        with_fake_home(|home| {
            std::fs::create_dir_all(home.join(".xencode")).unwrap();
            let root = home.join("portable");
            std::env::set_var("XCODE_CONFIG_DIR", &root);
            assert_eq!(settings_dir().unwrap(), root);
            assert_eq!(state_dir().unwrap(), root);
            assert_eq!(cache_dir().unwrap(), root.join("cache"));
            assert_eq!(data_dir().unwrap(), root.join("models"));
            // An override must also outrank a new location that already exists.
            std::fs::create_dir_all(home.join("xdg_config_home/xencode")).unwrap();
            assert_eq!(settings_dir().unwrap(), root);
            // An empty value is not an override.
            std::env::set_var("XCODE_CONFIG_DIR", "");
            assert_eq!(
                settings_dir().unwrap(),
                home.join("xdg_config_home").join("xencode")
            );
            // A relative name is still an override, and it resolves to an
            // absolute directory — every prefix guard compares against one.
            std::env::set_var("XCODE_CONFIG_DIR", "relative/xencode");
            assert_eq!(
                settings_dir().unwrap(),
                std::env::current_dir().unwrap().join("relative/xencode")
            );
            assert!(settings_dir().unwrap().is_absolute());
        });
    }

    /// The guard a project-writing tool asks. A path in any of the four
    /// directories is xencode's own, wherever it happens to resolve on this
    /// machine; a path inside the project is not.
    #[test]
    fn the_write_guard_covers_every_directory_not_just_the_settings_one() {
        with_fake_home(|home| {
            std::fs::create_dir_all(home.join(".xencode/cache")).unwrap();
            std::fs::create_dir_all(home.join(".xencode/models")).unwrap();
            std::fs::create_dir_all(home.join("xdg_state_home/xencode")).unwrap();
            assert!(is_internal(&home.join(".xencode/config.json")));
            assert!(is_internal(&home.join(".xencode/cache/entry.json")));
            assert!(is_internal(&home.join(".xencode/models/big.gguf")));
            assert!(is_internal(
                &home.join("xdg_state_home/xencode/audit.jsonl")
            ));
            assert!(!is_internal(
                &home.join("project/.xencode/cache/metrics.jsonl")
            ));
            assert!(!is_internal(Path::new("/tmp/elsewhere/config.json")));
            // A directory itself counts, not only a file inside it.
            assert!(is_internal(&settings_dir().unwrap()));
        });
    }

    /// A listing is what `xencode paths` prints and what the migration report is
    /// built from, so it has to name the kind, the place in use, and whether the
    /// old directory is still being read.
    #[test]
    fn the_listing_says_which_kinds_are_still_on_the_old_layout() {
        with_fake_home(|home| {
            std::fs::create_dir_all(home.join(".xencode/cache")).unwrap();
            std::fs::create_dir_all(home.join("xdg_data_home/xencode")).unwrap();
            let places = locations();
            assert_eq!(places.len(), 4);
            for place in &places {
                assert_eq!(place.in_use, dir(place.kind).unwrap());
            }
            let on_old_layout = |kind: Files| {
                places
                    .iter()
                    .find(|place| place.kind == kind)
                    .unwrap()
                    .legacy
            };
            assert!(on_old_layout(Files::Settings));
            assert!(on_old_layout(Files::State));
            assert!(on_old_layout(Files::Cache));
            assert!(
                !on_old_layout(Files::Data),
                "its new directory exists, so it is already read from there"
            );
            assert_eq!(
                places
                    .iter()
                    .find(|place| place.kind == Files::Settings)
                    .unwrap()
                    .modern,
                home.join("xdg_config_home/xencode")
            );
        });
    }

    /// A preview is what makes the real move safe to agree to: it names the same
    /// entries and the same destinations, and leaves the disk exactly as it was.
    #[test]
    fn a_preview_reports_the_same_moves_without_touching_anything() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            let planned = plan_migration().unwrap();
            assert_eq!(planned.moved.len(), 14, "{:?}", planned.moved);
            assert!(planned.left.is_empty(), "{:?}", planned.left);
            assert!(
                !planned.old_directory_removed,
                "a preview never removes the directory it is describing"
            );
            assert!(
                old.join("config.json").exists() && old.join("colab_ed25519").exists(),
                "the files are still where they were"
            );
            assert!(!home.join("xdg_config_home/xencode").exists());
            assert!(!home.join("xdg_state_home/xencode").exists());
            assert!(legacy_in_use(), "and xencode still reads the old layout");
            // Running it for real afterwards produces the same report, so the
            // preview is not a separate piece of guessing about the layout.
            let done = migrate().unwrap();
            assert_eq!(
                planned
                    .moved
                    .iter()
                    .map(|moved| moved.to.clone())
                    .collect::<Vec<_>>(),
                done.moved
                    .iter()
                    .map(|moved| moved.to.clone())
                    .collect::<Vec<_>>()
            );
        });
    }

    fn write(path: &Path, contents: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    /// A `~/.xencode` holding one of everything an installation builds up: the
    /// settings and their backups, the keypair, the records of past sessions,
    /// the response cache with a corpus inside it, and a downloaded model.
    fn a_legacy_installation(home: &Path) -> PathBuf {
        let old = home.join(".xencode");
        write(&old.join("config.json"), "{\"version\":1}");
        write(&old.join("config.json.bak.1700000000"), "{\"version\":0}");
        write(&old.join("layout.json"), "{\"pane\":2}");
        write(&old.join("model_advice.json"), "[]");
        write(&old.join("skills/deploy/SKILL.md"), "name: deploy");
        write(&old.join("colab_ed25519"), "private");
        write(&old.join("colab_ed25519.pub"), "public");
        write(&old.join("conversation_memory.json"), "{\"turns\":9}");
        write(&old.join("colab.json"), "{\"runtime\":\"x\"}");
        write(&old.join("audit.jsonl"), "{\"event\":\"a\"}\n");
        write(&old.join("last_panic.log"), "panicked at main.rs:1");
        write(&old.join("llamaserver.pid"), "4242");
        write(&old.join("cache/response.json"), "{\"cached\":true}");
        write(&old.join("cache/advisories/lookup.json"), "[]");
        write(&old.join("models/qwen3-4b.gguf"), "GGUF");
        old
    }

    /// Where a migration says one named entry ended up, by the name it has in
    /// the old directory.
    fn destination_of<'a>(report: &'a Migration, name: &str) -> Option<&'a Path> {
        let name = std::ffi::OsStr::new(name);
        report
            .moved
            .iter()
            .find(|moved| moved.from.file_name() == Some(name))
            .map(|moved| moved.to.as_path())
    }

    /// The whole old layout moves to the four directories, with its contents
    /// intact, and the directory itself is removed once nothing is in it.
    #[test]
    fn a_migration_moves_every_kind_of_file_to_where_it_belongs() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            let report = migrate().expect("the migration runs");
            assert!(report.left.is_empty(), "{:?}", report.left);
            assert_eq!(report.moved.len(), 14, "{:?}", report.moved);
            assert!(report.old_directory_removed);

            let settings = home.join("xdg_config_home/xencode");
            let state = home.join("xdg_state_home/xencode");
            let cache = home.join("xdg_cache_home/xencode");
            let data = home.join("xdg_data_home/xencode");
            for (path, contents) in [
                (settings.join("config.json"), "{\"version\":1}"),
                (
                    settings.join("config.json.bak.1700000000"),
                    "{\"version\":0}",
                ),
                (settings.join("layout.json"), "{\"pane\":2}"),
                (settings.join("model_advice.json"), "[]"),
                (settings.join("skills/deploy/SKILL.md"), "name: deploy"),
                (settings.join("colab_ed25519"), "private"),
                (settings.join("colab_ed25519.pub"), "public"),
                (state.join("conversation_memory.json"), "{\"turns\":9}"),
                (state.join("colab.json"), "{\"runtime\":\"x\"}"),
                (state.join("audit.jsonl"), "{\"event\":\"a\"}\n"),
                (state.join("last_panic.log"), "panicked at main.rs:1"),
                (state.join("llamaserver.pid"), "4242"),
                (cache.join("response.json"), "{\"cached\":true}"),
                (cache.join("advisories/lookup.json"), "[]"),
                (data.join("qwen3-4b.gguf"), "GGUF"),
            ] {
                assert_eq!(
                    std::fs::read_to_string(&path)
                        .unwrap_or_else(|error| panic!("{}: {error}", path.display())),
                    contents,
                );
            }
            assert!(!old.exists(), "the old directory is gone once emptied");
            assert!(!legacy_in_use());
            assert_eq!(settings_dir().unwrap(), settings);
            assert_eq!(state_dir().unwrap(), state);
            assert_eq!(cache_dir().unwrap(), cache);
            assert_eq!(data_dir().unwrap(), data);
            // The report says which kind each entry was sorted as, which is how
            // a person checks that their history went to state and their
            // keypair to settings rather than both somewhere plausible.
            assert_eq!(
                destination_of(&report, "conversation_memory.json").unwrap(),
                state.join("conversation_memory.json")
            );
            assert_eq!(
                destination_of(&report, "models"),
                Some(data.as_path()),
                "the weights move as one directory, onto the data directory itself"
            );
            assert_eq!(
                destination_of(&report, "cache"),
                Some(cache.as_path()),
                "and so does the response cache"
            );
            assert_eq!(
                report
                    .moved
                    .iter()
                    .find(|moved| moved.to.ends_with("colab_ed25519"))
                    .unwrap()
                    .kind,
                Files::Settings
            );
            assert_eq!(
                report
                    .moved
                    .iter()
                    .find(|moved| moved.to == data)
                    .unwrap()
                    .kind,
                Files::Data
            );
        });
    }

    /// An entry whose destination already holds something is refused, named, and
    /// left exactly where it was — a migration that guessed which of two
    /// `config.json` files to keep would destroy one of them.
    #[test]
    fn a_migration_never_overwrites_what_is_already_at_the_new_location() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            write(
                &home.join("xdg_config_home/xencode/config.json"),
                "{\"new\":true}",
            );
            let report = migrate().unwrap();
            assert_eq!(
                report.left.len(),
                1,
                "only the settings file conflicts: {:?}",
                report.left
            );
            assert!(report.left[0].path.ends_with("config.json"));
            assert!(report.left[0].reason.contains("already there"));
            assert_eq!(
                std::fs::read_to_string(home.join("xdg_config_home/xencode/config.json")).unwrap(),
                "{\"new\":true}",
                "the file that was already at the destination is untouched"
            );
            assert!(
                old.join("config.json").exists(),
                "and the one that could not move is still on disk"
            );
            assert_eq!(
                std::fs::read_to_string(home.join("xdg_config_home/xencode/layout.json")).unwrap(),
                "{\"pane\":2}",
                "everything else still moves"
            );
            assert!(
                !report.old_directory_removed,
                "the old directory is kept while it holds anything"
            );
            assert!(old.is_dir());
        });
    }

    /// The response cache and the downloaded weights already had a directory of
    /// their own, so when the new one is in use too their contents merge rather
    /// than the whole entry being refused.
    #[test]
    fn a_migration_merges_a_cache_directory_that_is_already_in_use() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            write(
                &home.join("xdg_cache_home/xencode/fresh.json"),
                "{\"cached\":false}",
            );
            let report = migrate().unwrap();
            let cache = home.join("xdg_cache_home/xencode");
            assert!(report.left.is_empty(), "{:?}", report.left);
            assert_eq!(
                std::fs::read_to_string(cache.join("fresh.json")).unwrap(),
                "{\"cached\":false}"
            );
            assert_eq!(
                std::fs::read_to_string(cache.join("response.json")).unwrap(),
                "{\"cached\":true}"
            );
            assert_eq!(
                std::fs::read_to_string(cache.join("advisories/lookup.json")).unwrap(),
                "[]"
            );
            assert!(!old.join("cache").exists(), "the emptied subdirectory goes");
        });
    }

    /// A conflicting name *inside* a directory being merged is refused on its
    /// own, and everything else from that directory still arrives.
    #[test]
    fn a_migration_refuses_one_conflicting_entry_inside_a_merged_directory() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            write(
                &home.join("xdg_cache_home/xencode/response.json"),
                "{\"cached\":false}",
            );
            let report = migrate().unwrap();
            assert_eq!(report.left.len(), 1, "{:?}", report.left);
            assert!(report.left[0].path.ends_with("cache/response.json"));
            assert_eq!(
                std::fs::read_to_string(home.join("xdg_cache_home/xencode/response.json")).unwrap(),
                "{\"cached\":false}"
            );
            assert!(home
                .join("xdg_cache_home/xencode/advisories/lookup.json")
                .exists());
            assert!(
                !report.old_directory_removed,
                "the refused file keeps the old directory alive"
            );
            assert!(old.join("cache/response.json").exists());
        });
    }

    /// With `$XCODE_CONFIG_DIR` set, this run is not reading `~/.xencode` at
    /// all, so moving it would empty a directory nothing here depends on.
    #[test]
    fn a_migration_refuses_to_run_while_the_environment_points_at_another_tree() {
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            std::env::set_var("XCODE_CONFIG_DIR", home.join("portable"));
            let error = migrate().expect_err("refused");
            assert!(matches!(error, MigrationError::Overridden), "{error}");
            assert!(
                old.join("config.json").exists(),
                "and nothing was touched: {}",
                old.display()
            );
            assert!(!home.join("xdg_config_home/xencode").exists());
        });
    }

    /// A new installation has nothing to migrate, and asking does not create the
    /// directories — the first real write does.
    #[test]
    fn a_migration_with_no_old_directory_reports_nothing_to_do() {
        with_fake_home(|home| {
            let report = migrate().unwrap();
            assert!(report.is_empty());
            assert!(!report.old_directory_removed);
            assert!(report.moved.is_empty());
            assert!(report.left.is_empty());
            assert!(!home.join("xdg_config_home/xencode").exists());
        });
    }

    /// A file that a person has kept `0600` because it holds a credential is
    /// still `0600` at its new path, and the directory it moved into takes the
    /// mode the old directory had rather than the process umask's guess.
    #[cfg(unix)]
    #[test]
    fn a_migration_keeps_the_permissions_a_file_already_had() {
        use std::os::unix::fs::PermissionsExt;
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            std::fs::set_permissions(
                old.join("config.json"),
                std::fs::Permissions::from_mode(0o600),
            )
            .unwrap();
            std::fs::set_permissions(old, std::fs::Permissions::from_mode(0o700)).unwrap();
            let report = migrate().unwrap();
            assert!(report.left.is_empty(), "{:?}", report.left);
            assert_eq!(
                std::fs::metadata(home.join("xdg_config_home/xencode/config.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(home.join("xdg_config_home/xencode"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700,
                "the settings directory is as private as the one it came from"
            );
        });
    }

    /// The destination's mode is never relaxed onto a directory that is already
    /// there: an old `0755` must not make a new `0700` group-readable.
    #[cfg(unix)]
    #[test]
    fn a_migration_does_not_loosen_a_new_directory_that_is_already_private() {
        use std::os::unix::fs::PermissionsExt;
        with_fake_home(|home| {
            let old = a_legacy_installation(home);
            std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o755)).unwrap();
            let new = home.join("xdg_config_home/xencode");
            std::fs::create_dir_all(&new).unwrap();
            std::fs::set_permissions(&new, std::fs::Permissions::from_mode(0o700)).unwrap();
            let report = migrate().unwrap();
            assert_eq!(
                std::fs::metadata(&new).unwrap().permissions().mode() & 0o777,
                0o700,
                "{:?}",
                report.left
            );
        });
    }

    /// The fallback a rename cannot answer — a destination on another
    /// filesystem — copies a whole tree, permissions and nested directories
    /// included, so that removing the source afterwards loses nothing. Which
    /// entries need this path is a property of the machine, so the routing into
    /// it is shown by a live run, not asserted here.
    #[test]
    fn the_copy_fallback_reproduces_a_directory_tree() {
        let scratch = std::env::temp_dir().join(format!(
            "xencode-copy-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        write(&scratch.join("src/a.json"), "{\"a\":1}");
        write(&scratch.join("src/nested/b.json"), "{\"b\":2}");
        write(&scratch.join("src/nested/deeper/c.json"), "{\"c\":3}");
        copy_all(&scratch.join("src"), &scratch.join("dst")).unwrap();
        for relative in ["a.json", "nested/b.json", "nested/deeper/c.json"] {
            assert_eq!(
                std::fs::read_to_string(scratch.join("dst").join(relative)).unwrap(),
                std::fs::read_to_string(scratch.join("src").join(relative)).unwrap(),
                "{relative}"
            );
        }
        assert!(scratch.join("dst/nested/deeper").is_dir());
        remove(&scratch.join("src")).unwrap();
        assert!(!scratch.join("src").exists());
        assert!(scratch.join("dst/a.json").exists(), "the copy survives");
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
