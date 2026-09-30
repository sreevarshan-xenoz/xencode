//! Installing a plugin from a git repository: clone, pin the commit, verify the
//! manifest, disclose what it declares, and only then copy it into the plugin
//! directory.
//!
//! The order is the point. A plugin whose manifest adds a `prompt_prefix` puts
//! text ahead of the agent's system prompt on every turn of every session, so
//! `install` shows that text before it reaches the directory the loader scans,
//! and `update` refuses to apply a change to what a plugin contributes until the
//! difference has been printed and acknowledged.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use similar::TextDiff;

use crate::manifest::{PluginManifest, KNOWN_PERMISSIONS};
use crate::registry::{is_safe_plugin_name, PluginRegistry};
use crate::runtime::{missing_permission_note, quote_list};

/// Recorded in each installed plugin directory: where it came from and which
/// commit its bytes are. `update` reads it to know what to fetch.
pub const SOURCE_FILE: &str = ".xencode-source.json";

/// The manifest file names the loader looks for, in order.
pub const MANIFEST_NAMES: &[&str] = &["plugin.json", "manifest.json"];

/// A `prompt_prefix` is a few paragraphs at most. Beyond this the diff is a file
/// dump, so it is cut with a note rather than flooding the terminal.
const DIFF_MAX_LINES: usize = 400;

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("{0}")]
    Git(String),
    #[error("{0}")]
    Install(String),
}

/// Where an installed plugin came from.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginSource {
    /// The repository it was cloned from. Absent for a copy from a local path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The full commit the installed files are from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// The branch, tag or commit asked for, which `update` follows again.
    #[serde(default, rename = "ref", skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    pub installed_at: String,
}

impl PluginSource {
    /// `file:///srv/plugins/guard.git @ 3f2a1c9 (from main)` — how `plugin list`
    /// and `/plugin` name the origin. A copy from a path has no line, and a
    /// plugin pinned by its own SHA is not named twice.
    pub fn summary(&self) -> Option<String> {
        let url = self.url.as_deref()?;
        let commit = self.commit.as_deref().map(short_commit);
        match (commit, self.git_ref.as_deref()) {
            (Some(commit), Some(rev)) if self.commit.as_deref() != Some(rev) => {
                Some(format!("{url} @ {commit} (from {rev})"))
            }
            (Some(commit), _) => Some(format!("{url} @ {commit}")),
            _ => Some(url.to_string()),
        }
    }
}

/// Read the origin record of an installed plugin, if it has one.
pub fn read_source(plugins_dir: &Path, name: &str) -> Option<PluginSource> {
    let dir = PluginRegistry::new(plugins_dir.to_path_buf()).plugin_path(name)?;
    let text = std::fs::read_to_string(dir.join(SOURCE_FILE)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_source(dir: &Path, source: &PluginSource) -> Result<(), InstallError> {
    let text = serde_json::to_string_pretty(source)
        .map_err(|e| InstallError::Install(format!("could not write the source record: {e}")))?;
    std::fs::write(dir.join(SOURCE_FILE), text + "\n")
        .map_err(|e| InstallError::Install(format!("could not record the source: {e}")))
}

pub fn short_commit(commit: &str) -> String {
    commit.chars().take(7).collect()
}

/// True when the argument names a repository to clone rather than a directory or
/// file to copy: a transport URL, a `.git` suffix, or the `host:path` form.
pub fn is_git_source(source: &str) -> bool {
    source.contains("://")
        || source.ends_with(".git")
        || (source.contains('@') && source.contains(':'))
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// The subcommand a git argument list is invoking, for the error message: the
/// first entry that is neither `-C <dir>` nor another option.
fn git_subcommand<'b>(args: &[&'b str]) -> &'b str {
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "-C" => index += 2,
            arg if arg.starts_with('-') => index += 1,
            arg => return arg,
        }
    }
    ""
}

fn run_git(args: &[&str]) -> Result<String, InstallError> {
    let output = Command::new("git")
        .args(args)
        // Installing a plugin is a command someone types; it must never stop to
        // ask for a password with no way to answer it.
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|e| {
            InstallError::Git(format!(
                "could not run git: {e} — installing a plugin from a repository needs git on PATH"
            ))
        })?;
    if !output.status.success() {
        let first_stderr = String::from_utf8_lossy(&output.stderr)
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_string();
        return Err(InstallError::Git(format!(
            "git {} failed: {}",
            git_subcommand(args),
            first_stderr
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// A repository checkout that exists only while this value is alive.
#[derive(Debug)]
pub struct Staged {
    /// Held, never read: dropping it deletes the checkout.
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    checkout: PathBuf,
    commit: String,
}

impl Staged {
    pub fn path(&self) -> &Path {
        &self.checkout
    }

    /// The full commit the checkout is at.
    pub fn commit(&self) -> &str {
        &self.commit
    }
}

/// Clone `url` into a temporary directory and resolve the commit to install.
///
/// Without a rev, this is a shallow clone of the default branch. With one, the
/// clone is complete and detached at exactly that commit, because a shallow
/// checkout cannot reach an arbitrary branch, tag or SHA.
pub fn clone_at(url: &str, rev: Option<&str>) -> Result<Staged, InstallError> {
    if let Some(rev) = rev {
        if rev.trim().is_empty() || rev.starts_with('-') {
            return Err(InstallError::Install(format!(
                "not a usable branch, tag or commit: {rev:?}"
            )));
        }
    }
    let staged = tempfile::tempdir().map_err(|e| {
        InstallError::Install(format!("could not create a directory to clone into: {e}"))
    })?;
    let target = staged.path().join("checkout");
    let target_text = target.display().to_string();
    match rev {
        None => {
            run_git(&["clone", "--quiet", "--depth", "1", "--", url, &target_text])?;
        }
        Some(rev) => {
            run_git(&["clone", "--quiet", "--", url, &target_text])?;
            // Resolve first, then detach at the SHA: this rejects a rev that names a
            // path or a non-commit object instead of checking out something else.
            let sha = run_git(&[
                "-C",
                &target_text,
                "rev-parse",
                "--verify",
                &format!("{rev}^{{commit}}"),
            ])
            .map_err(|e| {
                InstallError::Git(format!(
                    "{url} has no branch, tag or commit named {rev:?} — {e}"
                ))
            })?;
            run_git(&["-C", &target_text, "checkout", "--quiet", "--detach", &sha])?;
        }
    }
    let commit = run_git(&["-C", &target_text, "rev-parse", "HEAD"])?;
    Ok(Staged {
        dir: staged,
        checkout: target,
        commit,
    })
}

/// The manifest file a plugin directory is loaded by, in the loader's order.
pub fn find_manifest(dir: &Path) -> Option<PathBuf> {
    MANIFEST_NAMES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn read_manifest(path: &Path) -> Result<PluginManifest, InstallError> {
    PluginManifest::from_file(path).map_err(|e| {
        InstallError::Install(format!(
            "{} is not a readable manifest: {e}",
            path.display()
        ))
    })
}

/// Reject a manifest this build would refuse to load, before it is installed.
///
/// These are the loader's own checks (`PluginRuntime::load`): the name must be one
/// path component inside the plugin directory, the declared xencode version must
/// accept this build, and every capability used must be asked for. Installing a
/// plugin that cannot load only moves the failure into `/plugin`.
fn verify(manifest: &PluginManifest, xencode_version: &str) -> Result<(), InstallError> {
    if manifest.name.trim().is_empty() {
        return Err(InstallError::Install(
            "the manifest declares no name, so there is nothing to install it as".to_string(),
        ));
    }
    if !is_safe_plugin_name(&manifest.name) {
        return Err(InstallError::Install(format!(
            "the manifest names itself {:?}, which is not one directory name inside the plugin \
             directory — a plugin name cannot contain a path",
            manifest.name
        )));
    }
    if !manifest.is_compatible_with(xencode_version) {
        return Err(InstallError::Install(format!(
            "'{}' needs xencode {} (declared {}), so this build would not load it",
            manifest.name, xencode_version, manifest.xencode_version
        )));
    }
    let undeclared = manifest.undeclared_permissions();
    if !undeclared.is_empty() {
        return Err(InstallError::Install(format!(
            "'{}' {} in its manifest, so this build would not load it",
            manifest.name,
            missing_permission_note(&undeclared)
        )));
    }
    let unknown = manifest.unknown_permissions();
    if !unknown.is_empty() {
        return Err(InstallError::Install(format!(
            "'{}' requests permission{} xencode does not recognise: {}; a plugin can declare only {}",
            manifest.name,
            if unknown.len() == 1 { "" } else { "s" },
            quote_list(&unknown),
            KNOWN_PERMISSIONS.join(", ")
        )));
    }
    Ok(())
}

/// What this manifest will do to every agent turn, in words, for the installer to
/// print before any of it is in place.
pub fn disclose(manifest: &PluginManifest) -> String {
    let mut out = String::from("What it declares:\n");
    out.push_str(&format!(
        "  permissions: {}\n",
        if manifest.permissions.is_empty() {
            "none".to_string()
        } else {
            manifest.permissions.join(", ")
        }
    ));
    let prefix = manifest.prompt_prefix.trim();
    if prefix.is_empty() {
        out.push_str("  prompt prefix: none\n");
    } else {
        out.push_str(&format!(
            "  prompt prefix: {} line(s), put ahead of the agent's system prompt on every turn:\n",
            prefix.lines().count()
        ));
        for line in prefix.lines() {
            out.push_str(&format!("    | {line}\n"));
        }
    }
    if manifest.hooks.is_empty() {
        out.push_str("  hooks: none\n");
    } else {
        out.push_str("  hooks (each one runs a shell command in the workspace):\n");
        for (phase, table) in [
            ("before", &manifest.hooks.before),
            ("after", &manifest.hooks.after),
        ] {
            for (tool, command) in table {
                out.push_str(&format!("    {phase} {tool} → {command}\n"));
            }
        }
    }
    out.push_str("Nothing else happens: this build loads no plugin code.");
    out
}

fn destination_for(plugins_dir: &Path, manifest: &PluginManifest) -> Result<PathBuf, InstallError> {
    PluginRegistry::new(plugins_dir.to_path_buf())
        .plugin_path(&manifest.name)
        .ok_or_else(|| {
            InstallError::Install(format!("'{}' is not a usable plugin name", manifest.name))
        })
}

fn refuse_overwrite(name: &str, destination: &Path) -> Result<(), InstallError> {
    if destination.exists() {
        return Err(InstallError::Install(format!(
            "Plugin '{name}' is already installed at {} — run `xencode plugin update {name}` to \
             fetch a newer commit, or `xencode plugin remove {name}` first.",
            destination.display()
        )));
    }
    Ok(())
}

/// An install that has been verified and disclosed but not yet copied.
#[derive(Debug)]
pub struct Pending {
    manifest: PluginManifest,
    source: PluginSource,
    destination: PathBuf,
    contents: Contents,
}

#[derive(Debug)]
enum Contents {
    /// A temporary checkout, dropped when the install is committed.
    Repo(Staged),
    /// A directory or manifest file already on disk, copied as-is.
    Local(PathBuf),
}

impl Pending {
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    pub fn version(&self) -> &str {
        &self.manifest.version
    }

    /// The directory this plugin would be installed into.
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// The commit being pinned, for a repository install.
    pub fn commit(&self) -> Option<&str> {
        self.source.commit.as_deref()
    }

    /// The text to show before anything is copied.
    pub fn declaration(&self) -> String {
        disclose(&self.manifest)
    }

    /// Copy the verified files into the plugin directory and record where they
    /// came from. Consumes the pending install, which drops the checkout.
    pub fn apply(self) -> Result<InstallOutcome, InstallError> {
        let parent = self.destination.parent().ok_or_else(|| {
            InstallError::Install("the plugin directory has no parent".to_string())
        })?;
        std::fs::create_dir_all(parent).map_err(|e| {
            InstallError::Install(format!("could not create {}: {e}", parent.display()))
        })?;
        let copied = match &self.contents {
            Contents::Repo(staged) => copy_tree(staged.path(), &self.destination),
            Contents::Local(path) if path.is_dir() => copy_tree(path, &self.destination),
            Contents::Local(file) => std::fs::create_dir_all(&self.destination)
                .and_then(|()| std::fs::copy(file, self.destination.join("plugin.json")))
                .map(|_| ()),
        };
        if let Err(e) = copied {
            let _ = std::fs::remove_dir_all(&self.destination);
            return Err(InstallError::Install(format!(
                "could not copy the plugin into {}: {e}",
                self.destination.display()
            )));
        }
        let source = PluginSource {
            installed_at: now(),
            ..self.source.clone()
        };
        write_source(&self.destination, &source)?;
        Ok(InstallOutcome {
            name: self.manifest.name.clone(),
            version: self.manifest.version.clone(),
            destination: self.destination.clone(),
            source,
        })
    }
}

/// What a completed install put where.
#[derive(Debug, Clone)]
pub struct InstallOutcome {
    pub name: String,
    pub version: String,
    pub destination: PathBuf,
    pub source: PluginSource,
}

/// Fetch a repository and verify its manifest without writing anything into the
/// plugin directory yet.
pub fn prepare_git(
    plugins_dir: &Path,
    url: &str,
    rev: Option<&str>,
    xencode_version: &str,
) -> Result<Pending, InstallError> {
    let staged = clone_at(url, rev)?;
    let manifest_path = find_manifest(staged.path()).ok_or_else(|| {
        InstallError::Install(format!(
            "no {} at the root of {url} at commit {} — that repository is not a plugin",
            MANIFEST_NAMES.join(" or "),
            short_commit(staged.commit())
        ))
    })?;
    let manifest = read_manifest(&manifest_path)?;
    verify(&manifest, xencode_version)?;
    let destination = destination_for(plugins_dir, &manifest)?;
    refuse_overwrite(&manifest.name, &destination)?;
    Ok(Pending {
        source: PluginSource {
            url: Some(url.to_string()),
            commit: Some(staged.commit().to_string()),
            git_ref: rev.map(str::to_string),
            installed_at: String::new(),
        },
        manifest,
        destination,
        contents: Contents::Repo(staged),
    })
}

/// Verify a local plugin directory or manifest file without copying it yet.
pub fn prepare_path(
    plugins_dir: &Path,
    path: &Path,
    xencode_version: &str,
) -> Result<Pending, InstallError> {
    if !path.exists() {
        return Err(InstallError::Install(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }
    let manifest_path = if path.is_dir() {
        find_manifest(path).ok_or_else(|| {
            InstallError::Install(format!(
                "no {} in {} — that directory is not a plugin",
                MANIFEST_NAMES.join(" or "),
                path.display()
            ))
        })?
    } else {
        path.to_path_buf()
    };
    let manifest = read_manifest(&manifest_path)?;
    verify(&manifest, xencode_version)?;
    let destination = destination_for(plugins_dir, &manifest)?;
    refuse_overwrite(&manifest.name, &destination)?;
    Ok(Pending {
        source: PluginSource::default(),
        manifest,
        destination,
        contents: Contents::Local(path.to_path_buf()),
    })
}

/// Copy a tree, leaving out the git metadata and any previous origin record:
/// those describe where the bytes came from, and that is recorded separately.
fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" || name == SOURCE_FILE {
            continue;
        }
        let dest = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

/// How two manifests differ in what they contribute, in words. A version or
/// description change is not among them: it changes nothing about a turn.
fn declaration_changes(old: &PluginManifest, new: &PluginManifest) -> Vec<String> {
    let mut changes = Vec::new();
    let (before, after) = (old.prompt_prefix.trim(), new.prompt_prefix.trim());
    if before != after {
        if before.is_empty() {
            changes.push(format!(
                "adds a prompt prefix of {} line(s)",
                after.lines().count()
            ));
        } else if after.is_empty() {
            changes.push("removes its prompt prefix".to_string());
        } else {
            changes.push(format!(
                "changes its prompt prefix ({} line(s) → {})",
                before.lines().count(),
                after.lines().count()
            ));
        }
    }
    for phase in ["before", "after"] {
        let (old_table, new_table) = match phase {
            "before" => (&old.hooks.before, &new.hooks.before),
            _ => (&old.hooks.after, &new.hooks.after),
        };
        for tool in union_keys(old_table, new_table) {
            match (old_table.get(&tool), new_table.get(&tool)) {
                (None, Some(command)) => {
                    changes.push(format!("adds a {phase} hook for {tool}: {command}"))
                }
                (Some(previous), None) => {
                    changes.push(format!("removes the {phase} hook for {tool}: {previous}"))
                }
                (Some(previous), Some(command)) if previous != command => changes.push(format!(
                    "changes the {phase} hook for {tool}: {previous} → {command}"
                )),
                _ => {}
            }
        }
    }
    let mut dropped: Vec<&String> = old
        .permissions
        .iter()
        .filter(|p| !new.permissions.contains(p))
        .collect();
    let mut asked: Vec<&String> = new
        .permissions
        .iter()
        .filter(|p| !old.permissions.contains(p))
        .collect();
    dropped.sort();
    asked.sort();
    for permission in dropped {
        changes.push(format!("gives up the {permission} permission"));
    }
    for permission in asked {
        changes.push(format!("asks for the {permission} permission"));
    }
    if old.version != new.version {
        changes.push(format!("version {} → {}", old.version, new.version));
    }
    changes
}

/// Whether a set of changes reaches the agent loop. Everything the function
/// above reports does — prompt text, hooks, permissions — except a version bump,
/// which renames the plugin without changing what it does.
fn reaches_agent(changes: &[String]) -> bool {
    changes.iter().any(|change| !change.starts_with("version "))
}

fn union_keys(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> Vec<String> {
    let mut keys: Vec<String> = a.keys().chain(b.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    keys
}

/// A unified diff between the installed manifest and the one in the repository.
fn manifest_diff(old_text: &str, new_text: &str, file: &str, to_commit: &str) -> String {
    let diff = TextDiff::from_lines(old_text, new_text)
        .unified_diff()
        .context_radius(3)
        .header(
            &format!("{file} (installed)"),
            &format!("{file} (at {})", short_commit(to_commit)),
        )
        .to_string();
    let lines: Vec<&str> = diff.lines().collect();
    if lines.len() > DIFF_MAX_LINES {
        let mut kept = lines[..DIFF_MAX_LINES].join("\n");
        kept.push_str(&format!(
            "\n… diff truncated ({} of {} lines shown)",
            DIFF_MAX_LINES,
            lines.len()
        ));
        return kept;
    }
    diff
}

/// An update that has been fetched and compared but not yet applied.
#[derive(Debug)]
pub struct UpdatePlan {
    pub name: String,
    pub destination: PathBuf,
    /// Where the installed copy came from.
    pub from: PluginSource,
    /// The commit being offered, and the ref it was fetched at.
    pub to_commit: String,
    pub to_ref: Option<String>,
    pub url: String,
    pub from_version: String,
    pub to_version: String,
    /// In words, what changes about the plugin's contributions.
    pub changes: Vec<String>,
    /// The manifest difference itself.
    pub diff: String,
    /// Whether anything that reaches the agent turn changed.
    pub reaches_agent: bool,
    /// Nothing new at all: same commit, same declarations.
    pub current: bool,
    staged: Staged,
}

impl UpdatePlan {
    /// Replace the installed copy with the fetched one and re-pin it.
    pub fn apply(self) -> Result<PluginSource, InstallError> {
        let backup = backup_path(&self.destination);
        std::fs::rename(&self.destination, &backup).map_err(|e| {
            InstallError::Install(format!(
                "could not move {} aside to replace it: {e}",
                self.destination.display()
            ))
        })?;
        if let Err(e) = copy_tree(self.staged.path(), &self.destination) {
            let _ = std::fs::rename(&backup, &self.destination);
            return Err(InstallError::Install(format!(
                "could not copy the update into {} (the previous version is still installed): {e}",
                self.destination.display()
            )));
        }
        let _ = std::fs::remove_dir_all(&backup);
        let source = PluginSource {
            url: Some(self.url.clone()),
            commit: Some(self.to_commit.clone()),
            git_ref: self.to_ref.clone(),
            installed_at: now(),
        };
        write_source(&self.destination, &source)?;
        Ok(source)
    }
}

/// A dot-prefixed sibling, so the loader's directory scan never sees it as a
/// plugin even if this process is killed before the swap finishes.
fn backup_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "plugin".to_string());
    destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{name}.old-{}", std::process::id()))
}

/// Fetch what the plugin's own source record points at and compare it with what
/// is installed. Nothing is written: the caller prints the difference and decides.
///
/// With no `--rev`, the update follows the ref the plugin was installed at, so a
/// plugin installed from a branch keeps tracking that branch and one pinned to a
/// commit stays where it was.
pub fn prepare_update(
    plugins_dir: &Path,
    name: &str,
    rev: Option<&str>,
    xencode_version: &str,
) -> Result<UpdatePlan, InstallError> {
    let registry = PluginRegistry::new(plugins_dir.to_path_buf());
    let destination = registry
        .plugin_path(name)
        .filter(|path| path.exists())
        .ok_or_else(|| InstallError::Install(format!("Plugin '{name}' is not installed")))?;
    let source = read_source(plugins_dir, name).ok_or_else(|| {
        InstallError::Install(format!(
            "'{name}' has no record of where it came from: it was copied from a local path, so \
             there is nothing to fetch. Reinstall it with `xencode plugin install <path>`."
        ))
    })?;
    let url = source.url.clone().ok_or_else(|| {
        InstallError::Install(format!(
            "'{name}' was copied from a local path, so there is nothing to fetch. Reinstall it \
             with `xencode plugin install <path>`."
        ))
    })?;
    let to_ref = rev.map(str::to_string).or_else(|| source.git_ref.clone());
    let staged = clone_at(&url, to_ref.as_deref())?;
    let at = short_commit(staged.commit());
    let manifest_path = find_manifest(staged.path()).ok_or_else(|| {
        InstallError::Install(format!(
            "no {} at the root of {url} at commit {at} — that repository is not a plugin",
            MANIFEST_NAMES.join(" or ")
        ))
    })?;
    let new_manifest = read_manifest(&manifest_path)?;
    if new_manifest.name != name {
        return Err(InstallError::Install(format!(
            "{url} at commit {at} names itself '{}', but it is installed as '{name}' — the \
             repository renamed the plugin, so remove it and install it again",
            new_manifest.name
        )));
    }
    verify(&new_manifest, xencode_version)?;

    let installed_path = find_manifest(&destination).ok_or_else(|| {
        InstallError::Install(format!(
            "{} has no manifest to compare against — remove it and install it again",
            destination.display()
        ))
    })?;
    let old_manifest = read_manifest(&installed_path)?;
    let old_text = std::fs::read_to_string(&installed_path).unwrap_or_default();
    let new_text = std::fs::read_to_string(&manifest_path).unwrap_or_default();
    let changes = declaration_changes(&old_manifest, &new_manifest);
    let same_commit = source.commit.as_deref() == Some(staged.commit());
    let diff = if old_text == new_text {
        String::new()
    } else {
        manifest_diff(
            &old_text,
            &new_text,
            &installed_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "plugin.json".to_string()),
            staged.commit(),
        )
    };
    Ok(UpdatePlan {
        name: name.to_string(),
        destination,
        from: source,
        to_commit: staged.commit().to_string(),
        to_ref,
        url,
        from_version: old_manifest.version,
        to_version: new_manifest.version,
        current: same_commit && changes.is_empty(),
        reaches_agent: reaches_agent(&changes),
        changes,
        diff,
        staged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_in(dir: &Path, args: &[&str]) -> String {
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
            .expect("git");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A real git repository with a real commit, built by the git binary.
    fn repo_with(manifest: &str) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("guardrails-src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("plugin.json"), manifest).unwrap();
        std::fs::write(src.join("README.md"), "# guardrails\n").unwrap();
        git_in(&src, &["init", "-q", "-b", "main", "."]);
        git_in(&src, &["add", "-A"]);
        git_in(&src, &["commit", "-qm", "initial"]);
        (tmp, src)
    }

    fn url_of(repo: &Path) -> String {
        format!("file://{}", repo.display())
    }

    fn head(repo: &Path) -> String {
        git_in(repo, &["rev-parse", "HEAD"])
    }

    const GOOD: &str = r#"{
    "name": "guardrails",
    "version": "1.0.0",
    "permissions": ["prompt"],
    "prompt_prefix": "Run the tests before answering."
}"#;

    fn installed_json(plugins: &Path) -> String {
        std::fs::read_to_string(plugins.join("guardrails/plugin.json")).expect("installed manifest")
    }

    #[test]
    fn a_git_install_clones_verifies_and_names_the_pinned_commit() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();

        let pending = prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0").expect("prepare");
        // Nothing has been copied yet: the declaration is shown first.
        assert!(!pending.destination().exists());
        assert_eq!(pending.name(), "guardrails");
        assert_eq!(pending.commit(), Some(head(&repo).as_str()));
        let declaration = pending.declaration();
        assert!(declaration.contains("permissions: prompt"), "{declaration}");
        assert!(
            declaration.contains("| Run the tests before answering."),
            "{declaration}"
        );

        let outcome = pending.apply().expect("apply");
        assert_eq!(outcome.source.commit.as_deref(), Some(head(&repo).as_str()));
        assert!(outcome.destination.join("plugin.json").is_file());
        assert!(
            !outcome.destination.join(".git").exists(),
            "the clone's git metadata was copied into the plugin directory"
        );
        assert_eq!(
            read_source(plugins.path(), "guardrails")
                .expect("record")
                .commit,
            Some(head(&repo))
        );
    }

    #[test]
    fn an_installed_plugin_loads_and_its_prefix_is_the_one_that_was_shown() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");

        let runtime = crate::runtime::PluginRuntime::load(plugins.path(), "0.1.0");
        assert_eq!(runtime.loaded_count(), 1);
        assert_eq!(runtime.prompt_prefix(), "Run the tests before answering.");
        assert_eq!(
            runtime.reports()[0].prompt_text,
            "Run the tests before answering."
        );
    }

    #[test]
    fn an_install_at_a_rev_pins_exactly_that_commit() {
        let (_tmp, repo) = repo_with(GOOD);
        let first = head(&repo);
        std::fs::write(repo.join("plugin.json"), GOOD.replace("1.0.0", "1.1.0")).unwrap();
        git_in(&repo, &["add", "-A"]);
        git_in(&repo, &["commit", "-qm", "bump"]);
        assert_ne!(first, head(&repo));

        let plugins = tempfile::tempdir().unwrap();
        let outcome = prepare_git(plugins.path(), &url_of(&repo), Some(&first), "0.1.0")
            .expect("prepare at the first commit")
            .apply()
            .expect("apply");
        assert_eq!(outcome.source.commit.as_deref(), Some(first.as_str()));
        assert_eq!(outcome.source.git_ref.as_deref(), Some(first.as_str()));
        assert_eq!(outcome.version, "1.0.0");
        assert!(installed_json(plugins.path()).contains("1.0.0"));
    }

    #[test]
    fn a_rev_that_does_not_exist_is_refused_and_nothing_is_installed() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        let err = prepare_git(
            plugins.path(),
            &url_of(&repo),
            Some("no-such-thing"),
            "0.1.0",
        )
        .expect_err("a missing rev cannot be pinned");
        assert!(matches!(err, InstallError::Git(_)), "{err:?}");
        assert!(!plugins.path().join("guardrails").exists());
    }

    #[test]
    fn a_rev_cannot_be_smuggled_in_as_a_git_option() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        let err = prepare_git(plugins.path(), &url_of(&repo), Some("--help"), "0.1.0")
            .expect_err("an option was accepted as a rev");
        assert!(matches!(err, InstallError::Install(_)), "{err:?}");
    }

    #[test]
    fn a_repository_without_a_manifest_is_not_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("empty-src");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("README.md"), "nothing here").unwrap();
        git_in(&repo, &["init", "-q", "-b", "main", "."]);
        git_in(&repo, &["add", "-A"]);
        git_in(&repo, &["commit", "-qm", "initial"]);

        let plugins = tempfile::tempdir().unwrap();
        let err = prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect_err("a plain repository is not a plugin");
        let message = err.to_string();
        assert!(message.contains("plugin.json"), "{message}");
        assert!(std::fs::read_dir(plugins.path()).unwrap().next().is_none());
    }

    #[test]
    fn a_manifest_this_build_would_not_load_is_refused_before_the_copy() {
        let (_tmp, repo) = repo_with(
            r#"{ "name": "guardrails", "version": "1.0.0",
     "prompt_prefix": "Run the tests before answering." }"#,
        );
        let plugins = tempfile::tempdir().unwrap();
        let err = prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect_err("a prefix without its permission must not install");
        assert!(err.to_string().contains("did not declare"), "{err}");
        assert!(!plugins.path().join("guardrails").exists());
    }

    #[test]
    fn installing_the_same_plugin_twice_points_at_update_instead_of_overwriting() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect("first")
            .apply()
            .expect("apply");
        let err = prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect_err("a second install would replace it silently");
        assert!(err.to_string().contains("update"), "{err}");
    }

    #[test]
    fn a_manifest_that_gains_a_prompt_prefix_is_shown_as_a_diff() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        let before = installed_json(plugins.path());

        std::fs::write(
            repo.join("plugin.json"),
            GOOD.replace(
                "Run the tests before answering.",
                "Never commit without a test run.",
            ),
        )
        .unwrap();
        git_in(&repo, &["add", "-A"]);
        git_in(&repo, &["commit", "-qm", "new instructions"]);

        let plan = prepare_update(plugins.path(), "guardrails", None, "0.1.0").expect("prepare");
        assert!(!plan.current);
        assert!(plan.reaches_agent);
        assert!(
            plan.changes
                .iter()
                .any(|c| c.contains("changes its prompt prefix")),
            "{:?}",
            plan.changes
        );
        assert!(
            plan.diff
                .lines()
                .any(|line| line.starts_with('-') && line.contains("prompt_prefix")),
            "{}",
            plan.diff
        );
        assert!(
            plan.diff
                .lines()
                .any(|line| line.starts_with('+') && line.contains("prompt_prefix")),
            "{}",
            plan.diff
        );
        // Printing the difference has not applied it.
        assert_eq!(installed_json(plugins.path()), before);

        let source = plan.apply().expect("apply");
        assert!(installed_json(plugins.path()).contains("Never commit without a test run."));
        assert_eq!(source.commit, Some(head(&repo)));
    }

    #[test]
    fn a_version_only_update_applies_without_a_declaration_change() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        std::fs::write(repo.join("plugin.json"), GOOD.replace("1.0.0", "1.0.1")).unwrap();
        git_in(&repo, &["add", "-A"]);
        git_in(&repo, &["commit", "-qm", "typo"]);

        let plan = prepare_update(plugins.path(), "guardrails", None, "0.1.0").expect("prepare");
        assert!(!plan.reaches_agent, "{:?}", plan.changes);
        assert_eq!(plan.changes, vec!["version 1.0.0 → 1.0.1".to_string()]);
        assert_eq!(plan.to_version, "1.0.1");
        assert_eq!(plan.from_version, "1.0.0");
        plan.apply().expect("apply");
        assert!(installed_json(plugins.path()).contains("1.0.1"));
    }

    #[test]
    fn a_second_update_from_the_same_commit_says_there_is_nothing_new() {
        let (_tmp, repo) = repo_with(GOOD);
        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), None, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        let plan = prepare_update(plugins.path(), "guardrails", None, "0.1.0").expect("prepare");
        assert!(plan.current, "{:?}", plan.changes);
        assert!(plan.changes.is_empty(), "{:?}", plan.changes);
        assert!(plan.diff.is_empty(), "{}", plan.diff);
    }

    #[test]
    fn updating_follows_the_ref_the_plugin_was_installed_at() {
        let (_tmp, repo) = repo_with(GOOD);
        let first = head(&repo);
        // The branch moves on; a plugin pinned to the first commit stays there.
        std::fs::write(
            repo.join("plugin.json"),
            GOOD.replace(
                "Run the tests before answering.",
                "Never open a file you did not list.",
            ),
        )
        .unwrap();
        git_in(&repo, &["add", "-A"]);
        git_in(&repo, &["commit", "-qm", "stricter"]);

        let plugins = tempfile::tempdir().unwrap();
        prepare_git(plugins.path(), &url_of(&repo), Some(&first), "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        let plan = prepare_update(plugins.path(), "guardrails", None, "0.1.0").expect("prepare");
        assert!(plan.current, "{:?}", plan.changes);
        assert_eq!(plan.to_commit, first);

        // Naming the branch is what moves it.
        let moved =
            prepare_update(plugins.path(), "guardrails", Some("main"), "0.1.0").expect("rev");
        assert_ne!(moved.to_commit, first);
        assert!(!moved.current);
        assert!(moved.reaches_agent, "{:?}", moved.changes);
    }

    #[test]
    fn a_plugin_copied_from_a_path_has_nothing_to_update() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("guardrails");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("plugin.json"), GOOD).unwrap();
        let plugins = tempfile::tempdir().unwrap();
        prepare_path(plugins.path(), &src, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");

        let err = prepare_update(plugins.path(), "guardrails", None, "0.1.0")
            .expect_err("a path install has no source to fetch");
        assert!(err.to_string().contains("local path"), "{err}");
    }

    #[test]
    fn a_path_install_is_named_by_its_manifest_not_its_folder() {
        let tmp = tempfile::tempdir().unwrap();
        // The folder says one thing, the manifest another. The loader looks a
        // plugin up by its declared name, so that is the directory it lands in.
        let src = tmp.path().join("checkout-of-some-repo");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("plugin.json"), GOOD).unwrap();
        let plugins = tempfile::tempdir().unwrap();
        let outcome = prepare_path(plugins.path(), &src, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        assert_eq!(outcome.name, "guardrails");
        assert_eq!(outcome.destination, plugins.path().join("guardrails"));
        assert_eq!(
            read_source(plugins.path(), "guardrails")
                .expect("record")
                .url,
            None
        );
        let runtime = crate::runtime::PluginRuntime::load(plugins.path(), "0.1.0");
        assert_eq!(runtime.loaded_count(), 1, "{:?}", runtime.reports());
    }

    #[test]
    fn a_single_manifest_file_installs_by_its_declared_name() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("guardrails.json");
        std::fs::write(&file, GOOD).unwrap();
        let plugins = tempfile::tempdir().unwrap();
        let outcome = prepare_path(plugins.path(), &file, "0.1.0")
            .expect("prepare")
            .apply()
            .expect("apply");
        assert!(outcome.destination.join("plugin.json").is_file());
        let runtime = crate::runtime::PluginRuntime::load(plugins.path(), "0.1.0");
        assert_eq!(runtime.loaded_count(), 1);
    }

    #[test]
    fn disclosure_names_every_contribution_and_says_it_adds_no_code() {
        let manifest = PluginManifest::from_json(
            r#"{ "name": "g", "version": "1", "permissions": ["prompt", "hooks"],
                 "prompt_prefix": "line one\nline two",
                 "hooks": { "before": { "write_file": "cargo check" } } }"#,
        )
        .unwrap();
        let text = disclose(&manifest);
        assert!(text.contains("permissions: prompt, hooks"), "{text}");
        assert!(text.contains("2 line(s)"), "{text}");
        assert!(text.contains("| line two"), "{text}");
        assert!(text.contains("before write_file → cargo check"), "{text}");
        assert!(text.contains("loads no plugin code"), "{text}");
    }

    #[test]
    fn hook_and_permission_changes_are_each_named() {
        let old = PluginManifest::from_json(
            r#"{ "name": "g", "version": "2", "permissions": ["hooks"],
                 "hooks": { "before": { "write_file": "cargo check" }, "after": { "*": "cargo fmt" } } }"#,
        )
        .unwrap();
        let new = PluginManifest::from_json(
            r#"{ "name": "g", "version": "2", "permissions": ["hooks", "prompt"],
                 "prompt_prefix": "New.",
                 "hooks": { "before": { "write_file": "cargo clippy" } } }"#,
        )
        .unwrap();
        let changes = declaration_changes(&old, &new);
        assert!(
            changes.contains(&"adds a prompt prefix of 1 line(s)".to_string()),
            "{changes:?}"
        );
        assert!(
            changes
                .iter()
                .any(|c| c == "changes the before hook for write_file: cargo check → cargo clippy"),
            "{changes:?}"
        );
        assert!(
            changes
                .iter()
                .any(|c| c == "removes the after hook for *: cargo fmt"),
            "{changes:?}"
        );
        assert!(
            changes
                .iter()
                .any(|c| c == "asks for the prompt permission"),
            "{changes:?}"
        );
        assert!(reaches_agent(&changes));
    }

    #[test]
    fn a_permission_only_change_still_needs_reading() {
        let old = PluginManifest::from_json(r#"{ "name": "g", "version": "2" }"#).unwrap();
        let new = PluginManifest::from_json(
            r#"{ "name": "g", "version": "2", "permissions": ["prompt"] }"#,
        )
        .unwrap();
        let changes = declaration_changes(&old, &new);
        assert_eq!(changes, vec!["asks for the prompt permission".to_string()]);
        assert!(reaches_agent(&changes));
    }

    #[test]
    fn a_source_summary_reads_as_url_and_short_commit() {
        let full = "3f2a1c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f7";
        let source = PluginSource {
            url: Some("file:///srv/guard.git".to_string()),
            commit: Some(full.to_string()),
            git_ref: Some("main".to_string()),
            installed_at: "2026-09-30T12:00:00Z".to_string(),
        };
        assert_eq!(
            source.summary().as_deref(),
            Some("file:///srv/guard.git @ 3f2a1c9 (from main)")
        );
        // Pinned by its own SHA is not a branch worth naming again.
        let pinned = PluginSource {
            git_ref: Some(full.to_string()),
            ..source.clone()
        };
        assert_eq!(
            pinned.summary().as_deref(),
            Some("file:///srv/guard.git @ 3f2a1c9")
        );
        assert_eq!(PluginSource::default().summary(), None);
    }

    #[test]
    fn recognising_a_git_source_leaves_local_paths_alone() {
        assert!(is_git_source("https://github.com/a/b"));
        assert!(is_git_source("git@github.com:a/b.git"));
        assert!(is_git_source("file:///srv/plugins/guard"));
        assert!(is_git_source("/srv/plugins/guard.git"));
        assert!(!is_git_source("./my-plugin"));
        assert!(!is_git_source("/home/sree/src/my-plugin"));
    }
}
