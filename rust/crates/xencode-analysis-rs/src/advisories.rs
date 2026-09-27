//! `advisories` — what is known to be wrong with a crate you depend on.
//!
//! A 4B model cannot be expected to remember which `chrono` release segfaults,
//! and it will happily answer "no known vulnerabilities" from silence. Silence is
//! exactly the problem: the useful answer requires a corpus, and the corpus lives
//! on someone else's machine. This module keeps two of them on disk and reads
//! them without a network, so a lookup costs milliseconds and works on a plane.
//!
//! Two corpora, because they disagree about how much they know:
//!   - **RustSec** (`github.com/RustSec/advisory-db`) — 1 251 advisories over 942
//!     crates as this was written, 6.3 MB shallow-cloned. Every file carries a
//!     fenced ```` ```toml ```` block whose `[versions] patched` array is present
//!     in all 1 251 and whose `[versions] unaffected` array appears in 383. There
//!     is no `broken` key in the current database at all, so "which versions are
//!     bad" has to be worked out as *not in `unaffected`, not in `patched`*.
//!   - **OSV** (`crates.io/all.zip`, 3 490 826 bytes compressed, 2 856 records,
//!     124 of them withdrawn) — of which 732 have no link to RustSec at all and
//!     those cover 791 distinct crates. That is what the second corpus buys:
//!     GHSA-reviewed entries RustSec does not carry, plus 10 malicious-package
//!     records from the supply-chain feed. The other 2 124 are the reason the two
//!     have to be reconciled rather than concatenated, and a linked record can
//!     mirror a RustSec one in either of two ways — `zip` shows both:
//!     `RUSTSEC-2025-0168.json` uses the RustSec number as its own id, while
//!     `GHSA-94vh-gphv-8pm8.json` uses a GitHub number and lists
//!     `RUSTSEC-2025-0168` in `aliases`. Keeping both would print one finding
//!     twice, so the mirror is dropped — but a mirror often carries a severity
//!     word the curated record lacks, and 380 of the 822 RustSec advisories with
//!     no CVSS vector gain one that way, so that much is carried across.
//!
//! A record's `[affected.functions]` (188 files as a table, 87 more as an inline
//! `functions = { … }`) names the exact call path. That is the most actionable
//! line in the whole advisory and it is reported but never used to decide whether
//! you are hit — deciding that needs to know whether you call it, which a version
//! comparison cannot see.
//!
//! Refresh is a deliberate, separate, network-touching step ([`sync`], reached
//! from `xencode advisories sync`). Lookups never touch the network, and a missing
//! corpus is reported as *"advisory state unknown"* rather than as a clean bill of
//! health.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};

/// Which corpus a record came from. Decides how "affected" is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Corpus {
    RustSec,
    Osv,
}

impl Corpus {
    pub fn label(&self) -> &'static str {
        match self {
            Corpus::RustSec => "rustsec",
            Corpus::Osv => "osv",
        }
    }
}

pub const RUSTSEC_REPO: &str = "https://github.com/RustSec/advisory-db.git";
pub const OSV_ZIP_URL: &str =
    "https://osv-vulnerabilities.storage.googleapis.com/crates.io/all.zip";

/// Directory under the user's config dir that holds both corpora.
pub const CORPUS_DIRNAME: &str = "advisories";
pub const RUSTSEC_SUBDIR: &str = "advisory-db";
pub const OSV_SUBDIR: &str = "osv";
/// One `package<TAB>corpus<TAB>relative path` line per (crate, record) pair, so a
/// lookup reads a small text file instead of 4 000 records.
pub const INDEX_FILE: &str = "index.tsv";
pub const SYNC_FILE: &str = "sync.json";

/// The OSV dump is 3.4 MB today; 64 MB stops a wrong URL from filling the disk.
pub const MAX_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;

/// A clone/pull and a 3.4 MB download on a slow link.
pub const SYNC_TIMEOUT_SECS: u64 = 180;

#[derive(Debug, thiserror::Error)]
pub enum AdvisoryError {
    #[error("no advisory corpus at {0} — advisory state is unknown, not clean. Run `xencode advisories sync` (needs network once).")]
    Missing(PathBuf),
    #[error("sync failed: {0}")]
    Sync(String),
    #[error("{0} is not a version this tool can compare (needs major.minor.patch)")]
    BadVersion(String),
}

/// One advisory, as far as this tool can state it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Advisory {
    pub id: String,
    pub package: String,
    pub title: String,
    /// Published date as written in the corpus (YYYY-MM-DD text).
    pub date: String,
    pub url: Option<String>,
    /// A human rating when the corpus has one (OSV's GHSA review), else the raw
    /// CVSS vector from RustSec, prefixed so its origin is not mistaken for a score.
    pub severity: Option<String>,
    /// `unmaintained` / `unsound` / `notice` when the record is informational.
    pub informational: Option<String>,
    pub aliases: Vec<String>,
    /// Requirement strings that mean "this version is fine": RustSec's `patched`
    /// array as written, and OSV's `fixed` events turned into `>= <that version>`.
    pub patched: Vec<String>,
    /// Requirement strings that mean "never affected" (RustSec `unaffected`).
    pub unaffected: Vec<String>,
    /// Requirement strings that mean "affected" (built from OSV event pairs).
    pub affected_reqs: Vec<String>,
    /// Named call paths with the requirement text the corpus gives for them.
    pub functions: Vec<(String, String)>,
    pub withdrawn: bool,
    pub corpus: Corpus,
}

/// What a lookup concluded about one (advisory, version) pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The version sits in a range the corpus says is vulnerable.
    Vulnerable {
        /// The lowest version the corpus offers as safe, when it offers one.
        fix: Option<String>,
    },
    /// The version already matches a `patched` range — no action, but the
    /// advisory is worth naming because it is about this crate.
    Patched { since: Option<String> },
    /// Explicitly outside every affected range.
    Clear,
    /// Informational with nothing to upgrade to (typically `unmaintained`).
    Notice { kind: String },
    /// Retracted by its own database.
    Withdrawn,
}

impl Advisory {
    /// Decide this record against a concrete version.
    pub fn assess(&self, version: &str) -> Result<Outcome, AdvisoryError> {
        let version =
            Version::parse(version).map_err(|_| AdvisoryError::BadVersion(version.into()))?;
        if self.withdrawn {
            return Ok(Outcome::Withdrawn);
        }
        let matches_any = |reqs: &[String]| -> bool {
            reqs.iter()
                .filter_map(|text| parse_req(text))
                .any(|req| req.matches(&version))
        };

        if matches_any(&self.unaffected) {
            return Ok(Outcome::Clear);
        }
        match self.corpus {
            Corpus::RustSec => {
                if !self.patched.is_empty() {
                    if matches_any(&self.patched) {
                        return Ok(Outcome::Patched {
                            since: lowest_bound(&self.patched),
                        });
                    }
                    return Ok(Outcome::Vulnerable {
                        fix: lowest_bound(&self.patched),
                    });
                }
                // `patched = []`: there is nothing to upgrade to.
                match &self.informational {
                    Some(kind) => Ok(Outcome::Notice { kind: kind.clone() }),
                    None => Ok(Outcome::Vulnerable { fix: None }),
                }
            }
            Corpus::Osv => {
                if matches_any(&self.affected_reqs) {
                    Ok(Outcome::Vulnerable {
                        fix: lowest_bound(&self.patched),
                    })
                } else {
                    Ok(Outcome::Clear)
                }
            }
        }
    }
}

fn parse_req(text: &str) -> Option<VersionReq> {
    VersionReq::parse(text).ok()
}

/// The smallest version number mentioned in a set of `>= x` requirements — the
/// suggestion to print. A requirement without a lower bound contributes nothing.
fn lowest_bound(reqs: &[String]) -> Option<String> {
    let mut best: Option<Version> = None;
    for text in reqs {
        let Ok(req) = VersionReq::parse(text) else {
            continue;
        };
        for cmp in &req.comparators {
            if !matches!(
                cmp.op,
                semver::Op::GreaterEq | semver::Op::Greater | semver::Op::Exact
            ) {
                continue;
            }
            let version = Version {
                major: cmp.major,
                minor: cmp.minor.unwrap_or(0),
                patch: cmp.patch.unwrap_or(0),
                pre: cmp.pre.clone(),
                build: Default::default(),
            };
            if best.as_ref().is_none_or(|b| version < *b) {
                best = Some(version);
            }
        }
    }
    best.map(|v| v.to_string())
}

/// The first fenced ```` ```toml ```` block of an advisory file, if it has one.
fn toml_block(text: &str) -> Option<String> {
    let mut inside = false;
    let mut body = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !inside {
            if trimmed == "```toml" {
                inside = true;
            }
            continue;
        }
        if trimmed == "```" {
            return Some(body);
        }
        body.push_str(line);
        body.push('\n');
    }
    // An unterminated block is still the advisory's own toml; take what is there.
    if inside && !body.is_empty() {
        return Some(body);
    }
    None
}

/// The first `# ` heading after the block — the advisory's own title.
fn md_title(text: &str) -> String {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            let title = rest.trim();
            if !title.is_empty() {
                return collapse(title);
            }
        }
    }
    String::new()
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Deserialize)]
struct RsFile {
    advisory: RsAdvisory,
    #[serde(default)]
    versions: RsVersions,
    #[serde(default)]
    affected: RsAffected,
}

#[derive(Deserialize, Default)]
struct RsAdvisory {
    id: String,
    package: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    cvss: Option<String>,
    #[serde(default)]
    informational: Option<String>,
    #[serde(default)]
    withdrawn: Option<String>,
}

#[derive(Deserialize, Default)]
struct RsVersions {
    #[serde(default)]
    patched: Vec<String>,
    #[serde(default)]
    unaffected: Vec<String>,
}

#[derive(Deserialize, Default)]
struct RsAffected {
    /// Written both as an `[affected.functions]` table and as an inline
    /// `functions = { … }` inside `[affected]`; serde maps either to this field.
    #[serde(default)]
    functions: std::collections::BTreeMap<String, Vec<String>>,
}

/// Parse one RustSec `.md` file. Text, not a path, so a caller can feed a
/// hand-written record without a corpus on disk.
pub fn parse_rustsec(text: &str) -> Result<Advisory, String> {
    let block = toml_block(text).ok_or("no ```toml block in the advisory")?;
    let file: RsFile = toml::from_str(&block).map_err(|e| format!("invalid toml block: {e}"))?;
    let functions = file
        .affected
        .functions
        .iter()
        .map(|(path, reqs)| (path.clone(), reqs.join(" or ")))
        .collect();
    Ok(Advisory {
        id: file.advisory.id,
        package: file.advisory.package,
        title: md_title(text),
        date: file.advisory.date,
        url: file.advisory.url,
        severity: file.advisory.cvss.map(|c| format!("cvss {c}")),
        informational: file.advisory.informational,
        aliases: file.advisory.aliases,
        patched: file.versions.patched,
        unaffected: file.versions.unaffected,
        affected_reqs: Vec::new(),
        functions,
        withdrawn: file.advisory.withdrawn.is_some(),
        corpus: Corpus::RustSec,
    })
}

#[derive(Deserialize)]
struct OsvRecord {
    id: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    modified: String,
    #[serde(default)]
    published: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    withdrawn: Option<String>,
    #[serde(default)]
    affected: Vec<OsvAffected>,
    #[serde(default)]
    database_specific: Option<OsvDb>,
}

#[derive(Deserialize)]
struct OsvDb {
    #[serde(default)]
    severity: Option<String>,
}

#[derive(Deserialize)]
struct OsvAffected {
    #[serde(default)]
    package: OsvPackage,
    #[serde(default)]
    ranges: Vec<OsvRange>,
    #[serde(default)]
    database_specific: Option<OsvAffectedDb>,
}

#[derive(Deserialize, Default)]
struct OsvPackage {
    #[serde(default)]
    name: String,
}

#[derive(Deserialize)]
struct OsvAffectedDb {
    /// RustSec's own `informational` marker survives into OSV for the aliased
    /// records; it is why an `unmaintained` notice is not printed as a vulnerability.
    #[serde(default)]
    informational: Option<String>,
}

#[derive(Deserialize)]
struct OsvRange {
    #[serde(default)]
    events: Vec<std::collections::BTreeMap<String, String>>,
}

/// Turn `{introduced: 0.1.0} {fixed: 0.1.3}` into `">= 0.1.0, < 0.1.3"`, and
/// `{introduced: 0.1.0} {last_affected: 0.1.3}` into `">= 0.1.0, <= 0.1.3"`.
/// Events come in pairs within one range; several pairs mean several intervals.
fn osv_intervals(events: &[std::collections::BTreeMap<String, String>]) -> Vec<String> {
    let mut out = Vec::new();
    let mut start: Option<String> = None;
    for event in events {
        if let Some(introduced) = event.get("introduced") {
            start = Some(introduced.clone());
            continue;
        }
        if let Some(fixed) = event.get("fixed") {
            out.push(req_pair(start.take().as_deref(), "<", fixed));
            continue;
        }
        if let Some(last) = event.get("last_affected") {
            out.push(req_pair(start.take().as_deref(), "<=", last));
        }
    }
    if let Some(introduced) = start {
        // Introduced with no end: everything from there on.
        out.push(format!(">= {introduced}"));
    }
    out
}

fn req_pair(start: Option<&str>, op: &str, end: &str) -> String {
    match start {
        // `introduced: "0"` is how the dump writes "every version so far", and a
        // `>= 0` term says the same thing while costing the reader a line.
        Some(s) if s != "0" && s != "0.0.0" => format!(">= {s}, {op} {end}"),
        _ => format!("{op} {end}"),
    }
}

/// Parse one OSV `.json` record. A record can name several packages, so it yields
/// one [`Advisory`] per `affected` entry that belongs to crates.io.
pub fn parse_osv(text: &str) -> Result<Vec<Advisory>, String> {
    let record: OsvRecord =
        serde_json::from_str(text).map_err(|e| format!("invalid osv json: {e}"))?;
    let mut out = Vec::new();
    for affected in &record.affected {
        let ecosystem = crate_ecosystem(text, &affected.package.name);
        if matches!(ecosystem, Some(other) if other != "crates.io") {
            continue;
        }
        if affected.package.name.is_empty() {
            continue;
        }
        let events: Vec<_> = affected
            .ranges
            .iter()
            .flat_map(|r| r.events.clone())
            .collect();
        let affected_reqs = osv_intervals(&events);
        let fix = affected
            .ranges
            .iter()
            .flat_map(|r| r.events.iter())
            .filter_map(|e| e.get("fixed"))
            .map(|fixed| format!(">= {fixed}"))
            .collect::<Vec<_>>();
        out.push(Advisory {
            id: record.id.clone(),
            package: affected.package.name.clone(),
            title: collapse(&record.summary),
            date: date_of(if record.published.is_empty() {
                &record.modified
            } else {
                &record.published
            }),
            url: Some(format!("https://osv.dev/vulnerability/{}", record.id)),
            severity: record
                .database_specific
                .as_ref()
                .and_then(|d| d.severity.clone()),
            informational: affected
                .database_specific
                .as_ref()
                .and_then(|d| d.informational.clone()),
            aliases: record.aliases.clone(),
            patched: fix,
            unaffected: Vec::new(),
            affected_reqs,
            functions: Vec::new(),
            withdrawn: record.withdrawn.is_some(),
            corpus: Corpus::Osv,
        });
    }
    Ok(out)
}

/// The ecosystem for a package entry inside a record, read from the raw JSON
/// because `serde_json`'s value tree is already there and the field is optional.
fn crate_ecosystem(record_json: &str, package_name: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(record_json).ok()?;
    value
        .get("affected")?
        .as_array()?
        .iter()
        .find(|entry| {
            entry
                .pointer("/package/name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| n == package_name)
        })?
        .pointer("/package/ecosystem")
        .and_then(|e| e.as_str())
        .map(|e| e.to_string())
}

/// `2026-09-10T03:51:04Z` -> `2026-09-10`.
fn date_of(timestamp: &str) -> String {
    timestamp
        .split(['T', 'Z'])
        .next()
        .unwrap_or(timestamp)
        .trim()
        .to_string()
}

/// What `sync` recorded, and the only proof a lookup has of its own freshness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncInfo {
    pub synced_at_unix: u64,
    pub rustsec_revision: String,
    pub rustsec_advisories: usize,
    pub osv_records: usize,
    pub index_lines: usize,
}

#[derive(Debug, Clone)]
pub struct SyncOutcome {
    pub info: SyncInfo,
    /// False on the first run, when the corpus was cloned rather than pulled.
    pub rustsec_pulled: bool,
    pub osv_bytes: usize,
    pub corpus: PathBuf,
}

/// Lines of `index.tsv` for one crate name. Kept as text so a half-written index
/// is still readable by eye.
fn index_lines(text: &str) -> Vec<(String, Corpus, PathBuf)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split('\t');
        let (Some(package), Some(corpus), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let corpus = match corpus {
            "rustsec" => Corpus::RustSec,
            "osv" => Corpus::Osv,
            _ => continue,
        };
        out.push((package.to_string(), corpus, PathBuf::from(path)));
    }
    out
}

/// A lookup result: the records for a crate, and how recent they are.
#[derive(Debug, Clone)]
pub struct Lookup {
    pub crate_name: String,
    pub advisories: Vec<Advisory>,
    /// Records retracted by their own database, counted but not carried — the
    /// count is what tells a reader the listing is not the whole corpus.
    pub withdrawn: usize,
    pub sync: SyncInfo,
}

impl Lookup {
    pub fn is_empty(&self) -> bool {
        self.advisories.is_empty()
    }
}

/// Where the corpus lives for a given config directory.
pub fn corpus_dir(config_dir: &Path) -> PathBuf {
    config_dir.join(CORPUS_DIRNAME)
}

/// Read the sync record, or [`AdvisoryError::Missing`] when there is no usable
/// corpus. A directory that exists but has no `sync.json` is *not* a corpus: the
/// index and the data are written together, so a partial one says "unknown".
pub fn sync_info(corpus: &Path) -> Result<SyncInfo, AdvisoryError> {
    let path = corpus.join(SYNC_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => return Err(AdvisoryError::Missing(corpus.to_path_buf())),
    };
    serde_json::from_str(&text).map_err(|e| {
        AdvisoryError::Sync(format!(
            "{} is not readable ({e}) — advisory state is unknown, run `xencode advisories sync`",
            path.display()
        ))
    })
}

fn index_text(corpus: &Path) -> Result<String, AdvisoryError> {
    let path = corpus.join(INDEX_FILE);
    std::fs::read_to_string(&path).map_err(|_| AdvisoryError::Missing(corpus.to_path_buf()))
}

/// Every advisory the corpus holds for `crate_name`, both corpora, newest first.
pub fn advisories_for(corpus: &Path, crate_name: &str) -> Result<Lookup, AdvisoryError> {
    let sync = sync_info(corpus)?;
    let index = index_text(corpus)?;
    let mut kept = Vec::new();
    let mut pending_osv = Vec::new();
    let mut withdrawn = 0;
    for (package, corpus_kind, rel) in index_lines(&index) {
        if package != crate_name {
            continue;
        }
        let file = corpus.join(&rel);
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        let parsed: Vec<Advisory> = match corpus_kind {
            Corpus::RustSec => parse_rustsec(&text).map(|a| vec![a]).unwrap_or_default(),
            Corpus::Osv => parse_osv(&text).unwrap_or_default(),
        };
        for advisory in parsed {
            if advisory.package != package {
                continue;
            }
            if advisory.withdrawn {
                withdrawn += 1;
                continue;
            }
            match advisory.corpus {
                Corpus::RustSec => {
                    kept.push(advisory);
                }
                Corpus::Osv => pending_osv.push(advisory),
            }
        }
    }
    // RustSec is the curated half of the pair. An OSV record that names the same
    // advisory — either as its own id or under an alias — is the same finding
    // again with a different label, so it is dropped *only when* the RustSec
    // record it mirrors is actually here. What the mirror has and the curated
    // record often lacks is a one-word severity, so that is carried over.
    let rustsec_ids: std::collections::HashSet<String> = kept
        .iter()
        .flat_map(|a| std::iter::once(a.id.clone()).chain(a.aliases.iter().cloned()))
        .collect();
    let mut severity_of: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for advisory in pending_osv {
        let link = std::iter::once(advisory.id.clone())
            .chain(advisory.aliases.iter().cloned())
            .find(|name| name.starts_with("RUSTSEC-") && rustsec_ids.contains(name));
        match link {
            Some(id) => {
                if let Some(severity) = advisory.severity {
                    severity_of
                        .entry(id)
                        .or_insert_with(|| format!("GHSA {severity}"));
                }
            }
            None => kept.push(advisory),
        }
    }
    for advisory in kept.iter_mut() {
        if advisory.corpus != Corpus::RustSec || advisory.severity.is_some() {
            continue;
        }
        if let Some(severity) = severity_of.get(&advisory.id) {
            advisory.severity = Some(severity.clone());
        }
    }
    kept.sort_by(|a, b| b.date.cmp(&a.date).then_with(|| a.id.cmp(&b.id)));
    Ok(Lookup {
        crate_name: crate_name.to_string(),
        advisories: kept,
        withdrawn,
        sync,
    })
}

/// Look one crate up and assess every record against `version`.
pub fn advisories_for_version(
    corpus: &Path,
    crate_name: &str,
    version: &str,
) -> Result<Vec<(Advisory, Outcome)>, AdvisoryError> {
    let lookup = advisories_for(corpus, crate_name)?;
    Ok(lookup
        .advisories
        .into_iter()
        .filter_map(|a| a.assess(version).ok().map(|o| (a, o)))
        .collect())
}

/// Check every `[[package]]` entry of a Cargo.lock against the corpus. Only
/// crates that some advisory mentions are opened, so a 500-line lock costs a
/// handful of file reads rather than 500.
pub fn check_lockfile(
    corpus: &Path,
    lock_text: &str,
) -> Result<Vec<(String, String, Advisory, Outcome)>, AdvisoryError> {
    sync_info(corpus)?;
    let index = index_text(corpus)?;
    let rows = index_lines(&index);
    let mut hits = Vec::new();
    for (package, version) in locked_packages(lock_text) {
        if !rows.iter().any(|(named, _, _)| *named == package) {
            continue;
        }
        // Reuse the loader so the dedup rules are the same either way.
        let lookup = advisories_for(corpus, &package)?;
        for advisory in lookup.advisories {
            let Ok(outcome) = advisory.assess(&version) else {
                continue;
            };
            if matches!(outcome, Outcome::Vulnerable { .. } | Outcome::Notice { .. }) {
                hits.push((package.clone(), version.clone(), advisory, outcome));
            }
        }
    }
    hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.id.cmp(&b.2.id)));
    Ok(hits)
}

/// `[[package]] name`/`version` pairs from Cargo.lock text. Line-based on purpose:
/// see the note on the same reader in the TUI crate's `crate_sources`.
pub fn locked_packages(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut name: Option<String> = None;
    let mut version: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            if let (Some(n), Some(v)) = (name.take(), version.take()) {
                out.push((n, v));
            }
            continue;
        }
        if let Some(value) = lock_key(line, "name") {
            name = Some(value);
        } else if let Some(value) = lock_key(line, "version") {
            version = Some(value);
        }
    }
    if let (Some(n), Some(v)) = (name, version) {
        out.push((n, v));
    }
    out
}

fn lock_key(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim();
    let rest = rest.strip_prefix('=')?.trim();
    let value = rest.strip_prefix('"')?.strip_suffix('"')?;
    Some(value.to_string())
}

/// Fetch both corpora and rebuild the index. The only network this module does.
pub async fn sync(corpus: &Path) -> Result<SyncOutcome, AdvisoryError> {
    std::fs::create_dir_all(corpus).map_err(|e| AdvisoryError::Sync(e.to_string()))?;
    let rustsec_dir = corpus.join(RUSTSEC_SUBDIR);
    let rustsec_pulled = pull_rustsec(&rustsec_dir)?;
    let revision = git_revision(&rustsec_dir)?;

    let bytes = download(OSV_ZIP_URL).await?;
    let osv_dir = corpus.join(OSV_SUBDIR);
    let records = write_osv(&osv_dir, &bytes)?;

    let advisories = count_rustsec_advisories(&rustsec_dir)?;
    let index = build_index(corpus)?;

    let info = SyncInfo {
        synced_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        rustsec_revision: revision,
        rustsec_advisories: advisories,
        osv_records: records,
        index_lines: index,
    };
    let text = serde_json::to_string_pretty(&info)
        .map_err(|e| AdvisoryError::Sync(format!("could not encode sync state: {e}")))?;
    std::fs::write(corpus.join(SYNC_FILE), text + "\n")
        .map_err(|e| AdvisoryError::Sync(format!("could not write sync state: {e}")))?;
    Ok(SyncOutcome {
        info,
        rustsec_pulled,
        osv_bytes: bytes.len(),
        corpus: corpus.to_path_buf(),
    })
}

/// `git clone --depth 1` the first time, `git pull --ff-only` afterwards. `git` is
/// shelled out because a shallow clone over HTTP is exactly what it is good at,
/// and the alternative is a second git-object parser in this crate.
fn pull_rustsec(dir: &Path) -> Result<bool, AdvisoryError> {
    if dir.join(".git").is_dir() {
        run_git(&["-C", &dir.display().to_string(), "pull", "--ff-only"])?;
        return Ok(true);
    }
    let out = run_git(&[
        "clone",
        "--depth",
        "1",
        RUSTSEC_REPO,
        &dir.display().to_string(),
    ])?;
    if !dir.join("crates").is_dir() {
        return Err(AdvisoryError::Sync(format!(
            "the clone at {} has no crates/ directory ({out})",
            dir.display()
        )));
    }
    Ok(false)
}

fn run_git(args: &[&str]) -> Result<String, AdvisoryError> {
    let output = std::process::Command::new("git")
        .args(args)
        .output()
        .map_err(|e| {
            AdvisoryError::Sync(format!(
                "could not run git: {e} — installing git is a prerequisite for this command"
            ))
        })?;
    if !output.status.success() {
        return Err(AdvisoryError::Sync(format!(
            "git {} failed: {}",
            args.first().copied().unwrap_or(""),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_revision(dir: &Path) -> Result<String, AdvisoryError> {
    run_git(&["-C", &dir.display().to_string(), "rev-parse", "HEAD"])
}

async fn download(url: &str) -> Result<Vec<u8>, AdvisoryError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(SYNC_TIMEOUT_SECS))
        .user_agent("xencode-advisories (+https://github.com/xencode)")
        .build()
        .map_err(|e| AdvisoryError::Sync(e.to_string()))?;
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| AdvisoryError::Sync(format!("could not reach {url}: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(AdvisoryError::Sync(format!(
            "{url} answered {status} — no sync happened, the previous corpus is still in place"
        )));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| AdvisoryError::Sync(format!("download of {url} broke: {e}")))?
    {
        if bytes.len() + chunk.len() > MAX_DOWNLOAD_BYTES {
            return Err(AdvisoryError::Sync(format!(
                "{url} is over the {MAX_DOWNLOAD_BYTES}-byte cap — refused"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Extract the JSON records into `dir`, replacing whatever was there. A failed
/// extraction leaves the directory untouched rather than half-populated.
fn write_osv(dir: &Path, zip_bytes: &[u8]) -> Result<usize, AdvisoryError> {
    let staging = dir.with_extension("new");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)
        .map_err(|e| AdvisoryError::Sync(format!("could not create {}: {e}", staging.display())))?;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes))
        .map_err(|e| AdvisoryError::Sync(format!("the OSV archive is not readable: {e}")))?;
    let mut written = 0;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| AdvisoryError::Sync(format!("archive entry {i}: {e}")))?;
        let name = entry.name().to_string();
        if !name.ends_with(".json") || name.contains('/') || name.contains("..") {
            continue;
        }
        let mut reader = entry;
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf)
            .map_err(|e| AdvisoryError::Sync(format!("{name}: {e}")))?;
        std::fs::write(staging.join(&name), &buf)
            .map_err(|e| AdvisoryError::Sync(format!("could not write {name}: {e}")))?;
        written += 1;
    }
    if written == 0 {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(AdvisoryError::Sync(
            "the OSV archive held no JSON records".to_string(),
        ));
    }
    if dir.is_dir() {
        std::fs::remove_dir_all(dir).map_err(|e| {
            AdvisoryError::Sync(format!("could not replace {}: {e}", dir.display()))
        })?;
    }
    std::fs::rename(&staging, dir)
        .map_err(|e| AdvisoryError::Sync(format!("could not move the new OSV data in: {e}")))?;
    Ok(written)
}

fn count_rustsec_advisories(rustsec_dir: &Path) -> Result<usize, AdvisoryError> {
    let crates_dir = rustsec_dir.join("crates");
    let entries = std::fs::read_dir(&crates_dir).map_err(|e| {
        AdvisoryError::Sync(format!("could not list {}: {e}", crates_dir.display()))
    })?;
    let mut count = 0;
    for entry in entries.flatten() {
        if let Ok(files) = std::fs::read_dir(entry.path()) {
            count += files
                .flatten()
                .filter(|f| f.file_name().to_string_lossy().starts_with("RUSTSEC-"))
                .count();
        }
    }
    Ok(count)
}

/// Write `index.tsv`: every crate name the corpora mention, with the file that
/// mentions it. OSV records are expanded per package because one record can name
/// several crates.
pub fn build_index(corpus: &Path) -> Result<usize, AdvisoryError> {
    let mut lines: Vec<String> = Vec::new();
    let crates_dir = corpus.join(RUSTSEC_SUBDIR).join("crates");
    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(files) = std::fs::read_dir(entry.path()) else {
                continue;
            };
            for file in files.flatten() {
                let file_name = file.file_name().to_string_lossy().to_string();
                if !file_name.starts_with("RUSTSEC-") || !file_name.ends_with(".md") {
                    continue;
                }
                lines.push(format!(
                    "{name}\trustsec\t{RUSTSEC_SUBDIR}/crates/{name}/{file_name}"
                ));
            }
        }
    }
    let osv_dir = corpus.join(OSV_SUBDIR);
    if let Ok(entries) = std::fs::read_dir(&osv_dir) {
        let mut files: Vec<_> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".json"))
            .collect();
        files.sort();
        for file in files {
            let Ok(text) = std::fs::read_to_string(osv_dir.join(&file)) else {
                continue;
            };
            let Ok(record) = serde_json::from_str::<OsvRecord>(&text) else {
                continue;
            };
            for affected in &record.affected {
                if affected.package.name.is_empty() {
                    continue;
                }
                lines.push(format!(
                    "{}\tosv\t{OSV_SUBDIR}/{file}",
                    affected.package.name
                ));
            }
        }
    }
    lines.sort();
    lines.dedup();
    let text = lines.join("\n") + "\n";
    std::fs::write(corpus.join(INDEX_FILE), text)
        .map_err(|e| AdvisoryError::Sync(format!("could not write the advisory index: {e}")))?;
    Ok(lines.len())
}

/// Render a lookup as the text the model is handed.
pub fn render_lookup(lookup: &Lookup, version: Option<&str>) -> String {
    let age = age_days(lookup.sync.synced_at_unix);
    if lookup.advisories.is_empty() {
        return format!(
            "{}: no advisory in the local corpus ({} advisories over {} index lines, synced {} \
             at revision {})\nnote: absence of an advisory is not a statement that this crate is \
             safe.",
            lookup.crate_name,
            lookup.sync.rustsec_advisories,
            lookup.sync.index_lines,
            age,
            short_rev(&lookup.sync.rustsec_revision)
        );
    }
    let mut out = format!(
        "{} advisory record(s) for {} — corpus synced {}, rustsec revision {}\n",
        lookup.advisories.len(),
        lookup.crate_name,
        age,
        short_rev(&lookup.sync.rustsec_revision)
    );
    if lookup.withdrawn > 0 {
        out.push_str(&format!(
            "{} further record(s) were withdrawn by their database and are not listed\n",
            lookup.withdrawn
        ));
    }
    if let Some(version) = version {
        out.push_str(&format!("assessed against version {version}:\n"));
    }
    for advisory in &lookup.advisories {
        out.push_str(&render_advisory(advisory, version));
    }
    out
}

fn render_advisory(advisory: &Advisory, version: Option<&str>) -> String {
    let mut line = format!(
        "  {} [{}] {}: {}",
        advisory.id,
        advisory.corpus.label(),
        advisory.date,
        if advisory.title.is_empty() {
            "(no title)"
        } else {
            &advisory.title
        }
    );
    if let Some(severity) = &advisory.severity {
        line.push_str(&format!(" — severity {severity}"));
    }
    if let Some(kind) = &advisory.informational {
        line.push_str(&format!(" — informational: {kind}"));
    }
    line.push('\n');
    if let Some(url) = &advisory.url {
        line.push_str(&format!("      see: {url}\n"));
    }
    if let Some(version) = version {
        match advisory.assess(version) {
            Ok(outcome) => line.push_str(&format!(
                "      this version: {}\n",
                render_outcome(&outcome)
            )),
            Err(e) => line.push_str(&format!("      this version: {e}\n")),
        }
    } else if !advisory.patched.is_empty() {
        line.push_str(&format!(
            "      patched: {}\n",
            advisory.patched.join(" or ")
        ));
    }
    for (path, reqs) in advisory.functions.iter().take(4) {
        line.push_str(&format!("      affected function {path} in {reqs}\n"));
    }
    if advisory.functions.len() > 4 {
        line.push_str(&format!(
            "      ({} more named functions)\n",
            advisory.functions.len() - 4
        ));
    }
    line
}

fn render_outcome(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Vulnerable { fix: Some(fix) } => {
            format!("AFFECTED here — the corpus offers {fix} as safe")
        }
        Outcome::Vulnerable { fix: None } => {
            "AFFECTED here — the corpus offers no version that is safe".to_string()
        }
        Outcome::Patched { since } => match since {
            Some(s) => format!("not affected (this version is at or above the {s} fix)"),
            None => "not affected (this version matches the patched range)".to_string(),
        },
        Outcome::Clear => "not affected".to_string(),
        Outcome::Notice { kind } => {
            format!("informational notice ({kind}) — nothing to upgrade to")
        }
        Outcome::Withdrawn => "withdrawn by its database — ignored".to_string(),
    }
}

fn short_rev(revision: &str) -> String {
    revision.chars().take(9).collect()
}

/// "today" / "3 days ago" from a unix timestamp, without a calendar library.
pub fn age_days(unix: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if unix >= now {
        return "today".to_string();
    }
    let days = (now - unix) / 86_400;
    match days {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        2..=6 => format!("{days} days ago"),
        7..=30 => format!("{} weeks ago", days / 7),
        _ => format!("{} months ago", days / 30),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape is quoted from the corpus, not invented: chrono's segfault
    /// advisory, the one every RustSec tutorial cites.
    const CHRONO: &str = r#"```toml
[advisory]
id = "RUSTSEC-2020-0159"
package = "chrono"
date = "2020-11-10"
url = "https://github.com/chronotope/chrono/issues/499"
categories = ["code-execution", "memory-corruption"]
keywords = ["segfault"]
related = ["CVE-2020-26235", "RUSTSEC-2020-0071"]

[versions]
patched = [">=0.4.20"]
```

# Potential segfault in `localtime_r` invocations

### Impact

Unix-like operating systems may segfault.
"#;

    /// zip 2.x: a `patched` range, an `unaffected` range below it, and two named
    /// functions — the three things a bare `patched` list would not carry.
    const ZIP: &str = r#"```toml
[advisory]
id = "RUSTSEC-2025-0168"
package = "zip"
date = "2025-03-16"
url = "https://github.com/zip-rs/zip2/security/advisories/GHSA-94vh-gphv-8pm8"
cvss = "CVSS:4.0/AV:N/AC:H/AT:N/PR:L/UI:N/VC:L/VI:H/VA:N/SC:H/SI:H/SA:H"
keywords = ["file-overwrite", "symlink", "path-traversal"]
aliases = ["CVE-2025-29787", "GHSA-94vh-gphv-8pm8"]
license = "CC-BY-4.0"

[affected.functions]
"zip::unstable::stream::ZipStreamReader::extract" = ["< 2.3.0, >= 1.3.0"]
"zip::read::ZipArchive::extract" = ["< 2.3.0, >= 1.3.0"]

[versions]
patched = [">= 2.3.0"]
unaffected = ["< 1.3.0"]
```

# Incorrect path canonicalization during Archive Extraction
"#;

    const UNMAINTAINED: &str = r#"```toml
[advisory]
id = "RUSTSEC-2024-0337"
package = "zip_next"
date = "2024-04-20"
url = "https://github.com/zip-rs/zip/issues/446"
informational = "unmaintained"
[versions]
patched = []
```

# The crate `zip_next` has been renamed to `zip`.
"#;

    #[test]
    fn parses_a_patched_range_and_names_the_functions() {
        let advisory = parse_rustsec(ZIP).unwrap();
        assert_eq!(advisory.id, "RUSTSEC-2025-0168");
        assert_eq!(advisory.package, "zip");
        assert_eq!(advisory.date, "2025-03-16");
        assert_eq!(
            advisory.title,
            "Incorrect path canonicalization during Archive Extraction"
        );
        assert_eq!(advisory.patched, vec![">= 2.3.0".to_string()]);
        assert_eq!(advisory.unaffected, vec!["< 1.3.0".to_string()]);
        assert_eq!(advisory.functions.len(), 2, "both named functions survive");
        assert!(advisory
            .severity
            .as_deref()
            .unwrap()
            .starts_with("cvss CVSS:4.0"));

        // 1.2.0 is below `unaffected`'s floor of 1.3.0, so it is not covered by
        // that statement — and it is also not patched. What the corpus actually
        // asserts is "unaffected below 1.3.0", so a version under it is clear.
        assert_eq!(
            advisory.assess("1.2.0").unwrap(),
            Outcome::Clear,
            "the unaffected range is honoured"
        );
        assert_eq!(
            advisory.assess("2.2.0").unwrap(),
            Outcome::Vulnerable {
                fix: Some("2.3.0".to_string())
            }
        );
        assert_eq!(
            advisory.assess("2.3.0").unwrap(),
            Outcome::Patched {
                since: Some("2.3.0".to_string())
            }
        );
        assert_eq!(
            advisory.assess("9.9.9").unwrap(),
            Outcome::Patched {
                since: Some("2.3.0".to_string())
            }
        );
    }

    #[test]
    fn chrono_0_4_19_is_affected_and_0_4_20_is_not() {
        let advisory = parse_rustsec(CHRONO).unwrap();
        assert_eq!(advisory.functions.len(), 0);
        assert_eq!(
            advisory.assess("0.4.19").unwrap(),
            Outcome::Vulnerable {
                fix: Some("0.4.20".to_string())
            }
        );
        assert_eq!(
            advisory.assess("0.4.20").unwrap(),
            Outcome::Patched {
                since: Some("0.4.20".to_string())
            }
        );
        // A pre-release of the fixed version does not count as fixed.
        assert!(matches!(
            advisory.assess("0.4.20-rc.1").unwrap(),
            Outcome::Vulnerable { .. }
        ));
    }

    #[test]
    fn an_unmaintained_notice_is_not_reported_as_a_vulnerability() {
        let advisory = parse_rustsec(UNMAINTAINED).unwrap();
        assert_eq!(advisory.informational.as_deref(), Some("unmaintained"));
        assert_eq!(
            advisory.assess("2.2.1").unwrap(),
            Outcome::Notice {
                kind: "unmaintained".to_string()
            }
        );
    }

    #[test]
    fn a_broken_version_is_refused_rather_than_guessed() {
        let advisory = parse_rustsec(CHRONO).unwrap();
        let error = advisory.assess("1").unwrap_err();
        assert!(error.to_string().contains("not a version"), "{error}");
    }

    #[test]
    fn the_toml_block_is_taken_and_the_prose_is_not() {
        // A description containing another fenced block must not become the record.
        let text = format!(
            "{CHRONO}\n```rust\nlet x = 1;\n```\nmore prose\n\n```toml\n[advisory]\nid = \"BOGUS\"\npackage = \"x\"\n```"
        );
        let advisory = parse_rustsec(&text).unwrap();
        assert_eq!(advisory.id, "RUSTSEC-2020-0159");
    }

    #[test]
    fn osv_events_become_affected_intervals() {
        let json = r#"{
            "id": "GHSA-22w3-693w-x895",
            "modified": "2026-09-10T03:51:04.039013566Z",
            "published": "2026-05-01T00:00:00Z",
            "summary": "Origin validation mismatch",
            "database_specific": {"severity": "MODERATE"},
            "affected": [
              {"package": {"name": "webauthn-rs-core", "ecosystem": "crates.io"},
               "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "0.5.5"}]}]},
              {"package": {"name": "webauthn-authenticator-rs", "ecosystem": "crates.io"},
               "ranges": [{"type": "SEMVER", "events": [{"introduced": "0.5.0"}, {"last_affected": "0.5.4"}]}]}
            ]
        }"#;
        let advisories = parse_osv(json).unwrap();
        assert_eq!(advisories.len(), 2, "one record, two crates");
        let first = &advisories[0];
        assert_eq!(first.package, "webauthn-rs-core");
        assert_eq!(first.severity.as_deref(), Some("MODERATE"));
        assert_eq!(first.date, "2026-05-01");
        assert_eq!(first.affected_reqs, vec!["< 0.5.5".to_string()]);
        assert_eq!(
            first.assess("0.5.4").unwrap(),
            Outcome::Vulnerable {
                fix: Some("0.5.5".to_string())
            }
        );
        assert_eq!(first.assess("0.5.5").unwrap(), Outcome::Clear);
        let second = &advisories[1];
        assert_eq!(second.affected_reqs, vec![">= 0.5.0, <= 0.5.4".to_string()]);
        assert_eq!(second.assess("0.4.9").unwrap(), Outcome::Clear);
        assert_eq!(
            second.assess("0.5.4").unwrap(),
            Outcome::Vulnerable { fix: None }
        );
    }

    #[test]
    fn osv_partial_versions_compare_as_the_corpus_means_them_to() {
        // `fixed: "0.62"` appears in the real dump. semver treats an unspecified
        // component of `>=`/`<` as 0, so 0.61.9 is inside and 0.62.0 is outside.
        let json = r#"{"id":"GHSA-x","modified":"","summary":"s","affected":[
            {"package":{"name":"c","ecosystem":"crates.io"},
             "ranges":[{"type":"SEMVER","events":[{"introduced":"0"},{"fixed":"0.62"}]}]}]}"#;
        let advisory = &parse_osv(json).unwrap()[0];
        assert_eq!(advisory.affected_reqs, vec!["< 0.62".to_string()]);
        assert!(matches!(
            advisory.assess("0.61.9").unwrap(),
            Outcome::Vulnerable { .. }
        ));
        assert_eq!(advisory.assess("0.62.0").unwrap(), Outcome::Clear);
    }

    #[test]
    fn an_osv_record_for_another_ecosystem_is_not_mixed_in() {
        let json = r#"{"id":"GHSA-y","modified":"","summary":"s","affected":[
            {"package":{"name":"some-pip-thing","ecosystem":"PyPI"},
             "ranges":[{"type":"SEMVER","events":[{"introduced":"0"},{"fixed":"1"}]}]}]}"#;
        assert!(parse_osv(json).unwrap().is_empty());
    }

    #[test]
    fn a_withdrawn_record_is_ignored_but_still_named() {
        let text = CHRONO.replace(
            "keywords = [\"segfault\"]",
            "keywords = [\"segfault\"]\nwithdrawn = \"2026-01-01\"",
        );
        let advisory = parse_rustsec(&text).unwrap();
        assert!(advisory.withdrawn);
        assert_eq!(advisory.assess("0.4.19").unwrap(), Outcome::Withdrawn);
    }

    #[test]
    fn a_file_without_a_toml_block_is_rejected_by_name() {
        let error = parse_rustsec("# just a heading\n\nprose only").unwrap_err();
        assert!(error.contains("no ```toml block"), "{error}");
    }

    #[test]
    fn lockfile_entries_are_read_in_order_with_their_versions() {
        let lock = r#"version = 4

[[package]]
name = "chrono"
version = "0.4.19"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "zip"
version = "2.2.0"
"#;
        assert_eq!(
            locked_packages(lock),
            vec![
                ("chrono".to_string(), "0.4.19".to_string()),
                ("zip".to_string(), "2.2.0".to_string())
            ]
        );
    }

    /// A corpus on disk, built from hand-written records rather than a download:
    /// the lookup path is exercised end to end without any network.
    fn scratch_corpus(dir: &Path) -> SyncInfo {
        let chrono = CHRONO.replace("package = \"chrono\"", "package = \"scratchy\"");
        let path = dir.join(RUSTSEC_SUBDIR).join("crates").join("scratchy");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("RUSTSEC-2020-0159.md"), chrono).unwrap();
        let osv = dir.join(OSV_SUBDIR);
        std::fs::create_dir_all(&osv).unwrap();
        // Four shapes the real dump all contain, for the same one crate.
        let affected = serde_json::json!([{
            "package": {"name": "scratchy", "ecosystem": "crates.io"},
            "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "0.2"}]}]
        }]);
        let record = |id: &str, extra: serde_json::Value| -> String {
            let mut body =
                serde_json::json!({"id": id, "modified": "", "summary": id, "affected": affected});
            if let (Some(target), Some(source)) = (body.as_object_mut(), extra.as_object()) {
                for (key, value) in source {
                    target.insert(key.clone(), value.clone());
                }
            }
            body.to_string()
        };
        let records = [
            // linked by alias, and carrying the rating the RustSec record has no room for
            (
                "GHSA-by-alias.json",
                record(
                    "GHSA-by-alias",
                    serde_json::json!({"aliases": ["RUSTSEC-2020-0159"],
                                       "database_specific": {"severity": "HIGH"}}),
                ),
            ),
            // linked by having the RustSec number as its own id
            (
                "RUSTSEC-2020-0159.json",
                record("RUSTSEC-2020-0159", serde_json::json!({})),
            ),
            // no link to RustSec at all: the coverage the second corpus exists for
            (
                "GHSA-only-here.json",
                record("GHSA-only-here", serde_json::json!({})),
            ),
            // retracted by its own database
            (
                "GHSA-dead.json",
                record(
                    "GHSA-dead",
                    serde_json::json!({"withdrawn": "2026-01-01T00:00:00Z"}),
                ),
            ),
        ];
        for (name, body) in records {
            std::fs::write(osv.join(name), body).unwrap();
        }
        let lines = build_index(dir).unwrap();
        let info = SyncInfo {
            synced_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            rustsec_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            rustsec_advisories: 1,
            osv_records: 4,
            index_lines: lines,
        };
        std::fs::write(dir.join(SYNC_FILE), serde_json::to_string(&info).unwrap()).unwrap();
        info
    }

    #[test]
    fn a_missing_corpus_says_unknown_rather_than_clean() {
        let dir = std::env::temp_dir().join(format!("xencode-adv-missing-{}", std::process::id()));
        let error = advisories_for(&dir, "chrono").unwrap_err();
        assert!(
            error.to_string().contains("advisory state is unknown"),
            "{error}"
        );
        assert!(error.to_string().contains("sync"), "{error}");
    }

    #[test]
    fn a_lookup_reads_the_index_and_returns_both_corpora() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-adv-lookup-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        scratch_corpus(&dir);

        let lookup = advisories_for(&dir, "scratchy").unwrap();
        assert_eq!(
            lookup.advisories.len(),
            2,
            "both OSV mirrors of the RustSec number are dropped; the unlinked one stays"
        );
        assert_eq!(
            lookup.withdrawn, 1,
            "a retracted record is counted, not listed"
        );
        assert!(lookup
            .advisories
            .iter()
            .any(|a| a.id == "GHSA-only-here" && a.corpus == Corpus::Osv));
        let rustsec = lookup
            .advisories
            .iter()
            .find(|a| a.corpus == Corpus::RustSec)
            .unwrap();
        assert_eq!(rustsec.id, "RUSTSEC-2020-0159");
        assert_eq!(
            rustsec.severity.as_deref(),
            Some("GHSA HIGH"),
            "the mirror's one-word rating survives the dedup it loses"
        );

        let hits = advisories_for_version(&dir, "scratchy", "0.1.0").unwrap();
        assert_eq!(hits.len(), 2);
        assert!(
            hits.iter()
                .all(|(_, outcome)| matches!(outcome, Outcome::Vulnerable { .. })),
            "{:?}",
            hits.iter().map(|(a, o)| (&a.id, o)).collect::<Vec<_>>()
        );

        let rendered = render_lookup(&lookup, Some("0.1.0"));
        assert!(rendered.contains("RUSTSEC-2020-0159"), "{rendered}");
        assert!(rendered.contains("AFFECTED"), "{rendered}");
        assert!(rendered.contains("revision 012345678"), "{rendered}");
        assert!(
            rendered.contains("withdrawn by their database and are not listed"),
            "{rendered}"
        );

        // A crate nobody has reported on is answered as such, with the caveat.
        let none = advisories_for(&dir, "never_heard_of_it").unwrap();
        assert!(none.is_empty());
        let text = render_lookup(&none, None);
        assert!(text.contains("no advisory"), "{text}");
        assert!(
            text.contains("not a statement that this crate is safe"),
            "{text}"
        );

        // A lockfile check reports only what is actually hit.
        let lock = "[[package]]\nname = \"scratchy\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.200\"\n";
        let found = check_lockfile(&dir, lock).unwrap();
        assert_eq!(
            found.len(),
            2,
            "both surviving records for the one crate that has any"
        );
        assert!(found.iter().all(|(package, ..)| package == "scratchy"));
        assert!(matches!(found[0].3, Outcome::Vulnerable { .. }));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The real corpora, on the real network, once: this is the only test here
    /// that reaches outside the machine, and it is what the recorded numbers come
    /// from. Run with `cargo test -p xencode-analysis-rs -- --ignored --test sync`.
    #[tokio::test]
    #[ignore = "clones advisory-db and downloads the 3.4 MB OSV dump"]
    async fn syncing_the_real_corpora_makes_lookups_work() {
        let dir = std::env::temp_dir().join("xencode-advisories-live-sync");
        let _ = std::fs::remove_dir_all(&dir);
        let report = sync(&dir).await.unwrap();
        assert!(
            report.info.rustsec_advisories > 1000,
            "rustsec carried only {}",
            report.info.rustsec_advisories
        );
        assert!(report.osv_bytes > 1_000_000, "osv zip was small");
        assert!(report.info.index_lines > report.info.osv_records);

        // The two cases this pass measured by hand, now through the tool.
        let hit = advisories_for_version(&dir, "chrono", "0.4.19").unwrap();
        assert!(
            hit.iter().any(
                |(a, o)| a.id == "RUSTSEC-2020-0159" && matches!(o, Outcome::Vulnerable { .. })
            ),
            "chrono 0.4.19 should be reported by RUSTSEC-2020-0159, got {} records",
            hit.len()
        );
        let clear = advisories_for_version(&dir, "chrono", "0.4.20").unwrap();
        assert!(
            !clear.iter().any(
                |(a, o)| a.id == "RUSTSEC-2020-0159" && matches!(o, Outcome::Vulnerable { .. })
            ),
            "chrono 0.4.20 must not be reported as vulnerable"
        );
        let lookup = advisories_for(&dir, "chrono").unwrap();
        let ids: Vec<&str> = lookup.advisories.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            ids.iter().collect::<std::collections::HashSet<_>>().len(),
            "one advisory printed under two corpora: {ids:?}"
        );
        let mirrored = lookup
            .advisories
            .iter()
            .find(|a| a.id == "RUSTSEC-2020-0159" && a.corpus == Corpus::RustSec)
            .expect("the RustSec record for the segfault advisory");
        assert!(
            mirrored
                .url
                .as_deref()
                .is_some_and(|url| url.contains("chronotope/chrono")),
            "the record kept for a mirrored advisory is the curated one, not osv.dev: {:?}",
            mirrored.url
        );

        // tokio's RUSTSEC-2021-0072 has no CVSS vector of its own; its OSV mirror
        // carries a GHSA rating, and that rating is what a reader should see.
        let tokio = advisories_for(&dir, "tokio").unwrap();
        let tokio_record = tokio
            .advisories
            .iter()
            .find(|a| a.id == "RUSTSEC-2021-0072")
            .expect("tokio's empty vector / unpolled audit");
        assert_eq!(tokio_record.corpus, Corpus::RustSec);
        assert_eq!(
            tokio_record.severity.as_deref(),
            Some("GHSA MODERATE"),
            "the mirror's rating should survive the dedup that drops the mirror"
        );

        // Every RustSec file in the clone has to parse; a schema change is a
        // failure of this tool, not a silent skip.
        let mut files = 0;
        let mut unparsed = Vec::new();
        let crates_dir = dir.join(RUSTSEC_SUBDIR).join("crates");
        for entry in std::fs::read_dir(&crates_dir).unwrap().flatten() {
            for file in std::fs::read_dir(entry.path()).unwrap().flatten() {
                let name = file.file_name().to_string_lossy().to_string();
                if !name.starts_with("RUSTSEC-") {
                    continue;
                }
                files += 1;
                let text = std::fs::read_to_string(file.path()).unwrap();
                if let Err(e) = parse_rustsec(&text) {
                    unparsed.push(format!("{name}: {e}"));
                }
            }
        }
        assert_eq!(
            unparsed,
            Vec::<String>::new(),
            "{files} files read, {} failed to parse",
            unparsed.len()
        );

        let mut osv_files = 0;
        let mut osv_bad = Vec::new();
        for entry in std::fs::read_dir(dir.join(OSV_SUBDIR)).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".json") {
                continue;
            }
            osv_files += 1;
            if let Err(e) = parse_osv(&std::fs::read_to_string(entry.path()).unwrap()) {
                osv_bad.push(format!("{name}: {e}"));
            }
        }
        assert_eq!(
            osv_bad,
            Vec::<String>::new(),
            "{osv_files} osv records read, {} failed to parse",
            osv_bad.len()
        );
        println!(
            "corpus: {} rustsec advisories, {} osv records, {} index lines, revision {}",
            report.info.rustsec_advisories,
            report.info.osv_records,
            report.info.index_lines,
            report.info.rustsec_revision
        );
        println!("{} bytes osv", report.osv_bytes);
    }
}
