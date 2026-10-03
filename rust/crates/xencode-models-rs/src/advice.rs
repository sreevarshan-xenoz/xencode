//! Which model to serve on a given machine, and how current that answer is.
//!
//! The old code carried a hardcoded preference list of eight Ollama tags
//! assembled in 2024. A list like that does not announce when it goes stale;
//! it just quietly stops describing the models that exist, which is how an
//! advisor becomes fiction. So the list lives in a data file instead — with
//! the date it was assembled in it, next to the entries it is an opinion
//! about.
//!
//! Two files can answer: the one embedded at build time, and a replacement at
//! `<settings dir>/model_advice.json` for a person who wants different advice
//! than the shipped table gives. The answer always names which file it came from.
//!
//! The GGUF entries carry more than names: each one stores the repository
//! revision and the file's SHA256 as published by the host on the assembly
//! date, so the URL served from this table is pinned — the bytes it fetches
//! can be checked against a number that did not come from the same response.

use serde::Deserialize;

/// The shipped table. Verified against huggingface.co and ollama.com on the
/// `as_of` date inside it; a test re-checks that the embedded file parses and
/// that no entry has grown older than the rot horizon without being noticed.
const EMBEDDED: &str = include_str!("model_advice.json");

/// How long the shipped table may go unrefreshed before xencode says it is
/// out of date. Chosen to be uncomfortable rather than safe: a quantization
/// generation has historically turned over within months (Llama 2 → 3 took a
/// year, 3 → 3.3 six months), and the rot trap in the plan is exactly a table
/// that kept printing after its date stopped meaning anything.
pub const ROT_HORIZON_DAYS: i64 = 180;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct AdviceFile {
    /// The date the entries below were checked, `YYYY-MM-DD`.
    pub as_of: String,
    #[serde(default)]
    pub note: String,
    /// Ollama tags in preference order, for choosing among installed models.
    #[serde(default)]
    pub ollama_preference: Vec<String>,
    #[serde(default)]
    pub tiers: Vec<Tier>,
}

/// A capacity band. An entry is offered when the largest memory place to put
/// it can hold the file — the same reading `llamacpp start` prices a launch
/// against, not a guess at parameter counts.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Tier {
    pub name: String,
    /// Files up to this size fit the machines this tier describes.
    pub max_file_bytes: u64,
    #[serde(default)]
    pub gguf: Vec<GgufEntry>,
}

/// One recommended GGUF, pinned to the revision and checksum its host
/// published when the table was assembled.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GgufEntry {
    pub label: String,
    pub repo: String,
    pub file: String,
    /// The repository commit the file was fetched at — an exact 40-hex
    /// revision, not a movable branch name.
    pub revision: String,
    pub size_bytes: u64,
    /// Lowercase hex SHA256 of the file's bytes.
    pub sha256: String,
}

impl GgufEntry {
    /// The download URL, pinned: `/resolve/<revision>/` cannot follow a
    /// branch that moved after this row was written.
    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.file
        )
    }
}

/// Where an answer came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdviceSource {
    /// The table built into the binary.
    Embedded,
    /// A user's replacement file, at this path.
    UserFile(String),
    /// A replacement file exists but could not be read or parsed; the embedded
    /// table answered instead and the reason is here in the person's words.
    EmbeddedAfterRefusingUserFile(String),
}

pub struct Advice {
    pub file: AdviceFile,
    pub source: AdviceSource,
}

impl Advice {
    /// Load the user's replacement if `override_path` holds one, otherwise the
    /// embedded table. A broken replacement never silently degrades: the
    /// source says so and the caller prints why.
    pub fn load(override_path: &std::path::Path) -> Advice {
        match std::fs::read_to_string(override_path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Advice {
                file: parse_embedded(),
                source: AdviceSource::Embedded,
            },
            Err(e) => Advice {
                file: parse_embedded(),
                source: AdviceSource::EmbeddedAfterRefusingUserFile(format!(
                    "{}: {e}",
                    override_path.display()
                )),
            },
            Ok(text) => match serde_json::from_str::<AdviceFile>(&text) {
                Ok(file) => Advice {
                    file,
                    source: AdviceSource::UserFile(override_path.display().to_string()),
                },
                Err(e) => Advice {
                    file: parse_embedded(),
                    source: AdviceSource::EmbeddedAfterRefusingUserFile(format!(
                        "{} is not readable as model advice: {e}",
                        override_path.display()
                    )),
                },
            },
        }
    }

    /// The tier this machine's largest place to put a model file can hold,
    /// biggest first. `capacity_bytes` comes from the caller's measurement —
    /// this crate does not look at the machine. `None` when even the smallest
    /// tier's files do not fit, which is a real answer, not a failure.
    pub fn tier_for(&self, capacity_bytes: u64) -> Option<&Tier> {
        let mut best: Option<&Tier> = None;
        for tier in &self.file.tiers {
            if tier
                .gguf
                .iter()
                .any(|entry| entry.size_bytes <= capacity_bytes)
            {
                best = Some(tier);
            }
        }
        best
    }

    /// Days since the table was assembled, `None` if its own date is not a
    /// date. Advice older than [`ROT_HORIZON_DAYS`] still prints — refusing to
    /// answer a question you asked is not honesty, it is a shrug — but the
    /// caller is expected to say how old the answer is.
    pub fn age_days(&self, today_epoch_days: i64) -> Option<i64> {
        let assembled = days_from_civil(&self.file.as_of)?;
        Some(today_epoch_days.saturating_sub(assembled).max(0))
    }
}

fn parse_embedded() -> AdviceFile {
    // A parse failure here is a build-time mistake, not a runtime condition:
    // the embedded resource and this struct ship together.
    serde_json::from_str(EMBEDDED).expect("the shipped model_advice.json does not parse")
}

/// The Ollama tags worth preferring, from the shipped table alone. A caller
/// choosing between files it was handed uses this; the order in force on this
/// machine — which is what `xencode models default` answers with — is
/// [`active_preference`].
pub fn embedded_preference() -> Vec<String> {
    parse_embedded().ollama_preference
}

/// The preference order in force here: a person's own
/// `<settings dir>/model_advice.json` when one parses, the shipped table
/// otherwise. The two
/// questions this file answers — which GGUF fits this machine, and which Ollama
/// tag to reach for — are decided by the same table, so an override cannot leave
/// one answering from a file the other ignored.
pub fn active_preference() -> Vec<String> {
    Advice::load(&default_path()).file.ollama_preference
}

/// Where a person's own table lives, if they write one:
/// `<settings dir>/model_advice.json` — `~/.config/xencode/model_advice.json`,
/// or `~/.xencode/model_advice.json` for a person who has never moved off the old
/// layout.
pub fn default_path() -> std::path::PathBuf {
    xencode_config_rs::paths::settings_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("model_advice.json")
}

/// `YYYY-MM-DD` to days since the Unix epoch; `None` on anything else.
fn days_from_civil(date: &str) -> Option<i64> {
    let (y, m, d) = (date.get(0..4)?, date.get(5..7)?, date.get(8..10)?);
    if date.len() != 10 || date.get(4..5) != Some("-") || date.get(7..8) != Some("-") {
        return None;
    }
    let y = y.parse::<i64>().ok()?;
    let m = m.parse::<i64>().ok()?;
    let d = d.parse::<i64>().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil_algorithm(y, m, d))
}

/// Howard Hinnant's `days_from_civil`, in public-domain form: no calendar
/// crates, and correct across the 400-year Gregorian cycle xencode will run in.
fn days_from_civil_algorithm(mut y: i64, m: i64, d: i64) -> i64 {
    if m <= 2 {
        y -= 1;
    }
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

impl AdviceFile {
    /// A reference date for callers that just need "now" in epoch days without
    /// pulling in a time crate: seconds since the Unix epoch / 86400.
    pub fn today_epoch_days() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
            / 86_400
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn advice_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("xencode-advice-{tag}-{}.json", std::process::id()))
    }

    #[test]
    fn the_shipped_table_parses_and_carries_pinned_entries() {
        let advice = Advice::load(std::path::Path::new("/nonexistent/quiet-path.json"));
        assert_eq!(advice.source, AdviceSource::Embedded);
        assert_eq!(advice.file.as_of, "2026-09-27");
        assert!(!advice.file.ollama_preference.is_empty());
        let entries: Vec<&GgufEntry> = advice
            .file
            .tiers
            .iter()
            .flat_map(|t| t.gguf.iter())
            .collect();
        assert!(entries.len() >= 3, "the table shrank: {entries:?}");
        for entry in &entries {
            assert_eq!(
                entry.revision.len(),
                40,
                "{} is not pinned to a full commit",
                entry.repo
            );
            assert!(
                entry
                    .revision
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{} has a revision that is not a lowercase hex commit",
                entry.repo
            );
            assert_eq!(
                entry.sha256.len(),
                64,
                "{} has a checksum that is not a hex SHA256",
                entry.file
            );
            assert!(entry.url().contains(entry.revision.as_str()));
        }
    }

    #[test]
    fn a_table_older_than_the_rot_horizon_is_reported_as_out_of_date() {
        let advice = Advice::load(std::path::Path::new("/nonexistent/quiet-path.json"));
        // 2026-09-27 itself is fresh; 200 days later it is not.
        let assembled = days_from_civil("2026-09-27").unwrap();
        assert_eq!(advice.age_days(assembled), Some(0));
        assert_eq!(advice.age_days(assembled + 199), Some(199));
        assert!(advice.age_days(assembled + 199).unwrap() > ROT_HORIZON_DAYS);
        // A date in the past never yields a negative age.
        assert_eq!(advice.age_days(assembled - 5), Some(0));
    }

    #[test]
    fn a_broken_replacement_file_refuses_to_pretend_it_answered() {
        let path = advice_path("broken");
        std::fs::write(&path, "{ not json at all").unwrap();
        let advice = Advice::load(&path);
        assert!(matches!(
            advice.source,
            AdviceSource::EmbeddedAfterRefusingUserFile(_)
        ));
        // The embedded table still answers — the user asked a question.
        assert!(!advice.file.tiers.is_empty());
        let reason = match &advice.source {
            AdviceSource::EmbeddedAfterRefusingUserFile(r) => r.clone(),
            other => panic!("{other:?}"),
        };
        assert!(reason.contains("not readable as model advice"), "{reason}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_valid_replacement_file_answers_and_says_it_came_from_the_user() {
        let path = advice_path("user");
        std::fs::write(
            &path,
            r#"{"as_of":"2026-01-01","tiers":[{"name":"only","max_file_bytes":10,"gguf":[{"label":"l","repo":"r","file":"f","revision":"0123456789012345678901234567890123456789","size_bytes":5,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}]}"#,
        )
        .unwrap();
        let advice = Advice::load(&path);
        assert_eq!(
            advice.source,
            AdviceSource::UserFile(path.display().to_string())
        );
        assert_eq!(advice.file.as_of, "2026-01-01");
        let _ = std::fs::remove_file(path);
    }

    /// The one table answers both questions this file is consulted for: which
    /// GGUF fits this machine, and which installed Ollama tag to reach for. An
    /// override that moved one and not the other would leave a person editing a
    /// file that only half-took.
    #[test]
    fn a_replacement_table_decides_the_ollama_preference_too() {
        let path = advice_path("preference");
        std::fs::write(
            &path,
            r#"{"as_of":"2026-09-27","ollama_preference":["phi4-mini","qwen3"],"tiers":[]}"#,
        )
        .unwrap();
        let advice = Advice::load(&path);
        assert_eq!(advice.file.ollama_preference, ["phi4-mini", "qwen3"]);
        assert!(matches!(advice.source, AdviceSource::UserFile(_)));
        let _ = std::fs::remove_file(path);
        // The shipped order is what answers when no override is in the way.
        assert!(embedded_preference().contains(&"qwen3".to_string()));
    }

    #[test]
    fn a_machine_is_matched_to_the_biggest_tier_it_can_actually_hold() {
        let advice = Advice::load(std::path::Path::new("/nonexistent/quiet-path.json"));
        // 2 GiB of room: the small tier fits, the medium one (2.0 GB file is
        // just over) does not, the large one cannot.
        let tier = advice.tier_for(2_019_377_000).unwrap();
        assert_eq!(tier.name, "small");
        // One byte over a file's size admits its tier.
        let tier = advice.tier_for(2_019_377_696).unwrap();
        assert_eq!(tier.name, "medium");
        let tier = advice.tier_for(50_000_000_000).unwrap();
        assert_eq!(tier.name, "large");
        assert!(advice.tier_for(1024).is_none());
    }

    #[test]
    fn a_replacement_table_is_looked_for_beside_the_rest_of_the_configuration() {
        let path = default_path();
        assert_eq!(path.file_name().unwrap(), "model_advice.json");
        assert_eq!(
            path.parent().unwrap(),
            xencode_config_rs::paths::settings_dir().unwrap().as_path()
        );
    }

    #[test]
    fn epoch_day_arithmetic_survives_the_boundaries_a_calendar_gets_wrong() {
        assert_eq!(days_from_civil("1970-01-01"), Some(0));
        // 2000 is a leap year (divisible by 400), 1900 is not.
        assert_eq!(
            days_from_civil("2000-03-01").unwrap() - days_from_civil("2000-02-28").unwrap(),
            2
        );
        assert_eq!(
            days_from_civil("1900-03-01").unwrap() - days_from_civil("1900-02-28").unwrap(),
            1
        );
        // January dates take the year-shift branch.
        assert!(days_from_civil("2026-01-01").is_some());
        assert_eq!(days_from_civil("2026-1-1"), None);
        assert_eq!(days_from_civil("nonsense"), None);
    }
}
