//! Configuration drift as a deterministic check (`U-3`, W3 row).
//!
//! Every `env::var` / `env::var_os` key read in code, crossed with every key a
//! template declares. Both sides are machine facts — a key that is read and a
//! key that is listed — so the comparison never guesses, which is the whole
//! feature. Two wordings are load-bearing and both are enforced by tests:
//!
//! - A key read but undocumented is reported with its code reference and the
//!   list of sources searched. It is not called "required": nothing here knows
//!   whether the program needs it.
//! - A key documented but never read is reported as *unreferenced*, never as
//!   *unnecessary*. Deleting someone's documented key on a tool's say-so is how
//!   a rented machine breaks at 3 a.m.
//!
//! Well-known OS-provided keys (`HOME`, `PATH`, …) are a third bucket: provided
//! by the environment, not app config, so they are listed separately instead of
//! being reported as undocumented.
//!
//! The W11 row — a `doctor --json` surface — is not this module. This is the
//! graph; the surfacing is a separate placement and stays open.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A place in code that reads the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvRef {
    /// The key, e.g. `DATABASE_URL`.
    pub key: String,
    /// Repository-relative file.
    pub file: String,
    /// 1-based line number.
    pub line: u32,
    /// Whether the read panics when the key is absent (`.unwrap()`/`.expect(`).
    pub panics_when_absent: bool,
}

/// Keys the operating system provides, not the application.
///
/// Reading one of these without documenting it is normal, and reporting it as
/// "undocumented config" would be the guess this module exists to avoid.
pub fn is_os_provided(key: &str) -> bool {
    matches!(
        key,
        "HOME"
            | "PATH"
            | "USER"
            | "LOGNAME"
            | "SHELL"
            | "TERM"
            | "LANG"
            | "LC_ALL"
            | "PWD"
            | "OLDPWD"
            | "HOSTNAME"
            | "TMPDIR"
            | "TEMP"
            | "TMP"
            | "XDG_DATA_HOME"
            | "XDG_CONFIG_HOME"
            | "XDG_CACHE_HOME"
            | "XDG_STATE_HOME"
            | "XDG_RUNTIME_DIR"
            | "CARGO_HOME"
            | "RUSTUP_HOME"
            | "CI"
            // Signing and agent plumbing provided by the session, not the app.
            | "GPG_TTY"
            | "GNUPGHOME"
            | "GPG_AGENT_INFO"
            | "SSH_AUTH_SOCK"
            | "SSH_AGENT_PID"
            // Windows provides these the same way Unix provides HOME.
            | "USERPROFILE"
            | "HOMEDRIVE"
            | "HOMEPATH"
            | "SYSTEMROOT"
            | "SYSTEMDRIVE"
            | "WINDIR"
            | "COMPUTERNAME"
            | "USERNAME"
            | "APPDATA"
            | "LOCALAPPDATA"
            | "COMSPEC"
            | "PATHEXT"
            | "OS"
    )
}

/// Whether a path holds tests rather than shipped code.
fn is_test_path(relative: &str) -> bool {
    relative.starts_with("tests/")
        || relative.contains("/tests/")
        || relative.ends_with("_test.rs")
        || relative.ends_with("/tests.rs")
}

/// Extract every `env::var("KEY")` / `env::var_os("KEY")` from Rust sources.
///
/// Test paths are skipped: a key read only by tests is not configuration the
/// program needs at startup, and reporting it would send a reader after a key
/// that ships nowhere.
pub fn extract_env_refs(root: &Path) -> Vec<EnvRef> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path.extension().map(|e| e != "rs").unwrap_or(true) {
                continue;
            }
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            let relative = relative.to_string_lossy().replace('\\', "/");
            if is_test_path(&relative) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (index, line) in text.lines().enumerate() {
                // A key mentioned in a comment is not read. Without this the
                // module's own doc comment reports KEY as configuration.
                let stripped = line.trim_start();
                if stripped.starts_with("//") || stripped.starts_with('*') {
                    continue;
                }
                for key in env_keys_in_line(line) {
                    let rest = &line[line.find(&key).unwrap_or(0)..];
                    out.push(EnvRef {
                        key,
                        file: relative.clone(),
                        line: index as u32 + 1,
                        panics_when_absent: rest.contains(".unwrap()") || rest.contains(".expect("),
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then(a.file.cmp(&b.file))
            .then(a.line.cmp(&b.line))
    });
    out
}

/// Keys read on one line of Rust source.
fn env_keys_in_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for func in ["env::var", "env::var_os"] {
        let mut search = line;
        while let Some(start) = search.find(func) {
            let after = &search[start + func.len()..];
            let after = after.trim_start();
            if let Some(rest) = after.strip_prefix('(') {
                let rest = rest.trim_start();
                if let Some(quoted) = rest.strip_prefix('"') {
                    if let Some(end) = quoted.find('"') {
                        let key = &quoted[..end];
                        if !key.is_empty() && !out.iter().any(|k: &String| k == key) {
                            out.push(key.to_string());
                        }
                    }
                }
            }
            search = &search[start + func.len()..];
        }
    }
    out
}

/// Template files consulted, in order. The searched list is part of every
/// report, so "absent from every known source" names what was actually looked
/// at rather than gesturing at the unknown.
pub const TEMPLATE_FILES: &[&str] = &[".env.example", ".env.template"];

/// Keys a template file declares, plus which templates were found.
#[derive(Debug, Clone, Default)]
pub struct Templates {
    /// Declared keys.
    pub keys: BTreeSet<String>,
    /// Templates that existed and were read.
    pub found: Vec<String>,
}

/// Read every template that exists. A missing template is not an error — it is
/// itself the finding that nothing documents the configuration.
pub fn read_templates(root: &Path) -> Templates {
    let mut out = Templates::default();
    for name in TEMPLATE_FILES {
        let Ok(text) = std::fs::read_to_string(root.join(name)) else {
            continue;
        };
        out.found.push(name.to_string());
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("export ") {
                let line = line.strip_prefix("export ").unwrap_or(line).trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
            }
            if let Some((key, _)) = line.split_once('=') {
                let key = key.trim();
                if !key.is_empty()
                    && key
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
                {
                    out.keys.insert(key.to_string());
                }
            }
        }
    }
    out
}

/// The drift report: machine facts, no requiredness claims.
#[derive(Debug, Clone, Default)]
pub struct DriftReport {
    /// Read in code, absent from every template searched.
    pub undocumented: Vec<EnvRef>,
    /// Declared in a template, read nowhere. Unreferenced — never unnecessary.
    pub unreferenced: Vec<String>,
    /// OS-provided keys, listed separately so they are never misreported.
    pub os_provided: Vec<EnvRef>,
    /// Reads that panic when the key is absent, outside tests.
    pub panicking: Vec<EnvRef>,
    /// Templates that were actually read.
    pub sources_searched: Vec<String>,
}

/// Cross the references with the templates.
pub fn compare(refs: &[EnvRef], templates: &Templates) -> DriftReport {
    let mut report = DriftReport {
        sources_searched: templates.found.clone(),
        ..DriftReport::default()
    };
    let read: BTreeSet<&str> = refs.iter().map(|r| r.key.as_str()).collect();
    for r in refs {
        if is_os_provided(&r.key) {
            report.os_provided.push(r.clone());
        } else if !templates.keys.contains(&r.key) {
            report.undocumented.push(r.clone());
        }
        if r.panics_when_absent && !is_os_provided(&r.key) {
            report.panicking.push(r.clone());
        }
    }
    for key in &templates.keys {
        if !read.contains(key.as_str()) {
            report.unreferenced.push(key.clone());
        }
    }
    report
}

/// The one-line verdict vocabulary. `unreferenced` is deliberately never
/// `unnecessary`, and nothing here is ever `required`.
pub fn verdict_word(kind: &str) -> &'static str {
    match kind {
        "undocumented" => "read-but-undocumented",
        "unreferenced" => "documented-but-unreferenced",
        "os" => "os-provided",
        "panicking" => "panics-when-absent",
        _ => "unknown",
    }
}

/// Group references by key, preserving file:line order.
pub fn by_key(refs: &[EnvRef]) -> BTreeMap<&str, Vec<&EnvRef>> {
    let mut map: BTreeMap<&str, Vec<&EnvRef>> = BTreeMap::new();
    for r in refs {
        map.entry(r.key.as_str()).or_default().push(r);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Tree(std::path::PathBuf);

    impl Tree {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("xe-env-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(self, rel: &str, body: &str) -> Self {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(body.as_bytes()).unwrap();
            drop(f);
            self
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_new_key_absent_from_the_template_is_reported_with_its_reference() {
        let tree = Tree::new("newkey")
            .file(
                "src/main.rs",
                "fn main() {\n    let _ = std::env::var(\"NEW_KEY\").unwrap();\n}\n",
            )
            .file(".env.example", "OLD_KEY=1\n");
        let refs = extract_env_refs(tree.path());
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].key, "NEW_KEY");
        assert_eq!(refs[0].file, "src/main.rs");
        assert_eq!(refs[0].line, 2);
        assert!(refs[0].panics_when_absent);
        let templates = read_templates(tree.path());
        assert_eq!(templates.found, vec![".env.example".to_string()]);
        let report = compare(&refs, &templates);
        assert_eq!(report.undocumented.len(), 1);
        assert_eq!(report.panicking.len(), 1);
    }

    #[test]
    fn a_documented_key_read_nowhere_is_unreferenced_never_unnecessary() {
        let tree = Tree::new("unref")
            .file("src/main.rs", "fn main() {}\n")
            .file(".env.example", "STALE_KEY=1\n");
        let report = compare(&extract_env_refs(tree.path()), &read_templates(tree.path()));
        assert_eq!(report.unreferenced, vec!["STALE_KEY".to_string()]);
        assert_eq!(verdict_word("unreferenced"), "documented-but-unreferenced");
        assert_ne!(verdict_word("unreferenced"), "unnecessary");
    }

    #[test]
    fn os_keys_are_listed_separately_not_reported() {
        let tree = Tree::new("oskeys").file(
            "src/main.rs",
            "fn main() {\n    let _ = std::env::var_os(\"HOME\");\n    let _ = std::env::var(\"PATH\").unwrap_or_default();\n}\n",
        );
        let report = compare(&extract_env_refs(tree.path()), &read_templates(tree.path()));
        assert!(report.undocumented.is_empty(), "OS keys are not app config");
        assert_eq!(report.os_provided.len(), 2);
        assert!(
            report.panicking.is_empty(),
            "OS keys never count as panicking reads"
        );
        assert!(
            report.sources_searched.is_empty(),
            "no template existed, and the report says so"
        );
    }

    #[test]
    fn test_paths_do_not_count_as_configuration() {
        let tree = Tree::new("testpaths").file(
            "tests/integration.rs",
            "fn t() {\n    let _ = std::env::var(\"TEST_ONLY\").unwrap();\n}\n",
        );
        let refs = extract_env_refs(tree.path());
        assert!(refs.is_empty(), "test-only reads ship nowhere");
    }

    #[test]
    fn a_key_in_a_comment_is_not_read() {
        // The module's own doc comment tripped this: prose mentioning a key is
        // not a read, and reporting it would be a false positive by construction.
        let tree = Tree::new("comments").file(
            "src/a.rs",
            "/// Reads `env::var(\"COMMENT_KEY\")` when set.\nfn f() {}\n",
        );
        assert!(extract_env_refs(tree.path()).is_empty());
    }

    #[test]
    fn session_plumbing_is_os_provided() {
        let tree = Tree::new("plumb").file(
            "src/a.rs",
            "fn f() {\n    let _ = std::env::var(\"GPG_TTY\");\n    let _ = std::env::var(\"GNUPGHOME\");\n}\n",
        );
        let report = compare(&extract_env_refs(tree.path()), &read_templates(tree.path()));
        assert!(report.undocumented.is_empty());
        assert_eq!(report.os_provided.len(), 2);
    }

    #[test]
    fn windows_home_vars_are_os_provided() {
        let tree = Tree::new("win").file(
            "src/a.rs",
            "fn f() {\n    let _ = std::env::var_os(\"USERPROFILE\");\n}\n",
        );
        let report = compare(&extract_env_refs(tree.path()), &read_templates(tree.path()));
        assert!(report.undocumented.is_empty());
        assert_eq!(report.os_provided.len(), 1);
    }

    #[test]
    fn both_var_forms_are_read() {
        let tree = Tree::new("forms").file(
            "src/a.rs",
            "fn f() {\n    let _ = std::env::var(\"A_KEY\");\n    let _ = std::env::var_os(\"B_KEY\").expect(\"set it\");\n}\n",
        );
        let refs = extract_env_refs(tree.path());
        assert_eq!(refs.len(), 2);
        assert!(refs
            .iter()
            .any(|r| r.key == "A_KEY" && !r.panics_when_absent));
        assert!(refs
            .iter()
            .any(|r| r.key == "B_KEY" && r.panics_when_absent));
    }
}
