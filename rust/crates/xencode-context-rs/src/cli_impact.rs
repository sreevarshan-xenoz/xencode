//! Impact edges for CLI commands and documentation manuals (AE-4).
//!
//! When `main.rs` or a CLI entry-point changes, changing or renaming a subcommand
//! affects the clap `Commands` enum, `--help` output, and the documentation manuals
//! (`README.md`, `CLI_GUIDE.md`, `QUICK_START.md`) that document the commands.
//!
//! This module parses the clap `Commands` enum variants in `main.rs`, finds the
//! manual lines that document each command (anchored on backticked command names,
//! command tables, and variant names to avoid false positive word matches), and
//! reports stale manual references when a documented command does not match any
//! active variant in `Commands`.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// One reference to a CLI command in a documentation manual.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandDocRef {
    pub file: String,
    pub line: usize,
    pub text: String,
}

/// One CLI command defined in the clap `Commands` enum and its manual references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandImpact {
    /// The clap enum variant identifier (e.g. "Agents", "Scan", "ReleaseNotes").
    pub variant: String,
    /// The kebab-case subcommand name (e.g. "agents", "scan", "release-notes").
    pub subcommand: String,
    /// Line in the source file where the variant is defined (1-indexed).
    pub line: usize,
    /// References in manuals that document this command.
    pub doc_refs: Vec<CommandDocRef>,
}

/// A manual reference to a CLI command that no longer exists in `Commands`
/// (e.g. after a variant was renamed or removed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleDocRef {
    pub file: String,
    pub line: usize,
    pub subcommand: String,
    pub text: String,
}

/// The CLI command impact report for a file defining CLI subcommands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliImpact {
    pub source_file: String,
    pub commands: Vec<CommandImpact>,
    pub stale_docs: Vec<StaleDocRef>,
}

impl CliImpact {
    /// Total count of defined clap command variants.
    pub fn variants_count(&self) -> usize {
        self.commands.len()
    }

    /// Total count of manual references across all commands.
    pub fn doc_refs_count(&self) -> usize {
        self.commands.iter().map(|c| c.doc_refs.len()).sum()
    }

    /// Commands with at least one documentation reference.
    pub fn documented_commands(&self) -> Vec<&CommandImpact> {
        self.commands.iter().filter(|c| !c.doc_refs.is_empty()).collect()
    }

    /// Commands with no documentation references in any manual.
    pub fn undocumented_commands(&self) -> Vec<&CommandImpact> {
        self.commands.iter().filter(|c| c.doc_refs.is_empty()).collect()
    }
}

/// Convert a PascalCase identifier to kebab-case (clap's default subcommand naming).
pub fn to_kebab_case(s: &str) -> String {
    let mut result = String::new();
    let mut prev_is_upper = false;
    let mut chars = s.chars().peekable();
    let mut i = 0;
    while let Some(c) = chars.next() {
        if c.is_uppercase() {
            let next_is_lower = chars.peek().is_some_and(|next| next.is_lowercase());
            if i > 0 && (!prev_is_upper || next_is_lower) {
                result.push('-');
            }
            result.push(c.to_ascii_lowercase());
            prev_is_upper = true;
        } else {
            result.push(c);
            prev_is_upper = false;
        }
        i += 1;
    }
    result
}

/// A parsed variant from the `enum Commands` definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedVariant {
    pub variant: String,
    pub subcommand: String,
    pub line: usize,
}

/// Parse variants from `enum Commands` in Rust source text.
pub fn parse_commands_enum(content: &str) -> Vec<ParsedVariant> {
    let mut variants = Vec::new();
    let mut in_enum = false;
    let mut depth = 0;
    let mut has_opened = false;

    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let trimmed = line.trim();

        if !in_enum
            && (trimmed.starts_with("enum Commands")
                || trimmed.contains("enum Commands ")
                || trimmed.contains("enum Commands{"))
        {
            in_enum = true;
        }

        if in_enum {
            let open_braces = line.matches('{').count();
            let close_braces = line.matches('}').count();

            // At depth 1 inside enum Commands, extract variant identifiers
            if depth == 1
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("///")
                && !trimmed.starts_with("#[")
                && !trimmed.is_empty()
            {
                let first_word = trimmed
                    .split(|c: char| c.is_whitespace() || c == '{' || c == '(' || c == ',')
                    .next()
                    .unwrap_or("");
                if !first_word.is_empty()
                    && first_word.chars().next().is_some_and(|c| c.is_uppercase())
                    && first_word.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && first_word != "Some"
                    && first_word != "None"
                    && first_word != "Ok"
                    && first_word != "Err"
                    && first_word != "Commands"
                {
                    variants.push(ParsedVariant {
                        variant: first_word.to_string(),
                        subcommand: to_kebab_case(first_word),
                        line: line_num,
                    });
                }
            }

            depth = (depth + open_braces).saturating_sub(close_braces);
            if open_braces > 0 {
                has_opened = true;
            }
            if has_opened && depth == 0 {
                // Completed enum Commands definition
                break;
            }
        }
    }
    variants
}

/// Discover markdown manuals in the repository.
pub fn find_manual_files(repo_root: &Path) -> Vec<PathBuf> {
    let mut manuals = Vec::new();
    let candidate_names = [
        "README.md",
        "CLI_GUIDE.md",
        "QUICK_START.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
    ];
    for name in &candidate_names {
        let path = repo_root.join(name);
        if path.is_file() {
            manuals.push(path);
        }
    }
    let docs_dir = repo_root.join("docs");
    if docs_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&docs_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().is_some_and(|ext| ext == "md") {
                    manuals.push(p);
                }
            }
        }
    }
    manuals.sort();
    manuals.dedup();
    manuals
}

/// Find repository root from a start path (looking for README.md or .git).
pub fn find_repo_root(start: &Path) -> PathBuf {
    let mut cur = start.to_path_buf();
    loop {
        if cur.join("README.md").is_file() || cur.join(".git").exists() {
            return cur;
        }
        if let Some(parent) = cur.parent() {
            cur = parent.to_path_buf();
        } else {
            return start.to_path_buf();
        }
    }
}

fn is_placeholder_or_non_command(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "subcommand"
            | "subcommands"
            | "command"
            | "commands"
            | "cmd"
            | "flags"
            | "flag"
            | "options"
            | "option"
            | "args"
            | "arg"
            | "path"
            | "target"
            | "filter"
            | "file"
            | "prompt"
            | "task"
            | "cli"
            | "rs"
            | "xencode"
            | "help"
            | "version"
    )
}

/// Extract referenced subcommands from a markdown line.
///
/// Anchors strictly on backticked command calls (e.g. `` `xencode foo` ``),
/// `Commands::<Variant>` enum references, and command list enumerations,
/// avoiding prose and arbitrary table false positives.
pub fn extract_subcommand_references(line: &str) -> Vec<String> {
    let mut subcmds = Vec::new();
    let trimmed = line.trim();

    // Check backticked spans
    let mut rest = trimmed;
    while let Some(start) = rest.find('`') {
        let after_start = &rest[start + 1..];
        let Some(end) = after_start.find('`') else {
            break;
        };
        let code = after_start[..end].trim();
        rest = &after_start[end + 1..];

        // 1. Explicit `xencode <subcommand>`
        if let Some(cmd_part) = code.strip_prefix("xencode ") {
            let first_token = cmd_part.split_whitespace().next().unwrap_or("");
            if !first_token.starts_with('-')
                && !first_token.starts_with('<')
                && !first_token.starts_with('[')
            {
                let cmd_word = first_token
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                    .to_ascii_lowercase();
                if !cmd_word.is_empty() && !is_placeholder_or_non_command(&cmd_word) {
                    subcmds.push(cmd_word);
                }
            }
        }
        // 2. Explicit `Commands::<Variant>`
        else if let Some(variant_part) = code.strip_prefix("Commands::") {
            let variant_name = variant_part
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            if !variant_name.is_empty()
                && variant_name.chars().next().is_some_and(|c| c.is_uppercase())
            {
                subcmds.push(to_kebab_case(variant_name));
            }
        }
        // 3. Command listing context in README: "- CLI with ... among them: `scan`, ..."
        else if trimmed.contains("subcommands, among them:")
            || trimmed.contains("subcommands:")
        {
            let clean = code.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
            if !clean.is_empty()
                && !clean.starts_with('-')
                && !clean.starts_with('<')
                && clean.chars().all(|c| c.is_lowercase() || c == '-')
                && !is_placeholder_or_non_command(clean)
            {
                subcmds.push(clean.to_string());
            }
        }
    }

    subcmds.sort();
    subcmds.dedup();
    subcmds
}

/// Scan documentation manuals and match against known CLI subcommands.
pub fn scan_manuals_for_commands(
    repo_root: &Path,
    manuals: &[PathBuf],
    known_commands: &[ParsedVariant],
) -> (Vec<CommandImpact>, Vec<StaleDocRef>) {
    let known_map: BTreeMap<String, &ParsedVariant> = known_commands
        .iter()
        .map(|v| (v.subcommand.clone(), v))
        .collect();
    let known_set: BTreeSet<String> = known_map.keys().cloned().collect();

    let mut doc_refs_by_subcmd: BTreeMap<String, Vec<CommandDocRef>> = BTreeMap::new();
    let mut stale_docs = Vec::new();

    for manual in manuals {
        let Ok(content) = std::fs::read_to_string(manual) else {
            continue;
        };
        let rel_file = manual
            .strip_prefix(repo_root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| manual.display().to_string());

        let mut in_table = false;
        let mut command_col_idx: Option<usize> = None;

        for (line_idx, line) in content.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            let mut line_subcmds = extract_subcommand_references(line);

            if trimmed.starts_with('|') && trimmed.ends_with('|') {
                if !in_table {
                    // Header row
                    in_table = true;
                    command_col_idx = None;
                    let cells: Vec<&str> = trimmed
                        .trim_matches('|')
                        .split('|')
                        .map(|c| c.trim())
                        .collect();
                    for (idx, cell) in cells.iter().enumerate() {
                        let lower = cell.to_ascii_lowercase();
                        if lower == "command" || lower == "subcommand" {
                            command_col_idx = Some(idx);
                            break;
                        }
                    }
                } else if trimmed.contains("---") {
                    // Separator row: retain header info
                } else if let Some(cmd_col) = command_col_idx {
                    // Body row in a verified command table
                    let cells: Vec<&str> = trimmed
                        .trim_matches('|')
                        .split('|')
                        .map(|c| c.trim())
                        .collect();
                    if let Some(cell) = cells.get(cmd_col) {
                        let c_trimmed = cell.trim();
                        // If cell does not contain "xencode ", check if it is a bare backticked command
                        if !c_trimmed.contains("xencode ") {
                            if let Some(code) = c_trimmed.strip_prefix('`') {
                                if let Some(end) = code.find('`') {
                                    let sub = code[..end]
                                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                                        .to_ascii_lowercase();
                                    if !sub.is_empty()
                                        && !is_placeholder_or_non_command(&sub)
                                        && !sub.starts_with('-')
                                        && !sub.starts_with('<')
                                        && sub.chars().all(|c| c.is_lowercase() || c == '-')
                                    {
                                        line_subcmds.push(sub);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                in_table = false;
                command_col_idx = None;
            }

            line_subcmds.sort();
            line_subcmds.dedup();

            for subcmd in line_subcmds {
                if known_set.contains(&subcmd) {
                    doc_refs_by_subcmd
                        .entry(subcmd)
                        .or_default()
                        .push(CommandDocRef {
                            file: rel_file.clone(),
                            line: line_num,
                            text: line.trim().to_string(),
                        });
                } else {
                    // Documented command reference does not match any active clap variant
                    stale_docs.push(StaleDocRef {
                        file: rel_file.clone(),
                        line: line_num,
                        subcommand: subcmd,
                        text: line.trim().to_string(),
                    });
                }
            }
        }
    }

    let commands = known_commands
        .iter()
        .map(|v| {
            let refs = doc_refs_by_subcmd.remove(&v.subcommand).unwrap_or_default();
            CommandImpact {
                variant: v.variant.clone(),
                subcommand: v.subcommand.clone(),
                line: v.line,
                doc_refs: refs,
            }
        })
        .collect();

    (commands, stale_docs)
}

/// Analyze a CLI source file (e.g. `main.rs`) for clap commands and documentation impact.
pub fn detect_cli_impact(
    repo_root: &Path,
    target_path: &Path,
    symbol_filter: Option<&str>,
) -> Option<CliImpact> {
    let content = std::fs::read_to_string(target_path).ok()?;
    if !content.contains("enum Commands") {
        return None;
    }

    let parsed_variants = parse_commands_enum(&content);
    if parsed_variants.is_empty() {
        return None;
    }

    let manuals = find_manual_files(repo_root);
    let (mut commands, stale_docs) =
        scan_manuals_for_commands(repo_root, &manuals, &parsed_variants);

    if let Some(symbol) = symbol_filter {
        let sym_lower = symbol.to_ascii_lowercase();
        commands.retain(|c| {
            c.variant.eq_ignore_ascii_case(symbol) || c.subcommand == sym_lower
        });
    }

    let rel_source = target_path
        .strip_prefix(repo_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| target_path.display().to_string());

    Some(CliImpact {
        source_file: rel_source,
        commands,
        stale_docs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_kebab_case_converts_pascal_case_correctly() {
        assert_eq!(to_kebab_case("Scan"), "scan");
        assert_eq!(to_kebab_case("ReleaseNotes"), "release-notes");
        assert_eq!(to_kebab_case("Envcheck"), "envcheck");
        assert_eq!(to_kebab_case("Llamacpp"), "llamacpp");
        assert_eq!(to_kebab_case("Agents"), "agents");
        assert_eq!(to_kebab_case("Tui"), "tui");
    }

    #[test]
    fn parse_commands_enum_extracts_all_variants() {
        let source = r#"
#[derive(Subcommand)]
enum Commands {
    /// Scan a workspace and list all entries
    Scan {
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// List installed agents
    Agents {
        #[arg(long)]
        contract: bool,
    },

    /// Release notes generator
    ReleaseNotes {
        #[arg(long)]
        from: String,
    },

    Tui,
}
"#;
        let variants = parse_commands_enum(source);
        assert_eq!(variants.len(), 4);
        assert_eq!(variants[0].variant, "Scan");
        assert_eq!(variants[0].subcommand, "scan");
        assert_eq!(variants[1].variant, "Agents");
        assert_eq!(variants[1].subcommand, "agents");
        assert_eq!(variants[2].variant, "ReleaseNotes");
        assert_eq!(variants[2].subcommand, "release-notes");
        assert_eq!(variants[3].variant, "Tui");
        assert_eq!(variants[3].subcommand, "tui");
    }

    #[test]
    fn extract_subcommand_references_anchors_on_backticked_commands() {
        // Plain sentence talking about replay does NOT match:
        let plain = "This test demonstrates that we replay actions without mutating state.";
        assert!(extract_subcommand_references(plain).is_empty());

        // Backticked command matches:
        let doc1 = "Run `xencode replay` to inspect past sessions.";
        assert_eq!(extract_subcommand_references(doc1), vec!["replay"]);

        let doc2 = "### `xencode agents`\nList installed agents with `--contract`.";
        assert_eq!(extract_subcommand_references(doc2), vec!["agents"]);

        let doc3 = "- CLI with subcommands, among them: `scan`, `agents`, `release-notes`.";
        assert_eq!(
            extract_subcommand_references(doc3),
            vec!["agents", "release-notes", "scan"]
        );
    }

    #[test]
    fn renamed_variant_makes_corresponding_docs_row_appear_as_stale() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path();

        // 1. Create a CLI source file where `Agents` was renamed to `AgentRoster`:
        let main_rs = root.join("main.rs");
        std::fs::write(
            &main_rs,
            r#"
#[derive(Subcommand)]
enum Commands {
    Scan { path: String },
    AgentRoster { contract: bool },
}
"#,
        )
        .unwrap();

        // 2. Create a manual still referring to `xencode agents` and `xencode scan`:
        let manual = root.join("CLI_GUIDE.md");
        std::fs::write(
            &manual,
            r#"# Guide
Run `xencode scan` to scan directory.
Run `xencode agents` to inspect agents.
"#,
        )
        .unwrap();

        let parsed = parse_commands_enum(&std::fs::read_to_string(&main_rs).unwrap());
        let (commands, stale) = scan_manuals_for_commands(root, &[manual.clone()], &parsed);

        // Scan is documented:
        let scan_cmd = commands.iter().find(|c| c.subcommand == "scan").unwrap();
        assert_eq!(scan_cmd.doc_refs.len(), 1);
        assert_eq!(scan_cmd.doc_refs[0].file, "CLI_GUIDE.md");

        // AgentRoster is not yet documented under its new name:
        let roster_cmd = commands
            .iter()
            .find(|c| c.subcommand == "agent-roster")
            .unwrap();
        assert!(roster_cmd.doc_refs.is_empty());

        // But the old docs row for `agents` appears as STALE rather than absent:
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].subcommand, "agents");
        assert_eq!(stale[0].file, "CLI_GUIDE.md");
        assert_eq!(stale[0].line, 3);
    }

    #[test]
    fn command_table_and_variant_anchors_work() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path();

        let main_rs = root.join("main.rs");
        std::fs::write(
            &main_rs,
            r#"
#[derive(Subcommand)]
enum Commands {
    Query { prompt: String },
    Audit { path: Option<String> },
}
"#,
        )
        .unwrap();

        let readme = root.join("README.md");
        std::fs::write(
            &readme,
            r#"# README
| Area | Command | Purpose |
| --- | --- | --- |
| Query | `xencode query "..."` | Ask model |
| Audit | `audit` | Check server log |
| Old | `deprecated-cmd` | Gone |

Layout modes (NOT commands):
| Mode | Desc |
| `zen` | Zen view |
"#,
        )
        .unwrap();

        let parsed = parse_commands_enum(&std::fs::read_to_string(&main_rs).unwrap());
        let (commands, stale) = scan_manuals_for_commands(root, &[readme], &parsed);

        let query = commands.iter().find(|c| c.subcommand == "query").unwrap();
        assert_eq!(query.doc_refs.len(), 1);

        let audit = commands.iter().find(|c| c.subcommand == "audit").unwrap();
        assert_eq!(audit.doc_refs.len(), 1);

        // `deprecated-cmd` in Command column is stale:
        let stale_names: Vec<&str> = stale.iter().map(|s| s.subcommand.as_str()).collect();
        assert!(stale_names.contains(&"deprecated-cmd"));
        // `zen` in non-command table is NOT matched as command:
        assert!(!stale_names.contains(&"zen"));
    }
}
