//! Skills (M-3): `SKILL.md` documents found on disk, and the one-line menu the
//! prompt shows for them.
//!
//! A skill is a directory holding a `SKILL.md`: frontmatter naming it and
//! saying when to use it, then the instructions themselves. Only the name and
//! the summary line reach the system prompt; the body is read on demand by the
//! `load_skill` tool. That split is the whole point — a directory of thirty
//! skills costs a prompt thirty short lines, not thirty documents, and the
//! model still gets the full text of whichever one the task calls for.
//!
//! Two places are scanned, project last so it wins on a name clash: the user's
//! `~/.xencode/skills/` and `<workspace>/.xencode/skills/`.
//!
//! There is no YAML parser in this workspace, and adding one to read two keys
//! would cost more than it saves, so the frontmatter is handled by
//! [`parse_frontmatter`] below: `key: value` scalars, quoted values, `|` and
//! `>` block scalars, and `- item` lists. A field it cannot read is dropped,
//! not an error — a skill with unusual frontmatter still contributes its name
//! and its instructions.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The one file name that makes a directory a skill.
pub const SKILL_FILE: &str = "SKILL.md";

/// How much of a description the menu keeps. Thirty skills at this cap are
/// about 1.5k characters, which a prompt can afford as a listing and cannot
/// afford as thirty documents.
pub const MENU_DESCRIPTION_MAX_CHARS: usize = 220;

/// Which of the two scanned roots a skill came from. Project wins a clash, so a
/// repository can pin its own version of a skill a user also has installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillScope {
    /// `~/.xencode/skills`
    Home,
    /// `<workspace>/.xencode/skills`
    Project,
}

impl SkillScope {
    pub fn label(self) -> &'static str {
        match self {
            SkillScope::Home => "user",
            SkillScope::Project => "project",
        }
    }
}

/// One loaded skill: what the menu says about it, and what `load_skill` hands
/// back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    /// The frontmatter `description`, kept whole; only
    /// [`SkillRuntime::menu`] truncates.
    pub description: String,
    /// The directory holding `SKILL.md`, so the instructions' own relative file
    /// references can be resolved.
    pub directory: PathBuf,
    pub file: PathBuf,
    /// Everything after the frontmatter, as written.
    pub instructions: String,
    pub scope: SkillScope,
    /// Set when the description came from the document's first line because the
    /// frontmatter carried none, so `/skills` can say the summary is borrowed.
    pub description_inferred: bool,
}

/// A skill file that was found and then not loaded, with the reason. `/skills`
/// lists these so a broken install is visible instead of silently absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedSkill {
    pub file: PathBuf,
    pub reason: String,
}

/// The skills this session has. Read once at startup like the plugins, so the
/// menu in the system prompt is byte-identical across turns — the precondition
/// for the cached prefix the context assembler depends on.
#[derive(Debug, Clone, Default)]
pub struct SkillRuntime {
    skills: Vec<Skill>,
    rejected: Vec<RejectedSkill>,
    home: PathBuf,
    project: PathBuf,
    /// Names where a project skill replaced a user skill of the same name.
    shadowed: Vec<String>,
}

impl SkillRuntime {
    /// Scan both roots. A directory that does not exist contributes nothing,
    /// which is the normal case rather than a failure.
    pub fn load(home: &Path, project: &Path) -> Self {
        let mut rejected = Vec::new();
        let mut scanned = Vec::new();
        collect_skills(home, SkillScope::Home, &mut scanned, &mut rejected);
        collect_skills(project, SkillScope::Project, &mut scanned, &mut rejected);
        // Name order keeps the menu byte-identical run to run.
        scanned.sort_by(|a, b| a.name.cmp(&b.name));
        let mut skills: Vec<Skill> = Vec::new();
        let mut shadowed = Vec::new();
        for skill in scanned {
            match skills.iter_mut().find(|e| e.name == skill.name) {
                // Collected second, so a clash means this is the project one
                // and it replaces rather than sitting beside the user's.
                Some(existing) if skill.scope == SkillScope::Project => {
                    if existing.scope == SkillScope::Home {
                        shadowed.push(existing.name.clone());
                    }
                    *existing = skill;
                }
                Some(_) => {}
                None => skills.push(skill),
            }
        }
        SkillRuntime {
            skills,
            rejected,
            home: home.to_path_buf(),
            project: project.to_path_buf(),
            shadowed,
        }
    }

    /// Nothing loaded, with the roots recorded for `/skills` to report.
    pub fn empty(home: PathBuf, project: PathBuf) -> Self {
        SkillRuntime {
            skills: Vec::new(),
            rejected: Vec::new(),
            home,
            project,
            shadowed: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }

    pub fn rejected(&self) -> &[RejectedSkill] {
        &self.rejected
    }

    /// Project skills that replaced a user skill of the same name.
    pub fn shadowed(&self) -> &[String] {
        &self.shadowed
    }

    pub fn home_dir(&self) -> &Path {
        &self.home
    }

    pub fn project_dir(&self) -> &Path {
        &self.project
    }

    /// Find a skill by name: exact first, then case-folded against either the
    /// declared name or the directory, because the model copies the name out of
    /// the menu and a stray case shift should not read as "no such skill".
    pub fn get(&self, name: &str) -> Option<&Skill> {
        let needle = name.trim();
        if needle.is_empty() {
            return None;
        }
        self.skills
            .iter()
            .find(|skill| skill.name == needle)
            .or_else(|| {
                let lowered = needle.to_lowercase();
                self.skills.iter().find(|skill| {
                    skill.name.to_lowercase() == lowered || directory_name(&skill.file) == lowered
                })
            })
    }

    /// The text injected into the system prompt: the skill menu, or `None` when
    /// nothing is installed. `None` rather than an empty block is what keeps a
    /// machine with no skills sending exactly the prompt bytes it sent before
    /// this existed.
    pub fn menu(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        let mut out = String::from(SKILL_MENU_HEAD);
        for skill in &self.skills {
            out.push_str(&menu_line(skill));
        }
        Some(out)
    }

    /// The one menu line a skill costs every turn, kept public so `/skills` can
    /// show what the listing costs against what the document holds.
    pub fn menu_line(&self, name: &str) -> Option<String> {
        self.get(name).map(menu_line)
    }

    /// The `load_skill` answer for one skill: its own instructions, preceded by
    /// where they came from so relative file names inside them are resolvable.
    pub fn render(&self, skill: &Skill) -> String {
        format!(
            "skill: {} ({})\ndirectory: {}\n\n{}",
            skill.name,
            skill.scope.label(),
            skill.directory.display(),
            skill.instructions.trim_end()
        )
    }
}

/// What the menu says about itself, so the model knows the listing is not the
/// instructions and that reading one is a tool call away. Kept short: this text
/// sits in front of every turn.
const SKILL_MENU_HEAD: &str = "## Available skills\n\
     Each line is a skill's name and when to use it. The listing is not the \
     instructions: before following a skill, call load_skill(name) with its name \
     written exactly below and read what comes back.\n";

fn menu_line(skill: &Skill) -> String {
    format!(
        "- {}: {}\n",
        skill.name,
        cut_description(&skill.description)
    )
}

/// Cut a description to [`MENU_DESCRIPTION_MAX_CHARS`] on a character boundary,
/// marking that it was cut. Whitespace is folded first, because a folded block
/// in frontmatter carries the author's line breaks.
pub fn cut_description(description: &str) -> String {
    let flat = description.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MENU_DESCRIPTION_MAX_CHARS {
        return flat;
    }
    let kept: String = flat.chars().take(MENU_DESCRIPTION_MAX_CHARS).collect();
    format!("{kept}…")
}

fn directory_name(skill_file: &Path) -> String {
    skill_file
        .parent()
        .and_then(|parent| parent.file_name())
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// Every `*/SKILL.md` directly under `dir`, in the given scope. A directory
/// without one is not a skill and is passed over without comment.
fn collect_skills(
    dir: &Path,
    scope: SkillScope,
    out: &mut Vec<Skill>,
    rejected: &mut Vec<RejectedSkill>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    for skill_dir in dirs {
        let file = skill_dir.join(SKILL_FILE);
        if !file.is_file() {
            continue;
        }
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(error) => {
                rejected.push(RejectedSkill {
                    file,
                    reason: format!("could not be read: {error}"),
                });
                continue;
            }
        };
        let (fields, body, note) = parse_frontmatter(&text);
        if body.trim().is_empty() {
            rejected.push(RejectedSkill {
                file,
                reason: "has no instructions after its frontmatter".to_string(),
            });
            continue;
        }
        if let Some(note) = note {
            rejected.push(RejectedSkill {
                file: file.clone(),
                reason: note,
            });
        }
        let (description, inferred) = match field(&fields, "description") {
            Some(text) if !text.trim().is_empty() => (text, false),
            _ => (
                body.lines()
                    .map(str::trim)
                    .find(|line| !line.is_empty() && !line.starts_with('#'))
                    .unwrap_or_default()
                    .to_string(),
                true,
            ),
        };
        let name = field(&fields, "name")
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| {
                skill_dir
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default()
            });
        out.push(Skill {
            name,
            description,
            directory: skill_dir,
            file,
            instructions: body,
            scope,
            description_inferred: inferred,
        });
    }
}

/// Normalize a frontmatter key so `allowed-tools` and `allowed_tools` are the
/// same field, which is how these documents are written in practice.
fn normalize_key(key: &str) -> String {
    key.trim().to_lowercase().replace('-', "_")
}

fn field(fields: &BTreeMap<String, String>, key: &str) -> Option<String> {
    fields.get(key).filter(|text| !text.is_empty()).cloned()
}

/// Split a `SKILL.md` into its frontmatter fields and its instructions.
///
/// Returns `(fields, body, note)`. The note is set when the document opened a
/// frontmatter block that never closes: the fields read so far are kept and the
/// whole file is treated as the instructions, because a skill that half-works
/// beats one that vanishes — and the note makes the half-working visible.
pub fn parse_frontmatter(text: &str) -> (BTreeMap<String, String>, String, Option<String>) {
    let mut fields = BTreeMap::new();
    let trimmed = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = trimmed.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (fields, trimmed.to_string(), None);
    }
    let mut body: Vec<&str> = Vec::new();
    let mut open = true;
    // The field whose continuation lines (more-indented, or `- item`) are still
    // being collected.
    let mut pending: Option<PendingField> = None;
    for line in lines {
        if !open {
            body.push(line);
            continue;
        }
        match line.trim() {
            "---" | "..." => {
                open = false;
                flush_pending(&mut fields, &mut pending);
                continue;
            }
            _ => {}
        }
        if let Some(field_line) = read_field_line(line) {
            flush_pending(&mut fields, &mut pending);
            if field_line.value.is_empty() && field_line.block {
                pending = Some(PendingField {
                    key: field_line.key,
                    lines: Vec::new(),
                });
            } else {
                fields.insert(field_line.key, field_line.value);
            }
            continue;
        }
        if let Some(entry) = pending.as_mut() {
            if line.starts_with([' ', '\t']) {
                entry.lines.push(line.trim().to_string());
            }
        }
    }
    if open {
        flush_pending(&mut fields, &mut pending);
        return (
            fields,
            trimmed.to_string(),
            Some(
                "frontmatter opens with `---` but never closes; its fields were \
                 read and the whole file is treated as the instructions"
                    .to_string(),
            ),
        );
    }
    (fields, body.join("\n"), None)
}

struct PendingField {
    key: String,
    lines: Vec<String>,
}

struct FieldLine {
    key: String,
    value: String,
    /// The value was `|`, `>` or empty, so the real text is on the lines below.
    block: bool,
}

/// `key: value` at the left margin. Indented lines, comments, list items and
/// anything without a colon are not fields.
fn read_field_line(line: &str) -> Option<FieldLine> {
    if line.is_empty() || line.starts_with([' ', '\t', '#', '-']) {
        return None;
    }
    let (raw_key, raw_value) = line.split_once(':')?;
    let key = normalize_key(raw_key);
    if key.is_empty() || key.contains(' ') {
        return None;
    }
    let value = raw_value.trim();
    if matches!(value, "|" | "|-" | ">" | ">-" | "") {
        return Some(FieldLine {
            key,
            value: String::new(),
            block: true,
        });
    }
    Some(FieldLine {
        key,
        value: unquote(value),
        block: false,
    })
}

/// Strip one layer of matching quotes, which is how a description containing a
/// colon survives being written after the key's own colon.
fn unquote(value: &str) -> String {
    let first = value.chars().next();
    if matches!(first, Some('"') | Some('\'')) && value.chars().last() == first && value.len() > 1 {
        return value[1..value.len() - 1].to_string();
    }
    value.to_string()
}

/// Join the collected lines of a block scalar or list into one value. Folded
/// (`>`) and literal (`|`) blocks are both flattened here: every field this
/// loader cares about, `name` and `description`, is one line of prose in
/// practice, and the menu needs one line out of it.
fn flush_pending(fields: &mut BTreeMap<String, String>, pending: &mut Option<PendingField>) {
    let Some(entry) = pending.take() else {
        return;
    };
    let value = entry
        .lines
        .iter()
        .map(|line| line.strip_prefix("- ").unwrap_or(line.as_str()).trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    fields.insert(entry.key, value);
}

/// The user's skill directory: `$XCODE_SKILLS_DIR` when set (tests, portable
/// installs), else `<config dir>/skills` — which is `~/.xencode/skills` unless
/// `$XCODE_CONFIG_DIR` moved it. Same override shape as `$XCODE_PLUGIN_DIR`.
pub fn default_home_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XCODE_SKILLS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    config_home().join("skills")
}

/// `<workspace>/.xencode/skills` — the project's own skills, which win over the
/// user's by name.
pub fn project_skills_dir(workspace: &Path) -> PathBuf {
    workspace.join(".xencode").join("skills")
}

fn config_home() -> PathBuf {
    if let Ok(dir) = std::env::var("XCODE_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".xencode")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one skill and returns its directory.
    fn install(root: &Path, dir_name: &str, text: &str) -> PathBuf {
        let skill = root.join(dir_name);
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(skill.join(SKILL_FILE), text).unwrap();
        skill
    }

    fn skill_text(name: &str, description: &str, body: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    }

    #[test]
    fn frontmatter_reads_name_description_and_leaves_the_body_whole() {
        let (fields, body, note) = parse_frontmatter(&skill_text(
            "pdf-forms",
            "Fill a PDF form.",
            "Step one.\nStep two.",
        ));
        assert_eq!(fields.get("name").unwrap(), "pdf-forms");
        assert_eq!(fields.get("description").unwrap(), "Fill a PDF form.");
        assert_eq!(body, "Step one.\nStep two.");
        assert_eq!(note, None);
    }

    #[test]
    fn a_folded_description_and_a_quoted_one_both_parse() {
        let (fields, _, _) = parse_frontmatter(
            "---\nname: \"commit-hygiene\"\ndescription: >\n  Write the message\n  like a sentence.\n---\nbody\n",
        );
        assert_eq!(fields.get("name").unwrap(), "commit-hygiene");
        assert_eq!(
            fields.get("description").unwrap(),
            "Write the message, like a sentence."
        );
    }

    #[test]
    fn a_list_value_is_joined_so_a_hyphenated_key_is_not_dropped() {
        let (fields, _, _) = parse_frontmatter(
            "---\nname: t\nallowed-tools:\n  - read_file\n  - run_command\ndescription: d\n---\nb\n",
        );
        assert_eq!(
            fields.get("allowed_tools").unwrap(),
            "read_file, run_command"
        );
    }

    #[test]
    fn frontmatter_that_never_closes_keeps_its_fields_and_says_so() {
        let (fields, body, note) =
            parse_frontmatter("---\nname: unclosed\ndescription: no fence here\nno body fence\n");
        assert_eq!(fields.get("name").unwrap(), "unclosed");
        assert!(body.contains("no fence here"), "{body}");
        let note = note.expect("the unclosed block must be reported");
        assert!(note.contains("never closes"), "{note}");
    }

    #[test]
    fn a_document_with_no_frontmatter_still_loads_on_its_directory_name() {
        let home = tempfile::tempdir().unwrap();
        install(
            home.path(),
            "plain-skill",
            "# Title\nSummarise nothing, just do the thing.\n",
        );
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        let skill = runtime
            .get("plain-skill")
            .expect("directory name is the name");
        assert_eq!(skill.scope, SkillScope::Home);
        assert!(skill.description_inferred);
        assert_eq!(skill.description, "Summarise nothing, just do the thing.");
    }

    #[test]
    fn both_roots_are_scanned_and_the_project_wins_a_name_clash() {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        install(
            home.path(),
            "shared",
            &skill_text("shared", "user version", "USER BODY"),
        );
        install(
            home.path(),
            "only-home",
            &skill_text("only-home", "home only", "HOME BODY"),
        );
        let project_skills = project.path().join(".xencode").join("skills");
        install(
            &project_skills,
            "shared",
            &skill_text("shared", "project version", "PROJECT BODY"),
        );
        let runtime = SkillRuntime::load(home.path(), &project_skills);
        assert_eq!(runtime.len(), 2, "a clash replaces, it does not duplicate");
        let shared = runtime.get("shared").unwrap();
        assert_eq!(shared.scope, SkillScope::Project);
        assert_eq!(shared.description, "project version");
        assert_eq!(runtime.shadowed(), ["shared".to_string()]);
    }

    #[test]
    fn a_directory_without_a_skill_file_is_not_a_skill() {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join("not-a-skill")).unwrap();
        std::fs::write(home.path().join("not-a-skill").join("README.md"), "hi").unwrap();
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        assert!(runtime.is_empty());
        assert!(runtime.rejected().is_empty());
    }

    #[test]
    fn a_skill_with_no_instructions_is_reported_rather_than_loaded() {
        let home = tempfile::tempdir().unwrap();
        install(
            home.path(),
            "hollow",
            &skill_text("hollow", "nothing inside", ""),
        );
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        assert!(runtime.is_empty());
        assert_eq!(runtime.rejected().len(), 1);
        assert!(runtime.rejected()[0].reason.contains("no instructions"));
    }

    #[test]
    fn the_menu_lists_every_skill_and_no_skill_body() {
        let home = tempfile::tempdir().unwrap();
        install(
            home.path(),
            "alpha",
            &skill_text("alpha", "First skill.", "THE ALPHA INSTRUCTIONS"),
        );
        install(
            home.path(),
            "beta",
            &skill_text("beta", "Second skill.", "THE BETA INSTRUCTIONS"),
        );
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        let menu = runtime.menu().unwrap();
        assert!(menu.contains("- alpha: First skill."), "{menu}");
        assert!(menu.contains("- beta: Second skill."), "{menu}");
        assert!(!menu.contains("THE ALPHA INSTRUCTIONS"), "{menu}");
        assert!(!menu.contains("THE BETA INSTRUCTIONS"), "{menu}");
        assert!(menu.contains("load_skill"), "the menu must name the tool");
    }

    #[test]
    fn no_skills_means_no_menu_at_all() {
        let home = tempfile::tempdir().unwrap();
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        assert_eq!(runtime.menu(), None);
        assert!(runtime.is_empty());
    }

    /// The done-when for M-3 measured without a model: thirty installed skills
    /// add thirty lines and nothing else.
    #[test]
    fn thirty_skills_cost_the_prompt_a_menu_not_thirty_bodies() {
        let home = tempfile::tempdir().unwrap();
        for index in 0..30 {
            install(
                home.path(),
                &format!("skill-{index:02}"),
                &skill_text(
                    &format!("skill-{index:02}"),
                    &format!("Number {index}, with a description short enough to fit one line."),
                    &"read this line\n".repeat(60),
                ),
            );
        }
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        assert_eq!(runtime.len(), 30);
        let menu = runtime.menu().unwrap();
        let body_chars: usize = runtime
            .skills()
            .iter()
            .map(|skill| skill.instructions.chars().count())
            .sum();
        assert_eq!(
            menu.lines().count(),
            2 + 30,
            "the heading is two lines and each skill costs exactly one"
        );
        assert!(
            menu.chars().count() * 5 < body_chars,
            "menu {} chars vs bodies {body_chars} chars",
            menu.chars().count()
        );
    }

    #[test]
    fn a_long_description_is_cut_on_a_character_boundary_and_marked() {
        let multibyte = "é".repeat(MENU_DESCRIPTION_MAX_CHARS + 20);
        let cut = cut_description(&multibyte);
        assert!(cut.ends_with('…'), "{}", cut.len());
        assert_eq!(
            cut.chars().count(),
            MENU_DESCRIPTION_MAX_CHARS + 1,
            "the cap plus the mark that says it was cut"
        );
        assert!(cut.chars().nth(MENU_DESCRIPTION_MAX_CHARS - 1) == Some('é'));
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn lookup_tolerates_case_and_matches_the_directory_name() {
        let home = tempfile::tempdir().unwrap();
        install(
            home.path(),
            "kebab-case",
            &skill_text("Kebab-Case", "Named differently.", "BODY"),
        );
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        assert!(runtime.get("kebab-case").is_some());
        assert!(runtime.get("  KEBAB-CASE  ").is_some());
        assert!(runtime.get("nothing-like-it").is_none());
        assert!(runtime.get("").is_none());
    }

    #[test]
    fn render_returns_the_body_and_where_it_lives() {
        let home = tempfile::tempdir().unwrap();
        let dir = install(
            home.path(),
            "pdf-forms",
            &skill_text(
                "pdf-forms",
                "Fill forms.",
                "Open the file with scripts/fill.py.",
            ),
        );
        let runtime = SkillRuntime::load(home.path(), Path::new("/nonexistent-project"));
        let skill = runtime.get("pdf-forms").unwrap();
        let text = runtime.render(skill);
        assert!(
            text.contains("Open the file with scripts/fill.py."),
            "{text}"
        );
        assert!(
            text.contains(&dir.display().to_string()),
            "the directory must be named so relative references resolve: {text}"
        );
        assert!(text.contains("(user)"), "{text}");
    }

    #[test]
    fn the_project_directory_is_the_workspace_dot_xencode_skills() {
        assert_eq!(
            project_skills_dir(Path::new("/work/repo")),
            PathBuf::from("/work/repo/.xencode/skills")
        );
    }

    #[test]
    fn the_home_directory_is_xencode_skills_unless_the_environment_moves_it() {
        if std::env::var("XCODE_SKILLS_DIR").is_ok() {
            return;
        }
        let dir = default_home_dir();
        assert_eq!(dir.file_name().unwrap(), "skills");
        assert_eq!(dir.parent().unwrap().file_name().unwrap(), ".xencode");
    }
}
