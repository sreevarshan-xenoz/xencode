//! The command palette (AG-3, UX-6): one chord, `Ctrl+X`, reaches every
//! panel, every slash command and every setting.
//!
//! The palette owns no list of its own. Its entries are read from the three
//! tables that already describe those things: `focus::DESTINATIONS` for the
//! panels, the help overlay's command rows for the slash commands, and
//! `focus::SETTINGS_ITEMS` for the settings. Adding a row to any of them puts
//! it in the palette, and nothing here can drift from what the screen offers.
//!
//! Every panel is listed whatever the disclosure level, because the palette is
//! the way back to something the level hides.
//!
//! Ranking uses `nucleo-matcher`. Each typed word has to appear whole in the
//! entry, because letters matched anywhere put nonsense first: "undo" found
//! "rOUNDed bOrders" before this rule. A name that is exactly the query ranks
//! first, then a match in the name, then a match only in the description, so
//! "theme" lands on the Theme setting before a panel that mentions themes.
//! The start of a word is enough ("wor" finds the worktree panel). A query
//! whose words appear nowhere lists nothing, and the palette says "0 match"
//! rather than guessing.

use nucleo_matcher::pattern::{AtomKind, CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

use crate::focus::{FocusArea, DESTINATIONS, SETTINGS_ITEMS};

/// What choosing an entry does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteTarget {
    /// Focus this panel.
    Panel(FocusArea),
    /// Put this command, for example `/rewind`, into the composer.
    Command(&'static str),
    /// Open Settings on this row of `SETTINGS_ITEMS`.
    Setting(usize),
    /// Open one of the seven views (TD-3).
    Screen(crate::screens::Screen),
}

/// One row of the palette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteEntry {
    /// What the row is called on screen.
    pub label: &'static str,
    /// What a query is matched against first: the label, except for a
    /// command, whose label also lists every argument it takes.
    pub name: &'static str,
    /// One line saying what it is.
    pub detail: &'static str,
    pub target: PaletteTarget,
}

impl PaletteEntry {
    /// The word shown before the label so a person can tell the kinds apart
    /// without colour.
    pub fn kind(&self) -> &'static str {
        match self.target {
            PaletteTarget::Panel(_) => "panel",
            PaletteTarget::Command(_) => "command",
            PaletteTarget::Setting(_) => "setting",
            PaletteTarget::Screen(_) => "view",
        }
    }
}

/// Every entry, in table order: the seven views, then panels, then commands,
/// then settings.
pub fn entries() -> Vec<PaletteEntry> {
    let screens = crate::screens::Screen::ALL
        .into_iter()
        .map(|s| PaletteEntry {
            label: s.name(),
            name: s.command(),
            detail: s.description(),
            target: PaletteTarget::Screen(s),
        });
    let panels = DESTINATIONS.iter().map(|d| PaletteEntry {
        label: d.name,
        name: d.name,
        detail: d.description,
        target: PaletteTarget::Panel(d.area),
    });
    let commands = crate::help::COMMANDS.iter().map(|(usage, what)| {
        let name = usage.split_whitespace().next().unwrap_or(usage);
        PaletteEntry {
            label: usage,
            name,
            detail: what,
            target: PaletteTarget::Command(name),
        }
    });
    let settings = SETTINGS_ITEMS
        .iter()
        .enumerate()
        .map(|(i, row)| PaletteEntry {
            label: row.label,
            name: row.label,
            detail: row.section,
            target: PaletteTarget::Setting(i),
        });
    screens
        .chain(panels)
        .chain(commands)
        .chain(settings)
        .collect()
}

/// The entries that match `query`, best first, as indices into `all`.
///
/// An empty query lists everything in table order. Otherwise the order is:
/// a name equal to the query, then names holding every typed word, then
/// entries holding them only in their description. Within a group the higher
/// score wins, then the shorter name, then table order.
pub fn rank(query: &str, all: &[PaletteEntry]) -> Vec<usize> {
    let query = query.trim();
    if query.is_empty() {
        return (0..all.len()).collect();
    }
    let bare = query.trim_start_matches('/');
    let words = Pattern::new(
        query,
        CaseMatching::Ignore,
        Normalization::Smart,
        AtomKind::Substring,
    );
    let mut matcher = Matcher::new(Config::DEFAULT);
    let mut buf = Vec::new();
    // (group, score, name length, table index); lower group ranks first.
    let mut scored: Vec<(u8, u32, usize, usize)> = Vec::new();
    for (i, entry) in all.iter().enumerate() {
        let name = entry.name.trim_start_matches('/');
        if name.eq_ignore_ascii_case(bare) {
            scored.push((0, u32::MAX, name.len(), i));
        } else if let Some(score) = words.score(Utf32Str::new(entry.name, &mut buf), &mut matcher) {
            scored.push((1, score, name.len(), i));
        } else {
            let full = format!("{} {} {}", entry.label, entry.kind(), entry.detail);
            if let Some(score) = words.score(Utf32Str::new(&full, &mut buf), &mut matcher) {
                scored.push((2, score, name.len(), i));
            }
        }
    }
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    scored.into_iter().map(|(_, _, _, i)| i).collect()
}

/// The indexes of `paths` that match `query`, best first, for an `@` file
/// mention (TD-3). A file whose name starts with the query ranks first, then
/// one whose name contains it, then any path the letters can be found in, in
/// order; ties go to the shorter path.
pub fn rank_paths(query: &str, paths: &[String]) -> Vec<usize> {
    let query = query.trim();
    if query.is_empty() {
        return Vec::new();
    }
    let lower = query.to_lowercase();
    let fuzzy = Pattern::new(
        query,
        CaseMatching::Ignore,
        Normalization::Smart,
        AtomKind::Fuzzy,
    );
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let mut buf = Vec::new();
    let mut scored: Vec<(u8, u32, usize, usize)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let name = path
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(path)
            .to_lowercase();
        let group = if name.starts_with(&lower) {
            0
        } else if name.contains(&lower) {
            1
        } else {
            2
        };
        if group < 2 {
            scored.push((group, 0, path.len(), i));
        } else if let Some(score) = fuzzy.score(Utf32Str::new(path, &mut buf), &mut matcher) {
            scored.push((2, score, path.len(), i));
        }
    }
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    scored.into_iter().map(|(_, _, _, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_named_like_the_mention_ranks_first() {
        let paths: Vec<String> = [
            "src/scargo.rs",
            "docs/cargo-notes.md",
            "Cargo.toml",
            "x/c_a_r_g_o.txt",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let order: Vec<&str> = rank_paths("cargo", &paths)
            .into_iter()
            .map(|i| paths[i].as_str())
            .collect();
        assert_eq!(
            order,
            [
                "Cargo.toml",
                "docs/cargo-notes.md",
                "src/scargo.rs",
                "x/c_a_r_g_o.txt"
            ]
        );
        assert!(rank_paths("", &paths).is_empty());
    }

    fn top(query: &str) -> PaletteEntry {
        let all = entries();
        let ranked = rank(query, &all);
        let first = *ranked
            .first()
            .unwrap_or_else(|| panic!("nothing matched {query:?}"));
        all[first].clone()
    }

    #[test]
    fn every_panel_command_and_setting_is_listed_once() {
        let all = entries();
        assert_eq!(
            all.len(),
            crate::screens::Screen::ALL.len()
                + DESTINATIONS.len()
                + crate::help::COMMANDS.len()
                + SETTINGS_ITEMS.len()
        );
        for d in DESTINATIONS {
            let n = all
                .iter()
                .filter(|e| e.target == PaletteTarget::Panel(d.area))
                .count();
            assert_eq!(n, 1, "{:?} is listed {n} times", d.area);
        }
        for i in 0..SETTINGS_ITEMS.len() {
            assert!(all.iter().any(|e| e.target == PaletteTarget::Setting(i)));
        }
        for cmd in crate::app::SLASH_COMMANDS {
            assert!(
                all.iter().any(|e| e.target == PaletteTarget::Command(cmd)),
                "{cmd} is not in the palette"
            );
        }
    }

    #[test]
    fn an_empty_query_lists_everything_in_table_order() {
        let all = entries();
        assert_eq!(rank("", &all), (0..all.len()).collect::<Vec<_>>());
        assert_eq!(rank("   ", &all).len(), all.len());
    }

    #[test]
    fn ordinary_words_land_on_what_a_person_means() {
        assert_eq!(top("theme").target, PaletteTarget::Setting(0));
        assert_eq!(top("rewind").target, PaletteTarget::Command("/rewind"));
        // A view's own word lands on the view (TD-3).
        assert_eq!(
            top("models").target,
            PaletteTarget::Screen(crate::screens::Screen::Models)
        );
        assert_eq!(
            top("settings").target,
            PaletteTarget::Screen(crate::screens::Screen::Settings)
        );
        assert_eq!(
            top("worktree").target,
            PaletteTarget::Panel(FocusArea::WorktreePanel)
        );
        assert_eq!(
            top("git commit").target,
            PaletteTarget::Panel(FocusArea::GitCommit)
        );
        assert_eq!(top("ollama").kind(), "setting");
        assert_eq!(top("cost").target, PaletteTarget::Command("/cost"));
        assert_eq!(top("/cost").target, PaletteTarget::Command("/cost"));
        assert_eq!(top("help").target, PaletteTarget::Command("/help"));
        assert_eq!(
            top("background").target,
            PaletteTarget::Panel(FocusArea::TaskManager)
        );
        assert_eq!(
            top("health").target,
            PaletteTarget::Panel(FocusArea::ProviderHealth)
        );
        assert_eq!(
            top("wor").target,
            PaletteTarget::Panel(FocusArea::WorktreePanel)
        );
    }

    #[test]
    fn letters_scattered_through_a_name_do_not_count_as_a_match() {
        let all = entries();
        // "undo" is spelled out across "rOUNDed bOrders"; that is not a
        // match. It does find `/rewind`, whose description says "undo".
        let undo: Vec<_> = rank("undo", &all)
            .into_iter()
            .map(|i| all[i].name)
            .collect();
        assert!(!undo.contains(&"Rounded Borders"), "{undo:?}");
        assert_eq!(undo.first(), Some(&"/rewind"));
        // A long usage line is not matched for the letters in its arguments.
        let api: Vec<_> = rank("api key", &all)
            .into_iter()
            .map(|i| all[i].name)
            .collect();
        assert!(!api.contains(&"/orchestrator"), "{api:?}");
    }

    #[test]
    fn a_query_that_matches_nothing_lists_nothing() {
        assert!(rank("zzqqxxj", &entries()).is_empty());
    }
}
