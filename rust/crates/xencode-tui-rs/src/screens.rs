//! The seven views (TD-3): the conversation, and six screens the person
//! opens from it with Ctrl+P. Each view is a set of the existing panels; the
//! first one is where the view opens.

use crate::focus::FocusArea;

/// One of the views the TUI shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Screen {
    Conversation,
    Files,
    Changes,
    Agents,
    Models,
    Project,
    Settings,
}

impl Screen {
    pub const ALL: [Screen; 7] = [
        Screen::Conversation,
        Screen::Files,
        Screen::Changes,
        Screen::Agents,
        Screen::Models,
        Screen::Project,
        Screen::Settings,
    ];

    /// What the view is called on screen.
    pub fn name(self) -> &'static str {
        match self {
            Screen::Conversation => "Conversation",
            Screen::Files => "Files",
            Screen::Changes => "Changes",
            Screen::Agents => "Agents",
            Screen::Models => "Models",
            Screen::Project => "Project",
            Screen::Settings => "Settings",
        }
    }

    /// The word typed in the palette to reach it.
    pub fn command(self) -> &'static str {
        match self {
            Screen::Conversation => "conversation",
            Screen::Files => "files",
            Screen::Changes => "changes",
            Screen::Agents => "agents",
            Screen::Models => "models",
            Screen::Project => "project",
            Screen::Settings => "settings",
        }
    }

    /// One line saying what is in it.
    pub fn description(self) -> &'static str {
        match self {
            Screen::Conversation => "the chat with xencode, and ByteBot",
            Screen::Files => "the file tree and the editor",
            Screen::Changes => "review the diff, the pull request and commit",
            Screen::Agents => "worker agents, worktrees and background tasks",
            Screen::Models => "pick a model, your own models, provider health",
            Screen::Project => "insights, impact, analysis, security, languages",
            Screen::Settings => "every setting",
        }
    }

    /// The panels the view is made of, the one it opens on first.
    pub fn panels(self) -> &'static [FocusArea] {
        match self {
            Screen::Conversation => &[FocusArea::ChatInput, FocusArea::ByteBotPanel],
            Screen::Files => &[FocusArea::FileExplorer, FocusArea::CodeEditor],
            Screen::Changes => &[
                FocusArea::ReviewDashboard,
                FocusArea::CodeReview,
                FocusArea::GitCommit,
            ],
            Screen::Agents => &[
                FocusArea::WorkerPanel,
                FocusArea::WorktreePanel,
                FocusArea::TaskManager,
            ],
            Screen::Models => &[
                FocusArea::ModelSelector,
                FocusArea::CustomModels,
                FocusArea::ProviderHealth,
            ],
            Screen::Project => &[
                FocusArea::AdvisePanel,
                FocusArea::ImpactPanel,
                FocusArea::ProjectAnalyzer,
                FocusArea::SecurityAuditor,
                FocusArea::MultiLanguage,
            ],
            Screen::Settings => &[FocusArea::Settings],
        }
    }

    /// The view a panel belongs to. A panel in none of them (the ones being
    /// retired, and overlays such as the feature navigator) is drawn over the
    /// conversation, so it counts as the conversation.
    pub fn of(focus: FocusArea) -> Screen {
        Screen::ALL
            .into_iter()
            .find(|s| s.panels().contains(&focus))
            .unwrap_or(Screen::Conversation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_panel_belongs_to_at_most_one_view() {
        for area in crate::focus::FocusArea::ALL {
            let owners = Screen::ALL
                .iter()
                .filter(|s| s.panels().contains(&area))
                .count();
            assert!(owners <= 1, "{area:?} is in {owners} views");
        }
    }

    #[test]
    fn a_view_opens_on_its_first_panel() {
        for s in Screen::ALL {
            assert_eq!(Screen::of(s.panels()[0]), s);
        }
    }
}
