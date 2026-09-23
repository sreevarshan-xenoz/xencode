//! Focus-area enum, input-mode enum, feature list, and feature navigation.
//!
//! Single source of truth: `app.rs` and `ui.rs` import these from here.

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum InputMode {
    Normal,
    Editing,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FocusArea {
    ChatInput,
    FileExplorer,
    CodeEditor,
    ModelSelector,
    Settings,
    CodeReview,
    PerformanceDashboard,
    ProviderHealth,
    ProjectAnalyzer,
    GitCommit,
    FeatureNavigator,
    ByteBotPanel,
    CollaborationHub,
    VoiceInterface,
    TerminalAssistant,
    SecurityAuditor,
    PerformanceProfiler,
    CustomModels,
    LearningMode,
    MultiLanguage,
    ReviewDashboard,
    TaskManager,
    WorktreePanel,
    AdvisePanel,
}

impl FocusArea {
    /// Short human name for the header's focused-panel badge. Kept under
    /// ~12 columns so the right side survives narrow terminals.
    pub fn display_name(self) -> &'static str {
        match self {
            FocusArea::ChatInput => "Chat",
            FocusArea::FileExplorer => "Explorer",
            FocusArea::CodeEditor => "Editor",
            FocusArea::ModelSelector => "Models",
            FocusArea::Settings => "Settings",
            FocusArea::CodeReview => "Review",
            FocusArea::PerformanceDashboard => "Dashboard",
            FocusArea::ProviderHealth => "Providers",
            FocusArea::ProjectAnalyzer => "Analyzer",
            FocusArea::GitCommit => "Git Commit",
            FocusArea::FeatureNavigator => "Features",
            FocusArea::ByteBotPanel => "ByteBot",
            FocusArea::CollaborationHub => "Collab Hub",
            FocusArea::VoiceInterface => "Voice",
            FocusArea::TerminalAssistant => "Assistant",
            FocusArea::SecurityAuditor => "Security",
            FocusArea::PerformanceProfiler => "Profiler",
            FocusArea::CustomModels => "Custom Models",
            FocusArea::LearningMode => "Learning",
            FocusArea::MultiLanguage => "Languages",
            FocusArea::ReviewDashboard => "PR Review",
            FocusArea::TaskManager => "Tasks",
            FocusArea::WorktreePanel => "Worktrees",
            FocusArea::AdvisePanel => "Advice",
        }
    }
}

/// Sub-states of the WorktreePanel: while one is active every keystroke
/// belongs to the prompt instead of the list.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum WorktreePrompt {
    None,
    AddPath,
    AddBranch,
    ConfirmRemove,
}

/// Which Collaboration Hub connection field typed characters edit. Tab
/// cycles through these; the hub's form is an editing mode, not a focus.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CollabField {
    Server,
    Username,
    Session,
}

impl CollabField {
    pub fn next(self) -> Self {
        match self {
            CollabField::Server => CollabField::Username,
            CollabField::Username => CollabField::Session,
            CollabField::Session => CollabField::Server,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CollabField::Server => "Server",
            CollabField::Username => "User",
            CollabField::Session => "Session",
        }
    }
}

/// Which Multi-Language form field typed characters edit — the same shape as
/// `CollabField`: the panel's text entry is a mode, not a focus area.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LangField {
    Source,
    Target,
    Input,
}

impl LangField {
    pub fn next(self) -> Self {
        match self {
            LangField::Source => LangField::Target,
            LangField::Target => LangField::Input,
            LangField::Input => LangField::Source,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LangField::Source => "From",
            LangField::Target => "To",
            LangField::Input => "Text",
        }
    }
}

/// How a settings row responds to ←/→ and Enter. The row's behavior comes
/// from this table, not from its index — adding a row means adding an
/// entry, never renumbering the panel (H1-04).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingKind {
    /// ←/→ selects among option names (also the config values).
    Cycle(&'static [&'static str]),
    /// ←/→ flips a boolean.
    Toggle,
    /// ←/→ adjusts an integer by `step`, never below `min` nor above `max`.
    Stepped { step: u64, min: u64, max: u64 },
    /// Enter opens a text editor; commit stores the string as typed.
    Text,
    /// Enter opens a text editor like `Text`, but the value is an API key:
    /// it is echoed as bullets and stored as `Option<String>` (an empty
    /// commit clears it).
    Secret,
    /// Enter opens a text editor; commit parses the number.
    Number,
    /// Enter triggers it (Factory Reset).
    Action,
}

/// One row of the Settings panel: display order is table order, and the
/// section header prints once when the section changes.
pub struct SettingRow {
    pub label: &'static str,
    pub section: &'static str,
    pub kind: SettingKind,
}

/// Rows of the Settings panel in display order; the index is
/// `app.settings_cursor`. Navigation bounds derive from this list — adding
/// a row here makes it reachable without touching the key handler.
pub const SETTINGS_ITEMS: &[SettingRow] = &[
    SettingRow {
        label: "Theme",
        section: "Display",
        kind: SettingKind::Cycle(crate::theme::THEME_NAMES),
    },
    SettingRow {
        label: "Layout",
        section: "Display",
        kind: SettingKind::Cycle(crate::layout::LAYOUT_NAMES),
    },
    SettingRow {
        label: "Rounded Borders",
        section: "Display",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Show Scrollbars",
        section: "Display",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Line Numbers",
        section: "Display",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Agent Approval",
        section: "Agent",
        kind: SettingKind::Cycle(crate::agent_tools::APPROVAL_MODE_NAMES),
    },
    SettingRow {
        label: "Command Timeout",
        section: "Agent",
        kind: SettingKind::Stepped {
            step: 5,
            min: 5,
            max: 300,
        },
    },
    SettingRow {
        label: "Cache Enabled",
        section: "Performance",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Memory Enabled",
        section: "Performance",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Max Cache Size",
        section: "Limits",
        kind: SettingKind::Stepped {
            step: 10,
            min: 10,
            max: 1000,
        },
    },
    SettingRow {
        label: "Memory Items",
        section: "Limits",
        kind: SettingKind::Stepped {
            step: 5,
            min: 5,
            max: 500,
        },
    },
    SettingRow {
        label: "Response Timeout",
        section: "Limits",
        kind: SettingKind::Stepped {
            step: 5,
            min: 5,
            max: 300,
        },
    },
    SettingRow {
        label: "Ollama URL",
        section: "Connection",
        kind: SettingKind::Text,
    },
    SettingRow {
        label: "Llama.cpp URL",
        section: "Connection",
        kind: SettingKind::Text,
    },
    SettingRow {
        label: "Llama.cpp Model",
        section: "Connection",
        kind: SettingKind::Text,
    },
    SettingRow {
        label: "Llama Temp",
        section: "llama.cpp",
        kind: SettingKind::Number,
    },
    SettingRow {
        label: "Llama Top-K",
        section: "llama.cpp",
        kind: SettingKind::Number,
    },
    SettingRow {
        label: "Llama Min-P",
        section: "llama.cpp",
        kind: SettingKind::Number,
    },
    SettingRow {
        label: "Llama Max Tokens",
        section: "llama.cpp",
        kind: SettingKind::Number,
    },
    SettingRow {
        label: "Remote URL",
        section: "Providers",
        kind: SettingKind::Text,
    },
    SettingRow {
        label: "Remote Key",
        section: "Providers",
        kind: SettingKind::Secret,
    },
    SettingRow {
        label: "Gemini Key",
        section: "Providers",
        kind: SettingKind::Secret,
    },
    SettingRow {
        label: "Qwen Key",
        section: "Providers",
        kind: SettingKind::Secret,
    },
    SettingRow {
        label: "OpenRouter Key",
        section: "Providers",
        kind: SettingKind::Secret,
    },
    SettingRow {
        label: "Factory Reset",
        section: "Actions",
        kind: SettingKind::Action,
    },
];

/// Index of a settings row by its (stable) label — the test/keying
/// replacement for magic row numbers.
pub fn settings_row_index(label: &str) -> usize {
    SETTINGS_ITEMS
        .iter()
        .position(|r| r.label == label)
        .unwrap_or(0)
}

/// Column width the Settings panel pads its labels to.
pub const SETTINGS_LABEL_WIDTH: usize = 17;

/// Render a secret for the screen: bullets, with the last four characters
/// kept so two keys can be told apart. A value short enough to be fully
/// revealed by that tail is hidden entirely.
pub fn mask_secret(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 8 {
        return "••••".to_string();
    }
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{}{tail}", "•".repeat((chars.len() - 4).min(12)))
}

pub const FEATURE_LIST: &[(&str, &str)] = &[
    ("📊 Performance Dashboard", "Session stats & metrics"),
    ("🏥 Provider Health", "API connection status"),
    ("📈 Project Analyzer", "Workspace file breakdown"),
    ("📝 Git Commit", "Stage and commit changes"),
    ("🤖 ByteBot Agent", "Autonomous task execution"),
    ("👥 Collaboration Hub", "Team collaboration tools"),
    ("🎙️ Voice Interface", "Record a clip, transcribe if able"),
    ("💡 Terminal Assistant", "AI-powered shell helper"),
    ("🛡️ Security Auditor", "Vulnerability scanning"),
    ("⚡ Performance Profiler", "Code profiling tools"),
    ("🧩 Custom Models", "Model configuration & tuning"),
    ("📚 Learning Mode", "Lessons from this repo's own files"),
    ("🌐 Multi-Language", "Language detection & tools"),
    ("🔍 PR Review", "Per-file diff browsing"),
    ("⏳ Background Tasks", "Running & finished commands"),
    ("🌳 Worktrees", "List, create and remove git worktrees"),
    ("💡 Insights", "Live refactor suggestions & warnings"),
];

/// Maps a feature-navigator index to its target FocusArea.
pub fn navigate_feature(idx: usize) -> FocusArea {
    match idx {
        0 => FocusArea::PerformanceDashboard,
        1 => FocusArea::ProviderHealth,
        2 => FocusArea::ProjectAnalyzer,
        3 => FocusArea::GitCommit,
        4 => FocusArea::ByteBotPanel,
        5 => FocusArea::CollaborationHub,
        6 => FocusArea::VoiceInterface,
        7 => FocusArea::TerminalAssistant,
        8 => FocusArea::SecurityAuditor,
        9 => FocusArea::PerformanceProfiler,
        10 => FocusArea::CustomModels,
        11 => FocusArea::LearningMode,
        12 => FocusArea::MultiLanguage,
        13 => FocusArea::ReviewDashboard,
        14 => FocusArea::TaskManager,
        15 => FocusArea::WorktreePanel,
        16 => FocusArea::AdvisePanel,
        _ => FocusArea::ChatInput,
    }
}
