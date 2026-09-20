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

/// Rows of the Settings panel in display order; the index is
/// `app.settings_cursor`. Navigation bounds derive from this list — adding
/// a row here makes it reachable without touching the key handler.
pub const SETTINGS_ROWS: &[&str] = &[
    "Theme",
    "Cache Enabled",
    "Memory Enabled",
    "Max Cache Size",
    "Memory Items",
    "Response Timeout",
    "Ollama URL",
    "Llama.cpp URL",
    "Llama.cpp Model",
    "Llama Temp",
    "Llama Top-K",
    "Llama Min-P",
    "Llama Max Tokens",
    "Factory Reset",
];

/// Column width the Settings panel pads its labels to.
pub const SETTINGS_LABEL_WIDTH: usize = 17;

pub const FEATURE_LIST: &[(&str, &str)] = &[
    ("📊 Performance Dashboard", "Session stats & metrics"),
    ("🏥 Provider Health", "API connection status"),
    ("📈 Project Analyzer", "Workspace file breakdown"),
    ("📝 Git Commit", "Stage and commit changes"),
    ("🤖 ByteBot Agent", "Autonomous task execution"),
    ("👥 Collaboration Hub", "Team collaboration tools"),
    ("🎙️ Voice Interface", "Voice-to-code commands"),
    ("💡 Terminal Assistant", "AI-powered shell helper"),
    ("🛡️ Security Auditor", "Vulnerability scanning"),
    ("⚡ Performance Profiler", "Code profiling tools"),
    ("🧩 Custom Models", "Model configuration & tuning"),
    ("📚 Learning Mode", "Interactive code tutorials"),
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
