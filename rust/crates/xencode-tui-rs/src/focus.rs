//! Focus-area enum, input-mode enum, feature list, and feature navigation.
//!
//! Extracted from `app.rs` (Step 0 of the foundation refactor). Pure code move —
//! `navigate_feature` was an `&self` method on `App` but used no `App` state, so
//! it became a free function here.

#[derive(PartialEq, Clone, Copy)]
pub enum InputMode {
    Normal,
    Editing,
}

#[derive(PartialEq, Clone, Copy)]
pub enum FocusArea {
    ChatInput,
    FileExplorer,
    CodeEditor,
    ModelSelector,
    Settings,
    CodeReview,
    Terminal,
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
}

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
        _ => FocusArea::ChatInput,
    }
}
