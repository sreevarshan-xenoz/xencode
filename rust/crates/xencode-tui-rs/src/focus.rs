//! Focus-area enum, input-mode enum, feature list, and feature navigation.
//!
//! Single source of truth: `app.rs` and `ui.rs` import these from here.

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum InputMode {
    Normal,
    Editing,
}

/// The product mode (`X-2`): which *way of working* xencode is in. It is one
/// piece of state that both views read — the tasks, running agents, their
/// sessions, worktrees, diffs, pending approvals, event history, verification
/// results and git state are all held once on the `App`, never per-mode, so
/// switching cannot duplicate or drop any of them. `Coding` is the single-agent
/// editor the app has always been; `Orchestrator` is the same state seen as a
/// fleet to run, verified and merged.
#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Coding,
    Orchestrator,
}

impl Mode {
    /// The label shown in the status bar, kept short so the right-hand fields
    /// survive a narrow terminal.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Coding => "CODING",
            Mode::Orchestrator => "ORCHESTRATOR",
        }
    }

    pub fn toggled(self) -> Mode {
        match self {
            Mode::Coding => Mode::Orchestrator,
            Mode::Orchestrator => Mode::Coding,
        }
    }
}

impl Default for Mode {
    /// The app opens in Coding; the orchestrator is entered on purpose.
    fn default() -> Self {
        Mode::Coding
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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
    /// Blast radius of one file (`QD-2`): the fan-out tree QD-1's three layers
    /// project into. Opened by `/impact <file>`, traversed with the arrows.
    ImpactPanel,
    /// Why the screen is arranged as it is: this session's layout changes,
    /// each with the ask behind it (`V-9`).
    LayoutPanel,
    /// The worker panel (`OR-12`): the fleet, the tasks, the recorded graph, the
    /// costs, the events and what is waiting on a person — every row naming the
    /// record its figures were read from.
    WorkerPanel,
}

/// Four levels of progressive disclosure (AE-5).
/// Hide complexity until needed; four levels, never mixed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DisclosureLevel {
    /// Level 1: Core interaction (Chat, Editor, Explorer, Models, Settings).
    Level1 = 1,
    /// Level 2: Standard workflow (Review, Git Commit, Tasks, Worktrees, Advice).
    Level2 = 2,
    /// Level 3: Advanced inspection and diagnostics (Dashboards, Analyzers, Health, PR Review, Impact, Layout, Feature Navigator).
    Level3 = 3,
    /// Level 4: Specialist / deep tooling (ByteBot, Collab, Voice, Terminal Assistant, Security Auditor, Profiler, Custom Models, Learning, MultiLanguage).
    Level4 = 4,
}

impl DisclosureLevel {
    pub const fn rank(self) -> u8 {
        match self {
            DisclosureLevel::Level1 => 1,
            DisclosureLevel::Level2 => 2,
            DisclosureLevel::Level3 => 3,
            DisclosureLevel::Level4 => 4,
        }
    }

    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => DisclosureLevel::Level1,
            2 => DisclosureLevel::Level2,
            3 => DisclosureLevel::Level3,
            _ => DisclosureLevel::Level4,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            DisclosureLevel::Level1 => "Level 1 (Core)",
            DisclosureLevel::Level2 => "Level 2 (Workflow)",
            DisclosureLevel::Level3 => "Level 3 (Advanced)",
            DisclosureLevel::Level4 => "Level 4 (All / Specialist)",
        }
    }
}

impl FocusArea {
    /// All variants of FocusArea.
    pub const ALL: [FocusArea; 27] = [
        FocusArea::ChatInput,
        FocusArea::FileExplorer,
        FocusArea::CodeEditor,
        FocusArea::ModelSelector,
        FocusArea::Settings,
        FocusArea::CodeReview,
        FocusArea::PerformanceDashboard,
        FocusArea::ProviderHealth,
        FocusArea::ProjectAnalyzer,
        FocusArea::GitCommit,
        FocusArea::FeatureNavigator,
        FocusArea::ByteBotPanel,
        FocusArea::CollaborationHub,
        FocusArea::VoiceInterface,
        FocusArea::TerminalAssistant,
        FocusArea::SecurityAuditor,
        FocusArea::PerformanceProfiler,
        FocusArea::CustomModels,
        FocusArea::LearningMode,
        FocusArea::MultiLanguage,
        FocusArea::ReviewDashboard,
        FocusArea::TaskManager,
        FocusArea::WorktreePanel,
        FocusArea::AdvisePanel,
        FocusArea::ImpactPanel,
        FocusArea::LayoutPanel,
        FocusArea::WorkerPanel,
    ];

    /// Progressive disclosure tier required to see this destination in navigators / palettes.
    /// Exhaustive pattern match ensures any added FocusArea must explicitly define its tier.
    pub const fn disclosure_level(self) -> DisclosureLevel {
        match self {
            FocusArea::ChatInput
            | FocusArea::FileExplorer
            | FocusArea::CodeEditor
            | FocusArea::ModelSelector
            | FocusArea::Settings => DisclosureLevel::Level1,

            FocusArea::CodeReview
            | FocusArea::GitCommit
            | FocusArea::TaskManager
            | FocusArea::WorktreePanel
            | FocusArea::AdvisePanel => DisclosureLevel::Level2,

            FocusArea::PerformanceDashboard
            | FocusArea::ProviderHealth
            | FocusArea::ProjectAnalyzer
            | FocusArea::ReviewDashboard
            | FocusArea::ImpactPanel
            | FocusArea::LayoutPanel
            | FocusArea::WorkerPanel
            | FocusArea::FeatureNavigator => DisclosureLevel::Level3,

            FocusArea::ByteBotPanel
            | FocusArea::CollaborationHub
            | FocusArea::VoiceInterface
            | FocusArea::TerminalAssistant
            | FocusArea::SecurityAuditor
            | FocusArea::PerformanceProfiler
            | FocusArea::CustomModels
            | FocusArea::LearningMode
            | FocusArea::MultiLanguage => DisclosureLevel::Level4,
        }
    }

    /// Short human name for the header's focused-panel badge. Kept under
    /// ~12 columns so the right side survives narrow terminals.
    /// The one name this destination goes by everywhere — titles, the palette,
    /// `/goto` replies — read from [`DESTINATIONS`] so it cannot drift (TX-10).
    pub fn display_name(self) -> &'static str {
        find_destination(self).map(|d| d.name).unwrap_or("Panel")
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
    /// ←/→ selects among the layout names on offer: the three shipped presets
    /// plus the templates declared in config. Read at the keystroke rather than
    /// baked into the table, because a declared name only exists once the
    /// config that declares it has been loaded (`V-5`).
    CycleLayout,
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
        kind: SettingKind::CycleLayout,
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
        label: "Mouse Capture",
        section: "Display",
        kind: SettingKind::Toggle,
    },
    SettingRow {
        label: "Disclosure Level",
        section: "Display",
        kind: SettingKind::Stepped {
            step: 1,
            min: 1,
            max: 4,
        },
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
        label: "Llama Seed",
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
    // Consent sits below the keys on purpose: filling in a key says who you are
    // to a provider, this row says whether a prompt may reach one at all.
    SettingRow {
        label: "Cloud Models",
        section: "Providers",
        kind: SettingKind::Toggle,
    },
    // The other half of the same posture, and a different kind of consent: this
    // row is about work handed to another vendor's program, not bytes sent over
    // a socket, so it is its own row and cannot be read off the one above.
    SettingRow {
        label: "External Workers",
        section: "Providers",
        kind: SettingKind::Toggle,
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
///
/// The rule itself lives with the MCP client, which has to draw an access token
/// written into a server's url exactly this way, so there is one implementation
/// rather than two that can drift apart.
pub fn mask_secret(value: &str) -> String {
    xencode_mcp_rs::mask_secret(value)
}

/// Canonical metadata for a TUI destination.
/// One table in `focus.rs` is the single source of truth for both the palette
/// and the first-run recommendations (AE-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Destination {
    pub area: FocusArea,
    pub name: &'static str,
    pub description: &'static str,
    pub level: DisclosureLevel,
    pub shortcut: Option<&'static str>,
    pub command_name: &'static str,
}

/// Canonical table of all 27 TUI destinations.
/// Every variant of `FocusArea` is represented with its progressive disclosure tier.
pub const DESTINATIONS: &[Destination] = &[
    // ── Level 1: Core interaction ─────────────────────────────────────────────
    Destination {
        area: FocusArea::ChatInput,
        name: "Chat",
        description: "Interactive conversation & prompt instructions",
        level: DisclosureLevel::Level1,
        shortcut: Some("i"),
        command_name: "chat",
    },
    Destination {
        area: FocusArea::FileExplorer,
        name: "Explorer",
        description: "Workspace file tree navigation",
        level: DisclosureLevel::Level1,
        shortcut: Some("Tab"),
        command_name: "explorer",
    },
    Destination {
        area: FocusArea::CodeEditor,
        name: "Editor",
        description: "Focused file viewer & inline editor",
        level: DisclosureLevel::Level1,
        shortcut: Some("e"),
        command_name: "editor",
    },
    Destination {
        area: FocusArea::ModelSelector,
        name: "Models",
        description: "Select model provider & inference endpoint",
        level: DisclosureLevel::Level1,
        shortcut: Some("m"),
        command_name: "models",
    },
    Destination {
        area: FocusArea::Settings,
        name: "Settings",
        description: "Configuration & preferences panel",
        level: DisclosureLevel::Level1,
        shortcut: Some("s"),
        command_name: "settings",
    },
    // ── Level 2: Standard development workflow ─────────────────────────────────
    Destination {
        area: FocusArea::CodeReview,
        name: "Code Review",
        description: "AI review & suggestions on changes",
        level: DisclosureLevel::Level2,
        shortcut: Some("Ctrl+R"),
        command_name: "review",
    },
    Destination {
        area: FocusArea::GitCommit,
        name: "Git Commit",
        description: "Stage changes and write commits",
        level: DisclosureLevel::Level2,
        shortcut: Some("Ctrl+S"),
        command_name: "commit",
    },
    Destination {
        area: FocusArea::TaskManager,
        name: "Background Tasks",
        description: "Running & finished command tasks",
        level: DisclosureLevel::Level2,
        shortcut: Some("Ctrl+K"),
        command_name: "tasks",
    },
    Destination {
        area: FocusArea::WorktreePanel,
        name: "Worktrees",
        description: "List, create and remove git worktrees",
        level: DisclosureLevel::Level2,
        shortcut: Some("Ctrl+O"),
        command_name: "worktree",
    },
    Destination {
        area: FocusArea::AdvisePanel,
        name: "Insights",
        description: "Live refactor suggestions & warnings",
        level: DisclosureLevel::Level2,
        shortcut: Some("Ctrl+L"),
        command_name: "advise",
    },
    // ── Level 3: Advanced diagnostics & inspection ────────────────────────────
    Destination {
        area: FocusArea::PerformanceDashboard,
        name: "Performance Dashboard",
        description: "Session stats & metrics",
        level: DisclosureLevel::Level3,
        shortcut: None,
        command_name: "dashboard",
    },
    Destination {
        area: FocusArea::ProviderHealth,
        name: "Provider Health",
        description: "API connection status & latency",
        level: DisclosureLevel::Level3,
        shortcut: Some("Ctrl+H"),
        command_name: "health",
    },
    Destination {
        area: FocusArea::ProjectAnalyzer,
        name: "Project Analyzer",
        description: "Workspace file breakdown & analysis",
        level: DisclosureLevel::Level3,
        shortcut: None,
        command_name: "analyzer",
    },
    Destination {
        area: FocusArea::ReviewDashboard,
        name: "PR Review",
        description: "Per-file diff browsing against base",
        level: DisclosureLevel::Level3,
        shortcut: Some("Ctrl+Y"),
        command_name: "diff",
    },
    Destination {
        area: FocusArea::ImpactPanel,
        name: "Impact",
        description: "What a change to one file reaches",
        level: DisclosureLevel::Level3,
        shortcut: None,
        command_name: "impact",
    },
    Destination {
        area: FocusArea::LayoutPanel,
        name: "Layout History",
        description: "Why the screen is arranged this way",
        level: DisclosureLevel::Level3,
        shortcut: None,
        command_name: "layout",
    },
    Destination {
        area: FocusArea::WorkerPanel,
        name: "Worker Panel",
        description: "Workers, tasks, graph, costs, logs, approvals — each traced",
        level: DisclosureLevel::Level3,
        shortcut: Some("Ctrl+A"),
        command_name: "workers",
    },
    Destination {
        area: FocusArea::FeatureNavigator,
        name: "Feature Navigator",
        description: "Command palette and destination picker",
        level: DisclosureLevel::Level3,
        shortcut: Some("Ctrl+F"),
        command_name: "palette",
    },
    // ── Level 4: Specialist & deep tooling ────────────────────────────────────
    Destination {
        area: FocusArea::ByteBotPanel,
        name: "ByteBot",
        description: "Autonomous task execution loop",
        level: DisclosureLevel::Level4,
        shortcut: Some("/bytebot"),
        command_name: "bytebot",
    },
    Destination {
        area: FocusArea::CollaborationHub,
        name: "Collaboration Hub",
        description: "Team collaboration tools & sessions",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "collab",
    },
    Destination {
        area: FocusArea::VoiceInterface,
        name: "Voice Interface",
        description: "Record a clip, transcribe if able",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "voice",
    },
    Destination {
        area: FocusArea::TerminalAssistant,
        name: "Terminal Assistant",
        description: "AI-powered shell helper & commands",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "assistant",
    },
    Destination {
        area: FocusArea::SecurityAuditor,
        name: "Security Auditor",
        description: "Vulnerability scanning & dependencies",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "security",
    },
    Destination {
        area: FocusArea::PerformanceProfiler,
        name: "Performance Profiler",
        description: "Code profiling tools & CPU/mem gauges",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "profiler",
    },
    Destination {
        area: FocusArea::CustomModels,
        name: "Custom Models",
        description: "Model configuration & hyperparameter tuning",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "custom-models",
    },
    Destination {
        area: FocusArea::LearningMode,
        name: "Learning Mode",
        description: "Lessons from this repo's own files",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "learning",
    },
    Destination {
        area: FocusArea::MultiLanguage,
        name: "Multi-Language",
        description: "Language detection & translation tools",
        level: DisclosureLevel::Level4,
        shortcut: None,
        command_name: "languages",
    },
];

/// Find destination by FocusArea.
pub fn find_destination(area: FocusArea) -> Option<&'static Destination> {
    DESTINATIONS.iter().find(|d| d.area == area)
}

/// Find destination by name, label or slash command (case-insensitive).
pub fn find_destination_by_name(query: &str) -> Option<&'static Destination> {
    let q = query.trim().trim_start_matches('/').to_ascii_lowercase();
    DESTINATIONS.iter().find(|d| {
        d.command_name == q
            || d.name.eq_ignore_ascii_case(&q)
            || d.area.display_name().eq_ignore_ascii_case(&q)
            || (d.area == FocusArea::LearningMode && (q == "learn" || q == "learning"))
            || (d.area == FocusArea::ByteBotPanel
                && (q == "bytebot" || q == "agent" || q == "bytebot agent"))
            // Names these panels went by before they had one (TX-10).
            || (d.area == FocusArea::AdvisePanel && (q == "advice" || q == "insights & advice"))
            || (d.area == FocusArea::ImpactPanel && q == "blast radius")
    })
}

/// Filter destinations visible up to the given disclosure level.
pub fn destinations_for_level(
    max_level: DisclosureLevel,
) -> impl Iterator<Item = &'static Destination> {
    DESTINATIONS.iter().filter(move |d| d.level <= max_level)
}

/// Feature list items visible in the palette / navigator up to max disclosure level.
/// Excludes the navigator itself from its own options.
pub fn feature_items_for_level(
    max_level: DisclosureLevel,
) -> Vec<(&'static str, &'static str, FocusArea)> {
    DESTINATIONS
        .iter()
        .filter(|d| d.level <= max_level && d.area != FocusArea::FeatureNavigator)
        .map(|d| (d.name, d.description, d.area))
        .collect()
}

/// Formatted shortcuts text for the first-run welcome screen (AE-5).
/// Generates only shortcuts for destinations allowed by the disclosure level,
/// so novice levels never see Level 4 tooling.
pub fn first_run_shortcuts_line(max_level: DisclosureLevel) -> String {
    let mut parts = Vec::new();
    for d in DESTINATIONS.iter().filter(|d| d.level <= max_level) {
        if let Some(sc) = d.shortcut {
            if matches!(
                d.area,
                FocusArea::ModelSelector
                    | FocusArea::FileExplorer
                    | FocusArea::CodeReview
                    | FocusArea::GitCommit
                    | FocusArea::ByteBotPanel
            ) {
                parts.push(format!("{}={}", sc, d.name.to_ascii_lowercase()));
            }
        }
    }
    parts.join(", ")
}

pub const FEATURE_LIST: &[(&str, &str)] = &[
    ("Performance Dashboard", "Session stats & metrics"),
    ("Provider Health", "API connection status"),
    ("Project Analyzer", "Workspace file breakdown"),
    ("Git Commit", "Stage and commit changes"),
    ("ByteBot", "Autonomous task execution"),
    ("Collaboration Hub", "Team collaboration tools"),
    ("Voice Interface", "Record a clip, transcribe if able"),
    ("Terminal Assistant", "AI-powered shell helper"),
    ("Security Auditor", "Vulnerability scanning"),
    ("Performance Profiler", "Code profiling tools"),
    ("Custom Models", "Model configuration & tuning"),
    ("Learning Mode", "Lessons from this repo's own files"),
    ("Multi-Language", "Language detection & tools"),
    ("PR Review", "Per-file diff browsing"),
    ("Background Tasks", "Running & finished commands"),
    ("Worktrees", "List, create and remove git worktrees"),
    ("Insights", "Live refactor suggestions & warnings"),
    ("Impact", "What a change to one file reaches"),
    ("Layout History", "Why the screen is arranged this way"),
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
        17 => FocusArea::ImpactPanel,
        18 => FocusArea::LayoutPanel,
        _ => FocusArea::ChatInput,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TX-10: a panel goes by one name. The palette list, every title that
    /// reads `display_name`, and the destination table agree, and no two
    /// destinations share a name.
    #[test]
    fn every_destination_has_one_name_everywhere() {
        for (i, (name, _)) in super::FEATURE_LIST.iter().enumerate() {
            let area = super::navigate_feature(i);
            assert_eq!(
                *name,
                area.display_name(),
                "palette row {i} names {area:?} differently"
            );
        }
        let mut seen = std::collections::HashSet::new();
        for d in super::DESTINATIONS {
            assert_eq!(d.area.display_name(), d.name);
            assert!(
                seen.insert(d.name),
                "two destinations are called {}",
                d.name
            );
        }
        // The old names still find their panel.
        for (old, area) in [
            ("agent", super::FocusArea::ByteBotPanel),
            ("advice", super::FocusArea::AdvisePanel),
            ("blast radius", super::FocusArea::ImpactPanel),
        ] {
            assert_eq!(
                super::find_destination_by_name(old).map(|d| d.area),
                Some(area)
            );
        }
    }

    /// A settings row that renders its value must never render the secret:
    /// only a fixed-width bullet run plus a four-char tail, enough to tell two
    /// keys apart and nothing enough to read one.
    #[test]
    fn masked_secrets_keep_only_a_four_char_tail() {
        assert_eq!(mask_secret("sk-colab-abcdef123456"), "••••••••••••3456");
        let masked = mask_secret("sk-super-secret-value");
        assert!(masked.ends_with("lue"), "keeps the tail: {masked}");
        assert!(
            !masked.contains("super") && !masked.contains("secret"),
            "never leaks the middle: {masked}"
        );
        // Bullets stand in for the hidden prefix at a bounded width, so a long
        // key cannot push the row off the panel.
        assert!(mask_secret(&"a".repeat(200)).chars().count() <= 16);
    }

    #[test]
    fn short_and_empty_secrets_mask_to_a_fixed_placeholder() {
        assert_eq!(mask_secret(""), "••••");
        assert_eq!(mask_secret("abc"), "••••");
        // A value short enough to be fully revealed by the tail gets the bare
        // placeholder instead: the four-char tail is the whole key.
        assert_eq!(mask_secret("abcd"), "••••");
        assert_eq!(mask_secret("abcde"), "••••");
        assert_eq!(mask_secret("sk-12345"), "••••");
        // One char over the boundary is the first value that leaks a tail.
        assert_eq!(mask_secret("123456789"), "•••••6789");
    }

    #[test]
    fn every_focus_area_variant_has_a_disclosure_level_and_table_entry() {
        for area in FocusArea::ALL {
            let level = area.disclosure_level();
            assert!(
                level.rank() >= 1 && level.rank() <= 4,
                "FocusArea::{:?} must have a valid disclosure level (1..=4)",
                area
            );
            let dest = DESTINATIONS.iter().find(|d| d.area == area);
            assert!(
                dest.is_some(),
                "FocusArea::{:?} must have an entry in DESTINATIONS table",
                area
            );
            let dest = dest.unwrap();
            assert_eq!(
                dest.level, level,
                "FocusArea::{:?} disclosure_level() must match DESTINATIONS table level",
                area
            );
        }
    }

    #[test]
    fn destinations_table_has_all_27_focus_areas_uniquely() {
        assert_eq!(DESTINATIONS.len(), 27);
        let mut seen = std::collections::HashSet::new();
        for dest in DESTINATIONS {
            assert!(
                seen.insert(dest.area),
                "Duplicate FocusArea::{:?} in DESTINATIONS table",
                dest.area
            );
            assert!(!dest.name.is_empty());
            assert!(!dest.description.is_empty());
            assert!(!dest.command_name.is_empty());
        }
    }

    #[test]
    fn disclosure_filtering_hides_level_4_from_palette_and_first_run() {
        let l2_items = feature_items_for_level(DisclosureLevel::Level2);
        for (_name, _desc, area) in &l2_items {
            assert!(
                area.disclosure_level().rank() <= 2,
                "Level 2 palette must not contain {:?} (level {:?})",
                area,
                area.disclosure_level()
            );
        }

        let l2_shortcuts = first_run_shortcuts_line(DisclosureLevel::Level2);
        assert!(!l2_shortcuts.contains("bytebot"));
        assert!(!l2_shortcuts.contains("voice"));
        assert!(!l2_shortcuts.contains("collab"));

        // But level 4 palette contains all features
        let l4_items = feature_items_for_level(DisclosureLevel::Level4);
        assert!(l4_items
            .iter()
            .any(|(_, _, a)| *a == FocusArea::ByteBotPanel));
        assert!(l4_items
            .iter()
            .any(|(_, _, a)| *a == FocusArea::VoiceInterface));
    }

    #[test]
    fn deferred_destinations_remain_reachable_by_name() {
        // Even when deferred from discovery, all Level 4 destinations can be reached by name
        assert_eq!(
            find_destination_by_name("bytebot").map(|d| d.area),
            Some(FocusArea::ByteBotPanel)
        );
        assert_eq!(
            find_destination_by_name("/voice").map(|d| d.area),
            Some(FocusArea::VoiceInterface)
        );
        assert_eq!(
            find_destination_by_name("collab").map(|d| d.area),
            Some(FocusArea::CollaborationHub)
        );
        assert_eq!(
            find_destination_by_name("security").map(|d| d.area),
            Some(FocusArea::SecurityAuditor)
        );
    }
}
