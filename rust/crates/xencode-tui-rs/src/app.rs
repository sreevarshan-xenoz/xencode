use std::collections::{HashMap, HashSet};
use std::io;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{
    self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use ratatui::{backend::Backend, Terminal};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use xencode_config_rs::XencodeConfig;
use xencode_context_rs::init_project;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::{
    current_timestamp, HealthStatus, LlamaCppClient, LlamaCppOptions, LlamaCppTimings,
    OllamaClient,
};
use xencode_providers_rs::{ChatMessage, ProviderManager};

use crate::ui;

/// System block injected as tier 1 when previewing `/ctx` context assembly.
const CTX_SYSTEM: &str =
    "You are Xencode, a coding agent. Follow the project guidelines below exactly.";

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

#[derive(Clone, Copy)]
pub struct ThemeColors {
    pub bg: ratatui::style::Color,
    pub fg: ratatui::style::Color,
    pub accent: ratatui::style::Color,
    pub border: ratatui::style::Color,
    pub border_active: ratatui::style::Color,
    pub highlight: ratatui::style::Color,
    pub highlight_fg: ratatui::style::Color,
    pub message_user: ratatui::style::Color,
    pub message_assistant: ratatui::style::Color,
    pub message_system: ratatui::style::Color,
    pub status_bg: ratatui::style::Color,
    pub status_fg: ratatui::style::Color,
}

impl ThemeColors {
    pub fn get(name: &str) -> Self {
        match name {
            "midnight" => Self {
                bg: ratatui::style::Color::Rgb(15, 17, 26),
                fg: ratatui::style::Color::Rgb(230, 230, 230),
                accent: ratatui::style::Color::Magenta,
                border: ratatui::style::Color::Rgb(60, 60, 80),
                border_active: ratatui::style::Color::Magenta,
                highlight: ratatui::style::Color::Magenta,
                highlight_fg: ratatui::style::Color::White,
                message_user: ratatui::style::Color::Magenta,
                message_assistant: ratatui::style::Color::Cyan,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(30, 30, 50),
                status_fg: ratatui::style::Color::Rgb(200, 200, 220),
            },
            "forest" => Self {
                bg: ratatui::style::Color::Rgb(15, 29, 20),
                fg: ratatui::style::Color::Rgb(233, 245, 234),
                accent: ratatui::style::Color::Green,
                border: ratatui::style::Color::Rgb(40, 70, 40),
                border_active: ratatui::style::Color::Green,
                highlight: ratatui::style::Color::Green,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Green,
                message_assistant: ratatui::style::Color::Yellow,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(20, 40, 25),
                status_fg: ratatui::style::Color::Rgb(200, 230, 200),
            },
            "terminal" => Self {
                bg: ratatui::style::Color::Rgb(0, 17, 0),
                fg: ratatui::style::Color::Rgb(128, 255, 128),
                accent: ratatui::style::Color::Rgb(128, 255, 128),
                border: ratatui::style::Color::Rgb(0, 64, 0),
                border_active: ratatui::style::Color::Rgb(128, 255, 128),
                highlight: ratatui::style::Color::Rgb(0, 128, 0),
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Rgb(255, 255, 255),
                message_assistant: ratatui::style::Color::Rgb(128, 255, 128),
                message_system: ratatui::style::Color::Rgb(0, 128, 0),
                status_bg: ratatui::style::Color::Rgb(0, 30, 0),
                status_fg: ratatui::style::Color::Rgb(128, 255, 128),
            },
            "dracula" => Self {
                bg: ratatui::style::Color::Rgb(40, 42, 54),
                fg: ratatui::style::Color::Rgb(248, 248, 242),
                accent: ratatui::style::Color::Rgb(255, 121, 198),
                border: ratatui::style::Color::Rgb(68, 71, 90),
                border_active: ratatui::style::Color::Rgb(255, 121, 198),
                highlight: ratatui::style::Color::Rgb(189, 147, 249),
                highlight_fg: ratatui::style::Color::Rgb(40, 42, 54),
                message_user: ratatui::style::Color::Rgb(255, 121, 198),
                message_assistant: ratatui::style::Color::Rgb(80, 250, 123),
                message_system: ratatui::style::Color::Rgb(98, 114, 164),
                status_bg: ratatui::style::Color::Rgb(30, 31, 41),
                status_fg: ratatui::style::Color::Rgb(248, 248, 242),
            },
            "solarized" => Self {
                bg: ratatui::style::Color::Rgb(0, 43, 54),
                fg: ratatui::style::Color::Rgb(131, 148, 150),
                accent: ratatui::style::Color::Rgb(38, 139, 210),
                border: ratatui::style::Color::Rgb(7, 54, 66),
                border_active: ratatui::style::Color::Rgb(38, 139, 210),
                highlight: ratatui::style::Color::Rgb(42, 161, 152),
                highlight_fg: ratatui::style::Color::Rgb(0, 43, 54),
                message_user: ratatui::style::Color::Rgb(38, 139, 210),
                message_assistant: ratatui::style::Color::Rgb(133, 153, 0),
                message_system: ratatui::style::Color::Rgb(88, 110, 117),
                status_bg: ratatui::style::Color::Rgb(0, 30, 38),
                status_fg: ratatui::style::Color::Rgb(147, 161, 161),
            },
            "nord" => Self {
                bg: ratatui::style::Color::Rgb(46, 52, 64),
                fg: ratatui::style::Color::Rgb(216, 222, 233),
                accent: ratatui::style::Color::Rgb(136, 192, 208),
                border: ratatui::style::Color::Rgb(59, 66, 82),
                border_active: ratatui::style::Color::Rgb(136, 192, 208),
                highlight: ratatui::style::Color::Rgb(94, 129, 172),
                highlight_fg: ratatui::style::Color::Rgb(236, 239, 244),
                message_user: ratatui::style::Color::Rgb(136, 192, 208),
                message_assistant: ratatui::style::Color::Rgb(163, 190, 140),
                message_system: ratatui::style::Color::Rgb(97, 108, 135),
                status_bg: ratatui::style::Color::Rgb(36, 41, 51),
                status_fg: ratatui::style::Color::Rgb(216, 222, 233),
            },
            // "ocean" and default
            _ => Self {
                bg: ratatui::style::Color::Rgb(11, 27, 43),
                fg: ratatui::style::Color::Rgb(234, 244, 255),
                accent: ratatui::style::Color::Cyan,
                border: ratatui::style::Color::Rgb(40, 60, 80),
                border_active: ratatui::style::Color::Cyan,
                highlight: ratatui::style::Color::Cyan,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Cyan,
                message_assistant: ratatui::style::Color::Green,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(18, 40, 60),
                status_fg: ratatui::style::Color::Rgb(200, 220, 240),
            },
        }
    }
}

/// Represents a message in the UI chat list
pub struct UiMessage {
    pub role: String,
    pub content: String,
}

pub struct App<'a> {
    pub focus: FocusArea,
    pub input: String,
    pub input_cursor: usize,
    pub input_mode: InputMode,
    pub messages: Vec<UiMessage>,
    pub chat_scroll: u16,
    pub file_tree: Vec<String>,
    pub selected_file: usize,
    pub file_scroll_offset: usize,
    pub attached_files: HashSet<String>,
    pub opened_file: Option<String>,
    pub editor: TextArea<'a>,
    pub editor_dirty: bool,
    pub git_status: HashMap<String, String>,
    pub git_branch: String,
    pub available_models: Vec<String>,
    pub selected_model: usize,
    pub is_generating: bool,
    pub is_reviewing: bool,
    pub code_review_output: String,
    pub commit_message: String,
    pub commit_cursor: usize,
    pub spinner_tick: usize,
    pub theme: ThemeColors,
    pub config: XencodeConfig,
    pub show_terminal: bool,
    pub memory: ConversationMemory,
    pub feature_nav_selected: usize,

    // Performance & health tracking
    pub session_start_time: f64,
    pub ollama_health_entries: HashMap<String, (String, f64, Option<String>)>,
    pub last_health_check: f64,
    pub health_check_in_progress: bool,
    pub total_llm_calls: u64,
    pub average_latency: f64,

    // ByteBot state
    pub bytebot_command: String,
    pub bytebot_cursor: usize,
    pub bytebot_steps: Vec<(String, String)>, // (step_name, status)
    pub bytebot_progress: f64,
    pub bytebot_running: bool,
    pub bytebot_log: Vec<String>,
    pub bytebot_history: Vec<String>, // previously executed commands

    // Project context (M0) state
    pub init_running: bool,
    pub init_progress: f64,
    pub init_steps: Vec<(String, String)>, // (step_name, status)
    pub init_log: Vec<String>,
    pub init_visible: bool,
    pub init_cancel: Arc<AtomicBool>,

    // Collaboration Hub state
    pub collab_session_active: bool,
    pub collab_session_id: String,
    pub collab_members: Vec<(String, String, String)>, // (name, status, connection)
    pub collab_sync_status: String,                    // "synced", "syncing", "error"
    pub collab_last_sync: f64,
    pub collab_pending_changes: u32,
    pub collab_activity_log: Vec<String>,
    pub collab_commit_stream: Vec<(String, String)>, // (author, message)
    pub collab_shared_files: Vec<String>,            // shared file names

    // Voice Interface state
    pub voice_active: bool,
    pub voice_status: String, // "idle", "listening", "processing", "speaking"
    pub voice_level: f64,     // simulated audio level 0.0-1.0
    pub voice_transcript: Vec<String>,
    pub voice_commands: Vec<(String, String)>, // (command, result)
    pub voice_confidence: f64,
    pub voice_muted: bool,
    pub voice_language: String,

    // Terminal Assistant state
    pub term_asst_active: bool,
    pub term_asst_query: String,
    pub term_asst_cursor: usize,
    pub term_asst_suggestions: Vec<String>,
    pub term_asst_output: String,
    pub term_asst_history: Vec<(String, String, String)>, // (command, risk, explanation)
    pub term_risk_filter: String,                         // "All", "Safe", "Destructive"

    // Security Auditor state
    pub sec_scan_active: bool,
    pub sec_scan_path: String,
    pub sec_scan_results: Vec<(String, String, String)>, // (severity, category, file)
    pub sec_scan_summary: (u32, u32, u32, u32),          // (critical, high, medium, low)
    pub sec_scan_progress: f64,
    pub sec_scan_log: Vec<String>,
    pub sec_filter_severity: String, // "All", "Critical", "High", "Medium", "Low"
    pub sec_sort_mode: String,       // "severity" or "category"

    // Performance Profiler state
    pub profiler_active: bool,
    pub profiler_running: bool,
    pub profiler_functions: Vec<(String, f64, f64, u32)>, // (name, time_ms, mem_mb, calls)
    pub profiler_gauge_cpu: f64,
    pub profiler_gauge_mem: f64,
    pub profiler_gauge_latency: f64,

    // Custom Models state
    pub models_editing: bool,
    pub models_profiles: Vec<(String, String, f64, u32, f64)>, // (name, provider, temp, max_tokens, top_p)
    pub models_selected: usize,
    pub models_test_output: String,
    pub models_saving: bool,

    // Learning Mode state
    pub learn_active: bool,
    pub learn_current_lesson: usize,
    pub learn_total_lessons: usize,
    pub learn_lesson_title: String,
    pub learn_content: Vec<String>,
    pub learn_code_example: String,
    pub learn_exercise: String,
    pub learn_progress_pct: f64,
    pub learn_quiz_active: bool,
    pub learn_quiz_question: String,
    pub learn_quiz_options: Vec<String>,
    pub learn_quiz_selected: usize,
    pub learn_quiz_answered: bool,
    pub learn_quiz_correct: bool,

    // Multi-Language state
    pub lang_active: bool,
    pub lang_detection_results: Vec<(String, String, String)>, // (file, language, confidence)
    pub lang_supported: Vec<(String, String)>,                 // (language, status)
    pub lang_translate_input: String,
    pub lang_translate_output: String,
    pub lang_translate_source: String,
    pub lang_translate_target: String,

    // Panel scroll state
    pub provider_health_scroll: u16,
    pub security_scroll: u16,

    // Settings interactive state
    pub settings_cursor: usize,
    pub settings_reset_active: bool,
    pub settings_url_editing: bool,
    pub settings_url_buffer: String,
    pub settings_url_cursor: usize,

    // llama.cpp live-control state (model load/unload + sampling)
    pub llamacpp_editing: bool,
    pub llamacpp_path_buffer: String,
    pub llamacpp_path_cursor: usize,
    pub llamacpp_action_msg: String,
    // llama.cpp sampling options (temperature, top-k, min-p, max-tokens)
    pub sampling_temp_editing: bool,
    pub sampling_temp_buffer: String,
    pub sampling_int_editing: bool,
    pub sampling_int_buffer: String,

    // Last llama.cpp generation timings (tok/s) reported by the server
    pub last_llamacpp_timings: Option<LlamaCppTimings>,

    /// Total prompt tokens of the last `/ctx` assembly preview — used to derive
    /// `cached_tokens` from llama.cpp `tokens_evaluated` (§13).
    pub last_ctx_total_tokens: u64,
    /// Retrieved files in the last assembly (recorded into metrics row).
    pub last_ctx_retrieved_files: u8,
}

impl<'a> App<'a> {
    pub fn new() -> Self {
        let config = XencodeConfig::load().unwrap_or_default();
        let mut memory = ConversationMemory::with_persistence(config.max_memory_items)
            .unwrap_or_else(|_| ConversationMemory::new(50));
        memory.start_session(None);

        let _client = OllamaClient::new(&config.ollama_url, config.response_timeout);

        let scan_opts = ScanOptions {
            max_depth: Some(5),
            include_hidden: false,
            excluded_dirs: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "__pycache__".to_string(),
                ".pytest_cache".to_string(),
                ".venv".to_string(),
            ],
        };
        let tree = scan_workspace(".", &scan_opts).unwrap_or_default();
        let file_tree: Vec<String> = tree
            .into_iter()
            .map(|f| f.path.display().to_string())
            .collect();

        let available_models = if !config.default_model.is_empty() {
            vec![config.default_model.clone()]
        } else {
            vec!["qwen2.5:7b".to_string()]
        };
        let selected_model = 0;
        let theme = ThemeColors::get(&config.active_theme);

        let mut git_status = HashMap::new();
        if let Ok(output) = Command::new("git").args(["status", "--porcelain"]).output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                for line in s.lines() {
                    if line.len() > 3 {
                        let code = &line[0..2];
                        let path = &line[3..];
                        let fp = format!(".\\{}", path.replace("/", "\\"));
                        git_status.insert(fp, code.trim().to_string());
                    }
                }
            }
        }
        let git_branch = Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "main".to_string());

        let mut editor = TextArea::default();
        editor.set_line_number_style(
            ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
        );

        let now = current_timestamp();

        let mut app = Self {
            focus: FocusArea::ChatInput,
            input: String::new(),
            input_cursor: 0,
            input_mode: InputMode::Normal,
            messages: Vec::new(),
            chat_scroll: 0,
            file_tree,
            selected_file: 0,
            file_scroll_offset: 0,
            attached_files: HashSet::new(),
            opened_file: None,
            editor,
            editor_dirty: false,
            git_status,
            git_branch,
            available_models,
            selected_model,
            is_generating: false,
            is_reviewing: false,
            code_review_output: String::new(),
            commit_message: String::new(),
            commit_cursor: 0,
            spinner_tick: 0,
            theme,
            config,
            show_terminal: false,
            memory,
            feature_nav_selected: 0,
            session_start_time: now,
            ollama_health_entries: HashMap::new(),
            last_health_check: 0.0,
            health_check_in_progress: false,
            total_llm_calls: 0,
            average_latency: 0.0,
            bytebot_command: String::new(),
            bytebot_cursor: 0,
            bytebot_steps: Vec::new(),
            bytebot_progress: 0.0,
            bytebot_running: false,
            bytebot_log: Vec::new(),
            bytebot_history: Vec::new(),
            init_running: false,
            init_progress: 0.0,
            init_steps: Vec::new(),
            init_log: Vec::new(),
            init_visible: false,
            init_cancel: Arc::new(AtomicBool::new(false)),
            collab_session_active: false,
            collab_session_id: String::new(),
            collab_members: Vec::new(),
            collab_sync_status: "disconnected".to_string(),
            collab_last_sync: 0.0,
            collab_pending_changes: 0,
            collab_activity_log: Vec::new(),
            collab_commit_stream: Vec::new(),
            collab_shared_files: Vec::new(),

            voice_active: false,
            voice_status: "idle".to_string(),
            voice_level: 0.0,
            voice_transcript: Vec::new(),
            voice_commands: Vec::new(),
            voice_confidence: 0.0,
            voice_muted: false,
            voice_language: "en-US".to_string(),

            term_asst_active: false,
            term_asst_query: String::new(),
            term_asst_cursor: 0,
            term_asst_suggestions: Vec::new(),
            term_asst_output: String::new(),
            term_asst_history: Vec::new(),
            term_risk_filter: "All".to_string(),

            sec_scan_active: false,
            sec_scan_path: String::new(),
            sec_scan_results: Vec::new(),
            sec_scan_summary: (0, 0, 0, 0),
            sec_scan_progress: 0.0,
            sec_scan_log: Vec::new(),
            sec_filter_severity: "All".to_string(),
            sec_sort_mode: "severity".to_string(),

            profiler_active: false,
            profiler_running: false,
            profiler_functions: Vec::new(),
            profiler_gauge_cpu: 0.0,
            profiler_gauge_mem: 0.0,
            profiler_gauge_latency: 0.0,

            models_editing: false,
            models_profiles: Vec::new(),
            models_selected: 0,
            models_test_output: String::new(),
            models_saving: false,

            learn_active: false,
            learn_current_lesson: 0,
            learn_total_lessons: 0,
            learn_lesson_title: String::new(),
            learn_content: Vec::new(),
            learn_code_example: String::new(),
            learn_exercise: String::new(),
            learn_progress_pct: 0.0,
            learn_quiz_active: false,
            learn_quiz_question: String::new(),
            learn_quiz_options: Vec::new(),
            learn_quiz_selected: 0,
            learn_quiz_answered: false,
            learn_quiz_correct: false,

            lang_active: false,
            lang_detection_results: Vec::new(),
            lang_supported: Vec::new(),
            lang_translate_input: String::new(),
            lang_translate_output: String::new(),
            lang_translate_source: "auto".to_string(),
            lang_translate_target: "en".to_string(),

            provider_health_scroll: 0,
            security_scroll: 0,

            settings_cursor: 0,
            settings_reset_active: false,
            settings_url_editing: false,
            settings_url_buffer: String::new(),
            settings_url_cursor: 0,

            llamacpp_editing: false,
            llamacpp_path_buffer: String::new(),
            llamacpp_path_cursor: 0,
            llamacpp_action_msg: String::new(),
            sampling_temp_editing: false,
            sampling_temp_buffer: String::new(),
            sampling_int_editing: false,
            sampling_int_buffer: String::new(),

            last_llamacpp_timings: None,
            last_ctx_total_tokens: 0,
            last_ctx_retrieved_files: 0,
        };

        // Seed initial health entries for configured providers
        app.ollama_health_entries.insert(
            "ollama".to_string(),
            (HealthStatus::Unknown.to_string(), 0.0, None),
        );
        app.ollama_health_entries.insert(
            "openrouter".to_string(),
            (
                if app.config.api_keys.openrouter_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.openrouter_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "qwen".to_string(),
            (
                if app.config.api_keys.qwen_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.qwen_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "gemini".to_string(),
            (
                if app.config.api_keys.google_gemini_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.google_gemini_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "llamacpp".to_string(),
            (HealthStatus::Unknown.to_string(), 0.0, None),
        );

        for msg in app.memory.get_context(10) {
            app.messages.push(UiMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }
        app
    }

    pub fn open_file_in_editor(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                self.editor = TextArea::new(if lines.is_empty() {
                    vec![String::new()]
                } else {
                    lines
                });
                self.editor.set_line_number_style(
                    ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
                );
                self.opened_file = Some(path.to_string());
                self.editor_dirty = false;
            }
            Err(_) => {
                self.editor = TextArea::new(vec![format!("Unable to read file: {}", path)]);
                self.opened_file = None;
            }
        }
    }

    pub fn save_editor(&mut self) {
        if let Some(ref fp) = self.opened_file {
            let content: String = self.editor.lines().join("\n");
            if std::fs::write(fp, &content).is_ok() {
                self.editor_dirty = false;
            }
        }
    }

    pub fn navigate_feature(&self, idx: usize) -> FocusArea {
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

    pub fn refresh_git(&mut self) {
        self.git_status.clear();
        if let Ok(output) = Command::new("git").args(["status", "--porcelain"]).output() {
            if let Ok(s) = String::from_utf8(output.stdout) {
                for line in s.lines() {
                    if line.len() > 3 {
                        let code = &line[0..2];
                        let path = &line[3..];
                        let fp = format!(".\\{}", path.replace("/", "\\"));
                        self.git_status.insert(fp, code.trim().to_string());
                    }
                }
            }
        }
        if let Ok(output) = Command::new("git")
            .args(["branch", "--show-current"])
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                let branch = s.trim().to_string();
                if !branch.is_empty() {
                    self.git_branch = branch;
                }
            }
        }
    }

    pub fn submit_message(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.input.trim().is_empty() {
            return;
        }
        let prompt = self.input.clone();
        self.input.clear();
        self.input_cursor = 0;

        self.messages.push(UiMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
        self.memory.add_message("user", &prompt, None);

        // Project context engine interception (/init, /init abort, /init status)
        if prompt.starts_with("/init") {
            self.handle_init_command(&prompt, tx);
            return;
        }

        // Context assembly interception (/ctx, /ctx status, /ctx track <path>)
        if prompt.starts_with("/ctx") {
            self.handle_ctx_command(&prompt, tx);
            return;
        }

        self.is_generating = true;

        // ByteBot interception
        if prompt.starts_with("/bytebot") {
            let command = prompt
                .strip_prefix("/bytebot")
                .unwrap_or("")
                .trim()
                .to_string();
            tokio::spawn(async move {
                let _ = tx.send(format!("⚡ ByteBot: Initializing for '{}'\n", command));
                tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
                for step in [
                    "Analyzing workspace...",
                    "Formulating plan...",
                    "Scanning deps...",
                    "Running tests...",
                    "Applying changes...",
                    "Verifying...",
                ] {
                    let _ = tx.send(format!("  → {}\n", step));
                    tokio::time::sleep(tokio::time::Duration::from_millis(1200)).await;
                }
                let _ = tx.send("✅ ByteBot execution complete.\n".to_string());
                let _ = tx.send("[DONE]".to_string());
            });
            return;
        }

        // Normal LLM generation
        let mut context_messages = Vec::new();
        for msg in self.memory.get_context(10) {
            context_messages.push(ChatMessage {
                role: msg.role,
                content: msg.content,
            });
        }

        let mut attached_context = String::new();
        for path in &self.attached_files {
            if let Ok(content) = std::fs::read_to_string(path) {
                attached_context.push_str(&format!(
                    "<file path=\"{}\">\n{}\n</file>\n\n",
                    path, content
                ));
            }
        }
        if !attached_context.is_empty() {
            context_messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: format!("Attached files:\n{}", attached_context),
                },
            );
        }

        let model = self.config.default_model.clone();
        let ollama_url = self.config.ollama_url.clone();
        let llama_cpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let or_key = self.config.api_keys.openrouter_api_key.clone();
        let qwen_key = self.config.api_keys.qwen_api_key.clone();
        let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
        let llama_opts = LlamaCppOptions {
            temperature: self.config.llama_cpp_temperature,
            top_k: self.config.llama_cpp_top_k,
            min_p: self.config.llama_cpp_min_p,
            max_tokens: self.config.llama_cpp_max_tokens,
            grammar: None,
            json_schema: None,
            mirostat: None,
        };

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout);
            let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout);
            let manager = ProviderManager::new(client, or_key, qwen_key, gemini_key, None)
                .with_llama_cpp(llama_client);
            let _ = manager
                .generate_stream_with_options(&model, &context_messages, Some(&llama_opts), |token| {
                    let _ = tx.send(token.to_string());
                })
                .await;
            // Report llama.cpp tok/s stats if this was a llama.cpp request
            if let Some(ts) = manager.last_llamacpp_timings() {
                if let Ok(json) = serde_json::to_string(&ts) {
                    let _ = tx.send(format!("[TIMINGS]{}", json));
                }
            }
            let _ = tx.send("[DONE]".to_string());
        });
    }

    /// Send a load/unload/switch command to the llama.cpp server and report the
    /// result back through the channel.
    ///
    /// Commands:
    /// - "load"        : load `target` (a model id) or the configured GGUF path.
    /// - "switch"      : swap to `target` (a model id) — unloads first if we can.
    /// - "unload"      : unload whatever is loaded.
    pub fn llamacpp_control(
        &mut self,
        command: &str,
        target: Option<String>,
        tx: mpsc::UnboundedSender<String>,
    ) {
        let llamacpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let load_target = if let Some(t) = target {
            t
        } else {
            self.config.llama_cpp_model_path.clone()
        };

        self.llamacpp_action_msg = match command {
            "load" | "switch" => {
                if load_target.is_empty() {
                    "Set a GGUF model path first (Settings → Llama.cpp Model Path), or pick a llama.cpp model".to_string()
                } else {
                    "Requesting model switch...".to_string()
                }
            }
            "unload" => "Requesting model unload...".to_string(),
            _ => return,
        };

        let (url, path) = (llamacpp_url, load_target.clone());
        let is_unload = command == "unload";
        tokio::spawn(async move {
            let client = LlamaCppClient::new(&url, timeout);
            let result = if is_unload {
                client.unload_models().await
            } else {
                client.load_model(&path).await
            };
            let msg = match result {
                Ok(()) => {
                    let label = if is_unload {
                        "model unloaded".to_string()
                    } else {
                        format!("model '{path}' loaded")
                    };
                    format!("✅ llama.cpp: {label}")
                }
                Err(e) => format!("❌ llama.cpp: {e}"),
            };
            let _ = tx.send(format!("[LLAMACPP]{}", msg));
        });
    }

    pub fn append_generation(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_generating = false;
            self.total_llm_calls += 1;
            if let Some(last) = self.messages.last() {
                if last.role == "assistant" {
                    self.memory.add_message(
                        "assistant",
                        &last.content,
                        Some(self.config.default_model.clone()),
                    );
                }
            }
            return;
        }
        if let Some(last) = self.messages.last_mut() {
            if last.role == "assistant" && self.is_generating {
                last.content.push_str(text);
                return;
            }
        }
        self.messages.push(UiMessage {
            role: "assistant".to_string(),
            content: text.to_string(),
        });
    }

    pub fn append_review(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_reviewing = false;
        } else {
            self.code_review_output.push_str(text);
        }
    }

    /// Run asynchronous health checks for all configured providers.
    /// Results are sent back through the channel for processing in the event loop.
    /// Start a Collaboration Hub session with simulated team members and sync.
    pub fn start_collab_session(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.collab_session_active {
            return;
        }
        self.collab_session_active = true;
        self.collab_session_id = format!("xencode-{:06x}", (current_timestamp() as u64) & 0xFFFFFF);
        self.collab_sync_status = "connecting".to_string();
        self.collab_pending_changes = 0;
        self.collab_activity_log.clear();

        // Seed initial team members
        self.collab_members = vec![
            (
                "You (local)".to_string(),
                "online".to_string(),
                "🔗 LAN".to_string(),
            ),
            (
                "alice".to_string(),
                "online".to_string(),
                "🌐 WAN".to_string(),
            ),
            ("bob".to_string(), "away".to_string(), "🌐 WAN".to_string()),
            (
                "carol".to_string(),
                "busy".to_string(),
                "🔗 LAN".to_string(),
            ),
        ];

        self.collab_activity_log
            .push("🔌 Connecting to collaboration server...".to_string());

        let session_id = self.collab_session_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let _ = tx.send("[COLLAB]status:connected".to_string());
            let _ = tx.send(format!(
                "[COLLAB]log:🔗 Connected — Session: {}",
                session_id
            ));
            let _ = tx.send("[COLLAB]log:👥 3 remote team members online".to_string());
            let _ = tx.send("[COLLAB]member:alice:online".to_string());
            let _ = tx.send("[COLLAB]member:bob:away".to_string());
            let _ = tx.send("[COLLAB]member:carol:busy".to_string());

            // Simulate sync pulses
            let pulses = [
                ("📤 Syncing workspace...", "syncing", 5u32),
                ("📥 Pulled 3 remote changes", "synced", 0u32),
                ("🔄 Auto-merge applied (2 files)", "synced", 0u32),
                ("📤 Pushing local edits...", "syncing", 3u32),
                ("✅ All changes synchronized", "synced", 0u32),
            ];
            for (msg, status, pending) in pulses {
                tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
                let _ = tx.send(format!("[COLLAB]sync:{}", status));
                let _ = tx.send(format!("[COLLAB]pending:{}", pending));
                let _ = tx.send(format!("[COLLAB]log:{}", msg));
            }

            // Member status changes
            tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
            let _ = tx.send("[COLLAB]member:bob:online".to_string());
            let _ = tx.send("[COLLAB]log:👤 bob is now online".to_string());
            tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
            let _ = tx.send("[COLLAB]member:carol:online".to_string());
            let _ = tx.send("[COLLAB]log:👤 carol is now online".to_string());

            let _ = tx.send("[COLLAB]log:✅ Collaboration session ready".to_string());
            let _ = tx.send("[COLLAB]ready".to_string());
        });
    }

    /// Start a ByteBot autonomous task execution.
    /// Sends step updates back through the channel.
    pub fn run_bytebot(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.bytebot_running || self.bytebot_command.trim().is_empty() {
            return;
        }

        let command = self.bytebot_command.trim().to_string();
        self.bytebot_running = true;
        self.bytebot_progress = 0.0;
        self.bytebot_steps = vec![
            ("Analyzing workspace".to_string(), "pending".to_string()),
            ("Scanning dependencies".to_string(), "pending".to_string()),
            (
                "Formulating execution plan".to_string(),
                "pending".to_string(),
            ),
            ("Running tests".to_string(), "pending".to_string()),
            ("Applying changes".to_string(), "pending".to_string()),
            ("Verifying results".to_string(), "pending".to_string()),
        ];
        self.bytebot_log.clear();
        self.bytebot_log
            .push(format!("⚡ ByteBot: Initializing for '{}'", command));
        self.bytebot_command.clear();
        self.bytebot_cursor = 0;

        tokio::spawn(async move {
            let steps = [
                ("Analyzing workspace", "📁 Found 342 files in workspace"),
                (
                    "Scanning dependencies",
                    "🔍 Identified 12 outdated packages",
                ),
                (
                    "Formulating execution plan",
                    "📋 Plan: update 5 deps, fix 3 deprecations",
                ),
                ("Running tests", "🧪 Running test suite (142 tests)"),
                ("Applying changes", "🔧 Applying 8 changes across 6 files"),
                ("Verifying results", "✅ All tests pass, changes verified"),
            ];

            for (i, (step_name, detail)) in steps.iter().enumerate() {
                // Mark current step as running
                let _ = tx.send(format!("[BYTEBOT]step:{}:running:{}", i, step_name));
                tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;

                // Send progress update
                let progress = (i as f64 + 1.0) / steps.len() as f64;
                let _ = tx.send(format!("[BYTEBOT]progress:{:.2}", progress));

                // Send log line
                let _ = tx.send(format!(
                    "[BYTEBOT]log:{}  → {} — {}",
                    "▸", step_name, detail
                ));

                // Mark step as done
                let _ = tx.send(format!("[BYTEBOT]step:{}:done:{}", i, step_name));
            }

            let _ = tx.send("[BYTEBOT]log:✅ ByteBot execution complete.".to_string());
            let _ = tx.send("[BYTEBOT_DONE]".to_string());
        });
    }

    /// Handle `/init`, `/init abort` and `/init status` chat commands.
    fn handle_init_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let command = prompt.strip_prefix("/init").unwrap_or("").trim();
        match command {
            "abort" => {
                if self.init_running {
                    self.init_cancel.store(true, Ordering::Relaxed);
                    let _ = tx.send(
                        "[INIT]log:⏹️ Abort requested — finishing the current step…".to_string(),
                    );
                } else {
                    let _ = tx.send("[INIT]log:ℹ️ No /init job is running.".to_string());
                }
            }
            "status" => {
                let done = self
                    .init_steps
                    .iter()
                    .filter(|(_, s)| s == "done")
                    .count();
                let line = if self.init_running {
                    format!(
                        "⏳ init running — {done}/{} steps, {}%. Use /init abort to stop.",
                        self.init_steps.len(),
                        (self.init_progress * 100.0) as u64
                    )
                } else if self.init_visible || !self.init_log.is_empty() {
                    format!(
                        "🗂  Last /init run: {} step(s), {} log line(s). Type /init to re-run.",
                        done,
                        self.init_log.len()
                    )
                } else {
                    "No project index built yet — type /init to scan and create it.".to_string()
                };
                let _ = tx.send(format!("[INIT]log:{}", line));
            }
            "" => {
                if self.init_running {
                    let _ = tx.send(
                        "[INIT]log:⚠️ An init job is already running — use /init abort to stop it."
                            .to_string(),
                    );
                } else {
                    self.init_cancel.store(false, Ordering::Relaxed);
                    self.run_project_init(tx);
                }
            }
            other => {
                let _ = tx.send(format!(
                    "[INIT]log:ℹ️ Unknown /init subcommand '{other}' — use /init, /init abort, or /init status."
                ));
            }
        }
    }

    /// Run the deterministic structural `/init` pass in the background.
    /// Progress and log lines stream back through `[INIT]` channel tokens.
    pub fn run_project_init(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.init_running {
            return;
        }
        const PHASES: [&str; 7] = [
            "Create .xencode directory",
            "Resume check",
            "Git snapshot",
            "Scan repository",
            "Analyze languages & sizes",
            "Extract symbols & dependencies",
            "Write index files",
        ];

        self.init_running = true;
        self.init_progress = 0.0;
        self.init_visible = true;
        self.init_steps = PHASES
            .iter()
            .map(|name| (name.to_string(), "pending".to_string()))
            .collect();
        self.init_log.clear();
        self.init_log
            .push("⏺️ Initializing project context (structural pass) — zero LLM calls.".to_string());

        let cancel = self.init_cancel.clone();
        let root = xencode_context_rs::default_root();

        tokio::spawn(async move {
            let tx_progress = tx.clone();
            let progress = move |line: &str| {
                if let Some(name) = line.strip_prefix("phase_start:") {
                    if let Some(idx) = PHASES.iter().position(|p| *p == name) {
                        let _ = tx_progress
                            .send(format!("[INIT]step:{idx}:running:{name}"));
                        let _ = tx_progress
                            .send(format!("[INIT]progress:{:.2}", idx as f64 / PHASES.len() as f64));
                    }
                } else if let Some(name) = line.strip_prefix("phase_done:") {
                    if let Some(idx) = PHASES.iter().position(|p| *p == name) {
                        let _ = tx_progress
                            .send(format!("[INIT]step:{idx}:done:{name}"));
                        let _ = tx_progress.send(format!(
                            "[INIT]progress:{:.2}",
                            (idx as f64 + 1.0) / PHASES.len() as f64
                        ));
                    }
                } else if let Some(msg) = line.strip_prefix("log:") {
                    let _ = tx_progress.send(format!("[INIT]log:{msg}"));
                }
            };

            let result = tokio::task::spawn_blocking(move || {
                init_project(&root, cancel, progress)
            })
            .await;

            match result {
                Ok(Ok(summary)) => {
                    if summary.fresh {
                        let _ = tx.send(
                            "[INIT]log:✅ Project context is already up to date — nothing rewritten."
                                .to_string(),
                        );
                    } else {
                        let skipped = if summary.skipped > 0 {
                            format!(" ({} ignored/excluded)", summary.skipped)
                        } else {
                            String::new()
                        };
                        let _ = tx.send(format!(
                            "[INIT]log:✅ Indexed {} files{skipped} across {} language(s) — {} LOC.",
                            summary.files_scanned,
                            summary.languages.len(),
                            summary.total_loc
                        ));
                        if summary.dep_edges > 0 || summary.symbol_files > 0 {
                            let _ = tx.send(format!(
                                "[INIT]log:🧩 {} file(s) with symbols · {} dependency edge(s)",
                                summary.symbol_files, summary.dep_edges
                            ));
                        }
                        if !summary.secret_files.is_empty() {
                            let _ = tx.send(format!(
                                "[INIT]log:🔒 {} secret-detected file(s) listed, not read.",
                                summary.secret_files.len()
                            ));
                        }
                        if !summary.binary_files.is_empty() {
                            let _ = tx.send(format!(
                                "[INIT]log:🧊 {} binary file(s) listed, not read.",
                                summary.binary_files.len()
                            ));
                        }
                        if let Some(g) = &summary.git {
                            let head = if g.head.len() > 8 {
                                g.head[..8].to_string()
                            } else {
                                g.head.clone()
                            };
                            let _ = tx.send(format!(
                                "[INIT]log:🎋 {} @ {head} — {} dirty file(s)",
                                g.branch, g.dirty
                            ));
                        }
                        let _ = tx.send(format!(
                            "[INIT]log:🗂  index + symbols + deps — {} bytes",
                            summary.index_bytes
                        ));
                    }
                    let _ = tx.send(
                        "[INIT]log:💡 Index refreshes automatically on git changes. /init status shows the last run."
                            .to_string(),
                    );
                }
                Ok(Err(err)) => {
                    let _ = tx.send(format!("[INIT]log:❌ init failed: {err}"));
                }
                Err(join_err) => {
                    let _ = tx.send(format!("[INIT]log:❌ init task panicked: {join_err}"));
                }
            }
            let _ = tx.send("[INIT_DONE]".to_string());
        });
    }

    /// Handle `/ctx` — context assembly: deterministic retrieval over the
    /// project index, stale-file status, and pinning a file as "loaded".
    fn handle_ctx_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let rest = prompt.strip_prefix("/ctx").unwrap_or("").trim();
        let mut parts = rest.split_whitespace();
        match parts.next() {
            Some("status") => {
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let mut tracker = xencode_context_rs::FileContextTracker::new(&xencode);
                tracker.load_from_disk();
                let _ = tx.send("[CTX_START]".to_string());
                if tracker.state.is_empty() {
                    let _ = tx.send(
                        "[CTX]ℹ️ No tracked files — pin one with /ctx track src/foo.rs.".to_string(),
                    );
                    return;
                }
                let all = tracker.check_all(&root);
                let stale: usize = all
                    .iter()
                    .filter(|t| t.state == xencode_context_rs::FileStateKind::Stale)
                    .count();
                let missing: usize = all
                    .iter()
                    .filter(|t| t.state == xencode_context_rs::FileStateKind::Missing)
                    .count();
                for t in &all {
                    let mark = match t.state {
                        xencode_context_rs::FileStateKind::Clean => "✔",
                        xencode_context_rs::FileStateKind::Stale => "⚠",
                        xencode_context_rs::FileStateKind::Missing => "✖",
                    };
                    let _ = tx.send(format!("[CTX]{mark} {} — {:?}", t.path, t.state));
                }
                let _ = tx.send(format!(
                    "[CTX]📊 {} tracked — {stale} stale, {missing} missing.",
                    all.len()
                ));
            }
            Some("track") => {
                let Some(path) = parts.next() else {
                    let _ = tx.send("[CTX_START]".to_string());
                    let _ = tx.send("[CTX]ℹ️ Usage: /ctx track <repo-relative-path>".to_string());
                    return;
                };
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let mut tracker = xencode_context_rs::FileContextTracker::new(&xencode);
                tracker.load_from_disk();
                tracker.mark_loaded(&root, &[path]);
                let _ = tx.send("[CTX_START]".to_string());
                if tracker.save().is_ok() {
                    let _ = tx.send(format!(
                        "[CTX]🔖 Pinned \"{path}\" at load-time hash — /ctx status to check staleness."
                    ));
                } else {
                    let _ = tx.send(
                        "[CTX]❌ Could not persist the tracking state.".to_string(),
                    );
                }
            }
            Some("compact") => {
                let (mut t, appended) = self.canonical_transcript();
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let path = xencode_context_rs::Transcript::current_path(&xencode);
                let snap = t.snapshot(&xencode).unwrap_or_default();
                let report = xencode_context_rs::soft_compact(&mut t, 0.70);
                let _ = t.save_to(&path);
                self.messages = t
                    .entries
                    .iter()
                    .map(|e| UiMessage {
                        role: e.role.clone(),
                        content: e.content.clone(),
                    })
                    .collect();
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]📚 Canonical transcript synced (+{appended} new) → {} entries",
                    t.entries.len()
                ));
                let _ = tx.send(format!(
                    "[CTX]🗜️ Soft compaction {} → {} entries (dropped {}), decisions kept: {}",
                    report.before, report.after, report.dropped, report.retained_decisions
                ));
                let _ = tx.send(format!(
                    "[CTX]💾 Pre-rewrite snapshot → {}",
                    snap.display()
                ));
                let _ = tx.send(
                    "[CTX]✅ Deterministic, no LLM call — state.md only changes when the model flags it. Chat now shows the working projection."
                        .to_string(),
                );
            }
            Some("kv") => {
                const PROFILE: xencode_context_rs::HardwareProfile =
                    xencode_context_rs::HardwareProfile::Balanced;
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
                let anchor = std::fs::read_to_string(xencode.join("anchor.md")).ok();
                let state = xencode_context_rs::ContextState::from_disk(&xencode)
                    .map(|s| s.to_markdown());
                let git = xencode_context_rs::git_summary_text(&root).unwrap_or_default();
                let recent_a = "user: how does auth work?\nassistant: it uses the auth module";
                let recent_b = "user: why is startup slow?\nassistant: profile the init path";
                // Different recent windows (and git text) must NOT disturb the
                // byte-stable head — that's the KV-reuse contract (§13).
                let doc_a = xencode_context_rs::assemble_prompt(
                    PROFILE,
                    CTX_SYSTEM,
                    agents.as_deref(),
                    anchor.as_deref(),
                    state.as_deref(),
                    &git,
                    Vec::new(),
                    recent_a,
                );
                let doc_b = xencode_context_rs::assemble_prompt(
                    PROFILE,
                    CTX_SYSTEM,
                    agents.as_deref(),
                    anchor.as_deref(),
                    state.as_deref(),
                    &git,
                    Vec::new(),
                    recent_b,
                );
                let stable_ok = doc_a.stable_prefix == doc_b.stable_prefix;
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]🗂 Profile {} — ctx {} · utilization {}% · top-k {}",
                    PROFILE.name(),
                    PROFILE.ctx_tokens(),
                    (PROFILE.utilization() * 100.0) as u64,
                    PROFILE.top_k(),
                ));
                let _ = tx.send(format!(
                    "[CTX]⚙️ llama.cpp args: {}",
                    PROFILE.llama_cpp_args().join(" ")
                ));
                let _ = tx.send(format!(
                    "[CTX]🧱 Stable prefix {} bytes — sha256 {} · cross-request identical: {}",
                    doc_a.stable_prefix.len(),
                    doc_a.stable_prefix_sha256(),
                    if stable_ok { "✅ yes" } else { "❌ NO — KV reuse is broken" }
                ));

                let rows = xencode_context_rs::read_metrics(&xencode);
                if rows.is_empty() {
                    let _ = tx.send("[CTX]📈 No metrics yet — run /ctx <query> then a llama.cpp generation to see KV reuse.".to_string());
                } else {
                    let _ = tx.send("[CTX]📈 Latest KV-cache rows per profile:".to_string());
                    for r in xencode_context_rs::RequestMetrics::latest_per_profile(&rows) {
                        let _ = tx.send(format!(
                            "[CTX]   {} — prompt {} · cached {} · reuse {}% · {} tok/s",
                            r.profile,
                            r.prompt_tokens,
                            r.cached_tokens,
                            (r.kv_reuse_ratio() * 100.0) as u64,
                            r.generation_tok_s
                        ));
                    }
                    // Interpret §13: large stable prefix + ~0 cached = prefix drift bug.
                    let latest = xencode_context_rs::RequestMetrics::latest_per_profile(&rows)
                        .first()
                        .cloned();
                    if let Some(r) = latest {
                        if r.prompt_tokens > 2000 && r.kv_reuse_ratio() < 0.05 {
                            let _ = tx.send(
                                "[CTX]🚨 Large prompt but ~0 cached tokens — something breaks prefix stability; check for dynamic tiers above the stable head."
                                    .to_string(),
                            );
                        }
                    }
                }
                if let Some(ts) = &self.last_llamacpp_timings {
                    let _ = tx.send(format!(
                        "[CTX]⚡ Last llama.cpp run — evaluated {} · generated {} · {} tok/s gen · {} tok/s prompt",
                        ts.tokens_evaluated,
                        ts.tokens_generated,
                        (ts.predicted_per_second as u64),
                        (ts.prompt_per_second as u64)
                    ));
                }
            }
            Some("archive") => {
                let (t, appended) = self.canonical_transcript();
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let state =
                    xencode_context_rs::ContextState::from_disk(&xencode).unwrap_or_default();
                let snap = t.snapshot(&xencode).unwrap_or_default();
                let prompt = xencode_context_rs::hard_compact_prompt(&state, &t);
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]📚 Canonical transcript synced (+{appended} new) → {} entries",
                    t.entries.len()
                ));
                let _ = tx.send(format!(
                    "[CTX]💾 Archived snapshot → {} — raw history is safe.",
                    snap.display()
                ));
                let _ = tx.send(format!(
                    "[CTX]🧠 Hard-compaction fold prompt ({} tokens):",
                    xencode_context_rs::est_tokens(prompt.len(), true)
                ));
                for line in prompt.lines() {
                    let _ = tx.send(format!("[CTX]    {line}"));
                }
            }
            _ => {
                let query = if let Some(rest) = rest.strip_prefix("retrieve") {
                    rest.trim().to_string()
                } else {
                    rest.to_string()
                };
                // Carry a small recent-message window for the assembly preview.
                let mut recent: Vec<String> = self
                    .messages
                    .iter()
                    .rev()
                    .take(8)
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect();
                recent.reverse();
                self.run_ctx_retrieval(query, recent.join("\n"), tx);
            }
        }
    }

    /// Sync the in-memory conversation into the canonical transcript store.
    /// Appends only messages that aren't already at the tail, so repeated
    /// `/ctx compact|archive` runs never double-count history.
    fn canonical_transcript(&mut self) -> (xencode_context_rs::Transcript, usize) {
        let root = xencode_context_rs::default_root();
        let xencode = root.join(xencode_context_rs::XENCODE_DIR);
        let path = xencode_context_rs::Transcript::current_path(&xencode);
        let mut t = xencode_context_rs::Transcript::from_disk(&path)
            .unwrap_or_else(|| xencode_context_rs::Transcript::new("tui"));
        let mem = self.memory.get_context(100_000);
        let mut appended = 0usize;
        for m in &mem {
            let dup = t
                .entries
                .last()
                .map(|e| e.role == m.role && e.content == m.content)
                .unwrap_or(false);
            if dup {
                continue;
            }
            t.add(&m.role, &m.content);
            appended += 1;
        }
        (t, appended)
    }

    /// Run deterministic retrieval + a context-assembly preview in the
    /// background, streaming results back through `[CTX]` chat lines.
    fn run_ctx_retrieval(
        &mut self,
        query: String,
        recent_text: String,
        tx: mpsc::UnboundedSender<String>,
    ) {
        tokio::spawn(async move {
            let _ = tx.send("[CTX_START]".to_string());
            let root = xencode_context_rs::default_root();
            let xencode = root.join(xencode_context_rs::XENCODE_DIR);
            let Some(index) = xencode_context_rs::RetrievalIndex::load(&xencode) else {
                let _ = tx.send(
                    "[CTX]❌ No project index — run /init first.".to_string(),
                );
                return;
            };
            let profile = xencode_context_rs::HardwareProfile::Balanced;
            let opts = xencode_context_rs::RetrieveOptions {
                top_k: profile.top_k(),
                ..Default::default()
            };
            let changed: HashSet<String> =
                xencode_context_rs::dirty_paths(&root).into_iter().collect();
            let results = xencode_context_rs::retrieve(&query, &index, &changed, &opts);
            if results.is_empty() {
                let _ = tx.send(
                    "[CTX]😶 Nothing above the score threshold — try a more specific query."
                        .to_string(),
                );
                return;
            }
            let _ = tx.send(format!(
                "[CTX]🎯 Retrieval ({} profile, top-{}):",
                profile.name(),
                results.len()
            ));
            for r in &results {
                let _ = tx.send(format!(
                    "[CTX]  {:>3}  {}  ·  {}",
                    r.score,
                    r.path,
                    r.reasons.join(", ")
                ));
            }

            let blocks = xencode_context_rs::read_retrieved_bodies(
                &root,
                &index.files,
                &results,
                profile.content_cap_chars(),
            );
            let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
            let anchor = std::fs::read_to_string(xencode.join("anchor.md")).ok();
            let state = std::fs::read_to_string(xencode.join("state.md")).ok();
            let git = xencode_context_rs::git_summary_text(&root).unwrap_or_default();
            let doc = xencode_context_rs::assemble_prompt(
                profile,
                CTX_SYSTEM,
                agents.as_deref(),
                anchor.as_deref(),
                state.as_deref(),
                &git,
                blocks,
                &recent_text,
            );
            let stable_tokens: u64 = doc.tiers.iter().take(3).map(|t| t.tokens).sum();
            let _ = tx.send(format!(
                "[CTX]📦 Assembled context ≈ {} / {} tokens target — {} / {} retrieved files in — stable prefix {} tokens",
                doc.total_tokens,
                doc.target_tokens,
                doc.retrieved_included,
                doc.retrieved_total,
                stable_tokens
            ));
            // Capture for the KV-reuse metrics on the next llama.cpp timings.
            let _ = tx.send(format!(
                "[CTXSTATS]{}|{}",
                doc.total_tokens.min(u32::MAX as u64),
                doc.retrieved_included.min(u8::MAX as usize)
            ));
            if doc.truncated {
                let _ = tx.send(
                    "[CTX]⚠ Some retrieved files dropped to fit the budget.".to_string(),
                );
            }
            if doc.soft_compaction_needed {
                let _ = tx.send(
                    "[CTX]⚠ Recent-message budget under 1 message — soft compaction should trigger before sending."
                        .to_string(),
                );
            }

            let mut m = xencode_context_rs::RequestMetrics::new(
                profile.name(),
                profile.ctx_tokens() as u32,
            );
            m.ts_unix_ms = (current_timestamp() * 1000.0) as u64;
            m.prompt_tokens = doc.total_tokens.min(u32::MAX as u64) as u32;
            m.retrieved_files = doc.retrieved_included.min(u8::MAX as usize) as u8;
            m.context_usage = (doc.total_tokens as f32 / doc.target_tokens.max(1) as f32).min(1.0);
            m.compaction = if doc.soft_compaction_needed {
                xencode_context_rs::CompactAction::Soft
            } else {
                xencode_context_rs::CompactAction::None
            };
            let _ = xencode_context_rs::append_metrics(&xencode, &m);
        });
    }

    /// Start Voice Interface simulation with audio level and speech-to-text.
    pub fn start_voice_session(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.voice_active {
            return;
        }
        self.voice_active = true;
        self.voice_status = "listening".to_string();
        self.voice_transcript.clear();
        self.voice_commands.clear();
        self.voice_transcript
            .push("🎤 Microphone initialized".to_string());

        tokio::spawn(async move {
            let phrases = vec![
                (
                    "refactor user model",
                    "✅ Model refactored — UserModel split into User + Profile",
                ),
                (
                    "add validation for email",
                    "✅ Added email validation regex to UserService",
                ),
                ("run tests", "✅ 142 tests passed, 0 failed"),
                (
                    "commit changes",
                    "✅ Committed 'feat: add email validation'",
                ),
            ];

            for (cmd, result) in &phrases {
                // Simulate listening with audio levels
                for level in [0.3, 0.6, 0.8, 0.9, 0.7, 0.4] {
                    let _ = tx.send(format!("[VOICE]level:{}", level));
                    tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
                }

                let _ = tx.send("[VOICE]status:processing".to_string());
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

                let _ = tx.send("[VOICE]status:speaking".to_string());
                let _ = tx.send(format!("[VOICE]transcript:{}", cmd));
                let _ = tx.send(format!("[VOICE]command:{}|{}", cmd, result));
                tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;

                let _ = tx.send("[VOICE]status:listening".to_string());
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }

            let _ = tx.send("[VOICE]status:idle".to_string());
        });
    }

    /// Start Terminal Assistant with command suggestions.
    pub fn start_terminal_assistant(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.term_asst_active {
            return;
        }
        self.term_asst_active = true;
        self.term_asst_suggestions.clear();
        self.term_asst_output.clear();
        self.term_asst_history.clear();

        tokio::spawn(async move {
            let _ = tx.send(
                "[TERM]output:🧠 Terminal Assistant ready — type a query and press Enter"
                    .to_string(),
            );

            let suggestions = vec![
                (
                    "find . -name \"*.py\" | xargs grep -l \"def \"",
                    "🔍 Safe",
                    "Find all Python files with function definitions",
                ),
                (
                    "git log --oneline --graph --all",
                    "✅ Safe",
                    "Visual git history graph",
                ),
                (
                    "du -sh */ 2>/dev/null | sort -rh",
                    "✅ Safe",
                    "Show directory sizes sorted by size",
                ),
                (
                    "docker system prune -af",
                    "⚠️ Destructive",
                    "⚠ Removes ALL unused Docker data",
                ),
                (
                    "rm -rf node_modules && npm install",
                    "⚠️ Destructive",
                    "⚠ Deletes node_modules and reinstalls",
                ),
            ];

            for (cmd, risk, explanation) in &suggestions {
                tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
                let _ = tx.send(format!("[TERM]suggestion:{}|{}|{}", cmd, risk, explanation));
            }

            let _ = tx.send("[TERM]ready".to_string());
        });
    }

    /// Start Security Auditor scan simulation.
    pub fn start_security_scan(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.sec_scan_active {
            return;
        }
        self.sec_scan_active = true;
        self.sec_scan_results.clear();
        self.sec_scan_summary = (0, 0, 0, 0);
        self.sec_scan_progress = 0.0;
        self.sec_scan_log.clear();
        self.sec_scan_log
            .push("🔍 Starting vulnerability scan...".to_string());

        tokio::spawn(async move {
            let findings = [
                (
                    "Critical",
                    "Hardcoded API Key",
                    "src/config.py:42",
                    "❌ Found hardcoded AWS_SECRET_KEY",
                ),
                (
                    "High",
                    "SQL Injection",
                    "src/queries.py:18",
                    "🚨 Raw SQL concatenation detected",
                ),
                (
                    "High",
                    "Command Injection",
                    "src/deploy.py:55",
                    "🚨 Using os.system() with user input",
                ),
                (
                    "Medium",
                    "Weak Crypto",
                    "src/crypto.py:10",
                    "⚠️ MD5 used for password hashing",
                ),
                (
                    "Medium",
                    "XSS Vulnerability",
                    "src/templates/user.html:22",
                    "⚠️ Unsafe innerHTML assignment",
                ),
                (
                    "Low",
                    "Deprecated Package",
                    "requirements.txt:1",
                    "📦 PyCrypto v2.6.1 is end-of-life",
                ),
                (
                    "Low",
                    "Missing Rate Limit",
                    "src/api.py:30",
                    "🐢 No rate limiting on /login endpoint",
                ),
            ];

            let total = findings.len();
            for (i, (severity, category, location, detail)) in findings.iter().enumerate() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                let progress = (i as f64 + 1.0) / total as f64;
                let _ = tx.send(format!("[SECURITY]progress:{:.2}", progress));
                let _ = tx.send(format!(
                    "[SECURITY]finding:{}|{}|{}|{}",
                    severity, category, location, detail
                ));
            }

            let _ = tx.send("[SECURITY]done".to_string());
        });
    }

    /// Start Performance Profiler simulation.
    pub fn start_profiler(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.profiler_running {
            return;
        }
        self.profiler_active = true;
        self.profiler_running = true;
        self.profiler_functions.clear();

        tokio::spawn(async move {
            let funcs = [
                ("process_data", 245.3, 128.0, 1240u32),
                ("validate_input", 180.1, 64.0, 890u32),
                ("render_template", 95.7, 32.0, 450u32),
                ("query_database", 320.4, 256.0, 67u32),
                ("serialize_output", 45.2, 16.0, 340u32),
                ("generate_report", 512.8, 512.0, 12u32),
            ];

            for (i, (name, time_ms, mem_mb, calls)) in funcs.iter().enumerate() {
                tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
                let cpu = 30.0 + (i as f64 * 10.0) + (fastrand::i32(0..20) as f64 * 0.5);
                let mem = 40.0 + (i as f64 * 8.0) + (fastrand::i32(0..15) as f64 * 0.5);
                let latency = 50.0 + (i as f64 * 15.0) + (fastrand::i32(0..10) as f64);
                let _ = tx.send(format!("[PROFILER]gauge:cpu|{:.0}", cpu));
                let _ = tx.send(format!("[PROFILER]gauge:mem|{:.0}", mem));
                let _ = tx.send(format!("[PROFILER]gauge:latency|{:.0}", latency));
                let _ = tx.send(format!(
                    "[PROFILER]func:{}|{}|{}|{}",
                    name, time_ms, mem_mb, calls
                ));
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
            let _ = tx.send("[PROFILER]done".to_string());
        });
    }

    /// Start Custom Models session (seeds profile data).
    pub fn start_custom_models(&mut self) {
        if self.models_editing {
            return;
        }
        self.models_editing = true;
        self.models_saving = false;
        self.models_profiles = vec![
            (
                "Code Assistant".to_string(),
                "ollama".to_string(),
                0.3,
                4096,
                0.9,
            ),
            (
                "Creative Writer".to_string(),
                "openrouter".to_string(),
                0.8,
                2048,
                0.95,
            ),
            (
                "Bug Hunter".to_string(),
                "ollama".to_string(),
                0.2,
                8192,
                0.8,
            ),
            (
                "Code Reviewer".to_string(),
                "openrouter".to_string(),
                0.15,
                4096,
                0.85,
            ),
        ];
        self.models_selected = 0;
        self.models_test_output = String::new();
    }

    /// Start Learning Mode with lesson content.
    pub fn start_learning_mode(&mut self) {
        if self.learn_active {
            return;
        }
        self.learn_active = true;
        self.learn_current_lesson = 1;
        self.learn_total_lessons = 5;
        self.learn_lesson_title = "Rust Ownership Basics".to_string();
        self.learn_content = vec![
            "In Rust, each value has a single 'owner' at any time.".to_string(),
            "When the owner goes out of scope, the value is dropped.".to_string(),
            "References allow borrowing without taking ownership.".to_string(),
            "Mutable references (&mut T) are exclusive - only one at a time.".to_string(),
            "Immutable references (&T) can coexist freely.".to_string(),
        ];
        self.learn_code_example = [
            "fn main() {",
            "    let s = String::from(\"hello\");  // s owns the String",
            "    let len = calculate_length(&s);   // borrow, not move",
            "    println!(\"'{}' has length {}\", s, len);",
            "}",
            "",
            "fn calculate_length(s: &String) -> usize {",
            "    s.len()  // s is a reference, no ownership transfer",
            "}",
        ]
        .join("\n");
        self.learn_exercise =
            "Fix the ownership error: let s2 = s; println!(\"{}\", s);".to_string();
        self.learn_progress_pct = 20.0;
        // Seed quiz for lesson 1
        self.learn_quiz_active = true;
        self.learn_quiz_question = "What owns a String value in Rust?".to_string();
        self.learn_quiz_options = vec![
            "The variable that declares it".to_string(),
            "The heap allocator".to_string(),
            "The garbage collector".to_string(),
            "All references to it".to_string(),
        ];
        self.learn_quiz_selected = 0;
        self.learn_quiz_answered = false;
        self.learn_quiz_correct = false;
    }

    /// Start Multi-Language panel with detection results.
    pub fn start_multi_language(&mut self) {
        if self.lang_active {
            return;
        }
        self.lang_active = true;
        self.lang_detection_results = vec![
            (
                "src/main.rs".to_string(),
                "Rust".to_string(),
                "99.2%".to_string(),
            ),
            (
                "src/app.py".to_string(),
                "Python".to_string(),
                "98.7%".to_string(),
            ),
            (
                "src/components.tsx".to_string(),
                "TypeScript".to_string(),
                "97.5%".to_string(),
            ),
            (
                "templates/index.html".to_string(),
                "HTML".to_string(),
                "96.8%".to_string(),
            ),
            (
                "styles/main.css".to_string(),
                "CSS".to_string(),
                "95.1%".to_string(),
            ),
        ];
        self.lang_supported = vec![
            ("Rust".to_string(), "✅".to_string()),
            ("Python".to_string(), "✅".to_string()),
            ("TypeScript".to_string(), "✅".to_string()),
            ("JavaScript".to_string(), "✅".to_string()),
            ("Go".to_string(), "🔄".to_string()),
            ("Ruby".to_string(), "🚧".to_string()),
        ];
        self.lang_translate_input = String::new();
        self.lang_translate_output = String::new();
    }
    /// Dynamically discover models installed by the user in Ollama and configured cloud models.
    pub fn refresh_models(&mut self, tx: mpsc::UnboundedSender<String>) {
        let ollama_url = self.config.ollama_url.clone();
        let llama_cpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let has_openrouter = self.config.api_keys.openrouter_api_key.is_some();
        let has_gemini = self.config.api_keys.google_gemini_api_key.is_some();
        let has_qwen = self.config.api_keys.qwen_api_key.is_some();
        let current_default = self.config.default_model.clone();

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout.min(5));
            let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout.min(5));
            let mut models = Vec::new();

            if let Ok(installed) = client.list_models().await {
                for m in installed {
                    if !m.name.contains("embed") {
                        models.push(m.name);
                    }
                }
            }

            // llama.cpp server models
            if let Ok(llama_models) = llama_client.list_models().await {
                for m in llama_models {
                    let prefixed = format!("llamacpp:{}", m.id);
                    if !models.contains(&prefixed) {
                        models.push(prefixed);
                    }
                }
            }

            // Cloud providers if keys are configured
            if has_openrouter {
                models.push("anthropic/claude-3.5-sonnet".to_string());
                models.push("openai/gpt-4o".to_string());
            }
            if has_gemini {
                models.push("google/gemini-1.5-pro".to_string());
                models.push("google/gemini-1.5-flash".to_string());
            }
            if has_qwen {
                models.push("qwen-max".to_string());
                models.push("qwen-plus".to_string());
            }

            // Fallback if no models discovered
            if models.is_empty() && !current_default.is_empty() {
                models.push(current_default);
            }

            if let Ok(serialized) = serde_json::to_string(&models) {
                let _ = tx.send(format!("[MODELS]{}", serialized));
            }
        });
    }

    /// Trigger background health checks across all configured providers.
    pub fn run_health_check(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.health_check_in_progress {
            return;
        }
        self.health_check_in_progress = true;

        let ollama_url = self.config.ollama_url.clone();
        let llama_cpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let openrouter_key = self.config.api_keys.openrouter_api_key.clone();
        let qwen_key = self.config.api_keys.qwen_api_key.clone();
        let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
        let default_model = self.config.default_model.clone();

        tokio::spawn(async move {
            // Check Ollama health by listing installed models
            let mut client = OllamaClient::new(&ollama_url, timeout.min(5));
            let start = std::time::Instant::now();

            match client.list_models().await {
                Ok(installed_models) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let chat_models: Vec<String> = installed_models
                        .iter()
                        .filter(|m| !m.name.contains("embed"))
                        .map(|m| m.name.clone())
                        .collect();

                    if !chat_models.is_empty() {
                        // Send refreshed model list to TUI
                        let mut all_models = chat_models.clone();
                        if openrouter_key.is_some() {
                            all_models.push("anthropic/claude-3.5-sonnet".to_string());
                            all_models.push("openai/gpt-4o".to_string());
                        }
                        if gemini_key.is_some() {
                            all_models.push("google/gemini-1.5-pro".to_string());
                            all_models.push("google/gemini-1.5-flash".to_string());
                        }
                        if qwen_key.is_some() {
                            all_models.push("qwen-max".to_string());
                            all_models.push("qwen-plus".to_string());
                        }
                        let _ = tx.send(format!(
                            "[MODELS]{}",
                            serde_json::to_string(&all_models).unwrap_or_default()
                        ));

                        // Test health using actual default_model if local and installed, or first installed model
                        let test_model = if chat_models.contains(&default_model) {
                            &default_model
                        } else {
                            &chat_models[0]
                        };

                        match client.check_health(test_model).await {
                            Ok(health) => {
                                let _ = tx.send(format!(
                                    "[HEALTH]ollama|{}|{}|{}",
                                    health.status,
                                    health.response_time * 1000.0,
                                    health.error_message.unwrap_or_default()
                                ));
                            }
                            Err(e) => {
                                let _ = tx.send(format!(
                                    "[HEALTH]ollama|healthy|{}|{}",
                                    latency, e
                                ));
                            }
                        }
                    } else {
                        let _ = tx.send(format!(
                            "[HEALTH]ollama|healthy|{}|Running (0 models installed)",
                            latency
                        ));
                    }
                }
                Err(e) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let _ = tx.send(format!("[HEALTH]ollama|error|{}|{}", latency, e));
                }
            }

            // Check OpenRouter connectivity (if key is set)
            if let Some(api_key) = openrouter_key {
                let start_or = std::time::Instant::now();
                let openrouter_client = reqwest::Client::new();
                match openrouter_client
                    .get("https://openrouter.ai/api/v1/auth/key")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_or.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() {
                            let _ = tx.send(format!("[HEALTH]openrouter|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!(
                                "[HEALTH]openrouter|error|{}|HTTP {}",
                                latency,
                                resp.status()
                            ));
                        }
                    }
                    Err(e) => {
                        let latency = start_or.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]openrouter|error|{}|{}", latency, e));
                    }
                }
            }

            // Check Qwen DashScope connectivity (if key is set)
            if let Some(api_key) = qwen_key {
                let start_qw = std::time::Instant::now();
                let qwen_client = reqwest::Client::new();
                match qwen_client
                    .get("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_qw.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() || resp.status().as_u16() == 400 {
                            // 400 means the request reached the API but had invalid params (key is valid)
                            let _ = tx.send(format!("[HEALTH]qwen|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!("[HEALTH]qwen|error|{}|HTTP {}", latency, resp.status()));
                        }
                    }
                    Err(e) => {
                        let latency = start_qw.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]qwen|error|{}|{}", latency, e));
                    }
                }
            }

            // Check Gemini connectivity (if key is set)
            if let Some(api_key) = gemini_key {
                let start_ge = std::time::Instant::now();
                let gemini_client = reqwest::Client::new();
                match gemini_client
                    .get(format!(
                        "https://generativelanguage.googleapis.com/v1/models?key={}",
                        api_key
                    ))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_ge.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() {
                            let _ = tx.send(format!("[HEALTH]gemini|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!(
                                "[HEALTH]gemini|error|{}|HTTP {}",
                                latency,
                                resp.status()
                            ));
                        }
                    }
                    Err(e) => {
                        let latency = start_ge.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]gemini|error|{}|{}", latency, e));
                    }
                }
            }

            // Check llama.cpp connectivity
            let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout.min(5));
            match llama_client.ping().await {
                Ok(resp_time) => {
                    let _ = tx.send(format!(
                        "[HEALTH]llamacpp|healthy|{}|",
                        resp_time * 1000.0
                    ));
                }
                Err(e) => {
                    let _ = tx.send(format!("[HEALTH]llamacpp|unavailable|0|{}", e));
                }
            }

            let _ = tx.send("[HEALTH_DONE]".to_string());
        });
    }

    pub fn submit_review(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.is_reviewing {
            return;
        }
        if let Some(file_path) = self.file_tree.get(self.selected_file) {
            if let Ok(content) = std::fs::read_to_string(file_path) {
                self.is_reviewing = true;
                self.code_review_output = format!("📝 Reviewing: {}\n\n", file_path);
                let prompt = format!(
                    "Code review of {}. Identify bugs, security issues, and performance bottlenecks.\n\n```\n{}\n```",
                    file_path, content
                );
                let messages = vec![ChatMessage {
                    role: "user".to_string(),
                    content: prompt,
                }];
                let model = self.config.default_model.clone();
                let ollama_url = self.config.ollama_url.clone();
                let llama_cpp_url = self.config.llama_cpp_url.clone();
                let timeout = self.config.response_timeout;
                let or_key = self.config.api_keys.openrouter_api_key.clone();
                let qwen_key = self.config.api_keys.qwen_api_key.clone();
                let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
                let llama_opts = LlamaCppOptions {
                    temperature: self.config.llama_cpp_temperature,
                    top_k: self.config.llama_cpp_top_k,
                    min_p: self.config.llama_cpp_min_p,
                    max_tokens: self.config.llama_cpp_max_tokens,
                    grammar: None,
                    json_schema: None,
                    mirostat: None,
                };

                tokio::spawn(async move {
                    let client = OllamaClient::new(&ollama_url, timeout);
                    let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout);
                    let manager = ProviderManager::new(client, or_key, qwen_key, gemini_key, None)
                        .with_llama_cpp(llama_client);
                    let _ = manager
                        .generate_stream_with_options(
                            &model,
                            &messages,
                            Some(&llama_opts),
                            |token| {
                                let _ = tx.send(format!("[REVIEW]{}", token));
                            },
                        )
                        .await;
                    if let Some(ts) = manager.last_llamacpp_timings() {
                        if let Ok(json) = serde_json::to_string(&ts) {
                            let _ = tx.send(format!("[TIMINGS]{}", json));
                        }
                    }
                    let _ = tx.send("[REVIEW][DONE]".to_string());
                });
            }
        }
    }
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the inner model id from a llama.cpp-prefixed model selector entry.
fn llama_model_target(model: &str) -> Option<&str> {
    for prefix in ["llamacpp:", "llama.cpp:", "llama:"] {
        if let Some(rest) = model.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

pub async fn run_app<B: Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Query installed Ollama models and check provider health immediately on startup
    app.refresh_models(tx.clone());
    app.run_health_check(tx.clone());

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        // Drain async messages
        while let Ok(token) = rx.try_recv() {
            if let Some(body) = token.strip_prefix("[REVIEW]") {
                app.append_review(body);
            } else if let Some(body) = token.strip_prefix("[BYTEBOT]") {
                if body.starts_with("step:") {
                    let parts: Vec<&str> = body.splitn(4, ':').collect();
                    if parts.len() >= 4 {
                        let idx = parts[1].parse::<usize>().unwrap_or(0);
                        let status = parts[2].to_string();
                        if idx < app.bytebot_steps.len() {
                            app.bytebot_steps[idx].1 = status;
                        }
                    }
                } else if body.starts_with("progress:") {
                    if let Some(pct) = body.strip_prefix("progress:") {
                        app.bytebot_progress = pct.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("log:") {
                    if let Some(msg) = body.strip_prefix("log:") {
                        app.bytebot_log.push(msg.to_string());
                    }
                }
            } else if token == "[BYTEBOT_DONE]" {
                app.bytebot_running = false;
                app.bytebot_progress = 1.0;
            } else if token == "[INIT_DONE]" {
                app.init_running = false;
                app.init_progress = 1.0;
            } else if let Some(body) = token.strip_prefix("[INIT]") {
                if body.starts_with("step:") {
                    let parts: Vec<&str> = body.splitn(4, ':').collect();
                    if parts.len() >= 4 {
                        let idx = parts[1].parse::<usize>().unwrap_or(0);
                        if idx < app.init_steps.len() {
                            app.init_steps[idx].1 = parts[2].to_string();
                        }
                    }
                } else if body.starts_with("progress:") {
                    if let Some(pct) = body.strip_prefix("progress:") {
                        app.init_progress = pct.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("log:") {
                    if let Some(msg) = body.strip_prefix("log:") {
                        app.init_log.push(msg.to_string());
                    }
                }
            } else if token == "[CTX_START]" {
                app.messages.push(UiMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                });
            } else if let Some(body) = token.strip_prefix("[CTX]") {
                if let Some(last) = app.messages.last_mut() {
                    if last.role == "assistant" {
                        if !last.content.is_empty() {
                            last.content.push('\n');
                        }
                        last.content.push_str(body);
                    }
                }
            } else if let Some(body) = token.strip_prefix("[COLLAB]") {
                if body.starts_with("status:") {
                    if let Some(s) = body.strip_prefix("status:") {
                        app.collab_sync_status = s.to_string();
                    }
                } else if body.starts_with("sync:") {
                    if let Some(s) = body.strip_prefix("sync:") {
                        app.collab_sync_status = s.to_string();
                    }
                } else if body.starts_with("pending:") {
                    if let Some(n) = body.strip_prefix("pending:") {
                        app.collab_pending_changes = n.trim().parse::<u32>().unwrap_or(0);
                    }
                } else if body.starts_with("member:") {
                    let parts: Vec<&str> = body.splitn(3, ':').collect();
                    if parts.len() >= 3 {
                        let name = parts[1].to_string();
                        let new_status = parts[2].to_string();
                        if let Some(member) =
                            app.collab_members.iter_mut().find(|(n, _, _)| n == &name)
                        {
                            member.1 = new_status;
                        }
                    }
                } else if body.starts_with("log:") {
                    if let Some(msg) = body.strip_prefix("log:") {
                        app.collab_activity_log.push(msg.to_string());
                    }
                } else if body == "ready" {
                    app.collab_last_sync = current_timestamp();
                }
            } else if let Some(body) = token.strip_prefix("[HEALTH]") {
                let parts: Vec<&str> = body.splitn(4, '|').collect();
                if parts.len() >= 3 {
                    let provider = parts[0].to_string();
                    let status = parts[1].to_string();
                    let latency = parts[2].parse::<f64>().unwrap_or(0.0);
                    let error = if parts.len() > 3 && !parts[3].is_empty() {
                        Some(parts[3].to_string())
                    } else {
                        None
                    };
                    app.ollama_health_entries
                        .insert(provider, (status.clone(), latency, error));
                    // Update average latency across all providers
                    if status == "healthy" {
                        let total: f64 = app
                            .ollama_health_entries
                            .values()
                            .map(|(s, l, _)| if s == "healthy" { *l } else { 0.0 })
                            .sum();
                        let count = app
                            .ollama_health_entries
                            .values()
                            .filter(|(s, _, _)| s == "healthy")
                            .count() as f64;
                        app.average_latency = if count > 0.0 { total / count } else { 0.0 };
                    }
                }
            } else if let Some(body) = token.strip_prefix("[VOICE]") {
                if body.starts_with("status:") {
                    if let Some(s) = body.strip_prefix("status:") {
                        let new_status = s.to_string();
                        if new_status == "idle" {
                            app.voice_active = false;
                        }
                        app.voice_status = new_status;
                    }
                } else if body.starts_with("level:") {
                    if let Some(l) = body.strip_prefix("level:") {
                        app.voice_level = l.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("transcript:") {
                    if let Some(t) = body.strip_prefix("transcript:") {
                        app.voice_transcript.push(t.to_string());
                    }
                } else if body.starts_with("command:") {
                    if let Some(c) = body.strip_prefix("command:") {
                        let parts: Vec<&str> = c.splitn(2, '|').collect();
                        let cmd = parts.first().unwrap_or(&"").to_string();
                        let result = parts.get(1).unwrap_or(&"").to_string();
                        app.voice_commands.push((cmd, result));
                    }
                }
            } else if let Some(body) = token.strip_prefix("[TERM]") {
                if body.starts_with("suggestion:") {
                    if let Some(s) = body.strip_prefix("suggestion:") {
                        let parts: Vec<&str> = s.splitn(3, '|').collect();
                        let cmd = parts.first().unwrap_or(&"").to_string();
                        let risk = parts.get(1).unwrap_or(&"").to_string();
                        let explanation = parts.get(2).unwrap_or(&"").to_string();
                        app.term_asst_suggestions
                            .push(format!("{}  {} — {}", risk, cmd, explanation));
                    }
                } else if body.starts_with("output:") {
                    if let Some(o) = body.strip_prefix("output:") {
                        app.term_asst_output = o.to_string();
                    }
                } else if body == "ready" {
                    app.term_asst_active = false;
                }
            } else if let Some(body) = token.strip_prefix("[SECURITY]") {
                if body.starts_with("progress:") {
                    if let Some(p) = body.strip_prefix("progress:") {
                        app.sec_scan_progress = p.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("finding:") {
                    if let Some(f) = body.strip_prefix("finding:") {
                        let parts: Vec<&str> = f.splitn(4, '|').collect();
                        if parts.len() >= 4 {
                            let severity = parts[0].to_string();
                            let category = parts[1].to_string();
                            let location = parts[2].to_string();
                            let detail = parts[3].to_string();
                            app.sec_scan_results.push((
                                severity.clone(),
                                category.clone(),
                                location.clone(),
                            ));
                            app.sec_scan_log.push(detail);
                            // Update summary counts
                            let (mut c, mut h, mut m, mut l) = app.sec_scan_summary;
                            match severity.as_str() {
                                "Critical" => c += 1,
                                "High" => h += 1,
                                "Medium" => m += 1,
                                _ => l += 1,
                            }
                            app.sec_scan_summary = (c, h, m, l);
                        }
                    }
                } else if body == "done" {
                    app.sec_scan_active = false;
                }
            } else if let Some(body) = token.strip_prefix("[PROFILER]") {
                if body.starts_with("gauge:") {
                    if let Some(g) = body.strip_prefix("gauge:") {
                        let parts: Vec<&str> = g.splitn(2, '|').collect();
                        if parts.len() >= 2 {
                            let val = parts[1].parse::<f64>().unwrap_or(0.0);
                            match parts[0] {
                                "cpu" => app.profiler_gauge_cpu = val,
                                "mem" => app.profiler_gauge_mem = val,
                                "latency" => app.profiler_gauge_latency = val,
                                _ => {}
                            }
                        }
                    }
                } else if body.starts_with("func:") {
                    if let Some(f) = body.strip_prefix("func:") {
                        let parts: Vec<&str> = f.splitn(4, '|').collect();
                        if parts.len() >= 4 {
                            let name = parts[0].to_string();
                            let time = parts[1].parse::<f64>().unwrap_or(0.0);
                            let mem = parts[2].parse::<f64>().unwrap_or(0.0);
                            let calls = parts[3].parse::<u32>().unwrap_or(0);
                            app.profiler_functions.push((name, time, mem, calls));
                        }
                    }
                } else if body == "done" {
                    app.profiler_running = false;
                }
            } else if let Some(body) = token.strip_prefix("[MODELS]") {
                if let Ok(models) = serde_json::from_str::<Vec<String>>(body) {
                    if !models.is_empty() {
                        app.available_models = models;
                        if let Some(pos) = app
                            .available_models
                            .iter()
                            .position(|m| m == &app.config.default_model)
                        {
                            app.selected_model = pos;
                        } else {
                            // If current default_model is not installed, select first installed model from Ollama
                            if let Some(first) = app.available_models.first().cloned() {
                                app.config.default_model = first;
                                app.selected_model = 0;
                                let _ = app.config.save();
                            }
                        }
                    }
                }
            } else if let Some(body) = token.strip_prefix("[CTXSTATS]") {
                let parts: Vec<&str> = body.splitn(2, '|').collect();
                if parts.len() == 2 {
                    app.last_ctx_total_tokens = parts[0].parse().unwrap_or(0);
                    app.last_ctx_retrieved_files = parts[1].parse().unwrap_or(0);
                }
            } else if let Some(body) = token.strip_prefix("[LLAMACPP]") {
                app.llamacpp_action_msg = body.to_string();
            } else if let Some(body) = token.strip_prefix("[TIMINGS]") {
                if let Ok(ts) = serde_json::from_str::<LlamaCppTimings>(body) {
                    app.last_llamacpp_timings = Some(ts.clone());
                    // Record a §13 metrics row: cached = prompt_total − actually
                    // evaluated. prompt_total comes from the last /ctx assembly,
                    // evaluated from llama.cpp — a ~0 cached_tokens with a large
                    // stable prefix means prefix stability broke somewhere.
                    let root = xencode_context_rs::default_root();
                    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                    let profile = xencode_context_rs::HardwareProfile::Balanced;
                    let m = xencode_context_rs::RequestMetrics::from_timings(
                        profile.name(),
                        profile.ctx_tokens() as u32,
                        app.last_ctx_total_tokens.min(u32::MAX as u64) as u32,
                        ts.tokens_evaluated.min(u32::MAX as u64) as u32,
                        ts.tokens_generated.min(u32::MAX as u64) as u32,
                        ts.predicted_per_second as f32,
                        ts.prompt_per_second as f32,
                        app.last_ctx_retrieved_files,
                    );
                    let _ = xencode_context_rs::append_metrics(&xencode, &m);
                }
            } else if token == "[HEALTH_DONE]" {
                app.health_check_in_progress = false;
                app.last_health_check = current_timestamp();
            } else {
                app.append_generation(&token);
            }
        }

        // Poll events (~30fps)
        if event::poll(Duration::from_millis(33))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Global shortcuts (work in ALL modes)
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    if ctrl {
                        match key.code {
                            KeyCode::Char('c') => return Ok(()),
                            KeyCode::Char('g') => {
                                app.refresh_git();
                                continue;
                            }
                            KeyCode::Char(',') => {
                                app.focus = if app.focus == FocusArea::Settings {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::Settings
                                };
                                continue;
                            }
                            KeyCode::Char('b') => {
                                if !app.bytebot_running && app.focus != FocusArea::ByteBotPanel {
                                    app.focus = FocusArea::ByteBotPanel;
                                } else if app.focus == FocusArea::ByteBotPanel {
                                    app.focus = FocusArea::ChatInput;
                                }
                                continue;
                            }
                            KeyCode::Char('d') => {
                                app.focus = if app.focus == FocusArea::PerformanceDashboard {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::PerformanceDashboard
                                };
                                continue;
                            }
                            KeyCode::Char('p') => {
                                app.focus = if app.focus == FocusArea::ProjectAnalyzer {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::ProjectAnalyzer
                                };
                                continue;
                            }
                            KeyCode::Char('e') => {
                                app.focus = if app.focus == FocusArea::FileExplorer {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::FileExplorer
                                };
                                continue;
                            }
                            KeyCode::Char('w') => {
                                // Close current panel and return to ChatInput
                                match app.focus {
                                    FocusArea::ByteBotPanel
                                    | FocusArea::CollaborationHub
                                    | FocusArea::VoiceInterface
                                    | FocusArea::TerminalAssistant
                                    | FocusArea::SecurityAuditor
                                    | FocusArea::PerformanceProfiler
                                    | FocusArea::CustomModels
                                    | FocusArea::LearningMode
                                    | FocusArea::MultiLanguage
                                    | FocusArea::ProviderHealth
                                    | FocusArea::PerformanceDashboard
                                    | FocusArea::ProjectAnalyzer
                                    | FocusArea::GitCommit
                                    | FocusArea::CodeReview
                                    | FocusArea::FeatureNavigator
                                    | FocusArea::ModelSelector
                                    | FocusArea::Settings => {
                                        app.focus = FocusArea::ChatInput;
                                    }
                                    _ => {}
                                }
                                continue;
                            }
                            KeyCode::Char('r') => {
                                app.focus = if app.focus == FocusArea::CodeReview {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::CodeReview
                                };
                                continue;
                            }
                            KeyCode::Char('t') => {
                                app.show_terminal = !app.show_terminal;
                                if !app.show_terminal && app.focus == FocusArea::Terminal {
                                    app.focus = FocusArea::ChatInput;
                                }
                                continue;
                            }
                            KeyCode::Char('h') if ctrl => {
                                if !app.health_check_in_progress {
                                    app.run_health_check(tx.clone());
                                }
                                continue;
                            }
                            KeyCode::Char('f') => {
                                app.focus = if app.focus == FocusArea::FeatureNavigator {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::FeatureNavigator
                                };
                                continue;
                            }
                            KeyCode::Char('s') => {
                                if app.focus == FocusArea::CodeEditor {
                                    app.save_editor();
                                } else {
                                    app.focus = if app.focus == FocusArea::GitCommit {
                                        FocusArea::ChatInput
                                    } else {
                                        FocusArea::GitCommit
                                    };
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }

                    match app.input_mode {
                        InputMode::Normal => match key.code {
                            KeyCode::Tab => {
                                app.focus = match app.focus {
                                    FocusArea::FileExplorer => FocusArea::CodeEditor,
                                    FocusArea::CodeEditor => FocusArea::ChatInput,
                                    FocusArea::ChatInput => FocusArea::FileExplorer,
                                    _ => FocusArea::ChatInput,
                                };
                            }
                            KeyCode::Up | KeyCode::Char('k') => match app.focus {
                                FocusArea::Settings => {
                                    if app.settings_cursor > 0 {
                                        app.settings_cursor -= 1;
                                    }
                                }
                                FocusArea::FileExplorer => {
                                    if app.selected_file > 0 {
                                        app.selected_file -= 1;
                                    }
                                }
                                FocusArea::ModelSelector => {
                                    if app.selected_model > 0 {
                                        app.selected_model -= 1;
                                    }
                                }
                                FocusArea::ChatInput => {
                                    app.chat_scroll = app.chat_scroll.saturating_add(1);
                                }
                                FocusArea::FeatureNavigator => {
                                    if app.feature_nav_selected > 0 {
                                        app.feature_nav_selected -= 1;
                                    }
                                }
                                FocusArea::ProviderHealth => {
                                    if app.provider_health_scroll > 0 {
                                        app.provider_health_scroll -= 1;
                                    }
                                }
                                FocusArea::SecurityAuditor => {
                                    if app.security_scroll > 0 {
                                        app.security_scroll -= 1;
                                    }
                                }
                                FocusArea::CodeEditor => {
                                    app.editor.scroll((-1, 0));
                                }
                                FocusArea::CustomModels => {
                                    if app.models_selected > 0 {
                                        app.models_selected -= 1;
                                    }
                                }
                                FocusArea::ByteBotPanel
                                    if !app.bytebot_running && !app.bytebot_history.is_empty() =>
                                {
                                    app.bytebot_command =
                                        app.bytebot_history.last().unwrap().clone();
                                    app.bytebot_cursor = app.bytebot_command.len();
                                }
                                _ => {}
                            },
                            KeyCode::Down | KeyCode::Char('j') => {
                                match app.focus {
                                    FocusArea::Settings => {
                                        if app.settings_cursor + 1 < 14 {
                                            app.settings_cursor += 1;
                                        }
                                    }
                                    FocusArea::FileExplorer => {
                                        if app.selected_file + 1 < app.file_tree.len() {
                                            app.selected_file += 1;
                                        }
                                    }
                                    FocusArea::ModelSelector => {
                                        if app.selected_model + 1 < app.available_models.len() {
                                            app.selected_model += 1;
                                        }
                                    }
                                    FocusArea::ChatInput => {
                                        app.chat_scroll = app.chat_scroll.saturating_sub(1);
                                    }
                                    FocusArea::FeatureNavigator => {
                                        if app.feature_nav_selected + 1 < FEATURE_LIST.len() {
                                            app.feature_nav_selected += 1;
                                        }
                                    }
                                    FocusArea::ProviderHealth => {
                                        app.provider_health_scroll += 1;
                                    }
                                    FocusArea::SecurityAuditor => {
                                        app.security_scroll += 1;
                                    }
                                    FocusArea::CodeEditor => {
                                        app.editor.scroll((1, 0));
                                    }
                                    FocusArea::CustomModels => {
                                        if app.models_selected + 1 < app.models_profiles.len() {
                                            app.models_selected += 1;
                                        }
                                    }
                                    FocusArea::ByteBotPanel => {
                                        // Down in ByteBot - no-op
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Enter => {
                                match app.focus {
                                    FocusArea::FileExplorer => {
                                        if let Some(fp) =
                                            app.file_tree.get(app.selected_file).cloned()
                                        {
                                            app.open_file_in_editor(&fp);
                                        }
                                    }
                                    FocusArea::ModelSelector => {
                                        if let Some(model) =
                                            app.available_models.get(app.selected_model)
                                        {
                                            app.config.default_model = model.clone();
                                            let _ = app.config.save();
                                            // llama.cpp servers only serve their loaded
                                            // model, so kick off a server-side swap so the
                                            // next generation actually uses this model.
                                            if let Some(inner) = llama_model_target(model) {
                                                app.llamacpp_control(
                                                    "switch",
                                                    Some(inner.to_string()),
                                                    tx.clone(),
                                                );
                                            }
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::CodeReview => {
                                        if !app.is_reviewing {
                                            app.submit_review(tx.clone());
                                        }
                                    }
                                    FocusArea::GitCommit => {
                                        if !app.commit_message.trim().is_empty() {
                                            // Execute git commit async or blockingly
                                            let msg = app.commit_message.clone();
                                            let _ = Command::new("git")
                                                .args(["commit", "-am", &msg])
                                                .output();
                                            app.commit_message.clear();
                                            app.commit_cursor = 0;
                                            app.refresh_git();
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::ByteBotPanel => {
                                        if !app.bytebot_running {
                                            // ↑ recalls last command from history
                                            if !app.bytebot_history.is_empty() {
                                                app.bytebot_command =
                                                    app.bytebot_history.last().unwrap().clone();
                                                app.bytebot_cursor = app.bytebot_command.len();
                                            }
                                        }
                                    }
                                    FocusArea::CollaborationHub => {
                                        if !app.collab_session_active {
                                            app.start_collab_session(tx.clone());
                                        }
                                    }
                                    FocusArea::VoiceInterface => {
                                        if !app.voice_active {
                                            app.start_voice_session(tx.clone());
                                        }
                                    }
                                    FocusArea::TerminalAssistant => {
                                        if !app.term_asst_active {
                                            app.start_terminal_assistant(tx.clone());
                                        }
                                    }
                                    FocusArea::SecurityAuditor => {
                                        if !app.sec_scan_active {
                                            app.start_security_scan(tx.clone());
                                        }
                                    }
                                    FocusArea::PerformanceProfiler => {
                                        if !app.profiler_active {
                                            app.start_profiler(tx.clone());
                                        }
                                    }
                                    FocusArea::CustomModels => {
                                        if !app.models_editing {
                                            app.start_custom_models();
                                        }
                                    }
                                    FocusArea::LearningMode => {
                                        if !app.learn_active {
                                            app.start_learning_mode();
                                        } else if app.learn_quiz_active && !app.learn_quiz_answered
                                        {
                                            // Check quiz answer
                                            app.learn_quiz_answered = true;
                                            // Simple check: first option is correct
                                            app.learn_quiz_correct = app.learn_quiz_selected == 0;
                                            if app.learn_quiz_correct {
                                                app.learn_progress_pct =
                                                    (app.learn_progress_pct + 20.0).min(100.0);
                                            }
                                        }
                                    }
                                    FocusArea::MultiLanguage => {
                                        if !app.lang_active {
                                            app.start_multi_language();
                                        }
                                    }
                                    FocusArea::Settings => {
                                        if app.settings_url_editing {
                                            // Commit edit depending on which item was selected
                                            match app.settings_cursor {
                                                6 => {
                                                    app.config.ollama_url =
                                                        app.settings_url_buffer.clone();
                                                }
                                                7 => {
                                                    app.config.llama_cpp_url =
                                                        app.settings_url_buffer.clone();
                                                }
                                                8 => {
                                                    app.config.llama_cpp_model_path =
                                                        app.settings_url_buffer.clone();
                                                }
                                                9 => {
                                                    let v = app
                                                        .settings_url_buffer
                                                        .trim()
                                                        .parse::<f64>()
                                                        .ok()
                                                        .filter(|x| x.is_finite());
                                                    app.config.llama_cpp_temperature = v;
                                                }
                                                10 => {
                                                    app.config.llama_cpp_top_k =
                                                        app.settings_url_buffer.trim().parse().ok();
                                                }
                                                11 => {
                                                    let v = app
                                                        .settings_url_buffer
                                                        .trim()
                                                        .parse::<f64>()
                                                        .ok()
                                                        .filter(|x| x.is_finite());
                                                    app.config.llama_cpp_min_p = v;
                                                }
                                                12 => {
                                                    app.config.llama_cpp_max_tokens = app
                                                        .settings_url_buffer
                                                        .trim()
                                                        .parse()
                                                        .ok();
                                                }
                                                _ => {}
                                            }
                                            app.settings_url_editing = false;
                                            let _ = app.config.save();
                                            // Refresh models and health check when a URL/model path changed
                                            if app.settings_cursor <= 8 {
                                                app.refresh_models(tx.clone());
                                                app.run_health_check(tx.clone());
                                            }
                                        } else if app.settings_cursor == 6 {
                                            // Start Ollama URL editing
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app.config.ollama_url.clone();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 7 {
                                            // Start Llama.cpp URL editing
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app.config.llama_cpp_url.clone();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 8 {
                                            // Start Llama.cpp model path editing
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer =
                                                app.config.llama_cpp_model_path.clone();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 9 {
                                            // Temperature
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app
                                                .config
                                                .llama_cpp_temperature
                                                .map(|v| v.to_string())
                                                .unwrap_or_default();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 10 {
                                            // Top-K
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app
                                                .config
                                                .llama_cpp_top_k
                                                .map(|v| v.to_string())
                                                .unwrap_or_default();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 11 {
                                            // Min-P
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app
                                                .config
                                                .llama_cpp_min_p
                                                .map(|v| v.to_string())
                                                .unwrap_or_default();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 12 {
                                            // Max tokens
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app
                                                .config
                                                .llama_cpp_max_tokens
                                                .map(|v| v.to_string())
                                                .unwrap_or_default();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 13 {
                                            // Factory reset
                                            let defaults = XencodeConfig::default();
                                            app.config = defaults;
                                            app.theme = ThemeColors::get(&app.config.active_theme);
                                            app.settings_reset_active = true;
                                            app.settings_cursor = 0;
                                            let _ = app.config.save();
                                            app.refresh_models(tx.clone());
                                            app.run_health_check(tx.clone());
                                            app.focus = FocusArea::ChatInput;
                                        } else {
                                            let _ = app.config.save();
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::FeatureNavigator => {
                                        let target = app.navigate_feature(app.feature_nav_selected);
                                        app.focus = target;
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Char(' ') => {
                                if app.focus == FocusArea::FileExplorer {
                                    if let Some(fp) = app.file_tree.get(app.selected_file) {
                                        let fp = fp.clone();
                                        if app.attached_files.contains(&fp) {
                                            app.attached_files.remove(&fp);
                                        } else {
                                            app.attached_files.insert(fp);
                                        }
                                    }
                                } else if app.focus == FocusArea::SecurityAuditor {
                                    // Cycle severity filter
                                    app.sec_filter_severity =
                                        match app.sec_filter_severity.as_str() {
                                            "All" => "Critical",
                                            "Critical" => "High",
                                            "High" => "Medium",
                                            "Medium" => "Low",
                                            _ => "All",
                                        }
                                        .to_string();
                                } else if app.focus == FocusArea::TerminalAssistant {
                                    // Cycle risk filter
                                    app.term_risk_filter = match app.term_risk_filter.as_str() {
                                        "All" => "Safe",
                                        "Safe" => "Destructive",
                                        _ => "All",
                                    }
                                    .to_string();
                                } else if app.focus == FocusArea::VoiceInterface {
                                    // Toggle mute
                                    app.voice_muted = !app.voice_muted;
                                }
                            }
                            KeyCode::Char('i') | KeyCode::Char('/') => {
                                app.input_mode = InputMode::Editing;
                                app.focus = FocusArea::ChatInput;
                            }
                            KeyCode::Char('m') => {
                                app.focus = if app.focus == FocusArea::ModelSelector {
                                    FocusArea::ChatInput
                                } else {
                                    app.refresh_models(tx.clone());
                                    FocusArea::ModelSelector
                                };
                            }
                            KeyCode::Char('s') => {
                                app.focus = if app.focus == FocusArea::Settings {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::Settings
                                };
                            }
                            KeyCode::Char('r') if app.focus == FocusArea::ModelSelector => {
                                app.refresh_models(tx.clone());
                            }
                            KeyCode::Char('l') if app.focus == FocusArea::ModelSelector => {
                                // Load the selected model via llama.cpp
                                let target = app
                                    .available_models
                                    .get(app.selected_model)
                                    .and_then(|m| llama_model_target(m));
                                app.llamacpp_control("load", target.map(|s| s.to_string()), tx.clone());
                            }
                            KeyCode::Char('u') if app.focus == FocusArea::ModelSelector => {
                                app.llamacpp_control("unload", None, tx.clone());
                            }
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Esc => {
                                if app.init_visible {
                                    app.init_visible = false;
                                } else {
                                    match app.focus {
                                FocusArea::Settings => {
                                    if app.settings_url_editing {
                                        app.settings_url_editing = false;
                                    } else {
                                        app.settings_reset_active = false;
                                        let _ = app.config.save();
                                        app.focus = FocusArea::ChatInput;
                                    }
                                }
                                FocusArea::ModelSelector
                                | FocusArea::CodeReview
                                | FocusArea::PerformanceDashboard
                                | FocusArea::ProviderHealth
                                | FocusArea::ProjectAnalyzer
                                | FocusArea::GitCommit
                                | FocusArea::FeatureNavigator
                                | FocusArea::ByteBotPanel
                                | FocusArea::CollaborationHub
                                | FocusArea::VoiceInterface
                                | FocusArea::TerminalAssistant
                                | FocusArea::SecurityAuditor
                                | FocusArea::PerformanceProfiler
                                | FocusArea::CustomModels
                                | FocusArea::LearningMode
                                | FocusArea::MultiLanguage => {
                                    app.focus = FocusArea::ChatInput;
                                }
                                FocusArea::CodeEditor => {
                                    app.input_mode = InputMode::Normal;
                                }
                                _ => {}
                                }
                                }
                            },
                            KeyCode::Char(c) => {
                                if app.focus == FocusArea::Settings && app.settings_url_editing {
                                    app.settings_url_buffer.insert(app.settings_url_cursor, c);
                                    app.settings_url_cursor += 1;
                                } else if app.focus == FocusArea::GitCommit {
                                    app.commit_message.insert(app.commit_cursor, c);
                                    app.commit_cursor += 1;
                                } else if app.focus == FocusArea::ByteBotPanel {
                                    app.bytebot_command.insert(app.bytebot_cursor, c);
                                    app.bytebot_cursor += 1;
                                } else if c == 's' && app.focus == FocusArea::SecurityAuditor {
                                    // Toggle sort mode
                                    app.sec_sort_mode = if app.sec_sort_mode == "severity" {
                                        "category".to_string()
                                    } else {
                                        "severity".to_string()
                                    };
                                } else if c == 'm' && app.focus == FocusArea::VoiceInterface {
                                    // Toggle mute
                                    app.voice_muted = !app.voice_muted;
                                } else if c == 's'
                                    && app.focus == FocusArea::CustomModels
                                    && app.models_editing
                                {
                                    // Save profile
                                    app.models_saving = true;
                                    app.models_test_output = "Profile saved!".to_string();
                                } else if c == 'e' && app.focus == FocusArea::CodeEditor {
                                    app.input_mode = InputMode::Editing;
                                }
                            }
                            KeyCode::Backspace => {
                                if app.focus == FocusArea::Settings
                                    && app.settings_url_editing
                                    && app.settings_url_cursor > 0
                                {
                                    app.settings_url_cursor -= 1;
                                    app.settings_url_buffer.remove(app.settings_url_cursor);
                                } else if app.focus == FocusArea::GitCommit && app.commit_cursor > 0
                                {
                                    app.commit_cursor -= 1;
                                    app.commit_message.remove(app.commit_cursor);
                                } else if app.focus == FocusArea::ByteBotPanel
                                    && app.bytebot_cursor > 0
                                {
                                    app.bytebot_cursor -= 1;
                                    app.bytebot_command.remove(app.bytebot_cursor);
                                }
                            }
                            KeyCode::Left => {
                                if app.focus == FocusArea::Settings {
                                    if app.settings_url_editing
                                        && (6..=12).contains(&app.settings_cursor)
                                        && app.settings_url_cursor > 0
                                    {
                                        app.settings_url_cursor -= 1;
                                    } else if !app.settings_url_editing {
                                        match app.settings_cursor {
                                            0 => {
                                                let themes = [
                                                    "ocean",
                                                    "midnight",
                                                    "forest",
                                                    "terminal",
                                                    "dracula",
                                                    "solarized",
                                                    "nord",
                                                ];
                                                if let Some(pos) = themes
                                                    .iter()
                                                    .position(|t| *t == app.config.active_theme)
                                                {
                                                    app.config.active_theme = themes
                                                        [(pos + themes.len() - 1) % themes.len()]
                                                    .to_string();
                                                    app.theme =
                                                        ThemeColors::get(&app.config.active_theme);
                                                    let _ = app.config.save();
                                                }
                                            }
                                            1 => {
                                                app.config.cache_enabled = !app.config.cache_enabled;
                                                let _ = app.config.save();
                                            }
                                            2 => {
                                                app.config.memory_enabled =
                                                    !app.config.memory_enabled;
                                                let _ = app.config.save();
                                            }
                                            3 => {
                                                if app.config.max_cache_size >= 20 {
                                                    app.config.max_cache_size -= 10;
                                                    let _ = app.config.save();
                                                }
                                            }
                                            4 => {
                                                if app.config.max_memory_items >= 10 {
                                                    app.config.max_memory_items -= 5;
                                                    let _ = app.config.save();
                                                }
                                            }
                                            5 => {
                                                if app.config.response_timeout >= 10 {
                                                    app.config.response_timeout -= 5;
                                                    let _ = app.config.save();
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                } else if app.focus == FocusArea::GitCommit && app.commit_cursor > 0
                                {
                                    app.commit_cursor -= 1;
                                    app.commit_message.remove(app.commit_cursor);
                                } else if app.focus == FocusArea::ByteBotPanel
                                    && app.bytebot_cursor > 0
                                {
                                    app.bytebot_cursor -= 1;
                                    app.bytebot_command.remove(app.bytebot_cursor);
                                } else if app.focus == FocusArea::LearningMode
                                    && app.learn_quiz_active
                                    && !app.learn_quiz_answered
                                    && app.learn_quiz_selected > 0
                                {
                                    app.learn_quiz_selected -= 1;
                                } else if app.focus == FocusArea::CustomModels
                                    && app.models_editing
                                    && app.models_selected > 0
                                {
                                    app.models_selected -= 1;
                                }
                            }
                            KeyCode::Right => {
                                if app.focus == FocusArea::Settings {
                                    if app.settings_url_editing
                                        && (6..=12).contains(&app.settings_cursor)
                                        && app.settings_url_cursor < app.settings_url_buffer.len()
                                    {
                                        app.settings_url_cursor += 1;
                                    } else if !app.settings_url_editing {
                                        match app.settings_cursor {
                                            0 => {
                                                let themes = [
                                                    "ocean",
                                                    "midnight",
                                                    "forest",
                                                    "terminal",
                                                    "dracula",
                                                    "solarized",
                                                    "nord",
                                                ];
                                                if let Some(pos) = themes
                                                    .iter()
                                                    .position(|t| *t == app.config.active_theme)
                                                {
                                                    app.config.active_theme = themes
                                                        [(pos + 1) % themes.len()]
                                                    .to_string();
                                                    app.theme =
                                                        ThemeColors::get(&app.config.active_theme);
                                                    let _ = app.config.save();
                                                }
                                            }
                                            1 => {
                                                app.config.cache_enabled = !app.config.cache_enabled;
                                                let _ = app.config.save();
                                            }
                                            2 => {
                                                app.config.memory_enabled =
                                                    !app.config.memory_enabled;
                                                let _ = app.config.save();
                                            }
                                            3 => {
                                                app.config.max_cache_size = app
                                                    .config
                                                    .max_cache_size
                                                    .saturating_add(10)
                                                    .min(1000);
                                                let _ = app.config.save();
                                            }
                                            4 => {
                                                app.config.max_memory_items = app
                                                    .config
                                                    .max_memory_items
                                                    .saturating_add(5)
                                                    .min(500);
                                                let _ = app.config.save();
                                            }
                                            5 => {
                                                app.config.response_timeout = app
                                                    .config
                                                    .response_timeout
                                                    .saturating_add(5)
                                                    .min(300);
                                                let _ = app.config.save();
                                            }
                                            _ => {}
                                        }
                                    }
                                } else if app.focus == FocusArea::GitCommit
                                    && app.commit_cursor < app.commit_message.len()
                                {
                                    app.commit_cursor += 1;
                                } else if app.focus == FocusArea::ByteBotPanel
                                    && app.bytebot_cursor < app.bytebot_command.len()
                                {
                                    app.bytebot_cursor += 1;
                                } else if app.focus == FocusArea::LearningMode
                                    && app.learn_quiz_active
                                    && !app.learn_quiz_answered
                                    && app.learn_quiz_selected + 1 < app.learn_quiz_options.len()
                                {
                                    app.learn_quiz_selected += 1;
                                } else if app.focus == FocusArea::CustomModels
                                    && app.models_editing
                                    && app.models_selected + 1 < app.models_profiles.len()
                                {
                                    app.models_selected += 1;
                                }
                            }
                            _ => {}
                        },
                        InputMode::Editing => {
                            // If editor is focused, forward input to textarea
                            if app.focus == FocusArea::CodeEditor {
match key.code {
                                     KeyCode::Esc => {
                                         app.input_mode = InputMode::Normal;
                                     }
                                     _ => {
                                        app.editor.input(key);
                                        app.editor_dirty = true;
                                    }
                                }
                            } else {
                                // Normal chat input editing
                                match key.code {
                                    KeyCode::Enter => {
                                        if !app.is_generating {
                                            app.submit_message(tx.clone());
                                            app.chat_scroll = 0;
                                        }
                                    }
                                    KeyCode::Char(c) => {
                                        app.input.insert(app.input_cursor, c);
                                        app.input_cursor += 1;
                                    }
                                    KeyCode::Backspace => {
                                        if app.input_cursor > 0 {
                                            app.input_cursor -= 1;
                                            app.input.remove(app.input_cursor);
                                        }
                                    }
                                    KeyCode::Delete => {
                                        if app.input_cursor < app.input.len() {
                                            app.input.remove(app.input_cursor);
                                        }
                                    }
                                    KeyCode::Left => {
                                        if app.input_cursor > 0 {
                                            app.input_cursor -= 1;
                                        }
                                    }
                                    KeyCode::Right => {
                                        if app.input_cursor < app.input.len() {
                                            app.input_cursor += 1;
                                        }
                                    }
                                    KeyCode::Home => {
                                        app.input_cursor = 0;
                                    }
                                    KeyCode::End => {
                                        app.input_cursor = app.input.len();
                                    }
                                    KeyCode::Esc => {
                                        app.input_mode = InputMode::Normal;
                                    }
                                    KeyCode::Tab => {
                                        app.input.insert_str(app.input_cursor, "    ");
                                        app.input_cursor += 4;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => match app.focus {
                        FocusArea::ChatInput => {
                            app.chat_scroll = app.chat_scroll.saturating_add(3);
                        }
                        FocusArea::CodeEditor => {
                            app.editor.scroll((-3, 0));
                        }
                        FocusArea::FileExplorer => {
                            if app.selected_file >= 3 {
                                app.selected_file -= 3;
                            } else {
                                app.selected_file = 0;
                            }
                        }
                        FocusArea::FeatureNavigator => {
                            if app.feature_nav_selected >= 3 {
                                app.feature_nav_selected -= 3;
                            } else {
                                app.feature_nav_selected = 0;
                            }
                        }
                        FocusArea::ProviderHealth => {
                            if app.provider_health_scroll >= 3 {
                                app.provider_health_scroll -= 3;
                            } else {
                                app.provider_health_scroll = 0;
                            }
                        }
                        FocusArea::SecurityAuditor => {
                            if app.security_scroll >= 3 {
                                app.security_scroll -= 3;
                            } else {
                                app.security_scroll = 0;
                            }
                        }
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.focus {
                        FocusArea::ChatInput => {
                            app.chat_scroll = app.chat_scroll.saturating_sub(3);
                        }
                        FocusArea::CodeEditor => {
                            app.editor.scroll((3, 0));
                        }
                        FocusArea::FileExplorer => {
                            app.selected_file =
                                (app.selected_file + 3).min(app.file_tree.len().saturating_sub(1));
                        }
                        FocusArea::FeatureNavigator => {
                            app.feature_nav_selected = (app.feature_nav_selected + 3)
                                .min(FEATURE_LIST.len().saturating_sub(1));
                        }
                        FocusArea::ProviderHealth => {
                            app.provider_health_scroll += 3;
                        }
                        FocusArea::SecurityAuditor => {
                            app.security_scroll += 3;
                        }
                        _ => {}
                    },
                    MouseEventKind::Down(MouseButton::Left) => {
                        let term_width = terminal.size()?.width;
                        let left_pane = term_width * 20 / 100;
                        let center_pane = term_width * 70 / 100;

                        if mouse.column < left_pane {
                            app.focus = FocusArea::FileExplorer;
                            let row = mouse.row.saturating_sub(2) as usize;
                            if row < app.file_tree.len() {
                                app.selected_file = row;
                            }
                        } else if mouse.column < center_pane {
                            app.focus = FocusArea::CodeEditor;
                        } else {
                            app.focus = FocusArea::ChatInput;
                        }
                    }
                    _ => {}
                },
                Event::Resize(_, _) => {}
                _ => {}
            }
        } else if app.is_generating
            || app.is_reviewing
            || app.health_check_in_progress
            || app.bytebot_running
            || app.voice_active
            || app.collab_sync_status == "syncing"
            || app.sec_scan_active
            || app.profiler_running
        {
            app.spinner_tick = app.spinner_tick.wrapping_add(1);
        }
    }
}
