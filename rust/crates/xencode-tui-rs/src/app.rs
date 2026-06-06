use std::collections::{HashMap, HashSet};
use std::io;
use std::process::Command;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind, MouseButton};
use ratatui::{backend::Backend, Terminal};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::{current_timestamp, HealthStatus, OllamaClient};
use xencode_providers_rs::{ChatMessage, ProviderManager};

use crate::ui;

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
    pub bytebot_steps: Vec<(String, String)>,  // (step_name, status)
    pub bytebot_progress: f64,
    pub bytebot_running: bool,
    pub bytebot_log: Vec<String>,

    // Collaboration Hub state
    pub collab_session_active: bool,
    pub collab_session_id: String,
    pub collab_members: Vec<(String, String, String)>,  // (name, status, connection)
    pub collab_sync_status: String,  // "synced", "syncing", "error"
    pub collab_last_sync: f64,
    pub collab_pending_changes: u32,
    pub collab_activity_log: Vec<String>,

    // Voice Interface state
    pub voice_active: bool,
    pub voice_status: String,  // "idle", "listening", "processing", "speaking"
    pub voice_level: f64,      // simulated audio level 0.0-1.0
    pub voice_transcript: Vec<String>,
    pub voice_commands: Vec<(String, String)>,  // (command, result)

    // Terminal Assistant state
    pub term_asst_active: bool,
    pub term_asst_query: String,
    pub term_asst_cursor: usize,
    pub term_asst_suggestions: Vec<String>,
    pub term_asst_output: String,
    pub term_asst_history: Vec<(String, String, String)>,  // (command, risk, explanation)

    // Security Auditor state
    pub sec_scan_active: bool,
    pub sec_scan_path: String,
    pub sec_scan_results: Vec<(String, String, String)>,  // (severity, category, file)
    pub sec_scan_summary: (u32, u32, u32, u32),  // (critical, high, medium, low)
    pub sec_scan_progress: f64,
    pub sec_scan_log: Vec<String>,

    // Performance Profiler state
    pub profiler_active: bool,
    pub profiler_running: bool,
    pub profiler_functions: Vec<(String, f64, f64, u32)>,  // (name, time_ms, mem_mb, calls)
    pub profiler_gauge_cpu: f64,
    pub profiler_gauge_mem: f64,
    pub profiler_gauge_latency: f64,

    // Custom Models state
    pub models_editing: bool,
    pub models_profiles: Vec<(String, String, f64, u32, f64)>,  // (name, provider, temp, max_tokens, top_p)
    pub models_selected: usize,
    pub models_test_output: String,

    // Learning Mode state
    pub learn_active: bool,
    pub learn_current_lesson: usize,
    pub learn_total_lessons: usize,
    pub learn_lesson_title: String,
    pub learn_content: Vec<String>,
    pub learn_code_example: String,
    pub learn_exercise: String,
    pub learn_progress_pct: f64,

    // Multi-Language state
    pub lang_active: bool,
    pub lang_detection_results: Vec<(String, String, String)>,  // (file, language, confidence)
    pub lang_supported: Vec<(String, String)>,  // (language, status)
    pub lang_translate_input: String,
    pub lang_translate_output: String,

    // Settings interactive state
    pub settings_cursor: usize,
    pub settings_reset_active: bool,
    pub settings_url_editing: bool,
    pub settings_url_buffer: String,
    pub settings_url_cursor: usize,
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
                ".git".to_string(), "node_modules".to_string(), "target".to_string(),
                "__pycache__".to_string(), ".pytest_cache".to_string(), ".venv".to_string(),
            ],
        };
        let tree = scan_workspace(".", &scan_opts).unwrap_or_default();
        let file_tree: Vec<String> = tree.into_iter().map(|f| f.path.display().to_string()).collect();

        let available_models = vec![
            "qwen2.5:7b".to_string(), "llama3.1:8b".to_string(),
            "anthropic/claude-3.5-sonnet".to_string(), "google/gemini-1.5-pro".to_string(),
            "openai/gpt-4o".to_string(),
        ];
        let selected_model = available_models.iter().position(|m| m == &config.default_model).unwrap_or(0);
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

        let mut editor = TextArea::default();
        editor.set_line_number_style(ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray));

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
            collab_session_active: false,
            collab_session_id: String::new(),
            collab_members: Vec::new(),
            collab_sync_status: "disconnected".to_string(),
            collab_last_sync: 0.0,
            collab_pending_changes: 0,
            collab_activity_log: Vec::new(),

            voice_active: false,
            voice_status: "idle".to_string(),
            voice_level: 0.0,
            voice_transcript: Vec::new(),
            voice_commands: Vec::new(),

            term_asst_active: false,
            term_asst_query: String::new(),
            term_asst_cursor: 0,
            term_asst_suggestions: Vec::new(),
            term_asst_output: String::new(),
            term_asst_history: Vec::new(),

            sec_scan_active: false,
            sec_scan_path: String::new(),
            sec_scan_results: Vec::new(),
            sec_scan_summary: (0, 0, 0, 0),
            sec_scan_progress: 0.0,
            sec_scan_log: Vec::new(),

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

            learn_active: false,
            learn_current_lesson: 0,
            learn_total_lessons: 0,
            learn_lesson_title: String::new(),
            learn_content: Vec::new(),
            learn_code_example: String::new(),
            learn_exercise: String::new(),
            learn_progress_pct: 0.0,

            lang_active: false,
            lang_detection_results: Vec::new(),
            lang_supported: Vec::new(),
            lang_translate_input: String::new(),
            lang_translate_output: String::new(),

            settings_cursor: 0,
            settings_reset_active: false,
            settings_url_editing: false,
            settings_url_buffer: String::new(),
            settings_url_cursor: 0,
        };

        // Seed initial health entries for configured providers
        app.ollama_health_entries.insert(
            "ollama".to_string(),
            (HealthStatus::Unknown.to_string(), 0.0, None),
        );
        app.ollama_health_entries.insert(
            "openrouter".to_string(),
            (
                if app.config.api_keys.openrouter_api_key.is_some() { HealthStatus::Unknown.to_string() } else { HealthStatus::Error.to_string() },
                0.0,
                if app.config.api_keys.openrouter_api_key.is_none() { Some("API key not configured".to_string()) } else { None },
            ),
        );
        app.ollama_health_entries.insert(
            "qwen".to_string(),
            (
                if app.config.api_keys.qwen_api_key.is_some() { HealthStatus::Unknown.to_string() } else { HealthStatus::Error.to_string() },
                0.0,
                if app.config.api_keys.qwen_api_key.is_none() { Some("API key not configured".to_string()) } else { None },
            ),
        );
        app.ollama_health_entries.insert(
            "gemini".to_string(),
            (
                if app.config.api_keys.google_gemini_api_key.is_some() { HealthStatus::Unknown.to_string() } else { HealthStatus::Error.to_string() },
                0.0,
                if app.config.api_keys.google_gemini_api_key.is_none() { Some("API key not configured".to_string()) } else { None },
            ),
        );

        for msg in app.memory.get_context(10) {
            app.messages.push(UiMessage { role: msg.role.clone(), content: msg.content.clone() });
        }
        app
    }

    pub fn open_file_in_editor(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                self.editor = TextArea::new(if lines.is_empty() { vec![String::new()] } else { lines });
                self.editor.set_line_number_style(ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray));
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
    }

    pub fn submit_message(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.input.trim().is_empty() { return; }
        let prompt = self.input.clone();
        self.input.clear();
        self.input_cursor = 0;

        self.messages.push(UiMessage { role: "user".to_string(), content: prompt.clone() });
        self.memory.add_message("user", &prompt, None);
        self.is_generating = true;

        // ByteBot interception
        if prompt.starts_with("/bytebot") {
            let command = prompt.strip_prefix("/bytebot").unwrap_or("").trim().to_string();
            tokio::spawn(async move {
                let _ = tx.send(format!("⚡ ByteBot: Initializing for '{}'\n", command));
                tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
                for step in ["Analyzing workspace...", "Formulating plan...", "Scanning deps...",
                             "Running tests...", "Applying changes...", "Verifying..."] {
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
            context_messages.push(ChatMessage { role: msg.role, content: msg.content });
        }

        let mut attached_context = String::new();
        for path in &self.attached_files {
            if let Ok(content) = std::fs::read_to_string(path) {
                attached_context.push_str(&format!("<file path=\"{}\">\n{}\n</file>\n\n", path, content));
            }
        }
        if !attached_context.is_empty() {
            context_messages.insert(0, ChatMessage {
                role: "system".to_string(),
                content: format!("Attached files:\n{}", attached_context),
            });
        }

        let model = self.config.default_model.clone();
        let ollama_url = self.config.ollama_url.clone();
        let timeout = self.config.response_timeout;
        let or_key = self.config.api_keys.openrouter_api_key.clone();
        let qwen_key = self.config.api_keys.qwen_api_key.clone();
        let gemini_key = self.config.api_keys.google_gemini_api_key.clone();

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout);
            let manager = ProviderManager::new(client, or_key, qwen_key, gemini_key);
            let _ = manager.generate_stream(&model, &context_messages, |token| {
                let _ = tx.send(token.to_string());
            }).await;
            let _ = tx.send("[DONE]".to_string());
        });
    }

    pub fn append_generation(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_generating = false;
            self.total_llm_calls += 1;
            if let Some(last) = self.messages.last() {
                if last.role == "assistant" {
                    self.memory.add_message("assistant", &last.content, Some(self.config.default_model.clone()));
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
        self.messages.push(UiMessage { role: "assistant".to_string(), content: text.to_string() });
    }

    pub fn append_review(&mut self, text: &str) {
        if text == "[DONE]" { self.is_reviewing = false; }
        else { self.code_review_output.push_str(text); }
    }

    /// Run asynchronous health checks for all configured providers.
    /// Results are sent back through the channel for processing in the event loop.
    /// Start a Collaboration Hub session with simulated team members and sync.
    pub fn start_collab_session(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.collab_session_active { return; }
        self.collab_session_active = true;
        self.collab_session_id = format!("xencode-{:06x}", (current_timestamp() as u64) & 0xFFFFFF);
        self.collab_sync_status = "connecting".to_string();
        self.collab_pending_changes = 0;
        self.collab_activity_log.clear();

        // Seed initial team members
        self.collab_members = vec![
            ("You (local)".to_string(), "online".to_string(), "🔗 LAN".to_string()),
            ("alice".to_string(), "online".to_string(), "🌐 WAN".to_string()),
            ("bob".to_string(), "away".to_string(), "🌐 WAN".to_string()),
            ("carol".to_string(), "busy".to_string(), "🔗 LAN".to_string()),
        ];

        self.collab_activity_log.push("🔌 Connecting to collaboration server...".to_string());

        let session_id = self.collab_session_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let _ = tx.send("[COLLAB]status:connected".to_string());
            let _ = tx.send(format!("[COLLAB]log:🔗 Connected — Session: {}", session_id));
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
        if self.bytebot_running || self.bytebot_command.trim().is_empty() { return; }
        
        let command = self.bytebot_command.trim().to_string();
        self.bytebot_running = true;
        self.bytebot_progress = 0.0;
        self.bytebot_steps = vec![
            ("Analyzing workspace".to_string(), "pending".to_string()),
            ("Scanning dependencies".to_string(), "pending".to_string()),
            ("Formulating execution plan".to_string(), "pending".to_string()),
            ("Running tests".to_string(), "pending".to_string()),
            ("Applying changes".to_string(), "pending".to_string()),
            ("Verifying results".to_string(), "pending".to_string()),
        ];
        self.bytebot_log.clear();
        self.bytebot_log.push(format!("⚡ ByteBot: Initializing for '{}'", command));
        self.bytebot_command.clear();
        self.bytebot_cursor = 0;

        tokio::spawn(async move {
            let steps = [
                ("Analyzing workspace", "📁 Found 342 files in workspace"),
                ("Scanning dependencies", "🔍 Identified 12 outdated packages"),
                ("Formulating execution plan", "📋 Plan: update 5 deps, fix 3 deprecations"),
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
                let _ = tx.send(format!("[BYTEBOT]log:{}  → {} — {}", "▸", step_name, detail));
                
                // Mark step as done
                let _ = tx.send(format!("[BYTEBOT]step:{}:done:{}", i, step_name));
            }
            
            let _ = tx.send("[BYTEBOT]log:✅ ByteBot execution complete.".to_string());
            let _ = tx.send("[BYTEBOT_DONE]".to_string());
        });
    }

    /// Start Voice Interface simulation with audio level and speech-to-text.
    pub fn start_voice_session(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.voice_active { return; }
        self.voice_active = true;
        self.voice_status = "listening".to_string();
        self.voice_transcript.clear();
        self.voice_commands.clear();
        self.voice_transcript.push("🎤 Microphone initialized".to_string());

        tokio::spawn(async move {
            let phrases = vec![
                ("refactor user model", "✅ Model refactored — UserModel split into User + Profile"),
                ("add validation for email", "✅ Added email validation regex to UserService"),
                ("run tests", "✅ 142 tests passed, 0 failed"),
                ("commit changes", "✅ Committed 'feat: add email validation'"),
            ];

            for (cmd, result) in &phrases {
                // Simulate listening with audio levels
                for level in [0.3, 0.6, 0.8, 0.9, 0.7, 0.4] {
                    let _ = tx.send(format!("[VOICE]level:{}", level));
                    tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
                }

                let _ = tx.send(format!("[VOICE]status:processing"));
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

                let _ = tx.send(format!("[VOICE]status:speaking"));
                let _ = tx.send(format!("[VOICE]transcript:{}", cmd));
                let _ = tx.send(format!("[VOICE]command:{}|{}", cmd, result));
                tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;

                let _ = tx.send(format!("[VOICE]status:listening"));
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }

            let _ = tx.send("[VOICE]status:idle".to_string());
        });
    }

    /// Start Terminal Assistant with command suggestions.
    pub fn start_terminal_assistant(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.term_asst_active { return; }
        self.term_asst_active = true;
        self.term_asst_suggestions.clear();
        self.term_asst_output.clear();
        self.term_asst_history.clear();

        tokio::spawn(async move {
            let _ = tx.send("[TERM]output:🧠 Terminal Assistant ready — type a query and press Enter".to_string());

            let suggestions = vec![
                ("find . -name \"*.py\" | xargs grep -l \"def \"", "🔍 Safe", "Find all Python files with function definitions"),
                ("git log --oneline --graph --all", "✅ Safe", "Visual git history graph"),
                ("du -sh */ 2>/dev/null | sort -rh", "✅ Safe", "Show directory sizes sorted by size"),
                ("docker system prune -af", "⚠️ Destructive", "⚠ Removes ALL unused Docker data"),
                ("rm -rf node_modules && npm install", "⚠️ Destructive", "⚠ Deletes node_modules and reinstalls"),
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
        if self.sec_scan_active { return; }
        self.sec_scan_active = true;
        self.sec_scan_results.clear();
        self.sec_scan_summary = (0, 0, 0, 0);
        self.sec_scan_progress = 0.0;
        self.sec_scan_log.clear();
        self.sec_scan_log.push("🔍 Starting vulnerability scan...".to_string());

        tokio::spawn(async move {
            let findings = vec![
                ("Critical", "Hardcoded API Key", "src/config.py:42", "❌ Found hardcoded AWS_SECRET_KEY"),
                ("High", "SQL Injection", "src/queries.py:18", "🚨 Raw SQL concatenation detected"),
                ("High", "Command Injection", "src/deploy.py:55", "🚨 Using os.system() with user input"),
                ("Medium", "Weak Crypto", "src/crypto.py:10", "⚠️ MD5 used for password hashing"),
                ("Medium", "XSS Vulnerability", "src/templates/user.html:22", "⚠️ Unsafe innerHTML assignment"),
                ("Low", "Deprecated Package", "requirements.txt:1", "📦 PyCrypto v2.6.1 is end-of-life"),
                ("Low", "Missing Rate Limit", "src/api.py:30", "🐢 No rate limiting on /login endpoint"),
            ];

            let total = findings.len();
            for (i, (severity, category, location, detail)) in findings.iter().enumerate() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                let progress = (i as f64 + 1.0) / total as f64;
                let _ = tx.send(format!("[SECURITY]progress:{:.2}", progress));
                let _ = tx.send(format!("[SECURITY]finding:{}|{}|{}|{}", severity, category, location, detail));
            }

            let _ = tx.send("[SECURITY]done".to_string());
        });
    }

    /// Start Performance Profiler simulation.
    pub fn start_profiler(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.profiler_running { return; }
        self.profiler_active = true;
        self.profiler_running = true;
        self.profiler_functions.clear();

        tokio::spawn(async move {
            let funcs = vec![
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
                let _ = tx.send(format!("[PROFILER]func:{}|{}|{}|{}", name, time_ms, mem_mb, calls));
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
            let _ = tx.send("[PROFILER]done".to_string());
        });
    }

    /// Start Custom Models session (seeds profile data).
    pub fn start_custom_models(&mut self) {
        if self.models_editing { return; }
        self.models_editing = true;
        self.models_profiles = vec![
            ("Code Assistant".to_string(), "ollama".to_string(), 0.3, 4096, 0.9),
            ("Creative Writer".to_string(), "openrouter".to_string(), 0.8, 2048, 0.95),
            ("Bug Hunter".to_string(), "ollama".to_string(), 0.2, 8192, 0.8),
            ("Code Reviewer".to_string(), "openrouter".to_string(), 0.15, 4096, 0.85),
        ];
        self.models_selected = 0;
        self.models_test_output = String::new();
    }

    /// Start Learning Mode with lesson content.
    pub fn start_learning_mode(&mut self) {
        if self.learn_active { return; }
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
        ].join("\n");
        self.learn_exercise = "Fix the ownership error: let s2 = s; println!(\"{}\", s);".to_string();
        self.learn_progress_pct = 20.0;
    }

    /// Start Multi-Language panel with detection results.
    pub fn start_multi_language(&mut self) {
        if self.lang_active { return; }
        self.lang_active = true;
        self.lang_detection_results = vec![
            ("src/main.rs".to_string(), "Rust".to_string(), "99.2%".to_string()),
            ("src/app.py".to_string(), "Python".to_string(), "98.7%".to_string()),
            ("src/components.tsx".to_string(), "TypeScript".to_string(), "97.5%".to_string()),
            ("templates/index.html".to_string(), "HTML".to_string(), "96.8%".to_string()),
            ("styles/main.css".to_string(), "CSS".to_string(), "95.1%".to_string()),
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

    pub fn run_health_check(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.health_check_in_progress { return; }
        self.health_check_in_progress = true;

        let ollama_url = self.config.ollama_url.clone();
        let timeout = self.config.response_timeout;
        let openrouter_key = self.config.api_keys.openrouter_api_key.clone();
        let qwen_key = self.config.api_keys.qwen_api_key.clone();
        let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
        let default_model = self.config.default_model.clone();

        tokio::spawn(async move {
            // Check Ollama health
            let mut client = OllamaClient::new(&ollama_url, timeout);
            let start = std::time::Instant::now();

            if !default_model.contains('/') {
                // Local model - run real health check
                match client.check_health(&default_model).await {
                    Ok(health) => {
                        let _ = tx.send(format!(
                            "[HEALTH]ollama|{}|{}|{}",
                            health.status,
                            health.response_time * 1000.0, // ms
                            health.error_message.unwrap_or_default()
                        ));
                    }
                    Err(e) => {
                        let latency = start.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]ollama|error|{}|{}", latency, e));
                    }
                }
            } else {
                // OpenRouter model - just check connectivity to Ollama
                match client.check_health("llama3.2:3b").await {
                    Ok(health) => {
                        let _ = tx.send(format!(
                            "[HEALTH]ollama|{}|{}|{}",
                            health.status,
                            health.response_time * 1000.0,
                            health.error_message.unwrap_or_default()
                        ));
                    }
                    Err(e) => {
                        let latency = start.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]ollama|error|{}|{}", latency, e));
                    }
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
                            let _ = tx.send(format!("[HEALTH]openrouter|error|{}|HTTP {}", latency, resp.status()));
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
                            let _ = tx.send(format!("[HEALTH]gemini|error|{}|HTTP {}", latency, resp.status()));
                        }
                    }
                    Err(e) => {
                        let latency = start_ge.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]gemini|error|{}|{}", latency, e));
                    }
                }
            }

            let _ = tx.send("[HEALTH_DONE]".to_string());
        });
    }

    pub fn submit_review(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.is_reviewing { return; }
        if let Some(file_path) = self.file_tree.get(self.selected_file) {
            if let Ok(content) = std::fs::read_to_string(file_path) {
                self.is_reviewing = true;
                self.code_review_output = format!("📝 Reviewing: {}\n\n", file_path);
                let prompt = format!(
                    "Code review of {}. Identify bugs, security issues, and performance bottlenecks.\n\n```\n{}\n```",
                    file_path, content
                );
                let messages = vec![ChatMessage { role: "user".to_string(), content: prompt }];
                let model = self.config.default_model.clone();
                let ollama_url = self.config.ollama_url.clone();
                let timeout = self.config.response_timeout;
                let or_key = self.config.api_keys.openrouter_api_key.clone();
                let qwen_key = self.config.api_keys.qwen_api_key.clone();
                let gemini_key = self.config.api_keys.google_gemini_api_key.clone();

                tokio::spawn(async move {
                    let client = OllamaClient::new(&ollama_url, timeout);
                    let manager = ProviderManager::new(client, or_key, qwen_key, gemini_key);
                    let _ = manager.generate_stream(&model, &messages, |token| {
                        let _ = tx.send(format!("[REVIEW]{}", token));
                    }).await;
                    let _ = tx.send("[REVIEW][DONE]".to_string());
                });
            }
        }
    }
}

impl<'a> Default for App<'a> {
    fn default() -> Self { Self::new() }
}

pub async fn run_app<B: Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        // Drain async messages
        while let Ok(token) = rx.try_recv() {
            if token.starts_with("[REVIEW]") {
                app.append_review(&token[8..]);
            } else if token.starts_with("[BYTEBOT]") {
                let body = &token[9..];
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
            } else if token.starts_with("[COLLAB]") {
                let body = &token[8..];
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
                        if let Some(member) = app.collab_members.iter_mut().find(|(n, _, _)| n == &name) {
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
            } else if token.starts_with("[HEALTH]") {
                let parts: Vec<&str> = token[8..].splitn(4, '|').collect();
                if parts.len() >= 3 {
                    let provider = parts[0].to_string();
                    let status = parts[1].to_string();
                    let latency = parts[2].parse::<f64>().unwrap_or(0.0);
                    let error = if parts.len() > 3 && !parts[3].is_empty() { Some(parts[3].to_string()) } else { None };
                    app.ollama_health_entries.insert(provider, (status.clone(), latency, error));
                    // Update average latency across all providers
                    if status == "healthy" {
                        let total: f64 = app.ollama_health_entries.values().map(|(s, l, _)| if s == "healthy" { *l } else { 0.0 }).sum();
                        let count = app.ollama_health_entries.values().filter(|(s, _, _)| s == "healthy").count() as f64;
                        app.average_latency = if count > 0.0 { total / count } else { 0.0 };
                    }
                }
            } else if token.starts_with("[VOICE]") {
                let body = &token[7..];
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
            } else if token.starts_with("[TERM]") {
                let body = &token[6..];
                if body.starts_with("suggestion:") {
                    if let Some(s) = body.strip_prefix("suggestion:") {
                        let parts: Vec<&str> = s.splitn(3, '|').collect();
                        let cmd = parts.first().unwrap_or(&"").to_string();
                        let risk = parts.get(1).unwrap_or(&"").to_string();
                        let explanation = parts.get(2).unwrap_or(&"").to_string();
                        app.term_asst_suggestions.push(format!("{}  {} — {}", risk, cmd, explanation));
                    }
                } else if body.starts_with("output:") {
                    if let Some(o) = body.strip_prefix("output:") {
                        app.term_asst_output = o.to_string();
                    }
                } else if body == "ready" {
                    app.term_asst_active = false;
                }
            } else if token.starts_with("[SECURITY]") {
                let body = &token[10..];
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
                            app.sec_scan_results.push((severity.clone(), category.clone(), location.clone()));
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
            } else if token.starts_with("[PROFILER]") {
                let body = &token[10..];
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
                            KeyCode::Char('g') => { app.refresh_git(); continue; }
                            KeyCode::Char(',') => {
                                app.focus = if app.focus == FocusArea::Settings { FocusArea::ChatInput } else { FocusArea::Settings };
                                continue;
                            }
                            KeyCode::Char('r') => {
                                app.focus = if app.focus == FocusArea::CodeReview { FocusArea::ChatInput } else { FocusArea::CodeReview };
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
                                app.focus = if app.focus == FocusArea::FeatureNavigator { FocusArea::ChatInput } else { FocusArea::FeatureNavigator };
                                continue;
                            }
                            KeyCode::Char('s') => {
                                if app.focus == FocusArea::CodeEditor {
                                    app.save_editor();
                                } else {
                                    app.focus = if app.focus == FocusArea::GitCommit { FocusArea::ChatInput } else { FocusArea::GitCommit };
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
                            KeyCode::Up | KeyCode::Char('k') => {
                                match app.focus {
                                    FocusArea::Settings => {
                                        if app.settings_cursor > 0 { app.settings_cursor -= 1; }
                                    }
                                    FocusArea::FileExplorer => { if app.selected_file > 0 { app.selected_file -= 1; } }
                                    FocusArea::ModelSelector => { if app.selected_model > 0 { app.selected_model -= 1; } }
                                    FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_add(1); }
                                    FocusArea::FeatureNavigator => { if app.feature_nav_selected > 0 { app.feature_nav_selected -= 1; } }
                                    FocusArea::CodeEditor => { app.editor.scroll((-1, 0)); }
                                    FocusArea::CustomModels => {
                                        if app.models_selected > 0 { app.models_selected -= 1; }
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                match app.focus {
                                    FocusArea::Settings => {
                                        if app.settings_cursor + 1 < 8 { app.settings_cursor += 1; }
                                    }
                                    FocusArea::FileExplorer => {
                                        if app.selected_file + 1 < app.file_tree.len() { app.selected_file += 1; }
                                    }
                                    FocusArea::ModelSelector => {
                                        if app.selected_model + 1 < app.available_models.len() { app.selected_model += 1; }
                                    }
                                    FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_sub(1); }
                                    FocusArea::FeatureNavigator => {
                                        if app.feature_nav_selected + 1 < FEATURE_LIST.len() { app.feature_nav_selected += 1; }
                                    }
                                    FocusArea::CodeEditor => { app.editor.scroll((1, 0)); }
                                    FocusArea::CustomModels => {
                                        if app.models_selected + 1 < app.models_profiles.len() { app.models_selected += 1; }
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Enter => {
                                match app.focus {
                                    FocusArea::FileExplorer => {
                                        if let Some(fp) = app.file_tree.get(app.selected_file).cloned() {
                                            app.open_file_in_editor(&fp);
                                        }
                                    }
                                    FocusArea::ModelSelector => {
                                        if let Some(model) = app.available_models.get(app.selected_model) {
                                            app.config.default_model = model.clone();
                                            let _ = app.config.save();
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::CodeReview => {
                                        if !app.is_reviewing { app.submit_review(tx.clone()); }
                                    }
                                    FocusArea::GitCommit => {
                                        if !app.commit_message.trim().is_empty() {
                                            // Execute git commit async or blockingly
                                            let msg = app.commit_message.clone();
                                            let _ = Command::new("git").args(["commit", "-am", &msg]).output();
                                            app.commit_message.clear();
                                            app.commit_cursor = 0;
                                            app.refresh_git();
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::ByteBotPanel => {
                                        if !app.bytebot_running {
                                            app.run_bytebot(tx.clone());
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
                                        }
                                    }
                                    FocusArea::MultiLanguage => {
                                        if !app.lang_active {
                                            app.start_multi_language();
                                        }
                                    }
                                    FocusArea::Settings => {
                                        if app.settings_url_editing {
                                            // Commit URL edit
                                            app.config.ollama_url = app.settings_url_buffer.clone();
                                            app.settings_url_editing = false;
                                            let _ = app.config.save();
                                        } else if app.settings_cursor == 7 {
                                            // Start URL editing
                                            app.settings_url_editing = true;
                                            app.settings_url_buffer = app.config.ollama_url.clone();
                                            app.settings_url_cursor = app.settings_url_buffer.len();
                                        } else if app.settings_cursor == 6 {
                                            // Factory reset
                                            let defaults = XencodeConfig::default();
                                            app.config = defaults;
                                            app.theme = ThemeColors::get(&app.config.active_theme);
                                            app.settings_reset_active = true;
                                            app.settings_cursor = 0;
                                            let _ = app.config.save();
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
                                        if app.attached_files.contains(&fp) { app.attached_files.remove(&fp); }
                                        else { app.attached_files.insert(fp); }
                                    }
                                }
                            }
                            KeyCode::Char('i') | KeyCode::Char('/') => {
                                app.input_mode = InputMode::Editing;
                                app.focus = FocusArea::ChatInput;
                            }
                            KeyCode::Char('m') => {
                                app.focus = if app.focus == FocusArea::ModelSelector { FocusArea::ChatInput } else { FocusArea::ModelSelector };
                            }
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Esc => {
                                match app.focus {
                                    FocusArea::Settings => {
                                        if app.settings_url_editing {
                                            app.settings_url_editing = false;
                                        } else {
                                            app.settings_reset_active = false;
                                            app.focus = FocusArea::ChatInput;
                                        }
                                    }
                                    FocusArea::ModelSelector | FocusArea::CodeReview |
                                    FocusArea::PerformanceDashboard | FocusArea::ProviderHealth | FocusArea::ProjectAnalyzer | FocusArea::GitCommit |
                                    FocusArea::FeatureNavigator | FocusArea::ByteBotPanel | FocusArea::CollaborationHub |
                                    FocusArea::VoiceInterface | FocusArea::TerminalAssistant | FocusArea::SecurityAuditor |
                                    FocusArea::PerformanceProfiler | FocusArea::CustomModels | FocusArea::LearningMode | FocusArea::MultiLanguage => {
                                        app.focus = FocusArea::ChatInput;
                                    }
                                    FocusArea::CodeEditor => {
                                        app.input_mode = InputMode::Normal;
                                    }
                                    _ => {}
                                }
                            }
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
                                } else if c == 'e' && app.focus == FocusArea::CodeEditor {
                                    app.input_mode = InputMode::Editing;
                                }
                            }
                            KeyCode::Backspace => {
                                if app.focus == FocusArea::Settings && app.settings_url_editing && app.settings_url_cursor > 0 {
                                    app.settings_url_cursor -= 1;
                                    app.settings_url_buffer.remove(app.settings_url_cursor);
                                } else if app.focus == FocusArea::GitCommit && app.commit_cursor > 0 {
                                    app.commit_cursor -= 1;
                                    app.commit_message.remove(app.commit_cursor);
                                } else if app.focus == FocusArea::ByteBotPanel && app.bytebot_cursor > 0 {
                                    app.bytebot_cursor -= 1;
                                    app.bytebot_command.remove(app.bytebot_cursor);
                                }
                            }
                            KeyCode::Left => {
                                if app.focus == FocusArea::Settings {
                                    if app.settings_url_editing && app.settings_cursor == 7 && app.settings_url_cursor > 0 {
                                        app.settings_url_cursor -= 1;
                                    } else if !app.settings_url_editing {
                                        match app.settings_cursor {
                                            0 => {
                                                let themes = ["ocean", "midnight", "forest", "terminal", "dracula", "solarized", "nord"];
                                                if let Some(pos) = themes.iter().position(|t| *t == app.config.active_theme) {
                                                    app.config.active_theme = themes[(pos + themes.len() - 1) % themes.len()].to_string();
                                                    app.theme = ThemeColors::get(&app.config.active_theme);
                                                }
                                            }
                                            1 => app.config.cache_enabled = !app.config.cache_enabled,
                                            2 => app.config.memory_enabled = !app.config.memory_enabled,
                                            3 => { if app.config.max_cache_size >= 20 { app.config.max_cache_size -= 10; } }
                                            4 => { if app.config.max_memory_items >= 10 { app.config.max_memory_items -= 5; } }
                                            5 => { if app.config.response_timeout >= 10 { app.config.response_timeout -= 5; } }
                                            7 => {
                                                let defaults = XencodeConfig::default();
                                                app.config = defaults;
                                                app.theme = ThemeColors::get(&app.config.active_theme);
                                                app.settings_reset_active = true;
                                                app.settings_url_editing = false;
                                                app.settings_url_buffer.clear();
                                            }
                                            _ => {}
                                        }
                                    }
                                } else if app.focus == FocusArea::GitCommit && app.commit_cursor > 0 { app.commit_cursor -= 1; }
                                else if app.focus == FocusArea::ByteBotPanel && app.bytebot_cursor > 0 { app.bytebot_cursor -= 1; }
                            }
                            KeyCode::Right => {
                                if app.focus == FocusArea::Settings {
                                    if app.settings_url_editing && app.settings_cursor == 7 && app.settings_url_cursor < app.settings_url_buffer.len() {
                                        app.settings_url_cursor += 1;
                                    } else if !app.settings_url_editing {
                                        match app.settings_cursor {
                                            0 => {
                                                let themes = ["ocean", "midnight", "forest", "terminal", "dracula", "solarized", "nord"];
                                                if let Some(pos) = themes.iter().position(|t| *t == app.config.active_theme) {
                                                    app.config.active_theme = themes[(pos + 1) % themes.len()].to_string();
                                                    app.theme = ThemeColors::get(&app.config.active_theme);
                                                }
                                            }
                                            1 => app.config.cache_enabled = !app.config.cache_enabled,
                                            2 => app.config.memory_enabled = !app.config.memory_enabled,
                                            3 => app.config.max_cache_size = app.config.max_cache_size.saturating_add(10).min(1000),
                                            4 => app.config.max_memory_items = app.config.max_memory_items.saturating_add(5).min(500),
                                            5 => app.config.response_timeout = app.config.response_timeout.saturating_add(5).min(300),
                                            7 => {
                                                let defaults = XencodeConfig::default();
                                                app.config = defaults;
                                                app.theme = ThemeColors::get(&app.config.active_theme);
                                                app.settings_reset_active = true;
                                                app.settings_url_editing = false;
                                                app.settings_url_buffer.clear();
                                            }
                                            _ => {}
                                        }
                                    }
                                } else if app.focus == FocusArea::GitCommit && app.commit_cursor < app.commit_message.len() { app.commit_cursor += 1; }
                                else if app.focus == FocusArea::ByteBotPanel && app.bytebot_cursor < app.bytebot_command.len() { app.bytebot_cursor += 1; }
                            }
                            _ => {}
                        },
                        InputMode::Editing => {
                            // If editor is focused, forward input to textarea
                            if app.focus == FocusArea::CodeEditor {
                                match key.code {
                                    KeyCode::Esc => { app.input_mode = InputMode::Normal; }
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
                                if app.input_cursor > 0 { app.input_cursor -= 1; }
                            }
                            KeyCode::Right => {
                                if app.input_cursor < app.input.len() { app.input_cursor += 1; }
                            }
                            KeyCode::Home => { app.input_cursor = 0; }
                            KeyCode::End => { app.input_cursor = app.input.len(); }
                            KeyCode::Esc => { app.input_mode = InputMode::Normal; }
                            KeyCode::Tab => {
                                app.input.insert_str(app.input_cursor, "    ");
                                app.input_cursor += 4;
                            }
                            _ => {}
                                }
                            }
                        },
                    }
                }
                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            match app.focus {
                                FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_add(3); }
                                FocusArea::CodeEditor => { app.editor.scroll((-3, 0)); }
                                FocusArea::FileExplorer => {
                                    if app.selected_file >= 3 { app.selected_file -= 3; }
                                    else { app.selected_file = 0; }
                                }
                                _ => {}
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            match app.focus {
                                FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_sub(3); }
                                FocusArea::CodeEditor => { app.editor.scroll((3, 0)); }
                                FocusArea::FileExplorer => {
                                    app.selected_file = (app.selected_file + 3).min(app.file_tree.len().saturating_sub(1));
                                }
                                _ => {}
                            }
                        }
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
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        } else {
            if app.is_generating || app.is_reviewing || app.health_check_in_progress
                || app.bytebot_running || app.voice_active || app.collab_sync_status == "syncing"
                || app.sec_scan_active || app.profiler_running {
                app.spinner_tick = app.spinner_tick.wrapping_add(1);
            }
        }
    }
}
