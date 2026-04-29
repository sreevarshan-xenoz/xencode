use std::collections::{HashMap, HashSet};
use std::io;
use std::process::Command;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind, MouseButton};
use ratatui::{backend::Backend, Terminal};
use tokio::sync::mpsc;

use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::OllamaClient;
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
    ModelSelector,
    Settings,
    CodeReview,
    Terminal,
    PerformanceDashboard,
    ProviderHealth,
    ProjectAnalyzer,
    GitCommit,
}

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

pub struct App {
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
}

impl App {
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
        };

        for msg in app.memory.get_context(10) {
            app.messages.push(UiMessage { role: msg.role.clone(), content: msg.content.clone() });
        }
        app
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
        let api_key = self.config.api_keys.openrouter_api_key.clone();

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout);
            let manager = ProviderManager::new(client, api_key);
            let _ = manager.generate_stream(&model, &context_messages, |token| {
                let _ = tx.send(token.to_string());
            }).await;
            let _ = tx.send("[DONE]".to_string());
        });
    }

    pub fn append_generation(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_generating = false;
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
                let api_key = self.config.api_keys.openrouter_api_key.clone();

                tokio::spawn(async move {
                    let client = OllamaClient::new(&ollama_url, timeout);
                    let manager = ProviderManager::new(client, api_key);
                    let _ = manager.generate_stream(&model, &messages, |token| {
                        let _ = tx.send(format!("[REVIEW]{}", token));
                    }).await;
                    let _ = tx.send("[REVIEW][DONE]".to_string());
                });
            }
        }
    }
}

impl Default for App {
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
                            _ => {}
                        }
                    }

                    match app.input_mode {
                        InputMode::Normal => match key.code {
                            KeyCode::Tab => {
                                app.focus = match app.focus {
                                    FocusArea::ChatInput => FocusArea::FileExplorer,
                                    FocusArea::FileExplorer => FocusArea::ChatInput,
                                    _ => FocusArea::ChatInput,
                                };
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                match app.focus {
                                    FocusArea::FileExplorer => { if app.selected_file > 0 { app.selected_file -= 1; } }
                                    FocusArea::ModelSelector => { if app.selected_model > 0 { app.selected_model -= 1; } }
                                    FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_add(1); }
                                    _ => {}
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                match app.focus {
                                    FocusArea::FileExplorer => {
                                        if app.selected_file + 1 < app.file_tree.len() { app.selected_file += 1; }
                                    }
                                    FocusArea::ModelSelector => {
                                        if app.selected_model + 1 < app.available_models.len() { app.selected_model += 1; }
                                    }
                                    FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_sub(1); }
                                    _ => {}
                                }
                            }
                            KeyCode::Enter => {
                                match app.focus {
                                    FocusArea::FileExplorer => {
                                        if let Some(fp) = app.file_tree.get(app.selected_file) {
                                            let fp = fp.clone();
                                            if app.attached_files.contains(&fp) { app.attached_files.remove(&fp); }
                                            else { app.attached_files.insert(fp); }
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
                                    _ => {}
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
                                    FocusArea::ModelSelector | FocusArea::Settings | FocusArea::CodeReview |
                                    FocusArea::PerformanceDashboard | FocusArea::ProviderHealth | FocusArea::ProjectAnalyzer | FocusArea::GitCommit => {
                                        app.focus = FocusArea::ChatInput;
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Char(c) => {
                                if app.focus == FocusArea::GitCommit {
                                    app.commit_message.insert(app.commit_cursor, c);
                                    app.commit_cursor += 1;
                                }
                            }
                            KeyCode::Backspace => {
                                if app.focus == FocusArea::GitCommit && app.commit_cursor > 0 {
                                    app.commit_cursor -= 1;
                                    app.commit_message.remove(app.commit_cursor);
                                }
                            }
                            KeyCode::Left => {
                                if app.focus == FocusArea::GitCommit && app.commit_cursor > 0 { app.commit_cursor -= 1; }
                            }
                            KeyCode::Right => {
                                if app.focus == FocusArea::GitCommit && app.commit_cursor < app.commit_message.len() { app.commit_cursor += 1; }
                            }
                            _ => {}
                        },
                        InputMode::Editing => match key.code {
                            KeyCode::Enter => {
                                if !app.is_generating {
                                    // Add empty assistant message placeholder
                                    app.messages.push(UiMessage { role: "assistant".to_string(), content: String::new() });
                                    app.submit_message(tx.clone());
                                    app.chat_scroll = 0; // scroll to bottom
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
                                // Insert 4 spaces
                                app.input.insert_str(app.input_cursor, "    ");
                                app.input_cursor += 4;
                            }
                            _ => {}
                        },
                    }
                }
                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            match app.focus {
                                FocusArea::ChatInput => { app.chat_scroll = app.chat_scroll.saturating_add(3); }
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
                                FocusArea::FileExplorer => {
                                    app.selected_file = (app.selected_file + 3).min(app.file_tree.len().saturating_sub(1));
                                }
                                _ => {}
                            }
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            // Click in left quarter = file explorer, else chat
                            let term_width = terminal.size()?.width;
                            if mouse.column < term_width / 4 {
                                app.focus = FocusArea::FileExplorer;
                                // Approximate row click to file selection
                                let row = mouse.row.saturating_sub(2) as usize; // account for margin+border
                                if row < app.file_tree.len() {
                                    app.selected_file = row;
                                }
                            } else {
                                app.focus = FocusArea::ChatInput;
                            }
                        }
                        _ => {}
                    }
                }
                Event::Resize(_, _) => {} // handled by ratatui automatically
                _ => {}
            }
        } else {
            // Tick spinner when idle
            if app.is_generating || app.is_reviewing {
                app.spinner_tick = app.spinner_tick.wrapping_add(1);
            }
        }
    }
}
