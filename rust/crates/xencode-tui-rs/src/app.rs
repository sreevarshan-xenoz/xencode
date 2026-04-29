use std::collections::{HashMap, HashSet};
use std::io;
use std::process::Command;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{backend::Backend, Terminal};
use tokio::sync::mpsc;

use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderManager};

use crate::ui;

#[derive(PartialEq)]
pub enum InputMode {
    Normal,
    Editing,
}

#[derive(PartialEq)]
pub enum FocusArea {
    ChatInput,
    FileExplorer,
    ModelSelector,
    Settings,
    CodeReview,
    Terminal,
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
}

impl ThemeColors {
    pub fn get(name: &str) -> Self {
        match name {
            "midnight" => Self {
                bg: ratatui::style::Color::Rgb(15, 17, 26),
                fg: ratatui::style::Color::Rgb(230, 230, 230),
                accent: ratatui::style::Color::Magenta,
                border: ratatui::style::Color::DarkGray,
                border_active: ratatui::style::Color::Magenta,
                highlight: ratatui::style::Color::Magenta,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Magenta,
                message_assistant: ratatui::style::Color::Cyan,
                message_system: ratatui::style::Color::DarkGray,
            },
            "ocean" => Self {
                bg: ratatui::style::Color::Rgb(11, 27, 43),
                fg: ratatui::style::Color::Rgb(234, 244, 255),
                accent: ratatui::style::Color::Cyan,
                border: ratatui::style::Color::DarkGray,
                border_active: ratatui::style::Color::Cyan,
                highlight: ratatui::style::Color::Cyan,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Cyan,
                message_assistant: ratatui::style::Color::Green,
                message_system: ratatui::style::Color::DarkGray,
            },
            "forest" => Self {
                bg: ratatui::style::Color::Rgb(15, 29, 20),
                fg: ratatui::style::Color::Rgb(233, 245, 234),
                accent: ratatui::style::Color::Green,
                border: ratatui::style::Color::DarkGray,
                border_active: ratatui::style::Color::Green,
                highlight: ratatui::style::Color::Green,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Green,
                message_assistant: ratatui::style::Color::Yellow,
                message_system: ratatui::style::Color::DarkGray,
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
            },
            _ => Self { // default fallback
                bg: ratatui::style::Color::Reset,
                fg: ratatui::style::Color::Reset,
                accent: ratatui::style::Color::Cyan,
                border: ratatui::style::Color::Reset,
                border_active: ratatui::style::Color::Cyan,
                highlight: ratatui::style::Color::Cyan,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Blue,
                message_assistant: ratatui::style::Color::Green,
                message_system: ratatui::style::Color::DarkGray,
            }
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
    pub input_mode: InputMode,
    pub messages: Vec<UiMessage>,
    pub file_tree: Vec<String>,
    pub selected_file: usize,
    pub attached_files: HashSet<String>,
    pub git_status: HashMap<String, String>,
    pub available_models: Vec<String>,
    pub selected_model: usize,
    pub is_generating: bool,
    pub theme: ThemeColors,
    pub config: XencodeConfig,
    memory: ConversationMemory,
}

impl App {
    pub fn new() -> Self {
        let config = XencodeConfig::load().unwrap_or_default();
        let mut memory = ConversationMemory::with_persistence(config.max_memory_items)
            .unwrap_or_else(|_| ConversationMemory::new(50));
            
        memory.start_session(None);

        let client = OllamaClient::new(&config.ollama_url, config.response_timeout);
        // We only use the provider manager inside the spawned async task,
        // so we don't need to keep it in App state right now.

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
        let file_tree = tree.into_iter().map(|f| f.path.display().to_string()).collect();

        let available_models = vec![
            "qwen2.5:7b".to_string(),
            "llama3.1:8b".to_string(),
            "anthropic/claude-3.5-sonnet".to_string(),
            "google/gemini-1.5-pro".to_string(),
            "openai/gpt-4o".to_string(),
        ];
        
        let selected_model = available_models.iter().position(|m| m == &config.default_model).unwrap_or(0);
        let theme = ThemeColors::get(&config.active_theme);
        
        let mut git_status = HashMap::new();
        if let Ok(output) = Command::new("git").args(["status", "--porcelain"]).output() {
            if let Ok(status_str) = String::from_utf8(output.stdout) {
                for line in status_str.lines() {
                    if line.len() > 3 {
                        let status_code = &line[0..2];
                        let file_path = &line[3..];
                        // Convert to display path format matching file_tree
                        let formatted_path = format!(".\\{}", file_path.replace("/", "\\"));
                        git_status.insert(formatted_path, status_code.trim().to_string());
                    }
                }
            }
        }

        let mut app = Self {
            focus: FocusArea::ChatInput,
            input: String::new(),
            input_mode: InputMode::Normal,
            messages: Vec::new(),
            file_tree,
            selected_file: 0,
            attached_files: HashSet::new(),
            git_status,
            available_models,
            selected_model,
            is_generating: false,
            theme,
            config,
            memory,
        };

        // Load context into UI
        for msg in app.memory.get_context(10) {
            app.messages.push(UiMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }

        app
    }

    pub fn submit_message(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.input.trim().is_empty() {
            return;
        }

        let prompt = self.input.clone();
        self.input.clear();

        self.messages.push(UiMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });

        self.memory.add_message("user", &prompt, None);
        self.is_generating = true;

        // Start generation task
        let mut context_messages = Vec::new();
        for msg in self.memory.get_context(10) {
            context_messages.push(ChatMessage {
                role: msg.role,
                content: msg.content,
            });
        }

        // Inject attached files into context
        let mut attached_context = String::new();
        for path in &self.attached_files {
            if let Ok(content) = std::fs::read_to_string(path) {
                attached_context.push_str(&format!("<file path=\"{}\">\n{}\n</file>\n\n", path, content));
            }
        }
        
        if !attached_context.is_empty() {
            context_messages.insert(0, ChatMessage {
                role: "system".to_string(),
                content: format!("The following local files are attached for context:\n{}", attached_context),
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
            } else {
                self.messages.push(UiMessage {
                    role: "assistant".to_string(),
                    content: text.to_string(),
                });
            }
        } else {
            self.messages.push(UiMessage {
                role: "assistant".to_string(),
                content: text.to_string(),
            });
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn run_app<B: Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        // Handle async stream events
        if let Ok(token) = rx.try_recv() {
            app.append_generation(&token);
            continue;
        }

        // Handle input events
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.input_mode {
                        InputMode::Normal => match key.code {
                            KeyCode::Tab => {
                                app.focus = if app.focus == FocusArea::ChatInput {
                                    FocusArea::FileExplorer
                                } else {
                                    FocusArea::ChatInput
                                };
                            }
                            KeyCode::Up => {
                                if app.focus == FocusArea::FileExplorer && app.selected_file > 0 {
                                    app.selected_file -= 1;
                                } else if app.focus == FocusArea::ModelSelector && app.selected_model > 0 {
                                    app.selected_model -= 1;
                                }
                            }
                            KeyCode::Down => {
                                if app.focus == FocusArea::FileExplorer && app.selected_file + 1 < app.file_tree.len() {
                                    app.selected_file += 1;
                                } else if app.focus == FocusArea::ModelSelector && app.selected_model + 1 < app.available_models.len() {
                                    app.selected_model += 1;
                                }
                            }
                            KeyCode::Enter => {
                                if app.focus == FocusArea::FileExplorer {
                                    if let Some(file_path) = app.file_tree.get(app.selected_file) {
                                        let file_path_clone = file_path.clone();
                                        if app.attached_files.contains(&file_path_clone) {
                                            app.attached_files.remove(&file_path_clone);
                                        } else {
                                            app.attached_files.insert(file_path_clone);
                                        }
                                    }
                                } else if app.focus == FocusArea::ModelSelector {
                                    if let Some(model) = app.available_models.get(app.selected_model) {
                                        app.config.default_model = model.clone();
                                        let _ = app.config.save();
                                        app.focus = FocusArea::ChatInput;
                                    }
                                }
                            }
                            KeyCode::Char('i') => {
                                app.input_mode = InputMode::Editing;
                                app.focus = FocusArea::ChatInput;
                            }
                            KeyCode::Char('m') => {
                                app.focus = if app.focus == FocusArea::ModelSelector {
                                    FocusArea::ChatInput
                                } else {
                                    FocusArea::ModelSelector
                                };
                            }
                            KeyCode::Char('g') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                // Refresh git status
                                app.git_status.clear();
                                if let Ok(output) = Command::new("git").args(["status", "--porcelain"]).output() {
                                    if let Ok(status_str) = String::from_utf8(output.stdout) {
                                        for line in status_str.lines() {
                                            if line.len() > 3 {
                                                let status_code = &line[0..2];
                                                let file_path = &line[3..];
                                                let formatted_path = format!(".\\{}", file_path.replace("/", "\\"));
                                                app.git_status.insert(formatted_path, status_code.trim().to_string());
                                            }
                                        }
                                    }
                                }
                            }
                            KeyCode::Char(',') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                app.focus = if app.focus == FocusArea::Settings { FocusArea::ChatInput } else { FocusArea::Settings };
                            }
                            KeyCode::Char('r') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                app.focus = if app.focus == FocusArea::CodeReview { FocusArea::ChatInput } else { FocusArea::CodeReview };
                            }
                            KeyCode::Char('t') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                app.focus = if app.focus == FocusArea::Terminal { FocusArea::ChatInput } else { FocusArea::Terminal };
                            }
                            KeyCode::Char('q') | KeyCode::Esc => {
                                return Ok(());
                            }
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                return Ok(());
                            }
                            _ => {}
                        },
                        InputMode::Editing => match key.code {
                            KeyCode::Enter => {
                                if !app.is_generating {
                                    app.submit_message(tx.clone());
                                }
                            }
                            KeyCode::Char(c) => {
                                app.input.push(c);
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }
                            KeyCode::Esc => {
                                app.input_mode = InputMode::Normal;
                            }
                            _ => {}
                        },
                    }
                }
            }
        }
    }
}
