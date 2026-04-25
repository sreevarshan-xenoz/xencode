use std::io;
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
    pub is_generating: bool,
    config: XencodeConfig,
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

        let mut app = Self {
            focus: FocusArea::ChatInput,
            input: String::new(),
            input_mode: InputMode::Normal,
            messages: Vec::new(),
            file_tree,
            selected_file: 0,
            is_generating: false,
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

        let model = self.config.default_model.clone();
        let ollama_url = self.config.ollama_url.clone();
        let timeout = self.config.response_timeout;

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout);
            let manager = ProviderManager::new(client);
            
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
                                }
                            }
                            KeyCode::Down => {
                                if app.focus == FocusArea::FileExplorer && app.selected_file + 1 < app.file_tree.len() {
                                    app.selected_file += 1;
                                }
                            }
                            KeyCode::Char('i') => {
                                app.input_mode = InputMode::Editing;
                                app.focus = FocusArea::ChatInput;
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
