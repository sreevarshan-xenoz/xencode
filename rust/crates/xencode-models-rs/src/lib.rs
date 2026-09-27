pub mod download;
pub mod health;
pub mod llamacpp;
pub mod ollama;

pub use download::{
    discard_partial, fetch_gguf, fetch_model_file, human_bytes, partial_bytes, DownloadError,
    Downloaded, Progress,
};
pub use health::{current_timestamp, HealthStatus, HealthTracker, ModelHealth};
pub use llamacpp::{
    ctx_size_in, find_llama_server, launch_and_wait, resolve_gguf_model, start_llama_server,
    LaunchOutcome, LlamaCppClient, LlamaCppError, LlamaCppModelInfo, LlamaCppOptions,
    LlamaCppTimings, LlamaServerProcess, Patience, ServerExit, ServerStart,
};
pub use ollama::{ModelInfo, OllamaClient, OllamaError};
