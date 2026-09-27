pub mod advice;
pub mod download;
pub mod health;
pub mod llamacpp;
pub mod ollama;

pub use advice::{
    active_preference, embedded_preference, Advice, AdviceFile, AdviceSource, GgufEntry, Tier,
    ROT_HORIZON_DAYS,
};
pub use download::{
    check_model_file, discard_partial, fetch_gguf, fetch_model_file, human_bytes, partial_bytes,
    read_provenance, sha256_file, short_rev, DownloadError, Downloaded, FileCheck, Progress,
    Provenance,
};
pub use health::{current_timestamp, HealthStatus, HealthTracker, ModelHealth};
pub use llamacpp::{
    ctx_size_in, find_llama_server, launch_and_wait, resolve_gguf_model, start_llama_server,
    LaunchOutcome, LlamaCppClient, LlamaCppError, LlamaCppModelInfo, LlamaCppOptions,
    LlamaCppTimings, LlamaServerProcess, Patience, ServerExit, ServerStart,
};
pub use ollama::{ModelInfo, OllamaClient, OllamaError};
