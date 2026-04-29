# Rust Migration Status

Branch: `total-migiration-rust`

## Current Slice: Phase 6 (Context Injection, Cloud Providers, Advanced Panels)

The Rust migration has now achieved functional superiority over the Python baseline. We have added advanced context injection, cloud provider integrations via OpenRouter, and an interactive Model Selector.

## Added/Updated Crates

- `xencode-core-rs`: Workspace scanner skipping noisy directories.
- `xencode-config-rs`: **[UPDATED]** Added `api_keys` support for `openrouter_api_key`, `google_gemini_api_key`, etc.
- `xencode-cache-rs`: LRU + TTL response cache with disk persistence.
- `xencode-models-rs`: Async HTTP Ollama interactions.
- `xencode-memory-rs`: Conversation session persistence.
- `xencode-providers-rs`: **[UPDATED]** Upgraded to support dynamic routing. Added SSE streaming support for the OpenRouter completions API.
- `xencode-tui-rs`: **[UPDATED]** Added `attached_files` state, `ModelSelector` overlay, and injected file contents into system prompts.
- `xencode-cli`: **[UPDATED]** Passes API keys down from configuration.

## Key Features Added in Phase 6

1. **Context Injection (File Attachments)**: Users can select a file in the File Explorer and press `Enter` to attach it (`[x]`). Attached files are automatically read and injected as `<file>` blocks into the LLM context when querying.
2. **Cloud Provider Integration**: The `ProviderManager` now detects if a model requires OpenRouter (e.g., contains a slash `/`) and utilizes an SSE streaming client to stream responses from models like `anthropic/claude-3.5-sonnet` and `openai/gpt-4o`.
3. **Model Selector Panel**: Pressing `m` opens an interactive overlay that lists available models (both local and cloud). Selecting a model updates the application's configuration and instantly switches the backend.

## Local Verification Notes

Launch the TUI:

```powershell
cd rust
cargo run --release -p xencode-cli -- tui
```

Within the TUI:
- **File Explorer**: Press `Tab` to focus. Use `Up`/`Down` arrows to navigate. Press `Enter` to attach/detach a file to your LLM context.
- **Model Selector**: Press `m` to open the overlay. Use `Up`/`Down` to navigate. Press `Enter` to confirm model selection.
- **Chat**: Press `i` to enter Editing mode. Press `Enter` to submit. Press `Esc` to leave. Press `q` to quit.
