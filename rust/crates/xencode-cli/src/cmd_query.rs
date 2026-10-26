//! the query subcommand
//! `xencode query`, with its structured-answer and image helpers.

use super::*;

/// Parse `--json-schema` strictly: invalid JSON is a user error, not a
/// string to send. Pure — unit-tested.
pub(crate) fn parse_json_schema(
    schema: Option<String>,
) -> Result<Option<serde_json::Value>, String> {
    schema
        .map(|s| {
            serde_json::from_str(&s)
                .map_err(|e| format!("invalid --json-schema (must be JSON): {e}"))
        })
        .transpose()
}

/// Whether an answer may be given as one that satisfies `--json-schema`. With no
/// schema declared there is nothing to satisfy; with one, the text has to be that
/// JSON and nothing else.
pub(crate) fn structured_answer_fits(answer: &str, schema: Option<&serde_json::Value>) -> bool {
    match schema {
        None => true,
        Some(schema) => xencode_providers_rs::schema::read_answer(answer, schema).is_ok(),
    }
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to sampling options; a struct would just rename them
pub(crate) async fn run_query(
    prompt: String,
    model_override: Option<String>,
    no_cache: bool,
    session_id: Option<String>,
    temperature: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    mirostat: Option<i32>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    grammar: Option<String>,
    json_schema: Option<String>,
    images: Vec<String>,
    format: QueryFormat,
) -> Result<(), String> {
    let outcome = run_query_once(
        prompt,
        model_override,
        no_cache,
        session_id,
        temperature,
        top_k,
        min_p,
        mirostat,
        seed,
        max_tokens,
        grammar,
        json_schema,
        images,
        format,
    )
    .await;
    // One terminating line, whatever failed. Without this a script sees an
    // empty stream and cannot tell "no answer" from "still running".
    if let (Err(message), QueryFormat::Ndjson) = (&outcome, format) {
        println!("{}", query_stream::error(message));
        let _ = io::stdout().flush();
    }
    outcome
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to sampling options; a struct would just rename them
pub(crate) async fn run_query_once(
    prompt: String,
    model_override: Option<String>,
    no_cache: bool,
    session_id: Option<String>,
    temperature: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    mirostat: Option<i32>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    grammar: Option<String>,
    json_schema: Option<String>,
    images: Vec<String>,
    format: QueryFormat,
) -> Result<(), String> {
    let ndjson = format == QueryFormat::Ndjson;
    // Read before the cache is consulted or anything is printed: a schema that
    // is not JSON is the caller's mistake, and a schema that is one is a promise
    // about the answer. Same for images: an unreadable one is the caller's
    // mistake, and it is a mistake rather than a silent drop either way.
    let schema = parse_json_schema(json_schema)?;
    // Same intake the TUI's attach path uses: read, cap, verify it really is an
    // image, shrink to what a vision encoder can use, and report what changed —
    // so a re-encoded file is visible rather than quiet.
    let image_data_urls = images
        .iter()
        .map(|path| encode_query_image(path))
        .collect::<Result<Vec<_>, _>>()?;
    let started = std::time::Instant::now();
    let config = XencodeConfig::load().unwrap_or_default();
    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);

    // If no model override is provided, verify default model against Ollama's installed models
    let named_a_model = model_override.is_some();
    let model = match model_override {
        Some(m) => m,
        None => {
            if !config.default_model.contains('/') {
                if let Ok(installed) = client.list_models().await {
                    if !installed.is_empty()
                        && !installed.iter().any(|m| m.name == config.default_model)
                    {
                        // Configured default is not installed; use smart default or first installed
                        if let Ok(Some(smart)) = client.get_smart_default().await {
                            smart
                        } else if let Some(first) = installed.first() {
                            first.name.clone()
                        } else {
                            config.default_model.clone()
                        }
                    } else {
                        config.default_model.clone()
                    }
                } else {
                    config.default_model.clone()
                }
            } else {
                config.default_model.clone()
            }
        }
    };

    // A saved profile marked for the kind of work this prompt reads as takes the
    // turn (MI-7) — unless `--model` named one, which is an answer rather than a
    // question, and a rule does not overrule it. The sampling rides with the
    // profile only where a flag left it unset, for the same reason. Said on stderr
    // before anything about the request, so `-f ndjson` output stays parsable and a
    // script's `start` line still names the model that will actually answer.
    let mut model = model;
    let mut temperature = temperature;
    let mut max_tokens = max_tokens;
    if !named_a_model {
        let choice = xencode_tui_rs::task_profiles::choose_profile(&config, &model, &prompt);
        if let Some(profile) = choice.profile() {
            model = profile.model.clone();
            if temperature.is_none() {
                temperature = profile.temperature;
            }
            if max_tokens.is_none() {
                max_tokens = profile.max_tokens;
            }
        }
        if let Some(note) = choice.note() {
            eprintln!("profile: {note}");
        }
    }

    let mut cache = if config.cache_enabled && !no_cache {
        ResponseCache::with_persistence(config.max_cache_size, config.response_timeout as f64).ok()
    } else {
        None
    };

    let mut memory = if config.memory_enabled {
        ConversationMemory::with_persistence(config.max_memory_items).ok()
    } else {
        None
    };

    if let Some(ref mut mem) = memory {
        if let Some(ref sid) = session_id {
            mem.switch_session(sid);
        } else {
            mem.start_session(None);
        }
    }

    // Where the prompt is going, from the same prefix rules the router walks.
    // Reported before anything else so a stream that never finishes still says
    // which model it was waiting on.
    let routing = xencode_providers_rs::RoutingFacts {
        openrouter_key: config.api_keys.has_secret(SecretProvider::OpenRouter),
        remote_host: (!config.remote_base_url.is_empty())
            .then(|| xencode_providers_rs::url_host(&config.remote_base_url))
            .flatten(),
    };
    if ndjson {
        let session = memory.as_ref().and_then(|mem| mem.current_session());
        println!(
            "{}",
            query_stream::start(
                &model,
                xencode_providers_rs::provider_for(&model, routing),
                source_word(xencode_providers_rs::classify(&model, routing)),
                session.map(String::as_str),
            )
        );
        let _ = io::stdout().flush();
    }

    if let Some(ref mut c) = cache {
        if let Some(cached_resp) = c.get(&prompt, &model) {
            // A cached reply was written against whatever was asked the first
            // time, so it is only this run's answer if it still fits the schema
            // being declared now. If it does not, the model is asked again.
            if structured_answer_fits(&cached_resp, schema.as_ref()) {
                if ndjson {
                    // The answer travels as a `token` line too, so a script that
                    // concatenates tokens always has the whole reply — cached or
                    // not. Nothing was generated, so there are no token counts.
                    println!("{}", query_stream::token(&cached_resp));
                    println!(
                        "{}",
                        query_stream::done(&cached_resp, true, elapsed_ms(started), None, None,)
                    );
                } else {
                    println!("{}", cached_resp);
                }

                if let Some(ref mut mem) = memory {
                    mem.add_message("user", &prompt, None);
                    mem.add_message("assistant", &cached_resp, Some(model.clone()));
                }
                return Ok(());
            }
        }
    }

    // Project context injection, same engine as the TUI live path: a
    // byte-stable system head (KV-cacheable) + budgeted history turns +
    // retrieval/state/git riding in the final user turn (§10 tiers).
    // Memory is persisted after the generation below, so this snapshot holds
    // prior turns only — the current prompt arrives separately and unsqueezable.
    let history: Vec<(String, String)> = memory
        .as_ref()
        .map(|mem| {
            mem.get_context(25)
                .into_iter()
                .map(|m| (m.role, m.content))
                .collect()
        })
        .unwrap_or_default();
    let root = xencode_context_rs::default_root();
    // How much of this window project context may fill is a profile decision,
    // and the profile is no longer a constant: the config can name one, and if
    // it does not, the memory this machine reports chooses. What was chosen and
    // why is printed, because a budget that trims someone's context silently is
    // worse than a budget that says so.
    let hardware = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
    eprintln!("hardware: {}", hardware.describe());
    // A command that runs once has no earlier request to measure this turn's
    // prompt against, so it has nothing to scale retrieval from and keeps the
    // profile's own numbers. The interactive session sizes them per turn instead.
    let caps = xencode_context_rs::ContextCaps::from_profile(hardware.profile);
    eprintln!(
        "retrieval: up to {} files, {} characters each",
        caps.top_k, caps.content_cap_chars
    );
    let live = xencode_context_rs::collect_live_context(&root, &prompt, caps);
    // Which weights retrieval used, and the words in the prompt that chose them:
    // a turn read as bugfix scores differently from one read as a rename, and
    // without this line the difference looks like a random result.
    eprintln!(
        "read as {} work — {}",
        live.shape.shape,
        live.shape.reasons.join("; ")
    );
    // The window the server is actually running with, when the model is served
    // by a llama.cpp process that will say. A family table cannot know `-c`;
    // every other route keeps the table's answer.
    let probe = xencode_providers_rs::routes_to_llamacpp(&model)
        .then(|| LlamaCppClient::new(&config.llama_cpp_url, 3));
    let server_window = match &probe {
        Some(client) => client.context_window().await.unwrap_or(None),
        None => None,
    };
    if let Some(tokens) = server_window {
        eprintln!(
            "context: {tokens}-token window reported by the server at {}",
            config.llama_cpp_url
        );
    }
    // What the model itself says about the window its weights hold and whether
    // it claims it can think, asked of Ollama when this model goes there (MI-2).
    // The answer settles two numbers that have to be one number: the window
    // written into the request and the window the context below is filled for. A
    // turn budgeted past what the server was told to open is refused part-way
    // through with `exceed_context_size_error` rather than quietly truncated.
    let on_ollama = xencode_providers_rs::routes_to_ollama(&model);
    let ollama_show = if on_ollama {
        OllamaClient::new(&config.ollama_url, 3)
            .show_model(&model)
            .await
            .ok()
    } else {
        None
    };
    let mut ollama_request = xencode_providers_rs::OllamaRequest::from_settings(
        config.ollama_reasoning.as_deref(),
        config.ollama_keep_alive.as_deref(),
    )
    .map_err(|e| e.to_string())?;
    let mut context_window = xencode_providers_rs::effective_context_window(&model, server_window);
    if on_ollama {
        // What to ask for before the model gets a say: what this machine's
        // profile can serve, never the family table's advertised window.
        ollama_request.num_ctx = Some(xencode_providers_rs::ollama_window_asked(
            context_window,
            hardware.profile.ctx_tokens() as u32,
        ));
        let (decided, notes) =
            xencode_providers_rs::ollama_request_for(ollama_show.as_ref(), ollama_request);
        ollama_request = decided;
        context_window = ollama_request.num_ctx;
        if let Some(tokens) = context_window {
            eprintln!("context: asking Ollama for a {tokens}-token window");
        }
        for note in notes {
            eprintln!("context: {note}");
        }
        // Two controls Ollama's `/api/chat` has nowhere to put. Asking for one and
        // receiving an ordinary answer would be a silence nobody could find except
        // in the source, so each is said as it is dropped.
        if grammar.is_some() {
            eprintln!(
                "sampling: --grammar was not sent — Ollama takes a JSON schema in `format`, not a GBNF grammar"
            );
        }
        if mirostat.is_some() {
            eprintln!(
                "sampling: --mirostat was not sent — a running Ollama answers it as `invalid option provided`"
            );
        }
    }
    let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
        profile: hardware.profile,
        context_window,
        system: xencode_context_rs::prompts::AGENT_SYSTEM,
        agents_md: live.agents_md.as_deref(),
        anchor_md: live.anchor_md.as_deref(),
        scoped_md: live.scoped_md.as_deref(),
        state_md: live.state_md.as_deref(),
        notes_md: live.notes_md.as_deref(),
        git_summary: &live.git_summary,
        repo_map: &live.repo_map,
        retrieved: live.blocks,
        attached_block: "",
        history: &history,
        prompt: &prompt,
    });
    // What the prompt costs in the model's own vocabulary, asked of the server
    // once before the request goes out — the only moment a count exists at all,
    // since a reply's usage figures arrive after it has been paid for. The
    // budgeter's figure is printed beside it, and neither is a better number than
    // the other: a plain `/tokenize` of the whole prompt leaves out the framing a
    // chat template adds around each message — 13 tokens for the two-message
    // request measured here, out of 5,753 — while the budgeter's figure stops at
    // what it was allowed to spend, because the question and any attached files
    // are the parts it may not trim. Printed together, a run that is about to
    // overflow shows it. Measured on a 512-token server here: the arithmetic said
    // 384, the server counted 566 of the same turn, and the request was refused at
    // 579 tokens.
    let counted = match &probe {
        Some(client) => Some(
            client
                .count_tokens(&assembly.prompt_text())
                .await
                .map_err(|e| e.to_string()),
        ),
        None => None,
    };
    match counted {
        Some(Ok(Some(tokens))) => {
            eprintln!(
                "context: {tokens} tokens counted by the server, {} by character arithmetic",
                assembly.total_tokens
            );
            if let Some(window) = server_window {
                if tokens > window as u64 {
                    eprintln!(
                        "warning: the prompt was counted at {tokens} tokens, which is more than the {window}-token window this server is running with"
                    );
                }
            }
        }
        Some(Ok(None)) => {
            // The server answered the count request and refused it — a build
            // without /tokenize, or one that rejected this text.
            eprintln!(
                "context: {} tokens by character arithmetic (this server gave no count for /tokenize)",
                assembly.total_tokens
            );
        }
        Some(Err(reason)) => {
            // Nothing was ever counted, so the arithmetic line is not "the server
            // declined" — it is the only number there is because the ask failed.
            eprintln!(
                "context: {} tokens by character arithmetic (no count could be asked of this server: {reason})",
                assembly.total_tokens
            );
        }
        None => {}
    }
    if !live.index_present {
        eprintln!(
            "hint: run `xencode` → /init once for project-aware answers (continuing with guidelines + history only)."
        );
    }
    let mut context_messages: Vec<ChatMessage> = assembly
        .turns
        .into_iter()
        .map(|t| ChatMessage {
            role: t.role,
            content: t.content.into(),
        })
        .collect();
    // Images ride as message parts on the final user turn, never inlined into
    // the prompt text — same shape the TUI's attach path builds.
    if !image_data_urls.is_empty()
        && !attach_images_to_final_user_message(&mut context_messages, image_data_urls)
    {
        return Err(
            "the attached images could not be sent with this turn (no final user message) \
             — they were dropped, not seen by the model"
                .to_string(),
        );
    }

    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);

    let llamacpp_opts = LlamaCppOptions {
        temperature,
        top_k,
        min_p,
        mirostat,
        max_tokens,
        seed,
        grammar,
        json_schema: schema.clone(),
    };

    let provider = ProviderManager::new(
        client,
        api_key(&config, SecretProvider::OpenRouter),
        api_key(&config, SecretProvider::Qwen),
        api_key(&config, SecretProvider::Gemini),
        None,
    )
    .with_llama_cpp(llama_client)
    .with_remote(
        &config.remote_base_url,
        api_key(&config, SecretProvider::Remote),
    )
    .with_nvidia(api_key(&config, SecretProvider::Nvidia))
    .with_request_timeout(config.response_timeout)
    .with_ollama_request(ollama_request)
    .with_egress_policy(EgressPolicy::new(config.allow_cloud_models));

    let mut response_content = String::new();
    let result = provider
        .generate_stream_with_options(&model, &context_messages, Some(&llamacpp_opts), |token| {
            if ndjson {
                println!("{}", query_stream::token(token));
            } else {
                print!("{}", token);
            }
            let _ = io::stdout().flush();
            response_content.push_str(token);
        })
        .await;

    match result {
        Ok(_) => {
            // The tokens have already streamed, so this cannot unsay them; what
            // it can do is end the stream with a failure instead of a completion,
            // and keep an answer that does not fit out of the cache and the
            // conversation.
            if let Some(ref schema) = schema {
                if let Err(reason) =
                    xencode_providers_rs::schema::read_answer(&response_content, schema)
                {
                    return Err(format!("the answer does not fit --json-schema: {reason}"));
                }
            }
            let timings = provider.last_llamacpp_timings();
            if ndjson {
                println!(
                    "{}",
                    query_stream::done(
                        &response_content,
                        false,
                        elapsed_ms(started),
                        timings.as_ref().map(|t| t.tokens_generated),
                        timings.as_ref().map(|t| t.predicted_per_second),
                    )
                );
                let _ = io::stdout().flush();
            } else {
                println!(); // Ensure final newline
                if let Some(timings) = timings {
                    println!(
                        "\n(llama.cpp ~{} tok/s · {} tokens generated)",
                        timings.predicted_per_second as u64, timings.tokens_generated
                    );
                }
            }
            if let Some(ref mut c) = cache {
                c.set(&prompt, &model, &response_content);
            }
            if let Some(ref mut mem) = memory {
                mem.add_message("user", &prompt, None);
                mem.add_message("assistant", &response_content, Some(model));
            }
            Ok(())
        }
        Err(e) => Err(format!("Query failed: {}", e)),
    }
}

/// Fold image data URLs into the final user turn as content parts, keeping the
/// assembled prompt text ahead of them. Returns false when there is no user turn
/// to attach to, which the caller reports rather than sending a request the
/// model would answer without ever seeing the images.
pub(crate) fn attach_images_to_final_user_message(
    messages: &mut [ChatMessage],
    urls: Vec<String>,
) -> bool {
    let Some(last) = messages.iter_mut().rev().find(|m| m.role == "user") else {
        return false;
    };
    let text = last.text_content();
    let mut parts = Vec::with_capacity(urls.len() + 1);
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    parts.extend(urls.into_iter().map(|url| ContentPart::ImageUrl {
        image_url: ImageUrlPart { url, detail: None },
    }));
    last.content = MessageContent::Parts(parts);
    true
}

/// Read one `--image` path into the base64 data URL a vision request carries,
/// applying the same size cap, format check, and downscale the TUI's attach
/// path does. Errors name the file so a mistyped path is obvious.
pub(crate) fn encode_query_image(path: &str) -> Result<String, String> {
    use xencode_analysis_rs::{
        inspect_bytes, prepare_for_send, to_data_url, ImageError, MAX_IMAGE_BYTES,
    };
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read image {path}: {e}"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "{path} exceeds the {} MiB image cap",
            MAX_IMAGE_BYTES / 1024 / 1024
        ));
    }
    inspect_bytes(path, &bytes).map_err(|e| match e {
        ImageError::UnknownFormat(_) => format!("{path} is not a recognized image"),
        other => format!("{path}: {other}"),
    })?;
    let prepared = prepare_for_send(&bytes);
    if let Some(note) = prepared.summary() {
        eprintln!("image: {path} ({note})");
    }
    Ok(to_data_url(prepared.format, &prepared.bytes))
}

/// Milliseconds since this command started, including config load and context
/// build.
pub(crate) fn elapsed_ms(started: std::time::Instant) -> u64 {
    (started.elapsed().as_secs_f64() * 1_000.0) as u64
}

/// The destination word the stream writes. Kept as one function so a test can
/// pin it against the spelling the metrics rows already use.
pub(crate) fn source_word(egress: xencode_providers_rs::Egress) -> &'static str {
    match egress {
        xencode_providers_rs::Egress::Local => "local",
        xencode_providers_rs::Egress::Cloud => "cloud",
    }
}
