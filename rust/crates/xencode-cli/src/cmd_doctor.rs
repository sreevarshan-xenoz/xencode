//! the toolchain, doctor, selfcheck, bug-report, session, verify and envcheck subcommands
//! `xencode doctor` and the other machine and setup checks.

use super::*;

/// The `AR-1` interop probe.
///
/// The scratch fixture is built in a temporary directory and removed afterwards,
/// so running the probe never leaves anything in the workspace and never touches
/// a file the operator cares about.
pub(crate) fn run_toolchain(
    action: &str,
    allow_dirty: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::toolchain as kit;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = kit::manifest_dir(&root)?;
    let as_json = matches!(format, OutputFormat::Json);

    match action {
        "lint" => {
            let report = kit::clippy_report(&manifest)?;
            if as_json {
                let out = serde_json::json!({
                    "command": report.command,
                    "count": report.count(),
                    "by_lint": report.by_lint().iter().map(|(l, n)| serde_json::json!({
                        "lint": l, "count": n,
                    })).collect::<Vec<_>>(),
                    "diagnostics": report.diagnostics.iter().map(|d| serde_json::json!({
                        "lint": d.lint, "level": d.level, "message": d.message,
                        "file": d.file, "line": d.line, "suggestion": d.suggestion,
                    })).collect::<Vec<_>>(),
                    "notes": report.notes,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
            } else {
                println!("\n  {}", report.summary());
                for d in &report.diagnostics {
                    let at = match (&d.file, d.line) {
                        (Some(f), Some(l)) => format!("{f}:{l}"),
                        (Some(f), None) => f.clone(),
                        _ => "(no location)".to_string(),
                    };
                    let fixable = if d.suggestion {
                        " [machine-fixable]"
                    } else {
                        ""
                    };
                    println!("\n    {}{} — {} ({})", d.lint, fixable, d.message, at);
                }
                for note in &report.notes {
                    println!("\n  note: {note}");
                }
            }
            if report.count() == 0 {
                Ok(())
            } else {
                Err(format!("{} clippy diagnostic(s)", report.count()))
            }
        }
        "fix" => {
            let outcome = kit::cargo_fix(&manifest, &root, allow_dirty)?;
            if as_json {
                let out = serde_json::json!({
                    "command": outcome.command,
                    "changed": outcome.changed,
                    "diffstat": outcome.diffstat,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
            } else if outcome.changed {
                println!("\n  fix changed files:\n\n{}", outcome.diffstat);
            } else {
                println!("\n  fix changed nothing");
            }
            Ok(())
        }
        "fmt" => {
            let (clean, report) = kit::fmt_check_output(&manifest)?;
            if as_json {
                println!("{}", serde_json::json!({ "clean": clean }));
            } else if clean {
                println!("\n  formatting is clean");
            } else {
                println!("\n  formatting differs — run `cargo fmt`");
                if !report.is_empty() {
                    println!();
                    for line in report.lines().take(40) {
                        println!("    {line}");
                    }
                }
            }
            if clean {
                Ok(())
            } else {
                Err("formatting differs".to_string())
            }
        }
        "shear" => {
            let report = kit::shear(&manifest);
            if as_json {
                println!(
                    "{}",
                    serde_json::json!({ "clean": report.clean, "lines": report.lines })
                );
            } else if report.clean {
                println!("\n  no unused dependencies reported");
            } else {
                println!("\n  shear says:");
                for line in &report.lines {
                    println!("    {line}");
                }
            }
            for line in &report.lines {
                if line.contains("not installed") {
                    return Err("cargo-shear is not installed".to_string());
                }
            }
            if report.clean {
                Ok(())
            } else {
                Err("unused dependencies reported".to_string())
            }
        }
        other => Err(format!(
            "unknown toolchain action {other:?} — want lint, fix, fmt, or shear"
        )),
    }
}

pub(crate) async fn run_doctor(
    env: bool,
    deps: bool,
    selfcheck: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::doctor;

    if !env && !deps && !selfcheck {
        // Bare `xencode doctor` is the bug report: every check the other flags
        // ask about, plus the state files, permissions and free space a person
        // pastes into an issue. It is the superset, not a fourth thing.
        return run_bug_report(format).await;
    }
    if selfcheck {
        return run_selfcheck(format).await;
    }
    if deps {
        return run_doctor_deps(format);
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let facts = doctor::probe_env();

    // Colab route presence: a state file with a live forward pid. Absent
    // state is not an error — most machines have never run `colab up`.
    let (colab_state, colab_alive) = match xencode_colab_rs::ColabState::load() {
        Ok(Some(state)) => {
            let alive = state.forward_pid.is_some_and(xencode_colab_rs::pid_alive);
            (true, alive)
        }
        _ => (false, false),
    };

    // U-3's W11 row: the configuration drift result rides along as one row,
    // so `doctor --json` is the single shape DB-6 reads.
    let refs = xencode_analysis_rs::envdrift::extract_env_refs(&root);
    let templates = xencode_analysis_rs::envdrift::read_templates(&root);
    let drift = xencode_analysis_rs::envdrift::compare(&refs, &templates);

    // QK-4: a durable fact the code has contradicted leaves the prompt without a
    // word, which is the right thing for a model and nothing but confusion for the
    // person who wrote it. The same pass, reported. It rewrites nothing — a fact
    // stale on this checkout is often true again one branch over.
    let stale =
        xencode_context_rs::audit_durable_facts(&root.join(xencode_context_rs::XENCODE_DIR));
    let believed = xencode_context_rs::ContextState::from_markdown(&stale.text);
    let believed = believed.completed.len() + believed.decisions.len() + believed.unresolved.len();

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "machine": facts,
                "colab": {"state": colab_state, "forward_alive": colab_alive},
                "config_drift": {
                    "sources_searched": drift.sources_searched,
                    "undocumented": drift.undocumented.len(),
                    "unreferenced": drift.unreferenced,
                    "os_provided": drift.os_provided.len(),
                    "panicking": drift.panicking.len(),
                },
                "durable_facts": {
                    "believed": believed,
                    "unverifiable": stale.unverifiable,
                    "dropped": stale.dropped.iter().map(|fact| serde_json::json!({
                        "line": fact.line,
                        "reason": fact.problem.reason(),
                    })).collect::<Vec<_>>(),
                    "disagreeing": stale.disagreeing.iter().map(|odds| serde_json::json!({
                        "line": odds.line,
                        "name": odds.name,
                        "cited": odds.cited,
                        "declared_in": odds.declared_in,
                    })).collect::<Vec<_>>(),
                },
            })
        );
    } else {
        println!("\n  machine:");
        println!(
            "    cores: {}  mem_available: {}  psi: {}  cgroup_limit: {}",
            facts.nproc.map_or("?".to_string(), |n| n.to_string()),
            facts
                .mem_available_kib
                .map_or("?".to_string(), |k| format!("{k} KiB")),
            if facts.psi_readable {
                "readable"
            } else {
                "absent"
            },
            facts.cgroup_memory_limit.as_deref().unwrap_or("none"),
        );
        if facts.nvidia_gpus.is_empty() {
            println!("    gpus: none visible");
        } else {
            for gpu in &facts.nvidia_gpus {
                println!("    gpu: {gpu}");
            }
        }
        println!(
            "    journalctl: {}  dmesg_denied: {}",
            if facts.journalctl_readable {
                "readable"
            } else {
                "unreadable"
            },
            facts.dmesg_denied
        );
        println!(
            "\n  colab route: {}",
            match (colab_state, colab_alive) {
                (true, true) => "live forward".to_string(),
                (true, false) => "state file, no live forward".to_string(),
                _ => "none".to_string(),
            }
        );
        println!(
            "\n  config drift: {} undocumented, {} unreferenced, {} os-provided, {} panicking",
            drift.undocumented.len(),
            drift.unreferenced.len(),
            drift.os_provided.len(),
            drift.panicking.len(),
        );
        println!(
            "\n  durable facts: {} reaching the model, {} dropped, {} that could not be checked, {} that the code places in another file",
            believed,
            stale.dropped.len(),
            stale.unverifiable,
            stale.disagreeing.len()
        );
        // The reasons are the item: a count alone would say something was taken and
        // leave the person hunting for which line and why.
        for fact in stale.dropped.iter().take(10) {
            println!(
                "    dropped — {}: {}",
                fact.problem.reason(),
                xencode_context_rs::fact_prose(&fact.line)
            );
        }
        if stale.dropped.len() > 10 {
            println!("    … and {} more", stale.dropped.len() - 10);
        }
        // Not dropped, and not resolved either: the file the fact cites is untouched
        // and the name it was written against is still declared, in some other file.
        // Which one the fact meant is a question for whoever wrote it.
        for odds in stale.disagreeing.iter().take(10) {
            println!(
                "    disagrees — {} names {}, declared in {} rather than the cited {}",
                xencode_context_rs::fact_prose(&odds.line),
                odds.name,
                odds.declared_in.join(", "),
                odds.cited
            );
        }
        if stale.disagreeing.len() > 10 {
            println!("    … and {} more", stale.disagreeing.len() - 10);
        }
    }
    Ok(())
}

pub(crate) fn run_doctor_deps(format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::deps;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = xencode_context_rs::verify::manifest_dir(&root)?;
    let corpus = xencode_config_rs::paths::cache_dir()
        .map(|dir| xencode_analysis_rs::advisories::corpus_dir(&dir))
        .ok();
    let report = deps::doctor_deps(&manifest, corpus.as_deref())?;
    let rows = &report.rows;

    let state_word = |row: &deps::DepRow| match &row.state {
        deps::DepState::Clean => "clean",
        deps::DepState::Vulnerable(_) => "VULNERABLE",
        deps::DepState::Unknown(_) => "unknown",
    };
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "dependencies": rows.iter().map(|r| serde_json::json!({
                    "crate": r.krate,
                    "locked": r.locked,
                    "update_to": r.update_to,
                    "state": state_word(r),
                    "detail": match &r.state {
                        deps::DepState::Vulnerable(ids) => ids.join(","),
                        deps::DepState::Unknown(why) => why.clone(),
                        deps::DepState::Clean => String::new(),
                    },
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        let (mut vuln, mut unknown) = (0, 0);
        for row in rows {
            match &row.state {
                deps::DepState::Vulnerable(ids) => {
                    vuln += 1;
                    println!(
                        "\n  VULNERABLE {} {}: {}",
                        row.krate,
                        row.locked.as_deref().unwrap_or("?"),
                        ids.join(", ")
                    );
                }
                deps::DepState::Unknown(why) => {
                    unknown += 1;
                    println!("\n  unknown {}: {}", row.krate, why);
                }
                deps::DepState::Clean => {}
            }
            if let Some(to) = &row.update_to {
                println!(
                    "    update available: {} -> {to}",
                    row.locked.as_deref().unwrap_or("?")
                );
            }
        }
        println!(
            "\n  {} checked, {vuln} vulnerable, {unknown} unknown",
            rows.len()
        );
        if !report.updates_checked {
            println!("  update check did not run — the update column is unknown, not clean");
        }
        let lock_text = std::fs::read_to_string(manifest.join("Cargo.lock")).unwrap_or_default();
        let dups = deps::duplicate_versions(&lock_text);
        if !dups.is_empty() {
            println!("\n  duplicate majors (ask, never block):");
            let rev = deps::reverse_deps(&lock_text);
            for dup in dups.iter().take(10) {
                println!("    {}: {}", dup.krate, dup.versions.join(", "));
                for version in &dup.versions {
                    if let Some(owners) = rev.get(&(dup.krate.clone(), version.clone())) {
                        let shown: Vec<&str> = owners.iter().take(3).map(String::as_str).collect();
                        let more = if owners.len() > 3 {
                            format!(" +{} more", owners.len() - 3)
                        } else {
                            String::new()
                        };
                        println!("      {version} via {}{more}", shown.join(", "));
                    }
                }
            }
            if dups.len() > 10 {
                println!("    +{} more", dups.len() - 10);
            }
        }
        match deps::head_lock_text(&manifest) {
            Some((_, old)) => {
                let delta = deps::lock_delta(&old, &lock_text);
                if !delta.added.is_empty() || !delta.upgraded.is_empty() {
                    println!("\n  since HEAD (new vs already present):");
                    for (name, from, to) in &delta.upgraded {
                        println!("    {name}: {from} -> {to}");
                    }
                    for (name, version) in &delta.added {
                        println!("    + {name} {version}");
                    }
                }
            }
            None => println!("\n  no HEAD lock to compare: new-vs-present unknown"),
        }
    }
    if rows
        .iter()
        .any(|r| matches!(r.state, deps::DepState::Vulnerable(_)))
    {
        Err("vulnerable dependencies found".to_string())
    } else {
        Ok(())
    }
}

/// One address the configuration says a model comes from, and what would start
/// it if nothing answers. A cloud provider is the same shape with no start
/// command: the server belongs to someone else.
pub(crate) struct Endpoint {
    pub(crate) name: &'static str,
    pub(crate) url: String,
    pub(crate) start_command: Option<&'static str>,
}

/// Every endpoint this configuration would dial, in a fixed order so two runs of
/// the same report line up.
pub(crate) fn endpoints(config: &xencode_config_rs::XencodeConfig) -> Vec<Endpoint> {
    let mut list = vec![
        Endpoint {
            name: "ollama",
            url: config.ollama_url.clone(),
            start_command: Some("ollama serve"),
        },
        Endpoint {
            name: "llamacpp",
            url: config.llama_cpp_url.clone(),
            start_command: Some("llama-server --model <path>"),
        },
    ];
    if !config.remote_base_url.trim().is_empty() {
        list.push(Endpoint {
            name: "remote",
            url: config.remote_base_url.clone(),
            start_command: None,
        });
    }

    // A cloud provider is only worth dialling when a key is configured —
    // reaching an endpoint nobody set up proves nothing about this machine.
    // Env-resolved: a key living in NVIDIA_NIM_API_KEY counts as configured
    // here too, since the route would use it. Presence is asked the cheap way: a
    // cloud endpoint is worth dialling when a credential exists, whether it is
    // stored, stored as a command reference, or exported.
    for (provider, host) in [
        (SecretProvider::OpenAi, "api.openai.com"),
        (SecretProvider::OpenRouter, "openrouter.ai"),
        (SecretProvider::Gemini, "generativelanguage.googleapis.com"),
        (SecretProvider::Qwen, "chat.qwen.ai"),
        (SecretProvider::Nvidia, "integrate.api.nvidia.com"),
    ] {
        if config.api_keys.has_secret(provider) {
            list.push(Endpoint {
                name: provider.slug(),
                url: format!("https://{host}"),
                start_command: None,
            });
        }
    }
    list
}

/// Providers are dialled at the address the config points them at. This machine's
/// llama.cpp server is not on the port a remembered default would guess, so a
/// check that hardcoded an address proved nothing about the route the model
/// would actually take.
pub(crate) fn check_endpoints(config: &xencode_config_rs::XencodeConfig) -> Vec<doc::SelfCheck> {
    use std::time::Duration;
    endpoints(config)
        .into_iter()
        .map(|endpoint| match xencode_mcp_rs::address_of(&endpoint.url) {
            Some((host, port)) => doc::check_provider(
                endpoint.name,
                &host,
                port,
                Duration::from_secs(2),
                endpoint.start_command,
            ),
            None => doc::SelfCheck {
                name: format!("provider:{}", endpoint.name),
                state: "fail".to_string(),
                detail: format!("{} is not an http or https address to dial", endpoint.url),
                fix: Some(format!(
                    "xencode config set {}_url <http-or-https-address>",
                    endpoint.name
                )),
                raw: None,
            },
        })
        .collect()
}

/// The command that would actually work for each way Ollama can refuse the
/// default model. A model id carrying the `ollama:` prefix is something this
/// configuration can hold and Ollama cannot serve — it names its models without
/// a provider prefix — so the row says which name to use rather than telling
/// the reader to pull a model that will never exist.
pub(crate) fn ollama_fix(error: &xencode_models_rs::OllamaError, model: &str) -> String {
    match error {
        xencode_models_rs::OllamaError::ModelNotFound(name) => match name.strip_prefix("ollama:") {
            Some(bare) => format!(
                "Ollama names its models without a provider prefix — set the default to \
                 `{bare}` or `ollama pull {bare}`"
            ),
            None => format!("ollama pull {name}"),
        },
        xencode_models_rs::OllamaError::NotRunning(_) => "ollama serve".to_string(),
        _ => format!("xencode model health {model}"),
    }
}

/// Does the server that would serve the default model actually know it by name.
/// This is the question `xencode model health` asks, asked of the model every
/// turn starts on — and only of a server on this machine, because the answer for
/// a model that lives with a service is that service's provider row.
pub(crate) async fn model_check(config: &xencode_config_rs::XencodeConfig) -> doc::SelfCheck {
    let model = config.default_model.trim().to_string();
    if model.is_empty() {
        return doc::check_model(
            false,
            "no default model is configured".to_string(),
            Some("xencode config set default_model <name>".to_string()),
        );
    }
    // The same prefix chain the router walks, so the report asks the server that
    // a real turn would contact rather than the one a name looks like.
    let routing = xencode_providers_rs::RoutingFacts {
        openrouter_key: config.api_keys.has_secret(SecretProvider::OpenRouter),
        remote_host: (!config.remote_base_url.is_empty())
            .then(|| xencode_providers_rs::url_host(&config.remote_base_url))
            .flatten(),
    };
    match xencode_providers_rs::provider_for(&model, routing) {
        "llamacpp" => {
            let client = LlamaCppClient::new(&config.llama_cpp_url, 4);
            if client.model_ready().await {
                doc::check_model(
                    true,
                    format!(
                        "{model}: llama-server at {} has a model loaded",
                        config.llama_cpp_url
                    ),
                    None,
                )
            } else {
                doc::check_model(
                    false,
                    format!(
                        "llama-server at {} answers but has no model loaded",
                        config.llama_cpp_url
                    ),
                    Some("xencode llamacpp load <gguf-path>".to_string()),
                )
            }
        }
        "ollama" => {
            let client = OllamaClient::new(&config.ollama_url, 4);
            match client.show_model(&model).await {
                Ok(show) => {
                    let context = show
                        .trained_context_tokens
                        .map(|n| format!("{n} tokens"))
                        .unwrap_or_else(|| "no context length reported".to_string());
                    let capabilities = if show.capabilities.is_empty() {
                        "nothing it declared".to_string()
                    } else {
                        show.capabilities.join(", ")
                    };
                    doc::check_model(
                        true,
                        format!("{model} is in Ollama's list: {context}, can do {capabilities}"),
                        None,
                    )
                }
                Err(error) => doc::check_model(
                    false,
                    format!("Ollama at {}: {error}", config.ollama_url),
                    Some(ollama_fix(&error, &model)),
                )
                .with_raw(error.raw_message()),
            }
        }
        other => doc::SelfCheck {
            name: "model".to_string(),
            state: "absent".to_string(),
            detail: format!(
                "{model} is served by {other}; provider:{other} is the row that dials it"
            ),
            fix: None,
            raw: None,
        },
    }
}

/// The rows every `doctor` surface shares: the project's own state, the
/// configured endpoints, the default model, and each declared MCP server.
pub(crate) async fn spine_checks(
    root: &std::path::Path,
    config: &xencode_config_rs::XencodeConfig,
) -> Vec<doc::SelfCheck> {
    let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
    let mut checks = vec![
        doc::check_index(&xencode_dir),
        doc::check_git(root),
        doc::check_metrics(&xencode_dir),
        doc::check_cache_writable(&xencode_dir),
    ];
    checks.extend(check_endpoints(config));
    checks.push(model_check(config).await);

    // An MCP server is checked by the client the TUI uses: it is started, asked
    // to introduce itself, and killed. A pass is a completed handshake, and a
    // failure is the client's own sentence about why there was not one.
    for (name, server) in &config.mcp_servers {
        let spec = match xencode_tui_rs::mcp::spec_from_config(name, server) {
            Ok(spec) => spec,
            // A declaration that does not say how the server is reached, or says
            // both ways at once, is the named problem — there is nothing to try.
            Err(problem) => {
                checks.push(doc::check_mcp(
                    name,
                    false,
                    format!("the declaration {problem}"),
                    Some(format!(
                        "give the server `{name}` either a command or a url in the configuration"
                    )),
                ));
                continue;
            }
        };
        let fix = server.command.as_ref().map(|command| {
            format!("run `{command}` in a shell and check it answers, or remove `{name}`")
        });
        match xencode_mcp_rs::McpClient::start(&spec, std::time::Duration::from_secs(5)).await {
            Ok(client) => {
                let endpoint = client.endpoint();
                let declared = client.capabilities().declared();
                client.shutdown().await;
                checks.push(doc::check_mcp(
                    name,
                    true,
                    format!(
                        "{endpoint} started and answered the handshake; it offers {}",
                        if declared.is_empty() {
                            "nothing it declared".to_string()
                        } else {
                            declared.join(", ")
                        }
                    ),
                    None,
                ));
            }
            Err(error) => checks.push(doc::check_mcp(name, false, error.to_string(), fix)),
        }
    }
    checks
}

/// The preflight's rows, restated as report rows. The preflight names its own
/// checks "colab CLI" and "Colab API"; inside this report they are one group, so
/// the word is said once.
pub(crate) fn bridge_rows(report: xencode_colab_rs::PreflightReport) -> Vec<doc::SelfCheck> {
    report
        .checks
        .into_iter()
        .map(|check| {
            let short = check
                .name
                .strip_prefix("colab ")
                .or_else(|| check.name.strip_prefix("Colab "))
                .unwrap_or(check.name);
            doc::SelfCheck {
                name: format!("colab:{short}"),
                state: if check.ok { "pass" } else { "fail" }.to_string(),
                detail: check.detail,
                fix: check.fix,
                raw: None,
            }
        })
        .collect()
}

/// The Colab bridge, reported by the gate `xencode colab up` runs through. It is
/// delegated rather than re-implemented: that preflight owns the version floor
/// and the keypair, and a second copy of those judgements is a second truth. It
/// is only asked when the bridge is real on this machine, because its probes
/// reach the Google backend and a machine that never ran `colab up` has nothing
/// to report.
pub(crate) async fn colab_checks() -> Vec<doc::SelfCheck> {
    let installed = xencode_colab_rs::which("colab").is_some();
    let state_file = matches!(xencode_colab_rs::ColabState::load(), Ok(Some(_)));
    let keypair = xencode_colab_rs::preflight::key_paths()
        .map(|(private, _)| private.exists())
        .unwrap_or(false);
    if !(installed || state_file || keypair) {
        return vec![doc::SelfCheck {
            name: "colab".to_string(),
            state: "absent".to_string(),
            detail: "the Colab bridge is not installed here and has never been brought up"
                .to_string(),
            fix: None,
            raw: None,
        }];
    }
    // Never `generate_key`: a report reads the machine, it does not create
    // secret material on it.
    match xencode_colab_rs::preflight(false).await {
        Ok(report) => bridge_rows(report),
        Err(problem) => vec![doc::SelfCheck {
            name: "colab".to_string(),
            state: "fail".to_string(),
            detail: format!("the bridge could not be probed: {problem}"),
            fix: Some("xencode colab preflight".to_string()),
            raw: None,
        }],
    }
}

/// The configuration and the words for how it went, as one pair: the report and
/// the defaults it falls back to cannot then disagree about what was loaded.
pub(crate) fn load_config() -> (xencode_config_rs::XencodeConfig, doc::ConfigRead) {
    let Ok(path) = xencode_config_rs::XencodeConfig::config_path() else {
        return (
            xencode_config_rs::XencodeConfig::default(),
            doc::ConfigRead::Unparseable("no home directory to hold the config".to_string()),
        );
    };
    let outcome = xencode_config_rs::XencodeConfig::load();
    let read = match &outcome {
        Ok(_) if path.is_file() => doc::ConfigRead::Loaded,
        Ok(_) => doc::ConfigRead::Absent,
        Err(xencode_config_rs::ConfigError::NewerFile { found, known, .. }) => {
            doc::ConfigRead::TooNew {
                found: *found,
                known: *known,
            }
        }
        Err(xencode_config_rs::ConfigError::NotAConfig { found, .. }) => {
            doc::ConfigRead::Unparseable(format!(
                "it holds a JSON {found}, where an object of settings was expected"
            ))
        }
        // The loader's own sentence names the file, and the doctor row prefixes
        // it again, so this part takes the file out of what it says.
        Err(xencode_config_rs::ConfigError::Corrupt { problem, .. }) => {
            doc::ConfigRead::Unparseable(format!("it is not readable JSON — {problem}"))
        }
        Err(problem) => doc::ConfigRead::Unparseable(problem.to_string()),
    };
    (outcome.unwrap_or_default(), read)
}

/// The credential a provider is to use, read through the tiers: the value stored
/// in `config.json` (running it, when what is stored is a `command:` reference),
/// otherwise the variable named for that provider.
///
/// A reference that fails is said out loud. A keyring helper that is not on
/// `PATH` looks exactly like a provider nobody configured from the outside, and
/// the person would then go and paste the key in again.
pub(crate) fn api_key(config: &XencodeConfig, provider: SecretProvider) -> Option<String> {
    match config.api_keys.secret(provider) {
        Ok(value) => value,
        Err(problem) => {
            eprintln!(
                "xencode: the {} key is unusable — {problem}",
                provider.slug()
            );
            None
        }
    }
}

/// `xencode doctor --selfcheck`: the slice a person runs when xencode itself
/// looks broken.
pub(crate) async fn run_selfcheck(format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let (config, _) = load_config();
    let checks = spine_checks(&root, &config).await;
    render_checks("selfcheck", &checks, format)
}

/// `xencode doctor` with no flag: the whole bug report. Everything the slice
/// checks, plus the state files this install owns — does the configuration
/// parse, can its secrets be read by anyone else, is there room on the volume,
/// how much disk the response cache has taken, and whether the Colab bridge is
/// usable. The JSON is the report; the text is a rendering of the same rows.
pub(crate) async fn run_bug_report(format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let (config, read) = load_config();
    // Three directories, because three kinds of file live in them: the settings
    // this report is about, the records the free-space row cares about, and the
    // cache whose size is worth naming. Asking only for the settings directory
    // and guessing the rest with `join("cache")` is how a report ends up
    // measuring a directory that holds nothing.
    let settings_dir = xencode_config_rs::paths::settings_dir().ok();
    let state_dir = xencode_config_rs::paths::state_dir().ok();
    let cache_dir = xencode_config_rs::paths::cache_dir().ok();

    let mut checks = Vec::new();
    let config_path = settings_dir
        .as_ref()
        .map(|dir| dir.join("config.json"))
        .unwrap_or_default();
    checks.push(doc::check_config(&config_path, read));
    let still_old: Vec<String> = xencode_config_rs::paths::locations()
        .iter()
        .filter(|place| place.legacy)
        .map(|place| place.kind.label().to_string())
        .collect();
    checks.push(doc::check_layout(
        &still_old,
        xencode_config_rs::paths::override_root().as_deref(),
    ));
    if let Some(dir) = settings_dir.as_ref() {
        checks.push(doc::check_config_version(
            &config_path,
            xencode_config_rs::XencodeConfig::version_of(&config_path),
            xencode_config_rs::CURRENT_CONFIG_VERSION,
        ));
        checks.push(doc::check_permissions(
            "config",
            &dir.join("config.json"),
            doc::file_mode(&dir.join("config.json")),
        ));
        let (key_path, _) = xencode_colab_rs::preflight::key_paths().unwrap_or_else(|_| {
            (
                dir.join(xencode_colab_rs::KEY_FILENAME),
                dir.join("unused.pub"),
            )
        });
        checks.push(doc::check_permissions(
            "colab-key",
            &key_path,
            doc::file_mode(&key_path),
        ));
    }
    if let Some(dir) = state_dir.as_ref() {
        checks.push(doc::check_free_disk(
            "state",
            dir,
            xencode_context_rs::hwprobe::free_disk_bytes(&dir.display().to_string()),
        ));
    }
    if let Some(dir) = cache_dir.as_ref() {
        checks.push(doc::check_dir_size("cache", dir));
    }
    checks.extend(spine_checks(&root, &config).await);
    checks.extend(colab_checks().await);
    // QK-4: a durable fact the code has contradicted leaves the prompt without a
    // word. The row says which lines went, and why, in the report a person pastes.
    checks.push(doc::check_durable_facts(
        &root.join(xencode_context_rs::XENCODE_DIR),
    ));
    checks.push(doc::check_anchor(
        &root.join(xencode_context_rs::XENCODE_DIR),
    ));

    render_checks("report", &checks, format)
}

/// One list, two renderings. The rows are the report: `--format json` emits them
/// as they are, and the text listing is the same list with a mark in front.
pub(crate) fn render_checks(
    surface: &str,
    checks: &[doc::SelfCheck],
    format: OutputFormat,
) -> Result<(), String> {
    let failing: Vec<&str> = checks
        .iter()
        .filter(|check| check.state == "fail")
        .map(|check| check.name.as_str())
        .collect();
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "doctor": surface,
                "version": env!("CARGO_PKG_VERSION"),
                "ok": failing.is_empty(),
                "failing": failing,
                "checks": checks,
            })
        );
        if failing.is_empty() {
            return Ok(());
        } else {
            return Err(format!(
                "doctor found {} failing check{}",
                failing.len(),
                if failing.len() == 1 { "" } else { "s" }
            ));
        }
    }
    for check in checks {
        println!("  {:<6} {:<22} {}", check.mark(), check.name, check.detail);
        if let Some(fix) = &check.fix {
            println!("         {:<22} fix: {fix}", "");
        }
    }
    if !failing.is_empty() {
        println!("\n  failing: {}", failing.join(", "));
        return Err(format!(
            "doctor found {} failing check{}",
            failing.len(),
            if failing.len() == 1 { "" } else { "s" }
        ));
    }
    Ok(())
}

pub(crate) fn run_session(action: SessionAction) -> Result<(), String> {
    use xencode_context_rs::session as sess;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
    match action {
        SessionAction::Name { run, name } => {
            let run_id = sess::resolve_session(&xencode_dir, &run)?;
            sess::name_session(&xencode_dir, &name, &run_id)?;
            println!("  {name} now names run {run_id}");
            Ok(())
        }
        SessionAction::Resolve { target } => {
            let run_id = sess::resolve_session(&xencode_dir, &target)?;
            println!("{run_id}");
            Ok(())
        }
        SessionAction::Export { target, redacted } => {
            print!(
                "{}",
                sess::export_transcript(&xencode_dir, &target, redacted)?
            );
            Ok(())
        }
    }
}

pub(crate) fn run_verify(
    skip: Vec<String>,
    timeout: u64,
    session: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::toolchain as kit;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    for name in &skip {
        if !["test", "lint", "fmt"].contains(&name.as_str()) {
            return Err(format!(
                "cannot skip {name:?}: the checklist is test, lint, fmt"
            ));
        }
    }
    let target_session = session.or_else(|| {
        ConversationMemory::with_persistence(50)
            .ok()
            .and_then(|m| m.current_session().cloned())
    });
    let list = kit::run_checklist_for_session(&root, &skip, timeout, target_session.as_deref())?;
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "ok": list.ok(),
                "checks": list.checks.iter().map(|c| serde_json::json!({
                    "name": c.name, "ran": c.ran,
                    "exit": c.exit_code, "evidence": c.evidence_ref,
                })).collect::<Vec<_>>(),
                "failed": list.failed(),
                "skipped": list.skipped(),
            })
        );
    } else {
        for check in &list.checks {
            let state = if !check.ran {
                "SKIPPED".to_string()
            } else if check.passed() {
                "PASS".to_string()
            } else {
                format!("FAIL (exit {})", check.exit_code.unwrap_or(-1))
            };
            println!("  {:<7} {:<5} {}", check.name, state, check.evidence_ref);
        }
        if !list.skipped().is_empty() {
            println!("\n  skipped, not passed: {}", list.skipped().join(", "));
        }
    }
    if list.ok() {
        Ok(())
    } else {
        Err(format!("checklist failed: {}", list.failed().join(", ")))
    }
}

pub(crate) fn run_envcheck(format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::envdrift;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let refs = envdrift::extract_env_refs(&root);
    let templates = envdrift::read_templates(&root);
    let report = envdrift::compare(&refs, &templates);

    if matches!(format, OutputFormat::Json) {
        let out = serde_json::json!({
            "sources_searched": report.sources_searched,
            "undocumented": report.undocumented.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
                "panics_when_absent": r.panics_when_absent,
            })).collect::<Vec<_>>(),
            "unreferenced": report.unreferenced,
            "os_provided": report.os_provided.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
            })).collect::<Vec<_>>(),
            "panicking": report.panicking.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
    } else {
        if report.sources_searched.is_empty() {
            println!(
                "\n  no template found (.env.example, .env.template) — nothing documents the configuration"
            );
        } else {
            println!("\n  templates read: {}", report.sources_searched.join(", "));
        }
        if report.undocumented.is_empty() {
            println!("  no read-but-undocumented keys");
        } else {
            println!("\n  read-but-undocumented:");
            for (key, refs) in envdrift::by_key(&report.undocumented) {
                let at = refs
                    .iter()
                    .map(|r| format!("{}:{}", r.file, r.line))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("    {key}  (read at {at})");
            }
        }
        if !report.unreferenced.is_empty() {
            println!("\n  documented-but-unreferenced (not unnecessary):");
            for key in &report.unreferenced {
                println!("    {key}");
            }
        }
        if !report.panicking.is_empty() {
            println!("\n  panics-when-absent outside tests:");
            for r in &report.panicking {
                println!("    {}  ({}:{})", r.key, r.file, r.line);
            }
        }
        println!(
            "\n  {} os-provided key(s) listed separately, not reported",
            report.os_provided.len()
        );
    }
    Ok(())
}
