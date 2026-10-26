//! the colab, bootstrap, remote and computers subcommands
//! `xencode colab`, `bootstrap`, `remote` and `computers`.

use super::*;

pub(crate) async fn run_colab(action: ColabAction) -> Result<(), String> {
    match action {
        ColabAction::Preflight { generate_key } => run_colab_preflight(generate_key).await,
        ColabAction::Up {
            session,
            gpu,
            runtime,
            model,
            weights,
            quant,
            local_port,
            remote_port,
            reconnect,
            dry_run,
        } => {
            run_colab_up_cli(
                session,
                gpu,
                runtime,
                model,
                weights,
                quant,
                local_port,
                remote_port,
                reconnect,
                dry_run,
            )
            .await
        }
        ColabAction::Status => run_colab_status_cli().await,
        ColabAction::Down => run_colab_down_cli().await,
    }
}

/// What a successful `colab up` would persist to config.json, computed by the
/// same call the real bring-up makes — so the preview cannot disagree with the
/// thing it previews. Only the keys that would actually change are listed.
pub(crate) fn colab_up_config_preview(
    current: &XencodeConfig,
    runtime: &str,
    local_port: u16,
) -> Vec<String> {
    let mut preview = current.clone();
    xencode_colab_rs::point_config_at_forward(
        &mut preview,
        runtime,
        &format!("http://127.0.0.1:{local_port}"),
    );
    let mut lines = Vec::new();
    for (key, before, after) in [
        (
            "llama_cpp_url",
            current.llama_cpp_url.as_str(),
            preview.llama_cpp_url.as_str(),
        ),
        (
            "ollama_url",
            current.ollama_url.as_str(),
            preview.ollama_url.as_str(),
        ),
        (
            "remote_base_url",
            current.remote_base_url.as_str(),
            preview.remote_base_url.as_str(),
        ),
    ] {
        if before != after {
            lines.push(format!(
                "config.json would change: {key}: {before} → {after}"
            ));
        }
    }
    if lines.is_empty() {
        lines.push("config.json would not change".to_string());
    }
    lines
}

/// Write, or only report, the three files a project that has never run xencode
/// is usually missing. See `xencode_context_rs::bootstrap` for why every byte of
/// them is a fact or a blank, and never a guessed command.
pub(crate) fn run_bootstrap(
    path: &std::path::Path,
    check: bool,
    format: OutputFormat,
) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("not a directory: {}", path.display()));
    }
    // Every key the loader reads, at the value it ships with, from the same
    // struct this binary saves and reads: a template written by hand would drift
    // from the config the day a field is added, and a field added by hand would
    // teach a person a key that does not exist. The nine credential fields are
    // `None`, which serialises to `null`, so nothing but the shape of a real
    // settings file is written into somebody's repository.
    let template = serde_json::to_value(XencodeConfig::default())
        .map(|mut value| {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "_comment".to_string(),
                    serde_json::json!("Every key xencode reads, at the value this binary ships with. Copy the keys you want into `config.json` in the settings directory (`$XCODE_CONFIG_DIR`, or `~/.config/xencode`) — this file is an example, not a live setting, and the loader ignores a key it does not know. A credential is best left null here and supplied by `xencode config set <KEY> command:<program> <args>`, which stores the reference and not the secret, or by the provider's own environment variable. Nothing in this file was typed out by hand: it is generated from the struct the loader reads, so a key here exists and a key that is missing was added after this binary was built."),
                );
            }
            value
        })
        .and_then(|value| serde_json::to_string_pretty(&value))
        .map_err(|e| format!("could not render the settings template: {e}"))?;
    let report =
        xencode_context_rs::bootstrap(path, &template, check).map_err(|e| format!("{e}"))?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "root": report.root,
                "git": report.facts.is_git,
                "branch": report.facts.branch,
                "revision": report.facts.revision,
                "files": report.facts.files,
                "scrubbed": report.scrubbed,
                "check": report.check_only,
                "entries": report.entries,
            })
        );
        return Ok(());
    }

    println!("Project: {}", report.root);
    let where_from = if !report.facts.is_git {
        "not a git repository".to_string()
    } else {
        let revision = report.facts.revision.as_deref().unwrap_or("(no revision)");
        let files = report
            .facts
            .files
            .map(|n| n.to_string())
            .unwrap_or_else(|| "file count unknown".to_string());
        match report.facts.branch.as_deref() {
            Some(branch) => format!("git branch {branch} at {revision}, {files} files"),
            None => format!("git detached head at {revision}, {files} files"),
        }
    };
    println!("  {where_from}");
    for entry in &report.entries {
        println!(
            "  {:<12} {:<23} {}",
            if report.check_only && entry.action == "write" {
                "would write"
            } else {
                entry.action
            },
            entry.path,
            entry.reason
        );
    }
    let writing = report.writing().count();
    let keeping = report.keeping().count();
    if report.check_only {
        println!(
            "\nReport only: {writing} would be written, {keeping} already in place. Nothing created."
        );
    } else if writing == 0 {
        println!("\nNothing to write: all {keeping} files are already there.");
    } else {
        println!(
            "\n{writing} file{} written. Nothing that already existed was touched.",
            if writing == 1 { "" } else { "s" }
        );
    }
    if report.scrubbed {
        println!(
            "A name on this disk held something credential-shaped, and was written as [redacted]."
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to colab up flags; a struct would just rename them
pub(crate) async fn run_colab_up_cli(
    session: Option<String>,
    gpu: Option<String>,
    runtime: Option<String>,
    model: Option<String>,
    weights: Option<String>,
    quant: Option<String>,
    local_port: Option<u16>,
    remote_port: Option<u16>,
    reconnect: bool,
    dry_run: bool,
) -> Result<(), String> {
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    if !config.colab.enabled {
        return Err(
            "colab is disabled — set it: `xencode config set colab_enabled true`".to_string(),
        );
    }

    // Flags override config; config provides the defaults.
    let session = session
        .or(if config.colab.session.is_empty() {
            None
        } else {
            Some(config.colab.session.clone())
        })
        .unwrap_or("xencode-vm".to_string());
    let runtime = runtime.unwrap_or(config.colab.runtime.clone());
    let model = model
        .or(if config.colab.model.is_empty() {
            None
        } else {
            Some(config.colab.model.clone())
        })
        .unwrap_or("Qwen/Qwen2.5-7B-Instruct-GGUF".to_string());
    let weights_source = weights.unwrap_or(config.colab.weights_source.clone());
    let quant = quant.unwrap_or(config.colab.quant.clone());
    let local_port = local_port.unwrap_or(config.colab.local_port);
    let remote_port = remote_port.unwrap_or(config.colab.remote_port);

    if dry_run {
        // Deliberately before the preflight: preflight(true) would create the
        // SSH keypair, and a preview writes nothing anywhere.
        println!("colab up --dry-run — nothing was started and nothing was written.");
        println!("  session:   {session}");
        println!(
            "  runtime:   {runtime}   gpu: {}   model: {model}",
            gpu.as_deref().unwrap_or("T4")
        );
        println!(
            "  weights:   {weights_source}   quant: {}",
            if quant.is_empty() {
                "Q4_K_M (the default the VM serves)"
            } else {
                quant.as_str()
            }
        );
        println!(
            "  forward:   http://127.0.0.1:{local_port} → the VM's port {}",
            xencode_colab_rs::effective_remote_port(&runtime, remote_port)
        );
        for line in colab_up_config_preview(&config, &runtime, local_port) {
            println!("  {line}");
        }
        if reconnect {
            println!("  reconnect: reuses the URL in the recorded state when the forward is already live");
        }
        return Ok(());
    }

    // Gate on the bridge being usable; preflight also ensures the SSH key.
    let report = xencode_colab_rs::preflight(true)
        .await
        .map_err(|e| format!("colab preflight: {e}"))?;
    if !report.ready() {
        let failed: Vec<String> = report
            .checks
            .iter()
            .filter(|c| !c.ok)
            .map(|c| {
                format!(
                    "{} — {}",
                    c.name,
                    c.fix.as_deref().unwrap_or(c.detail.as_str())
                )
            })
            .collect();
        return Err(format!(
            "colab preflight not ready — run `xencode colab preflight` to see fixes:\n  {}",
            failed.join("\n  ")
        ));
    }

    let bins = xencode_colab_rs::resolve_binaries()?;
    let config_dir = XencodeConfig::config_dir().map_err(|e| e.to_string())?;
    let key = config_dir.join(xencode_colab_rs::KEY_FILENAME);

    let opts = xencode_colab_rs::UpOptions {
        session: session.clone(),
        gpu: gpu.unwrap_or("T4".to_string()),
        runtime: runtime.clone(),
        model: model.clone(),
        weights_source,
        quant,
        local_port,
        remote_port,
    };

    let url = if reconnect {
        xencode_colab_rs::run_colab_reconnect(&bins, &key, &opts).await?
    } else {
        xencode_colab_rs::run_colab_up(&bins, &key, &opts).await?
    };

    // Point the provider URLs at the forward so the picker and the
    // remote:/llama/ollama routes see the VM, then persist the change.
    let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
    let base = url.trim_end_matches("/v1");
    xencode_colab_rs::point_config_at_forward(&mut config, &runtime, base);
    config
        .save()
        .map_err(|e| format!("colab up: could not save config: {e}"))?;

    if reconnect {
        println!("Colab reconnected — {url}");
    } else {
        println!("Colab up — {url}");
    }
    println!("  session:   {session}");
    println!("  runtime:   {runtime}");
    println!("  model:     {model}");
    println!(
        "  provider:  {} -> {}",
        if runtime == "ollama" {
            "ollama_url"
        } else {
            "llama_cpp_url"
        },
        base
    );
    println!(
        "  remote:    {} (OpenAI-compatible)",
        config.remote_base_url
    );
    // What the VM's bootstrap warned about, such as a GPU build that ended up
    // serving on the CPU (K-6).
    if let Ok(Some(state)) = xencode_colab_rs::ColabState::load() {
        for warning in &state.warnings {
            println!("  {warning}");
        }
    }
    println!("  The VM endpoint is live. `xencode colab status` shows the bridge.");
    Ok(())
}

pub(crate) async fn run_colab_status_cli() -> Result<(), String> {
    let bins = xencode_colab_rs::resolve_binaries()?;
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    // An empty `colab.session` is not a session name; `status` prefers the
    // recorded colab.json and only falls back to what the caller asks for.
    let report = xencode_colab_rs::run_colab_status(&bins, &config.colab.session).await;
    for line in &report.lines {
        println!("  {line}");
    }
    Ok(())
}

pub(crate) async fn run_colab_down_cli() -> Result<(), String> {
    let bins = xencode_colab_rs::resolve_binaries()?;
    let summary = xencode_colab_rs::run_colab_down(&bins).await?;
    println!("{summary}");
    Ok(())
}

pub(crate) async fn run_colab_preflight(generate_key: bool) -> Result<(), String> {
    let report = xencode_colab_rs::preflight(generate_key)
        .await
        .map_err(|e| format!("colab preflight: {e}"))?;

    println!("Colab preflight — google-colab-cli bridge");
    for check in &report.checks {
        let mark = if check.ok { "[ ok ]" } else { "[FAIL]" };
        println!("  {mark} {:<18} {}", check.name, check.detail);
        if let Some(fix) = &check.fix {
            println!("         fix: {fix}");
        }
    }

    if report.ready() {
        println!("Ready — the Colab bridge can be brought up.");
        Ok(())
    } else {
        Err(format!(
            "colab preflight found {} failing check(s)",
            report.failed()
        ))
    }
}

pub(crate) fn run_remote(action: RemoteAction) -> Result<(), String> {
    use xencode_config_rs::remotes::{
        active, forget, list, load, parse_destination, save, set_active, Forgot, RemoteProfile,
    };
    match action {
        RemoteAction::Add {
            name,
            host,
            runtime,
            model,
            port,
            local_port,
            remote_port,
            force,
        } => {
            let dest = parse_destination(&host).map_err(|e| e.to_string())?;
            let ssh_port = port.or(dest.port).unwrap_or(22);
            let profile = RemoteProfile {
                name: name.clone(),
                host: dest.host,
                user: dest.user,
                port: ssh_port,
                runtime,
                model,
                local_port,
                remote_port,
            };
            let path = save(&profile, force).map_err(|e| e.to_string())?;
            println!(
                "Recorded remote profile `{}` in {}",
                profile.name,
                path.display()
            );
            Ok(())
        }
        RemoteAction::List => {
            let inventory = list().map_err(|e| e.to_string())?;
            let current = active().map_err(|e| e.to_string())?;
            let active_name = current.as_ref().map(|p| p.name.as_str());
            if inventory.is_empty() {
                println!(
                    "No remote profiles recorded. Add one with `xencode remote add <name> <user@host>`."
                );
                return Ok(());
            }
            println!("Remote profiles:");
            for profile in &inventory.profiles {
                let marker = if active_name == Some(profile.name.as_str()) {
                    "*"
                } else {
                    " "
                };
                println!("  {marker} {:<12} {}", profile.name, profile.summary());
            }
            for (bad, err) in &inventory.unreadable {
                println!("  ! {:<12} unreadable: {err}", bad);
            }
            Ok(())
        }
        RemoteAction::Use { name } => {
            set_active(&name).map_err(|e| e.to_string())?;
            println!("Active remote profile set to `{name}`.");
            Ok(())
        }
        RemoteAction::Forget { name } => match forget(&name).map_err(|e| e.to_string())? {
            Forgot::Removed => {
                println!("Removed remote profile `{name}`.");
                Ok(())
            }
            Forgot::RemovedWhileActive => {
                println!("Removed remote profile `{name}` (active profile cleared).");
                Ok(())
            }
            Forgot::NotRecorded => Err(format!("`{name}` is not a recorded remote profile")),
        },
        RemoteAction::Show { name } => {
            let target = match name {
                Some(n) => load(&n)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| format!("`{n}` is not a recorded remote profile"))?,
                None => active().map_err(|e| e.to_string())?.ok_or_else(|| {
                    "No active remote profile. Use `xencode remote use <name>` or specify a profile name."
                        .to_string()
                })?,
            };
            let is_active = active()
                .ok()
                .flatten()
                .map(|a| a.name == target.name)
                .unwrap_or(false);
            let active_marker = if is_active { " [active]" } else { "" };
            println!("Remote profile `{}`{active_marker}", target.name);
            println!("  destination: {}", target.destination());
            println!("  ssh port:    {}", target.port);
            println!("  runtime:     {}", target.runtime);
            if let Some(m) = &target.model {
                println!("  model:       {m}");
            }
            println!("  local port:  {}", target.local_port);
            let rport = if target.remote_port == 0 {
                "0 (runtime default)".to_string()
            } else {
                target.remote_port.to_string()
            };
            println!("  remote port: {rport}");
            Ok(())
        }
    }
}

pub(crate) async fn run_computers(
    action: Option<ComputersAction>,
    json: bool,
) -> Result<(), String> {
    let bins = match xencode_colab_rs::resolve_binaries() {
        Ok(b) => b,
        Err(_) => {
            let colab_path = xencode_colab_rs::which("colab")
                .unwrap_or_else(|| std::path::PathBuf::from("colab"));
            let ssh_path =
                xencode_colab_rs::which("ssh").unwrap_or_else(|| std::path::PathBuf::from("ssh"));
            xencode_colab_rs::Binaries {
                colab: colab_path,
                ssh: ssh_path,
            }
        }
    };
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    let key_path = XencodeConfig::config_dir()
        .map_err(|e| e.to_string())?
        .join(xencode_colab_rs::KEY_FILENAME);
    let registry = xencode_colab_rs::BackendRegistry::default();

    match action.unwrap_or(ComputersAction::List) {
        ComputersAction::List => {
            let list = registry.list_computers(&bins, &key_path);
            if json {
                let mut json_items = Vec::new();
                for c in &list {
                    json_items.push(serde_json::json!({
                        "id": c.id,
                        "kind": c.kind,
                        "description": c.description,
                        "available": c.available,
                        "detail": c.detail,
                        "active": c.id == config.computer_backend,
                    }));
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_items).map_err(|e| e.to_string())?
                );
                return Ok(());
            }

            println!("Registered computer backends:");
            for c in &list {
                let marker = if c.id == config.computer_backend {
                    "* "
                } else {
                    "  "
                };
                let status = if c.available {
                    "available"
                } else {
                    "unavailable"
                };
                println!(
                    "{marker}{:<8} ({:<6}) — {} [{}: {}]",
                    c.id, c.kind, c.description, status, c.detail
                );
            }
            Ok(())
        }
        ComputersAction::Show { name } => {
            let list = registry.list_computers(&bins, &key_path);
            let item = list.into_iter().find(|c| c.id == name).ok_or_else(|| {
                format!(
                    "unknown computer backend `{name}`; available: {}",
                    registry.available_backends().join(", ")
                )
            })?;
            if json {
                let doc = serde_json::json!({
                    "id": item.id,
                    "kind": item.kind,
                    "description": item.description,
                    "available": item.available,
                    "detail": item.detail,
                    "active": item.id == config.computer_backend,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            let active_str = if item.id == config.computer_backend {
                " (active)"
            } else {
                ""
            };
            println!("Computer `{}`{active_str}:", item.id);
            println!("  kind:        {}", item.kind);
            println!("  description: {}", item.description);
            println!(
                "  status:      {}",
                if item.available {
                    "available"
                } else {
                    "unavailable"
                }
            );
            println!("  detail:      {}", item.detail);
            Ok(())
        }
        ComputersAction::Use { name } => {
            if !registry.available_backends().contains(&name.as_str()) {
                return Err(format!(
                    "unknown computer backend `{name}`; available backends: {}",
                    registry.available_backends().join(", ")
                ));
            }
            let mut cfg = config;
            cfg.computer_backend = name.clone();
            cfg.save().map_err(|e| e.to_string())?;
            println!("Active computer backend set to `{name}`.");
            Ok(())
        }
        ComputersAction::Probe { name } => {
            let backend = registry.resolve(&name, &bins, &key_path)?;
            println!(
                "Probing computer backend `{name}` (kind: {})...",
                backend.kind()
            );
            let (avail, detail) = backend.is_available();
            if !avail {
                return Err(format!("computer `{name}` cannot be reached: {detail}"));
            }
            match backend.provision("probe-test", "none").await {
                Ok(()) => {
                    println!("Successfully connected to computer backend `{name}`.");
                    Ok(())
                }
                Err(e) => Err(format!("computer `{name}` cannot be reached: {e}")),
            }
        }
    }
}
