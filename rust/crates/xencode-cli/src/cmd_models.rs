//! the models, history, hardware and llamacpp subcommands
//! `xencode models`, `history`, `hw` and `llamacpp`.

use super::*;

pub(crate) async fn run_models(action: ModelAction) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let mut client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let mut llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);

    match action {
        ModelAction::List => {
            let ollama_models = client.list_models().await.unwrap_or_default();
            let llama_models = llama_client.list_models().await.unwrap_or_default();

            if ollama_models.is_empty() && llama_models.is_empty() {
                println!("No local models installed or running.");
                println!("For Ollama:   ollama pull qwen2.5:7b");
                println!("For llama.cpp: start llama-server with your GGUF model");
                return Ok(());
            }

            println!("{:<30} {:>12} PROVIDER", "MODEL", "SIZE");
            println!("{}", "-".repeat(60));
            for model in &ollama_models {
                let size_mb = model.size as f64 / 1_048_576.0;
                println!("{:<30} {:>9.1} MB [ollama]", model.name, size_mb);
            }
            for model in &llama_models {
                let display_name = format!("llamacpp:{}", model.id);
                println!("{:<30} {:>12} [llama.cpp]", display_name, "N/A");
            }
            let total = ollama_models.len() + llama_models.len();
            println!("\n{} model(s) available", total);
            Ok(())
        }
        ModelAction::Health { model } => {
            println!("Checking health of {model}...");
            if model.starts_with("llamacpp:")
                || model.starts_with("llama.cpp:")
                || model.starts_with("llama:")
            {
                let raw_model = model
                    .strip_prefix("llamacpp:")
                    .or_else(|| model.strip_prefix("llama.cpp:"))
                    .or_else(|| model.strip_prefix("llama:"))
                    .unwrap_or(&model);
                match llama_client.check_health(raw_model).await {
                    Ok(health) => {
                        println!("  Provider:      llama.cpp ({})", config.llama_cpp_url);
                        println!("  Status:        {}", health.status);
                        println!("  Response time: {:.3}s", health.response_time);
                        if let Some(ref err) = health.error_message {
                            println!("  Error:         {err}");
                        }
                    }
                    Err(e) => {
                        println!("  Provider:      llama.cpp ({})", config.llama_cpp_url);
                        println!("  Status:        unavailable");
                        println!("  Error:         {e}");
                    }
                }
            } else {
                let health = client
                    .check_health(&model)
                    .await
                    .map_err(|e| e.to_string())?;
                println!("  Provider:      Ollama ({})", config.ollama_url);
                println!("  Status:        {}", health.status);
                println!("  Response time: {:.3}s", health.response_time);
                if let Some(ref err) = health.error_message {
                    println!("  Error:         {err}");
                }
            }
            Ok(())
        }
        ModelAction::Default => {
            let default = client
                .get_smart_default()
                .await
                .map_err(|e| e.to_string())?;
            match default {
                Some(model) => println!("Smart default: {model}"),
                None => println!("No models available"),
            }
            Ok(())
        }
        ModelAction::Advice => run_models_advice(),
    }
}

/// Answer "which GGUF should this machine serve" from the advice table, and say
/// how old that answer is.
///
/// The capacity the table is matched against is the biggest single place a
/// model's bytes could go — one memory pool, not the sum of several, because
/// `llama-server` puts a model's weights in one place. That is the same reading
/// the launch preflight uses, so the two commands cannot disagree about whether
/// a model fits.
pub(crate) fn run_models_advice() -> Result<(), String> {
    use xencode_context_rs::hwprobe;
    use xencode_models_rs::advice;

    let advice = advice::Advice::load(&advice::default_path());
    if let advice::AdviceSource::EmbeddedAfterRefusingUserFile(reason) = &advice.source {
        println!("note:     {reason}");
    }

    let total_kib = xencode_context_rs::budget::total_memory_kib().unwrap_or(0);
    let available_kib = hwprobe::available_memory_kib().unwrap_or(0);
    let exe = xencode_models_rs::llamacpp::find_llama_server(None);
    let devices: Vec<hwprobe::ComputeDevice> = match &exe {
        Some(exe) => hwprobe::server_devices(exe)
            .as_deref()
            .map(hwprobe::parse_llama_devices)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let pools = hwprobe::memory_pools(&devices, total_kib / 1024, available_kib / 1024);
    let largest = pools.iter().max_by_key(|pool| pool.usable_mib);
    let capacity_bytes = largest
        .map(|pool| pool.usable_mib * 1024 * 1024)
        .unwrap_or(0);
    match largest {
        Some(pool) => println!(
            "room:     {} ({})",
            xencode_models_rs::human_bytes(capacity_bytes),
            pool.label
        ),
        None => println!(
            "room:     nothing measured — this machine reported no memory a model could go in"
        ),
    }

    let age = advice.age_days(advice::AdviceFile::today_epoch_days());
    println!("advice:   checked {}", advice.file.as_of);
    if let Some(age) = age {
        if age > advice::ROT_HORIZON_DAYS {
            println!(
                "          {age} days old, past the {} days this table is meant to be trusted \
                 for — the entries may name models that no longer exist or quants that have \
                 been beaten",
                advice::ROT_HORIZON_DAYS
            );
        } else {
            println!("          {age} days old");
        }
    }

    let source = match &advice.source {
        advice::AdviceSource::Embedded => "the table shipped with xencode".to_string(),
        advice::AdviceSource::UserFile(path) => path.clone(),
        advice::AdviceSource::EmbeddedAfterRefusingUserFile(_) => {
            "the table shipped with xencode".to_string()
        }
    };
    println!("from:     {source}");

    match advice.tier_for(capacity_bytes) {
        None => println!(
            "answer:   nothing in the table fits in {r}",
            r = xencode_models_rs::human_bytes(capacity_bytes)
        ),
        Some(tier) => {
            println!("tier:     {}", tier.name);
            for entry in &tier.gguf {
                println!("          {}", entry.label);
                println!(
                    "            size     {}",
                    xencode_models_rs::human_bytes(entry.size_bytes)
                );
                println!("            url      {}", entry.url());
                println!("            sha256   {}", entry.sha256);
            }
            println!(
                "\nto serve one of these:\n  \
                 xencode config set llama_cpp_model_url \"{}\"\n  \
                 xencode config set llama_cpp_model_sha256 \"{}\"\n  \
                 xencode config set llama_cpp_model_path \"{}\"\n  \
                 xencode llamacpp start\n\n\
                 The URL is pinned to a repository revision and the checksum is the one the \
                 host published for these bytes on {}, so a launch refuses the file rather than \
                 serving bytes that are not the ones this table described.",
                tier.gguf[0].url(),
                tier.gguf[0].sha256,
                default_gguf_path(&tier.gguf[0].file),
                advice.file.as_of
            );
        }
    }
    Ok(())
}

/// Where a fetched model lands by default: the directory xencode keeps the
/// things it fetched, which is not the cache — a weight costs a long download to
/// replace, and a cache cleaner is meant to be able to throw a cache away.
pub(crate) fn default_gguf_path(file_name: &str) -> String {
    xencode_config_rs::paths::data_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(file_name)
        .display()
        .to_string()
}

/// A duration git was measured taking, in milliseconds with one decimal,
/// because a warm repository answers in single-digit milliseconds and rounding
/// to whole numbers would print several probes as 0 ms.
pub(crate) fn history_ms(microseconds: u128) -> String {
    format!("{:.1} ms", microseconds as f64 / 1000.0)
}

/// How much a query handed back — what a model would have to pay to be shown
/// the history. Under a kibibyte the number is given in bytes, because "0 KiB"
/// for a line of text reads like "nothing".
pub(crate) fn history_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    }
}

/// Print one status page: what exists, what it costs, what to do about it.
pub(crate) fn print_history_status(status: &xencode_context_rs::HistoryStatus) {
    println!("repo:    {}", status.git_dir);
    println!(
        "history: {} reachable commit(s){}{}",
        status.reachable_commits,
        if status.shallow { ", shallow" } else { "" },
        status
            .partial_clone
            .as_ref()
            .map(|f| format!(", partial clone (filter {f})"))
            .unwrap_or_default(),
    );
    match &status.commit_graph {
        Some(graph) => println!(
            "commit-graph: {} ({}){}",
            graph.path,
            history_bytes(graph.size as usize),
            match graph.verifies {
                Some(true) => ", verifies",
                Some(false) => ", DOES NOT VERIFY against the objects here",
                None => "",
            }
        ),
        None => println!("commit-graph: none"),
    }
    match &status.pack_index {
        Some(index) => println!(
            "multi-pack-index: {} ({}) over {} pack(s)",
            index.path,
            history_bytes(index.size as usize),
            index.packs
        ),
        None => println!("multi-pack-index: none"),
    }
    println!("timed now:");
    for query in &status.queries {
        match &query.failed {
            Some(reason) => println!("  {:<46} failed: {reason}", query.label),
            None => println!(
                "  {:<46} {:>9}  {} out",
                query.label,
                history_ms(query.microseconds),
                history_bytes(query.bytes),
            ),
        }
    }
    for action in status.actions() {
        println!("todo:    {action}");
    }
}

/// `xencode history` — read what makes history queries fast here, or write it.
pub(crate) fn run_history(action: HistoryAction) -> Result<(), String> {
    use xencode_context_rs::{default_blame_target, history_setup, history_status};

    if let HistoryAction::Digest { path, file, json } = action {
        let digest = xencode_context_rs::history_digest(&path, &file);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "file": file,
                    "digest": digest,
                    "chars": digest.len(),
                    "cap": xencode_context_rs::DIGEST_CHAR_CAP,
                }))
                .unwrap_or_default()
            );
        } else {
            println!("{digest}");
        }
        return Ok(());
    }
    let (command, path, file, json) = match action {
        HistoryAction::Status { path, file, json } => ("status", path, file, json),
        HistoryAction::Setup { path, file, json } => ("setup", path, file, json),
        HistoryAction::Digest { .. } => unreachable!("handled above"),
    };
    let blame = match file {
        Some(file) => Some(file),
        None => default_blame_target(&path),
    };

    if command == "setup" {
        let setup = history_setup(&path, blame.as_deref())?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&setup).unwrap_or_default()
            );
            return Ok(());
        }
        print_history_status(&setup.before);
        println!();
        for line in &setup.wrote {
            println!("wrote:   {line}");
        }
        for line in &setup.refused {
            println!("skipped: {line}");
        }
        println!();
        println!(
            "after:  (timed in the same run, so this table also has the operating system's page \
             cache warmed by the one above — read it as an upper bound, not as the index's own \
             effect)"
        );
        print_history_status(&setup.after);
        // The point of the command is the speed, so say plainly whether it
        // moved rather than leaving the reader to compare two tables. Two
        // runs of the same query on the same machine differ by a couple of
        // milliseconds, so a claim needs to clear both a share and a floor.
        const MIN_RATIO: f64 = 1.2;
        const MIN_FLOOR_US: u128 = 2_000;
        let mut moved = false;
        for (before, after) in setup.before.queries.iter().zip(&setup.after.queries) {
            if before.failed.is_some() || after.failed.is_some() || after.microseconds == 0 {
                continue;
            }
            let gap = before.microseconds.abs_diff(after.microseconds);
            if gap < MIN_FLOOR_US {
                continue;
            }
            let ratio = before.microseconds as f64 / after.microseconds as f64;
            let words = if ratio >= MIN_RATIO {
                moved = true;
                format!("{ratio:.1}× faster")
            } else if ratio <= 1.0 / MIN_RATIO {
                moved = true;
                format!("{:.1}× slower", 1.0 / ratio)
            } else {
                continue;
            };
            println!(
                "{}: {} → {} ({words})",
                after.label,
                history_ms(before.microseconds),
                history_ms(after.microseconds),
            );
        }
        if !moved {
            println!(
                "no query changed by more than both 20% and 2 ms — on {} commits the commit \
                 chain was not the cost, so these indexes are there for the queries built on \
                 history rather than for the ones timed above",
                setup.before.reachable_commits
            );
        }
        return Ok(());
    }

    let status = history_status(&path, blame.as_deref())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status).unwrap_or_default()
        );
    } else {
        print_history_status(&status);
    }
    Ok(())
}

/// Report this machine as a place a model could be served, and the flags to
/// start a server with. Everything printed here is read from a named source at
/// run time, including the server's own device list — the two sources disagree
/// routinely and the point of the command is to show both, not to average them.
pub(crate) fn run_hw(action: HwAction) -> Result<(), String> {
    use xencode_context_rs::hwprobe;

    let HwAction::Probe { model, exec } = action;
    let config = XencodeConfig::load().unwrap_or_default();

    let total_kib = xencode_context_rs::budget::total_memory_kib();
    let available_kib = hwprobe::available_memory_kib();
    let fmt = |kib: Option<u64>| match kib {
        Some(kib) => format!("{:.1} GiB", kib as f64 / (1024.0 * 1024.0)),
        None => "unknown".to_string(),
    };
    println!(
        "ram:     {} total, {} available",
        fmt(total_kib),
        fmt(available_kib)
    );
    println!(
        "cores:   {}",
        hwprobe::cpu_core_count()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );

    let asked_exec = exec
        .as_deref()
        .or(if config.llama_cpp_executable.is_empty() {
            None
        } else {
            Some(config.llama_cpp_executable.as_str())
        });
    let exe = xencode_models_rs::llamacpp::find_llama_server(asked_exec);
    match &exe {
        Some(exe) => match hwprobe::server_version(exe) {
            Some(version) => println!("server:  {exe}  version {version}"),
            None => println!("server:  {exe}"),
        },
        None => println!(
            "server:  no llama-server found — set config llama_cpp_executable or pass --exec, \
             because what the binary can offload to cannot be read without asking it"
        ),
    }
    let devices: Vec<hwprobe::ComputeDevice> = match &exe {
        Some(exe) => match hwprobe::server_devices(exe) {
            Some(text) => hwprobe::parse_llama_devices(&text),
            None => {
                println!(
                    "devices: this server does not answer --list-devices, so what it can offload \
                     to is unknown rather than absent"
                );
                Vec::new()
            }
        },
        None => Vec::new(),
    };

    let cards = hwprobe::drm_cards(std::path::Path::new("/sys/class/drm"));
    if !cards.is_empty() {
        println!("cards:   the kernel's view, which does not report memory at all");
        for card in &cards {
            println!(
                "         {} · {} {} · driver {}{}",
                card.card,
                hwprobe::vendor_name(&card.vendor),
                card.device,
                if card.driver.is_empty() {
                    "none"
                } else {
                    &card.driver
                },
                if card.has_connector {
                    ", driving a display"
                } else {
                    ", no display attached"
                }
            );
        }
    }
    if !cards.is_empty() {
        println!(
            "         no memory figure is printed for a card here: PCI config space is what the \
             kernel exposes without a vendor tool, and on this box it reports a 256 MiB window \
             for a 2048 MiB device. The sizes below come from the server."
        );
    }
    if !devices.is_empty() {
        println!("devices  what the server itself can use:");
        let ram_total_mib = total_kib.map(|k| k / 1024).unwrap_or(0);
        for device in &devices {
            let kind = if !device.is_offload_target() {
                "the CPU path, reported as a device and not one"
            } else if hwprobe::is_shared_memory_device(device.total_mib, ram_total_mib) {
                "shares system memory — measured slower than the CPU here"
            } else {
                "its own memory"
            };
            println!(
                "         {:<9} {:<40} {:>6} MiB total, {:>6} MiB free · {}",
                device.id, device.name, device.total_mib, device.free_mib, kind
            );
        }
    }

    let model_path = model
        .clone()
        .or_else(|| {
            if config.llama_cpp_model_path.trim().is_empty() {
                None
            } else {
                Some(config.llama_cpp_model_path.clone())
            }
        })
        .or_else(|| xencode_models_rs::llamacpp::resolve_gguf_model(None, None));
    let shape = model_path.as_deref().and_then(hwprobe::read_gguf_shape);
    match (&model_path, &shape) {
        (Some(path), Some(shape)) => println!(
            "model:   {path}\n         {} · {} blocks · {} KV heads · head width {} · {} MiB of \
             weights",
            if shape.name.is_empty() {
                shape.architecture.as_str()
            } else {
                shape.name.as_str()
            },
            shape.block_count,
            shape.kv_head_count,
            shape.head_dim,
            shape.file_bytes / 1024 / 1024
        ),
        (Some(path), None) => println!("model:   {path} · no GGUF geometry could be read from it"),
        (None, _) => println!(
            "model:   none given — `--model <path.gguf>`, or set llama_cpp_model_path, and the \
             cache arithmetic below has a number under it"
        ),
    }
    if let Some(shape) = &shape {
        println!(
            "cache:   {:.1} KiB per token at 16-bit, {:.1} at {} keys and {} values — this is \
             what sits on top of the weights and it grows with every token of context",
            shape.kv_kib_per_token("f16", "f16"),
            shape.kv_kib_per_token(hwprobe::CACHE_K, hwprobe::CACHE_V),
            hwprobe::CACHE_K,
            hwprobe::CACHE_V
        );
    }
    if let (Some(available_kib), Some(shape)) = (available_kib, &shape) {
        let weights_mib = shape.file_bytes / 1024 / 1024;
        println!(
            "mmap:    the weights are read off disk as they are used; {} MiB available now \
             against {} MiB of model, and a build that fills the page cache is what turns a \
             serving model into a stalling one",
            available_kib / 1024,
            weights_mib
        );
    }

    let decision = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
    let ram_total_mib = total_kib.map(|k| k / 1024).unwrap_or(0);
    let rec = hwprobe::recommend(
        &devices,
        shape.as_ref(),
        ram_total_mib,
        decision.profile.ctx_tokens(),
    );
    println!(
        "budget:  {} · the token budget layer works with a {} token window",
        decision.describe(),
        decision.profile.ctx_tokens()
    );
    println!("recommend");
    for note in &rec.notes {
        println!("         · {note}");
    }
    let flags: Vec<&str> = rec.args.iter().map(String::as_str).collect();
    println!("         flags: {}", flags.join(" "));
    println!(
        "         to keep them: xencode config set llama_cpp_args \"{}\"\n         \
         (config args are passed last, and llama-server takes the later of a repeated flag, so \
         this overrides the profile's own --ctx-size)",
        flags.join(" ")
    );
    Ok(())
}

/// Bring a GGUF file onto the machine, saying how far it has got as it goes.
///
/// A model file is the one part of running a local server that takes minutes
/// rather than seconds, which makes it both the part worth interrupting and the
/// part whose interruption should not cost the entire transfer again. The bytes
/// land in a sidecar `.part` file and the next attempt asks the server for the
/// tail of it, so re-running the command after a Ctrl-C or a closed laptop lid
/// continues where it stopped.
/// `expected_sha256` is the checksum configured for this model, if the person
/// set one. It comes from outside the transfer, which is the only way comparing
/// with it proves anything: a digest taken from the server's own answer would
/// match whatever the server chose to send.
pub(crate) async fn fetch_model(
    url: &str,
    path: &str,
    expected_sha256: Option<&str>,
) -> Result<(), String> {
    use std::io::IsTerminal;
    let free = xencode_context_rs::hwprobe::free_disk_bytes(path);
    let left_off = xencode_models_rs::partial_bytes(path);
    if left_off > 0 {
        println!(
            "  {path} is not there yet, but a stopped download is: {} of its bytes are on disk.",
            xencode_models_rs::human_bytes(left_off)
        );
        if expected_sha256.is_some() {
            println!(
                "  those bytes will be hashed along with the rest, so a prefix that is not the \
                 file this checksum describes fails the whole transfer."
            );
        }
    } else {
        println!("  {path} is not there yet; fetching it now.");
    }
    println!("  from {url}");

    let tty = io::stdout().is_terminal();
    // The callback is shared as a plain `Fn`, so the throttle keeps its own
    // state rather than borrowing one from here.
    let last_percent = std::sync::atomic::AtomicI64::new(-1);
    let on_progress = |progress: xencode_models_rs::Progress| {
        let line = progress.label();
        if tty {
            print!("\r  {line}");
            let _ = io::stdout().flush();
        } else if let Some(fraction) = progress.fraction() {
            // One line every five percent when the output is being piped or
            // saved, where a carriage return would only be clutter.
            let percent = (fraction * 100.0) as i64;
            if percent >= last_percent.load(std::sync::atomic::Ordering::Relaxed) + 5 {
                last_percent.store(percent, std::sync::atomic::Ordering::Relaxed);
                println!("  {line}");
            }
        } else {
            println!("  {line}");
        }
    };
    let result =
        xencode_models_rs::fetch_model_file(url, path, free, expected_sha256, &on_progress).await;
    if tty {
        println!();
    }
    let got = result.map_err(|e| format!("the model download did not finish: {e}"))?;
    println!(
        "  model ready: {}",
        xencode_models_rs::human_bytes(got.bytes)
    );
    if got.verified {
        println!(
            "  checksum verified: the bytes hash to what was expected ({short}…)",
            short = xencode_models_rs::short_rev(&got.sha256)
        );
    } else {
        println!(
            "  unsigned: the bytes hash to {short}…, but nothing was expected, so that number \
             describes this file and proves nothing about where it came from. Set \
             llama_cpp_model_sha256 (or run xencode models advice for a pinned checksum) to \
             make a future launch check it.",
            short = xencode_models_rs::short_rev(&got.sha256)
        );
    }
    if got.resumed_from > 0 {
        println!(
            "  {} of that came from the bytes the earlier attempt had already fetched.",
            xencode_models_rs::human_bytes(got.resumed_from)
        );
    }
    if got.resume_refused {
        println!(
            "  note: the server sent the whole file rather than the part that was still \
             missing, so an interruption here would start the transfer over."
        );
    }
    Ok(())
}

pub(crate) async fn run_llamacpp(action: LlamacppAction) -> Result<(), String> {
    fn pid_file() -> std::path::PathBuf {
        xencode_config_rs::paths::state_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("llamaserver.pid")
    }

    fn write_pid_file(path: &std::path::Path, pid: u32) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, pid.to_string()).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn read_pid_file(path: &std::path::Path) -> Option<u32> {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
    }

    fn kill_pid(pid: u32) -> Result<(), String> {
        #[cfg(windows)]
        {
            let status = std::process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;
            if !status.status.success() {
                return Err(String::from_utf8_lossy(&status.stderr).to_string());
            }
        }
        #[cfg(not(windows))]
        {
            let status = std::process::Command::new("kill")
                .args([pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;
            if !status.status.success() {
                return Err(String::from_utf8_lossy(&status.stderr).to_string());
            }
        }
        Ok(())
    }

    match action {
        LlamacppAction::Status => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            match client.ping().await {
                Ok(resp_time) => {
                    println!("llama.cpp status: healthy  ({:.3}s)", resp_time);
                    println!("  URL: {}", config.llama_cpp_url);
                    let models = client.list_models().await.unwrap_or_default();
                    for m in &models {
                        println!("  Model: {}", m.id);
                    }
                    if let Some(t) = client.timings().await.ok().flatten() {
                        if t.tokens_generated > 0 {
                            println!(
                                "  Throughput: ~{:.0} tok/s · {} tokens (last gen)",
                                t.predicted_per_second, t.tokens_generated
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("llama.cpp status: unavailable");
                    println!("  URL: {}", config.llama_cpp_url);
                    println!("  Error: {e}");
                }
            }
            Ok(())
        }
        LlamacppAction::Start { model, port, exec } => {
            // Propagated, not defaulted: this branch ends in a save, and a
            // fallback here would write a full default block over the file the
            // person's keys are in.
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            let model_path = model
                .clone()
                .unwrap_or_else(|| config.llama_cpp_model_path.clone());
            if model_path.trim().is_empty() {
                return Err(
                    "no GGUF model path set (use --model or `xencode config set llama_cpp_model_path <path>`)"
                        .to_string(),
                );
            }
            // Nothing below this line means anything without a model on disk, so
            // the fetch comes first: it is the longest step of a bring-up and the
            // only one that can be interrupted and picked up again.
            // What the file is supposed to hash to, if anyone said. Empty config
            // means nothing is pinned, which is reported as such rather than
            // treated as a check that passed.
            let expected = config.llama_cpp_model_sha256.trim().to_string();
            let expected = if expected.is_empty() {
                None
            } else {
                Some(expected.as_str())
            };
            if !std::path::Path::new(&model_path).exists() {
                let url = config.llama_cpp_model_url.trim().to_string();
                if url.is_empty() {
                    return Err(format!(
                        "no model at {model_path}, and nowhere to get it from (set config llama_cpp_model_url to the HTTPS address of the GGUF)"
                    ));
                }
                fetch_model(&url, &model_path, expected).await?;
            } else {
                // A file that is already here has not been looked at since it
                // arrived. Checking it costs one read of the file — about what
                // the loader spends opening it — and turns "the host replaced
                // this file" into a refusal before a server is started rather
                // than into wrong answers afterwards.
                let check = xencode_models_rs::check_model_file(&model_path, expected);
                match &check {
                    xencode_models_rs::FileCheck::Mismatch { expected, actual } => {
                        return Err(format!(
                            "refusing to start: {model_path} hashes to {}, not the {} this \
                             configuration expects. The file is not the one that was pinned — \
                             delete it and start again to fetch it fresh, or set \
                             llama_cpp_model_sha256 to the checksum you now want.",
                            xencode_models_rs::short_rev(actual),
                            xencode_models_rs::short_rev(expected)
                        ));
                    }
                    xencode_models_rs::FileCheck::Unreadable { reason, .. } => {
                        return Err(format!("refusing to start: {reason}"));
                    }
                    other => {
                        let label = other.label();
                        println!("  model:   {model_path} ({label})");
                        // A file with no checksum configured has nothing to be
                        // checked against, but xencode may have written down
                        // where its bytes came from when it fetched them. That
                        // record is this program's own note, so it is printed as
                        // one and not as a passing test.
                        if matches!(other, xencode_models_rs::FileCheck::Unsigned { .. }) {
                            match xencode_models_rs::read_provenance(&model_path) {
                                Some(record) => {
                                    let now = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|since| since.as_secs())
                                        .unwrap_or(0);
                                    let age_days = now.saturating_sub(record.fetched_at) / 86_400;
                                    let revision = match &record.revision {
                                        Some(revision) => format!(
                                            "at revision {}",
                                            xencode_models_rs::short_rev(revision)
                                        ),
                                        None => "from a host that named no revision".to_string(),
                                    };
                                    println!(
                                        "  record:  xencode wrote this file down {} day(s) ago, \
                                         {revision}, {}",
                                        age_days,
                                        xencode_models_rs::human_bytes(record.bytes)
                                    );
                                }
                                None => println!(
                                    "  record:  no checksum is set for this file and xencode did \
                                     not download it, so nothing here says what its bytes are \
                                     supposed to be"
                                ),
                            }
                        }
                    }
                }
            }
            let exe = find_llama_server(exec.as_deref().or(
                if config.llama_cpp_executable.is_empty() {
                    None
                } else {
                    Some(config.llama_cpp_executable.as_str())
                },
            ));
            let exe = exe.ok_or_else(|| {
                "could not find llama-server on PATH (set config llama_cpp_executable)".to_string()
            })?;

            println!("Starting llama-server: {exe}");
            println!("  model: {model_path}");
            println!("  port:  {port}");

            // Same preset the TUI's auto-start uses: the profile's flags first,
            // then what the reasoning setting asks for, then the config's own
            // `llama_cpp_args` so they have the final say. Here a reasoning
            // setting that names nothing stops the command instead of being
            // shrugged off — this server was asked for by name, so saying "off"
            // and starting a server that thinks is the one thing not to do.
            let profile = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
            let asked = xencode_models_rs::llamacpp::ServerReport {
                context_tokens: Some(profile.profile.ctx_tokens() as u32),
                slots: Some(1),
            };
            let reasoning = xencode_models_rs::llamacpp::reasoning_launch_args(
                config.llama_cpp_reasoning.as_deref(),
            )?;
            let mut preset = profile.profile.llama_cpp_args();
            preset.extend_from_slice(&reasoning);
            let mut args = xencode_models_rs::llamacpp::server_launch_args(
                &preset,
                None,
                &config.llama_cpp_args,
            );

            // Ask the machine before the server, from the server's own device
            // list and the model's own header: a model no memory here can hold
            // is said so once, in the second before anything starts, instead of
            // being launched, waited on for two minutes, and reported as a
            // timeout that never happened.
            // The window the command line will actually run with is the last
            // `--ctx-size` in it, because the config's own flags come after the
            // preset and the later value is the one `llama-server` obeys.
            let window = xencode_models_rs::ctx_size_in(&args)
                .unwrap_or_else(|| profile.profile.ctx_tokens());
            let preflight =
                xencode_context_rs::hwprobe::launch_preflight(&exe, &model_path, window, &args);
            for line in &preflight.lines {
                println!("  {line}");
            }
            if let Some(reason) = preflight.refuse {
                return Err(reason);
            }
            if let Some(shorter) = preflight.window {
                // Appended last, which is the position that decides the value
                // `llama-server` runs with.
                args.push("--ctx-size".to_string());
                args.push(shorter.to_string());
            }
            println!("  flags: {}", args.join(" "));

            let mut server = match xencode_models_rs::launch_and_wait(
                &exe,
                &model_path,
                port,
                &args,
                xencode_models_rs::Patience {
                    tries: 120,
                    gap: std::time::Duration::from_millis(500),
                },
                &|| false,
                &xencode_context_rs::hwprobe::smaller_window,
            )
            .await
            {
                xencode_models_rs::LaunchOutcome::Started { server, notes } => {
                    for note in notes {
                        println!("  {note}");
                    }
                    server
                }
                xencode_models_rs::LaunchOutcome::Failed { lines } => {
                    return Err(lines.join("\n"));
                }
                xencode_models_rs::LaunchOutcome::Cancelled => {
                    return Err("stopped before the server answered".to_string());
                }
            };

            // Save PID once the server is known to be alive: a file naming a
            // process that died during load is a `xencode llamacpp stop` that
            // reports stopping something it never started.
            let _ = write_pid_file(&pid_file(), server.pid());
            let client = LlamaCppClient::new(&server.base_url, 5);

            config.llama_cpp_url = server.base_url.clone();
            config.llama_cpp_model_path = model_path.clone();
            config.save().map_err(|e| e.to_string())?;

            println!("llama-server ready at {}", server.base_url);
            println!("Model loaded: {}", model_path);
            // A server that answers is not yet a server that has loaded its
            // model, so ask until it says what it is running as.
            let report = client
                .report_when_ready(120, std::time::Duration::from_millis(500))
                .await;
            println!(
                "{}",
                xencode_models_rs::llamacpp::settings_check_line(
                    &format!("{} preset", profile.profile.name()),
                    asked,
                    report
                )
            );
            println!("Server will keep running; use `xencode llamacpp stop` to terminate.");
            // Keep the process alive under this CLI invocation.
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(604800)).await;
                }
            });
            let _ = tokio::signal::ctrl_c().await;
            let _ = server.stop();
            let _ = std::fs::remove_file(pid_file());
            Ok(())
        }
        LlamacppAction::Stop => {
            match read_pid_file(&pid_file()) {
                Some(pid) => {
                    println!("Stopping llama-server (PID {pid})...");
                    match kill_pid(pid) {
                        Ok(()) => {
                            let _ = std::fs::remove_file(pid_file());
                            println!("Stopped");
                        }
                        Err(e) => eprintln!("Failed to stop PID {pid}: {e}"),
                    }
                }
                None => {
                    println!("No xencode-managed llama-server PID file found.");
                }
            }
            Ok(())
        }
        LlamacppAction::Load { model } => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            println!("Loading model: {model}");
            match client.load_model(&model).await {
                Ok(()) => {
                    println!("Model loaded: {model}");
                    Ok(())
                }
                Err(e) => Err(format!("failed to load model: {e}")),
            }
        }
        LlamacppAction::Unload => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            match client.unload_models().await {
                Ok(()) => {
                    println!("Model unloaded");
                    Ok(())
                }
                Err(e) => Err(format!("failed to unload model: {e}")),
            }
        }
        LlamacppAction::List => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            let models = client.list_models().await.map_err(|e| e.to_string())?;
            for m in &models {
                println!("{}", m.id);
            }
            Ok(())
        }
        LlamacppAction::SetPath { path } => {
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            config.llama_cpp_model_path = path.clone();
            config.save().map_err(|e| e.to_string())?;
            println!("llama_cpp_model_path = {path}");
            Ok(())
        }
    }
}
