//! the server, fetch and review subcommands
//! `xencode server`, `xencode fetch` and `xencode review`.

use super::*;

/// What `xencode server` decided about its bind, before any socket exists.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ServerBind {
    pub(crate) addr: std::net::SocketAddr,
    /// (cert, key) when TLS was requested.
    pub(crate) tls: Option<(PathBuf, PathBuf)>,
    /// Non-fatal condition the operator must see on startup.
    pub(crate) warning: Option<String>,
}

/// Decide how the server binds. Pure — no sockets, no filesystem.
///
/// The loopback question is an `IpAddr` property, not a string prefix:
/// `::1` is loopback and `127.0.0.53` is too, while a `starts_with("127.")`
/// check would silently refuse the former and bless lookalikes.
pub(crate) fn resolve_bind(
    host: &str,
    port: u16,
    cert: Option<&std::path::Path>,
    key: Option<&std::path::Path>,
    allow_insecure_public: bool,
) -> Result<ServerBind, String> {
    let ip: std::net::IpAddr = match host {
        "localhost" => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        other => other.parse().map_err(|_| {
            format!("invalid --host '{host}': expected an IP address or 'localhost'")
        })?,
    };
    let tls = match (cert, key) {
        (Some(c), Some(k)) => Some((c.to_path_buf(), k.to_path_buf())),
        (Some(_), None) => return Err("--cert given without --key; TLS needs both".to_string()),
        (None, Some(_)) => return Err("--key given without --cert; TLS needs both".to_string()),
        (None, None) => None,
    };
    let public_plain = tls.is_none() && !ip.is_loopback();
    if public_plain && !allow_insecure_public {
        return Err(format!(
            "refusing to bind {ip} without TLS: pass --cert and --key, \
             or --allow-insecure-public to serve plain ws:// on a public interface"
        ));
    }
    let warning = public_plain.then(|| {
        format!(
            "serving WITHOUT TLS on {ip} because --allow-insecure-public was set: \
             tokens and activity travel in clear text"
        )
    });
    Ok(ServerBind {
        addr: std::net::SocketAddr::new(ip, port),
        tls,
        warning,
    })
}

/// Where the audit log goes: `--audit-path none` disables, an explicit path
/// is taken verbatim, and the default is the directory xencode keeps its
/// records in — `~/.local/state/xencode/audit.jsonl`, or `~/.xencode/audit.jsonl`
/// for a person still on the old layout. Sessions are not repo-scoped, so
/// neither is the trail.
pub(crate) fn resolve_audit_path(audit_path: Option<&str>) -> Result<Option<PathBuf>, String> {
    match audit_path {
        Some("none") => Ok(None),
        Some(p) => Ok(Some(PathBuf::from(p))),
        None => Ok(Some(
            xencode_config_rs::paths::state_dir()
                .map_err(|e| e.to_string())?
                .join("audit.jsonl"),
        )),
    }
}

/// The server is the one command whose libraries report through `tracing`:
/// sign-ins, peers dropped for a full send buffer, plugin events, and an audit
/// log that has stopped being written. Without a subscriber those lines were
/// dropped, so the server installs one that writes to standard error at `info`
/// and above, or at whatever level `RUST_LOG` names. Only the server does this:
/// in the TUI, lines written to the terminal would break the screen.
pub(crate) fn init_server_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

pub(crate) async fn run_server(
    port: u16,
    host: String,
    cert: Option<PathBuf>,
    key: Option<PathBuf>,
    audit_path: Option<String>,
    allow_insecure_public: bool,
) -> Result<(), String> {
    use xencode_server_rs::audit::AuditSink;

    init_server_logging();

    let bind = resolve_bind(
        &host,
        port,
        cert.as_deref(),
        key.as_deref(),
        allow_insecure_public,
    )?;
    if let Some(warning) = &bind.warning {
        println!("⚠ {warning}");
    }

    let audit = resolve_audit_path(audit_path.as_deref())?;
    if let Some(path) = &audit {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
    }
    let state = Arc::new(match &audit {
        Some(path) => ServerState::with_audit(Arc::new(AuditSink::to_file(path))),
        None => ServerState::new(),
    });
    let app = xencode_server_rs::build_app_with_state(state);

    let (http, ws) = if bind.tls.is_some() {
        ("https", "wss")
    } else {
        ("http", "ws")
    };
    println!("🚀 Xencode server starting on {http}://{}", bind.addr);
    println!("   WebSocket: {ws}://{}/ws/{{session_id}}", bind.addr);
    println!(
        "   audit: {}",
        audit
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );

    match bind.tls {
        Some((cert_path, key_path)) => {
            let rustls =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                    .map_err(|e| format!("TLS configuration failed: {e}"))?;
            axum_server::bind_rustls(bind.addr, rustls)
                .serve(app.into_make_service())
                .await
                .map_err(|e| e.to_string())
        }
        None => {
            let listener = tokio::net::TcpListener::bind(bind.addr)
                .await
                .map_err(|e| e.to_string())?;
            axum::serve(listener, app)
                .await
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }
}

/// One-line image inventory for text output. Pure — unit-tested.
pub(crate) fn format_image_text(meta: &ImageMeta) -> String {
    let dims = match (meta.width, meta.height) {
        (Some(w), Some(h)) => format!("{w}x{h}"),
        _ => "vector".to_string(),
    };
    format!(
        "{} — {} {} · {} bytes",
        meta.path,
        meta.format.mime(),
        dims,
        meta.bytes
    )
}

/// Cap for `--format text` page dumps: JSON keeps the whole body, but an
/// unbounded terminal dump helps nobody. The trailer says how much was cut.
/// The number is the one the fetch layer publishes, shared with the agent's
/// `web_fetch` tool so "the same page" is the same size to both readers.
pub const FETCH_TEXT_CAP_CHARS: usize = xencode_analysis_rs::DEFAULT_TEXT_CAP_CHARS;

/// One-screen research summary for a fetched page. Pure — unit-tested.
pub(crate) fn format_fetch_text(page: &FetchedPage) -> String {
    let title = page.title.as_deref().unwrap_or("(no title)");
    let (kept, dropped) = xencode_analysis_rs::cap_chars(&page.text, FETCH_TEXT_CAP_CHARS);
    let body = if dropped > 0 {
        format!("{kept}\n…[truncated to {FETCH_TEXT_CAP_CHARS} chars — use --format json for the full text]")
    } else {
        page.text.clone()
    };
    format!("{}\n{} ({} bytes)\n\n{}", page.url, title, page.bytes, body)
}

pub(crate) async fn run_fetch(url: String, format: OutputFormat) -> Result<(), String> {
    let page = fetch_url(&url).await.map_err(|e| e.to_string())?;
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&page).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            println!("{}", format_fetch_text(&page));
        }
    }
    Ok(())
}

/// One file in a review: diff stats plus working-tree analysis. `note` is
/// set instead of `issues` when the file cannot be analyzed (deleted,
/// binary, unreadable, oversized image) — visible, never silent.
#[derive(Debug)]
pub(crate) struct ReviewedFile {
    pub(crate) path: String,
    pub(crate) added: Option<u64>,
    pub(crate) deleted: Option<u64>,
    pub(crate) issues: Vec<CodeIssue>,
    pub(crate) note: Option<String>,
}

/// Review one diff entry against the working tree.
pub(crate) fn review_file(
    root: &std::path::Path,
    diff: &xencode_context_rs::DiffFile,
) -> ReviewedFile {
    let full = root.join(&diff.path);
    let mut file = ReviewedFile {
        path: diff.path.clone(),
        added: diff.added,
        deleted: diff.deleted,
        issues: Vec::new(),
        note: None,
    };
    if is_image_path(&full) {
        match analyze_image(&full) {
            Ok(meta) => file.note = Some(format_image_text(&meta)),
            Err(e) => file.note = Some(format!("image not readable: {e}")),
        }
        return file;
    }
    match CodeAnalyzer::analyze_file(&full) {
        Ok(issues) => file.issues = issues,
        Err(e) => file.note = Some(format!("not analyzed: {e}")),
    }
    file
}

/// Where the base branch for `xencode review` came from, reported in the
/// triage header so a person knows what baseline was used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BaseSource {
    Explicit,
    OriginHead,
    InitDefaultBranch(String),
    DefaultFallback,
}

impl BaseSource {
    pub(crate) fn header_suffix(&self) -> String {
        match self {
            BaseSource::Explicit => String::new(),
            BaseSource::OriginHead => " [base resolved from origin/HEAD]".to_string(),
            BaseSource::InitDefaultBranch(name) => {
                format!(" [no remote; fell back to init.defaultBranch ({name})]")
            }
            BaseSource::DefaultFallback => " [no remote; fell back to default 'main']".to_string(),
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            BaseSource::Explicit => "explicit",
            BaseSource::OriginHead => "origin/HEAD",
            BaseSource::InitDefaultBranch(_) => "init.defaultBranch",
            BaseSource::DefaultFallback => "fallback",
        }
    }
}

/// PR-level triage view: per-file stats plus issue counts. Pure — unit-tested.
pub(crate) fn format_review_text(
    base: &str,
    suffix: Option<&str>,
    files: &[ReviewedFile],
) -> String {
    let s = suffix.unwrap_or("");
    let mut out = format!("Review of diff {base}...HEAD ({} files){s}\n", files.len());
    for file in files {
        let stats = match (file.added, file.deleted) {
            (Some(a), Some(d)) => format!("+{a} -{d}"),
            _ => "binary".to_string(),
        };
        out.push_str(&format!("\n  {} ({})", file.path, stats));
        if let Some(note) = &file.note {
            out.push_str(&format!("\n    {note}"));
        }
        if !file.issues.is_empty() {
            out.push_str(&format!("\n    {} issue(s)", file.issues.len()));
        }
    }
    out
}

/// Repo root for diff paths: the enclosing git toplevel when inside a
/// repo (diff paths are repo-relative), else the working directory.
/// Pure over `cwd` — unit-tested with a temp repo.
pub(crate) fn resolve_review_root(cwd: &std::path::Path) -> std::path::PathBuf {
    let toplevel = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    toplevel
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| cwd.to_path_buf())
}

/// Base branch for `xencode review`: explicit `--base`, else `origin/HEAD`,
/// else `init.defaultBranch` (or existing `master`), else fallback `'main'`.
pub(crate) fn resolve_review_base(
    root: &std::path::Path,
    explicit_base: Option<&str>,
) -> (String, BaseSource) {
    if let Some(b) = explicit_base {
        return (b.to_string(), BaseSource::Explicit);
    }

    let ref_exists = |r: &str| -> bool {
        std::process::Command::new("git")
            .args(["rev-parse", "--verify", "--quiet", r])
            .current_dir(root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };

    // 1. Try git symbolic-ref refs/remotes/origin/HEAD
    let origin_head = std::process::Command::new("git")
        .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(sym_ref) = origin_head {
        let short_ref = sym_ref
            .strip_prefix("refs/remotes/")
            .unwrap_or(&sym_ref)
            .to_string();
        if ref_exists(&short_ref) || ref_exists(&sym_ref) {
            return (short_ref, BaseSource::OriginHead);
        }
    }

    // 2. Try git config --get init.defaultBranch
    let default_branch = std::process::Command::new("git")
        .args(["config", "--get", "init.defaultBranch"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(branch) = default_branch {
        if ref_exists(&branch) {
            return (branch.clone(), BaseSource::InitDefaultBranch(branch));
        }
    }

    // If init.defaultBranch was unset, check if local master exists
    if ref_exists("master") {
        return (
            "master".to_string(),
            BaseSource::InitDefaultBranch("master".to_string()),
        );
    }

    // 3. Fall back to "main"
    ("main".to_string(), BaseSource::DefaultFallback)
}
