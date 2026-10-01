//! The `up|status|down` lifecycle over any [`Backend`]: bring a machine's
//! inference server up through the bridge, report it, and tear it down —
//! always idempotent and always loopback-only on the far side (never a
//! public tunnel).
//!
//! Sequencing of `up` (identical for every backend, which is the point of
//! L-1):
//! 1. compute validated, created only when absent (`provision` also spawns
//!    whatever keep-alive the provider needs — for Colab, `colab new`'s
//!    official daemon; this feature runs none of its own),
//! 2. bootstrap script pushed over the backend's remote-command transport and
//!    run as `bash -s`,
//! 3. `ssh -N -L` forward spawned (the far endpoint appears on the laptop),
//! 4. `/v1/models` probed until the endpoint answers,
//! 5. state written to `~/.xencode/colab.json` (carrying which backend owns
//!    it, so a second backend never inherits a live Colab bridge).
//!
//! I/O is only ever argv/stdin/stdout against `colab`/`ssh` fakes on `$PATH`
//! in the hermetic tests; nothing here shells out through a user shell except
//! ssh's own `ProxyCommand` (which every value in is shell-quoted).
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use xencode_config_rs::XencodeConfig;

use crate::backend::{error_tail, Backend, TransportCmd};
use crate::bootstrap::bootstrap_script;
use crate::colab::ColabBackend;
use crate::orchestrate::{
    effective_remote_port, forward_url, now_rfc3339, probe_models, spawn_forward_cmd, terminate,
    validate_session_name, Binaries,
};
use crate::state::{load_state, remove_state, save_state, ColabState};

/// Timeout for one `colab sessions` / `colab new` / `colab stop` run lives
/// with the backend that runs them ([`crate::colab::CMD_TIMEOUT`]).
/// The VM bootstrap (runtime download + model download + load until the server
/// really serves) runs to completion before it reports `READY`. Measured live:
/// a 0.5B GGUF took ~20 s to fetch and 39 s to load on a T4; a 7-8B Q4 is
/// ~4.7 GB, so the budget has to be generous — and output is captured, so a
/// long bootstrap is silent but bounded.
const BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(2400);
/// `/v1/models` probes after the forward is up, ~30 s at 250 ms apart. The
/// bootstrap now only reports `READY` once the server serves, so this window
/// covers the forward itself — plus the Colab bridge's occasional `503` while
/// the runtime is loaded but the proxy has not caught up (seen live).
const PROBE_ATTEMPTS: u32 = 120;
/// Probes on reconnect's speculative forward-only attempt, ~5 s. Here we are
/// only asking "does the VM still serve", and a loaded runtime answers on the
/// first try — so a short window fails fast into the full repair instead of
/// spending the whole bring-up budget on a guess.
const FORWARD_ONLY_PROBE_ATTEMPTS: u32 = 20;
/// Colab gives a runtime exactly one SSH bridge, and the slot of a bridge that
/// just died takes a while to release (measured live on a free-tier T4:
/// `HTTP 429 … Already-active SSH session` survived ~45 s). Both the bootstrap
/// and the forward retry through that window instead of failing the bring-up.
const BRIDGE_ATTEMPTS: u32 = 8;
/// How long to wait between bridge attempts.
const BRIDGE_RETRY_GAP: Duration = Duration::from_secs(20);
/// How long the forward is given to prove it stayed up before it is trusted.
const FORWARD_SETTLE: Duration = Duration::from_secs(5);

/// GGUF quantization used when neither `--quant` nor `colab_quant` says:
/// small enough to fit a free-tier T4's 15 GB with a 7-8B model, good enough
/// to be worth talking to.
pub const DEFAULT_QUANT: &str = "Q4_K_M";

/// Everything `up` needs, resolved by the CLI from flags + config.
#[derive(Debug, Clone)]
pub struct UpOptions {
    /// Validated Colab session name.
    pub session: String,
    /// GPU accelerator (`colab new --gpu`): T4, L4, G4, H100, A100.
    pub gpu: String,
    /// "llama.cpp" or "ollama".
    pub runtime: String,
    /// Model repo/tag installed on the VM (HF repo id for llama.cpp, an
    /// ollama tag for ollama).
    pub model: String,
    /// Weight source ("hf" — llama.cpp only; drive/gcs are refused by the
    /// bootstrap with a fix).
    pub weights_source: String,
    /// GGUF quantization fragment to pick inside the model repo (llama.cpp
    /// only; ollama tags carry their own). Empty = "Q4_K_M".
    pub quant: String,
    /// Laptop-side port the forward exposes.
    pub local_port: u16,
    /// VM-side port; `0` = runtime native (18080 llama.cpp / 11434 ollama).
    pub remote_port: u16,
}

/// Bring the VM up. Returns the forward's URL (`http://127.0.0.1:PORT/v1`).
/// Fails with user-facing messages; the caller (CLI) prints and exits.
/// Thin wrapper: the sequencing is [`bring_up`] over the Colab backend, so a
/// second backend reuses it unchanged.
pub async fn run_colab_up(bins: &Binaries, key: &Path, opts: &UpOptions) -> Result<String, String> {
    bring_up(
        &ColabBackend::new(bins.clone(), key.to_path_buf()),
        "colab up",
        opts,
    )
    .await
}

/// The generic bring-up: validate, provision, bootstrap, forward, probe,
/// record. `op` prefixes every error (`"colab up"`, `"colab reconnect"`)
/// so the two callers' failures read as their own.
pub async fn bring_up<B: Backend>(
    backend: &B,
    op: &str,
    opts: &UpOptions,
) -> Result<String, String> {
    validate_session_name(&opts.session).map_err(|e| format!("{op}: {e}"))?;
    if opts.model.is_empty() {
        return Err(format!(
            "{op}: --model is required (what should the VM serve?)"
        ));
    }
    if opts.runtime != "llama.cpp" && opts.runtime != "ollama" {
        return Err(format!(
            "{op}: unknown runtime {:?} (expected \"llama.cpp\" or \"ollama\")",
            opts.runtime
        ));
    }

    let remote_port = effective_remote_port(&opts.runtime, opts.remote_port);

    backend
        .provision(&opts.session, &opts.gpu)
        .await
        .map_err(|e| format!("{op}: {e}"))?;

    let quant = if opts.quant.is_empty() {
        DEFAULT_QUANT
    } else {
        opts.quant.as_str()
    };
    let script = bootstrap_script(
        &opts.runtime,
        &opts.model,
        &opts.weights_source,
        quant,
        remote_port,
    )?;
    run_bootstrap(backend, op, &opts.session, &script).await?;

    let (url, child) =
        spawn_forward_ready(backend, op, &opts.session, opts.local_port, remote_port).await?;
    let forward_pid = child.id().expect("a just-spawned forward always has a pid");

    probe_models(opts.local_port, PROBE_ATTEMPTS, None)
        .await
        .map_err(|e| {
            terminate(forward_pid);
            format!("{op}: forward came up but the endpoint did not answer: {e}")
        })?;

    let state = ColabState {
        backend: Some(backend.id().to_string()),
        session: Some(opts.session.clone()),
        forward_pid: Some(forward_pid),
        keepalive_pid: None,
        local_port: Some(opts.local_port),
        remote_port: Some(remote_port),
        runtime: Some(opts.runtime.clone()),
        model: Some(opts.model.clone()),
        started_at: Some(now_rfc3339()),
        url: Some(url.clone()),
    };
    save_state(&state).map_err(|e| format!("{op}: {e}"))?;

    Ok(url)
}

/// One-key reconnect: repair a bridge whose forward died or whose VM got
/// reaped without re-asking for flags — the recorded `colab.json` decides.
///
/// 1. If a session is still listed server-side, reuse it (no `colab new`).
/// 2. If the endpoint on the recorded port already answers, we are done
///    (only the pid had died — a fast no-op reconnect).
/// 3. Re-spawn the forward and probe through it: a dead forward is the usual
///    breakage and the VM typically still serves, so this path must not pay
///    for a bootstrap (measured live: the full one re-fetches the weights).
/// 4. Only when a fresh forward reaches an empty port does the VM side get
///    rebuilt — the runtime died with the forward, or the VM was reaped.
pub async fn run_colab_reconnect(
    bins: &Binaries,
    key: &Path,
    opts: &UpOptions,
) -> Result<String, String> {
    reconnect_bridge(
        &ColabBackend::new(bins.clone(), key.to_path_buf()),
        "colab reconnect",
        opts,
    )
    .await
}

/// The generic one-key reconnect. `op` prefixes errors as in [`bring_up`].
pub async fn reconnect_bridge<B: Backend>(
    backend: &B,
    op: &str,
    opts: &UpOptions,
) -> Result<String, String> {
    let state = load_state()?
        .ok_or_else(|| format!("{op}: no colab.json — run `xencode colab up` first"))?;
    let session = opts.session.clone();
    validate_session_name(&session).map_err(|e| format!("{op}: {e}"))?;
    if opts.model.is_empty() {
        return Err(format!(
            "{op}: --model is required (what should the VM serve?)"
        ));
    }
    let remote_port = effective_remote_port(&opts.runtime, opts.remote_port);
    let local_port = opts.local_port;

    // The forward process may be gone while the VM still serves — nothing to
    // rebuild, just note the stale pid and move on.
    let endpoint_already = probe_models(local_port, 1, None).await.is_ok();
    if endpoint_already {
        let url = state.url.clone().unwrap_or_else(|| forward_url(local_port));
        return Ok(url);
    }

    // Session gone server-side (or the provider refusing to name it):
    // the VM was reaped. Recreate it like `up` would, then keep going.
    backend
        .provision(&session, opts.gpu.as_str())
        .await
        .map_err(|e| format!("{op}: {e}"))?;

    // Forward before bootstrap. The tunnel is cheap and the VM usually still
    // serves through a new one; rebuilding the VM side would re-download the
    // weights for nothing. On failure, fall through: an empty port behind a
    // live forward means the runtime itself has to come back.
    if let Ok((url, child)) =
        spawn_forward_ready(backend, op, &session, local_port, remote_port).await
    {
        let pid = child.id().expect("a just-spawned forward always has a pid");
        if probe_models(local_port, FORWARD_ONLY_PROBE_ATTEMPTS, None)
            .await
            .is_ok()
        {
            record_bridge(backend, op, pid, local_port, remote_port, opts, &url)?;
            return Ok(url);
        }
        // Release the runtime's single SSH slot for the bootstrap below.
        terminate(pid);
        tokio::time::sleep(BRIDGE_RETRY_GAP).await;
    }

    let quant = if opts.quant.is_empty() {
        DEFAULT_QUANT
    } else {
        opts.quant.as_str()
    };
    let script = bootstrap_script(
        &opts.runtime,
        &opts.model,
        &opts.weights_source,
        quant,
        remote_port,
    )?;
    run_bootstrap(backend, op, &session, &script).await?;

    let (url, child) = spawn_forward_ready(backend, op, &session, local_port, remote_port).await?;
    let forward_pid = child.id().expect("a just-spawned forward always has a pid");

    probe_models(local_port, PROBE_ATTEMPTS, None)
        .await
        .map_err(|e| {
            terminate(forward_pid);
            format!("{op}: forward came up but the endpoint did not answer: {e}")
        })?;

    record_bridge(
        backend,
        op,
        forward_pid,
        local_port,
        remote_port,
        opts,
        &url,
    )?;

    Ok(url)
}

/// Persist the bridge a reconnect rebuilt, so `status` / `down` / the next
/// reconnect all point at the live forward pid.
fn record_bridge<B: Backend>(
    backend: &B,
    op: &str,
    forward_pid: u32,
    local_port: u16,
    remote_port: u16,
    opts: &UpOptions,
    url: &str,
) -> Result<(), String> {
    save_state(&ColabState {
        backend: Some(backend.id().to_string()),
        session: Some(opts.session.clone()),
        forward_pid: Some(forward_pid),
        keepalive_pid: None,
        local_port: Some(local_port),
        remote_port: Some(remote_port),
        runtime: Some(opts.runtime.clone()),
        model: Some(opts.model.clone()),
        started_at: Some(now_rfc3339()),
        url: Some(url.to_string()),
    })
    .map_err(|e| format!("{op}: {e}"))
}

/// One line of `status` output.
#[derive(Debug)]
pub struct StatusReport {
    /// Whether the recorded forward's pid is currently alive.
    pub forward_alive: bool,
    /// Whether the session name is listed by `colab sessions` server-side.
    pub session_present: bool,
    /// Whether `/v1/models` on the forward answered 200.
    pub endpoint_ok: bool,
    /// The endpoint URL when a state was found.
    pub url: Option<String>,
    /// Human lines, in output order, always non-empty.
    pub lines: Vec<String>,
}

/// Report the bridge state. Never fails hard — everything degraded is
/// reported as a line so the CLI stays scriptable.
/// Thin wrapper over [`bridge_status`] with the Colab backend.
pub async fn run_colab_status(bins: &Binaries, asked_session: &str) -> StatusReport {
    // Status never builds a transport command, so it carries no key — the
    // empty path is never spawned from.
    bridge_status(
        &ColabBackend::new(bins.clone(), PathBuf::new()),
        asked_session,
    )
    .await
}

/// The generic status report. The backend answers the two provider questions
/// (is the session listed? why is an old bridge gone?) and the rest —
/// forward pid liveness, endpoint probe, recorded fields — is backend-free.
pub async fn bridge_status<B: Backend>(backend: &B, asked_session: &str) -> StatusReport {
    let mut lines = Vec::new();
    let state = load_state().unwrap_or_else(|e| {
        lines.push(format!(
            "{} status: could not read state — {e}",
            backend.id()
        ));
        None
    });

    let Some(state) = state else {
        lines.push("not up (no ~/.xencode/colab.json — run `xencode colab up`)".to_string());
        return StatusReport {
            forward_alive: false,
            session_present: false,
            endpoint_ok: false,
            url: None,
            lines,
        };
    };

    let url = state
        .url
        .clone()
        .or_else(|| state.local_port.map(forward_url_no_v1));
    let forward_alive = state
        .forward_pid
        .map(crate::orchestrate::pid_alive)
        .unwrap_or(false);
    let session = state.session.as_deref().unwrap_or(asked_session);
    let session_present = if session.is_empty() {
        false
    } else {
        match backend.list_sessions().await {
            Ok(names) => names.iter().any(|n| n == session),
            Err(e) => {
                lines.push(format!(
                    "{} status: {} sessions: {e}",
                    backend.id(),
                    backend.id()
                ));
                false
            }
        }
    };

    let endpoint_ok = if forward_alive {
        match state.local_port {
            Some(port) => match probe_models(port, 1, None).await {
                Ok(()) => true,
                Err(e) => {
                    lines.push(format!("colab status: endpoint check failed — {e}"));
                    false
                }
            },
            None => false,
        }
    } else {
        false
    };

    lines.push(match forward_alive {
        true => format!("forward: up (pid {})", state.forward_pid.unwrap_or(0)),
        false => "forward: down (pid not alive — run `xencode colab up --reconnect`)".to_string(),
    });
    lines.push(match session_present {
        true => format!("session: {session} (listed by {} sessions)", backend.id()),
        false => format!("session: {session} (not listed server-side)"),
    });
    lines.push(match endpoint_ok {
        true => "endpoint: /v1/models answered 200".to_string(),
        false => "endpoint: not answering".to_string(),
    });
    if let Some(u) = &url {
        lines.push(format!("url: {u}"));
    }
    if let Some(model) = &state.model {
        lines.push(format!("model: {model}"));
    }
    if let Some(ts) = &state.started_at {
        lines.push(format!("started: {ts}"));
    }
    if let Some(hint) = backend.reap_hint(state.started_at.as_deref(), endpoint_ok) {
        lines.push(hint);
    }

    StatusReport {
        forward_alive,
        session_present,
        endpoint_ok,
        url,
        lines,
    }
}

/// Tear the bridge down. Returns a summary string. Idempotent: no state, no
/// live pid, or an absent session are all "already down" style no-ops.
/// Thin wrapper over [`tear_down`] with the Colab backend.
pub async fn run_colab_down(bins: &Binaries) -> Result<String, String> {
    tear_down(&ColabBackend::new(bins.clone(), PathBuf::new())).await
}

/// The generic teardown: kill the recorded forward, release the compute
/// through the backend, clear the state.
pub async fn tear_down<B: Backend>(backend: &B) -> Result<String, String> {
    let state = match load_state()? {
        Some(s) => s,
        None => return Ok("nothing to tear down (no colab.json)".to_string()),
    };

    let mut parts = Vec::new();

    if let Some(pid) = state.forward_pid {
        if crate::orchestrate::pid_alive(pid) {
            terminate(pid);
            parts.push(format!("forward pid {pid} killed"));
        } else {
            parts.push(format!("forward pid {pid} already gone"));
        }
    }

    if let Some(session) = &state.session {
        match backend.deprovision(session).await {
            Ok(()) => parts.push(format!("{} stop {session}", backend.id())),
            Err(e) => parts.push(format!("{} stop {session}: {e}", backend.id())),
        }
    }

    remove_state()?;
    parts.push("state cleared".to_string());
    Ok(parts.join("; ").to_string())
}

/// Point the config's provider URLs at the forward so the model picker and the
/// `remote:`/llama/ollama routes see the VM. `base_url` is the forward's base
/// (`http://127.0.0.1:PORT`); the OpenAI-compatible remote root needs `/v1`.
pub fn point_config_at_forward(config: &mut XencodeConfig, runtime: &str, base_url: &str) {
    match runtime {
        "ollama" => config.ollama_url = base_url.to_string(),
        _ => config.llama_cpp_url = base_url.to_string(),
    }
    config.remote_base_url = format!("{}/v1", base_url.trim_end_matches('/'));
}

/// `http://127.0.0.1:{local_port}` — the forward's base (llama/ollama clients
/// append their own paths; the OpenAI-compatible remote adds `/v1`).
fn forward_url_no_v1(local_port: u16) -> String {
    format!("http://127.0.0.1:{local_port}")
}

/// Feed the bootstrap script to the backend's remote-command transport and
/// wait for a `READY` marker on stdout, retrying while the transport slot is
/// still held by whatever just died. Output is captured, not streamed: a slow
/// install is silent but bounded by [`BOOTSTRAP_TIMEOUT`]. What counts as
/// "still held" is the backend's call (`is_transient`).
async fn run_bootstrap<B: Backend>(
    backend: &B,
    op: &str,
    session: &str,
    script: &str,
) -> Result<(), String> {
    let cmd = backend.exec_command(session, "bash -s");
    let mut busy = String::new();
    for attempt in 1..=BRIDGE_ATTEMPTS {
        let result = bootstrap_once(&cmd, op, script).await;
        let Err(err) = result else { return Ok(()) };
        if !backend.is_transient(&err) {
            return Err(err);
        }
        // The previous bridge's slot is still held; wait it out and redo the
        // whole push (the script is idempotent: installs skip, downloads resume).
        busy = err;
        if attempt < BRIDGE_ATTEMPTS {
            tokio::time::sleep(BRIDGE_RETRY_GAP).await;
        }
    }
    Err(format!(
        "{op}: the runtime allows one SSH bridge and its slot never freed — {busy}"
    ))
}

async fn bootstrap_once(cmd: &TransportCmd, op: &str, script: &str) -> Result<(), String> {
    let mut child = tokio::process::Command::new(&cmd.exe)
        .args(cmd.argv.iter().skip(1))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{op}: could not spawn ssh bootstrap: {e}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("{op}: ssh stdin unavailable"))?;
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(script.as_bytes())
            .await
            .map_err(|e| format!("{op}: could not write bootstrap to ssh: {e}"))?;
        // Dropping stdin closes it -> ssh feeds EOF -> `bash -s` runs.
    }

    let output = tokio::time::timeout(BOOTSTRAP_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| {
            format!("{op}: VM bootstrap timed out (install may still be running — re-run up)")
        })?
        .map_err(|e| format!("{op}: bootstrap wait failed: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "{op}: VM bootstrap failed (exit {}){}",
            output.status.code().unwrap_or(-1),
            error_tail(&stderr)
                .or_else(|| error_tail(&stdout))
                .map(|t| format!(" — {t}"))
                .unwrap_or_default()
        ));
    }
    if !stdout.contains("READY") {
        return Err(format!("{op}: VM bootstrap did not report READY"));
    }
    Ok(())
}

/// Spawn the forward, tolerating a transport slot that frees slowly: the
/// process exits immediately when the far side refuses, so a forward that is
/// already dead after [`FORWARD_SETTLE`] is re-spawned until the slot frees.
async fn spawn_forward_ready<B: Backend>(
    backend: &B,
    op: &str,
    session: &str,
    local_port: u16,
    remote_port: u16,
) -> Result<(String, tokio::process::Child), String> {
    let cmd = backend.forward_command(session, local_port, remote_port);
    let mut last = String::from("forward exited before it settled");
    for attempt in 1..=BRIDGE_ATTEMPTS {
        let (url, mut child) = spawn_forward_cmd(&cmd, local_port).await?;
        tokio::time::sleep(FORWARD_SETTLE).await;
        match child.try_wait() {
            Ok(None) => return Ok((url, child)),
            Ok(Some(status)) => last = format!("forward exited ({status})"),
            Err(e) => last = format!("forward could not be waited on: {e}"),
        }
        if attempt < BRIDGE_ATTEMPTS {
            tokio::time::sleep(BRIDGE_RETRY_GAP).await;
        }
    }
    Err(format!(
        "{op}: the ssh forward never stayed up — {last} (the far side serves one SSH bridge; close any other bridge to it and retry)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrate::pid_alive;
    use crate::testutil::{temp_dir, with_env, write_script};

    /// A fake `colab` that answers `sessions`, records `new`/`stop` calls, and
    /// flips a marker file so tests can assert the session was created/stopped.
    /// `colab new` argv is `new --gpu <gpu> -s <session>`, so `$1` is the
    /// subcommand, `$3` the gpu, `$5` the session.
    fn fake_colab(dir: &Path) {
        write_script(
            dir,
            "colab",
            r#"#!/bin/sh
export PATH=/usr/bin:/bin
marker="$MARKER"
mkdir -p "$marker"
echo "$*" >> "$marker/calls"
case "$1" in
  sessions)
    if [ -f "$marker/session-online" ]; then
      echo "[$SESSION_NAME] http://colab-fake.web.app | Hardware: T4 | Shape: STANDARD | Variant: GPU"
    else
      echo "[colab] No active sessions found on server."
    fi
    ;;
  new)
    echo "$3 $5" > "$marker/new-args"
    : > "$marker/session-online"
    echo "[colab] Session ready."
    ;;
  stop)
    rm -f "$marker/session-online"
    echo "[colab] Session terminated."
    ;;
  *) exit 1 ;;
esac
"#,
        );
    }

    /// A fake `ssh`: `-N` (the forward) records its pid then Sleeps forever;
    /// anything else (the bootstrap exec) prints READY and exits. The forward
    /// must mirror the real argv (all values before `root@colab` are
    /// options/arguments; nothing extra matters to the fakes).
    fn fake_ssh(dir: &Path) {
        write_script(
            dir,
            "ssh",
            r#"#!/bin/sh
export PATH=/usr/bin:/bin
marker="$MARKER"
echo "$*" >> "$marker/ssh-calls"
case "$1" in
  -N)
    echo "$$" > "$marker/forward-pid"
    while [ : ]; do sleep 1; done
    ;;
  *)
    echo "READY 8080"
    ;;
esac
"#,
        );
    }

    /// A tiny HTTP responder on `port` answering `/v1/models` with 200 — what
    /// `probe_models` needs to see. Serves a handful of connections then stops.
    async fn serve_v1_models(port: u16) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = bind_v1_models(port).await;
        for _ in 0..5 {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut buf = [0u8; 256];
            let _ = sock.read(&mut buf).await;
            let _ = sock
                .write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            let _ = sock.shutdown().await;
        }
    }

    /// Bind the probe responder on `port` and return the listener: the binding
    /// is resolved before the caller proceeds, so a fast-path probe (single
    /// attempt) never races a lazy `tokio::spawn` inside the responder.
    async fn bind_v1_models(port: u16) -> tokio::net::TcpListener {
        tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .expect("bind probe port")
    }

    /// Serve the probe responder on an already-bound listener (owning the
    /// accept loop so the caller can hand the listener to a spawned task).
    async fn serve_bound(listener: tokio::net::TcpListener) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for _ in 0..5 {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut buf = [0u8; 256];
            let _ = sock.read(&mut buf).await;
            let _ = sock
                .write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            let _ = sock.shutdown().await;
        }
    }

    fn marker(session: &str) -> String {
        format!("/tmp/xencode-colab-life-{}-{}", session, std::process::id())
    }

    fn bins_in(dir: &Path) -> Binaries {
        Binaries {
            colab: dir.join("colab"),
            ssh: dir.join("ssh"),
        }
    }

    fn opts_for(session: &str, runtime: &str, local_port: u16) -> UpOptions {
        UpOptions {
            session: session.to_string(),
            gpu: "T4".to_string(),
            runtime: runtime.to_string(),
            model: if runtime == "ollama" {
                "qwen2.5:7b".to_string()
            } else {
                "Qwen/Qwen2.5-7B-Instruct-GGUF".to_string()
            },
            weights_source: "hf".to_string(),
            quant: String::new(),
            local_port,
            remote_port: 0,
        }
    }

    /// The fake `ssh -N` writes `$marker/ssh-calls` asynchronously after the
    /// parent's `spawn_forward` returns; the endpoint probe answers faster than
    /// that write lands. Poll briefly for the forward's argv line.
    async fn wait_for_ssh(argv_marker: &str) -> bool {
        for _ in 0..50 {
            if let Ok(calls) = std::fs::read_to_string(argv_marker) {
                if calls.contains("-N") {
                    return true;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        }
        false
    }

    async fn kill_forward(state: &ColabState) {
        if let Some(pid) = state.forward_pid {
            terminate(pid);
        }
    }

    #[tokio::test]
    async fn raw_probe_against_raw_listener_roundtrips() {
        // Bare TCP echo of the /v1/models probe against a minimal HTTP/1.0
        // responder — isolates probe_models from the fake-ssh plumbing.
        use crate::orchestrate::probe_models as pm;
        let _srv = tokio::spawn(async {
            let l = tokio::net::TcpListener::bind(("127.0.0.1", 18001))
                .await
                .unwrap();
            let (mut s, _) = l.accept().await.unwrap();
            let mut b = [0u8; 512];
            let _ = tokio::io::AsyncReadExt::read(&mut s, &mut b).await;
            let _ = tokio::io::AsyncWriteExt::write_all(
                &mut s,
                b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await;
        });
        match pm(18001, 3, Some(std::time::Duration::from_secs(2))).await {
            Ok(()) => eprintln!("PROBE OK"),
            Err(e) => eprintln!("PROBE ERR: {e}"),
        }
    }

    #[tokio::test]
    async fn up_creates_the_session_bootstraps_and_writes_state() {
        let session = "life1";
        let bin_dir = temp_dir(&format!("up-{session}"));
        let xcode_dir = temp_dir(&format!("up-cfg-{session}"));
        fake_colab(&bin_dir);
        fake_ssh(&bin_dir);
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        let srv = tokio::spawn(serve_v1_models(18000));
        let mut opts = opts_for(session, "llama.cpp", 18000);
        opts.remote_port = 8080; // explicit override path
        let url = run_colab_up(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect("up succeeds against fakes");
        assert_eq!(url, "http://127.0.0.1:18000/v1");

        let state = ColabState::load()
            .expect("state loaded")
            .expect("state exists");
        let summary = format!("{:?}", state);
        assert_eq!(state.session.as_deref(), Some(session));
        assert_eq!(
            state.backend.as_deref(),
            Some("colab"),
            "up records which backend owns the bridge: {summary}"
        );
        assert_eq!(state.local_port, Some(18000));
        assert_eq!(state.remote_port, Some(8080), "explicit override kept");
        assert_eq!(state.runtime.as_deref(), Some("llama.cpp"));
        assert_eq!(state.url.as_deref(), Some("http://127.0.0.1:18000/v1"));
        assert!(state.started_at.is_some(), "started_at stamped: {summary}");

        let mdir = marker(session);
        let calls = std::fs::read_to_string(format!("{mdir}/calls")).expect("calls");
        assert!(
            calls.contains("sessions"),
            "existence checked first: {calls}"
        );
        assert!(
            calls.contains("new --gpu T4 -s life1"),
            "created with gpu+s: {calls}"
        );
        assert!(
            std::fs::read_to_string(format!("{mdir}/new-args")).expect("new-args") == "T4 life1\n",
            "fake colab saw --gpu T4 -s life1"
        );
        assert!(
            wait_for_ssh(&format!("{mdir}/ssh-calls")).await,
            "forward spawned with -N"
        );
        assert!(
            std::fs::read_to_string(format!("{mdir}/forward-pid"))
                .expect("fwd")
                .trim()
                .parse::<u32>()
                .expect("pid")
                == state.forward_pid.expect("state pid"),
            "recorded pid is the forward process"
        );

        kill_forward(&state).await;
        srv.abort();
    }

    #[tokio::test]
    async fn up_reuses_an_existing_session_and_resolves_runtime_port() {
        let session = "life2";
        let bin_dir = temp_dir(&format!("up-{session}"));
        let xcode_dir = temp_dir(&format!("up-cfg-{session}"));
        fake_colab(&bin_dir);
        fake_ssh(&bin_dir);
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        // Session already exists server-side.
        std::fs::create_dir_all(marker(session)).expect("marker dir");
        std::fs::write(format!("{}/session-online", marker(session)), "yes").expect("online");

        let srv = tokio::spawn(serve_v1_models(18010));
        let opts = opts_for(session, "ollama", 18010); // remote_port 0 -> 11434
        run_colab_up(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect("up reuses session");

        let state = ColabState::load().expect("state").expect("exists");
        assert_eq!(
            state.remote_port,
            Some(11434),
            "ollama native port resolved"
        );
        assert_eq!(state.runtime.as_deref(), Some("ollama"));

        let calls = std::fs::read_to_string(format!("{}/calls", marker(session))).expect("calls");
        assert!(
            !calls.contains("new "),
            "no colab new when present: {calls}"
        );

        kill_forward(&state).await;
        srv.abort();
    }

    #[tokio::test]
    async fn reconnect_errors_without_colab_json() {
        let session = "life6";
        let bin_dir = temp_dir(&format!("rc-{session}"));
        let xcode_dir = temp_dir(&format!("rc-cfg-{session}"));
        let _g = with_env(&bin_dir, &xcode_dir);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        let opts = opts_for(session, "llama.cpp", 18030);
        let err = run_colab_reconnect(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect_err("reconnect without state must fail");
        assert!(
            err.contains("no colab.json"),
            "error names the missing state: {err}"
        );
        assert!(err.contains("colab up"), "error suggests the fix: {err}");
    }

    #[tokio::test]
    async fn reconnect_noops_when_the_endpoint_is_already_serving() {
        let session = "life7";
        let bin_dir = temp_dir(&format!("rc-{session}"));
        let xcode_dir = temp_dir(&format!("rc-cfg-{session}"));
        fake_colab(&bin_dir);
        fake_ssh(&bin_dir);
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        // A state whose forward pid is dead but whose endpoint already serves.
        let state = ColabState {
            backend: Some("colab".to_string()),
            session: Some(session.to_string()),
            forward_pid: Some(3), // unlikely to be alive
            keepalive_pid: None,
            local_port: Some(18040),
            remote_port: Some(8080),
            runtime: Some("llama.cpp".to_string()),
            model: Some("Qwen/Qwen2.5-7B-Instruct-GGUF".to_string()),
            started_at: Some("2099-01-02T03:04:05Z".to_string()),
            url: Some("http://127.0.0.1:18040/v1".to_string()),
        };
        save_state(&state).expect("seed state");

        // Bind synchronously (await) so the fast-path single probe has a live
        // listener by the time reconnect runs — no bind/accept race.
        let listener = bind_v1_models(18040).await;
        let srv = tokio::spawn(serve_bound(listener));
        let opts = opts_for(session, "llama.cpp", 18040);
        let url = run_colab_reconnect(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect("reconnect no-ops on a live endpoint");
        assert_eq!(url, "http://127.0.0.1:18040/v1");

        // The fast path must not touch colab/session/bootstrap at all.
        let mdir = marker(session);
        std::thread::sleep(std::time::Duration::from_millis(100));
        let calls = std::fs::read_to_string(format!("{mdir}/calls")).unwrap_or_default();
        assert!(
            calls.is_empty(),
            "no colab calls on no-op reconnect: {calls}"
        );
        assert!(
            !std::path::Path::new(&format!("{mdir}/ssh-calls")).exists(),
            "no ssh forward re-spawn on no-op reconnect"
        );

        srv.abort();
    }

    #[tokio::test]
    async fn reconnect_recovers_the_forward_when_only_the_pid_died() {
        let session = "life8";
        let bin_dir = temp_dir(&format!("rc-{session}"));
        let xcode_dir = temp_dir(&format!("rc-cfg-{session}"));
        fake_colab(&bin_dir);
        fake_ssh(&bin_dir);
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        // Session survives server-side, so reconnect must not `colab new`.
        std::fs::create_dir_all(marker(session)).expect("marker dir");
        std::fs::write(format!("{}/session-online", marker(session)), "yes").expect("online");

        // Stale state with a dead forward pid and no live endpoint.
        let state = ColabState {
            backend: Some("colab".to_string()),
            session: Some(session.to_string()),
            forward_pid: Some(3),
            keepalive_pid: None,
            local_port: Some(18050),
            remote_port: Some(8080),
            runtime: Some("llama.cpp".to_string()),
            model: Some("Qwen/Qwen2.5-7B-Instruct-GGUF".to_string()),
            started_at: Some("2099-01-02T03:04:05Z".to_string()),
            url: Some("http://127.0.0.1:18050/v1".to_string()),
        };
        save_state(&state).expect("seed state");

        // The endpoint comes up only *after* the dead forward would have been
        // rebuilt — so the initial single-attempt probe fails (reconnect must
        // not take the fast path), then the retry probe succeeds.
        let srv = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            serve_v1_models(18050).await;
        });
        let opts = opts_for(session, "llama.cpp", 18050);
        let url = run_colab_reconnect(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect("reconnect re-spawns the forward");
        assert_eq!(url, "http://127.0.0.1:18050/v1");

        let calls = std::fs::read_to_string(format!("{}/calls", marker(session))).expect("calls");
        assert!(
            !calls.contains("new "),
            "no colab new when session survives: {calls}"
        );
        let argv_calls = wait_for_ssh(&format!("{}/ssh-calls", marker(session))).await;
        assert!(argv_calls, "forward re-spawned");
        // Forward before bootstrap: the VM still served, so rebuilding it would
        // have re-fetched the weights for nothing.
        let pushed =
            std::fs::read_to_string(format!("{}/ssh-calls", marker(session))).expect("ssh calls");
        assert!(
            !pushed.contains("bash -s"),
            "reconnect must not re-bootstrap a serving VM: {pushed}"
        );

        let new_state = ColabState::load().expect("state").expect("exists");
        assert!(
            new_state.forward_pid.is_some_and(|pid| pid != 3),
            "forward pid refreshed, got {:?}",
            new_state.forward_pid
        );
        assert!(new_state.forward_pid.is_some_and(pid_alive));

        kill_forward(&new_state).await;
        srv.abort();
    }

    #[tokio::test]
    async fn up_refuses_unsafe_session_names() {
        let bin_dir = temp_dir("up-life3");
        let xcode_dir = temp_dir("up-cfg-life3");
        let _g = with_env(&bin_dir, &xcode_dir);
        let mut opts = opts_for("ok", "llama.cpp", 18020);
        opts.session = "evil; rm -rf /".to_string();
        let err = run_colab_up(&bins_in(&bin_dir), Path::new("/nope"), &opts)
            .await
            .expect_err("unsafe name rejected");
        assert!(err.contains("invalid session name"));
    }

    #[tokio::test]
    async fn up_fails_cleanly_when_bootstrap_does_not_report_ready() {
        let session = "life6";
        let bin_dir = temp_dir(&format!("up-{session}"));
        let xcode_dir = temp_dir(&format!("up-cfg-{session}"));
        fake_colab(&bin_dir);
        // An ssh that starts the forward but never prints READY.
        write_script(
            &bin_dir,
            "ssh",
            r#"#!/bin/sh
export PATH=/usr/bin:/bin
marker="$MARKER"
case "$1" in
  -N)
    echo "$$" > "$marker/forward-pid"
    while [ : ]; do sleep 1; done
    ;;
  *) echo "installing... (no READY)"; exit 1 ;;
esac
"#,
        );
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);
        let key = xcode_dir.join("colab_ed25519");
        std::fs::write(&key, "key").expect("write key");

        let opts = opts_for(session, "llama.cpp", 18030);
        let err = run_colab_up(&bins_in(&bin_dir), &key, &opts)
            .await
            .expect_err("bootstrap failure surfaced");
        assert!(
            err.contains("bootstrap") || err.contains("READY"),
            "helpful error: {err}"
        );
    }

    #[tokio::test]
    async fn down_kills_forward_stops_session_and_clears_state() {
        let session = "life4";
        let bin_dir = temp_dir(&format!("down-{session}"));
        let xcode_dir = temp_dir(&format!("down-cfg-{session}"));
        fake_colab(&bin_dir);
        fake_ssh(&bin_dir);
        let _g = with_env(&bin_dir, &xcode_dir);
        std::env::set_var("MARKER", marker(session));
        std::env::set_var("SESSION_NAME", session);

        let mdir = marker(session);
        std::fs::create_dir_all(&mdir).expect("marker dir");
        // Fake a live forward (our own sleep) + an online session.
        let mut sleep_child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("sleep");
        std::fs::write(format!("{mdir}/session-online"), "yes").expect("online");
        let state = ColabState {
            backend: Some("colab".to_string()),
            session: Some(session.to_string()),
            forward_pid: Some(sleep_child.id()),
            keepalive_pid: None,
            local_port: Some(18000),
            remote_port: Some(8080),
            runtime: Some("llama.cpp".to_string()),
            model: Some("m".to_string()),
            started_at: None,
            url: Some("http://127.0.0.1:18000/v1".to_string()),
        };
        state.save().expect("save state");

        let summary = run_colab_down(&bins_in(&bin_dir))
            .await
            .expect("down succeeds");
        assert!(summary.contains("killed"), "forward killed: {summary}");
        assert!(summary.contains("colab stop"), "session stopped: {summary}");
        assert!(
            !Path::new(&format!("{mdir}/session-online")).exists(),
            "colab stop cleared the session marker"
        );
        let status = sleep_child.wait().expect("reap forward");
        assert!(!status.success(), "forward terminated: {status:?}");
        assert!(ColabState::load().expect("load").is_none(), "state cleared");
    }

    #[tokio::test]
    async fn down_with_no_state_is_a_noop() {
        let bin_dir = temp_dir("down-life5");
        let xcode_dir = temp_dir("down-cfg-life5");
        let _g = with_env(&bin_dir, &xcode_dir);
        let summary = run_colab_down(&bins_in(&bin_dir)).await.expect("noop down");
        assert!(summary.contains("nothing to tear down"));
    }

    #[test]
    fn point_config_at_forward_sets_the_runtime_url_and_v1() {
        let mut config = XencodeConfig::default();
        point_config_at_forward(&mut config, "llama.cpp", "http://127.0.0.1:18000");
        assert_eq!(config.llama_cpp_url, "http://127.0.0.1:18000");
        assert_eq!(config.remote_base_url, "http://127.0.0.1:18000/v1");

        let mut config = XencodeConfig::default();
        point_config_at_forward(&mut config, "ollama", "http://127.0.0.1:18000");
        assert_eq!(config.ollama_url, "http://127.0.0.1:18000");
        assert_eq!(config.remote_base_url, "http://127.0.0.1:18000/v1");
    }

    #[test]
    fn forward_url_no_v1_is_the_base() {
        assert_eq!(forward_url_no_v1(18000), "http://127.0.0.1:18000");
    }
}
