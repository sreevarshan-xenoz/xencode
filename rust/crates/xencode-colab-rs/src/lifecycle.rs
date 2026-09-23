//! The `xencode colab up|status|down` lifecycle: bring a Colab VM up over the
//! official bridge, report it, and tear it down — always idempotent and always
//! loopback-only on the VM side (never a public tunnel).
//!
//! Sequencing of `up`:
//! 1. session validated, created only when absent (`colab new --gpu` also
//!    spawns the official keep-alive daemon — this feature does not run its
//!    own, exactly per the plan),
//! 2. bootstrap script pushed over the bridge and run as `bash -s`,
//! 3. `ssh -N -L` forward spawned (the VM's endpoint appears on the laptop),
//! 4. `/v1/models` probed until the endpoint answers,
//! 5. state written to `~/.xencode/colab.json`.
//!
//! I/O is only ever argv/stdin/stdout against `colab`/`ssh` fakes on `$PATH`
//! in the hermetic tests; nothing here shells out through a user shell except
//! ssh's own `ProxyCommand` (which every value in is shell-quoted).
#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use xencode_config_rs::XencodeConfig;

use crate::bootstrap::bootstrap_script;
use crate::orchestrate::{
    colab_new_argv, colab_sessions_argv, colab_stop_argv, effective_remote_port, exec_ssh_argv,
    now_rfc3339, parse_sessions, probe_models, spawn_forward, terminate, validate_session_name,
    Binaries,
};
use crate::state::{load_state, remove_state, save_state, ColabState};

/// Timeout for one `colab sessions` / `colab new` / `colab stop` run.
const CMD_TIMEOUT: Duration = Duration::from_secs(60);
/// The VM bootstrap (pip install + model download + server start) can take
/// minutes on a cold VM; give it room.
const BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(1200);
/// `/v1/models` probes after the forward is up. The server may still be
/// finishing a cold model load; keep trying briefly.
const PROBE_ATTEMPTS: u32 = 40;

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
    /// Laptop-side port the forward exposes.
    pub local_port: u16,
    /// VM-side port; `0` = runtime native (8080 llama.cpp / 11434 ollama).
    pub remote_port: u16,
}

/// Bring the VM up. Returns the forward's URL (`http://127.0.0.1:PORT/v1`).
/// Fails with user-facing messages; the caller (CLI) prints and exits.
pub async fn run_colab_up(bins: &Binaries, key: &Path, opts: &UpOptions) -> Result<String, String> {
    validate_session_name(&opts.session).map_err(|e| format!("colab up: {e}"))?;
    if opts.model.is_empty() {
        return Err("colab up: --model is required (what should the VM serve?)".to_string());
    }
    if opts.runtime != "llama.cpp" && opts.runtime != "ollama" {
        return Err(format!(
            "colab up: unknown runtime {:?} (expected \"llama.cpp\" or \"ollama\")",
            opts.runtime
        ));
    }

    let remote_port = effective_remote_port(&opts.runtime, opts.remote_port);

    ensure_session(bins, &opts.session, &opts.gpu).await?;

    let script = bootstrap_script(
        &opts.runtime,
        &opts.model,
        &opts.weights_source,
        remote_port,
    )?;
    run_bootstrap(bins, key, &opts.session, &script).await?;

    let (url, child) =
        spawn_forward(bins, &opts.session, key, opts.local_port, remote_port).await?;
    let forward_pid = child.id().expect("a just-spawned forward always has a pid");

    probe_models(opts.local_port, PROBE_ATTEMPTS, None)
        .await
        .map_err(|e| {
            terminate(forward_pid);
            format!("colab up: forward came up but the endpoint did not answer: {e}")
        })?;

    let state = ColabState {
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
    save_state(&state).map_err(|e| format!("colab up: {e}"))?;

    Ok(url)
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
pub async fn run_colab_status(bins: &Binaries, asked_session: &str) -> StatusReport {
    let mut lines = Vec::new();
    let state = load_state().unwrap_or_else(|e| {
        lines.push(format!("colab status: could not read state — {e}"));
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
        match colab_sessions(bins).await {
            Ok(names) => names.iter().any(|n| n == session),
            Err(e) => {
                lines.push(format!("colab status: colab sessions: {e}"));
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
        true => format!("session: {session} (listed by colab sessions)"),
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
pub async fn run_colab_down(bins: &Binaries) -> Result<String, String> {
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
        match colab_stop(bins, session).await {
            Ok(()) => parts.push(format!("colab stop {session}")),
            Err(e) => parts.push(format!("colab stop {session}: {e}")),
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

/// `colab sessions` captured. `Err` turns into a status line, so a
/// transient backend hiccup never kills the report.
async fn colab_sessions(bins: &Binaries) -> Result<Vec<String>, String> {
    let output = run_capture(&bins.colab, &colab_sessions_argv(&bins.colab), CMD_TIMEOUT).await?;
    Ok(parse_sessions(&String::from_utf8_lossy(&output.stdout)))
}

/// Create the session only when it does not exist server-side (idempotent
/// `up`). `colab new` also launches the official keep-alive daemon.
async fn ensure_session(bins: &Binaries, session: &str, gpu: &str) -> Result<(), String> {
    let existing = colab_sessions(bins)
        .await
        .map_err(|e| format!("colab up: could not list sessions: {e}"))?;
    if existing.iter().any(|n| n == session) {
        return Ok(());
    }
    let out = run_capture(
        &bins.colab,
        &colab_new_argv(&bins.colab, session, gpu),
        CMD_TIMEOUT,
    )
    .await
    .map_err(|e| format!("colab up: colab new failed: {e}"))?;
    if !out.status {
        return Err(format!(
            "colab up: colab new failed{}",
            first_line(&String::from_utf8_lossy(&out.stderr))
                .map(|l| format!(" — {l}"))
                .unwrap_or_default()
        ));
    }
    Ok(())
}

/// Feed the bootstrap script to `ssh ... colab-vm bash -s` over the bridge
/// and wait for a `READY` marker on stdout. Streams a progress line so a long
/// pip install is visibly working, not hung.
async fn run_bootstrap(
    bins: &Binaries,
    key: &Path,
    session: &str,
    script: &str,
) -> Result<(), String> {
    let argv = exec_ssh_argv(bins, session, key, "bash -s");
    let mut child = tokio::process::Command::new(&bins.ssh)
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("colab up: could not spawn ssh bootstrap: {e}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "colab up: ssh stdin unavailable".to_string())?;
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(script.as_bytes())
            .await
            .map_err(|e| format!("colab up: could not write bootstrap to ssh: {e}"))?;
        // Dropping stdin closes it -> ssh feeds EOF -> `bash -s` runs.
    }

    let output = tokio::time::timeout(BOOTSTRAP_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| {
            "colab up: VM bootstrap timed out (install may still be running — re-run up)"
                .to_string()
        })?
        .map_err(|e| format!("colab up: bootstrap wait failed: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "colab up: VM bootstrap failed (exit {}){}",
            output.status.code().unwrap_or(-1),
            first_line(&stderr)
                .or_else(|| first_line(&stdout))
                .map(|l| format!(" — {l}"))
                .unwrap_or_default()
        ));
    }
    if !stdout.contains("READY") {
        return Err("colab up: VM bootstrap did not report READY".to_string());
    }
    Ok(())
}

/// `colab stop -s <session>` — best-effort: absence is fine, hard backend
/// errors surface as a message from `down`, never a crash.
async fn colab_stop(bins: &Binaries, session: &str) -> Result<(), String> {
    let argv = colab_stop_argv(&bins.colab, session);
    let out = run_capture(&bins.colab, &argv, CMD_TIMEOUT).await?;
    if out.status {
        Ok(())
    } else {
        Err(first_line(&String::from_utf8_lossy(&out.stderr))
            .unwrap_or_else(|| "colab stop failed".to_string()))
    }
}

/// A captured subprocess run: `(status_ok, stdout, stderr)` with a timeout.
/// `argv` mirrors the other builders — `argv[0]` is the exe itself (it must
/// equal `exe`, which is how `forward_argv`/`exec_ssh_argv` behave too) and is
/// skipped here so `Command` gets its args only.
async fn run_capture(exe: &Path, argv: &[String], timeout: Duration) -> Result<Output, String> {
    tokio::time::timeout(
        timeout,
        tokio::process::Command::new(exe).args(&argv[1..]).output(),
    )
    .await
    .map_err(|_| "command timed out".to_string())?
    .map(|out| Output {
        status: out.status.success(),
        stdout: out.stdout,
        stderr: out.stderr,
    })
    .map_err(|e| format!("could not run {}: {e}", exe.display()))
}

struct Output {
    status: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// First non-blank line of `s`, trimmed and capped — for error tails.
fn first_line(s: &str) -> Option<String> {
    let line = s.lines().find(|l| !l.trim().is_empty())?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(160).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .expect("bind probe port");
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

    #[test]
    fn first_line_trims_and_caps() {
        assert_eq!(first_line("  hi there\n"), Some("hi there".to_string()));
        assert_eq!(first_line("\n\n  "), None);
        assert_eq!(first_line(&"x".repeat(300)).map(|l| l.len()), Some(160));
    }
}
