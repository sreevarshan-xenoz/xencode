//! Post-edit diagnostics from a language server (L-12).
//!
//! The cargo gate (L-7) only helps a Rust workspace; in any other language the
//! agent edits blind — nothing checks its work before the turn finishes. This
//! module talks just enough of the Language Server Protocol over stdio to open
//! the files a turn edited and read back the diagnostics the server publishes,
//! so a non-cargo workspace can fail a turn on a real compiler error the way a
//! cargo workspace fails on `cargo test`.
//!
//! Deliberately a client, not a language server host: it spawns one server,
//! performs the `initialize` / `textDocument/didOpen` handshake, collects the
//! `textDocument/publishDiagnostics` notifications for the opened files, and
//! tears the server down. There is no persistent editor session, no incremental
//! sync, and no semantic requests — only the publish path the gate needs.

use std::path::Path;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout};

/// One LSP server we know how to drive, and which language it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LspServer {
    /// Executable to spawn (must be resolvable on `PATH`).
    pub cmd: &'static str,
    pub args: &'static [&'static str],
}

/// clangd covers C and C++ out of the box and needs no project build file to
/// report syntax and type errors, which is what makes it safe to drive here.
const CLANGD: LspServer = LspServer {
    cmd: "clangd",
    args: &[
        "--background-index=false",
        "--clang-tidy=false",
        "--log=error",
    ],
};

/// The server for a file, chosen by extension, or `None` for a language we do
/// not yet drive. The `languageId` LSP wants for the same file is returned with
/// it so the caller never has to re-derive the mapping in two places.
pub fn server_for_file(path: &str) -> Option<(&'static LspServer, &'static str)> {
    let ext = Path::new(path).extension().and_then(|e| e.to_str())?;
    match ext.to_ascii_lowercase().as_str() {
        "c" => Some((&CLANGD, "c")),
        "h" => Some((&CLANGD, "c")),
        "cpp" | "cc" | "cxx" | "cppm" => Some((&CLANGD, "cpp")),
        "hpp" | "hh" | "hxx" => Some((&CLANGD, "cpp")),
        _ => None,
    }
}

/// True when `cmd` resolves on `PATH`, so the caller only ever spawns a server
/// that exists. Uses the same lookup a shell would.
fn command_exists(cmd: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    path.split(':').any(|dir| {
        let candidate = Path::new(dir).join(cmd);
        candidate.is_file()
    })
}

/// Whether an LSP gate is worth running for these files at all: the command of
/// the first edited file that maps to a server which is actually installed, or
/// `None` when nothing qualifies. Lets the caller skip the branch entirely for a
/// workspace it cannot check (only docs, only Rust) rather than reporting an
/// "unverified" that would be noise.
pub fn applies(files: &[String]) -> Option<&'static str> {
    files.iter().find_map(|f| {
        let (srv, _) = server_for_file(f)?;
        command_exists(srv.cmd).then_some(srv.cmd)
    })
}

/// How the gate should read an LSP diagnostic run, mirroring the cargo gate's
/// three-valued verdict so a missing server or a timeout is never a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspVerdict {
    /// The server answered and found no error on the edited files.
    Clean,
    /// The server reported at least one error; `report` explains each one.
    Errors,
    /// No usable answer: no server for these files, the binary is absent, the
    /// handshake failed, or the deadline passed with nothing published. Never a
    /// verdict on the edit itself.
    Unverifiable,
}

/// One diagnostic line from the server, reduced to what a model needs to fix
/// it: file, 1-based line, and message.
#[derive(Debug, Clone)]
struct Diagnostic {
    path: String,
    line: usize,
    severity: i64,
    message: String,
}

/// The outcome of a diagnostics run: the verdict plus the model-facing report
/// (empty unless `Errors`).
pub struct LspReport {
    pub verdict: LspVerdict,
    pub report: String,
}

/// Per-server file grouping: `path` is relative to `root` for display, `text`
/// is its current on-disk content.
struct OpenFile {
    uri: String,
    path: String,
    language_id: String,
    text: String,
}

/// Run one server over the edited files it covers and collect diagnostics.
///
/// `files` are workspace-relative paths the turn edited. Only files whose
/// extension maps to an *installed* server are opened; if none qualify the
/// caller gets `Unverifiable` without a spawn. The whole exchange is bounded by
/// `timeout_secs` so a wedged server cannot hang the agent turn.
pub async fn run(root: &Path, files: &[String], timeout_secs: u64) -> LspReport {
    // Group the edited files by the server that covers them. One language per
    // turn is the realistic case; multiple servers would each need their own
    // session, so the caller runs this once per server and we keep it single.
    let mut opens: Vec<OpenFile> = Vec::new();
    let mut server: Option<&'static LspServer> = None;
    for rel in files {
        let Some((srv, lang)) = server_for_file(rel) else {
            continue;
        };
        if !command_exists(srv.cmd) {
            continue;
        }
        if let Some(prev) = server {
            if prev.cmd != srv.cmd {
                // A second language in one turn: defer to the first server and
                // let the next turn pick up the rest. Mixing sessions here would
                // complicate the one thing this path must guarantee — a bounded,
                // teardown-safe run.
                continue;
            }
        } else {
            server = Some(srv);
        }
        let full = root.join(rel);
        let Ok(text) = std::fs::read_to_string(&full) else {
            continue;
        };
        opens.push(OpenFile {
            uri: file_uri(&full),
            path: rel.clone(),
            language_id: lang.to_string(),
            text,
        });
    }

    let Some(srv) = server else {
        return LspReport {
            verdict: LspVerdict::Unverifiable,
            report: String::new(),
        };
    };
    if opens.is_empty() {
        return LspReport {
            verdict: LspVerdict::Unverifiable,
            report: String::new(),
        };
    }

    let inner = diagnose(srv, root, opens, timeout_secs);
    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs.max(1)), inner).await {
        Err(_) => LspReport {
            verdict: LspVerdict::Unverifiable,
            report: String::new(),
        },
        Ok(Err(_)) => LspReport {
            verdict: LspVerdict::Unverifiable,
            report: String::new(),
        },
        Ok(Ok(diags)) => render(srv.cmd, &diags),
    }
}

/// Turn the collected diagnostics into a verdict and a model-facing report.
fn render(cmd: &str, diags: &[Diagnostic]) -> LspReport {
    let errors: Vec<&Diagnostic> = diags.iter().filter(|d| d.severity == 1).collect();
    if errors.is_empty() {
        return LspReport {
            verdict: LspVerdict::Clean,
            report: String::new(),
        };
    }
    let mut report = format!(
        "{cmd} reported {} error(s) in the file(s) you edited:\n",
        errors.len()
    );
    for d in errors.iter().take(12) {
        report.push_str(&format!("  {}:{}: {}\n", d.path, d.line, d.message));
    }
    if errors.len() > 12 {
        report.push_str(&format!("  … and {} more\n", errors.len() - 12));
    }
    report.push_str("Fix these in the same file and finish; the turn stays open until they clear.");
    LspReport {
        verdict: LspVerdict::Errors,
        report,
    }
}

async fn diagnose(
    srv: &LspServer,
    root: &Path,
    opens: Vec<OpenFile>,
    _timeout_secs: u64,
) -> Result<Vec<Diagnostic>, ()> {
    let mut child: Child = tokio::process::Command::new(srv.cmd)
        .args(srv.args)
        .current_dir(root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| ())?;

    let stdin: ChildStdin = child.stdin.take().ok_or(())?;
    let stdout: ChildStdout = child.stdout.take().ok_or(())?;
    let mut stdin = stdin;
    let mut reader = FramedRead { inner: stdout };

    let root_uri = file_uri(root);
    let init = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": { "relatedInformation": false }
                }
            }
        }
    });
    write_message(&mut stdin, &init).await?;

    // Wait for the initialize response so the server is past its own startup
    // before we push documents at it.
    loop {
        let msg = reader.next_message().await.ok_or(())?;
        if msg.get("id").and_then(Value::as_i64) == Some(1) {
            break;
        }
    }
    write_message(
        &mut stdin,
        &json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}),
    )
    .await?;

    let mut wanted: Vec<String> = opens.iter().map(|o| o.uri.clone()).collect();
    for o in &opens {
        write_message(
            &mut stdin,
            &json!({
                "jsonrpc": "2.0", "method": "textDocument/didOpen",
                "params": { "textDocument": {
                    "uri": o.uri, "languageId": o.language_id,
                    "version": 1, "text": o.text
                }}
            }),
        )
        .await?;
    }

    // Latest diagnostics per uri; clangd may publish more than once per file.
    let mut latest: std::collections::HashMap<String, Vec<Diagnostic>> =
        std::collections::HashMap::new();
    let mut reported: std::collections::HashSet<String> = std::collections::HashSet::new();
    let path_of = |uri: &str| -> String {
        opens
            .iter()
            .find(|o| o.uri == uri)
            .map(|o| o.path.clone())
            .unwrap_or_else(|| uri.to_string())
    };

    while !wanted.is_empty() {
        let msg = match reader.next_message().await {
            Some(m) => m,
            None => break,
        };
        if msg.get("method").and_then(Value::as_str) != Some("textDocument/publishDiagnostics") {
            continue;
        }
        let Some(params) = msg.get("params") else {
            continue;
        };
        let Some(uri) = params.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let mut diags = Vec::new();
        if let Some(list) = params.get("diagnostics").and_then(Value::as_array) {
            for d in list {
                let severity = d.get("severity").and_then(Value::as_i64).unwrap_or(1);
                let message = d
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                let line = d
                    .get("range")
                    .and_then(|r| r.get("start"))
                    .and_then(|s| s.get("line"))
                    .and_then(Value::as_i64)
                    .map(|l| (l as usize) + 1)
                    .unwrap_or(1);
                diags.push(Diagnostic {
                    path: path_of(uri),
                    line,
                    severity,
                    message,
                });
            }
        }
        latest.insert(uri.to_string(), diags);
        reported.insert(uri.to_string());
        // Consider a file answered once it has been published for — clangd
        // sends an empty list when a file is genuinely clean.
        wanted.retain(|u| !reported.contains(u));
    }

    // Ask the server to shut down cleanly, then drop (kill_on_drop covers a
    // server that ignores `exit`).
    let _ = write_message(
        &mut stdin,
        &json!({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": Value::Null}),
    )
    .await;
    let _ = write_message(
        &mut stdin,
        &json!({"jsonrpc": "2.0", "method": "exit", "params": Value::Null}),
    )
    .await;
    let _ = child.wait().await;

    Ok(latest.into_values().flatten().collect())
}

/// Write one `Content-Length`-framed JSON message.
async fn write_message(stdin: &mut ChildStdin, msg: &Value) -> Result<(), ()> {
    let body = serde_json::to_vec(msg).map_err(|_| ())?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin.write_all(header.as_bytes()).await.map_err(|_| ())?;
    stdin.write_all(&body).await.map_err(|_| ())?;
    stdin.flush().await.map_err(|_| ())?;
    Ok(())
}

/// Reads `Content-Length`-framed messages off the server's stdout.
struct FramedRead<R> {
    inner: R,
}

impl<R: AsyncReadExt + Unpin> FramedRead<R> {
    async fn next_message(&mut self) -> Option<Value> {
        // Headers: lines until a blank one. We only need Content-Length but must
        // consume every header line regardless of what the server sends.
        let mut content_length: Option<usize> = None;
        loop {
            let line = read_line(&mut self.inner).await?;
            if line.is_empty() {
                break;
            }
            if let Some(v) = parse_content_length(&line) {
                content_length = Some(v);
            }
        }
        let len = content_length?;
        let mut body = vec![0u8; len];
        self.inner.read_exact(&mut body).await.ok()?;
        serde_json::from_slice(&body).ok()
    }
}

/// Read one `\r\n`- or `\n`-terminated header line off the stream.
async fn read_line<R: AsyncReadExt + Unpin>(r: &mut R) -> Option<String> {
    let mut buf: Vec<u8> = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = r.read(&mut byte).await.ok()?;
        if n == 0 {
            if buf.is_empty() {
                return None;
            }
            return Some(String::from_utf8_lossy(&buf).into_owned());
        }
        if byte[0] == b'\n' {
            while buf.last() == Some(&b'\r') {
                buf.pop();
            }
            return Some(String::from_utf8_lossy(&buf).into_owned());
        }
        buf.push(byte[0]);
    }
}

fn parse_content_length(line: &str) -> Option<usize> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case("content-length") {
        v.trim().parse().ok()
    } else {
        None
    }
}

/// `file://` URI for a local path. Uses the path as-is (already absolute and
/// normalized by the caller) prefixed with the scheme.
fn file_uri(path: &Path) -> String {
    let s = path.to_string_lossy().replace(' ', "%20");
    format!("file://{s}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("xencode-lsp-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn extension_maps_to_server_and_language() {
        // C and C++ route to clangd with the right LSP languageId; a Rust file
        // (which the cargo gate already covers) and an unknown type route here.
        assert_eq!(server_for_file("a.c").map(|(_, l)| l), Some("c"));
        assert_eq!(server_for_file("a.h").map(|(_, l)| l), Some("c"));
        assert_eq!(server_for_file("a.cpp").map(|(_, l)| l), Some("cpp"));
        assert_eq!(server_for_file("a.hpp").map(|(_, l)| l), Some("cpp"));
        assert_eq!(server_for_file("A.CC").map(|(_, l)| l), Some("cpp"));
        assert_eq!(server_for_file("a.rs"), None);
        assert_eq!(server_for_file("notes.md"), None);
        assert_eq!(server_for_file("noext"), None);
    }

    #[test]
    fn content_length_header_parsed_case_insensitively() {
        assert_eq!(parse_content_length("Content-Length: 42"), Some(42));
        assert_eq!(parse_content_length("content-length:7"), Some(7));
        assert_eq!(parse_content_length("Content-Type: application/json"), None);
    }

    #[test]
    fn only_error_severity_produces_a_blocking_report() {
        let err = Diagnostic {
            path: "x.c".into(),
            line: 4,
            severity: 1,
            message: "undeclared identifier 'y'".into(),
        };
        let warn = Diagnostic {
            path: "x.c".into(),
            line: 9,
            severity: 2,
            message: "unused variable".into(),
        };
        // Warnings alone must not block the turn — only real errors do.
        let clean = render("clangd", &[warn]);
        assert_eq!(clean.verdict, LspVerdict::Clean);
        assert!(clean.report.is_empty());

        let blocked = render("clangd", &[err]);
        assert_eq!(blocked.verdict, LspVerdict::Errors);
        assert!(
            blocked.report.contains("clangd reported 1 error"),
            "{}",
            blocked.report
        );
        assert!(blocked.report.contains("x.c:4"), "{}", blocked.report);
        assert!(
            blocked.report.contains("undeclared identifier"),
            "{}",
            blocked.report
        );
    }

    #[test]
    fn file_uri_percent_encodes_spaces_only() {
        let got = file_uri(Path::new("/tmp/a dir/x.c"));
        assert_eq!(got, "file:///tmp/a dir/x.c".replace(' ', "%20"));
    }

    // The rest runs a *real* clangd over a *real* file. It is skipped only when
    // clangd is genuinely not installed — never mocked.
    #[tokio::test]
    async fn real_clangd_flags_a_type_error_that_cargo_cannot_see() {
        let root = unique_temp_dir("bad");
        std::fs::write(
            root.join("bad.c"),
            "int main(void) {\n    int x = \"not an int\";\n    return x;\n}\n",
        )
        .unwrap();
        if applies(&["bad.c".to_string()]).is_none() {
            eprintln!("clangd not installed; skipping live diagnostics test");
            std::fs::remove_dir_all(&root).unwrap();
            return;
        }
        let report = run(&root, &["bad.c".to_string()], 30).await;
        assert_eq!(
            report.verdict,
            LspVerdict::Errors,
            "expected clangd to flag the bad conversion, got {:?}",
            report.verdict
        );
        assert!(
            report.report.to_lowercase().contains("error") || report.report.contains("bad.c:2"),
            "{}",
            report.report
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn real_clangd_passes_a_clean_file() {
        let root = unique_temp_dir("good");
        std::fs::write(
            root.join("good.c"),
            "int add(int a, int b) {\n    return a + b;\n}\n",
        )
        .unwrap();
        if applies(&["good.c".to_string()]).is_none() {
            eprintln!("clangd not installed; skipping live diagnostics test");
            std::fs::remove_dir_all(&root).unwrap();
            return;
        }
        let report = run(&root, &["good.c".to_string()], 30).await;
        assert_eq!(
            report.verdict,
            LspVerdict::Clean,
            "a valid C file must not report errors: {:?}",
            report.report
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
