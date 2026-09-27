//! Bringing a model file onto the machine, and being able to stop halfway.
//!
//! A GGUF for a small model is a few hundred megabytes; for a large one it is
//! tens of gigabytes. That makes the download both the slowest step of bringing
//! a local server up and the one most likely to be interrupted — by a dropped
//! connection, by a laptop lid, by the person who started it deciding the file
//! they asked for is the wrong one. A downloader that restarts from zero on
//! every attempt turns each of those into throwing away the whole transfer.
//!
//! So the bytes go to `<target>.part` and are asked for with a `Range`, which
//! lets an interrupted file be continued from where it stopped. The finished
//! file is moved into place only at the end, so a path that exists always holds
//! a complete file and nothing downstream has to tell a truncated download apart
//! from a model.
//!
//! The size is priced against the filesystem before a byte is written, because
//! finding out at 90 % that the disk was never going to hold the file wastes
//! every minute of transfer that preceded it.
//!
//! ## What a checksum does and does not prove
//!
//! When the caller supplies an expected SHA256, the bytes are hashed as they
//! arrive and compared at the end, and a file that does not match is thrown away
//! rather than renamed into place. That comparison proves the transfer was
//! faithful to a number that came from *outside* the transfer — the shipped
//! advice table, or a checksum the person set themselves.
//!
//! It does not prove where the bytes came from. A checksum computed after the
//! fact, from the same server's own answer, is circular: whatever the host
//! serves would be declared genuine. That is why the pin lives in a table dated
//! and assembled separately (see [`crate::advice`]) and why the file's
//! provenance sidecar records `verified: false` when no outside checksum was
//! given, instead of quietly reporting the digest it just observed as if it
//! were a certificate.

use std::io::Write;
use std::path::Path;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How often progress is reported while the bytes are moving. Every chunk would
/// mean tens of thousands of updates for one model file, which is noise in a
/// terminal and a flooded queue in the TUI.
const REPORT_BYTES: u64 = 4 * 1024 * 1024;

/// How far a download has got, as reported to whoever is watching it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes on disk for this file now, including whatever an earlier attempt
    /// had already managed to write.
    pub received: u64,
    /// The size the server said the whole file is, once it has said.
    pub total: Option<u64>,
}

impl Progress {
    /// `None` while the total is unknown, since "a third of the way" is not
    /// answerable without knowing the length.
    pub fn fraction(&self) -> Option<f64> {
        let total = self.total?;
        if total == 0 {
            return None;
        }
        Some(self.received.min(total) as f64 / total as f64)
    }

    /// One line saying how far it has got, in the MiB every other memory figure
    /// in xencode is quoted in.
    pub fn label(&self) -> String {
        match self.total {
            Some(total) => format!(
                "{} of {} ({:.0} %)",
                human_bytes(self.received),
                human_bytes(total),
                self.fraction().unwrap_or(0.0) * 100.0
            ),
            None => format!("{} so far", human_bytes(self.received)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadError {
    /// The file cannot fit where it was asked to go. Nothing was written and
    /// nothing was created — not even the directory it would have lived in.
    NoRoom {
        path: String,
        /// Bytes the file will occupy once it is complete.
        needed_bytes: u64,
        /// Bytes the filesystem reports it can take.
        free_bytes: u64,
    },
    /// Anything else that stopped the transfer. A partial file is usually left
    /// behind on this path, so the same command started again continues rather
    /// than restarts.
    Failed(String),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::NoRoom {
                path,
                needed_bytes,
                free_bytes,
            } => write!(
                f,
                "the file is {} and the disk holding {} has {} free",
                human_bytes(*needed_bytes),
                Path::new(path).display(),
                human_bytes(*free_bytes)
            ),
            DownloadError::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DownloadError {}

/// A byte count in the unit a person reads it in: MiB below a GiB, GiB above.
/// Public because the download is not the only thing worth sizing in words.
pub fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes / GIB)
    } else {
        format!("{:.1} MiB", bytes / MIB)
    }
}

/// What a finished download left on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Downloaded {
    /// The path the file was asked for — not the `.part` it travelled through.
    pub path: String,
    pub bytes: u64,
    /// Bytes this call did not have to fetch because they were already there.
    /// Zero means either a fresh start or a partial file that had to be thrown
    /// away; `resume_refused` says which.
    pub resumed_from: u64,
    /// The server sent the whole file despite the range request, so the partial
    /// download was discarded and this transfer cannot be continued later.
    pub resume_refused: bool,
    /// Lowercase hex SHA256 of the bytes that landed, computed here from the
    /// transfer itself. Always available: hashing a few hundred megabytes costs
    /// far less than fetching them did.
    pub sha256: String,
    /// Whether that digest was compared against a checksum that came from
    /// somewhere other than this transfer. `false` means the digest above is an
    /// observation, not a proof.
    pub verified: bool,
    /// The repository revision the host said it served, when it says —
    /// huggingface.co answers with `x-repo-commit`. Recorded, not trusted: the
    /// server writes its own answer to this header.
    pub revision: Option<String>,
}

/// What a model file on disk records about where it came from. Written next to
/// the file as `<path>.provenance.json` once the transfer is complete, so the
/// question "where did this come from" is answerable later without asking the
/// network again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The URL the bytes were fetched from — the pinned one, when the caller
    /// asked for a revision rather than a branch.
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    pub revision: Option<String>,
    /// `true` only when `sha256` was checked against an expected value supplied
    /// by the caller. See the module docs: a digest that was not compared with
    /// anything external proves nothing about its source.
    pub verified: bool,
    /// Seconds since the Unix epoch when the file was completed.
    pub fetched_at: u64,
}

impl Provenance {
    /// A one-line description for a panel or a CLI answer.
    pub fn label(&self) -> String {
        let mut line = format!(
            "{} · {}",
            human_bytes(self.bytes),
            if self.verified {
                "checksum verified"
            } else {
                "checksum unverified"
            }
        );
        if let Some(revision) = &self.revision {
            line.push_str(&format!(" · revision {}", short_rev(revision)));
        }
        line
    }
}

/// The first eight characters of a commit, the way git quotes them.
pub fn short_rev(revision: &str) -> &str {
    revision.get(..8).unwrap_or(revision)
}

/// The verdict on a model file that is already on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileCheck {
    /// The bytes hash to the checksum they were pinned to.
    Verified { sha256: String },
    /// The bytes hash to something else. The file is not the one that was asked
    /// for, whether the host changed it or the disk did.
    Mismatch { expected: String, actual: String },
    /// The file was hashed but no checksum was configured, so nothing was
    /// proven. Not a failure — an honest absence of information.
    Unsigned { sha256: String },
    /// The file could not be read at all.
    Unreadable { path: String, reason: String },
}

impl FileCheck {
    /// Whether a local server should be allowed to open this file.
    pub fn is_verified(&self) -> bool {
        matches!(self, FileCheck::Verified { .. })
    }

    /// Short text for a status line or a launch refusal.
    pub fn label(&self) -> String {
        match self {
            FileCheck::Verified { sha256 } => format!("verified ({})", short_rev(sha256)),
            FileCheck::Mismatch { expected, actual } => format!(
                "does not match its checksum: expected {}, got {}",
                short_rev(expected),
                short_rev(actual)
            ),
            FileCheck::Unsigned { .. } => "unsigned".to_string(),
            FileCheck::Unreadable { reason, .. } => format!("unreadable: {reason}"),
        }
    }
}

/// Where a finished download's provenance is stored.
fn provenance_path(path: &str) -> String {
    format!("{path}.provenance.json")
}

/// The provenance recorded for this file, `None` if there is none or it cannot
/// be read — which is the normal state for a model that was placed on disk by
/// hand or by another tool.
pub fn read_provenance(path: &str) -> Option<Provenance> {
    let text = std::fs::read_to_string(provenance_path(path)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Read a file and hash it. Used before a server is started, so a file that
/// stopped matching its checksum is said so rather than loaded.
pub fn sha256_file(path: &str) -> Result<String, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| format!("{path}: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Hash `path` and compare it with `expected_sha256` (which may be `None`,
/// meaning nobody pinned this file). Reading the whole file costs about as much
/// as the loader opening it will anyway, so this is not a tax on top.
pub fn check_model_file(path: &str, expected_sha256: Option<&str>) -> FileCheck {
    let actual = match sha256_file(path) {
        Ok(digest) => digest,
        Err(reason) => {
            return FileCheck::Unreadable {
                path: path.to_string(),
                reason,
            }
        }
    };
    match expected_sha256 {
        Some(expected) if digest_matches(expected, &actual) => {
            FileCheck::Verified { sha256: actual }
        }
        Some(expected) => FileCheck::Mismatch {
            expected: expected.trim().to_lowercase(),
            actual,
        },
        None => FileCheck::Unsigned { sha256: actual },
    }
}

/// Whether a configured checksum names these bytes. Tolerates the ways a person
/// copies a digest: upper case, a `sha256-` prefix, surrounding whitespace.
fn digest_matches(expected: &str, actual: &str) -> bool {
    let expected = expected
        .trim()
        .trim_start_matches("sha256-")
        .trim_start_matches("SHA256-")
        .to_lowercase();
    !expected.is_empty() && expected == actual
}

/// Render a digest as lowercase hex.
fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        s.push(HEX[(byte >> 4) as usize] as char);
        s.push(HEX[(byte & 0x0f) as usize] as char);
    }
    s
}

/// Where the bytes of an unfinished download live.
fn partial_path(path: &str) -> String {
    format!("{path}.part")
}

/// What a `HEAD` is worth before a transfer: how big the file is, and which
/// repository revision the host says it is serving.
struct Probe {
    announced: Option<u64>,
    revision: Option<String>,
}

/// Ask the server about a file without taking it.
///
/// The revision has to be read *while* the redirects are being followed:
/// Hugging Face names it on the response that points at its content delivery
/// network, and the response that actually carries the bytes comes from that
/// network, which knows nothing about repositories. So the redirects are
/// walked by hand here — a client that followed them itself would only ever
/// show the last hop, which is the one without the answer.
async fn probe_file(client: &Client, url: &str) -> Probe {
    let Ok(start) = reqwest::Url::parse(url) else {
        return Probe {
            announced: None,
            revision: None,
        };
    };
    // The caller's client is only borrowed for its defaults, so a client of
    // this function's own does the asking: it must not follow redirects, or the
    // hop that names the commit disappears before it can be read.
    let prober = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap_or_else(|_| client.clone());
    let mut target = start;
    let mut revision = None;
    let mut announced = None;
    for _ in 0..MAX_REDIRECT_HOPS {
        let Ok(response) = prober.head(target.as_str()).send().await else {
            break;
        };
        if let Some(reported) = reported_revision(&response) {
            revision = Some(reported);
        }
        let status = response.status();
        if !status.is_redirection() {
            announced = content_length(&response);
            break;
        }
        let Some(location) = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
        else {
            break;
        };
        let Ok(next) = response.url().join(location.trim()) else {
            break;
        };
        target = next;
    }
    Probe {
        announced,
        revision,
    }
}

/// How many pointers down a chain this will walk before concluding the host is
/// not going to hand over a file. Ten is the limit `llama-server`'s own fetcher
/// uses and well past what Hugging Face's two-hop chain needs.
const MAX_REDIRECT_HOPS: u32 = 10;

/// What the server says the file weighs, from a response that transferred
/// nothing. `None` when it will not say — a host that rejects `HEAD`, or one
/// that answers without a length — which is a size the caller cannot price
/// before the bytes arrive.
fn content_length(response: &reqwest::Response) -> Option<u64> {
    if !response.status().is_success() {
        return None;
    }
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
}

/// The revision a response names, when it names one: a 40-character hex commit,
/// and nothing else. A host that answers with a branch name or a string of the
/// wrong shape is not reporting a revision.
fn reported_revision(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get("x-repo-commit")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| value.len() == 40 && value.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Unix seconds, for the provenance timestamp.
fn now_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Said whenever a transfer stops with bytes on disk worth keeping: the whole
/// point of writing to a `.part` is that the next attempt does not start over.
const RESUMABLE: &str =
    "the bytes already downloaded stay on disk, so the next attempt continues from there";

/// The whole size of the file a response is part of, read from `Content-Range`
/// (`bytes 4096-491400031/491400032`) — the header that says it only when the
/// response is a slice.
fn ranged_total(response: &reqwest::Response) -> Option<u64> {
    let value = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?;
    value.rsplit('/').next()?.trim().parse::<u64>().ok()
}

/// Fetch a GGUF into `path`, continuing a partial file where it stopped.
///
/// `free_bytes` is what the filesystem that will hold the file can take, from
/// [`xencode_context_rs::hwprobe::free_disk_bytes`] at the call site: this crate
/// reads model files and does not know where they are being put. `None` means
/// nobody measured, which is not the same as an empty disk — the transfer is
/// allowed to proceed and a full disk then shows up as an I/O failure.
///
/// `on_progress` is called at least at the start and the end of every 4 MiB of
/// new bytes.
///
/// `expected_sha256`, when given, is a digest the caller obtained somewhere
/// other than this transfer. The bytes are hashed as they arrive and compared
/// at the end; a mismatch throws the file away — including the partial, whose
/// prefix is now known to be wrong — because continuing from bytes that do not
/// hash correctly can never produce a file that does.
pub async fn fetch_gguf(
    client: &Client,
    url: &str,
    path: &str,
    free_bytes: Option<u64>,
    expected_sha256: Option<&str>,
    on_progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<Downloaded, DownloadError> {
    let part = partial_path(path);
    let already = std::fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);

    let probe = probe_file(client, url).await;
    let announced = probe.announced;
    if let Some(total) = announced {
        if let Some(free) = free_bytes {
            // The bytes already on disk are paid for; only the remainder has to
            // fit. Checking the whole file instead would refuse a download that
            // was three quarters finished and had plenty of room left.
            let room_needed = total.saturating_sub(already);
            if free < room_needed {
                return Err(DownloadError::NoRoom {
                    path: path.to_string(),
                    needed_bytes: total,
                    free_bytes: free,
                });
            }
        }
    }

    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DownloadError::Failed(format!("cannot create {}: {e}", parent.display()))
            })?;
        }
    }

    // A ranged request is retried once without the range: a partial file from a
    // different download, or one that ran past the end of this file, earns a
    // 416 rather than a continuation.
    let mut resume_from = already;
    let mut response = ranged_get(client, url, resume_from).await?;
    if response.status().as_u16() == 416 {
        resume_from = 0;
        response = ranged_get(client, url, 0).await?;
    }

    let status = response.status();
    if !status.is_success() {
        return Err(DownloadError::Failed(format!(
            "the server answered {status} for {url}"
        )));
    }

    // 206 is the server agreeing to send the tail. Anything else is the whole
    // file, which makes the bytes already on disk useless.
    let continued = status.as_u16() == 206 && resume_from > 0;
    let mut written = if continued { resume_from } else { 0 };
    let total = if continued {
        ranged_total(&response).or(announced)
    } else {
        announced
    };

    let mut file = if written == 0 {
        std::fs::File::create(&part)
    } else {
        std::fs::OpenOptions::new().append(true).open(&part)
    }
    .map_err(|e| DownloadError::Failed(format!("cannot write {}: {e}", part)))?;

    // Bytes from an earlier attempt are part of the file this one is building,
    // so they have to be inside the digest too — otherwise the checksum of a
    // resumed download would describe only its second half.
    let mut hasher = Sha256::new();
    if written > 0 {
        let hashed = hash_file_prefix(&part, &mut hasher).map_err(DownloadError::Failed)?;
        if hashed != written {
            return Err(DownloadError::Failed(format!(
                "the partial file shrank while being read ({hashed} of {written} bytes), so nothing can be trusted about it"
            )));
        }
    }
    let revision = reported_revision(&response).or_else(|| probe.revision.clone());

    on_progress(Progress {
        received: written,
        total,
    });
    let mut response = response;
    let mut next_report = written + REPORT_BYTES;
    loop {
        let chunk = response.chunk().await.map_err(|e| {
            DownloadError::Failed(format!(
                "the transfer stopped at {}: {e}; {RESUMABLE}",
                human_bytes(written)
            ))
        })?;
        let Some(chunk) = chunk else { break };
        file.write_all(&chunk)
            .map_err(|e| DownloadError::Failed(format!("the disk refused {path}: {e}")))?;
        hasher.update(&chunk);
        written += chunk.len() as u64;
        if written >= next_report {
            on_progress(Progress {
                received: written,
                total,
            });
            next_report = written + REPORT_BYTES;
        }
    }
    file.flush()
        .map_err(|e| DownloadError::Failed(format!("cannot finish writing {path}: {e}")))?;
    drop(file);

    if let Some(expected) = total {
        if written != expected {
            // Keep the partial file: this is the case a re-run exists to fix.
            return Err(DownloadError::Failed(format!(
                "the server sent {} of {}, so the file is incomplete; {RESUMABLE}",
                human_bytes(written),
                human_bytes(expected)
            )));
        }
    }

    let sha256 = hex_lower(&hasher.finalize());
    if let Some(expected) = expected_sha256 {
        if !digest_matches(expected, &sha256) {
            // Unlike a short transfer there is nothing worth keeping: the bytes
            // on disk are the wrong bytes, so no continuation of them could
            // ever hash to the expected value. Keeping them would only make the
            // next attempt fail the same way, faster.
            let _ = std::fs::remove_file(&part);
            return Err(DownloadError::Failed(format!(
                "{url} gave a file that hashes to {sha256} instead of the expected {}; the bad bytes were thrown away",
                expected.trim().to_lowercase()
            )));
        }
    }

    std::fs::rename(&part, path).map_err(|e| {
        DownloadError::Failed(format!("cannot move the finished file into place: {e}"))
    })?;
    let provenance = Provenance {
        url: url.to_string(),
        sha256: sha256.clone(),
        bytes: written,
        revision: revision.clone(),
        verified: expected_sha256.is_some(),
        fetched_at: now_unix_seconds(),
    };
    write_provenance(path, &provenance);
    on_progress(Progress {
        received: written,
        total,
    });

    Ok(Downloaded {
        path: path.to_string(),
        bytes: written,
        resumed_from: if continued { resume_from } else { 0 },
        resume_refused: already > 0 && !continued,
        sha256,
        verified: expected_sha256.is_some(),
        revision,
    })
}

/// Hash a file's whole contents into `hasher`, returning how many bytes went
/// in. Used to fold a resumed download's prefix into the digest of the file
/// being finished.
fn hash_file_prefix(path: &str, hasher: &mut Sha256) -> Result<u64, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hashed = 0u64;
    loop {
        let n = file.read(&mut buffer).map_err(|e| format!("{path}: {e}"))?;
        if n == 0 {
            return Ok(hashed);
        }
        hasher.update(&buffer[..n]);
        hashed += n as u64;
    }
}

/// Record where a finished file came from. A provenance sidecar that cannot be
/// written is a loss of metadata, not of the model: the file is complete and
/// usable, so the download is not failed over it.
fn write_provenance(model_path: &str, provenance: &Provenance) {
    let path = provenance_path(model_path);
    if let Ok(text) = serde_json::to_string_pretty(provenance) {
        let _ = std::fs::write(&path, text);
    }
}

/// [`fetch_gguf`] with a client built here, for callers that have no HTTP
/// client of their own. The client has no request timeout: a file that is
/// hundreds of megabytes long is allowed to take as long as it takes.
pub async fn fetch_model_file(
    url: &str,
    path: &str,
    free_bytes: Option<u64>,
    expected_sha256: Option<&str>,
    on_progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<Downloaded, DownloadError> {
    fetch_gguf(
        &Client::default(),
        url,
        path,
        free_bytes,
        expected_sha256,
        on_progress,
    )
    .await
}

async fn ranged_get(
    client: &Client,
    url: &str,
    from: u64,
) -> Result<reqwest::Response, DownloadError> {
    let mut request = client.get(url);
    if from > 0 {
        request = request.header("Range", format!("bytes={from}-"));
    }
    request
        .send()
        .await
        .map_err(|e| DownloadError::Failed(format!("cannot ask {url} for the file: {e}")))
}

/// Whether an interrupted download has anything to continue from.
pub fn partial_bytes(path: &str) -> u64 {
    std::fs::metadata(partial_path(path))
        .map(|meta| meta.len())
        .unwrap_or(0)
}

/// Throw away a partial file — used when the source changes and the bytes on
/// disk are for a different model.
pub fn discard_partial(path: &str) -> std::io::Result<()> {
    match std::fs::remove_file(partial_path(path)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real HTTP server on a real socket, serving one fixed file and honouring
    /// `Range` the way a file host does. The download under test talks HTTP over
    /// a loopback connection and writes to a real temporary directory: there is
    /// no fake client here, because a resume that only works against a fake
    /// server is not a resume.
    struct FileServer {
        base_url: String,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for FileServer {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `ignore_ranges` makes the server answer every request with the whole
    /// file, which is what a host without range support does.
    fn serve_file(payload: Vec<u8>, ignore_ranges: bool) -> FileServer {
        use std::io::Read;
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = stop.clone();
        std::thread::spawn(move || {
            while !server_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                };
                let mut probe = Vec::new();
                let mut buffer = [0u8; 1024];
                // Read just the request line and headers: the body of a GET is
                // empty, and reading further would wait for bytes that never come.
                loop {
                    match stream.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(n) => {
                            probe.extend_from_slice(&buffer[..n]);
                            if probe.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                            if probe.len() > 8192 {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let request = String::from_utf8_lossy(&probe).into_owned();
                let head_only = request.starts_with("HEAD ");
                let asked_from = request
                    .lines()
                    .find_map(|line| {
                        let (_, value) = line.split_once(':')?;
                        let value = value.trim();
                        let range = value.strip_prefix("bytes=")?;
                        range.split('-').next()?.trim().parse::<usize>().ok()
                    })
                    .unwrap_or(0);
                let (status, body, extra) = if ignore_ranges || asked_from == 0 {
                    (
                        "200 OK".to_string(),
                        payload.clone(),
                        format!(
                            "Content-Length: {}\r\nAccept-Ranges: bytes\r\n",
                            payload.len()
                        ),
                    )
                } else if asked_from > payload.len() {
                    (
                        "416 Range Not Satisfiable".to_string(),
                        Vec::new(),
                        "Content-Length: 0\r\n".to_string(),
                    )
                } else {
                    let end = payload.len() - 1;
                    (
                        "206 Partial Content".to_string(),
                        payload[asked_from..].to_vec(),
                        format!(
                            "Content-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\n",
                            payload.len() - asked_from,
                            asked_from,
                            end,
                            payload.len()
                        ),
                    )
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\n\
                     Connection: close\r\n{extra}\r\n"
                );
                let _ = stream.write_all(reply.as_bytes());
                if !head_only {
                    let _ = stream.write_all(&body);
                }
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Write);
            }
        });
        FileServer {
            base_url: format!("http://127.0.0.1:{port}"),
            stop,
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-download-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A host that owns a repository and answers a request for a file by
    /// pointing somewhere else for the bytes, naming the commit on that pointer.
    /// The destination says nothing about repositories — which is the shape that
    /// makes reading the revision out of the final response lose it.
    fn redirect_to(commit: &'static str, target: &FileServer) -> FileServer {
        use std::io::Read;
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = stop.clone();
        let location = format!("{}/model.gguf", target.base_url);
        std::thread::spawn(move || {
            while !server_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                };
                // Read one block of the request so the client is not writing
                // into a socket nobody is draining, then answer.
                let mut buffer = [0u8; 1024];
                let _ = stream.read(&mut buffer);
                let reply = format!(
                    "HTTP/1.1 302 Found\r\nLocation: {location}\r\n\
                     x-repo-commit: {commit}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(reply.as_bytes());
                let _ = stream.flush();
            }
        });
        FileServer {
            base_url: format!("http://127.0.0.1:{port}"),
            stop,
        }
    }

    /// A client with no timeout on purpose: a whole model file is allowed to
    /// take as long as it takes, which is the behaviour under test.
    fn client() -> Client {
        Client::default()
    }

    fn counted_progress() -> (
        std::sync::Arc<std::sync::Mutex<Vec<Progress>>>,
        impl Fn(Progress),
    ) {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        (seen, move |p| sink.lock().unwrap().push(p))
    }

    #[tokio::test]
    async fn a_whole_file_is_fetched_and_lands_at_the_path_it_was_asked_for() {
        let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("whole");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();

        let (seen, on_progress) = counted_progress();
        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            None,
            &on_progress,
        )
        .await
        .unwrap();

        assert_eq!(got.bytes, payload.len() as u64);
        assert_eq!(got.resumed_from, 0);
        assert!(!got.resume_refused);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        // The `.part` is gone: a path that exists is a complete file.
        assert!(!std::path::Path::new(&partial_path(&path_str)).exists());
        let seen = seen.lock().unwrap();
        assert!(
            seen.first().map(|p| p.received) == Some(0),
            "the first report should say nothing has arrived: {seen:?}"
        );
        assert_eq!(seen.last().map(|p| p.received), Some(got.bytes));
        assert_eq!(seen.last().and_then(|p| p.total), Some(got.bytes));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_revision_named_on_the_redirect_survives_a_transfer_that_ends_elsewhere() {
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let target = serve_file(payload.clone(), false);
        let commit = "9217f5db79a29953eb74d5343926648285ec7e67";
        let redirector = redirect_to(commit, &target);
        let dir = temp_dir("redirect");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();

        let (_, on_progress) = counted_progress();
        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", redirector.base_url),
            &path_str,
            None,
            None,
            &on_progress,
        )
        .await
        .unwrap();

        // The bytes came from the second host, which never mentioned a
        // repository; the revision still has to be the one the first host named.
        assert_eq!(got.revision.as_deref(), Some(commit));
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let record = read_provenance(&path_str).expect("the sidecar the download wrote");
        assert_eq!(record.revision.as_deref(), Some(commit));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_partial_file_is_continued_from_its_last_byte_and_not_restarted() {
        let payload: Vec<u8> = (0..500_000u32).map(|i| (i % 253) as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("resume");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        let cut = 200_000usize;
        std::fs::write(partial_path(&path_str), &payload[..cut]).unwrap();

        let (_, on_progress) = counted_progress();
        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            None,
            &on_progress,
        )
        .await
        .unwrap();

        assert_eq!(
            got.resumed_from, cut as u64,
            "the bytes already on disk should have been kept"
        );
        assert_eq!(got.bytes, payload.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_server_that_sends_the_whole_file_anyway_discards_the_partial_it_was_shown() {
        // The honest half of resume: when the host will not slice the file, the
        // bytes on disk are not a prefix of anything and continuing would write
        // a model with the first 200 000 bytes duplicated inside it.
        let payload: Vec<u8> = (0..300_000u32).map(|i| (i % 254) as u8).collect();
        let server = serve_file(payload.clone(), true);
        let dir = temp_dir("no-range");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        std::fs::write(partial_path(&path_str), vec![0xABu8; 100_000]).unwrap();

        let (_, on_progress) = counted_progress();
        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            None,
            &on_progress,
        )
        .await
        .unwrap();

        assert!(got.resume_refused);
        assert_eq!(got.resumed_from, 0);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_file_the_disk_cannot_hold_is_refused_before_anything_is_written() {
        let payload: Vec<u8> = (0..400_000u32).map(|i| i as u8).collect();
        let server = serve_file(payload, false);
        let dir = temp_dir("noroom");
        // A path whose parent does not exist yet: the refusal has to arrive
        // without creating it either.
        let path = dir.join("deeper").join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();

        let error = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            // One byte short of the file.
            Some(399_999),
            None,
            &|_| {},
        )
        .await
        .unwrap_err();

        match &error {
            DownloadError::NoRoom {
                needed_bytes,
                free_bytes,
                ..
            } => {
                assert_eq!(*needed_bytes, 400_000);
                assert_eq!(*free_bytes, 399_999);
            }
            other => panic!("expected a refusal about the disk, got {other:?}"),
        }
        assert!(!path.exists());
        assert!(!dir.join("deeper").exists(), "the refusal created {dir:?}");
        assert_eq!(partial_bytes(&path_str), 0);
        let text = error.to_string();
        assert!(text.contains("the file is 0.4 MiB"), "{text}");
        assert!(text.contains("has 0.4 MiB free"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_partial_file_only_needs_the_rest_of_the_disk_to_be_let_through() {
        // The other side of the same arithmetic: three quarters of the file is
        // already paid for, so a disk with room for the remainder is enough.
        let payload: Vec<u8> = (0..400_000u32).map(|i| i as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("remainder");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        std::fs::write(partial_path(&path_str), &payload[..300_000]).unwrap();

        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            // Enough for the last 100 000 bytes, nowhere near enough for the file.
            Some(100_000),
            None,
            &|_| {},
        )
        .await
        .unwrap();
        assert_eq!(got.resumed_from, 300_000);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_transfer_that_stops_early_keeps_the_bytes_it_got_says_so_and_can_be_finished() {
        let payload: Vec<u8> = (0..600_000u32).map(|i| (i % 255) as u8).collect();
        let truncated = payload[..400_000].to_vec();
        // A server that announces the full length and then sends less than it
        // promised: a connection cut mid-transfer, as it looks to a client.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let server_stop = stop.clone();
        let announced = payload.len();
        let body = truncated.clone();
        std::thread::spawn(move || {
            while !server_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                };
                let mut probe = Vec::new();
                let mut buffer = [0u8; 1024];
                loop {
                    match std::io::Read::read(&mut stream, &mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            probe.extend_from_slice(&buffer[..n]);
                            if probe.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let request = String::from_utf8_lossy(&probe).into_owned();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {announced}\r\nConnection: close\r\n\r\n"
                );
                if request.starts_with("HEAD ") {
                    let _ = stream.write_all(header.as_bytes());
                } else {
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                }
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Write);
            }
        });

        let dir = temp_dir("cut");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        let error = fetch_gguf(
            &client(),
            &format!("http://127.0.0.1:{port}/model.gguf"),
            &path_str,
            Some(10_000_000_000),
            None,
            &|_| {},
        )
        .await
        .unwrap_err();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);

        assert!(matches!(error, DownloadError::Failed(_)), "{error:?}");
        // The client notices a body that ends before the length it was
        // promised; what matters is that it says the next attempt continues.
        assert!(
            error.to_string().contains("the next attempt continues"),
            "the message should say the bytes are kept: {error}"
        );
        assert!(!path.exists(), "a short file must not be moved into place");
        assert_eq!(
            partial_bytes(&path_str),
            400_000,
            "the bytes that did arrive have to stay for the next attempt"
        );

        // The second half of the name: a fresh attempt against a server that
        // will answer properly finishes the file rather than restarting it.
        let whole = serve_file(payload.clone(), false);
        let finished = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", whole.base_url),
            &path_str,
            Some(10_000_000_000),
            None,
            &|_| {},
        )
        .await
        .unwrap();
        assert_eq!(
            finished.resumed_from, 400_000,
            "the second attempt should have asked for the tail only"
        );
        assert_eq!(finished.bytes, payload.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_partial_file_larger_than_the_file_is_not_trusted_with_a_range() {
        // A stale `.part` from a bigger model would get a 416 from any correct
        // server; the download has to fall back to fetching the whole thing
        // rather than reporting a failure.
        let payload: Vec<u8> = (0..50_000u32).map(|i| (i % 250) as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("stale");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        std::fs::write(partial_path(&path_str), vec![7u8; 80_000]).unwrap();

        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            None,
            &|_| {},
        )
        .await
        .unwrap();
        assert_eq!(got.bytes, payload.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_progress_line_says_a_share_when_the_length_is_known_and_silently_counts_when_it_is_not() {
        let known = Progress {
            received: 250_000_000,
            total: Some(1_000_000_000),
        };
        let label = known.label();
        assert!(label.contains("238.4 MiB"), "{label}");
        assert!(label.contains("953.7 MiB"), "{label}");
        assert!(label.contains("25 %"), "{label}");
        assert_eq!(known.fraction(), Some(0.25));

        let unknown = Progress {
            received: 4 * 1024 * 1024,
            total: None,
        };
        assert_eq!(unknown.fraction(), None);
        assert_eq!(unknown.label(), "4.0 MiB so far");

        // A zero-length answer is not a fraction of anything, and must not
        // divide by it.
        let empty = Progress {
            received: 0,
            total: Some(0),
        };
        assert_eq!(empty.fraction(), None);
    }

    #[test]
    fn byte_counts_are_quoted_in_the_unit_the_rest_of_xencode_uses() {
        assert_eq!(human_bytes(512), "0.0 MiB");
        assert_eq!(human_bytes(1024 * 1024), "1.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
        // A 469 MiB model file, the size the live check downloads.
        assert_eq!(human_bytes(491_400_032), "468.6 MiB");
    }

    #[test]
    fn discarding_a_partial_file_tolerates_there_not_being_one() {
        let dir = temp_dir("discard");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        assert_eq!(partial_bytes(&path_str), 0);
        discard_partial(&path_str).unwrap();
        std::fs::write(partial_path(&path_str), b"partial").unwrap();
        assert_eq!(partial_bytes(&path_str), 7);
        discard_partial(&path_str).unwrap();
        assert_eq!(partial_bytes(&path_str), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The digest of some bytes, worked out here rather than copied from the
    /// code under test, so a bug in the downloader's hashing cannot agree with
    /// itself.
    fn digest_of(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex_lower(&hasher.finalize())
    }

    #[tokio::test]
    async fn a_file_that_hashes_to_the_expected_checksum_is_verified_and_written_down() {
        let payload: Vec<u8> = (0..150_000u32).map(|i| (i % 251) as u8).collect();
        let expected = digest_of(&payload);
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("verified");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        let url = format!("{}/model.gguf", server.base_url);

        let got = fetch_gguf(
            &client(),
            &url,
            &path_str,
            Some(10_000_000_000),
            Some(&expected),
            &|_| {},
        )
        .await
        .unwrap();

        assert!(got.verified);
        assert_eq!(got.sha256, expected);

        let sidecar_name = format!("{path_str}.provenance.json");
        let sidecar = Path::new(&sidecar_name);
        assert!(sidecar.exists(), "the provenance file was not written");
        let written: Provenance =
            serde_json::from_str(&std::fs::read_to_string(sidecar).unwrap()).unwrap();
        assert_eq!(written.sha256, expected);
        assert_eq!(written.bytes, payload.len() as u64);
        assert_eq!(written.url, url);
        assert!(written.verified);
        assert!(written.fetched_at > 0);
        // The same answer through the reader the panels use.
        assert_eq!(read_provenance(&path_str), Some(written));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_file_that_does_not_hash_to_the_expected_checksum_is_thrown_away() {
        // The bytes arrive complete and the right length, so nothing else in the
        // downloader would notice. A host that silently replaced the file, or a
        // transfer corrupted in a way that preserves the length, has to be
        // caught by the checksum or not at all.
        let payload: Vec<u8> = (0..150_000u32).map(|i| (i % 251) as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("mismatch");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();

        let wrong = digest_of(b"the bytes some other file was made of");
        let error = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            Some(&wrong),
            &|_| {},
        )
        .await
        .unwrap_err();

        let message = error.to_string();
        assert!(matches!(error, DownloadError::Failed(_)), "{error:?}");
        assert!(message.contains("hashes to"), "{message}");
        assert!(message.contains(&digest_of(&payload)), "{message}");
        assert!(
            message.contains("thrown away"),
            "the message should say the bad bytes are gone: {message}"
        );
        assert!(!path.exists(), "the unverified file was left in place");
        assert_eq!(
            partial_bytes(&path_str),
            0,
            "bytes known to be wrong must not survive for the next attempt"
        );
        assert!(read_provenance(&path_str).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_resumed_download_is_hashed_over_the_bytes_it_did_not_refetch() {
        // Half the file was already on disk, so a checksum that only covered the
        // second half would reject a perfectly good resume.
        let payload: Vec<u8> = (0..400_000u32).map(|i| (i % 251) as u8).collect();
        let server = serve_file(payload.clone(), false);
        let dir = temp_dir("resume-hash");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        std::fs::write(partial_path(&path_str), &payload[..250_000]).unwrap();

        let got = fetch_gguf(
            &client(),
            &format!("{}/model.gguf", server.base_url),
            &path_str,
            Some(10_000_000_000),
            Some(&digest_of(&payload)),
            &|_| {},
        )
        .await
        .unwrap();

        assert_eq!(got.resumed_from, 250_000);
        assert!(got.verified);
        assert_eq!(got.sha256, digest_of(&payload));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_url_the_server_does_not_have_is_reported_as_the_thing_it_is() {
        // The pinned revision case: a commit that was garbage-collected, or a
        // file someone renamed upstream.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let server_stop = stop.clone();
        std::thread::spawn(move || {
            use std::io::Read;
            use std::io::Write;
            while !server_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                };
                let mut probe = Vec::new();
                let mut buffer = [0u8; 512];
                loop {
                    match stream.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            probe.extend_from_slice(&buffer[..n]);
                            if probe.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let reply = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                let _ = stream.write_all(reply.as_bytes());
                let _ = stream.flush();
            }
        });

        let dir = temp_dir("missing");
        let path = dir.join("model.gguf");
        let path_str = path.to_str().unwrap().to_string();
        let url = format!("http://127.0.0.1:{port}/model.gguf");
        let error = fetch_gguf(
            &client(),
            &url,
            &path_str,
            Some(10_000_000_000),
            None,
            &|_| {},
        )
        .await
        .unwrap_err();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);

        assert!(matches!(error, DownloadError::Failed(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("404"), "{message}");
        assert!(message.contains(&url), "{message}");
        assert!(!path.exists());
        assert_eq!(partial_bytes(&path_str), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_checksum_copied_off_a_web_page_still_matches() {
        let dir = temp_dir("checksum-forms");
        let path = dir.join("model.gguf");
        let payload = b"the bytes of a small model".to_vec();
        std::fs::write(&path, &payload).unwrap();
        let path_str = path.to_str().unwrap().to_string();
        let digest = digest_of(&payload);

        let plain = check_model_file(&path_str, Some(&digest));
        assert!(plain.is_verified(), "{plain:?}");
        let upper = check_model_file(&path_str, Some(&digest.to_uppercase()));
        assert!(upper.is_verified(), "{upper:?}");
        let spaced = check_model_file(&path_str, Some(&format!("  {digest}  ")));
        assert!(spaced.is_verified(), "{spaced:?}");
        let prefixed = check_model_file(&path_str, Some(&format!("sha256-{digest}")));
        assert!(prefixed.is_verified(), "{prefixed:?}");

        let bad = check_model_file(&path_str, Some(&digest_of(b"something else")));
        assert_eq!(
            bad,
            FileCheck::Mismatch {
                expected: digest_of(b"something else"),
                actual: digest.clone()
            }
        );
        assert!(bad.label().contains("does not match its checksum"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_file_nobody_pinned_reports_what_it_hashes_to_without_claiming_it_is_genuine() {
        let dir = temp_dir("unsigned");
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"bytes from an untracked source").unwrap();
        let path_str = path.to_str().unwrap().to_string();

        let check = check_model_file(&path_str, None);
        match &check {
            FileCheck::Unsigned { sha256 } => assert_eq!(sha256.len(), 64),
            other => panic!("an unpinned file must not report itself verified: {other:?}"),
        }
        assert!(!check.is_verified());
        assert_eq!(check.label(), "unsigned");

        // A file that is not there is not "unsigned", it is unreadable.
        let missing = check_model_file("/nonexistent/xencode-model.gguf", Some("ab12"));
        assert!(
            matches!(missing, FileCheck::Unreadable { .. }),
            "{missing:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_provenance_line_says_what_was_proved_and_what_was_only_observed() {
        let checked = Provenance {
            url: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/d7f5/q3.gguf".to_string(),
            sha256: "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897".to_string(),
            bytes: 1_107_409_472,
            revision: Some("d7f544eead698dbd1f15126ef60b45a1e1933222".to_string()),
            verified: true,
            fetched_at: 1_790_000_000,
        };
        let label = checked.label();
        assert!(label.contains("1.0 GiB"), "{label}");
        assert!(label.contains("checksum verified"), "{label}");
        assert!(label.contains("revision d7f544ee"), "{label}");

        let observed = Provenance {
            verified: false,
            ..checked.clone()
        };
        assert!(observed.label().contains("checksum unverified"));
        // A file from a host that does not report its commit still describes
        // itself, with the revision simply absent rather than guessed.
        let no_revision = Provenance {
            revision: None,
            ..checked
        };
        assert!(!no_revision.label().contains("revision"));
    }

    #[test]
    fn a_sidecar_that_is_not_there_reads_as_no_provenance_rather_than_an_error() {
        assert!(read_provenance("/nonexistent/xencode-model.gguf").is_none());
        let dir = temp_dir("bad-sidecar");
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"model").unwrap();
        std::fs::write(format!("{}.provenance.json", path.display()), b"{ broken").unwrap();
        assert!(read_provenance(path.to_str().unwrap()).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
