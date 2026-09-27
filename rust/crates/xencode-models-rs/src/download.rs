//! Bringing a model file onto the machine, and being able to stop halfway.
//!
//! A GGUF for a small model is a few hundred megabytes; for a large one it is
//! tens of gigabytes. That makes the download both the slowest step of bringing
//! a local server up and the one most likely to be interrupted — by a dropped
//! connection, by a laptop lid, by the person who started it deciding the file
//! they asked for is the wrong one. An downloader that restarts from zero on
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
//! What this does *not* do is verify the contents. It checks the byte count
//! against what the server announced and stops there: a file that is the right
//! size but wrong in some byte is a corrupt model, not a truncated download, and
//! the only thing that will notice is the loader reading its header.

use std::io::Write;
use std::path::Path;

use reqwest::Client;

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

/// Where the bytes of an unfinished download live.
fn partial_path(path: &str) -> String {
    format!("{path}.part")
}

/// Said whenever a transfer stops with bytes on disk worth keeping: the whole
/// point of writing to a `.part` is that the next attempt does not start over.
const RESUMABLE: &str =
    "the bytes already downloaded stay on disk, so the next attempt continues from there";

/// What the server says the file weighs, from a request that transfers nothing.
/// `None` when it will not say — a server that rejects `HEAD`, or one that
/// answers without a length — which is a size the caller cannot price in advance.
async fn announced_size(client: &Client, url: &str) -> Option<u64> {
    let head = client.head(url).send().await.ok()?;
    if !head.status().is_success() {
        return None;
    }
    head.headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
}

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
pub async fn fetch_gguf(
    client: &Client,
    url: &str,
    path: &str,
    free_bytes: Option<u64>,
    on_progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<Downloaded, DownloadError> {
    let part = partial_path(path);
    let already = std::fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);

    let announced = announced_size(client, url).await;
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

    std::fs::rename(&part, path).map_err(|e| {
        DownloadError::Failed(format!("cannot move the finished file into place: {e}"))
    })?;
    on_progress(Progress {
        received: written,
        total,
    });

    Ok(Downloaded {
        path: path.to_string(),
        bytes: written,
        resumed_from: if continued { resume_from } else { 0 },
        resume_refused: already > 0 && !continued,
    })
}

/// [`fetch_gguf`] with a client built here, for callers that have no HTTP
/// client of their own. The client has no request timeout: a file that is
/// hundreds of megabytes long is allowed to take as long as it takes.
pub async fn fetch_model_file(
    url: &str,
    path: &str,
    free_bytes: Option<u64>,
    on_progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<Downloaded, DownloadError> {
    fetch_gguf(&Client::default(), url, path, free_bytes, on_progress).await
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
                    "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\n{extra}\r\n"
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
                if request.starts_with("HEAD ") {
                    let reply = format!("HTTP/1.1 200 OK\r\nContent-Length: {announced}\r\n\r\n");
                    let _ = stream.write_all(reply.as_bytes());
                } else {
                    let reply = format!("HTTP/1.1 200 OK\r\nContent-Length: {announced}\r\n\r\n");
                    let _ = stream.write_all(reply.as_bytes());
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
}
