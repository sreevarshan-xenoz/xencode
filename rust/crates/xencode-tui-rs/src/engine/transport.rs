//! The local socket windows and the engine talk over (EN-2): JSON lines over a
//! named pipe on Windows or a socket file elsewhere, open only to the user's
//! own account.
//!
//! A pipe name can be guessed, so on Windows both ends check who is on the
//! other side: the engine creates its pipe as the first instance and refuses
//! to start if someone already holds the name, and a window refuses an
//! engine whose process belongs to another account. On Unix the socket lives
//! in a folder only the user can enter.

use std::io;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

use crate::engine::address::Address;

trait Stream: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Stream for T {}

/// One connection: whole lines in, whole lines out.
pub struct Conn {
    reader: BufReader<tokio::io::ReadHalf<Box<dyn Stream>>>,
    writer: tokio::io::WriteHalf<Box<dyn Stream>>,
}

/// The reading half of a connection, for a task of its own.
pub struct ConnReader(BufReader<tokio::io::ReadHalf<Box<dyn Stream>>>);

/// The writing half of a connection, for a task of its own.
pub struct ConnWriter(tokio::io::WriteHalf<Box<dyn Stream>>);

impl Conn {
    fn new(stream: Box<dyn Stream>) -> Conn {
        let (read, write) = tokio::io::split(stream);
        Conn {
            reader: BufReader::new(read),
            writer: write,
        }
    }

    pub async fn send(&mut self, line: &str) -> io::Result<()> {
        write_line(&mut self.writer, line).await
    }

    /// The next line, or `None` once the other side has gone.
    pub async fn recv(&mut self) -> io::Result<Option<String>> {
        read_line(&mut self.reader).await
    }

    pub fn split(self) -> (ConnReader, ConnWriter) {
        (ConnReader(self.reader), ConnWriter(self.writer))
    }
}

impl ConnReader {
    pub async fn recv(&mut self) -> io::Result<Option<String>> {
        read_line(&mut self.0).await
    }
}

impl ConnWriter {
    pub async fn send(&mut self, line: &str) -> io::Result<()> {
        write_line(&mut self.0, line).await
    }
}

async fn write_line(
    writer: &mut tokio::io::WriteHalf<Box<dyn Stream>>,
    line: &str,
) -> io::Result<()> {
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

async fn read_line(
    reader: &mut BufReader<tokio::io::ReadHalf<Box<dyn Stream>>>,
) -> io::Result<Option<String>> {
    let mut line = String::new();
    match reader.read_line(&mut line).await {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(line.trim_end_matches(['\r', '\n']).to_string())),
        // A pipe whose other end closed reports it as an error on Windows.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(windows)]
pub use self::windows::{current_user_sid, Listener};

#[cfg(unix)]
pub use self::unix::Listener;

/// Connect to the engine at `addr`. An address nobody listens on is an
/// error at once, so the caller can start an engine instead of waiting.
pub async fn connect(addr: &Address) -> io::Result<Conn> {
    #[cfg(windows)]
    {
        self::windows::connect(addr).await
    }
    #[cfg(unix)]
    {
        self::unix::connect(addr).await
    }
}

#[cfg(windows)]
mod windows {
    use std::io;
    use std::time::{Duration, Instant};

    use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeServer, ServerOptions};
    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, HANDLE};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, ConvertSidToStringSidW,
        ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo, SDDL_REVISION_1,
        SE_KERNEL_OBJECT,
    };
    use windows_sys::Win32::Security::{
        GetTokenInformation, TokenUser, DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
        SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
    };
    use windows_sys::Win32::System::Pipes::GetNamedPipeServerProcessId;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    use super::Conn;
    use crate::engine::address::Address;

    const ERROR_PIPE_BUSY: i32 = 231;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A NUL-terminated wide string from Windows, freed with `LocalFree`.
    unsafe fn take_wide(ptr: *mut u16) -> String {
        let mut len = 0;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
        LocalFree(ptr.cast());
        text
    }

    /// The string form (`S-1-5-…`) of the account that owns `process`.
    fn sid_of_process(process: HANDLE) -> io::Result<String> {
        // SAFETY: every pointer passed points at a local of the size given,
        // the token handle is closed on every path, and the strings Windows
        // allocates are freed by `take_wide`.
        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            if OpenProcessToken(process, TOKEN_QUERY, &mut token) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut needed = 0u32;
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut needed);
            // Eight-byte words, so the TOKEN_USER at the front is aligned.
            let mut buffer = vec![0u64; (needed as usize).div_ceil(8)];
            let ok = GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                needed,
                &mut needed,
            );
            // Read the error before CloseHandle can replace it.
            let failure = (ok == 0).then(io::Error::last_os_error);
            CloseHandle(token);
            if let Some(e) = failure {
                return Err(e);
            }
            let user = &*(buffer.as_ptr() as *const TOKEN_USER);
            let mut text: *mut u16 = std::ptr::null_mut();
            if ConvertSidToStringSidW(user.User.Sid, &mut text) == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(take_wide(text))
        }
    }

    /// The current user's account, as `S-1-5-…`.
    pub fn current_user_sid() -> io::Result<String> {
        // SAFETY: the pseudo-handle of the current process needs no closing.
        sid_of_process(unsafe { GetCurrentProcess() })
    }

    /// A security descriptor granting full access to `sid` and nobody else.
    struct Descriptor(PSECURITY_DESCRIPTOR);

    impl Descriptor {
        fn only_for(sid: &str) -> io::Result<Descriptor> {
            let sddl = wide(&format!("D:P(A;;GA;;;{sid})"));
            let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            // SAFETY: `sddl` is NUL-terminated and outlives the call; the
            // descriptor Windows allocates is freed in `Drop`.
            let ok = unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    sddl.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Descriptor(descriptor))
        }
    }

    impl Drop for Descriptor {
        fn drop(&mut self) {
            // SAFETY: allocated by Windows with LocalAlloc in `only_for`.
            unsafe {
                LocalFree(self.0);
            }
        }
    }

    // The descriptor is only read by Windows while pipes are created.
    unsafe impl Send for Descriptor {}
    unsafe impl Sync for Descriptor {}

    pub struct Listener {
        name: String,
        descriptor: Descriptor,
        next: NamedPipeServer,
    }

    impl Listener {
        fn create(name: &str, descriptor: &Descriptor, first: bool) -> io::Result<NamedPipeServer> {
            let mut attributes = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor.0,
                bInheritHandle: 0,
            };
            // SAFETY: `attributes` and the descriptor it points at outlive
            // the call, which copies what it needs.
            unsafe {
                ServerOptions::new()
                    .first_pipe_instance(first)
                    .reject_remote_clients(true)
                    .create_with_security_attributes_raw(
                        name,
                        (&mut attributes as *mut SECURITY_ATTRIBUTES).cast(),
                    )
            }
        }

        /// Create the pipe. Fails if any process already holds the name, so a
        /// pipe another account set up first is never taken over.
        pub async fn bind(addr: &Address) -> io::Result<Listener> {
            let Address::Pipe(name) = addr else {
                return Err(io::Error::other("a socket address on Windows"));
            };
            let descriptor = Descriptor::only_for(&current_user_sid()?)?;
            let next = Listener::create(name, &descriptor, true).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("the engine's pipe {name} is already held by another process: {e}"),
                )
            })?;
            Ok(Listener {
                name: name.clone(),
                descriptor,
                next,
            })
        }

        pub async fn accept(&mut self) -> io::Result<Conn> {
            self.next.connect().await?;
            let fresh = Listener::create(&self.name, &self.descriptor, false)?;
            let connected = std::mem::replace(&mut self.next, fresh);
            Ok(Conn::new(Box::new(connected)))
        }

        /// The pipe's access rule as Windows reports it, in SDDL.
        pub fn access_rule(&self) -> io::Result<String> {
            use std::os::windows::io::AsRawHandle;
            let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            // SAFETY: the handle is this listener's live pipe; the descriptor
            // and string Windows allocates are both freed below.
            unsafe {
                let status = GetSecurityInfo(
                    self.next.as_raw_handle(),
                    SE_KERNEL_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut descriptor,
                );
                if status != 0 {
                    return Err(io::Error::from_raw_os_error(status as i32));
                }
                let mut text: *mut u16 = std::ptr::null_mut();
                let ok = ConvertSecurityDescriptorToStringSecurityDescriptorW(
                    descriptor,
                    SDDL_REVISION_1,
                    DACL_SECURITY_INFORMATION,
                    &mut text,
                    std::ptr::null_mut(),
                );
                LocalFree(descriptor);
                if ok == 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(take_wide(text))
            }
        }
    }

    /// The account of the process serving the pipe this client opened.
    fn server_sid(client: &tokio::net::windows::named_pipe::NamedPipeClient) -> io::Result<String> {
        use std::os::windows::io::AsRawHandle;
        // SAFETY: the handles are live; the process handle is closed on
        // every path.
        unsafe {
            let mut pid = 0u32;
            if GetNamedPipeServerProcessId(client.as_raw_handle(), &mut pid) == 0 {
                return Err(io::Error::last_os_error());
            }
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if process.is_null() {
                return Err(io::Error::last_os_error());
            }
            let sid = sid_of_process(process);
            CloseHandle(process);
            sid
        }
    }

    pub async fn connect(addr: &Address) -> io::Result<Conn> {
        let Address::Pipe(name) = addr else {
            return Err(io::Error::other("a socket address on Windows"));
        };
        let start = Instant::now();
        let client = loop {
            match ClientOptions::new().open(name) {
                Ok(client) => break client,
                Err(e)
                    if e.raw_os_error() == Some(ERROR_PIPE_BUSY)
                        && start.elapsed() < Duration::from_secs(2) =>
                {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(e) => return Err(e),
            }
        };
        // The name can be guessed; who serves it cannot be faked.
        let theirs = server_sid(&client)?;
        let mine = current_user_sid()?;
        if theirs != mine {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("the engine's pipe {name} belongs to another account ({theirs}); not connecting"),
            ));
        }
        Ok(Conn::new(Box::new(client)))
    }
}

#[cfg(unix)]
mod unix {
    use std::io;
    use std::os::unix::fs::PermissionsExt;

    use tokio::net::{UnixListener, UnixStream};

    use super::Conn;
    use crate::engine::address::Address;

    pub struct Listener(UnixListener);

    impl Listener {
        /// Bind the socket in a folder only this user can enter. A socket
        /// file nobody answers on is left by a crashed engine and replaced.
        pub async fn bind(addr: &Address) -> io::Result<Listener> {
            let Address::Socket(path) = addr else {
                return Err(io::Error::other("a pipe address on Unix"));
            };
            let folder = path
                .parent()
                .ok_or_else(|| io::Error::other("a socket path with no folder"))?;
            std::fs::create_dir_all(folder)?;
            std::fs::set_permissions(folder, std::fs::Permissions::from_mode(0o700))?;
            if path.exists() {
                if UnixStream::connect(path).await.is_ok() {
                    return Err(io::Error::new(
                        io::ErrorKind::AddrInUse,
                        format!("an engine already listens on {}", path.display()),
                    ));
                }
                std::fs::remove_file(path)?;
            }
            let listener = UnixListener::bind(path)?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
            Ok(Listener(listener))
        }

        pub async fn accept(&mut self) -> io::Result<Conn> {
            let (stream, _) = self.0.accept().await?;
            Ok(Conn::new(Box::new(stream)))
        }
    }

    pub async fn connect(addr: &Address) -> io::Result<Conn> {
        let Address::Socket(path) = addr else {
            return Err(io::Error::other("a pipe address on Unix"));
        };
        Ok(Conn::new(Box::new(UnixStream::connect(path).await?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::address::Address;

    fn test_address() -> (tempfile::TempDir, Address) {
        let dir = tempfile::tempdir().unwrap();
        let name = format!("xencode-transport-test-{}-{}", std::process::id(), {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        });
        let addr = if cfg!(windows) {
            Address::Pipe(format!(r"\\.\pipe\{name}"))
        } else {
            Address::Socket(dir.path().join("engine").join(format!("{name}.sock")))
        };
        (dir, addr)
    }

    #[tokio::test]
    async fn lines_travel_both_ways_and_the_end_is_seen() {
        let (_dir, addr) = test_address();
        let mut listener = Listener::bind(&addr).await.unwrap();
        let server = tokio::spawn(async move {
            let mut conn = listener.accept().await.unwrap();
            for _ in 0..3 {
                let line = conn.recv().await.unwrap().unwrap();
                conn.send(&format!("echo {line}")).await.unwrap();
            }
            // The client goes away: the end of the stream is seen.
            assert_eq!(conn.recv().await.unwrap(), None);
        });
        let mut client = connect(&addr).await.unwrap();
        for word in ["one", "two", "three"] {
            client.send(word).await.unwrap();
            assert_eq!(client.recv().await.unwrap(), Some(format!("echo {word}")));
        }
        drop(client);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_second_client_is_accepted_while_the_first_stays() {
        let (_dir, addr) = test_address();
        let mut listener = Listener::bind(&addr).await.unwrap();
        let server = tokio::spawn(async move {
            let mut a = listener.accept().await.unwrap();
            let mut b = listener.accept().await.unwrap();
            a.send("to a").await.unwrap();
            b.send("to b").await.unwrap();
            (a, b)
        });
        let mut first = connect(&addr).await.unwrap();
        let mut second = connect(&addr).await.unwrap();
        assert_eq!(first.recv().await.unwrap(), Some("to a".into()));
        assert_eq!(second.recv().await.unwrap(), Some("to b".into()));
        drop(server.await.unwrap());
    }

    #[tokio::test]
    async fn nobody_listening_is_an_error_not_a_wait() {
        let (_dir, addr) = test_address();
        let start = std::time::Instant::now();
        assert!(connect(&addr).await.is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(3));
    }

    /// A pipe name some other process created first is never taken over:
    /// the engine refuses to start on it instead of serving beside it.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_pipe_name_already_held_is_refused() {
        let (_dir, addr) = test_address();
        let Address::Pipe(name) = &addr else {
            unreachable!()
        };
        let _squatter = tokio::net::windows::named_pipe::ServerOptions::new()
            .create(name)
            .unwrap();
        let err = match Listener::bind(&addr).await {
            Ok(_) => panic!("bound a pipe name another process already held"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("already held"), "{err}");
    }

    /// The pipe's access rule, read back from Windows, names only this user.
    #[cfg(windows)]
    #[tokio::test]
    async fn the_pipe_admits_only_this_user() {
        let (_dir, addr) = test_address();
        let listener = Listener::bind(&addr).await.unwrap();
        let dacl = listener.access_rule().unwrap();
        let me = current_user_sid().unwrap();
        assert!(dacl.contains(&me), "{dacl} does not name {me}");
        for broad in ["WD", "AU", "BU", "AN", "IU"] {
            assert!(
                !dacl.contains(&format!(";;;{broad})")),
                "{dacl} lets {broad} in"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_socket_file_is_readable_only_by_this_user() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, addr) = test_address();
        let _listener = Listener::bind(&addr).await.unwrap();
        let Address::Socket(path) = &addr else {
            unreachable!()
        };
        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let folder = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(folder, 0o700);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_stale_socket_file_is_replaced() {
        let (_dir, addr) = test_address();
        let Address::Socket(path) = &addr else {
            unreachable!()
        };
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"left by a crashed engine").unwrap();
        assert!(Listener::bind(&addr).await.is_ok());
    }
}
