//! OpenSSH remote computer backend (AF-4, L-2).
//!
//! Provides a machine reachable over standard SSH transport,
//! without requiring Google Colab or specific cloud tooling.

use std::path::PathBuf;
use std::time::Duration;

use crate::backend::{first_line, run_capture, Backend, BoxFuture, ComputerBackend, TransportCmd};
use crate::orchestrate::Binaries;

const SSH_TIMEOUT: Duration = Duration::from_secs(10);

/// Backend connecting to a remote machine over OpenSSH.
#[derive(Debug, Clone)]
pub struct SshBackend {
    pub bins: Binaries,
    pub key: Option<PathBuf>,
    pub destination: String,
    pub port: u16,
}

impl SshBackend {
    pub fn new(bins: Binaries, key: PathBuf) -> Self {
        Self {
            bins,
            key: if key.as_os_str().is_empty() {
                None
            } else {
                Some(key)
            },
            destination: "localhost".to_string(),
            port: 22,
        }
    }

    pub fn with_destination(mut self, dest: impl Into<String>) -> Self {
        self.destination = dest.into();
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
}

impl Backend for SshBackend {
    fn id(&self) -> &'static str {
        "ssh"
    }

    async fn provision(&self, session: &str, _gpu: &str) -> Result<(), String> {
        let mut argv = vec![
            self.bins.ssh.display().to_string(),
            "-o".to_string(),
            "BatchMode=yes".to_string(),
            "-o".to_string(),
            "ConnectTimeout=5".to_string(),
            "-p".to_string(),
            self.port.to_string(),
        ];
        if let Some(key) = &self.key {
            argv.push("-i".to_string());
            argv.push(key.display().to_string());
        }
        argv.push(self.destination.clone());
        argv.push(format!("echo xencode-ssh-probe-{session}"));

        let out = run_capture(&self.bins.ssh, &argv, SSH_TIMEOUT).await?;
        if !out.status {
            let err = first_line(&String::from_utf8_lossy(&out.stderr))
                .unwrap_or_else(|| "ssh connection failed".to_string());
            return Err(format!(
                "cannot reach ssh host `{}` (port {}): {err}",
                self.destination, self.port
            ));
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        Ok(vec![self.destination.clone()])
    }

    async fn deprovision(&self, _session: &str) -> Result<(), String> {
        Ok(())
    }

    fn forward_command(&self, _session: &str, local_port: u16, remote_port: u16) -> TransportCmd {
        let mut argv = vec![
            self.bins.ssh.display().to_string(),
            "-N".to_string(),
            "-p".to_string(),
            self.port.to_string(),
            "-L".to_string(),
            format!("{local_port}:127.0.0.1:{remote_port}"),
        ];
        if let Some(key) = &self.key {
            argv.push("-i".to_string());
            argv.push(key.display().to_string());
        }
        argv.push(self.destination.clone());
        TransportCmd {
            exe: self.bins.ssh.clone(),
            argv,
        }
    }

    fn exec_command(&self, _session: &str, command: &str) -> TransportCmd {
        let mut argv = vec![
            self.bins.ssh.display().to_string(),
            "-p".to_string(),
            self.port.to_string(),
        ];
        if let Some(key) = &self.key {
            argv.push("-i".to_string());
            argv.push(key.display().to_string());
        }
        argv.push(self.destination.clone());
        argv.push(command.to_string());
        TransportCmd {
            exe: self.bins.ssh.clone(),
            argv,
        }
    }

    fn reap_hint(&self, _started_at: Option<&str>, endpoint_ok: bool) -> Option<String> {
        if !endpoint_ok {
            Some(format!(
                "ssh bridge to `{}` unreachable or closed",
                self.destination
            ))
        } else {
            None
        }
    }
}

impl ComputerBackend for SshBackend {
    fn id(&self) -> &'static str {
        "ssh"
    }

    fn kind(&self) -> &'static str {
        "ssh"
    }

    fn description(&self) -> &'static str {
        "Remote computer over OpenSSH transport"
    }

    fn is_available(&self) -> (bool, String) {
        if self.bins.ssh.is_file() {
            (
                true,
                format!("OpenSSH client available at {}", self.bins.ssh.display()),
            )
        } else {
            (false, "OpenSSH client not found on PATH".to_string())
        }
    }

    fn provision<'a>(
        &'a self,
        session: &'a str,
        gpu: &'a str,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move { <Self as Backend>::provision(self, session, gpu).await })
    }

    fn list_sessions<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, String>> {
        Box::pin(async move { <Self as Backend>::list_sessions(self).await })
    }

    fn deprovision<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move { <Self as Backend>::deprovision(self, session).await })
    }

    fn forward_command(&self, session: &str, local_port: u16, remote_port: u16) -> TransportCmd {
        <Self as Backend>::forward_command(self, session, local_port, remote_port)
    }

    fn exec_command(&self, session: &str, command: &str) -> TransportCmd {
        <Self as Backend>::exec_command(self, session, command)
    }

    fn reap_hint(&self, started_at: Option<&str>, endpoint_ok: bool) -> Option<String> {
        <Self as Backend>::reap_hint(self, started_at, endpoint_ok)
    }
}
