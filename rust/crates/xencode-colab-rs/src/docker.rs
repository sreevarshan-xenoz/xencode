//! Docker container computer backend (AF-4).
//!
//! Provides a containerized machine using the local Docker engine.
//! If the Docker engine or socket is unreachable (e.g. daemon stopped or
//! permission denied on `/var/run/docker.sock`), all calls answer honestly
//! without faking green status or panicking.

use std::path::PathBuf;
use std::time::Duration;

use crate::backend::{
    first_line, run_capture, Backend, BoxFuture, ComputerBackend, TransportCmd,
};
use crate::preflight::which;

const DOCKER_TIMEOUT: Duration = Duration::from_secs(10);

/// Backend managing a containerized environment via Docker.
#[derive(Debug, Clone)]
pub struct DockerBackend {
    pub docker_bin: PathBuf,
    pub image: String,
}

impl Default for DockerBackend {
    fn default() -> Self {
        let docker_bin = which("docker").unwrap_or_else(|| PathBuf::from("docker"));
        Self {
            docker_bin,
            image: "alpine:latest".to_string(),
        }
    }
}

impl DockerBackend {
    pub fn new(docker_bin: PathBuf, image: String) -> Self {
        Self { docker_bin, image }
    }

    fn container_name(&self, session: &str) -> String {
        format!("xencode-{session}")
    }

    /// Check if the Docker engine is running and accessible.
    pub async fn check_daemon(&self) -> Result<(), String> {
        if !self.docker_bin.is_file() {
            return Err("docker executable not found on PATH".to_string());
        }
        let argv = vec![
            self.docker_bin.display().to_string(),
            "info".to_string(),
            "--format".to_string(),
            "{{.ServerVersion}}".to_string(),
        ];
        let out = run_capture(&self.docker_bin, &argv, Duration::from_secs(4)).await?;
        if !out.status {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let err = first_line(&stderr).unwrap_or_else(|| "docker info exited with failure".to_string());
            return Err(format!("docker engine unreachable: {err}"));
        }
        Ok(())
    }
}

impl Backend for DockerBackend {
    fn id(&self) -> &'static str {
        "docker"
    }

    async fn provision(&self, session: &str, _gpu: &str) -> Result<(), String> {
        self.check_daemon().await?;
        let name = self.container_name(session);

        // Check if container already exists
        let inspect_argv = vec![
            self.docker_bin.display().to_string(),
            "inspect".to_string(),
            "--type=container".to_string(),
            name.clone(),
        ];
        let inspect_out = run_capture(&self.docker_bin, &inspect_argv, DOCKER_TIMEOUT).await?;
        if inspect_out.status {
            // Container exists, ensure it is started
            let start_argv = vec![
                self.docker_bin.display().to_string(),
                "start".to_string(),
                name,
            ];
            let start_out = run_capture(&self.docker_bin, &start_argv, DOCKER_TIMEOUT).await?;
            if !start_out.status {
                let err = first_line(&String::from_utf8_lossy(&start_out.stderr))
                    .unwrap_or_else(|| "docker start failed".to_string());
                return Err(format!("could not start container: {err}"));
            }
            return Ok(());
        }

        // Run new container
        let run_argv = vec![
            self.docker_bin.display().to_string(),
            "run".to_string(),
            "-d".to_string(),
            "--name".to_string(),
            name,
            self.image.clone(),
            "tail".to_string(),
            "-f".to_string(),
            "/dev/null".to_string(),
        ];
        let run_out = run_capture(&self.docker_bin, &run_argv, DOCKER_TIMEOUT).await?;
        if !run_out.status {
            let err = first_line(&String::from_utf8_lossy(&run_out.stderr))
                .unwrap_or_else(|| "docker run failed".to_string());
            return Err(format!("could not provision docker container: {err}"));
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        self.check_daemon().await?;
        let argv = vec![
            self.docker_bin.display().to_string(),
            "ps".to_string(),
            "--filter".to_string(),
            "name=xencode-".to_string(),
            "--format".to_string(),
            "{{.Names}}".to_string(),
        ];
        let out = run_capture(&self.docker_bin, &argv, DOCKER_TIMEOUT).await?;
        if !out.status {
            let err = first_line(&String::from_utf8_lossy(&out.stderr))
                .unwrap_or_else(|| "docker ps failed".to_string());
            return Err(format!("docker ps failed: {err}"));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let sessions = stdout
            .lines()
            .filter_map(|l| l.strip_prefix("xencode-"))
            .map(|s| s.to_string())
            .collect();
        Ok(sessions)
    }

    async fn deprovision(&self, session: &str) -> Result<(), String> {
        self.check_daemon().await?;
        let name = self.container_name(session);
        let argv = vec![
            self.docker_bin.display().to_string(),
            "rm".to_string(),
            "-f".to_string(),
            name,
        ];
        let out = run_capture(&self.docker_bin, &argv, DOCKER_TIMEOUT).await?;
        if !out.status {
            let err = first_line(&String::from_utf8_lossy(&out.stderr))
                .unwrap_or_else(|| "docker rm failed".to_string());
            return Err(format!("docker deprovision failed: {err}"));
        }
        Ok(())
    }

    fn forward_command(&self, session: &str, _local_port: u16, remote_port: u16) -> TransportCmd {
        let name = self.container_name(session);
        TransportCmd {
            exe: self.docker_bin.clone(),
            argv: vec![
                self.docker_bin.display().to_string(),
                "port".to_string(),
                name,
                format!("{remote_port}/tcp"),
            ],
        }
    }

    fn exec_command(&self, session: &str, command: &str) -> TransportCmd {
        let name = self.container_name(session);
        TransportCmd {
            exe: self.docker_bin.clone(),
            argv: vec![
                self.docker_bin.display().to_string(),
                "exec".to_string(),
                "-i".to_string(),
                name,
                "sh".to_string(),
                "-c".to_string(),
                command.to_string(),
            ],
        }
    }

    fn reap_hint(&self, _started_at: Option<&str>, endpoint_ok: bool) -> Option<String> {
        if !endpoint_ok {
            Some("docker container is not responding or has stopped".to_string())
        } else {
            None
        }
    }
}

impl ComputerBackend for DockerBackend {
    fn id(&self) -> &'static str {
        "docker"
    }

    fn kind(&self) -> &'static str {
        "docker"
    }

    fn description(&self) -> &'static str {
        "Local or remote container computer backend"
    }

    fn is_available(&self) -> (bool, String) {
        if !self.docker_bin.is_file() {
            return (false, "docker executable not found on PATH".to_string());
        }
        // Synchronous probe of the docker binary / permission
        let output = std::process::Command::new(&self.docker_bin)
            .args(["ps"])
            .output();
        match output {
            Ok(out) => {
                if out.status.success() {
                    (true, "Docker engine ready and accessible".to_string())
                } else {
                    let err = first_line(&String::from_utf8_lossy(&out.stderr))
                        .unwrap_or_else(|| "docker ps failed".to_string());
                    (false, format!("docker daemon unreachable: {err}"))
                }
            }
            Err(e) => (false, format!("could not execute docker: {e}")),
        }
    }

    fn provision<'a>(&'a self, session: &'a str, gpu: &'a str) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            <Self as Backend>::provision(self, session, gpu).await
        })
    }

    fn list_sessions<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, String>> {
        Box::pin(async move {
            <Self as Backend>::list_sessions(self).await
        })
    }

    fn deprovision<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            <Self as Backend>::deprovision(self, session).await
        })
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
