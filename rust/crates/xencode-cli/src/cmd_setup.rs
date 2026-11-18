//! The top-level `xencode setup` command.
//!
//! Setup is deliberately thin. It strings together four building blocks that
//! already exist and are real: the advice table (via [`recommend_model`]), the
//! resumable GGUF downloader ([`fetch_model`]), the llama.cpp launcher
//! ([`run_llamacpp`]) and the one-shot query ([`run_query`]). It owns no network
//! or process code of its own beyond asking the OS for a free port and starting
//! the launcher on a task so a query can be asked after the server is up.
//!
//! The four boundaries are expressed as the [`SetupSteps`] trait. Production
//! runs it against [`RealSetup`]; tests run it against a recorder that writes
//! down each call and can fail at a chosen step. That lets the decision and
//! reporting logic be tested without pretending a download, a GPU or a server
//! exists. A step that cannot be verified here is left to the real run and is
//! reported as a failure by that run, never marked a success here.

use super::*;

/// The concrete choices the recommendation step made and the later steps use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetupPlan {
    pub(crate) model_path: String,
    pub(crate) model_url: String,
    pub(crate) model_sha256: String,
    /// The name the server will report and that `xencode query` will address.
    pub(crate) alias: String,
    /// Where the choice came from, printed so a recommendation is not mistaken
    /// for a measurement.
    pub(crate) source: String,
}

/// The four boundaries `setup` crosses, in order.
///
/// Each method is a real call in production and a real intent-recording call
/// in tests. The trait exists so the ordering and the failure reporting can be
/// exercised without a model, a GPU, `llama-server` or a network.
pub(crate) trait SetupSteps {
    fn recommend(&mut self) -> impl std::future::Future<Output = Result<SetupPlan, String>>;
    fn download(
        &mut self,
        plan: &SetupPlan,
    ) -> impl std::future::Future<Output = Result<(), String>>;
    fn start(&mut self, plan: &SetupPlan) -> impl std::future::Future<Output = Result<(), String>>;
    fn verify(&mut self, plan: &SetupPlan)
        -> impl std::future::Future<Output = Result<(), String>>;
}

/// Drive the four steps and name the one that failed.
///
/// Success is printed only after every step has returned `Ok`. Any `Err` is
/// wrapped with the step's position and name and returned immediately, so a
/// caller cannot read a partial run as a completed one.
pub(crate) async fn orchestrate_setup<S: SetupSteps>(steps: &mut S) -> Result<(), String> {
    println!("setup: 1/4 — detect the machine and recommend a model");
    let plan = steps
        .recommend()
        .await
        .map_err(|e| format!("step 1 of 4 (detect and recommend a model) failed: {e}"))?;
    println!("setup: chosen model {} ({})", plan.model_path, plan.source);

    println!("setup: 2/4 — download the GGUF if it is missing");
    steps
        .download(&plan)
        .await
        .map_err(|e| format!("step 2 of 4 (download the GGUF) failed: {e}"))?;

    println!("setup: 3/4 — start llama.cpp");
    steps
        .start(&plan)
        .await
        .map_err(|e| format!("step 3 of 4 (start llama.cpp) failed: {e}"))?;

    println!("setup: 4/4 — verify with one real xencode query");
    steps
        .verify(&plan)
        .await
        .map_err(|e| format!("step 4 of 4 (verify with xencode query) failed: {e}"))?;

    println!("setup: all four steps succeeded — xencode query can now use this model");
    Ok(())
}

fn alias_from_path(path: &str) -> String {
    let stem = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let alias: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let alias = alias.trim_matches(|c| c == '-' || c == '.');
    if alias.is_empty() {
        "model".to_string()
    } else {
        alias.to_string()
    }
}

/// Ask the OS for a port nothing holds right now.
///
/// The llama.cpp launcher in this tree binds a fixed port when none is given;
/// asking for one here keeps setup from colliding with another local program.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("could not ask the OS for a free local port: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("could not read the port the OS assigned: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn confirm_download(plan: &SetupPlan, yes: bool) -> Result<bool, String> {
    if yes {
        return Ok(true);
    }
    use std::io::IsTerminal;
    if !io::stdin().is_terminal() {
        println!("setup: standard input is not a terminal; proceeding without asking");
        return Ok(true);
    }
    println!(
        "setup: about to download {} from {}",
        plan.model_path, plan.model_url
    );
    print!("setup: download this model? [y/N] ");
    let _ = io::stdout().flush();
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .map_err(|e| format!("could not read the answer: {e}"))?;
    let answer = answer.trim();
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}

/// The production implementation of the four boundaries.
pub(crate) struct RealSetup {
    model_arg: Option<String>,
    yes: bool,
}

impl RealSetup {
    pub(crate) fn new(model_arg: Option<String>, yes: bool) -> Self {
        Self { model_arg, yes }
    }
}

impl SetupSteps for RealSetup {
    async fn recommend(&mut self) -> Result<SetupPlan, String> {
        let config = XencodeConfig::load().map_err(|e| format!("could not read config: {e}"))?;
        match self.model_arg.take() {
            Some(arg) => {
                let looks_like_path = arg.ends_with(".gguf")
                    || arg.contains(std::path::MAIN_SEPARATOR)
                    || std::path::Path::new(&arg).exists();
                if looks_like_path {
                    Ok(SetupPlan {
                        model_path: arg.clone(),
                        model_url: config.llama_cpp_model_url.clone(),
                        model_sha256: config.llama_cpp_model_sha256.clone(),
                        alias: alias_from_path(&arg),
                        source: "the path given with --model".to_string(),
                    })
                } else {
                    let alias = arg
                        .strip_prefix("llamacpp:")
                        .unwrap_or(&arg)
                        .trim()
                        .to_string();
                    if alias.is_empty() {
                        return Err("--model named no model".to_string());
                    }
                    let path = config.llama_cpp_model_path.clone();
                    if path.trim().is_empty() {
                        return Err(format!(
                            "model id '{arg}' needs a GGUF on disk; pass a .gguf path to --model \
                             or set llama_cpp_model_path first"
                        ));
                    }
                    Ok(SetupPlan {
                        model_path: path,
                        model_url: config.llama_cpp_model_url.clone(),
                        model_sha256: config.llama_cpp_model_sha256.clone(),
                        alias,
                        source: format!("the model id given with --model ({arg})"),
                    })
                }
            }
            None => {
                if !config.llama_cpp_model_path.trim().is_empty()
                    && !config.llama_cpp_model_url.trim().is_empty()
                {
                    return Ok(SetupPlan {
                        alias: alias_from_path(&config.llama_cpp_model_path),
                        model_path: config.llama_cpp_model_path.clone(),
                        model_url: config.llama_cpp_model_url.clone(),
                        model_sha256: config.llama_cpp_model_sha256.clone(),
                        source: "the model already in the config".to_string(),
                    });
                }
                let chosen = recommend_model()?;
                Ok(SetupPlan {
                    source: format!(
                        "the {} tier of the advice table ({})",
                        chosen.tier, chosen.label
                    ),
                    model_path: chosen.path,
                    model_url: chosen.url,
                    model_sha256: chosen.sha256,
                    alias: alias_from_path(&chosen.file_name),
                })
            }
        }
    }

    async fn download(&mut self, plan: &SetupPlan) -> Result<(), String> {
        if std::path::Path::new(&plan.model_path).exists() {
            println!(
                "setup: {} is already on disk; nothing to download",
                plan.model_path
            );
            return Ok(());
        }
        if plan.model_url.trim().is_empty() {
            return Err(format!(
                "no GGUF at {} and no URL to fetch it from; run `xencode models advice` or \
                 set llama_cpp_model_url",
                plan.model_path
            ));
        }
        if !confirm_download(plan, self.yes)? {
            return Err("download declined".to_string());
        }
        let expected = if plan.model_sha256.trim().is_empty() {
            None
        } else {
            Some(plan.model_sha256.as_str())
        };
        fetch_model(&plan.model_url, &plan.model_path, expected).await
    }

    async fn start(&mut self, plan: &SetupPlan) -> Result<(), String> {
        let mut config =
            XencodeConfig::load().map_err(|e| format!("could not read config: {e}"))?;
        config.llama_cpp_model_path = plan.model_path.clone();
        config.llama_cpp_model_url = plan.model_url.clone();
        config.llama_cpp_model_sha256 = plan.model_sha256.clone();
        config.default_model = format!("llamacpp:{}", plan.alias);
        config
            .save()
            .map_err(|e| format!("could not save config: {e}"))?;

        let port = free_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let mut launcher = tokio::spawn(run_llamacpp(LlamacppAction::Start {
            model: Some(plan.model_path.clone()),
            port: Some(port),
            exec: None,
        }));
        let client = LlamaCppClient::new(&base_url, 2);
        let mut ready = false;
        for _ in 0..130 {
            tokio::select! {
                finished = &mut launcher => {
                    launcher.abort();
                    let reason = match finished {
                        Ok(Err(e)) => e,
                        Ok(Ok(())) => "the launcher stopped before setup could verify it".to_string(),
                        Err(e) => format!("the launcher task ended unexpectedly: {e}"),
                    };
                    return Err(format!("llama.cpp did not start: {reason}"));
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
            }
            if client.ping().await.is_ok() {
                ready = true;
                break;
            }
        }
        if !ready {
            launcher.abort();
            return Err(format!(
                "no llama.cpp server answered at {base_url} within 65 seconds; its own messages \
                 above say why"
            ));
        }
        // The launcher waits for Ctrl-C by design. Setup has already observed a
        // real server, so release that wait and leave the server running for
        // later `xencode query` calls; its PID file and config entry stay in
        // place and `xencode llamacpp stop` can stop it.
        launcher.abort();
        let _ = launcher.await;
        println!("setup: llama.cpp is answering at {base_url}");
        Ok(())
    }

    async fn verify(&mut self, plan: &SetupPlan) -> Result<(), String> {
        run_query(
            "Reply with the single word: ready".to_string(),
            Some(format!("llamacpp:{}", plan.alias)),
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(16),
            None,
            None,
            Vec::new(),
            false,
            QueryFormat::Text,
        )
        .await
    }
}

/// Entry point for `xencode setup`.
pub(crate) async fn run_setup(model: Option<String>, yes: bool) -> Result<(), String> {
    println!("setup: bringing up a local model; each step is reported before it runs");
    let mut steps = RealSetup::new(model, yes);
    orchestrate_setup(&mut steps).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A boundary that records each call and the plan it was handed. It fails
    /// at a chosen step so the reporting logic can be observed. Nothing here
    /// touches a network, a GPU, a config file or a process.
    struct RecordingSetup {
        calls: Vec<&'static str>,
        plans: Vec<SetupPlan>,
        fail_at: Option<&'static str>,
    }

    impl RecordingSetup {
        fn new(fail_at: Option<&'static str>) -> Self {
            Self {
                calls: Vec::new(),
                plans: Vec::new(),
                fail_at,
            }
        }

        fn check(&mut self, step: &'static str, plan: &SetupPlan) -> Result<(), String> {
            self.calls.push(step);
            self.plans.push(plan.clone());
            if self.fail_at == Some(step) {
                Err(format!("{step} was refused by the test"))
            } else {
                Ok(())
            }
        }
    }

    fn plan() -> SetupPlan {
        SetupPlan {
            model_path: "/models/tiny.gguf".to_string(),
            model_url: "https://example.invalid/tiny.gguf".to_string(),
            model_sha256: String::new(),
            alias: "tiny".to_string(),
            source: "the test plan".to_string(),
        }
    }

    impl SetupSteps for RecordingSetup {
        async fn recommend(&mut self) -> Result<SetupPlan, String> {
            self.calls.push("recommend");
            if self.fail_at == Some("recommend") {
                return Err("recommend was refused by the test".to_string());
            }
            Ok(plan())
        }

        async fn download(&mut self, plan: &SetupPlan) -> Result<(), String> {
            self.check("download", plan)
        }

        async fn start(&mut self, plan: &SetupPlan) -> Result<(), String> {
            self.check("start", plan)
        }

        async fn verify(&mut self, plan: &SetupPlan) -> Result<(), String> {
            self.check("verify", plan)
        }
    }

    #[tokio::test]
    async fn a_full_run_calls_all_four_steps_in_order_with_the_same_plan() {
        let mut steps = RecordingSetup::new(None);
        orchestrate_setup(&mut steps)
            .await
            .expect("all four steps succeeded");
        assert_eq!(
            steps.calls,
            vec!["recommend", "download", "start", "verify"]
        );
        let expected = plan();
        assert!(steps.plans.iter().all(|p| p == &expected));
    }

    #[tokio::test]
    async fn a_failure_at_any_step_stops_the_run_names_the_step_and_never_claims_success() {
        for (step, label) in [
            ("recommend", "step 1 of 4 (detect and recommend a model)"),
            ("download", "step 2 of 4 (download the GGUF)"),
            ("start", "step 3 of 4 (start llama.cpp)"),
            ("verify", "step 4 of 4 (verify with xencode query)"),
        ] {
            let mut steps = RecordingSetup::new(Some(step));
            let error = orchestrate_setup(&mut steps)
                .await
                .expect_err("a failed step must fail the run");
            assert!(
                error.contains(label),
                "the error must name the failed step: {error}"
            );
            assert!(
                !error.contains("all four steps succeeded"),
                "a failed run must never report success: {error}"
            );
            let reached = steps.calls.len();
            assert_eq!(
                reached,
                steps
                    .calls
                    .iter()
                    .position(|s| *s == step)
                    .expect("the failing step was called")
                    + 1,
                "no step after the failure may run: {:?}",
                steps.calls
            );
        }
    }

    #[test]
    fn alias_from_path_is_stable_and_safe_for_a_model_id() {
        assert_eq!(
            alias_from_path("/models/Qwen3-0.6B-Q4_K_M.gguf"),
            "Qwen3-0.6B-Q4_K_M"
        );
        assert_eq!(alias_from_path("/models/my model.gguf"), "my-model");
        assert_eq!(alias_from_path("/"), "model");
    }
}
