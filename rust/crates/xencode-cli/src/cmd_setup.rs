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
use xencode_models_rs::advice::Advice;

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

/// Record where the server was actually started, not just which model it
/// serves. `verify` and later `xencode query` calls load the config afresh, so
/// leaving the default llama.cpp URL in place sends them to a port no setup
/// ever opened. Keeping this in one place lets a test assert the address the
/// query will use without starting a process.
fn setup_config_updates(config: &mut XencodeConfig, plan: &SetupPlan, base_url: &str) {
    config.llama_cpp_url = base_url.to_string();
    config.llama_cpp_model_path = plan.model_path.clone();
    config.llama_cpp_model_url = plan.model_url.clone();
    config.llama_cpp_model_sha256 = plan.model_sha256.clone();
    config.default_model = format!("llamacpp:{}", plan.alias);
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
        let port = free_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let mut config =
            XencodeConfig::load().map_err(|e| format!("could not read config: {e}"))?;
        setup_config_updates(&mut config, plan, &base_url);
        config
            .save()
            .map_err(|e| format!("could not save config: {e}"))?;

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
            // A socket that answers is not a model that is loaded. llama.cpp
            // returns 503 "Loading model" from /health until the weights and
            // the KV cache are in place, and ping() counts that as reachable.
            // Asking the stronger question here is what lets the very next
            // step be a real query instead of a 503.
            if client.model_ready().await {
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

/// Which route `setup --remote` takes once the shared decision has looked at
/// this machine. `Local` means a GGUF from the advice table fits; `Colab` means
/// none fits but the bridge is enabled and observed available; `None` means
/// neither holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetupRoute {
    Local,
    Colab,
    None,
}

/// A route with the reason it was picked, so every caller prints the same
/// answer instead of re-deriving one from the route alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetupRouteDecision {
    pub(crate) route: SetupRoute,
    pub(crate) reason: String,
}

/// Remote-backend availability, measured by the caller from the real backend
/// registry and passed into the decision as data. The pure decision runs no
/// probe, process or network call of its own.
#[derive(Debug, Clone, Default)]
pub(crate) struct RemoteAvailability {
    pub(crate) colab_available: bool,
    pub(crate) colab_detail: String,
}

/// Pure route decision for `setup --remote`, from facts already measured by the
/// caller. Local wins whenever a GGUF fits the largest single memory pool, even
/// if a remote bridge is available. Only when nothing fits is Colab taken, and
/// only when it is enabled in config and observed available. Otherwise no route
/// is chosen and the reason names every refusal. Kaggle is a separate slice and
/// is never considered here.
pub(crate) fn choose_setup_route(
    facts: &HardwareFacts,
    advice: &Advice,
    config: &XencodeConfig,
    availability: &RemoteAvailability,
) -> SetupRouteDecision {
    let decision = choose_default_model(
        facts,
        advice,
        config,
        xencode_context_rs::TaskShape::General,
    );
    let local_fits = decision
        .tier
        .as_ref()
        .and_then(|tier| tier.gguf.first())
        .is_some();
    let size_note = format!(
        "the largest memory pool can give one model file {}",
        xencode_models_rs::human_bytes(decision.capacity_bytes)
    );

    if local_fits {
        let tier = decision
            .tier
            .as_ref()
            .map(|t| t.name.as_str())
            .unwrap_or("?");
        return SetupRouteDecision {
            route: SetupRoute::Local,
            reason: format!(
                "a local GGUF from the {tier} tier of the advice table fits {size_note}, read \
                 from the hardware probe; the local route is used even if a remote bridge is \
                 available"
            ),
        };
    }

    let no_fit = format!("no GGUF in the advice table fits {size_note}");
    if config.colab.enabled && availability.colab_available {
        return SetupRouteDecision {
            route: SetupRoute::Colab,
            reason: format!(
                "{no_fit}; Colab is enabled and available: {}",
                availability.colab_detail
            ),
        };
    }

    let mut refusals = vec![no_fit];
    if !config.colab.enabled {
        refusals.push(
            "Colab is disabled — set it with `xencode config set colab_enabled true`".to_string(),
        );
    } else {
        refusals.push(format!(
            "Colab is enabled but not available: {}",
            availability.colab_detail
        ));
    }
    SetupRouteDecision {
        route: SetupRoute::None,
        reason: refusals.join("; "),
    }
}

/// Measure Colab bridge availability from the real backend registry once. This
/// is the only place the remote route touches the system for the decision; the
/// pure function above receives the result as plain data.
fn colab_availability() -> RemoteAvailability {
    let bins = xencode_colab_rs::resolve_binaries().unwrap_or_else(|_| {
        let colab = xencode_colab_rs::which("colab").unwrap_or_else(|| PathBuf::from("colab"));
        let ssh = xencode_colab_rs::which("ssh").unwrap_or_else(|| PathBuf::from("ssh"));
        xencode_colab_rs::Binaries { colab, ssh }
    });
    let key_path = XencodeConfig::config_dir()
        .map(|dir| dir.join(xencode_colab_rs::KEY_FILENAME))
        .unwrap_or_else(|_| PathBuf::from(xencode_colab_rs::KEY_FILENAME));
    let registry = xencode_colab_rs::BackendRegistry::default();
    match registry
        .list_computers(&bins, &key_path)
        .into_iter()
        .find(|info| info.id == "colab")
    {
        Some(info) => RemoteAvailability {
            colab_available: info.available,
            colab_detail: info.detail,
        },
        None => RemoteAvailability::default(),
    }
}

/// The model the remote route asks the VM to serve: the repository id of the
/// first GGUF in the largest tier of the same advice table the HR-1 local
/// decision reads. No local fit is required — the table's own pick is used even
/// when this machine cannot hold any of its files. It is a size-table choice,
/// not a claim that this model answers best.
fn remote_recommended_model(advice: &Advice) -> Option<String> {
    advice
        .file
        .tiers
        .iter()
        .filter(|tier| !tier.gguf.is_empty())
        .max_by_key(|tier| tier.max_file_bytes)
        .and_then(|tier| tier.gguf.first().map(|entry| entry.repo.clone()))
}

/// Entry point for `xencode setup`.
pub(crate) async fn run_setup(
    model: Option<String>,
    yes: bool,
    remote: bool,
    dry_run: bool,
) -> Result<(), String> {
    if remote {
        return run_remote_setup(model, yes, dry_run).await;
    }
    println!("setup: bringing up a local model; each step is reported before it runs");
    let mut steps = RealSetup::new(model, yes);
    orchestrate_setup(&mut steps).await
}

/// The `setup --remote` path: measure once, let the pure decision choose a
/// route, and follow it. Nothing here re-measures hardware or invents backend
/// state — the availability passed to the decision comes from the real backend
/// registry above.
async fn run_remote_setup(model: Option<String>, yes: bool, dry_run: bool) -> Result<(), String> {
    let config = XencodeConfig::load().map_err(|e| format!("could not read config: {e}"))?;
    let facts = measure_hardware(&config);
    let advice = Advice::load(&xencode_models_rs::advice::default_path());
    let availability = colab_availability();
    let route = choose_setup_route(&facts, &advice, &config, &availability);
    println!(
        "setup --remote: route is {:?} — {}",
        route.route, route.reason
    );

    match route.route {
        SetupRoute::Local => {
            println!("setup --remote: a local GGUF fits; bringing it up locally");
            let mut steps = RealSetup::new(model, yes);
            orchestrate_setup(&mut steps).await
        }
        SetupRoute::Colab => {
            // The Colab bridge is the only remote backend this slice handles.
            // Preflight is a real subprocess check of the installed CLI; in a
            // dry run it runs against whatever `colab` is on PATH, but the
            // bring-up below stops at the preview before any VM or network call.
            run_colab_preflight(false).await?;
            let model = remote_recommended_model(&advice).or(model);
            println!(
                "setup --remote: starting the Colab route with model {}",
                model.as_deref().unwrap_or("the configured default")
            );
            run_colab_up_cli(
                None, // session
                None, // gpu
                None, // runtime
                model, None,  // weights
                None,  // quant
                None,  // local_port
                None,  // remote_port
                false, // reconnect
                dry_run,
            )
            .await
        }
        SetupRoute::None => Err(format!(
            "setup --remote: no route is available — {}",
            route.reason
        )),
    }
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
    fn setup_points_the_query_at_the_server_it_actually_started() {
        let mut config = XencodeConfig::default();
        assert_ne!(config.llama_cpp_url, "http://127.0.0.1:40063");
        setup_config_updates(&mut config, &plan(), "http://127.0.0.1:40063");
        assert_eq!(config.llama_cpp_url, "http://127.0.0.1:40063");
        assert_eq!(config.default_model, "llamacpp:tiny");
        assert_eq!(config.llama_cpp_model_path, "/models/tiny.gguf");
        assert_eq!(
            config.llama_cpp_model_url,
            "https://example.invalid/tiny.gguf"
        );
        assert_eq!(config.llama_cpp_model_sha256, "");
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

    // Captured from hwprobe's own fixtures, the same readings the HR-1 decision
    // tests use (model_decision.rs:272-297). These are readings, not a model of
    // the machine, so the route decision stays pure and repeatable.
    const CAPTURED_DEVICES: &str = "Available devices:
  BLAS: OpenBLAS (0 MiB, 0 MiB free)
  Vulkan0: Intel(R) UHD Graphics (ICL GT1) (11822 MiB, 8987 MiB free)
  Vulkan1: NVIDIA GeForce MX250 (2294 MiB, 1156 MiB free)
";
    const CAPTURED_TOTAL_KIB: u64 = 16_141_084;
    const CAPTURED_AVAILABLE_KIB: u64 = 9_536_364;

    fn captured_facts() -> HardwareFacts {
        HardwareFacts {
            total_kib: Some(CAPTURED_TOTAL_KIB),
            available_kib: Some(CAPTURED_AVAILABLE_KIB),
            devices: xencode_context_rs::hwprobe::parse_llama_devices(CAPTURED_DEVICES),
            model_shape: None,
            window: Some(8192),
        }
    }

    /// Only 8 MiB available and no devices: nothing in the override table fits,
    /// so the local route is refused without pretending a GPU is present.
    fn no_fit_facts() -> HardwareFacts {
        HardwareFacts {
            total_kib: Some(CAPTURED_TOTAL_KIB),
            available_kib: Some(8 * 1024),
            devices: Vec::new(),
            ..HardwareFacts::default()
        }
    }

    /// A real override file loaded by the real `Advice::load`, in the shape
    /// advice.rs:318 uses. Returned with its path so the test removes it.
    fn advice_override(tag: &str) -> (Advice, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "xencode-rg3-advice-{tag}-{}.json",
            std::process::id()
        ));
        let body = r#"{"as_of":"2026-11-18","tiers":[
          {"name":"small","max_file_bytes":52428800,"gguf":[{"label":"Small GGUF","repo":"example/repo","file":"small.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":52428800,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]},
          {"name":"large","max_file_bytes":8589934592,"gguf":[{"label":"Large GGUF","repo":"example/large-repo","file":"large.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":8589934592,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}
        ]}"#
        .to_string();
        std::fs::write(&path, body).unwrap();
        (Advice::load(&path), path)
    }

    fn available_colab() -> RemoteAvailability {
        RemoteAvailability {
            colab_available: true,
            colab_detail: "google-colab-cli and OpenSSH available".to_string(),
        }
    }

    #[test]
    fn choose_setup_route_local_fit_wins_over_an_available_remote_bridge() {
        let (advice, path) = advice_override("local-first");
        let config = XencodeConfig {
            colab: xencode_config_rs::ColabConfig {
                enabled: true,
                ..Default::default()
            },
            ..XencodeConfig::default()
        };
        let decision = choose_setup_route(&captured_facts(), &advice, &config, &available_colab());
        assert_eq!(decision.route, SetupRoute::Local);
        assert!(
            decision.reason.contains("local GGUF"),
            "{}",
            decision.reason
        );
        assert!(
            decision
                .reason
                .contains("even if a remote bridge is available"),
            "{}",
            decision.reason
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn choose_setup_route_no_fit_with_colab_enabled_and_available_chooses_colab() {
        let (advice, path) = advice_override("colab");
        let config = XencodeConfig {
            colab: xencode_config_rs::ColabConfig {
                enabled: true,
                ..Default::default()
            },
            ..XencodeConfig::default()
        };
        let decision = choose_setup_route(&no_fit_facts(), &advice, &config, &available_colab());
        assert_eq!(decision.route, SetupRoute::Colab);
        assert!(decision.reason.contains("no GGUF"), "{}", decision.reason);
        assert!(
            decision.reason.contains("Colab is enabled and available"),
            "{}",
            decision.reason
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn choose_setup_route_no_fit_with_colab_disabled_chooses_no_route_and_names_the_setting() {
        let (advice, path) = advice_override("disabled");
        let config = XencodeConfig::default(); // colab.enabled is false by default
        assert!(!config.colab.enabled);
        let decision = choose_setup_route(&no_fit_facts(), &advice, &config, &available_colab());
        assert_eq!(decision.route, SetupRoute::None);
        assert!(
            decision.reason.contains("Colab is disabled"),
            "{}",
            decision.reason
        );
        assert!(
            decision.reason.contains("colab_enabled true"),
            "{}",
            decision.reason
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn choose_setup_route_no_fit_with_colab_unavailable_chooses_no_route_and_names_why() {
        let (advice, path) = advice_override("unavailable");
        let config = XencodeConfig {
            colab: xencode_config_rs::ColabConfig {
                enabled: true,
                ..Default::default()
            },
            ..XencodeConfig::default()
        };
        let unavailable = RemoteAvailability {
            colab_available: false,
            colab_detail: "google-colab-cli not found on PATH".to_string(),
        };
        let decision = choose_setup_route(&no_fit_facts(), &advice, &config, &unavailable);
        assert_eq!(decision.route, SetupRoute::None);
        assert!(
            decision
                .reason
                .contains("Colab is enabled but not available"),
            "{}",
            decision.reason
        );
        assert!(
            decision
                .reason
                .contains("google-colab-cli not found on PATH"),
            "{}",
            decision.reason
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn choose_setup_route_remote_model_is_the_advice_tables_largest_tier_repository() {
        let (advice, path) = advice_override("model");
        assert_eq!(
            remote_recommended_model(&advice).as_deref(),
            Some("example/large-repo")
        );
        let _ = std::fs::remove_file(path);
    }
}
