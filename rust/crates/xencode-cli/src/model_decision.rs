//! The one decision behind `models advice`, `setup` and the default model a
//! turn uses.
//!
//! Everything in the decision part of this module is pure: it reads facts
//! already measured by [`measure_hardware`] and an advice table already loaded
//! from disk, and returns an answer with the reason for it. It does not dial a
//! provider, start a server or touch the network. That is what lets the advice
//! command, setup and a routing caller ask the same question and get the same
//! answer — the only inputs that differ are the facts and, for a routing call,
//! the shape of the turn.
//!
//! The size match is exactly what [`Advice::tier_for`] says it is: a GGUF that
//! fits in the largest single place a model file can go. It says nothing about
//! how well any model answers, and no answer here claims a pass rate.

use xencode_config_rs::{SecretProvider, XencodeConfig};
use xencode_context_rs::{hwprobe, TaskShape};
use xencode_models_rs::advice::{Advice, AdviceSource, Tier};
use xencode_providers_rs::{classify, Egress, EgressPolicy, RoutingFacts};
use xencode_tui_rs::task_profiles::{choose_profile_for_shape, TurnChoice};

/// The machine as measured by [`measure_hardware`], before any decision is
/// made. Every field is a reading from a named source or `None` when that
/// source answered nothing; nothing here is an estimate of a model's quality.
#[derive(Debug, Clone, Default)]
pub(crate) struct HardwareFacts {
    /// Total system memory from `/proc/meminfo`, in kibibytes.
    pub(crate) total_kib: Option<u64>,
    /// Memory available right now from `/proc/meminfo`, in kibibytes.
    pub(crate) available_kib: Option<u64>,
    /// Devices the llama-server binary itself reported. An empty list means
    /// the binary was absent or did not answer, not that the machine has no
    /// devices.
    pub(crate) devices: Vec<hwprobe::ComputeDevice>,
    /// Geometry read from a GGUF already on disk, if one is configured.
    pub(crate) model_shape: Option<hwprobe::ModelShape>,
    /// The context window the profile would run with, if known.
    pub(crate) window: Option<u64>,
}

/// What the decision came to, with the evidence a caller needs to print the
/// same answer as every other caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Decision {
    /// The model id a request should name. The baseline is always local; a
    /// cloud or remote model is used only when a matching profile named it and
    /// `allow_cloud_models` permits it.
    pub(crate) model: String,
    /// Why this answer was reached, including every refusal that kept a route
    /// from being taken.
    pub(crate) reason: String,
    /// The largest single place a model file can go, in bytes.
    pub(crate) capacity_bytes: u64,
    /// How that place was described by the probe, when one was measured.
    pub(crate) pool_label: Option<String>,
    /// The band of the advice table that fits that place.
    pub(crate) tier: Option<Tier>,
}

/// Measure the machine once, from the same sources every caller reads.
///
/// The device list comes from the server binary rather than the kernel's card
/// list because only the binary says what it can offload to. Reading a GGUF
/// here is optional: the decision does not depend on a model being on disk,
/// but a caller that has one can name it when it explains the answer.
pub(crate) fn measure_hardware(config: &XencodeConfig) -> HardwareFacts {
    let total_kib = xencode_context_rs::budget::total_memory_kib();
    let available_kib = hwprobe::available_memory_kib();
    let exe = xencode_models_rs::llamacpp::find_llama_server(None);
    let devices: Vec<hwprobe::ComputeDevice> = match &exe {
        Some(exe) => hwprobe::server_devices(exe)
            .as_deref()
            .map(hwprobe::parse_llama_devices)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let model_path = if config.llama_cpp_model_path.trim().is_empty() {
        None
    } else {
        Some(config.llama_cpp_model_path.clone())
    }
    .or_else(|| xencode_models_rs::llamacpp::resolve_gguf_model(None, None));
    let model_shape = model_path.as_deref().and_then(hwprobe::read_gguf_shape);
    let window = Some(
        xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile)
            .profile
            .ctx_tokens(),
    );
    HardwareFacts {
        total_kib,
        available_kib,
        devices,
        model_shape,
        window,
    }
}

/// The capacity a single model file can use, and the probe's name for that
/// place. Pure arithmetic over the facts: a device keeps three quarters of what
/// is free on it, system memory is used as measured, and the largest pool wins.
/// No pool means zero capacity, which is a measurement to report rather than a
/// failure to guess around.
pub(crate) fn capacity_of(hw: &HardwareFacts) -> (u64, Option<String>) {
    let total_mib = hw.total_kib.unwrap_or(0) / 1024;
    let available_mib = hw.available_kib.unwrap_or(0) / 1024;
    let pools = hwprobe::memory_pools(&hw.devices, total_mib, available_mib);
    match pools.first() {
        Some(pool) => (
            pool.usable_mib.saturating_mul(1024 * 1024),
            Some(pool.label.clone()),
        ),
        None => (0, None),
    }
}

fn advice_source_name(advice: &Advice) -> String {
    match &advice.source {
        AdviceSource::Embedded | AdviceSource::EmbeddedAfterRefusingUserFile(_) => {
            "the table shipped with xencode".to_string()
        }
        AdviceSource::UserFile(path) => path.clone(),
    }
}

fn routing_facts(config: &XencodeConfig) -> RoutingFacts<'_> {
    RoutingFacts {
        openrouter_key: config.api_keys.has_secret(SecretProvider::OpenRouter),
        remote_host: (!config.remote_base_url.is_empty())
            .then(|| xencode_providers_rs::url_host(&config.remote_base_url))
            .flatten(),
    }
}

fn cloud_refusal(profile_name: &str, model: &str) -> String {
    format!(
        "profile {profile_name} was not used: it names {model}, which routes off-machine, and \
         cloud models are not allowed by the egress policy (`allow_cloud_models=false`; open it \
         with `xencode config set allow_cloud_models true`), so the local default stands"
    )
}

/// The default model for a turn, from facts already measured.
///
/// The baseline is always the local GGUF the advice table puts in the largest
/// fitting tier, named as the `llamacpp:` id setup would write. When the turn
/// reads as a bugfix and a saved profile is marked for bugfix work, that
/// profile takes the turn by the same rule the query path uses — including its
/// refusal to swap a llama.cpp server out from under a conversation. A profile
/// that routes off-machine is used only with `allow_cloud_models=true`; the
/// reason then names the refusal rather than quietly keeping or quietly taking
/// the cloud route.
pub(crate) fn choose_default_model(
    hw: &HardwareFacts,
    advice: &Advice,
    config: &XencodeConfig,
    shape: TaskShape,
) -> Decision {
    let (capacity_bytes, pool_label) = capacity_of(hw);
    let tier = advice.tier_for(capacity_bytes).cloned();

    let (mut model, mut reason) = match &tier {
        Some(tier) => match tier.gguf.first() {
            Some(entry) => {
                let id = format!(
                    "llamacpp:{}",
                    crate::cmd_models::server_alias(&entry.file, &[])
                );
                let place = pool_label
                    .as_deref()
                    .map(|label| format!("the {label}"))
                    .unwrap_or_else(|| "no measured memory pool".to_string());
                let window = hw
                    .window
                    .map(|w| format!(" at a {w}-token window"))
                    .unwrap_or_default();
                let shape_words = hw
                    .model_shape
                    .as_ref()
                    .map(|shape| {
                        let name = if shape.name.is_empty() {
                            shape.architecture.as_str()
                        } else {
                            shape.name.as_str()
                        };
                        format!(
                            " the {name} on disk carries {} MiB of weights, which a size match \
                             does not change;",
                            shape.file_bytes / 1024 / 1024
                        )
                    })
                    .unwrap_or_default();
                (
                    id,
                    format!(
                        "{place} is the largest place to put one model file ({}), read from the \
                         memory probe;{shape_words} the {} tier of {} fits it, so the local GGUF \
                         {} is the default{window}. This matches sizes only — it says nothing \
                         about how well any model answers.",
                        xencode_models_rs::human_bytes(capacity_bytes),
                        tier.name,
                        advice_source_name(advice),
                        entry.label,
                    ),
                )
            }
            None => (
                config.default_model.clone(),
                format!(
                    "the {} tier of {} names no GGUF file, so no local GGUF is proposed; the \
                     configured default stands and this layer chose no cloud route",
                    tier.name,
                    advice_source_name(advice)
                ),
            ),
        },
        None => (
            config.default_model.clone(),
            format!(
                "nothing in {} fits the {} the largest memory pool can give one model file, so \
                 no local GGUF is proposed; the configured default stands and this layer chose \
                 no cloud route",
                advice_source_name(advice),
                xencode_models_rs::human_bytes(capacity_bytes)
            ),
        ),
    };

    if shape == TaskShape::Bugfix {
        match choose_profile_for_shape(config, &model, shape) {
            TurnChoice::Taken(profile) => {
                let egress = classify(&profile.model, routing_facts(config));
                if EgressPolicy::new(config.allow_cloud_models)
                    .check(egress)
                    .is_err()
                {
                    reason.push_str("; ");
                    reason.push_str(&cloud_refusal(&profile.name, &profile.model));
                } else {
                    model = profile.model.clone();
                    reason.push_str(&format!("; {}", profile.reason));
                    if egress == Egress::Cloud {
                        reason.push_str(
                            " — this off-machine route was allowed by `allow_cloud_models=true`",
                        );
                    }
                }
            }
            TurnChoice::Refused { name, why } => {
                reason.push_str(&format!("; profile {name} was not used: {why}"));
            }
            TurnChoice::Unclaimed => {}
        }
    }

    Decision {
        model,
        reason,
        capacity_bytes,
        pool_label,
        tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_config_rs::ModelProfile;

    // Captured from hwprobe's own fixtures: the device lines at hwprobe.rs:1073
    // and the meminfo figures at hwprobe.rs:1114. These are readings, not a
    // model of a machine, so the decision stays pure and repeatable.
    const CAPTURED_DEVICES: &str = "Available devices:
  BLAS: OpenBLAS (0 MiB, 0 MiB free)
  Vulkan0: Intel(R) UHD Graphics (ICL GT1) (11822 MiB, 8987 MiB free)
  Vulkan1: NVIDIA GeForce MX250 (2294 MiB, 1156 MiB free)
";
    const CAPTURED_TOTAL_KIB: u64 = 16_141_084;
    const CAPTURED_AVAILABLE_KIB: u64 = 9_536_364;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn captured_facts() -> HardwareFacts {
        HardwareFacts {
            total_kib: Some(CAPTURED_TOTAL_KIB),
            available_kib: Some(CAPTURED_AVAILABLE_KIB),
            devices: hwprobe::parse_llama_devices(CAPTURED_DEVICES),
            model_shape: Some(hwprobe::ModelShape {
                architecture: "qwen3".into(),
                name: "Qwen3-0.6B".into(),
                block_count: 28,
                kv_head_count: 8,
                head_dim: 64,
                file_bytes: 396_705_472,
            }),
            window: Some(8192),
        }
    }

    /// A machine with too little room for even the smallest GGUF in the
    /// override table; no local GGUF is proposed, so the configured default is
    /// the baseline and a profile can be tested without a llama.cpp swap.
    fn no_fit_facts() -> HardwareFacts {
        HardwareFacts {
            total_kib: Some(CAPTURED_TOTAL_KIB),
            available_kib: Some(8 * 1024),
            devices: Vec::new(),
            ..HardwareFacts::default()
        }
    }

    /// An override written to a real file and loaded by the real `Advice::load`,
    /// in the shape advice.rs:314 uses. Returned with its path so each test
    /// removes the file when it has finished.
    fn advice_override(tag: &str) -> (Advice, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "xencode-hr1-advice-{tag}-{}.json",
            std::process::id()
        ));
        let body = r#"{"as_of":"2026-11-18","tiers":[
          {"name":"small","max_file_bytes":52428800,"gguf":[{"label":"Small GGUF","repo":"example/repo","file":"small.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":52428800,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]},
          {"name":"medium","max_file_bytes":2147483648,"gguf":[{"label":"Medium GGUF","repo":"example/repo","file":"medium.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":2147483648,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]},
          {"name":"large","max_file_bytes":8589934592,"gguf":[{"label":"Large GGUF","repo":"example/repo","file":"large.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":8589934592,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}
        ]}"#
        .to_string();
        std::fs::write(&path, body).unwrap();
        (Advice::load(&path), path)
    }

    fn bugfix_profile(name: &str, model: &str) -> ModelProfile {
        ModelProfile {
            name: name.to_string(),
            model: model.to_string(),
            temperature: None,
            max_tokens: None,
            for_task: Some("bugfix".to_string()),
        }
    }

    /// More room never moves the answer to a smaller band, and the band the
    /// decision prints is the one `Advice::tier_for` picks for the same bytes.
    #[test]
    fn capacity_chooses_a_monotonic_tier_from_the_override_table() {
        let (advice, path) = advice_override("monotonic");
        let capacities = [512 * MIB, 4 * GIB, 10 * GIB];
        let tiers: Vec<String> = capacities
            .iter()
            .map(|capacity| {
                let hw = HardwareFacts {
                    total_kib: Some(CAPTURED_TOTAL_KIB),
                    available_kib: Some(capacity / 1024),
                    devices: Vec::new(),
                    ..HardwareFacts::default()
                };
                let decision = choose_default_model(
                    &hw,
                    &advice,
                    &XencodeConfig::default(),
                    TaskShape::General,
                );
                assert_eq!(decision.capacity_bytes, *capacity);
                assert_eq!(
                    decision.tier.as_ref().map(|t| t.name.clone()),
                    advice.tier_for(*capacity).map(|t| t.name.clone()),
                    "the decision and the table must name the same band"
                );
                decision.tier.unwrap().name
            })
            .collect();
        assert_eq!(tiers, ["small", "medium", "large"]);
        let _ = std::fs::remove_file(path);
    }

    /// The local GGUF wins even when the config names a cloud model. This is a
    /// size decision from the machine, not a claim that the GGUF is better.
    #[test]
    fn the_default_is_local_even_when_config_names_a_cloud_model() {
        let (advice, path) = advice_override("local");
        let config = XencodeConfig {
            default_model: "anthropic:claude".to_string(),
            ..XencodeConfig::default()
        };
        let decision =
            choose_default_model(&captured_facts(), &advice, &config, TaskShape::General);
        assert!(
            decision.model.starts_with("llamacpp:"),
            "a configured cloud model must not become the default: {}",
            decision.model
        );
        assert!(decision.reason.contains("Large GGUF"));
        assert!(!decision.reason.contains("pass rate"));
        let _ = std::fs::remove_file(path);
    }

    /// A cloud profile stays refused until the owner opens egress, and the
    /// reason says which setting and which refusal kept it out.
    #[test]
    fn a_cloud_profile_is_refused_until_cloud_models_are_allowed() {
        let (advice, path) = advice_override("egress");
        let mut config = XencodeConfig {
            model_routing: true,
            default_model: "qwen2.5:7b".to_string(),
            model_profiles: vec![bugfix_profile("cloud", "qwen:72b")],
            ..XencodeConfig::default()
        };

        let refused = choose_default_model(&no_fit_facts(), &advice, &config, TaskShape::Bugfix);
        assert_eq!(refused.model, "qwen2.5:7b");
        assert!(
            refused.reason.contains("allow_cloud_models=false"),
            "{}",
            refused.reason
        );
        assert!(
            refused.reason.contains("egress policy"),
            "{}",
            refused.reason
        );

        config.allow_cloud_models = true;
        let allowed = choose_default_model(&no_fit_facts(), &advice, &config, TaskShape::Bugfix);
        assert_eq!(allowed.model, "qwen:72b");
        assert!(
            allowed.reason.contains("allow_cloud_models=true"),
            "{}",
            allowed.reason
        );
        let _ = std::fs::remove_file(path);
    }

    /// A local profile marked for bugfix work takes only a bugfix turn, and a
    /// profile that would swap a running llama.cpp model is reported and
    /// refused by the same rule the query path uses.
    #[test]
    fn a_bugfix_profile_overrides_but_never_swaps_llamacpp() {
        let (advice, path) = advice_override("bugfix");
        let config = XencodeConfig {
            model_routing: true,
            default_model: "qwen2.5:7b".to_string(),
            model_profiles: vec![bugfix_profile("quick", "smallmodel:latest")],
            ..XencodeConfig::default()
        };

        let general = choose_default_model(&no_fit_facts(), &advice, &config, TaskShape::General);
        assert_eq!(general.model, "qwen2.5:7b");

        let bugfix = choose_default_model(&no_fit_facts(), &advice, &config, TaskShape::Bugfix);
        assert_eq!(bugfix.model, "smallmodel:latest");
        assert!(bugfix.reason.contains("quick"), "{}", bugfix.reason);

        // With a fitted GGUF the baseline is llama.cpp: the same profile may
        // not replace it, because one server holds one model.
        let held = choose_default_model(&captured_facts(), &advice, &config, TaskShape::Bugfix);
        assert!(held.model.starts_with("llamacpp:"));
        assert!(
            held.reason.contains("one model at a time"),
            "{}",
            held.reason
        );
        let _ = std::fs::remove_file(path);
    }

    /// This test looks only at structural invariants of the live probe. It
    /// never names a model as best, and it never turns a reading into a claim
    /// about which model would answer well here.
    #[test]
    fn the_live_probe_reports_consistent_pools() {
        let total = xencode_context_rs::budget::total_memory_kib();
        let available = hwprobe::available_memory_kib();
        let pools = hwprobe::memory_pools(
            &[],
            total.unwrap_or(0) / 1024,
            available.unwrap_or(0) / 1024,
        );
        assert!(
            pools.windows(2).all(|w| w[0].usable_mib >= w[1].usable_mib),
            "pools must be largest first: {pools:?}"
        );
        assert!(pools.iter().all(|p| p.usable_mib > 0));
        match available {
            Some(kib) if kib > 0 => {
                let first = pools.first().expect("available memory gives a pool");
                assert_eq!(first.usable_mib, kib / 1024);
                assert!(first.label.contains("system memory"), "{}", first.label);
            }
            _ => assert!(pools.is_empty(), "no available memory means no pool"),
        }
        if let (Some(total), Some(available)) = (total, available) {
            assert!(available <= total, "available memory cannot exceed total");
        }
    }
}
