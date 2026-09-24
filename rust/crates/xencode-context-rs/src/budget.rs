//! Hardware profiles + deterministic token estimator (§10, §14).
//!
//! Profiles are data, never hardcoded logic — logic reads the fields here.
//! The estimator is deliberately cheap and only used for budgeting; real
//! token counts come from llama.cpp `usage` and are what the metrics layer
//! displays.

/// VRAM-based inference profiles (§14 defaults). Which one a given machine gets
/// is `ProfileDecision::resolve` — by total RAM, since a model without a GPU to
/// sit on lives in system memory, and the config may overrule it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareProfile {
    /// <4 GB — smallest context, aggressive compaction.
    Low,
    /// 4–8 GB — the Xencode default target.
    Balanced,
    /// 8 GB+ — room for larger contexts.
    High,
}

impl HardwareProfile {
    pub fn name(self) -> &'static str {
        match self {
            HardwareProfile::Low => "LOW",
            HardwareProfile::Balanced => "BALANCED",
            HardwareProfile::High => "HIGH",
        }
    }

    /// Model context window in tokens.
    pub fn ctx_tokens(self) -> u64 {
        match self {
            HardwareProfile::Low => 4096,
            HardwareProfile::Balanced => 8192,
            HardwareProfile::High => 16384,
        }
    }

    /// Fraction of the context window the context builder may fill.
    pub fn utilization(self) -> f64 {
        match self {
            HardwareProfile::Low => 0.60,
            HardwareProfile::Balanced => 0.75,
            HardwareProfile::High => 0.85,
        }
    }

    /// Retrieval top-K cap.
    pub fn top_k(self) -> usize {
        match self {
            HardwareProfile::Low => 3,
            HardwareProfile::Balanced => 5,
            HardwareProfile::High => 8,
        }
    }

    /// Per-file content cap in characters (injected file bodies).
    pub fn content_cap_chars(self) -> usize {
        match self {
            HardwareProfile::Low => 8_000,
            HardwareProfile::Balanced => 16_000,
            HardwareProfile::High => 24_000,
        }
    }

    /// (soft, hard) compaction trigger fractions — used by M3.
    pub fn compaction_pct(self) -> (f64, f64) {
        match self {
            HardwareProfile::Low => (0.60, 0.80),
            HardwareProfile::Balanced => (0.70, 0.90),
            HardwareProfile::High => (0.80, 0.90),
        }
    }

    /// Prompt batch size — how many tokens the server evaluates per step while
    /// reading a prompt in. A bigger batch costs memory and buys speed, so it
    /// follows the same band as the rest. These three values are reasoned from
    /// the memory bands, not measured against a model; `llama_cpp_args` in the
    /// config can overrule them, since `llama-server` runs with the last value
    /// given for a flag.
    pub fn batch_size(self) -> u32 {
        match self {
            HardwareProfile::Low => 512,
            HardwareProfile::Balanced => 2048,
            HardwareProfile::High => 4096,
        }
    }

    /// The flags this profile wants a self-spawned `llama-server` to start with.
    ///
    /// Measured on `llama-server` b10809 hosting a 1.5B Q4_K_M GGUF. Two things
    /// about that measurement matter here:
    ///
    /// - `--flash-attn` now takes a value (`on|off|auto`) and rejects itself
    ///   written bare: `--flash-attn --cache-type-k …` aborts the launch with
    ///   `unknown value for --flash-attn: '--cache-type-k'`. The preset used to
    ///   emit exactly that, so a server started from it would never have come up.
    /// - Of these flags, only `--ctx-size` and `--parallel` come back through
    ///   `/props` (as `default_generation_settings.n_ctx` and `total_slots`), so
    ///   only those two can be checked after the fact. `--cache-type-k`,
    ///   `--cache-type-v` and `--batch-size` are reported nowhere, and no claim
    ///   is made about them.
    ///
    /// KV quantization is a profile setting rather than a universal flag: LOW
    /// trades the value cache to `q4_0` to fit small memory alongside the key
    /// cache, BALANCED/HIGH run both at `q8_0`. `--parallel 1` is every
    /// profile's answer: `llama-server` splits `--ctx-size` across slots — a
    /// server on the BALANCED profile with two slots gives each conversation
    /// 4096 tokens while the budget is still filling for 8192 — and one slot is
    /// the only count the context builder's arithmetic survives.
    ///
    /// There is deliberately no `--n-predict`. Measured with `--n-predict 8`, a
    /// reply that asked for no limit of its own stopped after eight tokens with
    /// `finish_reason: "length"` — a generation cap is a property of the request,
    /// not of the machine's memory, and putting one here would quietly truncate
    /// every long answer.
    pub fn llama_cpp_args(self) -> Vec<String> {
        let cache_type_v = match self {
            HardwareProfile::Low => "q4_0",
            HardwareProfile::Balanced | HardwareProfile::High => "q8_0",
        };
        vec![
            "--flash-attn".to_string(),
            "on".to_string(),
            "--cache-type-k".to_string(),
            "q8_0".to_string(),
            "--cache-type-v".to_string(),
            cache_type_v.to_string(),
            "--ctx-size".to_string(),
            self.ctx_tokens().to_string(),
            "--batch-size".to_string(),
            self.batch_size().to_string(),
            "--parallel".to_string(),
            "1".to_string(),
        ]
    }

    /// The word this profile goes by in the config file's `hardware_profile`.
    pub fn key(self) -> &'static str {
        match self {
            HardwareProfile::Low => "low",
            HardwareProfile::Balanced => "balanced",
            HardwareProfile::High => "high",
        }
    }

    /// The profile a config word names, or `None` for anything else — including
    /// `auto`, which is not a profile but a request to ask the machine.
    pub fn from_key(value: &str) -> Option<HardwareProfile> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" => Some(HardwareProfile::Low),
            "balanced" => Some(HardwareProfile::Balanced),
            "high" => Some(HardwareProfile::High),
            _ => None,
        }
    }

    /// Choose a profile from total RAM in KiB, the unit `/proc/meminfo` reports.
    ///
    /// The sizes in this file's profile definitions are VRAM bands, and most
    /// machines running xencode have no usable VRAM to measure — this box has a
    /// 2 GB MX250 while the model it runs lives in system memory. So the choice
    /// is made against RAM, where the limit that actually bites is the model's
    /// own resident weight plus the context on top of it: under 8 GiB there is no
    /// room for both and the narrow window is the honest one; 24 GiB and up can
    /// hold a 16k window without the kernel evicting anything. These are reasoned
    /// thresholds, not measured ones, which is exactly why `ProfileDecision`
    /// reports which of them decided and why the config can overrule the probe.
    pub fn select_from_total_ram_kib(kib: u64) -> HardwareProfile {
        const GIB: u64 = 1024 * 1024;
        if kib < 8 * GIB {
            HardwareProfile::Low
        } else if kib < 24 * GIB {
            HardwareProfile::Balanced
        } else {
            HardwareProfile::High
        }
    }
}

/// Total RAM this machine reports, in KiB. `None` where the kernel's own file is
/// absent or unreadable — a non-Linux host, or a container with a stripped
/// `/proc` — which is the case the caller must handle rather than guess at.
pub fn total_memory_kib() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo_total_kib(&meminfo)
}

/// Read `MemTotal` out of `/proc/meminfo` text: `MemTotal:  16141080 kB` →
/// `Some(16141080)`. Separate from the reader so the arithmetic can be tested
/// without depending on the machine it runs on.
pub fn parse_meminfo_total_kib(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let value = line.split_whitespace().nth(1)?;
    value.parse().ok()
}

/// The profile chosen for this run, and the reason in words. The reason exists
/// because a budget that arrives by probe is a claim a user can only check if it
/// is printed: "BALANCED profile from 15.4 GiB of RAM" can be disagreed with,
/// and `hardware_profile` in the config is how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileDecision {
    pub profile: HardwareProfile,
    /// Completes the sentence `{PROFILE} profile {reason}`.
    pub reason: String,
}

impl ProfileDecision {
    /// Resolve the profile: the config word if it names one, then the machine's
    /// reported memory, then `Balanced` — which is the profile every build of
    /// xencode used before there was a probe, so a machine that cannot be read
    /// behaves exactly as it did.
    pub fn resolve(config_value: &str) -> ProfileDecision {
        let value = config_value.trim();
        if value.is_empty() || value.eq_ignore_ascii_case("auto") {
            return from_machine();
        }
        if let Some(profile) = HardwareProfile::from_key(value) {
            return ProfileDecision {
                profile,
                reason: "set in config".to_string(),
            };
        }
        // A word that names no profile is a typo, and a typo must not quietly
        // become someone's budget. The machine decides, and says what was refused.
        let decided = from_machine();
        ProfileDecision {
            profile: decided.profile,
            reason: format!(
                "{}, though the config said \"{value}\", which is not a profile",
                decided.reason
            ),
        }
    }

    /// One line, for the user who is about to have context trimmed by it.
    pub fn describe(&self) -> String {
        format!("{} profile {}", self.profile.name(), self.reason)
    }
}

fn from_machine() -> ProfileDecision {
    match total_memory_kib() {
        Some(kib) => ProfileDecision {
            profile: HardwareProfile::select_from_total_ram_kib(kib),
            reason: format!(
                "from {} of RAM",
                format_memory_size(kib).unwrap_or_else(|| format!("{kib} KiB"))
            ),
        },
        None => ProfileDecision {
            profile: HardwareProfile::Balanced,
            reason: "this machine reported no memory size, so the default applies".to_string(),
        },
    }
}

/// KiB as a human size: 16141080 KiB → "15.4 GiB". `None` for values that would
/// print as zero or negative, which are not sizes worth claiming.
pub fn format_memory_size(kib: u64) -> Option<String> {
    let gib = kib as f64 / (1024.0 * 1024.0);
    if gib < 0.05 {
        return None;
    }
    Some(format!("{gib:.1} GiB"))
}

/// Deterministic token estimate: `ceil(chars / 4)` prose, `ceil(chars / 3)`
/// for code bodies (§10).
pub fn est_tokens(chars: usize, is_code: bool) -> u64 {
    let divisor = if is_code { 3 } else { 4 };
    chars.div_ceil(divisor) as u64
}

/// Take the head of `text` up to `max_tokens`, cutting at a line boundary, and
/// return `(kept, kept_tokens)`.
pub fn truncate_to_tokens(text: &str, max_tokens: u64, is_code: bool) -> (String, u64) {
    if max_tokens == 0 {
        return (String::new(), 0);
    }
    let divisor = if is_code { 3 } else { 4 } as u64;
    let max_chars = (max_tokens * divisor) as usize;
    if text.len() <= max_chars {
        return (text.to_string(), est_tokens(text.len(), is_code));
    }
    let mut cut = max_chars.min(text.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    if let Some(rel) = text[..cut].rfind('\n') {
        cut = rel + 1;
    }
    let head = &text[..cut];
    (head.to_string(), est_tokens(head.len(), is_code))
}

/// Take the *tail* of `text` up to `max_tokens` so the most recent content
/// wins (§10 tier 7: "most-recent messages win (drop oldest first)").
pub fn truncate_tail_to_tokens(text: &str, max_tokens: u64, is_code: bool) -> (String, u64) {
    if max_tokens == 0 {
        return (String::new(), 0);
    }
    let divisor = if is_code { 3 } else { 4 } as u64;
    let max_chars = (max_tokens * divisor) as usize;
    if text.len() <= max_chars {
        return (text.to_string(), est_tokens(text.len(), is_code));
    }
    let mut start = text.len() - max_chars;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    let tail = &text[start..];
    // Back up to a line boundary for a cleaner cut when possible; with no
    // newline in the window keep the whole window (a mid-line cut still
    // beats dropping everything).
    let cut = tail.find('\n').map(|i| i + 1).unwrap_or(0);
    let tail = &tail[cut.min(tail.len())..];
    (tail.to_string(), est_tokens(tail.len(), is_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimators_are_ceil_based() {
        assert_eq!(est_tokens(0, false), 0);
        assert_eq!(est_tokens(4, false), 1);
        assert_eq!(est_tokens(5, false), 2);
        assert_eq!(est_tokens(4, true), 2);
        assert_eq!(est_tokens(3, true), 1);
    }

    #[test]
    fn truncate_cuts_at_line_boundary_and_under_cap() {
        let text = "aaaaaa\nbbbbbb\ncccccc\nddddd\n";
        let (kept, tokens) = truncate_to_tokens(text, 3, false);
        // 3 tokens prose = 12 chars max → only the first line fits.
        assert_eq!(kept, "aaaaaa\n");
        assert_eq!(kept.len(), 7);
        assert_eq!(tokens, est_tokens(kept.len(), false));
        assert!(tokens <= 3);
    }

    #[test]
    fn truncate_keeps_short_text_untouched() {
        let (kept, tokens) = truncate_to_tokens("short", 10, false);
        assert_eq!(kept, "short");
        assert_eq!(tokens, est_tokens(5, false));
    }

    #[test]
    fn truncate_tail_keeps_most_recent_content() {
        let text = "line1-old\nline2-mid\nline3-new\n";
        let (kept, _) = truncate_tail_to_tokens(text, 3, false);
        assert!(kept.contains("line3-new"));
        assert!(!kept.contains("line1-old"));
        // Short text passes through whole.
        let (kept, _) = truncate_tail_to_tokens(text, 100, false);
        assert_eq!(kept, text);
    }

    #[test]
    fn truncate_tail_single_line_hard_cuts_instead_of_empty() {
        let text = "x".repeat(200);
        let (kept, tokens) = truncate_tail_to_tokens(&text, 3, false);
        // 3 tokens prose = 12-char window; no newline, so the whole window
        // survives as a mid-line cut — never an empty string.
        assert_eq!(kept, "x".repeat(12));
        assert_eq!(tokens, est_tokens(12, false));
    }

    #[test]
    fn profiles_have_monotonic_resources() {
        assert!(HardwareProfile::Low.ctx_tokens() < HardwareProfile::Balanced.ctx_tokens());
        assert!(HardwareProfile::Balanced.ctx_tokens() < HardwareProfile::High.ctx_tokens());
        assert!(HardwareProfile::Low.top_k() < HardwareProfile::Balanced.top_k());
        assert_ne!(
            HardwareProfile::Balanced.top_k(),
            HardwareProfile::High.top_k()
        );
        assert!(HardwareProfile::Low.batch_size() < HardwareProfile::Balanced.batch_size());
        assert!(HardwareProfile::Balanced.batch_size() < HardwareProfile::High.batch_size());
    }

    /// The preset a self-spawned server starts from, written out in full for all
    /// three profiles. `llama-server` refuses to start at all on a flag it cannot
    /// parse — `--flash-attn` used to take no value and now takes `on|off|auto` —
    /// so this list is the difference between a server coming up and a TUI that
    /// says nothing happened.
    #[test]
    fn each_profile_starts_the_server_with_its_own_preset() {
        assert_eq!(
            HardwareProfile::Low.llama_cpp_args(),
            [
                "--flash-attn",
                "on",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q4_0",
                "--ctx-size",
                "4096",
                "--batch-size",
                "512",
                "--parallel",
                "1",
            ]
        );
        assert_eq!(
            HardwareProfile::Balanced.llama_cpp_args(),
            [
                "--flash-attn",
                "on",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--ctx-size",
                "8192",
                "--batch-size",
                "2048",
                "--parallel",
                "1",
            ]
        );
        assert_eq!(
            HardwareProfile::High.llama_cpp_args(),
            [
                "--flash-attn",
                "on",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--ctx-size",
                "16384",
                "--batch-size",
                "4096",
                "--parallel",
                "1",
            ]
        );
    }

    /// A flag with nothing after it is a flag that reads the *next* flag as its
    /// value, which is how `--flash-attn --cache-type-k …` aborts a launch on
    /// b10809 with `unknown value for --flash-attn`.
    #[test]
    fn every_flag_is_followed_by_the_value_it_takes() {
        for profile in [
            HardwareProfile::Low,
            HardwareProfile::Balanced,
            HardwareProfile::High,
        ] {
            let args = profile.llama_cpp_args();
            assert!(args.len() % 2 == 0, "{args:?} does not pair up");
            for pair in args.chunks(2) {
                assert!(pair[0].starts_with("--"), "{} has no value", pair[0]);
                assert!(
                    !pair[1].starts_with("--"),
                    "{} is missing its value",
                    pair[0]
                );
            }
        }
    }

    /// Measured with `--n-predict 8`: a reply that asked for no limit of its own
    /// stopped after eight tokens with `finish_reason: "length"`. How long an
    /// answer may run is not a property of the machine's memory, so no profile
    /// gets to decide it at launch.
    #[test]
    fn no_profile_caps_generation_length_at_launch() {
        for profile in [
            HardwareProfile::Low,
            HardwareProfile::Balanced,
            HardwareProfile::High,
        ] {
            assert!(
                !profile
                    .llama_cpp_args()
                    .iter()
                    .any(|arg| arg.starts_with("--n-predict")),
                "{} caps generation at launch",
                profile.name()
            );
        }
    }

    /// One slot on every profile, because that is what keeps the window the
    /// budget is filling: `llama-server` divides `--ctx-size` by the slot count.
    #[test]
    fn every_profile_runs_a_single_slot() {
        for profile in [
            HardwareProfile::Low,
            HardwareProfile::Balanced,
            HardwareProfile::High,
        ] {
            let args = profile.llama_cpp_args();
            let slot = args
                .iter()
                .position(|a| a == "--parallel")
                .and_then(|i| args.get(i + 1));
            assert_eq!(slot, Some(&"1".to_string()), "{}", profile.name());
        }
    }

    /// The bands are stated in whole GiB, so the boundary is where a machine
    /// changes budget without changing anything else.
    #[test]
    fn profile_bands_follow_total_ram() {
        let gib = 1024u64 * 1024;
        assert_eq!(
            HardwareProfile::select_from_total_ram_kib(4 * gib),
            HardwareProfile::Low
        );
        assert_eq!(
            HardwareProfile::select_from_total_ram_kib(8 * gib),
            HardwareProfile::Balanced
        );
        assert_eq!(
            HardwareProfile::select_from_total_ram_kib(24 * gib - 1),
            HardwareProfile::Balanced
        );
        assert_eq!(
            HardwareProfile::select_from_total_ram_kib(32 * gib),
            HardwareProfile::High
        );
    }

    /// A laptop's `MemTotal` — a real reading from the machine this was written
    /// on — has to land on the profile that machine behaved as before the probe
    /// existed. Anything else is a silent change of budget for everyone who
    /// upgraded.
    #[test]
    fn a_measured_laptop_lands_on_the_profile_it_had() {
        let meminfo = "MemTotal:       16141080 kB\nMemFree:         1234567 kB\n";
        let kib = parse_meminfo_total_kib(meminfo).expect("MemTotal should parse");
        assert_eq!(kib, 16_141_080);
        assert_eq!(
            HardwareProfile::select_from_total_ram_kib(kib),
            HardwareProfile::Balanced
        );
        assert_eq!(format_memory_size(kib).as_deref(), Some("15.4 GiB"));
    }

    #[test]
    fn meminfo_without_a_readable_total_is_not_an_answer() {
        assert_eq!(parse_meminfo_total_kib(""), None);
        assert_eq!(
            parse_meminfo_total_kib("MemFree: 123 kB"),
            None,
            "a file with no MemTotal must not decide a budget"
        );
        assert_eq!(
            parse_meminfo_total_kib("MemTotal: many kB"),
            None,
            "a value that is not a number is not a size"
        );
        assert_eq!(format_memory_size(0), None);
    }

    /// The config is the escape hatch, and it is also the thing that must be
    /// believed when it is wrong: a machine with 8 GiB told to budget like a
    /// 24 GiB one gets the wide window, because the user may know their model
    /// better than the probe does.
    #[test]
    fn config_overrides_the_probe_and_a_typo_does_not_become_a_budget() {
        for (value, expected) in [
            ("low", HardwareProfile::Low),
            ("BALANCED", HardwareProfile::Balanced),
            ("  high  ", HardwareProfile::High),
        ] {
            let decision = ProfileDecision::resolve(value);
            assert_eq!(decision.profile, expected, "{value} should be honoured");
            assert_eq!(decision.reason, "set in config");
            assert!(decision
                .describe()
                .starts_with(&format!("{} profile", expected.name())));
        }
        // "auto" and empty both mean: ask the machine, whatever it says.
        let auto = ProfileDecision::resolve("auto");
        let empty = ProfileDecision::resolve("");
        assert_eq!(auto, empty);
        // A word that is not a profile must not be read as one, and must say so.
        let typo = ProfileDecision::resolve("ballanced");
        assert_eq!(typo.profile, auto.profile);
        assert!(
            typo.reason.contains("\"ballanced\""),
            "the refused value has to appear in the reason: {typo:?}",
        );
    }

    /// Every profile survives a round trip through the config word for it, and
    /// `auto` is not one of them — it is the absence of a choice.
    #[test]
    fn profile_config_words_round_trip() {
        for profile in [
            HardwareProfile::Low,
            HardwareProfile::Balanced,
            HardwareProfile::High,
        ] {
            assert_eq!(HardwareProfile::from_key(profile.key()), Some(profile));
        }
        assert_eq!(HardwareProfile::from_key("auto"), None);
        assert_eq!(HardwareProfile::from_key("48gb"), None);
    }
}
