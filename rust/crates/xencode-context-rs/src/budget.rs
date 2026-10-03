//! Hardware profiles + deterministic token estimator (§10, §14).
//!
//! Profiles are data, never hardcoded logic — logic reads the fields here.
//! The estimator is deliberately cheap and is what the tier budget is filled
//! with; it is ±20–30 % on code, so anywhere a number has to be *right* the
//! server is asked instead: `llama.cpp`'s `/tokenize` for a prompt before it is
//! sent, and the `usage` on the answer after it. Those real counts are what the
//! metrics layer displays and what [`PromptOverhead`] averages.

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

    /// The profile one rung down, or `None` at the bottom rung.
    ///
    /// This is the whole of what a crossed daily budget can do to a turn: give
    /// up some of the next one's room rather than refuse it. A refusal strands a
    /// half-finished edit, and the profile is the one dial that changes every
    /// part of a turn at once — the window it fills, how many files it retrieves,
    /// how much of each file it sends.
    pub fn step_down(self) -> Option<HardwareProfile> {
        match self {
            HardwareProfile::High => Some(HardwareProfile::Balanced),
            HardwareProfile::Balanced => Some(HardwareProfile::Low),
            HardwareProfile::Low => None,
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

/// How many tokens the fill target leaves for one retrieved file before xencode
/// stops counting that file as worth asking for. 512 tokens is roughly 1.5 KiB of
/// source at the code estimate divisor — about a function — and a body smaller
/// than that has usually been cut to a fragment that answers nothing.
pub const TOKENS_PER_RETRIEVED_FILE: u64 = 512;

/// How much of a prompt a turn may fill with retrieved file bodies, and how big
/// one body may be.
///
/// These were profile constants only, which made them a guess about a window
/// xencode had not looked at: a laptop on a `Balanced` profile whose server runs
/// `--ctx-size 2048` was still asked to retrieve five files of up to 16 KiB each,
/// and the budgeter threw four of them away after the work of reading them was
/// done. [`ContextCaps::for_free_space`] derives both numbers from the space the
/// prompt actually leaves instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCaps {
    /// How many retrieved files to read bodies for.
    pub top_k: usize,
    /// Characters to keep per file body.
    pub content_cap_chars: usize,
}

impl ContextCaps {
    /// One file, the smallest body still worth sending. The floor exists because
    /// `top_k` of zero means "retrieve nothing", which is a different decision
    /// from "there was no room" and belongs to the caller.
    pub const MIN_TOP_K: usize = 1;
    /// Eight files, the widest ladder rung any profile has ever asked for. More
    /// than this is not derived from free space: no configuration past the widest
    /// measured rung has been run here, so a bigger number would be a guess.
    pub const MAX_TOP_K: usize = 8;
    /// About `TOKENS_PER_RETRIEVED_FILE` at the code estimate divisor, which is
    /// what one file's minimum share of the budget works out to.
    pub const MIN_CONTENT_CAP_CHARS: usize = 1_536;
    /// The largest body any profile was ever given.
    pub const MAX_CONTENT_CAP_CHARS: usize = 24_000;

    /// What this run asks for when nothing has been measured yet — the profile's
    /// own ladder, which is every build's behaviour before there was a reading.
    pub fn from_profile(profile: HardwareProfile) -> ContextCaps {
        ContextCaps {
            top_k: profile.top_k(),
            content_cap_chars: profile.content_cap_chars(),
        }
    }

    /// Split `free_tokens` — the fill target minus what a prompt is known to cost
    /// — between how many files to fetch and how much of each to keep.
    ///
    /// More free space means more files up to [`MAX_TOP_K`], and once the count is
    /// capped it means a bigger slice of each file. Both numbers move together so
    /// their product cannot exceed the space they were derived from.
    pub fn for_free_space(free_tokens: u64) -> ContextCaps {
        let top_k = usize::try_from(free_tokens / TOKENS_PER_RETRIEVED_FILE)
            .unwrap_or(usize::MAX)
            .clamp(Self::MIN_TOP_K, Self::MAX_TOP_K);
        // Three characters to a token is the estimate's divisor for code, the
        // conservative one: it under-promises room rather than over-promising it.
        let per_file_chars = free_tokens
            .saturating_mul(3)
            .checked_div(top_k as u64)
            .unwrap_or(Self::MAX_CONTENT_CAP_CHARS as u64);
        let content_cap_chars = usize::try_from(per_file_chars)
            .unwrap_or(Self::MAX_CONTENT_CAP_CHARS)
            .clamp(Self::MIN_CONTENT_CAP_CHARS, Self::MAX_CONTENT_CAP_CHARS);
        ContextCaps {
            top_k,
            content_cap_chars,
        }
    }

    /// The caps for one turn: from measured free space where a prompt size is
    /// known, from the profile ladder where it is not. `None` is not zero free
    /// space — it is the absence of a reading, and it keeps the old behaviour.
    pub fn for_turn(profile: HardwareProfile, free_tokens: Option<u64>) -> ContextCaps {
        match free_tokens {
            Some(free) => ContextCaps::for_free_space(free),
            None => ContextCaps::from_profile(profile),
        }
    }
}

/// What the parts of a prompt that are not retrieved files cost, in the server's
/// own tokens: the system head, the guidelines files, state, the git summary, the
/// conversation so far, the chat template's framing.
///
/// The total is measured and only the split is arithmetic: a completion reports
/// `prompt_tokens` for the whole prompt, and xencode knows how many characters of
/// it were retrieved bodies, so the share is `characters / characters`. Measured
/// on a 23,003-character prompt from this repository — 18,197 of it retrieved file
/// bodies — that came to 1,204 tokens against the 1,166 the server counted for the
/// same parts individually, three percent high. The estimator this replaces was
/// 32 % high on the retrieved half alone: `chars / 3` on those 18,197 characters
/// predicts 6,066 tokens where the server said 4,587.
///
/// It is tracked as an average rather than as the last reading because it changes
/// with the conversation — a long tool result in the history raises it for the
/// turns that follow — and the retrieval caps are asked for *before* the prompt is
/// built, so a number that jumped every turn would make the caps oscillate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptOverhead {
    tokens: Option<u64>,
}

impl PromptOverhead {
    /// An average that moves about a quarter of the way toward each new reading.
    /// Slow enough that one unusually large reply does not empty the next
    /// turn's retrieval, fast enough to follow a conversation that grows.
    const NEW_WEIGHT: u64 = 1;
    const SEEN_WEIGHT: u64 = 3;
    /// The step the average is reported in. Quantising is what stops a few dozen
    /// tokens of drift changing a budget: the assembly keeps `MARGIN_TOKENS` (200)
    /// of headroom anyway, so anything finer than that is noise being acted on.
    const STEP_TOKENS: u64 = 256;

    /// Record what a completion cost.
    ///
    /// `prompt_tokens` is the server's count for the whole prompt;
    /// `retrieved_chars` and `prompt_chars` are what xencode put in it, so
    /// `prompt_chars - retrieved_chars` is the part that is not file bodies.
    /// A reading with no prompt tokens, or a prompt of no characters, says
    /// nothing and is skipped rather than becoming a zero.
    pub fn observe(&mut self, prompt_tokens: u64, retrieved_chars: usize, prompt_chars: usize) {
        if prompt_tokens == 0 || prompt_chars == 0 {
            return;
        }
        let not_retrieved = prompt_chars.saturating_sub(retrieved_chars) as u64;
        let sample = prompt_tokens.saturating_mul(not_retrieved) / prompt_chars as u64;
        self.tokens = Some(match self.tokens {
            None => sample,
            Some(seen) => {
                (seen * Self::SEEN_WEIGHT + sample * Self::NEW_WEIGHT)
                    / (Self::SEEN_WEIGHT + Self::NEW_WEIGHT)
            }
        });
    }

    /// The average as it stands, rounded to the reported step. `None` until the
    /// first reading.
    pub fn tokens(&self) -> Option<u64> {
        self.tokens
            .map(|t| (t + Self::STEP_TOKENS / 2) / Self::STEP_TOKENS * Self::STEP_TOKENS)
    }

    /// What the fill target has left for retrieval once the overhead is paid.
    /// `None` when nothing has been measured — see [`ContextCaps::for_turn`].
    /// A prompt already over budget leaves nothing, which is what the subtraction
    /// says; the overflow itself is reported by the assembler, not hidden here.
    pub fn free_tokens(&self, target: u64) -> Option<u64> {
        self.tokens()
            .map(|overhead| target.saturating_sub(overhead))
    }
}

/// The number of tokens one turn's prompt is aimed at: the window it is going to,
/// times how much of it the profile is willing to fill. Shared by the assembler
/// and the caps so the two cannot disagree about what the budget was.
pub fn fill_target(profile: HardwareProfile, window_tokens: Option<u32>) -> u64 {
    let window = window_tokens.unwrap_or(profile.ctx_tokens() as u32);
    (window as f64 * profile.utilization()).floor() as u64
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

/// The four caps a person can set for one day, each in the unit its own source
/// reports. Every one is unset by default, and none of them stops a turn.
///
/// A cap that refused a turn would strand whatever that turn had already changed
/// on disk, which is the documented way agent frameworks lose people's work. So a
/// crossed cap buys down the next turn instead — see [`DailyBudgets::breach`] and
/// [`HardwareProfile::step_down`] — and the day keeps being spent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DailyBudgets {
    /// Prompt plus completion tokens.
    pub tokens: Option<u64>,
    /// Watt-hours the CPU package counted. Only a machine with a readable
    /// counter can ever cross this one; see [`crate::power::package_energy_uj`].
    pub energy_wh: Option<u64>,
    /// What the day's tokens cost at the rates in `pricing.json`, in millionths
    /// of a dollar. Provider spend alone: the electricity a local turn drew is
    /// what [`Self::energy_wh`] weighs, and the two are different documents.
    pub usd_micros: Option<u64>,
    /// Wall-clock the turns ran, in minutes. Time the interface sat open is not
    /// usage and is not counted.
    pub minutes: Option<u64>,
}

/// Which of the four caps was passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDimension {
    Tokens,
    Energy,
    ProviderCost,
    WallClock,
}

impl BudgetDimension {
    /// The word the cap is set by, for a message that has to be searchable in
    /// the manual.
    pub fn label(self) -> &'static str {
        match self {
            BudgetDimension::Tokens => "token",
            BudgetDimension::Energy => "energy",
            BudgetDimension::ProviderCost => "dollar",
            BudgetDimension::WallClock => "wall-clock",
        }
    }
}

/// A cap today's usage has reached, with both numbers worded in the same unit so
/// they read as one sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetBreach {
    pub dimension: BudgetDimension,
    pub used: String,
    pub cap: String,
    /// `used / cap`, which is how the worst of several crossings is picked.
    pub over: f64,
}

impl DailyBudgets {
    /// Whether any cap is set at all — the cheap answer that lets a caller skip
    /// the rest when nothing is configured.
    pub fn any_set(&self) -> bool {
        self.tokens.is_some()
            || self.energy_wh.is_some()
            || self.usd_micros.is_some()
            || self.minutes.is_some()
    }

    /// The cap today has passed by the widest margin, or `None` while every set
    /// cap still has room.
    ///
    /// `priced_micros` is what the day's tokens cost at `pricing.json` counting
    /// only the models the table knows. A model with no price contributes
    /// nothing, so a day of unpriced models cannot cross a dollar cap: the figure
    /// is a floor, and a floor is allowed to fire but never to say the day stayed
    /// under. `/cost` names the unpriced models; this does not invent them.
    ///
    /// The energy cap works the same way in reverse — it is compared against
    /// nothing when the machine reports no counter, which is a counter that
    /// stayed silent rather than a day that used no power.
    pub fn breach(
        &self,
        day: &crate::rollup::DayTotals,
        priced_micros: u64,
    ) -> Option<BudgetBreach> {
        let tokens = day.tokens.prompt_tokens + day.tokens.completion_tokens;
        let watt_hours = day.watt_hours();
        let minutes = day.active_ms as f64 / 60_000.0;
        [
            crossing(
                BudgetDimension::Tokens,
                tokens as f64,
                self.tokens.map(|cap| cap as f64),
                BudgetUnit::Tokens,
            ),
            crossing(
                BudgetDimension::Energy,
                watt_hours.unwrap_or(0.0),
                self.energy_wh.map(|cap| cap as f64),
                BudgetUnit::WattHours,
            ),
            crossing(
                BudgetDimension::ProviderCost,
                priced_micros as f64,
                self.usd_micros.map(|cap| cap as f64),
                BudgetUnit::Dollars,
            ),
            crossing(
                BudgetDimension::WallClock,
                minutes,
                self.minutes.map(|cap| cap as f64),
                BudgetUnit::Minutes,
            ),
        ]
        .into_iter()
        .flatten()
        .max_by(|a, b| a.over.total_cmp(&b.over))
    }

    /// Today's figure against every cap that is set, as the lines `/cost` prints
    /// so a bought-down turn can be checked against the numbers that bought it
    /// down. A dimension the day has no figure for is said as that, and not as
    /// zero: a machine publishing no energy counter has not used no power, and a
    /// model with no entry in `pricing.json` has not cost nothing.
    pub fn today_lines(
        &self,
        day: &crate::rollup::DayTotals,
        report: &crate::pricing::CostReport,
    ) -> Vec<String> {
        // Whether a day of unpriced models has anything to weigh against the
        // dollar cap. A partial table still weighs what it knows, which is a
        // floor, and the line says which models are missing.
        let priced =
            (report.complete() || report.known_micros > 0).then_some(report.known_micros as f64);
        let cost_note = if report.unpriced.is_empty() {
            String::new()
        } else {
            format!(
                " — price unknown for {} model{}",
                report.unpriced.len(),
                if report.unpriced.len() == 1 { "" } else { "s" }
            )
        };
        [
            (
                BudgetDimension::Tokens,
                self.tokens,
                Some((day.tokens.prompt_tokens + day.tokens.completion_tokens) as f64),
                String::new(),
            ),
            (BudgetDimension::Energy, self.energy_wh, day.watt_hours(), String::new()),
            (
                BudgetDimension::ProviderCost,
                self.usd_micros,
                priced,
                cost_note,
            ),
            (
                BudgetDimension::WallClock,
                self.minutes,
                Some(day.active_ms as f64 / 60_000.0),
                String::new(),
            ),
        ]
        .into_iter()
        .filter_map(|(dimension, cap, used, note)| {
            let cap = cap?;
            let unit = BudgetUnit::of(dimension);
            let cap_words = unit.words(cap as f64);
            Some(match used {
                Some(used) => format!(
                    "  • {} cap {cap_words} · today {} · {}",
                    dimension.label(),
                    unit.words(used),
                    if used >= cap as f64 {
                        "passed"
                    } else {
                        "room left"
                    }
                ) + &note,
                None => format!(
                    "  • {} cap {cap_words} · nothing this machine reported to weigh against it today",
                    dimension.label()
                ) + &note,
            })
        })
        .collect()
    }
}

/// The unit a dimension is written in, so a cap and its figure always read alike.
impl BudgetUnit {
    fn of(dimension: BudgetDimension) -> BudgetUnit {
        match dimension {
            BudgetDimension::Tokens => BudgetUnit::Tokens,
            BudgetDimension::Energy => BudgetUnit::WattHours,
            BudgetDimension::ProviderCost => BudgetUnit::Dollars,
            BudgetDimension::WallClock => BudgetUnit::Minutes,
        }
    }
}

/// One usage figure against one cap. `None` when no cap is set for the
/// dimension, or when the day has not reached it yet — and never for a day that
/// recorded nothing, since 0 against a 0-token cap has not been crossed, it has
/// not started.
fn crossing(
    dimension: BudgetDimension,
    used: f64,
    cap: Option<f64>,
    unit: BudgetUnit,
) -> Option<BudgetBreach> {
    let cap = cap?;
    if used <= 0.0 || used < cap {
        return None;
    }
    Some(BudgetBreach {
        dimension,
        used: unit.words(used),
        cap: unit.words(cap),
        over: used / cap,
    })
}

/// The unit a cap is written in, so `used` and `cap` always read the same way.
#[derive(Debug, Clone, Copy)]
enum BudgetUnit {
    Tokens,
    WattHours,
    Dollars,
    Minutes,
}

impl BudgetUnit {
    fn words(self, value: f64) -> String {
        match self {
            BudgetUnit::Tokens => format!("{} tokens", trim_count(value)),
            BudgetUnit::WattHours => format!("{} Wh", trim_count(value)),
            // Money goes through the same formatter `/cost` uses, because a
            // second one would be a second opinion about the same figure.
            BudgetUnit::Dollars => crate::pricing::format_usd(value.round() as u64),
            BudgetUnit::Minutes => format!("{} min", trim_count(value)),
        }
    }
}

/// A count that should not wear a decimal point it did not ask for: 12000 tokens
/// prints as `12000 tokens`, and 2.5 Wh prints as `2.5 Wh`.
fn trim_count(value: f64) -> String {
    if value >= 999_999.5 || (value - value.round()).abs() >= 0.05 {
        return format!("{value:.1}");
    }
    format!("{}", value.round() as u64)
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

    /// A run that has measured nothing asks for what its profile always asked
    /// for. This is the whole point of the fallback: turning AC-4 on must not
    /// change anyone's budget until there is a number behind it.
    #[test]
    fn an_unmeasured_run_keeps_its_profiles_numbers() {
        for profile in [
            HardwareProfile::Low,
            HardwareProfile::Balanced,
            HardwareProfile::High,
        ] {
            let caps = ContextCaps::for_turn(profile, None);
            assert_eq!(caps, ContextCaps::from_profile(profile));
        }
    }

    /// Free space is spent, not averaged: a wide window buys more files and then
    /// bigger ones, and a narrow one buys neither. The bounds are the rungs the
    /// profile ladder already had, so the widest ask in the codebase stays
    /// `ContextCaps::MAX_TOP_K` files of `MAX_CONTENT_CAP_CHARS` each.
    #[test]
    fn caps_follow_the_room_a_prompt_left() {
        // One function's worth of space: one small file, not nothing.
        assert_eq!(
            ContextCaps::for_free_space(128),
            ContextCaps {
                top_k: 1,
                content_cap_chars: ContextCaps::MIN_CONTENT_CAP_CHARS,
            }
        );
        // A narrow llama.cpp window still leaves room for a couple of files.
        assert_eq!(
            ContextCaps::for_free_space(1_200),
            ContextCaps {
                top_k: 2,
                content_cap_chars: 1_800,
            }
        );
        // Four files' worth: four files, each at its even share.
        assert_eq!(
            ContextCaps::for_free_space(4 * TOKENS_PER_RETRIEVED_FILE),
            ContextCaps {
                top_k: 4,
                content_cap_chars: (TOKENS_PER_RETRIEVED_FILE * 3) as usize,
            }
        );
        // Past the widest ladder the file count stops growing and the slice does.
        let wide = ContextCaps::for_free_space(100_000);
        assert_eq!(wide.top_k, ContextCaps::MAX_TOP_K);
        assert_eq!(wide.content_cap_chars, ContextCaps::MAX_CONTENT_CAP_CHARS);
    }

    /// The trap named in the plan is oscillation, so the checks are that more
    /// room never buys less, and that the retrieval being asked for never costs
    /// more than the room it was derived from.
    #[test]
    fn caps_never_ask_for_more_than_the_room_allows() {
        let mut previous_top_k = 0usize;
        for free in (0u64..40_000).step_by(64) {
            let caps = ContextCaps::for_free_space(free);
            assert!(caps.top_k >= ContextCaps::MIN_TOP_K);
            assert!(caps.top_k <= ContextCaps::MAX_TOP_K);
            assert!(caps.content_cap_chars >= ContextCaps::MIN_CONTENT_CAP_CHARS);
            assert!(caps.content_cap_chars <= ContextCaps::MAX_CONTENT_CAP_CHARS);
            assert!(
                caps.top_k >= previous_top_k,
                "{free} tokens buys fewer files than {} did",
                free - 64
            );
            previous_top_k = caps.top_k;
            // Files at their even share add up to the free space exactly. The one
            // place the ask overshoots is the minimum-sized-file floor, where it
            // overshoots to exactly one file and no more.
            let ask_tokens = (caps.top_k as u64) * (caps.content_cap_chars / 3) as u64;
            assert!(
                ask_tokens <= free.max(TOKENS_PER_RETRIEVED_FILE),
                "{free} tokens asked for {ask_tokens} worth"
            );
        }
    }

    /// What the arithmetic reaches for a prompt this repository actually builds:
    /// a 23,003-character turn — 4,742 of stable head, 18,197 of retrieved file
    /// bodies, 60 of question — which `llama-server` b10809 counted at 5,766
    /// tokens, against 1,156 + 10 = 1,166 tokens for the parts that are not
    /// retrieval.
    #[test]
    fn a_measured_prompt_buys_more_files_and_smaller_ones() {
        let profile = HardwareProfile::Balanced;
        let target = fill_target(profile, Some(profile.ctx_tokens() as u32));
        assert_eq!(target, 6_144);

        let mut overhead = PromptOverhead::default();
        overhead.observe(5_766, 18_197, 23_003);
        // Within two percent of the 1,166 the server says that prompt's
        // non-retrieval parts cost, and reported in steps of 256.
        assert_eq!(overhead.tokens(), Some(1_280));
        let unmeasured = ContextCaps::from_profile(profile);
        let caps = ContextCaps::for_turn(profile, overhead.free_tokens(target));
        // Room for nine files' worth, of which eight is the most anything asks
        // for, so each file is cut to its share — instead of the fifth file being
        // dropped by the budgeter after it had already been read from disk.
        assert_eq!(caps.top_k, ContextCaps::MAX_TOP_K);
        assert_eq!(caps.top_k, unmeasured.top_k + 3);
        assert_eq!(caps.content_cap_chars, (6_144 - 1_280) * 3 / 8);
        assert!(caps.content_cap_chars < unmeasured.content_cap_chars);

        // A prompt whose fixed parts already fill the target leaves nothing, and
        // one file's worth is the floor rather than zero.
        let mut crowded = PromptOverhead::default();
        crowded.observe(9_000, 900, 9_000);
        let caps = ContextCaps::for_turn(profile, crowded.free_tokens(target));
        assert_eq!(caps.top_k, ContextCaps::MIN_TOP_K);
    }

    /// A reading is only as good as the counts behind it: neither a server that
    /// said nothing nor a prompt of no characters may look like a free turn.
    #[test]
    fn a_reading_of_nothing_is_not_a_reading_of_zero() {
        let mut overhead = PromptOverhead::default();
        assert_eq!(overhead.tokens(), None);
        overhead.observe(0, 100, 5_000);
        assert_eq!(overhead.tokens(), None);
        overhead.observe(3_000, 100, 0);
        assert_eq!(overhead.tokens(), None);
    }

    /// The average moves toward a new reading instead of becoming it, and a
    /// change of a few dozen tokens is not a change of budget.
    #[test]
    fn the_overhead_average_moves_slowly_and_in_steps() {
        let mut overhead = PromptOverhead::default();
        overhead.observe(2_000, 6_000, 12_000);
        let first = overhead.tokens();
        assert_eq!(first, Some(1_024));
        // A turn twice as expensive lifts the average, but not to twice.
        overhead.observe(4_000, 6_000, 12_000);
        let second = overhead.tokens();
        assert!(second > first);
        assert!(second.unwrap() < 2_560, "{second:?} became the new reading");
        // Drift below half a step is reported as the same number.
        let mut steady = PromptOverhead::default();
        steady.observe(2_048, 6_000, 12_000);
        let before = steady.tokens();
        steady.observe(2_060, 6_000, 12_000);
        assert_eq!(before, steady.tokens(), "a few tokens of drift");
    }

    /// More than half of a prompt can be the fixed cost of it — a long history,
    /// a wide-open window — and the free space is then what is left, never less
    /// than nothing and never more than the target.
    #[test]
    fn free_space_is_clamped_to_the_target() {
        let mut overhead = PromptOverhead::default();
        overhead.observe(99_000, 1, 1_000);
        assert_eq!(overhead.free_tokens(6_144), Some(0));
        assert_eq!(
            ContextCaps::for_turn(HardwareProfile::Low, overhead.free_tokens(6_144)).top_k,
            ContextCaps::MIN_TOP_K
        );
        assert_eq!(PromptOverhead::default().free_tokens(6_144), None);
    }

    /// The window a server reported is the window the budget is aimed at, and
    /// the profile contributes only how much of it may be filled.
    #[test]
    fn the_target_is_the_reported_window_not_the_profiles_guess() {
        assert_eq!(
            fill_target(HardwareProfile::Balanced, None),
            fill_target(HardwareProfile::Balanced, Some(8_192))
        );
        assert_eq!(fill_target(HardwareProfile::Balanced, Some(2_048)), 1_536);
        assert_eq!(fill_target(HardwareProfile::Low, Some(2_048)), 1_228);
    }

    /// A day of usage, built in the units the caps are written in.
    fn a_day(tokens: u64, energy_uj: u64, active_ms: u64) -> crate::rollup::DayTotals {
        crate::rollup::DayTotals {
            tokens: crate::rollup::TokenTotals {
                requests: 1,
                prompt_tokens: tokens,
                cached_tokens: 0,
                completion_tokens: 0,
            },
            by_model: std::collections::BTreeMap::new(),
            energy_uj,
            est_cost_micros: 0,
            active_ms,
        }
    }

    #[test]
    fn the_profile_ladder_has_a_bottom_and_never_a_refusal() {
        assert_eq!(
            HardwareProfile::High.step_down(),
            Some(HardwareProfile::Balanced)
        );
        assert_eq!(
            HardwareProfile::Balanced.step_down(),
            Some(HardwareProfile::Low)
        );
        assert_eq!(HardwareProfile::Low.step_down(), None);
    }

    #[test]
    fn a_cap_fires_at_the_number_it_names_and_not_before() {
        let budgets = DailyBudgets {
            tokens: Some(1000),
            ..Default::default()
        };
        assert!(budgets.breach(&a_day(999, 0, 0), 0).is_none());
        let breach = budgets.breach(&a_day(1000, 0, 0), 0).expect("at the cap");
        assert_eq!(breach.dimension, BudgetDimension::Tokens);
        assert_eq!(breach.used, "1000 tokens");
        assert_eq!(breach.cap, "1000 tokens");

        // A day that recorded nothing has not crossed even a zero cap: nothing
        // was spent, which is not the same as the limit being reached.
        assert!(budgets.breach(&a_day(0, 0, 0), 0).is_none());
        // And a cap nobody set cannot fire.
        assert!(!DailyBudgets::default().any_set());
        assert!(DailyBudgets::default()
            .breach(&a_day(9_999_999, 0, 0), 0)
            .is_none());
    }

    #[test]
    fn the_widest_crossing_is_the_one_that_gets_said() {
        let budgets = DailyBudgets {
            tokens: Some(1000),
            minutes: Some(60),
            ..Default::default()
        };
        // 2000 of 1000 tokens is twice over; 66 of 60 minutes is 1.1 times over.
        let breach = budgets
            .breach(&a_day(2000, 0, 66 * 60_000), 0)
            .expect("both caps are passed");
        assert_eq!(breach.dimension, BudgetDimension::Tokens);
        assert!((breach.over - 2.0).abs() < 1e-9, "{}", breach.over);
    }

    #[test]
    fn a_figure_the_machine_or_the_price_table_never_reported_cannot_cross() {
        // An energy cap against a silent counter: the day used power, and there
        // is no reading to compare. This is not a pass, and it is never said as
        // one — the cap simply has nothing to fire on.
        let energy_only = DailyBudgets {
            energy_wh: Some(1),
            ..Default::default()
        };
        assert!(energy_only.breach(&a_day(0, 0, 0), 0).is_none());
        // 3.6 MJ is exactly one watt-hour, so the same cap fires on a reading.
        let breach = energy_only
            .breach(&a_day(0, 3_600_000_000, 0), 0)
            .expect("one Wh against a one Wh cap");
        assert_eq!(breach.dimension, BudgetDimension::Energy);
        assert_eq!(breach.used, "1 Wh");

        // A dollar cap with nothing priced is a floor of zero, and a floor of
        // zero cannot be over a limit. The same cap fires once the table knows
        // the model.
        let dollars = DailyBudgets {
            usd_micros: Some(500_000),
            ..Default::default()
        };
        assert!(dollars.breach(&a_day(10_000, 0, 0), 0).is_none());
        let breach = dollars
            .breach(&a_day(10_000, 0, 0), 600_000)
            .expect("$0.60 against a $0.50 budget");
        assert_eq!(breach.used, "$0.6");
        assert_eq!(breach.cap, "$0.5");
    }

    #[test]
    fn today_lines_weigh_every_cap_that_can_be_weighed_and_name_the_ones_that_cannot() {
        let budgets = DailyBudgets {
            tokens: Some(1000),
            energy_wh: Some(50),
            usd_micros: Some(500_000),
            minutes: Some(1),
        };
        let day = a_day(1200, 0, 95_000);
        let unpriced = crate::pricing::CostReport {
            per_model: Vec::new(),
            known_micros: 0,
            unpriced: vec!["qwen3-0.6b".to_string()],
            ..Default::default()
        };
        assert_eq!(
            budgets.today_lines(&day, &unpriced),
            vec![
                "  • token cap 1000 tokens · today 1200 tokens · passed",
                "  • energy cap 50 Wh · nothing this machine reported to weigh against it today",
                "  • dollar cap $0.5 · nothing this machine reported to weigh against it today — price unknown for 1 model",
                "  • wall-clock cap 1 min · today 1.6 min · passed",
            ]
        );

        // A day inside its caps says so in the same shape, and a cap nobody set
        // takes up no line at all.
        let priced = crate::pricing::CostReport {
            per_model: Vec::new(),
            known_micros: 486,
            unpriced: Vec::new(),
            ..Default::default()
        };
        let quiet = DailyBudgets {
            tokens: Some(10_000),
            usd_micros: Some(500_000),
            ..Default::default()
        };
        assert_eq!(
            quiet.today_lines(&day, &priced),
            vec![
                "  • token cap 10000 tokens · today 1200 tokens · room left",
                "  • dollar cap $0.5 · today $0.000486 · room left",
            ]
        );
    }

    #[test]
    fn a_count_wears_a_decimal_point_only_when_it_asked_for_one() {
        assert_eq!(trim_count(12_000.0), "12000");
        assert_eq!(trim_count(2.5), "2.5");
        assert_eq!(trim_count(41.49), "41.5");
        // Past a point a rounded figure is a lie about precision, so it is
        // printed as the size it is.
        assert_eq!(trim_count(1_500_000.0), "1500000.0");
    }
}
