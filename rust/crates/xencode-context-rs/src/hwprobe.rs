//! What this machine can actually serve, read from the machine.
//!
//! The budget layer (`budget.rs`) decides how much context a run may ask for.
//! This module answers the question underneath it: what is this box, and what
//! would a local `llama-server` do with that? Every field is read from a named
//! source, and the sources disagree, which is the reason the probe exists at
//! all. On the laptop this was written on:
//!
//! | source | what it said |
//! |---|---|
//! | `/proc/meminfo` | 16 141 084 KiB total, 9 536 364 KiB available |
//! | `/sys/class/drm` | two cards: `nvidia` (0x10de/0x1d13) and `i915` (0x8086/0x8a56) |
//! | PCI BARs | the NVIDIA card's largest BAR is **256 MiB** |
//! | `nvidia-smi` | the same card has **2048 MiB** |
//! | `llama-server --list-devices` | `Vulkan1: NVIDIA GeForce MX250 (2294 MiB, 1156 MiB free)` |
//!
//! The BAR figure is the trap: a 2 GiB card exposes a 256 MiB aperture, so
//! VRAM cannot be read off PCI config space. And the server's own list is the
//! only source that says what the *binary* can use — this build links no CUDA,
//! it offloads over Vulkan, and a probe that reasoned from `lspci` plus the
//! absence of `libcudart` would have said "no GPU path, serve on CPU" and been
//! wrong by 20%.
//!
//! Nothing here invents a per-parameter size table. Small models are dominated
//! by their embedding matrix — the 0.6B model used as the test case carries a
//! 151k-token vocabulary — so GiB-per-billion-parameters is not a constant, and
//! quoting one would be a guess dressed as arithmetic.

use std::path::Path;

/// A compute device as `llama-server --list-devices` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDevice {
    /// The id to pass to `--device`, e.g. `Vulkan1`.
    pub id: String,
    /// Human-readable name, e.g. `NVIDIA GeForce MX250`.
    pub name: String,
    /// Total memory in MiB. `0` marks the CPU fallback (`BLAS: OpenBLAS`),
    /// which the list reports as a device and is not one.
    pub total_mib: u64,
    /// Memory free right now, in MiB. Whatever else is running has already
    /// taken its share, which is why the probe reads this and not the total.
    pub free_mib: u64,
}

impl ComputeDevice {
    pub fn is_offload_target(&self) -> bool {
        self.total_mib > 0
    }
}

/// Parse the output of `llama-server --list-devices`.
///
/// The line shape is `  <id>: <name> (<total> MiB, <free> MiB free)` and the
/// name may itself hold parentheses — `Intel(R) UHD Graphics (ICL GT1)` — so
/// the memory group is found from the trailing `(` backwards rather than by
/// splitting on the first `(`.
pub fn parse_llama_devices(text: &str) -> Vec<ComputeDevice> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((id, rest)) = line.split_once(": ") else {
            continue;
        };
        let Some(open) = rest.rfind(" (") else {
            continue;
        };
        let name = &rest[..open];
        let Some(memory) = rest[open + 2..].strip_suffix(")") else {
            continue;
        };
        let mut parts = memory.split(',').map(|p| p.trim());
        let (Some(total), Some(free)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Some(total_mib), Some(free_mib)) = (
            total
                .strip_suffix(" MiB")
                .and_then(|v| v.trim().parse::<u64>().ok()),
            free.strip_suffix(" MiB free")
                .and_then(|v| v.trim().parse::<u64>().ok()),
        ) else {
            continue;
        };
        if id.is_empty() || name.is_empty() {
            continue;
        }
        out.push(ComputeDevice {
            id: id.to_string(),
            name: name.to_string(),
            total_mib,
            free_mib,
        });
    }
    out
}

/// Run `llama-server --list-devices` and return its stdout, or `None` when the
/// server is not installed or does not know the flag. An older server without
/// the flag exits non-zero with usage text; that is the `None` case, not a
/// missing GPU to report.
pub fn server_devices(executable: &str) -> Option<String> {
    let out = std::process::Command::new(executable)
        .arg("--list-devices")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The server's build line, so a recommendation can be traced to the binary it
/// was read from. `llama-server --version` writes it to **stderr**, not stdout
/// (checked against b10809), so both streams are searched.
pub fn server_version(executable: &str) -> Option<String> {
    let out = std::process::Command::new(executable)
        .arg("--version")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    stderr
        .lines()
        .chain(stdout.lines())
        .find(|l| l.starts_with("version:"))
        .map(|l| l.trim_start_matches("version:").trim().to_string())
}

/// A graphics card as the kernel sees it, from `/sys/class/drm`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrmCard {
    pub card: String,
    /// PCI vendor id as sysfs prints it, e.g. `0x10de`.
    pub vendor: String,
    pub device: String,
    /// The bound driver's directory name, e.g. `nvidia` or `i915`.
    pub driver: String,
    /// Whether a display connector (`card2-eDP-1`) hangs off this card. The
    /// card driving the screen is the one a server would have to share it with
    /// the compositor.
    pub has_connector: bool,
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read the DRM cards under `root` (usually `/sys/class/drm`). Connector
/// entries are named `card<N>-<type>-<n>` and are used only to set
/// `has_connector`; entries with no readable vendor are skipped, which covers
/// both a connector directory and a card whose device node is not there.
pub fn drm_cards(root: &Path) -> Vec<DrmCard> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let mut cards = Vec::new();
    for name in names
        .iter()
        // `renderD128` and the `card2-eDP-1` connector entries are not cards.
        .filter(|n| n.starts_with("card") && !n.contains('-'))
    {
        let device_dir = root.join(name).join("device");
        let Some(vendor) = read_trimmed(&device_dir.join("vendor")) else {
            continue;
        };
        let driver = std::fs::read_link(device_dir.join("driver"))
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        cards.push(DrmCard {
            card: name.clone(),
            vendor,
            device: read_trimmed(&device_dir.join("device")).unwrap_or_default(),
            has_connector: names.iter().any(|n| {
                n.len() > name.len() + 1 && n.starts_with(name) && n.as_bytes()[name.len()] == b'-'
            }),
            driver,
        });
    }
    cards.sort_by(|a, b| a.card.cmp(&b.card));
    cards
}

/// Vendor names worth printing. An unknown id stays an id: naming a card that
/// is not there is worse than naming nothing.
pub fn vendor_name(vendor_id: &str) -> &'static str {
    match vendor_id {
        "0x10de" => "NVIDIA",
        "0x8086" => "Intel",
        "0x1002" => "AMD",
        "0x15ad" => "VMware",
        "0x1af4" => "virtio",
        _ => "unknown vendor",
    }
}

/// `MemAvailable` in KiB — what a new process could actually get, unlike
/// `MemTotal`, which counts what the page cache and every running program are
/// holding. Separate from the reader so the arithmetic can be tested without
/// depending on the machine it runs on.
pub fn parse_meminfo_available_kib(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

pub fn available_memory_kib() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo_available_kib(&meminfo)
}

/// Turn `/sys/devices/system/cpu/online` (`0-7`, or `0-3,8-11`) into a count.
pub fn parse_cpu_online(text: &str) -> Option<usize> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut total = 0usize;
    for part in text.split(',') {
        match part.split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().ok()?;
                let b: usize = b.trim().parse().ok()?;
                total += b.checked_sub(a)? + 1;
            }
            None => {
                part.trim().parse::<usize>().ok()?;
                total += 1;
            }
        }
    }
    Some(total)
}

pub fn cpu_core_count() -> Option<usize> {
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    parse_cpu_online(&online)
}

/// The parts of a GGUF file's own header that decide what serving it costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub architecture: String,
    pub name: String,
    /// `block_count` — the number of transformer layers, which is also the
    /// largest useful `--n-gpu-layers`.
    pub block_count: u32,
    pub kv_head_count: u32,
    /// Width of one attention head: `embedding_length / head_count`.
    pub head_dim: u32,
    pub file_bytes: u64,
}

impl ModelShape {
    /// Elements stored per context token in one of the two caches.
    fn elements_per_token_per_cache(&self) -> u64 {
        self.block_count as u64 * self.kv_head_count as u64 * self.head_dim as u64
    }

    /// The key figure: what the KV cache adds on top of the weights, per token,
    /// for the two cache types. K and V are each `elements_per_token_per_cache`
    /// wide.
    ///
    /// `q8_0` and `q4_0` are block formats — 128 elements plus one 16-bit
    /// scale — so they are 1.0156 and 0.5156 bytes per element rather than a
    /// round 1 and 0.5.
    pub fn kv_kib_per_token(&self, cache_k: &str, cache_v: &str) -> f64 {
        let elements = self.elements_per_token_per_cache() as f64;
        elements * (cache_bytes_per_element(cache_k) + cache_bytes_per_element(cache_v)) / 1024.0
    }
}

/// Bytes per element for the cache types a launch preset can emit. An
/// unrecognised type is priced as 16-bit, the widest of the common ones, so a
/// type this module has not heard of cannot make a plan look cheaper than it is.
pub fn cache_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "q8_0" => 1.0156,
        "q4_0" => 0.5156,
        "q4_1" => 0.5625,
        _ => 2.0,
    }
}

/// Read the geometry out of a GGUF file's own metadata, plus its size on disk.
/// `None` when the file cannot be opened, is not a GGUF, or does not carry
/// `block_count` within the window read — an unknown geometry, which the caller
/// must report as unknown rather than fill in with a guess.
pub fn read_gguf_shape(path: &str) -> Option<ModelShape> {
    const WINDOW: usize = 1 << 20;
    let mut buf = vec![0u8; WINDOW];
    let n = {
        use std::io::Read;
        let mut file = std::fs::File::open(path).ok()?;
        file.read(&mut buf).ok()?
    };
    buf.truncate(n);
    let mut shape = parse_gguf_shape(&buf)?;
    shape.file_bytes = std::fs::metadata(path).ok()?.len();
    Some(shape)
}

/// The metadata section follows the header immediately, so the first window
/// covers the geometry for the models in use here — but a real file's metadata
/// continues into a 151k-entry vocabulary that does not fit in it. Running out
/// of window is therefore the end of the readable part, not a broken file: the
/// keys collected so far are used, and `None` means the geometry was not among
/// them.
fn parse_gguf_shape(bytes: &[u8]) -> Option<ModelShape> {
    let mut r = Reader { b: bytes, i: 0 };
    if r.take(4)? != b"GGUF" {
        return None;
    }
    let version = r.u32()?;
    if version < 2 {
        return None;
    }
    let _tensors = r.u64()?;
    let kv_count = r.u64()?;
    let mut keys: Vec<(String, MetaValue)> = Vec::new();
    for _ in 0..kv_count {
        let (Some(key), Some(kind)) = (r.string(), r.u32()) else {
            break;
        };
        let interesting = key == "general.architecture"
            || key == "general.name"
            || GEOMETRY_SUFFIXES.iter().any(|s| key.ends_with(s));
        let Some(value) = r.value(kind, interesting) else {
            break;
        };
        if !value.is_skipped() {
            keys.push((key, value));
        }
    }
    let text = |name: &str| {
        keys.iter()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| v.as_text())
    };
    let arch = text("general.architecture")?;
    let number = |suffix: &str| -> Option<u32> {
        keys.iter()
            .find(|(k, _)| k == &format!("{arch}.{suffix}"))
            .and_then(|(_, v)| v.as_u32())
    };
    let block_count = number("block_count")?;
    let kv_head_count = number("attention.head_count_kv")?;
    let head_count = number("attention.head_count")?;
    let embedding_length = number("embedding_length")?;
    if head_count == 0 || embedding_length == 0 {
        return None;
    }
    Some(ModelShape {
        architecture: arch.clone(),
        name: text("general.name").unwrap_or_default(),
        block_count,
        kv_head_count,
        head_dim: embedding_length / head_count,
        file_bytes: 0,
    })
}

const GEOMETRY_SUFFIXES: &[&str] = &[
    "block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "embedding_length",
];

#[derive(Debug, Clone)]
enum MetaValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Text(String),
    Skipped,
}

impl MetaValue {
    fn is_skipped(&self) -> bool {
        matches!(self, MetaValue::Skipped)
    }

    fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::Unsigned(v) => u32::try_from(*v).ok(),
            MetaValue::Signed(v) => u32::try_from(*v).ok(),
            // A conversion tool is free to store a count as a float, and one
            // that did should not read as no geometry at all. Only a whole one.
            MetaValue::Float(v) if v.fract() == 0.0 && *v >= 0.0 => u32::try_from(*v as u64).ok(),
            _ => None,
        }
    }

    fn as_text(&self) -> Option<String> {
        match self {
            MetaValue::Text(s) => Some(s.clone()),
            _ => None,
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.i.checked_add(n)?;
        if end > self.b.len() {
            return None;
        }
        let slice = &self.b[self.i..end];
        self.i = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<String> {
        let n = self.u64()? as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// Read (or, when `wanted` is false, step over) one value. Arrays recurse,
    /// which is what lets the tokenizer's 151k-entry vocabulary be skipped
    /// without holding it.
    fn value(&mut self, kind: u32, wanted: bool) -> Option<MetaValue> {
        let value = match kind {
            0 => MetaValue::Unsigned(self.take(1)?[0] as u64),
            1 => MetaValue::Signed(self.take(1)?[0] as i8 as i64),
            2 => MetaValue::Unsigned(u16::from_le_bytes(self.take(2)?.try_into().ok()?) as u64),
            3 => MetaValue::Signed(i16::from_le_bytes(self.take(2)?.try_into().ok()?) as i64),
            4 => MetaValue::Unsigned(self.u32()? as u64),
            5 => MetaValue::Signed(self.u32()? as i32 as i64),
            6 => MetaValue::Float(f32::from_le_bytes(self.take(4)?.try_into().ok()?) as f64),
            7 => MetaValue::Unsigned(self.take(1)?[0] as u64),
            8 => MetaValue::Text(self.string()?),
            9 => {
                let elem = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count.min(1_000_000) {
                    self.value(elem, false)?;
                }
                MetaValue::Skipped
            }
            10 => MetaValue::Unsigned(self.u64()?),
            11 => MetaValue::Signed(self.u64()? as i64),
            12 => MetaValue::Float(f64::from_le_bytes(self.take(8)?.try_into().ok()?)),
            _ => return None,
        };
        Some(if wanted { value } else { MetaValue::Skipped })
    }
}

/// The largest context whose weights + cache stays inside `usable_mib`,
/// rounded down to a whole number of 1024-token blocks.
///
/// The growth is linear in tokens, so this is a division — but the ceiling it
/// is divided against is not exact: llama.cpp's own compute buffers and
/// per-slot state are outside the sum, and on the 2 GiB card used to check
/// this they amount to roughly a fifth of reported free memory. Hence the
/// reserve the caller applies before calling.
pub fn largest_context(
    weights_mib: f64,
    kv_kib_per_token: f64,
    usable_mib: f64,
    floor: u64,
) -> u64 {
    if kv_kib_per_token <= 0.0 {
        return floor;
    }
    let room = usable_mib - weights_mib;
    if room <= 0.0 {
        return floor;
    }
    let tokens = (room * 1024.0 / kv_kib_per_token) as u64;
    let rounded = (tokens / 1024) * 1024;
    rounded.max(floor)
}

/// Whether a device of `total_mib` is sharing system memory rather than
/// carrying its own. Measured on the laptop in the module header: the
/// integrated device reported 11822 of 16141 MiB (73%), the dedicated card
/// 2294 MiB (14%). A card that owns its memory cannot hold twice what the
/// machine has; a card sharing it reports a slice of exactly that.
///
/// The test needs a machine whose RAM is comparable to the integrated window,
/// which a server with 256 GiB and no display is not. There the ratio says
/// nothing and the answer falls back to the free memory the server itself
/// reported, which already discounts whatever the desktop is holding.
pub fn is_shared_memory_device(total_mib: u64, ram_total_mib: u64) -> bool {
    ram_total_mib > 0 && total_mib * 2 >= ram_total_mib
}

/// The usable fraction of a device's free memory: three quarters. Measured, not
/// rounded for looks — on the MX250 with 1156 MiB free, a weights+cache sum of
/// 826 MiB loaded and 938 MiB did not, and 1156 × 3/4 = 867 falls between them.
pub const RESERVE_NUMERATOR: u64 = 3;
pub const RESERVE_DENOMINATOR: u64 = 4;

/// The recommendation: what to start a server with, and why, in one line each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recommendation {
    /// Flags in the order `llama-server` should receive them.
    pub args: Vec<String>,
    /// The device the offload was aimed at, or `None` for the CPU.
    pub device: Option<String>,
    pub ctx_tokens: u64,
    pub notes: Vec<String>,
}

/// Decide from the facts. `window_asked` is what the budget layer works with —
/// the recommendation never exceeds it, because context nobody asked for is
/// memory burned anyway, and on the card used to check this a window larger
/// than the model needed was measurably slower than the CPU (`16384` offloaded
/// answered at 43.7 tokens a second against 58.4 on the CPU). Fitting less than
/// what was asked is the interesting case, and it is said out loud.
pub fn recommend(
    devices: &[ComputeDevice],
    shape: Option<&ModelShape>,
    ram_total_mib: u64,
    window_asked: u64,
) -> Recommendation {
    let offload: Vec<&ComputeDevice> = devices.iter().filter(|d| d.is_offload_target()).collect();
    let dedicated: Vec<&ComputeDevice> = offload
        .iter()
        .copied()
        .filter(|d| !is_shared_memory_device(d.total_mib, ram_total_mib))
        .collect();
    let weights_mib = shape.map_or(0.0, |s| s.file_bytes as f64 / 1024.0 / 1024.0);

    let Some(chosen) = dedicated.iter().copied().max_by_key(|d| d.free_mib) else {
        let note = if offload.is_empty() {
            "the server reported no compute devices, so there is nothing to offload to and it \
             will serve on the CPU"
                .to_string()
        } else {
            let names: Vec<&str> = offload.iter().map(|d| d.name.as_str()).collect();
            format!(
                "{} reports {} MiB against a machine with {} MiB of RAM: that is system memory \
                 it is sharing, not memory of its own, and serving from an integrated device was \
                 measured slower here than serving on the CPU",
                names.join(", "),
                offload.iter().map(|d| d.total_mib).max().unwrap_or(0),
                ram_total_mib
            )
        };
        return Recommendation {
            args: vec!["--n-gpu-layers".into(), "0".into()],
            device: None,
            ctx_tokens: window_asked,
            notes: vec![note],
        };
    };

    let usable_mib = chosen.free_mib * RESERVE_NUMERATOR / RESERVE_DENOMINATOR;
    let kv_kib = shape.map_or(0.0, |s| s.kv_kib_per_token(CACHE_K, CACHE_V));
    // The smallest window worth starting a server for: below it the answer is
    // not a shorter context, it is that this device is the wrong device.
    const MIN_WINDOW: u64 = 1024;
    let fits = if kv_kib > 0.0 {
        largest_context(weights_mib, kv_kib, usable_mib as f64, MIN_WINDOW)
    } else {
        window_asked
    };
    let ctx = fits.min(window_asked);

    let mut notes = vec![format!(
        "{} has {} MiB free of {} MiB; this plan spends {} MiB of it, holding a quarter back \
         for the server's own buffers, which are not part of weights plus cache",
        chosen.name, chosen.free_mib, chosen.total_mib, usable_mib
    )];
    match shape {
        Some(shape) => notes.push(format!(
            "{}: {:.0} MiB of weights, {:.1} KiB of cache per token at {CACHE_K} keys and \
             {CACHE_V} values ({} blocks x {} KV heads x width {}), so the {} token window costs \
             {:.0} MiB all in",
            if shape.name.is_empty() {
                shape.architecture.as_str()
            } else {
                shape.name.as_str()
            },
            weights_mib,
            kv_kib,
            shape.block_count,
            shape.kv_head_count,
            shape.head_dim,
            ctx,
            weights_mib + ctx as f64 * kv_kib / 1024.0,
        )),
        None => notes.push(
            "no model geometry was read, so the cache is not sized and the window below is what \
             the budget layer asked for rather than a fit"
                .to_string(),
        ),
    }
    if ctx < window_asked {
        notes.push(format!(
            "{window_asked} tokens was asked for and only {fits} fits what is free on this \
             device once the weights are counted — that is the cache on top of the model, not a \
             setting to raise"
        ));
    }
    notes.push(
        "n-gpu-layers all is what was measured faster here than the server's own choice, which \
         left the model on the CPU; naming a number also switches the server's memory fitter off"
            .to_string(),
    );

    if weights_mib > usable_mib as f64 {
        notes.push(format!(
            "the weights alone are {weights_mib:.0} MiB against {usable_mib} MiB usable, so this \
             device cannot hold the model at any window; a smaller quant or a remote server is \
             the way out, not a smaller context",
        ));
        return Recommendation {
            args: vec!["--n-gpu-layers".into(), "0".into()],
            device: None,
            ctx_tokens: window_asked,
            notes,
        };
    }

    Recommendation {
        args: vec![
            "--n-gpu-layers".into(),
            "all".into(),
            "--device".into(),
            chosen.id.clone(),
            "--cache-type-k".into(),
            CACHE_K.into(),
            "--cache-type-v".into(),
            CACHE_V.into(),
            "--flash-attn".into(),
            "on".into(),
            "--ctx-size".into(),
            ctx.to_string(),
        ],
        device: Some(chosen.id.clone()),
        ctx_tokens: ctx,
        notes,
    }
}

/// The cache types the launch presets already emit, reused so the probe prices
/// the same server it describes.
pub const CACHE_K: &str = "q8_0";
pub const CACHE_V: &str = "q4_0";

/// Every source this probe reads, in the words the item asked for: a
/// recommendation is only as trustworthy as the figures under it, and two of
/// these disagree with each other on purpose.
pub const SOURCES: &[(&str, &str)] = &[
    (
        "/proc/meminfo",
        "MemTotal for the profile band, MemAvailable for what a server could get right now",
    ),
    (
        "/sys/devices/system/cpu/online",
        "core count, as a range list (`0-7`, `0-3,8-11`)",
    ),
    (
        "/sys/class/drm/card*/device/{vendor,device,driver}",
        "which graphics cards the kernel sees, and which driver owns each",
    ),
    (
        "/sys/class/drm/card*-*",
        "connector directories: the card driving a screen is the one to leave alone",
    ),
    (
        "llama-server --list-devices",
        "the only source that says what this binary can offload to, and how much of it is \
         free right now",
    ),
    (
        "llama-server --version",
        "the build, so a recommendation can be traced to the server it was read from",
    ),
    (
        "GGUF metadata, first MiB of the file",
        "general.architecture, general.name, block_count, attention.head_count, \
         attention.head_count_kv, embedding_length — the geometry the cache arithmetic is \
         built from",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICES: &str = "Available devices:
  BLAS: OpenBLAS (0 MiB, 0 MiB free)
  Vulkan0: Intel(R) UHD Graphics (ICL GT1) (11822 MiB, 8987 MiB free)
  Vulkan1: NVIDIA GeForce MX250 (2294 MiB, 1156 MiB free)
";

    fn qwen3() -> ModelShape {
        ModelShape {
            architecture: "qwen3".into(),
            name: "Qwen3-0.6B".into(),
            block_count: 28,
            kv_head_count: 8,
            head_dim: 64,
            file_bytes: 396_705_472,
        }
    }

    #[test]
    fn reads_the_real_device_list_including_a_name_with_parentheses() {
        let devices = parse_llama_devices(DEVICES);
        assert_eq!(devices.len(), 3, "got {devices:?}");
        assert_eq!(devices[0].id, "BLAS");
        assert!(
            !devices[0].is_offload_target(),
            "the CPU row is not a device"
        );
        assert_eq!(devices[1].name, "Intel(R) UHD Graphics (ICL GT1)");
        assert_eq!(devices[2].id, "Vulkan1");
        assert_eq!(devices[2].total_mib, 2294);
        assert_eq!(devices[2].free_mib, 1156);
    }

    #[test]
    fn ignores_lines_that_are_not_devices() {
        assert!(parse_llama_devices("Available devices:\n").is_empty());
        assert!(parse_llama_devices("usage: llama-server [options]\n").is_empty());
        assert!(parse_llama_devices("  Nope: something (x MiB, y MiB free)\n").is_empty());
    }

    #[test]
    fn meminfo_available_is_read_apart_from_total() {
        let text = "MemTotal:       16141084 kB\nMemFree:         1234567 kB\nMemAvailable:    9536364 kB\n";
        assert_eq!(parse_meminfo_available_kib(text), Some(9536364));
        assert_eq!(parse_meminfo_available_kib("MemTotal:       1 kB"), None);
    }

    #[test]
    fn counts_cores_from_a_range_list() {
        assert_eq!(parse_cpu_online("0-7\n"), Some(8));
        assert_eq!(parse_cpu_online("0-3,8-11"), Some(8));
        assert_eq!(parse_cpu_online("0"), Some(1));
        assert_eq!(parse_cpu_online(""), None);
        assert_eq!(parse_cpu_online("nope"), None);
    }

    #[test]
    fn a_card_holding_half_the_machines_memory_is_sharing_it() {
        // The measured pair from the module header: 11822 MiB "total" on a
        // 16141 MiB machine is the iGPU's slice of system RAM, 2294 MiB is a
        // card's own.
        assert!(is_shared_memory_device(11822, 16141));
        assert!(!is_shared_memory_device(2294, 16141));
        assert!(
            !is_shared_memory_device(11822, 0),
            "no RAM figure, no claim"
        );
    }

    #[test]
    fn the_reserve_sits_between_what_loaded_and_what_did_not() {
        // 826 MiB of weights+cache loaded, 938 MiB did not, against 1156 MiB
        // reported free.
        let usable = 1156 * RESERVE_NUMERATOR / RESERVE_DENOMINATOR;
        assert!((827..938).contains(&usable), "usable was {usable}");
    }

    #[test]
    fn cache_bytes_per_element_uses_the_block_formats_own_sizes() {
        assert!((cache_bytes_per_element("q8_0") - 1.0156).abs() < 1e-9);
        assert!((cache_bytes_per_element("q4_0") - 0.5156).abs() < 1e-9);
        assert_eq!(cache_bytes_per_element("f16"), 2.0);
        // Something unrecognised must not make a plan look cheap.
        assert_eq!(cache_bytes_per_element("iq2_xxs"), 2.0);
    }

    #[test]
    fn the_geometry_of_the_model_on_this_machine_computes_the_measured_cache() {
        let shape = qwen3();
        // 28 blocks x 8 KV heads x 64 wide, two caches: 56 KiB/token at 16-bit,
        // which is 448 MiB across 8192 tokens — the window that loaded next to
        // 378 MiB of weights — and 560 MiB across 10240, which did not.
        let plain = shape.kv_kib_per_token("f16", "f16");
        assert!(
            (plain - 56.0).abs() < 0.01,
            "plain cache was {plain} KiB/token"
        );
        let quantized = shape.kv_kib_per_token(CACHE_K, CACHE_V);
        assert!(
            (quantized - 21.4).abs() < 0.5,
            "quantized cache was {quantized} KiB/token"
        );
        assert_eq!(shape.file_bytes / 1024 / 1024, 378);
    }

    #[test]
    fn the_largest_window_is_rounded_down_to_a_thousand_block() {
        // 378 MiB weights, 21.4 KiB/token, 867 MiB usable: 489 MiB of room is
        // about 23 370 tokens, which must round down to 22 528.
        let ctx = largest_context(378.0, 21.4, 867.0, 4096);
        assert_eq!(ctx, 22_528, "got {ctx}");
        assert_eq!(ctx % 1024, 0);
    }

    #[test]
    fn a_model_with_no_room_left_gets_the_floors_window() {
        assert_eq!(largest_context(5000.0, 21.4, 867.0, 4096), 4096);
        assert_eq!(largest_context(100.0, 0.0, 867.0, 4096), 4096);
    }

    #[test]
    fn gguf_shape_is_parsed_from_a_header_written_here() {
        let shape = parse_gguf_shape(&sample_gguf()).expect("parses");
        assert_eq!(shape.architecture, "qwen3");
        assert_eq!(shape.name, "Qwen3-0.6B");
        assert_eq!(shape.block_count, 28);
        assert_eq!(shape.kv_head_count, 8);
        assert_eq!(shape.head_dim, 1024 / 16);
    }

    #[test]
    fn a_file_that_stops_before_the_geometry_is_no_geometry() {
        assert!(parse_gguf_shape(b"GPT2 something").is_none());
        // The two names, and nothing else.
        let mut body = Vec::new();
        str_kv(&mut body, "general.architecture", "qwen3");
        str_kv(&mut body, "general.name", "Qwen3-0.6B");
        let mut bytes = header(2);
        bytes.extend_from_slice(&body);
        assert!(
            parse_gguf_shape(&bytes).is_none(),
            "a file that stops before block_count must not invent one"
        );
    }

    #[test]
    fn metadata_running_past_the_window_keeps_the_geometry_read_before_it() {
        // A real file's metadata continues into a 151k-entry vocabulary no
        // bounded read covers, so stopping mid-value is the normal case. This
        // sample is cut inside its token list and must still report the
        // geometry that came first.
        let whole = sample_gguf();
        let cut = whole.len() - 4;
        let shape = parse_gguf_shape(&whole[..cut]).expect("read before the cut");
        assert_eq!(shape.block_count, 28);
        assert_eq!(shape.kv_head_count, 8);
    }

    #[test]
    fn drm_cards_are_the_card_directories_and_know_which_one_drives_a_screen() {
        let root =
            std::env::temp_dir().join(format!("xencode-hwprobe-drm-{}-{}", std::process::id(), {
                static NEXT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            }));
        let _ = std::fs::remove_dir_all(&root);
        // The shape /sys/class/drm has on the laptop in the module header: two
        // cards, one of them with a connector, plus render nodes that are not
        // cards. The driver symlink is left out — its target does not exist on
        // a synthetic tree, which is the case that must not panic.
        for (card, vendor, device) in [
            ("card1", "0x10de\n", "0x1d13\n"),
            ("card2", "0x8086\n", "0x8a56\n"),
        ] {
            let device_dir = root.join(card).join("device");
            std::fs::create_dir_all(&device_dir).unwrap();
            std::fs::write(device_dir.join("vendor"), vendor).unwrap();
            std::fs::write(device_dir.join("device"), device).unwrap();
        }
        std::fs::create_dir_all(root.join("card2-eDP-1")).unwrap();
        std::fs::create_dir_all(root.join("renderD128")).unwrap();

        let cards = drm_cards(&root);
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(cards.len(), 2, "got {cards:?}");
        assert_eq!(cards[0].card, "card1");
        assert_eq!(cards[0].vendor, "0x10de");
        assert_eq!(cards[0].device, "0x1d13");
        assert!(cards[0].driver.is_empty(), "no driver, no claim");
        assert!(!cards[0].has_connector, "the compute card drives nothing");
        assert!(cards[1].has_connector, "card2 owns the eDP connector");
        assert_eq!(vendor_name(&cards[0].vendor), "NVIDIA");
        assert_eq!(vendor_name(&cards[1].vendor), "Intel");
        assert_eq!(vendor_name("0xffff"), "unknown vendor");
    }

    fn header(kv_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&310u64.to_le_bytes());
        b.extend_from_slice(&kv_count.to_le_bytes());
        b
    }

    #[test]
    fn a_model_without_a_block_count_is_no_geometry_rather_than_a_guess() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        put_str(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut bytes, "qwen3");
        assert!(parse_gguf_shape(&bytes).is_none());
    }

    #[test]
    fn the_shared_integrated_device_is_not_chosen_when_a_dedicated_one_is_present() {
        let devices = parse_llama_devices(DEVICES);
        let rec = recommend(&devices, Some(&qwen3()), 16141, 8192);
        assert_eq!(rec.device.as_deref(), Some("Vulkan1"));
        let at = rec.args.iter().position(|a| a == "--device").unwrap();
        assert_eq!(rec.args[at + 1], "Vulkan1", "the flags must name the card");
        assert_eq!(rec.args[..2], ["--n-gpu-layers", "all"]);
        assert_eq!(rec.args[at - 2..at], ["--n-gpu-layers", "all"]);
        // The card holds this window with room to spare, so the window asked
        // for is the window recommended — and 8192 at `all` is the one that was
        // measured serving this model on this box.
        assert_eq!(rec.ctx_tokens, 8192, "notes were {:?}", rec.notes);
        assert!(
            !rec.notes.iter().any(|n| n.contains("only")),
            "nothing was short here: {:?}",
            rec.notes
        );
    }

    #[test]
    fn a_window_that_does_not_fit_once_the_cache_is_counted_is_named_as_shorter() {
        let devices = parse_llama_devices(DEVICES);
        // The same card, the same model, plain 16-bit caches and a 24 GiB ask:
        // the model plus its cache cannot be that wide.
        let wide = ModelShape {
            block_count: 64,
            kv_head_count: 8,
            head_dim: 128,
            ..qwen3()
        };
        let rec = recommend(&devices, Some(&wide), 16141, 32_768);
        assert!(rec.ctx_tokens < 32_768, "window was {}", rec.ctx_tokens);
        assert!(
            rec.notes
                .iter()
                .any(|n| n.contains("32768 tokens was asked for") && n.contains("cache on top")),
            "notes were {:?}",
            rec.notes
        );
    }

    #[test]
    fn with_only_a_shared_device_the_plan_says_cpu_and_says_why() {
        let devices = parse_llama_devices(
            "Available devices:
  BLAS: OpenBLAS (0 MiB, 0 MiB free)
  Vulkan0: Intel(R) UHD Graphics (ICL GT1) (11822 MiB, 8987 MiB free)
",
        );
        let rec = recommend(&devices, None, 16141, 4096);
        assert_eq!(rec.device, None);
        assert_eq!(rec.args, ["--n-gpu-layers", "0"]);
        assert!(
            rec.notes
                .iter()
                .any(|n| n.contains("slower") && n.contains("11822") && n.contains("16141")),
            "notes were {:?}",
            rec.notes
        );
    }

    #[test]
    fn with_no_device_at_all_the_note_says_so_instead_of_naming_numbers() {
        let devices =
            parse_llama_devices("Available devices:\n  BLAS: OpenBLAS (0 MiB, 0 MiB free)\n");
        let rec = recommend(&devices, None, 16141, 4096);
        assert_eq!(rec.args, ["--n-gpu-layers", "0"]);
        assert!(
            rec.notes[0].contains("no compute devices"),
            "{:?}",
            rec.notes
        );
    }

    #[test]
    fn weights_that_do_not_fit_are_refused_rather_than_run_into_a_small_window() {
        let devices = parse_llama_devices(DEVICES);
        let mut shape = qwen3();
        shape.file_bytes = 4096 * 1024 * 1024;
        let rec = recommend(&devices, Some(&shape), 16141, 4096);
        assert_eq!(rec.args, ["--n-gpu-layers", "0"]);
        assert!(
            rec.notes
                .iter()
                .any(|n| n.contains("smaller quant or a remote")),
            "notes were {:?}",
            rec.notes
        );
        assert!(
            rec.notes.iter().any(|n| n.contains("at any window")),
            "must blame the weights, not the window: {:?}",
            rec.notes
        );
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn str_kv(b: &mut Vec<u8>, key: &str, value: &str) {
        put_str(b, key);
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(b, value);
    }

    fn u32_kv(b: &mut Vec<u8>, key: &str, value: u32) {
        put_str(b, key);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
    }

    /// A metadata section in the shape the real file has: the two names, the
    /// four geometry keys, and an array of strings to be stepped over.
    fn sample_gguf() -> Vec<u8> {
        let mut body: Vec<u8> = Vec::new();
        str_kv(&mut body, "general.architecture", "qwen3");
        str_kv(&mut body, "general.name", "Qwen3-0.6B");
        u32_kv(&mut body, "qwen3.block_count", 28);
        u32_kv(&mut body, "qwen3.embedding_length", 1024);
        u32_kv(&mut body, "qwen3.attention.head_count", 16);
        u32_kv(&mut body, "qwen3.attention.head_count_kv", 8);
        put_str(&mut body, "tokenizer.ggml.tokens");
        body.extend_from_slice(&9u32.to_le_bytes()); // array
        body.extend_from_slice(&8u32.to_le_bytes()); // of strings
        body.extend_from_slice(&3u64.to_le_bytes()); // three of them
        for token in ["<|end|>", "a", "b"] {
            put_str(&mut body, token);
        }

        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&310u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&7u64.to_le_bytes()); // metadata count
        b.extend_from_slice(&body);
        b
    }
}
