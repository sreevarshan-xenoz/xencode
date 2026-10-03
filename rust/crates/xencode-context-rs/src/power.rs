//! What a generation cost in electricity, measured from the hardware.
//!
//! A local model is not free because no invoice arrives: the machine drew power
//! while it answered, and someone paid for that at the meter. This module reads
//! what the machine drew rather than modelling what it should have drawn, so the
//! number is an observation with all an observation's limits — which is why every
//! figure it produces is labelled an estimate before it is shown to anyone.
//!
//! Two sources, because a laptop and a tower disagree about where the work
//! happens:
//!
//! * **RAPL** (`/sys/class/powercap/intel-rapl:*`) counts the energy the CPU
//!   package has used since power-on. It is a *cumulative* counter, so a window's
//!   usage is the difference between two readings. It is also **package-wide**: a
//!   compile running beside xencode is in the same total, and nothing here can
//!   separate them.
//! * **`nvidia-smi`** reports the instantaneous draw of a discrete GPU. This is a
//!   poll, not an attribution, and many laptops — including the one this was
//!   written on — answer `[N/A]` for the connector that carries the chip. A GPU
//!   that will not say is reported as unknown, never as zero.
//!
//! An AMD or Arm machine without a RAPL package domain yields `None` from every
//! reader here. That is the honest result: the energy is not measurable from
//! userspace, and a figure invented for the gap would be a wrong number with the
//! confidence of a right one.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Where the kernel publishes the energy counters.
pub const POWERCAP_ROOT: &str = "/sys/class/powercap";

/// The command that asks an NVIDIA driver what the card is drawing right now.
pub const NVIDIA_QUERY_ARGS: &[&str] = &[
    "--query-gpu=power.draw",
    "--format=csv,noheader,nounits",
    "--id=0",
];

/// Parse one RAPL counter file. The kernel writes plain decimal microjoules with
/// a trailing newline; anything that is not a number — an empty file during a
/// driver reload, for instance — means no reading rather than a reading of zero.
pub fn parse_energy_uj(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse().ok()
}

/// Parse the output of [`NVIDIA_QUERY_ARGS`]: one decimal wattage per line, or
/// `[N/A]` where the driver cannot answer. The first number wins, because that is
/// the card `--id=0` was asked about; a machine with no GPU at all fails to run
/// the command and never reaches here.
pub fn parse_nvidia_power_w(text: &str) -> Option<f32> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && *line != "[N/A]")?
        .parse()
        .ok()
}

/// The cumulative limit of a RAPL domain, past which it wraps back to zero. A
/// laptop package domain is usually in the hundreds of kilojoules, so a wrap
/// inside a single generation is not a hypothetical: it happens on a long run on
/// a domain that counts a narrow range.
fn read_max_energy_uj(domain: &Path) -> Option<u64> {
    parse_energy_uj(&std::fs::read_to_string(domain.join("max_energy_range_uj")).ok()?)
}

/// The `energy_uj` reading of one domain.
fn read_domain_energy_uj(domain: &Path) -> Option<u64> {
    parse_energy_uj(&std::fs::read_to_string(domain.join("energy_uj")).ok()?)
}

/// Pick the package domain out of a powercap directory.
///
/// Top-level domains are named `intel-rapl:N` and carry a `name` file saying
/// `package-N` (or `platform-N`); the children, `intel-rapl:N:M`, are the parts
/// of that package — `core`, `uncore`, `dram` — and adding them to the package
/// counts the same joules twice. So only a directory whose own name has no second
/// colon and whose `name` begins with `package-` qualifies. A `platform-` domain
/// is deliberately not chosen: on the machines that publish both, it includes the
/// panel and the rest of the board, which is more than the work cost.
pub fn package_domain(root: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("intel-rapl:") && n.matches(':').count() == 1)
                .unwrap_or(false)
        })
        .filter(|path| {
            std::fs::read_to_string(path.join("name"))
                .map(|name| name.trim().starts_with("package-"))
                .unwrap_or(false)
        })
        .collect();
    // Lowest index first: on a multi-socket machine `package-0` is the one the
    // process scheduler starts from, and a second package is someone else's
    // electricity bill for this turn either way.
    found.sort();
    found.into_iter().next()
}

/// The energy the CPU package has used since power-on, with the limit it wraps at.
/// `None` on a machine with no readable package domain.
pub fn package_energy_uj(root: &str) -> Option<(u64, Option<u64>)> {
    let domain = package_domain(Path::new(root))?;
    let energy = read_domain_energy_uj(&domain)?;
    Some((energy, read_max_energy_uj(&domain)))
}

/// Ask the NVIDIA driver what the card is drawing. `None` when there is no card,
/// no driver, or a driver that answers `[N/A]` — which is what a laptop whose
/// discrete GPU is powered down at the connector does.
pub fn gpu_power_w() -> Option<f32> {
    let out = std::process::Command::new("nvidia-smi")
        .args(NVIDIA_QUERY_ARGS)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_power_w(&String::from_utf8_lossy(&out.stdout))
}

/// What a generation window drew, once it has closed.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PowerUse {
    /// How long the window was open. The cost in joules does not need this — the
    /// counter difference is already an integral — but a person reading the line
    /// wants the seconds beside the watt-hours.
    pub elapsed: Duration,
    /// Energy the CPU package used during the window, in microjoules, after any
    /// counter wrap has been taken back out.
    pub cpu_energy_uj: Option<u64>,
    /// Mean GPU draw over the window, from the two polls. `None` when the card
    /// would not say, which is not the same as a card that drew nothing.
    pub gpu_power_w: Option<f32>,
}

impl PowerUse {
    /// Watt-hours the package used, or `None` when it was never readable.
    pub fn cpu_watt_hours(&self) -> Option<f64> {
        self.cpu_energy_uj.map(|uj| uj as f64 / 3.6e9)
    }

    /// Watt-hours the GPU used over the window, from a mean draw held for the
    /// window's length. An average of two polls on a card that idles between
    /// layers, so the result is labelled an estimate everywhere it is shown.
    pub fn gpu_watt_hours(&self) -> Option<f64> {
        self.gpu_power_w
            .map(|w| f64::from(w) * self.elapsed.as_secs_f64() / 3600.0)
    }

    /// The whole window's draw, as far as this machine will report it. `None`
    /// means neither source answered, which is reported as unknown rather than
    /// as a generation that cost nothing.
    pub fn total_watt_hours(&self) -> Option<f64> {
        match (self.cpu_watt_hours(), self.gpu_watt_hours()) {
            (Some(cpu), Some(gpu)) => Some(cpu + gpu),
            (Some(cpu), None) => Some(cpu),
            (None, Some(gpu)) => Some(gpu),
            (None, None) => None,
        }
    }

    /// What [`Self::total_watt_hours`] would have cost at a given retail tariff,
    /// in micro-dollars: a kilowatt-hour at `cents_per_kwh` cents is
    /// `cents_per_kwh * 10_000` micro-dollars, so a watt-hour is
    /// `cents_per_kwh * 10` of them.
    ///
    /// The tariff is a number a person sets. When it is unset this is `None`, and
    /// the line shows the watt-hours without inventing a price for them.
    pub fn cost_micros(&self, cents_per_kwh: Option<f64>) -> Option<u64> {
        let wh = self.total_watt_hours()?;
        let cents = cents_per_kwh?;
        if !cents.is_finite() || cents < 0.0 {
            return None;
        }
        Some((wh * cents * 10.0).round() as u64)
    }

    /// Mean power over the window, in watts: everything this could see, divided
    /// by how long it was looking. `None` when nothing was measured — a window on
    /// a machine with no counters has no number, not a zero.
    pub fn mean_power_w(&self) -> Option<f32> {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return None;
        }
        self.total_watt_hours()
            .map(|wh| (wh * 3600.0 / seconds) as f32)
    }

    /// Stamp what this window measured onto the metrics row that describes the
    /// turn it covered, and say whether the energy figure went with it.
    ///
    /// The wall-clock always goes: how long a turn ran is true of the turn, not
    /// of the machine that happened to host the model, and a daily time budget
    /// has to be able to count a day spent against a cloud provider. The energy,
    /// the mean watts and their price go only onto a turn whose prompt stayed on
    /// this machine. A cloud turn's electricity went into a provider's meter, not
    /// this one, so attaching this window's reading to it would bill the same
    /// seconds twice — once for the work, once for a CPU that was only waiting on
    /// a socket.
    pub fn apply_to(
        &self,
        cents_per_kwh: Option<f64>,
        row: &mut crate::metrics::RequestMetrics,
    ) -> bool {
        row.elapsed_ms = Some(self.elapsed.as_millis() as u64);
        if !matches!(row.source, Some(crate::metrics::MetricSource::Local)) {
            return false;
        }
        row.energy_uj = self.cpu_energy_uj;
        row.power_w = self.mean_power_w();
        row.est_cost_micros = self.cost_micros(cents_per_kwh);
        true
    }
}

/// A power window: reads the counters when it opens so the closing reading has
/// something to be a difference from.
#[derive(Debug, Clone)]
pub struct PowerWindow {
    opened: Instant,
    /// The package domain this window reads, kept so the closing read uses the
    /// same counter — a driver reload could move it.
    domain: Option<PathBuf>,
    cpu_energy_uj: Option<u64>,
    max_energy_uj: Option<u64>,
    gpu_power_w: Option<f32>,
}

impl PowerWindow {
    /// Open a window on the real machine.
    pub fn begin() -> Self {
        Self::begin_at(POWERCAP_ROOT)
    }

    /// Open a window against a chosen powercap directory — the same code path, for
    /// a test that cannot wait for a laptop to burn two hundred kilojoules.
    pub fn begin_at(root: impl Into<String>) -> Self {
        let root = root.into();
        let domain = package_domain(Path::new(&root));
        let cpu_energy_uj = domain
            .as_ref()
            .and_then(|domain| read_domain_energy_uj(domain));
        let max_energy_uj = domain
            .as_ref()
            .and_then(|domain| read_max_energy_uj(domain));
        Self {
            opened: Instant::now(),
            domain,
            cpu_energy_uj,
            max_energy_uj,
            gpu_power_w: gpu_power_w(),
        }
    }

    /// Whether anything at all could be read when the window opened. A caller
    /// that is about to ask the same questions again for every round can skip
    /// them on a machine with no counters.
    pub fn is_measuring(&self) -> bool {
        self.cpu_energy_uj.is_some() || self.gpu_power_w.is_some()
    }

    /// Close the window and take what it used.
    pub fn finish(&self) -> PowerUse {
        let elapsed = self.opened.elapsed();
        let end = self
            .domain
            .as_ref()
            .and_then(|domain| read_domain_energy_uj(domain));
        let cpu_energy_uj = match (self.cpu_energy_uj, end) {
            (Some(start), Some(end)) => delta_uj(end, start, self.max_energy_uj),
            _ => None,
        };
        // Two polls at the ends of the window, averaged: a single generation is
        // seconds, and a card that spikes mid-layer and idles between them is
        // exactly the thing this cannot see.
        let gpu_power_w = match (self.gpu_power_w, gpu_power_w()) {
            (Some(start), Some(end)) => Some((start + end) / 2.0),
            (Some(start), None) => Some(start),
            (None, Some(end)) => Some(end),
            (None, None) => None,
        };
        PowerUse {
            elapsed,
            cpu_energy_uj,
            gpu_power_w,
        }
    }
}

/// The usage between two readings of a wrapping counter.
///
/// A wrap shows up as the end reading sitting *below* the start one. Adding the
/// domain's range back is what the kernel's own `energy_uj` documentation asks
/// for; without the range there is no way to know how far the counter went, and
/// the honest answer is that this window was not measured.
pub fn delta_uj(end: u64, start: u64, max_uj: Option<u64>) -> Option<u64> {
    if end >= start {
        return Some(end - start);
    }
    let max = max_uj?;
    max.checked_add(1).and_then(|range| {
        range
            .checked_sub(start)
            .and_then(|to_wrap| to_wrap.checked_add(end))
    })
}

/// Render the seconds a window stayed open, the way the plan's example does it:
/// a turn that ran is shown in seconds, a run that went past a minute in minutes
/// and seconds. Under ten seconds one decimal is kept, and under one second two,
/// so a window that closed almost at once reads as very short rather than as
/// having not happened.
pub fn elapsed_label(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 0.01 {
        "under 0.01 s".to_string()
    } else if secs < 1.0 {
        format!("{secs:.2} s")
    } else if secs < 10.0 {
        format!("{secs:.1} s")
    } else if secs < 60.0 {
        format!("{secs:.0} s")
    } else {
        let whole_minutes = elapsed.as_secs() / 60;
        let rest = elapsed.as_secs() % 60;
        format!("{whole_minutes} min {rest:02} s")
    }
}

/// Render the watt-hours with the `≈` the number has earned, or say that this
/// machine would not tell.
///
/// The unit follows the size of the figure on purpose. One generation is a few
/// hundredths of a watt-hour at most, and `≈ 0.00 Wh` is the only rendering worse
/// than no number at all: it reads as a turn that cost nothing. So below a
/// hundredth of a watt-hour the figure goes into milliwatt-hours, where it keeps
/// two digits that mean something.
pub fn watt_hours_label(wh: Option<f64>) -> String {
    match wh {
        Some(wh) if wh >= 1.0 => format!("≈ {wh:.1} Wh"),
        Some(wh) if wh >= 0.01 => format!("≈ {wh:.2} Wh"),
        Some(wh) if wh > 0.0 => {
            let milliwatt_hours = wh * 1000.0;
            if milliwatt_hours >= 0.01 {
                format!("≈ {milliwatt_hours:.2} mWh")
            } else {
                // Below a hundredth of a milliwatt-hour there is no unit left to
                // switch to; say it is tiny rather than saying it is nothing.
                "≈ under 0.01 mWh".to_string()
            }
        }
        Some(_) => "≈ 0 Wh".to_string(),
        None => "energy unknown".to_string(),
    }
}

/// Render the cost, keeping a local turn that has no tariff attached to it
/// visibly *unpriced* rather than free: the electricity was bought, the price
/// just was not written down. The figure itself goes through
/// [`crate::pricing::format_usd`], the same money formatter the token-price
/// report uses, so a fraction of a cent never prints as `$0.00`.
///
/// A turn short enough to cost less than one micro-dollar is the case that
/// formatter cannot say — it answers `$0` — so it is named here instead. A tariff
/// set to zero is the one genuine zero, and it says so with its reason attached,
/// because "zero-dollar" and "free" are different claims.
pub fn cost_label(cost_micros: Option<u64>, cents_per_kwh: Option<f64>) -> String {
    match (cost_micros, cents_per_kwh) {
        (Some(0), Some(0.0)) => "≈ $0 at a 0 ¢/kWh tariff".to_string(),
        (Some(0), Some(_)) => "≈ under $0.000001".to_string(),
        (Some(micros), _) => format!("≈ {}", crate::pricing::format_usd(micros)),
        (None, Some(_)) => "not priced".to_string(),
        (None, None) => "no $/kWh set".to_string(),
    }
}

/// The one-line form: `≈ 3.2 Wh · ≈ $0.0014 · 41 s · estimated, this machine
/// only`, plus whatever the sources could not see.
///
/// With nothing measured the price is left off the line entirely. A machine that
/// has a tariff set and no counter must not be told `no $/kWh set` — that would
/// send the reader to a setting that is already correct, when the real answer is
/// that this hardware publishes no energy at all. Nor does that line call itself
/// an estimate: there is no figure to qualify, only the seconds.
pub fn power_line(use_: &PowerUse, cents_per_kwh: Option<f64>) -> String {
    let Some(wh) = use_.total_watt_hours() else {
        return format!(
            "{} · {} — this machine reports no energy counter to read",
            watt_hours_label(None),
            elapsed_label(use_.elapsed)
        );
    };
    let mut line = format!(
        "{} · {} · {}",
        watt_hours_label(Some(wh)),
        cost_label(use_.cost_micros(cents_per_kwh), cents_per_kwh),
        elapsed_label(use_.elapsed)
    );
    if use_.gpu_power_w.is_none() {
        line.push_str(" — CPU package only; no graphics power was reported");
    }
    line.push_str(" · estimated, this machine only");
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A powercap directory that behaves like the kernel's: one package domain
    /// with two sub-domains beside it, exactly as `/sys/class/powercap` looks.
    fn fake_powercap(energy: u64, max: u64) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("intel-rapl:0");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("name"), "package-0\n").unwrap();
        std::fs::write(package.join("energy_uj"), format!("{energy}\n")).unwrap();
        std::fs::write(package.join("max_energy_range_uj"), format!("{max}\n")).unwrap();
        for (sub, sub_name) in [("intel-rapl:0:0", "core"), ("intel-rapl:0:1", "uncore")] {
            let child = dir.path().join(sub);
            std::fs::create_dir_all(&child).unwrap();
            std::fs::write(child.join("name"), format!("{sub_name}\n")).unwrap();
            std::fs::write(child.join("energy_uj"), "999999999999\n").unwrap();
            std::fs::write(child.join("max_energy_range_uj"), format!("{max}\n")).unwrap();
        }
        dir
    }

    #[test]
    fn a_counter_reads_as_the_number_the_kernel_wrote() {
        assert_eq!(parse_energy_uj("143837453369\n"), Some(143_837_453_369));
        assert_eq!(parse_energy_uj("  42 "), Some(42));
    }

    #[test]
    fn an_unreadable_counter_is_no_reading_rather_than_zero() {
        assert_eq!(parse_energy_uj(""), None);
        assert_eq!(parse_energy_uj("   \n"), None);
        assert_eq!(parse_energy_uj("not a number"), None);
    }

    #[test]
    fn a_wattage_is_read_and_a_refused_answer_is_not_one() {
        assert_eq!(parse_nvidia_power_w("27.93\n"), Some(27.93));
        assert_eq!(parse_nvidia_power_w("[N/A]\n"), None);
        assert_eq!(parse_nvidia_power_w(""), None);
        // A header-free, unit-free query answers one line; a blank first line is
        // skipped rather than parsed.
        assert_eq!(parse_nvidia_power_w("\n3.5\n"), Some(3.5));
    }

    #[test]
    fn the_package_domain_is_chosen_and_its_parts_are_left_alone() {
        let root = fake_powercap(1_000_000, 262_143_328_850);
        let chosen = package_domain(root.path()).unwrap();
        assert_eq!(
            chosen.file_name().unwrap().to_str().unwrap(),
            "intel-rapl:0"
        );
        // The sub-domains hold a different number on purpose: reading one of them
        // instead would report a fraction of the package as if it were all of it.
        assert_eq!(
            package_energy_uj(&root.path().display().to_string()).unwrap(),
            (1_000_000, Some(262_143_328_850))
        );
    }

    #[test]
    fn a_directory_with_no_package_domain_answers_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let core_only = dir.path().join("intel-rapl:0");
        std::fs::create_dir_all(&core_only).unwrap();
        std::fs::write(core_only.join("name"), "core\n").unwrap();
        std::fs::write(core_only.join("energy_uj"), "5\n").unwrap();
        assert_eq!(package_domain(dir.path()), None);
        assert_eq!(package_energy_uj(&dir.path().display().to_string()), None);
        // A path that is not there at all is the same answer, not a panic: this
        // is `/sys/class/powercap` on a machine whose vendor shipped no counters.
        assert_eq!(package_domain(Path::new("/nonexistent-powercap")), None);
    }

    #[test]
    fn a_window_measures_the_energy_between_its_two_readings() {
        let root = fake_powercap(1_000_000, 262_143_328_850);
        let root_path = root.path().display().to_string();
        let window = PowerWindow::begin_at(&root_path);
        assert!(window.is_measuring());
        let package = package_domain(Path::new(&root_path)).unwrap();
        std::fs::write(package.join("energy_uj"), "3700000\n").unwrap();
        let use_ = window.finish();
        assert_eq!(use_.cpu_energy_uj, Some(2_700_000));
        // 2.7 J is 0.00075 Wh: this is the arithmetic the whole module is for.
        assert!((use_.cpu_watt_hours().unwrap() - 7.5e-4).abs() < 1e-9);
    }

    #[test]
    fn a_window_on_a_machine_without_counters_reads_no_cpu_energy() {
        let dir = tempfile::tempdir().unwrap();
        let window = PowerWindow::begin_at(dir.path().display().to_string());
        let use_ = window.finish();
        // The GPU may or may not answer depending on what this box has, so only
        // the CPU half is claimed here; the arithmetic on both halves is covered
        // by `a_gpu_that_will_not_answer_is_left_out_of_the_total_rather_than_zeroed`.
        assert_eq!(use_.cpu_energy_uj, None);
        assert_eq!(use_.cpu_watt_hours(), None);
    }

    #[test]
    fn a_counter_that_wrapped_is_counted_around_and_not_backwards() {
        let max = 1_000_000_u64;
        assert_eq!(delta_uj(600, 400, Some(max)), Some(200));
        assert_eq!(delta_uj(400, 400, Some(max)), Some(0));
        // End below start: the counter passed `max` and came back round. The lap
        // is `max + 1` values wide, since the counter reaches `max` before it
        // returns to zero.
        assert_eq!(delta_uj(400, 900_000, Some(max)), Some(100_401));
        assert_eq!(delta_uj(0, max, Some(max)), Some(1));
        // With no known range there is no way to size the lap, so the window is
        // unmeasured rather than guessed at.
        assert_eq!(delta_uj(400, 900_000, None), None);
    }

    #[test]
    fn a_real_package_window_on_this_machine_reports_a_number_or_a_plain_absence() {
        // The two paths this module has to handle both come from the machine it
        // runs on, so the assertion has to hold either way — the difference is
        // that one of them is a figure a person is shown.
        let window = PowerWindow::begin();
        std::thread::sleep(Duration::from_millis(250));
        let use_ = window.finish();
        match use_.cpu_energy_uj {
            Some(uj) => {
                assert!(uj > 0, "a quarter second of a live CPU measured {uj} µJ");
                assert!(
                    use_.cpu_watt_hours().unwrap() > 0.0,
                    "a positive counter difference must be a positive watt-hour figure"
                );
            }
            None => assert_eq!(
                package_domain(Path::new(POWERCAP_ROOT)),
                None,
                "no reading came with a package domain present"
            ),
        }
    }

    #[test]
    fn a_tariff_price_multiplies_watt_hours_into_micro_dollars() {
        let use_ = PowerUse {
            elapsed: Duration::from_secs(41),
            cpu_energy_uj: Some(11_520_000_000), // 11.52 kJ is 3.2 Wh
            gpu_power_w: None,
        };
        assert_eq!(use_.total_watt_hours(), Some(3.2));
        // 3.2 Wh is 0.0032 kWh, and at 12 cents a kWh that is 0.0384 cents —
        // 384 micro-dollars.
        assert_eq!(use_.cost_micros(Some(12.0)), Some(384));
        assert_eq!(use_.cost_micros(Some(0.0)), Some(0));
        // No tariff is not a free hour; it is an hour nobody priced.
        assert_eq!(use_.cost_micros(None), None);
        assert_eq!(use_.cost_micros(Some(-1.0)), None);
        assert_eq!(use_.cost_micros(Some(f64::NAN)), None);
        assert_eq!(cost_label(Some(384), Some(12.0)), "≈ $0.000384");
        assert_eq!(cost_label(None, None), "no $/kWh set");
        // A turn too short to reach one micro-dollar is still not a free turn.
        assert_eq!(cost_label(Some(0), Some(12.0)), "≈ under $0.000001");
        assert_eq!(cost_label(Some(0), Some(0.0)), "≈ $0 at a 0 ¢/kWh tariff");
        assert_eq!(cost_label(None, Some(12.0)), "not priced");
    }

    #[test]
    fn a_small_energy_figure_never_prints_as_zero() {
        // The trap this whole formatter exists to avoid: one generation is a few
        // thousandths of a watt-hour, and `≈ 0.00 Wh` is the rendering that reads
        // as a turn that cost nothing.
        assert_eq!(watt_hours_label(Some(3.2)), "≈ 3.2 Wh");
        assert_eq!(watt_hours_label(Some(0.34)), "≈ 0.34 Wh");
        assert_eq!(watt_hours_label(Some(0.01)), "≈ 0.01 Wh");
        // Below a hundredth of a watt-hour the unit changes rather than the
        // digits running out: 0.0032 Wh is 3.2 mWh, and both say the same thing.
        assert_eq!(watt_hours_label(Some(0.0032)), "≈ 3.20 mWh");
        assert_eq!(watt_hours_label(Some(0.000001)), "≈ under 0.01 mWh");
        assert_eq!(watt_hours_label(Some(0.0)), "≈ 0 Wh");
        assert_eq!(watt_hours_label(None), "energy unknown");
        // The seconds keep a digit while they are small, so a window that closed
        // almost at once is short rather than absent.
        assert_eq!(elapsed_label(Duration::from_millis(4)), "under 0.01 s");
        assert_eq!(elapsed_label(Duration::from_millis(1400)), "1.4 s");
        assert_eq!(elapsed_label(Duration::from_secs(46)), "46 s");
    }

    #[test]
    fn a_gpu_that_will_not_answer_is_left_out_of_the_total_rather_than_zeroed() {
        let with_gpu = PowerUse {
            elapsed: Duration::from_secs(3600),
            cpu_energy_uj: Some(3_600_000_000), // 1 Wh
            gpu_power_w: Some(25.0),            // 25 W for an hour is 25 Wh
        };
        assert_eq!(with_gpu.total_watt_hours(), Some(26.0));
        let cpu_only = PowerUse {
            gpu_power_w: None,
            ..with_gpu
        };
        assert_eq!(cpu_only.total_watt_hours(), Some(1.0));
    }

    #[test]
    fn a_window_divides_energy_by_the_seconds_it_watched() {
        let use_ = PowerUse {
            elapsed: Duration::from_secs(40),
            cpu_energy_uj: Some(144_000_000), // 144 J over 40 s is 3.6 W
            gpu_power_w: None,
        };
        let watts = use_
            .mean_power_w()
            .expect("144 J over 40 s is a real figure");
        assert!(
            (watts - 3.6).abs() < 1e-5,
            "144 J over 40 s should read as 3.6 W, not {watts}"
        );
        // A window that measured nothing has no watts, and a window that lasted
        // no time cannot be divided by.
        let blind = PowerUse {
            elapsed: Duration::from_secs(40),
            cpu_energy_uj: None,
            gpu_power_w: None,
        };
        assert_eq!(blind.mean_power_w(), None);
        let instant = PowerUse {
            elapsed: Duration::ZERO,
            cpu_energy_uj: Some(144_000_000),
            gpu_power_w: None,
        };
        assert_eq!(instant.mean_power_w(), None);
    }

    #[test]
    fn only_a_turn_that_stayed_on_this_machine_is_priced_from_its_energy() {
        use crate::metrics::{MetricSource, RequestMetrics};

        let use_ = PowerUse {
            elapsed: Duration::from_secs(40),
            cpu_energy_uj: Some(11_520_000_000), // 3.2 Wh
            gpu_power_w: None,
        };
        let mut local = RequestMetrics::new("BALANCED", 8192);
        local.source = Some(MetricSource::Local);
        assert!(use_.apply_to(Some(12.0), &mut local));
        assert_eq!(local.energy_uj, Some(11_520_000_000));
        assert_eq!(local.elapsed_ms, Some(40_000));
        let local_watts = local.power_w.expect("3.2 Wh over 40 s is a real figure");
        assert!(
            (local_watts - 288.0).abs() < 1e-3,
            "3.2 Wh over 40 s should read as 288 W, not {local_watts}"
        );
        assert_eq!(local.est_cost_micros, Some(384));

        // A cloud turn's electricity was billed to the provider. Writing this
        // machine's idle watts onto it would be a figure nobody paid.
        let mut cloud = RequestMetrics::new("BALANCED", 8192);
        cloud.source = Some(MetricSource::Cloud);
        assert!(!use_.apply_to(Some(12.0), &mut cloud));
        assert_eq!(cloud.energy_uj, None);
        assert_eq!(cloud.power_w, None);
        assert_eq!(cloud.est_cost_micros, None);
        // The seconds are still the turn's own, so a daily time budget can count
        // a day spent against a provider.
        assert_eq!(cloud.elapsed_ms, Some(40_000));

        // A local turn with no tariff still reports what it drew; only the price
        // is unknown.
        let mut unpriced = RequestMetrics::new("BALANCED", 8192);
        unpriced.source = Some(MetricSource::Local);
        assert!(use_.apply_to(None, &mut unpriced));
        assert_eq!(unpriced.energy_uj, Some(11_520_000_000));
        assert_eq!(unpriced.est_cost_micros, None);
    }

    #[test]
    fn the_row_gains_two_fields_and_an_old_row_still_reads() {
        use crate::metrics::RequestMetrics;

        // The rows already on disk have no energy in them. They must keep parsing,
        // and the metrics log stays append-only across the change.
        let row = RequestMetrics::new("LOW", 4096);
        assert_eq!(row.energy_uj, None);
        assert_eq!(row.elapsed_ms, None);
        // A row written by this binary carries both; a row from before it carries
        // neither and reads back as unknown rather than as zero energy.
        let line = serde_json::to_string(&row).unwrap();
        assert!(line.contains("\"energy_uj\":null"), "{line}");
        let before: serde_json::Value = serde_json::from_str(&line).unwrap();
        let stripped = serde_json::Value::Object(
            before
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, _)| **key != "energy_uj" && **key != "elapsed_ms")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let read: RequestMetrics = serde_json::from_value(stripped).unwrap();
        assert_eq!(read.energy_uj, None);
        assert_eq!(read.elapsed_ms, None);
    }

    #[test]
    fn the_line_says_what_was_measured_and_what_the_machine_hid() {
        let priced = PowerUse {
            elapsed: Duration::from_secs(41),
            cpu_energy_uj: Some(11_520_000_000),
            gpu_power_w: Some(30.0),
        };
        let line = power_line(&priced, Some(12.0));
        // 3.2 Wh off the package counter plus 30 W held for 41 s off the card
        // (0.34 Wh), at 12 cents a kilowatt-hour.
        assert_eq!(
            line, "≈ 3.5 Wh · ≈ $0.000425 · 41 s · estimated, this machine only",
            "{line}"
        );
        assert!(!line.contains("CPU package only"), "{line}");
        assert!(
            line.ends_with("· estimated, this machine only"),
            "{line} — a figure this machine cannot attribute to one process has to say so"
        );

        let unpriced = PowerUse {
            gpu_power_w: None,
            ..priced
        };
        let line = power_line(&unpriced, Some(12.0));
        assert!(!line.contains("no $/kWh set"), "{line}");
        assert!(line.contains("≈ $"), "{line}");

        let no_tariff = power_line(&unpriced, None);
        assert!(no_tariff.contains("no $/kWh set"), "{no_tariff}");
        assert!(
            no_tariff.contains("CPU package only"),
            "{no_tariff} — a card that answered nothing must be named"
        );

        let blind = PowerUse {
            elapsed: Duration::from_secs(75),
            cpu_energy_uj: None,
            gpu_power_w: None,
        };
        let line = power_line(&blind, Some(12.0));
        assert!(line.contains("energy unknown"), "{line}");
        assert!(
            line.contains("no energy counter"),
            "{line} — the reason belongs on the line"
        );
        // A tariff is set on this machine; the counter is what is missing. Saying
        // `no $/kWh set` here would point at a setting that is already right.
        assert!(!line.contains('$'), "{line}");
        assert!(line.contains("1 min 15 s"), "{line}");
    }
}
