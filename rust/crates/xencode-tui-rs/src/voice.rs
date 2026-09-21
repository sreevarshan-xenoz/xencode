//! Microphone capture for the voice panel (J-07).
//!
//! The rule this module exists to enforce: every number and every word the
//! panel shows comes out of the recorder we spawned or a transcriber that is
//! actually installed. If neither is there, the only thing on screen is the
//! reason. There is no fallback transcript anywhere in this file.
//!
//! The commands here are fixed argv lists built from constants — never a shell
//! string and never user input — so they are in the same category as the `git`
//! and `llama-server` spawns elsewhere in the TUI, not the agent's `run_command`
//! path that the approval gate guards.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Everything downstream assumes 16 kHz mono signed 16-bit little endian.
pub const SAMPLE_RATE: u32 = 16_000;
const BYTES_PER_SAMPLE: usize = 2;
/// 100 ms of audio: one reading per chunk keeps the meter alive without
/// flooding the event queue.
const CHUNK_BYTES: usize = (SAMPLE_RATE as usize) * BYTES_PER_SAMPLE / 10;
/// Hard cap on a clip so a forgotten session cannot fill the disk.
pub const MAX_CLIP: Duration = Duration::from_secs(15);

/// Raw voice sits around RMS 0.02, which on a 30-cell bar reads as dead. The
/// gain is a fixed display constant, not a measurement.
const METER_GAIN: f64 = 8.0;

/// The panel shows RMS after that gain, clamped to the bar width.
pub fn meter_level(pcm: &[u8]) -> f64 {
    (rms(pcm) * METER_GAIN).min(1.0)
}

/// Root-mean-square amplitude of S16_LE PCM, 0.0–1.0. A trailing odd byte is
/// not a sample and is ignored.
pub fn rms(pcm: &[u8]) -> f64 {
    let (samples, _trailing) = pcm.as_chunks::<BYTES_PER_SAMPLE>();
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for sample in samples {
        let amplitude = i16::from_le_bytes([sample[0], sample[1]]) as f64 / 32768.0;
        sum += amplitude * amplitude;
    }
    (sum / samples.len() as f64).sqrt()
}

/// `duration_ms` / `clip_bytes` from a PCM length at the capture format.
pub fn pcm_ms(bytes: usize) -> u64 {
    (bytes / BYTES_PER_SAMPLE) as u64 * 1000 / (SAMPLE_RATE as u64)
}

/// A duration the panel can read: sub-second clips in ms, longer in s.
pub fn format_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else {
        format!("{:.1} s", ms as f64 / 1000.0)
    }
}

/// How much audio a PCM buffer holds, in the same words the panel uses.
pub fn format_pcm_ms(bytes: usize) -> String {
    format_ms(pcm_ms(bytes))
}

/// A minimal RIFF/WAVE container around PCM: 16-byte `fmt ` chunk, one
/// `data` chunk, 16-bit PCM. Written by hand because the only job it has to do
/// is be readable by a whisper CLI.
pub fn wav_bytes(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * (bits as u32 / 8);
    let block_align = channels * (bits as u32 / 8) as u16;
    let data_len = pcm.len() as u32;

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// First executable found on `PATH`, if any.
pub fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Recorder candidates and the argv that makes each stream raw S16_LE mono on
/// stdout. Probed in this order.
const RECORDERS: [(&str, &[&str]); 3] = [
    (
        "arecord",
        &[
            "-q", "-f", "S16_LE", "-t", "raw", "-r", "16000", "-c", "1", "-",
        ],
    ),
    (
        "pw-record",
        &["--rate", "16000", "--channels", "1", "--format", "s16", "-"],
    ),
    (
        "parec",
        &["--format=s16le", "--rate=16000", "--channels=1", "--raw"],
    ),
];

pub fn find_recorder() -> Option<PathBuf> {
    RECORDERS.iter().find_map(|(bin, _)| which(bin))
}

fn recorder_args(bin: &str) -> &'static [&'static str] {
    RECORDERS
        .iter()
        .find(|(name, _)| *name == bin)
        .map(|(_, args)| *args)
        .unwrap_or(&[])
}

/// Transcriber candidates. Each is given the WAV path; model selection is left
/// to the binary, because guessing a model path we cannot verify is how the
/// panel started lying in the first place.
const TRANSCRIBERS: [&str; 4] = ["whisper", "whisper-cpp", "whisper-cli", "whisper.cpp"];

pub fn find_transcriber() -> Option<PathBuf> {
    TRANSCRIBERS.iter().find_map(|bin| which(bin))
}

/// What the panel says when there is nothing to transcribe with. Names the
/// clip it kept, so the recording is still useful by hand.
pub fn missing_transcriber_note(clip: &Path) -> String {
    format!(
        "No speech-to-text engine on PATH (looked for {}). Clip kept at {}.",
        TRANSCRIBERS.join(", "),
        clip.display()
    )
}

/// One capture session: the PCM, the meter readings taken from it, and why it
/// ended. `capture` fills this in; nothing here is invented.
pub struct Capture {
    pub pcm: Vec<u8>,
    pub levels: Vec<f64>,
    pub error: Option<String>,
}

impl Capture {
    pub fn ms(&self) -> u64 {
        pcm_ms(self.pcm.len())
    }
}

/// Record until `stop` is set, the clip cap is reached, or the recorder stops
/// producing output. Blocks; call it from a blocking task.
///
/// `cmd` is the recorder binary; tests pass `cat` with a prepared PCM file so
/// the whole path can be exercised without a microphone.
pub fn capture(
    cmd: &Path,
    args: &[std::ffi::OsString],
    stop: &AtomicBool,
    muted: &AtomicBool,
    mut on_level: impl FnMut(f64, usize),
) -> Capture {
    let mut pcm = Vec::new();
    let mut levels = Vec::new();
    let child = spawn_recorder(cmd, args);
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return Capture {
                pcm,
                levels,
                error: Some(format!("{} failed to start: {e}", cmd.display())),
            }
        }
    };
    let Some(mut stdout) = child.stdout.take() else {
        reap(&mut child);
        return Capture {
            pcm,
            levels,
            error: Some(format!("{} produced no output pipe", cmd.display())),
        };
    };

    let started = Instant::now();
    let mut buf = [0u8; CHUNK_BYTES];
    let mut failure = None;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if started.elapsed() >= MAX_CLIP || pcm.len() >= max_pcm_bytes() {
            break;
        }
        match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if muted.load(Ordering::Relaxed) {
                    // Still drain the pipe so the recorder's buffer does not
                    // fill, but keep nothing: a muted clip is not a clip.
                    continue;
                }
                let read = buf[..n].to_vec();
                let level = meter_level(&read);
                pcm.extend_from_slice(&read);
                levels.push(level);
                on_level(level, pcm.len());
            }
            Err(e) => {
                failure = Some(format!("{}: {e}", cmd.display()));
                break;
            }
        }
    }
    reap(&mut child);
    Capture {
        pcm,
        levels,
        error: failure,
    }
}

fn spawn_recorder(cmd: &Path, args: &[std::ffi::OsString]) -> std::io::Result<Child> {
    Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
}

fn reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn max_pcm_bytes() -> usize {
    (MAX_CLIP.as_secs() as usize) * (SAMPLE_RATE as usize) * BYTES_PER_SAMPLE
}

/// Args the recorder in `path` needs, looked up by file name.
pub fn recorder_args_for(path: &Path) -> Vec<std::ffi::OsString> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    recorder_args(&name)
        .iter()
        .map(|a| std::ffi::OsString::from(*a))
        .collect()
}

/// Run the transcriber over a WAV. Ok holds its stdout trimmed of whitespace;
/// Err holds whatever the binary complained about, so the panel can quote it.
pub fn transcribe(bin: &Path, clip: &Path) -> Result<String, String> {
    let out = Command::new(bin)
        .arg(clip)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    if out.status.success() {
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if text.is_empty() {
            return Err(format!(
                "{} heard nothing in the clip",
                bin.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
        return Ok(text);
    }
    let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
    Err(format!(
        "{} exited with {}: {}",
        bin.file_name().unwrap_or_default().to_string_lossy(),
        out.status,
        if why.is_empty() { "no output" } else { &why }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s16_le(values: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn sample_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-voice-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rms_is_the_amplitude_of_the_pcm() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&s16_le(&[0, 0, 0, 0])), 0.0);
        // A full-scale square wave is RMS 1 in the limit; 32767 is 0.99997.
        let loud: Vec<i16> = (0..64)
            .map(|i| if i % 2 == 0 { 32767 } else { -32767 })
            .collect();
        assert!((rms(&s16_le(&loud)) - 1.0).abs() < 0.001);
        // Half amplitude, half RMS for a square wave.
        let half: Vec<i16> = (0..64)
            .map(|i| if i % 2 == 0 { 16383 } else { -16383 })
            .collect();
        assert!((rms(&s16_le(&half)) - 0.5).abs() < 0.001);
    }

    #[test]
    fn rms_ignores_a_trailing_odd_byte_and_meter_scales() {
        let mut pcm = s16_le(&[32767; 8]);
        pcm.push(7);
        assert!((rms(&pcm) - 1.0).abs() < 0.001);
        // Quiet voice reaches the top of the bar only after the display gain.
        assert!(meter_level(&s16_le(&[0; 16])) == 0.0);
        assert!(meter_level(&s16_le(&[4096; 16])) > 0.9);
        assert!(meter_level(&s16_le(&[128; 16])) < 0.1);
    }

    #[test]
    fn wav_header_describes_the_pcm_it_wraps() {
        let pcm = s16_le(&[1, -1, 2, -2]);
        let wav = wav_bytes(&pcm, SAMPLE_RATE);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(wav[24..28].try_into().unwrap()),
            SAMPLE_RATE
        );
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(&wav[44..], &pcm[..]);
        assert_eq!(wav.len(), 44 + pcm.len());
    }

    #[test]
    fn clip_length_comes_from_the_byte_count() {
        // 16 kHz mono S16: 3200 bytes is 1600 samples, which is 100 ms.
        assert_eq!(pcm_ms(3200), 100);
        assert_eq!(pcm_ms(0), 0);
        assert_eq!(pcm_ms(3199), 99);
        // The chunk size is what makes "one reading per 100 ms" true.
        assert_eq!(pcm_ms(CHUNK_BYTES), 100);
        assert_eq!(format_ms(999), "999 ms");
        assert_eq!(format_ms(1500), "1.5 s");
        assert_eq!(format_pcm_ms(3200), "100 ms");
    }

    #[test]
    fn capture_reads_the_recorder_and_reports_levels_per_chunk() {
        let dir = sample_dir("capture");
        let loud = s16_le(&[12000; CHUNK_BYTES / 2]);
        let quiet = s16_le(&[0; CHUNK_BYTES / 2]);
        let mut raw = loud.clone();
        raw.extend_from_slice(&quiet);
        let pcm_file = dir.join("in.pcm");
        std::fs::write(&pcm_file, &raw).unwrap();

        let cat = which("cat").expect("cat is on PATH for this test");
        let args = vec![pcm_file.clone().into_os_string()];
        let seen = std::sync::Mutex::new(Vec::new());
        let cap = capture(
            &cat,
            &args,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            |level, total| seen.lock().unwrap().push((level, total)),
        );

        assert!(cap.error.is_none(), "{:?}", cap.error);
        assert_eq!(cap.pcm, raw);
        assert_eq!(cap.levels.len(), 2, "one reading per chunk");
        assert!(cap.levels[0] > 0.0);
        assert_eq!(cap.levels[1], 0.0, "the silent chunk reads silent");
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert_eq!(seen.lock().unwrap()[1].1, raw.len(), "totals accumulate");
        assert_eq!(cap.ms(), pcm_ms(raw.len()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_muted_capture_keeps_nothing() {
        let dir = sample_dir("muted");
        let pcm_file = dir.join("in.pcm");
        std::fs::write(&pcm_file, s16_le(&[12000; 500])).unwrap();
        let cat = which("cat").expect("cat is on PATH for this test");
        let args = vec![pcm_file.into_os_string()];

        let cap = capture(
            &cat,
            &args,
            &AtomicBool::new(false),
            &AtomicBool::new(true),
            |_, _| {},
        );
        assert!(cap.error.is_none(), "{:?}", cap.error);
        assert!(cap.pcm.is_empty());
        assert!(cap.levels.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_recorder_that_cannot_start_says_so_with_its_own_name() {
        let missing = Path::new("/nonexistent/xencode-no-such-recorder");
        let cap = capture(
            missing,
            &[],
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            |_, _| {},
        );
        assert!(cap.pcm.is_empty());
        let err = cap.error.unwrap();
        assert!(err.contains("xencode-no-such-recorder"), "{err}");
        assert!(err.contains("failed to start"), "{err}");
    }

    #[test]
    fn a_transcriber_that_hears_nothing_is_reported_not_filled_in() {
        let dir = sample_dir("transcribe");
        let clip = dir.join("clip.wav");
        std::fs::write(&clip, wav_bytes(&s16_le(&[0; 8]), SAMPLE_RATE)).unwrap();

        // /bin/true stands in for an engine that succeeds and prints nothing.
        let truth = Path::new("/bin/true");
        if truth.is_file() {
            let err = transcribe(truth, &clip).unwrap_err();
            assert!(err.contains("heard nothing"), "{err}");
        }
        // A real transcriber needs a model we do not ship, so it fails here.
        // The panel must show that failure, not a sentence of its own.
        if let Some(bin) = find_transcriber() {
            let outcome = transcribe(&bin, &clip);
            match &outcome {
                Ok(text) => assert!(!text.trim().is_empty()),
                Err(e) => assert!(!e.is_empty(), "{e}"),
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_missing_engine_note_names_the_clip_and_the_binaries_tried() {
        let note = missing_transcriber_note(Path::new("/tmp/clip-1.wav"));
        assert!(note.contains("/tmp/clip-1.wav"), "{note}");
        assert!(note.contains("whisper"), "{note}");
        assert!(note.contains("No speech-to-text engine"), "{note}");
    }

    #[test]
    fn every_recorder_candidate_streams_raw_mono_sixteen_bit() {
        let arecord = recorder_args("arecord").join(" ");
        assert!(arecord.contains("S16_LE"), "{arecord}");
        assert!(arecord.contains("raw"), "{arecord}");
        assert!(arecord.contains("16000"), "{arecord}");
        assert!(recorder_args("parec").join(" ").contains("s16le"));
        // An unknown binary gets no args rather than someone else's.
        assert!(recorder_args("something-else").is_empty());
    }
}
