//! `xencode speak` — text-to-speech output through a local program.

pub(crate) fn run_speak(text: &str) -> Result<(), String> {
    xencode_tui_rs::voice_out::speak(text).map_err(|error| error.to_string())
}
