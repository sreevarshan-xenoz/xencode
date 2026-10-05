//! The commands that write `config.json` run against a file that a hand edit
//! has broken, driven through the real binary — because the defect this covers
//! was not a crash. It was a successful-looking command that quietly replaced
//! an unreadable config with the default block, settings and API keys gone.
//!
//! Every case here is about what the person sees and what is left on disk: a
//! non-zero exit, a message naming the file, and their own bytes still in it.

use std::path::Path;
use std::process::Output;

/// The damage a hand edit makes and the shape the guard has to catch: valid
/// settings, one trailing comma. `MYMARKERMODEL` is what a run that overwrote
/// the file would have erased.
const BROKEN: &[u8] =
    br#"{"default_model":"ollama:MYMARKERMODEL","openai_api_key":"sk-FAKE-NOT-A-REAL-TEST-KEY",}"#;

/// A config directory holding only that file, so the developer's own
/// `~/.config/xencode` is neither read nor written.
fn broken_config(dir: &Path) -> std::path::PathBuf {
    let config_dir = dir.join("xencode");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(config_dir.join("config.json"), BROKEN).unwrap();
    config_dir
}

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode is built by the time these tests run")
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-config-refusal-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The two settings the tests watch, out of the file's own text rather than by
/// loading it — a config that this binary refuses to read must be checked
/// without asking that binary what it thinks is in there.
fn their_settings_survive(text: &str) -> bool {
    text.contains("MYMARKERMODEL") && text.contains("sk-FAKE-NOT-A-REAL-TEST-KEY")
}

fn backups(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| {
            entry
                .ok()
                .map(|found| found.file_name().to_string_lossy().into_owned())
        })
        .filter(|name| name.starts_with("config.json.bak."))
        .collect();
    names.sort();
    names
}

#[test]
fn setting_a_value_stops_at_a_file_it_cannot_read() {
    let dir = temp_dir("set-path");
    let config_dir = broken_config(&dir);
    let path = config_dir.join("config.json");

    // This is the command that used to print success. It reports the path it is
    // about to store, and had then saved the whole default block over the file.
    let run = xencode(&config_dir, &["llamacpp", "set-path", "/tmp/some.gguf"]);
    assert!(
        !run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("config.json"), "{stderr}");
    assert!(stderr.contains("not readable JSON"), "{stderr}");
    assert!(stderr.contains("trailing comma"), "{stderr}");
    assert!(
        !stderr.contains("some.gguf"),
        "nothing was stored: {stderr}"
    );

    // Untouched — including the key, which is the reason the guard exists.
    let text = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
    assert!(
        their_settings_survive(&text),
        "the file was rewritten: {text}"
    );
    assert!(
        backups(&config_dir).is_empty(),
        "a refused save must not even copy the file aside"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_config_that_is_json_of_the_wrong_shape_is_refused_the_same_way() {
    let dir = temp_dir("array");
    let config_dir = dir.join("xencode");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(config_dir.join("config.json"), b"[]").unwrap();

    let run = xencode(&config_dir, &["llamacpp", "set-path", "/tmp/some.gguf"]);
    assert!(!run.status.success());
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("array"), "{stderr}");
    assert_eq!(
        std::fs::read(config_dir.join("config.json")).unwrap(),
        b"[]"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_file_that_holds_nothing_is_not_treated_as_damage() {
    let dir = temp_dir("empty");
    let config_dir = dir.join("xencode");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(config_dir.join("config.json"), b"").unwrap();

    // `touch`ed or truncated to nothing: there are no bytes to protect, and
    // refusing here would leave no way to start.
    let run = xencode(&config_dir, &["llamacpp", "set-path", "/tmp/some.gguf"]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = String::from_utf8(std::fs::read(config_dir.join("config.json")).unwrap()).unwrap();
    assert!(text.contains("/tmp/some.gguf"), "{text}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn reset_recovers_from_a_file_no_command_can_write_over() {
    let dir = temp_dir("reset");
    let config_dir = broken_config(&dir);
    let path = config_dir.join("config.json");

    // Refusing to overwrite an unreadable config is only survivable if there is
    // a command that means to. Reset is that command, and it keeps the damage.
    let run = xencode(&config_dir, &["config", "reset"]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
    assert!(
        !their_settings_survive(&text),
        "reset left the broken file: {text}"
    );

    let kept = backups(&config_dir);
    assert_eq!(kept.len(), 1, "the broken bytes were not copied aside");
    let copied = std::fs::read(config_dir.join(&kept[0])).unwrap();
    assert_eq!(
        copied, BROKEN,
        "the copy is of what was there, damage and all"
    );

    // And the config reads again.
    let show = xencode(&config_dir, &["config", "show"]);
    assert!(
        show.status.success(),
        "{}",
        String::from_utf8_lossy(&show.stderr)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
