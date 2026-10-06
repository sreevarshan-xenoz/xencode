//! The machines a person reaches over ssh (`L-2`): what `xencode remote add`
//! records and `xencode remote list` reads back.
//!
//! A profile is a *destination and a plan*, not a connection. Nothing in this
//! module opens a socket or spawns `ssh`: it stores the words that make the
//! later `up` possible, in the settings directory, one file per host, and
//! refuses on the way in anything that could not be handed to `ssh` safely.
//!
//! # Why the validation is here rather than at the spawn
//!
//! `xencode remote add lab '-oProxyCommand=touch /tmp/pwn'` would otherwise sit
//! in a file until some future command passed it as an argument, and an option
//! that early in an argv is read by `ssh` itself, not treated as a hostname. So
//! a name may only hold characters that make it a file name in this directory,
//! and a destination may only hold characters that make it a host: no leading
//! `-`, no whitespace, no shell metacharacters. A hostname nobody can type is
//! refused with the sentence that says why, and `~/.ssh/config` stays the way to
//! reach one — which is also what the row asks for, since an alias is a
//! supported destination.
//!
//! # The active host is a pointer, not a copy
//!
//! `use` writes one name into `remotes/active`. Nothing is duplicated, so there
//! is no second copy of a host to go stale, and `up` with no argument reads the
//! pointer rather than guessing which of the recorded machines was meant.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use xencode_core_rs::write_atomic;

/// The directory profiles live in, under the settings directory.
pub const PROFILE_DIRECTORY: &str = "remotes";

/// The file naming the host `up` uses when nobody says which.
pub const ACTIVE_FILE: &str = "active";

/// The runtime vocabulary, shared with `xencode colab`: these are the two
/// strings the bootstrap scripts and the model picker both understand.
pub const RUNTIMES: [&str; 2] = ["llama.cpp", "ollama"];

/// The port a recorded host serves on inside the machine, when the profile says
/// nothing: the runtime's own default, resolved by the same rule Colab uses.
pub const DEFAULT_LOCAL_PORT: u16 = 18_100;

/// Why something was refused, in a sentence a person can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteError(pub String);

impl std::fmt::Display for RemoteError {
    fn fmt(&self, form: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        form.write_str(&self.0)
    }
}

impl std::error::Error for RemoteError {}

impl From<std::io::Error> for RemoteError {
    fn from(problem: std::io::Error) -> Self {
        Self(format!("the profiles could not be read: {problem}"))
    }
}

impl From<crate::ConfigError> for RemoteError {
    fn from(problem: crate::ConfigError) -> Self {
        Self(format!(
            "the settings directory could not be resolved: {problem}"
        ))
    }
}

/// Where the profiles live. Resolved through [`crate::paths`] so a sandboxed
/// `HOME`, an `XCODE_CONFIG_DIR` and a machine still on `~/.xencode` all work,
/// and so a profile can never be written outside the directory xencode owns.
pub fn profiles_dir() -> Result<PathBuf, RemoteError> {
    Ok(crate::paths::settings_dir()?.join(PROFILE_DIRECTORY))
}

/// One machine. Every field is a string the CLI will later hand to `ssh` or to
/// a runtime, which is why each one is checked on the way in rather than at the
/// point of use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteProfile {
    /// The name this machine is known by, and the file it is stored in.
    pub name: String,
    /// The host or `~/.ssh/config` alias to connect to.
    pub host: String,
    /// The user to connect as. `None` lets `ssh` use the local one.
    #[serde(default)]
    pub user: Option<String>,
    /// The ssh port. `22` unless the host says otherwise.
    #[serde(default = "default_ssh_port")]
    pub port: u16,
    /// `"llama.cpp"` or `"ollama"` — see [`RUNTIMES`].
    pub runtime: String,
    /// The model to serve, in that runtime's form (`qwen2.5:7b` for ollama, a
    /// Hugging Face repo for llama.cpp). Optional: `up` can be told later.
    #[serde(default)]
    pub model: Option<String>,
    /// The local port the forward exposes.
    #[serde(default = "default_local_port")]
    pub local_port: u16,
    /// The port the server binds inside the machine. `0` means the runtime's
    /// own default, which is 18080 for llama.cpp and 11434 for ollama.
    #[serde(default)]
    pub remote_port: u16,
}

fn default_ssh_port() -> u16 {
    22
}

fn default_local_port() -> u16 {
    DEFAULT_LOCAL_PORT
}

impl RemoteProfile {
    /// `user@host` or `host`, the way it goes on a command line.
    pub fn destination(&self) -> String {
        match &self.user {
            Some(user) => format!("{user}@{}", self.host),
            None => self.host.clone(),
        }
    }

    /// The one-line description `list` prints for this profile.
    pub fn summary(&self) -> String {
        let serving = match &self.model {
            Some(model) => format!("{} · {model}", self.runtime),
            None => self.runtime.clone(),
        };
        format!(
            "{} · port {} · {serving} · local {}",
            self.destination(),
            self.port,
            self.local_port
        )
    }
}

/// A destination as it was typed, split into the three things `ssh` needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Destination {
    pub user: Option<String>,
    pub host: String,
    pub port: Option<u16>,
}

/// Characters a host, an alias or a user may hold. Deliberately short of what
/// a shell would accept: nothing here can quote its way out of an argument,
/// because nothing here can be a quote, a space or a semicolon in the first
/// place. A colon is excluded because `:port` is the separator this function
/// reads, so a host holding one is an IPv6 literal, which `ssh` wants in
/// brackets and this parser refuses in favour of `~/.ssh/config`.
fn is_plain_token(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('-')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// The name of a profile, checked as a file name: it becomes one, and `..` or a
/// slash would make it a file somewhere else.
pub fn validate_name(name: &str) -> Result<(), RemoteError> {
    let ok = name.len() <= 32
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !name.contains("..")
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(RemoteError(format!(
            "`{name}` is not a name xencode can store a host under: use up to 32 of a-z, 0-9, `_`, `-` and `.` \
             (no spaces, no `/`, nothing beginning with `-` or `.`). A name is a file in {}, so a slash \
             would be a file somewhere else.",
            crate::paths::settings_dir()
                .map(|dir| dir.display().to_string())
                .unwrap_or_else(|_| "the settings directory".to_string())
        )))
    }
}

/// `[user@]host[:port]`, as typed. An `ssh` alias is a host: the same characters
/// are allowed, and the alias is resolved by `ssh` later, not here.
pub fn parse_destination(spec: &str) -> Result<Destination, RemoteError> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(RemoteError(
            "a host is needed: `xencode remote add <name> <[user@]host[:port]>`, or the name of an \
             entry in `~/.ssh/config`"
                .to_string(),
        ));
    }
    let (user, rest) = match spec.split_once('@') {
        Some((user, rest)) => {
            if rest.contains('@') {
                return Err(RemoteError(format!(
                    "`{spec}` has two `@` signs: name the user once, as `user@host`"
                )));
            }
            (Some(user), rest)
        }
        None => (None, spec),
    };
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port_text)) => {
            let port = port_text.parse::<u16>().map_err(|_| {
                RemoteError(format!(
                    "`{port_text}` after the `:` in `{spec}` is not a port number (1 to 65535). If the \
                     host itself contains a colon, it is an IPv6 literal: put it in `~/.ssh/config` and \
                     give xencode the alias instead."
                ))
            })?;
            if port == 0 {
                return Err(RemoteError(format!(
                    "port 0 in `{spec}` is not a port anyone can connect to"
                )));
            }
            (host, Some(port))
        }
        None => (rest, None),
    };
    if !is_plain_token(host) {
        return Err(RemoteError(refuse_token(host, "host")));
    }
    if let Some(user) = user {
        if !is_plain_token(user) {
            return Err(RemoteError(refuse_token(user, "user")));
        }
    }
    Ok(Destination {
        user: user.map(str::to_string),
        host: host.to_string(),
        port,
    })
}

fn refuse_token(token: &str, what: &str) -> String {
    format!(
        "`{token}` is not a {what} xencode will hand to `ssh`: letters, digits, `.`, `_` and `-` only, \
         with no leading `-` (an option would be read by ssh itself) and no colon — a colon is where \
         the port starts, so an IPv6 address belongs in `~/.ssh/config` and xencode gets its alias."
    )
}

/// The runtime, checked against [`RUNTIMES`] rather than corrected. A typo here
/// is a bootstrap script that installs the wrong thing on somebody's machine.
pub fn validate_runtime(runtime: &str) -> Result<(), RemoteError> {
    if RUNTIMES.contains(&runtime) {
        Ok(())
    } else {
        Err(RemoteError(format!(
            "`{runtime}` is not a runtime xencode can start: `--runtime llama.cpp` or `--runtime ollama`"
        )))
    }
}

/// A model id, checked so that a space or a shell character cannot reach a
/// `docker pull`-style argument later. Both runtimes write ids with `/`, `:` and
/// `.` in them, so those are allowed.
pub fn validate_model(model: &str) -> Result<(), RemoteError> {
    let ok = !model.is_empty()
        && !model.starts_with('-')
        && model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':'));
    if ok {
        Ok(())
    } else {
        Err(RemoteError(format!(
            "`{model}` is not a model id xencode will pass to a runtime: letters, digits, `.`, `_`, `-`, \
             `/` and `:` only, with no spaces"
        )))
    }
}

fn profile_path(name: &str) -> Result<PathBuf, RemoteError> {
    validate_name(name)?;
    Ok(profiles_dir()?.join(format!("{name}.json")))
}

/// Record a host. An existing name is refused rather than overwritten, because
/// the profile is the thing a person typed once and a silent rewrite is how the
/// wrong machine gets used an hour later; `force` is the way to say otherwise.
pub fn save(profile: &RemoteProfile, force: bool) -> Result<PathBuf, RemoteError> {
    validate_name(&profile.name)?;
    parse_destination(&profile.destination())?;
    validate_runtime(&profile.runtime)?;
    if let Some(model) = &profile.model {
        validate_model(model)?;
    }
    if profile.port == 0 {
        return Err(RemoteError(
            "port 0 is not a port anyone can connect to".to_string(),
        ));
    }
    if profile.local_port == 0 {
        return Err(RemoteError(
            "the local port cannot be 0; pick a port the forward should listen on, or leave it out for 18100"
                .to_string(),
        ));
    }
    let path = profile_path(&profile.name)?;
    std::fs::create_dir_all(path.parent().expect("the profiles directory"))?;
    if path.exists() && !force {
        return Err(RemoteError(format!(
            "`{}` is already recorded in {}. `xencode remote forget {}` first, or pass --force to \
             replace it",
            profile.name,
            path.display(),
            profile.name
        )));
    }
    let json = serde_json::to_string_pretty(profile)
        .map_err(|problem| RemoteError(format!("the profile could not be written: {problem}")))?;
    write_atomic(&path, json.as_bytes())?;
    Ok(path)
}

/// The recorded host under `name`, if there is one.
pub fn load(name: &str) -> Result<Option<RemoteProfile>, RemoteError> {
    let path = profile_path(name)?;
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<RemoteProfile>(&text)
            .map(Some)
            .map_err(|problem| {
                RemoteError(format!(
                    "{} is not a profile xencode can read: {problem}",
                    path.display()
                ))
            }),
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(problem) => Err(problem.into()),
    }
}

/// What is recorded, in name order, plus any file in the directory that is not
/// a profile. A damaged entry is named rather than skipped quietly: a host that
/// has vanished from `list` looks the same as a host that was forgotten, and
/// those are different things to have happen.
pub fn list() -> Result<Inventory, RemoteError> {
    let dir = profiles_dir()?;
    let mut inventory = Inventory::default();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => return Ok(inventory),
        Err(problem) => return Err(problem.into()),
    };
    for entry in entries {
        let path = entry?.path();
        let Some(file) = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        else {
            continue;
        };
        let Some(stem) = file.strip_suffix(".json") else {
            continue;
        };
        match load(stem) {
            Ok(Some(profile)) => inventory.profiles.push(profile),
            Ok(None) => {}
            Err(problem) => inventory.unreadable.push((stem.to_string(), problem.0)),
        }
    }
    inventory.profiles.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(inventory)
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Inventory {
    pub profiles: Vec<RemoteProfile>,
    pub unreadable: Vec<(String, String)>,
}

impl Inventory {
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty() && self.unreadable.is_empty()
    }
}

/// Which host `up` should use when nobody names one. Refused for a name that is
/// not recorded, because a pointer to a machine that does not exist is what
/// makes the next command fail in a place far away from here.
pub fn set_active(name: &str) -> Result<(), RemoteError> {
    validate_name(name)?;
    if load(name)?.is_none() {
        return Err(RemoteError(format!(
            "`{name}` is not recorded. `xencode remote list` shows what is; `xencode remote add {name} \
             <user@host>` adds it"
        )));
    }
    let dir = profiles_dir()?;
    std::fs::create_dir_all(&dir)?;
    write_atomic(&dir.join(ACTIVE_FILE), format!("{name}\n").as_bytes())?;
    Ok(())
}

fn active_path(dir: &Path) -> PathBuf {
    dir.join(ACTIVE_FILE)
}

/// The name written in the pointer file, without reading the profile behind it.
/// `forget` needs exactly this: by the time it asks whether the host it is
/// removing was the chosen one, that host's own file is already gone, and a
/// check that reads the profile would find nothing and report a clean removal.
fn pointer_name(dir: &Path) -> Result<Option<String>, RemoteError> {
    match std::fs::read_to_string(active_path(dir)) {
        Ok(text) => {
            let name = text.trim();
            Ok(if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            })
        }
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(problem) => Err(problem.into()),
    }
}

/// The host in use, if one has been chosen. A pointer naming a host that is no
/// longer there reads as `None`: the choice is gone, and saying so is better
/// than returning a name with no profile behind it.
pub fn active() -> Result<Option<RemoteProfile>, RemoteError> {
    let dir = profiles_dir()?;
    match pointer_name(&dir)? {
        Some(name) => load(&name),
        None => Ok(None),
    }
}

/// What `forget` did, since clearing the pointer is worth saying out loud.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Forgot {
    Removed,
    RemovedWhileActive,
    NotRecorded,
}

/// Give up a recorded host. The file goes; the pointer goes with it if that host
/// was the one in use, so nothing is left naming a machine that has been
/// forgotten.
pub fn forget(name: &str) -> Result<Forgot, RemoteError> {
    let path = profile_path(name)?;
    if !path.exists() {
        return Ok(Forgot::NotRecorded);
    }
    let chosen = pointer_name(&profiles_dir()?)?;
    std::fs::remove_file(&path)?;
    if chosen.as_deref() == Some(name) {
        std::fs::remove_file(active_path(&profiles_dir()?)).ok();
        return Ok(Forgot::RemovedWhileActive);
    }
    Ok(Forgot::Removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Point the settings directory at one empty scratch directory, so a test
    /// writes profiles that belong to nobody else and can never read the real
    /// `~/.config/xencode`. The environment is process-global, so the same lock
    /// `crate::paths` uses for its own tests is taken here too.
    fn with_scratch<T>(body: impl FnOnce(&Path) -> T) -> T {
        let _guard = crate::paths::env_lock();
        let scratch = std::env::temp_dir().join(format!(
            "xencode-remotes-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        let previous = std::env::var_os("XCODE_CONFIG_DIR");
        std::env::set_var("XCODE_CONFIG_DIR", &scratch);
        let result = body(&scratch);
        match previous {
            Some(value) => std::env::set_var("XCODE_CONFIG_DIR", value),
            None => std::env::remove_var("XCODE_CONFIG_DIR"),
        }
        let _ = std::fs::remove_dir_all(&scratch);
        result
    }

    fn profile(name: &str) -> RemoteProfile {
        RemoteProfile {
            name: name.to_string(),
            host: "100.100.100.100".to_string(),
            user: Some("work".to_string()),
            port: 22,
            runtime: "llama.cpp".to_string(),
            model: Some("qwen2.5:7b".to_string()),
            local_port: DEFAULT_LOCAL_PORT,
            remote_port: 0,
        }
    }

    #[test]
    fn a_name_makes_a_file_and_nothing_else() {
        for refused in [
            "../elsewhere",
            "a/b",
            "-lead",
            ".hidden",
            "",
            &"n".repeat(33),
            "Bad Name",
        ] {
            let error = validate_name(refused)
                .expect_err(&format!("`{refused}` should not be a storable name"));
            assert!(
                error
                    .0
                    .contains("is not a name xencode can store a host under"),
                "the refusal said something else: {error}"
            );
        }
        for accepted in ["lab", "lab-box", "gpu2.box", "7"] {
            validate_name(accepted)
                .unwrap_or_else(|problem| panic!("`{accepted}` should be a name: {problem}"));
        }
    }

    #[test]
    fn a_destination_cannot_carry_an_option_to_ssh() {
        let refused = [
            "-oProxyCommand=touch /tmp/pwn",
            "two@@hops",
            "fe80::1",
            "host:0",
            "host:notaport",
            "host name",
            "user;-l",
            "",
        ];
        for spec in refused {
            let error = parse_destination(spec)
                .expect_err(&format!("`{spec}` should be refused before it reaches ssh"));
            assert!(!error.0.is_empty(), "`{spec}` was refused with no sentence");
        }
        assert_eq!(
            parse_destination("work@lab-box:2222").unwrap(),
            Destination {
                user: Some("work".to_string()),
                host: "lab-box".to_string(),
                port: Some(2222),
            }
        );
        assert_eq!(
            parse_destination("gpu-box").unwrap(),
            Destination {
                user: None,
                host: "gpu-box".to_string(),
                port: None,
            }
        );
    }

    #[test]
    fn a_recorded_host_reads_back_as_the_host_that_was_written() {
        with_scratch(|_| {
            let path = save(&profile("lab"), false).unwrap();
            assert!(
                path.ends_with("remotes/lab.json"),
                "unexpected path {path:?}"
            );
            let read = load("lab").unwrap().expect("the profile is there");
            assert_eq!(read, profile("lab"));
            assert_eq!(read.destination(), "work@100.100.100.100");
            assert!(
                read.summary().contains("llama.cpp · qwen2.5:7b"),
                "the summary hid the runtime or the model: {}",
                read.summary()
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn a_profile_is_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        with_scratch(|_| {
            let path = save(&profile("lab"), false).unwrap();
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "a profile is world-readable at {mode:o}");
        });
    }

    #[test]
    fn adding_the_same_name_twice_is_refused_until_it_is_meant() {
        with_scratch(|_| {
            save(&profile("lab"), false).unwrap();
            let mut moved = profile("lab");
            moved.host = "100.100.100.200".to_string();
            let error = save(&moved, false).expect_err("a second add must not overwrite");
            assert!(
                error.0.contains("--force"),
                "the refusal did not name the way round it: {error}"
            );
            assert_eq!(load("lab").unwrap().unwrap().host, "100.100.100.100");
            save(&moved, true).unwrap();
            assert_eq!(load("lab").unwrap().unwrap().host, "100.100.100.200");
        });
    }

    #[test]
    fn a_file_that_is_not_a_profile_is_named_rather_than_left_out() {
        with_scratch(|scratch| {
            save(&profile("lab"), false).unwrap();
            let dir = scratch.join("remotes");
            std::fs::write(dir.join("broken.json"), "{ not json").unwrap();
            let inventory = list().unwrap();
            assert_eq!(inventory.profiles.len(), 1);
            assert_eq!(inventory.unreadable.len(), 1, "the damaged file went quiet");
            assert_eq!(inventory.unreadable[0].0, "broken");
            assert!(inventory.unreadable[0].1.contains("broken.json"));
        });
    }

    #[test]
    fn the_host_in_use_is_a_pointer_and_only_a_pointer() {
        with_scratch(|scratch| {
            save(&profile("lab"), false).unwrap();
            save(&profile("study"), false).unwrap();
            assert!(active().unwrap().is_none(), "nothing was chosen yet");
            set_active("study").unwrap();
            assert_eq!(active().unwrap().unwrap().name, "study");
            assert_eq!(
                std::fs::read_to_string(scratch.join("remotes/active")).unwrap(),
                "study\n"
            );
            let error = set_active("nobody").expect_err("a pointer cannot name nothing");
            assert!(
                error.0.contains("xencode remote list"),
                "the refusal did not say what to do instead: {error}"
            );
        });
    }

    #[test]
    fn forgetting_a_host_forgets_the_choice_with_it() {
        with_scratch(|_| {
            save(&profile("lab"), false).unwrap();
            save(&profile("study"), false).unwrap();
            set_active("lab").unwrap();
            assert_eq!(forget("lab").unwrap(), Forgot::RemovedWhileActive);
            assert!(active().unwrap().is_none(), "a pointer survived its host");
            assert_eq!(forget("study").unwrap(), Forgot::Removed);
            assert_eq!(forget("study").unwrap(), Forgot::NotRecorded);
            assert!(list().unwrap().is_empty());
        });
    }

    #[test]
    fn only_the_two_runtimes_this_can_bootstrap_are_accepted() {
        for refused in ["Llama", "llama", "vllm", ""] {
            let error =
                validate_runtime(refused).expect_err(&format!("`{refused}` is not a runtime"));
            assert!(error.0.contains("--runtime llama.cpp"), "{error}");
        }
        for accepted in RUNTIMES {
            validate_runtime(accepted).unwrap_or_else(|problem| panic!("{accepted}: {problem}"));
        }
        with_scratch(|_| {
            let mut wrong = profile("lab");
            wrong.runtime = "vllm".to_string();
            assert!(save(&wrong, false).is_err(), "a bad runtime reached disk");
        });
    }

    #[test]
    fn a_model_id_cannot_hold_a_space_or_a_semicolon() {
        assert!(validate_model("TheBloke/X-Q4_K_M.gguf").is_ok());
        for refused in ["a model", "x;rm", "-x", ""] {
            assert!(
                validate_model(refused).is_err(),
                "`{refused}` was accepted as a model id"
            );
        }
        with_scratch(|_| {
            let mut spaced = profile("lab");
            spaced.model = Some("serve me".to_string());
            assert!(save(&spaced, false).is_err(), "a bad model id reached disk");
        });
    }

    #[test]
    fn a_port_of_zero_is_refused_wherever_it_is_found() {
        with_scratch(|_| {
            let mut local_zero = profile("lab");
            local_zero.local_port = 0;
            assert!(
                save(&local_zero, false)
                    .expect_err("a forward on port 0 is no forward")
                    .0
                    .contains("18100"),
                "the refusal did not offer the default"
            );
            let mut ssh_zero = profile("lab");
            ssh_zero.port = 0;
            assert!(save(&ssh_zero, false).is_err());
        });
    }
}
